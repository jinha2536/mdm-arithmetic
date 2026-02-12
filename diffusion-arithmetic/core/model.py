"""
Shared Transformer backbone.

  is_causal=True  → AR  (causal mask, left-to-right)
  is_causal=False → diffusion (bidirectional, RADD-style)

  pos_enc='absolute' → learned absolute position embedding (NanoGPT)
  pos_enc='rope'     → Rotary Position Embedding (RADD / LLaDA style)
  pos_enc='none'     → no positional encoding

RoPE helps length generalisation because relative positions remain
meaningful at unseen absolute positions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RoPE helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _rope_freqs(dim: int, max_len: int, base: float = 10000.0):
    """Pre-compute cos/sin tables for RoPE."""
    assert dim % 2 == 0
    half = dim // 2
    freqs = 1.0 / (base ** (torch.arange(0, half).float() / half))  # (half,)
    t = torch.arange(max_len).float()                                # (T,)
    angles = torch.outer(t, freqs)                                   # (T, half)
    return torch.cos(angles), torch.sin(angles)                      # each (T, half)


def _apply_rope(x, cos, sin):
    """
    x: (B, n_head, T, head_dim)
    cos, sin: (T, head_dim//2)  — will be broadcast
    """
    T = x.shape[2]
    cos = cos[:T].unsqueeze(0).unsqueeze(0)   # (1,1,T,half)
    sin = sin[:T].unsqueeze(0).unsqueeze(0)
    x1, x2 = x[..., ::2], x[..., 1::2]       # each (B,H,T,half)
    out = torch.stack([x1 * cos - x2 * sin,
                       x1 * sin + x2 * cos], dim=-1)
    return out.flatten(-2)                     # (B,H,T,head_dim)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Attention
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout, block_size,
                 is_causal=True, pos_enc='absolute'):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.is_causal = is_causal
        self.pos_enc = pos_enc

        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        if is_causal:
            self.register_buffer(
                "causal_mask",
                torch.tril(torch.ones(block_size, block_size))
                     .view(1, 1, block_size, block_size))

        if pos_enc == 'rope':
            cos, sin = _rope_freqs(self.head_dim, block_size)
            self.register_buffer('rope_cos', cos)
            self.register_buffer('rope_sin', sin)

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if self.pos_enc == 'rope':
            q = _apply_rope(q, self.rope_cos, self.rope_sin)
            k = _apply_rope(k, self.rope_cos, self.rope_sin)

        att = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        if self.is_causal:
            att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))
        att = self.attn_drop(F.softmax(att, dim=-1))
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.c_proj(y))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Block & Transformer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout, block_size,
                 is_causal, pos_enc):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(n_embd, n_head, dropout, block_size,
                                  is_causal, pos_enc)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.GELU(),
            nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class Transformer(nn.Module):
    """
    Args:
        pos_enc: 'absolute' | 'rope' | 'none'

    Note on RoPE + bidirectional (diffusion):
        RoPE provides position info only through Q·K attention rotation,
        not through token embeddings. MASK tokens at different positions
        share the same embedding entering the first layer, but attention
        weights still depend on position via RoPE. This matches LLaDA's
        architecture (LLaMA3 with causal mask removed, pure RoPE).
        If RoPE+diffusion underperforms absolute PE at small scale,
        that is itself an interesting finding about positional encoding
        requirements for masked diffusion models.
    """
    def __init__(self, vocab_size, block_size=256,
                 n_layer=6, n_head=6, n_embd=384, dropout=0.2,
                 is_causal=True, pos_enc='absolute'):
        super().__init__()
        self.block_size = block_size
        self.pos_enc = pos_enc

        self.wte = nn.Embedding(vocab_size, n_embd)
        if pos_enc == 'absolute':
            self.wpe = nn.Embedding(block_size, n_embd)
        else:
            self.wpe = None

        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            Block(n_embd, n_head, dropout, block_size, is_causal, pos_enc)
            for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        # weight tying
        self.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx):
        B, T = idx.size()
        x = self.wte(idx)
        if self.wpe is not None:
            x = x + self.wpe(torch.arange(T, device=idx.device))
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.ln_f(x))

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
