"""Bidirectional Transformer denoiser (pre-norm, learned absolute position
embeddings, GELU MLP with hidden size 3 x n_embd, tied input/output embeddings)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.attn_drop_p = dropout
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        drop_p = self.attn_drop_p if self.training else 0.0
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False, dropout_p=drop_p)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.c_proj(y))


class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(n_embd, n_head, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        mlp_dim = 3 * n_embd
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, mlp_dim), nn.GELU(),
            nn.Linear(mlp_dim, n_embd), nn.Dropout(dropout))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, vocab_size, block_size=256,
                 n_layer=6, n_head=6, n_embd=384, dropout=0.2):
        super().__init__()
        self.block_size = block_size

        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            Block(n_embd, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        # weight tying
        self.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        self.register_buffer('_pos_idx',
                             torch.arange(block_size, dtype=torch.long),
                             persistent=False)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx):
        B, T = idx.size()
        x = self.wte(idx) + self.wpe(self._pos_idx[:T])
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.ln_f(x))

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
