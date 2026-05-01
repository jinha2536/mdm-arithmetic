"""Toy MDM simulation for paper Section 3 (mechanism propositions).

Task: chain prediction in a bidirectional masked setting.
    x_0 ~ Uniform({0, ..., V-1})
    x_i = (x_{i-1} + i) mod V    for i = 1, ..., L-1

This is a minimal MDM-appropriate reasoning task:
  - Any cell determines all others bidirectionally.
  - x_0 is "hard" (no inductive bias from MDM pos-prior alone).
  - Chain interior cells are "easy" once neighbors unmasked.
  - Confidence-greedy decode reveals informative cells first;
    LSB-style oracle reveals x_0 first then propagates.

Three training modes are compared (matching exp_*.py code):
  1. Random masking baseline
  2. PAPL: weight = (1 + α·w_i)/(L-k) where w_i ∝ exp((1/τ) log P)
  3. PUMA: confidence-greedy forward process with K-schedule

Outputs:
  - probe_loss trajectory (random-masked test NLL over training)
  - per-position accuracy at confidence decode (final)
  - per-position accuracy at oracle decode (final)
  - accuracy vs chain length (test sets of varying L_eval)

Run: python toy_simulation.py [--out figures/]
Time: ~5-10 min on a single GPU, ~20 min on CPU.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────

CONFIG = dict(
    V=10,                 # vocab size
    L_train=12,           # chain length during training
    L_eval=[4, 8, 12, 16, 20],  # eval chain lengths (16, 20 are OOD)
    n_train=20000,        # training samples (regenerated each epoch)
    batch_size=256,
    n_iters=20000,        # training iterations
    eval_every=1000,
    lr=3e-4,
    n_layer=2,
    n_head=4,
    n_embd=64,
    dropout=0.1,
    # PAPL
    papl_alpha=1.0,
    papl_tau=1.0,
    # PUMA: step schedule from K_start to K_end over first 1/3 of training
    puma_k_start=2,
    puma_k_end=6,
    puma_k_step=1,
    seed=0,
)


# ─────────────────────────────────────────────────────────────────
# Tokenization: vocab = {0..V-1, MASK=V, BOS=V+1}
# ─────────────────────────────────────────────────────────────────

def make_tokens(V):
    return {'pad': V, 'mask': V, 'bos': V + 1, 'vocab_size': V + 2}


def gen_chain_data(n, L, V, device, generator):
    """x_0 ~ Uniform; x_i = (x_{i-1} + i) mod V."""
    x = torch.zeros(n, L, dtype=torch.long, device=device)
    x[:, 0] = torch.randint(0, V, (n,), generator=generator, device=device)
    for i in range(1, L):
        x[:, i] = (x[:, i - 1] + i) % V
    return x


# ─────────────────────────────────────────────────────────────────
# Tiny transformer (no causal mask — bidirectional)
# ─────────────────────────────────────────────────────────────────

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer,
                 max_seq_len, dropout):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(max_seq_len, n_embd)
        self.dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=n_embd, nhead=n_head, dim_feedforward=4 * n_embd,
            dropout=dropout, batch_first=True, activation='gelu',
            norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, num_layers=n_layer)
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        # tie input/output embeddings
        self.head.weight = self.tok_emb.weight

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = self.tok_emb(x) + self.pos_emb(pos)
        h = self.dropout(h)
        h = self.blocks(h)
        h = self.ln_f(h)
        return self.head(h)  # [B, T, V]


# ─────────────────────────────────────────────────────────────────
# Masking strategies
# ─────────────────────────────────────────────────────────────────

def random_mask(x, mask_id, generator):
    """Standard MDM: each position independently masked with random rate.
    Matches the per-sample-mean formulation used in exp_*.py."""
    B, T = x.shape
    rates = torch.rand(B, 1, generator=generator, device=x.device)
    rand = torch.rand(B, T, generator=generator, device=x.device)
    m = rand < rates
    # Ensure at least one position masked
    no_mask = ~m.any(dim=-1)
    if no_mask.any():
        idx = torch.randint(0, T, (no_mask.sum(),),
                            generator=generator, device=x.device)
        m[no_mask, idx] = True
    xm = x.clone()
    xm[m] = mask_id
    return xm, m


def puma_mask_step(model, x, mask_id, K, generator):
    """PUMA forward process: starting from fully masked state, iteratively
    reveal K most-confident positions. Returns (xm, mask_m) at a uniformly
    random midpoint of the trajectory (matching exp_*.py train_diffusion).

    Concretely:
      - Sample a target reveal count k ∈ {0, K, 2K, ..., L} uniformly
      - Run forward process for k/K steps, ground-truth-revealing top-K each
      - The state at this point is the training mask
    """
    B, T = x.shape
    n_steps = (T + K - 1) // K
    # Sample which step each sample is at
    target_step = torch.randint(0, n_steps + 1, (B,),
                                generator=generator, device=x.device)
    # Iteratively build state: start fully masked, reveal K positions per step
    xm = torch.full_like(x, mask_id)
    unmasked = torch.zeros(B, T, dtype=torch.bool, device=x.device)
    model.eval()
    with torch.no_grad():
        for step in range(n_steps):
            need = target_step > step
            if not need.any():
                break
            logits = model(xm)
            # confidence at masked positions
            log_probs = F.log_softmax(logits, dim=-1)
            top_logp = log_probs.max(dim=-1).values
            top_logp = top_logp.masked_fill(unmasked, -float('inf'))
            # select top K per sample
            _, top_idx = top_logp.topk(K, dim=-1)
            # only update samples that need this step
            for_update = need.unsqueeze(-1).expand_as(top_idx)
            ba = torch.arange(B, device=x.device).unsqueeze(-1).expand_as(top_idx)
            mask_for_update = for_update
            # Reveal ground truth for selected positions
            xm[ba[mask_for_update], top_idx[mask_for_update]] = \
                x[ba[mask_for_update], top_idx[mask_for_update]]
            unmasked[ba[mask_for_update], top_idx[mask_for_update]] = True
    model.train()
    m = ~unmasked
    return xm, m


def puma_K_schedule(it, n_iters, K_start, K_end, K_step):
    """Step schedule: K_start at iter 0, ramp to K_end over first n_iters/3."""
    ramp_iters = max(1, n_iters // 3)
    if it >= ramp_iters:
        return K_end
    n_increments = max(1, (K_end - K_start) // K_step)
    inc_period = ramp_iters // n_increments
    n_done = it // inc_period
    return min(K_start + K_step * n_done, K_end)


# ─────────────────────────────────────────────────────────────────
# Loss computation (matches exp_*.py exactly)
# ─────────────────────────────────────────────────────────────────

def compute_loss(model, x, mask_id, mode, alpha=1.0, tau=1.0,
                 mask_state=None):
    """mode ∈ {'random', 'papl', 'puma'}.
    mask_state can be precomputed (xm, m) or None to use random_mask."""
    if mask_state is not None:
        xm, m = mask_state
    else:
        xm, m = random_mask(x, mask_id, torch.Generator(device=x.device))
    if m.sum() == 0:
        return torch.tensor(0.0, device=x.device, requires_grad=True)
    logits = model(xm)
    log_probs = F.log_softmax(logits.float(), dim=-1)
    tlp = log_probs.gather(-1, x.unsqueeze(-1)).squeeze(-1)  # [B, T]
    nll = -tlp
    n_masked = m.sum(dim=-1).clamp_min(1).float()  # [B]
    if mode == 'papl':
        det = (tlp.detach() / tau).masked_fill(~m, float('-inf'))
        w_papl = F.softmax(det, dim=-1)
        base_w = (1.0 / n_masked).unsqueeze(-1)
        weights = base_w * (1.0 + alpha * w_papl)
        per_sample = (weights * nll * m.float()).sum(dim=-1)
    else:
        # 'random' and 'puma' both use vanilla per-sample mean NLL
        # (PUMA's intervention is in the *masking*, not the loss)
        per_sample = (nll * m.float()).sum(dim=-1) / n_masked
    return per_sample.mean()


# ─────────────────────────────────────────────────────────────────
# Generation (decode policies)
# ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate(model, B, L, mask_id, device, policy='confidence',
             oracle_order=None):
    """Generate from fully masked state. Returns predicted sequence + decode order.
    policy:
      - 'confidence': pick highest-confidence masked position each step
      - 'oracle_lsb': follow oracle_order (e.g., [0, 1, 2, ..., L-1])
    """
    model.eval()
    x = torch.full((B, L), mask_id, dtype=torch.long, device=device)
    unmasked = torch.zeros(B, L, dtype=torch.bool, device=device)
    decode_order = torch.zeros(B, L, dtype=torch.long, device=device)
    for step in range(L):
        logits = model(x)  # [B, T, V]
        if policy == 'confidence':
            top_logp = F.log_softmax(logits, dim=-1).max(dim=-1).values
            top_logp = top_logp.masked_fill(unmasked, -float('inf'))
            pos = top_logp.argmax(-1)  # [B]
        elif policy == 'oracle_lsb':
            pos = torch.full((B,), oracle_order[step], dtype=torch.long,
                             device=device)
        ba = torch.arange(B, device=device)
        toks = logits[ba, pos].argmax(-1)
        x[ba, pos] = toks
        unmasked[ba, pos] = True
        decode_order[:, step] = pos
    return x, decode_order


# ─────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_probe_loss(model, x_test, mask_id, n_samples=4):
    """Probe loss: NLL on uniform random masking. Measures how well
    model handles the random training distribution. Average over
    n_samples mask draws to reduce variance."""
    model.eval()
    losses = []
    for _ in range(n_samples):
        gen = torch.Generator(device=x_test.device)
        gen.manual_seed(int(torch.randint(0, 1_000_000, (1,)).item()))
        xm, m = random_mask(x_test, mask_id, gen)
        if m.sum() == 0:
            continue
        logits = model(xm)
        log_probs = F.log_softmax(logits, dim=-1)
        tlp = log_probs.gather(-1, x_test.unsqueeze(-1)).squeeze(-1)
        nll = -tlp
        # per-sample mean over masked
        n_masked = m.sum(dim=-1).clamp_min(1).float()
        per_sample = (nll * m.float()).sum(dim=-1) / n_masked
        losses.append(per_sample.mean().item())
    return float(np.mean(losses)) if losses else float('nan')


@torch.no_grad()
def eval_accuracy(model, x_test, L, mask_id, policy='confidence'):
    """Generate full sequences from scratch, return per-position accuracy."""
    B = x_test.shape[0]
    if policy == 'confidence':
        pred, _ = generate(model, B, L, mask_id, x_test.device,
                           policy='confidence')
    elif policy == 'oracle_lsb':
        order = list(range(L))  # x_0 first, then propagate
        pred, _ = generate(model, B, L, mask_id, x_test.device,
                           policy='oracle_lsb', oracle_order=order)
    correct = (pred == x_test).float()
    per_pos = correct.mean(dim=0)  # [L]
    overall = correct.all(dim=-1).float().mean()
    return per_pos.cpu().numpy(), float(overall)


# ─────────────────────────────────────────────────────────────────
# Training loop for one mode
# ─────────────────────────────────────────────────────────────────

def train_one(mode, cfg, x_train, x_test, device):
    print(f"\n{'='*60}\n  Training: {mode}\n{'='*60}")
    torch.manual_seed(cfg['seed'])
    tokens = make_tokens(cfg['V'])
    mask_id = tokens['mask']
    model = TinyTransformer(
        vocab_size=tokens['vocab_size'], n_embd=cfg['n_embd'],
        n_head=cfg['n_head'], n_layer=cfg['n_layer'],
        max_seq_len=max(cfg['L_eval']) + 1, dropout=cfg['dropout']).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'])
    history = {'iter': [], 'probe_loss': [],
               'acc_conf': {L: [] for L in cfg['L_eval']},
               'acc_oracle': {L: [] for L in cfg['L_eval']}}
    n_train = x_train.shape[0]
    gen = torch.Generator(device=device); gen.manual_seed(cfg['seed'])
    for it in range(cfg['n_iters']):
        idx = torch.randint(0, n_train, (cfg['batch_size'],),
                            generator=gen, device=device)
        x_batch = x_train[idx]
        if mode == 'puma':
            K = puma_K_schedule(it, cfg['n_iters'], cfg['puma_k_start'],
                                cfg['puma_k_end'], cfg['puma_k_step'])
            mask_state = puma_mask_step(model, x_batch, mask_id, K, gen)
        else:
            mask_state = random_mask(x_batch, mask_id, gen)
        loss = compute_loss(model, x_batch, mask_id, mode,
                            alpha=cfg['papl_alpha'], tau=cfg['papl_tau'],
                            mask_state=mask_state)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if it % cfg['eval_every'] == 0 or it == cfg['n_iters'] - 1:
            probe = eval_probe_loss(model, x_test, mask_id)
            history['iter'].append(it)
            history['probe_loss'].append(probe)
            for L in cfg['L_eval']:
                # Generate L-length test set
                gen_test = torch.Generator(device=device)
                gen_test.manual_seed(123 + L)
                x_eval = gen_chain_data(500, L, cfg['V'], device, gen_test)
                _, acc_c = eval_accuracy(model, x_eval, L, mask_id,
                                         policy='confidence')
                _, acc_o = eval_accuracy(model, x_eval, L, mask_id,
                                         policy='oracle_lsb')
                history['acc_conf'][L].append(acc_c)
                history['acc_oracle'][L].append(acc_o)
            extra = ''
            if mode == 'puma':
                extra = f' K={puma_K_schedule(it, cfg["n_iters"], cfg["puma_k_start"], cfg["puma_k_end"], cfg["puma_k_step"])}'
            print(f"  it={it:>6}  probe={probe:.3f}  loss={loss.item():.3f}  "
                  f"acc_conf(L=12)={history['acc_conf'][12][-1]:.3f}  "
                  f"acc_oracle(L=12)={history['acc_oracle'][12][-1]:.3f}{extra}")
    # Per-position accuracy at end of training
    final = {'per_pos_conf': {}, 'per_pos_oracle': {}}
    for L in cfg['L_eval']:
        gen_test = torch.Generator(device=device); gen_test.manual_seed(999 + L)
        x_eval = gen_chain_data(2000, L, cfg['V'], device, gen_test)
        per_pos_c, _ = eval_accuracy(model, x_eval, L, mask_id, policy='confidence')
        per_pos_o, _ = eval_accuracy(model, x_eval, L, mask_id, policy='oracle_lsb')
        final['per_pos_conf'][L] = per_pos_c
        final['per_pos_oracle'][L] = per_pos_o
    return model, history, final


# ─────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────

COLORS = {'random': '#1F3A5F', 'papl': '#E67E22', 'puma': '#16A085'}
LABELS = {'random': 'Random', 'papl': 'PAPL', 'puma': 'PUMA'}
MARKERS = {'random': 'o', 'papl': 's', 'puma': '^'}


def apply_paper_style():
    plt.rcParams.update({
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'axes.labelsize': 9, 'axes.titlesize': 10,
        'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'legend.fontsize': 8, 'figure.titlesize': 10,
        'axes.linewidth': 0.6, 'grid.linewidth': 0.4, 'grid.alpha': 0.3,
        'lines.linewidth': 1.4, 'lines.markersize': 4,
        'legend.frameon': False,
        'axes.spines.top': False, 'axes.spines.right': False,
    })


def plot_results(histories, finals, cfg, outdir):
    apply_paper_style()
    fig, axes = plt.subplots(2, 2, figsize=(6.75, 4.8))

    # (a) probe loss trajectory — Proposition 3
    ax = axes[0, 0]
    for mt in ['random', 'papl', 'puma']:
        h = histories[mt]
        ax.plot(h['iter'], h['probe_loss'],
                color=COLORS[mt], label=LABELS[mt],
                marker=MARKERS[mt], markersize=3, markeredgewidth=0)
    ax.set_xlabel('training iteration')
    ax.set_ylabel('probe NLL (uniform-masked test)')
    ax.set_title('(a) Out-of-distribution NLL trajectory', loc='left')
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)

    # (b) accuracy at confidence decode by chain length — at end of training
    ax = axes[0, 1]
    Ls = cfg['L_eval']
    for mt in ['random', 'papl', 'puma']:
        accs = [histories[mt]['acc_conf'][L][-1] for L in Ls]
        ax.plot(Ls, accs, color=COLORS[mt], label=LABELS[mt],
                marker=MARKERS[mt], markersize=4)
    ax.set_xlabel('chain length L')
    ax.set_ylabel('exact-match accuracy')
    ax.set_title('(b) Confidence decode (vs chain length)', loc='left')
    ax.set_ylim(-0.03, 1.05)
    ax.axvline(cfg['L_train'], color='gray', ls=':', lw=0.6, alpha=0.6)
    ax.text(cfg['L_train'] + 0.3, 0.05, 'train L', fontsize=7, color='gray')
    ax.grid(alpha=0.3)

    # (c) accuracy at oracle decode
    ax = axes[1, 0]
    for mt in ['random', 'papl', 'puma']:
        accs = [histories[mt]['acc_oracle'][L][-1] for L in Ls]
        ax.plot(Ls, accs, color=COLORS[mt], label=LABELS[mt],
                marker=MARKERS[mt], markersize=4)
    ax.set_xlabel('chain length L')
    ax.set_ylabel('exact-match accuracy')
    ax.set_title('(c) Oracle LSB decode (vs chain length)', loc='left')
    ax.set_ylim(-0.03, 1.05)
    ax.axvline(cfg['L_train'], color='gray', ls=':', lw=0.6, alpha=0.6)
    ax.grid(alpha=0.3)

    # (d) per-position accuracy at training-length L
    ax = axes[1, 1]
    L = cfg['L_train']
    pos = np.arange(L)
    for mt in ['random', 'papl', 'puma']:
        ax.plot(pos, finals[mt]['per_pos_conf'][L],
                color=COLORS[mt], label=f"{LABELS[mt]} (conf)",
                marker=MARKERS[mt], markersize=3, ls='-')
    ax.set_xlabel('chain position i')
    ax.set_ylabel('per-position accuracy')
    ax.set_title('(d) Per-position acc at L=12, confidence decode', loc='left')
    ax.set_ylim(-0.03, 1.05)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    pdf = Path(outdir) / 'toy_chain_simulation.pdf'
    fig.savefig(pdf, bbox_inches='tight')
    png = Path(outdir) / 'toy_chain_simulation.png'
    fig.savefig(png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n✓ Saved: {pdf}")
    print(f"✓ Saved: {png}")


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=str, default='figures/')
    p.add_argument('--n-iters', type=int, default=None)
    p.add_argument('--device', type=str, default=None)
    args = p.parse_args()

    cfg = dict(CONFIG)
    if args.n_iters:
        cfg['n_iters'] = args.n_iters
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Config: {cfg}")

    Path(args.out).mkdir(parents=True, exist_ok=True)

    # Generate train/test data (re-generated per epoch is overkill;
    # fix a large pool and resample)
    torch.manual_seed(cfg['seed'])
    gen = torch.Generator(device=device); gen.manual_seed(cfg['seed'])
    x_train = gen_chain_data(cfg['n_train'], cfg['L_train'], cfg['V'],
                             device, gen)
    gen_test = torch.Generator(device=device); gen_test.manual_seed(cfg['seed'] + 1)
    x_test = gen_chain_data(2000, cfg['L_train'], cfg['V'], device, gen_test)

    histories = {}; finals = {}
    for mode in ['random', 'papl', 'puma']:
        _, hist, final = train_one(mode, cfg, x_train, x_test, device)
        histories[mode] = hist
        finals[mode] = final

    plot_results(histories, finals, cfg, args.out)

    # Save raw data for paper
    import json
    summary = {'config': {k: (v if not isinstance(v, list) else list(v))
                          for k, v in cfg.items()}}
    for mt in histories:
        h = histories[mt]
        summary[f'{mt}_history'] = {
            'iter': h['iter'], 'probe_loss': h['probe_loss'],
            'acc_conf': {str(L): v for L, v in h['acc_conf'].items()},
            'acc_oracle': {str(L): v for L, v in h['acc_oracle'].items()},
        }
        summary[f'{mt}_final'] = {
            'per_pos_conf': {str(L): list(map(float, v))
                             for L, v in finals[mt]['per_pos_conf'].items()},
            'per_pos_oracle': {str(L): list(map(float, v))
                               for L, v in finals[mt]['per_pos_oracle'].items()},
        }
    with open(Path(args.out) / 'toy_chain_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved: {Path(args.out) / 'toy_chain_results.json'}")

    # Print summary table
    print(f"\n{'='*60}")
    print("Summary at end of training")
    print(f"{'='*60}")
    print(f"{'mode':<10} {'probe_loss':>11} {'acc_conf(L=12)':>16} {'acc_oracle(L=12)':>18} {'acc_conf(L=20)':>16}")
    for mt in ['random', 'papl', 'puma']:
        h = histories[mt]
        print(f"{mt:<10} {h['probe_loss'][-1]:>11.3f} "
              f"{h['acc_conf'][12][-1]:>16.3f} "
              f"{h['acc_oracle'][12][-1]:>18.3f} "
              f"{h['acc_conf'][20][-1]:>16.3f}")


if __name__ == '__main__':
    main()
