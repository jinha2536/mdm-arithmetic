"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Experiment 3 — Toy Distribution: Decoding Policy Analysis
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Colab:  %run experiments/exp_toy_distribution.py

  Train diffusion on toy distributions with KNOWN ground truth p(x).
  Compare decoding policies via exact distributional metrics.

  4 distributions (V=4, L=8 → 65536 sequences, fully enumerable):
    A: Near-independent  (no dependency)
    B: 2nd-order Markov  (strong local dependency)
    C: Global sum        (long-range coupling)
    D: Markov + Global   (local + global)

  3 policy families:
    Sequential (7): confidence, low_entropy, high_entropy, margin,
                    random, l2r, r2l  (all NFE=8)
    Adaptive threshold (3): τ=0.5, 0.7, 0.9 — decode all positions
                    above threshold simultaneously (variable NFE)
    Jacobi iteration (3): max_iter=5, 10, 20 — predict all at once,
                    iterate to fixed point (variable NFE)

  Key question: can adaptive/jacobi match sequential quality with
  fewer forward passes?  NFE vs TV scatter shows the answer.

  Decoding: sampling (not greedy — distributional metrics require
  stochasticity). Decode order tracked for first 10K samples only
  (speed optimization).
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                if '__file__' in dir() else '.')

import time, math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('Agg')
from collections import defaultdict
from itertools import product as iter_product
from scipy.stats import spearmanr

from core.model import Transformer
from core.train_utils import mount_drive, save_results, DEVICE

EXP_NAME = 'exp_toy_distribution'

# ── Config ──────────────────────────────────────────
V = 4            # vocab {0,1,2,3}
L = 8            # length → V^L = 65,536 possible sequences
MASK_ID = V      # = 4
TOTAL_V = V + 1  # = 5

N_TRAIN   = 200_000
N_SAMPLES = 500_000   # per policy — need >> V^L for reliable TV/KL
                       # TV noise ≈ sqrt(V^L / 2πN) ≈ 0.14 at 500K
BOOTSTRAP_K = 20       # bootstrap resamples for CI
MAX_EPOCHS = 60
PATIENCE_EPOCHS = 8
BATCH_SIZE = 1024      # larger batch for faster generation
LR = 3e-4

D_MODEL = 192; N_HEADS = 4; N_LAYERS = 4

# ── Distributions ───────────────────────────────────

class ToyDistribution:
    def __init__(self):
        self._full = None

    def log_prob(self, x):
        raise NotImplementedError

    def sample(self, n):
        all_seqs, probs = self.full_distribution()
        return all_seqs[torch.multinomial(probs, n, replacement=True)]

    def full_distribution(self):
        if self._full is not None:
            return self._full
        all_seqs = torch.tensor(list(iter_product(range(V), repeat=L)),
                                dtype=torch.long)
        lp = self.log_prob(all_seqs)
        lp = lp - torch.logsumexp(lp, dim=0)
        self._full = (all_seqs, torch.exp(lp))
        return self._full

    def entropy(self):
        _, p = self.full_distribution()
        return -(p * (p + 1e-30).log()).sum().item() / np.log(2)

    def optimal_masked_loss(self, n_eval=50_000):
        """
        Compute the theoretically optimal cross-entropy for a perfect
        masked predictor. This is the Bayes-optimal loss — no model of
        any size can beat this.

        For each (x, mask_config), the optimal predictor outputs
        p(x_i | x_{unmasked}) for each masked position i.
        We Monte-Carlo estimate this over random x and random masks.
        """
        all_seqs, p_true = self.full_distribution()
        samples = self.sample(n_eval)
        total_loss, total_tokens = 0.0, 0

        # For tractability, compute exact conditionals
        # p(x_i=v | x_unmasked) = sum_{x': x'_unmasked=x_unmasked, x'_i=v} p(x')
        #                        / sum_{x': x'_unmasked=x_unmasked} p(x')
        # This is expensive for large V^L, so we use a sampling approach:
        # sample mask configs, then for each, compute the conditional.

        # Faster approach: for random mask ratios, estimate expected CE
        rng = np.random.RandomState(0)
        for trial in range(min(2000, n_eval)):
            x = samples[trial]
            t = rng.uniform(0.1, 0.9)
            mask = torch.bernoulli(torch.full((L,), t)).bool()
            if not mask.any():
                mask[rng.randint(L)] = True

            unmasked_vals = x.clone()
            unmasked_vals[mask] = -1  # sentinel

            # Find all sequences matching unmasked positions
            match = torch.ones(len(all_seqs), dtype=torch.bool)
            for pos in range(L):
                if not mask[pos]:
                    match = match & (all_seqs[:, pos] == x[pos])

            if match.sum() == 0:
                continue

            cond_probs = p_true[match]
            cond_probs = cond_probs / cond_probs.sum()
            matching = all_seqs[match]

            for pos in mask.nonzero(as_tuple=True)[0]:
                true_val = x[pos].item()
                # p(x_pos = true_val | context)
                p_cond = cond_probs[matching[:, pos] == true_val].sum().item()
                if p_cond > 1e-10:
                    total_loss -= np.log(p_cond)
                else:
                    total_loss += 20  # cap
                total_tokens += 1

        return total_loss / max(total_tokens, 1)


class NearIndependent(ToyDistribution):
    """
    p(x) = prod p_i(x_i). Position-wise bias, no dependency.
    alpha controls peakedness: lower → more peaked marginals.
      alpha=3.0: near-uniform (H ≈ 1.9 bits/pos → cond_acc ~0.39)
      alpha=0.5: sparse/peaked (H ≈ 1.0 bits/pos → cond_acc ~0.65)
    """
    def __init__(self, alpha=0.5, seed=42):
        super().__init__()
        rng = np.random.RandomState(seed)
        self.pos_log_probs = torch.tensor(
            np.log(rng.dirichlet([alpha] * V, size=L) + 1e-30),
            dtype=torch.float32)

    def log_prob(self, x):
        x = x.reshape(-1, L)
        return sum(self.pos_log_probs[i][x[:, i]] for i in range(L))


class MarkovChain(ToyDistribution):
    """
    2nd-order Markov chain with peaked transition matrices.
    sparsity controls how peaked: lower → more deterministic transitions.
    """
    def __init__(self, order=2, sparsity=0.1, seed=42):
        super().__init__()
        self.k = order
        rng = np.random.RandomState(seed)
        self.trans = {}
        for ctx in iter_product(range(V), repeat=order):
            alpha = rng.uniform(sparsity * 0.5, sparsity, size=V)
            alpha[rng.randint(V)] += 3.0  # stronger peak
            self.trans[ctx] = torch.tensor(rng.dirichlet(alpha),
                                           dtype=torch.float32)
        self.init_lp = np.log(1.0 / (V ** order))

    def log_prob(self, x):
        x = x.reshape(-1, L); n = x.shape[0]
        lp = torch.full((n,), self.init_lp, dtype=torch.float32)
        for t in range(self.k, L):
            for i in range(n):
                ctx = tuple(x[i, t-self.k:t].tolist())
                lp[i] += self.trans[ctx][x[i, t].item()].log()
        return lp


class GlobalSumConstraint(ToyDistribution):
    """
    p(x) ∝ exp(-β|Σx_i - τ|²) × ∏ p_i(x_i).
    beta controls coupling strength: higher → stronger global dependency.
    """
    def __init__(self, beta=2.0, alpha=0.5, seed=42):
        super().__init__()
        self.beta, self.target = beta, (V-1)*L/2.0
        rng = np.random.RandomState(seed)
        self.base_lp = torch.tensor(
            np.log(rng.dirichlet([alpha]*V, size=L) + 1e-30), dtype=torch.float32)

    def log_prob(self, x):
        x = x.reshape(-1, L)
        lp = sum(self.base_lp[i][x[:, i]] for i in range(L))
        lp += -self.beta * (x.float().sum(-1) - self.target) ** 2
        return lp


class MarkovPlusGlobal(ToyDistribution):
    """Markov local + global sum constraint. Both strong."""
    def __init__(self, beta=1.5, order=2, sparsity=0.1, seed=42):
        super().__init__()
        self.markov = MarkovChain(order, sparsity, seed)
        self.beta, self.target = beta, (V-1)*L/2.0

    def log_prob(self, x):
        x = x.reshape(-1, L)
        lp = self.markov.log_prob(x)
        lp += -self.beta * (x.float().sum(-1) - self.target) ** 2
        return lp


# ── Train ───────────────────────────────────────────

def train_toy_model(dist, print_every=5):
    print(f"  Generating {N_TRAIN} samples...")
    train_data = dist.sample(N_TRAIN)

    model = Transformer(
        vocab_size=TOTAL_V, block_size=L + 4,
        n_layer=N_LAYERS, n_head=N_HEADS, n_embd=D_MODEL,
        dropout=0.1, is_causal=False, pos_enc='absolute',
    ).to(DEVICE)

    loader = DataLoader(train_data, batch_size=BATCH_SIZE,
                        shuffle=True, drop_last=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS * len(loader))

    history = {'loss': [], 'cond_acc': []}
    best_loss, wait = float('inf'), 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        epoch_loss, n_b = 0, 0
        for batch in loader:
            batch = batch.to(DEVICE); B, T = batch.shape
            t = torch.rand(B, device=DEVICE)
            mask = torch.bernoulli(t.unsqueeze(1).expand(B, T)).bool()
            no_m = ~mask.any(dim=1)
            if no_m.any():
                mask[no_m, torch.randint(0, T, (no_m.sum(),), device=DEVICE)] = True
            x_m = batch.clone(); x_m[mask] = MASK_ID
            logits = model(x_m)
            loss = F.cross_entropy(logits[mask], batch[mask])
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step()
            epoch_loss += loss.item(); n_b += 1

        avg_loss = epoch_loss / n_b
        history['loss'].append(avg_loss)

        model.eval()
        with torch.no_grad():
            x = dist.sample(2000).to(DEVICE)
            t = torch.rand(2000, device=DEVICE) * 0.8 + 0.1
            m = torch.bernoulli(t.unsqueeze(1).expand(2000, L)).bool()
            no_m = ~m.any(dim=1)
            if no_m.any():
                m[no_m, torch.randint(0, L, (no_m.sum(),), device=DEVICE)] = True
            x_m = x.clone(); x_m[m] = MASK_ID
            logits = model(x_m)
            acc = ((logits.argmax(-1) == x) & m).sum().float() / m.sum().float()
        history['cond_acc'].append(acc.item())

        if epoch % print_every == 0 or epoch == 1:
            print(f"    epoch {epoch:3d} | loss {avg_loss:.4f} | "
                  f"cond_acc {acc.item():.4f}")

        # Convergence check
        if avg_loss < best_loss - 1e-4:
            best_loss, wait = avg_loss, 0
        else:
            wait += 1
            if wait >= PATIENCE_EPOCHS:
                print(f"    ✓ Converged at epoch {epoch}")
                break

    model.eval()
    return model, history


# ── Decode ──────────────────────────────────────────

N_ORDER_SAMPLES = 10_000  # only track decode order for this many

@torch.no_grad()
def decode_sequential(model, n, policy='confidence', track_order=False):
    """Sequential decode. track_order=True records position chosen at each step."""
    x = torch.full((n, L), MASK_ID, dtype=torch.long, device=DEVICE)
    unmasked = torch.zeros(n, L, dtype=torch.bool, device=DEVICE)
    scores = torch.zeros(n, device=DEVICE)
    orders = torch.zeros(n, L, dtype=torch.long, device=DEVICE) if track_order else None

    for t in range(L):
        logits = model(x)
        logits[:, :, MASK_ID] = -float('inf')
        probs = F.softmax(logits, dim=-1)

        if policy == 'confidence':
            sc = probs.max(-1).values.clone(); sc[unmasked] = -1
            pos = sc.argmax(-1)
        elif policy == 'low_entropy':
            e = -(probs*(probs+1e-10).log()).sum(-1); e[unmasked] = 1e9
            pos = e.argmin(-1)
        elif policy == 'high_entropy':
            e = -(probs*(probs+1e-10).log()).sum(-1); e[unmasked] = -1e9
            pos = e.argmax(-1)
        elif policy == 'margin':
            t2 = probs.topk(2, dim=-1).values
            mg = t2[:,:,0]-t2[:,:,1]; mg[unmasked] = -1; pos = mg.argmax(-1)
        elif policy == 'random':
            pos = torch.zeros(n, dtype=torch.long, device=DEVICE)
            for b in range(n):
                mp = (~unmasked[b]).nonzero(as_tuple=True)[0]
                pos[b] = mp[torch.randint(len(mp), (1,))]
        elif policy == 'l2r':
            pos = torch.full((n,), t, dtype=torch.long, device=DEVICE)
        elif policy == 'r2l':
            pos = torch.full((n,), L-1-t, dtype=torch.long, device=DEVICE)
        else:
            raise ValueError(policy)

        if track_order:
            orders[:, t] = pos

        lp = logits[torch.arange(n, device=DEVICE), pos]
        tok = torch.multinomial(F.softmax(lp, dim=-1), 1).squeeze(-1)
        scores += F.log_softmax(lp, dim=-1)[torch.arange(n, device=DEVICE), tok]
        x[torch.arange(n, device=DEVICE), pos] = tok
        unmasked[torch.arange(n, device=DEVICE), pos] = True

    return x.cpu(), scores.cpu(), (orders.cpu() if track_order else None)


@torch.no_grad()
def decode_adaptive_threshold(model, n, tau=0.7):
    """
    Adaptive parallel: at each step, decode ALL positions whose max
    softmax probability ≥ τ.  If none qualify, fall back to the single
    most confident position (so it always makes progress).

    Naturally adapts: easy distributions → large batches → few NFE.
    Hard distributions → τ rarely exceeded → ~sequential → more NFE.
    """
    x = torch.full((n, L), MASK_ID, dtype=torch.long, device=DEVICE)
    unmasked = torch.zeros(n, L, dtype=torch.bool, device=DEVICE)
    scores = torch.zeros(n, device=DEVICE)
    n_steps = 0

    while not unmasked.all():
        logits = model(x)
        logits[:, :, MASK_ID] = -float('inf')
        probs = F.softmax(logits, dim=-1)
        conf = probs.max(-1).values  # (n, L)
        conf[unmasked] = -1

        for b in range(n):
            mp = (~unmasked[b]).nonzero(as_tuple=True)[0]
            if len(mp) == 0:
                continue
            # Select positions above threshold
            above = conf[b, mp] >= tau
            if above.any():
                sel = mp[above]
            else:
                # Fallback: single most confident
                sel = mp[conf[b, mp].argmax().unsqueeze(0)]
            for pos in sel:
                pos_logits = logits[b, pos]
                tok = torch.multinomial(
                    F.softmax(pos_logits.unsqueeze(0), dim=-1), 1).item()
                scores[b] += F.log_softmax(pos_logits, dim=-1)[tok]
                x[b, pos] = tok
                unmasked[b, pos] = True
        n_steps += 1

    return x.cpu(), scores.cpu(), n_steps


@torch.no_grad()
def decode_jacobi(model, n, max_iter=20):
    """
    Jacobi-style iteration: predict ALL masked positions at once,
    then re-predict with updated context, repeat until no position
    changes (fixed point) or max_iter reached.

    At convergence, the result is a fixed point of the model's
    conditional predictions — similar quality to sequential confidence
    but potentially much fewer forward passes.
    """
    # Initialize with single-pass prediction (all positions at once)
    x = torch.full((n, L), MASK_ID, dtype=torch.long, device=DEVICE)

    logits = model(x)
    logits[:, :, MASK_ID] = -float('inf')
    # Sample initial tokens
    for pos in range(L):
        tok = torch.multinomial(F.softmax(logits[:, pos], dim=-1), 1).squeeze(-1)
        x[:, pos] = tok

    n_steps = 1  # first pass
    scores = torch.zeros(n, device=DEVICE)

    for it in range(max_iter - 1):
        x_prev = x.clone()
        logits = model(x)
        logits[:, :, MASK_ID] = -float('inf')

        # Re-sample all positions
        for pos in range(L):
            tok = torch.multinomial(F.softmax(logits[:, pos], dim=-1), 1).squeeze(-1)
            x[:, pos] = tok

        n_steps += 1
        # Check convergence: if no tokens changed across batch
        if (x == x_prev).all():
            break

    # Compute final scores
    logits = model(x)
    logits[:, :, MASK_ID] = -float('inf')
    for pos in range(L):
        scores += F.log_softmax(logits[:, pos], dim=-1)[
            torch.arange(n, device=DEVICE), x[:, pos]]

    return x.cpu(), scores.cpu(), n_steps


# ── Metrics ─────────────────────────────────────────

def compute_metrics(samples, scores, dist):
    """
    Compute TV, KL, mode coverage, path score correlation.
    Includes bootstrap confidence intervals and ideal baseline TV.
    """
    all_seqs, p_true = dist.full_distribution()
    K = len(all_seqs)
    n = len(samples)

    # Build empirical distribution
    counts = defaultdict(int)
    for s in samples:
        counts[tuple(s.tolist())] += 1

    q_emp = np.zeros(K)
    for i in range(K):
        q_emp[i] = counts.get(tuple(all_seqs[i].tolist()), 0) / n
    p_np = p_true.numpy()

    # Point estimates
    tv = np.sum(np.abs(p_np - q_emp)) / 2
    kl = np.sum(p_np * np.log(p_np / np.maximum(q_emp, 1e-10) + 1e-30)
                * (p_np > 1e-10))

    # Mode coverage (top 100 modes)
    top_idx = p_true.topk(min(100, K)).indices.numpy()
    coverage = np.mean([q_emp[i] > 0 for i in top_idx])

    # Path score correlation
    true_lp = dist.log_prob(samples).numpy()
    scores_np = scores.numpy()
    valid = np.isfinite(scores_np) & np.isfinite(true_lp)
    sr = spearmanr(scores_np[valid], true_lp[valid])[0] if valid.sum() > 10 else 0

    # ── Bootstrap CI for TV ──
    tv_boots = []
    for _ in range(BOOTSTRAP_K):
        idx = np.random.choice(n, size=n, replace=True)
        boot_counts = defaultdict(int)
        for i in idx:
            boot_counts[tuple(samples[i].tolist())] += 1
        q_boot = np.zeros(K)
        for i in range(K):
            q_boot[i] = boot_counts.get(tuple(all_seqs[i].tolist()), 0) / n
        tv_boots.append(np.sum(np.abs(p_np - q_boot)) / 2)
    tv_boots = np.array(tv_boots)

    # ── Ideal baseline: TV from sampling p_true directly ──
    # (This is the irreducible estimation noise)
    ideal_tvs = []
    for _ in range(5):
        ideal_samples = dist.sample(n)
        ideal_counts = defaultdict(int)
        for s in ideal_samples:
            ideal_counts[tuple(s.tolist())] += 1
        q_ideal = np.zeros(K)
        for i in range(K):
            q_ideal[i] = ideal_counts.get(tuple(all_seqs[i].tolist()), 0) / n
        ideal_tvs.append(np.sum(np.abs(p_np - q_ideal)) / 2)
    baseline_tv = np.mean(ideal_tvs)

    return {
        'tv': float(tv),
        'tv_ci_lo': float(np.percentile(tv_boots, 2.5)),
        'tv_ci_hi': float(np.percentile(tv_boots, 97.5)),
        'tv_baseline': float(baseline_tv),  # irreducible noise
        'tv_excess': float(tv - baseline_tv),  # policy-induced error
        'kl': float(kl),
        'mode_coverage': float(coverage),
        'spearman_r': float(sr),
        'mean_true_lp': float(true_lp.mean()),
        'n_unique': int(len(counts)),
        'support_coverage': float(len(counts) / K),
    }


# ── Mutual information matrix ──────────────────────

def compute_mi(dist):
    all_s, probs = dist.full_distribution()
    MI = np.zeros((L, L))
    marginals = []
    for i in range(L):
        m = np.zeros(V)
        for v in range(V):
            m[v] = probs[all_s[:, i] == v].sum().item()
        marginals.append(m)
    for i in range(L):
        for j in range(i+1, L):
            joint = np.zeros((V, V))
            for vi in range(V):
                for vj in range(V):
                    joint[vi, vj] = probs[
                        (all_s[:,i]==vi) & (all_s[:,j]==vj)].sum().item()
            mi = sum(joint[vi,vj] * np.log(
                    joint[vi,vj]/(marginals[i][vi]*marginals[j][vj]+1e-30)+1e-30)
                for vi in range(V) for vj in range(V) if joint[vi,vj]>1e-30)
            MI[i,j] = MI[j,i] = mi
    return MI


def compute_mean_decode_order(orders_list):
    """
    From decode orders (n_samples × L), compute mean rank for each position.
    orders[i, t] = position decoded at step t for sample i.
    Returns: array of shape (L,) where entry j = mean step at which position j was decoded.
    """
    all_orders = torch.cat(orders_list, dim=0)  # (N, L)
    N = all_orders.shape[0]
    # Convert from "step→position" to "position→step"
    pos_ranks = torch.zeros(N, L)
    for b in range(N):
        for step in range(L):
            pos = all_orders[b, step].item()
            pos_ranks[b, pos] = step
    return pos_ranks.mean(dim=0).numpy()  # (L,)


def compute_mi_order_alignment(mean_ranks, MI):
    """
    Spearman correlation between "total MI of position" and "decode rank".
    High-MI positions should ideally be decoded first (lower rank).
    Returns negative correlation = MI-aligned (good).
    """
    total_mi = MI.sum(axis=1)  # total MI with all other positions
    rho, p = spearmanr(total_mi, mean_ranks)
    return {'rho': float(rho), 'p_value': float(p),
            'total_mi': total_mi.tolist(), 'mean_ranks': mean_ranks.tolist()}


# ── Main ────────────────────────────────────────────

def run():
    print("=" * 70)
    print("  EXP 3: Toy Distribution — Decoding Policy Analysis")
    print("=" * 70)
    mount_drive()
    torch.manual_seed(42); np.random.seed(42)

    distributions = {
        'A_Independent':   NearIndependent(alpha=0.5, seed=42),
        'B_Markov2':       MarkovChain(order=2, sparsity=0.1, seed=42),
        'C_GlobalSum':     GlobalSumConstraint(beta=2.0, alpha=0.5, seed=42),
        'D_Markov+Global': MarkovPlusGlobal(beta=1.5, order=2, sparsity=0.1, seed=42),
    }
    for name, dist in distributions.items():
        print(f"  {name}: H={dist.entropy():.2f} bits")

    print(f"\n  Sampling stats: V^L={V**L:,}, N_SAMPLES={N_SAMPLES:,}, "
          f"ratio={N_SAMPLES/V**L:.1f}x")
    print(f"  Expected TV noise ≈ {np.sqrt(V**L / (2*np.pi*N_SAMPLES)):.4f}")
    print(f"  Bootstrap resamples: {BOOTSTRAP_K}")

    # Compute theoretical optimal loss for each distribution
    print("\n  Computing Bayes-optimal masked loss (may take ~1min)...")
    for name, dist in distributions.items():
        opt_loss = dist.optimal_masked_loss(n_eval=2000)
        uniform_loss = np.log(V)
        print(f"    {name}: optimal_CE={opt_loss:.4f}, "
              f"uniform_CE={uniform_loss:.4f}, "
              f"gap={uniform_loss - opt_loss:.4f}")

    # MI matrices
    fig_mi, axes = plt.subplots(1, 4, figsize=(20, 4))
    for idx, (name, dist) in enumerate(distributions.items()):
        MI = compute_mi(dist)
        im = axes[idx].imshow(MI, cmap='YlOrRd', vmin=0)
        axes[idx].set_title(name, fontsize=10)
        plt.colorbar(im, ax=axes[idx], shrink=0.8)
    fig_mi.suptitle('Mutual Information I(X_i; X_j)', fontsize=13, y=1.02)
    fig_mi.tight_layout()

    # Train
    models, train_hists = {}, {}
    for name, dist in distributions.items():
        print(f"\n▶ Training: {name}")
        model, hist = train_toy_model(dist)
        models[name] = model
        train_hists[name] = hist

    # Evaluate policies (sampling only — greedy is deterministic, TV/KL meaningless)
    seq_policies = ['confidence', 'low_entropy', 'high_entropy', 'margin',
                    'random', 'l2r', 'r2l']
    # New parallel configs: adaptive threshold + jacobi iteration
    adaptive_taus = [0.5, 0.7, 0.9]
    jacobi_iters  = [5, 10, 20]

    all_results = {}
    all_orders = {}  # dist_name → policy → list of order tensors

    for dist_name, dist in distributions.items():
        model = models[dist_name]
        print(f"\n▶ Evaluating: {dist_name}")
        all_results[dist_name] = {}
        all_orders[dist_name] = {}

        # ── Sequential policies ──
        for pol in seq_policies:
            t0 = time.time()
            samples, scores, orders_list = [], [], []
            n_generated = 0
            for start in range(0, N_SAMPLES, BATCH_SIZE):
                bs = min(BATCH_SIZE, N_SAMPLES - start)
                need_order = (n_generated < N_ORDER_SAMPLES)
                s, sc, ords = decode_sequential(model, bs, pol,
                                                track_order=need_order)
                samples.append(s); scores.append(sc)
                if ords is not None:
                    orders_list.append(ords)
                n_generated += bs
            samples = torch.cat(samples); scores = torch.cat(scores)
            m = compute_metrics(samples, scores, dist)
            m['nfe'] = L
            all_results[dist_name][pol] = m
            all_orders[dist_name][pol] = orders_list
            print(f"    {pol:<20} TV={m['tv']:.4f} "
                  f"[{m['tv_ci_lo']:.4f},{m['tv_ci_hi']:.4f}] "
                  f"excess={m['tv_excess']:+.4f} "
                  f"cov={m['support_coverage']:.2%} ({time.time()-t0:.1f}s)")

        # ── Adaptive threshold ──
        for tau in adaptive_taus:
            key = f"adaptive_τ{tau}"
            t0 = time.time()
            samples, scores, total_steps = [], [], 0
            n_batches = 0
            for start in range(0, N_SAMPLES, BATCH_SIZE):
                bs = min(BATCH_SIZE, N_SAMPLES - start)
                s, sc, ns = decode_adaptive_threshold(model, bs, tau=tau)
                samples.append(s); scores.append(sc)
                total_steps += ns; n_batches += 1
            samples = torch.cat(samples); scores = torch.cat(scores)
            m = compute_metrics(samples, scores, dist)
            m['nfe'] = total_steps / max(n_batches, 1)
            all_results[dist_name][key] = m
            print(f"    {key:<20} TV={m['tv']:.4f} "
                  f"excess={m['tv_excess']:+.4f} "
                  f"NFE={m['nfe']:.1f} ({time.time()-t0:.1f}s)")

        # ── Jacobi iteration ──
        for mi in jacobi_iters:
            key = f"jacobi_i{mi}"
            t0 = time.time()
            samples, scores, total_steps = [], [], 0
            n_batches = 0
            for start in range(0, N_SAMPLES, BATCH_SIZE):
                bs = min(BATCH_SIZE, N_SAMPLES - start)
                s, sc, ns = decode_jacobi(model, bs, max_iter=mi)
                samples.append(s); scores.append(sc)
                total_steps += ns; n_batches += 1
            samples = torch.cat(samples); scores = torch.cat(scores)
            m = compute_metrics(samples, scores, dist)
            m['nfe'] = total_steps / max(n_batches, 1)
            all_results[dist_name][key] = m
            print(f"    {key:<20} TV={m['tv']:.4f} "
                  f"excess={m['tv_excess']:+.4f} "
                  f"NFE={m['nfe']:.1f} ({time.time()-t0:.1f}s)")

    # ── Compute MI matrices and alignment scores ──
    mi_matrices = {}
    mi_alignments = {}
    for dist_name, dist in distributions.items():
        MI = compute_mi(dist)
        mi_matrices[dist_name] = MI
        mi_alignments[dist_name] = {}
        for pol in seq_policies:
            if pol in all_orders[dist_name] and all_orders[dist_name][pol]:
                mean_ranks = compute_mean_decode_order(all_orders[dist_name][pol])
                alignment = compute_mi_order_alignment(mean_ranks, MI)
                mi_alignments[dist_name][pol] = alignment

    # ── Collect all policy names ──
    all_pol_names = (seq_policies
                     + [f"adaptive_τ{t}" for t in adaptive_taus]
                     + [f"jacobi_i{m}" for m in jacobi_iters])

    # ── Visualisation ──
    figs = {'mi_matrices': fig_mi}
    dist_names = list(distributions.keys())

    # Training curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for name, h in train_hists.items():
        axes[0].plot(h['loss'], label=name, alpha=0.8)
        axes[1].plot(h['cond_acc'], label=name, alpha=0.8)
    axes[0].set_title('Training Loss'); axes[0].set_xlabel('Epoch')
    axes[1].set_title('Conditional Accuracy'); axes[1].set_xlabel('Epoch')
    for ax in axes: ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.tight_layout(); figs['training_curves'] = fig

    # Print baseline TVs
    print("\n  Baseline TV (from sampling p_true directly):")
    for dn in dist_names:
        bl = all_results[dn][seq_policies[0]]['tv_baseline']
        print(f"    {dn}: {bl:.4f}")

    # ── FIG 1: TV excess heatmap (ALL policies) ──
    fig, ax = plt.subplots(figsize=(11, 8))
    data_excess = np.full((len(all_pol_names), len(dist_names)), np.nan)
    for j, d in enumerate(dist_names):
        for i, p in enumerate(all_pol_names):
            if p in all_results.get(d, {}):
                data_excess[i, j] = all_results[d][p]['tv_excess']
    im = ax.imshow(data_excess, cmap='RdYlGn_r', aspect='auto')
    ax.set_xticks(range(len(dist_names)))
    ax.set_xticklabels(dist_names, fontsize=8, rotation=20)
    ax.set_yticks(range(len(all_pol_names)))
    ax.set_yticklabels(all_pol_names, fontsize=8)
    for i in range(data_excess.shape[0]):
        for j in range(data_excess.shape[1]):
            if not np.isnan(data_excess[i, j]):
                ax.text(j, i, f"{data_excess[i,j]:+.3f}", ha='center',
                        va='center', fontsize=7,
                        color='white' if data_excess[i,j] > np.nanmedian(
                            data_excess[~np.isnan(data_excess)]) else 'black')
    # Draw separators
    ax.axhline(y=len(seq_policies)-0.5, color='blue', lw=1.5, ls='--')
    ax.axhline(y=len(seq_policies)+len(adaptive_taus)-0.5,
               color='orange', lw=1.5, ls='--')
    plt.colorbar(im, ax=ax, shrink=0.7)
    ax.set_title('Excess TV: Sequential vs Adaptive vs Jacobi (↓ better)',
                 fontsize=12)
    fig.tight_layout(); figs['tv_excess_heatmap'] = fig

    # ── FIG 2: NFE vs TV scatter (the meaningful Pareto) ──
    fig, axes = plt.subplots(1, len(dist_names), figsize=(5*len(dist_names), 4.5))
    if len(dist_names) == 1: axes = [axes]
    for col, dn in enumerate(dist_names):
        ax = axes[col]
        dr = all_results.get(dn, {})
        bl = dr[seq_policies[0]]['tv_baseline']
        ax.axhline(y=bl, color='gray', ls=':', alpha=0.5, label='baseline')

        # Sequential: all at NFE=L
        for p in seq_policies:
            if p in dr:
                c = '#2ecc71' if p == 'confidence' else '#95a5a6'
                ax.scatter(dr[p]['nfe'], dr[p]['tv'], marker='x', s=50,
                           color=c, zorder=5)
                ax.annotate(p, (dr[p]['nfe'], dr[p]['tv']), fontsize=5,
                            textcoords="offset points", xytext=(3, 3))

        # Adaptive threshold: different τ → different NFE
        a_nfes = [dr[f"adaptive_τ{t}"]['nfe'] for t in adaptive_taus
                  if f"adaptive_τ{t}" in dr]
        a_tvs  = [dr[f"adaptive_τ{t}"]['tv'] for t in adaptive_taus
                  if f"adaptive_τ{t}" in dr]
        if a_nfes:
            ax.plot(a_nfes, a_tvs, '-o', color='#e74c3c', label='adaptive',
                    markersize=6, zorder=10)
            for t, nf, tv in zip(adaptive_taus, a_nfes, a_tvs):
                ax.annotate(f'τ={t}', (nf, tv), fontsize=5,
                            textcoords="offset points", xytext=(4, 4))

        # Jacobi: different max_iter → different NFE
        j_nfes = [dr[f"jacobi_i{m}"]['nfe'] for m in jacobi_iters
                  if f"jacobi_i{m}" in dr]
        j_tvs  = [dr[f"jacobi_i{m}"]['tv'] for m in jacobi_iters
                  if f"jacobi_i{m}" in dr]
        if j_nfes:
            ax.plot(j_nfes, j_tvs, '-s', color='#3498db', label='jacobi',
                    markersize=6, zorder=10)
            for m, nf, tv in zip(jacobi_iters, j_nfes, j_tvs):
                ax.annotate(f'i={m}', (nf, tv), fontsize=5,
                            textcoords="offset points", xytext=(4, 4))

        ax.set_title(dn, fontsize=10)
        ax.set_xlabel('NFE (forward passes)')
        ax.set_ylabel('TV Distance')
        ax.grid(alpha=0.3)
        if col == 0: ax.legend(fontsize=7)
    fig.suptitle('Speed vs Quality: NFE vs TV', fontsize=13, y=1.02)
    fig.tight_layout(); figs['nfe_vs_tv'] = fig

    # ── FIG 3: Decode order heatmap (confidence only, per distribution) ──
    adaptive_pols = [p for p in seq_policies if p not in ('l2r', 'r2l')]
    fig, axes = plt.subplots(len(dist_names), len(adaptive_pols),
                             figsize=(3.5*len(adaptive_pols),
                                      2.5*len(dist_names)))
    if len(dist_names) == 1: axes = [axes]
    for row, dn in enumerate(dist_names):
        for col_idx, pol in enumerate(adaptive_pols):
            ax = axes[row][col_idx] if len(dist_names) > 1 else axes[col_idx]
            if (pol in all_orders[dn] and all_orders[dn][pol]):
                mr = compute_mean_decode_order(all_orders[dn][pol])
                ax.bar(range(L), mr, color='#3498db', alpha=0.85)
            ax.set_xticks(range(L))
            ax.set_title(f'{pol}' if row == 0 else '', fontsize=8)
            if col_idx == 0:
                ax.set_ylabel(dn.split('_')[-1], fontsize=8)
            ax.tick_params(labelsize=6)
            ax.grid(axis='y', alpha=0.3)
    fig.suptitle('Decode Order: Mean Step per Position (lower = earlier)',
                 fontsize=11, y=1.02)
    fig.tight_layout(); figs['decode_order'] = fig

    # ── FIG 4: MI-Order Alignment ──
    fig, ax = plt.subplots(figsize=(10, 5))
    align_data = np.full((len(seq_policies), len(dist_names)), np.nan)
    for j, dn in enumerate(dist_names):
        for i, pol in enumerate(seq_policies):
            if pol in mi_alignments.get(dn, {}):
                align_data[i, j] = mi_alignments[dn][pol]['rho']
    im = ax.imshow(align_data, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    ax.set_xticks(range(len(dist_names)))
    ax.set_xticklabels(dist_names, fontsize=8, rotation=20)
    ax.set_yticks(range(len(seq_policies)))
    ax.set_yticklabels(seq_policies, fontsize=9)
    for i in range(align_data.shape[0]):
        for j in range(align_data.shape[1]):
            if not np.isnan(align_data[i, j]):
                ax.text(j, i, f"{align_data[i,j]:.2f}", ha='center',
                        va='center', fontsize=8)
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title('MI-Order Alignment: ρ(total_MI, decode_rank)\n'
                 'Negative = high-MI positions decoded first (good)', fontsize=11)
    fig.tight_layout(); figs['mi_alignment'] = fig

    # ── FIG 5: Policy Ranking Table ──
    fig, ax = plt.subplots(figsize=(11, 7))
    rank_data = np.full((len(all_pol_names), len(dist_names)), np.nan)
    for j, dn in enumerate(dist_names):
        excesses = [(p, all_results[dn].get(p, {}).get('tv_excess', 99))
                    for p in all_pol_names]
        excesses.sort(key=lambda x: x[1])
        for rank, (p, _) in enumerate(excesses):
            i = all_pol_names.index(p)
            rank_data[i, j] = rank + 1
    im = ax.imshow(rank_data, cmap='YlGn_r', aspect='auto',
                   vmin=1, vmax=len(all_pol_names))
    ax.set_xticks(range(len(dist_names)))
    ax.set_xticklabels(dist_names, fontsize=8, rotation=20)
    ax.set_yticks(range(len(all_pol_names)))
    ax.set_yticklabels(all_pol_names, fontsize=8)
    for i in range(rank_data.shape[0]):
        for j in range(rank_data.shape[1]):
            if not np.isnan(rank_data[i, j]):
                ax.text(j, i, f"{int(rank_data[i,j])}", ha='center',
                        va='center', fontsize=9, fontweight='bold')
    ax.axhline(y=len(seq_policies)-0.5, color='blue', lw=1.5, ls='--')
    ax.axhline(y=len(seq_policies)+len(adaptive_taus)-0.5,
               color='orange', lw=1.5, ls='--')
    plt.colorbar(im, ax=ax, shrink=0.7, label='Rank (1=best)')
    ax.set_title('Policy Ranking by Excess TV (1=best)\n'
                 'Blue line: seq|adaptive, Orange: adaptive|jacobi', fontsize=11)
    fig.tight_layout(); figs['policy_ranking'] = fig

    # ── FIG 6: Spearman correlation heatmap ──
    corr_data = np.full((len(all_pol_names), len(dist_names)), np.nan)
    for j, d in enumerate(dist_names):
        for i, p in enumerate(all_pol_names):
            if p in all_results.get(d, {}):
                corr_data[i, j] = all_results[d][p]['spearman_r']
    fig, ax = plt.subplots(figsize=(11, 7))
    im = ax.imshow(corr_data, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)
    ax.set_xticks(range(len(dist_names)))
    ax.set_xticklabels(dist_names, fontsize=8, rotation=20)
    ax.set_yticks(range(len(all_pol_names)))
    ax.set_yticklabels(all_pol_names, fontsize=8)
    for i in range(corr_data.shape[0]):
        for j in range(corr_data.shape[1]):
            if not np.isnan(corr_data[i, j]):
                ax.text(j, i, f"{corr_data[i,j]:.2f}", ha='center',
                        va='center', fontsize=7)
    plt.colorbar(im, ax=ax, shrink=0.7)
    ax.set_title('Spearman Corr(path_score, log p_true)', fontsize=12)
    fig.tight_layout(); figs['correlation_heatmap'] = fig

    # JSON-safe results
    json_results = {}
    for d in all_results:
        json_results[d] = {}
        for p in all_results[d]:
            json_results[d][p] = {
                k: float(v) for k, v in all_results[d][p].items()}
    json_results['mi_alignments'] = {}
    for dn in mi_alignments:
        json_results['mi_alignments'][dn] = {}
        for pol in mi_alignments[dn]:
            a = mi_alignments[dn][pol]
            json_results['mi_alignments'][dn][pol] = {
                'rho': a['rho'], 'p_value': a['p_value']}

    save_results(EXP_NAME, json_results, figures=figs)

    # ── Summary ──
    print("\n" + "=" * 70)
    print("SUMMARY")
    print(f"  (N_SAMPLES={N_SAMPLES:,}, V^L={V**L:,}, "
          f"bootstrap_k={BOOTSTRAP_K})")
    print("=" * 70)

    # Combined table: excess TV + NFE
    header = f"{'Policy':<22} {'NFE':>5}"
    for dn in dist_names:
        header += f"  {dn:>14}"
    print(f"\n{header}")
    print("-" * len(header))
    for pol in all_pol_names:
        nfe = all_results[dist_names[0]].get(pol, {}).get('nfe', float('nan'))
        row = f"{pol:<22} {nfe:>5.1f}"
        for dn in dist_names:
            exc = all_results[dn].get(pol, {}).get('tv_excess', float('nan'))
            row += f"  {exc:>+13.4f}"
        print(row)

    # MI alignment summary
    print(f"\nMI-Order Alignment (ρ, negative = MI-aligned):")
    for dn in dist_names:
        best_pol = min(mi_alignments.get(dn, {}),
                       key=lambda p: mi_alignments[dn][p]['rho'],
                       default='N/A')
        if best_pol != 'N/A':
            a = mi_alignments[dn][best_pol]
            print(f"  {dn}: best={best_pol} ρ={a['rho']:+.3f}")

    # NFE efficiency summary
    print(f"\nNFE Efficiency (per distribution):")
    for dn in dist_names:
        conf_tv = all_results[dn].get('confidence', {}).get('tv_excess', 99)
        print(f"  {dn} (confidence baseline: excess={conf_tv:+.4f}, NFE=8):")
        for key in ([f"adaptive_τ{t}" for t in adaptive_taus]
                    + [f"jacobi_i{m}" for m in jacobi_iters]):
            if key in all_results[dn]:
                r = all_results[dn][key]
                delta = r['tv_excess'] - conf_tv
                speedup = L / r['nfe'] if r['nfe'] > 0 else 0
                print(f"    {key:<22} NFE={r['nfe']:.1f} "
                      f"({speedup:.1f}× faster) "
                      f"Δexcess={delta:+.4f}")

    plt.show()
    return all_results


if __name__ == '__main__':
    run()
