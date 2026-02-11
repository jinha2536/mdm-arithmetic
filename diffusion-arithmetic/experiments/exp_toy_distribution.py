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

  Decoding: sampling (not greedy — distributional metrics require stochasticity).

  Note on module 3's relationship to modules 1-2:
    Modules 1-2 use greedy (argmax) for exact match — appropriate for
    deterministic reasoning tasks. This module uses sampling because
    distributional metrics (TV, KL, mode coverage) require stochastic
    samples to be meaningful. Greedy produces one deterministic sequence,
    making distributional comparison impossible.
    Results here explain HOW policies behave on different dependency
    structures, but cannot directly predict arithmetic/tree performance.
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

@torch.no_grad()
def decode_sequential(model, n, policy='confidence'):
    x = torch.full((n, L), MASK_ID, dtype=torch.long, device=DEVICE)
    unmasked = torch.zeros(n, L, dtype=torch.bool, device=DEVICE)
    scores = torch.zeros(n, device=DEVICE)

    for t in range(L):
        logits = model(x)
        # exclude MASK token from predictions
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

        lp = logits[torch.arange(n, device=DEVICE), pos]
        tok = torch.multinomial(F.softmax(lp, dim=-1), 1).squeeze(-1)
        scores += F.log_softmax(lp, dim=-1)[torch.arange(n, device=DEVICE), tok]
        x[torch.arange(n, device=DEVICE), pos] = tok
        unmasked[torch.arange(n, device=DEVICE), pos] = True

    return x.cpu(), scores.cpu()


@torch.no_grad()
def decode_parallel(model, n, k, policy='parallel_random'):
    x = torch.full((n, L), MASK_ID, dtype=torch.long, device=DEVICE)
    unmasked = torch.zeros(n, L, dtype=torch.bool, device=DEVICE)
    scores = torch.zeros(n, device=DEVICE)
    n_steps = 0

    while not unmasked.all():
        logits = model(x)
        logits[:, :, MASK_ID] = -float('inf')
        probs = F.softmax(logits, dim=-1)
        for b in range(n):
            mp = (~unmasked[b]).nonzero(as_tuple=True)[0]
            if len(mp) == 0: continue
            ak = min(k, len(mp))
            if policy == 'parallel_random':
                sel = mp[torch.randperm(len(mp), device=DEVICE)[:ak]]
            elif policy == 'parallel_confidence':
                sel = mp[probs[b, mp].max(-1).values.topk(ak).indices]
            elif policy == 'parallel_low_dep':
                ln = F.normalize(logits[b, mp], dim=-1)
                conf = probs[b, mp].max(-1).values
                chosen = [conf.argmax().item()]
                for _ in range(ak-1):
                    sims = (ln @ ln[chosen].T).max(-1).values
                    for c in chosen: sims[c] = float('inf')
                    chosen.append(sims.argmin().item())
                sel = mp[torch.tensor(chosen, device=DEVICE)]
            else:
                raise ValueError(policy)
            for pos in sel:
                pos_logits = logits[b, pos]  # MASK already excluded above
                tok = torch.multinomial(
                    F.softmax(pos_logits.unsqueeze(0), dim=-1), 1).item()
                scores[b] += F.log_softmax(pos_logits, dim=-1)[tok]
                x[b, pos] = tok; unmasked[b, pos] = True
        n_steps += 1

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
    par_configs = [('parallel_random', 2), ('parallel_confidence', 2),
                   ('parallel_low_dep', 2), ('parallel_random', 4),
                   ('parallel_confidence', 4)]

    all_results = {}
    for dist_name, dist in distributions.items():
        model = models[dist_name]
        print(f"\n▶ Evaluating: {dist_name}")
        all_results[dist_name] = {}

        for pol in seq_policies:
            t0 = time.time()
            samples, scores = [], []
            for start in range(0, N_SAMPLES, BATCH_SIZE):
                bs = min(BATCH_SIZE, N_SAMPLES - start)
                s, sc = decode_sequential(model, bs, pol)
                samples.append(s); scores.append(sc)
            samples = torch.cat(samples); scores = torch.cat(scores)
            m = compute_metrics(samples, scores, dist)
            m['nfe'] = L
            all_results[dist_name][pol] = m
            print(f"    {pol:<20} TV={m['tv']:.4f} "
                  f"[{m['tv_ci_lo']:.4f},{m['tv_ci_hi']:.4f}] "
                  f"excess={m['tv_excess']:+.4f} "
                  f"cov={m['support_coverage']:.2%} ({time.time()-t0:.1f}s)")

        for pol, k in par_configs:
            key = f"{pol}_k{k}"
            t0 = time.time()
            samples, scores, total_steps = [], [], 0
            for start in range(0, N_SAMPLES, BATCH_SIZE):
                bs = min(BATCH_SIZE, N_SAMPLES - start)
                s, sc, ns = decode_parallel(model, bs, k, pol)
                samples.append(s); scores.append(sc); total_steps += ns
            samples = torch.cat(samples); scores = torch.cat(scores)
            m = compute_metrics(samples, scores, dist)
            m['nfe'] = total_steps / max(1, N_SAMPLES // BATCH_SIZE)
            all_results[dist_name][key] = m
            print(f"    {key:<20} TV={m['tv']:.4f} "
                  f"excess={m['tv_excess']:+.4f} "
                  f"NFE={m['nfe']:.1f} ({time.time()-t0:.1f}s)")

    # ── Visualisation ──
    figs = {'mi_matrices': fig_mi}

    # Training curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for name, h in train_hists.items():
        axes[0].plot(h['loss'], label=name, alpha=0.8)
        axes[1].plot(h['cond_acc'], label=name, alpha=0.8)
    axes[0].set_title('Training Loss'); axes[0].set_xlabel('Epoch')
    axes[1].set_title('Conditional Accuracy'); axes[1].set_xlabel('Epoch')
    for ax in axes: ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.tight_layout(); figs['training_curves'] = fig

    dist_names = list(distributions.keys())
    pol_names = seq_policies + [f"{p}_k{k}" for p, k in par_configs]

    # Print baseline TVs (irreducible noise)
    print("\n  Baseline TV (from sampling p_true directly):")
    for dn in dist_names:
        first_pol = seq_policies[0]
        bl = all_results[dn][first_pol]['tv_baseline']
        print(f"    {dn}: {bl:.4f}")

    # TV excess heatmap (raw TV minus baseline — shows actual policy error)
    fig, axes = plt.subplots(1, 2, figsize=(22, 7))

    # Left: raw TV
    data_raw = np.full((len(pol_names), len(dist_names)), np.nan)
    for j, d in enumerate(dist_names):
        for i, p in enumerate(pol_names):
            if p in all_results.get(d, {}):
                data_raw[i, j] = all_results[d][p]['tv']
    ax = axes[0]
    im = ax.imshow(data_raw, cmap='RdYlGn_r', aspect='auto')
    ax.set_xticks(range(len(dist_names)))
    ax.set_xticklabels(dist_names, fontsize=7, rotation=20)
    ax.set_yticks(range(len(pol_names)))
    ax.set_yticklabels(pol_names, fontsize=8)
    for i in range(data_raw.shape[0]):
        for j in range(data_raw.shape[1]):
            if not np.isnan(data_raw[i, j]):
                ax.text(j, i, f"{data_raw[i,j]:.3f}", ha='center',
                        va='center', fontsize=6,
                        color='white' if data_raw[i,j] > np.nanmedian(data_raw)
                        else 'black')
    ax.axhline(y=len(seq_policies)-0.5, color='blue', lw=2, ls='--')
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title('Raw TV Distance (includes estimation noise)', fontsize=10)

    # Right: excess TV (= TV - baseline)
    data_excess = np.full((len(pol_names), len(dist_names)), np.nan)
    for j, d in enumerate(dist_names):
        for i, p in enumerate(pol_names):
            if p in all_results.get(d, {}):
                data_excess[i, j] = all_results[d][p]['tv_excess']
    ax = axes[1]
    vmax = max(0.01, np.nanmax(np.abs(data_excess)))
    im = ax.imshow(data_excess, cmap='RdYlGn_r', aspect='auto',
                   vmin=-vmax*0.1, vmax=vmax)
    ax.set_xticks(range(len(dist_names)))
    ax.set_xticklabels(dist_names, fontsize=7, rotation=20)
    ax.set_yticks(range(len(pol_names)))
    ax.set_yticklabels(pol_names, fontsize=8)
    for i in range(data_excess.shape[0]):
        for j in range(data_excess.shape[1]):
            if not np.isnan(data_excess[i, j]):
                ax.text(j, i, f"{data_excess[i,j]:+.3f}", ha='center',
                        va='center', fontsize=6,
                        color='white' if data_excess[i,j] > np.nanmedian(
                            data_excess[~np.isnan(data_excess)]) else 'black')
    ax.axhline(y=len(seq_policies)-0.5, color='blue', lw=2, ls='--')
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title('Excess TV (policy error above baseline ↓ better)', fontsize=10)
    fig.tight_layout(); figs['tv_heatmap'] = fig

    # Spearman correlation heatmap
    corr_data = np.full((len(seq_policies), len(dist_names)), np.nan)
    for j, d in enumerate(dist_names):
        for i, p in enumerate(seq_policies):
            if p in all_results.get(d, {}):
                corr_data[i, j] = all_results[d][p]['spearman_r']
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(corr_data, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)
    ax.set_xticks(range(len(dist_names)))
    ax.set_xticklabels(dist_names, fontsize=8, rotation=20)
    ax.set_yticks(range(len(seq_policies)))
    ax.set_yticklabels(seq_policies, fontsize=8)
    for i in range(corr_data.shape[0]):
        for j in range(corr_data.shape[1]):
            if not np.isnan(corr_data[i, j]):
                ax.text(j, i, f"{corr_data[i,j]:.2f}", ha='center',
                        va='center', fontsize=8)
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title('Spearman Corr(path_score, log p_true)', fontsize=12)
    fig.tight_layout(); figs['correlation_heatmap'] = fig

    # Pareto: NFE vs TV per distribution
    fig, axes = plt.subplots(1, len(dist_names), figsize=(20, 4.5))
    color_map = {'random': 'tab:blue', 'confidence': 'tab:orange',
                 'low_dep': 'tab:green'}
    for col, dn in enumerate(dist_names):
        ax = axes[col]
        dr = all_results.get(dn, {})
        for p in seq_policies:
            if p in dr:
                ax.scatter(dr[p]['nfe'], dr[p]['tv'], marker='x', s=50,
                           color='gray', zorder=5)
                ax.annotate(p, (dr[p]['nfe'], dr[p]['tv']), fontsize=5,
                            textcoords="offset points", xytext=(3, 3))
        for ptype in ['parallel_random', 'parallel_confidence', 'parallel_low_dep']:
            nfes, tvs = [], []
            for k in [2, 4]:
                key = f"{ptype}_k{k}"
                if key in dr:
                    nfes.append(dr[key]['nfe']); tvs.append(dr[key]['tv'])
            if nfes:
                short = ptype.split('_')[-1]
                ax.plot(nfes, tvs, '-o', color=color_map.get(short, 'gray'),
                        label=short, markersize=5)
        ax.set_title(dn, fontsize=10); ax.set_xlabel('NFE')
        ax.set_ylabel('TV'); ax.grid(alpha=0.3)
        if col == 0: ax.legend(fontsize=7)
    fig.suptitle('Quality-Speed Pareto (NFE vs TV)', fontsize=13, y=1.02)
    fig.tight_layout(); figs['pareto'] = fig

    # JSON-safe
    json_results = {}
    for d in all_results:
        json_results[d] = {}
        for p in all_results[d]:
            json_results[d][p] = {
                k: float(v) for k, v in all_results[d][p].items()}

    save_results(EXP_NAME, json_results, figures=figs)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Best Sequential Policy per Distribution")
    print(f"  (N_SAMPLES={N_SAMPLES:,}, V^L={V**L:,}, "
          f"bootstrap_k={BOOTSTRAP_K})")
    print("=" * 70)
    for dn in dist_names:
        bl = all_results[dn][seq_policies[0]]['tv_baseline']
        seq = {p: all_results[dn].get(p, {}).get('tv_excess', 99)
               for p in seq_policies}
        best = min(seq, key=seq.get)
        worst = max(seq, key=seq.get)
        print(f"  {dn}:")
        print(f"    baseline TV = {bl:.4f} (irreducible estimation noise)")
        print(f"    best  = {best:<15} excess TV = {seq[best]:+.4f}")
        print(f"    worst = {worst:<15} excess TV = {seq[worst]:+.4f}")
        print(f"    gap   = {seq[worst]-seq[best]:.4f}")

    plt.show()
    return all_results


if __name__ == '__main__':
    run()
