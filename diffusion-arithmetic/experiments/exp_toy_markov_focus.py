"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Experiment 3b — Learned Conditional Analysis + pass@k
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Colab:  %run experiments/exp_toy_markov_focus.py

  Two analyses on two distributions (V=4, L=8, fully enumerable):

  ── Part 1: Conditional Analysis ─────────────────────
  Directly compare p_θ(x_t | x_S=s) vs p_true(x_t | x_S=s)
  for diverse conditioning patterns (causal, anticausal, gap, etc.)

  Metrics: mode_accuracy, true_answer_prob, KL, error_type

  ── Part 2: pass@k Analysis ──────────────────────────
  Fix prefix, generate k samples with different decode policies,
  measure pass@k and sample diversity.

  Key insight: even a well-calibrated model produces very different
  pass@k curves depending on decode order, because ORDER determines
  WHERE branching occurs in the sample tree.

  Distributions:
    A: HardMarkov       — order=2, p_peak=0.75 (moderate stochasticity)
    B: HardGlobalSum    — hard mod-V + peaked position bias (α=0.3)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                if '__file__' in dir() else '.')

import time
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from itertools import product as iter_product
from math import comb

from experiments.exp_toy_distribution import (
    V, L, MASK_ID, TOTAL_V,
    N_TRAIN, MAX_EPOCHS, PATIENCE_EPOCHS, BATCH_SIZE, LR,
    D_MODEL, N_HEADS, N_LAYERS,
    ToyDistribution, HardMarkov,
    train_toy_model, compute_mi,
)
from core.train_utils import mount_drive, save_results, DEVICE

EXP_NAME = 'exp_toy_markov_focus'


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Distributions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class HardGlobalSum(ToyDistribution):
    """
    p(x) ∝ 1[Σx_i ≡ 0 (mod V)] · ∏ b_i(x_i)

    Position-wise bias b_i ~ Dir(α) with α=0.3 (very peaked).
    Mod-V constraint creates symmetric global dependency.
    """
    def __init__(self, alpha=0.3, seed=42):
        super().__init__()
        rng = np.random.RandomState(seed)
        self.base_lp = torch.tensor(
            np.log(rng.dirichlet([alpha] * V, size=L) + 1e-30),
            dtype=torch.float32)

    def log_prob(self, x):
        x = x.reshape(-1, L)
        lp = sum(self.base_lp[i][x[:, i]] for i in range(L))
        valid = (x.sum(-1) % V == 0)
        return torch.where(valid, lp, torch.tensor(-70.0))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Part 1: Conditional Analysis
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@torch.no_grad()
def evaluate_conditional(model, dist, S_positions, target_pos,
                         prune_threshold=1e-10, batch_size=2048):
    """
    For each context s ∈ V^|S|, compare p_θ(x_t|s) vs p_true(x_t|s).
    All metrics are p(x_S=s)-weighted.
    """
    all_seqs, p_true = dist.full_distribution()
    p_np = p_true.numpy()
    S_np = all_seqs.numpy()
    eps = 1e-15

    if len(S_positions) == 0:
        contexts = [()]
    else:
        contexts = list(iter_product(range(V), repeat=len(S_positions)))

    # Marginal mode for error classification
    marginal_dist = np.zeros(V)
    for v in range(V):
        marginal_dist[v] = p_np[S_np[:, target_pos] == v].sum()
    marginal_dist /= marginal_dist.sum() + eps
    marginal_mode = int(np.argmax(marginal_dist))

    valid_contexts = []
    for s_val in contexts:
        mask = np.ones(len(p_np), dtype=bool)
        for pos, val in zip(S_positions, s_val):
            mask &= (S_np[:, pos] == val)
        p_ctx = float(p_np[mask].sum())
        if p_ctx < prune_threshold:
            continue
        p_true_t = np.zeros(V)
        for v in range(V):
            p_true_t[v] = p_np[mask & (S_np[:, target_pos] == v)].sum()
        total = p_true_t.sum()
        if total < eps:
            continue
        p_true_t /= total
        valid_contexts.append((s_val, p_ctx, p_true_t))

    if not valid_contexts:
        return None

    inputs = []
    for s_val, _, _ in valid_contexts:
        x_in = torch.full((L,), MASK_ID, dtype=torch.long)
        for pos, val in zip(S_positions, s_val):
            x_in[pos] = val
        inputs.append(x_in)
    inputs = torch.stack(inputs).to(DEVICE)

    all_model_probs = []
    for i in range(0, len(inputs), batch_size):
        batch = inputs[i:i + batch_size]
        logits = model(batch)
        logits[:, :, MASK_ID] = -float('inf')
        probs = F.softmax(logits[:, target_pos, :V], dim=-1)
        all_model_probs.append(probs.cpu())
    all_model_probs = torch.cat(all_model_probs, dim=0).numpy()

    total_kl = total_h_true = total_h_model = total_w = 0.0
    weighted_mode_acc = weighted_true_prob = 0.0
    n_correct = n_marginal_fb = n_wrong_resp = 0

    for idx, (s_val, p_ctx, p_true_t) in enumerate(valid_contexts):
        p_m = all_model_probs[idx]
        true_mode = int(np.argmax(p_true_t))
        model_mode = int(np.argmax(p_m))
        correct = (true_mode == model_mode)

        kl = sum(p_true_t[v] * np.log(p_true_t[v] / (p_m[v] + eps))
                 for v in range(V) if p_true_t[v] > eps)
        h_true = -sum(p_true_t[v] * np.log(p_true_t[v] + eps)
                      for v in range(V) if p_true_t[v] > eps)
        h_model = -sum(p_m[v] * np.log(p_m[v] + eps)
                       for v in range(V) if p_m[v] > eps)

        total_kl += p_ctx * kl
        total_h_true += p_ctx * h_true
        total_h_model += p_ctx * h_model
        total_w += p_ctx
        weighted_mode_acc += p_ctx * float(correct)
        weighted_true_prob += p_ctx * float(p_m[true_mode])

        if correct:
            n_correct += 1
        elif model_mode == marginal_mode:
            n_marginal_fb += 1
        else:
            n_wrong_resp += 1

    w = max(total_w, eps)
    n_total = len(valid_contexts)
    n_wrong = n_marginal_fb + n_wrong_resp

    return {
        'kl_forward': float(total_kl / w),
        'true_entropy': float(total_h_true / w),
        'model_entropy': float(total_h_model / w),
        'entropy_gap': float((total_h_model - total_h_true) / w),
        'mode_accuracy': float(weighted_mode_acc / w),
        'true_answer_prob': float(weighted_true_prob / w),
        'n_correct': n_correct,
        'n_marginal_fb': n_marginal_fb,
        'n_wrong_resp': n_wrong_resp,
        'n_contexts': n_total,
    }


def make_all_patterns():
    P = {}
    P['marginal'] = [([], t) for t in range(L)]
    P['single_left'] = [([t - 1], t) for t in range(1, L)]
    P['single_right'] = [([t + 1], t) for t in range(L - 1)]
    P['causal_w2'] = [([t - 2, t - 1], t) for t in range(2, L)]
    P['anticausal_w2'] = [([t + 1, t + 2], t) for t in range(L - 2)]
    P['sandwich'] = [([t - 1, t + 1], t) for t in range(1, L - 1)]
    P['causal_w3'] = [([t - 3, t - 2, t - 1], t) for t in range(3, L)]
    P['causal_gap1'] = [([t - 3, t - 1], t) for t in range(3, L)]
    P['causal_gap2'] = [([t - 4, t - 1], t) for t in range(4, L)]
    P['l2r_progressive'] = [(list(range(t)), t) for t in range(1, L)]
    P['r2l_progressive'] = [(list(range(t + 1, L)), t)
                             for t in range(L - 2, -1, -1)]
    P['distant'] = [([0, L - 1], t) for t in range(1, L - 1)]
    P['leave_one_out'] = [
        (list(range(t)) + list(range(t + 1, L)), t) for t in range(L)]
    return P


def avg_metric(cr_list, key):
    vals = [r[key] for r in cr_list if r is not None]
    return np.mean(vals) if vals else float('nan')


def error_counts(cr_list):
    nc = sum(r['n_correct'] for r in cr_list if r)
    nf = sum(r['n_marginal_fb'] for r in cr_list if r)
    nw = sum(r['n_wrong_resp'] for r in cr_list if r)
    return nc, nf, nw


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Part 2: pass@k
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@torch.no_grad()
def decode_with_prefix(model, n, prefix_pos, prefix_vals, policy):
    """
    Decode with some positions pre-filled.
    Remaining positions filled according to policy with temperature sampling.
    """
    x = torch.full((n, L), MASK_ID, dtype=torch.long, device=DEVICE)
    unmasked = torch.zeros(n, L, dtype=torch.bool, device=DEVICE)

    for pos, val in zip(prefix_pos, prefix_vals):
        x[:, pos] = val
        unmasked[:, pos] = True

    n_remaining = L - len(prefix_pos)
    for step in range(n_remaining):
        logits = model(x)
        logits[:, :, MASK_ID] = -float('inf')
        probs = F.softmax(logits, dim=-1)

        if policy == 'confidence':
            sc = probs.max(-1).values.clone(); sc[unmasked] = -1
            pos = sc.argmax(-1)
        elif policy == 'low_entropy':
            e = -(probs * (probs + 1e-10).log()).sum(-1)
            e[unmasked] = 1e9; pos = e.argmin(-1)
        elif policy == 'high_entropy':
            e = -(probs * (probs + 1e-10).log()).sum(-1)
            e[unmasked] = -1e9; pos = e.argmax(-1)
        elif policy == 'random':
            rand_sc = torch.rand(n, L, device=DEVICE)
            rand_sc[unmasked] = -1; pos = rand_sc.argmax(-1)
        elif policy == 'l2r':
            remaining = (~unmasked[0]).nonzero(as_tuple=True)[0]
            pos = remaining[0].expand(n)
        elif policy == 'r2l':
            remaining = (~unmasked[0]).nonzero(as_tuple=True)[0]
            pos = remaining[-1].expand(n)
        else:
            raise ValueError(policy)

        lp = logits[torch.arange(n, device=DEVICE), pos]
        tok = torch.multinomial(F.softmax(lp, dim=-1), 1).squeeze(-1)
        x[torch.arange(n, device=DEVICE), pos] = tok
        unmasked[torch.arange(n, device=DEVICE), pos] = True

    result = x.cpu()
    result = result.clamp(max=V - 1)
    return result


def compute_target_markov(dist, prefix_vals):
    """
    Given prefix (x0, x1), compute the mode sequence by greedily
    following the preferred token at each step.
    """
    order = dist.k
    seq = list(prefix_vals)
    for t in range(order, L):
        ctx_idx = sum(seq[t - order + i] * (V ** i) for i in range(order))
        seq.append(int(dist.preferred[ctx_idx]))
    return seq


def compute_target_globalsum(dist, prefix_pos, prefix_vals):
    """
    DP to find highest-probability completion satisfying sum mod V = 0.
    State: (position, running_sum mod V).
    """
    free_positions = [i for i in range(L) if i not in prefix_pos]
    prefix_sum = sum(prefix_vals) % V
    target_residual = (V - prefix_sum) % V  # free positions must sum to this mod V

    n_free = len(free_positions)
    base_lp = dist.base_lp.numpy()

    # DP forward: dp[step][residual] = max log-prob achievable
    NEG_INF = -1e30
    dp = [[NEG_INF] * V for _ in range(n_free + 1)]
    choice = [[0] * V for _ in range(n_free)]
    dp[0][0] = 0.0

    for step in range(n_free):
        pos = free_positions[step]
        for r_prev in range(V):
            if dp[step][r_prev] <= NEG_INF:
                continue
            for v in range(V):
                r_new = (r_prev + v) % V
                new_val = dp[step][r_prev] + base_lp[pos][v]
                if new_val > dp[step + 1][r_new]:
                    dp[step + 1][r_new] = new_val
                    choice[step][r_new] = v

    # Backtrack
    seq = [0] * L
    for pos, val in zip(prefix_pos, prefix_vals):
        seq[pos] = val

    r = target_residual
    for step in range(n_free - 1, -1, -1):
        v = choice[step][r]
        seq[free_positions[step]] = v
        r = (r - v) % V

    return seq


def pass_at_k(n_samples, n_correct, k):
    """Unbiased estimator from Chen et al. (2021)."""
    if n_samples - n_correct < k:
        return 1.0
    return 1.0 - comb(n_samples - n_correct, k) / comb(n_samples, k)


def evaluate_pass_k(model, dist, policies, n_samples=200,
                    n_prefixes=50, k_values=None, seed=42):
    """
    For random prefixes, generate n_samples per policy,
    compute pass@k and diversity metrics.

    Returns dict: policy → {pass_at_k: {k: value}, diversity: {...}}
    """
    if k_values is None:
        k_values = [1, 2, 5, 10, 20, 50, 100]

    rng = np.random.RandomState(seed)
    is_markov = isinstance(dist, HardMarkov)
    order = dist.k if is_markov else 2  # prefix size

    # Generate random prefixes (first `order` positions)
    prefix_pos = list(range(order))
    prefix_pool = [tuple(rng.randint(0, V, size=order))
                   for _ in range(n_prefixes)]

    # Compute targets
    targets = {}
    for pf in prefix_pool:
        if is_markov:
            targets[pf] = compute_target_markov(dist, list(pf))
        else:
            targets[pf] = compute_target_globalsum(
                dist, prefix_pos, list(pf))

    results = {}
    for pol in policies:
        all_pass = {k: [] for k in k_values}
        all_hamming = []
        all_unique = []
        t0 = time.time()

        for pf in prefix_pool:
            target = torch.tensor(targets[pf], dtype=torch.long)

            # Generate samples in batches
            samples_list = []
            remaining = n_samples
            while remaining > 0:
                bs = min(remaining, BATCH_SIZE)
                s = decode_with_prefix(
                    model, bs, prefix_pos, list(pf), pol)
                samples_list.append(s)
                remaining -= bs
            samples = torch.cat(samples_list, dim=0)[:n_samples]

            # Count correct (exact match)
            correct = (samples == target.unsqueeze(0)).all(dim=1)
            n_correct = correct.sum().item()

            for k in k_values:
                if k <= n_samples:
                    all_pass[k].append(pass_at_k(n_samples, n_correct, k))

            # Diversity: pairwise Hamming distance (subsample for speed)
            sub = samples[:min(100, n_samples)].numpy()
            if len(sub) > 1:
                total_ham = 0; n_pairs = 0
                for i in range(len(sub)):
                    for j in range(i + 1, min(i + 20, len(sub))):
                        total_ham += (sub[i] != sub[j]).sum()
                        n_pairs += 1
                avg_ham = total_ham / max(n_pairs, 1) / L
                all_hamming.append(avg_ham)

            # Unique sequences
            unique = len(set(tuple(s.tolist()) for s in samples))
            all_unique.append(unique / n_samples)

        dt = time.time() - t0
        pk = {k: float(np.mean(v)) for k, v in all_pass.items() if v}
        results[pol] = {
            'pass_at_k': pk,
            'diversity': float(np.mean(all_hamming)) if all_hamming else 0,
            'unique_ratio': float(np.mean(all_unique)),
            'time': dt,
        }
        pk_str = ' '.join(f"k={k}:{v:.3f}" for k, v in pk.items())
        print(f"    {pol:<15} {pk_str}  "
              f"div={results[pol]['diversity']:.3f}  "
              f"uniq={results[pol]['unique_ratio']:.2%}  "
              f"({dt:.1f}s)")

    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Run
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run():
    print("=" * 70)
    print("  EXP 3b: Conditional Analysis + pass@k")
    print("=" * 70)
    mount_drive()
    torch.manual_seed(42); np.random.seed(42)

    distributions = {
        'A_HardMarkov':    HardMarkov(order=2, p_peak=0.75, seed=42),
        'B_HardGlobalSum': HardGlobalSum(alpha=0.3, seed=42),
    }
    dist_names = list(distributions.keys())
    dist_colors = {'A_HardMarkov': '#e74c3c',
                   'B_HardGlobalSum': '#2ecc71'}

    # ── Distribution summary ──
    for name, dist in distributions.items():
        h = dist.entropy()
        _, p = dist.full_distribution()
        n_sup = (p > 1e-20).sum().item()
        extra = ''
        if isinstance(dist, HardMarkov):
            extra = (f"  H_l2r/step={dist.theoretical_l2r_entropy():.3f}"
                     f"  p_peak={dist.p_peak}"
                     f"  seq_mode_prob≈{dist.p_peak**(L-dist.k):.3f}")
        print(f"  {name}: H={h:.2f} bits  "
              f"support={n_sup}/{V**L} ({n_sup/V**L*100:.1f}%){extra}")

    # ── MI ──
    fig_mi, axes_mi = plt.subplots(
        1, len(distributions), figsize=(5 * len(distributions), 4))
    if len(distributions) == 1:
        axes_mi = [axes_mi]
    for idx, (name, dist) in enumerate(distributions.items()):
        MI = compute_mi(dist)
        im = axes_mi[idx].imshow(MI, cmap='YlOrRd', vmin=0)
        axes_mi[idx].set_title(name, fontsize=10)
        plt.colorbar(im, ax=axes_mi[idx], shrink=0.8)
    fig_mi.suptitle('Mutual Information', fontsize=13, y=1.02)
    fig_mi.tight_layout()
    figs = {'mi_matrices': fig_mi}

    # ── Train ──
    models, train_hists = {}, {}
    for name, dist in distributions.items():
        print(f"\n▶ Training: {name}")
        model, hist = train_toy_model(dist)
        models[name] = model
        train_hists[name] = hist

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for name, h in train_hists.items():
        axes[0].plot(h['loss'], label=name, alpha=0.8)
        axes[1].plot(h['cond_acc'], label=name, alpha=0.8)
    axes[0].set_title('Training Loss'); axes[0].set_xlabel('Epoch')
    axes[1].set_title('Conditional Accuracy'); axes[1].set_xlabel('Epoch')
    for ax in axes:
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.tight_layout(); figs['training_curves'] = fig

    # ══════════════════════════════════════════════════
    # PART 1: Conditional Analysis
    # ══════════════════════════════════════════════════

    print("\n" + "=" * 70)
    print("  PART 1: Conditional Analysis")
    print("=" * 70)

    patterns = make_all_patterns()
    cond_results = {}

    for dist_name, dist in distributions.items():
        model = models[dist_name]
        cond_results[dist_name] = {}
        print(f"\n▶ Evaluating: {dist_name}")

        for cat, instances in patterns.items():
            t0 = time.time()
            cat_results = []
            for S_pos, target in instances:
                r = evaluate_conditional(model, dist, S_pos, target)
                if r is not None:
                    r['S'] = S_pos; r['target'] = target
                    r['|S|'] = len(S_pos)
                    cat_results.append(r)
            cond_results[dist_name][cat] = cat_results
            dt = time.time() - t0
            if cat_results:
                nc, nf, nw = error_counts(cat_results)
                n_wrong = nf + nw
                fb_str = (f"  err→marginal {nf}/{n_wrong}"
                          if n_wrong > 0 else "  all correct")
                print(f"    {cat:<20}  "
                      f"mode_acc={avg_metric(cat_results, 'mode_accuracy'):.3f}  "
                      f"p(true)={avg_metric(cat_results, 'true_answer_prob'):.3f}  "
                      f"KL={avg_metric(cat_results, 'kl_forward'):.5f}"
                      f"{fb_str}  ({dt:.1f}s)")

    # ── Conditional Figures ──

    display_cats = ['marginal', 'single_left', 'single_right',
                    'causal_w2', 'anticausal_w2', 'sandwich',
                    'causal_w3', 'causal_gap1', 'leave_one_out']
    display_labels = ['marginal', 'single\nleft', 'single\nright',
                      'causal\n{t-2,t-1}', 'anti-causal\n{t+1,t+2}',
                      'sandwich\n{t-1,t+1}', 'causal\nw=3',
                      'causal\ngap=1', 'leave-\none-out']
    width = 0.35

    # FIG C1: Mode Accuracy
    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(len(display_cats))
    for i, dn in enumerate(dist_names):
        accs = [avg_metric(cond_results[dn].get(c, []), 'mode_accuracy')
                for c in display_cats]
        bars = ax.bar(x + i * width, accs, width,
                      label=dn, color=dist_colors[dn], alpha=0.85)
        for bar, val in zip(bars, accs):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height(), f'{val:.0%}',
                        ha='center', va='bottom', fontsize=6)
    ax.set_xticks(x + width / 2); ax.set_xticklabels(display_labels, fontsize=8)
    ax.set_ylabel('Mode Accuracy'); ax.set_ylim(0, 1.12)
    ax.axhline(y=1/V, color='gray', ls=':', alpha=0.5, label='chance')
    ax.legend(fontsize=7); ax.grid(axis='y', alpha=0.3)
    ax.set_title('Mode Accuracy by Conditioning Pattern', fontsize=12)
    fig.tight_layout(); figs['cond_mode_acc'] = fig

    # FIG C2: True Answer Prob
    fig, ax = plt.subplots(figsize=(13, 5))
    for i, dn in enumerate(dist_names):
        probs = [avg_metric(cond_results[dn].get(c, []), 'true_answer_prob')
                 for c in display_cats]
        bars = ax.bar(x + i * width, probs, width,
                      label=dn, color=dist_colors[dn], alpha=0.85)
        for bar, val in zip(bars, probs):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height(), f'{val:.2f}',
                        ha='center', va='bottom', fontsize=6)
    ax.set_xticks(x + width / 2); ax.set_xticklabels(display_labels, fontsize=8)
    ax.set_ylabel('p_θ(true answer)'); ax.set_ylim(0, 1.12)
    ax.axhline(y=1/V, color='gray', ls=':', alpha=0.5, label='chance')
    ax.legend(fontsize=7); ax.grid(axis='y', alpha=0.3)
    ax.set_title('True Answer Probability by Conditioning Pattern', fontsize=12)
    fig.tight_layout(); figs['cond_true_prob'] = fig

    # FIG C3: Error Type (stacked bar)
    fig, axes = plt.subplots(1, len(dist_names),
                             figsize=(6 * len(dist_names), 5))
    if len(dist_names) == 1:
        axes = [axes]
    for col, dn in enumerate(dist_names):
        ax = axes[col]
        cats_data = []
        for cat in display_cats:
            cr = cond_results[dn].get(cat, [])
            if cr:
                nc, nf, nw = error_counts(cr)
                total = nc + nf + nw
                cats_data.append((nc/total, nf/total, nw/total))
            else:
                cats_data.append((0, 0, 0))
        correct = [d[0] for d in cats_data]
        marg_fb = [d[1] for d in cats_data]
        wrong_r = [d[2] for d in cats_data]
        ax.bar(x, correct, 0.6, label='Correct', color='#2ecc71')
        ax.bar(x, marg_fb, 0.6, bottom=correct,
               label='→ marginal fallback', color='#f39c12')
        ax.bar(x, wrong_r, 0.6,
               bottom=[c+m for c, m in zip(correct, marg_fb)],
               label='→ wrong response', color='#e74c3c')
        ax.set_xticks(x); ax.set_xticklabels(display_labels, fontsize=7)
        ax.set_ylabel('Fraction'); ax.set_ylim(0, 1.05)
        ax.set_title(dn, fontsize=10)
        if col == 0:
            ax.legend(fontsize=7)
        ax.grid(axis='y', alpha=0.3)
    fig.suptitle('Error Type Breakdown', fontsize=12, y=1.02)
    fig.tight_layout(); figs['cond_error_type'] = fig

    # FIG C4: Direction asymmetry per position
    dir_cats = ['causal_w2', 'anticausal_w2', 'sandwich']
    dir_colors_map = {'causal_w2': '#2ecc71', 'anticausal_w2': '#e74c3c',
                      'sandwich': '#9b59b6'}
    dir_labels = {'causal_w2': 'causal', 'anticausal_w2': 'anti-causal',
                  'sandwich': 'sandwich'}

    fig, axes = plt.subplots(1, len(dist_names),
                             figsize=(6 * len(dist_names), 4.5))
    if len(dist_names) == 1:
        axes = [axes]
    for col, dn in enumerate(dist_names):
        ax = axes[col]
        for cat in dir_cats:
            cr = cond_results[dn].get(cat, [])
            if cr:
                ax.plot([r['target'] for r in cr],
                        [r['true_answer_prob'] for r in cr],
                        '-o', color=dir_colors_map[cat],
                        label=dir_labels[cat], markersize=5, alpha=0.85)
        mr = cond_results[dn].get('marginal', [])
        if mr:
            ax.plot([r['target'] for r in mr],
                    [r['true_answer_prob'] for r in mr],
                    '--', color='gray', alpha=0.5, label='marginal')
        ax.set_xlabel('Target position t')
        ax.set_ylabel('p_θ(true answer)')
        ax.set_title(dn, fontsize=10); ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
        if col == 0:
            ax.legend(fontsize=7)
    fig.suptitle('Direction Asymmetry', fontsize=12, y=1.02)
    fig.tight_layout(); figs['cond_direction'] = fig

    # FIG C5: Progressive chain
    fig, axes = plt.subplots(1, len(dist_names),
                             figsize=(6 * len(dist_names), 4.5))
    if len(dist_names) == 1:
        axes = [axes]
    for col, dn in enumerate(dist_names):
        ax = axes[col]
        for chain, color, label in [
            ('l2r_progressive', '#2ecc71', 'l2r'),
            ('r2l_progressive', '#e74c3c', 'r2l'),
        ]:
            cr = cond_results[dn].get(chain, [])
            if cr:
                ax.plot([r['|S|'] for r in cr],
                        [r['true_answer_prob'] for r in cr],
                        '-o', color=color, label=label,
                        markersize=5, alpha=0.85)
        ax.set_xlabel('Context size |S|')
        ax.set_ylabel('p_θ(true answer)')
        ax.set_title(dn, fontsize=10); ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
        if col == 0:
            ax.legend(fontsize=7)
    fig.suptitle('Progressive Chain', fontsize=12, y=1.02)
    fig.tight_layout(); figs['cond_progressive'] = fig

    # FIG C6: Leave-one-out per position
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    for dn in dist_names:
        cr = cond_results[dn].get('leave_one_out', [])
        if cr:
            ax.plot([r['target'] for r in cr],
                    [r['true_answer_prob'] for r in cr],
                    '-o', label=dn, markersize=5, color=dist_colors[dn])
    ax.set_xlabel('Position t'); ax.set_ylabel('p_θ(true answer)')
    ax.set_title('Leave-One-Out'); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = axes[1]
    for dn in dist_names:
        cr = cond_results[dn].get('leave_one_out', [])
        if cr:
            ax.plot([r['target'] for r in cr],
                    [r['true_entropy'] for r in cr],
                    '-s', label=dn, markersize=5, color=dist_colors[dn])
    ax.set_xlabel('Position t'); ax.set_ylabel('H_true(x_t | x_{-t})')
    ax.set_title('True Difficulty'); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.tight_layout(); figs['cond_leave_one_out'] = fig

    # ══════════════════════════════════════════════════
    # PART 2: pass@k
    # ══════════════════════════════════════════════════

    print("\n" + "=" * 70)
    print("  PART 2: pass@k Analysis")
    print("=" * 70)

    pass_k_policies = ['confidence', 'low_entropy', 'high_entropy',
                       'random', 'l2r', 'r2l']
    k_values = [1, 2, 5, 10, 20, 50, 100]
    N_SAMPLES_PK = 200     # samples per prefix
    N_PREFIXES = 50        # number of random prefixes

    pass_k_results = {}
    for dist_name, dist in distributions.items():
        model = models[dist_name]
        print(f"\n▶ pass@k: {dist_name}")

        # Theoretical baseline for well-calibrated model
        if isinstance(dist, HardMarkov):
            p_seq = dist.p_peak ** (L - dist.k)
            print(f"  Theoretical: P(mode sequence) = {p_seq:.3f}")
            for k in k_values:
                th = 1 - (1 - p_seq) ** k
                print(f"    ideal pass@{k} = {th:.3f} (independent samples)")

        pk = evaluate_pass_k(
            model, dist, pass_k_policies,
            n_samples=N_SAMPLES_PK, n_prefixes=N_PREFIXES,
            k_values=k_values)
        pass_k_results[dist_name] = pk

    # ── pass@k Figures ──

    pol_colors = {'confidence': '#e74c3c', 'low_entropy': '#e67e22',
                  'high_entropy': '#9b59b6', 'random': '#95a5a6',
                  'l2r': '#2ecc71', 'r2l': '#3498db'}

    # FIG P1: pass@k curves
    fig, axes = plt.subplots(1, len(dist_names),
                             figsize=(7 * len(dist_names), 5))
    if len(dist_names) == 1:
        axes = [axes]
    for col, dn in enumerate(dist_names):
        ax = axes[col]
        pk = pass_k_results[dn]

        # Theoretical independent baseline (for Markov)
        if isinstance(distributions[dn], HardMarkov):
            p_seq = distributions[dn].p_peak ** (L - distributions[dn].k)
            th_k = [1 - (1 - p_seq) ** k for k in k_values]
            ax.plot(k_values, th_k, 'k--', alpha=0.4,
                    label='independent (theory)', linewidth=2)

        for pol in pass_k_policies:
            if pol in pk:
                ks = sorted(pk[pol]['pass_at_k'].keys())
                vals = [pk[pol]['pass_at_k'][k] for k in ks]
                ax.plot(ks, vals, '-o', color=pol_colors[pol],
                        label=pol, markersize=4, alpha=0.85)

        ax.set_xlabel('k'); ax.set_ylabel('pass@k')
        ax.set_title(dn, fontsize=10)
        ax.set_xscale('log'); ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
        if col == 0:
            ax.legend(fontsize=6, loc='lower right')
    fig.suptitle('pass@k by Decode Policy (log scale)',
                 fontsize=12, y=1.02)
    fig.tight_layout(); figs['pass_k_curves'] = fig

    # FIG P2: Diversity vs pass@1 scatter
    fig, axes = plt.subplots(1, len(dist_names),
                             figsize=(6 * len(dist_names), 5))
    if len(dist_names) == 1:
        axes = [axes]
    for col, dn in enumerate(dist_names):
        ax = axes[col]
        pk = pass_k_results[dn]
        for pol in pass_k_policies:
            if pol in pk:
                p1 = pk[pol]['pass_at_k'].get(1, 0)
                div = pk[pol]['diversity']
                ax.scatter(div, p1, color=pol_colors[pol],
                           s=100, alpha=0.85, zorder=5)
                ax.annotate(pol, (div, p1), fontsize=7,
                            textcoords='offset points', xytext=(5, 5))
        ax.set_xlabel('Sample Diversity (avg Hamming / L)')
        ax.set_ylabel('pass@1')
        ax.set_title(dn, fontsize=10)
        ax.grid(alpha=0.3)
    fig.suptitle('Diversity vs pass@1: The Tradeoff',
                 fontsize=12, y=1.02)
    fig.tight_layout(); figs['diversity_vs_pass1'] = fig

    # FIG P3: pass@k improvement ratio (pass@k / pass@1)
    fig, axes = plt.subplots(1, len(dist_names),
                             figsize=(7 * len(dist_names), 5))
    if len(dist_names) == 1:
        axes = [axes]
    for col, dn in enumerate(dist_names):
        ax = axes[col]
        pk = pass_k_results[dn]
        for pol in pass_k_policies:
            if pol in pk:
                p1 = pk[pol]['pass_at_k'].get(1, 1e-10)
                ks = sorted(pk[pol]['pass_at_k'].keys())
                ratios = [pk[pol]['pass_at_k'][k] / max(p1, 1e-10)
                          for k in ks]
                ax.plot(ks, ratios, '-o', color=pol_colors[pol],
                        label=pol, markersize=4, alpha=0.85)
        ax.set_xlabel('k'); ax.set_ylabel('pass@k / pass@1 (scaling)')
        ax.set_title(dn, fontsize=10)
        ax.set_xscale('log'); ax.grid(alpha=0.3)
        if col == 0:
            ax.legend(fontsize=6)
    fig.suptitle('pass@k Scaling: How Much Does More Sampling Help?',
                 fontsize=12, y=1.02)
    fig.tight_layout(); figs['pass_k_scaling'] = fig

    # FIG P4: Summary bar — pass@1 and pass@10 side by side
    fig, axes = plt.subplots(1, len(dist_names),
                             figsize=(7 * len(dist_names), 5))
    if len(dist_names) == 1:
        axes = [axes]
    for col, dn in enumerate(dist_names):
        ax = axes[col]
        pk = pass_k_results[dn]
        pols = [p for p in pass_k_policies if p in pk]
        x_bar = np.arange(len(pols))
        w = 0.35
        p1s = [pk[p]['pass_at_k'].get(1, 0) for p in pols]
        p10s = [pk[p]['pass_at_k'].get(10, 0) for p in pols]
        ax.bar(x_bar - w/2, p1s, w, label='pass@1',
               color='#3498db', alpha=0.85)
        ax.bar(x_bar + w/2, p10s, w, label='pass@10',
               color='#e74c3c', alpha=0.85)
        ax.set_xticks(x_bar); ax.set_xticklabels(pols, fontsize=8)
        ax.set_ylabel('pass@k'); ax.set_ylim(0, 1.1)
        ax.set_title(dn, fontsize=10)
        ax.legend(fontsize=7); ax.grid(axis='y', alpha=0.3)
        # Annotate
        for i, (v1, v10) in enumerate(zip(p1s, p10s)):
            ax.text(i - w/2, v1, f'{v1:.2f}', ha='center',
                    va='bottom', fontsize=6)
            ax.text(i + w/2, v10, f'{v10:.2f}', ha='center',
                    va='bottom', fontsize=6)
    fig.suptitle('pass@1 vs pass@10', fontsize=12, y=1.02)
    fig.tight_layout(); figs['pass_k_summary'] = fig

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Save
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    json_results = {
        'conditional': {},
        'pass_k': {},
    }
    for dn in dist_names:
        json_results['conditional'][dn] = {}
        for cat, cr_list in cond_results[dn].items():
            json_results['conditional'][dn][cat] = [
                {k: v for k, v in r.items() if k != 'per_context'}
                for r in cr_list
            ]
        json_results['pass_k'][dn] = pass_k_results[dn]

    save_results(EXP_NAME, json_results, figures=figs)

    fig_dir = os.path.join('results', EXP_NAME)
    os.makedirs(fig_dir, exist_ok=True)
    for fname, fig_obj in figs.items():
        fig_obj.savefig(os.path.join(fig_dir, f'{fname}.png'),
                        dpi=150, bbox_inches='tight')
        print(f"  Saved {fname}.png")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Summary
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    # Conditional
    print("\n  ── Direction Asymmetry (|S|=2) ──")
    header = f"  {'Pattern':<20}"
    for dn in dist_names:
        header += f"  {'modeAcc':>7} {'p(true)':>7}"
    print(header)
    for cat in ['causal_w2', 'anticausal_w2', 'sandwich']:
        row = f"  {cat:<20}"
        for dn in dist_names:
            cr = cond_results[dn].get(cat, [])
            if cr:
                row += (f"  {avg_metric(cr, 'mode_accuracy'):>7.1%}"
                        f" {avg_metric(cr, 'true_answer_prob'):>7.3f}")
            else:
                row += f"  {'N/A':>7} {'N/A':>7}"
        print(row)

    # Error breakdown
    print("\n  ── Error Breakdown ──")
    for cat in display_cats:
        cr_any = any(cond_results[dn].get(cat) for dn in dist_names)
        if not cr_any:
            continue
        row = f"  {cat:<20}"
        for dn in dist_names:
            cr = cond_results[dn].get(cat, [])
            if cr:
                nc, nf, nw = error_counts(cr)
                n_wrong = nf + nw
                if n_wrong > 0:
                    row += f"  →marg {nf}/{n_wrong}"
                else:
                    row += f"  all correct  "
            else:
                row += f"  {'N/A':>14}"
        print(row)

    # Leave-one-out
    print("\n  ── Leave-One-Out p_θ(true answer) ──")
    for dn in dist_names:
        cr = cond_results[dn].get('leave_one_out', [])
        if cr:
            vals = ' '.join(f"t{r['target']}={r['true_answer_prob']:.3f}"
                            for r in cr)
            print(f"  {dn}: {vals}")

    # pass@k
    print("\n  ── pass@k ──")
    for dn in dist_names:
        pk = pass_k_results[dn]
        print(f"\n  {dn}:")
        for pol in pass_k_policies:
            if pol in pk:
                p1 = pk[pol]['pass_at_k'].get(1, 0)
                p10 = pk[pol]['pass_at_k'].get(10, 0)
                p50 = pk[pol]['pass_at_k'].get(50, 0)
                div = pk[pol]['diversity']
                uniq = pk[pol]['unique_ratio']
                print(f"    {pol:<15} "
                      f"@1={p1:.3f}  @10={p10:.3f}  @50={p50:.3f}  "
                      f"div={div:.3f}  uniq={uniq:.1%}")

    # Key insight
    print("\n  ── Key Insight ──")
    for dn in dist_names:
        pk = pass_k_results[dn]
        if 'confidence' in pk and 'l2r' in pk:
            conf_1 = pk['confidence']['pass_at_k'].get(1, 0)
            conf_10 = pk['confidence']['pass_at_k'].get(10, 0)
            l2r_1 = pk['l2r']['pass_at_k'].get(1, 0)
            l2r_10 = pk['l2r']['pass_at_k'].get(10, 0)
            print(f"  {dn}:")
            print(f"    confidence: @1={conf_1:.3f} → @10={conf_10:.3f} "
                  f"({conf_10/max(conf_1,1e-10):.1f}× scaling)")
            print(f"    l2r:        @1={l2r_1:.3f} → @10={l2r_10:.3f} "
                  f"({l2r_10/max(l2r_1,1e-10):.1f}× scaling)")

    plt.show()
    return cond_results, pass_k_results


if __name__ == '__main__':
    run()
