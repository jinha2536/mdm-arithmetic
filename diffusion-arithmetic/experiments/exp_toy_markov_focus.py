"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Experiment 3b — Conditional Analysis + pass@k
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Colab:  %run experiments/exp_toy_markov_focus.py

  V=4, L=8 (V^L = 65,536, fully enumerable).

  Two distributions, two analyses:

  ── A: HardMarkov (p_peak=0.99) ──────────────────────
  Near-deterministic 2nd-order Markov chain.
  → Conditional analysis only (causal vs anticausal, etc.)
  → pass@k not applicable (no binary "correct" criterion).

  ── B: HardGlobalSum (α=0.2, weighted mod-V) ────────
  p(x) ∝ 1[Σ w_i·x_i ≡ 0 (mod V)] · ∏ b_i(x_i)
  Weights w_i ∈ {1,...,V-1} per position (random, fixed).
  Very peaked position bias (α=0.2).
  → Conditional analysis
  → pass@k where "correct" = constraint satisfied.
     This is a natural binary verifiable criterion,
     analogous to reasoning tasks where the answer is checkable.

  Key question for pass@k:
    Decode order determines WHERE branching occurs.
    Confidence fills peaked positions first → k samples converge
    → low diversity → poor pass@k scaling.
    Can a different order maintain greedy quality AND scale better?
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
# Distribution
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class HardGlobalSum(ToyDistribution):
    """
    p(x) ∝ 1[Σ w_i·x_i ≡ 0 (mod V)] · ∏ b_i(x_i)

    Weighted mod constraint is harder to learn than simple sum:
    the model must discover position-specific weights AND do
    modular arithmetic.

    α=0.2 gives very peaked position bias, creating strong tension:
    following bias is "easy" but often violates the constraint.
    """
    def __init__(self, alpha=0.2, seed=42):
        super().__init__()
        rng = np.random.RandomState(seed)
        # Peaked position-wise bias
        self.base_lp = torch.tensor(
            np.log(rng.dirichlet([alpha] * V, size=L) + 1e-30),
            dtype=torch.float32)
        # Random weights per position, in {1, ..., V-1}
        self.weights = torch.tensor(
            rng.randint(1, V, size=L), dtype=torch.long)

    def log_prob(self, x):
        x = x.reshape(-1, L)
        lp = sum(self.base_lp[i][x[:, i]] for i in range(L))
        weighted_sum = (x * self.weights.unsqueeze(0)).sum(-1)
        valid = (weighted_sum % V == 0)
        return torch.where(valid, lp, torch.tensor(-70.0))

    def check_constraint(self, x):
        """Binary check: does sequence satisfy the weighted mod constraint?"""
        x = x.reshape(-1, L)
        weighted_sum = (x * self.weights.unsqueeze(0)).sum(-1)
        return (weighted_sum % V == 0)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Part 1: Conditional Analysis (both distributions)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@torch.no_grad()
def evaluate_conditional(model, dist, S_positions, target_pos,
                         prune_threshold=1e-10, batch_size=2048):
    all_seqs, p_true = dist.full_distribution()
    p_np = p_true.numpy(); S_np = all_seqs.numpy(); eps = 1e-15

    contexts = [()] if len(S_positions) == 0 else \
        list(iter_product(range(V), repeat=len(S_positions)))

    # Marginal mode
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
    w_mode_acc = w_true_prob = 0.0
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
        w_mode_acc += p_ctx * float(correct)
        w_true_prob += p_ctx * float(p_m[true_mode])
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
        'mode_accuracy': float(w_mode_acc / w),
        'true_answer_prob': float(w_true_prob / w),
        'n_correct': n_correct, 'n_marginal_fb': n_marginal_fb,
        'n_wrong_resp': n_wrong_resp, 'n_contexts': n_total,
    }


def make_all_patterns():
    P = {}
    P['marginal'] = [([], t) for t in range(L)]
    P['single_left'] = [([t-1], t) for t in range(1, L)]
    P['single_right'] = [([t+1], t) for t in range(L-1)]
    P['causal_w2'] = [([t-2, t-1], t) for t in range(2, L)]
    P['anticausal_w2'] = [([t+1, t+2], t) for t in range(L-2)]
    P['sandwich'] = [([t-1, t+1], t) for t in range(1, L-1)]
    P['causal_w3'] = [([t-3, t-2, t-1], t) for t in range(3, L)]
    P['causal_gap1'] = [([t-3, t-1], t) for t in range(3, L)]
    P['l2r_progressive'] = [(list(range(t)), t) for t in range(1, L)]
    P['r2l_progressive'] = [(list(range(t+1, L)), t)
                             for t in range(L-2, -1, -1)]
    P['leave_one_out'] = [
        (list(range(t)) + list(range(t+1, L)), t) for t in range(L)]
    return P


def avg_metric(cr, key):
    v = [r[key] for r in cr if r]; return np.mean(v) if v else float('nan')

def error_counts(cr):
    return (sum(r['n_correct'] for r in cr if r),
            sum(r['n_marginal_fb'] for r in cr if r),
            sum(r['n_wrong_resp'] for r in cr if r))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Part 2: pass@k (HardGlobalSum only)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@torch.no_grad()
def decode_with_prefix(model, n, prefix_pos, prefix_vals, policy):
    """Decode with prefix positions fixed, rest by policy + sampling."""
    x = torch.full((n, L), MASK_ID, dtype=torch.long, device=DEVICE)
    unmasked = torch.zeros(n, L, dtype=torch.bool, device=DEVICE)
    for pos, val in zip(prefix_pos, prefix_vals):
        x[:, pos] = val; unmasked[:, pos] = True

    for step in range(L - len(prefix_pos)):
        logits = model(x)
        logits[:, :, MASK_ID] = -float('inf')
        probs = F.softmax(logits, dim=-1)

        if policy == 'confidence':
            sc = probs.max(-1).values.clone(); sc[unmasked] = -1
            pos = sc.argmax(-1)
        elif policy == 'low_entropy':
            e = -(probs * (probs + 1e-10).log()).sum(-1)
            e[unmasked] = 1e9; pos = e.argmin(-1)
        elif policy == 'random':
            sc = torch.rand(n, L, device=DEVICE); sc[unmasked] = -1
            pos = sc.argmax(-1)
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

    return x.cpu().clamp(max=V-1)


def pass_at_k_estimator(n, c, k):
    """Unbiased estimator (Chen et al. 2021). n=total, c=correct, k=k."""
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def evaluate_pass_k(model, dist, policies, n_samples=200,
                    n_prefixes=100, k_values=None,
                    prefix_size=2, seed=42):
    """
    Generate n_samples per (prefix, policy).
    Correct = dist.check_constraint(sequence).
    """
    if k_values is None:
        k_values = [1, 2, 5, 10, 20, 50, 100]

    rng = np.random.RandomState(seed)
    prefix_pos = list(range(prefix_size))
    prefix_pool = [tuple(rng.randint(0, V, size=prefix_size))
                   for _ in range(n_prefixes)]

    results = {}
    for pol in policies:
        all_pass = {k: [] for k in k_values}
        all_sat_rate = []
        all_hamming = []
        all_unique_valid = []
        t0 = time.time()

        for pf in prefix_pool:
            # Generate samples
            samples_list = []
            rem = n_samples
            while rem > 0:
                bs = min(rem, BATCH_SIZE)
                s = decode_with_prefix(model, bs, prefix_pos, list(pf), pol)
                samples_list.append(s)
                rem -= bs
            samples = torch.cat(samples_list, dim=0)[:n_samples]

            # Check constraint
            valid = dist.check_constraint(samples).numpy()
            n_correct = valid.sum()
            sat_rate = n_correct / n_samples
            all_sat_rate.append(sat_rate)

            # pass@k
            for k in k_values:
                if k <= n_samples:
                    all_pass[k].append(
                        pass_at_k_estimator(n_samples, int(n_correct), k))

            # Diversity: pairwise Hamming (subsample)
            sub = samples[:min(100, n_samples)].numpy()
            if len(sub) > 1:
                total_h = 0; n_p = 0
                for i in range(len(sub)):
                    for j in range(i+1, min(i+20, len(sub))):
                        total_h += (sub[i] != sub[j]).sum()
                        n_p += 1
                all_hamming.append(total_h / max(n_p, 1) / L)

            # Unique valid sequences
            valid_seqs = samples[torch.tensor(valid, dtype=torch.bool)]
            n_unique_valid = len(set(tuple(s.tolist()) for s in valid_seqs))
            all_unique_valid.append(n_unique_valid)

        dt = time.time() - t0
        pk = {k: float(np.mean(v)) for k, v in all_pass.items() if v}
        results[pol] = {
            'pass_at_k': pk,
            'constraint_sat_rate': float(np.mean(all_sat_rate)),
            'diversity': float(np.mean(all_hamming)) if all_hamming else 0,
            'unique_valid_mean': float(np.mean(all_unique_valid)),
            'time': dt,
        }
        pk_str = ' '.join(f"@{k}={v:.3f}" for k, v in pk.items())
        print(f"    {pol:<15} sat={results[pol]['constraint_sat_rate']:.1%}  "
              f"{pk_str}  "
              f"div={results[pol]['diversity']:.3f}  "
              f"uniq_valid={results[pol]['unique_valid_mean']:.1f}  "
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
        'A_HardMarkov':    HardMarkov(order=2, p_peak=0.99, seed=42),
        'B_HardGlobalSum': HardGlobalSum(alpha=0.2, seed=42),
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
                     f"  p_peak={dist.p_peak}")
        if isinstance(dist, HardGlobalSum):
            extra = f"  weights={dist.weights.tolist()}"
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
    # PART 1: Conditional Analysis (both distributions)
    # ══════════════════════════════════════════════════

    print("\n" + "=" * 70)
    print("  PART 1: Conditional Analysis")
    print("=" * 70)

    patterns = make_all_patterns()
    cond_results = {}

    for dn, dist in distributions.items():
        model = models[dn]
        cond_results[dn] = {}
        print(f"\n▶ Evaluating: {dn}")
        for cat, instances in patterns.items():
            t0 = time.time()
            cr = []
            for S_pos, target in instances:
                r = evaluate_conditional(model, dist, S_pos, target)
                if r:
                    r['S'] = S_pos; r['target'] = target
                    r['|S|'] = len(S_pos)
                    cr.append(r)
            cond_results[dn][cat] = cr
            dt = time.time() - t0
            if cr:
                nc, nf, nw = error_counts(cr)
                nwrong = nf + nw
                fb = (f"  err→marg {nf}/{nwrong}" if nwrong > 0
                      else "  all correct")
                print(f"    {cat:<20}  "
                      f"mode_acc={avg_metric(cr, 'mode_accuracy'):.3f}  "
                      f"p(true)={avg_metric(cr, 'true_answer_prob'):.3f}  "
                      f"KL={avg_metric(cr, 'kl_forward'):.5f}"
                      f"{fb}  ({dt:.1f}s)")

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
        vals = [avg_metric(cond_results[dn].get(c, []), 'mode_accuracy')
                for c in display_cats]
        bars = ax.bar(x + i*width, vals, width,
                      label=dn, color=dist_colors[dn], alpha=0.85)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{v:.0%}', ha='center', va='bottom', fontsize=6)
    ax.set_xticks(x + width/2)
    ax.set_xticklabels(display_labels, fontsize=8)
    ax.set_ylabel('Mode Accuracy'); ax.set_ylim(0, 1.12)
    ax.axhline(y=1/V, color='gray', ls=':', alpha=0.5, label='chance')
    ax.legend(fontsize=7); ax.grid(axis='y', alpha=0.3)
    ax.set_title('Mode Accuracy by Conditioning Pattern')
    fig.tight_layout(); figs['cond_mode_acc'] = fig

    # FIG C2: True Answer Prob
    fig, ax = plt.subplots(figsize=(13, 5))
    for i, dn in enumerate(dist_names):
        vals = [avg_metric(cond_results[dn].get(c, []), 'true_answer_prob')
                for c in display_cats]
        bars = ax.bar(x + i*width, vals, width,
                      label=dn, color=dist_colors[dn], alpha=0.85)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{v:.2f}', ha='center', va='bottom', fontsize=6)
    ax.set_xticks(x + width/2)
    ax.set_xticklabels(display_labels, fontsize=8)
    ax.set_ylabel('p_θ(true answer)'); ax.set_ylim(0, 1.12)
    ax.legend(fontsize=7); ax.grid(axis='y', alpha=0.3)
    ax.set_title('True Answer Probability by Conditioning Pattern')
    fig.tight_layout(); figs['cond_true_prob'] = fig

    # FIG C3: Error Type
    fig, axes = plt.subplots(1, len(dist_names),
                             figsize=(6 * len(dist_names), 5))
    if len(dist_names) == 1: axes = [axes]
    for col, dn in enumerate(dist_names):
        ax = axes[col]
        data = []
        for cat in display_cats:
            cr = cond_results[dn].get(cat, [])
            if cr:
                nc, nf, nw = error_counts(cr)
                t = nc + nf + nw
                data.append((nc/t, nf/t, nw/t))
            else:
                data.append((0, 0, 0))
        c_ = [d[0] for d in data]
        m_ = [d[1] for d in data]
        w_ = [d[2] for d in data]
        ax.bar(x, c_, 0.6, label='Correct', color='#2ecc71')
        ax.bar(x, m_, 0.6, bottom=c_, label='→ marginal', color='#f39c12')
        ax.bar(x, w_, 0.6, bottom=[a+b for a, b in zip(c_, m_)],
               label='→ wrong', color='#e74c3c')
        ax.set_xticks(x); ax.set_xticklabels(display_labels, fontsize=7)
        ax.set_ylim(0, 1.05); ax.set_title(dn, fontsize=10)
        if col == 0: ax.legend(fontsize=7)
        ax.grid(axis='y', alpha=0.3)
    fig.suptitle('Error Type Breakdown', fontsize=12, y=1.02)
    fig.tight_layout(); figs['cond_error_type'] = fig

    # FIG C4: Direction per position
    dir_cats = ['causal_w2', 'anticausal_w2', 'sandwich']
    dc = {'causal_w2': '#2ecc71', 'anticausal_w2': '#e74c3c',
          'sandwich': '#9b59b6'}
    fig, axes = plt.subplots(1, len(dist_names),
                             figsize=(6*len(dist_names), 4.5))
    if len(dist_names) == 1: axes = [axes]
    for col, dn in enumerate(dist_names):
        ax = axes[col]
        for cat in dir_cats:
            cr = cond_results[dn].get(cat, [])
            if cr:
                ax.plot([r['target'] for r in cr],
                        [r['true_answer_prob'] for r in cr],
                        '-o', color=dc[cat], label=cat, markersize=5)
        mr = cond_results[dn].get('marginal', [])
        if mr:
            ax.plot([r['target'] for r in mr],
                    [r['true_answer_prob'] for r in mr],
                    '--', color='gray', alpha=0.5, label='marginal')
        ax.set_xlabel('Target position t'); ax.set_ylabel('p_θ(true answer)')
        ax.set_title(dn); ax.set_ylim(0, 1.05); ax.grid(alpha=0.3)
        if col == 0: ax.legend(fontsize=7)
    fig.suptitle('Direction Asymmetry', fontsize=12, y=1.02)
    fig.tight_layout(); figs['cond_direction'] = fig

    # FIG C5: Progressive
    fig, axes = plt.subplots(1, len(dist_names),
                             figsize=(6*len(dist_names), 4.5))
    if len(dist_names) == 1: axes = [axes]
    for col, dn in enumerate(dist_names):
        ax = axes[col]
        for ch, color, label in [('l2r_progressive', '#2ecc71', 'l2r'),
                                  ('r2l_progressive', '#e74c3c', 'r2l')]:
            cr = cond_results[dn].get(ch, [])
            if cr:
                ax.plot([r['|S|'] for r in cr],
                        [r['true_answer_prob'] for r in cr],
                        '-o', color=color, label=label, markersize=5)
        ax.set_xlabel('|S|'); ax.set_ylabel('p_θ(true answer)')
        ax.set_title(dn); ax.set_ylim(0, 1.05); ax.grid(alpha=0.3)
        if col == 0: ax.legend(fontsize=7)
    fig.suptitle('Progressive Chain', fontsize=12, y=1.02)
    fig.tight_layout(); figs['cond_progressive'] = fig

    # FIG C6: Leave-one-out
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for dn in dist_names:
        cr = cond_results[dn].get('leave_one_out', [])
        if cr:
            axes[0].plot([r['target'] for r in cr],
                         [r['true_answer_prob'] for r in cr],
                         '-o', label=dn, markersize=5, color=dist_colors[dn])
            axes[1].plot([r['target'] for r in cr],
                         [r['true_entropy'] for r in cr],
                         '-s', label=dn, markersize=5, color=dist_colors[dn])
    axes[0].set_xlabel('Position t'); axes[0].set_ylabel('p_θ(true answer)')
    axes[0].set_title('Leave-One-Out: p(true)')
    axes[0].legend(fontsize=7); axes[0].grid(alpha=0.3)
    axes[1].set_xlabel('Position t'); axes[1].set_ylabel('H_true')
    axes[1].set_title('Leave-One-Out: True Difficulty')
    axes[1].legend(fontsize=7); axes[1].grid(alpha=0.3)
    fig.tight_layout(); figs['cond_loo'] = fig

    # ══════════════════════════════════════════════════
    # PART 2: pass@k (HardGlobalSum only)
    # ══════════════════════════════════════════════════

    print("\n" + "=" * 70)
    print("  PART 2: pass@k (constraint satisfaction)")
    print("=" * 70)

    gs_dist = distributions['B_HardGlobalSum']
    gs_model = models['B_HardGlobalSum']

    # Baseline: random completion constraint satisfaction rate
    rng = np.random.RandomState(0)
    random_seqs = torch.tensor(rng.randint(0, V, size=(10000, L)),
                               dtype=torch.long)
    random_sat = gs_dist.check_constraint(random_seqs).float().mean().item()
    print(f"  Random baseline: constraint satisfaction = {random_sat:.1%}")
    print(f"  Weights: {gs_dist.weights.tolist()}")

    pk_policies = ['confidence', 'low_entropy', 'random', 'l2r', 'r2l']
    k_values = [1, 2, 5, 10, 20, 50, 100]

    print(f"\n▶ pass@k (n_samples=200, n_prefixes=100)")
    pk_results = evaluate_pass_k(
        gs_model, gs_dist, pk_policies,
        n_samples=200, n_prefixes=100, k_values=k_values)

    # ── pass@k Figures ──
    pol_colors = {'confidence': '#e74c3c', 'low_entropy': '#e67e22',
                  'random': '#95a5a6', 'l2r': '#2ecc71', 'r2l': '#3498db'}

    # FIG P1: pass@k curves
    fig, ax = plt.subplots(figsize=(8, 5))
    # Theoretical: if each sample has prob p of being correct independently
    avg_sat = np.mean([pk_results[p]['constraint_sat_rate']
                       for p in pk_policies])
    th_k = [1 - (1 - random_sat)**k for k in k_values]
    ax.plot(k_values, th_k, 'k:', alpha=0.4, lw=2,
            label=f'random baseline (p={random_sat:.2f})')
    for pol in pk_policies:
        if pol in pk_results:
            ks = sorted(pk_results[pol]['pass_at_k'].keys())
            vals = [pk_results[pol]['pass_at_k'][k] for k in ks]
            ax.plot(ks, vals, '-o', color=pol_colors[pol],
                    label=f"{pol} (sat={pk_results[pol]['constraint_sat_rate']:.0%})",
                    markersize=5, alpha=0.85)
    ax.set_xlabel('k'); ax.set_ylabel('pass@k')
    ax.set_title('pass@k: Constraint Satisfaction')
    ax.set_xscale('log'); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.tight_layout(); figs['pass_k_curves'] = fig

    # FIG P2: pass@k scaling (pass@k / pass@1)
    fig, ax = plt.subplots(figsize=(8, 5))
    for pol in pk_policies:
        if pol in pk_results:
            p1 = pk_results[pol]['pass_at_k'].get(1, 1e-10)
            ks = sorted(pk_results[pol]['pass_at_k'].keys())
            ratios = [pk_results[pol]['pass_at_k'][k] / max(p1, 1e-10)
                      for k in ks]
            ax.plot(ks, ratios, '-o', color=pol_colors[pol],
                    label=pol, markersize=4, alpha=0.85)
    ax.set_xlabel('k'); ax.set_ylabel('pass@k / pass@1')
    ax.set_title('pass@k Scaling (higher = sampling helps more)')
    ax.set_xscale('log'); ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout(); figs['pass_k_scaling'] = fig

    # FIG P3: Diversity vs Constraint Satisfaction scatter
    fig, ax = plt.subplots(figsize=(7, 5))
    for pol in pk_policies:
        if pol in pk_results:
            sat = pk_results[pol]['constraint_sat_rate']
            div = pk_results[pol]['diversity']
            ax.scatter(div, sat, color=pol_colors[pol],
                       s=120, alpha=0.85, zorder=5)
            ax.annotate(pol, (div, sat), fontsize=8,
                        textcoords='offset points', xytext=(5, 5))
    ax.axhline(y=random_sat, color='gray', ls=':', alpha=0.5,
               label=f'random baseline ({random_sat:.0%})')
    ax.set_xlabel('Sample Diversity (avg Hamming / L)')
    ax.set_ylabel('Constraint Satisfaction Rate')
    ax.set_title('Diversity vs Quality')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.tight_layout(); figs['diversity_vs_quality'] = fig

    # FIG P4: Summary bar
    fig, ax = plt.subplots(figsize=(10, 5))
    pols = [p for p in pk_policies if p in pk_results]
    x_bar = np.arange(len(pols))
    w = 0.2
    metrics_bar = [
        ('constraint_sat_rate', 'Satisfaction Rate', '#3498db'),
        ('pass@1', 'pass@1', '#2ecc71'),
        ('pass@10', 'pass@10', '#e74c3c'),
    ]
    for mi, (key, label, color) in enumerate(metrics_bar):
        vals = []
        for p in pols:
            if key == 'constraint_sat_rate':
                vals.append(pk_results[p][key])
            else:
                k = int(key.split('@')[1])
                vals.append(pk_results[p]['pass_at_k'].get(k, 0))
        bars = ax.bar(x_bar + mi*w, vals, w, label=label, color=color, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{v:.2f}', ha='center', va='bottom', fontsize=6)
    ax.set_xticks(x_bar + w); ax.set_xticklabels(pols, fontsize=9)
    ax.set_ylabel('Rate'); ax.set_ylim(0, 1.1)
    ax.axhline(y=random_sat, color='gray', ls=':', alpha=0.5)
    ax.legend(fontsize=7); ax.grid(axis='y', alpha=0.3)
    ax.set_title('Summary: Satisfaction Rate, pass@1, pass@10')
    fig.tight_layout(); figs['pass_k_summary'] = fig

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Save
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    json_results = {'conditional': {}, 'pass_k': {}}
    for dn in dist_names:
        json_results['conditional'][dn] = {}
        for cat, cr in cond_results[dn].items():
            json_results['conditional'][dn][cat] = [
                {k: v for k, v in r.items()} for r in cr]
    json_results['pass_k'] = pk_results
    json_results['pass_k']['_random_baseline'] = random_sat

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
    print("\n  ── Direction Asymmetry ──")
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

    print("\n  ── Leave-One-Out ──")
    for dn in dist_names:
        cr = cond_results[dn].get('leave_one_out', [])
        if cr:
            vals = ' '.join(f"t{r['target']}={r['true_answer_prob']:.3f}"
                            for r in cr)
            print(f"  {dn}: {vals}")

    # pass@k
    print(f"\n  ── pass@k (HardGlobalSum) ──")
    print(f"  Random baseline: {random_sat:.1%}")
    for pol in pk_policies:
        if pol in pk_results:
            r = pk_results[pol]
            p1 = r['pass_at_k'].get(1, 0)
            p10 = r['pass_at_k'].get(10, 0)
            p50 = r['pass_at_k'].get(50, 0)
            print(f"    {pol:<15} sat={r['constraint_sat_rate']:.1%}  "
                  f"@1={p1:.3f}  @10={p10:.3f}  @50={p50:.3f}  "
                  f"div={r['diversity']:.3f}")

    plt.show()
    return cond_results, pk_results


if __name__ == '__main__':
    run()
