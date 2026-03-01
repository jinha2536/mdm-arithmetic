"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Experiment 3b — Learned Conditional Analysis
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Colab:  %run experiments/exp_toy_markov_focus.py

  Directly probe how well the trained masked-diffusion model has
  learned conditional distributions, WITHOUT decoding or sampling:

      p_θ(x_t | x_S = s)  vs  p_true(x_t | x_S = s)

  Distributions (V=4, L=8, V^L = 65 536):
    A: Markov2            — soft 2nd-order Markov (sparsity=0.1)
    B: HardMarkov99       — near-deterministic Markov (p_peak=0.99)
    C: UniformGlobalSum   — hard mod-V constraint, uniform within valid

  Metrics per (S, t) pair:
    KL(p_true ‖ p_θ)           — how far off overall
    mode_accuracy               — did the model get the most likely token right?
    true_answer_prob            — how much probability on the correct answer?
    error_type                  — when wrong: marginal fallback or wrong response?

  Conditioning patterns:
    Direction:     causal_w2, anticausal_w2, sandwich
    Progressive:   l2r_progressive, r2l_progressive
    Gap:           causal_w2 (no gap), causal_gap1, causal_gap2
    Sufficiency:   single_left, causal_w2, causal_w3
    Extremes:      marginal (no context), leave_one_out (full context)
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

from experiments.exp_toy_distribution import (
    V, L, MASK_ID, TOTAL_V,
    N_TRAIN, MAX_EPOCHS, PATIENCE_EPOCHS, BATCH_SIZE, LR,
    D_MODEL, N_HEADS, N_LAYERS,
    ToyDistribution, MarkovChain, HardMarkov,
    train_toy_model, compute_mi,
)
from core.train_utils import mount_drive, save_results, DEVICE

EXP_NAME = 'exp_toy_markov_focus'


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Distribution
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class UniformGlobalSum(ToyDistribution):
    """
    Hard constraint: sum(x_i) mod V == 0.
    Uniform among valid sequences.

    Conditionals (analytically):
      |S| <= L-2  →  p(x_t | x_S) = 1/V   (always uniform)
      |S| = L-1   →  point mass at (-Σs) mod V
    """
    def __init__(self):
        super().__init__()

    def log_prob(self, x):
        x = x.reshape(-1, L)
        valid = (x.sum(-1) % V == 0)
        return torch.where(valid, torch.tensor(0.0), torch.tensor(-70.0))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Core analysis
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@torch.no_grad()
def evaluate_conditional(model, dist, S_positions, target_pos,
                         prune_threshold=1e-10, batch_size=2048):
    """
    For each context s ∈ V^|S| (weighted by p(x_S=s)):
      1. Forward pass with S positions set to s, rest MASK
      2. Read p_θ(x_t | x_S=s)
      3. Compare with p_true(x_t | x_S=s)

    Returns:
        kl_forward        E_s[KL(p_true ‖ p_θ)]
        true_entropy      H_true(x_t | x_S)
        model_entropy     H_model(x_t | x_S)
        mode_accuracy     P[argmax p_θ == argmax p_true]     (p(s)-weighted)
        true_answer_prob  E_s[p_θ(true_mode | s)]
        frac_correct      fraction of contexts where mode is correct
        frac_marginal_fb  fraction of wrong contexts → marginal fallback
        frac_wrong_resp   fraction of wrong contexts → neither correct nor marginal
    """
    all_seqs, p_true = dist.full_distribution()
    p_np = p_true.numpy()
    S_np = all_seqs.numpy()
    eps = 1e-15

    # Enumerate contexts
    if len(S_positions) == 0:
        contexts = [()]
    else:
        contexts = list(iter_product(range(V), repeat=len(S_positions)))

    # Compute marginal mode for error classification
    marginal_dist = np.zeros(V)
    for v in range(V):
        marginal_dist[v] = p_np[S_np[:, target_pos] == v].sum()
    marginal_dist /= marginal_dist.sum() + eps
    marginal_mode = int(np.argmax(marginal_dist))

    # Pre-compute true conditionals with pruning
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

    # Batch forward passes
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

    # ── Aggregate ──
    total_kl = 0.0
    total_h_true = total_h_model = 0.0
    total_w = 0.0

    # Error pattern tracking
    weighted_mode_acc = 0.0
    weighted_true_prob = 0.0
    n_correct = 0
    n_marginal_fb = 0
    n_wrong_resp = 0

    per_context = []

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
        else:
            if model_mode == marginal_mode:
                n_marginal_fb += 1
            else:
                n_wrong_resp += 1

        per_context.append({
            'true_mode': true_mode,
            'model_mode': model_mode,
            'correct': correct,
            'true_mode_prob_model': float(p_m[true_mode]),
            'true_mode_prob_true': float(p_true_t[true_mode]),
            'model_conf': float(p_m.max()),
            'kl': float(kl),
            'weight': float(p_ctx),
        })

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
        'frac_correct': n_correct / max(n_total, 1),
        'frac_marginal_fb': n_marginal_fb / max(n_wrong, 1),
        'frac_wrong_resp': n_wrong_resp / max(n_wrong, 1),
        'n_correct': n_correct,
        'n_marginal_fb': n_marginal_fb,
        'n_wrong_resp': n_wrong_resp,
        'n_contexts': n_total,
        'n_pruned': len(contexts) - n_total,
        'per_context': per_context,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Conditioning patterns
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def make_all_patterns():
    P = {}

    P['marginal'] = [([], t) for t in range(L)]

    # Direction (|S|=2)
    P['causal_w2'] = [([t - 2, t - 1], t) for t in range(2, L)]
    P['anticausal_w2'] = [([t + 1, t + 2], t) for t in range(L - 2)]
    P['sandwich'] = [([t - 1, t + 1], t) for t in range(1, L - 1)]

    # Single neighbour
    P['single_left'] = [([t - 1], t) for t in range(1, L)]
    P['single_right'] = [([t + 1], t) for t in range(L - 1)]

    # Progressive
    P['l2r_progressive'] = [(list(range(t)), t) for t in range(1, L)]
    P['r2l_progressive'] = [(list(range(t + 1, L)), t)
                             for t in range(L - 2, -1, -1)]

    # Sufficiency
    P['causal_w3'] = [([t - 3, t - 2, t - 1], t) for t in range(3, L)]

    # Gap
    P['causal_gap1'] = [([t - 3, t - 1], t) for t in range(3, L)]
    P['causal_gap2'] = [([t - 4, t - 1], t) for t in range(4, L)]

    # Distant
    P['distant'] = [([0, L - 1], t) for t in range(1, L - 1)]

    # Leave-one-out
    P['leave_one_out'] = [
        (list(range(t)) + list(range(t + 1, L)), t) for t in range(L)]

    return P


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def avg_metric(cr_list, key):
    vals = [r[key] for r in cr_list if r is not None]
    return np.mean(vals) if vals else float('nan')


def error_counts(cr_list):
    nc = sum(r['n_correct'] for r in cr_list if r)
    nf = sum(r['n_marginal_fb'] for r in cr_list if r)
    nw = sum(r['n_wrong_resp'] for r in cr_list if r)
    return nc, nf, nw


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Run
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run():
    print("=" * 70)
    print("  EXP 3b: Learned Conditional Analysis")
    print("=" * 70)
    mount_drive()
    torch.manual_seed(42); np.random.seed(42)

    distributions = {
        'A_Markov2':           MarkovChain(order=2, sparsity=0.1, seed=42),
        'B_HardMarkov99':      HardMarkov(order=2, p_peak=0.99, seed=42),
        'C_UniformGlobalSum':  UniformGlobalSum(),
    }
    dist_names = list(distributions.keys())

    # ── Distribution summary ──
    for name, dist in distributions.items():
        h = dist.entropy()
        _, p = dist.full_distribution()
        n_sup = (p > 1e-20).sum().item()
        extra = ''
        if hasattr(dist, 'theoretical_l2r_entropy'):
            extra = (f"  H_l2r/step={dist.theoretical_l2r_entropy():.3f}"
                     f"  greedy_acc={dist.expected_accuracy_l2r():.2%}")
        print(f"  {name}: H={h:.2f} bits  "
              f"support={n_sup}/{V**L} ({n_sup/V**L*100:.1f}%){extra}")

    # ── MI matrices ──
    fig_mi, axes_mi = plt.subplots(
        1, len(distributions), figsize=(5 * len(distributions), 4))
    if len(distributions) == 1:
        axes_mi = [axes_mi]
    mi_matrices = {}
    for idx, (name, dist) in enumerate(distributions.items()):
        MI = compute_mi(dist)
        mi_matrices[name] = MI
        im = axes_mi[idx].imshow(MI, cmap='YlOrRd', vmin=0)
        axes_mi[idx].set_title(name, fontsize=10)
        plt.colorbar(im, ax=axes_mi[idx], shrink=0.8)
    fig_mi.suptitle('Mutual Information I(X_i; X_j)', fontsize=13, y=1.02)
    fig_mi.tight_layout()
    figs = {'mi_matrices': fig_mi}

    # ── Train ──
    models, train_hists = {}, {}
    for name, dist in distributions.items():
        print(f"\n▶ Training: {name}")
        model, hist = train_toy_model(dist)
        models[name] = model
        train_hists[name] = hist

    # Training curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for name, h in train_hists.items():
        axes[0].plot(h['loss'], label=name, alpha=0.8)
        axes[1].plot(h['cond_acc'], label=name, alpha=0.8)
    axes[0].set_title('Training Loss'); axes[0].set_xlabel('Epoch')
    axes[1].set_title('Conditional Accuracy'); axes[1].set_xlabel('Epoch')
    for ax in axes:
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.tight_layout(); figs['training_curves'] = fig

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Evaluate
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    patterns = make_all_patterns()
    results = {}

    for dist_name, dist in distributions.items():
        model = models[dist_name]
        results[dist_name] = {}
        print(f"\n▶ Evaluating: {dist_name}")

        for cat, instances in patterns.items():
            t0 = time.time()
            cat_results = []
            for S_pos, target in instances:
                r = evaluate_conditional(model, dist, S_pos, target)
                if r is not None:
                    r['S'] = S_pos
                    r['target'] = target
                    r['|S|'] = len(S_pos)
                    cat_results.append(r)
            results[dist_name][cat] = cat_results
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

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Shared constants for figures
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    display_cats = ['marginal', 'single_left', 'single_right',
                    'causal_w2', 'anticausal_w2', 'sandwich',
                    'causal_w3', 'causal_gap1',
                    'leave_one_out']
    display_labels = ['marginal\n(no ctx)', 'single\nleft',
                      'single\nright',
                      'causal\n{t-2,t-1}', 'anti-causal\n{t+1,t+2}',
                      'sandwich\n{t-1,t+1}',
                      'causal\nw=3', 'causal\ngap=1',
                      'leave-\none-out']
    dist_colors = {'A_Markov2': '#3498db',
                   'B_HardMarkov99': '#e74c3c',
                   'C_UniformGlobalSum': '#2ecc71'}
    width = 0.25

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # FIG 1: Mode Accuracy by Pattern
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(display_cats))
    for i, dn in enumerate(dist_names):
        accs = [avg_metric(results[dn].get(c, []), 'mode_accuracy')
                for c in display_cats]
        bars = ax.bar(x + i * width, accs, width,
                      label=dn, color=dist_colors[dn], alpha=0.85)
        for bar, val in zip(bars, accs):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        f'{val:.0%}', ha='center', va='bottom', fontsize=6)
    ax.set_xticks(x + width)
    ax.set_xticklabels(display_labels, fontsize=8)
    ax.set_ylabel('Mode Accuracy (weighted)')
    ax.set_ylim(0, 1.12)
    ax.axhline(y=1/V, color='gray', ls=':', alpha=0.5, label='chance (1/V)')
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_title('Mode Accuracy: Does the Model Predict the Right Token?',
                 fontsize=12)
    fig.tight_layout(); figs['mode_accuracy'] = fig

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # FIG 2: True Answer Probability by Pattern
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    fig, ax = plt.subplots(figsize=(14, 5))
    for i, dn in enumerate(dist_names):
        probs = [avg_metric(results[dn].get(c, []), 'true_answer_prob')
                 for c in display_cats]
        bars = ax.bar(x + i * width, probs, width,
                      label=dn, color=dist_colors[dn], alpha=0.85)
        for bar, val in zip(bars, probs):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        f'{val:.2f}', ha='center', va='bottom', fontsize=6)
    ax.set_xticks(x + width)
    ax.set_xticklabels(display_labels, fontsize=8)
    ax.set_ylabel('E[p_θ(true answer | context)]')
    ax.set_ylim(0, 1.12)
    ax.axhline(y=1/V, color='gray', ls=':', alpha=0.5, label='chance (1/V)')
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_title('True Answer Probability: How Much Mass on the Right Token?',
                 fontsize=12)
    fig.tight_layout(); figs['true_answer_prob'] = fig

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # FIG 3: Error Type Breakdown (stacked bar, per dist)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    fig, axes = plt.subplots(1, len(dist_names),
                             figsize=(5.5 * len(dist_names), 5))
    if len(dist_names) == 1:
        axes = [axes]

    for col, dn in enumerate(dist_names):
        ax = axes[col]
        cats_data = []
        for cat in display_cats:
            cr = results[dn].get(cat, [])
            if cr:
                nc, nf, nw = error_counts(cr)
                total = nc + nf + nw
                cats_data.append((nc / total, nf / total, nw / total))
            else:
                cats_data.append((0, 0, 0))

        correct = [d[0] for d in cats_data]
        marg_fb = [d[1] for d in cats_data]
        wrong_r = [d[2] for d in cats_data]

        ax.bar(x, correct, 0.6, label='Correct', color='#2ecc71')
        ax.bar(x, marg_fb, 0.6, bottom=correct,
               label='Wrong → marginal fallback', color='#f39c12')
        ax.bar(x, wrong_r, 0.6,
               bottom=[c + m for c, m in zip(correct, marg_fb)],
               label='Wrong → other token', color='#e74c3c')

        ax.set_xticks(x)
        ax.set_xticklabels(display_labels, fontsize=7)
        ax.set_ylabel('Fraction of contexts')
        ax.set_ylim(0, 1.05)
        ax.set_title(dn, fontsize=10)
        if col == 0:
            ax.legend(fontsize=7, loc='lower right')
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Error Type: When Wrong, Is It Marginal Fallback?',
                 fontsize=12, y=1.02)
    fig.tight_layout(); figs['error_type'] = fig

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # FIG 4: Direction Asymmetry — per target position
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    dir_cats = ['causal_w2', 'anticausal_w2', 'sandwich']
    dir_colors_map = {'causal_w2': '#2ecc71', 'anticausal_w2': '#e74c3c',
                      'sandwich': '#9b59b6'}
    dir_labels = {'causal_w2': 'causal {t-2,t-1}→t',
                  'anticausal_w2': 'anti-causal {t+1,t+2}→t',
                  'sandwich': 'sandwich {t-1,t+1}→t'}

    fig, axes = plt.subplots(2, len(dist_names),
                             figsize=(6 * len(dist_names), 8))
    if len(dist_names) == 1:
        axes = axes.reshape(-1, 1)
    for col, dn in enumerate(dist_names):
        # Top: true answer prob
        ax = axes[0, col]
        for cat in dir_cats:
            cr = results[dn].get(cat, [])
            if cr:
                targets = [r['target'] for r in cr]
                vals = [r['true_answer_prob'] for r in cr]
                ax.plot(targets, vals, '-o', color=dir_colors_map[cat],
                        label=dir_labels[cat], markersize=5, alpha=0.85)
        mr = results[dn].get('marginal', [])
        if mr:
            ax.plot([r['target'] for r in mr],
                    [r['true_answer_prob'] for r in mr],
                    '--', color='gray', alpha=0.5, label='no context')
        ax.set_ylabel('p_θ(true answer)')
        ax.set_title(dn, fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
        if col == 0:
            ax.legend(fontsize=6)

        # Bottom: KL
        ax = axes[1, col]
        for cat in dir_cats:
            cr = results[dn].get(cat, [])
            if cr:
                targets = [r['target'] for r in cr]
                vals = [r['kl_forward'] for r in cr]
                ax.plot(targets, vals, '-o', color=dir_colors_map[cat],
                        label=dir_labels[cat], markersize=5, alpha=0.85)
        ax.set_xlabel('Target position t')
        ax.set_ylabel('KL(p_true ‖ p_θ)')
        ax.grid(alpha=0.3)

    fig.suptitle('Direction Asymmetry by Position', fontsize=12, y=1.01)
    fig.tight_layout(); figs['direction_per_position'] = fig

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # FIG 5: Progressive Chain
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    fig, axes = plt.subplots(2, len(dist_names),
                             figsize=(6 * len(dist_names), 8))
    if len(dist_names) == 1:
        axes = axes.reshape(-1, 1)
    chain_colors = {'l2r_progressive': '#2ecc71',
                    'r2l_progressive': '#e74c3c'}
    chain_labels = {'l2r_progressive': 'l2r {0,…,t-1}→t',
                    'r2l_progressive': 'r2l {t+1,…,7}→t'}

    for col, dn in enumerate(dist_names):
        # Top: true answer prob
        ax = axes[0, col]
        for chain in ['l2r_progressive', 'r2l_progressive']:
            cr = results[dn].get(chain, [])
            if cr:
                sizes = [r['|S|'] for r in cr]
                vals = [r['true_answer_prob'] for r in cr]
                ax.plot(sizes, vals, '-o', color=chain_colors[chain],
                        label=chain_labels[chain], markersize=5, alpha=0.85)
        ax.set_ylabel('p_θ(true answer)')
        ax.set_title(dn, fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
        if col == 0:
            ax.legend(fontsize=7)

        # Bottom: mode accuracy
        ax = axes[1, col]
        for chain in ['l2r_progressive', 'r2l_progressive']:
            cr = results[dn].get(chain, [])
            if cr:
                sizes = [r['|S|'] for r in cr]
                vals = [r['mode_accuracy'] for r in cr]
                ax.plot(sizes, vals, '-s', color=chain_colors[chain],
                        label=chain_labels[chain], markersize=5, alpha=0.85)
        ax.set_xlabel('Context size |S|')
        ax.set_ylabel('Mode accuracy')
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)

    fig.suptitle('Progressive Chain: l2r vs r2l Context Accumulation',
                 fontsize=12, y=1.01)
    fig.tight_layout(); figs['progressive_chain'] = fig

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # FIG 6: Gap Effect (Markov only)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    gap_cats = ['causal_w2', 'causal_gap1', 'causal_gap2']
    gap_labels_map = {'causal_w2': 'no gap {t-2,t-1}',
                      'causal_gap1': '1-gap {t-3,t-1}',
                      'causal_gap2': '2-gap {t-4,t-1}'}
    gap_colors = {'causal_w2': '#2ecc71', 'causal_gap1': '#f39c12',
                  'causal_gap2': '#e74c3c'}
    markov_dists = [d for d in dist_names if 'Markov' in d]

    if markov_dists:
        fig, axes = plt.subplots(2, len(markov_dists),
                                 figsize=(6 * len(markov_dists), 8))
        if len(markov_dists) == 1:
            axes = axes.reshape(-1, 1)
        for col, dn in enumerate(markov_dists):
            # Top: true answer prob
            ax = axes[0, col]
            for cat in gap_cats:
                cr = results[dn].get(cat, [])
                if cr:
                    targets = [r['target'] for r in cr]
                    vals = [r['true_answer_prob'] for r in cr]
                    ax.plot(targets, vals, '-o', color=gap_colors[cat],
                            label=gap_labels_map[cat],
                            markersize=5, alpha=0.85)
            ax.set_ylabel('p_θ(true answer)')
            ax.set_title(dn, fontsize=10)
            ax.grid(alpha=0.3)
            if col == 0:
                ax.legend(fontsize=7)

            # Bottom: mode accuracy
            ax = axes[1, col]
            for cat in gap_cats:
                cr = results[dn].get(cat, [])
                if cr:
                    targets = [r['target'] for r in cr]
                    vals = [r['mode_accuracy'] for r in cr]
                    ax.plot(targets, vals, '-s', color=gap_colors[cat],
                            label=gap_labels_map[cat],
                            markersize=5, alpha=0.85)
            ax.set_xlabel('Target position t')
            ax.set_ylabel('Mode accuracy')
            ax.set_ylim(0, 1.05)
            ax.grid(alpha=0.3)

        fig.suptitle('Gap Effect: Breaking the Markov Context',
                     fontsize=12, y=1.01)
        fig.tight_layout(); figs['gap_effect'] = fig

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # FIG 7: Context Sufficiency (window 1→2→3)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    suf_cats = ['single_left', 'causal_w2', 'causal_w3']
    suf_labels = ['w=1', 'w=2 (=order)', 'w=3']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    x_suf = np.arange(len(suf_cats))
    ax = axes[0]
    for i, dn in enumerate(dist_names):
        vals = [avg_metric(results[dn].get(c, []), 'mode_accuracy')
                for c in suf_cats]
        ax.bar(x_suf + i * width, vals, width,
               label=dn, color=dist_colors[dn], alpha=0.85)
    ax.set_xticks(x_suf + width)
    ax.set_xticklabels(suf_labels)
    ax.set_ylabel('Mode accuracy')
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=7); ax.grid(axis='y', alpha=0.3)
    ax.set_title('Mode Accuracy by Window Size')

    ax = axes[1]
    for i, dn in enumerate(dist_names):
        vals = [avg_metric(results[dn].get(c, []), 'true_answer_prob')
                for c in suf_cats]
        ax.bar(x_suf + i * width, vals, width,
               label=dn, color=dist_colors[dn], alpha=0.85)
    ax.set_xticks(x_suf + width)
    ax.set_xticklabels(suf_labels)
    ax.set_ylabel('p_θ(true answer)')
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=7); ax.grid(axis='y', alpha=0.3)
    ax.set_title('True Answer Prob by Window Size')

    fig.suptitle('Context Sufficiency', fontsize=12, y=1.01)
    fig.tight_layout(); figs['context_sufficiency'] = fig

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # FIG 8: Leave-one-out per position
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    ax = axes[0]
    for dn in dist_names:
        cr = results[dn].get('leave_one_out', [])
        if cr:
            ax.plot([r['target'] for r in cr],
                    [r['true_answer_prob'] for r in cr],
                    '-o', label=dn, markersize=5, alpha=0.85,
                    color=dist_colors[dn])
    ax.set_xlabel('Position t'); ax.set_ylabel('p_θ(true answer)')
    ax.set_title('Leave-One-Out: True Answer Prob')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = axes[1]
    for dn in dist_names:
        cr = results[dn].get('leave_one_out', [])
        if cr:
            ax.plot([r['target'] for r in cr],
                    [r['mode_accuracy'] for r in cr],
                    '-s', label=dn, markersize=5, alpha=0.85,
                    color=dist_colors[dn])
    ax.set_xlabel('Position t'); ax.set_ylabel('Mode accuracy')
    ax.set_title('Leave-One-Out: Mode Accuracy')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = axes[2]
    for dn in dist_names:
        cr = results[dn].get('leave_one_out', [])
        if cr:
            ax.plot([r['target'] for r in cr],
                    [r['true_entropy'] for r in cr],
                    '-^', label=dn, markersize=5, alpha=0.85,
                    color=dist_colors[dn])
    ax.set_xlabel('Position t'); ax.set_ylabel('H_true(x_t | x_{-t})')
    ax.set_title('Leave-One-Out: True Difficulty')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    fig.suptitle('Leave-One-Out Analysis', fontsize=12, y=1.01)
    fig.tight_layout(); figs['leave_one_out'] = fig

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # FIG 9: Summary heatmap (3 metrics side by side)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    all_cats = list(patterns.keys())

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    metric_cfgs = [
        ('mode_accuracy', 'Mode Accuracy (↑)', 'RdYlGn'),
        ('true_answer_prob', 'P(true answer) (↑)', 'RdYlGn'),
        ('kl_forward', 'KL(true ‖ model) (↓)', 'RdYlGn_r'),
    ]

    for ax_idx, (metric, title, cmap) in enumerate(metric_cfgs):
        ax = axes[ax_idx]
        hm = np.full((len(all_cats), len(dist_names)), np.nan)
        for j, dn in enumerate(dist_names):
            for i, cat in enumerate(all_cats):
                cr = results[dn].get(cat, [])
                if cr:
                    hm[i, j] = avg_metric(cr, metric)
        im = ax.imshow(hm, cmap=cmap, aspect='auto')
        ax.set_xticks(range(len(dist_names)))
        ax.set_xticklabels([d.split('_', 1)[1] for d in dist_names],
                           fontsize=8, rotation=20)
        ax.set_yticks(range(len(all_cats)))
        ax.set_yticklabels(all_cats, fontsize=7)
        for i in range(hm.shape[0]):
            for j in range(hm.shape[1]):
                if not np.isnan(hm[i, j]):
                    v = hm[i, j]
                    fmt = (f"{v:.0%}" if metric != 'kl_forward'
                           else f"{v:.4f}")
                    med = np.nanmedian(hm[~np.isnan(hm)])
                    use_white = (('kl' in metric and v > med) or
                                 ('kl' not in metric and v < med))
                    ax.text(j, i, fmt, ha='center', va='center',
                            fontsize=6,
                            color='white' if use_white else 'black')
        plt.colorbar(im, ax=ax, shrink=0.6)
        ax.set_title(title, fontsize=10)

    fig.suptitle('Summary: All Patterns × All Distributions',
                 fontsize=12, y=1.01)
    fig.tight_layout(); figs['summary_heatmap'] = fig

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Save
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    json_results = {}
    for dn in dist_names:
        json_results[dn] = {}
        for cat, cr_list in results[dn].items():
            json_results[dn][cat] = []
            for r in cr_list:
                entry = {k: v for k, v in r.items()
                         if k != 'per_context'}
                json_results[dn][cat].append(entry)

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

    # ── 1. Direction asymmetry ──
    print("\n  ── Direction Asymmetry (|S|=2) ──")
    header = f"  {'Pattern':<20}"
    for dn in dist_names:
        header += f"  {'modeAcc':>7} {'p(true)':>7}"
    print(header)
    for cat in ['causal_w2', 'anticausal_w2', 'sandwich']:
        row = f"  {cat:<20}"
        for dn in dist_names:
            cr = results[dn].get(cat, [])
            if cr:
                row += (f"  {avg_metric(cr, 'mode_accuracy'):>7.1%}"
                        f" {avg_metric(cr, 'true_answer_prob'):>7.3f}")
            else:
                row += f"  {'N/A':>7} {'N/A':>7}"
        print(row)

    # ── 2. Error type ──
    print("\n  ── Error Breakdown (wrong contexts only) ──")
    header = f"  {'Pattern':<20}"
    for dn in dist_names:
        header += f"  {'→marg':>6} {'→other':>6}"
    print(header)
    for cat in display_cats:
        cr_any = any(results[dn].get(cat) for dn in dist_names)
        if not cr_any:
            continue
        row = f"  {cat:<20}"
        for dn in dist_names:
            cr = results[dn].get(cat, [])
            if cr:
                nc, nf, nw = error_counts(cr)
                n_wrong = nf + nw
                if n_wrong > 0:
                    row += f"  {nf/n_wrong:>6.0%} {nw/n_wrong:>6.0%}"
                else:
                    row += f"  {'—':>6} {'—':>6}"
            else:
                row += f"  {'N/A':>6} {'N/A':>6}"
        print(row)

    # ── 3. Progressive chain ──
    print("\n  ── Progressive Chain (mode accuracy) ──")
    for dn in dist_names:
        print(f"\n  {dn}:")
        for chain in ['l2r_progressive', 'r2l_progressive']:
            cr = results[dn].get(chain, [])
            if cr:
                vals = ' '.join(
                    f"|S|={r['|S|']}:{r['mode_accuracy']:.0%}"
                    for r in cr)
                print(f"    {chain}: {vals}")

    # ── 4. Gap effect ──
    print("\n  ── Gap Effect (avg mode accuracy) ──")
    for dn in markov_dists:
        print(f"  {dn}:", end='')
        for cat in gap_cats:
            cr = results[dn].get(cat, [])
            if cr:
                print(f"  {gap_labels_map.get(cat, cat)}"
                      f"={avg_metric(cr, 'mode_accuracy'):.1%}", end='')
        print()

    # ── 5. Leave-one-out ──
    print("\n  ── Leave-One-Out (p_θ(true answer)) ──")
    for dn in dist_names:
        cr = results[dn].get('leave_one_out', [])
        if cr:
            vals = ' '.join(
                f"t{r['target']}={r['true_answer_prob']:.3f}"
                for r in cr)
            print(f"  {dn}: {vals}")

    # ── 6. Calibration ──
    print("\n  ── Calibration: entropy_gap = H_model − H_true ──")
    print(f"  {'Pattern':<20}", end='')
    for dn in dist_names:
        print(f"  {dn:>18}", end='')
    print()
    for cat in ['causal_w2', 'anticausal_w2', 'leave_one_out', 'marginal']:
        print(f"  {cat:<20}", end='')
        for dn in dist_names:
            cr = results[dn].get(cat, [])
            if cr:
                print(f"  {avg_metric(cr, 'entropy_gap'):>+18.4f}", end='')
            else:
                print(f"  {'N/A':>18}", end='')
        print()

    plt.show()
    return results


if __name__ == '__main__':
    run()
