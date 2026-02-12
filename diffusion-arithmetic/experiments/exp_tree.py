"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Experiment 2 — Parallel Independent Additions
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Colab:  %run experiments/exp_tree.py

  Tests whether diffusion models decode independent output segments
  in parallel, and whether this yields accuracy or generalisation
  advantages over AR.

  Format:  a1+b1|a2+b2|...|ak+bk=r1|r2|...|rk
  Example (k=3):  23+45|67+89|12+34=068|156|046

  Key insight:
    AR decodes the answer left-to-right across ALL segments.
    Diffusion can potentially fill all k segments simultaneously.

  Test splits (per model):
    test_k2   (k=2, in-distribution)
    test_k4   (k=4, in-distribution)
    test_k8   (k=8, out-of-distribution — more tasks than training)

  Analysis:
    - Accuracy scaling with k
    - Per-segment accuracy (position bias?)
    - Parallelism index (Kendall's τ on segment decode order)
    - k generalisation (train k∈{2,4} → test k=8)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                if '__file__' in dir() else '.')

import random, torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('Agg')
import numpy as np

from core.tokenizer import CharTokenizer
from core.train_utils import (
    mount_drive, save_results, train_model,
    generate_ar, generate_diffusion, DEVICE,
)

EXP_NAME = 'exp_parallel'

# ── Config ──────────────────────────────────────────
ND          = 2           # digits per operand (10–99)
ANS_WIDTH   = 3           # digits per answer (zero-padded; max 99+99=198)
K_TRAIN     = [2, 4]      # parallel tasks seen during training
K_TEST_OOD  = 8           # OOD: unseen number of parallel tasks
N_TRAIN     = 10_000
N_TEST      = 2_000
MAX_ITERS   = 15_000
PATIENCE    = 2_000
POS_ENCS    = ['absolute', 'rope']


# ── Data generation ─────────────────────────────────

def gen_sample(k, nd, rng):
    """Generate one sample: k independent nd-digit additions."""
    lo, hi = 10**(nd - 1), 10**nd - 1
    problems, answers = [], []
    for _ in range(k):
        a, b = rng.randint(lo, hi), rng.randint(lo, hi)
        problems.append(f"{a}+{b}")
        answers.append(str(a + b).zfill(ANS_WIDTH))
    return "|".join(problems) + "=" + "|".join(answers)


def gen_data_mixed(ks, n, seed=42):
    """Generate n samples with k chosen randomly from ks per sample."""
    rng = random.Random(seed)
    return [gen_sample(rng.choice(ks), ND, rng) for _ in range(n)]


def gen_data_fixed_k(k, n, seed=42):
    """Generate n samples, all with the same k."""
    rng = random.Random(seed)
    return [gen_sample(k, ND, rng) for _ in range(n)]


def get_answer(s):
    """Full answer string after '='."""
    return s.split('=', 1)[1]


def build_tok():
    chars = list('0123456789+=|')
    return CharTokenizer(chars, {'mask': 'M', 'pad': 'P'})


# ── Segment-level analysis helpers ──────────────────

def _kendall_tau(x, y):
    """Kendall's τ for small arrays (no scipy dependency)."""
    n = len(x)
    con, dis = 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            s = (x[i] - x[j]) * (y[i] - y[j])
            if s > 0:
                con += 1
            elif s < 0:
                dis += 1
    d = con + dis
    return (con - dis) / d if d > 0 else 0.0


def segment_accuracy(pred_str, gold_str, k):
    """Per-segment exact match.  Returns list of k booleans."""
    pred_segs = pred_str.split('|')
    gold_segs = gold_str.split('|')
    if len(pred_segs) != k:
        return [False] * k
    return [p == g for p, g in zip(pred_segs, gold_segs)]


def analyse_parallel_decode(decode_orders, ans_start, k,
                            seg_width=ANS_WIDTH):
    """
    Compute parallelism metrics from diffusion decode orders.

    Args
        decode_orders : (N, ans_tokens) — orders[b, step] = abs position
        ans_start     : index of first answer token
        k             : number of answer segments
        seg_width     : digits per segment

    Returns dict
        segment_mean_ranks : (k,) mean decode step per segment
        parallelism_tau    : mean Kendall's τ across samples
            ≈ 0  → parallel (no order preference among segments)
            ≈ 1  → sequential left-to-right
    """
    if decode_orders is None or len(decode_orders) == 0:
        return {}

    N, S = decode_orders.shape

    # Segment digit positions (relative to ans_start, excluding '|')
    # Answer layout:  seg0 | seg1 | ... | seg_{k-1}
    # Segment i starts at relative position i * (seg_width + 1)
    seg_positions = []
    for i in range(k):
        start = i * (seg_width + 1)
        seg_positions.append(list(range(start, start + seg_width)))

    # Invert step→position to position→rank (vectorised)
    rel_orders = (decode_orders - ans_start).long().clamp(0, S - 1)
    steps = torch.arange(S).unsqueeze(0).expand(N, S).float()
    pos_rank = torch.zeros(N, S)
    pos_rank.scatter_(1, rel_orders, steps)

    # Mean rank per segment (across all samples)
    seg_mean = np.zeros(k)
    for i, positions in enumerate(seg_positions):
        valid = [p for p in positions if p < S]
        if valid:
            seg_mean[i] = pos_rank[:, valid].mean().item()

    # Per-sample Kendall's τ
    taus = []
    seg_idx = list(range(k))
    for b in range(N):
        ranks = []
        for ps in seg_positions:
            valid = [p for p in ps if p < S]
            ranks.append(pos_rank[b, valid].mean().item() if valid else 0)
        taus.append(_kendall_tau(seg_idx, ranks))

    return {
        'segment_mean_ranks': seg_mean.tolist(),
        'parallelism_tau': float(np.mean(taus)),
        'tau_std': float(np.std(taus)),
        'n_samples': N,
    }


# ── Evaluation ──────────────────────────────────────

def evaluate_parallel(model, tokenizer, test_samples, objective, k,
                      batch_size=128):
    """
    Full evaluation for parallel additions:
      • whole-sequence exact match
      • per-segment exact match
      • decode-order parallelism analysis (diffusion only)
    """
    mask_id = tokenizer.special_ids['mask']
    model.eval()

    # All samples share the same k, so answer length is constant
    ex = test_samples[0]
    ans_len = len(tokenizer.encode(get_answer(ex)))
    ans_start = len(tokenizer.encode(ex.split('=')[0] + '='))

    all_correct, all_orders = [], []
    seg_correct = [[] for _ in range(k)]

    for start in range(0, len(test_samples), batch_size):
        batch = test_samples[start:start + batch_size]
        B = len(batch)
        golds = [get_answer(s) for s in batch]

        penc = [tokenizer.encode(s.split('=')[0] + '=') for s in batch]
        pmax = max(len(p) for p in penc)
        pids = torch.full((B, pmax), tokenizer.special_ids['pad'],
                          dtype=torch.long)
        for i, e in enumerate(penc):
            pids[i, :len(e)] = torch.tensor(e)

        with torch.no_grad():
            if objective == 'ar':
                gen = generate_ar(model, pids, ans_len, DEVICE)
                pred_ids = gen[:, pmax:pmax + ans_len]
                batch_orders = None
            else:
                gen, _, info = generate_diffusion(
                    model, pids, ans_len, mask_id,
                    policy='confidence', greedy=True, device=DEVICE)
                pred_ids = gen[:, pmax:pmax + ans_len]
                batch_orders = info.get('orders')

        if batch_orders is not None:
            all_orders.append(batch_orders)

        for i in range(B):
            pred_str = tokenizer.decode(pred_ids[i].cpu().tolist())
            gold_str = golds[i]
            all_correct.append(pred_str == gold_str)
            segs = segment_accuracy(pred_str, gold_str, k)
            for j in range(k):
                seg_correct[j].append(segs[j])

    result = {
        'exact_match': sum(all_correct) / max(len(all_correct), 1),
        'segment_accuracy': [
            sum(sc) / max(len(sc), 1) for sc in seg_correct],
    }

    if all_orders:
        orders = torch.cat(all_orders, dim=0)
        par = analyse_parallel_decode(orders, ans_start, k)
        result['parallelism'] = par

    return result


# ── Main ────────────────────────────────────────────

def run():
    print("=" * 70)
    print("  EXP 2: Parallel Independent Additions — AR vs Diffusion")
    print("=" * 70)
    mount_drive()
    torch.manual_seed(42)

    tok = build_tok()

    # ── Data ──
    train_data = gen_data_mixed(K_TRAIN, N_TRAIN, seed=42)
    test_splits = {}
    for k in K_TRAIN:
        test_splits[f'test_k{k}'] = gen_data_fixed_k(k, N_TEST,
                                                       seed=1000 + k)
    test_splits[f'test_k{K_TEST_OOD}'] = gen_data_fixed_k(
        K_TEST_OOD, N_TEST, seed=1000 + K_TEST_OOD)

    all_s = train_data + [s for v in test_splits.values() for s in v]
    max_len = max(len(tok.encode(s)) for s in all_s) + 1

    print(f"\n  Config: ND={ND}, K_TRAIN={K_TRAIN}, K_OOD={K_TEST_OOD}")
    print(f"  Train: {len(train_data)} samples (mixed k)")
    for name, data in test_splits.items():
        print(f"  {name}: {len(data)} samples  ex: {data[0]}")
    print(f"  max_len: {max_len}")

    all_results = {}
    all_histories = {}
    convergence_iters = {}

    for pos_enc in POS_ENCS:
        for objective in ['ar', 'diffusion']:
            key = f"{objective}_{pos_enc}"
            print(f"\n▶ {key}")

            model, hist, ml, conv_it = train_model(
                objective, tok, train_data, max_len=max_len,
                max_iters=MAX_ITERS, patience=PATIENCE,
                pos_enc=pos_enc, log_interval=500,
            )
            all_histories[key] = hist
            convergence_iters[key] = conv_it
            all_results[key] = {}

            for split_name, test_data in test_splits.items():
                k_val = int(split_name.split('k')[1])
                res = evaluate_parallel(
                    model, tok, test_data, objective, k_val)

                all_results[key][split_name] = res['exact_match']
                all_results[key][f'{split_name}_seg'] = \
                    res['segment_accuracy']

                seg_str = ' '.join(
                    f's{i}={a:.3f}'
                    for i, a in enumerate(res['segment_accuracy']))
                print(f"    {split_name}: "
                      f"{res['exact_match']:.4f}  [{seg_str}]")

                if 'parallelism' in res:
                    par = res['parallelism']
                    all_results[key][f'{split_name}_par'] = par
                    print(f"      τ={par['parallelism_tau']:.3f} "
                          f"(±{par['tau_std']:.3f})  "
                          f"ranks={[f'{r:.1f}' for r in par['segment_mean_ranks']]}")

            save_results(EXP_NAME, all_results, model=model, tag=key)

    all_results['convergence_iters'] = convergence_iters

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  Visualisation
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    figs = {}
    configs = [k for k in all_results
               if isinstance(all_results[k], dict)
               and 'test_k2' in all_results[k]]

    obj_color = {'ar': '#e74c3c', 'diffusion': '#3498db'}
    pe_style  = {'absolute': '-o', 'rope': '--s'}

    # 1) Accuracy vs k  ─────────────────────────────
    ks = sorted(set(K_TRAIN + [K_TEST_OOD]))
    fig, ax = plt.subplots(figsize=(8, 5))
    for cfg in configs:
        obj = 'ar' if 'ar' in cfg else 'diffusion'
        pe  = 'rope' if 'rope' in cfg else 'absolute'
        accs = [all_results[cfg].get(f'test_k{k}', 0) for k in ks]
        ax.plot(ks, accs, pe_style[pe], color=obj_color[obj],
                label=f'{obj}/{pe}', markersize=6)
    ax.axvline(x=max(K_TRAIN), color='gray', ls=':', alpha=0.5,
               label='train max k')
    ax.set_xlabel('k (parallel tasks)'); ax.set_ylabel('Exact Match')
    ax.set_xticks(ks); ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title('Accuracy Scaling with Parallel Tasks')
    fig.tight_layout(); figs['accuracy_vs_k'] = fig

    # 2) Per-segment accuracy at k=4  ───────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(4)
    w = 0.8 / max(len(configs), 1)
    for i, cfg in enumerate(configs):
        segs = all_results[cfg].get('test_k4_seg', [0] * 4)
        obj = 'ar' if 'ar' in cfg else 'diffusion'
        pe  = 'rope' if 'rope' in cfg else 'absolute'
        ax.bar(x + i * w, segs[:4], w, label=f'{obj}/{pe}',
               color=obj_color[obj],
               alpha=0.9 if pe == 'absolute' else 0.5,
               edgecolor='black' if pe == 'rope' else 'none',
               linewidth=0.5)
    ax.set_xlabel('Segment Position'); ax.set_ylabel('Segment Accuracy')
    ax.set_xticks(x + w * (len(configs) - 1) / 2)
    ax.set_xticklabels([f'seg {i}' for i in range(4)])
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)
    ax.set_title('Per-Segment Accuracy (k=4, ID)')
    fig.tight_layout(); figs['segment_accuracy_k4'] = fig

    # 3) Parallelism τ  ─────────────────────────────
    diff_cfgs = [c for c in configs if 'diffusion' in c]
    if diff_cfgs:
        test_keys = [f'test_k{k}' for k in ks if k > 2]
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(test_keys))
        w = 0.8 / max(len(diff_cfgs), 1)
        for i, cfg in enumerate(diff_cfgs):
            pe = 'rope' if 'rope' in cfg else 'absolute'
            vals = [all_results[cfg].get(f'{tk}_par', {})
                    .get('parallelism_tau', 0) for tk in test_keys]
            errs = [all_results[cfg].get(f'{tk}_par', {})
                    .get('tau_std', 0) for tk in test_keys]
            ax.bar(x + i * w, vals, w, yerr=errs,
                   label=f'diffusion/{pe}',
                   color=obj_color['diffusion'],
                   alpha=0.9 if pe == 'absolute' else 0.5,
                   capsize=3)
        ax.axhline(y=0, color='green', ls='--', alpha=0.5,
                   label='parallel (τ=0)')
        ax.axhline(y=1, color='red', ls='--', alpha=0.5,
                   label='sequential (τ=1)')
        ax.set_xlabel('Test Split'); ax.set_ylabel("Kendall's τ")
        ax.set_xticks(x + w * (len(diff_cfgs) - 1) / 2)
        ax.set_xticklabels(test_keys)
        ax.set_ylim(-0.3, 1.2)
        ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)
        ax.set_title('Parallelism Index (lower = more parallel)')
        fig.tight_layout(); figs['parallelism_tau'] = fig

    # 4) Segment decode-rank profile (k=4, diffusion)
    for cfg in diff_cfgs:
        par = all_results[cfg].get('test_k4_par')
        if par and 'segment_mean_ranks' in par:
            ranks = par['segment_mean_ranks']
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(range(len(ranks)), ranks,
                   color='#3498db', alpha=0.7)
            for bi, v in enumerate(ranks):
                ax.text(bi, v + 0.3, f'{v:.1f}', ha='center',
                        fontsize=9)
            ax.set_xlabel('Segment Index')
            ax.set_ylabel('Mean Decode Step')
            pe = 'rope' if 'rope' in cfg else 'absolute'
            ax.set_title(
                f'Segment Decode Order (k=4) — diffusion/{pe}')
            ax.grid(axis='y', alpha=0.3)
            fig.tight_layout()
            figs[f'decode_rank_{cfg}'] = fig

    save_results(EXP_NAME, all_results, figures=figs)

    # ── Summary ─────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    header = f"{'Config':<25}"
    for k in ks:
        header += f"  {'k='+str(k):>7}"
    header += f"  {'τ(k4)':>7}  {'τ(k'+str(K_TEST_OOD)+')':>7}"
    print(header)
    print("-" * len(header))

    for cfg in configs:
        r = all_results[cfg]
        vals = ''.join(f"  {r.get(f'test_k{k}', 0):>7.4f}"
                       for k in ks)
        tau4 = r.get('test_k4_par', {}).get('parallelism_tau')
        tau8 = r.get(f'test_k{K_TEST_OOD}_par', {}).get(
            'parallelism_tau')
        t4 = f"{tau4:.3f}" if tau4 is not None else "    —"
        t8 = f"{tau8:.3f}" if tau8 is not None else "    —"
        print(f"{cfg:<25}{vals}  {t4:>7}  {t8:>7}")

    # Segment detail for k=4
    print("\nSEGMENT ACCURACY (k=4):")
    for cfg in configs:
        segs = all_results[cfg].get('test_k4_seg', [])
        seg_str = '  '.join(f's{i}={v:.3f}'
                            for i, v in enumerate(segs))
        print(f"  {cfg:<25} {seg_str}")

    plt.show()
    return all_results


if __name__ == '__main__':
    run()
