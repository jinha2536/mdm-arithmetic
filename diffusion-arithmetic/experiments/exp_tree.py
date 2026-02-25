"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Experiment 2 — Parallel Independent Additions
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Colab:  %run experiments/exp_tree.py

  Tests whether diffusion models can decode independent output
  segments in parallel without accuracy loss.

  Format:  a1+b1=r1|a2+b2=r2|...|ak+bk=rk
  Example: 328+242=0570|854+204=1058

  Each problem–answer pair is self-contained. For AR, the model
  generates each answer immediately after seeing its problem.
  For diffusion, all answer positions are decoded (along with
  interleaved problem text in the answer region).

  3-digit operands → carry chains make each segment non-trivial.

  Decode strategies tested (same trained model):
    seq       — sequential confidence (1 token/step, fair vs AR)
    par_seg   — parallel_confidence, k=ANS_WIDTH (1 segment/step)
    par_all   — parallel_confidence, k=ans_len (all at once)

  Core question: if segments are independent, par_seg and par_all
  should match seq accuracy while using fewer forward passes.
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
ND          = 3           # digits per operand (100–999)
ANS_WIDTH   = 4           # digits per answer (zero-padded; max 1998)
KS          = [2, 4, 6, 8]  # parallel tasks (all in training)
N_TRAIN     = 20_000
N_TEST      = 2_000
MAX_ITERS   = 15_000
PATIENCE    = 2_000
POS_ENCS    = ['absolute', 'rope']

# Each segment: "XXX+YYY=ZZZZ" = 3+1+3+1+4 = 12 chars
SEG_LEN     = 2 * ND + 2 + ANS_WIDTH     # 12

# Diffusion decode strategies to compare
# (policy, parallel_k, label)
#   parallel_k: tokens per step (clamped to actual decode length)
DIFF_STRATEGIES = [
    ('confidence',          1,          'seq'),
    ('parallel_confidence', ANS_WIDTH,  'par_seg'),
    ('parallel_confidence', 9999,       'par_all'),   # clamped to n_decode
]


# ── Data generation ─────────────────────────────────

def gen_sample(k, nd, rng):
    """
    Generate one sample: k independent nd-digit additions.
    Format: a1+b1=r1|a2+b2=r2|...|ak+bk=rk
    """
    lo, hi = 10**(nd - 1), 10**nd - 1
    segments = []
    for _ in range(k):
        a, b = rng.randint(lo, hi), rng.randint(lo, hi)
        ans = str(a + b).zfill(ANS_WIDTH)
        segments.append(f"{a}+{b}={ans}")
    return "|".join(segments)


def gen_data_mixed(ks, n, seed=42):
    """Generate n samples with k chosen randomly from ks."""
    rng = random.Random(seed)
    return [gen_sample(rng.choice(ks), ND, rng) for _ in range(n)]


def gen_data_fixed_k(k, n, seed=42):
    """Generate n samples, all with the same k."""
    rng = random.Random(seed)
    return [gen_sample(k, ND, rng) for _ in range(n)]


def get_answer(s):
    """
    Extract the answer region (everything after the first '=').
    For '328+242=0570|854+204=1058' → '0570|854+204=1058'
    """
    return s.split('=', 1)[1]


def get_segment_answers(s, k):
    """
    Extract individual answers from the interleaved format.
    '328+242=0570|854+204=1058' → ['0570', '1058']
    """
    segments = s.split('|')
    return [seg.split('=')[1] for seg in segments]


def answer_offsets_in_region(k):
    """
    Compute where each segment's answer digits fall within the
    answer region (everything after the first '=').

    Answer region for k=2: '0570|854+204=1058'
      seg0 answer at offset 0..3 (ANS_WIDTH chars)
      seg1 answer at offset 4+8=12..15 (after '|854+204=')

    Returns list of (start, end) pairs into the answer region string.
    """
    offsets = []
    # First segment answer starts at offset 0
    offsets.append((0, ANS_WIDTH))
    # Subsequent segments: each preceded by '|' + problem + '='
    # problem = 'XXX+YYY' = 2*ND+1 chars, plus '|' and '='
    inter = 1 + (2 * ND + 1) + 1  # |XXX+YYY=
    pos = ANS_WIDTH
    for _ in range(1, k):
        pos += inter
        offsets.append((pos, pos + ANS_WIDTH))
        pos += ANS_WIDTH
    return offsets


def build_tok():
    chars = list('0123456789+=|')
    return CharTokenizer(chars, {'mask': 'M', 'pad': 'P'})


# ── Analysis helpers ────────────────────────────────

def _kendall_tau(x, y):
    """Kendall's τ for small arrays."""
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


def segment_accuracy(pred_ans_region, gold_ans_region, k):
    """
    Per-segment exact match from the answer region string.
    The answer region is '0570|854+204=1058' (interleaved format).
    Extract answer digits at known offsets.
    """
    offsets = answer_offsets_in_region(k)
    results = []
    for start, end in offsets:
        pred_seg = pred_ans_region[start:end] \
            if end <= len(pred_ans_region) else ''
        gold_seg = gold_ans_region[start:end] \
            if end <= len(gold_ans_region) else ''
        results.append(pred_seg == gold_seg)
    return results


def analyse_parallel_decode(decode_orders, ans_start, k):
    """
    Parallelism metrics from sequential decode orders.
    Uses answer_offsets_in_region to locate each segment's answer
    digit positions within the answer region.
    """
    if decode_orders is None or len(decode_orders) == 0:
        return {}

    N, S = decode_orders.shape

    # Map answer offsets to positions in decode order
    offsets = answer_offsets_in_region(k)
    seg_positions = []
    for start, end in offsets:
        seg_positions.append(list(range(start, end)))

    rel_orders = (decode_orders - ans_start).long().clamp(0, S - 1)
    steps = torch.arange(S).unsqueeze(0).expand(N, S).float()
    pos_rank = torch.zeros(N, S)
    pos_rank.scatter_(1, rel_orders, steps)

    seg_mean = np.zeros(k)
    for i, positions in enumerate(seg_positions):
        valid = [p for p in positions if p < S]
        if valid:
            seg_mean[i] = pos_rank[:, valid].mean().item()

    taus = []
    seg_idx = list(range(k))
    for b in range(N):
        ranks = []
        for ps in seg_positions:
            valid = [p for p in ps if p < S]
            ranks.append(
                pos_rank[b, valid].mean().item() if valid else 0)
        taus.append(_kendall_tau(seg_idx, ranks))

    return {
        'segment_mean_ranks': seg_mean.tolist(),
        'parallelism_tau': float(np.mean(taus)),
        'tau_std': float(np.std(taus)),
        'n_samples': N,
    }


# ── Evaluation ──────────────────────────────────────

def evaluate_parallel(model, tokenizer, test_samples, objective, k,
                      policy='confidence', parallel_k=1,
                      max_len=None, batch_size=128):
    """
    Evaluate with configurable decode strategy.

    For diffusion: masks the ENTIRE region after prefix (not just ans_len).
    The model must decode both answer tokens AND PAD tokens, proving it
    has learned the correct output length. This matches training where
    PAD positions are included in the diffusion loss.

    max_len: training-time total sequence length.
    """
    mask_id = tokenizer.special_ids['mask']
    pad_id = tokenizer.special_ids['pad']
    model.eval()

    ex = test_samples[0]
    ans_len = len(tokenizer.encode(get_answer(ex)))
    ans_start = len(tokenizer.encode(ex.split('=')[0] + '='))

    all_correct, all_orders = [], []
    seg_correct = [[] for _ in range(k)]
    total_steps = 0
    n_batches = 0

    for start in range(0, len(test_samples), batch_size):
        batch = test_samples[start:start + batch_size]
        B = len(batch)
        golds = [get_answer(s) for s in batch]

        penc = [tokenizer.encode(s.split('=')[0] + '=')
                for s in batch]
        pmax = max(len(p) for p in penc)
        pids = torch.full((B, pmax), pad_id, dtype=torch.long)
        for i, e in enumerate(penc):
            pids[i, :len(e)] = torch.tensor(e)

        with torch.no_grad():
            if objective == 'ar':
                gen = generate_ar(model, pids, ans_len, DEVICE)
                pred_ids = gen[:, pmax:pmax + ans_len]
                batch_orders = None
                batch_steps = ans_len
            else:
                # Decode ENTIRE remaining region (answer + PAD)
                n_decode = max_len - pmax if max_len else ans_len
                # Adjust parallel_k for par_all: decode all at once
                pk = min(parallel_k, n_decode)
                gen, _, info = generate_diffusion(
                    model, pids, n_decode, mask_id,
                    policy=policy, greedy=True,
                    parallel_k=pk, device=DEVICE)
                # Extract only the answer portion
                pred_ids = gen[:, pmax:pmax + ans_len]
                batch_orders = info.get('orders')
                batch_steps = info.get('n_steps', n_decode)

        total_steps += batch_steps
        n_batches += 1

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
        'avg_steps': total_steps / max(n_batches, 1),
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
    train_data = gen_data_mixed(KS, N_TRAIN, seed=42)
    test_splits = {}
    for k in KS:
        test_splits[f'test_k{k}'] = gen_data_fixed_k(
            k, N_TEST, seed=1000 + k)

    all_s = train_data + [s for v in test_splits.values() for s in v]
    max_len = max(len(tok.encode(s)) for s in all_s) + 1

    print(f"\n  Config: ND={ND}, KS={KS}")
    print(f"  Train: {len(train_data)} samples (mixed k)")
    for name, data in test_splits.items():
        print(f"  {name}: {len(data)} samples  ex: {data[0]}")
    print(f"  max_len: {max_len}")

    all_results = {}
    all_histories = {}
    convergence_iters = {}

    for pos_enc in POS_ENCS:
        for objective in ['ar', 'diffusion']:
            base_key = f"{objective}_{pos_enc}"
            print(f"\n▶ Training: {base_key}")

            model, hist, ml, conv_it = train_model(
                objective, tok, train_data, max_len=max_len,
                max_iters=MAX_ITERS, patience=PATIENCE,
                pos_enc=pos_enc, log_interval=500,
            )
            all_histories[base_key] = hist
            convergence_iters[base_key] = conv_it

            # Determine which decode strategies to test
            if objective == 'ar':
                strategies = [('confidence', 1, 'ar')]
            else:
                strategies = DIFF_STRATEGIES

            for policy, pk, strat_label in strategies:
                key = f"{objective}_{strat_label}_{pos_enc}"
                print(f"\n  ▸ Evaluating: {key}")
                all_results[key] = {}

                for split_name, test_data in test_splits.items():
                    k_val = int(split_name.split('k')[1])

                    res = evaluate_parallel(
                        model, tok, test_data, objective,
                        k_val, policy=policy, parallel_k=pk,
                        max_len=max_len)

                    all_results[key][split_name] = \
                        res['exact_match']
                    all_results[key][f'{split_name}_seg'] = \
                        res['segment_accuracy']
                    all_results[key][f'{split_name}_nfe'] = \
                        res['avg_steps']

                    seg_str = ' '.join(
                        f's{i}={a:.3f}'
                        for i, a in enumerate(
                            res['segment_accuracy']))
                    print(f"    {split_name}: "
                          f"{res['exact_match']:.4f}  "
                          f"NFE={res['avg_steps']:.0f}  "
                          f"[{seg_str}]")

                    if 'parallelism' in res:
                        par = res['parallelism']
                        all_results[key][f'{split_name}_par'] = par
                        print(
                            f"      τ={par['parallelism_tau']:.3f}"
                            f" (±{par['tau_std']:.3f})")

            save_results(EXP_NAME, all_results, model=model,
                         tag=base_key)

    all_results['convergence_iters'] = convergence_iters

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  Visualisation
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    figs = {}
    configs = [k for k in all_results
               if isinstance(all_results[k], dict)
               and 'test_k2' in all_results[k]]

    # Colour/style by strategy
    STYLE = {
        'ar_ar':           ('#e74c3c', '-o',  'AR'),
        'diffusion_seq':   ('#3498db', '-o',  'Diff (seq)'),
        'diffusion_par_seg': ('#2ecc71', '-s',  'Diff (par_seg)'),
        'diffusion_par_all': ('#9b59b6', '-^',  'Diff (par_all)'),
    }

    def _style(cfg):
        """Extract style key from config name like diffusion_seq_absolute."""
        parts = cfg.rsplit('_', 1)   # split off pos_enc
        return parts[0], parts[1]    # strategy_part, pos_enc

    # 1) Accuracy vs k — main result ────────────────
    for pe in POS_ENCS:
        fig, ax = plt.subplots(figsize=(8, 5))
        for cfg in configs:
            strat, cfg_pe = _style(cfg)
            if cfg_pe != pe:
                continue
            color, ls, label = STYLE.get(strat,
                                          ('#888', '-x', strat))
            accs = [all_results[cfg].get(f'test_k{k}', 0)
                    for k in KS]
            ax.plot(KS, accs, ls, color=color,
                    label=label, markersize=6)
        ax.set_xlabel('k (parallel tasks)')
        ax.set_ylabel('Exact Match')
        ax.set_xticks(KS); ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
        ax.set_title(f'Accuracy vs k — {pe}')
        fig.tight_layout()
        figs[f'accuracy_vs_k_{pe}'] = fig

    # 2) Per-segment accuracy at k=8  ───────────────
    k_main = max(KS)
    for pe in POS_ENCS:
        pe_cfgs = [c for c in configs if _style(c)[1] == pe]
        if not pe_cfgs:
            continue
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(k_main)
        w = 0.8 / max(len(pe_cfgs), 1)
        for i, cfg in enumerate(pe_cfgs):
            strat, _ = _style(cfg)
            color, _, label = STYLE.get(strat,
                                         ('#888', '', strat))
            segs = all_results[cfg].get(
                f'test_k{k_main}_seg', [0] * k_main)
            ax.bar(x + i * w, segs[:k_main], w,
                   label=label, color=color, alpha=0.8)
        ax.set_xlabel('Segment Position')
        ax.set_ylabel('Segment Accuracy')
        ax.set_xticks(x + w * (len(pe_cfgs) - 1) / 2)
        ax.set_xticklabels([f'seg {i}' for i in range(k_main)])
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)
        ax.set_title(f'Per-Segment Accuracy (k={k_main}) — {pe}')
        fig.tight_layout()
        figs[f'segment_accuracy_k{k_main}_{pe}'] = fig

    # 3) NFE vs Accuracy scatter ────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    for cfg in configs:
        strat, pe = _style(cfg)
        color, _, label = STYLE.get(strat,
                                     ('#888', '', strat))
        marker = 'o' if pe == 'absolute' else 's'
        for k_val in KS:
            acc = all_results[cfg].get(f'test_k{k_val}', 0)
            nfe = all_results[cfg].get(
                f'test_k{k_val}_nfe', 0)
            ax.scatter(nfe, acc, color=color, marker=marker,
                       s=40 + k_val * 8, alpha=0.7)
    # Manual legend
    from matplotlib.lines import Line2D
    handles = []
    for skey, (color, _, label) in STYLE.items():
        handles.append(Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=color, label=label,
                              markersize=8))
    handles.append(Line2D([0], [0], marker='o', color='w',
                          markerfacecolor='gray', label='absolute',
                          markersize=8))
    handles.append(Line2D([0], [0], marker='s', color='w',
                          markerfacecolor='gray', label='rope',
                          markersize=8))
    ax.legend(handles=handles, fontsize=8)
    ax.set_xlabel('NFE (forward passes)')
    ax.set_ylabel('Exact Match')
    ax.grid(alpha=0.3)
    ax.set_title('NFE vs Accuracy (size ∝ k)')
    fig.tight_layout(); figs['nfe_vs_accuracy'] = fig

    # 4) Parallelism τ (sequential only) ────────────
    seq_cfgs = [c for c in configs if 'seq' in c]
    if seq_cfgs:
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(KS))
        w = 0.8 / max(len(seq_cfgs), 1)
        for i, cfg in enumerate(seq_cfgs):
            _, pe = _style(cfg)
            vals = [all_results[cfg].get(f'test_k{k}_par', {})
                    .get('parallelism_tau', 0) for k in KS]
            errs = [all_results[cfg].get(f'test_k{k}_par', {})
                    .get('tau_std', 0) for k in KS]
            ax.bar(x + i * w, vals, w, yerr=errs,
                   label=f'diffusion/{pe}',
                   color='#3498db',
                   alpha=0.9 if pe == 'absolute' else 0.5,
                   capsize=3)
        ax.axhline(y=0, color='green', ls='--', alpha=0.5,
                   label='parallel (τ=0)')
        ax.axhline(y=1, color='red', ls='--', alpha=0.5,
                   label='sequential (τ=1)')
        ax.set_xlabel('k'); ax.set_ylabel("Kendall's τ")
        ax.set_xticks(x + w * (len(seq_cfgs) - 1) / 2)
        ax.set_xticklabels([f'k={k}' for k in KS])
        ax.set_ylim(-0.3, 1.2)
        ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)
        ax.set_title('Parallelism Index — Sequential Confidence')
        fig.tight_layout(); figs['parallelism_tau'] = fig

    save_results(EXP_NAME, all_results, figures=figs)

    # ── Summary ─────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    header = f"{'Config':<30}"
    for k in KS:
        header += f"  {'k='+str(k):>7}"
    header += f"  {'NFE(k'+str(max(KS))+')':>9}"
    print(header)
    print("-" * len(header))

    for cfg in sorted(configs):
        r = all_results[cfg]
        vals = ''.join(
            f"  {r.get(f'test_k{k}', 0):>7.4f}" for k in KS)
        nfe = r.get(f'test_k{max(KS)}_nfe', 0)
        print(f"{cfg:<30}{vals}  {nfe:>9.0f}")

    # Segment detail for k=max
    print(f"\nSEGMENT ACCURACY (k={k_main}):")
    for cfg in sorted(configs):
        segs = all_results[cfg].get(
            f'test_k{k_main}_seg', [])
        seg_str = '  '.join(
            f's{i}={v:.3f}' for i, v in enumerate(segs))
        print(f"  {cfg:<30} {seg_str}")

    plt.show()
    return all_results


if __name__ == '__main__':
    run()
