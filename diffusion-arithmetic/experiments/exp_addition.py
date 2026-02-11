"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Experiment 1 — Addition: AR vs Masked Diffusion
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Colab:  %run experiments/exp_addition.py

  Configs trained:
    {ar, diffusion} × {plain, reverse, scratchpad} × {absolute, rope}
  Test splits:
    test_id          (3d, operands from TRAIN_OPS)
    test_ood_number  (3d, ≥1 operand from HELD_OUT_OPS)
    test_ood_length  (4d, any operands)

  OOD design (matches Lee et al. 2023):
    Number-level holdout — specific operand VALUES excluded from
    training, not specific digits.  All 10 digits appear in training
    (via other operands and sums).  This tests LRMC-style
    generalisation: can the model compute a+b for unseen (a,b)?

  Key controls:
    - Greedy (argmax) decoding for BOTH objectives → fair comparison
    - Convergence-based early stopping
    - RoPE option for length generalisation
    - Scratchpad decode order analysis for diffusion
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                if '__file__' in dir() else '.')

import random, torch
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('Agg')
import numpy as np

from core.tokenizer import CharTokenizer
from core.train_utils import (
    mount_drive, save_results, train_model, evaluate,
    analyse_decode_order, DEVICE,
)

EXP_NAME = 'exp_addition'

# ── Config ──────────────────────────────────────────
N_TRAIN      = 10_000
N_TEST       = 2_000
MAX_ITERS    = 15_000
PATIENCE     = 2_000

# Number-level OOD: hold out 10% of 3-digit operand values (0-999).
# All 10 digits still appear in training via other operands and in
# answer tokens.  This mirrors Lee et al. (2023) Table 4.
_rng_split = random.Random(0)
_ALL_3D = list(range(1000))
_rng_split.shuffle(_ALL_3D)
N_HELD_OUT    = 100                          # 10% of operand space
HELD_OUT_OPS  = frozenset(_ALL_3D[:N_HELD_OUT])
TRAIN_OPS_3D  = sorted(set(_ALL_3D[N_HELD_OUT:]))
ALL_OPS_3D    = sorted(_ALL_3D)
HELD_OUT_LIST = sorted(HELD_OUT_OPS)

POS_ENCS     = ['absolute', 'rope']
FORMATS      = ['plain', 'reverse', 'scratchpad']

# ── Data generation ─────────────────────────────────

def _pad(n, w):
    return str(n).zfill(w)

def _fmt_plain(a, b, nd):
    return f"{_pad(a,nd)}+{_pad(b,nd)}={_pad(a+b,nd+1)}"

def _fmt_reverse(a, b, nd):
    return f"{_pad(a,nd)}+{_pad(b,nd)}={_pad(a+b,nd+1)[::-1]}"

def _fmt_scratchpad(a, b, nd):
    a_s, b_s = _pad(a, nd), _pad(b, nd)
    steps, carry = [], 0
    for i in range(nd - 1, -1, -1):
        s = int(a_s[i]) + int(b_s[i]) + carry
        carry, digit = divmod(s, 10)
        steps.append(f"C{carry}S{digit}")
    return f"{a_s}+{b_s}={''.join(steps)}>>{_pad(a+b, nd+1)}"

FMT_FN = {'plain': _fmt_plain, 'reverse': _fmt_reverse,
           'scratchpad': _fmt_scratchpad}


def _gen_from_set(nd, n, operand_list, fmt, seed):
    """Both operands sampled from operand_list."""
    rng = random.Random(seed)
    fn = FMT_FN[fmt]
    seen, out = set(), []
    for _ in range(n * 10):
        if len(out) >= n:
            break
        a, b = rng.choice(operand_list), rng.choice(operand_list)
        if (a, b) in seen:
            continue
        seen.add((a, b))
        out.append(fn(a, b, nd))
    return out


def _gen_ood_number(nd, n, held_list, all_list, fmt, seed):
    """At least one operand from held_list."""
    rng = random.Random(seed)
    fn = FMT_FN[fmt]
    seen, out = set(), []
    for _ in range(n * 10):
        if len(out) >= n:
            break
        # guarantee at least one held-out operand
        if rng.random() < 0.5:
            a, b = rng.choice(held_list), rng.choice(all_list)
        else:
            a, b = rng.choice(all_list), rng.choice(held_list)
        if (a, b) in seen:
            continue
        seen.add((a, b))
        out.append(fn(a, b, nd))
    return out


def _gen_any(nd, n, fmt, seed):
    """Operands sampled uniformly from [0, 10^nd)."""
    rng = random.Random(seed)
    fn = FMT_FN[fmt]
    mx = 10 ** nd
    seen, out = set(), []
    for _ in range(n * 10):
        if len(out) >= n:
            break
        a, b = rng.randint(0, mx - 1), rng.randint(0, mx - 1)
        if (a, b) in seen:
            continue
        seen.add((a, b))
        out.append(fn(a, b, nd))
    return out


def get_answer(s, fmt):
    if fmt == 'scratchpad': return s.split('>>')[-1]
    return s.split('=')[1]

def get_full_answer(s):
    return s.split('=', 1)[1]


def build_splits(fmt):
    return {
        'train':            _gen_from_set(3, N_TRAIN,  TRAIN_OPS_3D, fmt, 42),
        'test_id':          _gen_from_set(3, N_TEST,   TRAIN_OPS_3D, fmt, 1042),
        'test_ood_number':  _gen_ood_number(3, N_TEST, HELD_OUT_LIST, ALL_OPS_3D, fmt, 2042),
        'test_ood_length':  _gen_any(4, N_TEST, fmt, 3042),
    }

TEST_SPLITS = ['test_id', 'test_ood_number', 'test_ood_length']


def _parse_operands(s):
    """Extract (a, b) as ints from '123+456=...'."""
    prefix = s.split('=')[0]
    parts = prefix.split('+')
    return int(parts[0]), int(parts[1])


def breakdown_ood_number(test_samples, eval_result):
    """
    Break down test_ood_number into:
      - one_held:  exactly one operand in HELD_OUT_OPS
      - both_held: both operands in HELD_OUT_OPS
    """
    per_correct = eval_result['per_sample_correct']
    one_ok, both_ok = [], []
    for s, ok in zip(test_samples, per_correct):
        a, b = _parse_operands(s)
        a_in = a in HELD_OUT_OPS
        b_in = b in HELD_OUT_OPS
        if a_in and b_in:
            both_ok.append(ok)
        else:
            one_ok.append(ok)
    return {
        'one_held':   sum(one_ok)  / max(len(one_ok), 1),
        'both_held':  sum(both_ok) / max(len(both_ok), 1),
        'n_one':  len(one_ok),
        'n_both': len(both_ok),
    }


def build_tok(fmt):
    chars = list('0123456789+=')
    if fmt == 'scratchpad': chars.extend(['C', 'S', '>'])
    return CharTokenizer(chars, {'mask': 'M', 'pad': 'P'})


# ── Main ────────────────────────────────────────────

def run():
    print("=" * 70)
    print("  EXP 1: Addition — AR vs Diffusion × Format × PosEnc")
    print(f"  Number-level OOD: {N_HELD_OUT} operand values held out")
    print("=" * 70)
    mount_drive()
    torch.manual_seed(42)

    all_results = {}
    all_histories = {}
    convergence_iters = {}
    scratchpad_analysis = {}

    for pos_enc in POS_ENCS:
        for objective in ['ar', 'diffusion']:
            for fmt in FORMATS:
                key = f"{objective}_{fmt}_{pos_enc}"
                print(f"\n▶ {key}")

                splits = build_splits(fmt)
                tok = build_tok(fmt)
                print(f"  ex: {splits['train'][0]}")

                all_s = [s for v in splits.values() for s in v]
                max_len = max(len(tok.encode(s)) for s in all_s) + 1

                model, hist, ml, conv_it = train_model(
                    objective, tok, splits['train'], max_len=max_len,
                    max_iters=MAX_ITERS, patience=PATIENCE,
                    pos_enc=pos_enc, log_interval=500,
                )
                all_histories[key] = hist
                convergence_iters[key] = conv_it

                get_ans = lambda s, f=fmt: get_answer(s, f)
                all_results[key] = {}

                for sp in TEST_SPLITS:
                    res = evaluate(
                        model, tok, splits[sp], objective, get_ans,
                        get_full_answer, policy='confidence', greedy=True,
                    )
                    all_results[key][sp] = res['exact_match']
                    print(f"    {sp}: {res['exact_match']:.4f}")

                    # Breakdown for number OOD
                    if sp == 'test_ood_number':
                        bd = breakdown_ood_number(splits[sp], res)
                        all_results[key]['number_breakdown'] = bd
                        print(f"      → one held-out:  {bd['one_held']:.4f} "
                              f"({bd['n_one']} samples)")
                        print(f"      → both held-out: {bd['both_held']:.4f} "
                              f"({bd['n_both']} samples)")

                    # Scratchpad decode order analysis
                    if objective == 'diffusion' and fmt == 'scratchpad' \
                            and sp == 'test_id' and 'decode_orders' in res:
                        ex = splits[sp][0]
                        ans_start_pos = len(tok.encode(ex.split('=')[0])) + 1
                        sp_end = ans_start_pos + len(tok.encode(
                            ex.split('=', 1)[1].split('>>')[0] + '>>'))
                        total = len(tok.encode(ex))
                        sa = analyse_decode_order(
                            res['decode_orders'], ans_start_pos, sp_end, total)
                        scratchpad_analysis[key] = sa
                        print(f"      scratchpad_first_ratio: "
                              f"{sa.get('scratchpad_first_ratio', 'N/A')}")

                # Print diagnostic examples for first format
                if fmt == FORMATS[0] and pos_enc == POS_ENCS[0]:
                    print(f"    --- Diagnostic (first 3 test_id) ---")
                    for i in range(min(3, len(splits['test_id']))):
                        s = splits['test_id'][i]
                        print(f"      {s}")

                save_results(EXP_NAME, all_results, model=model, tag=key)

    # ── Summary table ──
    all_results['convergence_iters'] = convergence_iters
    all_results['scratchpad_analysis'] = {
        k: {kk: float(vv) if isinstance(vv, (int, float)) else vv
            for kk, vv in v.items()}
        for k, v in scratchpad_analysis.items()
    }

    # ── Visualization ──
    figs = {}
    configs = [k for k in all_results
               if isinstance(all_results[k], dict) and 'test_id' in all_results[k]]

    # 1) Training curves
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for pe_idx, pe in enumerate(POS_ENCS):
        ax = axes[pe_idx]
        for k, h in all_histories.items():
            if k.endswith(pe):
                ax.plot(h['iter'], h['loss'], label=k.replace(f'_{pe}', ''),
                        alpha=0.8)
        ax.set_xlabel('Iteration'); ax.set_ylabel('Loss')
        ax.set_title(f'Training Loss ({pe})'); ax.legend(fontsize=6)
        ax.grid(alpha=0.3)
    fig.tight_layout(); figs['training_curves'] = fig

    # 2) Convergence iterations
    fig, ax = plt.subplots(figsize=(10, 5))
    keys = [k for k in convergence_iters if k in configs]
    vals = [convergence_iters[k] for k in keys]
    colors = ['#e74c3c' if 'ar' in k else '#3498db' for k in keys]
    ax.barh(range(len(keys)), vals, color=colors, alpha=0.85)
    ax.set_yticks(range(len(keys))); ax.set_yticklabels(keys, fontsize=7)
    ax.set_xlabel('Best Iteration'); ax.set_title('Convergence Point')
    ax.invert_yaxis(); ax.grid(axis='x', alpha=0.3)
    fig.tight_layout(); figs['convergence'] = fig

    # 3) Accuracy: grouped by split
    labels = ['ID (3d, train ops)', 'OOD Number (3d, held-out ops)',
              'OOD Length (4d)']
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for idx, (sp, lb) in enumerate(zip(TEST_SPLITS, labels)):
        ax = axes[idx]
        vals = [all_results[k].get(sp, 0) for k in configs]
        colors = ['#e74c3c' if 'ar' in k else '#3498db' for k in configs]
        hatches = ['///' if 'rope' in k else '' for k in configs]
        bars = ax.bar(range(len(configs)), vals, color=colors, alpha=0.85)
        for bar, h in zip(bars, hatches):
            bar.set_hatch(h)
        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels(configs, fontsize=5, rotation=45, ha='right')
        ax.set_ylabel('Exact Match'); ax.set_title(lb); ax.set_ylim(0, 1.05)
        ax.grid(axis='y', alpha=0.3)
    fig.suptitle('Exp 1: Addition Accuracy (hatched = RoPE)', fontsize=13, y=1.02)
    fig.tight_layout(); figs['accuracy'] = fig

    # 4) RoPE vs Absolute on length OOD
    fig, ax = plt.subplots(figsize=(10, 5))
    for fmt in FORMATS:
        for obj in ['ar', 'diffusion']:
            abs_key = f"{obj}_{fmt}_absolute"
            rope_key = f"{obj}_{fmt}_rope"
            abs_val = all_results.get(abs_key, {}).get('test_ood_length', 0)
            rope_val = all_results.get(rope_key, {}).get('test_ood_length', 0)
            ax.scatter(abs_val, rope_val, s=80,
                       marker='o' if obj == 'ar' else 's',
                       label=f"{obj}_{fmt}")
            ax.annotate(f"{obj}_{fmt}", (abs_val, rope_val),
                        fontsize=6, textcoords="offset points", xytext=(3, 3))
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.set_xlabel('Absolute PE (4d)')
    ax.set_ylabel('RoPE (4d)')
    ax.set_title('Length Generalisation: RoPE vs Absolute')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
    fig.tight_layout(); figs['rope_vs_abs'] = fig

    save_results(EXP_NAME, all_results, figures=figs)

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Config':<35} {'ID':>8} {'OOD-Num':>8} {'OOD-Len':>8} {'conv':>8}")
    print("-" * 70)
    for k in configs:
        r = all_results[k]
        print(f"{k:<35} {r.get('test_id',0):>8.4f} "
              f"{r.get('test_ood_number',0):>8.4f} "
              f"{r.get('test_ood_length',0):>8.4f} "
              f"{convergence_iters.get(k,'?'):>8}")

    # Print number OOD breakdown
    print(f"\nNumber OOD Breakdown ({N_HELD_OUT} held-out operand values):")
    for k in configs:
        bd = all_results[k].get('number_breakdown')
        if bd:
            print(f"  {k}: one_held={bd['one_held']:.4f} ({bd['n_one']}), "
                  f"both_held={bd['both_held']:.4f} ({bd['n_both']})")

    if scratchpad_analysis:
        print("\nScratchpad Decode Order (diffusion):")
        for k, v in scratchpad_analysis.items():
            print(f"  {k}: scratchpad_first={v.get('scratchpad_first_ratio', 'N/A'):.3f}, "
                  f"avg_sp_rank={v.get('avg_scratchpad_rank', 'N/A'):.1f}, "
                  f"avg_final_rank={v.get('avg_final_rank', 'N/A'):.1f}")

    plt.show()
    return all_results


if __name__ == '__main__':
    run()
