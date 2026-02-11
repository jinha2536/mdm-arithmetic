"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Experiment 1 — Addition: AR vs Masked Diffusion
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Colab:  %run experiments/exp_addition.py

  Configs trained:
    {ar, diffusion} × {plain, reverse, scratchpad} × {absolute, rope}
  Test splits:
    test_id (3d, digits 0-7)
    test_ood_digit (3d, digits 0-9)
    test_ood_length (4d, digits 0-9)

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
TRAIN_DIGITS = [0, 1, 2, 3, 4, 6, 7, 8, 9]  # exclude 5 (mid-range, avoids edge bias)
ALL_DIGITS   = list(range(10))
HELD_OUT     = {5}
POS_ENCS     = ['absolute', 'rope']
FORMATS      = ['plain', 'reverse', 'scratchpad']

# ── Data generation ─────────────────────────────────

def _pad(n, w):
    return str(n).zfill(w)

def _operand(nd, digits, rng):
    return int(''.join(str(rng.choice(digits)) for _ in range(nd)))

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

def gen_addition(nd, n, digits, fmt, seed=42):
    rng = random.Random(seed)
    fn = FMT_FN[fmt]
    seen, out = set(), []
    for _ in range(n * 10):
        if len(out) >= n: break
        a, b = _operand(nd, digits, rng), _operand(nd, digits, rng)
        if (a, b) in seen: continue
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
        'train':              gen_addition(3, N_TRAIN, TRAIN_DIGITS, fmt, 42),
        'test_id':            gen_addition(3, N_TEST,  TRAIN_DIGITS, fmt, 1042),
        'test_ood_digit':     gen_addition(3, N_TEST,  ALL_DIGITS,   fmt, 2042),
        # Length OOD with IN-distribution digits → isolates length effect
        'test_ood_length':    gen_addition(4, N_TEST,  TRAIN_DIGITS, fmt, 3042),
        # Both OOD (hardest)
        'test_ood_both':      gen_addition(4, N_TEST,  ALL_DIGITS,   fmt, 4042),
    }

TEST_SPLITS = ['test_id', 'test_ood_digit', 'test_ood_length', 'test_ood_both']


def _has_ood_digit(s, held_out=HELD_OUT):
    """Check if a sample's operands contain held-out digits."""
    prefix = s.split('=')[0]  # e.g. "123+456"
    return any(int(c) in held_out for c in prefix if c.isdigit())



def build_tok(fmt):
    chars = list('0123456789+=')
    if fmt == 'scratchpad': chars.extend(['C', 'S', '>'])
    return CharTokenizer(chars, {'mask': 'M', 'pad': 'P'})


def breakdown_ood_digit_from_eval(test_samples, eval_result, get_ans_fn):
    """
    Break down OOD-digit eval into:
      - samples where operands use only train digits (no 5)
      - samples where operands contain held-out digit (5)
    """
    per_correct = eval_result['per_sample_correct']
    id_ok, ood_ok = [], []
    for s, ok in zip(test_samples, per_correct):
        if _has_ood_digit(s):
            ood_ok.append(ok)
        else:
            id_ok.append(ok)
    return {
        'no_held_out': sum(id_ok) / max(len(id_ok), 1),
        'has_held_out': sum(ood_ok) / max(len(ood_ok), 1),
        'n_no_held_out': len(id_ok),
        'n_has_held_out': len(ood_ok),
    }


# ── Main ────────────────────────────────────────────

def run():
    print("=" * 70)
    print("  EXP 1: Addition — AR vs Diffusion × Format × PosEnc")
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

                    # Breakdown for digit OOD: what fraction of errors
                    # are due to unseen digits vs genuine failure?
                    if sp == 'test_ood_digit':
                        bd = breakdown_ood_digit_from_eval(
                            splits[sp], res, get_ans)
                        all_results[key]['digit_breakdown'] = bd
                        print(f"      → no 5 in ops:  {bd['no_held_out']:.4f} "
                              f"({bd['n_no_held_out']} samples)")
                        print(f"      → has 5 in ops: {bd['has_held_out']:.4f} "
                              f"({bd['n_has_held_out']} samples)")

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
    split_order = TEST_SPLITS
    labels = ['ID (3d, no 5)', 'OOD Digit (3d, all)',
              'OOD Length (4d, no 5)', 'OOD Both (4d, all)']
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    for idx, (sp, lb) in enumerate(zip(split_order, labels)):
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
    ax.set_xlabel('Absolute PE (4d, no 5)')
    ax.set_ylabel('RoPE (4d, no 5)')
    ax.set_title('Pure Length Generalisation: RoPE vs Absolute')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
    fig.tight_layout(); figs['rope_vs_abs'] = fig

    save_results(EXP_NAME, all_results, figures=figs)

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Config':<35} {'ID':>8} {'OOD-d':>8} {'OOD-L':>8} {'Both':>8} {'conv':>8}")
    print("-" * 75)
    for k in configs:
        r = all_results[k]
        print(f"{k:<35} {r.get('test_id',0):>8.4f} "
              f"{r.get('test_ood_digit',0):>8.4f} "
              f"{r.get('test_ood_length',0):>8.4f} "
              f"{r.get('test_ood_both',0):>8.4f} "
              f"{convergence_iters.get(k,'?'):>8}")

    # Print digit breakdown
    print("\nDigit OOD Breakdown (held-out digit: 5):")
    for k in configs:
        bd = all_results[k].get('digit_breakdown')
        if bd:
            print(f"  {k}: no_5={bd['no_held_out']:.4f} ({bd['n_no_held_out']}), "
                  f"has_5={bd['has_held_out']:.4f} ({bd['n_has_held_out']})")

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
