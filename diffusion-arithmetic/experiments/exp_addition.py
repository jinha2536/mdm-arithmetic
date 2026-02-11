"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Experiment 1 — Addition: AR vs Masked Diffusion
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Colab:  %run experiments/exp_addition.py

  Multi-digit scaling: 3d, 5d, 7d to find difficulty frontier.
  Number-level OOD at 3d (paper comparison).
  Length OOD at each scale.

  Fixation order analysis: which answer digit does diffusion decode
  first?  Does it correlate with carry positions?

  Secondary: digit-position exclusion (Appendix B.2.1).
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                if '__file__' in dir() else '.')

import random, torch
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('Agg')
import numpy as np
from collections import defaultdict

from core.tokenizer import CharTokenizer
from core.train_utils import (
    mount_drive, save_results, train_model, evaluate,
    analyse_decode_order, DEVICE,
)

EXP_NAME = 'exp_addition'

# ── Config ──────────────────────────────────────────
MAX_ITERS    = 15_000
PATIENCE     = 2_000

# Multi-digit configs: (n_digits, n_train, n_test)
DIGIT_CONFIGS = [
    (3,  10_000, 2_000),
    (5,  30_000, 2_000),
    (7,  50_000, 2_000),
]

# Number-level OOD only for 3d (paper comparison)
_rng_split = random.Random(0)
_ALL_3D = list(range(1000))
_rng_split.shuffle(_ALL_3D)
N_HELD_OUT    = 100
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
    rng = random.Random(seed)
    fn = FMT_FN[fmt]
    seen, out = set(), []
    for _ in range(n * 10):
        if len(out) >= n: break
        a, b = rng.choice(operand_list), rng.choice(operand_list)
        if (a, b) in seen: continue
        seen.add((a, b))
        out.append(fn(a, b, nd))
    return out


def _gen_ood_number(nd, n, held_list, all_list, fmt, seed):
    rng = random.Random(seed)
    fn = FMT_FN[fmt]
    seen, out = set(), []
    for _ in range(n * 10):
        if len(out) >= n: break
        if rng.random() < 0.5:
            a, b = rng.choice(held_list), rng.choice(all_list)
        else:
            a, b = rng.choice(all_list), rng.choice(held_list)
        if (a, b) in seen: continue
        seen.add((a, b))
        out.append(fn(a, b, nd))
    return out


def _gen_any(nd, n, fmt, seed):
    rng = random.Random(seed)
    fn = FMT_FN[fmt]
    mx = 10 ** nd
    seen, out = set(), []
    for _ in range(n * 10):
        if len(out) >= n: break
        a, b = rng.randint(0, mx - 1), rng.randint(0, mx - 1)
        if (a, b) in seen: continue
        seen.add((a, b))
        out.append(fn(a, b, nd))
    return out


def get_answer(s, fmt):
    if fmt == 'scratchpad': return s.split('>>')[-1]
    return s.split('=')[1]

def get_full_answer(s):
    return s.split('=', 1)[1]


def build_splits_3d(fmt, n_train, n_test):
    """3d with number-level OOD (paper comparison)."""
    return {
        'train':            _gen_from_set(3, n_train, TRAIN_OPS_3D, fmt, 42),
        'test_id':          _gen_from_set(3, n_test,  TRAIN_OPS_3D, fmt, 1042),
        'test_ood_number':  _gen_ood_number(3, n_test, HELD_OUT_LIST, ALL_OPS_3D, fmt, 2042),
        'test_ood_length':  _gen_any(4, n_test, fmt, 3042),
    }


def build_splits_nd(nd, fmt, n_train, n_test):
    """nd with ID + length OOD only (for 5d, 7d)."""
    return {
        'train':            _gen_any(nd,   n_train, fmt, 42),
        'test_id':          _gen_any(nd,   n_test,  fmt, 1042),
        'test_ood_length':  _gen_any(nd+2, n_test,  fmt, 3042),
    }


def _parse_operands(s):
    prefix = s.split('=')[0]
    parts = prefix.split('+')
    return int(parts[0]), int(parts[1])


def breakdown_ood_number(test_samples, eval_result):
    per_correct = eval_result['per_sample_correct']
    one_ok, both_ok = [], []
    for s, ok in zip(test_samples, per_correct):
        a, b = _parse_operands(s)
        if a in HELD_OUT_OPS and b in HELD_OUT_OPS:
            both_ok.append(ok)
        else:
            one_ok.append(ok)
    return {
        'one_held':  sum(one_ok)  / max(len(one_ok), 1),
        'both_held': sum(both_ok) / max(len(both_ok), 1),
        'n_one': len(one_ok), 'n_both': len(both_ok),
    }


def build_tok(fmt):
    chars = list('0123456789+=')
    if fmt == 'scratchpad': chars.extend(['C', 'S', '>'])
    return CharTokenizer(chars, {'mask': 'M', 'pad': 'P'})


# ── Fixation order analysis ──────────────────────────
# For diffusion: track which answer digit position gets decoded at
# each step.  Answer positions are indexed 0=MSB to nd=LSB.
# We also annotate carry positions to test carry-fixation correlation.

def analyse_fixation_order(test_samples, eval_result, nd, fmt, tok):
    """
    Analyse which answer-digit positions are fixed first by diffusion.

    Returns:
        mean_rank[i]: average decode step at which answer position i
                      was fixed (0=MSB, nd=LSB of (nd+1)-digit answer)
        carry_corr:   Spearman correlation between carry presence
                      and fixation rank
    """
    if 'decode_orders' not in eval_result:
        return None

    orders = eval_result['decode_orders']  # list of list of positions
    ans_len = nd + 1  # answer has nd+1 digits

    # Find where the answer starts in token sequence
    ex = test_samples[0]
    if fmt == 'scratchpad':
        # answer is after ">>"
        before_ans = ex.split('>>')[0] + '>>'
        ans_start = len(tok.encode(before_ans))
    else:
        # answer is after "="
        before_ans = ex.split('=')[0] + '='
        ans_start = len(tok.encode(before_ans))

    # For each sample, extract decode rank for each answer position
    rank_by_pos = defaultdict(list)  # pos → list of ranks
    carry_flags = []  # (has_carry_at_pos, rank) pairs

    for idx, (sample_str, order) in enumerate(zip(test_samples, orders)):
        if order is None or len(order) == 0:
            continue

        # Get carry positions from the actual addition
        a, b = _parse_operands(sample_str)
        carries = _compute_carries(a, b, nd)

        # Map decode order to answer positions
        pos_to_rank = {}
        for rank, pos in enumerate(order):
            pos_to_rank[pos] = rank

        for digit_idx in range(ans_len):
            tok_pos = ans_start + digit_idx
            if tok_pos in pos_to_rank:
                rank = pos_to_rank[tok_pos]
                rank_by_pos[digit_idx].append(rank)

                # For carry correlation (skip MSB carry, index from LSB)
                # digit_idx 0=MSB, ans_len-1=LSB
                lsb_idx = ans_len - 1 - digit_idx
                has_carry = carries[lsb_idx] if lsb_idx < len(carries) else 0
                carry_flags.append((has_carry, rank))

    # Mean rank per answer position
    mean_rank = {}
    for pos in range(ans_len):
        if pos in rank_by_pos and len(rank_by_pos[pos]) > 0:
            mean_rank[pos] = np.mean(rank_by_pos[pos])

    # Carry-fixation correlation
    carry_corr = None
    if len(carry_flags) > 10:
        from scipy.stats import spearmanr
        c_arr = np.array(carry_flags)
        r, p = spearmanr(c_arr[:, 0], c_arr[:, 1])
        carry_corr = {'rho': float(r), 'p_value': float(p)}

    return {
        'mean_rank': mean_rank,
        'carry_corr': carry_corr,
        'n_samples': len(orders),
    }


def _compute_carries(a, b, nd):
    """Return list of carry flags, indexed from LSB (index 0)."""
    a_s, b_s = str(a).zfill(nd), str(b).zfill(nd)
    carries = []
    carry = 0
    for i in range(nd - 1, -1, -1):
        s = int(a_s[i]) + int(b_s[i]) + carry
        carry = s // 10
        carries.append(carry)
    carries.append(0)  # no carry into MSB overflow position
    return carries


# ── Main ────────────────────────────────────────────

def run():
    print("=" * 70)
    print("  EXP 1: Addition — Multi-Digit Scaling + Fixation Order")
    print("=" * 70)
    mount_drive()
    torch.manual_seed(42)

    all_results = {}
    all_histories = {}
    convergence_iters = {}
    fixation_data = {}

    for nd, n_train, n_test in DIGIT_CONFIGS:
        print(f"\n{'='*60}")
        print(f"  {nd}-digit addition (train={n_train}, test={n_test})")
        print(f"{'='*60}")

        for pos_enc in POS_ENCS:
            for objective in ['ar', 'diffusion']:
                for fmt in FORMATS:
                    key = f"{nd}d_{objective}_{fmt}_{pos_enc}"
                    print(f"\n▶ {key}")

                    # Build splits
                    if nd == 3:
                        splits = build_splits_3d(fmt, n_train, n_test)
                        test_splits = ['test_id', 'test_ood_number', 'test_ood_length']
                    else:
                        splits = build_splits_nd(nd, fmt, n_train, n_test)
                        test_splits = ['test_id', 'test_ood_length']

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

                    for sp in test_splits:
                        res = evaluate(
                            model, tok, splits[sp], objective, get_ans,
                            get_full_answer, policy='confidence', greedy=True,
                        )
                        all_results[key][sp] = res['exact_match']
                        print(f"    {sp}: {res['exact_match']:.4f}")

                        # Number OOD breakdown (3d only)
                        if sp == 'test_ood_number':
                            bd = breakdown_ood_number(splits[sp], res)
                            all_results[key]['number_breakdown'] = bd
                            print(f"      → one: {bd['one_held']:.4f} "
                                  f"({bd['n_one']}), both: {bd['both_held']:.4f} "
                                  f"({bd['n_both']})")

                        # Fixation order (diffusion, test_id only)
                        if objective == 'diffusion' and sp == 'test_id':
                            fix = analyse_fixation_order(
                                splits[sp], res, nd, fmt, tok)
                            if fix is not None:
                                fixation_data[key] = fix
                                mr = fix['mean_rank']
                                rank_str = ' '.join(
                                    f"pos{p}={mr[p]:.1f}" for p in sorted(mr))
                                print(f"      fixation: {rank_str}")
                                if fix['carry_corr']:
                                    print(f"      carry_corr: "
                                          f"rho={fix['carry_corr']['rho']:.3f}")

                    save_results(EXP_NAME, all_results, model=model, tag=key)

    # Store metadata
    all_results['convergence_iters'] = convergence_iters
    all_results['fixation_data'] = {
        k: {kk: (float(vv) if isinstance(vv, (int, float, np.floating))
                 else vv)
            for kk, vv in v.items()}
        for k, v in fixation_data.items()
    }

    # ── Visualization ──────────────────────────────────
    figs = {}
    configs = [k for k in all_results
               if isinstance(all_results[k], dict) and 'test_id' in all_results[k]]

    # 1) Difficulty curve: ID accuracy vs digit count
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    for fmt_idx, fmt in enumerate(FORMATS):
        ax = axes[fmt_idx]
        for obj in ['ar', 'diffusion']:
            for pe in POS_ENCS:
                xs, ys = [], []
                for nd, _, _ in DIGIT_CONFIGS:
                    k = f"{nd}d_{obj}_{fmt}_{pe}"
                    if k in all_results and isinstance(all_results[k], dict):
                        xs.append(nd)
                        ys.append(all_results[k].get('test_id', 0))
                ls = '-' if pe == 'absolute' else '--'
                color = '#e74c3c' if obj == 'ar' else '#3498db'
                label = f"{obj}_{pe}"
                ax.plot(xs, ys, ls, color=color, marker='o', label=label, alpha=0.8)
        ax.set_xlabel('Digits'); ax.set_ylabel('ID Accuracy')
        ax.set_title(f'{fmt}'); ax.set_ylim(-0.05, 1.05)
        ax.set_xticks([nd for nd, _, _ in DIGIT_CONFIGS])
        ax.legend(fontsize=6); ax.grid(alpha=0.3)
    fig.suptitle('Difficulty Scaling: ID Accuracy vs Digit Count', fontsize=13, y=1.02)
    fig.tight_layout(); figs['difficulty_curve'] = fig

    # 2) Format comparison at each digit count
    for nd, _, _ in DIGIT_CONFIGS:
        nd_configs = [k for k in configs if k.startswith(f"{nd}d_")]
        if not nd_configs:
            continue
        sps = ['test_id', 'test_ood_length']
        if nd == 3:
            sps = ['test_id', 'test_ood_number', 'test_ood_length']
        labels = sps
        fig, axes = plt.subplots(1, len(sps), figsize=(7*len(sps), 6))
        if len(sps) == 1:
            axes = [axes]
        for idx, sp in enumerate(sps):
            ax = axes[idx]
            vals = [all_results[k].get(sp, 0) for k in nd_configs]
            colors = ['#e74c3c' if 'ar' in k else '#3498db' for k in nd_configs]
            hatches = ['///' if 'rope' in k else '' for k in nd_configs]
            bars = ax.bar(range(len(nd_configs)), vals, color=colors, alpha=0.85)
            for bar, h in zip(bars, hatches):
                bar.set_hatch(h)
            ax.set_xticks(range(len(nd_configs)))
            ax.set_xticklabels([k.replace(f'{nd}d_','') for k in nd_configs],
                               fontsize=5, rotation=45, ha='right')
            ax.set_ylabel('Exact Match'); ax.set_title(sp); ax.set_ylim(0, 1.05)
            ax.grid(axis='y', alpha=0.3)
        fig.suptitle(f'{nd}-digit Addition (hatched=RoPE, red=AR, blue=Diff)',
                     fontsize=13, y=1.02)
        fig.tight_layout(); figs[f'accuracy_{nd}d'] = fig

    # 3) Fixation order heatmap for diffusion
    diff_fix_keys = [k for k in fixation_data
                     if 'diffusion' in k and 'plain' in k and 'absolute' in k]
    if diff_fix_keys:
        fig, axes = plt.subplots(1, len(diff_fix_keys),
                                 figsize=(5*len(diff_fix_keys), 4))
        if len(diff_fix_keys) == 1:
            axes = [axes]
        for idx, k in enumerate(sorted(diff_fix_keys)):
            ax = axes[idx]
            mr = fixation_data[k]['mean_rank']
            positions = sorted(mr.keys())
            ranks = [mr[p] for p in positions]
            nd_here = int(k.split('d_')[0])
            labels_pos = [f"MSB" if p == 0 else f"LSB" if p == nd_here
                          else f"pos{p}" for p in positions]
            bars = ax.bar(range(len(positions)), ranks, color='#3498db', alpha=0.85)
            ax.set_xticks(range(len(positions)))
            ax.set_xticklabels(labels_pos, fontsize=8)
            ax.set_ylabel('Mean Fixation Rank (lower = decoded earlier)')
            ax.set_title(f'{k}')
            ax.grid(axis='y', alpha=0.3)
        fig.suptitle('Diffusion Fixation Order: Which answer digit is decoded first?',
                     fontsize=12, y=1.02)
        fig.tight_layout(); figs['fixation_order'] = fig

    # 4) Fixation order comparison across formats
    for nd, _, _ in DIGIT_CONFIGS:
        fmt_keys = {}
        for fmt in FORMATS:
            k = f"{nd}d_diffusion_{fmt}_absolute"
            if k in fixation_data:
                fmt_keys[fmt] = fixation_data[k]
        if len(fmt_keys) >= 2:
            fig, ax = plt.subplots(figsize=(8, 5))
            for fmt, fix in fmt_keys.items():
                mr = fix['mean_rank']
                positions = sorted(mr.keys())
                ranks = [mr[p] for p in positions]
                ax.plot(positions, ranks, '-o', label=fmt, alpha=0.8)
            ax.set_xlabel('Answer Position (0=MSB → nd=LSB)')
            ax.set_ylabel('Mean Fixation Rank')
            ax.set_title(f'{nd}d: Fixation Order by Format')
            ax.legend(); ax.grid(alpha=0.3)
            fig.tight_layout(); figs[f'fixation_compare_{nd}d'] = fig

    save_results(EXP_NAME, all_results, figures=figs)

    # ── Summary ──
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for nd, _, _ in DIGIT_CONFIGS:
        nd_cfgs = [k for k in configs if k.startswith(f"{nd}d_")]
        if not nd_cfgs:
            continue
        sps = ['test_id']
        if nd == 3:
            sps += ['test_ood_number']
        sps += ['test_ood_length']
        header = f"{'Config':<40}" + ''.join(f" {s:>12}" for s in sps) + f" {'conv':>8}"
        print(f"\n{nd}-DIGIT:")
        print(header)
        print("-" * len(header))
        for k in nd_cfgs:
            r = all_results[k]
            vals = ''.join(f" {r.get(s, 0):>12.4f}" for s in sps)
            print(f"{k:<40}{vals} {convergence_iters.get(k,'?'):>8}")

    # Fixation summary
    if fixation_data:
        print("\nFIXATION ORDER (diffusion, lower rank = decoded earlier):")
        for k in sorted(fixation_data.keys()):
            fix = fixation_data[k]
            mr = fix['mean_rank']
            rank_str = ', '.join(f"pos{p}={mr[p]:.1f}" for p in sorted(mr))
            cc = fix.get('carry_corr')
            cc_str = f"carry_rho={cc['rho']:.3f}" if cc else "N/A"
            print(f"  {k}: [{rank_str}] {cc_str}")

    plt.show()
    return all_results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Secondary Analysis: Digit-Position Exclusion (Lee et al. B.2.1)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EXCL_DIGIT = 5
EXCL_POSITIONS = [0, 1, 2]
EXCL_FMT = 'plain'
EXCL_N_TRAIN = 10_000
EXCL_N_TEST  = 10_000
DIGITS_FULL  = list(range(10))
DIGITS_NO5   = [d for d in range(10) if d != EXCL_DIGIT]


def _gen_excl_operand(nd, excl_pos, rng):
    digits = []
    for p in range(nd):
        pool = DIGITS_NO5 if p == excl_pos else DIGITS_FULL
        digits.append(str(rng.choice(pool)))
    return int(''.join(reversed(digits)))


def _gen_excl_data(nd, n, excl_pos, fmt, seed):
    rng = random.Random(seed)
    fn = FMT_FN[fmt]
    seen, out = set(), []
    for _ in range(n * 10):
        if len(out) >= n: break
        a = _gen_excl_operand(nd, excl_pos, rng)
        b = _gen_excl_operand(nd, excl_pos, rng)
        if (a, b) in seen: continue
        seen.add((a, b))
        out.append(fn(a, b, nd))
    return out


def _has_digit_at_pos(s, digit, pos, nd=3):
    prefix = s.split('=')[0]
    parts = prefix.split('+')
    a_s, b_s = parts[0].zfill(nd), parts[1].zfill(nd)
    str_idx = nd - 1 - pos
    return int(a_s[str_idx]) == digit or int(b_s[str_idx]) == digit


def run_digit_position_analysis():
    print("\n" + "=" * 70)
    print("  SECONDARY: Digit-Position Exclusion (B.2.1)")
    print(f"  Exclude digit {EXCL_DIGIT} from one position at a time")
    print("=" * 70)
    mount_drive()
    torch.manual_seed(123)

    pos_names = {0: 'LSB (ones)', 1: '2nd (tens)', 2: 'MSB (hundreds)'}
    test_all = _gen_any(3, EXCL_N_TEST, EXCL_FMT, seed=9999)
    tok = build_tok(EXCL_FMT)
    results = {}

    for excl_pos in EXCL_POSITIONS:
        for objective in ['ar', 'diffusion']:
            key = f"{objective}_excl{excl_pos}"
            print(f"\n▶ {key} — exclude {EXCL_DIGIT} from {pos_names[excl_pos]}")

            train = _gen_excl_data(3, EXCL_N_TRAIN, excl_pos, EXCL_FMT,
                                   seed=42+excl_pos)
            all_s = train + test_all
            max_len = max(len(tok.encode(s)) for s in all_s) + 1

            model, hist, ml, conv_it = train_model(
                objective, tok, train, max_len=max_len,
                max_iters=MAX_ITERS, patience=PATIENCE,
                pos_enc='absolute', log_interval=500,
            )

            get_ans = lambda s: get_answer(s, EXCL_FMT)
            res = evaluate(
                model, tok, test_all, objective, get_ans,
                get_full_answer, policy='confidence', greedy=True,
            )

            per_correct = res['per_sample_correct']
            excl_ok, rest_ok = [], []
            for s, ok in zip(test_all, per_correct):
                if _has_digit_at_pos(s, EXCL_DIGIT, excl_pos):
                    excl_ok.append(ok)
                else:
                    rest_ok.append(ok)

            results[key] = {
                'excl_pos': excl_pos,
                'overall': res['exact_match'],
                'exclusion_acc': sum(excl_ok) / max(len(excl_ok), 1),
                'rest_acc': sum(rest_ok) / max(len(rest_ok), 1),
                'n_excl': len(excl_ok), 'n_rest': len(rest_ok),
                'conv_it': conv_it,
            }
            print(f"    overall: {results[key]['overall']:.4f}, "
                  f"excl_acc: {results[key]['exclusion_acc']:.4f}")

    # Summary + plot
    print("\n" + "=" * 70)
    print("DIGIT-POSITION EXCLUSION SUMMARY")
    print(f"{'Config':<25} {'Position':<18} {'Overall':>8} {'Excl':>8} {'Rest':>8}")
    print("-" * 70)
    for k in sorted(results):
        r = results[k]
        print(f"{k:<25} {pos_names[r['excl_pos']]:<18} "
              f"{r['overall']:>8.4f} {r['exclusion_acc']:>8.4f} {r['rest_acc']:>8.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax_idx, metric in enumerate(['overall', 'exclusion_acc']):
        ax = axes[ax_idx]
        x = np.arange(len(EXCL_POSITIONS))
        w = 0.35
        ar_vals = [results[f'ar_excl{p}'][metric] for p in EXCL_POSITIONS]
        diff_vals = [results[f'diffusion_excl{p}'][metric] for p in EXCL_POSITIONS]
        ax.bar(x - w/2, ar_vals, w, label='AR', color='#e74c3c', alpha=0.85)
        ax.bar(x + w/2, diff_vals, w, label='Diffusion', color='#3498db', alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([pos_names[p] for p in EXCL_POSITIONS])
        ax.set_ylabel('Exact Match')
        ax.set_title('Overall' if metric == 'overall' else 'Exclusion Accuracy')
        ax.set_ylim(0, 1.05); ax.legend(); ax.grid(axis='y', alpha=0.3)
    fig.suptitle(f'Digit-Position Exclusion: digit {EXCL_DIGIT}', fontsize=13, y=1.02)
    fig.tight_layout()
    save_results(f'{EXP_NAME}_digit_position', results, figures={'digit_pos': fig})
    plt.show()
    return results


if __name__ == '__main__':
    run()
