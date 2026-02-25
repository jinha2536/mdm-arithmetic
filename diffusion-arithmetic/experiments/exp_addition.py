"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Experiment 1 — Addition: AR vs Masked Diffusion  (v2)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Colab:  %run experiments/exp_addition.py

  Redesign following advisor feedback:

  1) Structured sampling (Lee et al.):
       • Digit-balanced: lower-digit operands over-represented
       • Carry-balanced: roughly equal 0,1,...,nd carries per nd
  2) Interpolation OOD: train nd∈{1,2,3,4,5,6,8}, test nd=7
       (avoids position-embedding confound of extrapolation OOD)
  3) Up to 8-digit operands — challenging even for AR
  4) Detailed per-sample error analysis:
       • Per answer-position accuracy
       • Carry-conditional accuracy
       • Error categorisation and examples
       • RoPE vs absolute side-by-side

  12 models: 2(objective) × 3(format) × 2(PE)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                if '__file__' in dir() else '.')

import random, torch, math
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('Agg')
import numpy as np
from collections import defaultdict

from core.tokenizer import CharTokenizer
from core.train_utils import (
    mount_drive, save_results, train_model,
    generate_ar, generate_diffusion, DEVICE,
)

EXP_NAME = 'exp_addition'

# ── Config ──────────────────────────────────────────
ND_OOD       = 7
ND_TRAIN_SET = [1, 2, 3, 4, 5, 6, 8]
ND_ALL       = [1, 2, 3, 4, 5, 6, 7, 8]

# Reduced training data — intentionally limited to expose
# learning difficulty differences between AR and diffusion.
# Original (Lee et al.) used ~50K; we use 2K to create a
# regime where neither model achieves 100%.
ND_ALLOC = {1: 50, 2: 100, 3: 200, 4: 300,
            5: 350, 6: 400, 8: 600}   # total 2,000

N_TEST       = 500      # per digit count
MAX_ITERS    = 15_000
PATIENCE     = 2_000

FORMATS  = ['plain', 'reverse']
POS_ENCS = ['absolute', 'rope']


# ── Format functions ────────────────────────────────

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


def get_answer(s, fmt):
    if fmt == 'scratchpad': return s.split('>>')[-1]
    return s.split('=')[1]

def get_full_answer(s):
    return s.split('=', 1)[1]

def _parse_operands(s):
    parts = s.split('=')[0].split('+')
    return int(parts[0]), int(parts[1])


# ── Carry helpers ───────────────────────────────────

def _count_carries(a, b, nd):
    """Count carry operations in nd-digit a + b."""
    a_s, b_s = str(a).zfill(nd), str(b).zfill(nd)
    carry, count = 0, 0
    for i in range(nd - 1, -1, -1):
        s = int(a_s[i]) + int(b_s[i]) + carry
        carry = s // 10
        count += carry
    return count


def _carry_positions(a, b, nd):
    """Return carry-out flags, indexed from LSB (pos 0 = ones)."""
    a_s, b_s = str(a).zfill(nd), str(b).zfill(nd)
    flags, carry = [], 0
    for i in range(nd - 1, -1, -1):
        s = int(a_s[i]) + int(b_s[i]) + carry
        carry = s // 10
        flags.append(bool(carry))
    return flags


def build_tok(fmt):
    chars = list('0123456789+=')
    if fmt == 'scratchpad':
        chars.extend(['C', 'S', '>'])
    return CharTokenizer(chars, {'mask': 'M', 'pad': 'P'})


# ── Structured data generation ──────────────────────

def _effective_digits(n):
    """Number of digits in n (0 has 1 digit)."""
    return max(1, len(str(n)))


def gen_pairs_balanced(nd, n, seed):
    """
    Structured sampling following Lee et al.:

    1) Digit-balanced: operands sampled from [0, 10^nd - 1] with
       uniform weight across effective digit counts 1..nd.
       E.g. for nd=5: equal mix of 1-digit (00001-00009),
       2-digit (00010-00099), ..., 5-digit (10000-99999).

    2) Carry-balanced: within each digit-pair stratum,
       roughly equal carry counts 0..nd.

    Returns n (a, b) pairs.
    """
    rng = random.Random(seed)

    # Digit-count strata ranges
    digit_ranges = []
    for d in range(1, nd + 1):
        lo = 1 if d == 1 else 10 ** (d - 1)
        hi = 10 ** d - 1
        digit_ranges.append((lo, hi))
    # Also include 0 (zero-digit operand) in the 1-digit bucket
    digit_ranges[0] = (0, 9)

    # Build pool: stratified by (a_eff_digits, b_eff_digits, n_carries)
    pool = defaultdict(list)
    seen = set()
    attempts = n * 100

    for _ in range(attempts):
        # Pick random effective digit counts for a and b
        da = rng.randint(1, nd)
        db = rng.randint(1, nd)
        lo_a, hi_a = digit_ranges[da - 1]
        lo_b, hi_b = digit_ranges[db - 1]
        a = rng.randint(lo_a, hi_a)
        b = rng.randint(lo_b, hi_b)
        if (a, b) in seen:
            continue
        seen.add((a, b))
        nc = _count_carries(a, b, nd)
        pool[nc].append((a, b))

    # Stratified by carry count (primary balancing axis)
    carry_counts = sorted(pool.keys())
    target = max(1, n // max(len(carry_counts), 1))

    out = []
    for nc in carry_counts:
        rng.shuffle(pool[nc])
        take = min(target, len(pool[nc]))
        out.extend(pool[nc][:take])

    # Fill remainder
    rng2 = random.Random(seed + 9999)
    hi = 10 ** nd - 1
    while len(out) < n:
        a, b = rng2.randint(0, hi), rng2.randint(0, hi)
        out.append((a, b))

    rng.shuffle(out)
    return out[:n]


def gen_train_mixed(alloc, fmt, seed):
    """
    Generate structured training set: digit-balanced + carry-balanced.

    alloc: {nd: n_samples} allocation per digit count.
    Returns list of formatted strings, shuffled.
    """
    fn = FMT_FN[fmt]
    out = []
    for nd, n in sorted(alloc.items()):
        pairs = gen_pairs_balanced(nd, n, seed=seed + nd * 100)
        for a, b in pairs:
            out.append(fn(a, b, nd))
    random.Random(seed).shuffle(out)
    return out


def gen_test(nd, n, fmt, seed):
    """Generate n carry-balanced test samples for nd-digit."""
    fn = FMT_FN[fmt]
    pairs = gen_pairs_balanced(nd, n, seed=seed)
    return [fn(a, b, nd) for a, b in pairs]


# ── Detailed evaluation ─────────────────────────────

def evaluate_detailed(model, tokenizer, test_samples, objective,
                      fmt, nd, max_len=None, batch_size=128):
    """
    Evaluate with full per-sample detail.

    For diffusion, decodes the FULL remaining region (answer + PAD)
    to match training-time context.

    Returns list of dicts, each containing:
      a, b, nd, max_eff_digits, gold_ans, pred_ans, correct,
      pos_correct (per answer digit), carries, n_carries,
      error_type (if wrong)
    """
    mask_id = tokenizer.special_ids['mask']
    pad_id = tokenizer.special_ids['pad']
    model.eval()

    # Compute answer length from an example
    ex = test_samples[0]
    full_ans = get_full_answer(ex)
    ans_len = len(tokenizer.encode(full_ans))

    results = []

    for start in range(0, len(test_samples), batch_size):
        batch = test_samples[start:start + batch_size]
        B = len(batch)

        penc = [tokenizer.encode(s.split('=')[0] + '=') for s in batch]
        pmax = max(len(p) for p in penc)
        pids = torch.full((B, pmax), pad_id, dtype=torch.long)
        for i, e in enumerate(penc):
            pids[i, :len(e)] = torch.tensor(e)

        with torch.no_grad():
            if objective == 'ar':
                gen = generate_ar(model, pids, ans_len, DEVICE)
                pred_ids = gen[:, pmax:pmax + ans_len]
            else:
                # Decode full remaining region (answer + PAD)
                n_decode = max_len - pmax if max_len else ans_len
                gen, _, info = generate_diffusion(
                    model, pids, n_decode, mask_id,
                    policy='confidence', greedy=True, device=DEVICE)
                pred_ids = gen[:, pmax:pmax + ans_len]

        for i in range(B):
            pred_full = tokenizer.decode(pred_ids[i].cpu().tolist())
            sample = batch[i]
            gold_ans = get_answer(sample, fmt)

            # Extract predicted final answer
            if fmt == 'scratchpad':
                if '>>' in pred_full:
                    pred_ans = pred_full.split('>>')[-1]
                else:
                    pred_ans = pred_full[-(nd + 1):]
            else:
                pred_ans = pred_full

            a, b_val = _parse_operands(sample)
            max_eff = max(_effective_digits(a), _effective_digits(b_val))

            # Per-position comparison on final answer
            pos_correct = []
            for j in range(len(gold_ans)):
                if j < len(pred_ans):
                    pos_correct.append(pred_ans[j] == gold_ans[j])
                else:
                    pos_correct.append(False)

            carries = _carry_positions(a, b_val, nd)
            correct = pred_ans == gold_ans

            # Classify error type
            error_type = None
            if not correct:
                if len(pred_ans) != len(gold_ans):
                    error_type = 'length'
                else:
                    wrong_pos = [j for j, c in enumerate(pos_correct)
                                 if not c]
                    # Check if errors align with carry positions
                    carry_pos_set = set()
                    for ci, cf in enumerate(carries):
                        if cf:
                            if fmt == 'reverse':
                                carry_pos_set.add(ci + 1)
                            else:
                                carry_pos_set.add(nd - (ci + 1))
                    if all(j in carry_pos_set for j in wrong_pos):
                        error_type = 'carry'
                    elif len(wrong_pos) == 1:
                        # off-by-one check
                        j = wrong_pos[0]
                        try:
                            diff = abs(int(pred_ans[j]) - int(gold_ans[j]))
                            error_type = 'off_by_1' if diff == 1 \
                                else 'single_digit'
                        except ValueError:
                            error_type = 'invalid_char'
                    else:
                        error_type = 'multi_digit'

            results.append({
                'a': a, 'b': b_val, 'nd': nd,
                'max_eff_digits': max_eff,
                'gold_ans': gold_ans, 'pred_ans': pred_ans,
                'correct': correct,
                'pos_correct': pos_correct,
                'carries': carries,
                'n_carries': sum(carries),
                'error_type': error_type,
            })

    return results


# ── Analysis aggregation ────────────────────────────

def analyse_results(detailed, fmt):
    """
    Aggregate detailed evaluation into structured analysis.

    Returns:
      accuracy, carry_accuracy{nc: (acc, count)},
      position_accuracy[j], carry_conditional{ci: {...}},
      eff_digit_accuracy{d: (acc, count)},
      error_types{type: count},
      error_examples[{input, gold, pred, wrong_pos, n_carries, type}]
    """
    if not detailed:
        return {'accuracy': 0, 'n_samples': 0}

    nd = detailed[0]['nd']
    ans_len = nd + 1
    n = len(detailed)

    # 1. Overall accuracy
    acc = sum(r['correct'] for r in detailed) / n

    # 2. Accuracy by carry count
    by_nc = defaultdict(list)
    for r in detailed:
        by_nc[r['n_carries']].append(r['correct'])
    carry_acc = {nc: (sum(v) / len(v), len(v))
                 for nc, v in sorted(by_nc.items())}

    # 3. Per answer-position accuracy (raw output order)
    pos_acc = []
    for j in range(ans_len):
        vals = [r['pos_correct'][j] for r in detailed
                if j < len(r['pos_correct'])]
        pos_acc.append(sum(vals) / max(len(vals), 1))

    # 4. Carry-conditional position accuracy
    carry_cond = {}
    for ci in range(nd):
        if fmt == 'reverse':
            ans_pos = ci + 1
        else:
            ans_pos = nd - (ci + 1)

        if ans_pos < 0 or ans_pos >= ans_len:
            continue

        with_c = [r['pos_correct'][ans_pos] for r in detailed
                  if ci < len(r['carries']) and r['carries'][ci]
                  and ans_pos < len(r['pos_correct'])]
        without_c = [r['pos_correct'][ans_pos] for r in detailed
                     if ci < len(r['carries']) and not r['carries'][ci]
                     and ans_pos < len(r['pos_correct'])]

        carry_cond[ci] = {
            'ans_pos': ans_pos,
            'acc_with': sum(with_c) / max(len(with_c), 1),
            'acc_without': sum(without_c) / max(len(without_c), 1),
            'n_with': len(with_c), 'n_without': len(without_c),
        }

    # 5. Accuracy by effective digit count (max of a, b)
    by_eff = defaultdict(list)
    for r in detailed:
        by_eff[r['max_eff_digits']].append(r['correct'])
    eff_digit_acc = {d: (sum(v) / len(v), len(v))
                     for d, v in sorted(by_eff.items())}

    # 6. Error type distribution
    error_types = defaultdict(int)
    for r in detailed:
        if not r['correct']:
            error_types[r.get('error_type', 'unknown')] += 1

    # 7. Per-position digit confusion (for errors only)
    digit_confusion = {}
    for j in range(ans_len):
        conf = defaultdict(int)
        for r in detailed:
            if (j < len(r['pos_correct']) and not r['pos_correct'][j]
                    and j < len(r['gold_ans'])
                    and j < len(r['pred_ans'])):
                gd, pd = r['gold_ans'][j], r['pred_ans'][j]
                conf[f"{gd}>{pd}"] += 1
        if conf:
            digit_confusion[j] = dict(conf)

    # 8. Error examples (up to 20)
    errors = []
    for r in detailed:
        if not r['correct']:
            wrong = [j for j, c in enumerate(r['pos_correct']) if not c]
            errors.append({
                'input': f"{_pad(r['a'], nd)}+{_pad(r['b'], nd)}",
                'gold': r['gold_ans'], 'pred': r['pred_ans'],
                'wrong_pos': wrong, 'n_carries': r['n_carries'],
                'eff_digits': r['max_eff_digits'],
                'type': r.get('error_type', 'unknown'),
            })

    return {
        'accuracy': acc,
        'n_samples': n,
        'carry_accuracy': carry_acc,
        'position_accuracy': pos_acc,
        'carry_conditional': carry_cond,
        'eff_digit_accuracy': eff_digit_acc,
        'error_types': dict(error_types),
        'digit_confusion': digit_confusion,
        'n_errors': len(errors),
        'error_examples': errors[:20],
    }


def print_analysis(analysis, nd):
    """Print formatted analysis for one configuration."""
    a = analysis

    # Carry breakdown
    if a.get('carry_accuracy'):
        parts = [f"{nc}c={acc:.3f}({n})"
                 for nc, (acc, n) in sorted(a['carry_accuracy'].items())]
        print(f"    By carries: {' '.join(parts)}")

    # Position accuracy
    if a.get('position_accuracy'):
        pa = a['position_accuracy']
        parts = [f"p{j}={v:.3f}" for j, v in enumerate(pa)]
        print(f"    By position: {' '.join(parts)}")

    # Effective digit accuracy
    if a.get('eff_digit_accuracy'):
        parts = [f"eff{d}={acc:.3f}({n})"
                 for d, (acc, n) in sorted(
                     a['eff_digit_accuracy'].items())]
        print(f"    By eff digits: {' '.join(parts)}")

    # Error type distribution
    if a.get('error_types'):
        parts = [f"{t}={c}" for t, c in
                 sorted(a['error_types'].items(),
                        key=lambda x: -x[1])]
        print(f"    Error types: {' '.join(parts)}")

    # Carry conditional (only show if interesting)
    if a.get('carry_conditional'):
        for ci, cc in sorted(a['carry_conditional'].items()):
            if cc['n_with'] > 5 and cc['n_without'] > 5:
                delta = cc['acc_with'] - cc['acc_without']
                if abs(delta) > 0.01:
                    print(f"      carry[{ci}]→pos{cc['ans_pos']}: "
                          f"with={cc['acc_with']:.3f}({cc['n_with']}) "
                          f"w/o={cc['acc_without']:.3f}({cc['n_without']}) "
                          f"Δ={delta:+.3f}")

    # Digit confusion (top confusions)
    if a.get('digit_confusion'):
        top_conf = []
        for pos, conf in a['digit_confusion'].items():
            for key, cnt in conf.items():
                g, p = key.split('>')
                top_conf.append((cnt, pos, g, p))
        top_conf.sort(reverse=True)
        if top_conf:
            show = min(5, len(top_conf))
            parts = [f"pos{p}:{g}→{pr}(×{c})"
                     for c, p, g, pr in top_conf[:show]]
            print(f"    Top confusions: {' '.join(parts)}")

    # Error examples
    if a.get('error_examples') and a['accuracy'] < 1.0:
        show = min(5, len(a['error_examples']))
        print(f"    Errors ({a['n_errors']}), first {show}:")
        for e in a['error_examples'][:show]:
            print(f"      {e['input']}: gold={e['gold']} "
                  f"pred={e['pred']} wrong@{e['wrong_pos']} "
                  f"{e['n_carries']}c [{e['type']}]")


# ── Main ────────────────────────────────────────────

def run():
    print("=" * 70)
    print("  EXP 1: Addition — Structured Sampling + Interpolation OOD")
    print("=" * 70)
    mount_drive()
    torch.manual_seed(42)

    all_results = {}
    all_histories = {}
    convergence_iters = {}

    for pos_enc in POS_ENCS:
        for objective in ['ar', 'diffusion']:
            for fmt in FORMATS:
                key = f"{objective}_{fmt}_{pos_enc}"
                print(f"\n{'─'*60}")
                print(f"▶ {key}")
                print(f"{'─'*60}")

                # ── Generate data ──
                train_data = gen_train_mixed(ND_ALLOC, fmt, seed=42)

                test_data = {}
                for nd in ND_ALL:
                    test_data[nd] = gen_test(
                        nd, N_TEST, fmt, seed=1000 + nd)

                tok = build_tok(fmt)

                all_s = train_data
                for v in test_data.values():
                    all_s = all_s + v
                max_len = max(len(tok.encode(s)) for s in all_s) + 1

                # ── Diagnostics ──
                carry_dist = defaultdict(lambda: defaultdict(int))
                eff_digit_dist = defaultdict(lambda: defaultdict(int))
                for s in train_data:
                    a, b = _parse_operands(s)
                    nd_s = len(s.split('=')[0].split('+')[0])
                    nc = _count_carries(a, b, nd_s)
                    carry_dist[nd_s][nc] += 1
                    me = max(_effective_digits(a), _effective_digits(b))
                    eff_digit_dist[nd_s][me] += 1

                print(f"  Train: {len(train_data)} samples, "
                      f"max_len={max_len}")
                for nd_s in sorted(carry_dist):
                    cd = dict(carry_dist[nd_s])
                    ed = dict(sorted(eff_digit_dist[nd_s].items()))
                    print(f"    {nd_s}d carries: {cd}")
                    print(f"    {nd_s}d eff_dig: {ed}")
                print(f"  Test: {N_TEST}/nd, OOD=nd={ND_OOD}")

                # ── Train ──
                model, hist, ml, conv_it = train_model(
                    objective, tok, train_data, max_len=max_len,
                    max_iters=MAX_ITERS, patience=PATIENCE,
                    pos_enc=pos_enc, log_interval=500,
                )
                all_histories[key] = hist
                convergence_iters[key] = conv_it

                # ── Evaluate per-nd ──
                all_results[key] = {}

                for nd in ND_ALL:
                    tag = 'OOD' if nd == ND_OOD else 'ID'
                    nd_key = f'{nd}d'

                    detailed = evaluate_detailed(
                        model, tok, test_data[nd], objective,
                        fmt, nd, max_len=max_len)
                    analysis = analyse_results(detailed, fmt)

                    all_results[key][nd_key] = analysis

                    acc = analysis['accuracy']
                    print(f"\n  [{tag}] {nd}d: {acc:.4f} "
                          f"({analysis['n_samples']})")
                    print_analysis(analysis, nd)

                save_results(EXP_NAME, all_results, model=model,
                             tag=key)

    all_results['convergence_iters'] = convergence_iters

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  Visualisation
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    figs = {}
    configs = [k for k in all_results
               if isinstance(all_results[k], dict)
               and '1d' in all_results[k]]

    obj_color = {'ar': '#e74c3c', 'diffusion': '#3498db'}

    # 1) Accuracy vs digit count — per format ───────
    for fmt in FORMATS:
        fig, ax = plt.subplots(figsize=(10, 6))
        for obj in ['ar', 'diffusion']:
            for pe in POS_ENCS:
                k = f"{obj}_{fmt}_{pe}"
                if k not in all_results:
                    continue
                accs = [all_results[k].get(f'{nd}d', {})
                        .get('accuracy', 0) for nd in ND_ALL]
                ls = '-' if pe == 'absolute' else '--'
                ax.plot(ND_ALL, accs, ls, marker='o',
                        color=obj_color[obj],
                        label=f'{obj}/{pe}', alpha=0.8)
        ax.axvspan(ND_OOD - 0.3, ND_OOD + 0.3,
                   alpha=0.15, color='orange',
                   label=f'OOD (nd={ND_OOD})')
        ax.set_xlabel('Digit Count (nd)')
        ax.set_ylabel('Exact Match')
        ax.set_xticks(ND_ALL); ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        ax.set_title(f'Accuracy vs Digit Count — {fmt}')
        fig.tight_layout(); figs[f'acc_vs_nd_{fmt}'] = fig

    # 2) Carry count breakdown at select nd ─────────
    for nd_show in [5, 8]:
        nd_key = f'{nd_show}d'
        fig, axes = plt.subplots(1, len(FORMATS),
                                  figsize=(6 * len(FORMATS), 5))
        for fi, fmt in enumerate(FORMATS):
            ax = axes[fi]
            for obj in ['ar', 'diffusion']:
                for pe in POS_ENCS:
                    k = f"{obj}_{fmt}_{pe}"
                    ca = all_results.get(k, {}).get(nd_key, {}) \
                        .get('carry_accuracy', {})
                    if not ca:
                        continue
                    ncs = sorted(ca.keys())
                    accs = [ca[nc][0] for nc in ncs]
                    ls = '-' if pe == 'absolute' else '--'
                    ax.plot(ncs, accs, ls + 'o',
                            color=obj_color[obj],
                            label=f'{obj}/{pe}', alpha=0.8)
            ax.set_xlabel('# Carries')
            ax.set_ylabel('Accuracy')
            ax.set_title(f'{fmt}')
            ax.set_ylim(-0.05, 1.05)
            ax.legend(fontsize=6); ax.grid(alpha=0.3)
        fig.suptitle(f'Accuracy by Carry Count — {nd_show}d',
                     fontsize=13, y=1.02)
        fig.tight_layout()
        figs[f'carry_acc_{nd_show}d'] = fig

    # 3) Per-position accuracy ──────────────────────
    for nd_show in [5, 8]:
        nd_key = f'{nd_show}d'
        fig, axes = plt.subplots(2, len(FORMATS),
                                  figsize=(6 * len(FORMATS), 10))
        for fi, fmt in enumerate(FORMATS):
            for oi, obj in enumerate(['ar', 'diffusion']):
                ax = axes[oi, fi]
                for pe in POS_ENCS:
                    k = f"{obj}_{fmt}_{pe}"
                    pa = all_results.get(k, {}).get(nd_key, {}) \
                        .get('position_accuracy', [])
                    if not pa:
                        continue
                    x = list(range(len(pa)))
                    ls = '-' if pe == 'absolute' else '--'
                    ax.plot(x, pa, ls + 's', color=obj_color[obj],
                            label=f'{pe}', alpha=0.8, markersize=5)
                ax.set_xlabel('Answer Position')
                ax.set_ylabel('Accuracy')
                ax.set_title(f'{obj} / {fmt}')
                ax.set_ylim(-0.05, 1.05)
                ax.legend(fontsize=7); ax.grid(alpha=0.3)
        fig.suptitle(f'Per-Position Accuracy — {nd_show}d',
                     fontsize=13, y=1.01)
        fig.tight_layout()
        figs[f'pos_acc_{nd_show}d'] = fig

    # 3b) Effective digit accuracy ──────────────────
    for nd_show in [5, 8]:
        nd_key = f'{nd_show}d'
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for oi, obj in enumerate(['ar', 'diffusion']):
            ax = axes[oi]
            for fmt in FORMATS:
                for pe in POS_ENCS:
                    k = f"{obj}_{fmt}_{pe}"
                    eda = all_results.get(k, {}).get(nd_key, {}) \
                        .get('eff_digit_accuracy', {})
                    if not eda:
                        continue
                    ds = sorted(eda.keys())
                    accs = [eda[d][0] for d in ds]
                    ls = '-' if pe == 'absolute' else '--'
                    marker = 'o' if fmt == 'plain' else \
                        ('s' if fmt == 'reverse' else '^')
                    ax.plot(ds, accs, ls, marker=marker,
                            label=f'{fmt}/{pe}', alpha=0.7)
            ax.set_xlabel('Max Effective Digits of Operands')
            ax.set_ylabel('Accuracy')
            ax.set_title(f'{obj}')
            ax.set_ylim(-0.05, 1.05)
            ax.legend(fontsize=6); ax.grid(alpha=0.3)
        fig.suptitle(f'Accuracy by Operand Size — {nd_show}d',
                     fontsize=13, y=1.02)
        fig.tight_layout()
        figs[f'eff_digit_acc_{nd_show}d'] = fig

    # 3c) Error type breakdown ──────────────────────
    for nd_show in [5, 7, 8]:
        nd_key = f'{nd_show}d'
        error_configs = []
        for k in configs:
            et = all_results.get(k, {}).get(nd_key, {}) \
                .get('error_types', {})
            if et:
                error_configs.append((k, et))
        if not error_configs:
            continue
        fig, ax = plt.subplots(figsize=(12, 6))
        all_types = sorted(set(t for _, et in error_configs
                               for t in et))
        x = np.arange(len(error_configs))
        w = 0.8 / max(len(all_types), 1)
        for ti, etype in enumerate(all_types):
            vals = [et.get(etype, 0) for _, et in error_configs]
            ax.bar(x + ti * w, vals, w, label=etype, alpha=0.8)
        ax.set_xticks(x + w * len(all_types) / 2)
        ax.set_xticklabels([k for k, _ in error_configs],
                           rotation=45, ha='right', fontsize=7)
        ax.set_ylabel('Error Count')
        ax.legend(fontsize=7); ax.grid(alpha=0.3, axis='y')
        tag = '[OOD]' if nd_show == ND_OOD else '[ID]'
        ax.set_title(f'Error Types — {nd_show}d {tag}')
        fig.tight_layout()
        figs[f'error_types_{nd_show}d'] = fig

    # 4) RoPE vs Absolute scatter ───────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for oi, obj in enumerate(['ar', 'diffusion']):
        ax = axes[oi]
        for fmt in FORMATS:
            k_abs = f"{obj}_{fmt}_absolute"
            k_rope = f"{obj}_{fmt}_rope"
            if k_abs not in all_results or k_rope not in all_results:
                continue
            xs, ys, labs = [], [], []
            for nd in ND_ALL:
                a_abs = all_results[k_abs].get(f'{nd}d', {}) \
                    .get('accuracy', 0)
                a_rope = all_results[k_rope].get(f'{nd}d', {}) \
                    .get('accuracy', 0)
                xs.append(a_abs); ys.append(a_rope)
                labs.append(f'{nd}d')
            marker = 'o' if fmt == 'plain' else \
                ('s' if fmt == 'reverse' else '^')
            ax.scatter(xs, ys, marker=marker, s=60,
                       label=fmt, alpha=0.7)
            for x, y, lab in zip(xs, ys, labs):
                ax.annotate(lab, (x, y), fontsize=6,
                            textcoords='offset points',
                            xytext=(4, 4))
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax.set_xlabel('Absolute PE Accuracy')
        ax.set_ylabel('RoPE Accuracy')
        ax.set_title(obj)
        ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        ax.set_aspect('equal')
    fig.suptitle('RoPE vs Absolute PE', fontsize=13, y=1.02)
    fig.tight_layout(); figs['rope_vs_abs'] = fig

    # 5) Training loss curves ───────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for fi, fmt in enumerate(FORMATS):
        for oi, obj in enumerate(['ar', 'diffusion']):
            ax = axes[oi, fi]
            for pe in POS_ENCS:
                k = f"{obj}_{fmt}_{pe}"
                h = all_histories.get(k, {})
                if 'loss' in h:
                    ls = '-' if pe == 'absolute' else '--'
                    ax.plot(h.get('iter', range(len(h['loss']))),
                            h['loss'], ls, label=pe, alpha=0.7)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Loss')
            ax.set_title(f'{obj}/{fmt}')
            ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.suptitle('Training Loss Curves', fontsize=13, y=1.01)
    fig.tight_layout(); figs['training_loss'] = fig

    save_results(EXP_NAME, all_results, figures=figs)

    # ── Summary ─────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY — Accuracy by Digit Count")
    print("=" * 70)

    header = f"{'Config':<32}"
    for nd in ND_ALL:
        tag = '*' if nd == ND_OOD else ' '
        header += f" {nd}d{tag:>5}"
    header += f"  {'conv':>6}"
    print(header)
    print("─" * len(header))

    for fmt in FORMATS:
        for obj in ['ar', 'diffusion']:
            for pe in POS_ENCS:
                k = f"{obj}_{fmt}_{pe}"
                if k not in all_results:
                    continue
                row = f"{k:<32}"
                for nd in ND_ALL:
                    a = all_results[k].get(f'{nd}d', {}) \
                        .get('accuracy', 0)
                    row += f" {a:>6.3f}"
                ci = convergence_iters.get(k, '?')
                row += f"  {ci:>6}"
                print(row)
        print()

    # OOD gap analysis
    print(f"\nOOD ({ND_OOD}d) vs Neighbours "
          f"({ND_OOD-1}d, {ND_OOD+1}d):")
    print(f"{'Config':<32}  {ND_OOD-1}d   {ND_OOD}d*  "
          f"{ND_OOD+1}d   gap")
    print("─" * 60)
    for k in sorted(configs):
        a_lo = all_results[k].get(f'{ND_OOD-1}d', {}) \
            .get('accuracy', 0)
        a_ood = all_results[k].get(f'{ND_OOD}d', {}) \
            .get('accuracy', 0)
        a_hi = all_results[k].get(f'{ND_OOD+1}d', {}) \
            .get('accuracy', 0)
        avg = (a_lo + a_hi) / 2
        gap = avg - a_ood
        print(f"{k:<32}  {a_lo:.3f}  {a_ood:.3f}  "
              f"{a_hi:.3f}  {gap:+.3f}")

    # Detailed error output for key conditions
    print("\n" + "=" * 70)
    print("DETAILED ERROR ANALYSIS")
    print("=" * 70)
    for nd in [5, 7, 8]:
        nd_key = f'{nd}d'
        tag = '[OOD]' if nd == ND_OOD else '[ID]'
        print(f"\n{'─'*60}")
        print(f"  {nd}d {tag}")
        print(f"{'─'*60}")
        for k in sorted(configs):
            a = all_results[k].get(nd_key, {})
            if not a or a.get('accuracy', 1) >= 0.999:
                continue
            print(f"\n  ▸ {k}: acc={a['accuracy']:.4f}")
            print_analysis(a, nd)

    plt.show()
    return all_results


if __name__ == '__main__':
    run()
