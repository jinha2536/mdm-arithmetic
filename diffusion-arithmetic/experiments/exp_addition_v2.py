"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Experiment 2 — Addition: Carry Dependency Learning Dynamics
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Core question: does the model learn the carry dependency graph?

  Fair comparison design:
    Training unit: EPOCH (1 epoch = 1 full pass over N samples)
    Both AR and Diffusion train for exactly MAX_EPOCHS
    Best model selected by held-out probe loss (fair across mask types)
    Three x-axes: epoch, iteration, token_gradients

  Training conditions:
    AR, Diff-random, Diff-oracle_lsb, Diff-confidence, Diff-puma
    Each diffusion model evaluated with decode policies: confidence, lsb
    Decode order analysis (GKP, chain→rank, easy→hard) for confidence only

  Carry-dependency analyses (post-training):
    - Confidence cascade: Δconf of dependent position when g/k/p is revealed
      (chain_end vs chain_mid — the key dependency graph test)
    - Chain length → decode rank / confidence
    - Counterfactual carry intervention (matched pairs, carry-in flip)
    - Carry-heavy adversarial test (long propagation chains)
    - No-propagate one-step solvability
    - Dependency context (p_above_g, p_above_k, p_above_p) confidence/accuracy
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import sys, os, time, math, json, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                if '__file__' in dir() else '.')

from core.tokenizer import CharTokenizer
from core.model import Transformer
from core.train_utils import (
    mount_drive, save_results,          # Drive persistence
    generate_ar, generate_diffusion,    # Generation (incl. l2r/r2l policies)
    encode_samples,                     # Data encoding
    DEVICE,
)
# NOT imported from core (custom in this experiment):
#   train_with_dynamics  — epoch-based loop + periodic probe + LSB/MSB masking
#   final_evaluate       — per-position & carry-conditional analysis
#   _analyse_orders      — position-level decode rank (core's is scratchpad-level)
#   probe_per_position   — per-position loss/acc/conf probe

EXP_NAME = 'exp_addition_v2_dynamics'

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Config (defaults; overridden by CLI args)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ND = 8
ANS_LEN = ND + 1                # 9

# These are defaults; parse_args() overrides them as module-level globals
N_TRAIN = 2000
N_TEST = 500
BATCH_SIZE = 200
MAX_EPOCHS = 1500
EVAL_EVERY = 50
LOG_EVERY = 20
GEN_EVAL_EVERY = 200
GEN_EVAL_N = 500
THRESHOLD = 0.99

FORMATS = ['plain', 'reverse']
MASK_TYPES = ['random', 'oracle_lsb', 'confidence', 'puma']
DECODE_POLICIES = ['confidence', 'lsb']

# Architecture
N_LAYER = 6
N_HEAD = 6
N_EMBD = 384
DROPOUT = 0.2
POS_ENC = 'absolute'

# Optimizer
LR = 1e-3
MIN_LR = 1e-4
WARMUP_EPOCHS = 10
GRAD_CLIP = 1.0

# PUMA-specific
PUMA_TAU = 0.9
PUMA_K_START = 3
PUMA_K_END = ANS_LEN

SEED = 42

# Whether to run AR baseline
RUN_AR = True


def parse_args():
    """Parse CLI args and update module-level config globals."""
    import argparse
    p = argparse.ArgumentParser(description='Addition learning dynamics experiment')

    # Experiment scope
    p.add_argument('--formats', nargs='+', default=None,
                   help='Formats to run (default: plain reverse)')
    p.add_argument('--masks', nargs='+', default=None,
                   help='Mask types to run (default: all)')
    p.add_argument('--decode', nargs='+', default=None,
                   help='Decode policies (default: confidence lsb)')
    p.add_argument('--no-ar', action='store_true',
                   help='Skip AR baseline')

    # Data
    p.add_argument('--nd', type=int, default=None,
                   help='Number of digits (default: 8)')
    p.add_argument('--n-train', type=int, default=None)
    p.add_argument('--n-test', type=int, default=None)

    # Training
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--batch-size', type=int, default=None)
    p.add_argument('--eval-every', type=int, default=None)
    p.add_argument('--gen-eval-every', type=int, default=None)

    # Architecture
    p.add_argument('--n-layer', type=int, default=None)
    p.add_argument('--n-head', type=int, default=None)
    p.add_argument('--n-embd', type=int, default=None)
    p.add_argument('--dropout', type=float, default=None)

    # PUMA
    p.add_argument('--puma-tau', type=float, default=None)
    p.add_argument('--puma-k-start', type=int, default=None)
    p.add_argument('--puma-k-end', type=int, default=None)

    # Tag for save directory
    p.add_argument('--tag', type=str, default='',
                   help='Suffix for experiment name (e.g. "puma_small")')
    p.add_argument('--seed', type=int, default=None,
                   help='Single seed (default: 42)')
    p.add_argument('--seeds', nargs='+', type=int, default=None,
                   help='Multiple seeds for repeated runs (e.g. --seeds 42 43 44)')

    args = p.parse_args()

    # Update globals
    g = globals()
    mapping = {
        'n_train': 'N_TRAIN', 'n_test': 'N_TEST', 'epochs': 'MAX_EPOCHS',
        'batch_size': 'BATCH_SIZE', 'eval_every': 'EVAL_EVERY',
        'gen_eval_every': 'GEN_EVAL_EVERY',
        'n_layer': 'N_LAYER', 'n_head': 'N_HEAD', 'n_embd': 'N_EMBD',
        'dropout': 'DROPOUT', 'puma_tau': 'PUMA_TAU',
        'puma_k_start': 'PUMA_K_START', 'puma_k_end': 'PUMA_K_END',
        'seed': 'SEED',
    }
    for arg_name, global_name in mapping.items():
        val = getattr(args, arg_name)
        if val is not None:
            g[global_name] = val

    if args.formats:
        g['FORMATS'] = args.formats
    if args.masks:
        g['MASK_TYPES'] = args.masks
    if args.decode:
        g['DECODE_POLICIES'] = args.decode
    if args.no_ar:
        g['RUN_AR'] = False
    # ND → recalculate derived globals
    if args.nd is not None:
        g['ND'] = args.nd
        g['ANS_LEN'] = args.nd + 1
        if args.puma_k_end is None:  # only auto-update if not explicitly set
            g['PUMA_K_END'] = args.nd + 1
    return args


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Data
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _pad(n, w):
    return str(n).zfill(w)

def _fmt_plain(a, b):
    return f"{_pad(a,ND)}+{_pad(b,ND)}={_pad(a+b,ANS_LEN)}"

def _fmt_reverse(a, b):
    return f"{_pad(a,ND)}+{_pad(b,ND)}={_pad(a+b,ANS_LEN)[::-1]}"

FMT_FN = {'plain': _fmt_plain, 'reverse': _fmt_reverse}

def get_answer(s, fmt):
    return s.split('=')[1]

def _parse_operands(s):
    parts = s.split('=')[0].split('+')
    return int(parts[0]), int(parts[1])

def _count_carries(a, b):
    a_s, b_s = _pad(a, ND), _pad(b, ND)
    carry, count = 0, 0
    for i in range(ND - 1, -1, -1):
        s = int(a_s[i]) + int(b_s[i]) + carry
        carry = s // 10
        count += carry
    return count

def _carry_positions(a, b):
    a_s, b_s = _pad(a, ND), _pad(b, ND)
    flags, carry = [], 0
    for i in range(ND - 1, -1, -1):
        s = int(a_s[i]) + int(b_s[i]) + carry
        carry = s // 10
        flags.append(bool(carry))
    return flags

def _carry_at_answer_pos(a, b, fmt):
    """Per-answer-position carry-in flag."""
    flags = _carry_positions(a, b)
    ci = [False] * ANS_LEN
    if fmt == 'plain':
        for k in range(ANS_LEN):
            lp = ND - k
            if lp == ND:
                ci[k] = flags[ND-1] if ND-1 < len(flags) else False
            elif 0 <= lp-1 < len(flags):
                ci[k] = flags[lp-1]
    else:
        for k in range(ANS_LEN):
            if k == 0: ci[k] = False
            elif k-1 < len(flags): ci[k] = flags[k-1]
    return ci


def _gkp_at_answer_pos(a, b, fmt):
    """Classify each answer position as generate/kill/propagate.
    Based on digit pair sum (without carry):
      generate (g): a_j + b_j >= 10 → always produces carry
      propagate (p): a_j + b_j == 9 → carry_out = carry_in
      kill (k): a_j + b_j <= 8 → never produces carry
    MSB (carry-out) position is classified as 'carry_out'.
    Returns list of ANS_LEN strings.
    """
    a_s, b_s = _pad(a, ND), _pad(b, ND)
    # digit_gkp[i] for i in 0..ND-1 (LSB=0)
    digit_gkp = []
    for i in range(ND - 1, -1, -1):
        s = int(a_s[i]) + int(b_s[i])
        if s >= 10:
            digit_gkp.append('g')
        elif s == 9:
            digit_gkp.append('p')
        else:
            digit_gkp.append('k')
    # digit_gkp[0]=LSB pair, digit_gkp[ND-1]=MSB pair
    out = ['?'] * ANS_LEN
    if fmt == 'plain':
        out[0] = 'carry_out'
        for j in range(ND):
            out[ND - j] = digit_gkp[j]
    else:
        out[ANS_LEN - 1] = 'carry_out'
        for j in range(ND):
            out[j] = digit_gkp[j]
    return out

def build_tok():
    return CharTokenizer(list('0123456789+='), {'mask': 'M', 'pad': 'P'})


def _carry_chain_length_at_pos(a, b, fmt):
    """For each answer position, compute the propagate chain length
    that must be resolved to determine carry-in at that position.
    Chain length 0: position is g or k (carry decided locally)
    Chain length n: n consecutive propagate positions below must be traced
    Returns list of ANS_LEN ints.
    """
    a_s, b_s = _pad(a, ND), _pad(b, ND)
    gkp = []
    for i in range(ND - 1, -1, -1):
        s = int(a_s[i]) + int(b_s[i])
        if s >= 10: gkp.append('g')
        elif s == 9: gkp.append('p')
        else: gkp.append('k')
    # gkp[0]=LSB, gkp[ND-1]=MSB
    chain_len = [0] * ND
    for d in range(ND):
        if gkp[d] in ('g', 'k'):
            chain_len[d] = 0
        else:  # propagate
            length = 0
            pos = d - 1
            while pos >= 0 and gkp[pos] == 'p':
                length += 1
                pos -= 1
            chain_len[d] = length + 1
    out = [0] * ANS_LEN
    if fmt == 'plain':
        out[0] = 0  # carry_out position
        for j in range(ND): out[ND - j] = chain_len[j]
    else:
        out[ANS_LEN - 1] = 0
        for j in range(ND): out[j] = chain_len[j]
    return out


def _dependency_context_at_pos(a, b, fmt):
    """Classify each answer position by its dependency context.

    Instead of flat g/k/p, captures what's BELOW each position:
      'g':           generate (carry-out=1 always, locally determined)
      'k':           kill (carry-out=0 always, locally determined)
      'p_above_g':   propagate, but chain terminates at g below → carry-in=1 certain
      'p_above_k':   propagate, but chain terminates at k below → carry-in=0 certain
      'p_above_p':   propagate, below is also p → must trace further
      'p_bottom':    propagate at LSB digit (no carry-in, so carry-in=0 → acts like k)
      'carry_out':   MSB overflow position

    The key insight: p_above_g and p_above_k are actually easy (carry resolved),
    while p_above_p is the hard case (uncertainty propagates).
    Returns list of ANS_LEN strings.
    """
    a_s, b_s = _pad(a, ND), _pad(b, ND)
    # gkp[d] for d=0(LSB)..ND-1(MSB)
    gkp = []
    for i in range(ND - 1, -1, -1):
        s = int(a_s[i]) + int(b_s[i])
        if s >= 10: gkp.append('g')
        elif s == 9: gkp.append('p')
        else: gkp.append('k')

    dep = ['?'] * ND
    for d in range(ND):
        if gkp[d] in ('g', 'k'):
            dep[d] = gkp[d]
        else:  # propagate
            if d == 0:
                dep[d] = 'p_bottom'  # LSB, no carry-in possible
            elif gkp[d-1] == 'g':
                dep[d] = 'p_above_g'
            elif gkp[d-1] == 'k':
                dep[d] = 'p_above_k'
            else:
                dep[d] = 'p_above_p'

    out = ['?'] * ANS_LEN
    if fmt == 'plain':
        out[0] = 'carry_out'
        for j in range(ND): out[ND - j] = dep[j]
    else:
        out[ANS_LEN - 1] = 'carry_out'
        for j in range(ND): out[j] = dep[j]
    return out


def _has_no_propagate(a, b):
    """Check if a pair has zero propagate positions (all g or k)."""
    a_s, b_s = _pad(a, ND), _pad(b, ND)
    for i in range(ND):
        if int(a_s[i]) + int(b_s[i]) == 9:
            return False
    return True


def gen_no_propagate_test(n, fmt, seed):
    """Generate test samples with NO propagate (p) positions.
    All digit pairs sum to ≤8 or ≥10, so every carry is locally determined.
    If the model can solve these in one step (fully-masked), it demonstrates
    that carry ambiguity (not position) is the real bottleneck.
    """
    rng = random.Random(seed)
    lo = 10**(ND - 1)
    hi = 10**ND - 1
    results = []
    attempts = 0
    while len(results) < n and attempts < n * 500:
        attempts += 1
        a = rng.randint(lo, hi)
        b = rng.randint(lo, hi)
        if _has_no_propagate(a, b):
            results.append(FMT_FN[fmt](a, b))
    if len(results) < n:
        print(f"    WARNING: only found {len(results)}/{n} no-propagate samples")
    return results

def _max_chain_len(a, b):
    """Max propagate chain length for a pair (format-independent, digit-level)."""
    a_s, b_s = _pad(a, ND), _pad(b, ND)
    gkp = []
    for i in range(ND - 1, -1, -1):
        s = int(a_s[i]) + int(b_s[i])
        if s >= 10: gkp.append('g')
        elif s == 9: gkp.append('p')
        else: gkp.append('k')
    # gkp[0]=LSB
    max_cl = 0
    cur_run = 0
    for d in range(ND):
        if gkp[d] == 'p':
            cur_run += 1
            max_cl = max(max_cl, cur_run)
        else:
            cur_run = 0
    return max_cl


def gen_pairs_balanced(n, seed):
    rng = random.Random(seed)
    pool = defaultdict(list)
    seen = set()
    for _ in range(max(n * 200, 100000)):
        da, db = rng.randint(1, ND), rng.randint(1, ND)
        lo_a = 0 if da == 1 else 10**(da-1)
        lo_b = 0 if db == 1 else 10**(db-1)
        a = rng.randint(lo_a, 10**da-1)
        b = rng.randint(lo_b, 10**db-1)
        if (a, b) in seen: continue
        seen.add((a, b))
        pool[_count_carries(a, b)].append((a, b))
    target = max(1, n // max(len(pool), 1))
    out = []
    for nc in sorted(pool):
        rng.shuffle(pool[nc]); out.extend(pool[nc][:target])
    rng2 = random.Random(seed + 9999)
    while len(out) < n:
        a, b = rng2.randint(0, 10**ND-1), rng2.randint(0, 10**ND-1)
        if (a, b) not in seen: out.append((a, b)); seen.add((a, b))
    rng.shuffle(out)
    return out[:n]

def gen_data(n, fmt, seed):
    return [FMT_FN[fmt](a, b) for a, b in gen_pairs_balanced(n, seed)]


def gen_test_pairs_full(n, seed):
    """Generate test pairs where both operands are full ND-digit numbers.
    Carry-balanced like gen_pairs_balanced, but no short operands.
    """
    rng = random.Random(seed)
    lo = 10**(ND - 1)
    hi = 10**ND - 1
    pool = defaultdict(list)
    seen = set()
    for _ in range(max(n * 200, 100000)):
        a = rng.randint(lo, hi)
        b = rng.randint(lo, hi)
        if (a, b) in seen: continue
        seen.add((a, b))
        pool[_count_carries(a, b)].append((a, b))
    target = max(1, n // max(len(pool), 1))
    out = []
    for nc in sorted(pool):
        rng.shuffle(pool[nc]); out.extend(pool[nc][:target])
    while len(out) < n:
        a, b = rng.randint(lo, hi), rng.randint(lo, hi)
        if (a, b) not in seen: out.append((a, b)); seen.add((a, b))
    rng.shuffle(out)
    return out[:n]


def gen_test_data(n, fmt, seed):
    return [FMT_FN[fmt](a, b) for a, b in gen_test_pairs_full(n, seed)]


def gen_counterfactual_pairs(n, seed):
    """Generate matched (a1,b1)/(a2,b2) pairs where at a target position j:
      - local digit pair (a_j, b_j) is identical
      - carry-in differs (0 vs 1)
    This isolates the causal effect of carry on model predictions.

    Returns list of dicts with keys:
      target_j: int (0=LSB digit position)
      pair: ((a1,b1), (a2,b2))
      carry_in: (bool, bool) — carry-in at target for each pair member
    """
    rng = random.Random(seed)
    lo = 10**(ND - 1)
    hi = 10**ND - 1
    results = []
    attempts = 0
    max_attempts = n * 500

    while len(results) < n and attempts < max_attempts:
        attempts += 1
        # Pick a target digit position (0=LSB, ND-1=MSB input digit)
        target_d = rng.randint(1, ND - 1)  # skip LSB (no carry-in) and MSB carry-out

        # Generate first operand pair
        a1 = rng.randint(lo, hi)
        b1 = rng.randint(lo, hi)
        a1_s, b1_s = _pad(a1, ND), _pad(b1, ND)

        # Get the digit pair at target position
        # target_d in 0..ND-1 where 0=LSB → string index = ND-1-target_d
        si = ND - 1 - target_d
        da, db = int(a1_s[si]), int(b1_s[si])

        # Compute carry-in at target_d for pair 1
        carry1 = 0
        for i in range(ND - 1, si, -1):  # from LSB up to (but not including) target
            s = int(a1_s[i]) + int(b1_s[i]) + carry1
            carry1 = s // 10

        # Now construct pair 2: same digits at target and above, different suffix
        # so that carry-in at target flips
        want_carry2 = 1 - carry1

        # Keep digits from target position upward the same, randomise below
        for _ in range(200):
            # Build new suffix (below target)
            a2_digits = list(a1_s[:si+1])  # keep target and above
            b2_digits = list(b1_s[:si+1])
            for i in range(si + 1, ND):
                a2_digits.append(str(rng.randint(0, 9)))
                b2_digits.append(str(rng.randint(0, 9)))
            a2 = int(''.join(a2_digits))
            b2 = int(''.join(b2_digits))
            a2_s, b2_s = _pad(a2, ND), _pad(b2, ND)

            # Check carry-in at target
            carry2 = 0
            for i in range(ND - 1, si, -1):
                s = int(a2_s[i]) + int(b2_s[i]) + carry2
                carry2 = s // 10
            if carry2 == want_carry2:
                results.append({
                    'target_d': target_d,
                    'digit_pair': (da, db),
                    'pair': ((a1, b1), (a2, b2)),
                    'carry_in': (bool(carry1), bool(carry2)),
                })
                break

    return results


@torch.no_grad()
def eval_counterfactual(model, tokenizer, cf_pairs, fmt, max_len, device=None):
    """Evaluate counterfactual carry pairs on a trained diffusion model.

    For each matched pair, compare fully-masked predictions at the target position.
    Measures:
      - confidence_delta: how much carry-in changes model confidence
      - prediction_flip: whether the predicted digit changes
      - accuracy per carry-in condition
    """
    if device is None: device = DEVICE
    model.eval()
    mask_id = tokenizer.special_ids['mask']

    deltas = []  # per-pair confidence difference
    flips = 0
    correct_c0, correct_c1 = 0, 0
    total_c0, total_c1 = 0, 0

    for cf in cf_pairs:
        target_d = cf['target_d']
        for mi, (a, b) in enumerate(cf['pair']):
            s = FMT_FN[fmt](a, b)
            ids = torch.tensor(tokenizer.encode(s), device=device).unsqueeze(0)
            ans_start = s.index('=') + 1

            # Fully mask answer
            xm = ids.clone()
            xm[0, ans_start:ans_start+ANS_LEN] = mask_id
            logits = model(xm)

            # Map target_d (0=LSB) to answer position index
            if fmt == 'plain':
                ans_j = ND - target_d  # plain: [MSB(0), ..., LSB(ND)]
            else:
                ans_j = target_d       # reverse: [LSB(0), ..., MSB(ND)]

            pos = ans_start + ans_j
            cl = logits[0, pos].clone()
            cl[mask_id] = -float('inf')
            probs = F.softmax(cl, dim=-1)
            pred = probs.argmax().item()
            conf = probs.max().item()
            gold = ids[0, pos].item()

            if mi == 0:
                pred0, conf0, correct0 = pred, conf, (pred == gold)
            else:
                pred1, conf1, correct1 = pred, conf, (pred == gold)

        ci0, ci1 = cf['carry_in']
        # ci0=False → no carry-in for pair 0
        if not ci0:
            correct_c0 += correct0; total_c0 += 1
            correct_c1 += correct1; total_c1 += 1
        else:
            correct_c1 += correct0; total_c1 += 1
            correct_c0 += correct1; total_c0 += 1

        deltas.append(abs(conf1 - conf0))
        if pred0 != pred1:
            flips += 1

    n = len(cf_pairs)
    return {
        'n_pairs': n,
        'mean_conf_delta': sum(deltas) / max(n, 1),
        'median_conf_delta': sorted(deltas)[n//2] if n > 0 else 0,
        'prediction_flip_rate': flips / max(n, 1),
        'acc_carry_in_0': correct_c0 / max(total_c0, 1),
        'acc_carry_in_1': correct_c1 / max(total_c1, 1),
    }


def gen_carry_heavy_test(n, fmt, seed, min_max_chain=3):
    """Generate test samples with long carry-propagation chains.
    Filters for samples where max carry chain length >= min_max_chain.
    Useful as adversarial eval: if model truly learns carry structure,
    performance should degrade gracefully on these vs random test.
    """
    rng = random.Random(seed)
    lo = 10**(ND - 1)
    hi = 10**ND - 1
    results = []
    attempts = 0
    while len(results) < n and attempts < n * 200:
        attempts += 1
        a = rng.randint(lo, hi)
        b = rng.randint(lo, hi)
        ccl = _carry_chain_length_at_pos(a, b, fmt)
        if max(ccl) >= min_max_chain:
            results.append(FMT_FN[fmt](a, b))
    # If not enough, relax threshold
    if len(results) < n:
        rng2 = random.Random(seed + 7777)
        while len(results) < n:
            a = rng2.randint(lo, hi)
            b = rng2.randint(lo, hi)
            results.append(FMT_FN[fmt](a, b))
    return results[:n]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Per-position eval probe
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@torch.no_grad()
def probe_per_position(model, tokenizer, test_samples, objective,
                       fmt, max_len, device=None):
    """
    AR: teacher-forced loss per position.
    Diffusion: fully-masked loss per position.

    For diffusion, also tracks dependency-context-level confidence and accuracy
    (piggybacks on the same forward pass — zero additional GPU cost).
    This enables tracking carry chain recognition vs accuracy over training.
    """
    if device is None: device = DEVICE
    model.eval()
    mask_id = tokenizer.special_ids['mask']

    ids_all, ans_all = encode_samples(test_samples, tokenizer, max_len)
    ids_all, ans_all = ids_all.to(device), ans_all.to(device)

    ci_flags = [_carry_at_answer_pos(*_parse_operands(s), fmt) for s in test_samples]

    # Precompute dependency contexts for all test samples (cheap, CPU)
    dep_ctxs = [_dependency_context_at_pos(*_parse_operands(s), fmt) for s in test_samples]
    ccl_all = [_carry_chain_length_at_pos(*_parse_operands(s), fmt) for s in test_samples]

    L  = torch.zeros(ANS_LEN, device=device)
    C  = torch.zeros(ANS_LEN, device=device)
    CF = torch.zeros(ANS_LEN, device=device)
    N  = torch.zeros(ANS_LEN, device=device)
    Lc, Ln = torch.zeros(ANS_LEN, device=device), torch.zeros(ANS_LEN, device=device)
    Cc, Cn = torch.zeros(ANS_LEN, device=device), torch.zeros(ANS_LEN, device=device)
    Nc, Nn = torch.zeros(ANS_LEN, device=device), torch.zeros(ANS_LEN, device=device)

    # Dependency context accumulators (diffusion only, same forward pass)
    dep_conf_sum = defaultdict(float)
    dep_acc_sum = defaultdict(float)
    dep_count = defaultdict(int)
    # Chain length accumulators (p positions only)
    cl_conf_sum = defaultdict(float)
    cl_acc_sum = defaultdict(float)
    cl_count = defaultdict(int)

    # Precompute carry/dep/chain as tensors for vectorized accumulation
    ci_tensor = torch.tensor(ci_flags, dtype=torch.bool, device=device)  # [N_test, ANS_LEN]
    # Encode dep contexts and chain lengths as integers for efficient grouping
    dep_ctx_names = ['g', 'k', 'p_above_g', 'p_above_k', 'p_above_p', 'p_bottom', 'carry_out']
    dep_to_id = {name: i for i, name in enumerate(dep_ctx_names)}
    dep_ids = torch.tensor([[dep_to_id.get(d, 0) for d in dep] for dep in dep_ctxs],
                           dtype=torch.long, device=device)  # [N_test, ANS_LEN]
    ccl_tensor = torch.tensor(ccl_all, dtype=torch.long, device=device)  # [N_test, ANS_LEN]

    for st in range(0, len(test_samples), 128):
        en = min(st+128, len(test_samples))
        ids, ans = ids_all[st:en], ans_all[st:en]
        B = ids.shape[0]
        T = ids.shape[1]

        # Build answer position indices [B, ANS_LEN]
        ans_pos = ans.unsqueeze(1) + torch.arange(ANS_LEN, device=device)
        ans_pos = ans_pos.clamp(max=T-1)
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand_as(ans_pos)

        if objective == 'ar':
            logits = model(ids[:, :-1])
            # For AR: predict position a_s+j from logits at a_s+j-1
            pred_pos = ans_pos - 1  # [B, ANS_LEN]
            valid = (pred_pos >= 0) & (pred_pos < logits.shape[1])
            pred_pos = pred_pos.clamp(min=0, max=logits.shape[1]-1)

            tgt_ids = ids[batch_idx, ans_pos]  # [B, ANS_LEN]
            log_probs = F.log_softmax(logits[batch_idx, pred_pos], dim=-1)  # [B, ANS_LEN, V]
            losses = -log_probs.gather(2, tgt_ids.unsqueeze(2)).squeeze(2)  # [B, ANS_LEN]
            preds = logits[batch_idx, pred_pos].argmax(dim=-1)  # [B, ANS_LEN]
            corrects = (preds == tgt_ids).float()  # [B, ANS_LEN]
            confs = F.softmax(logits[batch_idx, pred_pos], dim=-1).max(dim=-1).values

            # Zero out invalid positions
            losses = losses * valid.float()
            corrects = corrects * valid.float()
            confs = confs * valid.float()
            v_count = valid.float()

            for j in range(ANS_LEN):
                n_v = v_count[:, j].sum()
                L[j] += losses[:, j].sum(); C[j] += corrects[:, j].sum()
                CF[j] += confs[:, j].sum(); N[j] += n_v
            # Carry-conditioned (batched)
            ci_batch = ci_tensor[st:en]  # [B, ANS_LEN]
            for j in range(ANS_LEN):
                vm = valid[:, j]
                ci_j = ci_batch[:, j] & vm
                nc_j = ~ci_batch[:, j] & vm
                Lc[j] += losses[ci_j, j].sum(); Cc[j] += corrects[ci_j, j].sum(); Nc[j] += ci_j.sum()
                Ln[j] += losses[nc_j, j].sum(); Cn[j] += corrects[nc_j, j].sum(); Nn[j] += nc_j.sum()

        else:  # diffusion
            xm = ids.clone()
            xm[batch_idx, ans_pos] = mask_id
            logits = model(xm)

            # Extract logits at answer positions [B, ANS_LEN, V]
            ans_logits = logits[batch_idx, ans_pos]
            tgt_ids = ids[batch_idx, ans_pos]  # [B, ANS_LEN]

            log_probs = F.log_softmax(ans_logits, dim=-1)
            losses = -log_probs.gather(2, tgt_ids.unsqueeze(2)).squeeze(2)  # [B, ANS_LEN]

            # Confidence with MASK excluded
            cl = ans_logits.clone()
            cl[:, :, mask_id] = -float('inf')
            probs = F.softmax(cl, dim=-1)
            confs = probs.max(dim=-1).values  # [B, ANS_LEN]
            preds = probs.argmax(dim=-1)  # [B, ANS_LEN]
            corrects = (preds == tgt_ids).float()

            for j in range(ANS_LEN):
                L[j] += losses[:, j].sum(); C[j] += corrects[:, j].sum()
                CF[j] += confs[:, j].sum(); N[j] += B
            # Carry-conditioned
            ci_batch = ci_tensor[st:en]
            for j in range(ANS_LEN):
                ci_j = ci_batch[:, j]
                nc_j = ~ci_j
                Lc[j] += losses[ci_j, j].sum(); Cc[j] += corrects[ci_j, j].sum(); Nc[j] += ci_j.sum()
                Ln[j] += losses[nc_j, j].sum(); Cn[j] += corrects[nc_j, j].sum(); Nn[j] += nc_j.sum()

            # Dependency context tracking (vectorized accumulation)
            dep_batch = dep_ids[st:en]    # [B, ANS_LEN]
            ccl_batch = ccl_tensor[st:en]  # [B, ANS_LEN]
            confs_flat = confs.reshape(-1)
            corrects_flat = corrects.reshape(-1)
            dep_flat = dep_batch.reshape(-1)
            ccl_flat = ccl_batch.reshape(-1)
            for di, dname in enumerate(dep_ctx_names):
                mask = (dep_flat == di)
                if mask.any():
                    dep_conf_sum[dname] += confs_flat[mask].sum().item()
                    dep_acc_sum[dname] += corrects_flat[mask].sum().item()
                    dep_count[dname] += mask.sum().item()
            # Chain length stats (p positions: chain > 0)
            p_mask = (ccl_flat > 0)
            if p_mask.any():
                for cl_val in ccl_flat[p_mask].unique().tolist():
                    cl_mask = (ccl_flat == cl_val)
                    cl_conf_sum[cl_val] += confs_flat[cl_mask].sum().item()
                    cl_acc_sum[cl_val] += corrects_flat[cl_mask].sum().item()
                    cl_count[cl_val] += cl_mask.sum().item()

    s = N.clamp(1)
    pos_conf = (CF/s).cpu().tolist()
    # Carry-conditioned: use None where no samples exist (e.g. LSB has no carry-in)
    def _safe_div(num, den):
        out = []
        for n, d in zip(num.cpu().tolist(), den.cpu().tolist()):
            out.append(n / d if d > 0 else None)
        return out
    result = {
        'pos_loss': (L/s).cpu().tolist(),
        'pos_acc': (C/s).cpu().tolist(),
        'pos_conf': pos_conf,
        'pos_loss_carry_in': _safe_div(Lc, Nc),
        'pos_loss_no_carry': _safe_div(Ln, Nn),
        'pos_acc_carry_in': _safe_div(Cc, Nc),
        'pos_acc_no_carry': _safe_div(Cn, Nn),
        'overall_loss': (L.sum()/s.sum()).item(),
        'overall_acc': (C.sum()/s.sum()).item(),
    }

    # Dependency-context stats (diffusion only, piggybacked on existing pass)
    if objective == 'diffusion' and dep_count:
        dep_stats = {}
        for ctx in dep_count:
            n = dep_count[ctx]
            dep_stats[ctx] = {
                'conf': dep_conf_sum[ctx] / n,
                'acc': dep_acc_sum[ctx] / n,
                'n': n,
            }
        result['dep_context'] = dep_stats
        # Chain length stats (p positions only)
        cl_stats = {}
        for chain in sorted(cl_count):
            n = cl_count[chain]
            cl_stats[chain] = {
                'conf': cl_conf_sum[chain] / n,
                'acc': cl_acc_sum[chain] / n,
                'n': n,
            }
        result['chain_len_stats'] = cl_stats

    # Confidence-derived decode ordering metrics (diffusion only).
    # From fully-masked probe confidence, compute what ordering the model
    # would choose if it decoded right now via confidence policy.
    if objective == 'diffusion':
        # Confidence ranking: highest confidence = decoded first (rank 0)
        conf_ranking = sorted(range(ANS_LEN), key=lambda j: pos_conf[j], reverse=True)
        result['conf_ranking'] = conf_ranking
        # Pairwise concordance with LSB-first oracle
        # (same logic as _analyse_orders but from probe confidence, not generation)
        conc = 0
        for i in range(ANS_LEN):
            for j in range(i + 1, ANS_LEN):
                ri = conf_ranking.index(i)  # rank of position i
                rj = conf_ranking.index(j)  # rank of position j
                if fmt == 'plain':
                    conc += int(rj < ri)   # j (more LSB) decoded first
                else:
                    conc += int(ri < rj)   # i (more LSB) decoded first
        n_pairs = ANS_LEN * (ANS_LEN - 1) // 2
        result['conf_concordance'] = conc / n_pairs
        # Confidence spread: max - min (0 = uniform, high = strong ordering)
        result['conf_spread'] = max(pos_conf) - min(pos_conf)
    return result


@torch.no_grad()
def probe_gkp_detailed(model, tokenizer, test_samples, fmt, max_len, device=None):
    """Sample-level fully-masked confidence analysis by dependency context.

    For each sample and position, records the model's confidence in the
    fully-masked setting, tagged with:
      - g/k/p category
      - dependency context (p_above_g, p_above_k, p_above_p, etc.)
      - carry chain length
      - accuracy (correct or not)

    Returns dict with:
      'by_dep_context': {context: {'conf': [...], 'acc': [...]}}
      'by_chain_len':   {chain_len: {'conf': [...], 'acc': [...]}}
      'by_gkp':         {gkp: {'conf': [...], 'acc': [...]}}
      'chain_conf_corr': Pearson correlation(chain_len, confidence) across all p positions
      'no_p_samples':   {'n': int, 'one_step_acc': float, 'one_step_digit_acc': float}
    """
    if device is None: device = DEVICE
    model.eval()
    mask_id = tokenizer.special_ids['mask']

    ids_all, ans_all = encode_samples(test_samples, tokenizer, max_len)
    ids_all, ans_all = ids_all.to(device), ans_all.to(device)
    N_test = len(test_samples)
    T = ids_all.shape[1]

    # Precompute all metadata as tensors
    dep_ctx_names = ['g', 'k', 'p_above_g', 'p_above_k', 'p_above_p', 'p_bottom', 'carry_out']
    dep_to_id = {name: i for i, name in enumerate(dep_ctx_names)}
    gkp_names = ['g', 'k', 'p', 'carry_out']
    gkp_to_id = {name: i for i, name in enumerate(gkp_names)}

    all_dep = []
    all_gkp = []
    all_ccl = []
    all_has_p = []
    for s in test_samples:
        a, b = _parse_operands(s)
        dep = _dependency_context_at_pos(a, b, fmt)
        gkp = _gkp_at_answer_pos(a, b, fmt)
        ccl = _carry_chain_length_at_pos(a, b, fmt)
        all_dep.append([dep_to_id.get(d, 0) for d in dep])
        all_gkp.append([gkp_to_id.get(g, 0) for g in gkp])
        all_ccl.append(ccl)
        all_has_p.append(any(g == 'p' for g in gkp))

    dep_ids = torch.tensor(all_dep, dtype=torch.long, device=device)  # [N, ANS_LEN]
    gkp_ids = torch.tensor(all_gkp, dtype=torch.long, device=device)
    ccl_t = torch.tensor(all_ccl, dtype=torch.long, device=device)
    has_p = torch.tensor(all_has_p, dtype=torch.bool, device=device)

    # Accumulators
    all_confs = torch.zeros(N_test, ANS_LEN, device=device)
    all_corrects = torch.zeros(N_test, ANS_LEN, device=device)

    for st in range(0, N_test, 128):
        en = min(st + 128, N_test)
        ids, ans = ids_all[st:en], ans_all[st:en]
        B = ids.shape[0]

        ans_pos = ans.unsqueeze(1) + torch.arange(ANS_LEN, device=device)
        ans_pos = ans_pos.clamp(max=T-1)
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand_as(ans_pos)

        xm = ids.clone()
        xm[batch_idx, ans_pos] = mask_id
        logits = model(xm)

        ans_logits = logits[batch_idx, ans_pos]  # [B, ANS_LEN, V]
        tgt_ids = ids[batch_idx, ans_pos]
        cl = ans_logits.clone()
        cl[:, :, mask_id] = -float('inf')
        probs = F.softmax(cl, dim=-1)
        confs = probs.max(dim=-1).values
        preds = probs.argmax(dim=-1)
        corrects = (preds == tgt_ids).float()

        all_confs[st:en] = confs
        all_corrects[st:en] = corrects

    # Aggregate by dependency context
    dep_flat = dep_ids.reshape(-1)
    gkp_flat = gkp_ids.reshape(-1)
    ccl_flat = ccl_t.reshape(-1)
    conf_flat = all_confs.reshape(-1)
    acc_flat = all_corrects.reshape(-1)

    by_dep = {}
    for di, dname in enumerate(dep_ctx_names):
        mask = (dep_flat == di)
        if mask.any():
            by_dep[dname] = {'conf': conf_flat[mask].tolist(), 'acc': acc_flat[mask].tolist()}
    by_gkp = {}
    for gi, gname in enumerate(gkp_names):
        mask = (gkp_flat == gi)
        if mask.any():
            by_gkp[gname] = {'conf': conf_flat[mask].tolist(), 'acc': acc_flat[mask].tolist()}
    by_cl = {}
    for cl_val in ccl_flat.unique().tolist():
        mask = (ccl_flat == cl_val)
        by_cl[cl_val] = {'conf': conf_flat[mask].tolist(), 'acc': acc_flat[mask].tolist()}

    # Chain_len vs confidence correlation (p positions only)
    p_mask = (gkp_flat == gkp_to_id['p'])
    chain_conf_corr = None
    if p_mask.sum() >= 10:
        x = ccl_flat[p_mask].float()
        y = conf_flat[p_mask]
        xm, ym = x.mean(), y.mean()
        cov = ((x - xm) * (y - ym)).sum()
        sx = ((x - xm)**2).sum().sqrt()
        sy = ((y - ym)**2).sum().sqrt()
        chain_conf_corr = float(cov / (sx * sy)) if sx > 0 and sy > 0 else 0.0

    # No-propagate samples
    no_p_mask = ~has_p
    no_p_count = no_p_mask.sum().item()
    if no_p_count > 0:
        no_p_exact = (all_corrects[no_p_mask].sum(dim=1) == ANS_LEN).float().mean().item()
        no_p_digit = all_corrects[no_p_mask].mean().item()
    else:
        no_p_exact = 0.0
        no_p_digit = 0.0

    def _summarise(d):
        return {k: {'mean_conf': sum(v['conf'])/max(len(v['conf']),1),
                     'mean_acc': sum(v['acc'])/max(len(v['acc']),1),
                     'n': len(v['conf'])}
                for k, v in sorted(d.items(), key=lambda x: str(x[0]))}

    return {
        'by_dep_context': _summarise(by_dep),
        'by_chain_len': _summarise(by_cl),
        'by_gkp': _summarise(by_gkp),
        'chain_conf_corr': chain_conf_corr,
        'no_p_samples': {
            'n': no_p_count,
            'one_step_exact_acc': no_p_exact,
            'one_step_digit_acc': no_p_digit,
        },
    }



@torch.no_grad()
def analyse_confidence_cascade(model, tokenizer, test_samples, fmt, max_len,
                                n_samples=200, device=None):
    """Step-by-step generation with confidence tracking at each decode step.

    The core question: when position j (g or k) is revealed, does the
    confidence of the position ABOVE it (toward MSB, which receives j's carry)
    jump up? And when a p is revealed, does it NOT jump?

    This directly tests whether the model's generation dynamics reflect
    the carry dependency graph.

    Carry direction:
      plain:   carry flows right→left, so "above j" = position j-1
      reverse: carry flows left→right, so "above j" = position j+1

    Returns dict with:
      'delta_by_revealed_type': {g/k/p: mean Δconf of position above}
      'delta_by_revealed_type_all': {g/k/p: mean Δconf of ALL other positions}
      'step_confs': list of (step, {position: conf}) for trajectory visualization
      'cascade_pairs': list of (revealed_type, delta_above, delta_others) per event
    """
    if device is None: device = DEVICE
    model.eval()
    mask_id = tokenizer.special_ids['mask']
    pad_id = tokenizer.special_ids['pad']

    samples = test_samples[:n_samples]

    # Precompute metadata
    all_gkp = []
    all_dep = []
    for s in samples:
        a, b = _parse_operands(s)
        all_gkp.append(_gkp_at_answer_pos(a, b, fmt))
        all_dep.append(_dependency_context_at_pos(a, b, fmt))

    # "Above" direction: position that RECEIVES carry from position j
    # plain:  carry flows LSB(right)→MSB(left), above j = j-1
    # reverse: carry flows LSB(left)→MSB(right), above j = j+1
    def _above(j):
        if fmt == 'plain':
            return j - 1 if j > 0 else None
        else:
            return j + 1 if j < ANS_LEN - 1 else None

    # Accumulators
    delta_above = {'g': [], 'k': [], 'p': []}       # by type of revealed position
    delta_others = {'g': [], 'k': [], 'p': []}
    # Refined: by dep context of revealed position
    # "chain_end" = g, k, p_above_g, p_above_k, p_bottom → revealing resolves carry
    # "chain_mid" = p_above_p → revealing does NOT resolve carry (still ambiguous above)
    delta_above_resolved = []    # revealed position is chain terminator
    delta_above_unresolved = []  # revealed position is mid-chain p

    # Per-sample step-by-step generation
    for si, s in enumerate(samples):
        gkp = all_gkp[si]
        prefix = s.split('=')[0] + '='
        penc = tokenizer.encode(prefix)
        T_pre = len(penc)

        # Build input: prefix + MASK * ANS_LEN
        x = torch.full((1, T_pre + ANS_LEN), mask_id, dtype=torch.long, device=device)
        x[0, :T_pre] = torch.tensor(penc, device=device)
        unmasked = torch.zeros(T_pre + ANS_LEN, dtype=torch.bool, device=device)
        unmasked[:T_pre] = True

        prev_conf = torch.zeros(ANS_LEN, device=device)

        # Get initial confidence
        logits = model(x)
        for j in range(ANS_LEN):
            pos = T_pre + j
            cl = logits[0, pos].clone()
            cl[mask_id] = -float('inf')
            prev_conf[j] = F.softmax(cl, dim=-1).max()

        # Step-by-step decode (confidence policy)
        for step in range(ANS_LEN):
            logits = model(x)

            # Find highest-confidence masked position
            best_conf = -1.0
            best_j = -1
            for j in range(ANS_LEN):
                if unmasked[T_pre + j]:
                    continue
                pos = T_pre + j
                cl = logits[0, pos].clone()
                cl[mask_id] = -float('inf')
                c = F.softmax(cl, dim=-1).max().item()
                if c > best_conf:
                    best_conf = c
                    best_j = j

            if best_j < 0:
                break

            # Reveal this position
            pos = T_pre + best_j
            cl = logits[0, pos].clone()
            cl[mask_id] = -float('inf')
            tok = cl.argmax()
            x[0, pos] = tok
            unmasked[pos] = True

            # Get new confidences for remaining masked positions
            logits_new = model(x)
            new_conf = torch.zeros(ANS_LEN, device=device)
            for j in range(ANS_LEN):
                if unmasked[T_pre + j]:
                    new_conf[j] = 1.0  # already revealed
                    continue
                p = T_pre + j
                cl2 = logits_new[0, p].clone()
                cl2[mask_id] = -float('inf')
                new_conf[j] = F.softmax(cl2, dim=-1).max()

            # Record Δconf
            revealed_type = gkp[best_j]
            revealed_dep = all_dep[si][best_j]
            if revealed_type in ('g', 'k', 'p'):
                above_j = _above(best_j)

                # Δconf of the position directly above (dependent on carry)
                if above_j is not None and not unmasked[T_pre + above_j]:
                    d_above = (new_conf[above_j] - prev_conf[above_j]).item()
                    delta_above[revealed_type].append(d_above)

                    # Chain resolution grouping:
                    # g, k → carry-out known from input alone (chain terminator)
                    # p_above_g, p_above_k, p_bottom → carry resolved (chain end)
                    # p_above_p → carry NOT resolved (mid-chain)
                    if revealed_dep in ('g', 'k', 'p_above_g', 'p_above_k', 'p_bottom'):
                        delta_above_resolved.append(d_above)
                    elif revealed_dep == 'p_above_p':
                        delta_above_unresolved.append(d_above)

                # Δconf of all OTHER masked positions (not the one above)
                for j in range(ANS_LEN):
                    if unmasked[T_pre + j] or j == above_j:
                        continue
                    d = (new_conf[j] - prev_conf[j]).item()
                    delta_others[revealed_type].append(d)

            prev_conf = new_conf

    # Summarise
    def _mean(lst):
        return sum(lst) / len(lst) if lst else 0.0

    return {
        'delta_conf_above_by_type': {
            t: {'mean': _mean(v), 'n': len(v)} for t, v in delta_above.items()
        },
        'delta_conf_others_by_type': {
            t: {'mean': _mean(v), 'n': len(v)} for t, v in delta_others.items()
        },
        'delta_conf_above_resolved': {
            'chain_end': {'mean': _mean(delta_above_resolved),
                          'n': len(delta_above_resolved)},
            'chain_mid': {'mean': _mean(delta_above_unresolved),
                          'n': len(delta_above_unresolved)},
        },
        'summary': (
            f"By revealed type (Δconf of position above):\n"
            f"  g: {_mean(delta_above['g']):+.4f} (n={len(delta_above['g'])}), "
            f"  k: {_mean(delta_above['k']):+.4f} (n={len(delta_above['k'])}), "
            f"  p: {_mean(delta_above['p']):+.4f} (n={len(delta_above['p'])})\n"
            f"By chain resolution:\n"
            f"  chain_end (g/k/p_above_gk): {_mean(delta_above_resolved):+.4f} "
            f"(n={len(delta_above_resolved)})\n"
            f"  chain_mid (p_above_p):      {_mean(delta_above_unresolved):+.4f} "
            f"(n={len(delta_above_unresolved)})"
        ),
    }



# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Training (epoch-based, fixed budget)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _quick_gen_eval(model, tokenizer, test_samples, objective, fmt,
                    decode_policy='confidence', n=GEN_EVAL_N, device=None):
    """Lightweight generation eval on a subset. Returns full result dict."""
    if device is None: device = DEVICE
    subset = test_samples[:n]
    return final_evaluate(model, tokenizer, subset, objective, fmt,
                          decode_policy=decode_policy)


def train_with_dynamics(
    objective, tokenizer, train_samples, test_samples,
    max_len, fmt, mask_type='random', device=None,
):
    """
    Epoch-based training with periodic per-position probes.

    Replaces core.train_model for this experiment because:
      1. Epoch-based loop (core uses iteration-based while-loop)
      2. Periodic probe_per_position eval (core has no mid-training probes)
      3. Ordered masking support (core only has random masking)
      4. Token-gradient counting for fair AR/Diffusion comparison
    """
    if device is None: device = DEVICE

    train_ids, train_ans = encode_samples(train_samples, tokenizer, max_len)
    # Pre-move to GPU — eliminates CPU→GPU transfer every batch
    train_ids = train_ids.to(device)
    train_ans = train_ans.to(device)
    N = train_ids.shape[0]
    bpe = (N + BATCH_SIZE - 1) // BATCH_SIZE  # batches per epoch
    total_iters = MAX_EPOCHS * bpe

    mask_id = tokenizer.special_ids['mask']
    pad_id = tokenizer.special_ids['pad']
    is_causal = (objective == 'ar')

    model = Transformer(
        vocab_size=len(tokenizer), block_size=max_len + 8,
        n_layer=N_LAYER, n_head=N_HEAD, n_embd=N_EMBD,
        dropout=DROPOUT, is_causal=is_causal, pos_enc=POS_ENC,
    ).to(device)

    tag = objective + (f"/{mask_type}" if objective == 'diffusion' else "")
    print(f"  [{tag}|{POS_ENC}] params={model.n_params:,}, "
          f"seq_len={max_len}, {bpe} batches/epoch, {MAX_EPOCHS} epochs")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, betas=(0.9, 0.99), weight_decay=0.1)
    warmup_iters = WARMUP_EPOCHS * bpe

    def get_lr(it):
        if it < warmup_iters:
            return LR * it / max(warmup_iters, 1)
        ratio = (it - warmup_iters) / max(total_iters - warmup_iters, 1)
        return MIN_LR + 0.5 * (LR - MIN_LR) * (1 + math.cos(math.pi * min(ratio, 1.0)))

    dynamics = {'checkpoints': [], 'gen_checkpoints': [], 'train_loss': []}
    best_loss = float('inf')  # held-out probe loss (not training loss)
    best_state = None
    it = 0
    tg = 0  # token-gradient count
    t0 = time.time()

    # Precompute constant tensors (avoid re-creation every iteration)
    T = train_ids.shape[1]
    _arange_ans = torch.arange(ANS_LEN, device=device)
    _arange_T = torch.arange(T, device=device)
    _arange_Tm1 = torch.arange(T - 1, device=device)
    _refresh_all = list(range(BATCH_SIZE))

    def _pos_labels():
        if fmt == 'plain':
            return ['MSB'] + [f'p{j}' for j in range(1, ANS_LEN-1)] + ['LSB']
        return ['LSB'] + [f'p{j}' for j in range(1, ANS_LEN-1)] + ['MSB']

    def _do_eval(epoch):
        nonlocal best_loss, best_state
        probe = probe_per_position(
            model, tokenizer, test_samples, objective, fmt, max_len, device)
        dynamics['checkpoints'].append({
            'epoch': epoch, 'iter': it, 'token_gradients': tg, **probe})

        # Best model by held-out probe loss (fair across mask types)
        pl = probe['overall_loss']
        if pl < best_loss and epoch > 0:
            best_loss = pl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        labs = _pos_labels()
        acc_str = ' '.join(f"{labs[j]}={probe['pos_acc'][j]:.2f}"
                           for j in range(ANS_LEN))
        print(f"    [eval ep {epoch:4d}] loss={probe['overall_loss']:.4f} "
              f"acc={probe['overall_acc']:.4f}"
              + (f" conc={probe['conf_concordance']:.3f} spread={probe['conf_spread']:.4f}"
                 if 'conf_concordance' in probe else '')
              + f" | {time.time()-t0:.0f}s")
        print(f"      {acc_str}")
        # Compact dependency context summary (diffusion only)
        dc = probe.get('dep_context')
        if dc:
            parts = []
            for ctx in ['g', 'k', 'p_above_g', 'p_above_k', 'p_above_p']:
                if ctx in dc:
                    parts.append(f"{ctx}={dc[ctx]['conf']:.2f}/{dc[ctx]['acc']:.2f}")
            if parts:
                print(f"      dep(conf/acc): {' '.join(parts)}")

        # Generation accuracy tracking (less frequent, more expensive)
        # Front-loaded: denser in first 10% to catch early dynamics
        gen_eval_now = (epoch % GEN_EVAL_EVERY == 0)
        if not gen_eval_now and epoch < MAX_EPOCHS * 0.1:
            gen_eval_now = (epoch % max(GEN_EVAL_EVERY // 4, 1) == 0)
        elif not gen_eval_now and epoch < MAX_EPOCHS * 0.3:
            gen_eval_now = (epoch % max(GEN_EVAL_EVERY // 2, 1) == 0)
        if gen_eval_now:
            gen_entry = {'epoch': epoch}
            if objective == 'ar':
                r = _quick_gen_eval(
                    model, tokenizer, test_samples, 'ar', fmt, device=device)
                gen_entry['gen_acc'] = {'ar': r['accuracy']}
                gen_entry['gen_pos_acc'] = {'ar': r['position_accuracy']}
                gen_entry['gen_carry_out'] = {'ar': {
                    'carry': r.get('pos_acc_carry_in', [None]*ANS_LEN),
                    'no_carry': r.get('pos_acc_no_carry', [None]*ANS_LEN)}}
                print(f"      [gen] ar={r['accuracy']:.3f}")
            else:
                ga_strs = []
                # Only confidence decode during training (lsb evaluated at final)
                for dp in ['confidence']:
                    r = _quick_gen_eval(
                        model, tokenizer, test_samples, 'diffusion', fmt,
                        decode_policy=dp, device=device)
                    gen_entry.setdefault('gen_acc', {})[dp] = r['accuracy']
                    gen_entry.setdefault('gen_pos_acc', {})[dp] = r['position_accuracy']
                    gen_entry.setdefault('gen_carry_out', {})[dp] = {
                        'carry': r.get('pos_acc_carry_in', [None]*ANS_LEN),
                        'no_carry': r.get('pos_acc_no_carry', [None]*ANS_LEN)}
                    oa = r.get('decode_order_analysis')
                    if oa:
                        gen_entry.setdefault('concordance', {})[dp] = oa['pairwise_concordance']
                        gen_entry.setdefault('mean_rank', {})[dp] = oa['mean_rank']
                        if 'gkp_mean_rank' in oa:
                            gen_entry.setdefault('gkp_mean_rank', {})[dp] = oa['gkp_mean_rank']
                        if 'easy_before_hard' in oa:
                            gen_entry.setdefault('easy_before_hard', {})[dp] = oa['easy_before_hard']
                        if 'chain_len_vs_rank' in oa:
                            gen_entry.setdefault('chain_len_vs_rank', {})[dp] = oa['chain_len_vs_rank']
                        if 'carry_source_rank' in oa:
                            gen_entry.setdefault('carry_source_rank', {})[dp] = oa['carry_source_rank']
                    ga_strs.append(f"{dp}={r['accuracy']:.3f}")
                print(f"      [gen] {' '.join(ga_strs)}")
            dynamics['gen_checkpoints'].append(gen_entry)

    # ── Eval at epoch 0 ──
    model.eval(); _do_eval(0); model.train()

    # ── Streaming buffer (for mask_type in {puma, oracle_lsb}) ──
    uses_streaming = mask_type in ('puma', 'oracle_lsb')
    puma_x0 = puma_z = puma_ans = puma_stage = None
    puma_pool_idx = 0
    if uses_streaming:
        puma_all_ids, puma_all_ans = encode_samples(train_samples, tokenizer, max_len)
        puma_all_ids = puma_all_ids.to(device)
        puma_all_ans = puma_all_ans.to(device)
        puma_perm = torch.randperm(len(train_samples))

    def _puma_refresh(indices):
        """Initialize or refresh streaming buffer entries (vectorized)."""
        nonlocal puma_x0, puma_z, puma_ans, puma_stage, puma_pool_idx, puma_perm
        if puma_x0 is None:
            B = len(indices)
            T = puma_all_ids.shape[1]
            puma_x0 = torch.zeros(B, T, dtype=torch.long, device=device)
            puma_z = torch.zeros(B, T, dtype=torch.long, device=device)
            puma_ans = torch.zeros(B, dtype=torch.long, device=device)
            puma_stage = torch.zeros(B, dtype=torch.long, device=device)
        idx_t = torch.tensor(indices, device=device)
        n = len(indices)
        # Gather pool indices, handle wrap-around
        pool_end = puma_pool_idx + n
        if pool_end <= len(puma_perm):
            si = puma_perm[puma_pool_idx:pool_end]
            puma_pool_idx = pool_end
        else:
            part1 = puma_perm[puma_pool_idx:]
            puma_perm = torch.randperm(len(train_samples))
            remaining = n - len(part1)
            part2 = puma_perm[:remaining]
            si = torch.cat([part1, part2])
            puma_pool_idx = remaining
        si = si.to(device)
        puma_x0[idx_t] = puma_all_ids[si]
        puma_z[idx_t] = puma_all_ids[si].clone()
        puma_ans[idx_t] = puma_all_ans[si]
        puma_stage[idx_t] = 0
        # Mask answer regions (vectorized)
        ans_pos = puma_ans[idx_t].unsqueeze(1) + torch.arange(ANS_LEN, device=device)
        ans_pos = ans_pos.clamp(max=puma_z.shape[1]-1)
        bi_exp = idx_t.unsqueeze(1).expand_as(ans_pos)
        puma_z[bi_exp, ans_pos] = mask_id

    def _chain_advance(logits_det, K_cur):
        """Advance chains (vectorized). Policy depends on mask_type:
        - puma: reveal highest-confidence positions (adaptive)
        - oracle_lsb: reveal LSB-first positions (fixed order)
        """
        nonlocal puma_z, puma_stage
        B = puma_z.shape[0]
        T = puma_z.shape[1]

        # Build answer position indices [B, ANS_LEN]
        ans_pos = puma_ans.unsqueeze(1) + torch.arange(ANS_LEN, device=device)
        ans_pos = ans_pos.clamp(max=T-1)
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand_as(ans_pos)

        # Which answer positions are still masked? [B, ANS_LEN]
        is_masked = (puma_z[batch_idx, ans_pos] == mask_id)
        n_masked = is_masked.sum(dim=1)  # [B]

        # How many to reveal per sample
        K_rem = (K_cur - puma_stage).clamp(min=1)  # [B]
        n_reveal = (n_masked.float() / K_rem.float()).ceil().long().clamp(min=1)  # [B]

        if mask_type == 'puma':
            # Get confidence at answer positions
            lp = logits_det[batch_idx, ans_pos].clone()  # [B, ANS_LEN, V]
            lp[:, :, mask_id] = -float('inf')
            confs = F.softmax(lp, dim=-1).max(dim=-1).values  # [B, ANS_LEN]
            # Non-masked positions get -inf so they're never selected
            confs[~is_masked] = -float('inf')
            # Rank descending by confidence
            ranked = confs.argsort(dim=1, descending=True)  # [B, ANS_LEN]
            rank_of_pos = torch.zeros_like(ranked)
            rank_of_pos.scatter_(1, ranked, torch.arange(ANS_LEN, device=device).expand(B, -1))
            # Reveal if rank < n_reveal OR confidence > tau
            reveal = (rank_of_pos < n_reveal.unsqueeze(1)) | (confs > PUMA_TAU)
            reveal = reveal & is_masked
        else:
            # Oracle LSB-first: assign priority by position
            if fmt == 'plain':
                # plain: higher index = more LSB → reveal first → priority = index
                priority = torch.arange(ANS_LEN, device=device).expand(B, -1).float()
            else:
                # reverse: lower index = more LSB → reveal first → priority = -index
                priority = -torch.arange(ANS_LEN, device=device).expand(B, -1).float()
            priority[~is_masked] = -float('inf')
            ranked = priority.argsort(dim=1, descending=True)
            rank_of_pos = torch.zeros_like(ranked)
            rank_of_pos.scatter_(1, ranked, torch.arange(ANS_LEN, device=device).expand(B, -1))
            reveal = (rank_of_pos < n_reveal.unsqueeze(1)) & is_masked

        # Apply reveals
        reveal_abs = ans_pos[reveal]
        batch_reveal = batch_idx[reveal]
        puma_z[batch_reveal, reveal_abs] = puma_x0[batch_reveal, reveal_abs]
        puma_stage += 1

        # Check which samples are done (no masks left or stage >= K)
        still_masked = (puma_z[batch_idx, ans_pos] == mask_id).any(dim=1)
        done = (~still_masked) | (puma_stage >= K_cur)
        if done.any():
            _puma_refresh(done.nonzero(as_tuple=True)[0].tolist())

    if uses_streaming:
        _puma_refresh(list(range(BATCH_SIZE)))

    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_loss_t = torch.tensor(0.0, device=device)
        epoch_tg = torch.tensor(0, dtype=torch.long, device=device)
        epoch_n = 0
        # K scheduling for streaming methods
        if uses_streaming:
            puma_K_cur = PUMA_K_START + int((PUMA_K_END - PUMA_K_START)
                                             * epoch / MAX_EPOCHS)
        else:
            puma_K_cur = 0

        if uses_streaming:
            # ── Streaming buffer iterations (PUMA / oracle_lsb) ──
            for _ in range(bpe):
                for pg in optimizer.param_groups:
                    pg['lr'] = get_lr(it)
                m = (puma_z == mask_id)
                if m.sum() == 0:
                    _puma_refresh(list(range(BATCH_SIZE)))
                    m = (puma_z == mask_id)
                logits = model(puma_z)
                loss = F.cross_entropy(logits[m], puma_x0[m])
                epoch_tg += m.sum()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                _chain_advance(logits.detach(), puma_K_cur)

                epoch_loss_t += loss.detach()
                epoch_n += 1
                it += 1
        else:
            # ── Standard mask types (random/confidence) ──
            # GPU-side shuffle — no DataLoader, no CPU→GPU transfer
            perm = torch.randperm(N, device=device)
            for bi in range(bpe):
                for pg in optimizer.param_groups:
                    pg['lr'] = get_lr(it)

                idx = perm[bi*BATCH_SIZE : min((bi+1)*BATCH_SIZE, N)]
                ids = train_ids[idx]
                ans_starts = train_ans[idx]
                B, T = ids.shape

                if objective == 'ar':
                    logits = model(ids[:, :-1])
                    targets = ids[:, 1:]
                    pos = torch.arange(T-1, device=device).unsqueeze(0)
                    lm = pos >= (ans_starts.unsqueeze(1) - 1)
                    lm = lm & (targets != pad_id)
                    if lm.sum() == 0: it += 1; continue
                    loss = F.cross_entropy(logits[lm], targets[lm])
                    epoch_tg += lm.sum()

                else:  # diffusion (non-streaming)
                    pos = torch.arange(T, device=device).unsqueeze(0)
                    ans_mask = pos >= ans_starts.unsqueeze(1)
                    # Precompute answer position indices [B, ANS_LEN]
                    ans_pos = ans_starts.unsqueeze(1) + torch.arange(ANS_LEN, device=device)
                    ans_pos = ans_pos.clamp(max=T-1)

                    if mask_type == 'random':
                        t_ratio = torch.rand(B, device=device)
                        m_probs = t_ratio.unsqueeze(1) * ans_mask.float()
                        m = torch.bernoulli(m_probs).bool()
                        # Guarantee at least one mask per sample (vectorized)
                        no_m = ~(m.any(dim=1))
                        if no_m.any():
                            # Pick random answer position for each no-mask sample
                            rand_j = torch.randint(ANS_LEN, (no_m.sum(),), device=device)
                            fix_pos = ans_pos[no_m].gather(1, rand_j.unsqueeze(1)).squeeze(1)
                            m[no_m, fix_pos] = True

                    elif mask_type == 'confidence':
                        # Single-probe adaptive: fully-masked → rank → mask bottom (vectorized)
                        xm_probe = ids.clone()
                        # Batch mask all answer regions at once
                        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand_as(ans_pos)
                        xm_probe[batch_idx, ans_pos] = mask_id
                        model.eval()
                        with torch.no_grad():
                            logits_probe = model(xm_probe)
                        model.train()

                        # Extract logits at answer positions [B, ANS_LEN, V]
                        lp = logits_probe[batch_idx, ans_pos]
                        lp[:, :, mask_id] = -float('inf')
                        confs = F.softmax(lp, dim=-1).max(dim=-1).values  # [B, ANS_LEN]
                        # Rank by ascending confidence
                        ranked = confs.argsort(dim=1)  # [B, ANS_LEN]
                        # Number to mask per sample
                        t_ratio = torch.rand(B, device=device)
                        nm = (t_ratio * ANS_LEN).ceil().long().clamp(min=1)  # [B]
                        # Build mask: positions with rank < nm[b] get masked
                        rank_of_pos = torch.zeros_like(ranked)
                        rank_of_pos.scatter_(1, ranked, torch.arange(ANS_LEN, device=device).expand(B, -1))
                        to_mask = rank_of_pos < nm.unsqueeze(1)  # [B, ANS_LEN]
                        m = torch.zeros(B, T, dtype=torch.bool, device=device)
                        m[batch_idx, ans_pos] = to_mask

                    xm = ids.clone(); xm[m] = mask_id
                    logits = model(xm)
                    if m.sum() == 0: it += 1; continue
                    loss = F.cross_entropy(logits[m], ids[m])
                    epoch_tg += m.sum()

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

                epoch_loss_t += loss.detach()
                epoch_n += 1
                it += 1

        # ── End of epoch (single .item() call here, not per-iteration) ──
        tg += epoch_tg.item()
        avg_loss = epoch_loss_t.item() / max(epoch_n, 1)

        if epoch % LOG_EVERY == 0:
            dynamics['train_loss'].append((epoch, avg_loss))
            print(f"    ep {epoch:4d}/{MAX_EPOCHS} | loss {avg_loss:.4f} | "
                  f"lr {get_lr(it):.1e} | tg {tg:,} | {time.time()-t0:.0f}s")

        if epoch % EVAL_EVERY == 0 and epoch < MAX_EPOCHS:
            model.eval(); _do_eval(epoch); model.train()
        elif epoch < MAX_EPOCHS * 0.1 and epoch % max(EVAL_EVERY // 5, 1) == 0:
            # Dense eval in first 10% of training — catch early dynamics
            model.eval(); _do_eval(epoch); model.train()
        elif epoch < MAX_EPOCHS * 0.3 and epoch % max(EVAL_EVERY // 2, 1) == 0:
            # Medium-dense eval in next 20%
            model.eval(); _do_eval(epoch); model.train()

    # ── Load best model ──
    print(f"    ✓ Done {MAX_EPOCHS} epochs (best probe loss: {best_loss:.4f})")
    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    _do_eval(MAX_EPOCHS)
    return model, dynamics


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Final evaluation (generation-based)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _lsb_policy(fmt):
    """Map format to core generate_diffusion policy for LSB-first decoding.
    plain:   answer is [MSB..LSB] → decode right-to-left = 'r2l'
    reverse: answer is [LSB..MSB] → decode left-to-right = 'l2r'
    """
    return 'r2l' if fmt == 'plain' else 'l2r'


def final_evaluate(model, tokenizer, test_samples, objective, fmt,
                   decode_policy='confidence', batch_size=128, device=None):
    """
    Generation-based evaluation with per-position & carry analysis.

    Uses core generate_ar / generate_diffusion for generation.
    Adds per-position accuracy and carry-count breakdown on top.
    (core.evaluate doesn't track these — it's for exact match only.)
    """
    if device is None: device = DEVICE
    mask_id = tokenizer.special_ids['mask']
    pad_id = tokenizer.special_ids['pad']
    model.eval()
    results, all_orders = [], []

    for st in range(0, len(test_samples), batch_size):
        batch = test_samples[st:st+batch_size]; B = len(batch)
        penc = [tokenizer.encode(s.split('=')[0]+'=') for s in batch]
        pm = max(len(p) for p in penc)
        pids = torch.full((B, pm), pad_id, dtype=torch.long)
        for i, e in enumerate(penc): pids[i,:len(e)] = torch.tensor(e)

        with torch.no_grad():
            if objective == 'ar':
                gen = generate_ar(model, pids, ANS_LEN, device)
                pred_ids = gen[:, pm:pm+ANS_LEN]; bo = None
            else:
                # Both 'confidence' and 'lsb' use core generate_diffusion
                policy = _lsb_policy(fmt) if decode_policy == 'lsb' else 'confidence'
                gen, _, info = generate_diffusion(
                    model, pids, ANS_LEN, mask_id,
                    policy=policy, greedy=True, device=device)
                pred_ids = gen[:, pm:pm+ANS_LEN]
                bo = info.get('orders')
        if bo is not None: all_orders.append(bo)

        for i in range(B):
            ps = tokenizer.decode(pred_ids[i].cpu().tolist())
            gs = get_answer(batch[i], fmt)
            a, b = _parse_operands(batch[i])
            pc = [ps[j]==gs[j] if j<len(ps) else False for j in range(len(gs))]
            ci = _carry_at_answer_pos(a, b, fmt)
            gkp = _gkp_at_answer_pos(a, b, fmt)
            ccl = _carry_chain_length_at_pos(a, b, fmt)
            results.append({'correct': ps==gs, 'pos_correct': pc,
                           'n_carries': _count_carries(a, b),
                           'carry_flags': ci, 'gkp': gkp, 'chain_len': ccl})

    n = len(results)
    acc = sum(r['correct'] for r in results) / max(n, 1)
    pos_acc = [sum(r['pos_correct'][j] for r in results)/max(n,1) for j in range(ANS_LEN)]
    by_nc = defaultdict(list)
    for r in results: by_nc[r['n_carries']].append(r['correct'])
    carry_acc = {nc: (sum(v)/len(v), len(v)) for nc, v in sorted(by_nc.items())}

    # Per-position carry-conditioned generation accuracy
    pos_acc_carry_in = []
    pos_acc_no_carry = []
    for j in range(ANS_LEN):
        ci_correct = [r['pos_correct'][j] for r in results if r['carry_flags'][j]]
        nc_correct = [r['pos_correct'][j] for r in results if not r['carry_flags'][j]]
        pos_acc_carry_in.append(sum(ci_correct)/len(ci_correct) if ci_correct else None)
        pos_acc_no_carry.append(sum(nc_correct)/len(nc_correct) if nc_correct else None)

    oa = None
    if all_orders:
        oc = torch.cat(all_orders, dim=0)
        pl = len(tokenizer.encode(test_samples[0].split('=')[0]+'='))
        gkp_all = [r['gkp'] for r in results]
        ccl_all = [r['chain_len'] for r in results]
        oa = _analyse_orders(oc, pl, fmt, gkp_all, ccl_all)

    return {'accuracy': acc, 'n_samples': n, 'position_accuracy': pos_acc,
            'carry_accuracy': carry_acc, 'decode_order_analysis': oa,
            'pos_acc_carry_in': pos_acc_carry_in,
            'pos_acc_no_carry': pos_acc_no_carry}


def _analyse_orders(decode_orders, prefix_len, fmt,
                    gkp_per_sample=None, chain_len_per_sample=None):
    """Per-position decode rank analysis (vectorized).
    Computes:
      mean_rank:          mean decode step per position
      pairwise_conc:      fraction of position pairs decoded in LSB→MSB order
      rank_histogram:     (ANS_LEN, S) count matrix
      gkp_mean_rank:      mean decode rank by g/k/p category
      easy_before_hard:   fraction of (g/k, p) pairs where g/k decoded first
      chain_len_vs_rank:  {chain_len: mean_decode_rank} — key analysis
    """
    N, S = decode_orders.shape

    # Build rop [N, ANS_LEN]: for each sample & position, what decode step was it?
    # decode_orders[i, s] = absolute position decoded at step s
    # We want: rop[i, j] = step at which relative position j was decoded
    rel_orders = decode_orders - prefix_len  # [N, S]
    rop = torch.full((N, ANS_LEN), float('nan'))
    # Vectorized: for each step, assign step number to the corresponding position
    valid = (rel_orders >= 0) & (rel_orders < ANS_LEN)
    for s in range(S):
        v = valid[:, s]
        if v.any():
            positions = rel_orders[v, s].long()
            rop[v.nonzero(as_tuple=True)[0], positions] = s

    # Mean rank per position
    mr = []
    for j in range(ANS_LEN):
        v = rop[:, j][~rop[:, j].isnan()]
        mr.append(v.mean().item() if len(v) > 0 else -1)

    # Pairwise concordance (vectorized)
    # Valid rows = no NaN in any position
    valid_mask = ~rop.isnan().any(dim=1)  # [N]
    rop_valid = rop[valid_mask]  # [N_valid, ANS_LEN]
    n_valid = rop_valid.shape[0]
    n_pairs = ANS_LEN * (ANS_LEN - 1) // 2

    if n_valid > 0:
        # For all pairs (i, j) with i < j, count how many times
        # the LSB-side position is decoded first
        # Generate all pair indices
        ii, jj = torch.triu_indices(ANS_LEN, ANS_LEN, offset=1)
        ri = rop_valid[:, ii]  # [N_valid, n_pairs]
        rj = rop_valid[:, jj]  # [N_valid, n_pairs]
        if fmt == 'plain':
            # j is more LSB → concordant if rj < ri (j decoded first)
            conc_per_pair = (rj < ri).float()
        else:
            # i is more LSB → concordant if ri < rj (i decoded first)
            conc_per_pair = (ri < rj).float()
        pw_conc = conc_per_pair.mean().item()
    else:
        pw_conc = -1

    # Rank histogram
    rh = torch.zeros(ANS_LEN, S)
    for j in range(ANS_LEN):
        vals = rop[:, j][~rop[:, j].isnan()].long()
        if len(vals) > 0:
            rh[j].scatter_add_(0, vals.clamp(max=S-1), torch.ones_like(vals, dtype=torch.float))

    result = {'mean_rank': mr, 'pairwise_concordance': pw_conc,
              'rank_histogram': rh}

    # G/K/P analysis (vectorized)
    if gkp_per_sample is not None and len(gkp_per_sample) >= N:
        gkp_names = ['g', 'k', 'p', 'carry_out']
        gkp_to_id = {name: i for i, name in enumerate(gkp_names)}
        gkp_ids = torch.tensor([[gkp_to_id.get(g, 0) for g in gkp[:ANS_LEN]]
                                 for gkp in gkp_per_sample[:N]])  # [N, ANS_LEN]

        rop_v = rop_valid
        gkp_v = gkp_ids[valid_mask]  # [N_valid, ANS_LEN]
        rop_flat = rop_v.reshape(-1)
        gkp_flat = gkp_v.reshape(-1)

        gkp_mean_rank = {}
        for gi, gname in enumerate(gkp_names):
            mask = (gkp_flat == gi)
            if mask.any():
                gkp_mean_rank[gname] = rop_flat[mask].mean().item()
            else:
                gkp_mean_rank[gname] = None
        result['gkp_mean_rank'] = gkp_mean_rank

        # Easy before hard (vectorized over pairs within each sample)
        if n_valid > 0:
            is_easy = (gkp_v <= 1)  # g=0, k=1 → easy
            is_p = (gkp_v == 2)     # p=2 → hard
            # For each sample, count pairs where easy position decoded before p position
            # Use broadcasting: [N_valid, ANS_LEN, 1] vs [N_valid, 1, ANS_LEN]
            easy_rank = rop_v.unsqueeze(2) * is_easy.float().unsqueeze(2)  # rank of easy pos
            p_rank = rop_v.unsqueeze(1) * is_p.float().unsqueeze(1)        # rank of p pos
            pair_valid = is_easy.unsqueeze(2) & is_p.unsqueeze(1)  # [N_valid, ANS_LEN, ANS_LEN]
            if pair_valid.any():
                easy_first = (rop_v.unsqueeze(2) < rop_v.unsqueeze(1)) & pair_valid
                result['easy_before_hard'] = easy_first.sum().item() / pair_valid.sum().item()
            else:
                result['easy_before_hard'] = None
        else:
            result['easy_before_hard'] = None

    # Carry chain length vs decode rank (vectorized)
    if chain_len_per_sample is not None and len(chain_len_per_sample) >= N:
        ccl = torch.tensor([c[:ANS_LEN] for c in chain_len_per_sample[:N]])  # [N, ANS_LEN]
        ccl_v = ccl[valid_mask].reshape(-1)
        rank_v = rop_valid.reshape(-1)
        cl_vs_rank = {}
        for cl_val in ccl_v.unique().tolist():
            mask = (ccl_v == cl_val)
            cl_vs_rank[cl_val] = rank_v[mask].mean().item()
        result['chain_len_vs_rank'] = cl_vs_rank

    # Carry-source-type → decode rank
    # For each position j, what is the g/k/p of its carry SOURCE (below)?
    # This tests: "does model decode j earlier when its source is g/k (carry resolved)
    # vs p (carry unresolved)?"
    # Key difference from gkp_mean_rank: that groups by j's OWN type,
    # this groups by j's DEPENDENCY (what determines j's difficulty).
    if gkp_per_sample is not None and len(gkp_per_sample) >= N and n_valid > 0:
        # Carry source: plain → j+1 (LSB is at high index),
        #               reverse → j-1 (LSB is at low index)
        source_type = torch.full((n_valid, ANS_LEN), -1, dtype=torch.long)  # -1 = no source
        for i in range(n_valid):
            orig_i = valid_mask.nonzero(as_tuple=True)[0][i].item()
            gkp = gkp_per_sample[orig_i]
            for j in range(ANS_LEN):
                if fmt == 'plain':
                    src_j = j + 1  # carry comes from more-LSB = higher index
                else:
                    src_j = j - 1  # carry comes from more-LSB = lower index
                if 0 <= src_j < ANS_LEN and gkp[src_j] in ('g', 'k', 'p'):
                    source_type[i, j] = gkp_to_id[gkp[src_j]]

        src_flat = source_type.reshape(-1)
        rank_flat_all = rop_valid.reshape(-1)
        source_rank = {}
        for gi, gname in enumerate(gkp_names):
            mask = (src_flat == gi)
            if mask.any():
                source_rank[f'source_{gname}'] = rank_flat_all[mask].mean().item()
        # Also: positions with no carry source (LSB or carry_out)
        no_src = (src_flat == -1)
        if no_src.any():
            source_rank['source_none'] = rank_flat_all[no_src].mean().item()
        result['carry_source_rank'] = source_rank

    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Visualization
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _pos_labels(fmt):
    if fmt == 'plain':
        return ['MSB'] + [f'p{j}' for j in range(1, ANS_LEN-1)] + ['LSB']
    return ['LSB'] + [f'p{j}' for j in range(1, ANS_LEN-1)] + ['MSB']

COLORS = {
    'ar': '#e74c3c',
    'diff-ran-con': '#3498db', 'diff-ran-lsb': '#2ecc71',
    'diff-ora-con': '#9b59b6', 'diff-ora-lsb': '#e67e22',
    'diff-con-con': '#1abc9c', 'diff-con-lsb': '#16a085',
    'diff-pum-con': '#8e44ad', 'diff-pum-lsb': '#6c3483',
}

def _ck(obj, mt='', dp=''):
    if obj == 'ar':
        return 'ar'
    return f"diff-{mt[:3]}-{dp[:3]}"

def _fk(obj, fmt, mt='', dp=''):
    if obj == 'ar':
        return f"ar_{fmt}"
    return f"{obj}_{fmt}_{mt}_{dp}"


def make_figures(all_dyn, all_final):
    figs = {}
    cmap = plt.cm.coolwarm

    for fmt in FORMATS:
        labels = _pos_labels(fmt)
        def pc(j):
            return cmap(1.0 - j/(ANS_LEN-1)) if fmt=='plain' else cmap(j/(ANS_LEN-1))

        conds = [('ar', '')] + [('diffusion', mt) for mt in MASK_TYPES]
        nc = len(conds)

        # Helper: get x from checkpoints
        def get_x(dyn, axis='epoch'):
            return [c[axis] for c in dyn['checkpoints']]

        # ── Fig 1: Per-position LOSS (x=epoch) ──
        fig, axes = plt.subplots(1, nc, figsize=(6*nc, 5), squeeze=False)
        axes = axes[0]
        for ai, (obj, mt) in enumerate(conds):
            key = _fk(obj, fmt, mt, 'confidence')
            dyn = all_dyn.get(key)
            if not dyn: continue
            ax = axes[ai]
            xs = get_x(dyn)
            for j in range(ANS_LEN):
                ax.plot(xs, [c['pos_loss'][j] for c in dyn['checkpoints']],
                        '-', color=pc(j), label=labels[j], lw=1.5)
            ax.set_xlabel('Epoch'); ax.set_ylabel('CE Loss')
            ax.set_title(_ck(obj, mt, 'confidence'))
            ax.legend(fontsize=5, ncol=3, loc='upper right'); ax.grid(alpha=0.3)
        fig.suptitle(f'Per-Position Loss — {fmt}', fontsize=13, y=1.02)
        fig.tight_layout(); figs[f'pos_loss_{fmt}'] = fig

        # ── Fig 2: Per-position ACC (x=epoch) ──
        fig, axes = plt.subplots(1, nc, figsize=(6*nc, 5), squeeze=False)
        axes = axes[0]
        for ai, (obj, mt) in enumerate(conds):
            key = _fk(obj, fmt, mt, 'confidence')
            dyn = all_dyn.get(key)
            if not dyn: continue
            ax = axes[ai]
            xs = get_x(dyn)
            for j in range(ANS_LEN):
                ax.plot(xs, [c['pos_acc'][j] for c in dyn['checkpoints']],
                        '-', color=pc(j), label=labels[j], lw=1.5)
            ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy')
            ax.set_ylim(-0.05, 1.05); ax.set_title(_ck(obj, mt, 'confidence'))
            ax.legend(fontsize=5, ncol=3, loc='lower right'); ax.grid(alpha=0.3)
        fig.suptitle(f'Per-Position Accuracy — {fmt}', fontsize=13, y=1.02)
        fig.tight_layout(); figs[f'pos_acc_{fmt}'] = fig

        # ── Fig 2b: Per-position ACC (x=token_gradients) ──
        fig, axes = plt.subplots(1, nc, figsize=(6*nc, 5), squeeze=False)
        axes = axes[0]
        for ai, (obj, mt) in enumerate(conds):
            key = _fk(obj, fmt, mt, 'confidence')
            dyn = all_dyn.get(key)
            if not dyn: continue
            ax = axes[ai]
            xs = get_x(dyn, 'token_gradients')
            for j in range(ANS_LEN):
                ax.plot(xs, [c['pos_acc'][j] for c in dyn['checkpoints']],
                        '-', color=pc(j), label=labels[j], lw=1.5)
            ax.set_xlabel('Token-Gradients'); ax.set_ylabel('Accuracy')
            ax.set_ylim(-0.05, 1.05); ax.set_title(_ck(obj, mt, 'confidence'))
            ax.legend(fontsize=5, ncol=3, loc='lower right'); ax.grid(alpha=0.3)
            ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
        fig.suptitle(f'Per-Position Accuracy — {fmt}  (x = token-gradient count)',
                     fontsize=12, y=1.02)
        fig.tight_layout(); figs[f'pos_acc_tg_{fmt}'] = fig

        # ── Fig 3: Carry gap ──
        fig, axes = plt.subplots(1, nc, figsize=(6*nc, 5), squeeze=False)
        axes = axes[0]
        for ai, (obj, mt) in enumerate(conds):
            key = _fk(obj, fmt, mt, 'confidence')
            dyn = all_dyn.get(key)
            if not dyn: continue
            ax = axes[ai]
            xs = get_x(dyn)
            for j in range(ANS_LEN):
                def _carry_gap(c, j):
                    ci = c['pos_loss_carry_in'][j]
                    nc = c['pos_loss_no_carry'][j]
                    if ci is None or nc is None:
                        return float('nan')
                    return ci - nc
                ax.plot(xs, [_carry_gap(c, j) for c in dyn['checkpoints']],
                        '-', color=pc(j), label=labels[j], lw=1.5)
            ax.axhline(0, color='k', lw=0.5, ls='--')
            ax.set_xlabel('Epoch'); ax.set_ylabel('Loss(carry) − Loss(no carry)')
            ax.set_title(_ck(obj, mt, 'confidence'))
            ax.legend(fontsize=5, ncol=3); ax.grid(alpha=0.3)
        fig.suptitle(f'Carry Effect — {fmt}  (+ = carry harder)',
                     fontsize=12, y=1.02)
        fig.tight_layout(); figs[f'carry_gap_{fmt}'] = fig

        # ── Fig 4: Confidence evolution ──
        dconds = [(mt, _fk('diffusion', fmt, mt, 'confidence'))
                  for mt in MASK_TYPES
                  if _fk('diffusion', fmt, mt, 'confidence') in all_dyn]
        if dconds:
            fig, axes = plt.subplots(1, len(dconds), figsize=(6*len(dconds), 5), squeeze=False)
            axes = axes[0]
            for ai, (mt, key) in enumerate(dconds):
                dyn = all_dyn[key]; ax = axes[ai]
                xs = get_x(dyn)
                for j in range(ANS_LEN):
                    ax.plot(xs, [c['pos_conf'][j] for c in dyn['checkpoints']],
                            '-', color=pc(j), label=labels[j], lw=1.5)
                ax.set_xlabel('Epoch'); ax.set_ylabel('Confidence (fully masked)')
                ax.set_ylim(0, 1.05); ax.set_title(f'mask={mt}')
                ax.legend(fontsize=5, ncol=3, loc='lower right'); ax.grid(alpha=0.3)
            fig.suptitle(f'Confidence Evolution — {fmt}', fontsize=13, y=1.02)
            fig.tight_layout(); figs[f'conf_evo_{fmt}'] = fig

        # ── Fig 4b: Probe confidence concordance + spread evolution ──
        if dconds:
            fig, axes = plt.subplots(1, 2, figsize=(16, 5))

            # Left: Probe conf concordance (dense) vs actual decode concordance (sparse)
            ax = axes[0]
            for mt, key in dconds:
                dyn = all_dyn[key]
                ck = _ck('diffusion', mt)
                col = COLORS.get(ck+'con', COLORS.get(ck, '#333'))
                # Dense: from probe checkpoints (every EVAL_EVERY)
                xs_p = [c['epoch'] for c in dyn['checkpoints']
                        if 'conf_concordance' in c]
                ys_p = [c['conf_concordance'] for c in dyn['checkpoints']
                        if 'conf_concordance' in c]
                if xs_p:
                    ax.plot(xs_p, ys_p, '-', color=col, label=f'{mt} (probe)',
                            lw=1.5, alpha=0.7)
                # Sparse: from gen_checkpoints (every GEN_EVAL_EVERY)
                gc = dyn.get('gen_checkpoints', [])
                xs_g = [g['epoch'] for g in gc
                        if 'confidence' in g.get('concordance', {})]
                ys_g = [g['concordance']['confidence'] for g in gc
                        if 'confidence' in g.get('concordance', {})]
                if xs_g:
                    ax.plot(xs_g, ys_g, 'o', color=col, ms=6, alpha=0.9,
                            label=f'{mt} (gen)')
            ax.axhline(0.5, color='grey', ls=':', lw=1, alpha=0.5)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Pairwise Concordance (LSB-first)')
            ax.set_ylim(-0.05, 1.05)
            ax.set_title('Probe vs Actual Decode Concordance')
            ax.legend(fontsize=5, ncol=2); ax.grid(alpha=0.3)

            # Right: Confidence spread (max - min) over epochs
            ax = axes[1]
            for mt, key in dconds:
                dyn = all_dyn[key]
                ck = _ck('diffusion', mt)
                col = COLORS.get(ck+'con', COLORS.get(ck, '#333'))
                xs = [c['epoch'] for c in dyn['checkpoints']
                      if 'conf_spread' in c]
                ys = [c['conf_spread'] for c in dyn['checkpoints']
                      if 'conf_spread' in c]
                if xs:
                    ax.plot(xs, ys, '-', color=col, label=f'mask={mt}',
                            lw=1.5, alpha=0.8)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Confidence Spread (max − min)')
            ax.set_title('Confidence Ordering Strength')
            ax.legend(fontsize=6); ax.grid(alpha=0.3)

            fig.suptitle(f'Confidence Ordering Dynamics — {fmt}', fontsize=13, y=1.02)
            fig.tight_layout(); figs[f'conf_ordering_{fmt}'] = fig

        # ── Fig 4c: Dependency context confidence & accuracy evolution ──
        # Tracks whether carry chain RECOGNITION forms before/after ACCURACY
        if dconds:
            ctx_order = ['g', 'k', 'p_above_g', 'p_above_k', 'p_above_p', 'p_bottom']
            ctx_colors = {'g': '#2ecc71', 'k': '#3498db', 'p_above_g': '#27ae60',
                          'p_above_k': '#2980b9', 'p_above_p': '#e74c3c',
                          'p_bottom': '#f39c12', 'carry_out': '#95a5a6'}
            n_dc = len(dconds)
            fig, axes = plt.subplots(2, n_dc, figsize=(7*n_dc, 10), squeeze=False)
            for ai, (mt, key) in enumerate(dconds):
                dyn = all_dyn[key]
                cps = dyn['checkpoints']
                xs = [c['epoch'] for c in cps if 'dep_context' in c]
                if not xs: continue
                # Top row: confidence by context
                ax = axes[0][ai]
                for ctx in ctx_order:
                    ys = [c['dep_context'][ctx]['conf']
                          for c in cps if 'dep_context' in c and ctx in c['dep_context']]
                    if ys and len(ys) == len(xs):
                        ax.plot(xs, ys, '-', color=ctx_colors.get(ctx, '#333'),
                                label=ctx, lw=1.5, alpha=0.8)
                ax.set_xlabel('Epoch'); ax.set_ylabel('Mean Confidence')
                ax.set_ylim(0, 1.05); ax.set_title(f'mask={mt} — Confidence')
                ax.legend(fontsize=5, ncol=2); ax.grid(alpha=0.3)
                # Bottom row: accuracy by context
                ax = axes[1][ai]
                for ctx in ctx_order:
                    ys = [c['dep_context'][ctx]['acc']
                          for c in cps if 'dep_context' in c and ctx in c['dep_context']]
                    if ys and len(ys) == len(xs):
                        ax.plot(xs, ys, '-', color=ctx_colors.get(ctx, '#333'),
                                label=ctx, lw=1.5, alpha=0.8)
                ax.set_xlabel('Epoch'); ax.set_ylabel('Mean Accuracy')
                ax.set_ylim(-0.05, 1.05); ax.set_title(f'mask={mt} — Accuracy')
                ax.legend(fontsize=5, ncol=2); ax.grid(alpha=0.3)
            fig.suptitle(f'Carry Chain Recognition vs Accuracy Over Training — {fmt}',
                         fontsize=13, y=1.02)
            fig.tight_layout(); figs[f'dep_ctx_evo_{fmt}'] = fig

        # ── Fig 4d: Recognition gap evolution ──
        # conf(g/k) − conf(p_above_p): when does the model learn to distinguish?
        if dconds:
            fig, axes = plt.subplots(1, 2, figsize=(16, 5))
            # Left: confidence gap
            ax = axes[0]
            for mt, key in dconds:
                dyn = all_dyn[key]
                cps = dyn['checkpoints']
                xs, ys = [], []
                for c in cps:
                    dc = c.get('dep_context')
                    if dc and 'g' in dc and 'p_above_p' in dc:
                        gk_conf = (dc['g']['conf'] * dc['g']['n']
                                   + dc['k']['conf'] * dc['k']['n']) / (dc['g']['n'] + dc['k']['n'])
                        gap = gk_conf - dc['p_above_p']['conf']
                        xs.append(c['epoch']); ys.append(gap)
                if xs:
                    ck = _ck('diffusion', mt)
                    ax.plot(xs, ys, '-', color=COLORS.get(ck+'con', COLORS.get(ck, '#333')),
                            label=f'mask={mt}', lw=1.5, alpha=0.8)
            ax.axhline(0, color='grey', ls=':', lw=1, alpha=0.5)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('conf(g/k) − conf(p_above_p)')
            ax.set_title('Confidence Recognition Gap')
            ax.legend(fontsize=6); ax.grid(alpha=0.3)
            # Right: accuracy gap
            ax = axes[1]
            for mt, key in dconds:
                dyn = all_dyn[key]
                cps = dyn['checkpoints']
                xs, ys = [], []
                for c in cps:
                    dc = c.get('dep_context')
                    if dc and 'g' in dc and 'p_above_p' in dc:
                        gk_acc = (dc['g']['acc'] * dc['g']['n']
                                  + dc['k']['acc'] * dc['k']['n']) / (dc['g']['n'] + dc['k']['n'])
                        gap = gk_acc - dc['p_above_p']['acc']
                        xs.append(c['epoch']); ys.append(gap)
                if xs:
                    ck = _ck('diffusion', mt)
                    ax.plot(xs, ys, '-', color=COLORS.get(ck+'con', COLORS.get(ck, '#333')),
                            label=f'mask={mt}', lw=1.5, alpha=0.8)
            ax.axhline(0, color='grey', ls=':', lw=1, alpha=0.5)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('acc(g/k) − acc(p_above_p)')
            ax.set_title('Accuracy Gap')
            ax.legend(fontsize=6); ax.grid(alpha=0.3)
            fig.suptitle(f'Recognition vs Computation Gap — {fmt}\n'
                         '(conf gap rises → model knows what\'s hard; '
                         'acc gap rises → model can\'t yet solve hard cases)',
                         fontsize=11, y=1.05)
            fig.tight_layout(); figs[f'recognition_gap_{fmt}'] = fig

        # ── Fig 5: Final per-position accuracy ──
        fig, axes = plt.subplots(1, 3, figsize=(21, 5))
        diff_conf = [('diffusion', mt, 'confidence') for mt in MASK_TYPES]
        diff_lsb = [('diffusion', mt, 'lsb') for mt in MASK_TYPES]
        for ai, (title, cs) in enumerate([
            ('AR vs Diffusion (conf decode)', [('ar','','')] + diff_conf),
            ('Training mask comparison (conf decode)', diff_conf),
            ('LSB decode comparison', diff_lsb),
        ]):
            ax = axes[ai]
            for obj, mt, dp in cs:
                key = _fk(obj, fmt, mt, dp)
                r = all_final.get(key)
                if r and r.get('position_accuracy'):
                    ck = _ck(obj, mt, dp)
                    ax.plot(range(ANS_LEN), r['position_accuracy'], '-o',
                            color=COLORS.get(ck,'#333'),
                            label=f"{ck} ({r['accuracy']:.3f})", alpha=0.8, ms=4)
            ax.set_xticks(range(ANS_LEN)); ax.set_xticklabels(labels, fontsize=7)
            ax.set_ylim(-0.05, 1.05); ax.set_ylabel('Gen. Accuracy')
            ax.set_title(title); ax.legend(fontsize=6); ax.grid(alpha=0.3)
        fig.suptitle(f'Final Position Accuracy — {fmt}', fontsize=13, y=1.02)
        fig.tight_layout(); figs[f'final_acc_{fmt}'] = fig

        # ── Fig 6: Decode order heatmaps ──
        od = [(mt, all_final.get(_fk('diffusion',fmt,mt,'confidence'),{}).get('decode_order_analysis'))
              for mt in MASK_TYPES]
        od = [(mt, oa) for mt, oa in od if oa]
        if od:
            fig, axes = plt.subplots(1, len(od), figsize=(7*len(od), 5), squeeze=False)
            axes = axes[0]
            for ax, (mt, oa) in zip(axes, od):
                rh = oa['rank_histogram'].float()
                rn = rh / rh.sum(1, keepdim=True).clamp(1)
                im = ax.imshow(rn.numpy(), aspect='auto', cmap='YlOrRd')
                ax.set_xlabel('Decode Step'); ax.set_ylabel('Position')
                ax.set_yticks(range(ANS_LEN)); ax.set_yticklabels(labels, fontsize=8)
                plt.colorbar(im, ax=ax, label='Fraction')
                ax.set_title(f'mask={mt} | LSB-conc={oa["pairwise_concordance"]:.2f}')
            fig.suptitle(f'Decode Order — {fmt}', fontsize=13, y=1.02)
            fig.tight_layout(); figs[f'decode_ord_{fmt}'] = fig

        # ── Fig 8: Training loss ──
        fig, ax = plt.subplots(figsize=(10, 6))
        for obj, mt in conds:
            key = _fk(obj, fmt, mt, 'confidence')
            dyn = all_dyn.get(key)
            if not dyn or not dyn['train_loss']: continue
            ck = _ck(obj, mt, 'confidence')
            ax.plot([x[0] for x in dyn['train_loss']],
                    [x[1] for x in dyn['train_loss']],
                    '-', color=COLORS.get(ck,'#333'), label=ck, alpha=0.7)
        ax.set_xlabel('Epoch'); ax.set_ylabel('Training Loss (each model\'s own)')
        ax.set_title(f'Training Loss — {fmt}')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        fig.tight_layout(); figs[f'train_loss_{fmt}'] = fig

        # ── Fig 9: Generation accuracy over training ──
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        for ai, dp in enumerate(DECODE_POLICIES):
            ax = axes[ai]
            for obj, mt in conds:
                key = _fk(obj, fmt, mt, 'confidence')
                dyn = all_dyn.get(key)
                if not dyn or not dyn.get('gen_checkpoints'): continue
                gc = dyn['gen_checkpoints']
                dp_key = 'ar' if obj == 'ar' else dp
                xs = [g['epoch'] for g in gc if dp_key in g.get('gen_acc', {})]
                ys = [g['gen_acc'][dp_key] for g in gc if dp_key in g.get('gen_acc', {})]
                if xs:
                    ck = _ck(obj, mt, dp)
                    ax.plot(xs, ys, '-o', color=COLORS.get(ck, '#333'),
                            label=ck, alpha=0.8, ms=3)
            ax.set_xlabel('Epoch'); ax.set_ylabel('Gen. Accuracy')
            ax.set_ylim(-0.05, 1.05)
            ax.set_title(f'decode={dp}')
            ax.legend(fontsize=6); ax.grid(alpha=0.3)
        fig.suptitle(f'Generation Accuracy During Training — {fmt}',
                     fontsize=13, y=1.02)
        fig.tight_layout(); figs[f'gen_acc_curve_{fmt}'] = fig

        # ── Fig 10: Time-to-threshold per position ──
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        for ai, (title, metric) in enumerate([
            (f'Probe acc ≥ {THRESHOLD}', 'pos_acc'),
        ]):
            ax = axes[ai]
            bar_data = {}
            for obj, mt in conds:
                key = _fk(obj, fmt, mt, 'confidence')
                dyn = all_dyn.get(key)
                if not dyn: continue
                ck = _ck(obj, mt)
                t2t = []
                for j in range(ANS_LEN):
                    epoch_hit = None
                    for c in dyn['checkpoints']:
                        if c[metric][j] >= THRESHOLD:
                            epoch_hit = c['epoch']; break
                    t2t.append(epoch_hit)
                bar_data[ck] = t2t
            if bar_data:
                x = range(ANS_LEN)
                w = 0.8 / max(len(bar_data), 1)
                for ki, (ck, t2t) in enumerate(bar_data.items()):
                    vals = [v if v is not None else MAX_EPOCHS*1.1 for v in t2t]
                    ax.bar([xi + ki*w for xi in x], vals, w,
                           color=COLORS.get(ck+'con', COLORS.get(ck, '#333')),
                           label=ck, alpha=0.8)
                ax.set_xticks([xi + w*len(bar_data)/2 for xi in x])
                ax.set_xticklabels(labels, fontsize=7)
                ax.set_ylabel('Epoch'); ax.set_title(title)
                ax.legend(fontsize=5, ncol=2); ax.grid(alpha=0.3, axis='y')
        # Second panel: gen accuracy time-to-threshold
        ax = axes[1]
        bar_data = {}
        for obj, mt in conds:
            key = _fk(obj, fmt, mt, 'confidence')
            dyn = all_dyn.get(key)
            if not dyn or not dyn.get('gen_checkpoints'): continue
            ck = _ck(obj, mt)
            dp_key = 'ar' if obj == 'ar' else 'confidence'
            epoch_hit = None
            for g in dyn['gen_checkpoints']:
                if g.get('gen_acc', {}).get(dp_key, 0) >= THRESHOLD:
                    epoch_hit = g['epoch']; break
            bar_data[ck] = epoch_hit
        if bar_data:
            cks = list(bar_data.keys())
            vals = [bar_data[ck] if bar_data[ck] is not None else MAX_EPOCHS*1.1 for ck in cks]
            colors = [COLORS.get(ck+'con', COLORS.get(ck, '#333')) for ck in cks]
            ax.bar(range(len(cks)), vals, color=colors, alpha=0.8)
            ax.set_xticks(range(len(cks))); ax.set_xticklabels(cks, fontsize=7, rotation=30)
            ax.set_ylabel('Epoch'); ax.set_title(f'Gen acc ≥ {THRESHOLD} (conf decode)')
            ax.grid(alpha=0.3, axis='y')
        fig.suptitle(f'Time to {THRESHOLD} — {fmt}', fontsize=13, y=1.02)
        fig.tight_layout(); figs[f'time_to_thresh_{fmt}'] = fig

        # ── Fig 11: Carry-out analysis (generation accuracy) ──
        # MSB position = position 0 (plain) or ANS_LEN-1 (reverse)
        msb_j = 0 if fmt == 'plain' else ANS_LEN - 1
        all_conds_fig = [('ar','','')]
        for mt in MASK_TYPES:
            for dp in DECODE_POLICIES:
                all_conds_fig.append(('diffusion', mt, dp))
        carry_data = []
        for obj, mt, dp in all_conds_fig:
            key = _fk(obj, fmt, mt, dp)
            r = all_final.get(key)
            if not r: continue
            ck = _ck(obj, mt, dp)
            ci = r.get('pos_acc_carry_in', [None]*ANS_LEN)[msb_j]
            nc = r.get('pos_acc_no_carry', [None]*ANS_LEN)[msb_j]
            carry_data.append((ck, ci, nc))
        if carry_data:
            fig, ax = plt.subplots(figsize=(12, 5))
            x = range(len(carry_data))
            w = 0.35
            ci_vals = [d[1] if d[1] is not None else 0 for d in carry_data]
            nc_vals = [d[2] if d[2] is not None else 0 for d in carry_data]
            ax.bar([xi-w/2 for xi in x], ci_vals, w, label='carry-out=1', color='#e74c3c', alpha=0.8)
            ax.bar([xi+w/2 for xi in x], nc_vals, w, label='carry-out=0', color='#3498db', alpha=0.8)
            ax.set_xticks(list(x)); ax.set_xticklabels([d[0] for d in carry_data],
                                                         fontsize=7, rotation=30)
            ax.set_ylim(0, 1.05); ax.set_ylabel('MSB Position Accuracy')
            ax.set_title(f'Carry-Out Effect on MSB — {fmt}')
            ax.legend(); ax.grid(alpha=0.3, axis='y')
            fig.tight_layout(); figs[f'carry_out_{fmt}'] = fig

        # ── Fig 12: Per-position gen accuracy evolution ──
        for obj, mt in conds:
            key = _fk(obj, fmt, mt, 'confidence')
            dyn = all_dyn.get(key)
            if not dyn or not dyn.get('gen_checkpoints'): continue
            gc = dyn['gen_checkpoints']
            dp_key = 'ar' if obj == 'ar' else 'confidence'
            gcs = [g for g in gc if dp_key in g.get('gen_pos_acc', {})]
            if not gcs: continue
            fig, ax = plt.subplots(figsize=(10, 5))
            xs = [g['epoch'] for g in gcs]
            for j in range(ANS_LEN):
                ys = [g['gen_pos_acc'][dp_key][j] for g in gcs]
                ax.plot(xs, ys, '-o', color=pc(j), label=labels[j], ms=3, lw=1.5)
            ax.set_xlabel('Epoch'); ax.set_ylabel('Gen. Position Accuracy')
            ax.set_ylim(-0.05, 1.05)
            ck = _ck(obj, mt)
            ax.set_title(f'{ck} — Per-Position Gen Accuracy ({fmt})')
            ax.legend(fontsize=5, ncol=3); ax.grid(alpha=0.3)
            fig.tight_layout(); figs[f'gen_pos_evo_{fmt}_{ck}'] = fig

        # ── Fig 13: Decode concordance evolution ──
        if dconds:
            fig, ax = plt.subplots(figsize=(10, 5))
            for mt, key in dconds:
                dyn = all_dyn[key]
                gc = dyn.get('gen_checkpoints', [])
                dp = 'confidence'
                xs = [g['epoch'] for g in gc if dp in g.get('concordance', {})]
                ys = [g['concordance'][dp] for g in gc if dp in g.get('concordance', {})]
                if xs:
                    ck = _ck('diffusion', mt)
                    ax.plot(xs, ys, '-o', color=COLORS.get(ck+'con', COLORS.get(ck, '#333')),
                            label=f'mask={mt}', ms=3, lw=2, alpha=0.8)
            ax.axhline(0.5, color='grey', ls=':', lw=1, alpha=0.5, label='random')
            ax.set_xlabel('Epoch'); ax.set_ylabel('Pairwise Concordance (LSB-first)')
            ax.set_ylim(-0.05, 1.05); ax.set_title(f'Decode Order Evolution — {fmt}')
            ax.legend(fontsize=7); ax.grid(alpha=0.3)
            fig.tight_layout(); figs[f'concordance_evo_{fmt}'] = fig

        # ── Fig 14: Carry-out MSB gen accuracy evolution ──
        msb_j = 0 if fmt == 'plain' else ANS_LEN - 1
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        for ai, label_dp in enumerate([('carry-out=1', 'carry'), ('carry-out=0', 'no_carry')]):
            title, carry_key = label_dp
            ax = axes[ai]
            for obj, mt in conds:
                key = _fk(obj, fmt, mt, 'confidence')
                dyn = all_dyn.get(key)
                if not dyn or not dyn.get('gen_checkpoints'): continue
                dp_key = 'ar' if obj == 'ar' else 'confidence'
                gc = dyn['gen_checkpoints']
                xs, ys = [], []
                for g in gc:
                    co = g.get('gen_carry_out', {}).get(dp_key, {})
                    val = co.get(carry_key, [None]*ANS_LEN)[msb_j]
                    if val is not None:
                        xs.append(g['epoch']); ys.append(val)
                if xs:
                    ck = _ck(obj, mt)
                    ax.plot(xs, ys, '-o', color=COLORS.get(ck+'con', COLORS.get(ck, '#333')),
                            label=ck, ms=3, lw=1.5, alpha=0.8)
            ax.set_xlabel('Epoch'); ax.set_ylabel('MSB Gen Accuracy')
            ax.set_ylim(-0.05, 1.05); ax.set_title(f'{title} — {fmt}')
            ax.legend(fontsize=6); ax.grid(alpha=0.3)
        fig.suptitle(f'Carry-Out Effect on MSB During Training — {fmt}', fontsize=12, y=1.02)
        fig.tight_layout(); figs[f'carry_out_evo_{fmt}'] = fig

        # ── Fig 15b: Easy-before-hard evolution ──
        if dconds:
            fig, ax = plt.subplots(figsize=(10, 5))
            for mt, key in dconds:
                dyn = all_dyn[key]
                gc = dyn.get('gen_checkpoints', [])
                dp = 'confidence'
                xs = [g['epoch'] for g in gc if dp in g.get('easy_before_hard', {})]
                ys = [g['easy_before_hard'][dp] for g in gc
                      if dp in g.get('easy_before_hard', {})]
                if xs:
                    ck = _ck('diffusion', mt)
                    ax.plot(xs, ys, '-o', color=COLORS.get(ck+'con', COLORS.get(ck, '#333')),
                            label=f'mask={mt}', ms=3, lw=1.5, alpha=0.8)
            ax.axhline(0.5, color='grey', ls=':', lw=1, alpha=0.5, label='random')
            ax.set_xlabel('Epoch'); ax.set_ylabel('P(easy before hard)')
            ax.set_ylim(-0.05, 1.05)
            ax.set_title(f'Easy (g/k) Before Hard (p) Evolution — {fmt}')
            ax.legend(fontsize=6); ax.grid(alpha=0.3)
            fig.tight_layout(); figs[f'easy_hard_evo_{fmt}'] = fig

        # ── Fig 15c: Carry source type → decode rank evolution ──
        # Key test: does the model decode positions earlier when their
        # carry source is g/k (resolved) vs p (unresolved)?
        if dconds:
            fig, axes = plt.subplots(1, len(dconds), figsize=(7*len(dconds), 5), squeeze=False)
            axes = axes[0]
            src_colors = {'source_g': '#2ecc71', 'source_k': '#3498db', 'source_p': '#e74c3c'}
            for ai, (mt, key) in enumerate(dconds):
                ax = axes[ai]
                dyn = all_dyn[key]
                gc = dyn.get('gen_checkpoints', [])
                dp = 'confidence'
                for src_type in ['source_g', 'source_k', 'source_p']:
                    xs, ys = [], []
                    for g in gc:
                        csr = g.get('carry_source_rank', {}).get(dp, {})
                        if src_type in csr:
                            xs.append(g['epoch']); ys.append(csr[src_type])
                    if xs:
                        ax.plot(xs, ys, '-o', color=src_colors.get(src_type, '#333'),
                                label=src_type, ms=3, lw=1.5, alpha=0.8)
                ax.set_xlabel('Epoch'); ax.set_ylabel('Mean Decode Rank')
                ax.set_title(f'mask={mt}')
                ax.legend(fontsize=7); ax.grid(alpha=0.3)
            fig.suptitle(
                f'Decode Rank by Carry Source Type — {fmt}\n'
                'source_g/k = carry resolved (should be decoded earlier) '
                'vs source_p = unresolved',
                fontsize=11, y=1.05)
            fig.tight_layout(); figs[f'source_rank_evo_{fmt}'] = fig

        # ── Fig 16: Chain length → decode rank evolution ──
        if dconds:
            fig, axes = plt.subplots(1, len(dconds), figsize=(7*len(dconds), 5), squeeze=False)
            axes = axes[0]
            chain_cmap = plt.cm.viridis
            for ai, (mt, key) in enumerate(dconds):
                ax = axes[ai]
                dyn = all_dyn[key]
                gc = dyn.get('gen_checkpoints', [])
                dp = 'confidence'
                # Collect all chain lengths seen across epochs
                all_cls = set()
                for g in gc:
                    clvr = g.get('chain_len_vs_rank', {}).get(dp, {})
                    all_cls.update(clvr.keys())
                all_cls = sorted(all_cls)
                if all_cls:
                    max_cl = max(max(all_cls), 1)
                    for cl in all_cls:
                        xs, ys = [], []
                        for g in gc:
                            clvr = g.get('chain_len_vs_rank', {}).get(dp, {})
                            if cl in clvr:
                                xs.append(g['epoch']); ys.append(clvr[cl])
                        if xs:
                            ax.plot(xs, ys, '-o', color=chain_cmap(cl / max_cl),
                                    label=f'chain={cl}', ms=3, lw=1.5, alpha=0.8)
                ax.set_xlabel('Epoch'); ax.set_ylabel('Mean Decode Rank')
                ax.set_title(f'mask={mt}')
                ax.legend(fontsize=5, ncol=2); ax.grid(alpha=0.3)
            fig.suptitle(f'Chain Length → Decode Rank Evolution — {fmt}', fontsize=12, y=1.02)
            fig.tight_layout(); figs[f'chain_rank_evo_{fmt}'] = fig

        # ── Fig 17: Counterfactual carry analysis ──
        cf_data = []
        for mt in MASK_TYPES:
            key = _fk('diffusion', fmt, mt, 'counterfactual')
            cf = all_final.get(key)
            if cf:
                cf_data.append((mt, cf))
        if cf_data:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            mts = [d[0] for d in cf_data]
            x = range(len(mts))

            # Panel 1: prediction flip rate
            ax = axes[0]
            vals = [d[1]['prediction_flip_rate'] for d in cf_data]
            colors = [COLORS.get(_ck('diffusion', mt)+'con',
                      COLORS.get(_ck('diffusion', mt), '#333')) for mt in mts]
            ax.bar(x, vals, color=colors, alpha=0.8)
            ax.set_xticks(list(x)); ax.set_xticklabels(mts, fontsize=8, rotation=20)
            ax.set_ylabel('Prediction Flip Rate')
            ax.set_title('How often carry-in flips the prediction')
            ax.set_ylim(0, 1.05); ax.grid(alpha=0.3, axis='y')

            # Panel 2: confidence delta
            ax = axes[1]
            vals = [d[1]['mean_conf_delta'] for d in cf_data]
            ax.bar(x, vals, color=colors, alpha=0.8)
            ax.set_xticks(list(x)); ax.set_xticklabels(mts, fontsize=8, rotation=20)
            ax.set_ylabel('Mean |Δconfidence|')
            ax.set_title('Confidence shift from carry-in change')
            ax.grid(alpha=0.3, axis='y')

            # Panel 3: accuracy by carry-in condition
            ax = axes[2]
            w = 0.35
            c0 = [d[1]['acc_carry_in_0'] for d in cf_data]
            c1 = [d[1]['acc_carry_in_1'] for d in cf_data]
            ax.bar([xi-w/2 for xi in x], c0, w, label='carry-in=0', color='#3498db', alpha=0.8)
            ax.bar([xi+w/2 for xi in x], c1, w, label='carry-in=1', color='#e74c3c', alpha=0.8)
            ax.set_xticks(list(x)); ax.set_xticklabels(mts, fontsize=8, rotation=20)
            ax.set_ylim(0, 1.05); ax.set_ylabel('Accuracy')
            ax.set_title('Accuracy by carry-in condition')
            ax.legend(); ax.grid(alpha=0.3, axis='y')

            fig.suptitle(f'Counterfactual Carry Analysis — {fmt}', fontsize=12, y=1.02)
            fig.tight_layout(); figs[f'counterfactual_{fmt}'] = fig

        # ── Fig 18: Standard vs carry-heavy comparison ──
        heavy_data = []
        for mt in MASK_TYPES:
            for dp in DECODE_POLICIES:
                key_std = _fk('diffusion', fmt, mt, dp)
                key_hvy = _fk('diffusion', fmt, mt, f'heavy_{dp}')
                r_std = all_final.get(key_std)
                r_hvy = all_final.get(key_hvy)
                if r_std and r_hvy:
                    heavy_data.append((_ck('diffusion', mt, dp),
                                       r_std['accuracy'], r_hvy['accuracy']))
        if heavy_data:
            fig, ax = plt.subplots(figsize=(12, 5))
            x = range(len(heavy_data))
            w = 0.35
            ax.bar([xi-w/2 for xi in x], [d[1] for d in heavy_data], w,
                   label='Standard test', color='#3498db', alpha=0.8)
            ax.bar([xi+w/2 for xi in x], [d[2] for d in heavy_data], w,
                   label='Carry-heavy test', color='#e74c3c', alpha=0.8)
            ax.set_xticks(list(x))
            ax.set_xticklabels([d[0] for d in heavy_data], fontsize=7, rotation=30)
            ax.set_ylim(0, 1.05); ax.set_ylabel('Exact Match Accuracy')
            ax.set_title(f'Standard vs Carry-Heavy — {fmt}')
            ax.legend(); ax.grid(alpha=0.3, axis='y')
            fig.tight_layout(); figs[f'carry_heavy_{fmt}'] = fig

        # ── Fig 19: Dependency context confidence + accuracy ──
        gkp_results = {}
        for mt in MASK_TYPES:
            key = _fk('diffusion', fmt, mt, 'gkp_detail')
            gkp_d = all_final.get(key)
            if gkp_d:
                gkp_results[mt] = gkp_d
        if gkp_results:
            n_mt = len(gkp_results)
            fig, axes = plt.subplots(2, n_mt, figsize=(7*n_mt, 10), squeeze=False)
            ctx_order = ['g', 'k', 'p_above_g', 'p_above_k', 'p_above_p', 'p_bottom', 'carry_out']
            ctx_colors = {'g': '#2ecc71', 'k': '#3498db', 'p_above_g': '#27ae60',
                          'p_above_k': '#2980b9', 'p_above_p': '#e74c3c',
                          'p_bottom': '#f39c12', 'carry_out': '#95a5a6'}
            for ai, (mt, gd) in enumerate(gkp_results.items()):
                bdc = gd['by_dep_context']
                cats = [c for c in ctx_order if c in bdc]
                x = range(len(cats))
                confs = [bdc[c]['mean_conf'] for c in cats]
                accs = [bdc[c]['mean_acc'] for c in cats]
                colors = [ctx_colors.get(c, '#333') for c in cats]

                ax = axes[0][ai]
                ax.bar(x, confs, color=colors, alpha=0.8)
                ax.set_xticks(list(x)); ax.set_xticklabels(cats, fontsize=7, rotation=30)
                ax.set_ylim(0, 1.05); ax.set_ylabel('Mean Confidence')
                ax.set_title(f'mask={mt}'); ax.grid(alpha=0.3, axis='y')

                ax = axes[1][ai]
                ax.bar(x, accs, color=colors, alpha=0.8)
                ax.set_xticks(list(x)); ax.set_xticklabels(cats, fontsize=7, rotation=30)
                ax.set_ylim(0, 1.05); ax.set_ylabel('Mean Accuracy')
                ax.set_title(f'mask={mt}'); ax.grid(alpha=0.3, axis='y')

            fig.suptitle(f'Fully-Masked Confidence & Accuracy by Dependency Context — {fmt}',
                         fontsize=12, y=1.02)
            fig.tight_layout(); figs[f'dep_context_{fmt}'] = fig

        # ── Fig 20: Chain length vs confidence for p positions ──
        if gkp_results:
            fig, axes = plt.subplots(1, len(gkp_results), figsize=(7*len(gkp_results), 5),
                                     squeeze=False)
            axes = axes[0]
            for ai, (mt, gd) in enumerate(gkp_results.items()):
                ax = axes[ai]
                bcl = gd['by_chain_len']
                # Only chain_len >= 1 (p positions)
                cls = sorted([cl for cl in bcl if cl >= 1])
                if cls:
                    confs = [bcl[cl]['mean_conf'] for cl in cls]
                    accs = [bcl[cl]['mean_acc'] for cl in cls]
                    ax.plot(cls, confs, '-o', color='#e74c3c', label='confidence', lw=2, ms=5)
                    ax.plot(cls, accs, '-s', color='#3498db', label='accuracy', lw=2, ms=5)
                    # Also show chain=0 (g/k) as reference
                    if 0 in bcl:
                        ax.axhline(bcl[0]['mean_conf'], color='#e74c3c', ls=':', alpha=0.5,
                                   label=f'g/k conf={bcl[0]["mean_conf"]:.3f}')
                        ax.axhline(bcl[0]['mean_acc'], color='#3498db', ls=':', alpha=0.5,
                                   label=f'g/k acc={bcl[0]["mean_acc"]:.3f}')
                corr = gd.get('chain_conf_corr')
                corr_str = f' (r={corr:.3f})' if corr is not None else ''
                ax.set_xlabel('Carry Chain Length')
                ax.set_ylabel('Value')
                ax.set_ylim(0, 1.05)
                ax.set_title(f'mask={mt}{corr_str}')
                ax.legend(fontsize=7); ax.grid(alpha=0.3)
            fig.suptitle(f'Chain Length → Confidence & Accuracy (p positions) — {fmt}',
                         fontsize=12, y=1.02)
            fig.tight_layout(); figs[f'chain_conf_{fmt}'] = fig

        # ── Fig 21: No-propagate one-step accuracy ──
        nop_data_fig = []
        for mt in MASK_TYPES:
            # From standard test (no_p_samples within test set)
            key = _fk('diffusion', fmt, mt, 'gkp_detail')
            gd = all_final.get(key)
            if gd:
                nps = gd['no_p_samples']
                nop_data_fig.append((mt, nps['one_step_exact_acc'],
                                     nps['one_step_digit_acc'], nps['n'], 'in-test'))
            # From dedicated no-p test
            key_np = _fk('diffusion', fmt, mt, 'no_prop')
            gd_np = all_final.get(key_np)
            if gd_np:
                nps = gd_np['no_p_samples']
                nop_data_fig.append((mt, nps['one_step_exact_acc'],
                                     nps['one_step_digit_acc'], nps['n'], 'dedicated'))
        if nop_data_fig:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            # Group by source
            for ai, source in enumerate(['in-test', 'dedicated']):
                ax = axes[ai]
                items = [(mt, ex, dg, n) for mt, ex, dg, n, src in nop_data_fig if src == source]
                if not items: continue
                x = range(len(items))
                w = 0.35
                ax.bar([xi-w/2 for xi in x], [it[1] for it in items], w,
                       label='Exact match', color='#e74c3c', alpha=0.8)
                ax.bar([xi+w/2 for xi in x], [it[2] for it in items], w,
                       label='Digit acc', color='#3498db', alpha=0.8)
                ax.set_xticks(list(x))
                ax.set_xticklabels([f'{it[0]}\n(n={it[3]})' for it in items],
                                   fontsize=7)
                ax.set_ylim(0, 1.05); ax.set_ylabel('Accuracy')
                ax.set_title(f'No-Propagate One-Step ({source})')
                ax.legend(); ax.grid(alpha=0.3, axis='y')
            fig.suptitle(f'Can the Model Solve Without Carry Ambiguity? — {fmt}',
                         fontsize=12, y=1.02)
            fig.tight_layout(); figs[f'no_prop_acc_{fmt}'] = fig

        # ── Fig 22: Confidence cascade — the key dependency graph test ──
        cascade_data = {}
        for mt in MASK_TYPES:
            key = _fk('diffusion', fmt, mt, 'cascade')
            c = all_final.get(key)
            if c:
                cascade_data[mt] = c
        if cascade_data:
            n_mt = len(cascade_data)
            fig, axes = plt.subplots(2, n_mt, figsize=(7*n_mt, 9), squeeze=False)
            for ai, (mt, cd) in enumerate(cascade_data.items()):
                # Top row: by g/k/p type of revealed position
                ax = axes[0][ai]
                types = ['g', 'k', 'p']
                type_colors = {'g': '#2ecc71', 'k': '#3498db', 'p': '#e74c3c'}
                x = range(len(types))
                above_vals = [cd['delta_conf_above_by_type'].get(t, {}).get('mean', 0) for t in types]
                above_n = [cd['delta_conf_above_by_type'].get(t, {}).get('n', 0) for t in types]
                other_vals = [cd['delta_conf_others_by_type'].get(t, {}).get('mean', 0) for t in types]
                w = 0.35
                ax.bar([xi-w/2 for xi in x], above_vals, w,
                       color=[type_colors[t] for t in types], alpha=0.9,
                       label='position above')
                ax.bar([xi+w/2 for xi in x], other_vals, w,
                       color=[type_colors[t] for t in types], alpha=0.3,
                       label='other positions')
                ax.set_xticks(list(x))
                ax.set_xticklabels([f'{t} (n={above_n[i]})' for i, t in enumerate(types)])
                ax.axhline(0, color='grey', ls='-', lw=0.5)
                ax.set_ylabel('Mean Δconf'); ax.set_title(f'mask={mt} — by revealed type')
                ax.legend(fontsize=7); ax.grid(alpha=0.3, axis='y')

                # Bottom row: chain_end vs chain_mid (the key test)
                ax = axes[1][ai]
                cr = cd.get('delta_conf_above_resolved', {})
                cats = ['chain_end', 'chain_mid']
                cat_labels = ['chain end\n(g/k/p_above_gk)', 'chain mid\n(p_above_p)']
                cat_colors = ['#2ecc71', '#e74c3c']
                vals = [cr.get(c, {}).get('mean', 0) for c in cats]
                ns = [cr.get(c, {}).get('n', 0) for c in cats]
                ax.bar(range(2), vals, color=cat_colors, alpha=0.8)
                ax.set_xticks(range(2))
                ax.set_xticklabels([f'{cat_labels[i]}\n(n={ns[i]})' for i in range(2)],
                                   fontsize=8)
                ax.axhline(0, color='grey', ls='-', lw=0.5)
                ax.set_ylabel('Mean Δconf of position above')
                ax.set_title(f'mask={mt} — chain end vs mid')
                ax.grid(alpha=0.3, axis='y')

            fig.suptitle(
                f'Confidence Cascade Analysis — {fmt}\n'
                'Top: Δconf by g/k/p type | Bottom: chain_end (carry resolved) vs chain_mid (still ambiguous)',
                fontsize=11, y=1.03)
            fig.tight_layout(); figs[f'cascade_{fmt}'] = fig

    return figs


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run(tag=''):
    exp_name = EXP_NAME + (f'_{tag}' if tag else '')

    print("=" * 70)
    print(f"  EXP 2: 8-Digit Addition — Learning Dynamics")
    if tag:
        print(f"  Tag: {tag}")
    print("=" * 70)
    mount_drive()
    torch.manual_seed(SEED)

    tok = build_tok()
    sample = _fmt_plain(12345678, 87654321)
    max_len = len(tok.encode(sample))
    print(f"  ND={ND}, ANS_LEN={ANS_LEN}, max_len={max_len}")
    print(f"  Model: {N_LAYER}L/{N_HEAD}H/{N_EMBD}D, dropout={DROPOUT}")
    print(f"  Data:  N_TRAIN={N_TRAIN}, N_TEST={N_TEST}")
    print(f"  Budget: {MAX_EPOCHS} epochs × {N_TRAIN//BATCH_SIZE} batches "
          f"= {MAX_EPOCHS * (N_TRAIN//BATCH_SIZE):,} iters")
    print(f"  Eval every {EVAL_EVERY} epochs (dense early: /{EVAL_EVERY//5} first 10%, "
          f"/{EVAL_EVERY//2} next 20%)")
    print(f"  Gen eval every {GEN_EVAL_EVERY} epochs (dense early: /{GEN_EVAL_EVERY//4} first 10%)")
    print(f"  Formats: {FORMATS}  Masks: {MASK_TYPES}  Decode: {DECODE_POLICIES}")
    print(f"  AR: {'yes' if RUN_AR else 'skip'}")

    all_dyn, all_final = {}, {}

    for fmt in FORMATS:
        train_data = gen_data(N_TRAIN, fmt, seed=SEED)
        test_data = gen_test_data(N_TEST, fmt, seed=9000)
        cd = defaultdict(int)
        mcl = defaultdict(int)
        for s in train_data:
            a, b = _parse_operands(s)
            cd[_count_carries(a, b)] += 1
            mcl[_max_chain_len(a, b)] += 1
        print(f"\n  [{fmt}] train N={len(train_data)}")
        print(f"    carries={dict(sorted(cd.items()))}")
        print(f"    max_chain={dict(sorted(mcl.items()))}")
        cd_test = defaultdict(int)
        mcl_test = defaultdict(int)
        for s in test_data:
            a, b = _parse_operands(s)
            cd_test[_count_carries(a, b)] += 1
            mcl_test[_max_chain_len(a, b)] += 1
        print(f"  [{fmt}] test  N={len(test_data)} (full {ND}-digit)")
        print(f"    carries={dict(sorted(cd_test.items()))}")
        print(f"    max_chain={dict(sorted(mcl_test.items()))}")

        # ── AR ──
        if RUN_AR:
            key = _fk('ar', fmt)
            print(f"\n{'━'*60}\n▶ {key}\n{'━'*60}")
            m, d = train_with_dynamics('ar', tok, train_data, test_data, max_len, fmt)
            all_dyn[key] = d
            r = final_evaluate(m, tok, test_data, 'ar', fmt)
            all_final[key] = r
            print(f"  Final gen acc: {r['accuracy']:.4f}")
            del m; torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # ── Diffusion ──
        for mt in MASK_TYPES:
            kb = _fk('diffusion', fmt, mt, 'confidence')
            print(f"\n{'━'*60}\n▶ {kb}\n{'━'*60}")
            m, d = train_with_dynamics('diffusion', tok, train_data, test_data,
                                       max_len, fmt, mask_type=mt)
            all_dyn[kb] = d

            for dp in DECODE_POLICIES:
                key = _fk('diffusion', fmt, mt, dp)
                print(f"\n  Final eval: {key}")
                r = final_evaluate(m, tok, test_data, 'diffusion', fmt, decode_policy=dp)
                all_final[key] = r
                print(f"  Acc: {r['accuracy']:.4f}")
                oa = r.get('decode_order_analysis')
                # Decode order analysis only meaningful for confidence decode
                # (for fixed-order policies like lsb, the order is tautological)
                if oa and dp == 'confidence':
                    labs = _pos_labels(fmt); mr = oa['mean_rank']
                    print(f"    LSB concordance: {oa['pairwise_concordance']:.3f}")
                    print(f"    Rank: {' '.join(f'{labs[j]}={mr[j]:.1f}' for j in range(ANS_LEN))}")
                    gkp_mr = oa.get('gkp_mean_rank')
                    ebh = oa.get('easy_before_hard')
                    if gkp_mr:
                        parts = [f"{cat}={v:.2f}" if v is not None else f"{cat}=n/a"
                                 for cat, v in gkp_mr.items()]
                        print(f"    GKP rank: {' '.join(parts)}"
                              + (f"  easy→hard: {ebh:.3f}" if ebh is not None else ''))
                    clvr = oa.get('chain_len_vs_rank')
                    if clvr:
                        parts = [f"cl={cl}→{r:.2f}" for cl, r in sorted(clvr.items())]
                        print(f"    Chain→rank: {' '.join(parts)}")
                    csr = oa.get('carry_source_rank')
                    if csr:
                        parts = [f"{k}={v:.2f}" for k, v in sorted(csr.items())]
                        print(f"    Source→rank: {' '.join(parts)}")

            # ── Counterfactual carry eval (once per trained model) ──
            print(f"  Counterfactual carry eval...")
            cf_pairs = gen_counterfactual_pairs(200, seed=SEED + 42)
            cf_result = eval_counterfactual(m, tok, cf_pairs, fmt, max_len, device=DEVICE)
            all_final[_fk('diffusion', fmt, mt, 'counterfactual')] = cf_result
            print(f"    pairs={cf_result['n_pairs']} "
                  f"flip_rate={cf_result['prediction_flip_rate']:.3f} "
                  f"Δconf={cf_result['mean_conf_delta']:.4f} "
                  f"acc(c=0)={cf_result['acc_carry_in_0']:.3f} "
                  f"acc(c=1)={cf_result['acc_carry_in_1']:.3f}")

            # ── Carry-heavy adversarial eval ──
            print(f"  Carry-heavy eval...")
            heavy_data = gen_carry_heavy_test(N_TEST, fmt, seed=7000, min_max_chain=3)
            for dp in DECODE_POLICIES:
                key_h = _fk('diffusion', fmt, mt, f'heavy_{dp}')
                r_h = final_evaluate(m, tok, heavy_data, 'diffusion', fmt, decode_policy=dp)
                all_final[key_h] = r_h
                key_std = _fk('diffusion', fmt, mt, dp)
                std_acc = all_final[key_std]['accuracy']
                print(f"    heavy {dp}: {r_h['accuracy']:.4f} (standard: {std_acc:.4f}, "
                      f"Δ={r_h['accuracy']-std_acc:+.4f})")

            # ── GKP dependency-context probe (sample-level) ──
            print(f"  GKP dependency probe...")
            gkp_detail = probe_gkp_detailed(m, tok, test_data, fmt, max_len, device=DEVICE)
            all_final[_fk('diffusion', fmt, mt, 'gkp_detail')] = gkp_detail
            # Print dependency context summary
            for ctx, stats in sorted(gkp_detail['by_dep_context'].items()):
                print(f"    {ctx:<12s}: conf={stats['mean_conf']:.3f} "
                      f"acc={stats['mean_acc']:.3f} (n={stats['n']})")
            corr = gkp_detail['chain_conf_corr']
            print(f"    chain_len↔confidence corr: {corr:.3f}" if corr is not None else
                  f"    chain_len↔confidence corr: n/a")
            np_info = gkp_detail['no_p_samples']
            print(f"    No-p samples in test: n={np_info['n']} "
                  f"one-step exact={np_info['one_step_exact_acc']:.3f} "
                  f"digit={np_info['one_step_digit_acc']:.3f}")

            # ── No-propagate dedicated test set ──
            nop_data = gen_no_propagate_test(N_TEST, fmt, seed=8888)
            if nop_data:
                nop_detail = probe_gkp_detailed(m, tok, nop_data, fmt, max_len, device=DEVICE)
                all_final[_fk('diffusion', fmt, mt, 'no_prop')] = nop_detail
                npi = nop_detail['no_p_samples']
                print(f"    No-p dedicated test (n={len(nop_data)}): "
                      f"one-step exact={npi['one_step_exact_acc']:.3f} "
                      f"digit={npi['one_step_digit_acc']:.3f}")

            # ── Confidence cascade analysis ──
            print(f"  Confidence cascade analysis...")
            cascade = analyse_confidence_cascade(
                m, tok, test_data, fmt, max_len, n_samples=200, device=DEVICE)
            all_final[_fk('diffusion', fmt, mt, 'cascade')] = cascade
            print(f"    {cascade['summary']}")

            del m; torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── Figures ──
    print(f"\n{'='*70}\n  Generating figures...\n{'='*70}")
    figs = make_figures(all_dyn, all_final)

    # ── Save ──
    sd = {'config': {
        'ND': ND, 'ANS_LEN': ANS_LEN, 'N_TRAIN': N_TRAIN,
        'N_TEST': N_TEST, 'MAX_EPOCHS': MAX_EPOCHS, 'BATCH_SIZE': BATCH_SIZE,
        'N_LAYER': N_LAYER, 'N_HEAD': N_HEAD, 'N_EMBD': N_EMBD,
        'DROPOUT': DROPOUT, 'MASK_TYPES': MASK_TYPES,
        'DECODE_POLICIES': DECODE_POLICIES, 'FORMATS': FORMATS,
        'RUN_AR': RUN_AR, 'tag': tag,
    }}
    for k, d_val in all_dyn.items():
        sd[f'dyn_{k}'] = {'checkpoints': d_val['checkpoints'],
                          'gen_checkpoints': d_val.get('gen_checkpoints', []),
                          'train_loss': d_val['train_loss']}
    for k, r in all_final.items():
        sr = {kk: vv for kk, vv in r.items() if kk != 'decode_order_analysis'}
        oa = r.get('decode_order_analysis')
        if oa: sr['decode_order'] = {kk: vv for kk, vv in oa.items() if kk != 'rank_histogram'}
        sd[f'final_{k}'] = sr
    save_results(exp_name, sd, figures=figs)

    # ── Summary ──
    print(f"\n{'='*70}\n  SUMMARY\n{'='*70}")
    for fmt in FORMATS:
        labs = _pos_labels(fmt)
        print(f"\n  ━━ {fmt} ━━")
        print(f"  {'Condition':<25} {'Acc':>6}  {'TG':>12}  Position Accuracy")
        print(f"  {'─'*90}")
        all_conds = []
        if RUN_AR: all_conds.append(('ar','',''))
        for mt in MASK_TYPES:
            for dp in DECODE_POLICIES:
                all_conds.append(('diffusion', mt, dp))
        for obj, mt, dp in all_conds:
            key = _fk(obj, fmt, mt, dp)
            r = all_final.get(key)
            if not r: continue
            dk = _fk(obj, fmt, mt, 'confidence')
            dyn = all_dyn.get(dk, {})
            tg = dyn.get('checkpoints', [{}])[-1].get('token_gradients', '?')
            pa = r.get('position_accuracy', [])
            print(f"  {_ck(obj,mt,dp):<25} {r['accuracy']:>6.3f}  {tg:>12,}  "
                  f"{' '.join(f'{v:.2f}' for v in pa)}")

    plt.show()
    return all_dyn, all_final


if __name__ == '__main__':
    args = parse_args()
    seeds = args.seeds if args.seeds else [SEED]
    if len(seeds) == 1:
        globals()['SEED'] = seeds[0]
        run(tag=args.tag)
    else:
        print(f"Multi-seed run: {seeds}")
        for si, seed in enumerate(seeds):
            globals()['SEED'] = seed
            seed_tag = f"{args.tag}_s{seed}" if args.tag else f"s{seed}"
            print(f"\n{'#'*70}\n# Seed {seed} ({si+1}/{len(seeds)})\n{'#'*70}")
            run(tag=seed_tag)
