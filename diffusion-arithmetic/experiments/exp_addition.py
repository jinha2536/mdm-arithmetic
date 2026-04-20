"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Addition — Carry Dependency Learning + PUMA Coverage Deficit
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Training: random vs puma masking (iteration-based, EMA)
  Decode:   confidence (model) vs lsb (oracle) vs random
  Analyses: GKP dependency, carry rarity × accuracy, PUMA coverage,
            corner cases, confidence cascade, counterfactual carry
  Continuation: random→puma, puma→random (representation persistence)
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
from core.train_utils import (
    mount_drive, save_results, save_checkpoint, encode_samples,
    train_diffusion, puma_k_fixed, puma_k_linear, puma_k_step, generate_diffusion,
    simulate_reveal_trajectory, compute_reveal_vs_order_tau, DEVICE,
)

EXP_NAME = 'exp_addition'

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Config
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ND = 32; ANS_LEN = ND + 1
# N_TEST increased: natural distribution has chain≥24 at ~0.1-1% rate, so N=1000
# barely reaches 10 extreme samples. N=10000 gives ~100 which is statistically meaningful.
# Constructive tests (chain sweep, corner cases) are per-bucket and do not scale with N_TEST.
N_TRAIN = 20000; N_TEST = 10000; BATCH_SIZE = 256
# Per-bucket count for constructed (chain sweeps, corner cases) — shared across all analyses
N_PER_BUCKET = 500
# ~78 iters/epoch at N_TRAIN=20000/BS=256 → 400k iters ≈ 5000 epochs
MAX_ITERS = 400000; EVAL_EVERY = 5000; LOG_EVERY = 1000
GEN_EVAL_EVERY = 10000; GEN_EVAL_N = 500
# Reveal trajectory: K stages, match PUMA K_END for apples-to-apples
REVEAL_K_DEFAULT = 16
# Reveal-vs-reasoning tau diagnostic (PUMA-only, stratified across chain length).
# Per-stratum cap ensures all chain-length buckets present in training data are
# represented in the tracked subset; the final τ is decomposed by stratum so
# easy/hard cases can be compared separately.
REVEAL_TAU_N_TRACKED = 100        # total cap across strata (per-stratum = cap/num_strata)
REVEAL_TAU_EVERY = 20000           # less frequent than eval; trajectory is slow-changing
REVEAL_TAU_K_THRESHOLD_FRAC = 0.7  # only run once K_cur >= K_final * this
MASK_TYPES = ['random', 'puma']
DECODE_POLICIES = ['confidence', 'lsb']
N_LAYER = 3; N_HEAD = 3; N_EMBD = 192; DROPOUT = 0.1; POS_ENC = 'absolute'
LR = 1e-3; MIN_LR = 1e-4; WARMUP_ITERS = 2000; GRAD_CLIP = 1.0
WEIGHT_DECAY = 0.1; EMA_DECAY = 0.9999
PUMA_TAU = 0.9; PUMA_K = 8  # fixed K (unused when K_START is set)
# Step schedule: K_START, +K_STEP every K_EVERY iters, cap K_END. The K range
# is chosen so reveal-per-step = ans_len/K aligns with confidence informativeness:
# start at ~10 tokens/step (random-like, confidence rank uninformative early in
# training) and ramp to ~2 tokens/step (fine-grained once confidence is reliable).
# For addition (ans_len=33): K=3 → 11 tokens/step, K=16 → ~2 tokens/step.
PUMA_K_START = 3; PUMA_K_END = 16
PUMA_K_STEP = 3; PUMA_K_EVERY = None  # None = auto (ramp over first 1/3 of training)
SEED = 42
NO_AMP = False
DATA_MODE = 'natural'

# Early stopping: patience in iters (None = disabled, run full max_iters)
# ~50k iters ≈ 640 epochs of patience
PATIENCE = 50000

# Continuation training: ~100-200 epochs ≈ 10k iters
CONTINUATION_ITERS = 10000


def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--nd', type=int); p.add_argument('--n-train', type=int)
    p.add_argument('--n-test', type=int); p.add_argument('--max-iters', type=int)
    p.add_argument('--batch-size', type=int); p.add_argument('--eval-every', type=int)
    p.add_argument('--gen-eval-every', type=int)
    p.add_argument('--n-layer', type=int); p.add_argument('--n-head', type=int)
    p.add_argument('--n-embd', type=int); p.add_argument('--dropout', type=float)
    p.add_argument('--lr', type=float)
    p.add_argument('--weight-decay', type=float)
    p.add_argument('--patience', type=int)
    p.add_argument('--puma-tau', type=float)
    p.add_argument('--puma-k', type=int)
    p.add_argument('--puma-k-start', type=int)
    p.add_argument('--puma-k-end', type=int)
    p.add_argument('--puma-k-step', type=int)
    p.add_argument('--puma-k-every', type=int)
    p.add_argument('--masks', nargs='+'); p.add_argument('--decode', nargs='+')
    p.add_argument('--data-mode', choices=['balanced', 'natural'])
    p.add_argument('--continuation-iters', type=int)
    p.add_argument('--no-continuation', action='store_true')
    p.add_argument('--no-amp', action='store_true')
    p.add_argument('--tag', type=str, default=''); p.add_argument('--seed', type=int)
    p.add_argument('--seeds', nargs='+', type=int)
    try:
        args, _ = p.parse_known_args()
    except SystemExit:
        args, _ = p.parse_known_args([])
    g = globals()
    for a, gl in {'n_train': 'N_TRAIN', 'n_test': 'N_TEST', 'max_iters': 'MAX_ITERS',
                   'batch_size': 'BATCH_SIZE', 'eval_every': 'EVAL_EVERY',
                   'gen_eval_every': 'GEN_EVAL_EVERY', 'n_layer': 'N_LAYER',
                   'n_head': 'N_HEAD', 'n_embd': 'N_EMBD', 'dropout': 'DROPOUT',
                   'lr': 'LR', 'weight_decay': 'WEIGHT_DECAY',
                   'patience': 'PATIENCE', 'puma_tau': 'PUMA_TAU',
                   'puma_k': 'PUMA_K', 'puma_k_start': 'PUMA_K_START',
                   'puma_k_end': 'PUMA_K_END', 'puma_k_step': 'PUMA_K_STEP',
                   'puma_k_every': 'PUMA_K_EVERY',
                   'seed': 'SEED', 'no_amp': 'NO_AMP', 'continuation_iters': 'CONTINUATION_ITERS'}.items():
        v = getattr(args, a, None)
        if v is not None: g[gl] = v
    if args.nd:
        g['ND'] = args.nd; g['ANS_LEN'] = args.nd + 1
    if args.masks: g['MASK_TYPES'] = args.masks
    if args.decode: g['DECODE_POLICIES'] = args.decode
    if args.data_mode: g['DATA_MODE'] = args.data_mode
    return args


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Data helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _pad(n, w): return str(n).zfill(w)
def _fmt_plain(a, b): return f"{_pad(a,ND)}+{_pad(b,ND)}={_pad(a+b,ANS_LEN)}"
def get_answer(s): return s.split('=')[1]
def _parse_operands(s):
    parts = s.split('=')[0].split('+'); return int(parts[0]), int(parts[1])

def _count_carries(a, b):
    a_s, b_s = _pad(a, ND), _pad(b, ND)
    carry, count = 0, 0
    for i in range(ND - 1, -1, -1):
        s = int(a_s[i]) + int(b_s[i]) + carry; carry = s // 10; count += carry
    return count

def _carry_at_answer_pos(a, b):
    """Per-answer-position carry-in flag (plain format)."""
    a_s, b_s = _pad(a, ND), _pad(b, ND)
    flags, carry = [], 0
    for i in range(ND - 1, -1, -1):
        s = int(a_s[i]) + int(b_s[i]) + carry; carry = s // 10; flags.append(bool(carry))
    ci = [False] * ANS_LEN
    for k in range(ANS_LEN):
        lp = ND - k
        if lp == ND: ci[k] = flags[ND-1] if ND-1 < len(flags) else False
        elif 0 <= lp-1 < len(flags): ci[k] = flags[lp-1]
    return ci

def _gkp_at_answer_pos(a, b):
    """Classify each answer position as g/k/p/carry_out (plain format)."""
    a_s, b_s = _pad(a, ND), _pad(b, ND)
    digit_gkp = []
    for i in range(ND - 1, -1, -1):
        s = int(a_s[i]) + int(b_s[i])
        digit_gkp.append('g' if s >= 10 else ('p' if s == 9 else 'k'))
    out = ['?'] * ANS_LEN
    out[0] = 'carry_out'
    for j in range(ND): out[ND - j] = digit_gkp[j]
    return out

def _dependency_context_at_pos(a, b):
    """Dependency context: g, k, p_above_g, p_above_k, p_above_p, p_bottom, carry_out."""
    a_s, b_s = _pad(a, ND), _pad(b, ND)
    gkp = []
    for i in range(ND - 1, -1, -1):
        s = int(a_s[i]) + int(b_s[i])
        gkp.append('g' if s >= 10 else ('p' if s == 9 else 'k'))
    dep = ['?'] * ND
    for d in range(ND):
        if gkp[d] in ('g', 'k'): dep[d] = gkp[d]
        elif d == 0: dep[d] = 'p_bottom'
        elif gkp[d-1] == 'g': dep[d] = 'p_above_g'
        elif gkp[d-1] == 'k': dep[d] = 'p_above_k'
        else: dep[d] = 'p_above_p'
    out = ['?'] * ANS_LEN
    out[0] = 'carry_out'
    for j in range(ND): out[ND - j] = dep[j]
    return out

def _chain_stats(a, b):
    """Rich carry-chain stats."""
    a_s, b_s = _pad(a, ND), _pad(b, ND)
    gkp = []
    for i in range(ND - 1, -1, -1):
        s = int(a_s[i]) + int(b_s[i])
        gkp.append('g' if s >= 10 else ('p' if s == 9 else 'k'))
    carry = 0
    for i in range(ND - 1, -1, -1):
        s = int(a_s[i]) + int(b_s[i]) + carry; carry = s // 10
    chains, rs, rl = [], None, 0
    for d in range(ND):
        if gkp[d] == 'p':
            if rs is None: rs, rl = d, 1
            else: rl += 1
        else:
            if rs is not None: chains.append(rl)
            rs, rl = None, 0
    if rs is not None: chains.append(rl)
    reaches_msb = (gkp[ND-1] == 'p')
    msb_cl = 0
    if reaches_msb:
        d = ND-1
        while d >= 0 and gkp[d] == 'p': msb_cl += 1; d -= 1
    return {'max_chain_len': max(chains, default=0), 'n_propagate': sum(1 for g in gkp if g == 'p'),
            'chain_reaches_msb': reaches_msb, 'msb_carry_out': bool(carry),
            'msb_chain_len': msb_cl, 'gkp': gkp}

def _max_chain_len(a, b): return _chain_stats(a, b)['max_chain_len']

def _pos_labels():
    return ['MSB'] + [f'p{j}' for j in range(1, ANS_LEN-1)] + ['LSB']

def build_tok():
    return CharTokenizer(list('0123456789+='), {'mask': 'M', 'pad': 'P'})


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Data generation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def gen_data_natural(n, seed):
    """Natural distribution — no carry balancing."""
    rng = random.Random(seed)
    lo, hi = 10**(ND-1), 10**ND - 1
    return [_fmt_plain(rng.randint(lo, hi), rng.randint(lo, hi)) for _ in range(n)]

def gen_data_balanced(n, seed):
    """Carry-balanced training data."""
    rng = random.Random(seed)
    pool = defaultdict(list); seen = set()
    for _ in range(max(n * 200, 100000)):
        da, db = rng.randint(1, ND), rng.randint(1, ND)
        lo_a = 0 if da == 1 else 10**(da-1); lo_b = 0 if db == 1 else 10**(db-1)
        a, b = rng.randint(lo_a, 10**da-1), rng.randint(lo_b, 10**db-1)
        if (a, b) in seen: continue
        seen.add((a, b)); pool[_count_carries(a, b)].append((a, b))
    target = max(1, n // max(len(pool), 1)); out = []
    for nc in sorted(pool): rng.shuffle(pool[nc]); out.extend(pool[nc][:target])
    while len(out) < n:
        a, b = rng.randint(0, 10**ND-1), rng.randint(0, 10**ND-1)
        if (a, b) not in seen: out.append((a, b)); seen.add((a, b))
    rng.shuffle(out)
    return [_fmt_plain(a, b) for a, b in out[:n]]

def gen_train_data(n, seed):
    return gen_data_natural(n, seed) if DATA_MODE == 'natural' else gen_data_balanced(n, seed)

def gen_test_data(n, seed):
    return gen_data_natural(n, seed) if DATA_MODE == 'natural' else gen_data_balanced(n, seed)

def gen_corner_case_test(n, seed, category='msb_chain'):
    """Construct samples matching structural corner categories.
    Supported: 'msb_chain' (chain reaches MSB), 'full_propagate' (all-p digits).
    """
    rng = random.Random(seed); lo, hi = 10**(ND-1), 10**ND-1; results = []
    for _ in range(n * 3000):
        if category == 'full_propagate':
            ad = [rng.randint(1, 8)] + [rng.randint(0, 9) for _ in range(ND-1)]
            bd = [9-d for d in ad]
            if bd[0] < 1: ad[0] = rng.randint(1, 8); bd[0] = 9-ad[0]
            a, b = int(''.join(str(d) for d in ad)), int(''.join(str(d) for d in bd))
        else:
            a, b = rng.randint(lo, hi), rng.randint(lo, hi)
        st = _chain_stats(a, b)
        if category == 'msb_chain' and st['chain_reaches_msb']:
            results.append(_fmt_plain(a, b))
        elif category == 'full_propagate' and st['n_propagate'] == ND:
            results.append(_fmt_plain(a, b))
        if len(results) >= n: break
    if len(results) < n: print(f"    WARNING: corner/{category}: {len(results)}/{n}")
    return results[:n]

def gen_min_chain_test(n, seed, min_chain):
    """Construct samples with a propagate chain of exactly min_chain length."""
    rng = random.Random(seed); results = []; seen = set()
    for _ in range(n * 50):
        if len(results) >= n: break
        max_start = ND - min_chain
        if max_start < 0: break
        chain_start = rng.randint(0, max_start)
        a_digits = [0] * ND; b_digits = [0] * ND
        for d in range(ND):
            if chain_start <= d < chain_start + min_chain:
                a_d = rng.randint(0, 9); b_d = 9 - a_d
            else:
                a_d = rng.randint(0, 9); b_d = rng.randint(0, 9)
                while a_d + b_d == 9: b_d = rng.randint(0, 9)
            a_digits[d] = a_d; b_digits[d] = b_d
        if a_digits[ND-1] == 0: a_digits[ND-1] = rng.randint(1, 9)
        if b_digits[ND-1] == 0: b_digits[ND-1] = rng.randint(1, 9)
        if chain_start <= ND-1 < chain_start + min_chain:
            a_digits[ND-1] = rng.randint(1, 4); b_digits[ND-1] = 9 - a_digits[ND-1]
        a_str = ''.join(str(d) for d in reversed(a_digits))
        b_str = ''.join(str(d) for d in reversed(b_digits))
        a, b = int(a_str), int(b_str)
        if (a, b) in seen: continue
        seen.add((a, b))
        if _max_chain_len(a, b) >= min_chain:
            results.append(_fmt_plain(a, b))
    return results[:n]

def gen_counterfactual_pairs(n, seed):
    rng = random.Random(seed); lo, hi = 10**(ND-1), 10**ND-1; results = []
    for _ in range(n * 500):
        if len(results) >= n: break
        target_d = rng.randint(1, ND-1); a1, b1 = rng.randint(lo, hi), rng.randint(lo, hi)
        a1_s, b1_s = _pad(a1, ND), _pad(b1, ND)
        si = ND - 1 - target_d
        carry1 = 0
        for i in range(ND-1, si, -1): s = int(a1_s[i]) + int(b1_s[i]) + carry1; carry1 = s // 10
        for _ in range(200):
            a2d = list(a1_s[:si+1]) + [str(rng.randint(0,9)) for _ in range(si+1, ND)]
            b2d = list(b1_s[:si+1]) + [str(rng.randint(0,9)) for _ in range(si+1, ND)]
            a2, b2 = int(''.join(a2d)), int(''.join(b2d))
            a2_s, b2_s = _pad(a2, ND), _pad(b2, ND)
            c2 = 0
            for i in range(ND-1, si, -1): s = int(a2_s[i]) + int(b2_s[i]) + c2; c2 = s // 10
            if c2 == 1 - carry1:
                results.append({'target_d': target_d, 'pair': ((a1,b1),(a2,b2)),
                               'carry_in': (bool(carry1), bool(c2))}); break
    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Unified test suite — all analyses slice from here
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _annotate_sample(s):
    """Pre-compute all structural annotations needed by downstream analyses."""
    a, b = _parse_operands(s)
    return {
        'a': a, 'b': b,
        'chain_stats': _chain_stats(a, b),
        'gkp_at_pos': _gkp_at_answer_pos(a, b),
        'dep_ctx': _dependency_context_at_pos(a, b),
        'n_carries': _count_carries(a, b),
        'carry_flags': _carry_at_answer_pos(a, b),
    }


def _bucket_from_samples(samples, tokenizer, max_len):
    """Package a sample list with encoded ids and annotations."""
    metas = [_annotate_sample(s) for s in samples]
    ids, ans = encode_samples(samples, tokenizer, max_len)
    return {
        'samples': samples,
        'metas': metas,
        'ids': ids,
        'ans_starts': ans,
        'n': len(samples),
    }


def build_test_suite(tokenizer, max_len, seed=None):
    """
    Build a unified test suite with pre-computed annotations.

    All evaluations/analyses slice from this single source so that:
      - Sample base is consistent across analyses (enables cross-referencing)
      - Statistical power is shared (large natural + constructed extremes)
      - No redundant re-generation and re-annotation

    Structure:
        suite['natural']                     — N_TEST natural distribution
        suite['constructed']['chain_{cl}']   — min_chain ≥ cl (N_PER_BUCKET)
        suite['constructed']['full_propagate'] — all-p corner (N_PER_BUCKET)
        suite['constructed']['msb_chain']    — chain reaches MSB (N_PER_BUCKET)
        suite['counterfactual']              — CF pairs (raw dict list)
    Each bucket: {samples, metas, ids, ans_starts, n}
    """
    if seed is None: seed = SEED + 1000

    suite = {}

    # Natural distribution — the common test base
    nat_samples = gen_test_data(N_TEST, seed)
    suite['natural'] = _bucket_from_samples(nat_samples, tokenizer, max_len)

    # Constructed: chain-length sweep buckets (used for sweep + extreme analyses)
    suite['constructed'] = {}
    sweep_lengths = [2, 3, 4, 6, 8, 12]
    if ND >= 24: sweep_lengths += [16, 20]
    if ND >= 32: sweep_lengths += [24, 28]
    sweep_lengths = [cl for cl in sweep_lengths if cl <= ND]
    for min_cl in sweep_lengths:
        sp = gen_min_chain_test(N_PER_BUCKET, seed=seed + 500 + min_cl, min_chain=min_cl)
        if sp:
            suite['constructed'][f'chain_{min_cl}'] = _bucket_from_samples(sp, tokenizer, max_len)

    # Corner cases
    for cat in ['full_propagate', 'msb_chain']:
        sp = gen_corner_case_test(N_PER_BUCKET, seed=seed + 600, category=cat)
        if sp:
            suite['constructed'][cat] = _bucket_from_samples(sp, tokenizer, max_len)

    # Counterfactual pairs — raw dicts, uses natural-style samples
    suite['counterfactual'] = gen_counterfactual_pairs(200, seed=seed + 42)

    print(f"  Test suite built:")
    print(f"    natural: {suite['natural']['n']}")
    for k, v in suite['constructed'].items():
        print(f"    constructed/{k}: {v['n']}")
    print(f"    counterfactual: {len(suite['counterfactual'])}")
    return suite


def filter_natural(suite, pred):
    """Filter natural bucket by predicate on meta dict; return a bucket dict.
    Used to slice natural test data by chain length, carry pattern, etc.
    """
    nat = suite['natural']
    idx = [i for i, m in enumerate(nat['metas']) if pred(m)]
    if not idx:
        return {'samples': [], 'metas': [], 'ids': torch.empty(0, nat['ids'].shape[1], dtype=torch.long),
                'ans_starts': torch.empty(0, dtype=torch.long), 'n': 0}
    return {
        'samples': [nat['samples'][i] for i in idx],
        'metas': [nat['metas'][i] for i in idx],
        'ids': nat['ids'][idx],
        'ans_starts': nat['ans_starts'][idx],
        'n': len(idx),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Probes & analyses
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@torch.no_grad()
def probe_per_position(model, tokenizer, test_samples, max_len, device=None):
    """Fully-masked probe: per-position loss/acc/conf + dependency context."""
    if device is None: device = DEVICE
    model.eval(); mask_id = tokenizer.special_ids['mask']
    ids_all, ans_all = encode_samples(test_samples, tokenizer, max_len)
    ids_all, ans_all = ids_all.to(device), ans_all.to(device)
    ci_tensor = torch.tensor([_carry_at_answer_pos(*_parse_operands(s)) for s in test_samples],
                              dtype=torch.bool, device=device)
    dep_ctx_names = ['g', 'k', 'p_above_g', 'p_above_k', 'p_above_p', 'p_bottom', 'carry_out']
    dep_to_id = {n: i for i, n in enumerate(dep_ctx_names)}
    dep_ids = torch.tensor([[dep_to_id.get(d, 0) for d in _dependency_context_at_pos(*_parse_operands(s))]
                             for s in test_samples], dtype=torch.long, device=device)

    L = torch.zeros(ANS_LEN, device=device); C = torch.zeros(ANS_LEN, device=device)
    CF = torch.zeros(ANS_LEN, device=device); N = torch.zeros(ANS_LEN, device=device)
    Lc, Ln = torch.zeros(ANS_LEN, device=device), torch.zeros(ANS_LEN, device=device)
    Cc, Cn = torch.zeros(ANS_LEN, device=device), torch.zeros(ANS_LEN, device=device)
    Nc, Nn = torch.zeros(ANS_LEN, device=device), torch.zeros(ANS_LEN, device=device)
    dep_conf_sum = defaultdict(float); dep_acc_sum = defaultdict(float); dep_count = defaultdict(int)
    _arange = torch.arange(ANS_LEN, device=device)

    for st in range(0, len(test_samples), 128):
        en = min(st+128, len(test_samples))
        ids, ans = ids_all[st:en], ans_all[st:en]; B, T = ids.shape
        ans_pos = (ans.unsqueeze(1) + _arange).clamp(max=T-1)
        bi = torch.arange(B, device=device).unsqueeze(1).expand_as(ans_pos)

        xm = ids.clone(); xm[bi, ans_pos] = mask_id
        logits = model(xm); al = logits[bi, ans_pos]; tgt = ids[bi, ans_pos]
        lp = F.log_softmax(al, dim=-1)
        losses = -lp.gather(2, tgt.unsqueeze(2)).squeeze(2)
        cl = al.clone(); cl[:, :, mask_id] = -float('inf')
        probs = F.softmax(cl, dim=-1)
        confs = probs.max(dim=-1).values; preds = probs.argmax(dim=-1)
        corrects = (preds == tgt).float()

        # Vectorized accumulation (no per-position Python loop)
        L += losses.sum(dim=0)
        C += corrects.sum(dim=0)
        CF += confs.sum(dim=0)
        N += B

        ci_b = ci_tensor[st:en]  # [B, ANS_LEN]
        Lc += (losses * ci_b).sum(dim=0)
        Cc += (corrects * ci_b).sum(dim=0)
        Nc += ci_b.sum(dim=0).float()
        nc_b = ~ci_b
        Ln += (losses * nc_b).sum(dim=0)
        Cn += (corrects * nc_b).sum(dim=0)
        Nn += nc_b.sum(dim=0).float()

        dep_b = dep_ids[st:en].reshape(-1); cf = confs.reshape(-1); co = corrects.reshape(-1)
        for di, dn in enumerate(dep_ctx_names):
            m = (dep_b == di)
            if m.any():
                dep_conf_sum[dn] += cf[m].sum().item()
                dep_acc_sum[dn] += co[m].sum().item()
                dep_count[dn] += m.sum().item()

    s = N.clamp(1); pos_conf = (CF/s).cpu().tolist()
    def _sd(num, den): return [n/d if d > 0 else None for n, d in zip(num.cpu().tolist(), den.cpu().tolist())]
    result = {'pos_loss': (L/s).cpu().tolist(), 'pos_acc': (C/s).cpu().tolist(), 'pos_conf': pos_conf,
              'pos_loss_carry_in': _sd(Lc, Nc), 'pos_loss_no_carry': _sd(Ln, Nn),
              'pos_acc_carry_in': _sd(Cc, Nc), 'pos_acc_no_carry': _sd(Cn, Nn),
              'overall_loss': (L.sum()/s.sum()).item(), 'overall_acc': (C.sum()/s.sum()).item()}
    if dep_count:
        result['dep_context'] = {ctx: {'conf': dep_conf_sum[ctx]/n, 'acc': dep_acc_sum[ctx]/n, 'n': n}
                                  for ctx, n in dep_count.items() if n > 0}
    cr = sorted(range(ANS_LEN), key=lambda j: pos_conf[j], reverse=True)
    conc = 0; n_p = ANS_LEN * (ANS_LEN - 1) // 2
    for i in range(ANS_LEN):
        for j in range(i+1, ANS_LEN):
            conc += int(cr.index(j) < cr.index(i))
    result['conf_concordance'] = conc / n_p if n_p > 0 else 0
    result['conf_spread'] = max(pos_conf) - min(pos_conf)
    return result


@torch.no_grad()
def eval_counterfactual(model, tokenizer, cf_pairs, max_len, device=None):
    if device is None: device = DEVICE
    model.eval(); mask_id = tokenizer.special_ids['mask']

    # Batch all samples: each cf pair → 2 samples
    all_strings = []
    all_target_aj = []  # answer-relative position to check
    for cf in cf_pairs:
        td = cf['target_d']; aj = ND - td
        for a, b in cf['pair']:
            all_strings.append(_fmt_plain(a, b))
            all_target_aj.append(aj)

    # Encode all at once
    ids_all, ans_all = encode_samples(all_strings, tokenizer, max_len)
    ids_all, ans_all = ids_all.to(device), ans_all.to(device)
    target_aj = torch.tensor(all_target_aj, dtype=torch.long, device=device)
    target_pos = ans_all + target_aj  # absolute position

    # Batched forward pass (fully masked answer region)
    _arange = torch.arange(ANS_LEN, device=device)
    all_preds = []; all_confs = []; all_correct = []
    for st in range(0, len(all_strings), 128):
        en = min(st + 128, len(all_strings))
        ids = ids_all[st:en]; ans = ans_all[st:en]; B = ids.shape[0]
        ans_pos = (ans.unsqueeze(1) + _arange).clamp(max=ids.shape[1] - 1)
        bi = torch.arange(B, device=device).unsqueeze(1).expand_as(ans_pos)
        xm = ids.clone(); xm[bi, ans_pos] = mask_id
        logits = model(xm)
        tp = target_pos[st:en]
        tgt_logits = logits[torch.arange(B, device=device), tp]
        tgt_logits[:, mask_id] = -float('inf')
        probs = F.softmax(tgt_logits, dim=-1)
        pred_tok = probs.argmax(dim=-1)
        gold_tok = ids[torch.arange(B, device=device), tp]
        all_preds.extend(pred_tok.cpu().tolist())
        all_confs.extend(probs.max(dim=-1).values.cpu().tolist())
        all_correct.extend((pred_tok == gold_tok).cpu().tolist())

    # Aggregate per pair
    flips = 0; deltas = []; c0_ok, c1_ok, n0, n1 = 0, 0, 0, 0
    for pi, cf in enumerate(cf_pairs):
        i0, i1 = pi * 2, pi * 2 + 1
        deltas.append(abs(all_confs[i1] - all_confs[i0]))
        if all_preds[i0] != all_preds[i1]: flips += 1
        for mi, idx in enumerate([i0, i1]):
            if cf['carry_in'][mi]: c1_ok += all_correct[idx]; n1 += 1
            else: c0_ok += all_correct[idx]; n0 += 1
    n = len(cf_pairs)
    return {'n_pairs': n, 'mean_conf_delta': sum(deltas)/max(n,1),
            'prediction_flip_rate': flips/max(n,1),
            'acc_carry_in_0': c0_ok/max(n0,1), 'acc_carry_in_1': c1_ok/max(n1,1)}


@torch.no_grad()
def gen_eval_with_stats(model, tokenizer, test_samples, max_len,
                        decode_policy='confidence', device=None):
    """Per-sample generation with chain stats + error positions."""
    if device is None: device = DEVICE
    mask_id = tokenizer.special_ids['mask']; pad_id = tokenizer.special_ids['pad']
    model.eval(); out = []
    lsb_policy = 'r2l'  # plain format: LSB is rightmost
    for st in range(0, len(test_samples), 128):
        batch = test_samples[st:min(st+128, len(test_samples))]; B = len(batch)
        penc = [tokenizer.encode(s.split('=')[0]+'=') for s in batch]
        pm = max(len(p) for p in penc)
        pids = torch.full((B, pm), pad_id, dtype=torch.long)
        for i, e in enumerate(penc): pids[i, :len(e)] = torch.tensor(e)
        policy = lsb_policy if decode_policy == 'lsb' else decode_policy
        gen, _, info = generate_diffusion(model, pids, ANS_LEN, mask_id,
                                          policy=policy, greedy=True, device=device)
        pred_ids = gen[:, pm:pm+ANS_LEN]
        for i in range(B):
            s = batch[i]; ps = tokenizer.decode(pred_ids[i].cpu().tolist())
            gs = get_answer(s); a, b = _parse_operands(s)
            pc = [ps[j] == gs[j] if j < len(ps) else False for j in range(len(gs))]
            errs = [j for j in range(len(gs)) if j >= len(ps) or ps[j] != gs[j]]
            out.append({'correct': ps==gs, 'pos_correct': pc, 'error_positions': errs,
                       'chain_stats': _chain_stats(a, b), 'gkp_at_pos': _gkp_at_answer_pos(a, b),
                       'dep_ctx': _dependency_context_at_pos(a, b),
                       'n_carries': _count_carries(a, b), 'carry_flags': _carry_at_answer_pos(a, b)})
    return out


def stratify_results(per_sample):
    """Stratify by chain properties."""
    def _mcl(mcl):
        if mcl == 0: return 'cl=0'
        if mcl <= 2: return 'cl=1-2'
        if mcl <= 4: return 'cl=3-4'
        return 'cl=5+'
    strata = {
        'reaches_msb': lambda st: 'msb_yes' if st['chain_reaches_msb'] else 'msb_no',
        'max_chain': lambda st: _mcl(st['max_chain_len']),
        'msb_x_carry': lambda st: ('msb+carry' if st['chain_reaches_msb'] and st['msb_carry_out']
                                    else 'msb+nocarry' if st['chain_reaches_msb']
                                    else 'nomsb+carry' if st['msb_carry_out'] else 'nomsb+nocarry'),
    }
    out = {}
    for name, fn in strata.items():
        bk = defaultdict(list)
        for r in per_sample: bk[fn(r['chain_stats'])].append(r['correct'])
        out[name] = {k: {'acc': sum(v)/len(v), 'n': len(v)} for k, v in sorted(bk.items())}
    return out


def analyse_carry_rarity(per_sample, test_samples):
    """Per-position carry-in base rate × conditional accuracy."""
    ci_flags = [_carry_at_answer_pos(*_parse_operands(s)) for s in test_samples]
    N = len(per_sample); per_pos = []
    for j in range(ANS_LEN):
        ci1, ci0 = [], []
        for i in range(N):
            c = per_sample[i]['pos_correct'][j] if j < len(per_sample[i]['pos_correct']) else False
            (ci1 if ci_flags[i][j] else ci0).append(c)
        br = len(ci1)/max(N,1); a1 = sum(ci1)/len(ci1) if ci1 else None
        a0 = sum(ci0)/len(ci0) if ci0 else None
        gap = (a0-a1) if a0 is not None and a1 is not None else None
        per_pos.append({'position': j, 'base_rate': br, 'acc_carry_1': a1, 'acc_carry_0': a0, 'acc_gap': gap})
    bins = {'rare(<15%)': lambda r: r < 0.15, 'low(15-30%)': lambda r: 0.15 <= r < 0.30,
            'mid(30-50%)': lambda r: 0.30 <= r < 0.50, 'high(>=50%)': lambda r: r >= 0.50}
    binned = {}
    for bn, bfn in bins.items():
        ps = [p for p in per_pos if p['acc_gap'] is not None and bfn(p['base_rate'])]
        if ps: binned[bn] = {'n_pos': len(ps), 'mean_gap': sum(p['acc_gap'] for p in ps)/len(ps),
                              'mean_acc1': sum(p['acc_carry_1'] for p in ps)/len(ps),
                              'mean_acc0': sum(p['acc_carry_0'] for p in ps)/len(ps)}
    valid = [(p['base_rate'], p['acc_gap']) for p in per_pos if p['acc_gap'] is not None]
    corr = None
    if len(valid) >= 3:
        brs, gs = [v[0] for v in valid], [v[1] for v in valid]
        mb, mg = sum(brs)/len(brs), sum(gs)/len(gs)
        c = sum((b-mb)*(g-mg) for b, g in zip(brs, gs))
        sb = sum((b-mb)**2 for b in brs)**0.5; sg = sum((g-mg)**2 for g in gs)**0.5
        corr = c/(sb*sg) if sb > 0 and sg > 0 else 0.0
    return {'per_position': per_pos, 'binned': binned, 'corr': corr}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Reveal trajectory analysis — the PUMA-specific failure diagnostic
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@torch.no_grad()
def analyse_reveal_patterns(model, tokenizer, bucket, max_len,
                             K=REVEAL_K_DEFAULT, tau=PUMA_TAU, device=None):
    """
    For a given test bucket (typically extreme class like chain>=24), run the
    PUMA forward process and aggregate reveal behavior by structural role.

    The core finding this produces: on extreme examples, PUMA defers (or never
    reaches) specific structural positions — these are the positions the model
    cannot learn because training signal is starved.

    Returns:
        per_position_reveal: [ANS_LEN, K] fraction of examples where position j
                             is still masked at START of stage k (i.e., not yet revealed)
        by_role: {role_name: [K] array, mean still-masked fraction per stage for positions of that role}
                 role taken from dep_ctx (g/k/p_above_g/p_above_k/p_above_p/p_bottom/carry_out)
        by_chain_position: [max_chain_seen, K] — for positions within the longest chain
                          of each example, still-masked fraction indexed by "distance from chain bottom"
        never_revealed: [ANS_LEN] fraction of examples where position j is still
                        masked at the end of stage K
        representative_traces: 3 example traces (reveal_stage arrays with annotation)
    """
    if device is None: device = DEVICE
    if bucket['n'] == 0:
        return {'n': 0}

    traj = simulate_reveal_trajectory(
        model, tokenizer, bucket['ids'], bucket['ans_starts'], ANS_LEN,
        blank_masks=None, K=K, tau=tau, device=device)

    rs = traj['reveal_stage']           # [N, ANS_LEN]
    smm = traj['still_masked_start']    # [N, K+1, ANS_LEN]
    N = bucket['n']

    # (1) Per-position masked-fraction over stages
    # smm[:, k, j] = True if position j still masked at start of stage k
    per_position_reveal = smm[:, :K, :].float().mean(dim=0).T.tolist()  # [ANS_LEN, K]

    # (2) Aggregate by dependency role
    dep_role_names = ['g', 'k', 'p_above_g', 'p_above_k', 'p_above_p', 'p_bottom', 'carry_out']
    by_role_raw = {r: [] for r in dep_role_names}
    for i, m in enumerate(bucket['metas']):
        dc = m['dep_ctx']
        for j, role in enumerate(dc):
            if role in by_role_raw:
                by_role_raw[role].append(smm[i, :K, j].float())
    by_role = {}
    for r, xs in by_role_raw.items():
        if xs:
            by_role[r] = {
                'still_masked_per_stage': torch.stack(xs).mean(dim=0).tolist(),
                'n_positions': len(xs),
            }

    # (3) By position-within-longest-chain (the core extreme-case view).
    # For each example, find its longest propagate chain; for each chain position
    # (0 = chain bottom, LSB side; L-1 = chain top, MSB side), record the reveal stage.
    max_chain_rank = 0
    chain_acc = defaultdict(list)  # chain_rank → list of (reveal_stage_value, still_masked_per_stage array)
    for i, m in enumerate(bucket['metas']):
        gkp_digit = m['chain_stats']['gkp']  # digits LSB→MSB
        # Find the longest p-run
        best = (-1, 0, 0)  # (length, start, end_exclusive)
        cur_s, cur_l = -1, 0
        for d, g in enumerate(gkp_digit):
            if g == 'p':
                if cur_s == -1: cur_s, cur_l = d, 1
                else: cur_l += 1
            else:
                if cur_l > best[0]: best = (cur_l, cur_s, cur_s + cur_l)
                cur_s, cur_l = -1, 0
        if cur_l > best[0]: best = (cur_l, cur_s, cur_s + cur_l)
        if best[0] <= 0: continue
        L_chain, s_d, e_d = best
        max_chain_rank = max(max_chain_rank, L_chain)
        # Map digit indices (LSB=0) to answer positions: answer[ND - d] is digit d (ND - 0 = ND = LSB, ND-(ND-1)=1 = near MSB)
        for rank, d in enumerate(range(s_d, e_d)):   # rank 0 = bottom of chain (LSB side)
            aj = ND - d
            if 0 <= aj < ANS_LEN:
                chain_acc[rank].append(smm[i, :K, aj].float())
    by_chain_position = []
    for rank in range(max_chain_rank):
        xs = chain_acc.get(rank, [])
        if xs:
            by_chain_position.append({
                'rank': rank,
                'still_masked_per_stage': torch.stack(xs).mean(dim=0).tolist(),
                'n': len(xs),
            })

    # (4) Never-revealed fraction per position
    never = (rs >= K).float().mean(dim=0).tolist()  # [ANS_LEN]

    # (5) Representative traces — 3 examples spanning short/medium/long max_chain
    order_by_chain = sorted(range(N), key=lambda i: bucket['metas'][i]['chain_stats']['max_chain_len'])
    picks = [order_by_chain[len(order_by_chain) // 6],
             order_by_chain[len(order_by_chain) // 2],
             order_by_chain[-1]] if N >= 3 else list(range(N))
    reps = []
    for i in picks:
        m = bucket['metas'][i]
        reps.append({
            'a': m['a'], 'b': m['b'],
            'max_chain': m['chain_stats']['max_chain_len'],
            'n_propagate': m['chain_stats']['n_propagate'],
            'gkp': m['gkp_at_pos'],
            'dep_ctx': m['dep_ctx'],
            'reveal_stage': rs[i].tolist(),
        })

    return {
        'n': N, 'K': K, 'tau': tau,
        'per_position_reveal': per_position_reveal,
        'by_role': by_role,
        'by_chain_position': by_chain_position,
        'never_revealed': never,
        'representative_traces': reps,
    }


def analyse_error_localization(per_sample):
    """Where in the chain do errors occur?"""
    cats = defaultdict(int); total_errs = 0
    for r in per_sample:
        if r['correct']: continue
        gkp = r['gkp_at_pos']; dep = r['dep_ctx']
        for j in r['error_positions']:
            if j >= len(gkp): continue
            total_errs += 1
            if gkp[j] == 'carry_out': cats['overflow'] += 1
            elif gkp[j] in ('g', 'k'): cats['gk_position'] += 1
            elif dep[j] == 'p_bottom': cats['p_chain_bottom'] += 1
            elif dep[j] == 'p_above_g': cats['p_above_g'] += 1
            elif dep[j] == 'p_above_k': cats['p_above_k'] += 1
            elif dep[j] == 'p_above_p': cats['p_chain_interior'] += 1
    if total_errs == 0: return {'total_errors': 0}
    return {'total_errors': total_errs,
            **{k: v/total_errs for k, v in cats.items()}}


@torch.no_grad()
def analyse_confidence_calibration(model, tokenizer, test_samples, max_len, device=None):
    """Per chain-length bin: mean confidence for correct vs wrong predictions."""
    if device is None: device = DEVICE
    ps = gen_eval_with_stats(model, tokenizer, test_samples, max_len,
                             decode_policy='confidence', device=device)
    mask_id = tokenizer.special_ids['mask']
    ids_all, ans_all = encode_samples(test_samples, tokenizer, max_len)
    ids_all, ans_all = ids_all.to(device), ans_all.to(device)
    _arange = torch.arange(ANS_LEN, device=device)

    confs_per = []
    for st in range(0, len(test_samples), 128):
        en = min(st+128, len(test_samples))
        ids, ans = ids_all[st:en], ans_all[st:en]; B = ids.shape[0]
        ans_pos = (ans.unsqueeze(1) + _arange).clamp(max=ids.shape[1]-1)
        bi = torch.arange(B, device=device).unsqueeze(1).expand_as(ans_pos)
        xm = ids.clone(); xm[bi, ans_pos] = mask_id
        logits = model(xm); al = logits[bi, ans_pos]
        cl = al.clone(); cl[:, :, mask_id] = -float('inf')
        confs_per.extend(F.softmax(cl, dim=-1).max(dim=-1).values.mean(dim=1).cpu().tolist())

    def _cl_bin(cl):
        if cl <= 2: return 'cl<=2'
        if cl <= 4: return 'cl<=4'
        return 'cl>=5'

    bins = defaultdict(lambda: {'correct': [], 'wrong': []})
    for i, r in enumerate(ps):
        cl = r['chain_stats']['max_chain_len']
        bn = _cl_bin(cl)
        (bins[bn]['correct'] if r['correct'] else bins[bn]['wrong']).append(
            confs_per[i] if i < len(confs_per) else 0.5)

    result = {}
    for bn, data in sorted(bins.items()):
        mc = sum(data['correct'])/len(data['correct']) if data['correct'] else None
        mw = sum(data['wrong'])/len(data['wrong']) if data['wrong'] else None
        oc = sum(1 for c in data['wrong'] if c > 0.8)/len(data['wrong']) if data['wrong'] else None
        result[bn] = {'mean_conf_correct': mc, 'mean_conf_wrong': mw,
                      'overconfident_wrong': oc, 'n_correct': len(data['correct']),
                      'n_wrong': len(data['wrong'])}
    return result


def _quick_gen(model, tokenizer, test_samples, max_len, decode_policy='confidence',
               n=None, device=None):
    if n is None: n = GEN_EVAL_N
    if device is None: device = DEVICE
    subset = test_samples[:n]; mask_id = tokenizer.special_ids['mask']
    pad_id = tokenizer.special_ids['pad']
    lsb_policy = 'r2l'
    results = []
    for st in range(0, len(subset), 128):
        batch = subset[st:st+128]; B = len(batch)
        penc = [tokenizer.encode(s.split('=')[0]+'=') for s in batch]
        pm = max(len(p) for p in penc)
        pids = torch.full((B, pm), pad_id, dtype=torch.long)
        for i, e in enumerate(penc): pids[i, :len(e)] = torch.tensor(e)
        policy = lsb_policy if decode_policy == 'lsb' else decode_policy
        gen, _, _ = generate_diffusion(model, pids, ANS_LEN, mask_id,
                                       policy=policy, greedy=True, device=device)
        pred = gen[:, pm:pm+ANS_LEN]
        for i in range(B):
            ps = tokenizer.decode(pred[i].cpu().tolist()); gs = get_answer(batch[i])
            results.append({'correct': ps==gs})
    return {'accuracy': sum(r['correct'] for r in results)/max(len(results),1), 'n': len(results)}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Training stratum construction
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Chain-length stratum boundaries (upper-exclusive).
# Natural uniform ND=32 distribution puts ~0 samples at chain≥24, so we use 4
# strata instead of 5 and let the extreme bucket (chain_16plus) catch the long
# tail — that's where the coverage-deficit prediction is most testable without
# requiring constructive (synthetic) training data.
STRATUM_BOUNDS = [(0, 4), (4, 8), (8, 16), (16, ND + 1)]
STRATUM_NAMES = ['chain_0_3', 'chain_4_7', 'chain_8_15', 'chain_16plus']


def _chain_to_stratum(max_chain):
    for i, (lo, hi) in enumerate(STRATUM_BOUNDS):
        if lo <= max_chain < hi:
            return i
    return len(STRATUM_BOUNDS) - 1


def build_training_strata(train_samples):
    """Compute stratum id per training sample (by max_chain_len).
    Returns: long tensor [N], names list[str], counts list[int]
    """
    strata = [_chain_to_stratum(_max_chain_len(*_parse_operands(s))) for s in train_samples]
    counts = [strata.count(i) for i in range(len(STRATUM_NAMES))]
    return torch.tensor(strata, dtype=torch.long), STRATUM_NAMES, counts


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Training wrapper (uses unified train_diffusion)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def train_model(mask_type, tokenizer, train_samples, suite, max_len,
                max_iters=None, init_state=None, device=None):
    """Wrapper around train_diffusion for addition experiment.

    Passes per-sample stratum tensor to train_diffusion so that training-time
    masked-token loss is logged per chain-length stratum. This is the
    training-side diagnostic: the trajectory shows which strata PUMA fails
    to improve on over iterations.

    `suite` is used for eval probes on natural distribution; eval_fn does
    NOT compute stratified test-gen-acc (that analysis is done post-hoc after
    training using the held-out constructed buckets).
    """
    if device is None: device = DEVICE
    if max_iters is None: max_iters = MAX_ITERS

    train_ids, train_ans = encode_samples(train_samples, tokenizer, max_len)
    train_ids, train_ans = train_ids.to(device), train_ans.to(device)

    # Stratum tensor — drives training-side loss trajectory
    sample_strata, stratum_names, stratum_counts = build_training_strata(train_samples)
    print(f"  Training strata counts: " +
          ', '.join(f"{n}={c}" for n, c in zip(stratum_names, stratum_counts)))

    natural_samples = suite['natural']['samples']

    # ── Reveal-vs-reasoning tau diagnostic setup (PUMA only) ──
    # Track a fixed subset of extreme training samples (long carry chains) and,
    # at late-training checkpoints where K has grown large, compare PUMA's
    # confidence-induced reveal order against the canonical reasoning order (r2l).
    # This is the central zebra-inspired diagnostic — does what PUMA *learns* to
    # unmask first actually align with the direction that logical reasoning would
    # proceed?
    reveal_tracked_ids = None
    reveal_tracked_ans = None
    reveal_reasoning_order = None
    reveal_blanks = None
    reveal_tracked_strata = None
    if mask_type == 'puma':
        # Per-stratum tracked indices: gather up to per_stratum_cap samples
        # from each chain-length stratum. This lets us decompose the τ
        # trajectory — e.g. chain_0_3 should trend toward τ ≈ +1 (r2l learned)
        # while chain_24plus should remain near τ ≈ 0 (extreme case misalign).
        per_stratum_cap = max(REVEAL_TAU_N_TRACKED // len(STRATUM_NAMES), 10)
        tracked_by_stratum = {sn: [] for sn in STRATUM_NAMES}
        for i, s in enumerate(train_samples):
            cl = _max_chain_len(*_parse_operands(s))
            si = _chain_to_stratum(cl)
            sn = STRATUM_NAMES[si]
            if len(tracked_by_stratum[sn]) < per_stratum_cap:
                tracked_by_stratum[sn].append((i, si))
        tracked_flat = [t for v in tracked_by_stratum.values() for t in v]
        strata_counts = {sn: len(v) for sn, v in tracked_by_stratum.items()}
        if sum(strata_counts.values()) >= 30:
            tracked_idx = [t[0] for t in tracked_flat]
            tracked_strata = [t[1] for t in tracked_flat]
            reveal_tracked_ids = train_ids[tracked_idx]
            reveal_tracked_ans = train_ans[tracked_idx]
            reveal_tracked_strata = torch.tensor(tracked_strata, dtype=torch.long)
            N_tr = len(tracked_idx)
            # Reasoning order for addition: r2l — LSB (answer position ANS_LEN-1)
            # decoded first, MSB (answer position 0) last. Rank[j] = ANS_LEN-1-j.
            reveal_reasoning_order = torch.arange(
                ANS_LEN - 1, -1, -1, dtype=torch.long).unsqueeze(0).expand(
                N_tr, -1).contiguous()
            reveal_blanks = torch.ones(N_tr, ANS_LEN, dtype=torch.bool)
            print(f"  Reveal-τ tracking (stratified): total={N_tr}, by stratum: " +
                  ', '.join(f"{sn}={c}" for sn, c in strata_counts.items() if c > 0))
        else:
            print(f"  Reveal-τ tracking: SKIPPED "
                  f"(total tracked < 30: {strata_counts})")

    # PUMA K schedule
    k_sched = None
    if mask_type == 'puma':
        if PUMA_K_START is not None and PUMA_K_END is not None:
            k_step = PUMA_K_STEP or 3
            if PUMA_K_EVERY is not None:
                k_every = PUMA_K_EVERY
            else:
                n_increments = max(1, (PUMA_K_END - PUMA_K_START) // k_step)
                k_every = max(1000, (max_iters // 3) // n_increments)
            k_sched = puma_k_step(PUMA_K_START, PUMA_K_END, k_step, k_every)
            final_k = k_sched(max_iters)
            print(f"  PUMA K: step {PUMA_K_START}→{final_k} (+{k_step} every {k_every//1000}k, cap={PUMA_K_END})")
        else:
            k_sched = puma_k_fixed(PUMA_K)
            print(f"  PUMA K: fixed {PUMA_K}")

    K_final_for_tau = k_sched(max_iters) if k_sched else None

    # Eval callback — probe on natural + optional gen eval + reveal-τ diagnostic
    def eval_fn(model, it, tg):
        probe = probe_per_position(model, tokenizer, natural_samples, max_len, device)
        dc = probe.get('dep_context', {})
        parts = [f"{c}={dc[c]['acc']:.2f}" for c in ['g','k','p_above_g','p_above_k','p_above_p']
                 if c in dc]
        print(f"    [eval it {it}] loss={probe['overall_loss']:.4f} "
              f"acc={probe['overall_acc']:.4f} {' '.join(parts)}")
        if it > 0 and it % GEN_EVAL_EVERY == 0:
            r = _quick_gen(model, tokenizer, natural_samples, max_len, 'confidence', device=device)
            probe['gen_acc_confidence'] = r['accuracy']
            print(f"      [gen] natural confidence={r['accuracy']:.3f}")

        # Reveal-vs-reasoning tau (PUMA only, past K threshold, on fixed cadence)
        if (reveal_tracked_ids is not None and it > 0
                and it % REVEAL_TAU_EVERY == 0 and K_final_for_tau is not None):
            K_cur = k_sched(it)
            if K_cur >= K_final_for_tau * REVEAL_TAU_K_THRESHOLD_FRAC:
                traj = simulate_reveal_trajectory(
                    model, tokenizer, reveal_tracked_ids, reveal_tracked_ans, ANS_LEN,
                    blank_masks=reveal_blanks, K=K_cur, tau=PUMA_TAU,
                    batch_size=64, device=device)
                taus = compute_reveal_vs_order_tau(
                    traj['reveal_stage'], reveal_reasoning_order, reveal_blanks)
                import numpy as _np
                valid_mask = ~_np.isnan(taus)
                stratum_np = reveal_tracked_strata.cpu().numpy()
                # Per-stratum decomposition
                per_stratum = {}
                for si, sn in enumerate(STRATUM_NAMES):
                    m = valid_mask & (stratum_np == si)
                    vs = taus[m]
                    if len(vs) >= 3:
                        per_stratum[sn] = {
                            'n': int(len(vs)),
                            'mean': float(vs.mean()),
                            'q25': float(_np.percentile(vs, 25)),
                            'q50': float(_np.percentile(vs, 50)),
                            'q75': float(_np.percentile(vs, 75)),
                        }
                valid = taus[valid_mask]
                if len(valid) > 0:
                    probe['reveal_tau'] = {
                        'K_cur': K_cur,
                        'n': int(len(valid)),
                        'mean': float(valid.mean()),
                        'q25': float(_np.percentile(valid, 25)),
                        'q50': float(_np.percentile(valid, 50)),
                        'q75': float(_np.percentile(valid, 75)),
                        'min': float(valid.min()),
                        'max': float(valid.max()),
                        'per_stratum': per_stratum,
                    }
                    strata_str = ' | '.join(
                        f"{sn}={d['q50']:+.2f}(n{d['n']})"
                        for sn, d in per_stratum.items())
                    print(f"      [reveal-τ] K={K_cur} overall={probe['reveal_tau']['q50']:+.3f} "
                          f"| {strata_str}")
        return probe

    model, dynamics = train_diffusion(
        train_ids=train_ids, train_ans=train_ans, ans_len=ANS_LEN, tokenizer=tokenizer,
        mask_type=mask_type, blank_masks=None,
        puma_tau=PUMA_TAU, puma_k_schedule=k_sched,
        n_layer=N_LAYER, n_head=N_HEAD, n_embd=N_EMBD, dropout=DROPOUT, pos_enc=POS_ENC,
        max_iters=max_iters, batch_size=BATCH_SIZE,
        lr=LR, min_lr=MIN_LR, warmup_iters=WARMUP_ITERS,
        grad_clip=GRAD_CLIP, weight_decay=WEIGHT_DECAY, ema_decay=EMA_DECAY,
        eval_fn=eval_fn, eval_every=EVAL_EVERY, log_every=LOG_EVERY,
        patience=PATIENCE,
        sample_strata=sample_strata, stratum_names=stratum_names,
        init_state=init_state, device=device,
        use_amp=False if NO_AMP else None,
    )
    return model, dynamics


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Figures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def make_figures(all_dyn, all_final):
    figs = {}; labels = _pos_labels()
    cmap = plt.cm.coolwarm
    def pc(j): return cmap(1.0 - j/(ANS_LEN-1))

    # Fig 1: Per-position accuracy over training
    nc = len(MASK_TYPES)
    fig, axes = plt.subplots(1, nc, figsize=(6*nc, 5), squeeze=False); axes = axes[0]
    for ai, mt in enumerate(MASK_TYPES):
        dyn = all_dyn.get(mt)
        if not dyn: continue
        ax = axes[ai]; xs = [c['iter'] for c in dyn['checkpoints']]
        for j in range(ANS_LEN):
            ys = [c['pos_acc'][j] for c in dyn['checkpoints'] if 'pos_acc' in c]
            if ys: ax.plot(xs[:len(ys)], ys, '-', color=pc(j), label=labels[j], lw=1.2)
        ax.set_xlabel('Iteration'); ax.set_ylabel('Accuracy'); ax.set_ylim(-0.05, 1.05)
        ax.set_title(mt); ax.legend(fontsize=4, ncol=4); ax.grid(alpha=0.3)
    fig.suptitle('Per-Position Probe Accuracy', y=1.02); fig.tight_layout()
    figs['pos_acc'] = fig

    # Fig 2: Chain sweep comparison
    sweep_lengths = [2, 3, 4, 6, 8, 12]
    if ND >= 24: sweep_lengths += [16, 20]
    if ND >= 32: sweep_lengths += [24, 28]
    sweep_lengths = [cl for cl in sweep_lengths if cl <= ND]
    for dp in DECODE_POLICIES:
        fig, ax = plt.subplots(figsize=(10, 5))
        for mt, col, mk in [('random', '#3498db', 'o'), ('puma', '#8e44ad', 's')]:
            accs = []
            for cl in sweep_lengths:
                r = all_final.get(f'{mt}_chain_sweep_{cl}_{dp}')
                accs.append(r['accuracy'] if r else None)
            valid = [(cl, a) for cl, a in zip(sweep_lengths, accs) if a is not None]
            if valid:
                ax.plot([v[0] for v in valid], [v[1] for v in valid], f'-{mk}',
                        color=col, label=mt, lw=2, markersize=8)
        ax.set_xlabel('Min carry chain length'); ax.set_ylabel('Accuracy')
        ax.set_title(f'Chain Length Sweep — {dp} decode'); ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout(); figs[f'chain_sweep_{dp}'] = fig

    # Fig 3: Carry rarity × accuracy gap
    fig, ax = plt.subplots(figsize=(8, 5))
    for mt, color, mk in [('random', '#3498db', 'o'), ('puma', '#8e44ad', 's')]:
        r = all_final.get(f'{mt}_carry_rarity')
        if not r: continue
        brs = [p['base_rate'] for p in r['per_position'] if p['acc_gap'] is not None]
        gaps = [p['acc_gap'] for p in r['per_position'] if p['acc_gap'] is not None]
        ax.scatter(brs, gaps, c=color, marker=mk, s=50, alpha=0.7,
                   label=f"{mt} (r={r['corr']:.2f})" if r['corr'] else mt)
    ax.axhline(0, color='gray', ls=':'); ax.set_xlabel('Carry-in base rate')
    ax.set_ylabel('acc(c=0) - acc(c=1)'); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); figs['carry_rarity'] = fig

    # Fig 4: Stratum loss trajectory — core training-side diagnostic
    # Shows, per chain-length stratum, the mean masked-token loss over training.
    # The central finding: PUMA's chain_24plus stratum plateaus while Random's decreases.
    mt_with_strat = [mt for mt in MASK_TYPES
                     if all_dyn.get(mt, {}).get('stratified_loss')]
    if mt_with_strat:
        nm = len(mt_with_strat)
        fig, axes = plt.subplots(1, nm, figsize=(7 * nm, 5), squeeze=False)
        axes = axes[0]
        s_cmap = plt.cm.viridis
        for ai, mt in enumerate(mt_with_strat):
            dyn = all_dyn[mt]
            sl = dyn['stratified_loss']
            names = dyn.get('stratum_names', [])
            S = len(names) if names else (len(sl[0]['per_stratum_loss']) if sl else 0)
            ax = axes[ai]
            xs = [e['iter'] for e in sl]
            for si in range(S):
                ys = [e['per_stratum_loss'][si] if e['per_stratum_n'][si] > 0 else None
                      for e in sl]
                label = names[si] if si < len(names) else f's{si}'
                valid_xy = [(x, y) for x, y in zip(xs, ys) if y is not None]
                if valid_xy:
                    ax.plot([p[0] for p in valid_xy], [p[1] for p in valid_xy],
                            '-', color=s_cmap(si / max(S - 1, 1)), label=label, lw=1.5)
            ax.set_xlabel('Iteration'); ax.set_ylabel('Masked-token loss (training)')
            ax.set_title(f'{mt}: stratum loss trajectory'); ax.set_yscale('log')
            ax.legend(fontsize=8); ax.grid(alpha=0.3)
        fig.suptitle('Training-time masked-token loss, stratified by chain length', y=1.02)
        fig.tight_layout(); figs['stratum_loss'] = fig

    # Fig 5: Reveal-trajectory heatmap on extreme bucket — PUMA failure mechanism
    # x-axis = PUMA stage, y-axis = chain-position rank (0 = chain bottom / LSB side,
    # higher = chain top / MSB side). Cell value = fraction of examples where that
    # chain-position is STILL MASKED at the start of that stage. High values at late
    # stages = positions the forward process fails to train on.
    extreme_key = None
    for k in ['chain_24', 'chain_20', 'chain_16']:
        if any(f'{mt}_reveal_{k}' in all_final for mt in MASK_TYPES):
            extreme_key = k; break
    if extreme_key is not None:
        mts_with_rev = [mt for mt in MASK_TYPES
                        if f'{mt}_reveal_{extreme_key}' in all_final]
        nm = len(mts_with_rev)
        if nm > 0:
            fig, axes = plt.subplots(1, nm, figsize=(6 * nm, 5), squeeze=False)
            axes = axes[0]
            for ai, mt in enumerate(mts_with_rev):
                rev = all_final[f'{mt}_reveal_{extreme_key}']
                bcp = rev.get('by_chain_position', [])
                if not bcp: continue
                K = rev.get('K', REVEAL_K_DEFAULT)
                mat = [row['still_masked_per_stage'] for row in bcp]
                import numpy as np
                arr = np.array(mat)  # [chain_ranks, K]
                ax = axes[ai]
                im = ax.imshow(arr, aspect='auto', origin='lower',
                               cmap='Reds', vmin=0, vmax=1, interpolation='nearest')
                ax.set_xlabel(f'PUMA stage (K={K})')
                ax.set_ylabel('Chain position rank (0 = LSB side of chain)')
                ax.set_title(f'{mt}: still-masked fraction ({extreme_key}, N={rev["n"]})')
                plt.colorbar(im, ax=ax, label='still masked')
            fig.suptitle(f'Reveal trajectory — {extreme_key} extreme bucket', y=1.02)
            fig.tight_layout(); figs[f'reveal_{extreme_key}'] = fig

    # Fig 6: Never-revealed fraction per answer position (PUMA failure map)
    if extreme_key is not None:
        fig, ax = plt.subplots(figsize=(10, 4))
        for mt, col in [('random', '#3498db'), ('puma', '#8e44ad')]:
            rev = all_final.get(f'{mt}_reveal_{extreme_key}')
            if not rev: continue
            ys = rev.get('never_revealed', [])
            if ys:
                ax.plot(range(len(ys)), ys, '-o', color=col, label=mt,
                        markersize=4, lw=1.5)
        ax.set_xlabel('Answer position (0=MSB, ANS_LEN-1=LSB)')
        ax.set_ylabel(f'Fraction never-revealed in {REVEAL_K_DEFAULT} stages')
        ax.set_title(f'Never-revealed map ({extreme_key})')
        ax.set_ylim(-0.02, 1.02); ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout(); figs[f'never_revealed_{extreme_key}'] = fig

    # Fig 7: Reveal-vs-reasoning Kendall τ trajectory (PUMA only)
    # Central training-time diagnostic. Overlays overall trajectory (shaded IQR)
    # with per-stratum median trajectories, so the paper argument "PUMA learns
    # r2l order for easy chains but fails to align for chain≥24" is a single
    # figure: chain_0_3 line converges to +1, chain_24plus stays near 0.
    puma_dyn = all_dyn.get('puma', {})
    tau_pts = [c for c in puma_dyn.get('checkpoints', []) if 'reveal_tau' in c]
    if tau_pts:
        fig, ax = plt.subplots(figsize=(10, 5))
        # Overall (shaded IQR)
        xs = [c['iter'] for c in tau_pts]
        mids = [c['reveal_tau']['q50'] for c in tau_pts]
        q25s = [c['reveal_tau']['q25'] for c in tau_pts]
        q75s = [c['reveal_tau']['q75'] for c in tau_pts]
        ax.fill_between(xs, q25s, q75s, alpha=0.2, color='#8e44ad', label='overall IQR')
        ax.plot(xs, mids, '--', color='#5e2d6e', lw=1.5, label='overall median')
        # Per-stratum trajectories
        s_cmap = plt.cm.plasma
        for si, sn in enumerate(STRATUM_NAMES):
            xs_s, mids_s = [], []
            for c in tau_pts:
                ps = c['reveal_tau'].get('per_stratum', {}).get(sn)
                if ps is not None:
                    xs_s.append(c['iter']); mids_s.append(ps['q50'])
            if xs_s:
                ax.plot(xs_s, mids_s, '-o',
                        color=s_cmap(si / max(len(STRATUM_NAMES) - 1, 1)),
                        label=sn, lw=2, markersize=5)
        ax.axhline(0, color='gray', ls=':', alpha=0.7, label='τ=0')
        ax.axhline(1, color='green', ls='--', alpha=0.4, label='τ=+1 (r2l)')
        ax.axhline(-1, color='red', ls='--', alpha=0.4, label='τ=−1 (reversed)')
        ax.set_xlabel('Training iteration')
        ax.set_ylabel('Kendall τ (reveal vs r2l reasoning order)')
        ax.set_title('PUMA reveal-order alignment, stratified by chain length')
        ax.set_ylim(-1.05, 1.05); ax.legend(loc='best', fontsize=8); ax.grid(alpha=0.3)
        fig.tight_layout(); figs['reveal_tau_stratified'] = fig

    return figs


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Run
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run(tag=''):
    exp_name = f"{EXP_NAME}_{tag}" if tag else EXP_NAME
    mount_drive()
    torch.manual_seed(SEED); random.seed(SEED)
    tok = build_tok()

    print(f"\n{'='*70}")
    print(f"  Addition ND={ND} | masks={MASK_TYPES} | decode={DECODE_POLICIES}")
    print(f"  N_TRAIN={N_TRAIN} N_TEST={N_TEST} MAX_ITERS={MAX_ITERS}")
    print(f"  arch: {N_LAYER}L/{N_HEAD}H/{N_EMBD}D | data: {DATA_MODE}")
    print(f"{'='*70}\n")

    train_data = gen_train_data(N_TRAIN, seed=SEED)
    max_len = max(len(tok.encode(s)) for s in train_data)

    # Unified test suite — all evals/analyses slice from here
    suite = build_test_suite(tok, max_len, seed=SEED + 1000)
    natural_samples = suite['natural']['samples']

    # "Heavy" subset = natural filtered by max_chain ≥ 3 (no fresh generation)
    heavy_bucket = filter_natural(suite, lambda m: m['chain_stats']['max_chain_len'] >= 3)

    all_dyn = {}; all_final = {}
    saved_states = {}

    # ── Main training ──
    for mt in MASK_TYPES:
        print(f"\n{'━'*60}\n▶ {mt}\n{'━'*60}")
        m, d = train_model(mt, tok, train_data, suite, max_len)
        all_dyn[mt] = d
        saved_states[mt] = {k: v.cpu().clone() for k, v in m.state_dict().items()}
        save_checkpoint(exp_name, saved_states[mt], tag=mt)

        # Standard + heavy (both from suite / filter)
        for dp in DECODE_POLICIES:
            for name, data in [('standard', natural_samples),
                               ('heavy', heavy_bucket['samples'])]:
                if not data: continue
                ps = gen_eval_with_stats(m, tok, data, max_len, decode_policy=dp, device=DEVICE)
                acc = sum(r['correct'] for r in ps) / len(ps)
                all_final[f'{mt}_{name}_{dp}'] = {
                    'accuracy': acc, 'n': len(ps), 'stratified': stratify_results(ps)}
                print(f"    {name} {dp}: {acc:.4f}  (n={len(ps)})")

        # Corner cases from suite
        for cat in ['msb_chain', 'full_propagate']:
            bucket = suite['constructed'].get(cat)
            if not bucket: continue
            for dp in DECODE_POLICIES:
                ps = gen_eval_with_stats(m, tok, bucket['samples'], max_len,
                                          decode_policy=dp, device=DEVICE)
                acc = sum(r['correct'] for r in ps) / len(ps)
                all_final[f'{mt}_corner_{cat}_{dp}'] = {'accuracy': acc, 'n': len(ps)}
                print(f"    corner/{cat} {dp}: {acc:.4f}")

        # Chain length sweep from suite
        print(f"  Chain length sweep...")
        sweep_keys = sorted([k for k in suite['constructed'] if k.startswith('chain_')],
                            key=lambda k: int(k.split('_')[1]))
        for key in sweep_keys:
            min_cl = int(key.split('_')[1])
            bucket = suite['constructed'][key]
            for dp in DECODE_POLICIES:
                ps = gen_eval_with_stats(m, tok, bucket['samples'], max_len,
                                          decode_policy=dp, device=DEVICE)
                acc = sum(r['correct'] for r in ps) / len(ps)
                all_final[f'{mt}_chain_sweep_{min_cl}_{dp}'] = {'accuracy': acc, 'n': len(ps)}
                print(f"    chain>={min_cl:2d} {dp}: {acc:.4f}")

        # Counterfactual (from suite)
        cfr = eval_counterfactual(m, tok, suite['counterfactual'], max_len, device=DEVICE)
        all_final[f'{mt}_cf'] = cfr
        print(f"    CF: flip={cfr['prediction_flip_rate']:.3f} "
              f"acc(c=0)={cfr['acc_carry_in_0']:.3f} acc(c=1)={cfr['acc_carry_in_1']:.3f}")

        # Error localization per chain bucket (suite slice, no regeneration)
        for min_cl in [4, 8, 12, 16, 20]:
            key = f'chain_{min_cl}'
            if key not in suite['constructed']: continue
            ps = gen_eval_with_stats(m, tok, suite['constructed'][key]['samples'],
                                      max_len, decode_policy='confidence', device=DEVICE)
            all_final[f'{mt}_error_loc_{min_cl}'] = analyse_error_localization(ps)

        # Carry rarity (natural)
        ps_conf = gen_eval_with_stats(m, tok, natural_samples, max_len,
                                       decode_policy='confidence', device=DEVICE)
        rarity = analyse_carry_rarity(ps_conf, natural_samples)
        all_final[f'{mt}_carry_rarity'] = rarity
        corr_str = f"{rarity['corr']:.3f}" if rarity.get('corr') is not None else "N/A"
        print(f"    Rarity corr: {corr_str}")

        # Reveal trajectory on extreme bucket — PUMA-specific failure diagnostic
        # (for both mask types: Random gives baseline shape to contrast against)
        extreme_key = None
        for k in ['chain_24', 'chain_20', 'chain_16']:
            if k in suite['constructed']:
                extreme_key = k; break
        if extreme_key is not None:
            rev = analyse_reveal_patterns(
                m, tok, suite['constructed'][extreme_key], max_len,
                K=REVEAL_K_DEFAULT, tau=PUMA_TAU, device=DEVICE)
            all_final[f'{mt}_reveal_{extreme_key}'] = rev
            nv = sum(1 for v in rev.get('never_revealed', []) if v > 0.5)
            print(f"    Reveal on {extreme_key}: {rev.get('n', 0)} examples, "
                  f"{nv}/{ANS_LEN} positions never-revealed >50% of time")

        # Confidence calibration on heavy (from filter)
        if heavy_bucket['samples']:
            cal = analyse_confidence_calibration(m, tok, heavy_bucket['samples'],
                                                  max_len, device=DEVICE)
            all_final[f'{mt}_calibration'] = cal

        del m; torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── Continuation training ──
    args = parse_args()
    if not getattr(args, 'no_continuation', False) and len(MASK_TYPES) >= 2:
        cont_pairs = []
        if 'random' in saved_states and 'puma' in MASK_TYPES:
            cont_pairs.append(('random', 'puma'))
        if 'puma' in saved_states and 'random' in MASK_TYPES:
            cont_pairs.append(('puma', 'random'))

        for src, tgt in cont_pairs:
            label = f'{src}_to_{tgt}'
            print(f"\n{'━'*60}\n▶ Continuation: {label} ({CONTINUATION_ITERS} iters)\n{'━'*60}")
            m, d = train_model(tgt, tok, train_data, suite, max_len,
                               max_iters=CONTINUATION_ITERS,
                               init_state=saved_states[src])
            all_dyn[label] = d

            # Same eval as main, from suite
            for dp in DECODE_POLICIES:
                for name, data in [('standard', natural_samples),
                                   ('heavy', heavy_bucket['samples'])]:
                    if not data: continue
                    ps = gen_eval_with_stats(m, tok, data, max_len, decode_policy=dp, device=DEVICE)
                    acc = sum(r['correct'] for r in ps) / len(ps)
                    all_final[f'{label}_{name}_{dp}'] = {'accuracy': acc, 'n': len(ps)}
                    print(f"    {name} {dp}: {acc:.4f}")

            for cat in ['msb_chain', 'full_propagate']:
                bucket = suite['constructed'].get(cat)
                if not bucket: continue
                for dp in DECODE_POLICIES:
                    ps = gen_eval_with_stats(m, tok, bucket['samples'], max_len,
                                              decode_policy=dp, device=DEVICE)
                    acc = sum(r['correct'] for r in ps) / len(ps)
                    all_final[f'{label}_corner_{cat}_{dp}'] = {'accuracy': acc, 'n': len(ps)}
                    print(f"    corner/{cat} {dp}: {acc:.4f}")

            for key in sweep_keys:
                min_cl = int(key.split('_')[1])
                bucket = suite['constructed'][key]
                for dp in DECODE_POLICIES:
                    ps = gen_eval_with_stats(m, tok, bucket['samples'], max_len,
                                              decode_policy=dp, device=DEVICE)
                    acc = sum(r['correct'] for r in ps) / len(ps)
                    all_final[f'{label}_chain_sweep_{min_cl}_{dp}'] = {
                        'accuracy': acc, 'n': len(ps)}
                    print(f"    chain>={min_cl:2d} {dp}: {acc:.4f}")

            del m; torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── Figures & save ──
    figs = make_figures(all_dyn, all_final)
    sd = {'config': {k: globals()[k] for k in
           ['ND','ANS_LEN','N_TRAIN','N_TEST','MAX_ITERS',
            'BATCH_SIZE','N_LAYER','N_HEAD','N_EMBD','MASK_TYPES','DECODE_POLICIES','DATA_MODE']}}
    for k, v in all_dyn.items():
        sd[f'dyn_{k}'] = {
            'checkpoints': v['checkpoints'],
            'train_loss': v['train_loss'],
            'stratified_loss': v.get('stratified_loss', []),
            'stratum_names': v.get('stratum_names', []),
        }
    for k, v in all_final.items():
        sd[f'final_{k}'] = v
    save_results(exp_name, sd, figures=figs)

    # Summary
    print(f"\n{'='*70}\n  SUMMARY\n{'='*70}")
    all_conditions = list(MASK_TYPES)
    if not getattr(args, 'no_continuation', False):
        for src, tgt in [('random', 'puma'), ('puma', 'random')]:
            if f'{src}_to_{tgt}' in all_dyn:
                all_conditions.append(f'{src}_to_{tgt}')

    print(f"\n  {'Test':<35s}", end='')
    for mt in all_conditions: print(f" {mt:>14s}", end='')
    print()
    for dp in DECODE_POLICIES:
        for tt in ['standard', 'heavy', 'corner_msb_chain', 'corner_full_propagate']:
            accs = [all_final.get(f'{mt}_{tt}_{dp}', {}).get('accuracy') for mt in all_conditions]
            if any(a is not None for a in accs):
                print(f"  {tt+'_'+dp:<35s}", end='')
                for a in accs: print(f" {a:>14.4f}" if a is not None else f" {'N/A':>14s}", end='')
                print()
        for min_cl in [4, 8, 12, 16, 20, 24, 28]:
            accs = [all_final.get(f'{mt}_chain_sweep_{min_cl}_{dp}', {}).get('accuracy')
                    for mt in all_conditions]
            if any(a is not None for a in accs):
                print(f"  {'chain>='+str(min_cl)+'_'+dp:<35s}", end='')
                for a in accs: print(f" {a:>14.4f}" if a is not None else f" {'N/A':>14s}", end='')
                print()

    return all_dyn, all_final


if __name__ == '__main__':
    args = parse_args()
    seeds = args.seeds if args.seeds else [SEED]
    for si, seed in enumerate(seeds):
        globals()['SEED'] = seed
        t = f"{args.tag}_s{seed}" if args.tag and len(seeds) > 1 else args.tag
        if len(seeds) > 1: print(f"\n{'#'*70}\n# Seed {seed} ({si+1}/{len(seeds)})\n{'#'*70}")
        run(tag=t)
