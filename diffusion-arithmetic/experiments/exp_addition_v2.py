"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Addition v2 — Carry Dependency Learning + PUMA Coverage Deficit
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Training: random vs puma (+ optional oracle_lsb, confidence)
  Decode:   confidence (model) vs lsb (oracle) vs random
  Analyses: GKP dependency, carry rarity × accuracy, PUMA coverage,
            corner cases, confidence cascade, counterfactual carry
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
    mount_drive, save_results, generate_ar, generate_diffusion,
    encode_samples, DEVICE,
)

EXP_NAME = 'exp_addition_v2'

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Config
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ND = 16; ANS_LEN = ND + 1
N_TRAIN = 10000; N_TEST = 1000; BATCH_SIZE = 200
MAX_EPOCHS = 5000; EVAL_EVERY = 100; LOG_EVERY = 50
GEN_EVAL_EVERY = 200; GEN_EVAL_N = 500
FORMATS = ['plain']; MASK_TYPES = ['random', 'puma']
DECODE_POLICIES = ['confidence', 'lsb']
N_LAYER = 2; N_HEAD = 2; N_EMBD = 128; DROPOUT = 0.2; POS_ENC = 'absolute'
LR = 1e-3; MIN_LR = 1e-4; WARMUP_EPOCHS = 10; GRAD_CLIP = 1.0
PUMA_TAU = 0.9; PUMA_K_START = 3; PUMA_K_END = ANS_LEN
SEED = 42; RUN_AR = False
DATA_MODE = 'natural'  # 'balanced' or 'natural'

# Curriculum: list of (label, schedule) where schedule = [(frac, mask_type), ...]
# frac = fraction of total epochs at which to switch TO this mask_type
# Example: [('puma2rand_50', [(0.0,'puma'),(0.5,'random')])]
CURRICULUM_MODES = [
    ('puma2rand_20', [(0.0, 'puma'), (0.20, 'random')]),
    ('puma2rand_50', [(0.0, 'puma'), (0.50, 'random')]),
    ('puma2rand_70', [(0.0, 'puma'), (0.70, 'random')]),
    ('interleaved',  'interleaved'),   # alternate every batch
]
RUN_CURRICULUM = True


def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--nd', type=int); p.add_argument('--n-train', type=int)
    p.add_argument('--n-test', type=int); p.add_argument('--epochs', type=int)
    p.add_argument('--batch-size', type=int); p.add_argument('--eval-every', type=int)
    p.add_argument('--gen-eval-every', type=int)
    p.add_argument('--n-layer', type=int); p.add_argument('--n-head', type=int)
    p.add_argument('--n-embd', type=int); p.add_argument('--dropout', type=float)
    p.add_argument('--puma-tau', type=float)
    p.add_argument('--puma-k-start', type=int); p.add_argument('--puma-k-end', type=int)
    p.add_argument('--formats', nargs='+'); p.add_argument('--masks', nargs='+')
    p.add_argument('--decode', nargs='+'); p.add_argument('--no-ar', action='store_true')
    p.add_argument('--ar', action='store_true')
    p.add_argument('--data-mode', choices=['balanced', 'natural'])
    p.add_argument('--curriculum', nargs='+', help='Curriculum switchover fracs, e.g. 20 50')
    p.add_argument('--no-curriculum', action='store_true')
    p.add_argument('--tag', type=str, default=''); p.add_argument('--seed', type=int)
    p.add_argument('--seeds', nargs='+', type=int)
    # Colab/IPython safe parsing
    try:
        args, _ = p.parse_known_args()
    except SystemExit:
        args, _ = p.parse_known_args([])
    g = globals()
    for a, gl in {'n_train': 'N_TRAIN', 'n_test': 'N_TEST', 'epochs': 'MAX_EPOCHS',
                   'batch_size': 'BATCH_SIZE', 'eval_every': 'EVAL_EVERY',
                   'gen_eval_every': 'GEN_EVAL_EVERY', 'n_layer': 'N_LAYER',
                   'n_head': 'N_HEAD', 'n_embd': 'N_EMBD', 'dropout': 'DROPOUT',
                   'puma_tau': 'PUMA_TAU', 'puma_k_start': 'PUMA_K_START',
                   'puma_k_end': 'PUMA_K_END', 'seed': 'SEED'}.items():
        v = getattr(args, a, None)
        if v is not None: g[gl] = v
    if args.nd:
        g['ND'] = args.nd; g['ANS_LEN'] = args.nd + 1
        if not args.puma_k_end: g['PUMA_K_END'] = args.nd + 1
    if args.formats: g['FORMATS'] = args.formats
    if args.masks: g['MASK_TYPES'] = args.masks
    if args.decode: g['DECODE_POLICIES'] = args.decode
    if args.no_ar: g['RUN_AR'] = False
    if args.ar: g['RUN_AR'] = True
    if args.data_mode: g['DATA_MODE'] = args.data_mode
    if args.no_curriculum: g['RUN_CURRICULUM'] = False
    if args.curriculum:
        modes = []
        for pct in args.curriculum:
            p_int = int(pct); frac = p_int / 100.0
            modes.append((f'puma2rand_{p_int}', [(0.0, 'puma'), (frac, 'random')]))
        g['CURRICULUM_MODES'] = modes
    return args


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Data helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _pad(n, w): return str(n).zfill(w)

def _fmt_plain(a, b): return f"{_pad(a,ND)}+{_pad(b,ND)}={_pad(a+b,ANS_LEN)}"
def _fmt_reverse(a, b): return f"{_pad(a,ND)}+{_pad(b,ND)}={_pad(a+b,ANS_LEN)[::-1]}"
FMT_FN = {'plain': _fmt_plain, 'reverse': _fmt_reverse}

def get_answer(s, fmt): return s.split('=')[1]
def _parse_operands(s):
    parts = s.split('=')[0].split('+'); return int(parts[0]), int(parts[1])

def _count_carries(a, b):
    a_s, b_s = _pad(a, ND), _pad(b, ND)
    carry, count = 0, 0
    for i in range(ND - 1, -1, -1):
        s = int(a_s[i]) + int(b_s[i]) + carry; carry = s // 10; count += carry
    return count

def _carry_at_answer_pos(a, b, fmt):
    """Per-answer-position carry-in flag."""
    a_s, b_s = _pad(a, ND), _pad(b, ND)
    flags, carry = [], 0
    for i in range(ND - 1, -1, -1):
        s = int(a_s[i]) + int(b_s[i]) + carry; carry = s // 10; flags.append(bool(carry))
    ci = [False] * ANS_LEN
    if fmt == 'plain':
        for k in range(ANS_LEN):
            lp = ND - k
            if lp == ND: ci[k] = flags[ND-1] if ND-1 < len(flags) else False
            elif 0 <= lp-1 < len(flags): ci[k] = flags[lp-1]
    else:
        for k in range(ANS_LEN):
            if k == 0: ci[k] = False
            elif k-1 < len(flags): ci[k] = flags[k-1]
    return ci

def _gkp_at_answer_pos(a, b, fmt):
    """Classify each answer position as g/k/p/carry_out."""
    a_s, b_s = _pad(a, ND), _pad(b, ND)
    digit_gkp = []
    for i in range(ND - 1, -1, -1):
        s = int(a_s[i]) + int(b_s[i])
        digit_gkp.append('g' if s >= 10 else ('p' if s == 9 else 'k'))
    out = ['?'] * ANS_LEN
    if fmt == 'plain':
        out[0] = 'carry_out'
        for j in range(ND): out[ND - j] = digit_gkp[j]
    else:
        out[ANS_LEN - 1] = 'carry_out'
        for j in range(ND): out[j] = digit_gkp[j]
    return out

def _dependency_context_at_pos(a, b, fmt):
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
    if fmt == 'plain':
        out[0] = 'carry_out'
        for j in range(ND): out[ND - j] = dep[j]
    else:
        out[ANS_LEN - 1] = 'carry_out'
        for j in range(ND): out[j] = dep[j]
    return out

def _chain_stats(a, b):
    """Rich carry-chain stats: max_chain_len, chain_reaches_msb, msb_carry_out, etc."""
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

def _pos_labels(fmt):
    if fmt == 'plain': return ['MSB'] + [f'p{j}' for j in range(1, ANS_LEN-1)] + ['LSB']
    return ['LSB'] + [f'p{j}' for j in range(1, ANS_LEN-1)] + ['MSB']

def _lsb_policy(fmt): return 'r2l' if fmt == 'plain' else 'l2r'

def build_tok(): return CharTokenizer(list('0123456789+='), {'mask': 'M', 'pad': 'P'})


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Data generation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def gen_data(n, fmt, seed):
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
    return [FMT_FN[fmt](a, b) for a, b in out[:n]]

def gen_data_natural(n, fmt, seed):
    """Natural distribution training data — NO carry balancing.
    At ND=32, MSB carry-in base rate drops to ~3-5%, creating the
    rare-event gradient needed for coverage deficit analysis.
    """
    rng = random.Random(seed)
    lo, hi = 10**(ND-1), 10**ND - 1
    return [FMT_FN[fmt](rng.randint(lo, hi), rng.randint(lo, hi)) for _ in range(n)]

def gen_test_data(n, fmt, seed):
    """Full ND-digit test data. Uses DATA_MODE for consistency with training."""
    if DATA_MODE == 'natural':
        return gen_data_natural(n, fmt, seed)
    rng = random.Random(seed); lo, hi = 10**(ND-1), 10**ND-1
    pool = defaultdict(list); seen = set()
    for _ in range(max(n*200, 100000)):
        a, b = rng.randint(lo, hi), rng.randint(lo, hi)
        if (a, b) in seen: continue
        seen.add((a, b)); pool[_count_carries(a, b)].append((a, b))
    target = max(1, n // max(len(pool), 1)); out = []
    for nc in sorted(pool): rng.shuffle(pool[nc]); out.extend(pool[nc][:target])
    while len(out) < n:
        a, b = rng.randint(lo, hi), rng.randint(lo, hi)
        if (a, b) not in seen: out.append((a, b)); seen.add((a, b))
    rng.shuffle(out)
    return [FMT_FN[fmt](a, b) for a, b in out[:n]]

def gen_carry_heavy_test(n, fmt, seed, min_max_chain=3):
    rng = random.Random(seed); lo, hi = 10**(ND-1), 10**ND-1; results = []
    for _ in range(n * 200):
        a, b = rng.randint(lo, hi), rng.randint(lo, hi)
        if _max_chain_len(a, b) >= min_max_chain: results.append(FMT_FN[fmt](a, b))
        if len(results) >= n: break
    return results[:n]

def gen_corner_case_test(n, fmt, seed, category='msb_chain'):
    rng = random.Random(seed); lo, hi = 10**(ND-1), 10**ND-1; results = []
    def _accept(a, b):
        st = _chain_stats(a, b)
        if category == 'msb_chain': return st['chain_reaches_msb']
        if category == 'full_propagate': return st['n_propagate'] == ND
        if category == 'long_chain': return st['max_chain_len'] >= ND // 2
        return False
    for _ in range(n * 3000):
        if category == 'full_propagate':
            ad = [rng.randint(1, 8)] + [rng.randint(0, 9) for _ in range(ND-1)]
            bd = [9-d for d in ad]
            if bd[0] < 1: ad[0] = rng.randint(1, 8); bd[0] = 9-ad[0]
            a, b = int(''.join(str(d) for d in ad)), int(''.join(str(d) for d in bd))
        else:
            a, b = rng.randint(lo, hi), rng.randint(lo, hi)
        if _accept(a, b): results.append(FMT_FN[fmt](a, b))
        if len(results) >= n: break
    if len(results) < n: print(f"    WARNING: corner/{category}: {len(results)}/{n}")
    return results[:n]

def gen_min_chain_test(n, fmt, seed, min_chain):
    """Construct samples with a propagate chain of exactly min_chain length.
    Places min_chain consecutive p-positions (digit pairs summing to 9)
    at a random location, fills the rest with g/k positions.
    """
    rng = random.Random(seed); results = []; seen = set()
    for _ in range(n * 50):
        if len(results) >= n: break
        # Choose where to place the p-chain
        max_start = ND - min_chain
        if max_start < 0: break
        chain_start = rng.randint(0, max_start)  # digit index (LSB=0)

        a_digits = [0] * ND; b_digits = [0] * ND
        for d in range(ND):
            if chain_start <= d < chain_start + min_chain:
                # p-position: a+b = 9
                a_d = rng.randint(0, 9); b_d = 9 - a_d
            else:
                # g or k position: a+b != 9 (and a+b < 10 for k, >= 10 for g)
                a_d = rng.randint(0, 9); b_d = rng.randint(0, 9)
                while a_d + b_d == 9:
                    b_d = rng.randint(0, 9)
            a_digits[d] = a_d; b_digits[d] = b_d

        # Ensure MSB is nonzero
        if a_digits[ND-1] == 0: a_digits[ND-1] = rng.randint(1, 9)
        if b_digits[ND-1] == 0: b_digits[ND-1] = rng.randint(1, 9)
        # Fix MSB if it was in chain and got overwritten
        if chain_start <= ND-1 < chain_start + min_chain:
            a_digits[ND-1] = rng.randint(1, 8)
            b_digits[ND-1] = 9 - a_digits[ND-1]

        # Convert digit arrays (LSB=index 0) to integers
        a = int(''.join(str(a_digits[d]) for d in range(ND-1, -1, -1)))
        b = int(''.join(str(b_digits[d]) for d in range(ND-1, -1, -1)))

        # Verify actual chain length
        st = _chain_stats(a, b)
        if st['max_chain_len'] >= min_chain and (a, b) not in seen:
            seen.add((a, b)); results.append(FMT_FN[fmt](a, b))

    if len(results) < n:
        print(f"    WARNING: chain>={min_chain}: {len(results)}/{n}")
    return results[:n]

def gen_counterfactual_pairs(n, seed):
    rng = random.Random(seed); lo, hi = 10**(ND-1), 10**ND-1; results = []
    for _ in range(n * 500):
        if len(results) >= n: break
        target_d = rng.randint(1, ND-1); a1, b1 = rng.randint(lo, hi), rng.randint(lo, hi)
        a1_s, b1_s = _pad(a1, ND), _pad(b1, ND)
        si = ND - 1 - target_d; da, db = int(a1_s[si]), int(b1_s[si])
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
# Probes & analyses
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@torch.no_grad()
def probe_per_position(model, tokenizer, test_samples, objective, fmt, max_len, device=None):
    """Fully-masked probe: per-position loss/acc/conf + dependency context tracking."""
    if device is None: device = DEVICE
    model.eval(); mask_id = tokenizer.special_ids['mask']
    ids_all, ans_all = encode_samples(test_samples, tokenizer, max_len)
    ids_all, ans_all = ids_all.to(device), ans_all.to(device)
    ci_tensor = torch.tensor([_carry_at_answer_pos(*_parse_operands(s), fmt) for s in test_samples],
                              dtype=torch.bool, device=device)
    dep_ctx_names = ['g', 'k', 'p_above_g', 'p_above_k', 'p_above_p', 'p_bottom', 'carry_out']
    dep_to_id = {n: i for i, n in enumerate(dep_ctx_names)}
    dep_ids = torch.tensor([[dep_to_id.get(d, 0) for d in _dependency_context_at_pos(*_parse_operands(s), fmt)]
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

        if objective == 'ar':
            logits = model(ids[:, :-1]); pred_pos = ans_pos - 1
            valid = (pred_pos >= 0) & (pred_pos < logits.shape[1])
            pred_pos = pred_pos.clamp(min=0, max=logits.shape[1]-1)
            tgt = ids[bi, ans_pos]; lp = F.log_softmax(logits[bi, pred_pos], dim=-1)
            losses = -lp.gather(2, tgt.unsqueeze(2)).squeeze(2) * valid.float()
            preds = logits[bi, pred_pos].argmax(dim=-1)
            corrects = (preds == tgt).float() * valid.float()
            confs = F.softmax(logits[bi, pred_pos], dim=-1).max(dim=-1).values * valid.float()
            w = valid.float()
        else:
            xm = ids.clone(); xm[bi, ans_pos] = mask_id
            logits = model(xm); al = logits[bi, ans_pos]; tgt = ids[bi, ans_pos]
            lp = F.log_softmax(al, dim=-1)
            losses = -lp.gather(2, tgt.unsqueeze(2)).squeeze(2)
            cl = al.clone(); cl[:, :, mask_id] = -float('inf')
            probs = F.softmax(cl, dim=-1)
            confs = probs.max(dim=-1).values; preds = probs.argmax(dim=-1)
            corrects = (preds == tgt).float(); w = torch.ones(B, ANS_LEN, device=device)

        for j in range(ANS_LEN):
            L[j] += losses[:, j].sum(); C[j] += corrects[:, j].sum()
            CF[j] += confs[:, j].sum(); N[j] += w[:, j].sum()
        ci_b = ci_tensor[st:en]
        for j in range(ANS_LEN):
            ci_j = ci_b[:, j] & (w[:, j] > 0); nc_j = ~ci_b[:, j] & (w[:, j] > 0)
            Lc[j] += losses[ci_j, j].sum(); Cc[j] += corrects[ci_j, j].sum(); Nc[j] += ci_j.sum()
            Ln[j] += losses[nc_j, j].sum(); Cn[j] += corrects[nc_j, j].sum(); Nn[j] += nc_j.sum()

        if objective != 'ar':
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
    if objective != 'ar' and dep_count:
        result['dep_context'] = {ctx: {'conf': dep_conf_sum[ctx]/n, 'acc': dep_acc_sum[ctx]/n, 'n': n}
                                  for ctx, n in dep_count.items() if n > 0}
    if objective != 'ar':
        cr = sorted(range(ANS_LEN), key=lambda j: pos_conf[j], reverse=True)
        conc = 0; n_p = ANS_LEN * (ANS_LEN - 1) // 2
        for i in range(ANS_LEN):
            for j in range(i+1, ANS_LEN):
                ri, rj = cr.index(i), cr.index(j)
                conc += int(rj < ri) if fmt == 'plain' else int(ri < rj)
        result['conf_concordance'] = conc / n_p
        result['conf_spread'] = max(pos_conf) - min(pos_conf)
    return result


@torch.no_grad()
def eval_counterfactual(model, tokenizer, cf_pairs, fmt, max_len, device=None):
    if device is None: device = DEVICE
    model.eval(); mask_id = tokenizer.special_ids['mask']
    deltas, flips = [], 0; c0_ok, c1_ok, n0, n1 = 0, 0, 0, 0
    for cf in cf_pairs:
        td = cf['target_d']; preds_confs = []
        for a, b in cf['pair']:
            s = FMT_FN[fmt](a, b); ids = torch.tensor(tokenizer.encode(s), device=device).unsqueeze(0)
            ans_s = s.index('=') + 1; xm = ids.clone()
            xm[0, ans_s:ans_s+ANS_LEN] = mask_id; logits = model(xm)
            aj = ND - td if fmt == 'plain' else td; pos = ans_s + aj
            cl = logits[0, pos].clone(); cl[mask_id] = -float('inf')
            probs = F.softmax(cl, dim=-1)
            preds_confs.append((probs.argmax().item(), probs.max().item(), probs.argmax().item() == ids[0, pos].item()))
        deltas.append(abs(preds_confs[1][1] - preds_confs[0][1]))
        if preds_confs[0][0] != preds_confs[1][0]: flips += 1
        for mi, (_, _, ok) in enumerate(preds_confs):
            if cf['carry_in'][mi]: c1_ok += ok; n1 += 1
            else: c0_ok += ok; n0 += 1
    n = len(cf_pairs)
    return {'n_pairs': n, 'mean_conf_delta': sum(deltas)/max(n,1),
            'prediction_flip_rate': flips/max(n,1),
            'acc_carry_in_0': c0_ok/max(n0,1), 'acc_carry_in_1': c1_ok/max(n1,1)}


@torch.no_grad()
def gen_eval_with_stats(model, tokenizer, test_samples, fmt, max_len,
                        decode_policy='confidence', device=None):
    """Per-sample generation with chain stats + error positions."""
    if device is None: device = DEVICE
    mask_id = tokenizer.special_ids['mask']; pad_id = tokenizer.special_ids['pad']
    model.eval(); out = []
    for st in range(0, len(test_samples), 128):
        batch = test_samples[st:min(st+128, len(test_samples))]; B = len(batch)
        penc = [tokenizer.encode(s.split('=')[0]+'=') for s in batch]
        pm = max(len(p) for p in penc)
        pids = torch.full((B, pm), pad_id, dtype=torch.long)
        for i, e in enumerate(penc): pids[i, :len(e)] = torch.tensor(e)
        policy = _lsb_policy(fmt) if decode_policy == 'lsb' else 'confidence'
        gen, _, info = generate_diffusion(model, pids, ANS_LEN, mask_id,
                                          policy=policy, greedy=True, device=device)
        pred_ids = gen[:, pm:pm+ANS_LEN]; orders = info.get('orders')
        for i in range(B):
            s = batch[i]; ps = tokenizer.decode(pred_ids[i].cpu().tolist())
            gs = get_answer(s, fmt); a, b = _parse_operands(s)
            pc = [ps[j] == gs[j] if j < len(ps) else False for j in range(len(gs))]
            errs = [j for j in range(len(gs)) if j >= len(ps) or ps[j] != gs[j]]
            out.append({'correct': ps==gs, 'pos_correct': pc, 'error_positions': errs,
                       'chain_stats': _chain_stats(a, b), 'gkp_at_pos': _gkp_at_answer_pos(a, b, fmt),
                       'dep_ctx': _dependency_context_at_pos(a, b, fmt),
                       'n_carries': _count_carries(a, b), 'carry_flags': _carry_at_answer_pos(a, b, fmt)})
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


def analyse_carry_rarity(per_sample, test_samples, fmt):
    """Per-position carry-in base rate × conditional accuracy."""
    ci_flags = [_carry_at_answer_pos(*_parse_operands(s), fmt) for s in test_samples]
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


@torch.no_grad()
def simulate_puma_coverage(model, tokenizer, test_samples, fmt, max_len,
                           K=None, tau=None, n_samples=200, device=None):
    """Measure PUMA chain coverage × carry-in condition."""
    if device is None: device = DEVICE
    if K is None: K = PUMA_K_END
    if tau is None: tau = PUMA_TAU
    model.eval(); mask_id = tokenizer.special_ids['mask']
    ci_flags = [_carry_at_answer_pos(*_parse_operands(s), fmt) for s in test_samples[:n_samples]]
    N = min(len(test_samples), n_samples)
    c1s, c0s, c1n, c0n = torch.zeros(ANS_LEN), torch.zeros(ANS_LEN), \
                          torch.zeros(ANS_LEN, dtype=torch.long), torch.zeros(ANS_LEN, dtype=torch.long)
    for si in range(N):
        s = test_samples[si]; ci = ci_flags[si]
        prefix = s.split('=')[0]+'='; answer = get_answer(s, fmt)
        penc = tokenizer.encode(prefix); aenc = tokenizer.encode(answer); T_pre = len(penc)
        x = torch.tensor(penc + [mask_id]*ANS_LEN, dtype=torch.long, device=device).unsqueeze(0)
        x0 = torch.tensor(aenc[:ANS_LEN], dtype=torch.long, device=device)
        is_m = torch.ones(ANS_LEN, dtype=torch.bool, device=device)
        steps_m = torch.zeros(ANS_LEN); total = 0
        for step in range(K):
            if not is_m.any(): break
            total += 1; steps_m += is_m.cpu().float()
            logits = model(x); nm = is_m.sum().item()
            nr = max(1, int(math.ceil(nm / max(K-step, 1))))
            confs = torch.full((ANS_LEN,), -float('inf'), device=device)
            for j in range(ANS_LEN):
                if is_m[j]:
                    cl = logits[0, T_pre+j].clone(); cl[mask_id] = -float('inf')
                    confs[j] = F.softmax(cl, dim=-1).max()
            ranked = confs.argsort(descending=True)
            reveal = torch.zeros(ANS_LEN, dtype=torch.bool, device=device)
            for ri in range(ANS_LEN):
                j = ranked[ri].item()
                if not is_m[j]: continue
                if reveal.sum() < nr or confs[j] > tau: reveal[j] = True
            for j in range(ANS_LEN):
                if reveal[j]: x[0, T_pre+j] = x0[j]; is_m[j] = False
        if total == 0: continue
        frac = steps_m / total
        for j in range(ANS_LEN):
            if ci[j]: c1s[j] += frac[j]; c1n[j] += 1
            else: c0s[j] += frac[j]; c0n[j] += 1
    per_pos = []
    for j in range(ANS_LEN):
        n1, n0 = c1n[j].item(), c0n[j].item()
        cv1 = c1s[j].item()/n1 if n1 > 0 else None; cv0 = c0s[j].item()/n0 if n0 > 0 else None
        deficit = (cv0 - cv1) if cv0 is not None and cv1 is not None else None
        br = n1/max(n1+n0, 1)
        per_pos.append({'position': j, 'base_rate': br, 'cov_c1': cv1, 'cov_c0': cv0, 'deficit': deficit})
    valid = [(p['base_rate'], p['deficit']) for p in per_pos if p['deficit'] is not None]
    corr = None
    if len(valid) >= 3:
        brs, ds = [v[0] for v in valid], [v[1] for v in valid]
        mb, md = sum(brs)/len(brs), sum(ds)/len(ds)
        c = sum((b-mb)*(d-md) for b, d in zip(brs, ds))
        sb = sum((b-mb)**2 for b in brs)**0.5; sd = sum((d-md)**2 for d in ds)**0.5
        corr = c/(sb*sd) if sb > 0 and sd > 0 else 0.0
    return {'per_position': per_pos, 'corr': corr}


def analyse_error_localization(per_sample, fmt):
    """Where in the chain do errors occur? Maps errors to chain structure.
    Returns: {overflow_errors, chain_top_errors, chain_mid_errors,
              chain_bottom_errors, gk_errors} as fractions of total errors.
    """
    cats = defaultdict(int); total_errs = 0
    for r in per_sample:
        if r['correct']: continue
        gkp = r['gkp_at_pos']; cs = r['chain_stats']; dep = r['dep_ctx']
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
            **{k: v/total_errs for k, v in sorted(cats.items())}}


@torch.no_grad()
def analyse_confidence_calibration(model, tokenizer, test_samples, fmt, max_len,
                                   decode_policy='confidence', device=None):
    """When the model is wrong, is it confident or uncertain?
    Returns per-chain-length: mean confidence for correct/incorrect predictions.
    """
    if device is None: device = DEVICE
    model.eval(); mask_id = tokenizer.special_ids['mask']
    pad_id = tokenizer.special_ids['pad']
    by_cl = defaultdict(lambda: {'conf_correct': [], 'conf_wrong': []})

    for st in range(0, len(test_samples), 128):
        batch = test_samples[st:min(st+128, len(test_samples))]; B = len(batch)
        penc = [tokenizer.encode(s.split('=')[0]+'=') for s in batch]
        pm = max(len(p) for p in penc)
        pids = torch.full((B, pm), pad_id, dtype=torch.long)
        for i, e in enumerate(penc): pids[i, :len(e)] = torch.tensor(e)

        # Single forward pass with all answer positions masked
        full_enc = [tokenizer.encode(s) for s in batch]
        ml = max(len(e) for e in full_enc)
        ids = torch.full((B, ml), pad_id, dtype=torch.long, device=device)
        for i, e in enumerate(full_enc):
            ids[i, :len(e)] = torch.tensor(e, device=device)
        ans_starts = torch.tensor([s.index('=')+1 for s in batch], device=device)
        _ar = torch.arange(ANS_LEN, device=device)
        ap = (ans_starts.unsqueeze(1) + _ar).clamp(max=ml-1)
        bi = torch.arange(B, device=device).unsqueeze(1).expand_as(ap)
        xm = ids.clone(); xm[bi, ap] = mask_id
        logits = model(xm); al = logits[bi, ap]
        cl = al.clone(); cl[:, :, mask_id] = -float('inf')
        probs = F.softmax(cl, dim=-1)
        confs = probs.max(dim=-1).values; preds = cl.argmax(dim=-1)
        tgt = ids[bi, ap]; correct = (preds == tgt)

        for i in range(B):
            a, b = _parse_operands(batch[i])
            cs = _chain_stats(a, b)
            mcl = cs['max_chain_len']
            cl_bin = 0 if mcl == 0 else (2 if mcl <= 2 else (4 if mcl <= 4 else
                     (8 if mcl <= 8 else (16 if mcl <= 16 else 32))))
            for j in range(ANS_LEN):
                c = confs[i, j].item()
                if correct[i, j]: by_cl[cl_bin]['conf_correct'].append(c)
                else: by_cl[cl_bin]['conf_wrong'].append(c)

    result = {}
    for cl_bin in sorted(by_cl):
        cc = by_cl[cl_bin]['conf_correct']
        cw = by_cl[cl_bin]['conf_wrong']
        result[f'cl<={cl_bin}'] = {
            'mean_conf_correct': sum(cc)/len(cc) if cc else None,
            'mean_conf_wrong': sum(cw)/len(cw) if cw else None,
            'n_correct': len(cc), 'n_wrong': len(cw),
            'overconfident_wrong': sum(1 for c in cw if c > 0.8)/max(len(cw),1) if cw else None,
        }
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Training
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def train_model(objective, tokenizer, train_samples, test_samples, max_len,
                fmt, mask_type='random', curriculum=None, device=None):
    """Train with optional curriculum schedule.
    curriculum: None → use mask_type throughout
                list of (frac, type) → switch at epoch = frac * MAX_EPOCHS
                'interleaved' → alternate puma/random every batch
    """
    if device is None: device = DEVICE
    train_ids, train_ans = encode_samples(train_samples, tokenizer, max_len)
    train_ids, train_ans = train_ids.to(device), train_ans.to(device)
    N, T = train_ids.shape; bpe = (N + BATCH_SIZE - 1) // BATCH_SIZE
    total_iters = MAX_EPOCHS * bpe
    mask_id = tokenizer.special_ids['mask']; pad_id = tokenizer.special_ids['pad']
    is_causal = (objective == 'ar')
    model = Transformer(vocab_size=len(tokenizer), block_size=max_len+8,
                        n_layer=N_LAYER, n_head=N_HEAD, n_embd=N_EMBD,
                        dropout=DROPOUT, is_causal=is_causal, pos_enc=POS_ENC).to(device)
    label = mask_type if not curriculum else (curriculum if isinstance(curriculum, str)
             else '+'.join(f"{t}@{int(f*100)}%" for f, t in curriculum))
    tag = objective + (f"/{label}" if objective == 'diffusion' else "")
    print(f"  [{tag}] params={model.n_params:,}, {bpe} batches/epoch, {MAX_EPOCHS} epochs")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.99), weight_decay=0.1)
    warmup_iters = WARMUP_EPOCHS * bpe
    def get_lr(it):
        if it < warmup_iters: return LR * it / max(warmup_iters, 1)
        ratio = (it - warmup_iters) / max(total_iters - warmup_iters, 1)
        return MIN_LR + 0.5 * (LR - MIN_LR) * (1 + math.cos(math.pi * min(ratio, 1.0)))

    dynamics = {'checkpoints': [], 'gen_checkpoints': [], 'train_loss': []}
    best_loss, best_state = float('inf'), None; it = 0; tg = 0; t0 = time.time()
    _arange = torch.arange(ANS_LEN, device=device)

    def _get_active_mask(epoch, batch_in_epoch=0):
        """Determine which mask type to use right now."""
        if curriculum is None: return mask_type
        if curriculum == 'interleaved':
            return 'puma' if batch_in_epoch % 2 == 0 else 'random'
        # Schedule: list of (frac, type)
        active = curriculum[0][1]
        for frac, mt in curriculum:
            if epoch >= frac * MAX_EPOCHS: active = mt
        return active

    # PUMA streaming buffer (always allocate; only used when active_mask is puma)
    puma_x0 = torch.zeros(BATCH_SIZE, T, dtype=torch.long, device=device)
    puma_z = torch.zeros(BATCH_SIZE, T, dtype=torch.long, device=device)
    puma_ans = torch.zeros(BATCH_SIZE, dtype=torch.long, device=device)
    puma_stage = torch.zeros(BATCH_SIZE, dtype=torch.long, device=device)
    pool = torch.randperm(N); pool_ptr = 0
    puma_initialized = False

    def _refresh(indices):
        nonlocal pool_ptr, pool
        idx_t = torch.tensor(indices, device=device); n = len(indices)
        if pool_ptr + n > len(pool): pool = torch.randperm(N); pool_ptr = 0
        si = pool[pool_ptr:pool_ptr+n].to(device); pool_ptr += n
        puma_x0[idx_t] = train_ids[si]; puma_z[idx_t] = train_ids[si].clone()
        puma_ans[idx_t] = train_ans[si]; puma_stage[idx_t] = 0
        ap = (puma_ans[idx_t].unsqueeze(1) + _arange).clamp(max=T-1)
        bii = idx_t.unsqueeze(1).expand_as(ap)
        puma_z[bii, ap] = mask_id

    def _advance(logits, K_cur):
        nonlocal puma_stage
        B = BATCH_SIZE; ap = (puma_ans.unsqueeze(1) + _arange).clamp(max=T-1)
        bi = torch.arange(B, device=device).unsqueeze(1).expand_as(ap)
        is_m = (puma_z[bi, ap] == mask_id)
        if not is_m.any(): _refresh(list(range(B))); return
        nm = is_m.sum(dim=1).float()
        K_rem = (K_cur - puma_stage).clamp(min=1)
        nr = (nm / K_rem.float()).ceil().long().clamp(min=1)
        lp = logits[bi, ap].clone(); lp[:, :, mask_id] = -float('inf')
        confs = F.softmax(lp, dim=-1).max(dim=-1).values; confs[~is_m] = -float('inf')
        ranked = confs.argsort(dim=1, descending=True)
        rop = torch.zeros_like(ranked); rop.scatter_(1, ranked, _arange.expand(B, -1))
        reveal = ((rop < nr.unsqueeze(1)) | (confs > PUMA_TAU)) & is_m
        puma_z[bi[reveal], ap[reveal]] = puma_x0[bi[reveal], ap[reveal]]
        puma_stage += 1
        done = (~(puma_z[bi, ap] == mask_id).any(dim=1)) | (puma_stage >= K_cur)
        if done.any(): _refresh(done.nonzero(as_tuple=True)[0].tolist())

    def _do_eval(epoch):
        nonlocal best_loss, best_state
        probe = probe_per_position(model, tokenizer, test_samples, objective, fmt, max_len, device)
        dynamics['checkpoints'].append({'epoch': epoch, 'iter': it, 'tg': tg, **probe})
        if probe['overall_loss'] < best_loss and epoch > 0:
            best_loss = probe['overall_loss']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        dc = probe.get('dep_context', {})
        parts = [f"{c}={dc[c]['acc']:.2f}" for c in ['g','k','p_above_g','p_above_k','p_above_p'] if c in dc]
        print(f"    [eval ep {epoch}] loss={probe['overall_loss']:.4f} acc={probe['overall_acc']:.4f} "
              + (' '.join(parts) if parts else '') + f" | {time.time()-t0:.0f}s")

    def _do_gen(epoch):
        if objective != 'diffusion': return
        for dp in ['confidence']:
            r = _quick_gen(model, tokenizer, test_samples, objective, fmt, dp, device=device)
            entry = dynamics['gen_checkpoints'][-1] if dynamics['gen_checkpoints'] and \
                    dynamics['gen_checkpoints'][-1].get('epoch') == epoch else {'epoch': epoch}
            entry.setdefault('gen_acc', {})[dp] = r['accuracy']
            if entry not in dynamics['gen_checkpoints']: dynamics['gen_checkpoints'].append(entry)
            print(f"      [gen] {dp}={r['accuracy']:.3f}")

    model.eval(); _do_eval(0); model.train()
    prev_active = None

    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_loss = torch.tensor(0.0, device=device); epoch_tg = 0; epoch_n = 0
        K_cur = PUMA_K_START + int((PUMA_K_END - PUMA_K_START) * epoch / MAX_EPOCHS)

        perm = torch.randperm(N, device=device)
        for bi_idx in range(bpe):
            active = _get_active_mask(epoch, bi_idx)

            # Initialize PUMA buffer on first use or re-entry
            if active == 'puma' and not puma_initialized:
                _refresh(list(range(BATCH_SIZE))); puma_initialized = True
            if active != prev_active and prev_active is not None:
                phase_str = f"[→{active}]" if active != prev_active else ""
                if phase_str:
                    print(f"    ep {epoch}: switching to {active}")
                if active == 'puma' and not puma_initialized:
                    _refresh(list(range(BATCH_SIZE))); puma_initialized = True
            prev_active = active

            for pg in optimizer.param_groups: pg['lr'] = get_lr(it)

            if active == 'puma':
                m = (puma_z == mask_id)
                if m.sum() == 0: _refresh(list(range(BATCH_SIZE))); m = (puma_z == mask_id)
                logits = model(puma_z); loss = F.cross_entropy(logits[m], puma_x0[m])
                epoch_tg += m.sum().item()
                optimizer.zero_grad(set_to_none=True); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP); optimizer.step()
                _advance(logits.detach(), K_cur)
            elif objective == 'ar':
                idx = perm[bi_idx*BATCH_SIZE:min((bi_idx+1)*BATCH_SIZE, N)]
                ids = train_ids[idx]; ans_s = train_ans[idx]; B_b = ids.shape[0]
                logits = model(ids[:, :-1]); targets = ids[:, 1:]
                pos = torch.arange(T-1, device=device).unsqueeze(0)
                lm = (pos >= (ans_s.unsqueeze(1) - 1)) & (targets != pad_id)
                if lm.sum() == 0: it += 1; continue
                loss = F.cross_entropy(logits[lm], targets[lm]); epoch_tg += lm.sum().item()
                optimizer.zero_grad(set_to_none=True); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP); optimizer.step()
            else:  # random masking
                idx = perm[bi_idx*BATCH_SIZE:min((bi_idx+1)*BATCH_SIZE, N)]
                ids = train_ids[idx]; ans_s = train_ans[idx]; B_b = ids.shape[0]
                ap = (ans_s.unsqueeze(1) + _arange).clamp(max=T-1)
                bii = torch.arange(B_b, device=device).unsqueeze(1).expand_as(ap)
                ans_mask = torch.zeros(B_b, T, dtype=torch.bool, device=device)
                for j in range(ANS_LEN): ans_mask[range(B_b), ap[:, j]] = True
                t_r = torch.rand(B_b, device=device)
                m_probs = t_r.unsqueeze(1) * ans_mask.float()
                m = torch.bernoulli(m_probs).bool()
                no_m = ~m.any(dim=1)
                if no_m.any():
                    rj = torch.randint(ANS_LEN, (no_m.sum(),), device=device)
                    fp = ap[no_m].gather(1, rj.unsqueeze(1)).squeeze(1)
                    m[no_m.nonzero(as_tuple=True)[0], fp] = True
                xm = ids.clone(); xm[m] = mask_id; logits = model(xm)
                if m.sum() == 0: it += 1; continue
                loss = F.cross_entropy(logits[m], ids[m]); epoch_tg += m.sum().item()
                optimizer.zero_grad(set_to_none=True); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP); optimizer.step()
            epoch_loss += loss.detach(); epoch_n += 1; it += 1

        tg += epoch_tg
        if epoch % LOG_EVERY == 0:
            dynamics['train_loss'].append((epoch, epoch_loss.item()/max(epoch_n, 1)))
            print(f"    ep {epoch:4d}/{MAX_EPOCHS} | loss {epoch_loss.item()/max(epoch_n,1):.4f} | "
                  f"lr {get_lr(it):.1e} | tg {tg:,} | {time.time()-t0:.0f}s")
        do_eval = (epoch % EVAL_EVERY == 0) or \
                  (epoch < MAX_EPOCHS*0.1 and epoch % max(EVAL_EVERY//5, 1) == 0) or \
                  (epoch < MAX_EPOCHS*0.3 and epoch % max(EVAL_EVERY//2, 1) == 0)
        if do_eval and epoch < MAX_EPOCHS:
            model.eval(); _do_eval(epoch)
            if epoch % GEN_EVAL_EVERY == 0: _do_gen(epoch)
            model.train()

    if best_state: model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval(); _do_eval(MAX_EPOCHS); _do_gen(MAX_EPOCHS)
    return model, dynamics


def _quick_gen(model, tokenizer, test_samples, objective, fmt,
               decode_policy='confidence', n=None, device=None):
    if n is None: n = GEN_EVAL_N
    if device is None: device = DEVICE
    subset = test_samples[:n]; mask_id = tokenizer.special_ids['mask']
    pad_id = tokenizer.special_ids['pad']; results = []
    for st in range(0, len(subset), 128):
        batch = subset[st:st+128]; B = len(batch)
        penc = [tokenizer.encode(s.split('=')[0]+'=') for s in batch]
        pm = max(len(p) for p in penc)
        pids = torch.full((B, pm), pad_id, dtype=torch.long)
        for i, e in enumerate(penc): pids[i, :len(e)] = torch.tensor(e)
        if objective == 'ar':
            gen = generate_ar(model, pids, ANS_LEN, device); pred = gen[:, pm:pm+ANS_LEN]
        else:
            policy = _lsb_policy(fmt) if decode_policy == 'lsb' else 'confidence'
            gen, _, _ = generate_diffusion(model, pids, ANS_LEN, mask_id, policy=policy, greedy=True, device=device)
            pred = gen[:, pm:pm+ANS_LEN]
        for i in range(B):
            ps = tokenizer.decode(pred[i].cpu().tolist()); gs = get_answer(batch[i], fmt)
            pc = [ps[j]==gs[j] if j<len(ps) else False for j in range(len(gs))]
            a, b = _parse_operands(batch[i])
            results.append({'correct': ps==gs, 'pos_correct': pc, 'carry_flags': _carry_at_answer_pos(a,b,fmt)})
    n_r = len(results)
    pos_acc = [sum(r['pos_correct'][j] for r in results)/max(n_r,1) for j in range(ANS_LEN)]
    return {'accuracy': sum(r['correct'] for r in results)/max(n_r,1), 'position_accuracy': pos_acc, 'n': n_r}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Figures (compact)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def make_figures(all_dyn, all_final, fmt):
    figs = {}; labels = _pos_labels(fmt)
    cmap = plt.cm.coolwarm
    def pc(j): return cmap(1.0 - j/(ANS_LEN-1)) if fmt=='plain' else cmap(j/(ANS_LEN-1))
    conds = ([('ar', '')] if RUN_AR else []) + [('diffusion', mt) for mt in MASK_TYPES]
    COLORS = {'ar': '#e74c3c', 'diff-ran': '#3498db', 'diff-pum': '#8e44ad',
              'diff-ora': '#9b59b6', 'diff-con': '#1abc9c'}
    def _ck(obj, mt=''): return 'ar' if obj=='ar' else f"diff-{mt[:3]}"
    def _fk(obj, mt='', dp=''): return f"ar_{fmt}" if obj=='ar' else f"{obj}_{fmt}_{mt}_{dp}"

    # Fig 1: Per-position accuracy over training
    nc = len(conds)
    fig, axes = plt.subplots(1, nc, figsize=(6*nc, 5), squeeze=False); axes = axes[0]
    for ai, (obj, mt) in enumerate(conds):
        key = _fk(obj, mt, 'confidence'); dyn = all_dyn.get(key)
        if not dyn: continue
        ax = axes[ai]; xs = [c['epoch'] for c in dyn['checkpoints']]
        for j in range(ANS_LEN):
            ax.plot(xs, [c['pos_acc'][j] for c in dyn['checkpoints']], '-', color=pc(j), label=labels[j], lw=1.2)
        ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy'); ax.set_ylim(-0.05, 1.05)
        ax.set_title(_ck(obj, mt)); ax.legend(fontsize=4, ncol=4); ax.grid(alpha=0.3)
    fig.suptitle(f'Per-Position Probe Accuracy — {fmt}', y=1.02); fig.tight_layout()
    figs[f'pos_acc_{fmt}'] = fig

    # Fig 2: Dep context evolution
    dconds = [(mt, _fk('diffusion', mt, 'confidence')) for mt in MASK_TYPES
              if _fk('diffusion', mt, 'confidence') in all_dyn]
    ctx_colors = {'g': '#2ecc71', 'k': '#3498db', 'p_above_g': '#27ae60',
                  'p_above_k': '#2980b9', 'p_above_p': '#e74c3c', 'p_bottom': '#f39c12'}
    if dconds:
        fig, axes = plt.subplots(2, len(dconds), figsize=(7*len(dconds), 10), squeeze=False)
        for ai, (mt, key) in enumerate(dconds):
            dyn = all_dyn[key]; cps = dyn['checkpoints']; xs = [c['epoch'] for c in cps]
            for ri, metric in enumerate(['conf', 'acc']):
                ax = axes[ri][ai]
                for ctx in ['g', 'k', 'p_above_g', 'p_above_k', 'p_above_p']:
                    ys = [c.get('dep_context', {}).get(ctx, {}).get(metric, float('nan')) for c in cps]
                    if any(not math.isnan(y) for y in ys):
                        ax.plot(xs, ys, '-', color=ctx_colors.get(ctx), label=ctx, lw=1.5)
                ax.set_xlabel('Epoch'); ax.set_ylabel(metric.capitalize()); ax.set_ylim(0, 1.05)
                ax.set_title(f'{mt} — {metric}'); ax.legend(fontsize=6); ax.grid(alpha=0.3)
        fig.suptitle(f'Dependency Context Over Training — {fmt}', y=1.02); fig.tight_layout()
        figs[f'dep_ctx_{fmt}'] = fig

    # Fig 3: Standard vs carry-heavy vs corner cases
    test_types = ['standard', 'heavy'] + [f'corner_{c}' for c in ['msb_chain', 'full_propagate', 'long_chain']]
    for dp in DECODE_POLICIES:
        fig, ax = plt.subplots(figsize=(12, 5))
        for mi, mt in enumerate(MASK_TYPES):
            accs, lbls = [], []
            for tt in test_types:
                key = _fk('diffusion', mt, f'{tt}_{dp}')
                r = all_final.get(key)
                if r: accs.append(r['accuracy']); lbls.append(tt.replace('corner_',''))
            if accs:
                x = range(len(lbls)); w = 0.35; off = -w/2 if mi == 0 else w/2
                col = '#3498db' if mt == 'random' else '#8e44ad'
                ax.bar([i+off for i in x], accs, w, label=mt, color=col, alpha=0.8)
        if lbls:
            ax.set_xticks(range(len(lbls))); ax.set_xticklabels(lbls, fontsize=8, rotation=20)
        ax.set_ylabel('Accuracy'); ax.set_title(f'{dp} decode — {fmt}')
        ax.legend(); ax.grid(alpha=0.3, axis='y')
        all_a = [all_final.get(_fk('diffusion', mt, f'{tt}_{dp}'), {}).get('accuracy', 1)
                 for mt in MASK_TYPES for tt in test_types]
        all_a = [a for a in all_a if a is not None]
        if all_a: ax.set_ylim(max(0, min(all_a)-0.05), 1.005)
        fig.tight_layout(); figs[f'test_types_{dp}_{fmt}'] = fig

    # Fig 4: Carry rarity × accuracy gap
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    for mt, color, mk in [('random', '#3498db', 'o'), ('puma', '#8e44ad', 's')]:
        r = all_final.get(_fk('diffusion', mt, 'carry_rarity'))
        if not r: continue
        brs = [p['base_rate'] for p in r['per_position'] if p['acc_gap'] is not None]
        gaps = [p['acc_gap'] for p in r['per_position'] if p['acc_gap'] is not None]
        ax.scatter(brs, gaps, c=color, marker=mk, s=50, alpha=0.7,
                   label=f"{mt} (r={r['corr']:.2f})" if r['corr'] else mt)
    ax.axhline(0, color='gray', ls=':'); ax.set_xlabel('Carry-in base rate')
    ax.set_ylabel('acc(c=0) - acc(c=1)'); ax.legend(); ax.grid(alpha=0.3)
    ax = axes[1]
    bins_order = ['rare(<15%)', 'low(15-30%)', 'mid(30-50%)', 'high(>=50%)']
    for mi, (mt, col) in enumerate([('random', '#3498db'), ('puma', '#8e44ad')]):
        r = all_final.get(_fk('diffusion', mt, 'carry_rarity'))
        if not r: continue
        bl = [b for b in bins_order if b in r['binned']]; gs = [r['binned'][b]['mean_gap'] for b in bl]
        if bl:
            x = range(len(bl)); w = 0.35; off = -w/2 if mi == 0 else w/2
            ax.bar([i+off for i in x], gs, w, label=mt, color=col, alpha=0.8)
            ax.set_xticks(range(len(bl))); ax.set_xticklabels(bl, fontsize=7)
    ax.axhline(0, color='gray', ls=':'); ax.set_ylabel('Mean acc gap'); ax.legend(); ax.grid(alpha=0.3)
    fig.suptitle(f'Carry Rarity × Accuracy — {fmt}', y=1.02); fig.tight_layout()
    figs[f'carry_rarity_{fmt}'] = fig

    # Fig 5: PUMA coverage deficit
    cov = all_final.get(_fk('diffusion', 'puma', 'coverage'))
    if cov:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        ax = axes[0]
        c1 = [p['cov_c1'] or 0 for p in cov['per_position']]
        c0 = [p['cov_c0'] or 0 for p in cov['per_position']]
        x = range(ANS_LEN)
        ax.bar([i-0.2 for i in x], c0, 0.4, label='carry=0', color='#2ecc71', alpha=0.7)
        ax.bar([i+0.2 for i in x], c1, 0.4, label='carry=1', color='#e74c3c', alpha=0.7)
        ax.axhline(0.5, color='gray', ls='--', alpha=0.5, label='random baseline')
        ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=6, rotation=45)
        ax.set_ylabel('Coverage'); ax.legend(fontsize=7); ax.grid(alpha=0.3)
        ax = axes[1]
        brs = [p['base_rate'] for p in cov['per_position'] if p['deficit'] is not None]
        ds = [p['deficit'] for p in cov['per_position'] if p['deficit'] is not None]
        ax.scatter(brs, ds, c='#8e44ad', s=60, alpha=0.7); ax.axhline(0, color='gray', ls=':')
        ax.set_xlabel('Carry-in base rate'); ax.set_ylabel('Coverage deficit')
        if cov['corr'] is not None: ax.set_title(f"r={cov['corr']:.2f}")
        ax.grid(alpha=0.3)
        fig.suptitle(f'PUMA Coverage Deficit — {fmt}', y=1.02); fig.tight_layout()
        figs[f'coverage_{fmt}'] = fig

    # Fig 6: Chain length sweep (generalization boundary)
    sweep_cls = sorted(set(int(k.split('_')[-2]) for k in all_final
                           if 'chain_sweep' in k and k.endswith('confidence')))
    if sweep_cls:
        fig, ax = plt.subplots(figsize=(10, 6))
        for mt, col, mk in [('puma', '#8e44ad', 's'), ('random', '#3498db', 'o')]:
            cls, accs = [], []
            for cl in sweep_cls:
                r = all_final.get(_fk('diffusion', mt, f'chain_sweep_{cl}_confidence'))
                if r:
                    cls.append(cl); accs.append(r['accuracy'])
            if cls:
                ax.plot(cls, accs, f'-{mk}', color=col, label=mt, ms=8, lw=2, alpha=0.8)
        # Add full_propagate as rightmost point
        for mt, col, mk in [('puma', '#8e44ad', 's'), ('random', '#3498db', 'o')]:
            fp = all_final.get(_fk('diffusion', mt, f'corner_full_propagate_confidence'))
            if fp:
                ax.plot(ANS_LEN, fp['accuracy'], mk, color=col, ms=12, alpha=0.8,
                        markeredgecolor='black', markeredgewidth=1.5)
                ax.annotate(f"full_prop\n{fp['accuracy']:.3f}",
                           (ANS_LEN, fp['accuracy']), fontsize=7,
                           textcoords='offset points', xytext=(10, 5))
        ax.set_xlabel('Min chain length'); ax.set_ylabel('Accuracy')
        ax.set_title(f'Chain Length Sweep — Generalization Boundary')
        ax.legend(fontsize=10); ax.grid(alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        fig.tight_layout(); figs[f'chain_sweep_{fmt}'] = fig

    return figs


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Run
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _fk(obj, fmt, mt='', dp=''):
    return f"ar_{fmt}" if obj == 'ar' else f"{obj}_{fmt}_{mt}_{dp}"

def run(tag=''):
    exp_name = f"{EXP_NAME}_{tag}" if tag else EXP_NAME
    print(f"\n{'='*70}\n  {exp_name}")
    print(f"  ND={ND} Model={N_LAYER}L/{N_EMBD}D/{N_HEAD}H data={DATA_MODE}")
    print(f"  N_TRAIN={N_TRAIN} N_TEST={N_TEST} epochs={MAX_EPOCHS}")
    print(f"  masks={MASK_TYPES} decode={DECODE_POLICIES}")
    print(f"{'='*70}")
    mount_drive(); torch.manual_seed(SEED); random.seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed(SEED)
    tok = build_tok(); sample = _fmt_plain(10**(ND-1), 10**(ND-1))
    max_len = len(tok.encode(sample))
    all_dyn, all_final = {}, {}

    for fmt in FORMATS:
        if DATA_MODE == 'natural':
            train_data = gen_data_natural(N_TRAIN, fmt, seed=SEED)
        else:
            train_data = gen_data(N_TRAIN, fmt, seed=SEED)
        test_data = gen_test_data(N_TEST, fmt, seed=9000)
        # Show carry-in base rate distribution
        ci_counts = [0] * ANS_LEN; ci_totals = [0] * ANS_LEN
        for s in test_data:
            a, b = _parse_operands(s); ci = _carry_at_answer_pos(a, b, fmt)
            for j in range(ANS_LEN):
                ci_totals[j] += 1
                if ci[j]: ci_counts[j] += 1
        labs = _pos_labels(fmt)
        br_str = ' '.join(f"{labs[j]}={ci_counts[j]/max(ci_totals[j],1):.2f}" for j in range(ANS_LEN))
        print(f"\n  [{fmt}] train={len(train_data)} test={len(test_data)} mode={DATA_MODE}")
        print(f"    carry-in base rates: {br_str}")

        if RUN_AR:
            key = _fk('ar', fmt); print(f"\n▶ {key}")
            m, d = train_model('ar', tok, train_data, test_data, max_len, fmt)
            all_dyn[key] = d
            r = _quick_gen(m, tok, test_data, 'ar', fmt, device=DEVICE)
            all_final[_fk('ar', fmt, '', 'ar')] = r
            print(f"  AR acc: {r['accuracy']:.4f}")
            del m; torch.cuda.empty_cache() if torch.cuda.is_available() else None

        for mt in MASK_TYPES:
            kb = _fk('diffusion', fmt, mt, 'confidence')
            print(f"\n{'━'*60}\n▶ {kb}\n{'━'*60}")
            m, d = train_model('diffusion', tok, train_data, test_data, max_len, fmt, mask_type=mt)
            all_dyn[kb] = d

            # Standard + heavy eval with all decode policies
            heavy = gen_carry_heavy_test(N_TEST, fmt, seed=7000, min_max_chain=3)
            for dp in DECODE_POLICIES:
                for name, data in [('standard', test_data), ('heavy', heavy)]:
                    ps = gen_eval_with_stats(m, tok, data, fmt, max_len, decode_policy=dp, device=DEVICE)
                    acc = sum(r['correct'] for r in ps) / len(ps)
                    key = _fk('diffusion', fmt, mt, f'{name}_{dp}')
                    all_final[key] = {'accuracy': acc, 'n': len(ps), 'stratified': stratify_results(ps)}
                    print(f"    {name} {dp}: {acc:.4f}")
                    if name == 'standard':
                        for sn, bk in stratify_results(ps).items():
                            parts = [f"{k}={v['acc']:.4f}(n={v['n']})" for k, v in bk.items()]
                            print(f"      {sn}: {' '.join(parts)}")

            # Corner cases
            for cat in ['msb_chain', 'full_propagate', 'long_chain']:
                cc = gen_corner_case_test(N_TEST, fmt, seed=6000, category=cat)
                if not cc: continue
                for dp in DECODE_POLICIES:
                    ps = gen_eval_with_stats(m, tok, cc, fmt, max_len, decode_policy=dp, device=DEVICE)
                    acc = sum(r['correct'] for r in ps) / len(ps)
                    all_final[_fk('diffusion', fmt, mt, f'corner_{cat}_{dp}')] = {'accuracy': acc, 'n': len(cc)}
                    print(f"    corner/{cat} {dp}: {acc:.4f} (n={len(cc)})")

            # ── Chain length sweep (PUMA generalization boundary) ──
            print(f"  Chain length sweep...")
            sweep_lengths = [2, 3, 4, 6, 8, 12]
            if ND >= 24: sweep_lengths += [16, 20]
            if ND >= 32: sweep_lengths += [24, 28]
            sweep_lengths = [cl for cl in sweep_lengths if cl <= ND]
            sweep_n = min(500, N_TEST)
            for min_cl in sweep_lengths:
                cc = gen_min_chain_test(sweep_n, fmt, seed=6500+min_cl, min_chain=min_cl)
                if not cc: continue
                for dp in DECODE_POLICIES:
                    ps = gen_eval_with_stats(m, tok, cc, fmt, max_len, decode_policy=dp, device=DEVICE)
                    acc = sum(r['correct'] for r in ps) / len(ps)
                    key = _fk('diffusion', fmt, mt, f'chain_sweep_{min_cl}_{dp}')
                    all_final[key] = {'accuracy': acc, 'n': len(cc), 'min_chain': min_cl}
                    print(f"    chain>={min_cl:2d} {dp}: {acc:.4f} (n={len(cc)})")

            # Counterfactual
            cf = gen_counterfactual_pairs(200, seed=SEED+42)
            cfr = eval_counterfactual(m, tok, cf, fmt, max_len, device=DEVICE)
            all_final[_fk('diffusion', fmt, mt, 'cf')] = cfr
            print(f"    CF: flip={cfr['prediction_flip_rate']:.3f} acc(c=0)={cfr['acc_carry_in_0']:.3f} "
                  f"acc(c=1)={cfr['acc_carry_in_1']:.3f}")

            # ── Error localization on chain sweep failures ──
            print(f"  Error localization...")
            for min_cl in [4, 8, 12, 16, 20, 24]:
                if min_cl > ND: continue
                cc = gen_min_chain_test(500, fmt, seed=6500+min_cl, min_chain=min_cl)
                if not cc: continue
                ps = gen_eval_with_stats(m, tok, cc, fmt, max_len, decode_policy='confidence', device=DEVICE)
                el = analyse_error_localization(ps, fmt)
                key = _fk('diffusion', fmt, mt, f'error_loc_{min_cl}')
                all_final[key] = el
                if el['total_errors'] > 0:
                    parts = [f"{k}={v:.2f}" for k, v in el.items() if k != 'total_errors' and isinstance(v, float)]
                    print(f"    chain>={min_cl}: {el['total_errors']} errors — {' '.join(parts)}")

            # ── Confidence calibration ──
            print(f"  Confidence calibration...")
            # Use heavy + chain sweep samples for enough wrong predictions
            cal_samples = gen_carry_heavy_test(N_TEST, fmt, seed=7000, min_max_chain=3)
            cal = analyse_confidence_calibration(m, tok, cal_samples, fmt, max_len, device=DEVICE)
            all_final[_fk('diffusion', fmt, mt, 'calibration')] = cal
            for cl_bin, info in cal.items():
                mc = info['mean_conf_correct']; mw = info['mean_conf_wrong']
                oc = info.get('overconfident_wrong')
                if mw is not None:
                    print(f"    {cl_bin}: conf_ok={mc:.3f} conf_err={mw:.3f} "
                          f"overconf={oc:.1%} (n_err={info['n_wrong']})")

            # Carry rarity
            ps_conf = gen_eval_with_stats(m, tok, test_data, fmt, max_len, decode_policy='confidence', device=DEVICE)
            rarity = analyse_carry_rarity(ps_conf, test_data, fmt)
            all_final[_fk('diffusion', fmt, mt, 'carry_rarity')] = rarity
            print(f"    Rarity corr: {rarity['corr']:.3f}" if rarity['corr'] else "    Rarity corr: N/A")
            for bn, bd in rarity['binned'].items():
                print(f"      {bn}: gap={bd['mean_gap']:+.4f}")

            # PUMA coverage (puma only)
            if mt == 'puma':
                cov = simulate_puma_coverage(m, tok, test_data, fmt, max_len, device=DEVICE)
                all_final[_fk('diffusion', fmt, mt, 'coverage')] = cov
                print(f"    Coverage corr: {cov['corr']:.3f}" if cov['corr'] else "    Coverage corr: N/A")
                labs = _pos_labels(fmt)
                for p in cov['per_position']:
                    if p['deficit'] is not None:
                        print(f"      {labs[p['position']]}: br={p['base_rate']:.2f} "
                              f"cov(c1)={p['cov_c1']:.3f} deficit={p['deficit']:+.4f}")

            del m; torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # ── Curriculum training (PUMA→Random switchover sweep) ──
        if RUN_CURRICULUM:
            print(f"\n{'━'*60}\n  Curriculum Training\n{'━'*60}")
            for cur_label, cur_schedule in CURRICULUM_MODES:
                kb = _fk('diffusion', fmt, cur_label, 'confidence')
                print(f"\n▶ curriculum: {cur_label}")
                m, d = train_model('diffusion', tok, train_data, test_data, max_len,
                                   fmt, mask_type='puma', curriculum=cur_schedule, device=DEVICE)
                all_dyn[kb] = d
                # Eval: standard + corner + chain sweep
                for dp in DECODE_POLICIES:
                    for name, data in [('standard', test_data)]:
                        ps = gen_eval_with_stats(m, tok, data, fmt, max_len, decode_policy=dp, device=DEVICE)
                        acc = sum(r['correct'] for r in ps) / len(ps)
                        key = _fk('diffusion', fmt, cur_label, f'{name}_{dp}')
                        all_final[key] = {'accuracy': acc, 'n': len(ps)}
                        print(f"    {name} {dp}: {acc:.4f}")
                for cat in ['msb_chain', 'full_propagate']:
                    cc = gen_corner_case_test(N_TEST, fmt, seed=6000, category=cat)
                    if not cc: continue
                    for dp in DECODE_POLICIES:
                        ps = gen_eval_with_stats(m, tok, cc, fmt, max_len, decode_policy=dp, device=DEVICE)
                        acc = sum(r['correct'] for r in ps) / len(ps)
                        all_final[_fk('diffusion', fmt, cur_label, f'corner_{cat}_{dp}')] = {'accuracy': acc, 'n': len(cc)}
                        print(f"    corner/{cat} {dp}: {acc:.4f}")
                for min_cl in sweep_lengths:
                    cc = gen_min_chain_test(sweep_n, fmt, seed=6500+min_cl, min_chain=min_cl)
                    if not cc: continue
                    ps = gen_eval_with_stats(m, tok, cc, fmt, max_len, decode_policy='confidence', device=DEVICE)
                    acc = sum(r['correct'] for r in ps) / len(ps)
                    all_final[_fk('diffusion', fmt, cur_label, f'chain_sweep_{min_cl}_confidence')] = {
                        'accuracy': acc, 'n': len(cc), 'min_chain': min_cl}
                    print(f"    chain>={min_cl:2d}: {acc:.4f}")
                del m; torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Figures
        figs = make_figures(all_dyn, all_final, fmt)

    # Save
    sd = {'config': {k: globals()[k] for k in ['ND','ANS_LEN','N_TRAIN','N_TEST','MAX_EPOCHS',
           'BATCH_SIZE','N_LAYER','N_HEAD','N_EMBD','MASK_TYPES','DECODE_POLICIES','DATA_MODE']}}
    for k, v in all_dyn.items():
        sd[f'dyn_{k}'] = {'checkpoints': v['checkpoints'], 'train_loss': v['train_loss'],
                          'gen_checkpoints': v.get('gen_checkpoints', [])}
    for k, v in all_final.items(): sd[f'final_{k}'] = v
    save_results(exp_name, sd, figures=figs)

    # Summary
    print(f"\n{'='*70}\n  SUMMARY\n{'='*70}")
    for fmt in FORMATS:
        all_conditions = list(MASK_TYPES)
        if RUN_CURRICULUM:
            all_conditions += [c[0] for c in CURRICULUM_MODES]
        print(f"\n  ── {fmt} ──")
        print(f"  {'Test':<35s}", end='')
        for mt in all_conditions: print(f" {mt:>12s}", end='')
        print()
        for dp in DECODE_POLICIES:
            for tt in ['standard', 'heavy', 'corner_msb_chain', 'corner_full_propagate']:
                key_parts = [_fk('diffusion', fmt, mt, f'{tt}_{dp}') for mt in all_conditions]
                accs = [all_final.get(k, {}).get('accuracy') for k in key_parts]
                if any(a is not None for a in accs):
                    print(f"  {tt+'_'+dp:<35s}", end='')
                    for a in accs: print(f" {a:>12.4f}" if a is not None else f" {'N/A':>12s}", end='')
                    print()
            # Chain sweep
            for min_cl in [2, 3, 4, 6, 8, 12, 16, 20, 24, 28]:
                key_parts = [_fk('diffusion', fmt, mt, f'chain_sweep_{min_cl}_{dp}') for mt in all_conditions]
                accs = [all_final.get(k, {}).get('accuracy') for k in key_parts]
                if any(a is not None for a in accs):
                    print(f"  {'chain>='+str(min_cl)+'_'+dp:<35s}", end='')
                    for a in accs: print(f" {a:>12.4f}" if a is not None else f" {'N/A':>12s}", end='')
                    print()

        # Error localization summary
        print(f"\n  ── Error Localization ──")
        for mt in MASK_TYPES:
            print(f"  {mt}:")
            for min_cl in [4, 8, 12, 16, 20, 24]:
                el = all_final.get(_fk('diffusion', fmt, mt, f'error_loc_{min_cl}'))
                if el and el.get('total_errors', 0) > 0:
                    parts = [f"{k}={v:.2f}" for k, v in el.items() if k != 'total_errors' and isinstance(v, float)]
                    print(f"    cl>={min_cl}: n={el['total_errors']} {' '.join(parts)}")

    return all_dyn, all_final


if __name__ == '__main__':
    args = parse_args()
    seeds = args.seeds if args.seeds else [SEED]
    for si, seed in enumerate(seeds):
        globals()['SEED'] = seed
        t = f"{args.tag}_s{seed}" if args.tag and len(seeds) > 1 else args.tag
        if len(seeds) > 1: print(f"\n{'#'*70}\n# Seed {seed} ({si+1}/{len(seeds)})\n{'#'*70}")
        run(tag=t)
