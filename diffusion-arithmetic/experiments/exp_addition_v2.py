"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Experiment 2 — 8-Digit Addition: Learning Dynamics Analysis
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Fair comparison design:

    Training unit: EPOCH (1 epoch = 1 full pass over N=2000 samples)
      - BATCH_SIZE=200 → 10 batches/epoch, no data waste
      - Both AR and Diffusion train for exactly MAX_EPOCHS
      - Logging, eval, everything in epoch units

    Three x-axes tracked per checkpoint:
      epoch           : data exposure (same for both)
      iteration       : gradient updates (same for both, = epoch × 10)
      token_gradients : actual supervised signals
                        AR ≈ 9/sample, Diff ≈ 4.5/sample → AR gets ~2×

    Plotting: default x=epoch, secondary x=token_gradients
    Stopping: fixed budget (no early stopping) + best model by train loss

  Conditions (×2 formats = 14 total):
    AR, Diff-random-conf, Diff-random-lsb,
    Diff-lsb-conf, Diff-lsb-lsb,
    Diff-confidence-conf, Diff-confidence-lsb,   ← single-probe adaptive
    Diff-msb-conf, Diff-msb-lsb                  ← anti-oracle baseline
    Diff-puma-conf, Diff-puma-lsb                 ← teacher-forced chain
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import sys, os, time, math, json, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
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
MASK_TYPES = ['random', 'lsb', 'confidence', 'msb', 'puma']
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
    p.add_argument('--seed', type=int, default=None)

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

def build_tok():
    return CharTokenizer(list('0123456789+='), {'mask': 'M', 'pad': 'P'})

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
    lo = 10**(ND - 1)    # 10000000 for ND=8
    hi = 10**ND - 1       # 99999999
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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Per-position eval probe
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@torch.no_grad()
def probe_per_position(model, tokenizer, test_samples, objective,
                       fmt, max_len, device=None):
    """
    AR: teacher-forced loss per position.
    Diffusion: fully-masked loss per position.
    """
    if device is None: device = DEVICE
    model.eval()
    mask_id = tokenizer.special_ids['mask']

    ids_all, ans_all = encode_samples(test_samples, tokenizer, max_len)
    ids_all, ans_all = ids_all.to(device), ans_all.to(device)

    ci_flags = [_carry_at_answer_pos(*_parse_operands(s), fmt) for s in test_samples]

    L  = torch.zeros(ANS_LEN, device=device)
    C  = torch.zeros(ANS_LEN, device=device)
    CF = torch.zeros(ANS_LEN, device=device)
    N  = torch.zeros(ANS_LEN, device=device)
    Lc, Ln = torch.zeros(ANS_LEN, device=device), torch.zeros(ANS_LEN, device=device)
    Cc, Cn = torch.zeros(ANS_LEN, device=device), torch.zeros(ANS_LEN, device=device)
    Nc, Nn = torch.zeros(ANS_LEN, device=device), torch.zeros(ANS_LEN, device=device)

    for st in range(0, len(test_samples), 128):
        en = min(st+128, len(test_samples))
        ids, ans = ids_all[st:en], ans_all[st:en]
        B = ids.shape[0]

        if objective == 'ar':
            logits = model(ids[:, :-1])
            for b in range(B):
                a_s = ans[b].item()
                ci = ci_flags[st+b]
                for j in range(ANS_LEN):
                    sp = a_s + j - 1
                    if sp < 0 or sp >= logits.shape[1]: continue
                    tgt = ids[b, a_s+j]
                    lp = F.log_softmax(logits[b, sp], dim=-1)
                    lj = -lp[tgt].item()
                    pred = logits[b, sp].argmax().item()
                    cj = F.softmax(logits[b, sp], dim=-1).max().item()
                    L[j] += lj; C[j] += float(pred == tgt.item()); CF[j] += cj; N[j] += 1
                    if ci[j]: Lc[j] += lj; Cc[j] += float(pred == tgt.item()); Nc[j] += 1
                    else:     Ln[j] += lj; Cn[j] += float(pred == tgt.item()); Nn[j] += 1
        else:
            xm = ids.clone()
            for b in range(B):
                a_s = ans[b].item()
                xm[b, a_s:a_s+ANS_LEN] = mask_id
            logits = model(xm)
            for b in range(B):
                a_s = ans[b].item()
                ci = ci_flags[st+b]
                for j in range(ANS_LEN):
                    pos = a_s + j
                    tgt = ids[b, pos]
                    lp = F.log_softmax(logits[b, pos], dim=-1)
                    lj = -lp[tgt].item()
                    pred = logits[b, pos].argmax().item()
                    # Exclude MASK token from confidence
                    cl = logits[b, pos].clone()
                    cl[mask_id] = -float('inf')
                    cj = F.softmax(cl, dim=-1).max().item()
                    L[j] += lj; C[j] += float(pred == tgt.item()); CF[j] += cj; N[j] += 1
                    if ci[j]: Lc[j] += lj; Cc[j] += float(pred == tgt.item()); Nc[j] += 1
                    else:     Ln[j] += lj; Cn[j] += float(pred == tgt.item()); Nn[j] += 1

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
    # Confidence ranking is only meaningful for diffusion (fully-masked probe).
    # AR's pos_conf is teacher-forced confidence — a different quantity entirely.
    return result


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
    loader = DataLoader(TensorDataset(train_ids, train_ans),
                        batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    bpe = len(loader)  # batches per epoch
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
    best_loss = float('inf')
    best_state = None
    it = 0
    tg = 0  # token-gradient count
    t0 = time.time()

    def _pos_labels():
        if fmt == 'plain':
            return ['MSB'] + [f'p{j}' for j in range(1, ANS_LEN-1)] + ['LSB']
        return ['LSB'] + [f'p{j}' for j in range(1, ANS_LEN-1)] + ['MSB']

    def _do_eval(epoch):
        probe = probe_per_position(
            model, tokenizer, test_samples, objective, fmt, max_len, device)
        dynamics['checkpoints'].append({
            'epoch': epoch, 'iter': it, 'token_gradients': tg, **probe})
        labs = _pos_labels()
        acc_str = ' '.join(f"{labs[j]}={probe['pos_acc'][j]:.2f}"
                           for j in range(ANS_LEN))
        print(f"    [eval ep {epoch:4d}] loss={probe['overall_loss']:.4f} "
              f"acc={probe['overall_acc']:.4f} | {time.time()-t0:.0f}s")
        print(f"      {acc_str}")

        # Generation accuracy tracking (less frequent, more expensive)
        if epoch % GEN_EVAL_EVERY == 0:
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
                for dp in DECODE_POLICIES:
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
                    ga_strs.append(f"{dp}={r['accuracy']:.3f}")
                print(f"      [gen] {' '.join(ga_strs)}")
            dynamics['gen_checkpoints'].append(gen_entry)

    # ── Eval at epoch 0 ──
    model.eval(); _do_eval(0); model.train()

    # ── PUMA streaming buffer (only used if mask_type == 'puma') ──
    puma_x0 = puma_z = puma_ans = puma_stage = None
    puma_pool_idx = 0  # index into pre-encoded training data for refreshing
    if mask_type == 'puma':
        # Pre-encode all training data for fast refresh
        puma_all_ids, puma_all_ans = encode_samples(train_samples, tokenizer, max_len)
        puma_all_ids = puma_all_ids.to(device)
        puma_all_ans = puma_all_ans.to(device)
        puma_perm = torch.randperm(len(train_samples))  # shuffle order

    def _puma_refresh(indices):
        """Initialize or refresh PUMA buffer entries from training pool."""
        nonlocal puma_x0, puma_z, puma_ans, puma_stage, puma_pool_idx, puma_perm
        if puma_x0 is None:
            # First-time init: allocate full buffer
            B = len(indices)
            T = puma_all_ids.shape[1]
            puma_x0 = torch.zeros(B, T, dtype=torch.long, device=device)
            puma_z = torch.zeros(B, T, dtype=torch.long, device=device)
            puma_ans = torch.zeros(B, dtype=torch.long, device=device)
            puma_stage = torch.zeros(B, dtype=torch.long, device=device)
        for bi in indices:
            si = puma_perm[puma_pool_idx % len(puma_perm)].item()
            puma_pool_idx += 1
            if puma_pool_idx >= len(puma_perm):
                puma_perm = torch.randperm(len(train_samples))
                puma_pool_idx = 0
            puma_x0[bi] = puma_all_ids[si]
            puma_z[bi] = puma_all_ids[si].clone()
            puma_ans[bi] = puma_all_ans[si]
            a_s = puma_all_ans[si].item()
            puma_z[bi, a_s:min(a_s + ANS_LEN, puma_z.shape[1])] = mask_id
            puma_stage[bi] = 0

    def _puma_advance(logits_det, K_cur):
        """Advance PUMA chains using detached logits. Zero overhead."""
        nonlocal puma_z, puma_stage
        B = puma_z.shape[0]
        refresh_list = []
        for bi in range(B):
            mpos = (puma_z[bi] == mask_id).nonzero(as_tuple=True)[0]
            if len(mpos) == 0:
                refresh_list.append(bi); continue
            # Vectorized confidence (excl MASK)
            lp = logits_det[bi, mpos].clone()
            lp[:, mask_id] = -float('inf')
            confs = F.softmax(lp, dim=-1).max(dim=-1).values
            # How many to reveal this step
            K_rem = max(K_cur - puma_stage[bi].item(), 1)
            n_reveal = max(1, math.ceil(len(mpos) / K_rem))
            # Threshold: also reveal positions with conf > τ
            above_tau = confs > PUMA_TAU
            _, topk_idx = confs.topk(min(n_reveal, len(mpos)))
            reveal = torch.zeros(len(mpos), dtype=torch.bool, device=device)
            reveal[topk_idx] = True
            reveal = reveal | above_tau
            # Reveal with ground truth
            reveal_pos = mpos[reveal]
            puma_z[bi, reveal_pos] = puma_x0[bi, reveal_pos]
            puma_stage[bi] += 1
            # Chain complete?
            if puma_stage[bi] >= K_cur or not (puma_z[bi] == mask_id).any():
                refresh_list.append(bi)
        if refresh_list:
            _puma_refresh(refresh_list)

    # Initialize PUMA buffer
    if mask_type == 'puma':
        _puma_refresh(list(range(BATCH_SIZE)))

    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_loss = 0.0
        epoch_n = 0
        # K scheduling for PUMA
        puma_K_cur = PUMA_K_START + int((PUMA_K_END - PUMA_K_START)
                                         * epoch / MAX_EPOCHS) if mask_type == 'puma' else 0

        if mask_type == 'puma':
            # ── PUMA: streaming buffer iterations ──
            for _ in range(bpe):
                for pg in optimizer.param_groups:
                    pg['lr'] = get_lr(it)
                # Current mask = wherever z has MASK tokens
                m = (puma_z == mask_id)
                if m.sum() == 0:
                    _puma_refresh(list(range(BATCH_SIZE)))
                    m = (puma_z == mask_id)
                # Forward on current chain state
                logits = model(puma_z)
                loss = F.cross_entropy(logits[m], puma_x0[m])
                tg += m.sum().item()
                # Backward
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                # Advance chains using same logits (zero overhead)
                _puma_advance(logits.detach(), puma_K_cur)

                epoch_loss += loss.item()
                epoch_n += 1
                it += 1
        else:
            # ── Standard mask types (random/lsb/msb/confidence) ──
            for batch in loader:
                for pg in optimizer.param_groups:
                    pg['lr'] = get_lr(it)

                ids = batch[0].to(device)
                ans_starts = batch[1].to(device)
                B, T = ids.shape

                if objective == 'ar':
                    logits = model(ids[:, :-1])
                    targets = ids[:, 1:]
                    pos = torch.arange(T-1, device=device).unsqueeze(0)
                    lm = pos >= (ans_starts.unsqueeze(1) - 1)
                    lm = lm & (targets != pad_id)
                    if lm.sum() == 0: it += 1; continue
                    loss = F.cross_entropy(logits[lm], targets[lm])
                    tg += lm.sum().item()

                else:  # diffusion (non-PUMA)
                    pos = torch.arange(T, device=device).unsqueeze(0)
                    ans_mask = pos >= ans_starts.unsqueeze(1)

                    if mask_type == 'random':
                        t_ratio = torch.rand(B, device=device)
                        m_probs = t_ratio.unsqueeze(1) * ans_mask.float()
                        m = torch.bernoulli(m_probs).bool()
                        no_m = ~(m.any(dim=1))
                        for bi in no_m.nonzero(as_tuple=True)[0]:
                            v = ans_mask[bi].nonzero(as_tuple=True)[0]
                            if len(v) > 0: m[bi, v[torch.randint(len(v),(1,))]] = True

                    elif mask_type == 'lsb':
                        t_ratio = torch.rand(B, device=device)
                        m = torch.zeros(B, T, dtype=torch.bool, device=device)
                        for bi in range(B):
                            a_s = ans_starts[bi].item()
                            a_pos = list(range(a_s, min(a_s + ANS_LEN, T)))
                            na = len(a_pos)
                            nm = max(1, int(math.ceil(t_ratio[bi].item() * na)))
                            if fmt == 'plain':
                                for p in a_pos[:nm]: m[bi, p] = True
                            else:
                                for p in a_pos[na-nm:]: m[bi, p] = True

                    elif mask_type == 'msb':
                        # Anti-oracle: unmask MSB first (opposite of 'lsb')
                        t_ratio = torch.rand(B, device=device)
                        m = torch.zeros(B, T, dtype=torch.bool, device=device)
                        for bi in range(B):
                            a_s = ans_starts[bi].item()
                            a_pos = list(range(a_s, min(a_s + ANS_LEN, T)))
                            na = len(a_pos)
                            nm = max(1, int(math.ceil(t_ratio[bi].item() * na)))
                            if fmt == 'plain':
                                for p in a_pos[na-nm:]: m[bi, p] = True
                            else:
                                for p in a_pos[:nm]: m[bi, p] = True

                    elif mask_type == 'confidence':
                        # Single-probe adaptive: fully-masked → rank → mask bottom
                        xm_probe = ids.clone()
                        for bi in range(B):
                            a_s = ans_starts[bi].item()
                            xm_probe[bi, a_s:min(a_s + ANS_LEN, T)] = mask_id
                        model.eval()
                        with torch.no_grad():
                            logits_probe = model(xm_probe)
                        model.train()

                        t_ratio = torch.rand(B, device=device)
                        m = torch.zeros(B, T, dtype=torch.bool, device=device)
                        for bi in range(B):
                            a_s = ans_starts[bi].item()
                            a_pos = list(range(a_s, min(a_s + ANS_LEN, T)))
                            na = len(a_pos)
                            nm = max(1, int(math.ceil(t_ratio[bi].item() * na)))
                            lp = logits_probe[bi, torch.tensor(a_pos, device=device)].clone()
                            lp[:, mask_id] = -float('inf')
                            confs = F.softmax(lp, dim=-1).max(dim=-1).values
                            ranked = confs.argsort()  # ascending
                            for k in range(nm):
                                m[bi, a_pos[ranked[k].item()]] = True

                    xm = ids.clone(); xm[m] = mask_id
                    logits = model(xm)
                    if m.sum() == 0: it += 1; continue
                    loss = F.cross_entropy(logits[m], ids[m])
                    tg += m.sum().item()

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

                epoch_loss += loss.item()
                epoch_n += 1
                it += 1

        # ── End of epoch ──
        avg_loss = epoch_loss / max(epoch_n, 1)

        # Best model snapshot (by own training loss)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % LOG_EVERY == 0:
            dynamics['train_loss'].append((epoch, avg_loss))
            print(f"    ep {epoch:4d}/{MAX_EPOCHS} | loss {avg_loss:.4f} | "
                  f"lr {get_lr(it):.1e} | tg {tg:,} | {time.time()-t0:.0f}s")

        if epoch % EVAL_EVERY == 0 and epoch < MAX_EPOCHS:
            model.eval(); _do_eval(epoch); model.train()

    # ── Load best model ──
    print(f"    ✓ Done {MAX_EPOCHS} epochs (best train loss: {best_loss:.4f})")
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
            results.append({'correct': ps==gs, 'pos_correct': pc,
                           'n_carries': _count_carries(a, b),
                           'carry_flags': ci})

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
        oa = _analyse_orders(oc, pl, fmt)

    return {'accuracy': acc, 'n_samples': n, 'position_accuracy': pos_acc,
            'carry_accuracy': carry_acc, 'decode_order_analysis': oa,
            'pos_acc_carry_in': pos_acc_carry_in,
            'pos_acc_no_carry': pos_acc_no_carry}


def _analyse_orders(decode_orders, prefix_len, fmt):
    """Per-position decode rank analysis.
    Computes:
      mean_rank:     mean decode step per position (over all samples)
      pairwise_conc: fraction of position pairs decoded in LSB→MSB order,
                     averaged over all 36 pairs × all samples
      rank_histogram: (ANS_LEN, S) count matrix
    """
    N, S = decode_orders.shape
    rop = torch.full((N, ANS_LEN), float('nan'))
    for i in range(N):
        for s in range(S):
            r = decode_orders[i, s].item() - prefix_len
            if 0 <= r < ANS_LEN: rop[i, r] = s
    mr = []
    for j in range(ANS_LEN):
        v = rop[:, j][~rop[:, j].isnan()]
        mr.append(v.mean().item() if len(v) > 0 else -1)

    # Pairwise concordance with LSB-first oracle.
    # For each pair (i, j), oracle says the more-LSB position decodes first.
    # plain:   higher index = more LSB → oracle: j before i (for i<j)
    # reverse: lower index  = more LSB → oracle: i before j (for i<j)
    total_conc = 0.0
    total_valid = 0
    for n in range(N):
        row = rop[n]
        if row.isnan().any():
            continue
        conc = 0
        for i in range(ANS_LEN):
            for j in range(i + 1, ANS_LEN):
                if fmt == 'plain':
                    conc += int(row[j].item() < row[i].item())
                else:
                    conc += int(row[i].item() < row[j].item())
        total_conc += conc
        total_valid += 1
    n_pairs = ANS_LEN * (ANS_LEN - 1) // 2  # 36
    pw_conc = total_conc / (total_valid * n_pairs) if total_valid > 0 else -1

    rh = torch.zeros(ANS_LEN, S)
    for j in range(ANS_LEN):
        for r in range(S): rh[j, r] = (rop[:, j] == r).sum()
    return {'mean_rank': mr, 'pairwise_concordance': pw_conc,
            'rank_histogram': rh}


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
    'diff-lsb-con': '#9b59b6', 'diff-lsb-lsb': '#e67e22',
    'diff-con-con': '#1abc9c', 'diff-con-lsb': '#16a085',
    'diff-msb-con': '#c0392b', 'diff-msb-lsb': '#d35400',
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

        # ── Fig 5: Final per-position accuracy ──
        fig, axes = plt.subplots(1, 3, figsize=(21, 5))
        for ai, (title, cs) in enumerate([
            ('AR vs Diffusion (conf decode)', [('ar','',''),
             ('diffusion','random','confidence'), ('diffusion','lsb','confidence'),
             ('diffusion','confidence','confidence'), ('diffusion','msb','confidence'),
             ('diffusion','puma','confidence')]),
            ('Training mask comparison (conf decode)', [
             ('diffusion','random','confidence'), ('diffusion','lsb','confidence'),
             ('diffusion','confidence','confidence'), ('diffusion','msb','confidence'),
             ('diffusion','puma','confidence')]),
            ('LSB decode comparison', [
             ('diffusion','random','lsb'), ('diffusion','lsb','lsb'),
             ('diffusion','confidence','lsb'), ('diffusion','msb','lsb'),
             ('diffusion','puma','lsb')]),
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
    print(f"  Eval every {EVAL_EVERY} epochs, gen eval every {GEN_EVAL_EVERY} epochs")
    print(f"  Formats: {FORMATS}  Masks: {MASK_TYPES}  Decode: {DECODE_POLICIES}")
    print(f"  AR: {'yes' if RUN_AR else 'skip'}")

    all_dyn, all_final = {}, {}

    for fmt in FORMATS:
        train_data = gen_data(N_TRAIN, fmt, seed=SEED)
        test_data = gen_test_data(N_TEST, fmt, seed=9000)
        cd = defaultdict(int)
        for s in train_data: a, b = _parse_operands(s); cd[_count_carries(a, b)] += 1
        print(f"\n  [{fmt}] train N={len(train_data)}, carries={dict(sorted(cd.items()))}")
        cd_test = defaultdict(int)
        for s in test_data: a, b = _parse_operands(s); cd_test[_count_carries(a, b)] += 1
        print(f"  [{fmt}] test  N={len(test_data)} (full {ND}-digit), carries={dict(sorted(cd_test.items()))}")

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
                if oa:
                    labs = _pos_labels(fmt); mr = oa['mean_rank']
                    print(f"    LSB concordance: {oa['pairwise_concordance']:.3f}")
                    print(f"    Rank: {' '.join(f'{labs[j]}={mr[j]:.1f}' for j in range(ANS_LEN))}")
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
    run(tag=args.tag)
