"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Experiment 2 — 8-Digit Addition: Learning Dynamics Analysis
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Focus: HOW do AR vs diffusion learn addition, position by position?

  Research Questions:
    Q1. Per-position learning order: Does diffusion learn LSB first
        even with random masking? How does this compare to AR?
    Q2. Carry effect on learning: How much slower are carry positions?
    Q3. Confidence evolution: Which positions does diffusion become
        confident about first? Does this match the optimal (LSB→MSB)?
    Q4. Ordered masking: Does it accelerate the natural LSB→MSB
        curriculum, or does random masking already discover it?
    Q5. Format effect: Does reverse format change learning dynamics?
    Q6. Forced LSB decoding: Does it help regardless of training?

  Experimental conditions (10 total, absolute PE):
    ┌──────────────────┬───────────────┬──────────────────┐
    │ Condition         │ Training      │ Decoding         │
    ├──────────────────┼───────────────┼──────────────────┤
    │ AR                │ AR loss       │ L→R greedy       │
    │ Diff-rand-conf    │ Random mask   │ Confidence       │
    │ Diff-rand-lsb     │ Random mask   │ Forced LSB→MSB   │
    │ Diff-ord-conf     │ Ordered mask  │ Confidence       │
    │ Diff-ord-lsb      │ Ordered mask  │ Forced LSB→MSB   │
    └──────────────────┴───────────────┴──────────────────┘
    × 2 formats (plain, reverse)

  Key feature: periodic eval every EVAL_EVERY iterations captures
  per-position loss, accuracy, and confidence throughout training.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import sys, os, time, math, json, random, copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                if '__file__' in dir() else '.')

from core.tokenizer import CharTokenizer
from core.model import Transformer
from core.train_utils import (
    mount_drive, save_results,
    generate_ar, generate_diffusion, encode_samples,
    DEVICE,
)

EXP_NAME = 'exp_addition_v2_dynamics'

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Config
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ND = 8
ANS_LEN = ND + 1

N_TRAIN = 2000
N_TEST = 500

MAX_ITERS = 20_000
PATIENCE = 3_000
EVAL_EVERY = 500
LOG_INTERVAL = 200

FORMATS = ['plain', 'reverse']

N_LAYER = 6
N_HEAD = 6
N_EMBD = 384
DROPOUT = 0.2
POS_ENC = 'absolute'

BATCH_SIZE = 256
LR = 1e-3
MIN_LR = 1e-4
WARMUP = 100
GRAD_CLIP = 1.0

MASK_TYPES = ['random', 'ordered']
DECODE_POLICIES = ['confidence', 'lsb']

SEED = 42


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
    """
    Per-answer-position carry-in flag.
    carry_in[j] = True if position j receives a carry from its right neighbor.
    """
    flags = _carry_positions(a, b)  # flags[i] = carry out of LSB position i
    carry_in = [False] * ANS_LEN
    if fmt == 'plain':
        # ans = [overflow, digit_{ND-1}, ..., digit_0]
        # ans[0] = overflow ← carry out of position ND-1
        # ans[k] for k>=1 = digit at LSB position (ND-k) ← carry out of (ND-k-1)
        for k in range(ANS_LEN):
            lsb_pos = ND - k
            if lsb_pos == ND:
                carry_in[k] = flags[ND - 1] if ND - 1 < len(flags) else False
            elif 0 <= lsb_pos - 1 < len(flags):
                carry_in[k] = flags[lsb_pos - 1]
    else:  # reverse
        # ans = [digit_0(ones), digit_1, ..., digit_{ND-1}, overflow]
        # ans[0] = ones digit ← no carry in
        # ans[k] for k>=1 = digit at LSB position k ← carry out of (k-1)
        for k in range(ANS_LEN):
            if k == 0:
                carry_in[k] = False
            elif k - 1 < len(flags):
                carry_in[k] = flags[k - 1]
    return carry_in


def build_tok():
    chars = list('0123456789+=')
    return CharTokenizer(chars, {'mask': 'M', 'pad': 'P'})


def gen_pairs_balanced(n, seed):
    rng = random.Random(seed)
    pool = defaultdict(list)
    seen = set()
    attempts = max(n * 200, 100000)
    for _ in range(attempts):
        da = rng.randint(1, ND)
        db = rng.randint(1, ND)
        lo_a = 0 if da == 1 else 10 ** (da - 1)
        hi_a = 10 ** da - 1
        lo_b = 0 if db == 1 else 10 ** (db - 1)
        hi_b = 10 ** db - 1
        a = rng.randint(lo_a, hi_a)
        b = rng.randint(lo_b, hi_b)
        if (a, b) in seen:
            continue
        seen.add((a, b))
        nc = _count_carries(a, b)
        pool[nc].append((a, b))
    carry_counts = sorted(pool.keys())
    target = max(1, n // max(len(carry_counts), 1))
    out = []
    for nc in carry_counts:
        rng.shuffle(pool[nc])
        out.extend(pool[nc][:target])
    rng2 = random.Random(seed + 9999)
    hi = 10 ** ND - 1
    while len(out) < n:
        a, b = rng2.randint(0, hi), rng2.randint(0, hi)
        if (a, b) not in seen:
            out.append((a, b))
            seen.add((a, b))
    rng.shuffle(out)
    return out[:n]


def gen_data(n, fmt, seed):
    fn = FMT_FN[fmt]
    pairs = gen_pairs_balanced(n, seed)
    return [fn(a, b) for a, b in pairs]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Per-position eval probe
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@torch.no_grad()
def probe_per_position(model, tokenizer, test_samples, objective,
                       fmt, max_len, device=None):
    """
    Per-answer-position metrics WITHOUT generation.
    AR: teacher-forced loss.  Diffusion: fully-masked loss.
    """
    if device is None:
        device = DEVICE
    model.eval()

    mask_id = tokenizer.special_ids['mask']
    pad_id = tokenizer.special_ids['pad']

    ids_all, ans_starts_all = encode_samples(test_samples, tokenizer, max_len)
    ids_all = ids_all.to(device)
    ans_starts_all = ans_starts_all.to(device)

    carry_in_flags = []
    for s in test_samples:
        a, b = _parse_operands(s)
        carry_in_flags.append(_carry_at_answer_pos(a, b, fmt))

    pos_loss_sum = torch.zeros(ANS_LEN, device=device)
    pos_correct_sum = torch.zeros(ANS_LEN, device=device)
    pos_conf_sum = torch.zeros(ANS_LEN, device=device)
    pos_count = torch.zeros(ANS_LEN, device=device)

    pos_loss_carry = torch.zeros(ANS_LEN, device=device)
    pos_loss_nocarry = torch.zeros(ANS_LEN, device=device)
    pos_correct_carry = torch.zeros(ANS_LEN, device=device)
    pos_correct_nocarry = torch.zeros(ANS_LEN, device=device)
    pos_count_carry = torch.zeros(ANS_LEN, device=device)
    pos_count_nocarry = torch.zeros(ANS_LEN, device=device)

    bs = 128
    for start in range(0, len(test_samples), bs):
        end = min(start + bs, len(test_samples))
        ids = ids_all[start:end]
        ans_starts = ans_starts_all[start:end]
        B = ids.shape[0]

        if objective == 'ar':
            logits = model(ids[:, :-1])
            for b in range(B):
                a_start = ans_starts[b].item()
                ci = carry_in_flags[start + b]
                for j in range(ANS_LEN):
                    orig_pos = a_start + j
                    shift_pos = orig_pos - 1
                    if shift_pos < 0 or shift_pos >= logits.shape[1]:
                        continue
                    target = ids[b, orig_pos]
                    lp = F.log_softmax(logits[b, shift_pos], dim=-1)
                    loss_j = -lp[target].item()
                    pred = logits[b, shift_pos].argmax().item()
                    conf_j = F.softmax(logits[b, shift_pos], dim=-1).max().item()

                    pos_loss_sum[j] += loss_j
                    pos_correct_sum[j] += float(pred == target.item())
                    pos_conf_sum[j] += conf_j
                    pos_count[j] += 1
                    if ci[j]:
                        pos_loss_carry[j] += loss_j
                        pos_correct_carry[j] += float(pred == target.item())
                        pos_count_carry[j] += 1
                    else:
                        pos_loss_nocarry[j] += loss_j
                        pos_correct_nocarry[j] += float(pred == target.item())
                        pos_count_nocarry[j] += 1
        else:
            x_masked = ids.clone()
            for b in range(B):
                a_start = ans_starts[b].item()
                x_masked[b, a_start:a_start + ANS_LEN] = mask_id
            logits = model(x_masked)
            for b in range(B):
                a_start = ans_starts[b].item()
                ci = carry_in_flags[start + b]
                for j in range(ANS_LEN):
                    pos = a_start + j
                    target = ids[b, pos]
                    lp = F.log_softmax(logits[b, pos], dim=-1)
                    loss_j = -lp[target].item()
                    pred = logits[b, pos].argmax().item()
                    conf_j = F.softmax(logits[b, pos], dim=-1).max().item()

                    pos_loss_sum[j] += loss_j
                    pos_correct_sum[j] += float(pred == target.item())
                    pos_conf_sum[j] += conf_j
                    pos_count[j] += 1
                    if ci[j]:
                        pos_loss_carry[j] += loss_j
                        pos_correct_carry[j] += float(pred == target.item())
                        pos_count_carry[j] += 1
                    else:
                        pos_loss_nocarry[j] += loss_j
                        pos_correct_nocarry[j] += float(pred == target.item())
                        pos_count_nocarry[j] += 1

    safe = pos_count.clamp(min=1)
    safe_c = pos_count_carry.clamp(min=1)
    safe_nc = pos_count_nocarry.clamp(min=1)

    return {
        'pos_loss': (pos_loss_sum / safe).cpu().tolist(),
        'pos_acc': (pos_correct_sum / safe).cpu().tolist(),
        'pos_conf': (pos_conf_sum / safe).cpu().tolist(),
        'pos_loss_carry_in': (pos_loss_carry / safe_c).cpu().tolist(),
        'pos_loss_no_carry': (pos_loss_nocarry / safe_nc).cpu().tolist(),
        'pos_acc_carry_in': (pos_correct_carry / safe_c).cpu().tolist(),
        'pos_acc_no_carry': (pos_correct_nocarry / safe_nc).cpu().tolist(),
        'overall_loss': (pos_loss_sum.sum() / safe.sum()).item(),
        'overall_acc': (pos_correct_sum.sum() / safe.sum()).item(),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Partial mask confidence probe (diffusion only)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@torch.no_grad()
def probe_partial_mask_confidence(model, tokenizer, test_samples,
                                  fmt, max_len, mask_fractions=None,
                                  device=None):
    if device is None:
        device = DEVICE
    if mask_fractions is None:
        mask_fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
    model.eval()
    mask_id = tokenizer.special_ids['mask']

    ids_all, ans_starts_all = encode_samples(test_samples, tokenizer, max_len)
    ids_all = ids_all.to(device)
    ans_starts_all = ans_starts_all.to(device)

    results = {}
    for frac in mask_fractions:
        n_mask = max(1, int(math.ceil(frac * ANS_LEN)))
        conf_sum = torch.zeros(ANS_LEN, device=device)
        acc_sum = torch.zeros(ANS_LEN, device=device)
        count = torch.zeros(ANS_LEN, device=device)

        for start in range(0, len(test_samples), 128):
            end = min(start + 128, len(test_samples))
            ids = ids_all[start:end]
            ans_starts = ans_starts_all[start:end]
            B = ids.shape[0]
            x = ids.clone()
            mask_flags = torch.zeros(B, ANS_LEN, dtype=torch.bool, device=device)
            for b in range(B):
                a_start = ans_starts[b].item()
                if fmt == 'plain':
                    for j in range(min(n_mask, ANS_LEN)):
                        x[b, a_start + j] = mask_id
                        mask_flags[b, j] = True
                else:
                    for j in range(min(n_mask, ANS_LEN)):
                        pos = a_start + ANS_LEN - 1 - j
                        x[b, pos] = mask_id
                        mask_flags[b, ANS_LEN - 1 - j] = True
            logits = model(x)
            for b in range(B):
                a_start = ans_starts[b].item()
                for j in range(ANS_LEN):
                    if mask_flags[b, j]:
                        pos = a_start + j
                        target = ids[b, pos]
                        probs = F.softmax(logits[b, pos], dim=-1)
                        conf_sum[j] += probs.max().item()
                        acc_sum[j] += float(probs.argmax().item() == target.item())
                        count[j] += 1
        safe = count.clamp(min=1)
        results[frac] = {
            'pos_conf': (conf_sum / safe).cpu().tolist(),
            'pos_acc': (acc_sum / safe).cpu().tolist(),
            'n_masked': n_mask,
        }
    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Training with periodic eval
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def train_with_dynamics(
    objective, tokenizer, train_samples, test_samples,
    max_len, fmt,
    mask_type='random',
    eval_every=EVAL_EVERY,
    n_layer=N_LAYER, n_head=N_HEAD, n_embd=N_EMBD,
    dropout=DROPOUT, pos_enc=POS_ENC,
    batch_size=BATCH_SIZE, lr=LR,
    max_iters=MAX_ITERS, warmup_iters=WARMUP,
    min_lr=MIN_LR, grad_clip=GRAD_CLIP,
    patience=PATIENCE, min_delta=1e-4,
    log_interval=LOG_INTERVAL, device=None,
):
    if device is None:
        device = DEVICE

    train_ids, train_ans = encode_samples(train_samples, tokenizer, max_len)
    loader = DataLoader(
        TensorDataset(train_ids, train_ans),
        batch_size=batch_size, shuffle=True, drop_last=True)

    mask_id = tokenizer.special_ids['mask']
    pad_id = tokenizer.special_ids['pad']
    is_causal = (objective == 'ar')

    model = Transformer(
        vocab_size=len(tokenizer), block_size=max_len + 8,
        n_layer=n_layer, n_head=n_head, n_embd=n_embd,
        dropout=dropout, is_causal=is_causal, pos_enc=pos_enc,
    ).to(device)

    tag = f"{objective}" + (f"/{mask_type}" if objective == 'diffusion' else "")
    print(f"  [{tag}|{pos_enc}] params={model.n_params:,}, "
          f"seq_len={max_len}, eval_every={eval_every}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=0.1)

    def get_lr(it):
        if it < warmup_iters:
            return lr * it / max(warmup_iters, 1)
        ratio = (it - warmup_iters) / max(max_iters - warmup_iters, 1)
        return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * min(ratio, 1.0)))

    dynamics = {'checkpoints': [], 'train_loss': [], 'converged_at': None}
    best_loss, best_iter = float('inf'), 0
    best_state = None
    model.train()
    it = 0
    t0 = time.time()

    def _pos_labels():
        if fmt == 'plain':
            return ['MSB'] + [f'p{j}' for j in range(1, ANS_LEN-1)] + ['LSB']
        return ['LSB'] + [f'p{j}' for j in range(1, ANS_LEN-1)] + ['MSB']

    def _do_eval():
        probe = probe_per_position(
            model, tokenizer, test_samples, objective, fmt, max_len, device)
        dynamics['checkpoints'].append({'iter': it, **probe})
        elapsed = time.time() - t0
        print(f"    [eval@{it:5d}] loss={probe['overall_loss']:.4f} "
              f"acc={probe['overall_acc']:.4f} | {elapsed:.0f}s")
        labs = _pos_labels()
        acc_str = ' '.join(f"{labs[j]}={probe['pos_acc'][j]:.2f}"
                           for j in range(ANS_LEN))
        print(f"             acc: {acc_str}")

    _do_eval()

    while it < max_iters:
        for batch in loader:
            if it >= max_iters:
                break
            for pg in optimizer.param_groups:
                pg['lr'] = get_lr(it)

            ids = batch[0].to(device)
            ans_starts = batch[1].to(device)
            B, T = ids.shape

            if objective == 'ar':
                logits = model(ids[:, :-1])
                targets = ids[:, 1:]
                pos = torch.arange(T - 1, device=device).unsqueeze(0)
                loss_mask = pos >= (ans_starts.unsqueeze(1) - 1)
                loss_mask = loss_mask & (targets != pad_id)
                if loss_mask.sum() == 0:
                    it += 1; continue
                loss = F.cross_entropy(logits[loss_mask], targets[loss_mask])
            else:
                pos = torch.arange(T, device=device).unsqueeze(0)
                ans_mask = pos >= ans_starts.unsqueeze(1)

                if mask_type == 'random':
                    t_ratio = torch.rand(B, device=device)
                    m_probs = t_ratio.unsqueeze(1) * ans_mask.float()
                    m = torch.bernoulli(m_probs).bool()
                    no_m = ~(m.any(dim=1))
                    for b_idx in no_m.nonzero(as_tuple=True)[0]:
                        valid = ans_mask[b_idx].nonzero(as_tuple=True)[0]
                        if len(valid) > 0:
                            m[b_idx, valid[torch.randint(len(valid), (1,))]] = True
                elif mask_type == 'ordered':
                    t_ratio = torch.rand(B, device=device)
                    m = torch.zeros(B, T, dtype=torch.bool, device=device)
                    for b_idx in range(B):
                        a_start = ans_starts[b_idx].item()
                        a_positions = list(range(a_start, min(a_start + ANS_LEN, T)))
                        n_ans = len(a_positions)
                        n_mask = max(1, int(math.ceil(t_ratio[b_idx].item() * n_ans)))
                        if fmt == 'plain':
                            mask_positions = a_positions[:n_mask]
                        else:
                            mask_positions = a_positions[n_ans - n_mask:]
                        for p in mask_positions:
                            m[b_idx, p] = True
                else:
                    raise ValueError(f"Unknown mask_type: {mask_type}")

                x_m = ids.clone()
                x_m[m] = mask_id
                logits = model(x_m)
                if m.sum() == 0:
                    it += 1; continue
                loss = F.cross_entropy(logits[m], ids[m])

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            cur_loss = loss.item()
            if it % log_interval == 0:
                dynamics['train_loss'].append((it, cur_loss))
                if it % eval_every != 0:
                    elapsed = time.time() - t0
                    print(f"    it {it:5d} | loss {cur_loss:.4f} | "
                          f"lr {get_lr(it):.1e} | {elapsed:.0f}s")

            if it > 0 and it % eval_every == 0:
                model.eval()
                _do_eval()
                model.train()

            if cur_loss < best_loss - min_delta:
                best_loss = cur_loss
                best_iter = it
                best_state = {k: v.cpu().clone()
                              for k, v in model.state_dict().items()}
            elif it - best_iter >= patience and it > warmup_iters + patience:
                print(f"    ✓ Converged at it {it} (best={best_iter})")
                dynamics['converged_at'] = it
                if best_state:
                    model.load_state_dict(
                        {k: v.to(device) for k, v in best_state.items()})
                model.eval()
                _do_eval()
                return model, dynamics

            it += 1

    print(f"    ✓ Reached max_iters={max_iters}")
    dynamics['converged_at'] = max_iters
    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    _do_eval()
    return model, dynamics


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Final evaluation (generation-based)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@torch.no_grad()
def generate_diffusion_lsb(model, prefix_ids, n_tokens, mask_id,
                           fmt='plain', greedy=True, device=None):
    if device is None:
        device = DEVICE
    model.eval()
    B = prefix_ids.shape[0]
    T_pre = prefix_ids.shape[1]
    T = T_pre + n_tokens
    x = torch.full((B, T), mask_id, dtype=torch.long, device=device)
    x[:, :T_pre] = prefix_ids.to(device)
    if fmt == 'plain':
        order = list(range(T_pre + n_tokens - 1, T_pre - 1, -1))
    else:
        order = list(range(T_pre, T_pre + n_tokens))
    orders = []
    for pos in order:
        logits = model(x)
        pos_logits = logits[:, pos, :].clone()
        pos_logits[:, mask_id] = -float('inf')
        tok = pos_logits.argmax(-1) if greedy else \
            torch.multinomial(F.softmax(pos_logits, dim=-1), 1).squeeze(-1)
        x[:, pos] = tok
        orders.append(torch.full((B,), pos, dtype=torch.long))
    return x, torch.stack(orders, dim=1)


def final_evaluate(model, tokenizer, test_samples, objective, fmt,
                   decode_policy='confidence', batch_size=128, device=None):
    if device is None:
        device = DEVICE
    mask_id = tokenizer.special_ids['mask']
    pad_id = tokenizer.special_ids['pad']
    model.eval()

    results = []
    all_orders = []

    for start in range(0, len(test_samples), batch_size):
        batch = test_samples[start:start + batch_size]
        B = len(batch)
        prefixes = [s.split('=')[0] + '=' for s in batch]
        penc = [tokenizer.encode(p) for p in prefixes]
        pmax = max(len(p) for p in penc)
        pids = torch.full((B, pmax), pad_id, dtype=torch.long)
        for i, e in enumerate(penc):
            pids[i, :len(e)] = torch.tensor(e)

        with torch.no_grad():
            if objective == 'ar':
                gen = generate_ar(model, pids, ANS_LEN, device)
                pred_ids = gen[:, pmax:pmax + ANS_LEN]
                batch_orders = None
            elif decode_policy == 'lsb':
                gen, batch_orders = generate_diffusion_lsb(
                    model, pids, ANS_LEN, mask_id,
                    fmt=fmt, greedy=True, device=device)
                pred_ids = gen[:, pmax:pmax + ANS_LEN]
            else:
                gen, _, info = generate_diffusion(
                    model, pids, ANS_LEN, mask_id,
                    policy='confidence', greedy=True, device=device)
                pred_ids = gen[:, pmax:pmax + ANS_LEN]
                batch_orders = info.get('orders')

        if batch_orders is not None:
            all_orders.append(batch_orders)

        for i in range(B):
            pred_str = tokenizer.decode(pred_ids[i].cpu().tolist())
            gold_ans = get_answer(batch[i], fmt)
            a, b_val = _parse_operands(batch[i])
            pos_correct = [pred_str[j] == gold_ans[j]
                           if j < len(pred_str) else False
                           for j in range(len(gold_ans))]
            results.append({
                'correct': pred_str == gold_ans,
                'pos_correct': pos_correct,
                'n_carries': _count_carries(a, b_val),
            })

    n = len(results)
    acc = sum(r['correct'] for r in results) / max(n, 1)
    pos_acc = []
    for j in range(ANS_LEN):
        vals = [r['pos_correct'][j] for r in results]
        pos_acc.append(sum(vals) / max(len(vals), 1))
    by_nc = defaultdict(list)
    for r in results:
        by_nc[r['n_carries']].append(r['correct'])
    carry_acc = {nc: (sum(v)/len(v), len(v)) for nc, v in sorted(by_nc.items())}

    decode_order_analysis = None
    if all_orders:
        orders_cat = torch.cat(all_orders, dim=0)
        prefix_len = len(tokenizer.encode(test_samples[0].split('=')[0] + '='))
        decode_order_analysis = _analyse_orders(orders_cat, prefix_len, fmt)

    return {
        'accuracy': acc, 'n_samples': n,
        'position_accuracy': pos_acc,
        'carry_accuracy': carry_acc,
        'decode_order_analysis': decode_order_analysis,
    }


def _analyse_orders(decode_orders, prefix_len, fmt):
    N, S = decode_orders.shape
    rank_of_pos = torch.full((N, ANS_LEN), float('nan'))
    for i in range(N):
        for step in range(S):
            rel = decode_orders[i, step].item() - prefix_len
            if 0 <= rel < ANS_LEN:
                rank_of_pos[i, rel] = step

    mean_rank, median_rank = [], []
    for j in range(ANS_LEN):
        valid = rank_of_pos[:, j][~rank_of_pos[:, j].isnan()]
        mean_rank.append(valid.mean().item() if len(valid) > 0 else -1)
        median_rank.append(valid.median().item() if len(valid) > 0 else -1)

    lsb_idx = ANS_LEN - 1 if fmt == 'plain' else 0
    msb_idx = 0 if fmt == 'plain' else ANS_LEN - 1
    valid_both = ~(rank_of_pos[:, lsb_idx].isnan() | rank_of_pos[:, msb_idx].isnan())
    lsb_first = (rank_of_pos[valid_both, lsb_idx] <
                 rank_of_pos[valid_both, msb_idx]).float().mean().item() if valid_both.any() else -1

    rank_hist = torch.zeros(ANS_LEN, S)
    for j in range(ANS_LEN):
        for r in range(S):
            rank_hist[j, r] = (rank_of_pos[:, j] == r).sum()

    return {
        'mean_rank': mean_rank, 'median_rank': median_rank,
        'lsb_first_ratio': lsb_first, 'rank_histogram': rank_hist,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Visualization
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _pos_labels(fmt):
    if fmt == 'plain':
        return ['MSB'] + [f'p{j}' for j in range(1, ANS_LEN-1)] + ['LSB']
    return ['LSB'] + [f'p{j}' for j in range(1, ANS_LEN-1)] + ['MSB']

COLORS = {
    'ar':             '#e74c3c',
    'diff-rand-conf': '#3498db',
    'diff-rand-lsb':  '#2ecc71',
    'diff-ord-conf':  '#9b59b6',
    'diff-ord-lsb':   '#e67e22',
}

def _cond_key(obj, mt, dp):
    if obj == 'ar': return 'ar'
    return f"diff-{mt[:3]}-{dp[:3]}"

def _full_key(obj, fmt, mt, dp):
    return f"{obj}_{fmt}_{mt}_{dp}"


def make_figures(all_dynamics, all_final, all_partial_conf):
    figs = {}
    # Use a colormap for per-position lines
    pos_cmap = plt.cm.coolwarm  # blue(LSB-easy) → red(MSB-hard)

    for fmt in FORMATS:
        labels = _pos_labels(fmt)

        def pos_color(j):
            """Color by semantic difficulty: LSB=blue, MSB=red."""
            if fmt == 'plain':
                # j=0 is MSB(hard), j=8 is LSB(easy) → reverse
                return pos_cmap(1.0 - j / (ANS_LEN - 1))
            else:
                # j=0 is LSB(easy), j=8 is MSB(hard)
                return pos_cmap(j / (ANS_LEN - 1))

        # ── Fig 1: Per-position LOSS trajectory ──────
        conditions = []
        for obj in ['ar', 'diffusion']:
            for mt in (['random'] if obj == 'ar' else MASK_TYPES):
                conditions.append((obj, mt))
        n_cond = len(conditions)

        fig, axes = plt.subplots(1, n_cond, figsize=(6 * n_cond, 5))
        if n_cond == 1: axes = [axes]
        for ax_i, (obj, mt) in enumerate(conditions):
            dp = 'confidence'
            key = _full_key(obj, fmt, mt, dp)
            dyn = all_dynamics.get(key)
            if not dyn: continue
            ax = axes[ax_i]
            ck = _cond_key(obj, mt, dp)
            iters = [c['iter'] for c in dyn['checkpoints']]
            for j in range(ANS_LEN):
                losses = [c['pos_loss'][j] for c in dyn['checkpoints']]
                ax.plot(iters, losses, '-', color=pos_color(j),
                        label=labels[j], linewidth=1.5)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('CE Loss')
            ax.set_title(ck)
            ax.legend(fontsize=5, ncol=3, loc='upper right')
            ax.grid(alpha=0.3)
        fig.suptitle(f'Per-Position Loss Trajectory — {fmt}', fontsize=13, y=1.02)
        fig.tight_layout()
        figs[f'pos_loss_traj_{fmt}'] = fig

        # ── Fig 2: Per-position ACCURACY trajectory ──
        fig, axes = plt.subplots(1, n_cond, figsize=(6 * n_cond, 5))
        if n_cond == 1: axes = [axes]
        for ax_i, (obj, mt) in enumerate(conditions):
            dp = 'confidence'
            key = _full_key(obj, fmt, mt, dp)
            dyn = all_dynamics.get(key)
            if not dyn: continue
            ax = axes[ax_i]
            ck = _cond_key(obj, mt, dp)
            iters = [c['iter'] for c in dyn['checkpoints']]
            for j in range(ANS_LEN):
                accs = [c['pos_acc'][j] for c in dyn['checkpoints']]
                ax.plot(iters, accs, '-', color=pos_color(j),
                        label=labels[j], linewidth=1.5)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Accuracy (teacher-forced/fully-masked)')
            ax.set_ylim(-0.05, 1.05)
            ax.set_title(ck)
            ax.legend(fontsize=5, ncol=3, loc='lower right')
            ax.grid(alpha=0.3)
        fig.suptitle(f'Per-Position Accuracy Trajectory — {fmt}', fontsize=13, y=1.02)
        fig.tight_layout()
        figs[f'pos_acc_traj_{fmt}'] = fig

        # ── Fig 3: Carry gap trajectory ──────────────
        fig, axes = plt.subplots(1, n_cond, figsize=(6 * n_cond, 5))
        if n_cond == 1: axes = [axes]
        for ax_i, (obj, mt) in enumerate(conditions):
            dp = 'confidence'
            key = _full_key(obj, fmt, mt, dp)
            dyn = all_dynamics.get(key)
            if not dyn: continue
            ax = axes[ax_i]
            ck = _cond_key(obj, mt, dp)
            iters = [c['iter'] for c in dyn['checkpoints']]
            for j in range(ANS_LEN):
                gaps = [c['pos_loss_carry_in'][j] - c['pos_loss_no_carry'][j]
                        for c in dyn['checkpoints']]
                ax.plot(iters, gaps, '-', color=pos_color(j),
                        label=labels[j], linewidth=1.5)
            ax.axhline(0, color='black', linewidth=0.5, ls='--')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Loss(carry) − Loss(no carry)')
            ax.set_title(ck)
            ax.legend(fontsize=5, ncol=3)
            ax.grid(alpha=0.3)
        fig.suptitle(f'Carry Effect on Loss — {fmt}  (+ = carry harder)',
                     fontsize=12, y=1.02)
        fig.tight_layout()
        figs[f'carry_gap_{fmt}'] = fig

        # ── Fig 4: Confidence evolution (diffusion) ──
        diff_conds = [(mt, _full_key('diffusion', fmt, mt, 'confidence'))
                      for mt in MASK_TYPES
                      if _full_key('diffusion', fmt, mt, 'confidence') in all_dynamics]
        if diff_conds:
            fig, axes = plt.subplots(1, len(diff_conds),
                                      figsize=(6 * len(diff_conds), 5))
            if len(diff_conds) == 1: axes = [axes]
            for ax_i, (mt, key) in enumerate(diff_conds):
                dyn = all_dynamics[key]
                ax = axes[ax_i]
                iters = [c['iter'] for c in dyn['checkpoints']]
                for j in range(ANS_LEN):
                    confs = [c['pos_conf'][j] for c in dyn['checkpoints']]
                    ax.plot(iters, confs, '-', color=pos_color(j),
                            label=labels[j], linewidth=1.5)
                ax.set_xlabel('Iteration')
                ax.set_ylabel('Confidence (fully masked)')
                ax.set_ylim(0, 1.05)
                ax.set_title(f'mask={mt}')
                ax.legend(fontsize=5, ncol=3, loc='lower right')
                ax.grid(alpha=0.3)
            fig.suptitle(f'Diffusion Confidence Evolution — {fmt}',
                         fontsize=13, y=1.02)
            fig.tight_layout()
            figs[f'conf_evolution_{fmt}'] = fig

        # ── Fig 5: Final per-position accuracy ───────
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax_i, (title, conds) in enumerate([
            ('AR vs Diffusion', [
                ('ar', 'random', 'confidence'),
                ('diffusion', 'random', 'confidence'),
                ('diffusion', 'ordered', 'confidence'),
            ]),
            ('Decoding policy', [
                ('diffusion', 'random', 'confidence'),
                ('diffusion', 'random', 'lsb'),
                ('diffusion', 'ordered', 'confidence'),
                ('diffusion', 'ordered', 'lsb'),
            ]),
        ]):
            ax = axes[ax_i]
            for obj, mt, dp in conds:
                key = _full_key(obj, fmt, mt, dp)
                r = all_final.get(key)
                if r and r.get('position_accuracy'):
                    ck = _cond_key(obj, mt, dp)
                    pa = r['position_accuracy']
                    ax.plot(range(ANS_LEN), pa, '-o',
                            color=COLORS.get(ck, '#333'),
                            label=f"{ck} ({r['accuracy']:.3f})",
                            alpha=0.8, markersize=4)
            ax.set_xticks(range(ANS_LEN))
            ax.set_xticklabels(labels, fontsize=7)
            ax.set_ylim(-0.05, 1.05)
            ax.set_ylabel('Generation Accuracy')
            ax.set_title(title)
            ax.legend(fontsize=7); ax.grid(alpha=0.3)
        fig.suptitle(f'Final Per-Position Accuracy — {fmt}', fontsize=13, y=1.02)
        fig.tight_layout()
        figs[f'final_pos_acc_{fmt}'] = fig

        # ── Fig 6: Decode order heatmaps ─────────────
        order_data = []
        for mt in MASK_TYPES:
            key = _full_key('diffusion', fmt, mt, 'confidence')
            r = all_final.get(key)
            if r and r.get('decode_order_analysis'):
                order_data.append((mt, r['decode_order_analysis']))
        if order_data:
            fig, axes = plt.subplots(1, len(order_data),
                                      figsize=(7 * len(order_data), 5))
            if len(order_data) == 1: axes = [axes]
            for ax, (mt, oa) in zip(axes, order_data):
                rh = oa['rank_histogram'].float()
                rh_norm = rh / rh.sum(dim=1, keepdim=True).clamp(min=1)
                im = ax.imshow(rh_norm.numpy(), aspect='auto',
                               cmap='YlOrRd', interpolation='nearest')
                ax.set_xlabel('Decode Step (0=first)')
                ax.set_ylabel('Answer Position')
                ax.set_yticks(range(ANS_LEN))
                ax.set_yticklabels(labels, fontsize=8)
                plt.colorbar(im, ax=ax, label='Fraction')
                ax.set_title(f'mask={mt} | LSB-first={oa["lsb_first_ratio"]:.2f}')
            fig.suptitle(f'Confidence Decode Order — {fmt}', fontsize=13, y=1.02)
            fig.tight_layout()
            figs[f'decode_order_{fmt}'] = fig

        # ── Fig 7: Partial mask confidence ────────────
        for mt in MASK_TYPES:
            key = _full_key('diffusion', fmt, mt, 'confidence')
            pc = all_partial_conf.get(key)
            if not pc: continue
            fracs = sorted(pc.keys())
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            for ax, metric, ylabel in [
                (axes[0], 'pos_conf', 'Confidence'),
                (axes[1], 'pos_acc', 'Accuracy'),
            ]:
                for frac in fracs:
                    vals = pc[frac][metric]
                    ax.plot(range(ANS_LEN), vals, '-o', markersize=3,
                            label=f'mask={frac:.0%} ({pc[frac]["n_masked"]}pos)',
                            alpha=0.8)
                ax.set_xticks(range(ANS_LEN))
                ax.set_xticklabels(labels, fontsize=7)
                ax.set_ylabel(ylabel)
                if metric == 'pos_acc': ax.set_ylim(-0.05, 1.05)
                ax.legend(fontsize=6); ax.grid(alpha=0.3)
            fig.suptitle(f'Partial Mask Analysis — {fmt}, mask={mt}\n'
                         f'(MSB masked first → LSB visible)',
                         fontsize=12, y=1.08)
            fig.tight_layout()
            figs[f'partial_mask_{fmt}_{mt}'] = fig

        # ── Fig 8: Training loss curves ──────────────
        fig, ax = plt.subplots(figsize=(10, 6))
        for obj, mt in conditions:
            dp = 'confidence'
            key = _full_key(obj, fmt, mt, dp)
            dyn = all_dynamics.get(key)
            if not dyn: continue
            ck = _cond_key(obj, mt, dp)
            iters_l = [x[0] for x in dyn['train_loss']]
            losses = [x[1] for x in dyn['train_loss']]
            ax.plot(iters_l, losses, '-', color=COLORS.get(ck, '#333'),
                    label=ck, alpha=0.7)
        ax.set_xlabel('Iteration'); ax.set_ylabel('Training Loss')
        ax.set_title(f'Training Loss — {fmt}')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        fig.tight_layout()
        figs[f'train_loss_{fmt}'] = fig

    return figs


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run():
    print("=" * 70)
    print("  EXP 2: 8-Digit Addition — Learning Dynamics")
    print("=" * 70)
    mount_drive()
    torch.manual_seed(SEED)

    tok = build_tok()
    sample = _fmt_plain(12345678, 87654321)
    max_len = len(tok.encode(sample))
    print(f"  ND={ND}, ANS_LEN={ANS_LEN}, max_len={max_len}")
    print(f"  Plain:   {sample}")
    print(f"  Reverse: {_fmt_reverse(12345678, 87654321)}")

    all_dynamics = {}
    all_final = {}
    all_partial_conf = {}

    for fmt in FORMATS:
        train_data = gen_data(N_TRAIN, fmt, seed=SEED)
        test_data = gen_data(N_TEST, fmt, seed=9000)

        carry_dist = defaultdict(int)
        for s in train_data:
            a, b = _parse_operands(s)
            carry_dist[_count_carries(a, b)] += 1
        print(f"\n  [{fmt}] Train: {len(train_data)}, "
              f"carries: {dict(sorted(carry_dist.items()))}")

        # ── AR ──
        key = _full_key('ar', fmt, 'random', 'confidence')
        print(f"\n{'━'*60}")
        print(f"▶ {key}")
        print(f"{'━'*60}")
        model_ar, dyn_ar = train_with_dynamics(
            'ar', tok, train_data, test_data, max_len=max_len, fmt=fmt)
        all_dynamics[key] = dyn_ar
        res = final_evaluate(model_ar, tok, test_data, 'ar', fmt)
        all_final[key] = res
        print(f"  Final gen accuracy: {res['accuracy']:.4f}")
        del model_ar
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # ── Diffusion variants ──
        for mask_type in MASK_TYPES:
            key_base = _full_key('diffusion', fmt, mask_type, 'confidence')
            print(f"\n{'━'*60}")
            print(f"▶ {key_base}")
            print(f"{'━'*60}")
            model_diff, dyn_diff = train_with_dynamics(
                'diffusion', tok, train_data, test_data,
                max_len=max_len, fmt=fmt, mask_type=mask_type)
            all_dynamics[key_base] = dyn_diff

            print(f"  Partial mask analysis...")
            pc = probe_partial_mask_confidence(
                model_diff, tok, test_data, fmt, max_len)
            all_partial_conf[key_base] = pc

            for dp in DECODE_POLICIES:
                key = _full_key('diffusion', fmt, mask_type, dp)
                print(f"\n  Final eval: {key}")
                res = final_evaluate(
                    model_diff, tok, test_data, 'diffusion', fmt,
                    decode_policy=dp)
                all_final[key] = res
                print(f"  Accuracy: {res['accuracy']:.4f}")
                oa = res.get('decode_order_analysis')
                if oa:
                    labs = _pos_labels(fmt)
                    mr = oa['mean_rank']
                    print(f"    LSB-first: {oa['lsb_first_ratio']:.3f}")
                    print(f"    Rank: {' '.join(f'{labs[j]}={mr[j]:.1f}' for j in range(ANS_LEN))}")

            del model_diff
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── Figures ──
    print(f"\n{'='*70}")
    print("  Generating figures...")
    figs = make_figures(all_dynamics, all_final, all_partial_conf)

    # ── Save ──
    save_data = {'config': {
        'ND': ND, 'ANS_LEN': ANS_LEN, 'N_TRAIN': N_TRAIN,
        'N_TEST': N_TEST, 'MAX_ITERS': MAX_ITERS,
    }}
    for key, dyn in all_dynamics.items():
        save_data[f'dynamics_{key}'] = {
            'checkpoints': dyn['checkpoints'],
            'train_loss': dyn['train_loss'],
            'converged_at': dyn['converged_at'],
        }
    for key, res in all_final.items():
        save_res = {k: v for k, v in res.items() if k != 'decode_order_analysis'}
        oa = res.get('decode_order_analysis')
        if oa:
            save_res['decode_order'] = {
                'mean_rank': oa['mean_rank'],
                'median_rank': oa['median_rank'],
                'lsb_first_ratio': oa['lsb_first_ratio'],
            }
        save_data[f'final_{key}'] = save_res
    for key, pc in all_partial_conf.items():
        save_data[f'partial_conf_{key}'] = pc
    save_results(EXP_NAME, save_data, figures=figs)

    # ── Summary ──
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")

    for fmt in FORMATS:
        print(f"\n  ━━ {fmt} ━━")
        labs = _pos_labels(fmt)
        print(f"  {'Condition':<25} {'Acc':>6}  Position Accuracy")
        print(f"  {'─'*25} {'─'*6}  {'─'*60}")
        for obj, mt, dp in [
            ('ar', 'random', 'confidence'),
            ('diffusion', 'random', 'confidence'),
            ('diffusion', 'random', 'lsb'),
            ('diffusion', 'ordered', 'confidence'),
            ('diffusion', 'ordered', 'lsb'),
        ]:
            key = _full_key(obj, fmt, mt, dp)
            r = all_final.get(key)
            if not r: continue
            ck = _cond_key(obj, mt, dp)
            pa = r.get('position_accuracy', [])
            pa_str = ' '.join(f"{pa[j]:.2f}" for j in range(len(pa)))
            print(f"  {ck:<25} {r['accuracy']:>6.3f}  {pa_str}")

        print(f"\n  Decode order (confidence):")
        for mt in MASK_TYPES:
            key = _full_key('diffusion', fmt, mt, 'confidence')
            r = all_final.get(key)
            oa = r.get('decode_order_analysis') if r else None
            if oa:
                mr = oa['mean_rank']
                print(f"    mask={mt}: " + ' '.join(
                    f"{labs[j]}={mr[j]:.1f}" for j in range(ANS_LEN)))
                print(f"    LSB-first ratio: {oa['lsb_first_ratio']:.3f}")

    # ── Key findings ──
    print(f"\n{'='*70}")
    print("  LEARNING ORDER (iteration to reach 90% per-pos acc)")
    print(f"{'='*70}")
    for fmt in FORMATS:
        labs = _pos_labels(fmt)
        print(f"\n  [{fmt}]")
        for obj in ['ar', 'diffusion']:
            for mt in (['random'] if obj == 'ar' else MASK_TYPES):
                dp = 'confidence'
                key = _full_key(obj, fmt, mt, dp)
                dyn = all_dynamics.get(key)
                if not dyn: continue
                ck = _cond_key(obj, mt, dp)
                first_90 = {}
                for j in range(ANS_LEN):
                    for c in dyn['checkpoints']:
                        if c['pos_acc'][j] >= 0.9:
                            first_90[j] = c['iter']
                            break
                parts = [f"{labs[j]}@{first_90.get(j, '>max')}"
                         for j in range(ANS_LEN)]
                print(f"    {ck:<20} {' '.join(parts)}")

    plt.show()
    return all_dynamics, all_final, all_partial_conf


if __name__ == '__main__':
    run()
