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
    Diff-ordered-conf, Diff-ordered-lsb,
    Diff-confidence-conf, Diff-confidence-lsb,   ← PUMA-style adaptive
    Diff-msb-conf, Diff-msb-lsb                  ← anti-oracle baseline
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
#   train_with_dynamics  — epoch-based loop + periodic probe + ordered masking
#   final_evaluate       — per-position & carry-conditional analysis
#   _analyse_orders      — position-level decode rank (core's is scratchpad-level)
#   probe_per_position   — per-position loss/acc/conf probe
#   probe_partial_mask   — partial-mask confidence analysis

EXP_NAME = 'exp_addition_v2_dynamics'

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Config
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ND = 8
ANS_LEN = ND + 1                # 9

N_TRAIN = 2000
N_TEST = 500

# ── Training (epoch-based) ──
BATCH_SIZE = 200                # 2000 / 200 = 10 batches/epoch
MAX_EPOCHS = 1500               # fixed budget, no early stopping
EVAL_EVERY = 50                 # probe eval every 50 epochs
LOG_EVERY = 20                  # print train loss every 20 epochs

FORMATS = ['plain', 'reverse']

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

MASK_TYPES = ['random', 'ordered', 'confidence', 'msb']
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


@torch.no_grad()
def probe_partial_mask(model, tokenizer, test_samples, fmt, max_len,
                       fracs=None, device=None):
    """Diffusion: confidence at different masking levels (MSB masked first)."""
    if device is None: device = DEVICE
    if fracs is None: fracs = [0.2, 0.4, 0.6, 0.8, 1.0]
    model.eval()
    mask_id = tokenizer.special_ids['mask']
    ids_all, ans_all = encode_samples(test_samples, tokenizer, max_len)
    ids_all, ans_all = ids_all.to(device), ans_all.to(device)

    results = {}
    for frac in fracs:
        nm = max(1, int(math.ceil(frac * ANS_LEN)))
        conf_s = torch.zeros(ANS_LEN, device=device)
        acc_s = torch.zeros(ANS_LEN, device=device)
        cnt = torch.zeros(ANS_LEN, device=device)
        for st in range(0, len(test_samples), 128):
            en = min(st+128, len(test_samples))
            ids, ans = ids_all[st:en], ans_all[st:en]
            B = ids.shape[0]
            x = ids.clone()
            mf = torch.zeros(B, ANS_LEN, dtype=torch.bool, device=device)
            for b in range(B):
                a = ans[b].item()
                if fmt == 'plain':
                    for j in range(min(nm, ANS_LEN)):
                        x[b, a+j] = mask_id; mf[b, j] = True
                else:
                    for j in range(min(nm, ANS_LEN)):
                        x[b, a+ANS_LEN-1-j] = mask_id; mf[b, ANS_LEN-1-j] = True
            logits = model(x)
            for b in range(B):
                a = ans[b].item()
                for j in range(ANS_LEN):
                    if mf[b, j]:
                        cl = logits[b, a+j].clone()
                        cl[mask_id] = -float('inf')
                        p = F.softmax(cl, dim=-1)
                        conf_s[j] += p.max().item()
                        acc_s[j] += float(p.argmax().item() == ids[b, a+j].item())
                        cnt[j] += 1
        results[frac] = {
            'pos_conf': [conf_s[j].item()/cnt[j].item() if cnt[j]>0 else None
                         for j in range(ANS_LEN)],
            'pos_acc':  [acc_s[j].item()/cnt[j].item() if cnt[j]>0 else None
                         for j in range(ANS_LEN)],
            'n_masked': nm}
    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Training (epoch-based, fixed budget)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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

    dynamics = {'checkpoints': [], 'train_loss': []}
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

    # ── Eval at epoch 0 ──
    model.eval(); _do_eval(0); model.train()

    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_loss = 0.0
        epoch_n = 0

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

            else:  # diffusion
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

                elif mask_type == 'ordered':
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
                    # Anti-oracle: unmask MSB first → mask LSB side
                    # Opposite direction of 'ordered' (LSB-first)
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
                    # PUMA-style: mask positions the model is least
                    # confident about (= unmask high-confidence first).
                    # Probe in eval mode (no dropout noise) for clean ranking.
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
                        # rank by confidence (ascending) → mask least confident
                        # Exclude MASK token from confidence to avoid bias
                        confs = []
                        for p in a_pos:
                            lp = logits_probe[bi, p].clone()
                            lp[mask_id] = -float('inf')
                            confs.append(F.softmax(lp, dim=-1).max().item())
                        ranked = sorted(range(na), key=lambda k: confs[k])
                        for k in range(nm):
                            m[bi, a_pos[ranked[k]]] = True

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
            results.append({'correct': ps==gs, 'pos_correct': pc,
                           'n_carries': _count_carries(a, b)})

    n = len(results)
    acc = sum(r['correct'] for r in results) / max(n, 1)
    pos_acc = [sum(r['pos_correct'][j] for r in results)/max(n,1) for j in range(ANS_LEN)]
    by_nc = defaultdict(list)
    for r in results: by_nc[r['n_carries']].append(r['correct'])
    carry_acc = {nc: (sum(v)/len(v), len(v)) for nc, v in sorted(by_nc.items())}

    oa = None
    if all_orders:
        oc = torch.cat(all_orders, dim=0)
        pl = len(tokenizer.encode(test_samples[0].split('=')[0]+'='))
        oa = _analyse_orders(oc, pl, fmt)

    return {'accuracy': acc, 'n_samples': n, 'position_accuracy': pos_acc,
            'carry_accuracy': carry_acc, 'decode_order_analysis': oa}


def _analyse_orders(decode_orders, prefix_len, fmt):
    """Per-position decode rank analysis.
    Different from core.analyse_decode_order which is scratchpad-vs-final.
    This computes mean/median rank per answer position + LSB-first ratio.
    """
    N, S = decode_orders.shape
    rop = torch.full((N, ANS_LEN), float('nan'))
    for i in range(N):
        for s in range(S):
            r = decode_orders[i, s].item() - prefix_len
            if 0 <= r < ANS_LEN: rop[i, r] = s
    mr, mdr = [], []
    for j in range(ANS_LEN):
        v = rop[:, j][~rop[:, j].isnan()]
        mr.append(v.mean().item() if len(v)>0 else -1)
        mdr.append(v.median().item() if len(v)>0 else -1)
    li = ANS_LEN-1 if fmt=='plain' else 0
    mi = 0 if fmt=='plain' else ANS_LEN-1
    vb = ~(rop[:, li].isnan() | rop[:, mi].isnan())
    lf = (rop[vb, li] < rop[vb, mi]).float().mean().item() if vb.any() else -1
    rh = torch.zeros(ANS_LEN, S)
    for j in range(ANS_LEN):
        for r in range(S): rh[j, r] = (rop[:, j]==r).sum()
    return {'mean_rank': mr, 'median_rank': mdr,
            'lsb_first_ratio': lf, 'rank_histogram': rh}


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
    'diff-ord-con': '#9b59b6', 'diff-ord-lsb': '#e67e22',
    'diff-con-con': '#1abc9c', 'diff-con-lsb': '#16a085',
    'diff-msb-con': '#c0392b', 'diff-msb-lsb': '#d35400',
}

def _ck(obj, mt='', dp=''):
    if obj == 'ar':
        return 'ar'
    return f"diff-{mt[:3]}-{dp[:3]}"

def _fk(obj, fmt, mt='', dp=''):
    if obj == 'ar':
        return f"ar_{fmt}"
    return f"{obj}_{fmt}_{mt}_{dp}"


def make_figures(all_dyn, all_final, all_partial):
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
             ('diffusion','random','confidence'), ('diffusion','ordered','confidence'),
             ('diffusion','confidence','confidence'), ('diffusion','msb','confidence')]),
            ('Training mask comparison (conf decode)', [
             ('diffusion','random','confidence'), ('diffusion','ordered','confidence'),
             ('diffusion','confidence','confidence'), ('diffusion','msb','confidence')]),
            ('LSB decode comparison', [
             ('diffusion','random','lsb'), ('diffusion','ordered','lsb'),
             ('diffusion','confidence','lsb'), ('diffusion','msb','lsb')]),
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
                ax.set_title(f'mask={mt} | LSB-1st={oa["lsb_first_ratio"]:.2f}')
            fig.suptitle(f'Decode Order — {fmt}', fontsize=13, y=1.02)
            fig.tight_layout(); figs[f'decode_ord_{fmt}'] = fig

        # ── Fig 7: Partial mask ──
        for mt in MASK_TYPES:
            key = _fk('diffusion', fmt, mt, 'confidence')
            pcd = all_partial.get(key)
            if not pcd: continue
            fracs = sorted(pcd.keys())
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            for ax, m, yl in [(axes[0],'pos_conf','Confidence'),(axes[1],'pos_acc','Accuracy')]:
                for f in fracs:
                    ax.plot(range(ANS_LEN), pcd[f][m], '-o', ms=3,
                            label=f'mask={f:.0%}', alpha=0.8)
                ax.set_xticks(range(ANS_LEN)); ax.set_xticklabels(labels, fontsize=7)
                ax.set_ylabel(yl)
                if m=='pos_acc': ax.set_ylim(-0.05, 1.05)
                ax.legend(fontsize=6); ax.grid(alpha=0.3)
            fig.suptitle(f'Partial Mask — {fmt}, mask={mt}', fontsize=12, y=1.05)
            fig.tight_layout(); figs[f'partial_{fmt}_{mt}'] = fig

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
    print(f"  Budget:  {MAX_EPOCHS} epochs × {N_TRAIN//BATCH_SIZE} batches "
          f"= {MAX_EPOCHS * (N_TRAIN//BATCH_SIZE):,} iters")
    print(f"  Eval every {EVAL_EVERY} epochs → {MAX_EPOCHS//EVAL_EVERY} checkpoints")

    all_dyn, all_final, all_partial = {}, {}, {}

    for fmt in FORMATS:
        train_data = gen_data(N_TRAIN, fmt, seed=SEED)
        test_data = gen_data(N_TEST, fmt, seed=9000)
        cd = defaultdict(int)
        for s in train_data: a, b = _parse_operands(s); cd[_count_carries(a, b)] += 1
        print(f"\n  [{fmt}] N={len(train_data)}, carries={dict(sorted(cd.items()))}")

        # ── AR ──
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
            print(f"  Partial mask probe...")
            all_partial[kb] = probe_partial_mask(m, tok, test_data, fmt, max_len)

            for dp in DECODE_POLICIES:
                key = _fk('diffusion', fmt, mt, dp)
                print(f"\n  Final eval: {key}")
                r = final_evaluate(m, tok, test_data, 'diffusion', fmt, decode_policy=dp)
                all_final[key] = r
                print(f"  Acc: {r['accuracy']:.4f}")
                oa = r.get('decode_order_analysis')
                if oa:
                    labs = _pos_labels(fmt); mr = oa['mean_rank']
                    print(f"    LSB-1st: {oa['lsb_first_ratio']:.3f}")
                    print(f"    Rank: {' '.join(f'{labs[j]}={mr[j]:.1f}' for j in range(ANS_LEN))}")
            del m; torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── Figures ──
    print(f"\n{'='*70}\n  Generating figures...\n{'='*70}")
    figs = make_figures(all_dyn, all_final, all_partial)

    # ── Save ──
    sd = {'config': {'ND': ND, 'ANS_LEN': ANS_LEN, 'N_TRAIN': N_TRAIN,
                     'N_TEST': N_TEST, 'MAX_EPOCHS': MAX_EPOCHS, 'BATCH_SIZE': BATCH_SIZE}}
    for k, d in all_dyn.items():
        sd[f'dyn_{k}'] = {'checkpoints': d['checkpoints'], 'train_loss': d['train_loss']}
    for k, r in all_final.items():
        sr = {kk: vv for kk, vv in r.items() if kk != 'decode_order_analysis'}
        oa = r.get('decode_order_analysis')
        if oa: sr['decode_order'] = {kk: vv for kk, vv in oa.items() if kk != 'rank_histogram'}
        sd[f'final_{k}'] = sr
    for k, p in all_partial.items(): sd[f'partial_{k}'] = p
    save_results(EXP_NAME, sd, figures=figs)

    # ── Summary ──
    print(f"\n{'='*70}\n  SUMMARY\n{'='*70}")
    for fmt in FORMATS:
        labs = _pos_labels(fmt)
        print(f"\n  ━━ {fmt} ━━")
        print(f"  {'Condition':<25} {'Acc':>6}  {'TG':>12}  Position Accuracy")
        print(f"  {'─'*90}")
        all_conds = [('ar','','')]
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

    # ── Learning order ──
    print(f"\n  LEARNING ORDER (epoch to 90% per-pos acc)")
    for fmt in FORMATS:
        labs = _pos_labels(fmt)
        print(f"\n  [{fmt}]")
        for obj, mt in [('ar','')]+[('diffusion',mt) for mt in MASK_TYPES]:
            key = _fk(obj, fmt, mt, 'confidence')
            dyn = all_dyn.get(key)
            if not dyn: continue
            f90 = {}
            for j in range(ANS_LEN):
                for c in dyn['checkpoints']:
                    if c['pos_acc'][j] >= 0.9: f90[j] = c['epoch']; break
            parts = [f"{labs[j]}@ep{f90[j]}" if j in f90 else f"{labs[j]}@-"
                     for j in range(ANS_LEN)]
            print(f"    {_ck(obj,mt,'confidence'):<20} {' '.join(parts)}")

    plt.show()
    return all_dyn, all_final, all_partial

if __name__ == '__main__':
    run()
