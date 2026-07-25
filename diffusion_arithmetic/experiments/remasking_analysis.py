"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Remasking analysis on addition (inference-only, existing checkpoints)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Rebuttal experiment for 3GQB Q1 / meta-review item 1. Evaluates whether
remasking-capable decoders can correct the premature high-confidence commits
identified in the paper, at the CEILING of the remasking class: the exact
leave-one-out (LOO) conditional q_j = p(x^j = y^j | y ⊕ m_j), computed
directly with one forward pass per position (the provably-optimal target of
learned per-token-quality heads; no such head can exceed it).

Components
  (A) Observability: replay the confidence decode trajectory, computing the
      LOO score of every committed position at regular audit points. For the
      canonical failure (single ±1 error at the chain-dependent g/k cell),
      measures WHEN the error first becomes detectable (q < τ) relative to
      when it was committed and when the refuting evidence (topmost chain
      p-cell) arrives. Correct-cell scores provide the false-positive control.
  (B) Final-state audit: classifies failure end-states — single-cell error at
      the chain-dependent cell with the chain below "zipped up" correctly
      (evidence present, error locally inconsistent → detectable in principle)
      vs. cascaded / multi-error states.
  (C) Remasking decoders, applied post-hoc in rounds to the base decode:
        loo          exact-q threshold remasking (τ sweep, round budget)
        renoise      uniform random re-noising (no-information class,
                     budget-matched representative of ReMDM-cap/loop)
        conf_frozen  commit-time-confidence rule (ReMDM-conf class; frozen
                     score, refreshed only when a position is re-committed)
      Metrics per chain bucket: exact-match accuracy per round, remask counts,
      re-commitment rate (same token re-committed after remask — "correction
      without new information"), convergence/stuck flags.

Usage (Colab, repo layout diffusion_arithmetic/{core,experiments}):

  %run experiments/remasking_analysis.py \
      --checkpoint-dir /content/drive/MyDrive/.../exp_addition_result41 \
      --seeds 41 --methods random papl puma --iter 300000 \
      --chain-sweep 4 8 12 16 20 24 28 --n-per-bucket 300 \
      --out /content/drive/MyDrive/.../remasking_seed41.json

  # Table R2 (ZTfz Q2, stochastic decoding) — separate run & output:
  %run experiments/remasking_analysis.py --mode stochastic \
      --checkpoint-dir ... --seeds 41 --methods random papl puma \
      --out .../stochastic_seed41.json

  Multiple seeds in one dir: --checkpoint-dir DIR --seeds 41 42 43
  (template: checkpoint_seed{seed}_{method}_iter{iter:06d}.pt)

All instances in a bucket are shared across seeds/methods (fixed --test-seed),
so numbers are directly comparable. Decoding is deterministic (argmax,
max-logit position selection — identical to generate_diffusion semantics).
"""
from __future__ import annotations

import argparse, json, math, os, sys, time
from collections import defaultdict

import torch
import torch.nn.functional as F

if '__file__' in dir():
    _here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(_here))   # repo root
    sys.path.insert(0, _here)                    # experiments dir
else:
    sys.path.insert(0, '.')

import exp_addition as A
from addition_decode_analysis import load_model  # arch inference + EMA workaround

DETECT_TAUS = [0.5]  # observability: q < 1/2 = model deems token more likely wrong than right


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['remask', 'stochastic', 'all'],
                   default='remask',
                   help='remask → Table R1 (3GQB Q1); stochastic → Table R2 '
                        '(ZTfz Q2); all → both from one base-decode pass')
    p.add_argument('--checkpoint-dir', type=str, required=True)
    p.add_argument('--ckpt-template', type=str,
                   default='checkpoint_seed{seed}_{method}_iter{iter:06d}.pt')
    p.add_argument('--seeds', nargs='+', type=int, default=[41])
    p.add_argument('--methods', nargs='+', default=['random', 'papl', 'puma'])
    p.add_argument('--iter', type=int, default=300000)
    p.add_argument('--nd', type=int, default=32)
    p.add_argument('--n-head', type=int, default=None,
                   help='Override inferred n_head (load_model default: 2)')
    p.add_argument('--chain-sweep', nargs='+', type=int,
                   default=[4, 8, 12, 16, 20, 24, 28])
    p.add_argument('--n-per-bucket', type=int, default=300)
    p.add_argument('--natural-n', type=int, default=500,
                   help='0 disables the natural bucket')
    p.add_argument('--test-seed', type=int, default=1042,
                   help='Bucket generation seed — FIXED across seeds/methods')
    p.add_argument('--audit-every', type=int, default=4,
                   help='LOO audit every N commits during base decode; 0 = off')
    p.add_argument('--loo-chunk', type=int, default=4096)
    p.add_argument('--fp-tau-grid', nargs='+', type=float,
                   default=[0.3, 0.5, 0.7, 0.9, 0.99])
    # ── Stochastic decoding arm (ZTfz Q2) ──
    # Token-sampling axis: temperature and nucleus (top-p) are SEPARATE,
    # independently sweepable knobs. Position selection stays clean and
    # deterministic (max clean logit) unless a score variant or positional
    # Gumbel noise is explicitly requested.
    p.add_argument('--temps', nargs='+', type=float, default=[0.7, 1.0, 1.5, 2.0])
    p.add_argument('--top-ps', nargs='+', type=float, default=[1.0, 0.9],
                   help='1.0 = nucleus off; grid is temps × top_ps')
    p.add_argument('--nucleus-order', choices=['temp_first', 'nucleus_first'],
                   default='temp_first',
                   help='temp_first = HF convention (scale logits by T, then '
                        'truncate); nucleus_first = truncate on clean probs, '
                        'then temper survivors')
    p.add_argument('--n-samples', type=int, default=8)
    p.add_argument('--extra-scores', nargs='+', default=['sampled_prob'],
                   choices=['sampled_prob', 'margin', 'neg_entropy'],
                   help='Position-score variants run at T=1, p=1: sampled_prob '
                        '= LLaDA/MaskGIT ranking by clean prob of the DRAWN '
                        'token (mode filter); margin = top1−top2; neg_entropy '
                        '= −H (sign-corrected peakedness)')
    p.add_argument('--gumbel-scales', nargs='+', type=float, default=[1.0, 4.0],
                   help='Positional-order noise: score = clean max-logit + '
                        's·Gumbel, tokens greedy. Isolates the ORDER axis.')
    p.add_argument('--stoch-buckets', nargs='+', default=None,
                   help='Bucket names for the stochastic arm (default: '
                        'natural + two largest chain buckets)')
    p.add_argument('--out', type=str, default='remasking_results.json')
    p.add_argument('--device', type=str, default=None)
    return p.parse_args()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Data / encoding
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def set_nd(nd):
    A.ND = nd
    A.ANS_LEN = nd + 1
    return nd, nd + 1


def max_p_run(gkp):
    """Longest run of 'p' in gkp (index 0 = LSB). Returns (d_lo, d_hi) or None."""
    best, cur_lo = None, None
    for d, g in enumerate(gkp):
        if g == 'p':
            if cur_lo is None:
                cur_lo = d
        else:
            if cur_lo is not None:
                if best is None or (d - 1 - cur_lo) > (best[1] - best[0]):
                    best = (cur_lo, d - 1)
                cur_lo = None
    if cur_lo is not None:
        if best is None or (len(gkp) - 1 - cur_lo) > (best[1] - best[0]):
            best = (cur_lo, len(gkp) - 1)
    return best


def make_meta(s, nd, ans_len):
    """Per-instance structural metadata for the canonical failure analysis."""
    a, b = A._parse_operands(s)
    cs = A._chain_stats(a, b)
    gkp = cs['gkp']                      # index 0 = LSB
    run = max_p_run(gkp)
    meta = {'chain_len': cs['max_chain_len']}
    if run is not None:
        d_lo, d_hi = run
        dd = d_hi + 1                    # digit whose carry-in crosses the run
        # answer-cell index of digit d (0 = MSB text cell): ans_len-1-d
        meta['dep_digit'] = dd
        meta['dep_cell'] = ans_len - 1 - dd if dd < nd else 0   # 0 = carry_out
        meta['dep_role'] = gkp[dd] if dd < nd else 'carry_out'
        meta['evidence_cell'] = ans_len - 1 - d_hi   # topmost chain p-cell
        meta['run_cells'] = [ans_len - 1 - d for d in range(d_lo, d_hi + 1)]
    else:
        meta['dep_cell'] = None
    return meta


def build_bucket(tok, samples, device):
    """Encode full strings; return x0 (answer masked), gold ids, ans_start."""
    mask_id = tok.special_ids['mask']
    enc = [tok.encode(s) for s in samples]
    T = len(enc[0])
    assert all(len(e) == T for e in enc), 'fixed-width format expected'
    ids = torch.tensor(enc, dtype=torch.long, device=device)
    ans_start = samples[0].index('=') + 1
    AL = A.ANS_LEN
    gold = ids[:, ans_start:ans_start + AL].clone()
    x0 = ids.clone()
    x0[:, ans_start:ans_start + AL] = mask_id
    return x0, gold, ans_start


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Core: confidence decode replay + LOO scoring
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@torch.no_grad()
def loo_q(model, x, ans_start, AL, active, mask_id, chunk):
    """Exact leave-one-out score q[b,j] = p(x^j = current | x ⊕ m_j) for every
    (b,j) with active[b,j] True (position currently committed). NaN elsewhere.
    Uses the same mask-token exclusion as decoding."""
    B = x.shape[0]
    device = x.device
    pairs = active.nonzero(as_tuple=False)            # [M, 2] (b, j)
    q = torch.full((B, AL), float('nan'), device=device)
    if pairs.numel() == 0:
        return q
    M = pairs.shape[0]
    for st in range(0, M, chunk):
        pk = pairs[st:st + chunk]
        bs, js = pk[:, 0], pk[:, 1]
        xv = x[bs].clone()
        pos = ans_start + js
        cur = xv[torch.arange(len(bs), device=device), pos].clone()
        xv[torch.arange(len(bs), device=device), pos] = mask_id
        logits = model(xv)                             # [m, T, V]
        lg = logits[torch.arange(len(bs), device=device), pos]
        lg[:, mask_id] = -float('inf')
        probs = F.softmax(lg, dim=-1)
        q[bs, js] = probs[torch.arange(len(bs), device=device), cur]
    return q


@torch.no_grad()
def confidence_fill(model, x, masked, ans_start, AL, mask_id,
                    record_conf=False):
    """Fill all masked answer positions one at a time by confidence
    (max-logit position selection, argmax token — generate_diffusion
    semantics). Mutates x. Returns (commit_rank, commit_conf) as [B, AL]
    (rank = step index at which position was committed; NaN/-1 where the
    position was not masked at entry)."""
    B = x.shape[0]
    device = x.device
    ar = torch.arange(B, device=device)
    commit_rank = torch.full((B, AL), -1, dtype=torch.long, device=device)
    commit_conf = torch.full((B, AL), float('nan'), device=device)
    masked = masked.clone()
    n_steps = int(masked.sum(dim=1).max().item())
    for t in range(n_steps):
        any_m = masked.any(dim=1)
        if not any_m.any():
            break
        logits = model(x)
        al = logits[:, ans_start:ans_start + AL, :].clone()
        al[:, :, mask_id] = -float('inf')
        ml = al.max(dim=-1).values                    # [B, AL]
        ml[~masked] = -float('inf')
        pos = ml.argmax(dim=-1)                       # [B]
        row_l = al[ar, pos]                           # [B, V]
        tokv = row_l.argmax(dim=-1)
        if record_conf:
            conf = F.softmax(row_l, dim=-1)[ar, tokv]
        upd = any_m                                    # rows still working
        x[ar[upd], ans_start + pos[upd]] = tokv[upd]
        commit_rank[ar[upd], pos[upd]] = t
        if record_conf:
            commit_conf[ar[upd], pos[upd]] = conf[upd]
        masked[ar[upd], pos[upd]] = False
    return commit_rank, commit_conf


@torch.no_grad()
def base_decode_with_audits(model, x0, gold, ans_start, AL, mask_id,
                            audit_every, chunk):
    """Full confidence decode of x0 (answer fully masked), recording commit
    order/confidence and LOO audits of all committed cells every
    `audit_every` commits. Returns dict."""
    x = x0.clone()
    B = x.shape[0]
    device = x.device
    ar = torch.arange(B, device=device)
    masked = torch.ones(B, AL, dtype=torch.bool, device=device)
    commit_rank = torch.full((B, AL), -1, dtype=torch.long, device=device)
    commit_conf = torch.full((B, AL), float('nan'), device=device)
    audits = []                                        # (t, q[B,AL] cpu)
    for t in range(AL):
        logits = model(x)
        al = logits[:, ans_start:ans_start + AL, :].clone()
        al[:, :, mask_id] = -float('inf')
        ml = al.max(dim=-1).values
        ml[~masked] = -float('inf')
        pos = ml.argmax(dim=-1)
        row_l = al[ar, pos]
        tokv = row_l.argmax(dim=-1)
        conf = F.softmax(row_l, dim=-1)[ar, tokv]
        x[ar, ans_start + pos] = tokv
        commit_rank[ar, pos] = t
        commit_conf[ar, pos] = conf
        masked[ar, pos] = False
        if audit_every and ((t + 1) % audit_every == 0) and (t + 1) < AL:
            q = loo_q(model, x, ans_start, AL, ~masked, mask_id, chunk)
            audits.append((t, q.cpu()))
    q_final = loo_q(model, x, ans_start, AL,
                    torch.ones(B, AL, dtype=torch.bool, device=device),
                    mask_id, chunk)
    audits.append((AL - 1, q_final.cpu()))
    pred = x[:, ans_start:ans_start + AL]
    return {
        'x': x, 'pred': pred,
        'correct': (pred == gold).all(dim=1),
        'pos_correct': (pred == gold),
        'commit_rank': commit_rank, 'commit_conf': commit_conf,
        'audits': audits, 'q_final': q_final,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# (B) Final-state audit + (A) observability post-processing
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def digit_map(tok):
    return {tok.encode(str(d))[0]: d for d in range(10)}


def audit_and_observe(base, gold, metas, tok, taus):
    """Classify failure end-states; extract detection-delay records for the
    canonical single-error-at-dependent-cell failures."""
    dm = digit_map(tok)
    B = gold.shape[0]
    pos_ok = base['pos_correct'].cpu()
    correct = base['correct'].cpu()
    cr = base['commit_rank'].cpu()
    cc = base['commit_conf'].cpu()
    audits = base['audits']
    out = {'n': B, 'n_fail': int((~correct).sum()),
           'err_count_hist': defaultdict(int),
           'single_at_dep': 0, 'single_elsewhere': 0, 'multi': 0,
           'zip_correct': 0, 'dir_by_role': defaultdict(lambda: defaultdict(int)),
           'records': []}
    for b in range(B):
        if correct[b]:
            continue
        errs = (~pos_ok[b]).nonzero(as_tuple=True)[0].tolist()
        out['err_count_hist'][len(errs)] += 1
        m = metas[b]
        dep = m.get('dep_cell')
        single = len(errs) == 1
        at_dep = single and dep is not None and errs[0] == dep
        if single and at_dep:
            out['single_at_dep'] += 1
        elif single:
            out['single_elsewhere'] += 1
        else:
            out['multi'] += 1
        if at_dep:
            run_ok = all(pos_ok[b, c] for c in m['run_cells'])
            out['zip_correct'] += int(run_ok)
            p_tok = int(base['pred'][b, dep].cpu())
            g_tok = int(gold[b, dep].cpu())
            if p_tok in dm and g_tok in dm:
                d = (dm[p_tok] - dm[g_tok])
                if d > 5: d -= 10
                if d < -5: d += 10
                out['dir_by_role'][m['dep_role']][d] += 1
            rec = {
                'b': b,
                'chain_len': m['chain_len'],
                'commit_rank': int(cr[b, dep]),
                'commit_conf': float(cc[b, dep]),
                'evidence_rank': int(cr[b, m['evidence_cell']]),
                'final_q': float(base['q_final'][b, dep].cpu()),
                'detect_rank': {},
            }
            for tau in taus:
                dr = None
                for (t, q) in audits:
                    v = q[b, dep].item()
                    if not math.isnan(v) and v < tau:
                        dr = t
                        break
                rec['detect_rank'][str(tau)] = dr
            out['records'].append(rec)
    # correct-cell control: final LOO of all cells in correct rows
    ctrl = base['q_final'].cpu()[correct]
    out['control_q'] = ctrl.flatten().tolist()[:5000]
    out['err_count_hist'] = dict(out['err_count_hist'])
    out['dir_by_role'] = {k: dict(v) for k, v in out['dir_by_role'].items()}
    return out


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# (C) Remasking decoders (post-hoc rounds on the base final state)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@torch.no_grad()
def decoder_loo(model, base, gold, ans_start, AL, mask_id, chunk):
    """Minimal PRISM-ideal corrector, one equation:
        T_t = { argmin_j  p(x^j = y^j | y ⊕ m_j) }
    Remask the single cell whose exact leave-one-out conditional (PRISM's
    provably-optimal per-token-quality target, computed directly with one
    forward pass per position) is lowest; refill by the same confidence
    decoding; iterate to the fixed point (the refilled token equals the
    removed one), capped at L rounds. No thresholds, no budgets."""
    x = base['x'].clone()
    B = x.shape[0]
    device = x.device
    ar = torch.arange(B, device=device)
    active = torch.ones(B, dtype=torch.bool, device=device)
    rounds = torch.zeros(B, dtype=torch.long, device=device)
    n_changed = torch.zeros(B, dtype=torch.long, device=device)
    for r in range(AL):
        if not active.any():
            break
        q = loo_q(model, x, ans_start, AL,
                  active.unsqueeze(1).expand(B, AL).clone(),
                  mask_id, chunk)
        q = torch.nan_to_num(q, nan=float('inf'))
        pos = q.argmin(dim=1)                              # [B]
        old = x[ar, ans_start + pos].clone()
        x[ar[active], ans_start + pos[active]] = mask_id
        T = torch.zeros(B, AL, dtype=torch.bool, device=device)
        T[ar[active], pos[active]] = True
        confidence_fill(model, x, T, ans_start, AL, mask_id)
        new = x[ar, ans_start + pos]
        changed = (new != old) & active
        n_changed += changed.long()
        rounds += active.long()
        active = changed                                   # fixed point: stop
    pred = x[:, ans_start:ans_start + AL]
    return {
        'acc_final': float((pred == gold).all(dim=1).float().mean()),
        'rounds_mean': float(rounds.float().mean()),
        'rounds_max': int(rounds.max()),
        'cells_changed_mean': float(n_changed.float().mean()),
        'rescued': int(((pred == gold).all(dim=1) & ~base['correct']).sum()),
        'broken': int((~(pred == gold).all(dim=1) & base['correct']).sum()),
    }


@torch.no_grad()
def decoder_renoise(model, base, gold, ans_start, AL, mask_id, rng):
    """Minimal ReMDM-style re-noising, one equation:
        T_t = { j ~ Unif(committed) }
    Identical loop to decoder_loo — one cell per round, refill by confidence,
    L rounds total — differing ONLY in the selection score (uniform random
    instead of the leave-one-out conditional)."""
    x = base['x'].clone()
    B = x.shape[0]
    device = x.device
    ar = torch.arange(B, device=device)
    recommit = 0
    for r in range(AL):
        pos = torch.randint(0, AL, (B,), generator=rng).to(device)
        old = x[ar, ans_start + pos].clone()
        x[ar, ans_start + pos] = mask_id
        T = torch.zeros(B, AL, dtype=torch.bool, device=device)
        T[ar, pos] = True
        confidence_fill(model, x, T, ans_start, AL, mask_id)
        recommit += int((x[ar, ans_start + pos] == old).sum())
    pred = x[:, ans_start:ans_start + AL]
    return {
        'acc_final': float((pred == gold).all(dim=1).float().mean()),
        'n_remask_total': B * AL,
        'recommit_rate': recommit / (B * AL),
        'rescued': int(((pred == gold).all(dim=1) & ~base['correct']).sum()),
        'broken': int((~(pred == gold).all(dim=1) & base['correct']).sum()),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Stochastic decoding arm (ZTfz Q2)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Two INDEPENDENT axes, kept separate throughout:
#   token axis    — how the token at the selected position is drawn
#                   (temperature; nucleus/top-p truncation, either order)
#   position axis — how the position to commit is selected
#                   (clean max-logit by default = deterministic and
#                    temperature-invariant by construction; variants:
#                    sampled_prob / margin / neg_entropy scores, or
#                    additive Gumbel noise on the score)

def _gumbel_like(shape, gen, device):
    u = torch.rand(shape, generator=gen).clamp_(1e-20, 1 - 1e-20)
    return (-torch.log(-torch.log(u))).to(device)


def _nucleus_mask(logits, probs, top_p):
    """Set logits outside the smallest set with cumulative prob ≥ top_p
    (computed from `probs`) to -inf. Standard inclusive nucleus."""
    sp, si = probs.sort(dim=-1, descending=True)
    cum = sp.cumsum(dim=-1)
    drop_sorted = (cum - sp) >= top_p          # tokens strictly after crossing
    drop = torch.zeros_like(drop_sorted).scatter(-1, si, drop_sorted)
    return logits.masked_fill(drop, -float('inf'))


def _draw_tokens(al, T, top_p, order, gen):
    """Sample a token per position from answer-region logits `al` [B, AL, V]
    (mask token already excluded). T=0 → greedy (nucleus irrelevant)."""
    if T <= 0:
        return al.argmax(dim=-1)
    if order == 'temp_first':
        lt = al / T
        if top_p < 1.0:
            lt = _nucleus_mask(lt, F.softmax(lt, dim=-1), top_p)
    else:  # nucleus_first: truncate on the CLEAN distribution, then temper
        lt = al
        if top_p < 1.0:
            lt = _nucleus_mask(lt, F.softmax(lt, dim=-1), top_p)
        lt = lt / T
    return (lt + _gumbel_like(lt.shape, gen, lt.device)).argmax(dim=-1)


@torch.no_grad()
def stochastic_decode(model, x0, ans_start, AL, mask_id, cfg, gen):
    """One full decode under a stochastic config.
    cfg: dict(T, top_p, order, score, gpos). Returns (pred, commit_rank)."""
    x = x0.clone()
    B = x.shape[0]
    device = x.device
    ar = torch.arange(B, device=device)
    masked = torch.ones(B, AL, dtype=torch.bool, device=device)
    commit_rank = torch.full((B, AL), -1, dtype=torch.long, device=device)
    for t in range(AL):
        logits = model(x)
        al = logits[:, ans_start:ans_start + AL, :].clone()
        al[:, :, mask_id] = -float('inf')
        cand = _draw_tokens(al, cfg['T'], cfg['top_p'], cfg['order'], gen)
        # position score
        sc = cfg['score']
        if sc == 'clean_maxlogit':
            s = al.max(dim=-1).values
        elif sc == 'sampled_prob':      # LLaDA/MaskGIT: clean prob of drawn token
            p_clean = F.softmax(al, dim=-1)
            s = p_clean.gather(-1, cand.unsqueeze(-1)).squeeze(-1)
        elif sc == 'margin':
            top2 = F.softmax(al, dim=-1).topk(2, dim=-1).values
            s = top2[..., 0] - top2[..., 1]
        elif sc == 'neg_entropy':
            p_clean = F.softmax(al, dim=-1)
            s = (p_clean * torch.log(p_clean + 1e-10)).sum(dim=-1)   # = -H
        else:
            raise ValueError(sc)
        if cfg['gpos'] > 0:
            s = s + cfg['gpos'] * _gumbel_like(s.shape, gen, device)
        s = s.masked_fill(~masked, -float('inf'))
        pos = s.argmax(dim=-1)
        tok = cand[ar, pos]
        x[ar, ans_start + pos] = tok
        commit_rank[ar, pos] = t
        masked[ar, pos] = False
    return x[:, ans_start:ans_start + AL], commit_rank


def stochastic_configs(args):
    cfgs = []
    for T in args.temps:
        for tp in args.top_ps:
            cfgs.append({'name': f'T={T}, p={tp}', 'T': T, 'top_p': tp,
                         'order': args.nucleus_order,
                         'score': 'clean_maxlogit', 'gpos': 0.0})
    for sc in args.extra_scores:
        cfgs.append({'name': f'score={sc} (T=1,p=1)', 'T': 1.0, 'top_p': 1.0,
                     'order': args.nucleus_order, 'score': sc, 'gpos': 0.0})
    for g in args.gumbel_scales:
        if g > 0:
            cfgs.append({'name': f'order-noise s={g} (greedy tok)', 'T': 0.0,
                         'top_p': 1.0, 'order': args.nucleus_order,
                         'score': 'clean_maxlogit', 'gpos': g})
    return cfgs


def run_stochastic_bucket(model, base, gold, ans_start, AL, mask_id,
                          audit, args, seed, bucket_name):
    """All stochastic configs × N samples on one bucket. Uses the base decode
    for trap-instance identity (canonical single@dep failures) and for the
    base-correct set (corruption side)."""
    x0 = base['x'].clone()
    x0[:, ans_start:ans_start + AL] = mask_id            # re-mask answer
    device = x0.device
    base_correct = base['correct']
    n_bc = int(base_correct.sum())
    recs = audit['records']
    if recs:
        dep_b = torch.tensor([r['b'] for r in recs], dtype=torch.long,
                             device=device)
        # canonical failures are single-error: the dep cell is the wrong pos
        dep_cell = torch.tensor(
            [int((~base['pos_correct'][r['b']]).nonzero(as_tuple=True)[0][0])
             for r in recs], dtype=torch.long, device=device)
        base_rank = torch.tensor([r['commit_rank'] for r in recs],
                                 dtype=torch.long, device=device)

    # sanity: T→0, p=1, no order noise ≡ deterministic base (bit-identical)
    gen0 = torch.Generator().manual_seed(0)
    pred0, _ = stochastic_decode(
        model, x0, ans_start, AL, mask_id,
        {'T': 0.0, 'top_p': 1.0, 'order': args.nucleus_order,
         'score': 'clean_maxlogit', 'gpos': 0.0}, gen0)
    assert torch.equal(pred0, base['pred']), \
        'T→0 limit does not reproduce deterministic base decode'

    out = {'sanity_T0_equals_base': True, 'configs': {}}
    bsum = sum(ord(c) for c in bucket_name)
    for ci, cfg in enumerate(stochastic_configs(args)):
        accs, escapes, rescues, corrupt, flips, rank_same = [], [], [], [], [], []
        ok_all = []
        for n in range(args.n_samples):
            gen = torch.Generator().manual_seed(
                7_000_000 + seed * 10_000 + ci * 100 + n * 7 + bsum)
            pred, cr = stochastic_decode(model, x0, ans_start, AL, mask_id,
                                         cfg, gen)
            ok = (pred == gold).all(dim=1)
            ok_all.append(ok)
            accs.append(float(ok.float().mean()))
            if n_bc:
                corrupt.append(float((~ok[base_correct]).float().mean()))
                flips.append(float(
                    (pred[base_correct] != gold[base_correct]).float().mean()))
            if recs:
                pd = pred[dep_b, dep_cell]
                gd = gold[dep_b, dep_cell]
                escapes.append(float((pd == gd).float().mean()))
                rescues.append(float(ok[dep_b].float().mean()))
                rank_same.append(float(
                    (cr[dep_b, dep_cell] == base_rank).float().mean()))
        ok_any = torch.stack(ok_all, dim=0).any(dim=0)
        mean = lambda v: (sum(v) / len(v)) if v else None
        out['configs'][cfg['name']] = {
            'acc_mean': mean(accs), 'acc_min': min(accs), 'acc_max': max(accs),
            'pass_at_n': float(ok_any.float().mean()),
            'trap_pass_at_n': (float(ok_any[dep_b].float().mean())
                               if recs else None),
            'trap_escape_rate': mean(escapes),
            'trap_full_rescue_rate': mean(rescues),
            'trap_rank_unchanged': mean(rank_same),
            'corruption_rate': mean(corrupt),
            'cell_flip_rate': mean(flips),
            'n_trap': len(recs), 'n_base_correct': n_bc,
        }
    return out


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, torch.Tensor):
        return o.tolist()
    if isinstance(o, float) and math.isnan(o):
        return None
    return o


def fp_rate(control_q, tau):
    if not control_q:
        return None
    return sum(1 for v in control_q if v < tau) / len(control_q)


def run_analysis(args):
    device = torch.device(args.device) if args.device else (
        torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    nd, AL = set_nd(args.nd)
    if args.n_head is not None:
        import addition_decode_analysis as ADA
        ADA.N_HEAD_OVERRIDE = args.n_head
    tok = A.build_tok()
    mask_id = tok.special_ids['mask']

    # ── Buckets: shared across seeds/methods (fixed test seed) ──
    buckets = {}
    if args.natural_n > 0:
        nat = A.gen_data_natural(args.natural_n, seed=args.test_seed)
        buckets['natural'] = nat
    for cl in args.chain_sweep:
        sp = A.gen_min_chain_test(args.n_per_bucket,
                                  seed=args.test_seed + 500 + cl,
                                  min_chain=cl)
        if sp:
            buckets[f'chain>={cl}'] = sp
    enc_buckets = {}
    for name, samples in buckets.items():
        x0, gold, ans_start = build_bucket(tok, samples, device)
        metas = [make_meta(s, nd, AL) for s in samples]
        enc_buckets[name] = (x0, gold, ans_start, metas)
        print(f"  bucket {name}: n={len(samples)}")

    results = {'config': _jsonable(vars(args)), 'nd': nd}
    # stochastic-arm bucket selection: natural + two largest chain buckets
    if args.stoch_buckets:
        stoch_names = set(args.stoch_buckets)
    else:
        chains = sorted([b for b in enc_buckets if b.startswith('chain')],
                        key=lambda s: int(s.split('>=')[1]))
        stoch_names = set((['natural'] if 'natural' in enc_buckets else [])
                          + chains[-2:])
    if args.mode in ('stochastic', 'all'):
        print(f"  stochastic buckets: {sorted(stoch_names)}")
    for seed in args.seeds:
        for method in args.methods:
            ck = args.ckpt_template.format(seed=seed, method=method,
                                           iter=args.iter)
            path = os.path.join(args.checkpoint_dir, ck)
            print(f"\n{'━'*66}\n  seed {seed} | {method} | {path}\n{'━'*66}")
            if not os.path.exists(path):
                print(f"  !! checkpoint missing, skipping")
                continue
            model = load_model(path, device)
            model.eval()
            key = f'seed{seed}_{method}'
            results[key] = {}
            for name, (x0, gold, ans_start, metas) in enc_buckets.items():
                t0 = time.time()
                base = base_decode_with_audits(
                    model, x0, gold, ans_start, AL, mask_id,
                    args.audit_every, args.loo_chunk)
                obs = audit_and_observe(base, gold, metas, tok, DETECT_TAUS)
                entry = {
                    'base_acc': float(base['correct'].float().mean()),
                    'audit': {k: v for k, v in obs.items()
                              if k not in ('control_q',)},
                    'fp_rate': {str(t): fp_rate(obs['control_q'], t)
                                for t in args.fp_tau_grid},
                    'decoders': {},
                }
                if args.mode in ('remask', 'all'):
                    entry['decoders']['loo'] = decoder_loo(
                        model, base, gold, ans_start, AL, mask_id,
                        args.loo_chunk)
                    rng = torch.Generator().manual_seed(
                        10_000 + seed * 97 + sum(ord(c) for c in name))
                    entry['decoders']['renoise'] = decoder_renoise(
                        model, base, gold, ans_start, AL, mask_id, rng)
                if args.mode in ('stochastic', 'all') and name in stoch_names:
                    entry['stochastic'] = run_stochastic_bucket(
                        model, base, gold, ans_start, AL, mask_id, obs,
                        args, seed, name)
                results[key][name] = entry
                dt = time.time() - t0
                dstr = ' '.join(f"{k}={v['acc_final']:.3f}"
                                for k, v in entry['decoders'].items())
                print(f"  {name:>12s}  base={entry['base_acc']:.3f}  {dstr}"
                      f"  ({dt:.0f}s, fails={obs['n_fail']},"
                      f" single@dep={obs['single_at_dep']},"
                      f" zip={obs['zip_correct']})")
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ── Summary tables ──
    if args.mode in ('remask', 'all'):
        print(f"\n{'='*70}\n  TABLE R1 — remasking decoders (3GQB Q1)\n{'='*70}")
        dec_cols = None
        for key in [k for k in results if k.startswith('seed')]:
            print(f"\n── {key} ──")
            rows = results[key]
            if dec_cols is None:
                any_b = next(iter(rows.values()))
                dec_cols = list(any_b['decoders'].keys())
            print(f"  {'bucket':>12s} {'base':>7s}" +
                  ''.join(f" {c:>12s}" for c in dec_cols))
            for name, e in rows.items():
                print(f"  {name:>12s} {e['base_acc']:>7.3f}" +
                      ''.join(f" {e['decoders'][c]['acc_final']:>12.3f}"
                              for c in dec_cols))
            for name, e in rows.items():
                recs = e['audit']['records']
                if not recs:
                    continue
                for tau in DETECT_TAUS:
                    ds = [r['detect_rank'][str(tau)] for r in recs]
                    nd_ = sum(1 for d in ds if d is None)
                    dd = [r['detect_rank'][str(tau)] - r['commit_rank']
                          for r in recs
                          if r['detect_rank'][str(tau)] is not None]
                    ev = [r['evidence_rank'] - r['commit_rank'] for r in recs]
                    if dd:
                        print(f"    {name} τ={tau}: detect-delay median="
                              f"{sorted(dd)[len(dd)//2]}, never-detected="
                              f"{nd_}/{len(recs)}, evidence-delay median="
                              f"{sorted(ev)[len(ev)//2]}")

    if args.mode in ('stochastic', 'all'):
        print(f"\n{'='*70}\n  TABLE R2 — stochastic decoding (ZTfz Q2)\n{'='*70}")
        for key in [k for k in results if k.startswith('seed')]:
            for name, e in results[key].items():
                st = e.get('stochastic')
                if not st:
                    continue
                print(f"\n── {key} | {name} (base={e['base_acc']:.3f}, "
                      f"n_trap={next(iter(st['configs'].values()))['n_trap']}) ──")
                print(f"  {'config':>26s} {'acc':>6s} {'Δbase':>7s} "
                      f"{'corrupt':>8s} {'escape':>7s} {'rescue':>7s} "
                      f"{'rank=':>6s} {'pass@N':>7s}")
                for cn, c in st['configs'].items():
                    fmt = lambda v, w=7: (f"{v:>{w}.3f}" if v is not None
                                          else f"{'—':>{w}s}")
                    print(f"  {cn:>26s} {c['acc_mean']:>6.3f} "
                          f"{c['acc_mean']-e['base_acc']:>+7.3f} "
                          f"{fmt(c['corruption_rate'],8)} "
                          f"{fmt(c['trap_escape_rate'])} "
                          f"{fmt(c['trap_full_rescue_rate'])} "
                          f"{fmt(c['trap_rank_unchanged'],6)} "
                          f"{c['pass_at_n']:>7.3f}")

    with open(args.out, 'w') as f:
        json.dump(_jsonable(results), f, indent=1)
    print(f"\n  💾 {args.out}")
    return results


if __name__ == '__main__':
    run_analysis(parse_args())
