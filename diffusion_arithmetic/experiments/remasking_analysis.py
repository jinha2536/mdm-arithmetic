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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def parse_args():
    p = argparse.ArgumentParser()
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
    p.add_argument('--decoders', nargs='+',
                   default=['loo', 'renoise', 'conf_frozen'])
    p.add_argument('--taus', nargs='+', type=float, default=[0.5, 0.9])
    p.add_argument('--loo-rounds', type=int, default=8)
    p.add_argument('--renoise-k', type=int, default=5)
    p.add_argument('--renoise-rounds', type=int, default=8)
    p.add_argument('--conf-k', type=int, default=2)
    p.add_argument('--conf-rounds', type=int, default=4)
    p.add_argument('--audit-every', type=int, default=4,
                   help='LOO audit every N commits during base decode; 0 = off')
    p.add_argument('--loo-chunk', type=int, default=4096)
    p.add_argument('--fp-tau-grid', nargs='+', type=float,
                   default=[0.3, 0.5, 0.7, 0.9, 0.99])
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
def run_rounds(model, base, gold, ans_start, AL, mask_id, select_fn,
               max_rounds, chunk, need_q, stop_when_empty):
    """Generic round loop. select_fn(state) -> bool [B, AL] set to remask.
    state carries x, q (if need_q), commit_conf (refreshed on re-commit),
    round index. Returns metrics."""
    x = base['x'].clone()
    B = x.shape[0]
    device = x.device
    commit_conf = base['commit_conf'].clone()
    acc_rounds, remask_counts, recommit_counts = [], [], []
    prev_T = None
    stuck = False
    rounds_used = 0
    for r in range(max_rounds):
        q = loo_q(model, x, ans_start, AL,
                  torch.ones(B, AL, dtype=torch.bool, device=device),
                  mask_id, chunk) if need_q else None
        T = select_fn({'x': x, 'q': q, 'commit_conf': commit_conf,
                       'round': r, 'ans_start': ans_start})
        nT = int(T.sum())
        if stop_when_empty and nT == 0:
            break
        if prev_T is not None and torch.equal(T, prev_T):
            stuck = True
            break
        prev_T = T.clone()
        rounds_used = r + 1
        old_tok = x[:, ans_start:ans_start + AL].clone()
        # remask T
        bpos = T.nonzero(as_tuple=False)
        x[bpos[:, 0], ans_start + bpos[:, 1]] = mask_id
        # refill by confidence; refresh commit_conf at refilled positions
        _, new_conf = confidence_fill(model, x, T.clone(), ans_start, AL,
                                      mask_id, record_conf=True)
        commit_conf[T] = new_conf[T]
        new_tok = x[:, ans_start:ans_start + AL]
        recommit = ((new_tok == old_tok) & T).sum()
        remask_counts.append(nT)
        recommit_counts.append(int(recommit))
        acc_rounds.append(float((new_tok == gold).all(dim=1).float().mean()))
    final_pred = x[:, ans_start:ans_start + AL]
    total_remask = sum(remask_counts)
    return {
        'acc_final': float((final_pred == gold).all(dim=1).float().mean()),
        'acc_rounds': acc_rounds,
        'rounds_used': rounds_used,
        'stuck': stuck,
        'n_remask_total': total_remask,
        'n_remask_rounds': remask_counts,
        'recommit_rate': (sum(recommit_counts) / total_remask)
                         if total_remask else None,
    }


def decoder_loo(model, base, gold, ans_start, AL, mask_id, tau, rounds, chunk):
    def sel(st):
        return st['q'] < tau
    return run_rounds(model, base, gold, ans_start, AL, mask_id, sel,
                      rounds, chunk, need_q=True, stop_when_empty=True)


def decoder_renoise(model, base, gold, ans_start, AL, mask_id, k, rounds,
                    chunk, rng):
    def sel(st):
        B = st['x'].shape[0]
        scores = torch.rand(B, AL, generator=rng, device='cpu').to(st['x'].device)
        thresh = scores.topk(k, dim=1).values[:, -1:]
        return scores >= thresh
    return run_rounds(model, base, gold, ans_start, AL, mask_id, sel,
                      rounds, chunk, need_q=False, stop_when_empty=False)


def decoder_conf_frozen(model, base, gold, ans_start, AL, mask_id, k, rounds,
                        chunk):
    def sel(st):
        cc = st['commit_conf']
        thresh = cc.topk(k, dim=1, largest=False).values[:, -1:]
        return cc <= thresh
    return run_rounds(model, base, gold, ans_start, AL, mask_id, sel,
                      rounds, chunk, need_q=False, stop_when_empty=False)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Orchestration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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
                obs = audit_and_observe(base, gold, metas, tok, args.taus)
                entry = {
                    'base_acc': float(base['correct'].float().mean()),
                    'audit': {k: v for k, v in obs.items()
                              if k not in ('control_q',)},
                    'fp_rate': {str(t): fp_rate(obs['control_q'], t)
                                for t in args.fp_tau_grid},
                    'decoders': {},
                }
                for dec in args.decoders:
                    if dec == 'loo':
                        for tau in args.taus:
                            r = decoder_loo(model, base, gold, ans_start, AL,
                                            mask_id, tau, args.loo_rounds,
                                            args.loo_chunk)
                            entry['decoders'][f'loo_tau{tau}'] = r
                    elif dec == 'renoise':
                        rng = torch.Generator().manual_seed(
                            10_000 + seed * 97 + sum(ord(c) for c in name))
                        r = decoder_renoise(model, base, gold, ans_start, AL,
                                            mask_id, args.renoise_k,
                                            args.renoise_rounds,
                                            args.loo_chunk, rng)
                        entry['decoders']['renoise'] = r
                    elif dec == 'conf_frozen':
                        r = decoder_conf_frozen(model, base, gold, ans_start,
                                                AL, mask_id, args.conf_k,
                                                args.conf_rounds,
                                                args.loo_chunk)
                        entry['decoders']['conf_frozen'] = r
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
    print(f"\n{'='*70}\n  SUMMARY (exact-match)\n{'='*70}")
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
        # detection-delay digest
        for name, e in rows.items():
            recs = e['audit']['records']
            if not recs:
                continue
            for tau in args.taus:
                ds = [r['detect_rank'][str(tau)] for r in recs]
                nd_ = sum(1 for d in ds if d is None)
                dd = [r['detect_rank'][str(tau)] - r['commit_rank']
                      for r in recs if r['detect_rank'][str(tau)] is not None]
                ev = [r['evidence_rank'] - r['commit_rank'] for r in recs]
                if dd:
                    print(f"    {name} τ={tau}: detect-delay median="
                          f"{sorted(dd)[len(dd)//2]}, never-detected={nd_}/"
                          f"{len(recs)}, evidence-delay median="
                          f"{sorted(ev)[len(ev)//2]}")

    with open(args.out, 'w') as f:
        json.dump(_jsonable(results), f, indent=1)
    print(f"\n  💾 {args.out}")
    return results


if __name__ == '__main__':
    run_analysis(parse_args())
