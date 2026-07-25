"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Maze remasking analysis (one-cell retrained checkpoints)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Addition-parallel remasking analysis for the maze domain, with the two
maze-specific structures made first-class:

  • Only OPEN cells (puzzle '.') are ever masked, decoded, audited, or
    remasked — walls/S/E are given context.
  • Failure audit classifies errors by DIRECTION — deletion (gold on-path
    '1' → predicted off '0') vs. addition (gold '0' → predicted '1') — and
    by STRUCTURE: connected components under 4-adjacency. A single-cell
    error is a local inconsistency (the addition-style mode, repairable in
    principle); a coherent multi-cell component is a self-consistent fake /
    deleted path segment whose cells may mutually endorse each other under
    leave-one-out — measured via the final-state LOO score of error cells,
    stratified by component size. If coherent components self-endorse
    (q high), even the LOO ceiling cannot repair them: the maze analog of
    a representation-level failure at inference time.

Decoders (identical minimal pair as the addition table):
  loo      T_t = { argmin_{j ∈ open} p(x^j = y^j | y ⊕ m_j) }, refill by the
           same confidence decoding, iterate to fixed point (cap: #open).
  renoise  T_t = { j ~ Unif(open cells) }, same loop, #open rounds.

Base decode mirrors exp_maze.generate_blanks one-cell semantics exactly
(softmax-prob confidence, no mask-logit exclusion, one cell per forward);
verified bit-identical against generate_blanks in the smoke test.

Usage (Colab):
  %run experiments/maze_remasking.py \\
      --checkpoint-dir <drive>/exp_maze_ms1 \\
      --seeds 41 42 43 --methods random papl puma --grid-n 10 \\
      --out <drive>/maze_remasking_ms1.json
"""
from __future__ import annotations

import argparse, json, math, os, sys, time
from collections import defaultdict

import torch
import torch.nn.functional as F

if '__file__' in dir():
    _here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(_here))
    sys.path.insert(0, _here)
else:
    sys.path.insert(0, '.')

import exp_maze as M
from addition_decode_analysis import load_model
from remasking_analysis import loo_q, confidence_fill, _jsonable

DETECT_TAU = 0.5


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint-dir', type=str, required=True)
    p.add_argument('--ckpt-template', type=str,
                   default='checkpoint_seed{seed}_{method}.pt')
    p.add_argument('--seeds', nargs='+', type=int, default=[41])
    p.add_argument('--methods', nargs='+', default=['random', 'papl', 'puma'])
    p.add_argument('--grid-n', type=int, default=10)
    p.add_argument('--n-head', type=int, default=None)
    # Agreed bucket design (hard-tail discussion):
    #   natural-stratified corridor bins  — pure stratification of the natural
    #       pool by max corridor length (no straightness-bias confound)
    #   extrapolation band — constructed mazes beyond the natural p99 (~75)
    #   pure corridor — labeled representation-boundary bucket
    p.add_argument('--natural-pool', type=int, default=3000,
                   help='Natural mazes generated then stratified')
    p.add_argument('--strata-edges', nargs='+', type=int,
                   default=[24, 32, 40, 56],
                   help='max-corridor bin edges over the natural pool')
    p.add_argument('--xtrap-sweep', nargs='+', type=int, default=[50, 65, 80],
                   help='Constructed extrapolation band (bias-generated; '
                        'labeled as such)')
    p.add_argument('--pure-n', type=int, default=100,
                   help='Pure-corridor boundary bucket size (0 disables)')
    p.add_argument('--n-per-bucket', type=int, default=200)
    p.add_argument('--natural-n', type=int, default=300,
                   help='Unstratified natural bucket (0 disables)')
    p.add_argument('--test-seed', type=int, default=2042,
                   help='FIXED across seeds/methods — shared buckets')
    p.add_argument('--audit-every', type=int, default=25,
                   help='LOO audit every N commits during base decode; 0=off')
    p.add_argument('--loo-chunk', type=int, default=2048)
    p.add_argument('--out', type=str, default='maze_remasking.json')
    p.add_argument('--device', type=str, default=None)
    return p.parse_args()


def set_grid(n):
    M.GRID_N = n
    M.GRID_H = M.GRID_W = 2 * n + 1
    M.CELL_N = M.GRID_H * M.GRID_W
    M.ANS_LEN = M.CELL_N
    return M.CELL_N


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Encoding
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_bucket(tok, entries, device):
    strings = [e['string'] for e in entries]
    enc = [tok.encode(s) for s in strings]
    T = len(enc[0])
    assert all(len(x) == T for x in enc), 'fixed-width maze strings expected'
    ids = torch.tensor(enc, dtype=torch.long, device=device)
    ans_start = strings[0].index('=') + 1
    AL = M.ANS_LEN
    gold = ids[:, ans_start:ans_start + AL].clone()
    open_mask = torch.zeros(len(entries), AL, dtype=torch.bool, device=device)
    for b, s in enumerate(strings):
        puz = s[:ans_start - 1]
        for j, c in enumerate(puz):
            if c == '.':
                open_mask[b, j] = True
    mask_id = tok.special_ids['mask']
    x0 = ids.clone()
    pos = open_mask.nonzero(as_tuple=False)
    x0[pos[:, 0], ans_start + pos[:, 1]] = mask_id
    return x0, gold, open_mask, ans_start


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Base decode (mirrors exp_maze.generate_blanks one-cell exactly:
# softmax-prob confidence, no mask-logit exclusion, one cell/forward)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@torch.no_grad()
def base_decode(model, x0, gold, open_mask, ans_start, AL, mask_id,
                audit_every, chunk):
    x = x0.clone()
    B = x.shape[0]
    device = x.device
    ar = torch.arange(B, device=device)
    masked = open_mask.clone()
    commit_rank = torch.full((B, AL), -1, dtype=torch.long, device=device)
    commit_conf = torch.full((B, AL), float('nan'), device=device)
    audits = []
    n_steps = int(masked.sum(1).max())
    for t in range(n_steps):
        upd = masked.any(dim=1)
        if not upd.any():
            break
        logits = model(x)
        al = logits[:, ans_start:ans_start + AL, :]
        probs = F.softmax(al, dim=-1)
        confs, preds = probs.max(dim=-1)
        confs = confs.masked_fill(~masked, -float('inf'))
        pos = confs.argmax(dim=-1)
        tokv = preds[ar, pos]
        x[ar[upd], ans_start + pos[upd]] = tokv[upd]
        commit_rank[ar[upd], pos[upd]] = t
        commit_conf[ar[upd], pos[upd]] = probs[ar, pos].max(-1).values[upd]
        masked[ar[upd], pos[upd]] = False
        if audit_every and ((t + 1) % audit_every == 0) and masked.any():
            q = loo_q(model, x, ans_start, AL, open_mask & ~masked,
                      mask_id, chunk)
            audits.append((t, q.cpu()))
    q_final = loo_q(model, x, ans_start, AL, open_mask.clone(),
                    mask_id, chunk)
    audits.append((n_steps - 1, q_final.cpu()))
    pred = x[:, ans_start:ans_start + AL]
    return {'x': x, 'pred': pred,
            'correct': (pred == gold).all(dim=1),
            'pos_correct': (pred == gold),
            'commit_rank': commit_rank, 'commit_conf': commit_conf,
            'audits': audits, 'q_final': q_final}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Audit: direction (add/delete) × structure (connected components)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _components(cells, W):
    cs = set(cells)
    comps = []
    while cs:
        seed = cs.pop()
        comp, stack = [seed], [seed]
        while stack:
            c = stack.pop()
            r, col = divmod(c, W)
            for nb in (c - W, c + W, c - 1 if col > 0 else -1,
                       c + 1 if col < W - 1 else -1):
                if nb in cs:
                    cs.remove(nb)
                    comp.append(nb)
                    stack.append(nb)
        comps.append(comp)
    return comps


def audit_maze(base, gold, open_mask, entries, tok):
    id1 = tok.encode('1')[0]
    id0 = tok.encode('0')[0]
    W = M.GRID_W
    B = gold.shape[0]
    pos_ok = base['pos_correct'].cpu()
    correct = base['correct'].cpu()
    pred = base['pred'].cpu()
    gd = gold.cpu()
    qf = base['q_final'].cpu()
    out = {'n': B, 'n_fail': int((~correct).sum()),
           'err_count_hist': defaultdict(int),
           'comp_count_hist': defaultdict(int),
           'comp_size_hist': defaultdict(int),
           'n_single_cell': 0, 'n_coherent': 0, 'n_multi_comp': 0,
           'dir_totals': {'add': 0, 'delete': 0, 'other': 0},
           'comp_purity': {'all_add': 0, 'all_delete': 0, 'mixed': 0},
           'finalq_err_size1': [], 'finalq_err_sizege2': [],
           'records': []}
    for b in range(B):
        if correct[b]:
            continue
        err = ((~pos_ok[b]) & open_mask[b].cpu()).nonzero(as_tuple=True)[0]
        err = [int(c) for c in err]
        out['err_count_hist'][len(err)] += 1
        n_add = n_del = n_oth = 0
        for c in err:
            g_, p_ = int(gd[b, c]), int(pred[b, c])
            if g_ == id0 and p_ == id1:
                n_add += 1
            elif g_ == id1 and p_ == id0:
                n_del += 1
            else:
                n_oth += 1
        out['dir_totals']['add'] += n_add
        out['dir_totals']['delete'] += n_del
        out['dir_totals']['other'] += n_oth
        comps = _components(err, W)
        out['comp_count_hist'][len(comps)] += 1
        if len(err) == 1:
            out['n_single_cell'] += 1
        elif len(comps) == 1:
            out['n_coherent'] += 1
        else:
            out['n_multi_comp'] += 1
        for comp in comps:
            out['comp_size_hist'][len(comp)] += 1
            dirs = set()
            for c in comp:
                g_, p_ = int(gd[b, c]), int(pred[b, c])
                dirs.add('add' if (g_ == id0 and p_ == id1) else
                         'delete' if (g_ == id1 and p_ == id0) else 'other')
            if dirs == {'add'}:
                out['comp_purity']['all_add'] += 1
            elif dirs == {'delete'}:
                out['comp_purity']['all_delete'] += 1
            else:
                out['comp_purity']['mixed'] += 1
        qs = [float(qf[b, c]) for c in err]
        # LOO blindness: rank of the best (lowest-q) error cell among ALL
        # open cells of this instance. rank 0 → argmin targets an error in
        # round 1; large rank → the errors are q-endorsed above correct
        # cells and the leave-one-out corrector is structurally blind.
        oc = open_mask[b].cpu().nonzero(as_tuple=True)[0].tolist()
        allq = sorted(float(qf[b, c]) for c in oc)
        best_err = min(qs)
        import bisect
        err_rank = bisect.bisect_left(allq, best_err)
        for c in err:
            v = float(qf[b, c])
            comp_of = next(cc for cc in comps if c in cc)
            (out['finalq_err_size1'] if len(comp_of) == 1
             else out['finalq_err_sizege2']).append(v)
        out['records'].append({
            'b': b, 'err_q_rank': err_rank, 'n_open': len(oc),
            'n_err': len(err), 'n_comp': len(comps),
            'max_comp': max(len(c) for c in comps),
            'n_add': n_add, 'n_delete': n_del,
            'finalq_med': sorted(qs)[len(qs) // 2],
            'finalq_min': min(qs),
            'detectable': any(v < DETECT_TAU for v in qs),
            'max_corridor_len':
                entries[b]['corridor_stats']['max_corridor_len'],
        })
    ctrl = []
    for b in range(B):
        if correct[b]:
            oc = open_mask[b].cpu().nonzero(as_tuple=True)[0]
            ctrl.extend(float(qf[b, c]) for c in oc[:20])
        if len(ctrl) >= 4000:
            break
    out['control_q'] = ctrl[:4000]
    for k in ('err_count_hist', 'comp_count_hist', 'comp_size_hist'):
        out[k] = dict(out[k])
    # keep score lists bounded
    out['finalq_err_size1'] = out['finalq_err_size1'][:4000]
    out['finalq_err_sizege2'] = out['finalq_err_sizege2'][:4000]
    return out


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Correctors (open-cell aware, minimal pair)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@torch.no_grad()
def decoder_loo(model, base, gold, open_mask, ans_start, AL, mask_id, chunk):
    x = base['x'].clone()
    B = x.shape[0]
    device = x.device
    ar = torch.arange(B, device=device)
    active = torch.ones(B, dtype=torch.bool, device=device)
    rounds = torch.zeros(B, dtype=torch.long, device=device)
    n_changed = torch.zeros(B, dtype=torch.long, device=device)
    max_rounds = int(open_mask.sum(1).max())
    for r in range(max_rounds):
        if not active.any():
            break
        act_cells = open_mask & active.unsqueeze(1)
        q = loo_q(model, x, ans_start, AL, act_cells.clone(), mask_id, chunk)
        q = torch.nan_to_num(q, nan=float('inf'))
        pos = q.argmin(dim=1)
        old = x[ar, ans_start + pos].clone()
        x[ar[active], ans_start + pos[active]] = mask_id
        T = torch.zeros(B, AL, dtype=torch.bool, device=device)
        T[ar[active], pos[active]] = True
        confidence_fill(model, x, T, ans_start, AL, mask_id)
        new = x[ar, ans_start + pos]
        changed = (new != old) & active
        n_changed += changed.long()
        rounds += active.long()
        active = changed
    pred = x[:, ans_start:ans_start + AL]
    ok = (pred == gold).all(dim=1)
    return {'acc_final': float(ok.float().mean()),
            'rounds_mean': float(rounds.float().mean()),
            'rounds_max': int(rounds.max()),
            'cells_changed_mean': float(n_changed.float().mean()),
            'rescued': int((ok & ~base['correct']).sum()),
            'broken': int((~ok & base['correct']).sum())}


@torch.no_grad()
def decoder_renoise(model, base, gold, open_mask, ans_start, AL, mask_id,
                    rng):
    x = base['x'].clone()
    B = x.shape[0]
    device = x.device
    ar = torch.arange(B, device=device)
    # per-row open-cell index tables for uniform sampling
    n_open = open_mask.sum(1)
    max_open = int(n_open.max())
    idx_tab = torch.zeros(B, max_open, dtype=torch.long, device=device)
    for b in range(B):
        oc = open_mask[b].nonzero(as_tuple=True)[0]
        idx_tab[b, :len(oc)] = oc
    recommit = 0
    for r in range(max_open):
        u = torch.rand(B, generator=rng).to(device)
        pick = (u * n_open.float()).long().clamp(max=max_open - 1)
        pos = idx_tab[ar, pick]
        old = x[ar, ans_start + pos].clone()
        x[ar, ans_start + pos] = mask_id
        T = torch.zeros(B, AL, dtype=torch.bool, device=device)
        T[ar, pos] = True
        confidence_fill(model, x, T, ans_start, AL, mask_id)
        recommit += int((x[ar, ans_start + pos] == old).sum())
    pred = x[:, ans_start:ans_start + AL]
    ok = (pred == gold).all(dim=1)
    tot = B * max_open
    return {'acc_final': float(ok.float().mean()),
            'n_remask_total': tot,
            'recommit_rate': recommit / tot,
            'rescued': int((ok & ~base['correct']).sum()),
            'broken': int((~ok & base['correct']).sum())}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Orchestration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run(args):
    device = torch.device(args.device) if args.device else (
        torch.device('cuda') if torch.cuda.is_available()
        else torch.device('cpu'))
    set_grid(args.grid_n)
    # n_head is NOT inferable from checkpoint shapes; default to the maze
    # config's own value (exp_maze.N_HEAD = 4) rather than the addition
    # loader's default of 2 — a mismatch loads cleanly and silently
    # produces garbage.
    import addition_decode_analysis as ADA
    ADA.N_HEAD_OVERRIDE = args.n_head if args.n_head is not None else M.N_HEAD
    print(f"  building models with n_head={ADA.N_HEAD_OVERRIDE} "
          f"(exp_maze default {M.N_HEAD}; override with --n-head)")
    tok = M.build_tok()
    mask_id = tok.special_ids['mask']

    buckets = {}
    if args.natural_n > 0:
        buckets['natural'] = M.gen_test_data(args.natural_n,
                                             seed=args.test_seed)
    # natural-stratified corridor bins (no bias confound)
    if args.natural_pool > 0 and args.strata_edges:
        pool = M.gen_test_data(args.natural_pool, seed=args.test_seed + 1)
        edges = sorted(args.strata_edges)
        bins = [(0, edges[0])] + list(zip(edges, edges[1:])) + [(edges[-1],
                                                                10**9)]
        for lo, hi in bins:
            sel = [e for e in pool
                   if lo <= e['corridor_stats']['max_corridor_len'] < hi]
            if len(sel) >= 30:
                nm = (f'nat[{lo},{hi})' if hi < 10**9 else f'nat[{lo}+]')
                buckets[nm] = sel[:args.n_per_bucket]
    # constructed extrapolation band (straightness-bias generated — labeled)
    for cl in args.xtrap_sweep:
        sp = M.gen_min_corridor_test(args.n_per_bucket,
                                     seed=args.test_seed + 700 + cl,
                                     min_corridor=cl)
        if sp:
            buckets[f'xtrap>={cl}'] = sp
    # pure-corridor representation boundary
    if args.pure_n > 0:
        pc = M.gen_corner_case_test(args.pure_n, seed=args.test_seed + 9,
                                    category='pure_corridor')
        if pc:
            buckets['pure_corridor'] = pc
    enc = {}
    for name, entries in buckets.items():
        x0, gold, open_mask, ans_start = build_bucket(tok, entries, device)
        enc[name] = (x0, gold, open_mask, ans_start, entries)
        print(f"  bucket {name}: n={len(entries)}, "
              f"open/maze≈{float(open_mask.sum(1).float().mean()):.0f}")

    results = {'config': _jsonable(vars(args))}
    for seed in args.seeds:
        for method in args.methods:
            ck = args.ckpt_template.format(seed=seed, method=method)
            path = os.path.join(args.checkpoint_dir, ck)
            print(f"\n{'━'*66}\n  seed {seed} | {method} | {path}\n{'━'*66}")
            if not os.path.exists(path):
                print("  !! checkpoint missing, skipping")
                continue
            model = load_model(path, device)
            model.eval()
            key = f'seed{seed}_{method}'
            results[key] = {}
            for name, (x0, gold, open_mask, ans_start, entries) in enc.items():
                t0 = time.time()
                base = base_decode(model, x0, gold, open_mask, ans_start,
                                   M.ANS_LEN, mask_id, args.audit_every,
                                   args.loo_chunk)
                obs = audit_maze(base, gold, open_mask, entries, tok)
                rng = torch.Generator().manual_seed(
                    30_000 + seed * 113 + sum(ord(c) for c in name))
                entry = {
                    'base_acc': float(base['correct'].float().mean()),
                    'audit': obs,
                    'decoders': {
                        'loo': decoder_loo(model, base, gold, open_mask,
                                           ans_start, M.ANS_LEN, mask_id,
                                           args.loo_chunk),
                        'renoise': decoder_renoise(model, base, gold,
                                                   open_mask, ans_start,
                                                   M.ANS_LEN, mask_id, rng),
                    }}
                results[key][name] = entry
                a = obs
                print(f"  {name:>13s}  base={entry['base_acc']:.3f} "
                      f"loo={entry['decoders']['loo']['acc_final']:.3f} "
                      f"ren={entry['decoders']['renoise']['acc_final']:.3f}"
                      f"  ({time.time()-t0:.0f}s | fails={a['n_fail']} "
                      f"1cell={a['n_single_cell']} coh={a['n_coherent']} "
                      f"multi={a['n_multi_comp']} "
                      f"add/del={a['dir_totals']['add']}/"
                      f"{a['dir_totals']['delete']})")
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\n{'='*70}\n  MAZE — base | LOO ceiling† | re-noise†\n{'='*70}")
    for key in [k for k in results if k.startswith('seed')]:
        print(f"\n── {key} ──")
        rows = results[key]
        print(f"  {'bucket':>13s} {'base':>7s} {'loo':>7s} {'renoise':>8s} "
              f"{'coh-frac':>8s} {'q_med(1cell)':>12s} {'q_med(comp)':>11s} "
              f"{'q_med(ok)':>9s} {'blind%':>6s}")
        for name, e in rows.items():
            a = e['audit']
            nf = max(1, a['n_fail'])
            q1 = sorted(a['finalq_err_size1'])
            q2 = sorted(a['finalq_err_sizege2'])
            f1 = f"{q1[len(q1)//2]:.3f}" if q1 else '—'
            f2 = f"{q2[len(q2)//2]:.3f}" if q2 else '—'
            cq = sorted(a.get('control_q', []))
            cm = f"{cq[len(cq)//2]:.3f}" if cq else '—'
            recs = a['records']
            blind = (sum(1 for r in recs if r['err_q_rank'] > 0)
                     / max(1, len(recs)))
            print(f"  {name:>13s} {e['base_acc']:>7.3f} "
                  f"{e['decoders']['loo']['acc_final']:>7.3f} "
                  f"{e['decoders']['renoise']['acc_final']:>8.3f} "
                  f"{(a['n_coherent']+a['n_multi_comp'])/nf:>8.2f} "
                  f"{f1:>12s} {f2:>11s} {cm:>9s} {blind:>6.0%}")
    with open(args.out, 'w') as f:
        json.dump(_jsonable(results), f, indent=1)
    print(f"\n  💾 {args.out}")
    return results


if __name__ == '__main__':
    run(parse_args())
