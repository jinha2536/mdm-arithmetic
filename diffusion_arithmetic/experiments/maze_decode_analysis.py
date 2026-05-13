"""Comprehensive analysis of confidence vs dead-end-filling decoding on maze."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path

import torch
import torch.nn.functional as F

# ── sys.path setup (mirror exp_maze.py) ──────────────────────────────────────
import os, sys
if '__file__' in dir():
    _here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(_here))  # repo root → finds core/
    sys.path.insert(0, _here)                    # this dir → finds exp_maze
else:
    sys.path.insert(0, '.')

from core.train_utils import (  # type: ignore
    encode_samples, generate_diffusion, simulate_reveal_trajectory, DEVICE,
)

def _maybe_override_grid_n_from_argv():
    """Look at sys.argv for --checkpoint_dir and infer GRID_N from any checkpoint's wpe.weight shape."""
    import argparse
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--checkpoint_dir", type=str)
    pre.add_argument("--grid_n", type=int, default=None,
                     help="Force GRID_N (override checkpoint inference).")
    args, _ = pre.parse_known_args()

    import exp_maze
    if args.grid_n is not None:
        grid_n = args.grid_n
    elif args.checkpoint_dir is None:
        return  # nothing to do
    else:
        from pathlib import Path
        import torch as _torch
        ckpt_dir = Path(args.checkpoint_dir)
        ckpt_files = list(ckpt_dir.glob("checkpoint_*.pt"))
        if not ckpt_files:
            print(f"[warn] no checkpoint found in {ckpt_dir}; using default GRID_N")
            return
        sd = _torch.load(ckpt_files[0], map_location="cpu", weights_only=True)
        if "wpe.weight" not in sd:
            print("[warn] no wpe.weight in checkpoint; using default GRID_N")
            return
        block_size = sd["wpe.weight"].shape[0]
        cell_n = (block_size - 10) // 2
        import math
        grid_n = (int(math.isqrt(cell_n)) - 1) // 2
        if (2 * grid_n + 1) ** 2 != cell_n:
            print(f"[warn] block_size={block_size} doesn't yield clean GRID_N; "
                  f"using exp_maze default GRID_N={exp_maze.GRID_N}")
            return

    if grid_n != exp_maze.GRID_N:
        print(f"  [maze] overriding GRID_N: {exp_maze.GRID_N} → {grid_n}")
    exp_maze.GRID_N = grid_n
    exp_maze.GRID_H = 2 * grid_n + 1
    exp_maze.GRID_W = 2 * grid_n + 1
    exp_maze.CELL_N = exp_maze.GRID_H * exp_maze.GRID_W
    exp_maze.ANS_LEN = exp_maze.CELL_N


# Run BEFORE importing names from exp_maze
_maybe_override_grid_n_from_argv()

from exp_maze import (  # type: ignore
    GRID_H, GRID_W, CELL_N, ANS_LEN, build_tok,
    classify_path_cells, compute_corridor_segments, compute_corridor_stats,
    compute_dead_end_filling_order, find_path_bfs,
    gen_min_corridor_test, gen_corner_case_test, gen_maze_dfs,
    _make_entry, _maze_to_strings, _bucket_from_entries,
    build_test_suite,
)


# ── Configuration ────────────────────────────────────────────────────────────
METHODS = ["random", "papl", "puma"]
N_HEAD_OVERRIDE = None  # CLI-overridable. Default 8 (exp_maze.py: N_HEAD = 8)


#  Model loading (state-dict reconstruction via core.model.Transformer)
def load_model(ckpt_path, device):
    """Load checkpoint saved as state_dict and reconstruct via core.model.Transformer."""
    sd = torch.load(ckpt_path, map_location=device, weights_only=True)
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    if any(k.startswith("module.") for k in sd):
        sd = {k.removeprefix("module."): v for k, v in sd.items()}
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}

    vocab_size, n_embd = sd["wte.weight"].shape
    has_wpe = "wpe.weight" in sd
    block_size = sd["wpe.weight"].shape[0] if has_wpe else 512
    n_layer = 0
    while f"blocks.{n_layer}.attn.c_attn.weight" in sd:
        n_layer += 1
    n_head = N_HEAD_OVERRIDE if N_HEAD_OVERRIDE is not None else 8  # exp_maze default
    pos_enc = "absolute" if has_wpe else "rope"

    print(f"  inferred arch: vocab={vocab_size} n_embd={n_embd} "
          f"block_size={block_size} n_layer={n_layer} n_head={n_head} "
          f"pos_enc={pos_enc}")

    from core.model import Transformer  # type: ignore
    model = Transformer(
        vocab_size=vocab_size, block_size=block_size,
        n_layer=n_layer, n_head=n_head, n_embd=n_embd,
        dropout=0.0, is_causal=False, pos_enc=pos_enc,
    )
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    return model


#  Decode helper for maze (uses generate_diffusion with a r2l-style oracle
#  policy isn't applicable; instead we apply a custom oracle order from
#  dead-end-filling using the `static_order` mechanism in generate_diffusion).
@torch.no_grad()
def _decode_maze(model, tokenizer, entries, policy, device=None):
    """Maze decode helper."""
    device = device or DEVICE
    pad_id = tokenizer.special_ids['pad']
    mask_id = tokenizer.special_ids['mask']
    dot_id = tokenizer.encode('.')[0]

    B = len(entries)
    if B == 0:
        return (torch.empty(0, ANS_LEN, dtype=torch.long),
                torch.empty(0, ANS_LEN, dtype=torch.long),
                torch.empty(0, ANS_LEN, dtype=torch.bool),
                torch.empty(0, dtype=torch.long))

    strings = [e['string'] for e in entries]
    ids_all, ans_all = encode_samples(strings, tokenizer, max_len=None)
    ids_all = ids_all.to(device); ans_all = ans_all.to(device)
    pm = ans_all[0].item()
    assert (ans_all == pm).all(), "Variable ans_starts not supported"

    blank_mask_ans = (ids_all[:, :ANS_LEN] == dot_id)
    gold_ans = ids_all[:, pm:pm + ANS_LEN].cpu()

    # Pre-compute dead-end-filling rank tensor if needed
    de_rank = None
    if policy == 'dead_end_filling':
        de_rank = torch.full((B, ANS_LEN), 9999, dtype=torch.long, device=device)
        for i, e in enumerate(entries):
            order = e.get('de_filling_order')
            if order is None:
                grid = e.get('grid'); start = e.get('start'); end = e.get('end')
                if grid is None:
                    raise ValueError(
                        f"entry[{i}] missing 'de_filling_order' and grid for fallback")
                order, _ = compute_dead_end_filling_order(grid, start, end, GRID_H, GRID_W)
            for cell_idx, rank in order.items():
                if cell_idx < ANS_LEN:
                    de_rank[i, cell_idx] = int(rank)

    # Set up x: copy ids, mask answer region open cells
    x = ids_all.clone()
    for i in range(B):
        for j in range(ANS_LEN):
            if blank_mask_ans[i, j]:
                x[i, pm + j] = mask_id
    unmasked = torch.ones_like(x, dtype=torch.bool)
    for i in range(B):
        for j in range(ANS_LEN):
            if blank_mask_ans[i, j]:
                unmasked[i, pm + j] = False

    n_blanks = blank_mask_ans.sum(dim=1).cpu().tolist()
    max_stages = max(n_blanks) if n_blanks else 0

    for stage in range(max_stages):
        if not (~unmasked).any().item(): break
        logits = model(x)
        logits[:, :, mask_id] = -float("inf")
        max_logit, top1_tok = logits.max(dim=-1)

        for i in range(B):
            ans_um = unmasked[i, pm:pm + ANS_LEN]
            still = (~ans_um).nonzero(as_tuple=True)[0]
            if still.numel() == 0:
                continue
            abs_pos = pm + still
            confs = max_logit[i, abs_pos]
            if policy == 'confidence':
                chosen_local = confs.argmax().item()
            elif policy == 'dead_end_filling':
                # Among still-masked, pick smallest de_rank; tie-break by confidence
                ranks = de_rank[i, still]                         # [n_masked]
                # Use rank as primary (lower=better), -confidence as secondary
                # Add small epsilon * negative confidence so confidence breaks ties
                score = ranks.float() - 1e-6 * confs              # smaller=better
                chosen_local = score.argmin().item()
            else:
                raise ValueError(policy)
            chosen_off = still[chosen_local].item()
            chosen_abs = pm + chosen_off
            pred_tok = top1_tok[i, chosen_abs].item()
            x[i, chosen_abs] = pred_tok
            unmasked[i, chosen_abs] = True

    pred_ans = x[:, pm:pm + ANS_LEN].cpu()
    return pred_ans, gold_ans, blank_mask_ans.cpu(), ans_all.cpu()


#  Cell role classification (for stratifying analyses)
def cell_role_per_position(entry):
    """For each open cell index (in answer region), compute a fine-grained role analogous to addition's gkp partition."""
    grid = entry.get('grid')
    start = entry.get('start')
    end = entry.get('end')
    path = entry.get('path')
    if grid is None or start is None or end is None or path is None:
        # Try to recover from string
        return {}
    path_set = set(path)
    H, W = GRID_H, GRID_W

    # Existing classifications from exp_maze
    base_roles = classify_path_cells(grid, path_set, start, end, H, W)
    de_order, n_fillable = compute_dead_end_filling_order(grid, start, end, H, W)

    def _open_nbrs(i):
        r, c = i // W, i % W
        nbrs = []
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W:
                ni = nr * W + nc
                if grid[ni] != '#':
                    nbrs.append(ni)
        return nbrs

    open_cells = [i for i in range(H * W) if grid[i] != '#']
    junctions = {i for i in open_cells if i in path_set
                 and len(_open_nbrs(i)) >= 3}

    # BFS rank from start (depth along on-path cells)
    bfs_depth = {start: 0}
    q = deque([start])
    while q:
        u = q.popleft()
        for v in _open_nbrs(u):
            if v in path_set and v not in bfs_depth:
                bfs_depth[v] = bfs_depth[u] + 1
                q.append(v)

    role = {}
    for i in range(H * W):
        if grid[i] == '#':
            role[i] = 'wall'; continue
        if i == start:
            role[i] = 'start'; continue
        if i == end:
            role[i] = 'end'; continue

        if i in path_set:
            # On-path cell
            if i in junctions:
                # On-path junction: backbone if it's not dead-end-fillable
                role[i] = 'backbone_junction' if i not in de_order \
                          else 'junction'
            else:
                # Corridor cell on path
                nbrs_open = _open_nbrs(i)
                adj_to_junction = any(nb in junctions for nb in nbrs_open)
                adj_to_se = any(nb in (start, end) for nb in nbrs_open)
                if adj_to_se:
                    role[i] = 'corridor_endpoint'
                elif adj_to_junction:
                    role[i] = 'corridor_entrance'
                else:
                    # Backbone corridor (not removable by dead-end filling)
                    role[i] = 'backbone_corridor' if i not in de_order \
                              else 'corridor_interior'
        else:
            # Off-path open cell
            r = de_order.get(i, None)
            if r is None:
                role[i] = 'off_path_other'
            elif r < 3:
                role[i] = 'dead_end_tip'
            else:
                role[i] = 'dead_end_interior'

    return role


#  M1  Per-instance correctness comparison
@torch.no_grad()
def m1_per_instance(model, tokenizer, entries, device=None):
    """For each instance, record correctness under confidence vs dead-end-filling decode."""
    B = len(entries)
    if B == 0:
        return {"cross": {}, "per_corridor": {}, "examples": {}, "n": 0}

    pred_conf, gold, blank_mask, ans_starts = _decode_maze(
        model, tokenizer, entries, "confidence", device)
    pred_oracle, _, _, _ = _decode_maze(
        model, tokenizer, entries, "dead_end_filling", device)

    cross = {"both_correct": 0, "only_oracle": 0, "only_conf": 0, "neither": 0}
    per_corridor = defaultdict(lambda: dict(cross))
    failure_examples = {"only_oracle": [], "only_conf": [], "neither": []}

    for i in range(B):
        # Correctness only over decoded (open-cell) positions
        bm = blank_mask[i]
        c_correct = bool(torch.equal(pred_conf[i][bm], gold[i][bm]))
        o_correct = bool(torch.equal(pred_oracle[i][bm], gold[i][bm]))
        if c_correct and o_correct:
            cat = "both_correct"
        elif o_correct and not c_correct:
            cat = "only_oracle"
        elif c_correct and not o_correct:
            cat = "only_conf"
        else:
            cat = "neither"
        cross[cat] += 1
        cstats = entries[i].get('corridor_stats', {})
        L = cstats.get('max_corridor_len', 0)
        per_corridor[L][cat] += 1
        if cat in failure_examples and len(failure_examples[cat]) < 20:
            failure_examples[cat].append({
                "instance_idx": i,
                "max_corridor_len": L,
                "n_corridor_segments": cstats.get('n_corridor_segments', 0),
                "max_dead_end_len": cstats.get('max_dead_end_len', 0),
                "conf_pred": pred_conf[i].tolist(),
                "oracle_pred": pred_oracle[i].tolist(),
                "gold": gold[i].tolist(),
                "blank_mask": bm.tolist(),
            })

    return {
        "cross": cross,
        "per_corridor": {k: dict(v) for k, v in sorted(per_corridor.items())},
        "examples": failure_examples,
        "n": B,
    }


#  M2  Reveal-order vs dead-end-filling-order Kendall tau
@torch.no_grad()
def m2_kendall_tau(model, tokenizer, entries, K=16, tau=0.9, device=None):
    """Kendall tau between confidence-greedy reveal order (per instance) and dead-end-filling order, computed only over decoded (open-cell)..."""
    if not entries:
        return {"by_corridor": {}, "overall_mean_tau": None}

    strings = [e['string'] for e in entries]
    ids_all, ans_all = encode_samples(strings, tokenizer, max_len=None)
    ids_all = ids_all.to(device); ans_all = ans_all.to(device)
    dot_id = tokenizer.encode('.')[0]
    blank_masks = (ids_all[:, :ANS_LEN] == dot_id)

    traj = simulate_reveal_trajectory(
        model, tokenizer, ids_all, ans_all, ANS_LEN,
        blank_masks=blank_masks, K=K, tau=tau, device=device)

    reveal_stage = traj['reveal_stage']  # [N, ANS_LEN]
    N = reveal_stage.shape[0]

    by_corridor = defaultdict(list)
    for i in range(N):
        rs = reveal_stage[i].cpu().numpy()
        bm = blank_masks[i].cpu().numpy()
        decoded_positions = [j for j in range(ANS_LEN) if bm[j]]
        if not decoded_positions:
            continue

        # Reveal order: sort decoded positions by reveal_stage ascending
        actual_pairs = [(j, rs[j]) for j in decoded_positions
                        if rs[j] < 9999]
        actual_pairs.sort(key=lambda x: (x[1], x[0]))
        actual_order = [p[0] for p in actual_pairs]

        # Oracle order: sort by dead-end-filling rank
        de_order = entries[i].get('de_filling_order')
        if de_order is None:
            grid = entries[i].get('grid')
            start = entries[i].get('start')
            end = entries[i].get('end')
            if not grid: continue
            de_order, _ = compute_dead_end_filling_order(
                grid, start, end, GRID_H, GRID_W)
        # Cells in de_order have explicit rank; backbone cells (not in
        # de_order) get rank = max(de_order)+1 (decoded last).
        max_rank = max(de_order.values()) + 1 if de_order else 0
        oracle_pairs = []
        for j in decoded_positions:
            r = de_order.get(j, max_rank)
            oracle_pairs.append((j, r))
        oracle_pairs.sort(key=lambda x: (x[1], x[0]))
        oracle_order = [p[0] for p in oracle_pairs]

        tau_val = _kendall_tau(actual_order, oracle_order)
        cstats = entries[i].get('corridor_stats', {})
        L = cstats.get('max_corridor_len', 0)
        by_corridor[L].append(tau_val)

    all_taus = [t for vs in by_corridor.values() for t in vs]
    return {
        "by_corridor": {
            k: {"mean_tau": sum(v)/len(v), "n": len(v),
                "min": min(v), "max": max(v)}
            for k, v in sorted(by_corridor.items())
        },
        "overall_mean_tau": (sum(all_taus)/len(all_taus)) if all_taus else None,
        "K": K,
    }


def _kendall_tau(perm_a, perm_b):
    """Normalized Kendall tau-b for two equal-length permutations."""
    pos_a = {x: i for i, x in enumerate(perm_a)}
    pos_b = {x: i for i, x in enumerate(perm_b)}
    keys = sorted(set(perm_a) & set(perm_b))
    concordant = discordant = 0
    for i, x in enumerate(keys):
        for y in keys[i + 1:]:
            da = pos_a[x] - pos_a[y]
            db = pos_b[x] - pos_b[y]
            if da * db > 0: concordant += 1
            elif da * db < 0: discordant += 1
    total = concordant + discordant
    return (concordant - discordant) / total if total > 0 else 0.0


#  M3  Per-stage commit correctness by cell role
#  M4  Confidence calibration by corridor-length bin × role
#  (combined manual decode loop, as in addition A3/A4)
@torch.no_grad()
def m3_m4_decode_trace(model, tokenizer, entries, K=None, device=None):
    """Manual confidence-greedy decode over open cells."""
    device = device or DEVICE
    pad_id = tokenizer.special_ids['pad']
    mask_id = tokenizer.special_ids['mask']
    dot_id = tokenizer.encode('.')[0]

    B = len(entries)
    if B == 0:
        return {"m3": {}, "m4": {}}

    strings = [e['string'] for e in entries]
    ids_all, ans_all = encode_samples(strings, tokenizer, max_len=None)
    ids_all = ids_all.to(device); ans_all = ans_all.to(device)
    pm = ans_all[0].item()
    T = ids_all.shape[1]

    blank_mask_ans = (ids_all[:, :ANS_LEN] == dot_id)  # [B, ANS_LEN]

    # Build masked input: replace open-cell positions in answer region
    # with mask_id; keep walls/S/E intact.
    x = ids_all.clone()
    for i in range(B):
        for j in range(ANS_LEN):
            if blank_mask_ans[i, j]:
                x[i, pm + j] = mask_id

    # unmasked: True for positions whose token is fixed (not to be decoded)
    unmasked = torch.ones_like(x, dtype=torch.bool)
    for i in range(B):
        for j in range(ANS_LEN):
            if blank_mask_ans[i, j]:
                unmasked[i, pm + j] = False

    roles_per_instance = [cell_role_per_position(e) for e in entries]
    n_blanks = blank_mask_ans.sum(dim=1).cpu().tolist()  # for each instance

    K_eff = K if K is not None else max(n_blanks)  # one-cell-per-stage

    # M3 storage: role → stage → [n_correct, n_total]
    m3_acc = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    # M4 storage: corridor_bin → role → {correct: [confs], wrong: [confs]}
    m4_cal = defaultdict(lambda: defaultdict(lambda: {"correct": [], "wrong": []}))

    def _corridor_bin(L):
        if L <= 4:    return "<=4"
        if L <= 12:   return "5-12"
        if L <= 20:   return "13-20"
        if L <= 30:   return "21-30"
        return ">=31"

    for stage in range(K_eff):
        # If no instance has any masked position left, break
        any_masked = (~unmasked).any().item()
        if not any_masked:
            break
        logits = model(x)
        logits[:, :, mask_id] = -float("inf")
        max_logit, top1_tok = logits.max(dim=-1)  # [B, T]

        for i in range(B):
            if n_blanks[i] == 0:
                continue
            # Find still-masked positions in answer region
            ans_unmasked = unmasked[i, pm:pm + ANS_LEN]
            still = (~ans_unmasked).nonzero(as_tuple=True)[0]
            if still.numel() == 0:
                continue
            abs_pos = pm + still
            confs = max_logit[i, abs_pos]
            chosen_local = confs.argmax().item()
            chosen_off = still[chosen_local].item()
            chosen_abs = pm + chosen_off

            pred_tok = top1_tok[i, chosen_abs].item()
            gold_tok = ids_all[i, chosen_abs].item()
            correct = (pred_tok == gold_tok)

            role = roles_per_instance[i].get(chosen_off, '?')
            cstats = entries[i].get('corridor_stats', {})
            L = cstats.get('max_corridor_len', 0)
            cbin = _corridor_bin(L)

            # M3
            m3_acc[role][stage][1] += 1
            m3_acc[role][stage][0] += int(correct)
            # M4
            top1_prob = F.softmax(logits[i, chosen_abs], dim=-1).max().item()
            key = "correct" if correct else "wrong"
            m4_cal[cbin][role][key].append(top1_prob)

            # Commit
            x[i, chosen_abs] = pred_tok
            unmasked[i, chosen_abs] = True

    # Format outputs
    m3_out = {}
    for role, stages in m3_acc.items():
        m3_out[role] = {
            f"stage_{s}": {"n_correct": v[0], "n_total": v[1],
                          "acc": v[0] / max(v[1], 1)}
            for s, v in sorted(stages.items())
        }
    m4_out = {}
    for cbin, roles in m4_cal.items():
        m4_out[cbin] = {}
        for role, d in roles.items():
            cs = d["correct"]; ws = d["wrong"]
            m4_out[cbin][role] = {
                "n_correct": len(cs), "n_wrong": len(ws),
                "mean_conf_correct": sum(cs) / len(cs) if cs else None,
                "mean_conf_wrong":   sum(ws) / len(ws) if ws else None,
            }
    return {"m3": m3_out, "m4": m4_out}


#  M5  Constructed long-corridor slice
@torch.no_grad()
def m5_constructed_slice(model, tokenizer, n_per_bucket=300, device=None):
    """Sweep min_corridor in {4, 8, 12, 16, 20, 24, 28}; evaluate confidence and dead-end-filling decoding."""
    out = {}
    for L in [4, 8, 12, 16, 20, 24, 28]:
        entries = gen_min_corridor_test(n_per_bucket, seed=2026 + L,
                                        min_corridor=L)
        if not entries:
            continue
        for policy, label in [("confidence", "confidence"),
                              ("dead_end_filling", "oracle")]:
            pred, gold, bm, _ = _decode_maze(model, tokenizer, entries,
                                             policy, device)
            correct = 0
            for i in range(pred.shape[0]):
                if torch.equal(pred[i][bm[i]], gold[i][bm[i]]):
                    correct += 1
            out[f"L{L}_{label}"] = {
                "accuracy": correct / max(pred.shape[0], 1),
                "n": pred.shape[0],
                "min_corridor_len": L,
            }
    return out


#  M8  Confidence-failure dissection (full per-example trace)
@torch.no_grad()
def m8_failure_dissection(model, tokenizer, entries, max_examples=50, device=None):
    """For each instance, run BOTH confidence and dead-end-filling decode manually, logging per-stage commit info."""
    device = device or DEVICE
    pad_id = tokenizer.special_ids['pad']
    mask_id = tokenizer.special_ids['mask']
    dot_id = tokenizer.encode('.')[0]
    tok_0 = tokenizer.encode('0')[0]
    tok_1 = tokenizer.encode('1')[0]

    B = len(entries)
    if B == 0:
        return {"summary": {}, "only_oracle_examples": [], "n_total": 0}

    strings = [e['string'] for e in entries]
    ids_all, ans_all = encode_samples(strings, tokenizer, max_len=None)
    ids_all = ids_all.to(device); ans_all = ans_all.to(device)
    pm = ans_all[0].item()
    T = ids_all.shape[1]
    blank_mask_ans = (ids_all[:, :ANS_LEN] == dot_id)
    gold_ans = ids_all[:, pm:pm + ANS_LEN].cpu()

    roles_per_instance = [cell_role_per_position(e) for e in entries]
    de_orders_per_instance = []
    for e in entries:
        order = e.get('de_filling_order')
        if order is None:
            grid = e.get('grid'); start = e.get('start'); end = e.get('end')
            if grid is not None:
                order, _ = compute_dead_end_filling_order(
                    grid, start, end, GRID_H, GRID_W)
            else:
                order = {}
        de_orders_per_instance.append(order)

    def _trace_decode(policy):
        x = ids_all.clone()
        for i in range(B):
            for j in range(ANS_LEN):
                if blank_mask_ans[i, j]:
                    x[i, pm + j] = mask_id
        unmasked = torch.ones_like(x, dtype=torch.bool)
        for i in range(B):
            for j in range(ANS_LEN):
                if blank_mask_ans[i, j]:
                    unmasked[i, pm + j] = False

        traces = [[] for _ in range(B)]
        n_blanks = blank_mask_ans.sum(dim=1).cpu().tolist()

        max_stages = max(n_blanks)
        for stage in range(max_stages):
            any_left = (~unmasked).any().item()
            if not any_left: break
            logits = model(x)
            logits[:, :, mask_id] = -float("inf")
            max_logit, top1_tok = logits.max(dim=-1)

            for i in range(B):
                ans_um = unmasked[i, pm:pm + ANS_LEN]
                still = (~ans_um).nonzero(as_tuple=True)[0]
                if still.numel() == 0:
                    continue
                abs_pos = pm + still

                if policy == 'confidence':
                    chosen_local = max_logit[i, abs_pos].argmax().item()
                elif policy == 'dead_end_filling':
                    # Pick still-masked with smallest dead-end-filling rank;
                    # break ties by argmax confidence.
                    de = de_orders_per_instance[i]
                    max_rank = (max(de.values()) + 1) if de else 0
                    ranks = torch.tensor(
                        [de.get(j.item(), max_rank) for j in still],
                        device=device)
                    # Lower rank first; tie-break by max-logit
                    confs = max_logit[i, abs_pos]
                    # sort key: (rank, -conf)
                    score = ranks.float() - 1e-6 * confs
                    chosen_local = score.argmin().item()
                else:
                    raise ValueError(policy)

                chosen_off = still[chosen_local].item()
                chosen_abs = pm + chosen_off
                pred_tok = top1_tok[i, chosen_abs].item()
                gold_tok = ids_all[i, chosen_abs].item()
                probs = F.softmax(logits[i, chosen_abs], dim=-1)
                top2 = probs.topk(2)
                top1_prob = top2.values[0].item()
                top2_prob = top2.values[1].item()
                gold_prob = probs[gold_tok].item()
                role = roles_per_instance[i].get(chosen_off, '?')

                # Direction of error (binary):
                if pred_tok != gold_tok:
                    if gold_tok == tok_1 and pred_tok == tok_0:
                        direction = 'path_to_off'
                    elif gold_tok == tok_0 and pred_tok == tok_1:
                        direction = 'off_to_path'
                    else:
                        direction = 'other'
                else:
                    direction = 'correct'

                traces[i].append({
                    "stage": stage,
                    "chosen_off": chosen_off,
                    "chosen_row": chosen_off // GRID_W,
                    "chosen_col": chosen_off % GRID_W,
                    "role": role,
                    "de_rank": de_orders_per_instance[i].get(chosen_off, None),
                    "committed_tok": pred_tok,
                    "gold_tok": gold_tok,
                    "is_correct": (pred_tok == gold_tok),
                    "direction": direction,
                    "top1_prob": top1_prob,
                    "top2_prob": top2_prob,
                    "gold_prob": gold_prob,
                    "margin": top1_prob - top2_prob,
                })

                x[i, chosen_abs] = pred_tok
                unmasked[i, chosen_abs] = True

        return traces, x[:, pm:pm + ANS_LEN].cpu()

    print("    [m8] running confidence decode trace...")
    conf_traces, conf_preds = _trace_decode('confidence')
    print("    [m8] running dead-end-filling decode trace...")
    oracle_traces, oracle_preds = _trace_decode('dead_end_filling')

    only_oracle_examples = []
    only_conf_examples = []
    neither_examples = []
    summary = {"both_correct": 0, "only_oracle": 0, "only_conf": 0, "neither": 0}

    for i in range(B):
        bm = blank_mask_ans[i].cpu()
        c_correct = bool(torch.equal(conf_preds[i][bm], gold_ans[i][bm]))
        o_correct = bool(torch.equal(oracle_preds[i][bm], gold_ans[i][bm]))
        if c_correct and o_correct:
            cat = "both_correct"
        elif o_correct and not c_correct:
            cat = "only_oracle"
        elif c_correct and not o_correct:
            cat = "only_conf"
        else:
            cat = "neither"
        summary[cat] += 1

        if cat == "both_correct":
            continue

        # First wrong commit (in answer-region offset)
        wrong_offs = [j for j in range(ANS_LEN)
                      if bm[j] and conf_preds[i][j].item() != gold_ans[i][j].item()]
        if not wrong_offs:
            continue
        first_wrong = wrong_offs[0]

        conf_commit = next((t for t in conf_traces[i]
                            if t["chosen_off"] == first_wrong), None)
        oracle_commit = next((t for t in oracle_traces[i]
                              if t["chosen_off"] == first_wrong), None)
        conf_stage_w = conf_commit["stage"] if conf_commit else None
        preceding = ([{"stage": t["stage"], "chosen_off": t["chosen_off"],
                       "row": t["chosen_row"], "col": t["chosen_col"],
                       "role": t["role"], "de_rank": t["de_rank"],
                       "tok": t["committed_tok"], "correct": t["is_correct"],
                       "top1_prob": t["top1_prob"]}
                      for t in conf_traces[i] if t["stage"] < conf_stage_w]
                     if conf_stage_w is not None else [])

        cstats = entries[i].get('corridor_stats', {})
        record = {
            "instance_idx": i,
            "max_corridor_len": cstats.get('max_corridor_len', 0),
            "n_corridor_segments": cstats.get('n_corridor_segments', 0),
            "max_dead_end_len": cstats.get('max_dead_end_len', 0),
            "all_conf_wrong_offs": wrong_offs,
            "first_wrong_off": first_wrong,
            "first_wrong_role": roles_per_instance[i].get(first_wrong, '?'),
            "first_wrong_row": first_wrong // GRID_W,
            "first_wrong_col": first_wrong % GRID_W,
            "first_wrong_de_rank": de_orders_per_instance[i].get(first_wrong, None),
            "conf_commit": conf_commit,
            "oracle_commit": oracle_commit,
            "conf_preceding_reveals": preceding,
            "n_conf_wrong_total": len(wrong_offs),
        }
        target = (only_oracle_examples if cat == "only_oracle"
                  else only_conf_examples if cat == "only_conf"
                  else neither_examples)
        if len(target) < max_examples:
            target.append(record)

    return {
        "summary": summary,
        "only_oracle_examples": only_oracle_examples,
        "only_conf_examples": only_conf_examples,
        "neither_examples": neither_examples,
        "n_total": B,
    }


#  M9  Confidence-ranking margin diagnostic
@torch.no_grad()
def m9_ranking_margin(model, tokenizer, entries, max_examples=50, device=None):
    """At each conf decode stage, log chosen and runner-up positions (max-logit ranking) , same as A9 in addition."""
    device = device or DEVICE
    pad_id = tokenizer.special_ids['pad']
    mask_id = tokenizer.special_ids['mask']
    dot_id = tokenizer.encode('.')[0]

    B = len(entries)
    if B == 0:
        return {"summary": {}, "records": [], "n_total": 0}

    strings = [e['string'] for e in entries]
    ids_all, ans_all = encode_samples(strings, tokenizer, max_len=None)
    ids_all = ids_all.to(device); ans_all = ans_all.to(device)
    pm = ans_all[0].item()
    T = ids_all.shape[1]
    blank_mask_ans = (ids_all[:, :ANS_LEN] == dot_id)
    gold_ans = ids_all[:, pm:pm + ANS_LEN].cpu()

    roles_per_instance = [cell_role_per_position(e) for e in entries]

    x = ids_all.clone()
    for i in range(B):
        for j in range(ANS_LEN):
            if blank_mask_ans[i, j]:
                x[i, pm + j] = mask_id
    unmasked = torch.ones_like(x, dtype=torch.bool)
    for i in range(B):
        for j in range(ANS_LEN):
            if blank_mask_ans[i, j]:
                unmasked[i, pm + j] = False

    traces = [[] for _ in range(B)]
    stage_of_first_wrong = [None] * B
    full_ranking_at_wrong = [None] * B
    final_pred = torch.full((B, ANS_LEN), -1, dtype=torch.long, device=device)

    n_blanks = blank_mask_ans.sum(dim=1).cpu().tolist()
    max_stages = max(n_blanks) if n_blanks else 0

    for stage in range(max_stages):
        if not (~unmasked).any().item(): break
        logits = model(x)
        logits[:, :, mask_id] = -float("inf")
        max_logit, top1_tok = logits.max(dim=-1)
        top1_prob_all = F.softmax(logits, dim=-1).max(dim=-1).values

        for i in range(B):
            ans_um = unmasked[i, pm:pm + ANS_LEN]
            still = (~ans_um).nonzero(as_tuple=True)[0]
            if still.numel() == 0: continue
            abs_pos = pm + still
            scores = max_logit[i, abs_pos]
            sorted_scores, sorted_idx = scores.sort(descending=True)

            chosen_local = sorted_idx[0].item()
            chosen_off = still[chosen_local].item()
            chosen_abs = pm + chosen_off
            chosen_role = roles_per_instance[i].get(chosen_off, '?')
            chosen_top1 = top1_prob_all[i, chosen_abs].item()
            chosen_logit = sorted_scores[0].item()

            runner_top1 = runner_role = runner_off = None
            runner_logit = None
            if sorted_scores.numel() > 1:
                runner_local = sorted_idx[1].item()
                runner_off = still[runner_local].item()
                runner_role = roles_per_instance[i].get(runner_off, '?')
                runner_top1 = top1_prob_all[i, pm + runner_off].item()
                runner_logit = sorted_scores[1].item()

            margin_logit = (chosen_logit - runner_logit) if runner_logit is not None else None
            margin_prob  = (chosen_top1 - runner_top1) if runner_top1 is not None else None

            pred_tok = top1_tok[i, chosen_abs].item()
            gold_tok = ids_all[i, chosen_abs].item()
            is_wrong = (pred_tok != gold_tok)

            traces[i].append({
                "stage": stage,
                "chosen_off": chosen_off,
                "chosen_role": chosen_role,
                "chosen_top1": chosen_top1,
                "chosen_logit": chosen_logit,
                "chosen_correct": (not is_wrong),
                "runner_off": runner_off,
                "runner_role": runner_role,
                "runner_top1": runner_top1,
                "runner_logit": runner_logit,
                "margin_logit": margin_logit,
                "margin_prob":  margin_prob,
            })

            if is_wrong and stage_of_first_wrong[i] is None:
                stage_of_first_wrong[i] = stage
                top_k = min(10, sorted_scores.numel())
                ranking = []
                for r in range(top_k):
                    local = sorted_idx[r].item()
                    off = still[local].item()
                    rl = roles_per_instance[i].get(off, '?')
                    ranking.append({
                        "rank": r,
                        "off": off, "row": off // GRID_W, "col": off % GRID_W,
                        "role": rl,
                        "top1_prob": top1_prob_all[i, pm + off].item(),
                        "max_logit": sorted_scores[r].item(),
                        "would_be_correct": (
                            top1_tok[i, pm + off].item()
                            == ids_all[i, pm + off].item()
                        ),
                    })
                full_ranking_at_wrong[i] = ranking

            x[i, chosen_abs] = pred_tok
            unmasked[i, chosen_abs] = True
            final_pred[i, chosen_off] = pred_tok

    summary = {"both_correct": 0, "wrong": 0}
    records = []
    for i in range(B):
        bm = blank_mask_ans[i].cpu()
        ok = bool(torch.equal(final_pred[i].cpu()[bm], gold_ans[i][bm]))
        if ok:
            summary["both_correct"] += 1
            continue
        summary["wrong"] += 1
        if len(records) >= max_examples: continue
        records.append({
            "instance_idx": i,
            "max_corridor_len": entries[i].get('corridor_stats', {}).get('max_corridor_len', 0),
            "stage_of_first_wrong": stage_of_first_wrong[i],
            "full_ranking_at_wrong_stage": full_ranking_at_wrong[i],
            "preceding_trace": traces[i][: (stage_of_first_wrong[i] or 0) + 1],
        })

    # Aggregate margin stats
    all_logit = []; wrong_logit = []
    for i in range(B):
        for t in traces[i]:
            if t.get("margin_logit") is not None:
                all_logit.append(t["margin_logit"])
        if stage_of_first_wrong[i] is not None:
            s = stage_of_first_wrong[i]
            if s < len(traces[i]) and traces[i][s].get("margin_logit") is not None:
                wrong_logit.append(traces[i][s]["margin_logit"])

    def _stats(xs):
        if not xs: return {"mean": None, "min": None, "max": None}
        return {"mean": sum(xs)/len(xs), "min": min(xs), "max": max(xs)}

    return {
        "summary": summary,
        "records": records,
        "n_total": B,
        "all_stage_n": len(all_logit),
        "wrong_stage_n": len(wrong_logit),
        "all_stage_margin_logit": _stats(all_logit),
        "wrong_stage_margin_logit": _stats(wrong_logit),
    }


#  M10  Spatial pattern of wrong commits
@torch.no_grad()
def m10_spatial_pattern(model, tokenizer, entries, device=None):
    """Aggregate: where do confidence-decode wrong commits land in the maze?"""
    device = device or DEVICE
    mask_id = tokenizer.special_ids['mask']
    dot_id = tokenizer.encode('.')[0]
    tok_0 = tokenizer.encode('0')[0]
    tok_1 = tokenizer.encode('1')[0]

    B = len(entries)
    if B == 0:
        return {"by_role": {}, "by_de_rank_bin": {}, "by_direction": {}}

    strings = [e['string'] for e in entries]
    ids_all, ans_all = encode_samples(strings, tokenizer, max_len=None)
    ids_all = ids_all.to(device); ans_all = ans_all.to(device)
    pm = ans_all[0].item()
    blank_mask_ans = (ids_all[:, :ANS_LEN] == dot_id)

    roles_per_instance = [cell_role_per_position(e) for e in entries]
    de_orders = []
    for e in entries:
        order = e.get('de_filling_order')
        if order is None:
            grid = e.get('grid'); start = e.get('start'); end = e.get('end')
            if grid is not None:
                order, _ = compute_dead_end_filling_order(grid, start, end, GRID_H, GRID_W)
            else:
                order = {}
        de_orders.append(order)

    x = ids_all.clone()
    for i in range(B):
        for j in range(ANS_LEN):
            if blank_mask_ans[i, j]:
                x[i, pm + j] = mask_id
    unmasked = torch.ones_like(x, dtype=torch.bool)
    for i in range(B):
        for j in range(ANS_LEN):
            if blank_mask_ans[i, j]:
                unmasked[i, pm + j] = False

    by_role = defaultdict(lambda: {"correct": 0, "wrong": 0,
                                   "wrong_path_to_off": 0,
                                   "wrong_off_to_path": 0})
    de_rank_buckets = defaultdict(lambda: {"correct": 0, "wrong": 0})

    n_blanks = blank_mask_ans.sum(dim=1).cpu().tolist()
    max_stages = max(n_blanks) if n_blanks else 0

    for stage in range(max_stages):
        if not (~unmasked).any().item(): break
        logits = model(x)
        logits[:, :, mask_id] = -float("inf")
        max_logit, top1_tok = logits.max(dim=-1)
        for i in range(B):
            ans_um = unmasked[i, pm:pm + ANS_LEN]
            still = (~ans_um).nonzero(as_tuple=True)[0]
            if still.numel() == 0: continue
            abs_pos = pm + still
            chosen_local = max_logit[i, abs_pos].argmax().item()
            chosen_off = still[chosen_local].item()
            chosen_abs = pm + chosen_off
            pred_tok = top1_tok[i, chosen_abs].item()
            gold_tok = ids_all[i, chosen_abs].item()
            correct = (pred_tok == gold_tok)
            role = roles_per_instance[i].get(chosen_off, '?')
            de = de_orders[i]
            de_max = max(de.values()) + 1 if de else 0
            de_rank = de.get(chosen_off, de_max)
            # Bucket de_rank for backbone visualization
            if de_rank == de_max:
                bucket = "backbone"
            elif de_rank < 3:
                bucket = "tip(0-2)"
            elif de_rank < 10:
                bucket = "fill(3-9)"
            else:
                bucket = "fill(>=10)"

            by_role[role]["correct" if correct else "wrong"] += 1
            de_rank_buckets[bucket]["correct" if correct else "wrong"] += 1
            if not correct:
                if gold_tok == tok_1 and pred_tok == tok_0:
                    by_role[role]["wrong_path_to_off"] += 1
                elif gold_tok == tok_0 and pred_tok == tok_1:
                    by_role[role]["wrong_off_to_path"] += 1
            x[i, chosen_abs] = pred_tok
            unmasked[i, chosen_abs] = True

    return {
        "by_role": dict(by_role),
        "by_de_rank_bin": dict(de_rank_buckets),
    }


#  M11  Branch decision errors at junctions
@torch.no_grad()
def m11_branch_decisions(model, tokenizer, entries, device=None):
    """At each junction in the maze (node with degree ≥ 3 on the open-cell graph), examine whether confidence decode commits the correct branch as..."""
    device = device or DEVICE
    dot_id = tokenizer.encode('.')[0]
    mask_id = tokenizer.special_ids['mask']
    tok_0 = tokenizer.encode('0')[0]
    tok_1 = tokenizer.encode('1')[0]

    if not entries:
        return {"junctions_total": 0, "by_branch_count": {}}

    # Run confidence decode once, get final predictions
    pred_conf, gold, blank_mask, ans_starts = _decode_maze(
        model, tokenizer, entries, "confidence", device)
    B = pred_conf.shape[0]

    by_branch_count = defaultdict(lambda: {"junctions": 0, "wrong_decisions": 0})
    junctions_total = 0
    junctions_with_error = 0

    H, W = GRID_H, GRID_W
    for i in range(B):
        e = entries[i]
        grid = e.get('grid')
        path = e.get('path')
        start = e.get('start')
        end = e.get('end')
        if grid is None or path is None: continue
        path_set = set(path)

        def _open_nbrs(c):
            r, col = c // W, c % W
            nbrs = []
            for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
                nr, nc = r+dr, col+dc
                if 0 <= nr < H and 0 <= nc < W and grid[nr*W+nc] != '#':
                    nbrs.append(nr*W+nc)
            return nbrs

        # Find junctions on the path
        for j in path:
            nbrs = _open_nbrs(j)
            if len(nbrs) >= 3:
                junctions_total += 1
                # For each branch off this junction, check whether the
                # commit at the next cell is consistent with gold.
                wrong_here = 0
                for nb in nbrs:
                    if nb in (start, end): continue
                    pred_tok = pred_conf[i, nb].item()
                    gold_tok = gold[i, nb].item()
                    if pred_tok != gold_tok:
                        wrong_here += 1
                by_branch_count[len(nbrs)]["junctions"] += 1
                by_branch_count[len(nbrs)]["wrong_decisions"] += wrong_here
                if wrong_here > 0:
                    junctions_with_error += 1

    return {
        "junctions_total": junctions_total,
        "junctions_with_error": junctions_with_error,
        "junction_error_rate": (junctions_with_error / max(junctions_total, 1)),
        "by_branch_count": {k: dict(v) for k, v in sorted(by_branch_count.items())},
    }


#  Driver
ALL_ANALYSES = {
    "m1":  "Per-instance correctness comparison",
    "m2":  "Reveal-order Kendall-tau vs dead-end-filling",
    "m34": "Per-stage commit correctness + calibration (combined)",
    "m5":  "Constructed long-corridor slice",
    "m8":  "Confidence-failure dissection",
    "m9":  "Confidence-ranking margin diagnostic",
    "m10": "Spatial pattern of wrong commits",
    "m11": "Branch decision errors at junctions",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", required=True, type=Path)
    ap.add_argument("--out_dir", default=Path("./decode_analysis_maze"), type=Path)
    ap.add_argument("--analyses", default="all",
                    help="Comma-separated subset of " + ",".join(ALL_ANALYSES))
    ap.add_argument("--n_per_bucket", default=300, type=int)
    ap.add_argument("--K_puma", default=32, type=int,
                    help="K for simulate_reveal_trajectory in M2. Default 32 "
                         "matches the upper end of maze's PUMA K-schedule "
                         "(K_START=12, step=3, auto-derived end ~32-40 "
                         "depending on corridor stats). Override via CLI to "
                         "match your specific training run's final K.")
    ap.add_argument("--tau", default=0.9, type=float)
    ap.add_argument("--n_head", default=None, type=int,
                    help="Override n_head (default 8 for exp_maze).")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.n_head is not None:
        global N_HEAD_OVERRIDE
        N_HEAD_OVERRIDE = args.n_head

    selected = (list(ALL_ANALYSES) if args.analyses == "all"
                else args.analyses.split(","))
    tokenizer = build_tok()
    device = DEVICE
    print(f"Device: {device}")
    print(f"Selected analyses: {selected}")

    # Build corridor-length stratified buckets
    print("\nBuilding test buckets (corridor sweep)...")
    corridor_buckets = {}
    for L in [4, 8, 12, 16, 20, 24, 28]:
        ent = gen_min_corridor_test(args.n_per_bucket, seed=5000 + L,
                                    min_corridor=L)
        if ent:
            corridor_buckets[L] = ent
            print(f"  corridor>={L}: {len(ent)} entries")

    for method in METHODS:
        ckpt = args.checkpoint_dir / f"checkpoint_{method}.pt"
        if not ckpt.exists():
            print(f"\n[skip] {ckpt} not found")
            continue
        print(f"\n=== {method} ===")
        model = load_model(ckpt, device)
        results = {"method": method, "n_per_bucket": args.n_per_bucket}

        if "m1" in selected:
            print("  M1: per-instance correctness comparison")
            results["m1"] = {}
            for L, ent in corridor_buckets.items():
                results["m1"][f"corridor_{L}"] = m1_per_instance(
                    model, tokenizer, ent, device=device)

        if "m2" in selected:
            print("  M2: Kendall tau vs dead-end-filling order")
            results["m2"] = {}
            for L, ent in corridor_buckets.items():
                results["m2"][f"corridor_{L}"] = m2_kendall_tau(
                    model, tokenizer, ent, K=args.K_puma, tau=args.tau, device=device)

        if "m34" in selected:
            print("  M3+M4: per-stage commit correctness + calibration")
            all_ent = [e for ent in corridor_buckets.values() for e in ent]
            out34 = m3_m4_decode_trace(model, tokenizer, all_ent, K=None, device=device)
            results["m3"] = out34["m3"]
            results["m4"] = out34["m4"]

        if "m5" in selected:
            print("  M5: constructed long-corridor slice")
            results["m5"] = m5_constructed_slice(model, tokenizer,
                                                 n_per_bucket=args.n_per_bucket,
                                                 device=device)

        if "m8" in selected:
            print("  M8: failure dissection")
            results["m8"] = {}
            for L, ent in corridor_buckets.items():
                results["m8"][f"corridor_{L}"] = m8_failure_dissection(
                    model, tokenizer, ent, max_examples=50, device=device)

        if "m9" in selected:
            print("  M9: ranking-margin diagnostic")
            results["m9"] = {}
            for L, ent in corridor_buckets.items():
                results["m9"][f"corridor_{L}"] = m9_ranking_margin(
                    model, tokenizer, ent, max_examples=50, device=device)

        if "m10" in selected:
            print("  M10: spatial pattern of wrong commits")
            results["m10"] = {}
            for L, ent in corridor_buckets.items():
                results["m10"][f"corridor_{L}"] = m10_spatial_pattern(
                    model, tokenizer, ent, device=device)

        if "m11" in selected:
            print("  M11: branch decision errors")
            results["m11"] = {}
            for L, ent in corridor_buckets.items():
                results["m11"][f"corridor_{L}"] = m11_branch_decisions(
                    model, tokenizer, ent, device=device)

        out_path = args.out_dir / f"analysis_{method}.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"  wrote {out_path}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nDone.")


if __name__ == "__main__":
    main()
