"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Maze v2 — Corridor Dependency Learning + PUMA Coverage Deficit
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Core question: Does PUMA's coverage deficit on rare long corridors
  mirror the addition carry-chain phenomenon?

  Dependency analog:
    Addition carry chain  ←→  Maze corridor (no-fork path segment)
    g/k position          ←→  Junction (3+ open neighbors on path)
    p position            ←→  Corridor cell (exactly 2 open neighbors)
    LSB oracle order      ←→  BFS-from-start oracle order

  Training: random vs puma (blank-only masking on open cells)
  Decode:   confidence | bfs_oracle | random
  Analyses: corridor stratification, BFS depth × accuracy,
            PUMA coverage, error localization, corridor sweep
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import sys, os, time, math, json, random
from collections import defaultdict, deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                if '__file__' in dir() else '.')
from core.tokenizer import CharTokenizer
from core.model import Transformer
from core.train_utils import mount_drive, save_results, encode_samples, DEVICE

EXP_NAME = 'exp_maze_v2'

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Config
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GRID_N = 7                  # logical size → actual (2*7+1)=15 → 15×15=225 cells
GRID_H = 2 * GRID_N + 1
GRID_W = GRID_H
CELL_N = GRID_H * GRID_W   # = 225
ANS_LEN = CELL_N            # solution is same size as puzzle

N_TRAIN = 50000; N_TEST = 1000; BATCH_SIZE = 128
MAX_ITERS = 200000; EVAL_EVERY = 5000; LOG_EVERY = 1000
GEN_EVAL_EVERY = 10000; GEN_EVAL_N = 200

MASK_TYPES = ['random', 'puma']
DECODE_POLICIES = ['confidence', 'bfs_oracle', 'random']

N_LAYER = 4; N_HEAD = 4; N_EMBD = 128; DROPOUT = 0.1; POS_ENC = 'absolute'
LR = 3e-4; MIN_LR = 1e-5; WARMUP_ITERS = 2000; GRAD_CLIP = 1.0
WEIGHT_DECAY = 0.01; EMA_DECAY = 0.9999

PUMA_TAU = 0.9; PUMA_K = 12
SEED = 42
STRAIGHTNESS_BIAS = 0.0  # 0.0=normal DFS, higher=longer corridors in training

# Corridor sweep lengths for evaluation
CORRIDOR_SWEEP = [2, 4, 6, 8, 10, 12, 15, 20, 25, 30]


def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--grid-n', type=int)
    p.add_argument('--n-train', type=int); p.add_argument('--n-test', type=int)
    p.add_argument('--max-iters', type=int); p.add_argument('--batch-size', type=int)
    p.add_argument('--eval-every', type=int); p.add_argument('--gen-eval-every', type=int)
    p.add_argument('--n-layer', type=int); p.add_argument('--n-head', type=int)
    p.add_argument('--n-embd', type=int); p.add_argument('--dropout', type=float)
    p.add_argument('--lr', type=float)
    p.add_argument('--puma-tau', type=float); p.add_argument('--puma-k', type=int)
    p.add_argument('--masks', nargs='+'); p.add_argument('--decode', nargs='+')
    p.add_argument('--straightness-bias', type=float)
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
                   'lr': 'LR', 'puma_tau': 'PUMA_TAU', 'puma_k': 'PUMA_K',
                   'seed': 'SEED', 'straightness_bias': 'STRAIGHTNESS_BIAS'}.items():
        v = getattr(args, a, None)
        if v is not None: g[gl] = v
    if args.grid_n:
        g['GRID_N'] = args.grid_n
        g['GRID_H'] = 2 * args.grid_n + 1; g['GRID_W'] = g['GRID_H']
        g['CELL_N'] = g['GRID_H'] * g['GRID_W']; g['ANS_LEN'] = g['CELL_N']
    if args.masks: g['MASK_TYPES'] = args.masks
    if args.decode: g['DECODE_POLICIES'] = args.decode
    return args


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Maze Generation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _neighbors_2(r, c, H, W):
    """Passage neighbors 2 steps away (for DFS carving)."""
    for dr, dc in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
        nr, nc = r + dr, c + dc
        if 0 < nr < H and 0 < nc < W:
            yield nr, nc, dr, dc


def gen_maze_dfs(grid_n, rng, straightness_bias=0.0):
    """Generate perfect maze via randomized DFS.
    grid_n: logical size → actual grid (2*grid_n+1)^2
    straightness_bias: 0.0=random, 0.5+=prefer continuing in same direction
    Returns: (grid, start, end) where grid is flat list of '#'/'.', start/end are indices.
    """
    H = W = 2 * grid_n + 1
    grid = ['#'] * (H * W)

    def _set(r, c, v): grid[r * W + c] = v
    def _get(r, c): return grid[r * W + c]

    start_rc = (1, 1)
    end_rc = (H - 2, W - 2)
    _set(*start_rc, '.')
    stack = [start_rc]
    visited = {start_rc}
    last_dir = None

    while stack:
        r, c = stack[-1]
        nbrs = [(nr, nc, dr, dc) for nr, nc, dr, dc in _neighbors_2(r, c, H, W)
                if (nr, nc) not in visited]
        if not nbrs:
            stack.pop(); last_dir = None; continue

        # Apply straightness bias
        chosen = None
        if last_dir and straightness_bias > 0:
            straight = [(nr, nc, dr, dc) for nr, nc, dr, dc in nbrs
                        if (dr, dc) == last_dir]
            if straight and rng.random() < straightness_bias:
                chosen = straight[0]
        if chosen is None:
            chosen = nbrs[rng.randint(0, len(nbrs) - 1)]

        nr, nc, dr, dc = chosen
        _set(r + dr // 2, c + dc // 2, '.')  # carve wall between
        _set(nr, nc, '.')
        visited.add((nr, nc))
        stack.append((nr, nc))
        last_dir = (dr, dc)

    si = start_rc[0] * W + start_rc[1]
    ei = end_rc[0] * W + end_rc[1]
    return grid, si, ei


def gen_maze_snake(grid_n, rng):
    """Generate snake/spiral maze — pure corridor, no forks.
    The path winds through the grid row by row (boustrophedon).
    This is the full_propagate analog.
    """
    H = W = 2 * grid_n + 1
    grid = ['#'] * (H * W)

    def _set(r, c, v): grid[r * W + c] = v

    # Create boustrophedon path through passage cells
    passage_order = []
    for row_idx in range(grid_n):
        r = 1 + 2 * row_idx
        cols = list(range(grid_n))
        if row_idx % 2 == 1:
            cols = cols[::-1]
        for col_idx in cols:
            c = 1 + 2 * col_idx
            passage_order.append((r, c))

    # Carve the path
    for i, (r, c) in enumerate(passage_order):
        _set(r, c, '.')
        if i > 0:
            pr, pc = passage_order[i - 1]
            # Carve wall between consecutive passages
            _set((r + pr) // 2, (c + pc) // 2, '.')

    si = passage_order[0][0] * W + passage_order[0][1]
    ei = passage_order[-1][0] * W + passage_order[-1][1]
    return grid, si, ei


def find_path_bfs(grid, start, end, H, W):
    """BFS shortest path. Returns list of cell indices on path, or [] if none."""
    queue = deque([(start, [start])])
    visited = {start}
    while queue:
        ci, path = queue.popleft()
        if ci == end:
            return path
        r, c = ci // W, ci % W
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            ni = nr * W + nc
            if 0 <= nr < H and 0 <= nc < W and ni not in visited and grid[ni] != '#':
                visited.add(ni)
                queue.append((ni, path + [ni]))
    return []


def compute_bfs_depth(grid, start, H, W):
    """BFS depth from start for all reachable open cells.
    Returns dict: cell_index → depth."""
    depths = {start: 0}
    queue = deque([start])
    while queue:
        ci = queue.popleft()
        r, c = ci // W, ci % W
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            ni = nr * W + nc
            if 0 <= nr < H and 0 <= nc < W and ni not in depths and grid[ni] != '#':
                depths[ni] = depths[ci] + 1
                queue.append(ni)
    return depths


def _open_neighbors(grid, ci, H, W):
    """Count open (non-wall) neighbors of cell ci."""
    r, c = ci // W, ci % W
    count = 0
    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < H and 0 <= nc < W and grid[nr * W + nc] != '#':
            count += 1
    return count


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Corridor Analysis (carry chain analog)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PATH_ROLES = ['wall', 'off_path', 'start', 'end', 'junction', 'corridor']
PATH_ROLE_TO_ID = {n: i for i, n in enumerate(PATH_ROLES)}

# Dependency context (analog of addition's dependency_context)
DEP_CONTEXTS = [
    'wall', 'off_path', 'start', 'end',
    'junction',           # ≥ 3 open neighbors in maze (= g/k position)
    'corridor_shallow',   # corridor cell, BFS depth < median (= p_above_g)
    'corridor_deep',      # corridor cell, BFS depth >= median (= p_above_p)
    'corridor_entrance',  # corridor cell adjacent to junction (= p_above_k)
]
DEP_CTX_TO_ID = {n: i for i, n in enumerate(DEP_CONTEXTS)}


def classify_path_cells(grid, path_set, start, end, H, W):
    """Classify each cell: wall, off_path, start, end, junction, corridor."""
    roles = {}
    for i in range(H * W):
        if grid[i] == '#':
            roles[i] = 'wall'
        elif i == start:
            roles[i] = 'start'
        elif i == end:
            roles[i] = 'end'
        elif i not in path_set:
            roles[i] = 'off_path'
        else:
            n_open = _open_neighbors(grid, i, H, W)
            roles[i] = 'junction' if n_open >= 3 else 'corridor'
    return roles


def compute_corridor_segments(path, roles):
    """Extract corridor segments from the path.
    A corridor segment is a maximal sequence of consecutive path cells
    classified as 'corridor' (not junction/start/end).
    Returns list of (start_path_idx, length).
    """
    segments = []
    seg_start = None
    seg_len = 0
    for pi, ci in enumerate(path):
        if roles.get(ci) == 'corridor':
            if seg_start is None:
                seg_start = pi
            seg_len += 1
        else:
            if seg_start is not None:
                segments.append((seg_start, seg_len))
            seg_start = None
            seg_len = 0
    if seg_start is not None:
        segments.append((seg_start, seg_len))
    return segments


def compute_corridor_stats(path, roles):
    """Corridor statistics for a maze path (= chain_stats analog)."""
    segs = compute_corridor_segments(path, roles)
    lengths = [s[1] for s in segs]
    n_junctions = sum(1 for ci in path if roles.get(ci) == 'junction')
    n_corridor = sum(1 for ci in path if roles.get(ci) == 'corridor')
    return {
        'path_length': len(path),
        'max_corridor_len': max(lengths, default=0),
        'n_corridor_segments': len(segs),
        'n_corridor_cells': n_corridor,
        'n_junction_cells': n_junctions,
        'corridor_lengths': lengths,
        'is_pure_corridor': n_junctions == 0,
    }


def compute_dependency_context(grid, path, path_set, roles, bfs_depths, start, end, H, W):
    """Per-cell dependency context (= addition's _dependency_context_at_pos).
    Returns dict: cell_index → context_name."""
    # Median BFS depth of corridor cells
    corridor_depths = [bfs_depths.get(ci, 0) for ci in path if roles.get(ci) == 'corridor']
    median_depth = sorted(corridor_depths)[len(corridor_depths) // 2] if corridor_depths else 0

    ctx = {}
    for i in range(H * W):
        role = roles.get(i, 'wall')
        if role in ('wall', 'off_path', 'start', 'end'):
            ctx[i] = role
        elif role == 'junction':
            ctx[i] = 'junction'
        elif role == 'corridor':
            # Check if adjacent to a junction on the path
            r, c = i // W, i % W
            adj_junction = False
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                ni = nr * W + nc
                if 0 <= nr < H and 0 <= nc < W and ni in path_set and roles.get(ni) == 'junction':
                    adj_junction = True; break
            if adj_junction:
                ctx[i] = 'corridor_entrance'
            elif bfs_depths.get(i, 0) >= median_depth:
                ctx[i] = 'corridor_deep'
            else:
                ctx[i] = 'corridor_shallow'
        else:
            ctx[i] = 'off_path'
    return ctx


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Data Formatting
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _maze_to_strings(grid, path_set, start, end):
    """Convert maze to (puzzle_str, solution_str)."""
    H, W = GRID_H, GRID_W
    puzzle_chars = []
    sol_chars = []
    for i in range(H * W):
        if grid[i] == '#':
            puzzle_chars.append('#'); sol_chars.append('#')
        elif i == start:
            puzzle_chars.append('S'); sol_chars.append('S')
        elif i == end:
            puzzle_chars.append('E'); sol_chars.append('E')
        else:
            puzzle_chars.append('.')
            sol_chars.append('1' if i in path_set else '0')
    return ''.join(puzzle_chars), ''.join(sol_chars)


def _make_entry(grid, start, end, compute_meta=False):
    """Create one data entry from a generated maze."""
    H, W = GRID_H, GRID_W
    path = find_path_bfs(grid, start, end, H, W)
    if not path:
        return None
    path_set = set(path)
    puzzle_str, sol_str = _maze_to_strings(grid, path_set, start, end)
    entry = {'string': f"{puzzle_str}={sol_str}", 'path_length': len(path)}

    roles = classify_path_cells(grid, path_set, start, end, H, W)
    cstats = compute_corridor_stats(path, roles)
    entry['corridor_stats'] = cstats

    if compute_meta:
        bfs_depths = compute_bfs_depth(grid, start, H, W)
        dep_ctx = compute_dependency_context(grid, path, path_set, roles, bfs_depths,
                                              start, end, H, W)
        meta = {}
        for i in range(H * W):
            meta[i] = {
                'is_wall': grid[i] == '#',
                'is_open': puzzle_str[i] == '.',
                'is_start': i == start,
                'is_end': i == end,
                'on_path': i in path_set,
                'bfs_depth': bfs_depths.get(i, -1),
                'path_role': roles.get(i, 'wall'),
                'dep_context': dep_ctx.get(i, 'wall'),
                'path_index': path.index(i) if i in path_set else -1,
            }
        entry['meta'] = meta
        # BFS oracle order: path cells sorted by BFS depth
        path_by_depth = sorted(
            [i for i in range(H * W) if puzzle_str[i] == '.'],
            key=lambda i: bfs_depths.get(i, 9999)
        )
        oracle_order = {}
        for rank, ci in enumerate(path_by_depth):
            oracle_order[ci] = rank
        entry['bfs_oracle_order'] = oracle_order
    return entry


def build_tok():
    return CharTokenizer(list('#.01SE='), {'mask': 'M', 'pad': 'P'})


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Data Generation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def gen_train_data(n, seed, straightness_bias=None):
    """Generate training mazes (lightweight, no full metadata)."""
    if straightness_bias is None:
        straightness_bias = STRAIGHTNESS_BIAS
    rng = random.Random(seed)
    data = []
    for _ in range(int(n * 1.2)):
        if len(data) >= n: break
        grid, si, ei = gen_maze_dfs(GRID_N, rng, straightness_bias)
        entry = _make_entry(grid, si, ei, compute_meta=False)
        if entry: data.append(entry)
    if len(data) < n:
        print(f"  WARNING: gen_train_data: {len(data)}/{n}")
    return data[:n]


def gen_test_data(n, seed, straightness_bias=None):
    """Generate test mazes with full metadata."""
    if straightness_bias is None:
        straightness_bias = STRAIGHTNESS_BIAS
    rng = random.Random(seed)
    data = []
    for _ in range(int(n * 1.5)):
        if len(data) >= n: break
        grid, si, ei = gen_maze_dfs(GRID_N, rng, straightness_bias)
        entry = _make_entry(grid, si, ei, compute_meta=True)
        if entry: data.append(entry)
    if len(data) < n:
        print(f"  WARNING: gen_test_data: {len(data)}/{n}")
    return data[:n]


def gen_corner_case_test(n, seed, category='pure_corridor'):
    """Generate corner case test mazes.
    Categories:
      pure_corridor:   snake maze, no forks (= full_propagate)
      long_corridor:   max corridor >= GRID_N
      deep_path:       path length >= 2 * GRID_N^2
    """
    rng = random.Random(seed)
    data = []

    if category == 'pure_corridor':
        for _ in range(int(n * 1.2)):
            if len(data) >= n: break
            grid, si, ei = gen_maze_snake(GRID_N, rng)
            entry = _make_entry(grid, si, ei, compute_meta=True)
            if entry: data.append(entry)
    elif category == 'long_corridor':
        # High straightness bias → longer corridors
        for _ in range(n * 20):
            if len(data) >= n: break
            grid, si, ei = gen_maze_dfs(GRID_N, rng, straightness_bias=0.85)
            entry = _make_entry(grid, si, ei, compute_meta=True)
            if entry and entry['corridor_stats']['max_corridor_len'] >= GRID_N:
                data.append(entry)
    elif category == 'deep_path':
        threshold = 2 * GRID_N * GRID_N
        for _ in range(n * 20):
            if len(data) >= n: break
            grid, si, ei = gen_maze_dfs(GRID_N, rng, straightness_bias=0.5)
            entry = _make_entry(grid, si, ei, compute_meta=True)
            if entry and entry['path_length'] >= threshold:
                data.append(entry)

    if len(data) < n:
        print(f"  WARNING: corner/{category}: {len(data)}/{n}")
    return data[:n]


def gen_min_corridor_test(n, seed, min_corridor):
    """Generate test set with max corridor length >= min_corridor.
    Analogous to gen_min_chain_test in addition."""
    rng = random.Random(seed)
    data = []
    bias = min(0.95, 0.3 + min_corridor * 0.03)  # higher bias for longer corridors
    for _ in range(n * 50):
        if len(data) >= n: break
        grid, si, ei = gen_maze_dfs(GRID_N, rng, straightness_bias=bias)
        entry = _make_entry(grid, si, ei, compute_meta=True)
        if entry and entry['corridor_stats']['max_corridor_len'] >= min_corridor:
            data.append(entry)
    if len(data) < n:
        print(f"  WARNING: corridor>={min_corridor}: {len(data)}/{n}")
    return data[:n]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Probe (fully-masked, per-cell)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@torch.no_grad()
def probe_per_cell(model, tokenizer, test_data, max_len, device=None):
    """Fully-masked probe on open cells with per-context tracking."""
    if device is None: device = DEVICE
    model.eval()
    mask_id = tokenizer.special_ids['mask']
    dot_id = tokenizer.encode('.')[0]

    strings = [d['string'] for d in test_data]
    ids_all, ans_all = encode_samples(strings, tokenizer, max_len)
    ids_all, ans_all = ids_all.to(device), ans_all.to(device)
    N = len(test_data)

    # Blank masks: open cells only (puzzle[i] == '.')
    _arange = torch.arange(ANS_LEN, device=device)
    blank_masks = torch.zeros(N, ANS_LEN, dtype=torch.bool, device=device)
    for si, d in enumerate(test_data):
        ps = d['string'].split('=')[0]
        for j in range(min(ANS_LEN, len(ps))):
            if ps[j] == '.':
                blank_masks[si, j] = True

    # Dependency context IDs (if metadata available)
    has_meta = 'meta' in test_data[0]
    ctx_ids = torch.zeros(N, ANS_LEN, dtype=torch.long, device=device)
    if has_meta:
        for si, d in enumerate(test_data):
            for j in range(ANS_LEN):
                ctx_name = d['meta'][j]['dep_context']
                ctx_ids[si, j] = DEP_CTX_TO_ID.get(ctx_name, 0)

    n_ctx = len(DEP_CONTEXTS)
    ctx_conf = torch.zeros(n_ctx, device=device)
    ctx_correct = torch.zeros(n_ctx, device=device)
    ctx_count = torch.zeros(n_ctx, dtype=torch.long, device=device)
    total_loss = torch.tensor(0.0, device=device)
    total_n = torch.tensor(0, dtype=torch.long, device=device)

    for st in range(0, N, 64):
        en = min(st + 64, N)
        ids, ans = ids_all[st:en], ans_all[st:en]
        B, T = ids.shape
        ans_pos = (ans.unsqueeze(1) + _arange).clamp(max=T - 1)
        bi = torch.arange(B, device=device).unsqueeze(1).expand_as(ans_pos)
        bl = blank_masks[st:en]

        # Mask open cells in solution
        xm = ids.clone()
        xm[bi[bl], ans_pos[bl]] = mask_id
        logits = model(xm)
        al = logits[bi, ans_pos]
        tgt = ids[bi, ans_pos]
        lp = F.log_softmax(al, dim=-1)
        losses = -lp.gather(2, tgt.unsqueeze(2)).squeeze(2)
        cl = al.clone(); cl[:, :, mask_id] = -float('inf')
        probs = F.softmax(cl, dim=-1)
        confs = probs.max(dim=-1).values
        corrects = (probs.argmax(dim=-1) == tgt).float()
        w = bl.float()

        total_loss += (losses * w).sum()
        total_n += w.sum().long()

        if has_meta:
            cat_b = ctx_ids[st:en]
            w_f = w.reshape(-1); cf_f = confs.reshape(-1)
            co_f = corrects.reshape(-1); ca_f = cat_b.reshape(-1)
            valid = w_f > 0
            if valid.any():
                vc = ca_f[valid]
                ctx_conf.scatter_add_(0, vc, cf_f[valid])
                ctx_correct.scatter_add_(0, vc, co_f[valid])
                ctx_count.scatter_add_(0, vc, torch.ones_like(vc, dtype=torch.long))

    overall_loss = (total_loss / total_n.clamp(1)).item()
    overall_acc = ctx_correct.sum().item() / ctx_count.sum().clamp(1).item() if has_meta else 0
    dep_context = {}
    if has_meta:
        for ci, cn in enumerate(DEP_CONTEXTS):
            nc = ctx_count[ci].item()
            if nc > 0:
                dep_context[cn] = {
                    'mean_conf': ctx_conf[ci].item() / nc,
                    'mean_acc': ctx_correct[ci].item() / nc,
                    'n': nc,
                }
    return {'overall_loss': overall_loss, 'overall_acc': overall_acc,
            'dep_context': dep_context}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Generation (multi-policy decode)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@torch.no_grad()
def generate_blanks(model, tokenizer, test_data, decode_policy='confidence',
                    batch_size=32, device=None):
    """Iterative unmasking of open cells with configurable policy."""
    if device is None: device = DEVICE
    mask_id = tokenizer.special_ids['mask']
    pad_id = tokenizer.special_ids['pad']
    model.eval()
    results = []

    for st in range(0, len(test_data), batch_size):
        batch = test_data[st:st + batch_size]; B = len(batch)
        full_enc = [tokenizer.encode(d['string']) for d in batch]
        ml = max(len(e) for e in full_enc)
        ids = torch.full((B, ml), pad_id, dtype=torch.long, device=device)
        for i, e in enumerate(full_enc):
            ids[i, :len(e)] = torch.tensor(e, device=device)

        # Find answer start positions (after '=')
        eq_id = tokenizer.encode('=')[0]
        ans_starts = torch.zeros(B, dtype=torch.long, device=device)
        for i in range(B):
            for t in range(ml):
                if ids[i, t].item() == eq_id:
                    ans_starts[i] = t + 1; break
        _ar = torch.arange(ANS_LEN, device=device)
        ap = (ans_starts.unsqueeze(1) + _ar).clamp(max=ml - 1)
        bi = torch.arange(B, device=device).unsqueeze(1).expand_as(ap)

        # Mask open cells (puzzle[j] == '.')
        x = ids.clone()
        blank_m = torch.zeros(B, ANS_LEN, dtype=torch.bool, device=device)
        for i in range(B):
            ps = batch[i]['string'].split('=')[0]
            for j in range(ANS_LEN):
                if j < len(ps) and ps[j] == '.':
                    x[i, ans_starts[i] + j] = mask_id
                    blank_m[i, j] = True
        n_blanks = blank_m.sum(dim=1)
        max_steps = n_blanks.max().item()

        # Static decode order (for oracle/random)
        static_order = None
        if decode_policy in ('bfs_oracle', 'random'):
            static_order = torch.full((B, ANS_LEN), 9999, dtype=torch.long, device=device)
            for i in range(B):
                blank_js = [j for j in range(ANS_LEN) if blank_m[i, j]]
                if decode_policy == 'random':
                    random.shuffle(blank_js)
                    for rank, j in enumerate(blank_js):
                        static_order[i, j] = rank
                elif decode_policy == 'bfs_oracle' and 'bfs_oracle_order' in batch[i]:
                    oracle = batch[i]['bfs_oracle_order']
                    for j in blank_js:
                        static_order[i, j] = oracle.get(j, 9999)

        # Iterative decode
        for step in range(max_steps):
            is_m = (blank_m & (x[bi, ap] == mask_id))
            if not is_m.any(): break
            logits = model(x)
            al = logits[bi, ap].clone()
            al[:, :, mask_id] = -float('inf')
            confs = F.softmax(al, dim=-1).max(dim=-1).values
            confs[~is_m] = -float('inf')

            if decode_policy == 'confidence':
                # Reveal most confident
                best_j = confs.argmax(dim=1)
            else:
                # Static order: reveal the one with lowest rank among still-masked
                rank_vals = torch.where(is_m, static_order, torch.tensor(9999, device=device))
                best_j = rank_vals.argmin(dim=1)

            preds = F.softmax(al, dim=-1).argmax(dim=-1)
            for i in range(B):
                j = best_j[i].item()
                if is_m[i, j]:
                    x[i, ans_starts[i] + j] = preds[i, j]

        # Collect results
        for i in range(B):
            pred_ids = x[i, ans_starts[i]:ans_starts[i] + ANS_LEN]
            pred_str = tokenizer.decode(pred_ids.cpu().tolist())
            gold_str = batch[i]['string'].split('=')[1]
            ps = batch[i]['string'].split('=')[0]
            pc = [pred_str[j] == gold_str[j] if j < len(pred_str) and j < len(gold_str) else False
                  for j in range(ANS_LEN)]
            errs = [j for j in range(ANS_LEN) if j < len(pred_str) and j < len(gold_str)
                    and pred_str[j] != gold_str[j] and ps[j] == '.']
            results.append({
                'correct': pred_str == gold_str,
                'pos_correct': pc,
                'error_positions': errs,
                'corridor_stats': batch[i].get('corridor_stats', {}),
            })
    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Analysis Functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def stratify_by_corridor(per_sample):
    """Stratify accuracy by corridor properties (= stratify_results in addition)."""
    def _mcl(mcl):
        if mcl <= 2: return 'cl=0-2'
        if mcl <= 5: return 'cl=3-5'
        if mcl <= 10: return 'cl=6-10'
        if mcl <= 20: return 'cl=11-20'
        return 'cl=21+'

    strata = {
        'max_corridor': lambda st: _mcl(st.get('max_corridor_len', 0)),
        'pure_corridor': lambda st: 'pure' if st.get('is_pure_corridor') else 'mixed',
    }
    out = {}
    for name, fn in strata.items():
        bk = defaultdict(list)
        for r in per_sample:
            bk[fn(r['corridor_stats'])].append(r['correct'])
        out[name] = {k: {'acc': sum(v) / len(v), 'n': len(v)} for k, v in sorted(bk.items())}
    return out


def analyse_corridor_rarity(per_sample, test_data):
    """BFS depth bin × accuracy (= carry rarity × accuracy in addition).
    Groups open cells by their BFS depth and measures per-group accuracy."""
    if not test_data or 'meta' not in test_data[0]:
        return {'binned': {}, 'corr': None}

    depth_bins = {
        'shallow(0-5)': lambda d: 0 <= d <= 5,
        'mid(6-15)': lambda d: 6 <= d <= 15,
        'deep(16-30)': lambda d: 16 <= d <= 30,
        'very_deep(31+)': lambda d: d >= 31,
    }
    bin_correct = defaultdict(list)
    bin_on_path = defaultdict(list)

    for si, (r, d) in enumerate(zip(per_sample, test_data)):
        ps = d['string'].split('=')[0]
        for j in range(ANS_LEN):
            if j >= len(ps) or ps[j] != '.': continue
            depth = d['meta'][j]['bfs_depth']
            on_path = d['meta'][j]['on_path']
            correct = r['pos_correct'][j] if j < len(r['pos_correct']) else False
            for bn, bfn in depth_bins.items():
                if bfn(depth):
                    bin_correct[bn].append(correct)
                    bin_on_path[bn].append(on_path)
                    break

    binned = {}
    for bn in depth_bins:
        if bn in bin_correct and bin_correct[bn]:
            cs = bin_correct[bn]
            ps = bin_on_path[bn]
            acc_path = sum(c for c, p in zip(cs, ps) if p) / max(sum(ps), 1)
            acc_off = sum(c for c, p in zip(cs, ps) if not p) / max(sum(1 for p in ps if not p), 1)
            binned[bn] = {
                'n_cells': len(cs),
                'accuracy': sum(cs) / len(cs),
                'acc_on_path': acc_path,
                'acc_off_path': acc_off,
                'path_rate': sum(ps) / len(ps),
                'acc_gap': acc_off - acc_path if sum(ps) > 0 else None,
            }
    # Correlation: depth vs accuracy gap
    valid = [(v['path_rate'], v.get('acc_gap', 0)) for v in binned.values()
             if v.get('acc_gap') is not None]
    corr = None
    if len(valid) >= 3:
        rs, gs = [v[0] for v in valid], [v[1] for v in valid]
        mr, mg = sum(rs) / len(rs), sum(gs) / len(gs)
        c = sum((r - mr) * (g - mg) for r, g in zip(rs, gs))
        sr = sum((r - mr) ** 2 for r in rs) ** 0.5
        sg = sum((g - mg) ** 2 for g in gs) ** 0.5
        corr = c / (sr * sg) if sr > 0 and sg > 0 else 0.0
    return {'binned': binned, 'corr': corr}


@torch.no_grad()
def simulate_puma_coverage(model, tokenizer, test_data, max_len,
                           K=None, tau=None, n_samples=200, device=None):
    """Measure PUMA chain coverage per depth/context category
    (= simulate_puma_coverage in addition)."""
    if device is None: device = DEVICE
    if K is None: K = PUMA_K
    if tau is None: tau = PUMA_TAU
    model.eval()
    mask_id = tokenizer.special_ids['mask']

    N = min(len(test_data), n_samples)
    n_ctx = len(DEP_CONTEXTS)
    cov_sum = torch.zeros(n_ctx)
    cov_count = torch.zeros(n_ctx, dtype=torch.long)

    for si in range(N):
        d = test_data[si]
        ps = d['string'].split('=')[0]
        sol = d['string'].split('=')[1]
        meta = d.get('meta', {})

        prefix_enc = tokenizer.encode(ps + '=')
        sol_enc = tokenizer.encode(sol[:ANS_LEN])
        T_pre = len(prefix_enc)
        x = torch.tensor(prefix_enc + [mask_id] * ANS_LEN, dtype=torch.long, device=device).unsqueeze(0)
        x0 = torch.tensor(sol_enc, dtype=torch.long, device=device)

        blank_js = [j for j in range(ANS_LEN) if j < len(ps) and ps[j] == '.']
        is_m = torch.zeros(ANS_LEN, dtype=torch.bool, device=device)
        for j in blank_js:
            is_m[j] = True
            x[0, T_pre + j] = mask_id

        # For non-open cells, fill in ground truth
        for j in range(ANS_LEN):
            if not is_m[j] and j < len(x0):
                x[0, T_pre + j] = x0[j]

        steps_m = torch.zeros(ANS_LEN)
        total = 0
        for step in range(K):
            if not is_m.any(): break
            total += 1
            steps_m += is_m.cpu().float()
            logits = model(x)
            nm = is_m.sum().item()
            nr = max(1, int(math.ceil(nm / max(K - step, 1))))
            confs = torch.full((ANS_LEN,), -float('inf'), device=device)
            for j in range(ANS_LEN):
                if is_m[j]:
                    cl = logits[0, T_pre + j].clone()
                    cl[mask_id] = -float('inf')
                    confs[j] = F.softmax(cl, dim=-1).max()
            ranked = confs.argsort(descending=True)
            reveal = torch.zeros(ANS_LEN, dtype=torch.bool, device=device)
            for ri in range(ANS_LEN):
                j = ranked[ri].item()
                if not is_m[j]: continue
                if reveal.sum() < nr or confs[j] > tau:
                    reveal[j] = True
            for j in range(ANS_LEN):
                if reveal[j] and j < len(x0):
                    x[0, T_pre + j] = x0[j]
                    is_m[j] = False

        if total == 0: continue
        frac = steps_m / total
        for j in blank_js:
            ctx_name = meta.get(j, {}).get('dep_context', 'off_path')
            ctx_id = DEP_CTX_TO_ID.get(ctx_name, 1)
            cov_sum[ctx_id] += frac[j].item()
            cov_count[ctx_id] += 1

    per_ctx = {}
    for ci, cn in enumerate(DEP_CONTEXTS):
        nc = cov_count[ci].item()
        if nc > 0:
            per_ctx[cn] = {'mean_coverage': cov_sum[ci].item() / nc, 'n': nc}
    return per_ctx


def analyse_error_localization(per_sample, test_data):
    """Where in the corridor structure do errors occur?
    (= analyse_error_localization in addition)."""
    cats = defaultdict(int)
    total_errs = 0
    for r, d in zip(per_sample, test_data):
        if r['correct']: continue
        meta = d.get('meta', {})
        for j in r['error_positions']:
            total_errs += 1
            role = meta.get(j, {}).get('path_role', 'off_path')
            ctx = meta.get(j, {}).get('dep_context', 'off_path')
            cats[ctx] += 1
    if total_errs == 0:
        return {'total_errors': 0}
    result = {'total_errors': total_errs}
    for k, v in cats.items():
        result[k] = v / total_errs
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Training
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def train(mask_type, tokenizer, train_data, test_data, max_len, device=None):
    """Iteration-based training with EMA. Returns (model, ema_state, dynamics)."""
    if device is None: device = DEVICE
    strings = [d['string'] for d in train_data]
    train_ids, train_ans = encode_samples(strings, tokenizer, max_len)
    train_ids, train_ans = train_ids.to(device), train_ans.to(device)
    N, T = train_ids.shape
    mask_id = tokenizer.special_ids['mask']
    pad_id = tokenizer.special_ids['pad']

    # Blank masks [N, ANS_LEN] — open cells only
    blank_masks = torch.zeros(N, ANS_LEN, dtype=torch.bool, device=device)
    for si, d in enumerate(train_data):
        ps = d['string'].split('=')[0]
        for j in range(min(ANS_LEN, len(ps))):
            if ps[j] == '.':
                blank_masks[si, j] = True

    _arange = torch.arange(ANS_LEN, device=device)
    model = Transformer(vocab_size=len(tokenizer), block_size=max_len + 8,
                        n_layer=N_LAYER, n_head=N_HEAD, n_embd=N_EMBD,
                        dropout=DROPOUT, is_causal=False, pos_enc=POS_ENC).to(device)
    ema_state = {k: v.clone() for k, v in model.state_dict().items()}
    print(f"  [{mask_type}] params={model.n_params:,}, N={N}, max_len={max_len}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                   betas=(0.9, 0.99), weight_decay=WEIGHT_DECAY)

    def get_lr(it):
        if it < WARMUP_ITERS: return LR * it / max(WARMUP_ITERS, 1)
        ratio = (it - WARMUP_ITERS) / max(MAX_ITERS - WARMUP_ITERS, 1)
        return MIN_LR + 0.5 * (LR - MIN_LR) * (1 + math.cos(math.pi * min(ratio, 1.0)))

    dynamics = {'checkpoints': [], 'train_loss': []}
    best_loss, best_ema = float('inf'), None
    t0 = time.time(); tg = 0

    # ── PUMA streaming buffer ──
    uses_streaming = (mask_type == 'puma')
    if uses_streaming:
        buf_z = torch.zeros(BATCH_SIZE, T, dtype=torch.long, device=device)
        buf_x0 = torch.zeros(BATCH_SIZE, T, dtype=torch.long, device=device)
        buf_ans = torch.zeros(BATCH_SIZE, dtype=torch.long, device=device)
        buf_stage = torch.zeros(BATCH_SIZE, dtype=torch.long, device=device)
        buf_pool = torch.randperm(N); buf_ptr = 0

        def _refresh(indices):
            nonlocal buf_ptr, buf_pool
            idx_t = torch.tensor(indices, device=device); n = len(indices)
            if buf_ptr + n > len(buf_pool):
                buf_pool = torch.randperm(N); buf_ptr = 0
            si = buf_pool[buf_ptr:buf_ptr + n].to(device); buf_ptr += n
            buf_x0[idx_t] = train_ids[si]
            buf_z[idx_t] = train_ids[si].clone()
            buf_ans[idx_t] = train_ans[si]
            buf_stage[idx_t] = 0
            # Mask blank positions
            ap = (buf_ans[idx_t].unsqueeze(1) + _arange).clamp(max=T - 1)
            bii = idx_t.unsqueeze(1).expand_as(ap)
            bl = blank_masks[si]
            buf_z[bii[bl], ap[bl]] = mask_id

        def _advance(logits):
            nonlocal buf_stage
            B_buf = BATCH_SIZE
            ap = (buf_ans.unsqueeze(1) + _arange).clamp(max=T - 1)
            bi = torch.arange(B_buf, device=device).unsqueeze(1).expand_as(ap)
            is_m = (buf_z[bi, ap] == mask_id)
            if not is_m.any():
                _refresh(list(range(B_buf))); return
            lp = logits[bi, ap].clone()
            lp[:, :, mask_id] = -float('inf')
            confs = F.softmax(lp, dim=-1).max(dim=-1).values
            confs[~is_m] = -float('inf')
            nm = is_m.sum(dim=1).float()
            K_rem = (PUMA_K - buf_stage).clamp(min=1).float()
            nr = (nm / K_rem).ceil().long().clamp(min=1)
            ranked = confs.argsort(dim=1, descending=True)
            rop = torch.zeros_like(ranked)
            rop.scatter_(1, ranked, _arange.expand(B_buf, -1))
            reveal = ((rop < nr.unsqueeze(1)) | (confs > PUMA_TAU)) & is_m
            buf_z[bi[reveal], ap[reveal]] = buf_x0[bi[reveal], ap[reveal]]
            buf_stage += 1
            done = (~(buf_z[bi, ap] == mask_id).any(dim=1)) | (buf_stage >= PUMA_K)
            if done.any():
                _refresh(done.nonzero(as_tuple=True)[0].tolist())

        _refresh(list(range(BATCH_SIZE)))

    # ── Training loop ──
    perm = torch.randperm(N, device=device); perm_ptr = 0

    def _next_batch():
        nonlocal perm, perm_ptr
        if perm_ptr + BATCH_SIZE > N:
            perm = torch.randperm(N, device=device); perm_ptr = 0
        idx = perm[perm_ptr:perm_ptr + BATCH_SIZE]; perm_ptr += BATCH_SIZE
        return idx

    def _do_eval(it_num):
        nonlocal best_loss, best_ema
        orig = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(ema_state); model.eval()
        probe = probe_per_cell(model, tokenizer, test_data, max_len, device)
        dynamics['checkpoints'].append({'iter': it_num, 'tg': tg, **probe})
        dc = probe.get('dep_context', {})
        parts = [f"{c}={dc[c]['mean_acc']:.3f}" for c in
                 ['junction', 'corridor_entrance', 'corridor_shallow', 'corridor_deep']
                 if c in dc]
        print(f"    [eval it {it_num}] loss={probe['overall_loss']:.4f} "
              f"acc={probe['overall_acc']:.4f} {' '.join(parts)}")
        if probe['overall_loss'] < best_loss:
            best_loss = probe['overall_loss']
            best_ema = {k: v.clone() for k, v in ema_state.items()}
        model.load_state_dict(orig); model.train()

    model.eval(); _do_eval(0); model.train()

    for it in range(1, MAX_ITERS + 1):
        for pg in optimizer.param_groups: pg['lr'] = get_lr(it)

        if uses_streaming:
            m = (buf_z == mask_id)
            if m.sum() == 0:
                _refresh(list(range(BATCH_SIZE))); m = (buf_z == mask_id)
            logits = model(buf_z)
            loss = F.cross_entropy(logits[m], buf_x0[m])
            tg += m.sum().item()
        else:
            idx = _next_batch()
            ids = train_ids[idx]; ans_starts = train_ans[idx]
            B_b = ids.shape[0]
            ap = (ans_starts.unsqueeze(1) + _arange).clamp(max=T - 1)
            bi = torch.arange(B_b, device=device).unsqueeze(1).expand_as(ap)
            bl = blank_masks[idx]
            t_ratio = torch.rand(B_b, device=device)
            m_probs = torch.zeros(B_b, T, dtype=torch.float, device=device)
            m_probs[bi, ap] = t_ratio.unsqueeze(1) * bl.float()
            m = torch.bernoulli(m_probs).bool()
            # Ensure at least one mask per sample
            no_m = ~m.any(dim=1)
            if no_m.any():
                rs = torch.rand_like(bl[no_m].float()); rs[~bl[no_m]] = -1.0
                cj = rs.argmax(dim=1)
                ca = ap[no_m].gather(1, cj.unsqueeze(1)).squeeze(1)
                m[no_m.nonzero(as_tuple=True)[0], ca] = True
            xm = ids.clone(); xm[m] = mask_id
            logits = model(xm)
            if m.sum() == 0: continue
            loss = F.cross_entropy(logits[m], ids[m])
            tg += m.sum().item()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        # EMA update
        with torch.no_grad():
            for k, v in model.state_dict().items():
                ema_state[k].lerp_(v, 1 - EMA_DECAY)

        if uses_streaming:
            _advance(logits.detach())

        if it % LOG_EVERY == 0:
            dynamics['train_loss'].append((it, loss.item()))
            print(f"    it {it:6d}/{MAX_ITERS} | loss {loss.item():.4f} | "
                  f"lr {get_lr(it):.1e} | tg {tg:,} | {time.time() - t0:.0f}s")

        if it % EVAL_EVERY == 0 or (it <= MAX_ITERS * 0.1 and it % max(EVAL_EVERY // 5, 1) == 0):
            _do_eval(it)
            model.train()

    if best_ema:
        model.load_state_dict({k: v.to(device) for k, v in best_ema.items()})
    model.eval()
    print(f"  Done {MAX_ITERS} iters (best probe loss: {best_loss:.4f})")
    return model, ema_state, dynamics


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Figures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def make_figures(all_dyn, all_final):
    figs = {}
    COLORS = {'random': '#3498db', 'puma': '#8e44ad'}

    # Fig 1: Dep context accuracy over training
    ctx_show = ['junction', 'corridor_entrance', 'corridor_shallow', 'corridor_deep']
    ctx_colors = {'junction': '#2ecc71', 'corridor_entrance': '#3498db',
                  'corridor_shallow': '#f39c12', 'corridor_deep': '#e74c3c'}
    dconds = [(mt, f'dyn_{mt}') for mt in MASK_TYPES if f'dyn_{mt}' in all_dyn]
    if dconds:
        fig, axes = plt.subplots(1, len(dconds), figsize=(7 * len(dconds), 5), squeeze=False)
        axes = axes[0]
        for ai, (mt, key) in enumerate(dconds):
            dyn = all_dyn[key]; cps = dyn['checkpoints']; xs = [c['iter'] for c in cps]
            ax = axes[ai]
            for ctx in ctx_show:
                ys = [c.get('dep_context', {}).get(ctx, {}).get('mean_acc', float('nan')) for c in cps]
                if any(not math.isnan(y) for y in ys):
                    ax.plot(xs, ys, '-', color=ctx_colors.get(ctx, 'gray'), label=ctx, lw=1.5)
            ax.set_xlabel('Iteration'); ax.set_ylabel('Accuracy'); ax.set_ylim(0, 1.05)
            ax.set_title(mt); ax.legend(fontsize=7); ax.grid(alpha=0.3)
        fig.suptitle('Dependency Context Accuracy Over Training', y=1.02)
        fig.tight_layout(); figs['dep_ctx'] = fig

    # Fig 2: Standard vs corner cases
    test_types = ['standard'] + [f'corner_{c}' for c in ['pure_corridor', 'long_corridor', 'deep_path']]
    for dp in DECODE_POLICIES:
        fig, ax = plt.subplots(figsize=(12, 5))
        for mi, mt in enumerate(MASK_TYPES):
            accs, lbls = [], []
            for tt in test_types:
                key = f'{mt}_{tt}_{dp}'
                r = all_final.get(key)
                if r:
                    accs.append(r['accuracy']); lbls.append(tt.replace('corner_', ''))
            if accs:
                x = range(len(lbls)); w = 0.35; off = -w / 2 if mi == 0 else w / 2
                ax.bar([i + off for i in x], accs, w, label=mt, color=COLORS.get(mt, 'gray'), alpha=0.8)
        if lbls:
            ax.set_xticks(range(len(lbls))); ax.set_xticklabels(lbls, fontsize=8, rotation=20)
        ax.set_ylabel('Accuracy'); ax.set_title(f'{dp} decode')
        ax.legend(); ax.grid(alpha=0.3, axis='y')
        fig.tight_layout(); figs[f'test_types_{dp}'] = fig

    # Fig 3: Corridor sweep
    for dp in DECODE_POLICIES:
        fig, ax = plt.subplots(figsize=(10, 5))
        for mt in MASK_TYPES:
            xs_plot, ys_plot = [], []
            for min_cl in CORRIDOR_SWEEP:
                key = f'{mt}_corridor_sweep_{min_cl}_{dp}'
                r = all_final.get(key)
                if r:
                    xs_plot.append(min_cl); ys_plot.append(r['accuracy'])
            if xs_plot:
                ax.plot(xs_plot, ys_plot, 'o-', color=COLORS.get(mt, 'gray'), label=mt, lw=2)
        ax.set_xlabel('Min corridor length'); ax.set_ylabel('Accuracy')
        ax.set_title(f'Corridor Sweep — {dp}'); ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout(); figs[f'corridor_sweep_{dp}'] = fig

    # Fig 4: BFS depth × accuracy
    fig, ax = plt.subplots(figsize=(10, 5))
    depth_bins_order = ['shallow(0-5)', 'mid(6-15)', 'deep(16-30)', 'very_deep(31+)']
    for mi, mt in enumerate(MASK_TYPES):
        r = all_final.get(f'{mt}_corridor_rarity')
        if not r: continue
        bl = [b for b in depth_bins_order if b in r['binned']]
        accs = [r['binned'][b]['accuracy'] for b in bl]
        if bl:
            x = range(len(bl)); w = 0.35; off = -w / 2 if mi == 0 else w / 2
            ax.bar([i + off for i in x], accs, w, label=mt, color=COLORS.get(mt, 'gray'), alpha=0.8)
            ax.set_xticks(range(len(bl))); ax.set_xticklabels(bl, fontsize=7)
    ax.set_ylabel('Accuracy'); ax.set_title('BFS Depth × Accuracy')
    ax.legend(); ax.grid(alpha=0.3, axis='y')
    fig.tight_layout(); figs['depth_accuracy'] = fig

    return figs


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Run
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run(tag=''):
    exp_name = f"{EXP_NAME}_{tag}" if tag else EXP_NAME
    print(f"\n{'='*70}\n  {exp_name}\n{'='*70}")
    print(f"  Grid: {GRID_H}×{GRID_W} ({CELL_N} cells), bias={STRAIGHTNESS_BIAS}")
    print(f"  Model: L={N_LAYER} H={N_HEAD} E={N_EMBD}")
    print(f"  Masks: {MASK_TYPES}, Decode: {DECODE_POLICIES}")

    torch.manual_seed(SEED); random.seed(SEED)
    tok = build_tok()
    max_len = 2 * CELL_N + 2  # puzzle + '=' + solution + margin

    # ── Data ──
    print(f"\n  Generating {N_TRAIN} train mazes...")
    t0 = time.time()
    train_data = gen_train_data(N_TRAIN, seed=SEED, straightness_bias=STRAIGHTNESS_BIAS)
    print(f"  Train: {len(train_data)} mazes in {time.time()-t0:.1f}s")
    # Profile corridor distribution
    cl_dist = defaultdict(int)
    for d in train_data:
        mcl = d['corridor_stats']['max_corridor_len']
        if mcl <= 2: cl_dist['0-2'] += 1
        elif mcl <= 5: cl_dist['3-5'] += 1
        elif mcl <= 10: cl_dist['6-10'] += 1
        elif mcl <= 20: cl_dist['11-20'] += 1
        else: cl_dist['21+'] += 1
    print(f"  Train corridor dist: {dict(sorted(cl_dist.items()))}")

    print(f"\n  Generating {N_TEST} test mazes (with metadata)...")
    t0 = time.time()
    test_data = gen_test_data(N_TEST, seed=SEED + 1000)
    print(f"  Test: {len(test_data)} mazes in {time.time()-t0:.1f}s")

    all_dyn = {}; all_final = {}

    # ── Train ──
    for mt in MASK_TYPES:
        print(f"\n{'━'*60}\n  Training: {mt}\n{'━'*60}")
        m, ema, dyn = train(mt, tok, train_data, test_data, max_len, device=DEVICE)
        all_dyn[f'dyn_{mt}'] = dyn

        # ── Eval: standard test ──
        for dp in DECODE_POLICIES:
            print(f"  Eval: {mt} × {dp}")
            ps = generate_blanks(m, tok, test_data, decode_policy=dp, device=DEVICE)
            acc = sum(r['correct'] for r in ps) / len(ps)
            strat = stratify_by_corridor(ps)
            all_final[f'{mt}_standard_{dp}'] = {'accuracy': acc, 'n': len(ps), 'stratified': strat}
            print(f"    standard: {acc:.4f}")
            for sn, sv in strat.items():
                parts = [f"{k}={v['acc']:.3f}({v['n']})" for k, v in sv.items()]
                print(f"      {sn}: {' '.join(parts)}")

        # ── Corner cases ──
        for cat in ['pure_corridor', 'long_corridor', 'deep_path']:
            cc = gen_corner_case_test(min(N_TEST, 500), seed=SEED + 6000, category=cat)
            if not cc:
                print(f"    corner/{cat}: no samples"); continue
            for dp in DECODE_POLICIES:
                ps = generate_blanks(m, tok, cc, decode_policy=dp, device=DEVICE)
                acc = sum(r['correct'] for r in ps) / len(ps)
                all_final[f'{mt}_corner_{cat}_{dp}'] = {'accuracy': acc, 'n': len(cc)}
                print(f"    corner/{cat} {dp}: {acc:.4f} (n={len(cc)})")

        # ── Corridor sweep ──
        print(f"  Corridor sweep...")
        for min_cl in CORRIDOR_SWEEP:
            if min_cl > GRID_N * 2: continue
            cc = gen_min_corridor_test(min(500, N_TEST), seed=SEED + 6500 + min_cl,
                                       min_corridor=min_cl)
            if not cc: continue
            for dp in DECODE_POLICIES:
                ps = generate_blanks(m, tok, cc, decode_policy=dp, device=DEVICE)
                acc = sum(r['correct'] for r in ps) / len(ps)
                all_final[f'{mt}_corridor_sweep_{min_cl}_{dp}'] = {
                    'accuracy': acc, 'n': len(cc), 'min_corridor': min_cl}
                print(f"    corridor>={min_cl:2d} {dp}: {acc:.4f} (n={len(cc)})")

        # ── Corridor rarity ──
        print(f"  Corridor rarity analysis...")
        ps_conf = generate_blanks(m, tok, test_data, decode_policy='confidence', device=DEVICE)
        rarity = analyse_corridor_rarity(ps_conf, test_data)
        all_final[f'{mt}_corridor_rarity'] = rarity
        for bn, bd in rarity['binned'].items():
            gap_s = f"gap={bd['acc_gap']:+.4f}" if bd.get('acc_gap') is not None else "gap=N/A"
            print(f"      {bn}: acc={bd['accuracy']:.3f} {gap_s}")

        # ── PUMA coverage (puma only) ──
        if mt == 'puma':
            print(f"  PUMA coverage simulation...")
            cov = simulate_puma_coverage(m, tok, test_data, max_len, device=DEVICE)
            all_final[f'{mt}_coverage'] = cov
            for cn, cv in cov.items():
                print(f"    {cn}: coverage={cv['mean_coverage']:.3f} (n={cv['n']})")

        # ── Error localization ──
        print(f"  Error localization...")
        el = analyse_error_localization(ps_conf, test_data)
        all_final[f'{mt}_error_loc'] = el
        if el['total_errors'] > 0:
            parts = [f"{k}={v:.2f}" for k, v in el.items()
                     if k != 'total_errors' and isinstance(v, float)]
            print(f"    {el['total_errors']} errors: {' '.join(parts)}")

        del m; torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── Figures ──
    figs = make_figures(all_dyn, all_final)

    # ── Save ──
    sd = {'config': {k: globals()[k] for k in [
        'GRID_N', 'GRID_H', 'GRID_W', 'CELL_N', 'N_TRAIN', 'N_TEST', 'MAX_ITERS',
        'BATCH_SIZE', 'N_LAYER', 'N_HEAD', 'N_EMBD', 'MASK_TYPES', 'DECODE_POLICIES',
        'PUMA_K', 'PUMA_TAU', 'STRAIGHTNESS_BIAS']}}
    for k, v in all_dyn.items():
        sd[k] = {'checkpoints': v['checkpoints'], 'train_loss': v['train_loss']}
    for k, v in all_final.items():
        sd[f'final_{k}'] = v
    save_results(exp_name, sd, figures=figs)

    # ── Summary ──
    print(f"\n{'='*70}\n  SUMMARY\n{'='*70}")
    print(f"\n  {'Test':<40s}", end='')
    for mt in MASK_TYPES: print(f" {mt:>14s}", end='')
    print()
    for dp in DECODE_POLICIES:
        for tt in ['standard', 'corner_pure_corridor', 'corner_long_corridor', 'corner_deep_path']:
            key_parts = [f'{mt}_{tt}_{dp}' for mt in MASK_TYPES]
            accs = [all_final.get(k, {}).get('accuracy') for k in key_parts]
            if any(a is not None for a in accs):
                print(f"  {tt+'_'+dp:<40s}", end='')
                for a in accs:
                    print(f" {a:>14.4f}" if a is not None else f" {'N/A':>14s}", end='')
                print()
        # Corridor sweep
        for min_cl in CORRIDOR_SWEEP:
            key_parts = [f'{mt}_corridor_sweep_{min_cl}_{dp}' for mt in MASK_TYPES]
            accs = [all_final.get(k, {}).get('accuracy') for k in key_parts]
            if any(a is not None for a in accs):
                print(f"  {'corridor>='+str(min_cl)+'_'+dp:<40s}", end='')
                for a in accs:
                    print(f" {a:>14.4f}" if a is not None else f" {'N/A':>14s}", end='')
                print()

    # Error localization summary
    print(f"\n  ── Error Localization ──")
    for mt in MASK_TYPES:
        el = all_final.get(f'{mt}_error_loc', {})
        if el.get('total_errors', 0) > 0:
            parts = [f"{k}={v:.2f}" for k, v in el.items()
                     if k != 'total_errors' and isinstance(v, float)]
            print(f"  {mt}: n={el['total_errors']} {' '.join(parts)}")

    return all_dyn, all_final


if __name__ == '__main__':
    args = parse_args()
    seeds = args.seeds if args.seeds else [SEED]
    for si, seed in enumerate(seeds):
        globals()['SEED'] = seed
        t = f"{args.tag}_s{seed}" if args.tag and len(seeds) > 1 else args.tag
        if len(seeds) > 1:
            print(f"\n{'#'*70}\n# Seed {seed} ({si+1}/{len(seeds)})\n{'#'*70}")
        run(tag=t)
