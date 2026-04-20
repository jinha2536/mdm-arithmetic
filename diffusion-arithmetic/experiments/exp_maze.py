"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Maze — Corridor Dependency Learning + PUMA Coverage Deficit
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
from core.train_utils import (
    mount_drive, save_results, save_checkpoint, encode_samples,
    train_diffusion, puma_k_step, generate_diffusion,
    simulate_reveal_trajectory, compute_reveal_vs_order_tau, DEVICE,
)

EXP_NAME = 'exp_maze'

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Config
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GRID_N = 7                  # logical size → actual (2*7+1)=15 → 15×15=225 cells
GRID_H = 2 * GRID_N + 1
GRID_W = GRID_H
CELL_N = GRID_H * GRID_W   # = 225
ANS_LEN = CELL_N            # solution is same size as puzzle

N_TRAIN = 50000; N_TEST = 5000; BATCH_SIZE = 128
# Per-bucket count for constructed tests (sweeps, corners) — shared across analyses
N_PER_BUCKET = 300
MAX_ITERS = 200000; EVAL_EVERY = 5000; LOG_EVERY = 1000
GEN_EVAL_EVERY = 10000; GEN_EVAL_N = 200

MASK_TYPES = ['random', 'puma']
DECODE_POLICIES = ['confidence', 'bfs_oracle', 'dead_end_filling', 'random']

N_LAYER = 4; N_HEAD = 4; N_EMBD = 128; DROPOUT = 0.1; POS_ENC = 'absolute'
LR = 3e-4; MIN_LR = 1e-5; WARMUP_ITERS = 2000; GRAD_CLIP = 1.0
WEIGHT_DECAY = 0.01; EMA_DECAY = 0.9999

PUMA_TAU = 0.9
# PUMA K schedule. K range chosen so reveal-per-step aligns with confidence
# strategy: coarse start when model is random (~10 cells/step) → finer at the
# end when confidence is informative (~5 cells/step for maze, which has longer
# ans_len than addition so we don't go as fine as 2 per step).
# K_END=None → auto: target ~5 cells/step at final K.
# K_EVERY=None → auto: ramp over first 1/3 of training.
PUMA_K_START = 12; PUMA_K_END = None; PUMA_K_STEP = 3; PUMA_K_EVERY = None
SEED = 42
NO_AMP = False
STRAIGHTNESS_BIAS = 0.0  # 0.0=normal DFS, higher=longer corridors in training

# Continuation training
CONTINUATION_ITERS = 5000

# Early stopping (None = disabled)
PATIENCE = None

# Corridor sweep lengths for evaluation
CORRIDOR_SWEEP = [2, 4, 6, 8, 10, 12, 15, 20, 25, 30]

# Dead-end branch size sweep for evaluation
DEAD_END_SWEEP = [2, 4, 6, 8, 10, 15, 20]

# Backbone length sweep — NEW, primary extreme axis (analog of addition chain sweep)
# Adjusted by grid size. GRID_N=7 (15x15): backbone typically 10-25
# GRID_N=10 (21x21): backbone typically 20-40
BACKBONE_SWEEP = [4, 8, 12, 16, 20, 25, 30]

# Reveal trajectory: K stages for per-domain diagnostic
REVEAL_K_DEFAULT = 30  # matches typical maze K_END; will be overridden to actual K_end
# Reveal-vs-reasoning tau (PUMA-only, backbone-extreme training subset)
REVEAL_TAU_MIN_BACKBONE = 16     # samples with backbone ≥ this are tracked
REVEAL_TAU_N_TRACKED = 100
REVEAL_TAU_EVERY = 20000
REVEAL_TAU_K_THRESHOLD_FRAC = 0.7


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
    p.add_argument('--puma-tau', type=float)
    p.add_argument('--puma-k-start', type=int); p.add_argument('--puma-k-end', type=int)
    p.add_argument('--puma-k-step', type=int); p.add_argument('--puma-k-every', type=int)
    p.add_argument('--masks', nargs='+'); p.add_argument('--decode', nargs='+')
    p.add_argument('--straightness-bias', type=float)
    p.add_argument('--continuation-iters', type=int)
    p.add_argument('--patience', type=int)
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
                   'lr': 'LR', 'puma_tau': 'PUMA_TAU',
                   'puma_k_start': 'PUMA_K_START', 'puma_k_end': 'PUMA_K_END',
                   'puma_k_step': 'PUMA_K_STEP', 'puma_k_every': 'PUMA_K_EVERY',
                   'seed': 'SEED', 'no_amp': 'NO_AMP', 'straightness_bias': 'STRAIGHTNESS_BIAS',
                   'continuation_iters': 'CONTINUATION_ITERS',
                   'patience': 'PATIENCE'}.items():
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


def compute_corridor_stats(path, roles, grid=None, path_set=None, H=None, W=None):
    """Corridor + dead-end + junction statistics for a maze (= chain_stats analog)."""
    segs = compute_corridor_segments(path, roles)
    lengths = [s[1] for s in segs]
    n_junctions = sum(1 for ci in path if roles.get(ci) == 'junction')
    n_corridor = sum(1 for ci in path if roles.get(ci) == 'corridor')

    stats = {
        'path_length': len(path),
        'max_corridor_len': max(lengths, default=0),
        'n_corridor_segments': len(segs),
        'n_corridor_cells': n_corridor,
        'n_junction_cells': n_junctions,
        'corridor_lengths': lengths,
        'is_pure_corridor': n_junctions == 0,
    }

    # Dead-end and total junction stats (require grid)
    if grid is not None and path_set is not None and H is not None:
        de = compute_dead_end_stats(grid, path_set, H, W)
        stats['max_dead_end_len'] = de['max_dead_end_len']
        stats['n_dead_end_branches'] = de['n_dead_end_branches']
        stats['dead_end_lengths'] = de['dead_end_lengths']
        stats['total_off_path'] = de['total_off_path']
        stats['n_junctions_total'] = compute_junction_count(grid, H, W)
    else:
        stats['max_dead_end_len'] = 0
        stats['n_dead_end_branches'] = 0
        stats['dead_end_lengths'] = []
        stats['total_off_path'] = 0
        stats['n_junctions_total'] = n_junctions

    return stats


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
# Dead-End Branch Analysis (secondary difficulty axis)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compute_dead_end_stats(grid, path_set, H, W):
    """Compute dead-end branch statistics.
    A dead-end branch is a connected component of open cells NOT on the
    solution path. In a perfect maze (tree), each component is a subtree
    hanging off a junction on the solution path.

    Returns dict with:
      max_dead_end_len:   size of largest off-path component (cells)
      n_dead_end_branches: number of off-path components
      dead_end_lengths:   sorted list of component sizes (descending)
      total_off_path:     total off-path open cells
    """
    off_path = set()
    for i in range(H * W):
        if grid[i] != '#' and i not in path_set:
            off_path.add(i)

    if not off_path:
        return {'max_dead_end_len': 0, 'n_dead_end_branches': 0,
                'dead_end_lengths': [], 'total_off_path': 0}

    visited = set()
    branches = []
    for seed in off_path:
        if seed in visited:
            continue
        comp = set()
        queue = deque([seed])
        visited.add(seed)
        while queue:
            ci = queue.popleft()
            comp.add(ci)
            r, c = ci // W, ci % W
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                ni = nr * W + nc
                if 0 <= nr < H and 0 <= nc < W and ni in off_path and ni not in visited:
                    visited.add(ni)
                    queue.append(ni)
        branches.append(len(comp))

    branches.sort(reverse=True)
    return {
        'max_dead_end_len': branches[0] if branches else 0,
        'n_dead_end_branches': len(branches),
        'dead_end_lengths': branches,
        'total_off_path': len(off_path),
    }


def compute_dead_end_filling_order(grid, start, end, H, W):
    """Dead-end filling algorithm → per-open-cell decode order.

    Algorithm:
      1. Build adjacency among open cells
      2. Iteratively remove degree-1 cells (not start/end)
      3. Track removal round per cell
      4. Remaining cells (solution backbone) ordered by BFS from start

    Returns:
        order: dict cell_index → rank (lower = decoded first = easier).
               Dead-end tips get lowest ranks; solution path backbone gets
               highest.
        n_fillable: int — rank count of cells dead-end-fillable.
                    Cells with rank >= n_fillable are BACKBONE (structurally
                    essential — cannot be reduced by dead-end filling).

    Backbone analog to addition's p-chain: the set of positions whose values
    cannot be determined from local structure alone and must be resolved via
    sequential decisions along the path.
    """
    open_cells = set()
    adj = defaultdict(set)
    for i in range(H * W):
        if grid[i] == '#':
            continue
        open_cells.add(i)
        r, c = i // W, i % W
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            ni = nr * W + nc
            if 0 <= nr < H and 0 <= nc < W and grid[ni] != '#':
                adj[i].add(ni)

    remaining = set(open_cells)
    degree = {i: len(adj[i] & remaining) for i in remaining}

    order = {}
    rank = 0

    while True:
        leaves = [i for i in remaining
                  if degree.get(i, 0) == 1 and i != start and i != end]
        if not leaves:
            break
        for leaf in leaves:
            order[leaf] = rank
            rank += 1
            remaining.discard(leaf)
            for nb in adj[leaf]:
                if nb in remaining:
                    degree[nb] -= 1

    n_fillable = rank  # ranks [0, n_fillable) = dead-end-fillable; [n_fillable, ...) = backbone

    # Remaining = solution path backbone → order by BFS from start
    bfs_q = deque([start])
    bfs_visited = {start}
    while bfs_q:
        ci = bfs_q.popleft()
        if ci in remaining and ci not in order:
            order[ci] = rank
            rank += 1
        for nb in adj[ci]:
            if nb in remaining and nb not in bfs_visited:
                bfs_visited.add(nb)
                bfs_q.append(nb)

    return order, n_fillable


def compute_backbone_length(path, dead_end_order, n_fillable):
    """Length of the backbone subset of the shortest path.

    Backbone = path cells that survive iterative dead-end filling.
    Analog to addition's max_chain_len: positions forming a sequential
    dependency structure that local elimination cannot break.

    Args:
        path: list of cell indices on shortest path (start → end)
        dead_end_order: dict from compute_dead_end_filling_order
        n_fillable: threshold from same function

    Returns: int — number of path cells that are backbone.
    """
    return sum(1 for ci in path if dead_end_order.get(ci, -1) >= n_fillable)


def compute_junction_count(grid, H, W):
    """Count total junctions (open cells with ≥3 open neighbors) in the maze."""
    count = 0
    for i in range(H * W):
        if grid[i] == '#':
            continue
        if _open_neighbors(grid, i, H, W) >= 3:
            count += 1
    return count


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
    cstats = compute_corridor_stats(path, roles, grid, path_set, H, W)
    entry['corridor_stats'] = cstats

    if compute_meta:
        bfs_depths = compute_bfs_depth(grid, start, H, W)
        dep_ctx = compute_dependency_context(grid, path, path_set, roles, bfs_depths,
                                              start, end, H, W)
        # Pre-built path index lookup (avoids O(path_len) per call)
        path_idx = {ci: pi for pi, ci in enumerate(path)}
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
                'path_index': path_idx.get(i, -1),
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

        # Dead-end filling oracle order + backbone length
        de_order, n_fillable = compute_dead_end_filling_order(grid, start, end, H, W)
        entry['de_filling_order'] = de_order
        entry['n_fillable'] = n_fillable
        entry['backbone_length'] = compute_backbone_length(path, de_order, n_fillable)
        # Add to corridor_stats for stratification usage
        entry['corridor_stats']['backbone_length'] = entry['backbone_length']
    else:
        # Lightweight path: still compute de_filling_order + backbone_length
        # (needed for training-data stratification and reveal-τ reasoning order).
        # This is cheap (O(open_cells) per maze) so we always do it.
        de_order, n_fillable = compute_dead_end_filling_order(grid, start, end, H, W)
        entry['de_filling_order'] = de_order
        entry['n_fillable'] = n_fillable
        entry['backbone_length'] = compute_backbone_length(path, de_order, n_fillable)
        entry['corridor_stats']['backbone_length'] = entry['backbone_length']
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
        # Snake mazes have near-maximal path length (Hamiltonian)
        # Also try high-straightness DFS for variety
        for _ in range(n * 2):
            if len(data) >= n: break
            if rng.random() < 0.5:
                grid, si, ei = gen_maze_snake(GRID_N, rng)
            else:
                grid, si, ei = gen_maze_dfs(GRID_N, rng, straightness_bias=0.9)
            entry = _make_entry(grid, si, ei, compute_meta=True)
            if entry:
                data.append(entry)
        # Sort by path length descending, keep longest n
        data.sort(key=lambda d: d['path_length'], reverse=True)
        data = data[:n]

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


def gen_min_dead_end_test(n, seed, min_dead_end):
    """Generate test set with max dead-end branch size >= min_dead_end."""
    rng = random.Random(seed)
    data = []
    for _ in range(n * 50):
        if len(data) >= n: break
        grid, si, ei = gen_maze_dfs(GRID_N, rng, straightness_bias=0.0)
        entry = _make_entry(grid, si, ei, compute_meta=True)
        if entry and entry['corridor_stats'].get('max_dead_end_len', 0) >= min_dead_end:
            data.append(entry)
    if len(data) < n:
        print(f"  WARNING: dead_end>={min_dead_end}: {len(data)}/{n}")
    return data[:n]


def gen_min_backbone_test(n, seed, min_backbone):
    """Generate test set with backbone_length >= min_backbone.

    Backbone = path cells that survive iterative dead-end filling. This is
    the primary extreme-case axis for maze: mazes where dead-end heuristics
    reduce the problem the least, leaving the longest sequential decision
    chain on the solution path. Analog of addition's min_chain test.
    """
    rng = random.Random(seed)
    data = []
    for _ in range(n * 100):  # backbone≥X is rarer than corridor≥X
        if len(data) >= n: break
        grid, si, ei = gen_maze_dfs(GRID_N, rng, straightness_bias=0.0)
        entry = _make_entry(grid, si, ei, compute_meta=True)
        if entry and entry.get('backbone_length', 0) >= min_backbone:
            data.append(entry)
    if len(data) < n:
        print(f"  WARNING: backbone>={min_backbone}: {len(data)}/{n}")
    return data[:n]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Unified test suite — all analyses slice from here
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _bucket_from_entries(entries, tokenizer, max_len):
    """Package entry list with encoded ids for batched analysis."""
    strings = [e['string'] for e in entries]
    if not strings:
        return {'entries': [], 'strings': [], 'ids': torch.empty(0, max_len, dtype=torch.long),
                'ans_starts': torch.empty(0, dtype=torch.long), 'n': 0}
    ids, ans = encode_samples(strings, tokenizer, max_len)
    return {
        'entries': entries, 'strings': strings,
        'ids': ids, 'ans_starts': ans, 'n': len(entries),
    }


def build_test_suite(tokenizer, max_len, seed=None):
    """Unified test suite for maze.

    Structure:
        suite['natural']:                  N_TEST mazes, natural DFS distribution
        suite['constructed']['corridor_{L}']: corridor ≥ L bucket
        suite['constructed']['backbone_{L}']: backbone ≥ L bucket (primary extreme axis)
        suite['constructed']['dead_end_{L}']: dead-end branch ≥ L
        suite['constructed']['pure_corridor']: snake maze (zero junction)
    All entries carry 'meta' field with per-cell dependency_context, bfs_oracle_order,
    de_filling_order, backbone_length.
    """
    if seed is None: seed = SEED + 1000
    suite = {}
    nat = gen_test_data(N_TEST, seed)
    suite['natural'] = _bucket_from_entries(nat, tokenizer, max_len)

    suite['constructed'] = {}
    for L in CORRIDOR_SWEEP:
        ent = gen_min_corridor_test(N_PER_BUCKET, seed=seed + 300 + L, min_corridor=L)
        if ent:
            suite['constructed'][f'corridor_{L}'] = _bucket_from_entries(ent, tokenizer, max_len)
    for L in BACKBONE_SWEEP:
        ent = gen_min_backbone_test(N_PER_BUCKET, seed=seed + 400 + L, min_backbone=L)
        if ent:
            suite['constructed'][f'backbone_{L}'] = _bucket_from_entries(ent, tokenizer, max_len)
    for L in DEAD_END_SWEEP:
        ent = gen_min_dead_end_test(N_PER_BUCKET, seed=seed + 500 + L, min_dead_end=L)
        if ent:
            suite['constructed'][f'dead_end_{L}'] = _bucket_from_entries(ent, tokenizer, max_len)
    pc = gen_corner_case_test(N_PER_BUCKET, seed=seed + 600, category='pure_corridor')
    if pc:
        suite['constructed']['pure_corridor'] = _bucket_from_entries(pc, tokenizer, max_len)

    print(f"  Maze test suite built:")
    print(f"    natural: {suite['natural']['n']}")
    for k, v in suite['constructed'].items():
        print(f"    constructed/{k}: {v['n']}")
    return suite


def filter_natural(suite, pred):
    """Filter natural bucket by predicate on entry; returns same-shape bucket."""
    nat = suite['natural']
    idx = [i for i, e in enumerate(nat['entries']) if pred(e)]
    if not idx:
        return {'entries': [], 'strings': [],
                'ids': torch.empty(0, nat['ids'].shape[1], dtype=torch.long),
                'ans_starts': torch.empty(0, dtype=torch.long), 'n': 0}
    return {
        'entries': [nat['entries'][i] for i in idx],
        'strings': [nat['strings'][i] for i in idx],
        'ids': nat['ids'][idx],
        'ans_starts': nat['ans_starts'][idx],
        'n': len(idx),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Training stratum construction
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRATUM_BOUNDS_BB = [(0, 6), (6, 12), (12, 18), (18, 24), (24, 999)]
STRATUM_NAMES = ['bb_0_5', 'bb_6_11', 'bb_12_17', 'bb_18_23', 'bb_24plus']


def _backbone_to_stratum(bb):
    for i, (lo, hi) in enumerate(STRATUM_BOUNDS_BB):
        if lo <= bb < hi:
            return i
    return len(STRATUM_BOUNDS_BB) - 1


def build_training_strata(train_data):
    """Compute stratum id per training entry by backbone_length."""
    strata = [_backbone_to_stratum(e.get('backbone_length', 0)) for e in train_data]
    counts = [strata.count(i) for i in range(len(STRATUM_NAMES))]
    return torch.tensor(strata, dtype=torch.long), STRATUM_NAMES, counts


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Probe (fully-masked, per-cell)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@torch.no_grad()
def probe_per_cell(model, tokenizer, test_data, max_len, device=None):
    """Fully-masked probe on open cells with per-context tracking."""
    if device is None: device = DEVICE
    model.eval()
    mask_id = tokenizer.special_ids['mask']

    strings = [d['string'] for d in test_data]
    ids_all, ans_all = encode_samples(strings, tokenizer, max_len)
    ids_all, ans_all = ids_all.to(device), ans_all.to(device)
    N = len(test_data)

    # Blank masks: vectorized via token comparison
    dot_id = tokenizer.encode('.')[0]
    _arange = torch.arange(ANS_LEN, device=device)
    blank_masks = (ids_all[:, :ANS_LEN] == dot_id).to(device)

    # Dependency context IDs (if metadata available)
    has_meta = 'meta' in test_data[0]
    ctx_ids = torch.zeros(N, ANS_LEN, dtype=torch.long, device=device)
    if has_meta:
        ctx_lists = [[DEP_CTX_TO_ID.get(d['meta'][j]['dep_context'], 0)
                       for j in range(ANS_LEN)] for d in test_data]
        ctx_ids = torch.tensor(ctx_lists, dtype=torch.long, device=device)

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
                    n_decode_steps=None, batch_size=32, device=None):
    """Iterative unmasking with BATCHED multi-reveal per step.
    Reveals ceil(n_remaining / K_remaining) cells per forward pass.
    For ANS_LEN=225 with ~100 blanks, this reduces ~100 forward passes to ~12."""
    if device is None: device = DEVICE
    if n_decode_steps is None: n_decode_steps = globals().get('_PUMA_K_FINAL', PUMA_K_END or 24)
    mask_id = tokenizer.special_ids['mask']
    pad_id = tokenizer.special_ids['pad']
    dot_id = tokenizer.encode('.')[0]
    model.eval(); results = []
    _ar = torch.arange(ANS_LEN, device=device)

    for st in range(0, len(test_data), batch_size):
        batch = test_data[st:st + batch_size]; B = len(batch)
        full_enc = [tokenizer.encode(d['string']) for d in batch]
        ml = max(len(e) for e in full_enc)
        ids = torch.full((B, ml), pad_id, dtype=torch.long, device=device)
        for i, e in enumerate(full_enc):
            ids[i, :len(e)] = torch.tensor(e, device=device)

        # Vectorized '=' search
        eq_id = tokenizer.encode('=')[0]
        ans_starts = (ids == eq_id).long().argmax(dim=1) + 1
        ap = (ans_starts.unsqueeze(1) + _ar).clamp(max=ml - 1)
        bi = torch.arange(B, device=device).unsqueeze(1).expand_as(ap)

        # Vectorized blank mask
        blank_m = (ids[:, :ANS_LEN] == dot_id)
        x = ids.clone()
        x[bi[blank_m], ap[blank_m]] = mask_id

        # Static decode order
        static_order = None
        if decode_policy in ('bfs_oracle', 'dead_end_filling', 'random'):
            static_order = torch.full((B, ANS_LEN), 9999, dtype=torch.long, device=device)
            for i in range(B):
                blank_js = blank_m[i].nonzero(as_tuple=True)[0].tolist()
                if decode_policy == 'random':
                    random.shuffle(blank_js)
                    for rank, j in enumerate(blank_js): static_order[i, j] = rank
                elif decode_policy == 'bfs_oracle' and 'bfs_oracle_order' in batch[i]:
                    oracle = batch[i]['bfs_oracle_order']
                    for j in blank_js: static_order[i, j] = oracle.get(j, 9999)
                elif decode_policy == 'dead_end_filling' and 'de_filling_order' in batch[i]:
                    oracle = batch[i]['de_filling_order']
                    for j in blank_js: static_order[i, j] = oracle.get(j, 9999)

        # Multi-reveal iterative decode
        for step in range(n_decode_steps):
            is_m = blank_m & (x[bi, ap] == mask_id)
            if not is_m.any(): break
            logits = model(x)
            al = logits[bi, ap].clone(); al[:, :, mask_id] = -float('inf')
            probs = F.softmax(al, dim=-1)
            confs = probs.max(dim=-1).values; preds = probs.argmax(dim=-1)
            confs[~is_m] = -float('inf')

            nm = is_m.sum(dim=1).float()
            K_rem = max(n_decode_steps - step, 1)
            nr = (nm / K_rem).ceil().long().clamp(min=1)

            if decode_policy == 'confidence':
                ranked = confs.argsort(dim=1, descending=True)
            else:
                rank_vals = torch.where(is_m, static_order,
                                        torch.tensor(9999, dtype=torch.long, device=device))
                ranked = rank_vals.argsort(dim=1)
            rop = torch.zeros_like(ranked)
            rop.scatter_(1, ranked, _ar.expand(B, -1))
            reveal = (rop < nr.unsqueeze(1)) & is_m
            x[bi[reveal], ap[reveal]] = preds[reveal]

        # Vectorized result collection
        pred_at_ans = x[bi, ap]; gold_at_ans = ids[bi, ap]
        pos_correct = (pred_at_ans == gold_at_ans)
        open_correct = pos_correct | ~blank_m
        sample_correct = open_correct.all(dim=1)

        for i in range(B):
            errs = ((~pos_correct[i]) & blank_m[i]).nonzero(as_tuple=True)[0].tolist()
            results.append({
                'correct': sample_correct[i].item(),
                'pos_correct': pos_correct[i].tolist(),
                'error_positions': errs,
                'corridor_stats': batch[i].get('corridor_stats', {}),
            })
    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Analysis Functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def stratify_by_corridor(per_sample):
    """Stratify accuracy by corridor, dead-end, and junction properties."""
    def _mcl(mcl):
        if mcl <= 2: return 'cl=0-2'
        if mcl <= 5: return 'cl=3-5'
        if mcl <= 10: return 'cl=6-10'
        if mcl <= 20: return 'cl=11-20'
        return 'cl=21+'

    def _mdel(mdel):
        if mdel <= 0: return 'de=0'
        if mdel <= 3: return 'de=1-3'
        if mdel <= 8: return 'de=4-8'
        if mdel <= 15: return 'de=9-15'
        return 'de=16+'

    def _nj(nj):
        if nj <= 2: return 'jn=0-2'
        if nj <= 5: return 'jn=3-5'
        if nj <= 10: return 'jn=6-10'
        return 'jn=11+'

    def _bb(bb):
        if bb <= 4: return 'bb=0-4'
        if bb <= 10: return 'bb=5-10'
        if bb <= 16: return 'bb=11-16'
        if bb <= 22: return 'bb=17-22'
        return 'bb=23+'

    strata = {
        'max_corridor': lambda st: _mcl(st.get('max_corridor_len', 0)),
        'pure_corridor': lambda st: 'pure' if st.get('is_pure_corridor') else 'mixed',
        'max_dead_end': lambda st: _mdel(st.get('max_dead_end_len', 0)),
        'n_junctions': lambda st: _nj(st.get('n_junctions_total', st.get('n_junction_cells', 0))),
        'backbone': lambda st: _bb(st.get('backbone_length', 0)),
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
def analyse_reveal_patterns(model, tokenizer, bucket, max_len,
                             K=REVEAL_K_DEFAULT, tau=PUMA_TAU, device=None):
    """For extreme-case bucket, run PUMA forward and aggregate reveal behavior
    by (a) dependency context, (b) backbone-path rank, (c) never-revealed map.

    Core failure diagnostic: on hard mazes PUMA defers structurally-critical
    backbone cells, while Random covers them uniformly.
    """
    if device is None: device = DEVICE
    if bucket['n'] == 0:
        return {'n': 0}

    # Compute blank_masks (open-cell mask) for PUMA forward
    dot_id = tokenizer.encode('.')[0]
    blank_masks = (bucket['ids'][:, :ANS_LEN] == dot_id)

    traj = simulate_reveal_trajectory(
        model, tokenizer, bucket['ids'], bucket['ans_starts'], ANS_LEN,
        blank_masks=blank_masks, K=K, tau=tau, device=device)

    rs = traj['reveal_stage']              # [N, ANS_LEN]
    smm = traj['still_masked_start']       # [N, K+1, ANS_LEN]
    N = bucket['n']

    # (1) By dependency context (junction / corridor_deep / corridor_entrance / etc.)
    by_role_raw = {r: [] for r in DEP_CONTEXTS}
    for i, ent in enumerate(bucket['entries']):
        meta = ent.get('meta', {})
        for j in range(ANS_LEN):
            role = meta.get(j, {}).get('dep_context', 'wall')
            if role not in ('wall', 'off_path'):   # only open cells on/near path
                by_role_raw.setdefault(role, []).append(smm[i, :K, j].float())
    by_role = {}
    for r, xs in by_role_raw.items():
        if xs:
            by_role[r] = {
                'still_masked_per_stage': torch.stack(xs).mean(dim=0).tolist(),
                'n_positions': len(xs),
            }

    # (2) By backbone-path rank — primary axis (analog of addition chain-position)
    # For each entry, sort backbone cells by their BFS depth from start and track
    # reveal timing by rank-from-start.
    bb_acc = defaultdict(list)
    max_bb_rank = 0
    for i, ent in enumerate(bucket['entries']):
        meta = ent.get('meta', {})
        de_order = ent.get('de_filling_order', {})
        n_fillable = ent.get('n_fillable', 0)
        if not de_order: continue
        # backbone cells on path, sorted by their BFS depth from start
        bb_cells = [j for j in range(ANS_LEN)
                    if meta.get(j, {}).get('on_path')
                    and de_order.get(j, -1) >= n_fillable
                    and meta.get(j, {}).get('bfs_depth', -1) >= 0]
        bb_cells.sort(key=lambda j: meta[j]['bfs_depth'])
        for rank, j in enumerate(bb_cells):
            max_bb_rank = max(max_bb_rank, rank)
            bb_acc[rank].append(smm[i, :K, j].float())
    by_backbone_rank = []
    for rank in range(max_bb_rank + 1):
        xs = bb_acc.get(rank, [])
        if xs:
            by_backbone_rank.append({
                'rank': rank,
                'still_masked_per_stage': torch.stack(xs).mean(dim=0).tolist(),
                'n': len(xs),
            })

    # (3) Never-revealed fraction per cell position
    never = (rs >= K).float().mean(dim=0).tolist()

    # (4) Representative traces
    order_by_bb = sorted(range(N), key=lambda i: bucket['entries'][i].get('backbone_length', 0))
    picks = [order_by_bb[len(order_by_bb) // 6],
             order_by_bb[len(order_by_bb) // 2],
             order_by_bb[-1]] if N >= 3 else list(range(N))
    reps = []
    for i in picks:
        ent = bucket['entries'][i]
        reps.append({
            'backbone_length': ent.get('backbone_length', 0),
            'path_length': ent.get('path_length', 0),
            'corridor_stats': ent.get('corridor_stats', {}),
            'reveal_stage': rs[i].tolist(),
        })

    return {
        'n': N, 'K': K, 'tau': tau,
        'by_role': by_role,
        'by_backbone_rank': by_backbone_rank,
        'never_revealed': never,
        'representative_traces': reps,
    }


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

def train_model(mask_type, tokenizer, train_data, suite, max_len,
                max_iters=None, init_state=None, device=None):
    """Train maze model. train_data is list of entries (must have backbone_length).
    `suite` is the unified test suite (for eval probe on natural + reveal-τ on extreme).
    """
    if device is None: device = DEVICE
    if max_iters is None: max_iters = MAX_ITERS

    strings = [d['string'] for d in train_data]
    train_ids, train_ans = encode_samples(strings, tokenizer, max_len)
    train_ids, train_ans = train_ids.to(device), train_ans.to(device)

    # Blank masks: only '.' cells are maskable
    dot_id = tokenizer.encode('.')[0]
    blank_masks = (train_ids[:, :ANS_LEN] == dot_id)

    # Training strata (backbone length buckets)
    sample_strata, stratum_names, stratum_counts = build_training_strata(train_data)
    print(f"  Training strata counts: " +
          ', '.join(f"{n}={c}" for n, c in zip(stratum_names, stratum_counts)))

    # Natural probe target (from suite)
    natural_entries = suite['natural']['entries']

    # Reveal-τ tracked subset: stratified by backbone_length.
    # Reasoning order = dead-end-filling order (lower rank = unmask first).
    # Per-stratum trajectories will show: bb_0_5 → τ ≈ +1 (dead-end filling
    # order learned), bb_24plus → τ ≈ 0 (extreme case misalignment).
    reveal_tracked_ids = None
    reveal_tracked_ans = None
    reveal_reasoning_order = None
    reveal_blanks = None
    reveal_tracked_strata = None
    if mask_type == 'puma':
        per_stratum_cap = max(REVEAL_TAU_N_TRACKED // len(STRATUM_NAMES), 10)
        tracked_by_stratum = {sn: [] for sn in STRATUM_NAMES}
        for i, e in enumerate(train_data):
            bb = e.get('backbone_length', 0)
            si = _backbone_to_stratum(bb)
            sn = STRATUM_NAMES[si]
            if len(tracked_by_stratum[sn]) < per_stratum_cap:
                tracked_by_stratum[sn].append((i, si))
        tracked_flat = [t for v in tracked_by_stratum.values() for t in v]
        strata_counts = {sn: len(v) for sn, v in tracked_by_stratum.items()}
        if sum(strata_counts.values()) >= 30:
            tracked = [t[0] for t in tracked_flat]
            tracked_strata = [t[1] for t in tracked_flat]
            reveal_tracked_ids = train_ids[tracked]
            reveal_tracked_ans = train_ans[tracked]
            reveal_tracked_strata = torch.tensor(tracked_strata, dtype=torch.long)
            N_tr = len(tracked)
            ro = torch.full((N_tr, ANS_LEN), ANS_LEN, dtype=torch.long)
            bm = torch.zeros(N_tr, ANS_LEN, dtype=torch.bool)
            for i, idx in enumerate(tracked):
                ent = train_data[idx]
                de_order = ent.get('de_filling_order', {})
                if not de_order:
                    continue
                for j in range(ANS_LEN):
                    if j in de_order:
                        ro[i, j] = de_order[j]
                bm[i] = blank_masks[idx]
            reveal_reasoning_order = ro
            reveal_blanks = bm
            print(f"  Reveal-τ tracking (stratified): total={N_tr}, by stratum: " +
                  ', '.join(f"{sn}={c}" for sn, c in strata_counts.items() if c > 0))
        else:
            print(f"  Reveal-τ tracking: SKIPPED "
                  f"(total tracked < 30: {strata_counts})")

    # PUMA K schedule
    k_sched = None
    if mask_type == 'puma':
        avg_blanks = blank_masks.sum(dim=1).float().mean().item()
        if PUMA_K_END is not None:
            k_end = PUMA_K_END
        else:
            # Target ~5 cells revealed per step at final K (see config comment
            # for rationale: reveal-per-step alignment with confidence strategy)
            k_raw = avg_blanks / 5.0
            k_end = max(PUMA_K_START, int(round(k_raw / PUMA_K_STEP) * PUMA_K_STEP))
            k_end = min(k_end, int(avg_blanks) // 3)
        n_increments = max(1, (k_end - PUMA_K_START) // PUMA_K_STEP)
        if PUMA_K_EVERY is not None:
            k_every = PUMA_K_EVERY
        else:
            k_every = max(1000, (max_iters // 3) // n_increments)
        k_sched = puma_k_step(PUMA_K_START, k_end, PUMA_K_STEP, k_every)
        final_k = k_sched(max_iters)
        print(f"  PUMA K: {PUMA_K_START} → {final_k} (+{PUMA_K_STEP} every {k_every//1000}k, "
              f"cap={k_end}, avg blanks={avg_blanks:.0f}, target ~{avg_blanks/final_k:.1f} cells/step)")
        globals()['_PUMA_K_FINAL'] = final_k
    K_final_for_tau = k_sched(max_iters) if k_sched else None

    def eval_fn(model, it, tg):
        probe = probe_per_cell(model, tokenizer, natural_entries, max_len, device)
        dc = probe.get('dep_context', {})
        parts = [f"{c}={dc[c]['mean_acc']:.3f}" for c in
                 ['junction', 'corridor_entrance', 'corridor_shallow', 'corridor_deep']
                 if c in dc]
        print(f"    [eval it {it}] loss={probe['overall_loss']:.4f} "
              f"acc={probe['overall_acc']:.4f} {' '.join(parts)}")

        # Reveal-τ (PUMA only, past K threshold) — stratified by backbone stratum
        if (reveal_tracked_ids is not None and it > 0
                and it % REVEAL_TAU_EVERY == 0 and K_final_for_tau is not None):
            K_cur = k_sched(it)
            if K_cur >= K_final_for_tau * REVEAL_TAU_K_THRESHOLD_FRAC:
                traj = simulate_reveal_trajectory(
                    model, tokenizer, reveal_tracked_ids, reveal_tracked_ans, ANS_LEN,
                    blank_masks=reveal_blanks, K=K_cur, tau=PUMA_TAU,
                    batch_size=32, device=device)
                taus = compute_reveal_vs_order_tau(
                    traj['reveal_stage'], reveal_reasoning_order, reveal_blanks)
                import numpy as _np
                valid_mask = ~_np.isnan(taus)
                stratum_np = reveal_tracked_strata.cpu().numpy()
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
                        'K_cur': K_cur, 'n': int(len(valid)),
                        'mean': float(valid.mean()),
                        'q25': float(_np.percentile(valid, 25)),
                        'q50': float(_np.percentile(valid, 50)),
                        'q75': float(_np.percentile(valid, 75)),
                        'min': float(valid.min()), 'max': float(valid.max()),
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
        mask_type=mask_type, blank_masks=blank_masks,
        puma_tau=PUMA_TAU, puma_k_schedule=k_sched,
        n_layer=N_LAYER, n_head=N_HEAD, n_embd=N_EMBD, dropout=DROPOUT, pos_enc=POS_ENC,
        max_iters=max_iters, batch_size=BATCH_SIZE,
        lr=LR, min_lr=MIN_LR, warmup_iters=WARMUP_ITERS,
        grad_clip=GRAD_CLIP, weight_decay=WEIGHT_DECAY, ema_decay=EMA_DECAY,
        eval_fn=eval_fn, eval_every=EVAL_EVERY, log_every=LOG_EVERY,
        patience=globals().get('PATIENCE'),
        sample_strata=sample_strata, stratum_names=stratum_names,
        init_state=init_state, device=device,
        use_amp=False if NO_AMP else None,
    )
    return model, dynamics


# Backward-compat alias — some scripts may still import `train`
train = train_model


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

    # Fig 3b: Dead-end sweep
    for dp in DECODE_POLICIES:
        has_data = False
        for mt in MASK_TYPES:
            for mdel in DEAD_END_SWEEP:
                if f'{mt}_dead_end_sweep_{mdel}_{dp}' in all_final:
                    has_data = True; break
            if has_data: break
        if not has_data: continue
        fig, ax = plt.subplots(figsize=(10, 5))
        for mt in MASK_TYPES:
            xs_plot, ys_plot = [], []
            for mdel in DEAD_END_SWEEP:
                key = f'{mt}_dead_end_sweep_{mdel}_{dp}'
                r = all_final.get(key)
                if r:
                    xs_plot.append(mdel); ys_plot.append(r['accuracy'])
            if xs_plot:
                ax.plot(xs_plot, ys_plot, 'o-', color=COLORS.get(mt, 'gray'), label=mt, lw=2)
        ax.set_xlabel('Min dead-end branch size'); ax.set_ylabel('Accuracy')
        ax.set_title(f'Dead-End Sweep — {dp}'); ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout(); figs[f'dead_end_sweep_{dp}'] = fig

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

    # Backbone sweep — PRIMARY extreme axis
    for dp in DECODE_POLICIES:
        fig, ax = plt.subplots(figsize=(10, 5))
        for mt, col, mk in [('random', '#3498db', 'o'), ('puma', '#8e44ad', 's')]:
            xs, ys = [], []
            for L in BACKBONE_SWEEP:
                r = all_final.get(f'{mt}_backbone_sweep_{L}_{dp}')
                if r is not None:
                    xs.append(L); ys.append(r['accuracy'])
            if xs:
                ax.plot(xs, ys, f'-{mk}', color=col, label=mt, lw=2, markersize=8)
        ax.set_xlabel('Min backbone length (dead-end filling remainder)')
        ax.set_ylabel('Accuracy'); ax.set_title(f'Backbone Length Sweep — {dp} decode')
        ax.legend(); ax.grid(alpha=0.3); ax.set_ylim(-0.05, 1.05)
        fig.tight_layout(); figs[f'backbone_sweep_{dp}'] = fig

    # Stratum loss trajectory — training-side diagnostic
    mts_with_strat = [mt for mt in MASK_TYPES
                     if all_dyn.get(f'dyn_{mt}', {}).get('stratified_loss')]
    if mts_with_strat:
        nm = len(mts_with_strat)
        fig, axes = plt.subplots(1, nm, figsize=(7 * nm, 5), squeeze=False); axes = axes[0]
        s_cmap = plt.cm.viridis
        for ai, mt in enumerate(mts_with_strat):
            dyn = all_dyn[f'dyn_{mt}']
            sl = dyn['stratified_loss']
            names = dyn.get('stratum_names', [])
            S = len(names) if names else (len(sl[0]['per_stratum_loss']) if sl else 0)
            ax = axes[ai]
            xs = [e['iter'] for e in sl]
            for si in range(S):
                ys = [e['per_stratum_loss'][si] if e['per_stratum_n'][si] > 0 else None
                      for e in sl]
                label = names[si] if si < len(names) else f's{si}'
                valid = [(x, y) for x, y in zip(xs, ys) if y is not None]
                if valid:
                    ax.plot([p[0] for p in valid], [p[1] for p in valid],
                            '-', color=s_cmap(si / max(S - 1, 1)), label=label, lw=1.5)
            ax.set_xlabel('Iteration'); ax.set_ylabel('Masked-token loss (training)')
            ax.set_title(f'{mt}: stratum loss trajectory'); ax.set_yscale('log')
            ax.legend(fontsize=8); ax.grid(alpha=0.3)
        fig.suptitle('Training-time loss by backbone-length stratum', y=1.02)
        fig.tight_layout(); figs['stratum_loss'] = fig

    # Reveal heatmap on extreme backbone bucket
    extreme_key = None
    for L in sorted(BACKBONE_SWEEP, reverse=True):
        if any(f'{mt}_reveal_backbone_{L}' in all_final for mt in MASK_TYPES):
            extreme_key = f'backbone_{L}'; break
    if extreme_key is not None:
        mts_with_rev = [mt for mt in MASK_TYPES
                        if f'{mt}_reveal_{extreme_key}' in all_final]
        nm = len(mts_with_rev)
        if nm > 0:
            fig, axes = plt.subplots(1, nm, figsize=(6 * nm, 5), squeeze=False); axes = axes[0]
            import numpy as _np
            for ai, mt in enumerate(mts_with_rev):
                rev = all_final[f'{mt}_reveal_{extreme_key}']
                br = rev.get('by_backbone_rank', [])
                if not br: continue
                K = rev.get('K', REVEAL_K_DEFAULT)
                mat = [row['still_masked_per_stage'] for row in br]
                arr = _np.array(mat)
                ax = axes[ai]
                im = ax.imshow(arr, aspect='auto', origin='lower',
                               cmap='Reds', vmin=0, vmax=1, interpolation='nearest')
                ax.set_xlabel(f'PUMA stage (K={K})')
                ax.set_ylabel('Backbone rank (0 = nearest to start)')
                ax.set_title(f'{mt}: still-masked fraction ({extreme_key}, N={rev["n"]})')
                plt.colorbar(im, ax=ax, label='still masked')
            fig.suptitle(f'Reveal trajectory — {extreme_key}', y=1.02)
            fig.tight_layout(); figs[f'reveal_{extreme_key}'] = fig

    # Reveal-vs-reasoning τ trajectory (PUMA only, stratified by backbone).
    # Central training-time diagnostic: dead-end-filling order alignment for
    # easy strata (bb_0_5 → τ ≈ +1) vs hard ones (bb_24plus → τ ≈ 0).
    puma_dyn = all_dyn.get('dyn_puma', {})
    tau_pts = [c for c in puma_dyn.get('checkpoints', []) if 'reveal_tau' in c]
    if tau_pts:
        fig, ax = plt.subplots(figsize=(10, 5))
        # Overall IQR shaded
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
        ax.axhline(1, color='green', ls='--', alpha=0.4, label='τ=+1 (solver-aligned)')
        ax.axhline(-1, color='red', ls='--', alpha=0.4, label='τ=−1 (reversed)')
        ax.set_xlabel('Training iteration')
        ax.set_ylabel('Kendall τ (reveal vs dead-end-filling order)')
        ax.set_title('PUMA reveal-order alignment, stratified by backbone length')
        ax.set_ylim(-1.05, 1.05); ax.legend(loc='best', fontsize=8); ax.grid(alpha=0.3)
        fig.tight_layout(); figs['reveal_tau_stratified'] = fig

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
    # Training distribution profile — focus on backbone (primary extreme axis)
    bb_dist = defaultdict(int)
    for d in train_data:
        bb = d.get('backbone_length', 0)
        if bb <= 5: bb_dist['0-5'] += 1
        elif bb <= 11: bb_dist['6-11'] += 1
        elif bb <= 17: bb_dist['12-17'] += 1
        elif bb <= 23: bb_dist['18-23'] += 1
        else: bb_dist['24+'] += 1
    print(f"  Train backbone dist: {dict(sorted(bb_dist.items()))}")

    # Unified test suite — all evals slice from here
    print(f"\n  Building test suite...")
    t0 = time.time()
    suite = build_test_suite(tok, max_len, seed=SEED + 1000)
    print(f"  Suite built in {time.time()-t0:.1f}s")
    natural_entries = suite['natural']['entries']

    all_dyn = {}; all_final = {}
    saved_states = {}

    # ── Train ──
    for mt in MASK_TYPES:
        print(f"\n{'━'*60}\n  Training: {mt}\n{'━'*60}")
        m, dyn = train_model(mt, tok, train_data, suite, max_len, device=DEVICE)
        all_dyn[f'dyn_{mt}'] = dyn
        saved_states[mt] = {k: v.cpu().clone() for k, v in m.state_dict().items()}
        save_checkpoint(exp_name, saved_states[mt], tag=mt)

        # Standard eval on natural
        for dp in DECODE_POLICIES:
            ps = generate_blanks(m, tok, natural_entries, decode_policy=dp, device=DEVICE)
            acc = sum(r['correct'] for r in ps) / len(ps)
            strat = stratify_by_corridor(ps)
            all_final[f'{mt}_standard_{dp}'] = {'accuracy': acc, 'n': len(ps), 'stratified': strat}
            print(f"    standard {dp}: {acc:.4f}")

        # Corner cases (pure_corridor from suite)
        for cat_key in ['pure_corridor']:
            bucket = suite['constructed'].get(cat_key)
            if not bucket: continue
            for dp in DECODE_POLICIES:
                ps = generate_blanks(m, tok, bucket['entries'], decode_policy=dp, device=DEVICE)
                acc = sum(r['correct'] for r in ps) / len(ps)
                all_final[f'{mt}_corner_{cat_key}_{dp}'] = {'accuracy': acc, 'n': len(ps)}
                print(f"    corner/{cat_key} {dp}: {acc:.4f}")

        # Corridor sweep from suite
        print(f"  Corridor sweep...")
        for L in CORRIDOR_SWEEP:
            key = f'corridor_{L}'
            if key not in suite['constructed']: continue
            bucket = suite['constructed'][key]
            for dp in DECODE_POLICIES:
                ps = generate_blanks(m, tok, bucket['entries'], decode_policy=dp, device=DEVICE)
                acc = sum(r['correct'] for r in ps) / len(ps)
                all_final[f'{mt}_corridor_sweep_{L}_{dp}'] = {
                    'accuracy': acc, 'n': len(ps), 'min_corridor': L}
                print(f"    corridor>={L:2d} {dp}: {acc:.4f}")

        # Backbone sweep (PRIMARY EXTREME AXIS) from suite
        print(f"  Backbone sweep...")
        for L in BACKBONE_SWEEP:
            key = f'backbone_{L}'
            if key not in suite['constructed']: continue
            bucket = suite['constructed'][key]
            for dp in DECODE_POLICIES:
                ps = generate_blanks(m, tok, bucket['entries'], decode_policy=dp, device=DEVICE)
                acc = sum(r['correct'] for r in ps) / len(ps)
                all_final[f'{mt}_backbone_sweep_{L}_{dp}'] = {
                    'accuracy': acc, 'n': len(ps), 'min_backbone': L}
                print(f"    backbone>={L:2d} {dp}: {acc:.4f}")

        # Dead-end sweep from suite
        print(f"  Dead-end sweep...")
        for L in DEAD_END_SWEEP:
            key = f'dead_end_{L}'
            if key not in suite['constructed']: continue
            bucket = suite['constructed'][key]
            for dp in DECODE_POLICIES:
                ps = generate_blanks(m, tok, bucket['entries'], decode_policy=dp, device=DEVICE)
                acc = sum(r['correct'] for r in ps) / len(ps)
                all_final[f'{mt}_dead_end_sweep_{L}_{dp}'] = {
                    'accuracy': acc, 'n': len(ps), 'min_dead_end': L}
                print(f"    dead_end>={L:2d} {dp}: {acc:.4f}")

        # Corridor rarity (natural)
        ps_conf = generate_blanks(m, tok, natural_entries, decode_policy='confidence', device=DEVICE)
        rarity = analyse_corridor_rarity(ps_conf, natural_entries)
        all_final[f'{mt}_corridor_rarity'] = rarity

        # Reveal trajectory on extreme backbone bucket — core failure diagnostic
        extreme_key = None
        for L in sorted(BACKBONE_SWEEP, reverse=True):
            if f'backbone_{L}' in suite['constructed']:
                extreme_key = f'backbone_{L}'; break
        if extreme_key is not None:
            K_for_reveal = globals().get('_PUMA_K_FINAL', REVEAL_K_DEFAULT)
            rev = analyse_reveal_patterns(
                m, tok, suite['constructed'][extreme_key], max_len,
                K=K_for_reveal, tau=PUMA_TAU, device=DEVICE)
            all_final[f'{mt}_reveal_{extreme_key}'] = rev
            nv = sum(1 for v in rev.get('never_revealed', []) if v > 0.5)
            print(f"    Reveal on {extreme_key}: n={rev.get('n', 0)}, "
                  f"{nv} positions never-revealed >50% of the time")

        # Error localization (natural)
        el = analyse_error_localization(ps_conf, natural_entries)
        all_final[f'{mt}_error_loc'] = el
        if el.get('total_errors', 0) > 0:
            parts = [f"{k}={v:.2f}" for k, v in el.items()
                     if k != 'total_errors' and isinstance(v, float)]
            print(f"    {el['total_errors']} errors: {' '.join(parts)}")

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
                               init_state=saved_states[src], device=DEVICE)
            all_dyn[f'dyn_{label}'] = d

            for dp in DECODE_POLICIES:
                ps = generate_blanks(m, tok, natural_entries, decode_policy=dp, device=DEVICE)
                acc = sum(r['correct'] for r in ps) / len(ps)
                all_final[f'{label}_standard_{dp}'] = {'accuracy': acc, 'n': len(ps)}
                print(f"    standard {dp}: {acc:.4f}")

            # Backbone sweep for continuation
            for L in BACKBONE_SWEEP:
                key = f'backbone_{L}'
                if key not in suite['constructed']: continue
                for dp in ['confidence']:
                    ps = generate_blanks(m, tok, suite['constructed'][key]['entries'],
                                          decode_policy=dp, device=DEVICE)
                    acc = sum(r['correct'] for r in ps) / len(ps)
                    all_final[f'{label}_backbone_sweep_{L}_{dp}'] = {
                        'accuracy': acc, 'n': len(ps)}
                    print(f"    backbone>={L:2d} {dp}: {acc:.4f}")

            del m; torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── Figures ──
    figs = make_figures(all_dyn, all_final)

    # ── Save ──
    sd = {'config': {k: globals()[k] for k in [
        'GRID_N', 'GRID_H', 'GRID_W', 'CELL_N', 'N_TRAIN', 'N_TEST', 'MAX_ITERS',
        'BATCH_SIZE', 'N_LAYER', 'N_HEAD', 'N_EMBD', 'MASK_TYPES', 'DECODE_POLICIES',
        'PUMA_K_START', 'PUMA_K_END', 'PUMA_K_STEP', 'PUMA_K_EVERY', 'PUMA_TAU',
        'STRAIGHTNESS_BIAS', 'BACKBONE_SWEEP', 'DEAD_END_SWEEP', 'CORRIDOR_SWEEP']}}
    for k, v in all_dyn.items():
        sd[k] = {
            'checkpoints': v['checkpoints'],
            'train_loss': v['train_loss'],
            'stratified_loss': v.get('stratified_loss', []),
            'stratum_names': v.get('stratum_names', []),
        }
    for k, v in all_final.items():
        sd[f'final_{k}'] = v
    save_results(exp_name, sd, figures=figs)

    # ── Summary ──
    print(f"\n{'='*70}\n  SUMMARY\n{'='*70}")
    print(f"\n  {'Test':<40s}", end='')
    for mt in MASK_TYPES: print(f" {mt:>14s}", end='')
    print()
    for dp in DECODE_POLICIES:
        for tt in ['standard', 'corner_pure_corridor']:
            key_parts = [f'{mt}_{tt}_{dp}' for mt in MASK_TYPES]
            accs = [all_final.get(k, {}).get('accuracy') for k in key_parts]
            if any(a is not None for a in accs):
                print(f"  {tt+'_'+dp:<40s}", end='')
                for a in accs:
                    print(f" {a:>14.4f}" if a is not None else f" {'N/A':>14s}", end='')
                print()
        for L in BACKBONE_SWEEP:
            key_parts = [f'{mt}_backbone_sweep_{L}_{dp}' for mt in MASK_TYPES]
            accs = [all_final.get(k, {}).get('accuracy') for k in key_parts]
            if any(a is not None for a in accs):
                print(f"  {'backbone>='+str(L)+'_'+dp:<40s}", end='')
                for a in accs:
                    print(f" {a:>14.4f}" if a is not None else f" {'N/A':>14s}", end='')
                print()

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
