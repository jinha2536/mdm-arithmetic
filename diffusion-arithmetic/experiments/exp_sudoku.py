"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Sudoku — Technique-Level Analysis of PUMA vs Random Masking
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Primary axis: technique_level (what solving technique a cell requires)
    tl_0: naked/hidden singles  → strong stepping stone
    tl_1: pairs/triples/pointing → moderate stepping stone
    tl_2: X-wing/swordfish/wings → weak stepping stone
    tl_3: forcing chains        → logical stepping stone
    tl_4: bifurcation/search    → NO stepping stone (= addition full_propagate)

  Prediction: tl_4 cells show random > puma crossover.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import sys, os, time, math, json, random, copy
from itertools import combinations
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                if '__file__' in dir() else '.')
from core.tokenizer import CharTokenizer
from core.train_utils import (
    mount_drive, save_results, save_checkpoint, encode_samples,
    train_diffusion, puma_k_fixed, generate_diffusion, DEVICE,
)

EXP_NAME = 'exp_sudoku'

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Config
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ANS_LEN = 81

# Data — training distribution
# DIFFICULTY_DECAY controls the rarity gradient:
#   None → use all available data per tier (balanced, like carry-balanced addition)
#   0.1  → easy,medium at 100%, then hard×0.1, very_hard×0.01, ... (like natural addition)
DIFFICULTY_DECAY = None
# Per-tier test counts: more easy/medium to balance technique_level distribution
# (hard/extreme puzzles are almost entirely tl_4; easy/medium have more tl_0-3)
N_TEST_PER_TIER_DICT = {
    'easy': 1000, 'medium': 1000, 'hard': 500, 'very_hard': 200, 'extreme': 100,
}
N_TEST_PER_TIER = 500  # fallback for non-dict usage
DATA_SOURCE = 'hf'      # 'hf', 'csv', 'generate'
CSV_PATH = ''

# Model (PUMA paper Sudoku config)
N_LAYER = 8; N_HEAD = 8; N_EMBD = 256
DROPOUT = 0.0; POS_ENC = 'absolute'

# Training
MAX_ITERS = 400000; BATCH_SIZE = 64
LR = 3e-4; MIN_LR = 1e-5; WARMUP_ITERS = 2000
GRAD_CLIP = 1.0; WEIGHT_DECAY = 0.01
EMA_DECAY = 0.9999
EVAL_EVERY = 8000; LOG_EVERY = 2000; GEN_EVAL_EVERY = 20000; GEN_EVAL_N = 100

MASK_TYPES = ['random', 'puma']
DECODE_POLICIES = ['confidence', 'oracle_solver', 'oracle_technique', 'random']
PUMA_TAU = 0.9; PUMA_K = 8
SEED = 42

# Continuation training
CONTINUATION_ITERS = 10000

# Rating tiers — auto-determined from data via _compute_rating_tiers()
# Fallback fixed tiers (overridden by actual data distribution)
RATING_TIERS = {
    'easy': (0, 0), 'medium': (1, 9), 'hard': (10, 49),
    'very_hard': (50, 149), 'extreme': (150, 99999),
}


def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data-source', default=None, choices=['hf', 'csv', 'generate'])
    p.add_argument('--csv-path', default=None)
    p.add_argument('--difficulty-decay', type=float, default=None,
                   help='Exponential decay for hard tiers. None=balanced, 0.1=natural')
    p.add_argument('--n-test', type=int, default=None)
    p.add_argument('--max-iters', type=int, default=None)
    p.add_argument('--batch-size', type=int, default=None)
    p.add_argument('--n-layer', type=int, default=None)
    p.add_argument('--n-head', type=int, default=None)
    p.add_argument('--n-embd', type=int, default=None)
    p.add_argument('--dropout', type=float, default=None)
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--masks', nargs='+', default=None)
    p.add_argument('--decode', nargs='+', default=None)
    p.add_argument('--puma-k', type=int, default=None)
    p.add_argument('--puma-tau', type=float, default=None)
    p.add_argument('--continuation-iters', type=int, default=None)
    p.add_argument('--no-continuation', action='store_true')
    p.add_argument('--tag', type=str, default='')
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--seeds', nargs='+', type=int, default=None)
    try:
        args, _ = p.parse_known_args()
    except SystemExit:
        args, _ = p.parse_known_args([])
    g = globals()
    for a, gl in {'data_source': 'DATA_SOURCE', 'csv_path': 'CSV_PATH',
                   'n_test': 'N_TEST_PER_TIER', 'max_iters': 'MAX_ITERS',
                   'batch_size': 'BATCH_SIZE', 'n_layer': 'N_LAYER', 'n_head': 'N_HEAD',
                   'n_embd': 'N_EMBD', 'dropout': 'DROPOUT', 'lr': 'LR',
                   'puma_k': 'PUMA_K', 'puma_tau': 'PUMA_TAU', 'seed': 'SEED',
                   'difficulty_decay': 'DIFFICULTY_DECAY',
                   'continuation_iters': 'CONTINUATION_ITERS'}.items():
        v = getattr(args, a, None)
        if v is not None: g[gl] = v
    if args.masks: g['MASK_TYPES'] = args.masks
    if args.decode: g['DECODE_POLICIES'] = args.decode
    return args


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sudoku Core
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_rows = [[(r, c) for c in range(9)] for r in range(9)]
_cols = [[(r, c) for r in range(9)] for c in range(9)]
_boxes = [[(br+dr, bc+dc) for dr in range(3) for dc in range(3)]
          for br in range(0, 9, 3) for bc in range(0, 9, 3)]
ALL_GROUPS = _rows + _cols + _boxes
PEERS_FLAT = [None] * 81
UNITS_FLAT = [None] * 81
for _r in range(9):
    for _c in range(9):
        _i = _r * 9 + _c
        _units = [u for u in ALL_GROUPS if (_r, _c) in u]
        UNITS_FLAT[_i] = [[rr * 9 + cc for rr, cc in u] for u in _units]
        _ps = set()
        for u in _units:
            for rr, cc in u: _ps.add(rr * 9 + cc)
        _ps.discard(_i)
        PEERS_FLAT[_i] = list(_ps)

ALL_9 = 0x1FF
VAL_BIT = [0] + [1 << (v - 1) for v in range(1, 10)]
BIT_VAL = {1 << i: i + 1 for i in range(9)}
POPCOUNT = [bin(i).count('1') for i in range(512)]
LOWEST_BIT = [0] * 512
for _i in range(1, 512): LOWEST_BIT[_i] = _i & (-_i)

# Flat unit lookups for technique solver
ROW_OF = [i // 9 for i in range(81)]
COL_OF = [i % 9 for i in range(81)]
BOX_OF = [(i // 9 // 3) * 3 + (i % 9 // 3) for i in range(81)]
ROW_CELLS = [[r * 9 + c for c in range(9)] for r in range(9)]
COL_CELLS = [[r * 9 + c for r in range(9)] for c in range(9)]
BOX_CELLS = [[(br + dr) * 9 + (bc + dc) for dr in range(3) for dc in range(3)]
             for br in range(0, 9, 3) for bc in range(0, 9, 3)]
ALL_UNITS = ROW_CELLS + COL_CELLS + BOX_CELLS  # 27 units, flat indices

# For each cell, which box does it belong to, and which row/col cells are in that box
BOX_ROW_INTER = {}   # (box, row) → list of cells in intersection
BOX_COL_INTER = {}
for b in range(9):
    for cell in BOX_CELLS[b]:
        r, c = ROW_OF[cell], COL_OF[cell]
        BOX_ROW_INTER.setdefault((b, r), []).append(cell)
        BOX_COL_INTER.setdefault((b, c), []).append(cell)

def _bm_init(grid):
    cands = [ALL_9] * 81
    for i in range(81):
        if grid[i] != 0:
            if not _bm_assign(cands, i, grid[i]): return None
    return cands

def _bm_assign(cands, i, val):
    others = cands[i] & ~VAL_BIT[val]
    while others:
        ob = LOWEST_BIT[others]
        if not _bm_elim(cands, i, BIT_VAL[ob]): return False
        others &= ~ob
    return True

def _bm_elim(cands, i, val):
    bit = VAL_BIT[val]
    if not (cands[i] & bit): return True
    cands[i] &= ~bit
    c = cands[i]
    if c == 0: return False
    if POPCOUNT[c] == 1:
        for p in PEERS_FLAT[i]:
            if not _bm_elim(cands, p, BIT_VAL[c]): return False
    for unit in UNITS_FLAT[i]:
        places = [j for j in unit if cands[j] & bit]
        if len(places) == 0: return False
        if len(places) == 1:
            if not _bm_assign(cands, places[0], val): return False
    return True

def _bm_solved(cands):
    return all(POPCOUNT[cands[i]] == 1 for i in range(81))

def _bm_search(cands, solutions, max_sol):
    if len(solutions) >= max_sol: return
    best_i, best_n = -1, 10
    for i in range(81):
        n = POPCOUNT[cands[i]]
        if 1 < n < best_n: best_i, best_n = i, n
    if best_i == -1:
        solutions.append([BIT_VAL[cands[i]] for i in range(81)]); return
    c = cands[best_i]
    while c:
        bit = LOWEST_BIT[c]; cp = list(cands)
        if _bm_assign(cp, best_i, BIT_VAL[bit]): _bm_search(cp, solutions, max_sol)
        c &= ~bit

def compute_prop_depth(puzzle_flat):
    """Iterative constraint propagation depth for each blank cell."""
    grid = list(puzzle_flat)
    blanks = {i for i in range(81) if grid[i] == 0}
    depths = {}
    current_depth = 0
    def _cands(i):
        if grid[i] != 0: return set()
        return set(range(1, 10)) - {grid[p] for p in PEERS_FLAT[i]} - {0}
    while True:
        newly = []
        for i in blanks - set(depths):
            c = _cands(i)
            if len(c) == 1:
                grid[i] = c.pop(); depths[i] = current_depth; newly.append(i)
        for group_cells in ALL_GROUPS:
            gi = [r*9+c for r, c in group_cells]
            unfilled = [i for i in gi if grid[i] == 0 and i not in depths]
            if not unfilled: continue
            for val in range(1, 10):
                if any(grid[i] == val for i in gi): continue
                possible = [i for i in unfilled if val in _cands(i)]
                if len(possible) == 1 and possible[0] not in depths:
                    grid[possible[0]] = val; depths[possible[0]] = current_depth
                    newly.append(possible[0])
        if not newly: break
        current_depth += 1
    for i in blanks:
        if i not in depths: depths[i] = -1
    return depths

def compute_n_candidates(puzzle_flat):
    result = {}
    for i in range(81):
        if puzzle_flat[i] == 0:
            result[i] = 9 - len({puzzle_flat[p] for p in PEERS_FLAT[i]} - {0})
    return result

def compute_n_cands_after_cp(puzzle_flat):
    cands = _bm_init(puzzle_flat)
    if cands is None: return {}
    return {i: POPCOUNT[cands[i]] for i in range(81)
            if puzzle_flat[i] == 0 and POPCOUNT[cands[i]] > 1}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Technique-Level Solver
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Level 0: Naked singles + Hidden singles
# Level 1: Naked/Hidden Pairs & Triples, Pointing Pairs, Box-Line Reduction
# Level 2: X-Wing, Swordfish, XY-Wing
# Level 3: Simple Forcing Chains (depth-1 hypothesis → contradiction)
# Level 4: Remaining (requires bifurcation/search — no stepping stone)

def _elim(cands, i, val):
    """Eliminate val from cands[i]. Returns False if contradiction."""
    bit = VAL_BIT[val]
    if not (cands[i] & bit): return True
    cands[i] &= ~bit
    return cands[i] != 0

def _singles_propagate(cands):
    """Apply naked + hidden singles with full cascade (uses _bm_assign).
    Returns (changed, valid) — changed=set of cells newly determined, valid=bool."""
    changed = set()
    seen = set(i for i in range(81) if POPCOUNT[cands[i]] == 1)
    progress = True
    while progress:
        progress = False
        # Naked singles: cells with exactly one candidate
        for i in range(81):
            if POPCOUNT[cands[i]] == 1 and i not in seen:
                seen.add(i); changed.add(i); progress = True
        # Hidden singles: value appears in only one cell in a unit
        for unit in ALL_UNITS:
            for v in range(1, 10):
                bit = VAL_BIT[v]
                places = [c for c in unit if cands[c] & bit]
                if len(places) == 0: return changed, False
                if len(places) == 1 and POPCOUNT[cands[places[0]]] > 1:
                    c = places[0]
                    if not _bm_assign(cands, c, v): return changed, False
                    changed.add(c); seen.add(c); progress = True
    return changed, True

def _apply_naked_subsets(cands, unit, size):
    """Find naked pairs (size=2) or triples (size=3) in a unit. Returns True if any elimination."""
    unsolved = [c for c in unit if POPCOUNT[cands[c]] > 1]
    if len(unsolved) < size: return False
    elim_any = False
    # Find subsets of `size` cells whose union of candidates has exactly `size` values
    # For efficiency, only check cells with <= size candidates
    eligible = [c for c in unsolved if POPCOUNT[cands[c]] <= size]
    if len(eligible) < size: return False
    for combo in combinations(eligible, size):
        union = 0
        for c in combo: union |= cands[c]
        if POPCOUNT[union] == size:
            # Found naked subset — eliminate these values from other cells in unit
            for c in unsolved:
                if c not in combo and (cands[c] & union):
                    cands[c] &= ~union
                    if cands[c] == 0: return True  # contradiction handled upstream
                    elim_any = True
    return elim_any

def _apply_hidden_subsets(cands, unit, size):
    """Find hidden pairs/triples in a unit. Returns True if any elimination."""
    unsolved = [c for c in unit if POPCOUNT[cands[c]] > 1]
    if len(unsolved) <= size: return False
    elim_any = False
    # For each subset of `size` values, check if they appear in exactly `size` cells
    vals_in_unit = set()
    for c in unsolved:
        m = cands[c]
        while m:
            vals_in_unit.add(BIT_VAL[LOWEST_BIT[m]]); m &= m - 1
    if len(vals_in_unit) < size: return False
    for val_combo in combinations(vals_in_unit, size):
        val_mask = 0
        for v in val_combo: val_mask |= VAL_BIT[v]
        # Which cells contain any of these values?
        cells_with = [c for c in unsolved if cands[c] & val_mask]
        if len(cells_with) == size:
            # Hidden subset — remove all OTHER candidates from these cells
            for c in cells_with:
                if cands[c] & ~val_mask:
                    cands[c] &= val_mask
                    elim_any = True
    return elim_any

def _apply_pointing(cands):
    """Pointing pairs/triples: value confined to one row/col within a box."""
    elim_any = False
    for b in range(9):
        for v in range(1, 10):
            bit = VAL_BIT[v]
            cells = [c for c in BOX_CELLS[b] if cands[c] & bit]
            if len(cells) < 2: continue
            rows = set(ROW_OF[c] for c in cells)
            cols = set(COL_OF[c] for c in cells)
            if len(rows) == 1:
                # All in one row — eliminate from rest of row outside box
                r = rows.pop()
                for c in ROW_CELLS[r]:
                    if BOX_OF[c] != b and (cands[c] & bit):
                        cands[c] &= ~bit; elim_any = True
                        if cands[c] == 0: return True
            if len(cols) == 1:
                c_col = cols.pop()
                for c in COL_CELLS[c_col]:
                    if BOX_OF[c] != b and (cands[c] & bit):
                        cands[c] &= ~bit; elim_any = True
                        if cands[c] == 0: return True
    return elim_any

def _apply_box_line(cands):
    """Box-line reduction: value in a row/col confined to one box."""
    elim_any = False
    for v in range(1, 10):
        bit = VAL_BIT[v]
        # Check rows
        for r in range(9):
            cells = [c for c in ROW_CELLS[r] if cands[c] & bit]
            if len(cells) < 2: continue
            boxes = set(BOX_OF[c] for c in cells)
            if len(boxes) == 1:
                b = boxes.pop()
                for c in BOX_CELLS[b]:
                    if ROW_OF[c] != r and (cands[c] & bit):
                        cands[c] &= ~bit; elim_any = True
                        if cands[c] == 0: return True
        # Check cols
        for co in range(9):
            cells = [c for c in COL_CELLS[co] if cands[c] & bit]
            if len(cells) < 2: continue
            boxes = set(BOX_OF[c] for c in cells)
            if len(boxes) == 1:
                b = boxes.pop()
                for c in BOX_CELLS[b]:
                    if COL_OF[c] != co and (cands[c] & bit):
                        cands[c] &= ~bit; elim_any = True
                        if cands[c] == 0: return True
    return elim_any

def _apply_xwing(cands):
    """X-Wing: value in exactly 2 cols in 2 rows → eliminate from those cols."""
    elim_any = False
    for v in range(1, 10):
        bit = VAL_BIT[v]
        # Row-based X-Wing
        row_cols = {}  # row → frozenset of cols where v appears
        for r in range(9):
            cols = frozenset(COL_OF[c] for c in ROW_CELLS[r] if cands[c] & bit)
            if len(cols) == 2: row_cols[r] = cols
        rows_list = list(row_cols.keys())
        for i in range(len(rows_list)):
            for j in range(i + 1, len(rows_list)):
                r1, r2 = rows_list[i], rows_list[j]
                if row_cols[r1] == row_cols[r2]:
                    for co in row_cols[r1]:
                        for c in COL_CELLS[co]:
                            if ROW_OF[c] != r1 and ROW_OF[c] != r2 and (cands[c] & bit):
                                cands[c] &= ~bit; elim_any = True
                                if cands[c] == 0: return True
        # Col-based X-Wing
        col_rows = {}
        for co in range(9):
            rows = frozenset(ROW_OF[c] for c in COL_CELLS[co] if cands[c] & bit)
            if len(rows) == 2: col_rows[co] = rows
        cols_list = list(col_rows.keys())
        for i in range(len(cols_list)):
            for j in range(i + 1, len(cols_list)):
                c1, c2 = cols_list[i], cols_list[j]
                if col_rows[c1] == col_rows[c2]:
                    for r in col_rows[c1]:
                        for c in ROW_CELLS[r]:
                            if COL_OF[c] != c1 and COL_OF[c] != c2 and (cands[c] & bit):
                                cands[c] &= ~bit; elim_any = True
                                if cands[c] == 0: return True
    return elim_any

def _apply_swordfish(cands):
    """Swordfish: value in ≤3 cols across 3 rows (and vice versa)."""
    elim_any = False
    for v in range(1, 10):
        bit = VAL_BIT[v]
        # Row-based
        row_cols = {}
        for r in range(9):
            cols = frozenset(COL_OF[c] for c in ROW_CELLS[r] if cands[c] & bit)
            if 2 <= len(cols) <= 3: row_cols[r] = cols
        for combo in combinations(row_cols.keys(), 3):
            union = row_cols[combo[0]] | row_cols[combo[1]] | row_cols[combo[2]]
            if len(union) <= 3:
                for co in union:
                    for c in COL_CELLS[co]:
                        if ROW_OF[c] not in combo and (cands[c] & bit):
                            cands[c] &= ~bit; elim_any = True
                            if cands[c] == 0: return True
        # Col-based
        col_rows = {}
        for co in range(9):
            rows = frozenset(ROW_OF[c] for c in COL_CELLS[co] if cands[c] & bit)
            if 2 <= len(rows) <= 3: col_rows[co] = rows
        for combo in combinations(col_rows.keys(), 3):
            union = col_rows[combo[0]] | col_rows[combo[1]] | col_rows[combo[2]]
            if len(union) <= 3:
                for r in union:
                    for c in ROW_CELLS[r]:
                        if COL_OF[c] not in combo and (cands[c] & bit):
                            cands[c] &= ~bit; elim_any = True
                            if cands[c] == 0: return True
    return elim_any

def _apply_xy_wing(cands):
    """XY-Wing: pivot {x,y} + pincer1 {x,z} + pincer2 {y,z} → eliminate z."""
    elim_any = False
    bivalue = [i for i in range(81) if POPCOUNT[cands[i]] == 2]
    peers_set = [set(PEERS_FLAT[i]) for i in range(81)]
    for pivot in bivalue:
        pv = cands[pivot]  # {x, y}
        pivot_peers = [p for p in bivalue if p in peers_set[pivot] and p != pivot]
        for pi in range(len(pivot_peers)):
            p1 = pivot_peers[pi]
            c1 = cands[p1]
            shared1 = pv & c1
            if POPCOUNT[shared1] != 1: continue  # must share exactly one value
            z1 = c1 & ~shared1  # the non-shared value from p1
            for pj in range(pi + 1, len(pivot_peers)):
                p2 = pivot_peers[pj]
                if p2 in peers_set[p1]: continue  # pincers must NOT see each other
                c2 = cands[p2]
                shared2 = pv & c2
                if POPCOUNT[shared2] != 1: continue
                if shared1 == shared2: continue  # must share DIFFERENT values with pivot
                z2 = c2 & ~shared2
                if z1 != z2: continue  # pincers must share the same non-pivot value z
                z_bit = z1
                # Eliminate z from cells that see BOTH pincers
                for c in range(81):
                    if c != p1 and c != p2 and c != pivot:
                        if c in peers_set[p1] and c in peers_set[p2]:
                            if cands[c] & z_bit:
                                cands[c] &= ~z_bit; elim_any = True
                                if cands[c] == 0: return True
    return elim_any

def _apply_forcing_chains(cands):
    """Simple forcing chains (depth-1): if assigning value v to cell i leads to
    contradiction via singles propagation → eliminate v.
    Only tests bivalue/trivalue cells for efficiency (captures most real patterns).
    """
    elim_any = False
    for i in range(81):
        pc = POPCOUNT[cands[i]]
        if pc < 2 or pc > 3: continue  # only bi/trivalue cells
        c = cands[i]
        while c:
            bit = LOWEST_BIT[c]; c &= c - 1
            val = BIT_VAL[bit]
            # Hypothesize: assign val to cell i
            hyp = list(cands)
            if not _bm_assign(hyp, i, val):
                # Contradiction → eliminate this candidate
                cands[i] &= ~bit; elim_any = True
                if cands[i] == 0: return True
                break  # restart this cell since cands changed
    return elim_any


TECHNIQUE_LEVELS = ['tl_0_singles', 'tl_1_subsets', 'tl_2_fish_wing',
                     'tl_3_chains', 'tl_4_search']
TECHNIQUE_LEVEL_TO_ID = {n: i for i, n in enumerate(TECHNIQUE_LEVELS)}

def _tl_to_cat(meta):
    tl = meta.get('technique_level', -1)
    if meta['is_given']: return 'given'
    if tl == 0: return 'tl_0_singles'
    if tl == 1: return 'tl_1_subsets'
    if tl == 2: return 'tl_2_fish_wing'
    if tl == 3: return 'tl_3_chains'
    return 'tl_4_search'


def compute_technique_level(puzzle_flat):
    """Per-cell technique level via hierarchical solver.
    Returns dict: cell → level (0-4), and solve_order dict.

    Level 0: Naked/Hidden singles
    Level 1: Naked/Hidden pairs/triples, Pointing, Box-line reduction
    Level 2: X-Wing, Swordfish, XY-Wing
    Level 3: Forcing chains (depth-1 hypothesis → contradiction)
    Level 4: Remaining (requires search/bifurcation — TRUE no stepping stone)
    """
    blanks_set = set(i for i in range(81) if puzzle_flat[i] == 0)
    cands = _bm_init(puzzle_flat)
    if cands is None:
        return {i: 4 for i in blanks_set}, {i: idx for idx, i in enumerate(blanks_set)}

    cell_level = {}
    solve_order = {}
    oc = [0]

    def _record(level):
        for c in range(81):
            if c in blanks_set and c not in cell_level and POPCOUNT[cands[c]] == 1:
                cell_level[c] = level
                solve_order[c] = oc[0]; oc[0] += 1

    # ── Level 0: _bm_init already did full singles propagation ──
    _record(0)
    if len(cell_level) == len(blanks_set):
        return cell_level, solve_order

    # ── Level 1: Subsets + Intersection ──
    for _ in range(30):  # cap iterations
        prog = False
        for unit in ALL_UNITS:
            prog |= _apply_naked_subsets(cands, unit, 2)
            prog |= _apply_naked_subsets(cands, unit, 3)
            prog |= _apply_hidden_subsets(cands, unit, 2)
            prog |= _apply_hidden_subsets(cands, unit, 3)
        prog |= _apply_pointing(cands)
        prog |= _apply_box_line(cands)
        det, valid = _singles_propagate(cands)
        if not valid: break
        if det: prog = True
        _record(1)
        if not prog or len(cell_level) == len(blanks_set): break

    if len(cell_level) == len(blanks_set):
        return cell_level, solve_order

    # ── Level 2: Fish + Wings ──
    for _ in range(20):
        prog = False
        prog |= _apply_xwing(cands)
        prog |= _apply_swordfish(cands)
        prog |= _apply_xy_wing(cands)
        # Cascade lower techniques
        for unit in ALL_UNITS:
            prog |= _apply_naked_subsets(cands, unit, 2)
            prog |= _apply_naked_subsets(cands, unit, 3)
        prog |= _apply_pointing(cands)
        prog |= _apply_box_line(cands)
        det, valid = _singles_propagate(cands)
        if not valid: break
        if det: prog = True
        _record(2)
        if not prog or len(cell_level) == len(blanks_set): break

    if len(cell_level) == len(blanks_set):
        return cell_level, solve_order

    # ── Level 3: Forcing chains ──
    for _ in range(15):
        prog = _apply_forcing_chains(cands)
        # Cascade all lower
        for unit in ALL_UNITS:
            prog |= _apply_naked_subsets(cands, unit, 2)
        prog |= _apply_pointing(cands)
        prog |= _apply_xwing(cands)
        det, valid = _singles_propagate(cands)
        if not valid: break
        if det: prog = True
        _record(3)
        if not prog or len(cell_level) == len(blanks_set): break

    # ── Level 4: Remaining ──
    for i in blanks_set:
        if i not in cell_level:
            cell_level[i] = 4
            solve_order[i] = oc[0]; oc[0] += 1

    return cell_level, solve_order


def compute_guess_depth(puzzle_flat):
    """Per-cell guess depth + solve order via MRV backtracking solver.
    Returns:
      cell_depth: dict cell → guess_depth (0=CP, 1+=after k-th guess)
      solve_order: dict cell → rank in solver's determination sequence (0=first determined)
      total_guesses: int (≈ tdoku rating)
    The solve_order is the TRUE oracle ordering — the Sudoku analog of LSB in addition.
    """
    cands = _bm_init(puzzle_flat)
    blanks = [i for i in range(81) if puzzle_flat[i] == 0]
    if cands is None:
        return ({i: -1 for i in blanks}, {i: i for i in blanks}, 0)

    cell_depth = {}
    solve_order = {}
    order_counter = [0]
    total_guesses = [0]

    # Mark cells determined by initial CP (guess_depth=0)
    for i in blanks:
        if POPCOUNT[cands[i]] == 1:
            cell_depth[i] = 0
            solve_order[i] = order_counter[0]; order_counter[0] += 1

    def _search(cands_state, guess_count):
        best_i, best_n = -1, 10
        for i in range(81):
            n = POPCOUNT[cands_state[i]]
            if 1 < n < best_n:
                best_i, best_n = i, n
        if best_i == -1:
            # Solved — mark any remaining unassigned blanks
            for i in blanks:
                if i not in cell_depth and POPCOUNT[cands_state[i]] == 1:
                    cell_depth[i] = guess_count
                    solve_order[i] = order_counter[0]; order_counter[0] += 1
            return True

        guess_count += 1; total_guesses[0] += 1
        c = cands_state[best_i]
        while c:
            bit = LOWEST_BIT[c]
            cp = list(cands_state)
            if _bm_assign(cp, best_i, BIT_VAL[bit]):
                newly = []
                for i in blanks:
                    if i not in cell_depth and POPCOUNT[cp[i]] == 1:
                        cell_depth[i] = guess_count
                        solve_order[i] = order_counter[0]; order_counter[0] += 1
                        newly.append(i)
                if _search(cp, guess_count):
                    return True
                # Backtrack
                for i in newly:
                    del cell_depth[i]; del solve_order[i]
                order_counter[0] -= len(newly)
            c &= ~bit
        return False

    _search(list(cands), 0)
    for i in blanks:
        if i not in cell_depth:
            cell_depth[i] = -1
            solve_order[i] = order_counter[0]; order_counter[0] += 1
    return cell_depth, solve_order, total_guesses[0]


def compute_live_candidates(grid_flat):
    """Compute candidate count for each unfilled cell in a partially-solved grid.
    Used by adaptive_n_cands decode: recomputes at every decode step.
    Returns dict: cell_index → n_candidates (only for cells with value 0).
    """
    result = {}
    for i in range(81):
        if grid_flat[i] == 0:
            used = {grid_flat[p] for p in PEERS_FLAT[i]} - {0}
            result[i] = 9 - len(used)
    return result

DEPTH_CATS = ['given', 'depth_0', 'depth_1', 'depth_2', 'depth_3plus', 'depth_hard']
DEPTH_CAT_TO_ID = {n: i for i, n in enumerate(DEPTH_CATS)}

GUESS_CATS = ['guess_0', 'guess_1', 'guess_2', 'guess_3plus']
GUESS_CAT_TO_ID = {n: i for i, n in enumerate(GUESS_CATS)}

def _depth_to_cat(meta):
    if meta['is_given']: return 'given'
    d = meta['prop_depth']
    if d == 0: return 'depth_0'
    if d == 1: return 'depth_1'
    if d == 2: return 'depth_2'
    if d >= 3: return 'depth_3plus'
    return 'depth_hard'

def _guess_to_cat(meta):
    gd = meta.get('guess_depth', 0)
    if gd <= 0: return 'guess_0'
    if gd == 1: return 'guess_1'
    if gd == 2: return 'guess_2'
    return 'guess_3plus'


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Data Loading
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _compute_puzzle_meta(puzzle_str, sol_str):
    """Compute metadata for one puzzle. Works with both '0' and '.' for blanks."""
    puzzle_flat = [int(c) if c.isdigit() else 0 for c in puzzle_str]
    n_blanks = sum(1 for v in puzzle_flat if v == 0)
    depths = compute_prop_depth(puzzle_flat)
    n_cands = compute_n_candidates(puzzle_flat)
    n_cands_cp = compute_n_cands_after_cp(puzzle_flat)
    # Per-cell guess depth via backtracking
    guess_depths, solve_order, total_guesses = compute_guess_depth(puzzle_flat)
    # Per-cell technique level via hierarchical solver
    tech_levels, tech_order = compute_technique_level(puzzle_flat)
    meta = {}
    for i in range(81):
        is_given = puzzle_flat[i] != 0
        meta[i] = {
            'is_given': is_given,
            'prop_depth': -99 if is_given else depths.get(i, -1),
            'n_candidates': 0 if is_given else n_cands.get(i, 0),
            'n_cands_after_cp': 0 if is_given else n_cands_cp.get(i, 0),
            'guess_depth': 0 if is_given else guess_depths.get(i, -1),
            'solve_order': -1 if is_given else solve_order.get(i, 999),
            'technique_level': 0 if is_given else tech_levels.get(i, 4),
            'technique_order': -1 if is_given else tech_order.get(i, 999),
        }
    # Normalize puzzle string to use '0' for blanks
    pstr = ''.join(str(v) for v in puzzle_flat)
    return {'string': f"{pstr}={sol_str}", 'meta': meta, 'n_blanks': n_blanks,
            'total_guesses': total_guesses}


def _compute_meta_worker(args):
    """Worker for parallel meta computation."""
    puzzle_str, sol_str, rating = args[:3]
    source = args[3] if len(args) > 3 else ''
    d = _compute_puzzle_meta(puzzle_str, sol_str)
    d['rating'] = rating
    d['source'] = source
    return d


def load_hf_data(n_test, decay=None, seed=42, cache_dir='.sudoku_cache'):
    """Load from HuggingFace sapientinc/sudoku-extreme.
    decay: None → use all available per tier (balanced)
           0.1  → easy,medium at 100%, hard×0.1, very_hard×0.01, ... (natural)
    """
    print("  Loading HuggingFace sudoku-extreme dataset...")
    from datasets import load_dataset
    ds = load_dataset('sapientinc/sudoku-extreme', cache_dir=cache_dir)

    rng = random.Random(seed)
    train_raw = ds['train']
    test_raw = ds['test']

    # ── Profile actual rating distribution ──
    print("  Profiling rating distribution...")
    all_ratings = train_raw['rating']
    n_total = len(all_ratings)
    sorted_r = sorted(all_ratings)
    r_max = sorted_r[-1]
    print(f"    Total: {n_total:,} | min={sorted_r[0]} max={r_max}")
    # Histogram
    bins = [0, 1, 5, 10, 20, 50, 100, 200, 500, r_max+1]
    for i in range(len(bins)-1):
        lo, hi = bins[i], bins[i+1]
        cnt = sum(1 for r in all_ratings if lo <= r < hi)
        if cnt > 0:
            label = f"[{lo},{hi})" if hi <= r_max else f"[{lo},{r_max}]"
            print(f"      rating {label:<12s}: {cnt:>10,} ({cnt/n_total:.1%})")
    for p in [50, 75, 90, 95, 99, 99.9]:
        idx = min(int(p/100 * n_total), n_total-1)
        print(f"    p{p}: {sorted_r[idx]}")

    # Auto-determine tier boundaries based on actual data
    # Keep easy=0, then split non-zero into meaningful groups
    global RATING_TIERS
    nonzero = [r for r in all_ratings if r > 0]
    if nonzero:
        nonzero_sorted = sorted(nonzero)
        nn = len(nonzero_sorted)
        p50 = nonzero_sorted[nn//2]
        p75 = nonzero_sorted[int(nn*0.75)]
        p90 = nonzero_sorted[int(nn*0.90)]
        p99 = nonzero_sorted[min(int(nn*0.99), nn-1)]
        RATING_TIERS = {
            'easy': (0, 0),
            'medium': (1, max(p50, 1)),
            'hard': (max(p50, 1)+1, p75),
            'very_hard': (p75+1, p90),
            'extreme': (p90+1, p99),
            'top1pct': (p99+1, r_max),
        }
        # Remove empty tiers
        RATING_TIERS = {k: v for k, v in RATING_TIERS.items() if v[0] <= v[1]}
        print(f"    Auto-determined tiers:")
        for tn, (lo, hi) in RATING_TIERS.items():
            cnt = sum(1 for r in all_ratings if lo <= r <= hi)
            print(f"      {tn}: [{lo}, {hi}] → {cnt:,} train samples")

    # Efficient sampling: batch-read ratings first, then filter
    def _sample_by_rating(data, lo, hi, n, rng_inst):
        ratings = data['rating']
        indices = [i for i, r in enumerate(ratings) if lo <= r <= hi]
        actual_n = min(len(indices), n)
        if len(indices) > n:
            indices = rng_inst.sample(indices, n)
        print(f"    rating [{lo},{hi}]: {len(indices)} available → sampling {actual_n}")
        if not indices: return []
        rows = data.select(indices)
        return [(rows[i]['question'], rows[i]['answer'], rows[i]['rating'],
                 rows[i].get('source', ''))
                for i in range(len(rows))]

    # Train sampling with difficulty decay
    print(f"  Sampling train data (decay={decay})...")
    train_tuples = []
    tier_counts = {}
    tier_names = list(RATING_TIERS.keys())
    # Find index where decay starts (after easy and medium)
    base_tiers = {'easy'}
    decay_idx = 0
    for i, tn in enumerate(tier_names):
        if tn not in base_tiers:
            decay_idx = i; break

    for i, (tier_name, (lo, hi)) in enumerate(RATING_TIERS.items()):
        # Count available in this tier
        available = sum(1 for r in all_ratings if lo <= r <= hi)
        if available == 0: continue

        if decay is None:
            # Balanced: use all available
            n_tier = available
        elif tier_name in base_tiers:
            # Base tiers: use all available
            n_tier = available
        else:
            # Apply exponential decay: λ^(tier_position - decay_start + 1)
            exp = i - decay_idx + 1
            if decay <= 0:
                n_tier = 0
            else:
                n_tier = max(1, int(available * (decay ** exp)))

        if n_tier <= 0:
            tier_counts[tier_name] = 0; continue
        samples = _sample_by_rating(train_raw, lo, hi, n_tier, rng)
        train_tuples.extend(samples)
        tier_counts[tier_name] = len(samples)
    rng.shuffle(train_tuples)

    # Test: separate tiers (per-tier counts for technique-level balance)
    print("  Sampling test data...")
    rng2 = random.Random(seed + 1000)
    test_tiers = {}
    for tier_name, (lo, hi) in RATING_TIERS.items():
        n_tier_test = N_TEST_PER_TIER_DICT.get(tier_name, n_test)
        samples = _sample_by_rating(test_raw, lo, hi, n_tier_test, rng2)
        if not samples:
            print(f"    → fallback to train set for {tier_name}")
            samples = _sample_by_rating(train_raw, lo, hi, n_tier_test, rng2)
        test_tiers[tier_name] = samples

    total = sum(tier_counts.values())
    dist_str = ' + '.join(f"{v} {k}" for k, v in tier_counts.items())
    print(f"  Train: {dist_str} = {total}")
    if total > 0:
        pct_str = ' '.join(f"{k}={v/total:.1%}" for k, v in tier_counts.items())
        print(f"  Train distribution: {pct_str}")
    for tn, ts in test_tiers.items():
        print(f"  Test/{tn}: {len(ts)}")

    return train_tuples, test_tiers


def load_csv_data(csv_path, total_n, n_test, seed=42):
    """Load from local CSV (puzzle,solution,rating or puzzle,solution,difficulty)."""
    import csv
    print(f"  Loading CSV: {csv_path}")
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            p = row.get('puzzle', row.get('quizzes', row.get('question', '')))
            s = row.get('solution', row.get('solutions', row.get('answer', '')))
            r = int(row.get('rating', row.get('difficulty', 0)))
            if len(p) == 81 and len(s) == 81:
                rows.append((p, s, r))
    print(f"  Loaded {len(rows)} puzzles from CSV")
    rng = random.Random(seed)
    rng.shuffle(rows)
    train_tuples = rows[:total_n]
    rest = rows[total_n:]
    test_tiers = {}
    for tn, (lo, hi) in RATING_TIERS.items():
        n_tier_test = N_TEST_PER_TIER_DICT.get(tn, n_test)
        test_tiers[tn] = [r for r in rest if lo <= r[2] <= hi][:n_tier_test]
    return train_tuples, test_tiers


def _gen_grid(rng):
    grid = [0]*81; base = list(range(1,10)); rng.shuffle(base)
    shifts = [0,3,6,1,4,7,2,5,8]
    for r in range(9):
        for c in range(9): grid[r*9+c] = base[(c+shifts[r])%9]
    for _ in range(30):
        t = rng.randint(0,4)
        if t < 2:
            band = rng.randint(0,2); a,b = rng.sample(range(band*3,band*3+3),2)
            for k in range(9):
                if t==0: grid[a*9+k], grid[b*9+k] = grid[b*9+k], grid[a*9+k]
                else: grid[k*9+a], grid[k*9+b] = grid[k*9+b], grid[k*9+a]
        elif t==4:
            perm = list(range(1,10)); rng.shuffle(perm)
            grid = [perm[v-1] for v in grid]
    return grid

def _make_puzzle(sol, nb, rng):
    puzzle = list(sol); cells = list(range(81)); rng.shuffle(cells)
    removed = 0
    for i in cells:
        if removed >= nb: break
        val = puzzle[i]; puzzle[i] = 0
        cands = _bm_init(puzzle)
        if cands and _bm_solved(cands): removed += 1
        else:
            sols = []
            if cands: _bm_search(cands, sols, 2)
            if len(sols) == 1: removed += 1
            else: puzzle[i] = val
    return puzzle if removed >= nb else None

def _gen_one(args):
    seed_val, min_bl, max_bl = args
    rng = random.Random(seed_val)
    sol = _gen_grid(rng)
    nb = rng.randint(min_bl, max_bl)
    puzzle = _make_puzzle(sol, nb, rng)
    if puzzle is None: return None
    ps = ''.join(str(v) for v in puzzle)
    ss = ''.join(str(v) for v in sol)
    return (ps, ss, 0)

def gen_fallback_data(n, seed=42, min_bl=45, max_bl=58):
    """Self-generate puzzles (all rating=0 since we can't compute tdoku rating)."""
    print(f"  Self-generating {n} puzzles...")
    args_list = [(seed*100000+i, min_bl, max_bl) for i in range(int(n*1.5)+200)]
    results = []
    workers = min(os.cpu_count() or 1, 16)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for r in pool.map(_gen_one, args_list, chunksize=max(1, len(args_list)//(workers*4))):
            if r: results.append(r)
            if len(results) >= n: break
    print(f"  Generated {len(results)} puzzles")
    return results[:n]


def _make_train_entry(puzzle_str, sol_str, rating, source=''):
    """Lightweight train entry — NO metadata computation."""
    pf = [int(c) if c.isdigit() else 0 for c in puzzle_str]
    pstr = ''.join(str(v) for v in pf)
    return {'string': f"{pstr}={sol_str}", 'rating': rating,
            'n_blanks': sum(1 for v in pf if v == 0)}


def prepare_datasets(source=None, seed=42):
    """Load data. Train = lightweight (no meta), Test = full metadata."""
    source = source or DATA_SOURCE
    if source == 'hf':
        train_tuples, test_tiers = load_hf_data(N_TEST_PER_TIER, decay=DIFFICULTY_DECAY, seed=seed)
    elif source == 'csv':
        train_tuples, test_tiers = load_csv_data(
            CSV_PATH, 500000, N_TEST_PER_TIER, seed)
    else:
        raw = gen_fallback_data(100000, seed)
        train_tuples = raw
        test_raw = gen_fallback_data(N_TEST_PER_TIER * 2, seed + 5000)
        test_tiers = {'all': test_raw}

    # Train data: lightweight (just string + rating, NO metadata)
    print(f"  Preparing {len(train_tuples)} train entries (lightweight)...")
    t0 = time.time()
    train_data = [_make_train_entry(*t[:3], t[3] if len(t) > 3 else '')
                  for t in train_tuples]
    print(f"  Train prepared in {time.time()-t0:.1f}s")

    # Test data: full metadata (only ~2K puzzles, fast)
    test_tuples = [t for ts in test_tiers.values() for t in ts]
    print(f"  Computing metadata for {len(test_tuples)} test puzzles...")
    t0 = time.time()
    workers = min(os.cpu_count() or 1, 16)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        test_entries = list(pool.map(_compute_meta_worker, test_tuples,
                                     chunksize=max(1, len(test_tuples) // (workers*4))))
    test_data = {}; idx = 0
    for tn, ts in test_tiers.items():
        test_data[tn] = test_entries[idx:idx+len(ts)]; idx += len(ts)
    print(f"  Test metadata computed in {time.time()-t0:.1f}s")

    # Diagnostics
    train_ratings = [d.get('rating', 0) for d in train_data]
    print(f"  [train] n={len(train_data)} rating: "
          f"min={min(train_ratings)} med={sorted(train_ratings)[len(train_ratings)//2]} "
          f"max={max(train_ratings)}")
    for name, data in test_data.items():
        if not data: continue
        ratings = [d.get('rating', 0) for d in data]
        depth_dist = defaultdict(int)
        guess_dist = defaultdict(int)
        for d in data:
            for j in range(81):
                depth_dist[_depth_to_cat(d['meta'][j])] += 1
                if not d['meta'][j]['is_given']:
                    guess_dist[_guess_to_cat(d['meta'][j])] += 1
        tg = [d.get('total_guesses', 0) for d in data]
        print(f"  [{name}] n={len(data)} rating: "
              f"min={min(ratings)} med={sorted(ratings)[len(ratings)//2]} max={max(ratings)}")
        print(f"    depth: {dict(sorted(depth_dist.items()))}")
        print(f"    guess: {dict(sorted(guess_dist.items()))}")
        if tg:
            print(f"    total_guesses: min={min(tg)} med={sorted(tg)[len(tg)//2]} max={max(tg)}")

    return train_data, test_data


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tokenizer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_tok():
    return CharTokenizer(list('0123456789='), {'mask': 'M', 'pad': 'P'})


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Probe (fully-masked, per-cell)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@torch.no_grad()
def probe_per_cell(model, tokenizer, test_data, max_len, device=None):
    """Fully-masked probe with per-depth-category and per-rating tracking."""
    if device is None: device = DEVICE
    model.eval()
    mask_id = tokenizer.special_ids['mask']
    strings = [d['string'] for d in test_data]
    ids_all, ans_all = encode_samples(strings, tokenizer, max_len)
    ids_all, ans_all = ids_all.to(device), ans_all.to(device)
    N = len(test_data)

    # Precompute blank masks and category ids
    blank_masks = torch.zeros(N, ANS_LEN, dtype=torch.bool, device=device)
    cat_ids = torch.zeros(N, ANS_LEN, dtype=torch.long, device=device)
    for si, d in enumerate(test_data):
        ps = d['string'].split('=')[0]
        for j in range(ANS_LEN):
            if ps[j] == '0': blank_masks[si, j] = True
            cat_ids[si, j] = DEPTH_CAT_TO_ID[_depth_to_cat(d['meta'][j])]

    n_cats = len(DEPTH_CATS)
    cat_conf = torch.zeros(n_cats, device=device)
    cat_correct = torch.zeros(n_cats, device=device)
    cat_count = torch.zeros(n_cats, dtype=torch.long, device=device)
    total_loss = torch.tensor(0.0, device=device)
    total_n = torch.tensor(0, dtype=torch.long, device=device)

    _arange = torch.arange(ANS_LEN, device=device)
    for st in range(0, N, 128):
        en = min(st+128, N)
        ids, ans = ids_all[st:en], ans_all[st:en]
        B, T = ids.shape
        ans_pos = (ans.unsqueeze(1) + _arange).clamp(max=T-1)
        bi = torch.arange(B, device=device).unsqueeze(1).expand_as(ans_pos)
        bl = blank_masks[st:en]

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

        # Per-category (vectorized scatter)
        cat_b = cat_ids[st:en]
        w_f, cf_f, co_f, ca_f = w.reshape(-1), confs.reshape(-1), corrects.reshape(-1), cat_b.reshape(-1)
        valid = w_f > 0
        if valid.any():
            vc = ca_f[valid]
            cat_conf.scatter_add_(0, vc, cf_f[valid])
            cat_correct.scatter_add_(0, vc, co_f[valid])
            cat_count.scatter_add_(0, vc, torch.ones_like(vc, dtype=torch.long))

    overall_loss = (total_loss / total_n.clamp(1)).item()
    overall_acc = cat_correct.sum().item() / cat_count.sum().clamp(1).item()
    depth_context = {}
    for ci, cn in enumerate(DEPTH_CATS):
        n = cat_count[ci].item()
        if n > 0:
            depth_context[cn] = {'mean_conf': cat_conf[ci].item()/n,
                                  'mean_acc': cat_correct[ci].item()/n, 'n': n}
    return {'overall_loss': overall_loss, 'overall_acc': overall_acc,
            'depth_context': depth_context}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Training
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def train(mask_type, tokenizer, train_data, test_data_dict, max_len,
          max_iters=None, init_state=None, device=None):
    """Train sudoku model using unified train_diffusion. Returns (model, dynamics)."""
    if device is None: device = DEVICE
    if max_iters is None: max_iters = MAX_ITERS

    strings = [d['string'] for d in train_data]
    train_ids, train_ans = encode_samples(strings, tokenizer, max_len)
    train_ids, train_ans = train_ids.to(device), train_ans.to(device)

    # Blank masks: only '0' cells are maskable
    zero_id = tokenizer.encode('0')[0]
    blank_masks = (train_ids[:, :ANS_LEN] == zero_id)

    # Pick one test tier for periodic eval
    eval_tier = 'hard' if 'hard' in test_data_dict else list(test_data_dict.keys())[0]
    eval_data = test_data_dict[eval_tier]

    # Eval callback
    def eval_fn(model, it, tg):
        probe = probe_per_cell(model, tokenizer, eval_data, max_len, device)
        dc = probe.get('depth_context', {})
        parts = [f"{c}={dc[c]['mean_acc']:.3f}" for c in DEPTH_CATS if c in dc and c != 'given']
        print(f"    [eval it {it}] loss={probe['overall_loss']:.4f} "
              f"acc={probe['overall_acc']:.4f} {' '.join(parts)}")
        return probe

    model, dynamics = train_diffusion(
        train_ids=train_ids, train_ans=train_ans, ans_len=ANS_LEN, tokenizer=tokenizer,
        mask_type=mask_type, blank_masks=blank_masks,
        puma_tau=PUMA_TAU, puma_k_schedule=puma_k_fixed(PUMA_K) if mask_type == 'puma' else None,
        n_layer=N_LAYER, n_head=N_HEAD, n_embd=N_EMBD, dropout=DROPOUT, pos_enc=POS_ENC,
        max_iters=max_iters, batch_size=BATCH_SIZE,
        lr=LR, min_lr=MIN_LR, warmup_iters=WARMUP_ITERS,
        grad_clip=GRAD_CLIP, weight_decay=WEIGHT_DECAY, ema_decay=EMA_DECAY,
        eval_fn=eval_fn, eval_every=EVAL_EVERY, log_every=LOG_EVERY,
        init_state=init_state, device=device,
    )
    return model, dynamics


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Generation (multi-policy decode)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@torch.no_grad()
def generate_blanks(model, tokenizer, test_data, decode_policy='confidence',
                    batch_size=32, device=None):
    """Decode blank cells with configurable policy.
    Policies:
      'confidence':       model max prob (adaptive, standard MDM)
      'oracle_solver':     backtracking solver determination order (true oracle)
                          (static oracle — Sudoku analog of addition's LSB)
      'adaptive_n_cands': fewest candidates in current grid state (adaptive)
      'n_cands':          fewest initial candidates (static)
      'n_cands_cp':       fewest CP candidates (static)
      'random':           random order (baseline)
    """
    if device is None: device = DEVICE
    mask_id = tokenizer.special_ids['mask']
    pad_id = tokenizer.special_ids['pad']
    model.eval()
    results = []
    digit_ids = {tokenizer.encode(str(d))[0]: d for d in range(10)}

    for st in range(0, len(test_data), batch_size):
        batch = test_data[st:st+batch_size]; B = len(batch)
        full_enc = [tokenizer.encode(d['string']) for d in batch]
        ml = max(len(e) for e in full_enc)
        ids = torch.full((B, ml), pad_id, dtype=torch.long, device=device)
        for i, e in enumerate(full_enc):
            ids[i, :len(e)] = torch.tensor(e, device=device)
        eq_id = tokenizer.encode('=')[0]
        ans_starts = torch.zeros(B, dtype=torch.long, device=device)
        for i in range(B):
            for t in range(ml):
                if ids[i, t].item() == eq_id: ans_starts[i] = t+1; break
        _ar = torch.arange(ANS_LEN, device=device)
        ap = (ans_starts.unsqueeze(1) + _ar).clamp(max=ml-1)
        bi = torch.arange(B, device=device).unsqueeze(1).expand_as(ap)

        # Pre-fill givens, mask blanks
        x = ids.clone()
        blank_m = torch.zeros(B, ANS_LEN, dtype=torch.bool, device=device)
        for i in range(B):
            ps = batch[i]['string'].split('=')[0]
            for j in range(ANS_LEN):
                if ps[j] == '0': x[i, ans_starts[i]+j] = mask_id; blank_m[i,j] = True
        n_blanks = blank_m.sum(dim=1)
        max_steps = n_blanks.max().item()

        # Precompute static decode order for static policies
        static_order = None
        if decode_policy in ('oracle_solver', 'oracle_depth', 'oracle_technique',
                             'n_cands', 'n_cands_cp', 'random'):
            static_order = torch.full((B, ANS_LEN), 9999, dtype=torch.long, device=device)
            for i in range(B):
                meta = batch[i]['meta']
                blank_js = [j for j in range(ANS_LEN) if not meta[j]['is_given']]
                if decode_policy == 'random':
                    random.shuffle(blank_js)
                    for rank, j in enumerate(blank_js): static_order[i, j] = rank
                elif decode_policy == 'oracle_solver':
                    # TRUE oracle: solver's actual determination order
                    blank_js.sort(key=lambda j: meta[j].get('solve_order', 999))
                    for rank, j in enumerate(blank_js): static_order[i, j] = rank
                elif decode_policy == 'oracle_technique':
                    # Technique-level oracle: level 0 first, then 1, 2, 3, 4
                    # Within same level, by technique_order
                    blank_js.sort(key=lambda j: (meta[j].get('technique_level', 4),
                                                  meta[j].get('technique_order', 999)))
                    for rank, j in enumerate(blank_js): static_order[i, j] = rank
                elif decode_policy == 'oracle_depth':
                    def _depth_key(j):
                        d = meta[j]['prop_depth']
                        nc = meta[j].get('n_cands_after_cp', 9) or 9
                        if d == -1: return (1000, nc)
                        return (d, nc)
                    blank_js.sort(key=_depth_key)
                    for rank, j in enumerate(blank_js): static_order[i, j] = rank
                elif decode_policy == 'n_cands':
                    blank_js.sort(key=lambda j: meta[j]['n_candidates'])
                    for rank, j in enumerate(blank_js): static_order[i, j] = rank
                elif decode_policy == 'n_cands_cp':
                    blank_js.sort(key=lambda j: (meta[j]['n_cands_after_cp']
                                                  if meta[j]['n_cands_after_cp'] > 0
                                                  else meta[j]['n_candidates']))
                    for rank, j in enumerate(blank_js): static_order[i, j] = rank

        decode_orders = torch.full((B, ANS_LEN), -1, dtype=torch.long, device=device)

        for step in range(max_steps):
            logits = model(x)
            al = logits[bi, ap]; cl = al.clone(); cl[:, :, mask_id] = -float('inf')
            probs = F.softmax(cl, dim=-1)
            still_m = (x[bi, ap] == mask_id)

            if decode_policy == 'confidence':
                confs = probs.max(dim=-1).values
                confs[~still_m] = -float('inf')
                best_j = confs.argmax(dim=1)

            elif decode_policy == 'adaptive_n_cands':
                # Reconstruct current grid per sample, compute live candidates
                priority = torch.full((B, ANS_LEN), 9999, dtype=torch.long, device=device)
                for i in range(B):
                    if not still_m[i].any(): continue
                    # Extract current grid from token sequence
                    grid = [0] * 81
                    a_s = ans_starts[i].item()
                    for j in range(81):
                        tok_id = x[i, a_s + j].item()
                        grid[j] = digit_ids.get(tok_id, 0)
                    live = compute_live_candidates(grid)
                    for j in range(81):
                        if still_m[i, j].item() and j in live:
                            priority[i, j] = live[j]
                best_j = priority.argmin(dim=1)

            else:
                # Static order policies
                order = static_order.clone()
                order[~still_m] = 9999
                best_j = order.argmin(dim=1)

            best_pos = ap[torch.arange(B, device=device), best_j]
            best_probs = probs[torch.arange(B, device=device), best_j]
            pred_toks = best_probs.argmax(dim=1)
            has = still_m.any(dim=1)
            for i in range(B):
                if has[i]:
                    x[i, best_pos[i]] = pred_toks[i]
                    decode_orders[i, step] = best_pos[i]

        # Collect results
        pred_ids = x[bi, ap]
        for i in range(B):
            ps = tokenizer.decode(pred_ids[i].cpu().tolist())
            gs = batch[i]['string'].split('=')[1]
            pc = [ps[j] == gs[j] if j < len(ps) else False for j in range(len(gs))]
            results.append({'correct': ps==gs, 'pos_correct': pc,
                           'meta': batch[i]['meta'], 'n_blanks': batch[i]['n_blanks'],
                           'rating': batch[i].get('rating', 0)})
    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Evaluation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def evaluate(model, tokenizer, test_data, decode_policy='confidence',
             batch_size=32, device=None):
    """Rating-stratified generation evaluation."""
    results = generate_blanks(model, tokenizer, test_data, decode_policy, batch_size, device)
    n = len(results)
    acc = sum(r['correct'] for r in results) / max(n, 1)

    # Blank cell accuracy
    bl_correct, bl_total = 0, 0
    for r in results:
        for j in range(81):
            if not r['meta'][j]['is_given']:
                bl_total += 1; bl_correct += r['pos_correct'][j]
    bl_acc = bl_correct / max(bl_total, 1)

    # Per depth category
    cat_acc = defaultdict(list)
    for r in results:
        for j in range(81):
            cat = _depth_to_cat(r['meta'][j])
            if cat != 'given': cat_acc[cat].append(r['pos_correct'][j])
    cat_acc = {c: sum(v)/len(v) for c, v in cat_acc.items() if v}

    # Per guess_depth category
    guess_acc = defaultdict(list)
    for r in results:
        for j in range(81):
            if not r['meta'][j]['is_given']:
                gc = _guess_to_cat(r['meta'][j])
                guess_acc[gc].append(r['pos_correct'][j])
    guess_acc = {c: sum(v)/len(v) for c, v in guess_acc.items() if v}

    # Per rating tier
    rating_acc = {}
    for tn, (lo, hi) in RATING_TIERS.items():
        tier_r = [r for r in results if lo <= r.get('rating', 0) <= hi]
        if tier_r:
            rating_acc[tn] = {
                'exact': sum(r['correct'] for r in tier_r) / len(tier_r),
                'cell': sum(sum(r['pos_correct'][j] for j in range(81)
                               if not r['meta'][j]['is_given'])
                           for r in tier_r) / max(sum(r['n_blanks'] for r in tier_r), 1),
                'n': len(tier_r),
                'mean_blanks': sum(r['n_blanks'] for r in tier_r) / len(tier_r),
            }

    # Per n_blanks bin
    blanks_acc = {}
    blank_bins = [(20, 35), (36, 45), (46, 55), (56, 65)]
    for lo_b, hi_b in blank_bins:
        bin_r = [r for r in results if lo_b <= r['n_blanks'] <= hi_b]
        if bin_r:
            blanks_acc[f'{lo_b}-{hi_b}'] = {
                'exact': sum(r['correct'] for r in bin_r) / len(bin_r),
                'cell': sum(sum(r['pos_correct'][j] for j in range(81)
                               if not r['meta'][j]['is_given'])
                           for r in bin_r) / max(sum(r['n_blanks'] for r in bin_r), 1),
                'n': len(bin_r),
            }

    # Per technique level
    tech_acc = defaultdict(list)
    for r in results:
        for j in range(81):
            if not r['meta'][j]['is_given']:
                tl_cat = _tl_to_cat(r['meta'][j])
                tech_acc[tl_cat].append(r['pos_correct'][j])
    tech_acc = {c: sum(v)/len(v) for c, v in tech_acc.items() if v}

    return {'accuracy': acc, 'blank_cell_acc': bl_acc, 'n': n,
            'category_accuracy': cat_acc, 'guess_accuracy': guess_acc,
            'technique_accuracy': tech_acc,
            'rating_accuracy': rating_acc, 'blanks_accuracy': blanks_acc}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Analysis
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def analyse_rating_correlation(data):
    """Correlate our depth/n_cands metrics with external tdoku rating.
    Also analyzes by source (puzzle origin) for SE-rating-equivalent grouping."""
    features, ratings, sources = [], [], []
    for d in data:
        r = d.get('rating', 0)
        src = d.get('source', '')
        meta = d['meta']
        n_hard = sum(1 for j in range(81) if meta[j]['prop_depth'] == -1)
        blanks = [j for j in range(81) if not meta[j]['is_given']]
        if not blanks: continue
        n_cands_cp = [meta[j]['n_cands_after_cp'] for j in blanks if meta[j]['n_cands_after_cp'] > 0]
        max_depth = max((meta[j]['prop_depth'] for j in blanks if meta[j]['prop_depth'] >= 0), default=0)
        guess_depths = [meta[j].get('guess_depth', 0) for j in blanks]
        max_guess = max(guess_depths) if guess_depths else 0
        mean_guess = sum(guess_depths)/len(guess_depths) if guess_depths else 0
        n_guess_hard = sum(1 for gd in guess_depths if gd >= 2)
        features.append({
            'n_depth_hard': n_hard,
            'n_blanks': len(blanks),
            'mean_n_cands_cp': sum(n_cands_cp)/len(n_cands_cp) if n_cands_cp else 0,
            'max_depth': max_depth,
            'frac_hard': n_hard / max(len(blanks), 1),
            'max_guess_depth': max_guess,
            'mean_guess_depth': mean_guess,
            'n_guess_hard': n_guess_hard,
        })
        ratings.append(r)
        sources.append(src)

    if len(ratings) < 10: return {}

    def _corr(xs, ys):
        n = len(xs); mx, my = sum(xs)/n, sum(ys)/n
        c = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
        sx = sum((x-mx)**2 for x in xs)**0.5
        sy = sum((y-my)**2 for y in ys)**0.5
        return c/(sx*sy) if sx > 0 and sy > 0 else 0.0

    correlations = {}
    for feat_name in ['n_depth_hard', 'n_blanks', 'mean_n_cands_cp', 'max_depth', 'frac_hard',
                       'max_guess_depth', 'mean_guess_depth', 'n_guess_hard']:
        xs = [f[feat_name] for f in features]
        correlations[feat_name] = _corr(xs, ratings)

    # Rating distribution by tdoku backtrack ranges → conceptual SE mapping
    # tdoku rating=0 ↔ CP-solvable ↔ SE ~1-4 (singles only)
    # tdoku rating=1-9 ↔ light search ↔ SE ~4-7 (pairs, triples)
    # tdoku rating=10-99 ↔ moderate search ↔ SE ~7-10 (X-wing, coloring)
    # tdoku rating=100+ ↔ deep search ↔ SE ~10+ (forcing chains, unique rect)
    se_mapping = {
        'rating=0 (CP-solvable, ~SE 1-4)': sum(1 for r in ratings if r == 0),
        'rating=1-9 (light search, ~SE 4-7)': sum(1 for r in ratings if 1 <= r <= 9),
        'rating=10-99 (mod search, ~SE 7-10)': sum(1 for r in ratings if 10 <= r <= 99),
        'rating=100+ (deep search, ~SE 10+)': sum(1 for r in ratings if r >= 100),
    }

    # Source-based analysis: map puzzle source to difficulty profile
    source_stats = defaultdict(lambda: {'ratings': [], 'n_hard_cells': [], 'n': 0})
    for i in range(len(ratings)):
        src = sources[i] if sources[i] else 'unknown'
        # Simplify source name (remove path prefixes)
        src_short = src.split('/')[-1] if '/' in src else src
        source_stats[src_short]['ratings'].append(ratings[i])
        source_stats[src_short]['n_hard_cells'].append(features[i]['n_depth_hard'])
        source_stats[src_short]['n'] += 1
    source_summary = {}
    for src, stats in source_stats.items():
        rs = stats['ratings']
        source_summary[src] = {
            'n': stats['n'],
            'mean_rating': sum(rs)/len(rs),
            'median_rating': sorted(rs)[len(rs)//2],
            'mean_n_hard_cells': sum(stats['n_hard_cells'])/len(stats['n_hard_cells']),
            'frac_cp_solvable': sum(1 for r in rs if r == 0) / len(rs),
        }

    return {'correlations': correlations, 'se_mapping': se_mapping,
            'source_summary': source_summary,
            'features': features, 'ratings': ratings}


@torch.no_grad()
def probe_selective_masking(model, tokenizer, test_data, max_len, device=None):
    """Technique-level-aware selective masking probe.

    Conditions:
      all_blank:       all blank cells masked (standard)
      tl4_only:        only tl_4 cells masked, others get ground truth
                       (= stepping stone benefit removed for the hardest cells)
      tl4_no_stepping: all blank cells masked, but only MEASURE tl_4
                       (= how well model does on tl_4 without stepping stone info)
      tl01_only:       only tl_0/1 cells masked (easy cells only)
    """
    if device is None: device = DEVICE
    model.eval()
    mask_id = tokenizer.special_ids['mask']
    strings = [d['string'] for d in test_data]
    metas = [d['meta'] for d in test_data]
    ids_all, ans_all = encode_samples(strings, tokenizer, max_len)
    ids_all, ans_all = ids_all.to(device), ans_all.to(device)
    N = len(test_data); _ar = torch.arange(ANS_LEN, device=device)

    conditions = {
        'all_blank':       lambda m: not m['is_given'],
        'tl4_only':        lambda m: m.get('technique_level', 0) == 4,
        'tl4_no_stepping': lambda m: not m['is_given'],  # mask all, measure tl4
        'tl01_only':       lambda m: not m['is_given'] and m.get('technique_level', 0) <= 1,
    }
    results = {}
    for cn, fn in conditions.items():
        tl_correct = defaultdict(list)
        for st in range(0, N, 64):
            en = min(st+64, N); B = en - st
            ids, ans = ids_all[st:en], ans_all[st:en]
            T = ids.shape[1]
            ap = (ans.unsqueeze(1) + _ar).clamp(max=T-1)
            bi = torch.arange(B, device=device).unsqueeze(1).expand_as(ap)
            mc = torch.zeros(B, ANS_LEN, dtype=torch.bool, device=device)
            for b in range(B):
                for j in range(ANS_LEN):
                    if fn(metas[st+b][j]): mc[b, j] = True
            xm = ids.clone(); xm[bi[mc], ap[mc]] = mask_id
            logits = model(xm); al = logits[bi, ap]
            cl = al.clone(); cl[:, :, mask_id] = -float('inf')
            preds = cl.argmax(dim=-1); tgt = ids[bi, ap]
            corrects = (preds == tgt)
            for b in range(B):
                for j in range(ANS_LEN):
                    meta_j = metas[st+b][j]
                    if meta_j['is_given']: continue
                    tl_cat = _tl_to_cat(meta_j)
                    if cn == 'tl4_no_stepping':
                        # Mask everything, but only record tl_4 accuracy
                        if meta_j.get('technique_level', 0) == 4:
                            tl_correct[tl_cat].append(corrects[b,j].item())
                    elif mc[b, j]:
                        tl_correct[tl_cat].append(corrects[b,j].item())
        results[cn] = {c: sum(v)/len(v) for c, v in tl_correct.items() if v}
    return results


def filter_by_tl4_frac(test_data, min_frac):
    """Filter puzzles by fraction of blank cells that are tl_4 (search-required).
    Sudoku analog of addition's chain length sweep.
    min_frac=0.5 → some stepping stones remain
    min_frac=0.9 → almost no stepping stones
    min_frac=1.0 → ALL blanks require search (= addition full_propagate)
    """
    results = []
    for d in test_data:
        meta = d['meta']
        blanks = [j for j in range(81) if not meta[j]['is_given']]
        if not blanks: continue
        n_tl4 = sum(1 for j in blanks if meta[j].get('technique_level', 0) == 4)
        frac = n_tl4 / len(blanks)
        if frac >= min_frac:
            d_copy = dict(d); d_copy['tl4_frac'] = frac
            results.append(d_copy)
    return results


def filter_by_hard_frac(test_data, min_hard_frac):
    """Filter puzzles by fraction of blank cells that are depth_hard (legacy)."""
    results = []
    for d in test_data:
        meta = d['meta']
        blanks = [j for j in range(81) if not meta[j]['is_given']]
        if not blanks: continue
        n_hard = sum(1 for j in blanks if meta[j]['prop_depth'] == -1)
        frac = n_hard / len(blanks)
        if frac >= min_hard_frac:
            d_copy = dict(d); d_copy['hard_frac'] = frac; results.append(d_copy)
    return results


@torch.no_grad()
def analyse_decode_strategy(model, tokenizer, test_data, max_len,
                            n_samples=200, device=None):
    """Uncover the model's solving strategy by correlating confidence decode rank
    with multiple cell features.

    Features per blank cell:
      prop_depth:       CP propagation depth (-1→99 for depth_hard)
      n_candidates:     initial candidate count
      n_cands_cp:       candidate count after CP
      n_given_peers:    how many of 20 peers are given (constraint density)
      box_occupancy:    given cells in same box
      row_occupancy:    given cells in same row
      col_occupancy:    given cells in same column
      min_unit_vacancy: min(empty in row, col, box) — closest to unit completion

    If |ρ(n_given_peers, rank)| > |ρ(prop_depth, rank)| → model uses global constraint
    density rather than sequential propagation. (Sudoku carry-lookahead analog.)
    """
    if device is None: device = DEVICE
    model.eval()
    mask_id = tokenizer.special_ids['mask']
    samples = test_data[:n_samples]

    all_ranks = []
    all_feats = defaultdict(list)

    for si, d in enumerate(samples):
        s = d['string']; meta = d['meta']
        puzzle_str = s.split('=')[0]
        puzzle_flat = [int(c) if c.isdigit() else 0 for c in puzzle_str]

        enc = tokenizer.encode(s)
        ids = torch.tensor(enc, dtype=torch.long, device=device).unsqueeze(0)
        T = ids.shape[1]
        eq_id = tokenizer.encode('=')[0]
        ans_s = 0
        for t in range(T):
            if ids[0, t].item() == eq_id: ans_s = t + 1; break

        x = ids.clone()
        blank_js = []
        for j in range(ANS_LEN):
            if puzzle_flat[j] == 0:
                x[0, ans_s + j] = mask_id; blank_js.append(j)
        if not blank_js: continue

        logits = model(x)
        confs = torch.zeros(ANS_LEN, device=device)
        for j in blank_js:
            cl = logits[0, ans_s + j].clone(); cl[mask_id] = -float('inf')
            confs[j] = F.softmax(cl, dim=-1).max()

        blank_confs = sorted([(j, confs[j].item()) for j in blank_js], key=lambda x: -x[1])
        rank_of = {j: rank for rank, (j, _) in enumerate(blank_confs)}

        for j in blank_js:
            r, c = j // 9, j % 9
            box_r, box_c = (r // 3) * 3, (c // 3) * 3
            given_peers = sum(1 for p in PEERS_FLAT[j] if puzzle_flat[p] != 0)
            row_occ = sum(1 for cc in range(9) if puzzle_flat[r*9+cc] != 0)
            col_occ = sum(1 for rr in range(9) if puzzle_flat[rr*9+c] != 0)
            box_occ = sum(1 for dr in range(3) for dc in range(3)
                         if puzzle_flat[(box_r+dr)*9+(box_c+dc)] != 0)
            row_vac = 9 - row_occ; col_vac = 9 - col_occ
            box_vac = sum(1 for dr in range(3) for dc in range(3)
                         if puzzle_flat[(box_r+dr)*9+(box_c+dc)] == 0)
            min_vac = min(row_vac, col_vac, box_vac)

            pd = meta[j]['prop_depth']
            nc = meta[j]['n_candidates']
            nc_cp = meta[j].get('n_cands_after_cp', nc) or nc

            all_ranks.append(rank_of[j])
            all_feats['prop_depth'].append(pd if pd >= 0 else 99)
            all_feats['n_candidates'].append(nc)
            all_feats['n_cands_cp'].append(nc_cp)
            all_feats['n_given_peers'].append(given_peers)
            all_feats['box_occupancy'].append(box_occ)
            all_feats['row_occupancy'].append(row_occ)
            all_feats['col_occupancy'].append(col_occ)
            all_feats['min_unit_vacancy'].append(min_vac)
            all_feats['guess_depth'].append(meta[j].get('guess_depth', 0))
            all_feats['technique_level'].append(meta[j].get('technique_level', 4))

    if len(all_ranks) < 20: return {}

    def _rank(xs):
        sx = sorted(range(len(xs)), key=lambda i: xs[i])
        ranks = [0.0] * len(xs)
        for r, i in enumerate(sx): ranks[i] = r
        return ranks

    def _spearman(xs, ys):
        n = len(xs); rx, ry = _rank(xs), _rank(ys)
        mx, my = sum(rx)/n, sum(ry)/n
        c = sum((a-mx)*(b-my) for a, b in zip(rx, ry))
        sx = sum((a-mx)**2 for a in rx)**0.5
        sy = sum((b-my)**2 for b in ry)**0.5
        return c/(sx*sy) if sx > 0 and sy > 0 else 0.0

    correlations = {}
    for feat_name, feat_vals in all_feats.items():
        correlations[feat_name] = _spearman(feat_vals, all_ranks)

    sorted_feats = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    binned = {}
    for feat_name in ['prop_depth', 'n_given_peers', 'n_candidates', 'min_unit_vacancy', 'guess_depth']:
        bins = defaultdict(list)
        for i in range(len(all_ranks)):
            bins[all_feats[feat_name][i]].append(all_ranks[i])
        binned[feat_name] = {k: sum(v)/len(v) for k, v in sorted(bins.items()) if len(v) >= 5}

    return {'correlations': correlations, 'sorted_features': sorted_feats,
            'binned': binned, 'n_cells': len(all_ranks)}


@torch.no_grad()
def simulate_puma_coverage(model, tokenizer, test_data, max_len,
                           n_samples=100, device=None):
    """Measure PUMA chain coverage by depth category."""
    if device is None: device = DEVICE
    model.eval()
    mask_id = tokenizer.special_ids['mask']
    n_cats = len(DEPTH_CATS)
    cov_sum = torch.zeros(n_cats); cov_n = torch.zeros(n_cats, dtype=torch.long)

    for si in range(min(len(test_data), n_samples)):
        d = test_data[si]; s = d['string']; meta = d['meta']
        ps, sol = s.split('='); penc = tokenizer.encode(ps + '=')
        T_pre = len(penc)
        x = torch.tensor(penc + [mask_id]*ANS_LEN, dtype=torch.long, device=device).unsqueeze(0)
        x0 = torch.tensor(tokenizer.encode(sol)[:ANS_LEN], dtype=torch.long, device=device)
        is_m = torch.zeros(ANS_LEN, dtype=torch.bool, device=device)
        for j in range(ANS_LEN):
            if ps[j] == '0': is_m[j] = True
            else: x[0, T_pre+j] = x0[j]  # fill givens
        steps_m = torch.zeros(ANS_LEN); total = 0

        for step in range(PUMA_K):
            if not is_m.any(): break
            total += 1; steps_m += is_m.cpu().float()
            logits = model(x)
            nm = is_m.sum().item()
            nr = max(1, int(math.ceil(nm / max(PUMA_K - step, 1))))
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
                if reveal.sum() < nr or confs[j] > PUMA_TAU: reveal[j] = True
            for j in range(ANS_LEN):
                if reveal[j]: x[0, T_pre+j] = x0[j]; is_m[j] = False

        if total == 0: continue
        frac = steps_m / total
        for j in range(ANS_LEN):
            cat_id = DEPTH_CAT_TO_ID[_depth_to_cat(meta[j])]
            cov_sum[cat_id] += frac[j]; cov_n[cat_id] += 1

    result = {}
    for ci, cn in enumerate(DEPTH_CATS):
        if cov_n[ci] > 0:
            result[cn] = {'coverage': cov_sum[ci].item()/cov_n[ci].item(), 'n': cov_n[ci].item()}
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Figures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DEPTH_COLORS = {'given': '#95a5a6', 'depth_0': '#2ecc71', 'depth_1': '#3498db',
                'depth_2': '#e67e22', 'depth_3plus': '#e74c3c', 'depth_hard': '#8e44ad'}

def make_figures(all_results, all_dyn, corr_data):
    figs = {}

    # ── Fig 1: Accuracy by rating tier × decode policy × mask type ──
    tiers = [t for t in RATING_TIERS if any(
        t in all_results.get(f'{mt}_{dp}', {}).get('rating_accuracy', {})
        for mt in MASK_TYPES for dp in DECODE_POLICIES)]
    if tiers:
        fig, axes = plt.subplots(1, len(tiers), figsize=(6*len(tiers), 5), squeeze=False)
        axes = axes[0]
        for ti, tier in enumerate(tiers):
            ax = axes[ti]
            labels, vals = [], []
            for mt in MASK_TYPES:
                for dp in DECODE_POLICIES:
                    key = f'{mt}_{dp}'
                    ra = all_results.get(key, {}).get('rating_accuracy', {}).get(tier)
                    if ra:
                        labels.append(f'{mt[:3]}+{dp[:4]}')
                        vals.append(ra['exact'])
            if vals:
                colors = ['#3498db' if 'ran' in l else '#8e44ad' for l in labels]
                ax.bar(range(len(vals)), vals, color=colors, alpha=0.8)
                ax.set_xticks(range(len(vals))); ax.set_xticklabels(labels, fontsize=7, rotation=30)
                ax.set_ylim(0, 1.05); ax.set_ylabel('Exact Match Accuracy')
                ax.set_title(f'{tier} (rating {RATING_TIERS[tier][0]}-{RATING_TIERS[tier][1]})')
                ax.grid(alpha=0.3, axis='y')
        fig.suptitle('Accuracy by Rating Tier × Decode Policy', fontsize=12, y=1.02)
        fig.tight_layout(); figs['acc_by_tier'] = fig

    # ── Fig 2: Per-category accuracy comparison ──
    fig, axes = plt.subplots(1, len(MASK_TYPES), figsize=(7*len(MASK_TYPES), 5), squeeze=False)
    axes = axes[0]
    for mi, mt in enumerate(MASK_TYPES):
        ax = axes[mi]
        for dp in DECODE_POLICIES:
            key = f'{mt}_{dp}'
            ca = all_results.get(key, {}).get('category_accuracy', {})
            cats = [c for c in DEPTH_CATS if c in ca]
            if cats:
                vals = [ca[c] for c in cats]
                x = range(len(cats))
                ax.plot(x, vals, '-o', label=dp, ms=5, lw=1.5, alpha=0.8)
                ax.set_xticks(list(x)); ax.set_xticklabels(cats, fontsize=7, rotation=30)
        ax.set_ylim(0, 1.05); ax.set_ylabel('Cell Accuracy'); ax.set_title(f'mask={mt}')
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.suptitle('Per-Category Accuracy by Decode Policy', fontsize=12, y=1.02)
    fig.tight_layout(); figs['cat_acc'] = fig

    # ── Fig 2b: Per-guess-depth accuracy comparison (PUMA vs Random) ──
    has_guess = any(all_results.get(f'{mt}_confidence', {}).get('guess_accuracy')
                    for mt in MASK_TYPES)
    if has_guess:
        fig, ax = plt.subplots(figsize=(10, 5))
        mt_colors = {'random': '#3498db', 'puma': '#8e44ad'}
        for mt in MASK_TYPES:
            ga = all_results.get(f'{mt}_confidence', {}).get('guess_accuracy', {})
            cats = [c for c in GUESS_CATS if c in ga]
            if cats:
                vals = [ga[c] for c in cats]; x = range(len(cats))
                ax.plot(x, vals, '-o', label=mt, color=mt_colors.get(mt, '#333'),
                        ms=8, lw=2, alpha=0.8)
                ax.set_xticks(list(x)); ax.set_xticklabels(cats, fontsize=9)
        ax.set_ylim(0, 1.05); ax.set_ylabel('Cell Accuracy')
        ax.set_title('Accuracy by Guess Depth (confidence decode)')
        ax.legend(fontsize=10); ax.grid(alpha=0.3)
        fig.tight_layout(); figs['guess_depth_acc'] = fig

    # ── Fig 3: Rating correlation scatter ──
    if corr_data and 'features' in corr_data:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        feat_names = ['n_depth_hard', 'frac_hard', 'mean_n_cands_cp']
        for ai, fn in enumerate(feat_names):
            ax = axes[ai]
            xs = [f[fn] for f in corr_data['features']]
            ys = corr_data['ratings']
            ax.scatter(xs, ys, s=5, alpha=0.3, c='#8e44ad')
            r = corr_data['correlations'].get(fn, 0)
            ax.set_xlabel(fn); ax.set_ylabel('tdoku rating (backtracks)')
            ax.set_title(f'r = {r:.3f}'); ax.grid(alpha=0.3)
        fig.suptitle('Our Metrics vs External Difficulty Rating', fontsize=12, y=1.02)
        fig.tight_layout(); figs['rating_corr'] = fig

    # ── Fig 4: Training loss curves ──
    if all_dyn:
        fig, ax = plt.subplots(figsize=(10, 5))
        for mt, dyn in all_dyn.items():
            tl = dyn['train_loss']
            if tl: ax.plot([x[0] for x in tl], [x[1] for x in tl], '-', label=mt, alpha=0.8)
        ax.set_xlabel('Iteration'); ax.set_ylabel('Loss')
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout(); figs['train_loss'] = fig

    # ── Fig 5: Depth category accuracy over training ──
    if all_dyn:
        fig, axes = plt.subplots(1, len(all_dyn), figsize=(7*len(all_dyn), 5), squeeze=False)
        axes = axes[0]
        for ai, (mt, dyn) in enumerate(all_dyn.items()):
            ax = axes[ai]; cps = dyn['checkpoints']
            xs = [c['iter'] for c in cps]
            for cat in DEPTH_CATS:
                ys = [c.get('depth_context', {}).get(cat, {}).get('mean_acc', float('nan'))
                      for c in cps]
                if any(not math.isnan(y) for y in ys):
                    ax.plot(xs, ys, '-', color=DEPTH_COLORS.get(cat, '#333'), label=cat, lw=1.5)
            ax.set_xlabel('Iteration'); ax.set_ylabel('Accuracy')
            ax.set_ylim(-0.05, 1.05); ax.set_title(mt); ax.legend(fontsize=7); ax.grid(alpha=0.3)
        fig.suptitle('Accuracy by Depth Category Over Training', fontsize=12, y=1.02)
        fig.tight_layout(); figs['depth_acc_evo'] = fig

    # ── Fig 6: PUMA coverage by depth category ──
    cov_data = {mt: all_results.get(f'{mt}_coverage', {})
                for mt in MASK_TYPES if f'{mt}_coverage' in all_results}
    if cov_data:
        fig, ax = plt.subplots(figsize=(10, 5))
        cats = [c for c in DEPTH_CATS if c != 'given']
        for mt, cov in cov_data.items():
            vals = [cov.get(c, {}).get('coverage', 0) for c in cats]
            ax.plot(range(len(cats)), vals, '-o', label=f'{mt} PUMA coverage', ms=6, lw=2)
        ax.axhline(0.5, color='gray', ls='--', alpha=0.5, label='random baseline (0.5)')
        ax.set_xticks(range(len(cats))); ax.set_xticklabels(cats, fontsize=8)
        ax.set_ylabel('Coverage (frac steps masked)')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        fig.suptitle('PUMA Coverage by Depth Category', fontsize=12, y=1.02)
        fig.tight_layout(); figs['coverage'] = fig

    # ── Fig 7: Selective masking comparison ──
    sel_data = {mt: all_results.get(f'{mt}_selective', {})
                for mt in MASK_TYPES if f'{mt}_selective' in all_results}
    if sel_data:
        fig, axes = plt.subplots(1, len(sel_data), figsize=(7*len(sel_data), 5), squeeze=False)
        axes = axes[0]
        for ai, (mt, sel) in enumerate(sel_data.items()):
            ax = axes[ai]
            for cn in ['all_blank', 'hard_only', 'easy_only']:
                if cn not in sel: continue
                cats = [c for c in DEPTH_CATS if c in sel[cn]]
                vals = [sel[cn][c] for c in cats]
                ax.plot(range(len(cats)), vals, '-o', label=cn, ms=5, lw=1.5)
            ax.set_xticks(range(len(cats))); ax.set_xticklabels(cats, fontsize=7, rotation=20)
            ax.set_ylim(0, 1.05); ax.set_ylabel('Cell Accuracy'); ax.set_title(f'mask={mt}')
            ax.legend(fontsize=7); ax.grid(alpha=0.3)
        fig.suptitle('Selective Masking Probe (Parallel vs Sequential Reasoning)', fontsize=12, y=1.02)
        fig.tight_layout(); figs['selective'] = fig

    # ── Fig 8: Hard fraction sweep (generalization boundary) ──
    hard_fracs = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0]
    sweep_data = {}
    for mt in MASK_TYPES:
        for dp in ['confidence', 'oracle_solver']:
            xs, ys = [], []
            for hf in hard_fracs:
                r = all_results.get(f'{mt}_hardfrac_{hf:.2f}_{dp}', {})
                if r and 'blank_cell_acc' in r:
                    xs.append(hf); ys.append(r['blank_cell_acc'])
            if xs: sweep_data[f'{mt}_{dp}'] = (xs, ys)
    if sweep_data:
        fig, ax = plt.subplots(figsize=(10, 6))
        styles = {'puma_confidence': ('#8e44ad', 's', '-'),
                  'puma_oracle_solver': ('#8e44ad', 's', '--'),
                  'random_confidence': ('#3498db', 'o', '-'),
                  'random_oracle_solver': ('#3498db', 'o', '--')}
        for key, (xs, ys) in sweep_data.items():
            col, mk, ls = styles.get(key, ('#333', 'x', '-'))
            ax.plot(xs, ys, f'{ls}', color=col, marker=mk, label=key, ms=8, lw=2, alpha=0.8)
        ax.set_xlabel('Min hard fraction (depth_hard / n_blanks)')
        ax.set_ylabel('Blank Cell Accuracy')
        ax.set_title('Hard Fraction Sweep — PUMA Generalization Boundary')
        ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_ylim(0, 1.05)
        fig.tight_layout(); figs['hard_frac_sweep'] = fig

    return figs


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Run
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run(tag=''):
    exp_name = f"{EXP_NAME}_{tag}" if tag else EXP_NAME
    print(f"\n{'='*70}")
    print(f"  {exp_name}")
    decay_str = f"decay={DIFFICULTY_DECAY}" if DIFFICULTY_DECAY else "balanced (all data)"
    print(f"  Model: {N_LAYER}L/{N_EMBD}D/{N_HEAD}H  Data: {decay_str}")
    print(f"  Training: {MAX_ITERS} iters, batch={BATCH_SIZE}")
    print(f"  Masks: {MASK_TYPES}  Decode: {DECODE_POLICIES}")
    print(f"  PUMA: K={PUMA_K}, tau={PUMA_TAU}")
    print(f"{'='*70}")

    mount_drive()
    torch.manual_seed(SEED); random.seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed(SEED)

    tok = build_tok()
    train_data, test_data = prepare_datasets(seed=SEED)
    max_len = max(len(tok.encode(d['string'])) for d in train_data)
    print(f"  max_len={max_len}")

    # ── Rating correlation analysis (test data only — has metadata) ──
    print(f"\n{'━'*60}\n  Rating Correlation Analysis\n{'━'*60}")
    all_test_for_corr = [d for ds in test_data.values() for d in ds]
    corr = analyse_rating_correlation(all_test_for_corr)
    if corr.get('correlations'):
        for fn, r in corr['correlations'].items():
            print(f"    corr({fn}, tdoku_rating) = {r:.3f}")
    if corr.get('se_mapping'):
        for desc, n in corr['se_mapping'].items():
            print(f"    {desc}: n={n}")
    if corr.get('source_summary'):
        print(f"  Source-based difficulty profile:")
        for src, info in sorted(corr['source_summary'].items(),
                                 key=lambda x: x[1]['mean_rating']):
            print(f"    {src:<35s}: n={info['n']:>6d} "
                  f"mean_rating={info['mean_rating']:>6.1f} "
                  f"mean_hard_cells={info['mean_n_hard_cells']:>4.1f} "
                  f"cp_solvable={info['frac_cp_solvable']:.0%}")

    # ── Merge all test tiers for eval ──
    # Keep separate by tier for rating-stratified eval
    all_test = [d for ds in test_data.values() for d in ds]

    all_results = {}
    all_dyn = {}
    saved_states = {}

    for mt in MASK_TYPES:
        print(f"\n{'━'*60}\n▶ Training: {mt}\n{'━'*60}")
        model, dyn = train(mt, tok, train_data, test_data, max_len, device=DEVICE)
        all_dyn[mt] = dyn
        saved_states[mt] = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        save_checkpoint(exp_name, saved_states[mt], tag=mt)

        # ── Multi-policy eval ──
        for dp in DECODE_POLICIES:
            key = f'{mt}_{dp}'
            print(f"\n  Eval: {key}")
            r = evaluate(model, tok, all_test, decode_policy=dp, batch_size=32, device=DEVICE)
            all_results[key] = r
            print(f"    acc={r['accuracy']:.4f}  blank_cell={r['blank_cell_acc']:.4f}")
            # Technique accuracy (primary)
            ta = r.get('technique_accuracy', {})
            if ta:
                parts = [f"{c}={v:.3f}" for c, v in sorted(ta.items())]
                print(f"    Tech:  {' '.join(parts)}")
            # Rating tiers (secondary)
            ra = r.get('rating_accuracy', {})
            for tn, info in ra.items():
                print(f"    {tn}: exact={info['exact']:.4f} cell={info['cell']:.4f} "
                      f"(n={info['n']}, blanks={info.get('mean_blanks', '?'):.1f})")

        # ── Selective masking (technique-aware) ──
        print(f"  Selective masking probe (technique-aware)...")
        sel = probe_selective_masking(model, tok, all_test[:200], max_len, device=DEVICE)
        all_results[f'{mt}_selective'] = sel
        for cn, accs in sel.items():
            parts = [f"{c}={v:.3f}" for c, v in sorted(accs.items())]
            print(f"    {cn}: {' '.join(parts)}")

        # ── TL4 fraction sweep (technique-level analog of chain length sweep) ──
        print(f"  TL4 fraction sweep...")
        tl4_fracs = [0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0]
        for tf in tl4_fracs:
            subset = filter_by_tl4_frac(all_test, tf)
            if len(subset) < 10:
                print(f"    tl4_frac>={tf:.0%}: {len(subset)} puzzles (skipped)")
                continue
            subset_eval = subset[:min(200, len(subset))]
            r = evaluate(model, tok, subset_eval, decode_policy='confidence',
                        batch_size=16, device=DEVICE)
            key = f'{mt}_tl4frac_{tf:.2f}_confidence'
            all_results[key] = r
            ta = r.get('technique_accuracy', {})
            tl4_s = f"tl4={ta.get('tl_4_search', 0):.3f}" if 'tl_4_search' in ta else ""
            print(f"    tl4_frac>={tf:.0%}: cell={r['blank_cell_acc']:.4f} "
                  f"exact={r['accuracy']:.4f} {tl4_s} "
                  f"(n={len(subset_eval)}, available={len(subset)})")

        # ── Legacy hard fraction sweep (for backward compat) ──
        print(f"  Hard fraction sweep (legacy)...")
        for hf in [0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0]:
            subset = filter_by_hard_frac(all_test, hf)
            if len(subset) < 10: continue
            subset_eval = subset[:min(200, len(subset))]
            r = evaluate(model, tok, subset_eval, decode_policy='confidence',
                        batch_size=16, device=DEVICE)
            all_results[f'{mt}_hardfrac_{hf:.2f}_confidence'] = r
            print(f"    hf>={hf:.0%}: cell={r['blank_cell_acc']:.4f} "
                  f"(n={len(subset_eval)}, available={len(subset)})")

        # ── Decode strategy analysis (what does the model actually look at?) ──
        print(f"  Decode strategy analysis...")
        strat = analyse_decode_strategy(model, tok, all_test[:200], max_len, device=DEVICE)
        all_results[f'{mt}_decode_strategy'] = strat
        if strat.get('sorted_features'):
            print(f"    Feature importance (|ρ| with decode rank):")
            for feat, rho in strat['sorted_features']:
                print(f"      {feat:<20s}: ρ={rho:+.3f}")
            # Key comparison
            rho_depth = strat['correlations'].get('prop_depth', 0)
            rho_peers = strat['correlations'].get('n_given_peers', 0)
            rho_ncands = strat['correlations'].get('n_candidates', 0)
            rho_minvac = strat['correlations'].get('min_unit_vacancy', 0)
            print(f"    → Model strategy: ", end='')
            best = strat['sorted_features'][0]
            if best[0] in ('n_given_peers', 'box_occupancy', 'row_occupancy', 'col_occupancy'):
                print(f"CONSTRAINT DENSITY (ρ={best[1]:+.3f}) > prop_depth (ρ={rho_depth:+.3f})")
            elif best[0] == 'min_unit_vacancy':
                print(f"UNIT COMPLETION (ρ={best[1]:+.3f}) > prop_depth (ρ={rho_depth:+.3f})")
            elif best[0] == 'n_candidates':
                print(f"CANDIDATE COUNT (ρ={best[1]:+.3f}) > prop_depth (ρ={rho_depth:+.3f})")
            else:
                print(f"CP-LIKE (prop_depth ρ={rho_depth:+.3f})")

        # ── PUMA coverage (puma model only) ──
        if mt == 'puma':
            print(f"  PUMA coverage simulation...")
            hard_test = test_data.get('hard', test_data.get('all', all_test))[:100]
            cov = simulate_puma_coverage(model, tok, hard_test, max_len, device=DEVICE)
            all_results[f'{mt}_coverage'] = cov
            for cn, info in cov.items():
                print(f"    {cn}: coverage={info['coverage']:.3f} (n={info['n']})")

        # ── Probe vs Generation gap (technique-level) ──
        print(f"  Probe vs Generation gap...")
        last_probe = dyn['checkpoints'][-1] if dyn['checkpoints'] else {}
        gen_conf = all_results.get(f'{mt}_confidence', {})
        probe_dc = last_probe.get('depth_context', {})
        gen_ta = gen_conf.get('technique_accuracy', {})
        if gen_ta:
            print(f"    {'TechLevel':<18s} {'gen_cell':>10s}")
            for tl in TECHNIQUE_LEVELS:
                g_acc = gen_ta.get(tl)
                if g_acc is not None:
                    print(f"    {tl:<18s} {g_acc:>10.4f}")

        del model; torch.cuda.empty_cache() if torch.cuda.is_available() else None

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
            model, d = train(tgt, tok, train_data, test_data, max_len,
                             max_iters=CONTINUATION_ITERS,
                             init_state=saved_states[src], device=DEVICE)
            all_dyn[label] = d

            # Evaluate continuation with key decode policies
            for dp in ['confidence', 'oracle_technique']:
                if dp not in DECODE_POLICIES and dp != 'confidence':
                    continue
                key = f'{label}_{dp}'
                r = evaluate(model, tok, all_test, decode_policy=dp, batch_size=32, device=DEVICE)
                all_results[key] = r
                print(f"    {dp}: acc={r['accuracy']:.4f} blank_cell={r['blank_cell_acc']:.4f}")
                ta = r.get('technique_accuracy', {})
                if ta:
                    parts = [f"{c}={v:.3f}" for c, v in sorted(ta.items())]
                    print(f"      Tech: {' '.join(parts)}")

            del model; torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── Figures ──
    print(f"\n{'='*70}\n  Generating figures...\n{'='*70}")
    figs = make_figures(all_results, all_dyn, corr)

    # ── Summary table (exact match + cell accuracy) ──
    print(f"\n{'='*70}\n  SUMMARY — decay={DIFFICULTY_DECAY}\n{'='*70}")

    # ── 1. TECHNIQUE LEVEL ACCURACY (PRIMARY) ──
    print(f"\n  ── Technique Level Accuracy (confidence decode) ──")
    print(f"  {'Level':<18}", end='')
    for mt in MASK_TYPES: print(f" {mt:>12}", end='')
    if len(MASK_TYPES) >= 2: print(f" {'Δ(R-P)':>10}", end='')
    print()
    print(f"  {'─'*65}")
    for tl in TECHNIQUE_LEVELS:
        accs = []
        for mt in MASK_TYPES:
            ta = all_results.get(f'{mt}_confidence', {}).get('technique_accuracy', {})
            accs.append(ta.get(tl))
        if any(a is not None for a in accs):
            print(f"  {tl:<18}", end='')
            for a in accs: print(f" {a:>12.4f}" if a is not None else f" {'N/A':>12}", end='')
            if len(accs) >= 2 and all(a is not None for a in accs[:2]):
                delta = accs[0] - accs[1]  # random - puma (if MASK_TYPES=[random,puma])
                marker = " ←★" if delta > 0 and tl == 'tl_4_search' else ""
                print(f" {delta:>+10.4f}{marker}", end='')
            print()

    # ── 2. TL4 FRACTION SWEEP (crossover search) ──
    tl4_fracs = [0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0]
    has_tl4 = any(f'{mt}_tl4frac_{tf:.2f}_confidence' in all_results
                  for mt in MASK_TYPES for tf in tl4_fracs)
    if has_tl4:
        print(f"\n  ── TL4 Fraction Sweep (cell acc, confidence decode) ──")
        print(f"  {'tl4_frac':<12}", end='')
        for mt in MASK_TYPES: print(f" {mt:>10}", end='')
        if len(MASK_TYPES) >= 2: print(f" {'Δ(R-P)':>10}", end='')
        print()
        print(f"  {'─'*50}")
        for tf in tl4_fracs:
            accs = []
            for mt in MASK_TYPES:
                r = all_results.get(f'{mt}_tl4frac_{tf:.2f}_confidence', {})
                accs.append(r.get('blank_cell_acc'))
            if any(a is not None for a in accs):
                print(f"  {'>=' + str(int(tf*100)) + '%':<12}", end='')
                for a in accs: print(f" {a:>10.4f}" if a is not None else f" {'N/A':>10}", end='')
                if len(accs) >= 2 and all(a is not None for a in accs[:2]):
                    d = accs[0] - accs[1]
                    marker = " ←" if d > 0 else ""
                    print(f" {d:>+10.4f}{marker}", end='')
                print()

    # ── 3. SELECTIVE MASKING (stepping stone effect) ──
    has_sel = any(f'{mt}_selective' in all_results for mt in MASK_TYPES)
    if has_sel:
        print(f"\n  ── Selective Masking — tl_4 accuracy ──")
        for cn in ['all_blank', 'tl4_only', 'tl4_no_stepping']:
            accs = []
            for mt in MASK_TYPES:
                sel = all_results.get(f'{mt}_selective', {}).get(cn, {})
                accs.append(sel.get('tl_4_search'))
            if any(a is not None for a in accs):
                print(f"  {cn:<25s}", end='')
                for mt, a in zip(MASK_TYPES, accs):
                    print(f" {mt}={a:.4f}" if a is not None else f" {mt}=N/A", end='')
                if len(accs) >= 2 and all(a is not None for a in accs[:2]):
                    d = accs[0] - accs[1]
                    print(f"  Δ(R-P)={d:+.4f}", end='')
                print()

    # ── 4. RATING TIER ACCURACY (secondary) ──
    print(f"\n  ── Rating-Stratified (confidence decode) ──")
    print(f"  {'Tier':<12} {'blanks':>6}", end='')
    for mt in MASK_TYPES: print(f" {mt+'_cell':>12}", end='')
    print()
    for tn in RATING_TIERS:
        row_parts = []; has = False; bl = '?'
        for mt in MASK_TYPES:
            ra = all_results.get(f'{mt}_confidence', {}).get('rating_accuracy', {}).get(tn)
            if ra: has = True; bl = f"{ra.get('mean_blanks',0):.0f}"; row_parts.append(f" {ra['cell']:>12.4f}")
            else: row_parts.append(f" {'N/A':>12}")
        if has:
            print(f"  {tn:<12} {bl:>6}", end='')
            for p in row_parts: print(p, end='')
            print()

    # ── 5. TECHNIQUE DISTRIBUTION ──
    tl_dist = defaultdict(int); tl_total = 0
    for d in all_test:
        for j in range(81):
            if not d['meta'][j]['is_given']:
                tl_dist[_tl_to_cat(d['meta'][j])] += 1; tl_total += 1
    if tl_total > 0:
        print(f"\n  ── Test data technique distribution ──")
        for tl in TECHNIQUE_LEVELS:
            n = tl_dist.get(tl, 0)
            print(f"    {tl:<20s}: {n:>8,} ({n/tl_total:.1%})")

    # ── Save ──
    sd = {'config': {k: globals()[k] for k in ['DIFFICULTY_DECAY',
           'N_LAYER', 'N_EMBD', 'N_HEAD', 'MAX_ITERS', 'BATCH_SIZE',
           'MASK_TYPES', 'DECODE_POLICIES', 'PUMA_K', 'PUMA_TAU']},
          'rating_tiers': RATING_TIERS}
    for k, v in all_results.items(): sd[f'result_{k}'] = v
    for k, v in all_dyn.items():
        sd[f'dyn_{k}'] = {'checkpoints': v['checkpoints'], 'train_loss': v['train_loss']}
    sd['correlation'] = corr
    save_results(exp_name, sd, figures=figs)
    return all_results, all_dyn


if __name__ == '__main__':
    args = parse_args()
    seeds = args.seeds if args.seeds else [SEED]
    for si, seed in enumerate(seeds):
        globals()['SEED'] = seed
        seed_tag = f"{args.tag}_s{seed}" if args.tag and len(seeds) > 1 else args.tag
        if len(seeds) > 1:
            print(f"\n{'#'*70}\n# Seed {seed} ({si+1}/{len(seeds)})\n{'#'*70}")
        run(tag=seed_tag)
