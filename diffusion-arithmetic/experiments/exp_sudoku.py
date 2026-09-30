"""Sudoku (9x9) from the sudoku-extreme dataset.

The prompt is the puzzle (blanks as 0); the answer is the full 81-cell
solution, with only the blank cells masked. Test puzzles are stratified by the
tdoku rating (number of guesses of an MRV backtracking solver) into tiers, and
every blank cell is classified by the deepest solving technique it requires:
    level 0  naked / hidden singles
    level 1  naked / hidden pairs and triples, pointing pairs, box-line reduction
    level 2  X-wing, swordfish, XY-wing
    level 3  single-step forcing chains
    level 4  search (bifurcation) required
Decoding (one cell per forward pass): confidence, solver order (determination
order of the MRV backtracking solver), technique order (increasing technique
level) and uniform random. Accuracy is reported on the blank cells: overall,
per rating tier, and on the puzzles with >= TL4_FRAC_MIN of level-4 blank cells.
"""
import sys, os, time, random
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                if '__file__' in dir() else '.')
from core.tokenizer import CharTokenizer
from core.train_utils import (
    prepare_results_dir, save_results, save_checkpoint, encode_samples,
    train_diffusion, puma_k_step, DEVICE,
)

EXP_NAME = 'exp_sudoku'

# Config
ANS_LEN = 81

# Training distribution over rating tiers:
#   None  -> all available puzzles per tier
#   d     -> easy tier in full, then tier i (in increasing difficulty) sub-sampled
#            at rate d**i (exponential decay towards the hard tiers)
DIFFICULTY_DECAY = 0.01
# per-tier test counts
N_TEST_PER_TIER_DICT = {
    'easy': 1000, 'medium': 1000, 'hard': 500, 'very_hard': 200, 'extreme': 100,
    'top1pct': 500,
}
N_TEST_PER_TIER = 500  # fallback for tiers not listed above
# TL4 stratum: the first TL4_MAX_N test puzzles whose fraction of level-4
# blank cells is >= TL4_FRAC_MIN
TL4_FRAC_MIN = 0.95
TL4_MAX_N = 200

# Model
N_LAYER = 8; N_HEAD = 8; N_EMBD = 256
DROPOUT = 0.0

# Training
MAX_ITERS = 300000; BATCH_SIZE = 256
LR = 3e-4; MIN_LR = 1e-5; WARMUP_ITERS = 2000
GRAD_CLIP = 1.0; WEIGHT_DECAY = 0.01
EMA_DECAY = 0.9999
EVAL_EVERY = 8000; LOG_EVERY = 2000

MASK_TYPES = ['random', 'papl', 'puma']
DECODE_POLICIES = ['confidence', 'oracle_solver', 'oracle_technique', 'random']
PUMA_TAU = 0.9
PUMA_K_START = 8; PUMA_K_END = 40     # 81 cells: ~10 -> ~2 tokens per step
PUMA_K_STEP = 5; PUMA_K_EVERY = None  # None = ramp over the first 1/3 of training
PAPL_TAU = 1.0; PAPL_ALPHA = 5.0
SEED = 42
NO_AMP = True   # bf16 autocast off (fp32 training)

# Rating tiers; replaced by data-driven quantile boundaries in load_hf_data()
RATING_TIERS = {
    'easy': (0, 0), 'medium': (1, 9), 'hard': (10, 49),
    'very_hard': (50, 149), 'extreme': (150, 99999),
}


def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--difficulty-decay', type=float, default=None,
                   help='sub-sampling rate of the harder training tiers (see DIFFICULTY_DECAY)')
    p.add_argument('--max-iters', type=int, default=None)
    p.add_argument('--batch-size', type=int, default=None)
    p.add_argument('--n-layer', type=int, default=None)
    p.add_argument('--n-head', type=int, default=None)
    p.add_argument('--n-embd', type=int, default=None)
    p.add_argument('--dropout', type=float, default=None)
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--masks', nargs='+', default=None)
    p.add_argument('--decode', nargs='+', default=None)
    p.add_argument('--puma-k-start', type=int, default=None)
    p.add_argument('--puma-k-end', type=int, default=None)
    p.add_argument('--puma-k-step', type=int, default=None)
    p.add_argument('--puma-k-every', type=int, default=None)
    p.add_argument('--puma-tau', type=float, default=None)
    p.add_argument('--papl-tau', type=float, default=None)
    p.add_argument('--papl-alpha', type=float, default=None)
    p.add_argument('--no-amp', action='store_true')
    p.add_argument('--tag', type=str, default='')
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--seeds', nargs='+', type=int, default=None)
    args, _ = p.parse_known_args()
    g = globals()
    for a, gl in {'max_iters': 'MAX_ITERS',
                   'batch_size': 'BATCH_SIZE', 'n_layer': 'N_LAYER', 'n_head': 'N_HEAD',
                   'n_embd': 'N_EMBD', 'dropout': 'DROPOUT', 'lr': 'LR',
                   'puma_tau': 'PUMA_TAU',
                   'puma_k_start': 'PUMA_K_START', 'puma_k_end': 'PUMA_K_END',
                   'puma_k_step': 'PUMA_K_STEP', 'puma_k_every': 'PUMA_K_EVERY',
                   'papl_tau': 'PAPL_TAU', 'papl_alpha': 'PAPL_ALPHA',
                   'seed': 'SEED', 'difficulty_decay': 'DIFFICULTY_DECAY'}.items():
        v = getattr(args, a, None)
        if v is not None: g[gl] = v
    if args.no_amp: g['NO_AMP'] = True
    if args.masks: g['MASK_TYPES'] = args.masks
    if args.decode: g['DECODE_POLICIES'] = args.decode
    return args


# Sudoku core

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

# flat unit lookups for the technique solver
ROW_OF = [i // 9 for i in range(81)]
COL_OF = [i % 9 for i in range(81)]
BOX_OF = [(i // 9 // 3) * 3 + (i % 9 // 3) for i in range(81)]
ROW_CELLS = [[r * 9 + c for c in range(9)] for r in range(9)]
COL_CELLS = [[r * 9 + c for r in range(9)] for c in range(9)]
BOX_CELLS = [[(br + dr) * 9 + (bc + dc) for dr in range(3) for dc in range(3)]
             for br in range(0, 9, 3) for bc in range(0, 9, 3)]
ALL_UNITS = ROW_CELLS + COL_CELLS + BOX_CELLS  # 27 units, flat indices

BOX_ROW_INTER = {}   # (box, row) -> list of cells in intersection
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


# Technique-level solver
# Level 0: naked / hidden singles
# Level 1: naked / hidden pairs and triples, pointing pairs, box-line reduction
# Level 2: X-wing, swordfish, XY-wing
# Level 3: single-step forcing chains (hypothesis -> contradiction)
# Level 4: remaining cells (search required)


def _singles_propagate(cands):
    """Apply naked and hidden singles with full cascade.
    Returns (changed, valid): the set of newly determined cells and a validity flag."""
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
            # Found naked subset - eliminate these values from other cells in unit
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
            # Hidden subset - remove all OTHER candidates from these cells
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
                # All in one row - eliminate from rest of row outside box
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
    """X-Wing: value in exactly 2 cols in 2 rows -> eliminate from those cols."""
    elim_any = False
    for v in range(1, 10):
        bit = VAL_BIT[v]
        # Row-based X-Wing
        row_cols = {}  # row -> frozenset of cols where v appears
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
    """Swordfish: value in at most 3 columns across 3 rows (and vice versa)."""
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
    """XY-Wing: pivot {x,y} + pincer1 {x,z} + pincer2 {y,z} -> eliminate z."""
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
    contradiction via singles propagation -> eliminate v.
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
                # Contradiction -> eliminate this candidate
                cands[i] &= ~bit; elim_any = True
                if cands[i] == 0: return True
                break  # restart this cell since cands changed
    return elim_any


def compute_technique_level(puzzle_flat):
    """Per-cell technique level via hierarchical solver.
    Returns dict: cell -> level (0-4), and solve_order dict.

    Level 0: Naked/Hidden singles
    Level 1: Naked/Hidden pairs/triples, Pointing, Box-line reduction
    Level 2: X-Wing, Swordfish, XY-Wing
    Level 3: Forcing chains (depth-1 hypothesis -> contradiction)
    Level 4: Remaining (search required)
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

    # Level 0: singles (already propagated by _bm_init)
    _record(0)
    if len(cell_level) == len(blanks_set):
        return cell_level, solve_order

    # Level 1: subsets + intersections
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

    # Level 2: fish + wings
    for _ in range(20):
        prog = False
        prog |= _apply_xwing(cands)
        prog |= _apply_swordfish(cands)
        prog |= _apply_xy_wing(cands)
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

    # Level 3: forcing chains
    for _ in range(15):
        prog = _apply_forcing_chains(cands)
        for unit in ALL_UNITS:
            prog |= _apply_naked_subsets(cands, unit, 2)
        prog |= _apply_pointing(cands)
        prog |= _apply_xwing(cands)
        det, valid = _singles_propagate(cands)
        if not valid: break
        if det: prog = True
        _record(3)
        if not prog or len(cell_level) == len(blanks_set): break

    # Level 4: remaining cells
    for i in blanks_set:
        if i not in cell_level:
            cell_level[i] = 4
            solve_order[i] = oc[0]; oc[0] += 1

    return cell_level, solve_order


def compute_guess_depth(puzzle_flat):
    """Per-cell guess depth and determination order of an MRV backtracking solver.
    Returns:
      cell_depth: dict cell -> guess_depth (0 = constraint propagation, k = after the k-th guess)
      solve_order: dict cell -> rank in the solver's determination sequence
      total_guesses: int (the tdoku-style rating)
    solve_order is the order used by the 'oracle_solver' decode policy.
    """
    cands = _bm_init(puzzle_flat)
    blanks = [i for i in range(81) if puzzle_flat[i] == 0]
    if cands is None:
        return ({i: -1 for i in blanks}, {i: i for i in blanks}, 0)

    cell_depth = {}
    solve_order = {}
    order_counter = [0]
    total_guesses = [0]

    # cells determined by constraint propagation (guess_depth = 0)
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
                # backtrack
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


# Data loading

def _compute_puzzle_meta(puzzle_str, sol_str):
    """Per-cell metadata of one puzzle ('0' or '.' for blanks): given flag,
    determination order of the backtracking solver, technique level and order."""
    puzzle_flat = [int(c) if c.isdigit() else 0 for c in puzzle_str]
    n_blanks = sum(1 for v in puzzle_flat if v == 0)
    _, solve_order, total_guesses = compute_guess_depth(puzzle_flat)
    tech_levels, tech_order = compute_technique_level(puzzle_flat)
    meta = {}
    for i in range(81):
        is_given = puzzle_flat[i] != 0
        meta[i] = {
            'is_given': is_given,
            'solve_order': -1 if is_given else solve_order.get(i, 999),
            'technique_level': 0 if is_given else tech_levels.get(i, 4),
            'technique_order': -1 if is_given else tech_order.get(i, 999),
        }
    pstr = ''.join(str(v) for v in puzzle_flat)
    return {'string': f"{pstr}={sol_str}", 'meta': meta, 'n_blanks': n_blanks,
            'total_guesses': total_guesses}


def _compute_meta_worker(args):
    """Worker for parallel meta computation."""
    puzzle_str, sol_str, rating = args[:3]
    d = _compute_puzzle_meta(puzzle_str, sol_str)
    d['rating'] = rating
    return d


def load_hf_data(n_test, decay=None, seed=42, cache_dir='.sudoku_cache'):
    """Load the sudoku-extreme dataset from the HuggingFace hub.
    decay: None -> all available puzzles per tier;
           d    -> easy tier in full, tier i (medium = 1, hard = 2, ...) at rate d**i
    """
    print("  Loading HuggingFace sudoku-extreme dataset...")
    from datasets import load_dataset
    ds = load_dataset('sapientinc/sudoku-extreme', cache_dir=cache_dir)

    rng = random.Random(seed)
    train_raw = ds['train']
    test_raw = ds['test']

    # rating distribution
    print("  Profiling rating distribution...")
    all_ratings = train_raw['rating']
    n_total = len(all_ratings)
    sorted_r = sorted(all_ratings)
    r_max = sorted_r[-1]
    print(f"    Total: {n_total:,} | min={sorted_r[0]} max={r_max}")
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

    # tier boundaries: easy = rating 0; the non-zero ratings are split at their
    # p50 / p80 / p95 / p99 quantiles
    global RATING_TIERS
    nonzero = [r for r in all_ratings if r > 0]
    if nonzero:
        nonzero_sorted = sorted(nonzero)
        nn = len(nonzero_sorted)
        p50 = nonzero_sorted[nn//2]
        p80 = nonzero_sorted[int(nn*0.80)]
        p95 = nonzero_sorted[min(int(nn*0.95), nn-1)]
        p99 = nonzero_sorted[min(int(nn*0.99), nn-1)]
        RATING_TIERS = {
            'easy': (0, 0),
            'medium': (1, max(p50, 1)),
            'hard': (max(p50, 1)+1, p80),
            'very_hard': (p80+1, p95),
            'extreme': (p95+1, p99),
            'top1pct': (p99+1, r_max),
        }
        RATING_TIERS = {k: v for k, v in RATING_TIERS.items() if v[0] <= v[1]}
        print("    Rating tiers:")
        for tn, (lo, hi) in RATING_TIERS.items():
            cnt = sum(1 for r in all_ratings if lo <= r <= hi)
            print(f"      {tn}: [{lo}, {hi}] -> {cnt:,} train samples")

    def _sample_by_rating(data, lo, hi, n, rng_inst):
        ratings = data['rating']
        indices = [i for i, r in enumerate(ratings) if lo <= r <= hi]
        actual_n = min(len(indices), n)
        if len(indices) > n:
            indices = rng_inst.sample(indices, n)
        print(f"    rating [{lo},{hi}]: {len(indices)} available -> sampling {actual_n}")
        if not indices: return []
        rows = data.select(indices)
        return [(rows[i]['question'], rows[i]['answer'], rows[i]['rating'],
                 rows[i].get('source', ''))
                for i in range(len(rows))]

    # training sample with difficulty decay
    print(f"  Sampling train data (decay={decay})...")
    train_tuples = []
    tier_counts = {}
    tier_names = list(RATING_TIERS.keys())
    # decay starts after the base tiers
    base_tiers = {'easy'}
    decay_idx = 0
    for i, tn in enumerate(tier_names):
        if tn not in base_tiers:
            decay_idx = i; break

    for i, (tier_name, (lo, hi)) in enumerate(RATING_TIERS.items()):
        available = sum(1 for r in all_ratings if lo <= r <= hi)
        if available == 0: continue

        if decay is None:
            n_tier = available
        elif tier_name in base_tiers:
            n_tier = available
        else:
            # decay ** (tier position after the base tiers)
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

    # test tiers
    print("  Sampling test data...")
    rng2 = random.Random(seed + 1000)
    test_tiers = {}
    for tier_name, (lo, hi) in RATING_TIERS.items():
        n_tier_test = N_TEST_PER_TIER_DICT.get(tier_name, n_test)
        samples = _sample_by_rating(test_raw, lo, hi, n_tier_test, rng2)
        if not samples:
            print(f"    -> fallback to train set for {tier_name}")
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

def _make_train_entry(puzzle_str, sol_str, rating, source=''):
    """Training entry without metadata."""
    pf = [int(c) if c.isdigit() else 0 for c in puzzle_str]
    pstr = ''.join(str(v) for v in pf)
    return {'string': f"{pstr}={sol_str}", 'rating': rating,
            'n_blanks': sum(1 for v in pf if v == 0)}


def prepare_datasets(seed=42):
    """Training entries (no metadata) and test tiers (with per-cell metadata)."""
    train_tuples, test_tiers = load_hf_data(N_TEST_PER_TIER, decay=DIFFICULTY_DECAY, seed=seed)
    train_data = [_make_train_entry(*t[:3], t[3] if len(t) > 3 else '')
                  for t in train_tuples]

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
    return train_data, test_data


# Tokenizer

def build_tok():
    return CharTokenizer(list('0123456789='), {'mask': 'M', 'pad': 'P'})


# Probe (fully-masked, blank cells)

@torch.no_grad()
def probe_per_cell(model, tokenizer, test_data, max_len, device=None):
    """Fully-masked probe: every blank cell masked, one forward pass; mean NLL
    and argmax accuracy (mask token excluded) over the blank cells."""
    if device is None: device = DEVICE
    model.eval()
    mask_id = tokenizer.special_ids['mask']
    strings = [d['string'] for d in test_data]
    ids_all, ans_all = encode_samples(strings, tokenizer, max_len)
    ids_all, ans_all = ids_all.to(device), ans_all.to(device)
    N = len(test_data)

    blank_masks = torch.zeros(N, ANS_LEN, dtype=torch.bool, device=device)
    for si, d in enumerate(test_data):
        ps = d['string'].split('=')[0]
        for j in range(ANS_LEN):
            if ps[j] == '0': blank_masks[si, j] = True

    total_loss = torch.tensor(0.0, device=device)
    total_n = torch.tensor(0, dtype=torch.long, device=device)
    total_correct = 0.0

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
        corrects = (cl.argmax(dim=-1) == tgt).float()
        w = bl.float()

        total_loss += (losses * w).sum()
        total_n += w.sum().long()
        total_correct += (corrects * w).sum().item()

    return {'overall_loss': (total_loss / total_n.clamp(1)).item(),
            'overall_acc': total_correct / max(total_n.item(), 1)}


# Training

def train(mask_type, tokenizer, train_data, test_data_dict, max_len, device=None):
    """train_diffusion with the blank cells as the maskable positions and the
    fully-masked probe on the 'hard' test tier as eval callback."""
    if device is None: device = DEVICE
    max_iters = MAX_ITERS

    strings = [d['string'] for d in train_data]
    train_ids, train_ans = encode_samples(strings, tokenizer, max_len)
    train_ids, train_ans = train_ids.to(device), train_ans.to(device)

    # only blank ('0') cells are maskable
    zero_id = tokenizer.encode('0')[0]
    blank_masks = (train_ids[:, :ANS_LEN] == zero_id)

    eval_tier = 'hard' if 'hard' in test_data_dict else list(test_data_dict.keys())[0]
    eval_data = test_data_dict[eval_tier]

    def eval_fn(model, it, tg):
        probe = probe_per_cell(model, tokenizer, eval_data, max_len, device)
        print(f"    [eval it {it}] loss={probe['overall_loss']:.4f} "
              f"acc={probe['overall_acc']:.4f}")
        return probe

    k_sched = None
    if mask_type == 'puma':
        k_step = PUMA_K_STEP or 5
        if PUMA_K_EVERY is not None:
            k_every = PUMA_K_EVERY
        else:
            n_inc = max(1, (PUMA_K_END - PUMA_K_START) // k_step)
            k_every = max(1000, (max_iters // 3) // n_inc)
        k_sched = puma_k_step(PUMA_K_START, PUMA_K_END, k_step, k_every)
        print(f"  PUMA K: step {PUMA_K_START}->{k_sched(max_iters)} "
              f"(+{k_step} every {k_every // 1000}k, cap={PUMA_K_END})")

    return train_diffusion(
        train_ids=train_ids, train_ans=train_ans, ans_len=ANS_LEN, tokenizer=tokenizer,
        mask_type=mask_type, blank_masks=blank_masks,
        puma_tau=PUMA_TAU, puma_k_schedule=k_sched,
        papl_tau=PAPL_TAU, papl_alpha=PAPL_ALPHA,
        n_layer=N_LAYER, n_head=N_HEAD, n_embd=N_EMBD, dropout=DROPOUT,
        max_iters=max_iters, batch_size=BATCH_SIZE,
        lr=LR, min_lr=MIN_LR, warmup_iters=WARMUP_ITERS,
        grad_clip=GRAD_CLIP, weight_decay=WEIGHT_DECAY, ema_decay=EMA_DECAY,
        eval_fn=eval_fn, eval_every=EVAL_EVERY, log_every=LOG_EVERY,
        device=device, use_amp=False if NO_AMP else None,
    )


# Decoding

@torch.no_grad()
def generate_blanks(model, tokenizer, test_data, decode_policy='confidence',
                    batch_size=32, device=None):
    """Decode the blank cells one per forward pass. Position selection:
      'confidence':       max softmax probability among masked cells
      'oracle_solver':    determination order of the backtracking solver
      'oracle_technique': increasing technique level, then technique order
      'random':           uniformly random order
    The token is always the argmax. Returns per-puzzle per-cell correctness."""
    if device is None: device = DEVICE
    mask_id = tokenizer.special_ids['mask']
    pad_id = tokenizer.special_ids['pad']
    model.eval()
    results = []

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

        # givens visible, blanks masked
        x = ids.clone()
        blank_m = torch.zeros(B, ANS_LEN, dtype=torch.bool, device=device)
        for i in range(B):
            ps = batch[i]['string'].split('=')[0]
            for j in range(ANS_LEN):
                if ps[j] == '0': x[i, ans_starts[i]+j] = mask_id; blank_m[i,j] = True
        max_steps = blank_m.sum(dim=1).max().item()

        # fixed decode order for the non-confidence policies
        static_order = None
        if decode_policy in ('oracle_solver', 'oracle_technique', 'random'):
            static_order = torch.full((B, ANS_LEN), 9999, dtype=torch.long, device=device)
            for i in range(B):
                meta = batch[i]['meta']
                blank_js = [j for j in range(ANS_LEN) if not meta[j]['is_given']]
                if decode_policy == 'random':
                    random.shuffle(blank_js)
                elif decode_policy == 'oracle_solver':
                    blank_js.sort(key=lambda j: meta[j].get('solve_order', 999))
                else:
                    blank_js.sort(key=lambda j: (meta[j].get('technique_level', 4),
                                                  meta[j].get('technique_order', 999)))
                for rank, j in enumerate(blank_js): static_order[i, j] = rank

        for step in range(max_steps):
            logits = model(x)
            al = logits[bi, ap]; cl = al.clone(); cl[:, :, mask_id] = -float('inf')
            probs = F.softmax(cl, dim=-1)
            still_m = (x[bi, ap] == mask_id)

            if decode_policy == 'confidence':
                confs = probs.max(dim=-1).values
                confs[~still_m] = -float('inf')
                best_j = confs.argmax(dim=1)
            else:
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

        pred_ids = x[bi, ap]
        for i in range(B):
            ps = tokenizer.decode(pred_ids[i].cpu().tolist())
            gs = batch[i]['string'].split('=')[1]
            pc = [ps[j] == gs[j] if j < len(ps) else False for j in range(len(gs))]
            results.append({'correct': ps == gs, 'pos_correct': pc})
    return results


# Evaluation

def tl4_fraction(entry):
    """Fraction of the blank cells that need search (technique level 4)."""
    meta = entry['meta']
    blanks = [j for j in range(81) if not meta[j]['is_given']]
    if not blanks: return None
    return sum(1 for j in blanks if meta[j].get('technique_level', 0) == 4) / len(blanks)


def _cell_acc(results, entries):
    n_blank = sum(e['n_blanks'] for e in entries)
    n_ok = sum(sum(r['pos_correct'][j] for j in range(81) if not e['meta'][j]['is_given'])
               for r, e in zip(results, entries))
    return n_ok / max(n_blank, 1)


def evaluate(model, tokenizer, test_data, decode_policy='confidence',
             batch_size=32, device=None):
    """Exact-match and blank-cell accuracy: overall, per rating tier, and on the
    TL4 stratum (first TL4_MAX_N puzzles with tl4_fraction >= TL4_FRAC_MIN)."""
    results = generate_blanks(model, tokenizer, test_data, decode_policy, batch_size, device)
    n = len(results)

    rating_acc = {}
    for tn, (lo, hi) in RATING_TIERS.items():
        idx = [i for i, e in enumerate(test_data) if lo <= e.get('rating', 0) <= hi]
        if idx:
            rs = [results[i] for i in idx]; es = [test_data[i] for i in idx]
            rating_acc[tn] = {'exact': sum(r['correct'] for r in rs) / len(rs),
                              'cell': _cell_acc(rs, es), 'n': len(rs),
                              'mean_blanks': sum(e['n_blanks'] for e in es) / len(es)}

    fracs = [tl4_fraction(e) for e in test_data]
    idx = [i for i, fr in enumerate(fracs) if fr is not None and fr >= TL4_FRAC_MIN][:TL4_MAX_N]
    tl4 = None
    if idx:
        rs = [results[i] for i in idx]; es = [test_data[i] for i in idx]
        tl4 = {'exact': sum(r['correct'] for r in rs) / len(rs),
               'cell': _cell_acc(rs, es), 'n': len(rs)}

    return {'accuracy': sum(r['correct'] for r in results) / max(n, 1),
            'blank_cell_acc': _cell_acc(results, test_data), 'n': n,
            'rating_accuracy': rating_acc, 'tl4_stratum': tl4}


# Run

def run(tag=''):
    exp_name = f"{EXP_NAME}_{tag}" if tag else EXP_NAME
    print(f"\n{'='*70}")
    print(f"  {exp_name}")
    print(f"  Model: {N_LAYER}L/{N_EMBD}D/{N_HEAD}H  Data: decay={DIFFICULTY_DECAY}")
    print(f"  Training: {MAX_ITERS} iters, batch={BATCH_SIZE}")
    print(f"  Masks: {MASK_TYPES}  Decode: {DECODE_POLICIES}")
    print(f"  PUMA: K={PUMA_K_START}->{PUMA_K_END} step={PUMA_K_STEP}, tau={PUMA_TAU}")
    print(f"{'='*70}")

    prepare_results_dir()
    torch.manual_seed(SEED); random.seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed(SEED)

    tok = build_tok()
    train_data, test_data = prepare_datasets(seed=SEED)
    max_len = max(len(tok.encode(d['string'])) for d in train_data)
    all_test = [d for ds in test_data.values() for d in ds]

    all_results = {}
    all_dyn = {}
    for mt in MASK_TYPES:
        print(f"\n{'='*60}\nTraining: {mt}\n{'='*60}")
        model, dyn = train(mt, tok, train_data, test_data, max_len, device=DEVICE)
        all_dyn[mt] = dyn
        save_checkpoint(exp_name, {k: v.cpu().clone() for k, v in model.state_dict().items()},
                        tag=mt)

        for dp in DECODE_POLICIES:
            r = evaluate(model, tok, all_test, decode_policy=dp, batch_size=32, device=DEVICE)
            all_results[f'{mt}_{dp}'] = r
            t4 = r['tl4_stratum']
            print(f"  {dp}: exact={r['accuracy']:.4f} cell={r['blank_cell_acc']:.4f}"
                  + (f" | TL4>={TL4_FRAC_MIN}: cell={t4['cell']:.4f} (n={t4['n']})" if t4 else ''))
            for tn, info in r['rating_accuracy'].items():
                print(f"    {tn}: exact={info['exact']:.4f} cell={info['cell']:.4f} (n={info['n']})")

        del model; torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print(f"\n{'='*70}\n  SUMMARY (blank-cell accuracy)\n{'='*70}")
    print(f"  {'Test':<34s}" + ''.join(f" {mt:>10s}" for mt in MASK_TYPES))
    for dp in DECODE_POLICIES:
        rows = [('overall', lambda r: r.get('blank_cell_acc'))]
        rows += [(tn, lambda r, tn=tn: r.get('rating_accuracy', {}).get(tn, {}).get('cell'))
                 for tn in RATING_TIERS]
        rows += [(f'TL4>={TL4_FRAC_MIN}', lambda r: (r.get('tl4_stratum') or {}).get('cell'))]
        for name, fn in rows:
            vals = [fn(all_results.get(f'{mt}_{dp}', {})) for mt in MASK_TYPES]
            if any(v is not None for v in vals):
                print(f"  {dp + ' ' + name:<34s}" +
                      ''.join(f" {v:>10.4f}" if v is not None else f" {'N/A':>10s}" for v in vals))

    sd = {'config': {k: globals()[k] for k in ['DIFFICULTY_DECAY',
           'N_LAYER', 'N_EMBD', 'N_HEAD', 'MAX_ITERS', 'BATCH_SIZE',
           'MASK_TYPES', 'DECODE_POLICIES', 'PUMA_TAU', 'PUMA_K_START', 'PUMA_K_END',
           'PUMA_K_STEP', 'PAPL_TAU', 'PAPL_ALPHA', 'TL4_FRAC_MIN', 'TL4_MAX_N', 'SEED']},
          'rating_tiers': RATING_TIERS}
    for k, v in all_results.items(): sd[f'result_{k}'] = v
    for k, v in all_dyn.items():
        sd[f'dyn_{k}'] = {'checkpoints': v['checkpoints'], 'train_loss': v['train_loss']}
    save_results(exp_name, sd)
    return all_results, all_dyn


if __name__ == '__main__':
    args = parse_args()
    seeds = args.seeds if args.seeds else [SEED]
    for si, seed in enumerate(seeds):
        globals()['SEED'] = seed
        seed_tag = (f"{args.tag}_s{seed}" if args.tag else f"s{seed}") if len(seeds) > 1 else args.tag
        if len(seeds) > 1:
            print(f"\n{'#'*70}\n# Seed {seed} ({si+1}/{len(seeds)})\n{'#'*70}")
        run(tag=seed_tag)
