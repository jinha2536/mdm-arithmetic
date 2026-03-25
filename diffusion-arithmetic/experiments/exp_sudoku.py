"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Experiment — Sudoku: Constraint Dependency Learning Dynamics
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Core question: does a masked diffusion model learn the constraint
  propagation graph when solving Sudoku?

  Data format:  puzzle_string=solution_string  (163 tokens)
    puzzle:   81 chars, '0' = blank, '1'-'9' = given clue
    solution: 81 chars, all '1'-'9'

  Cell difficulty classification (analogous to addition g/k/p):
    given:    clue cell — trivially solvable (copy from puzzle)
    depth_0:  naked/hidden single from initial clues
    depth_1:  solvable after filling depth_0 cells
    depth_k:  solvable after filling depth_0..k-1 cells
    depth_-1: requires advanced techniques (pairs, X-wing, etc.)

  Key analyses:
    1. prop_depth vs decode_rank — does model decode easy cells first?
    2. constraint cascade — does revealing a cell boost same-group cells?
    3. difficulty concordance — fraction of (easy, hard) pairs decoded correctly
    4. learning dynamics — depth category accuracy over training
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import sys, os, time, math, json, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

EXP_NAME = 'exp_sudoku_dynamics'

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Config (defaults; overridden by CLI)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ANS_LEN = 81

N_TRAIN = 5000
N_TEST = 500
MIN_BLANKS = 30
MAX_BLANKS = 55

BATCH_SIZE = 64
MAX_EPOCHS = 3000
EVAL_EVERY = 50
LOG_EVERY = 20
GEN_EVAL_EVERY = 200
GEN_EVAL_N = 200
THRESHOLD = 0.95

MASK_TYPES = ['random', 'puma']
DECODE_POLICIES = ['confidence']

N_LAYER = 4
N_HEAD = 4
N_EMBD = 256
DROPOUT = 0.2
POS_ENC = 'absolute'

LR = 1e-3
MIN_LR = 1e-4
WARMUP_EPOCHS = 10
GRAD_CLIP = 1.0

PUMA_TAU = 0.9
PUMA_K_START = 5
PUMA_K_END = ANS_LEN

SEED = 42
RUN_AR = False


def parse_args():
    import argparse
    p = argparse.ArgumentParser(description='Sudoku constraint learning experiment')
    p.add_argument('--n-train', type=int, default=None)
    p.add_argument('--n-test', type=int, default=None)
    p.add_argument('--min-blanks', type=int, default=None)
    p.add_argument('--max-blanks', type=int, default=None)
    p.add_argument('--masks', nargs='+', default=None)
    p.add_argument('--decode', nargs='+', default=None)
    p.add_argument('--no-ar', action='store_true')
    p.add_argument('--ar', action='store_true')
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--batch-size', type=int, default=None)
    p.add_argument('--eval-every', type=int, default=None)
    p.add_argument('--gen-eval-every', type=int, default=None)
    p.add_argument('--n-layer', type=int, default=None)
    p.add_argument('--n-head', type=int, default=None)
    p.add_argument('--n-embd', type=int, default=None)
    p.add_argument('--dropout', type=float, default=None)
    p.add_argument('--puma-tau', type=float, default=None)
    p.add_argument('--puma-k-start', type=int, default=None)
    p.add_argument('--puma-k-end', type=int, default=None)
    p.add_argument('--tag', type=str, default='')
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--seeds', nargs='+', type=int, default=None)
    args = p.parse_args()

    g = globals()
    mapping = {
        'n_train': 'N_TRAIN', 'n_test': 'N_TEST', 'epochs': 'MAX_EPOCHS',
        'batch_size': 'BATCH_SIZE', 'eval_every': 'EVAL_EVERY',
        'gen_eval_every': 'GEN_EVAL_EVERY',
        'n_layer': 'N_LAYER', 'n_head': 'N_HEAD', 'n_embd': 'N_EMBD',
        'dropout': 'DROPOUT', 'puma_tau': 'PUMA_TAU',
        'puma_k_start': 'PUMA_K_START', 'puma_k_end': 'PUMA_K_END',
        'seed': 'SEED', 'min_blanks': 'MIN_BLANKS', 'max_blanks': 'MAX_BLANKS',
    }
    for arg_name, global_name in mapping.items():
        val = getattr(args, arg_name)
        if val is not None:
            g[global_name] = val
    if args.masks:
        g['MASK_TYPES'] = args.masks
    if args.decode:
        g['DECODE_POLICIES'] = args.decode
    if args.no_ar:
        g['RUN_AR'] = False
    if args.ar:
        g['RUN_AR'] = True
    return args


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sudoku Core: Solver, Generator, Depth
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _peers(r, c):
    br, bc = (r // 3) * 3, (c // 3) * 3
    ps = set()
    for i in range(9):
        ps.add((r, i)); ps.add((i, c))
    for dr in range(3):
        for dc in range(3):
            ps.add((br + dr, bc + dc))
    ps.discard((r, c))
    return ps

PEERS = {(r, c): _peers(r, c) for r in range(9) for c in range(9)}
ROWS = [[(r, c) for c in range(9)] for r in range(9)]
COLS = [[(r, c) for r in range(9)] for c in range(9)]
BOXES = [[(br+dr, bc+dc) for dr in range(3) for dc in range(3)]
         for br in range(0, 9, 3) for bc in range(0, 9, 3)]
ALL_GROUPS = ROWS + COLS + BOXES

# Cell → indices of groups it belongs to
CELL_GROUP_IDS = {}
for r in range(9):
    for c in range(9):
        CELL_GROUP_IDS[(r, c)] = [gi for gi, g in enumerate(ALL_GROUPS) if (r, c) in g]

# Cell → set of all peer cells (sharing at least one group)
CELL_PEERS_SET = {(r, c): _peers(r, c) for r in range(9) for c in range(9)}


def _get_candidates(grid, r, c):
    if grid[r][c] != 0:
        return set()
    used = {grid[pr][pc] for pr, pc in PEERS[(r, c)]} - {0}
    return set(range(1, 10)) - used


def _solve(grid, max_solutions=2):
    grid = [row[:] for row in grid]
    solutions = []
    def _bt():
        if len(solutions) >= max_solutions:
            return
        best, best_cands = None, None
        for r in range(9):
            for c in range(9):
                if grid[r][c] == 0:
                    cands = _get_candidates(grid, r, c)
                    if len(cands) == 0:
                        return
                    if best is None or len(cands) < len(best_cands):
                        best = (r, c)
                        best_cands = cands
        if best is None:
            solutions.append([row[:] for row in grid])
            return
        r, c = best
        for v in best_cands:
            grid[r][c] = v
            _bt()
            grid[r][c] = 0
    _bt()
    return solutions


def _generate_complete_grid(rng):
    grid = [[0]*9 for _ in range(9)]
    base = list(range(1, 10))
    rng.shuffle(base)
    shifts = [0, 3, 6, 1, 4, 7, 2, 5, 8]
    for r in range(9):
        for c in range(9):
            grid[r][c] = base[(c + shifts[r]) % 9]
    for _ in range(30):
        t = rng.randint(0, 4)
        if t == 0:
            band = rng.randint(0, 2)
            r1, r2 = rng.sample(range(band*3, band*3+3), 2)
            grid[r1], grid[r2] = grid[r2], grid[r1]
        elif t == 1:
            stack = rng.randint(0, 2)
            c1, c2 = rng.sample(range(stack*3, stack*3+3), 2)
            for r in range(9):
                grid[r][c1], grid[r][c2] = grid[r][c2], grid[r][c1]
        elif t == 2:
            b1, b2 = rng.sample(range(3), 2)
            for i in range(3):
                grid[b1*3+i], grid[b2*3+i] = grid[b2*3+i], grid[b1*3+i]
        elif t == 3:
            s1, s2 = rng.sample(range(3), 2)
            for r in range(9):
                for i in range(3):
                    grid[r][s1*3+i], grid[r][s2*3+i] = grid[r][s2*3+i], grid[r][s1*3+i]
        else:
            perm = list(range(1, 10))
            rng.shuffle(perm)
            mp = {i+1: perm[i] for i in range(9)}
            grid = [[mp[grid[r][c]] for c in range(9)] for r in range(9)]
    return grid


def _make_puzzle(solution, n_blanks, rng):
    puzzle = [row[:] for row in solution]
    cells = [(r, c) for r in range(9) for c in range(9)]
    rng.shuffle(cells)
    removed = 0
    for r, c in cells:
        if removed >= n_blanks:
            break
        val = puzzle[r][c]
        puzzle[r][c] = 0
        sols = _solve(puzzle, max_solutions=2)
        if len(sols) == 1:
            removed += 1
        else:
            puzzle[r][c] = val
    return puzzle if removed >= n_blanks else None


def compute_prop_depth(puzzle):
    """Propagation depth using naked + hidden singles.
    Returns dict: (row, col) → depth (0, 1, 2, ..., or -1)."""
    grid = [row[:] for row in puzzle]
    blanks = {(r, c) for r in range(9) for c in range(9) if grid[r][c] == 0}
    depths = {}
    current_depth = 0
    while True:
        newly_solved = []
        for r, c in blanks - set(depths.keys()):
            cands = _get_candidates(grid, r, c)
            if len(cands) == 1:
                grid[r][c] = cands.pop()
                depths[(r, c)] = current_depth
                newly_solved.append((r, c))
        for group in ALL_GROUPS:
            unfilled = [(r, c) for r, c in group
                        if grid[r][c] == 0 and (r, c) not in depths]
            if not unfilled:
                continue
            for val in range(1, 10):
                if any(grid[r][c] == val for r, c in group):
                    continue
                possible = [(r, c) for r, c in unfilled
                            if val in _get_candidates(grid, r, c)]
                if len(possible) == 1:
                    r, c = possible[0]
                    if (r, c) not in depths:
                        grid[r][c] = val
                        depths[(r, c)] = current_depth
                        newly_solved.append((r, c))
        if not newly_solved:
            break
        current_depth += 1
    for r, c in blanks:
        if (r, c) not in depths:
            depths[(r, c)] = -1
    return depths


def compute_n_candidates(puzzle):
    result = {}
    for r in range(9):
        for c in range(9):
            if puzzle[r][c] == 0:
                result[(r, c)] = len(_get_candidates(puzzle, r, c))
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Data Format & Generation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _grid_to_str(grid):
    return ''.join(str(grid[r][c]) for r in range(9) for c in range(9))

def _str_to_grid(s):
    return [[int(s[r*9+c]) for c in range(9)] for r in range(9)]

def _flat_idx_to_rc(idx):
    return idx // 9, idx % 9

def _rc_to_flat_idx(r, c):
    return r * 9 + c


def _fmt_sudoku(puzzle_grid, solution_grid):
    return f"{_grid_to_str(puzzle_grid)}={_grid_to_str(solution_grid)}"


def _parse_sudoku(s):
    """Parse formatted string → (puzzle_str, solution_str)."""
    parts = s.split('=')
    return parts[0], parts[1]


def _get_cell_metadata(puzzle_grid, solution_grid):
    """Compute per-cell metadata for analysis.
    Returns dict with flat_idx → {is_given, prop_depth, n_candidates, ...}
    """
    depths = compute_prop_depth(puzzle_grid)
    n_cands = compute_n_candidates(puzzle_grid)

    meta = {}
    for r in range(9):
        for c in range(9):
            idx = _rc_to_flat_idx(r, c)
            is_given = puzzle_grid[r][c] != 0
            meta[idx] = {
                'is_given': is_given,
                'prop_depth': -99 if is_given else depths.get((r, c), -1),
                'n_candidates': 0 if is_given else n_cands.get((r, c), 0),
                'row': r, 'col': c,
                'box': (r // 3) * 3 + c // 3,
            }
    return meta


def gen_sudoku_data(n, seed, min_blanks=None, max_blanks=None):
    """Generate n Sudoku puzzles with metadata.
    Returns list of dicts: {string, meta, n_blanks}.
    """
    if min_blanks is None: min_blanks = MIN_BLANKS
    if max_blanks is None: max_blanks = MAX_BLANKS

    rng = random.Random(seed)
    results = []
    attempts = 0
    max_attempts = n * 10

    while len(results) < n and attempts < max_attempts:
        attempts += 1
        sol = _generate_complete_grid(rng)
        nb = rng.randint(min_blanks, max_blanks)
        puzzle = _make_puzzle(sol, nb, rng)
        if puzzle is None:
            continue
        actual_blanks = sum(1 for r in range(9) for c in range(9) if puzzle[r][c] == 0)
        s = _fmt_sudoku(puzzle, sol)
        meta = _get_cell_metadata(puzzle, sol)
        results.append({
            'string': s,
            'meta': meta,
            'n_blanks': actual_blanks,
        })

    if len(results) < n:
        print(f"  WARNING: generated only {len(results)}/{n} puzzles")
    return results


def build_tok():
    return CharTokenizer(list('0123456789='), {'mask': 'M', 'pad': 'P'})


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Analysis: Cell Difficulty Categories
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Categories for grouping (analogous to addition's g/k/p)
DEPTH_CATS = ['given', 'depth_0', 'depth_1', 'depth_2', 'depth_3plus', 'depth_hard']

def _depth_to_cat(meta_entry):
    """Map cell metadata to difficulty category."""
    if meta_entry['is_given']:
        return 'given'
    d = meta_entry['prop_depth']
    if d == 0: return 'depth_0'
    if d == 1: return 'depth_1'
    if d == 2: return 'depth_2'
    if d >= 3: return 'depth_3plus'
    return 'depth_hard'  # d == -1

DEPTH_CAT_TO_ID = {name: i for i, name in enumerate(DEPTH_CATS)}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Per-position Probe (adapted from addition)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@torch.no_grad()
def probe_per_position(model, tokenizer, test_data, objective,
                       max_len, device=None):
    """Fully-masked probe with difficulty-category tracking.

    Returns dict with per-position and per-category metrics.
    """
    if device is None: device = DEVICE
    model.eval()
    mask_id = tokenizer.special_ids['mask']

    strings = [d['string'] for d in test_data]
    metas = [d['meta'] for d in test_data]

    ids_all, ans_all = encode_samples(strings, tokenizer, max_len)
    ids_all, ans_all = ids_all.to(device), ans_all.to(device)
    N_test = len(test_data)

    # Precompute category IDs per sample per position [N, 81]
    cat_ids = torch.zeros(N_test, ANS_LEN, dtype=torch.long, device=device)
    for si in range(N_test):
        for j in range(ANS_LEN):
            m = metas[si][j]
            cat_ids[si, j] = DEPTH_CAT_TO_ID[_depth_to_cat(m)]

    # Accumulators
    total_loss = torch.zeros(ANS_LEN, device=device)
    total_correct = torch.zeros(ANS_LEN, device=device)
    total_conf = torch.zeros(ANS_LEN, device=device)
    total_n = torch.zeros(ANS_LEN, device=device)

    # Per-category accumulators
    cat_conf_sum = defaultdict(float)
    cat_acc_sum = defaultdict(float)
    cat_count = defaultdict(int)

    for st in range(0, N_test, 64):
        en = min(st + 64, N_test)
        ids = ids_all[st:en]
        ans = ans_all[st:en]
        B = ids.shape[0]
        T = ids.shape[1]

        ans_pos = ans.unsqueeze(1) + torch.arange(ANS_LEN, device=device)
        ans_pos = ans_pos.clamp(max=T-1)
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand_as(ans_pos)

        if objective == 'ar':
            logits = model(ids[:, :-1])
            pred_pos = ans_pos - 1
            valid = (pred_pos >= 0) & (pred_pos < logits.shape[1])
            pred_pos = pred_pos.clamp(min=0, max=logits.shape[1]-1)
            tgt_ids = ids[batch_idx, ans_pos]
            log_probs = F.log_softmax(logits[batch_idx, pred_pos], dim=-1)
            losses = -log_probs.gather(2, tgt_ids.unsqueeze(2)).squeeze(2) * valid.float()
            preds = logits[batch_idx, pred_pos].argmax(dim=-1)
            corrects = (preds == tgt_ids).float() * valid.float()
            confs = F.softmax(logits[batch_idx, pred_pos], dim=-1).max(dim=-1).values * valid.float()
            v_count = valid.float()
        else:
            xm = ids.clone()
            xm[batch_idx, ans_pos] = mask_id
            logits = model(xm)
            ans_logits = logits[batch_idx, ans_pos]
            tgt_ids = ids[batch_idx, ans_pos]
            log_probs = F.log_softmax(ans_logits, dim=-1)
            losses = -log_probs.gather(2, tgt_ids.unsqueeze(2)).squeeze(2)
            cl = ans_logits.clone()
            cl[:, :, mask_id] = -float('inf')
            probs = F.softmax(cl, dim=-1)
            confs = probs.max(dim=-1).values
            preds = probs.argmax(dim=-1)
            corrects = (preds == tgt_ids).float()
            v_count = torch.ones(B, ANS_LEN, device=device)

        for j in range(ANS_LEN):
            total_loss[j] += losses[:, j].sum()
            total_correct[j] += corrects[:, j].sum()
            total_conf[j] += confs[:, j].sum()
            total_n[j] += v_count[:, j].sum()

        # Per-category (vectorized)
        cat_batch = cat_ids[st:en]
        confs_flat = confs.reshape(-1)
        corrects_flat = corrects.reshape(-1)
        cat_flat = cat_batch.reshape(-1)
        for ci, cname in enumerate(DEPTH_CATS):
            mask = (cat_flat == ci)
            if mask.any():
                cat_conf_sum[cname] += confs_flat[mask].sum().item()
                cat_acc_sum[cname] += corrects_flat[mask].sum().item()
                cat_count[cname] += mask.sum().item()

    N_safe = total_n.clamp(min=1)
    pos_loss = (total_loss / N_safe).tolist()
    pos_acc = (total_correct / N_safe).tolist()
    pos_conf = (total_conf / N_safe).tolist()
    overall_loss = total_loss.sum().item() / total_n.sum().item()
    overall_acc = total_correct.sum().item() / total_n.sum().item()

    # Per-category summary
    depth_context = {}
    for cname in DEPTH_CATS:
        n = cat_count.get(cname, 0)
        if n > 0:
            depth_context[cname] = {
                'mean_conf': cat_conf_sum[cname] / n,
                'mean_acc': cat_acc_sum[cname] / n,
                'n': n,
            }

    return {
        'pos_loss': pos_loss, 'pos_acc': pos_acc, 'pos_conf': pos_conf,
        'overall_loss': overall_loss, 'overall_acc': overall_acc,
        'depth_context': depth_context,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Constraint Cascade Analysis
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@torch.no_grad()
def analyse_constraint_cascade(model, tokenizer, test_data, max_len,
                                n_samples=100, device=None):
    """Step-by-step decode tracking constraint group relationships.

    When cell X is revealed:
      - Δconf of cells sharing a constraint group (row/col/box) with X
      - Δconf of cells sharing NO group with X

    If same_group_delta >> diff_group_delta, the model recognizes
    the constraint network topology.

    Also tracks by depth category of revealed cell:
      - revealing a 'given' or 'depth_0' cell (easy) should cascade differently
        than revealing a 'depth_hard' cell.
    """
    if device is None: device = DEVICE
    model.eval()
    mask_id = tokenizer.special_ids['mask']

    samples = test_data[:n_samples]

    # Accumulators
    delta_same_group = []     # Δconf of cells in same row/col/box
    delta_diff_group = []     # Δconf of cells sharing no group
    # By category of revealed cell
    delta_by_cat = {cat: {'same': [], 'diff': []} for cat in DEPTH_CATS}
    # By category of observed cell (whose conf changes)
    delta_obs_by_cat = {cat: [] for cat in DEPTH_CATS}

    for si, sample in enumerate(samples):
        s = sample['string']
        meta = sample['meta']

        puzzle_str, sol_str = _parse_sudoku(s)
        prefix = puzzle_str + '='
        penc = tokenizer.encode(prefix)
        T_pre = len(penc)

        # Build input: prefix + MASK * 81
        x = torch.full((1, T_pre + ANS_LEN), mask_id, dtype=torch.long, device=device)
        x[0, :T_pre] = torch.tensor(penc, device=device)
        unmasked = torch.zeros(T_pre + ANS_LEN, dtype=torch.bool, device=device)
        unmasked[:T_pre] = True

        # Precompute cell peer sets (flat index → set of flat indices)
        peer_flat = {}
        for j in range(81):
            r, c = _flat_idx_to_rc(j)
            peer_flat[j] = {_rc_to_flat_idx(pr, pc) for pr, pc in PEERS[(r, c)]}

        # Initial confidence
        logits = model(x)
        prev_conf = torch.zeros(ANS_LEN, device=device)
        for j in range(ANS_LEN):
            cl = logits[0, T_pre + j].clone()
            cl[mask_id] = -float('inf')
            prev_conf[j] = F.softmax(cl, dim=-1).max()

        # Step-by-step decode (confidence policy)
        for step in range(ANS_LEN):
            logits = model(x)

            # Find highest-confidence masked position
            best_conf, best_j = -1.0, -1
            for j in range(ANS_LEN):
                if unmasked[T_pre + j]:
                    continue
                cl = logits[0, T_pre + j].clone()
                cl[mask_id] = -float('inf')
                c = F.softmax(cl, dim=-1).max().item()
                if c > best_conf:
                    best_conf = c
                    best_j = j
            if best_j < 0:
                break

            # Reveal
            pos = T_pre + best_j
            cl = logits[0, pos].clone()
            cl[mask_id] = -float('inf')
            x[0, pos] = cl.argmax()
            unmasked[pos] = True

            # New confidences
            logits_new = model(x)
            new_conf = torch.zeros(ANS_LEN, device=device)
            for j in range(ANS_LEN):
                if unmasked[T_pre + j]:
                    new_conf[j] = 1.0
                    continue
                cl2 = logits_new[0, T_pre + j].clone()
                cl2[mask_id] = -float('inf')
                new_conf[j] = F.softmax(cl2, dim=-1).max()

            # Classify revealed cell
            rev_cat = _depth_to_cat(meta[best_j])
            rev_peers = peer_flat[best_j]

            # Record Δconf for remaining masked cells
            for j in range(ANS_LEN):
                if unmasked[T_pre + j] or j == best_j:
                    continue
                delta = (new_conf[j] - prev_conf[j]).item()
                obs_cat = _depth_to_cat(meta[j])

                if j in rev_peers:
                    delta_same_group.append(delta)
                    delta_by_cat[rev_cat]['same'].append(delta)
                else:
                    delta_diff_group.append(delta)
                    delta_by_cat[rev_cat]['diff'].append(delta)

                delta_obs_by_cat[obs_cat].append(delta)

            prev_conf = new_conf

    def _mean(lst):
        return sum(lst) / len(lst) if lst else 0.0

    # Summarise by revealed category
    by_rev_cat = {}
    for cat in DEPTH_CATS:
        s_list = delta_by_cat[cat]['same']
        d_list = delta_by_cat[cat]['diff']
        if s_list or d_list:
            by_rev_cat[cat] = {
                'same_group_delta': _mean(s_list), 'n_same': len(s_list),
                'diff_group_delta': _mean(d_list), 'n_diff': len(d_list),
            }

    by_obs_cat = {cat: {'mean_delta': _mean(v), 'n': len(v)}
                  for cat, v in delta_obs_by_cat.items() if v}

    result = {
        'same_group_delta': _mean(delta_same_group),
        'diff_group_delta': _mean(delta_diff_group),
        'n_same': len(delta_same_group),
        'n_diff': len(delta_diff_group),
        'by_revealed_cat': by_rev_cat,
        'by_observed_cat': by_obs_cat,
    }
    sg = result['same_group_delta']
    dg = result['diff_group_delta']
    if dg != 0:
        ratio_str = f"ratio={sg/dg:.2f}x"
    else:
        ratio_str = "ratio=inf"
    result['summary'] = (
        f"Constraint cascade: same_group={sg:+.4f} (n={result['n_same']}), "
        f"diff_group={dg:+.4f} (n={result['n_diff']}), {ratio_str}"
    )
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Training Loop (epoch-based, adapted from addition)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def train_with_dynamics(
    objective, tokenizer, train_data, test_data,
    max_len, mask_type='random', device=None,
):
    if device is None: device = DEVICE

    train_strings = [d['string'] for d in train_data]
    train_ids, train_ans = encode_samples(train_strings, tokenizer, max_len)
    train_ids = train_ids.to(device)
    train_ans = train_ans.to(device)
    N = train_ids.shape[0]
    bpe = (N + BATCH_SIZE - 1) // BATCH_SIZE
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

    dynamics = {'checkpoints': [], 'gen_checkpoints': [], 'train_loss': []}
    best_loss = float('inf')
    best_state = None
    it = 0
    tg = 0
    t0 = time.time()

    T = train_ids.shape[1]
    uses_streaming = mask_type in ('puma', 'oracle_lsb')

    # ── PUMA/oracle streaming buffer ──
    if uses_streaming:
        puma_z = torch.full((BATCH_SIZE, T), mask_id, dtype=torch.long, device=device)
        puma_x0 = torch.zeros(BATCH_SIZE, T, dtype=torch.long, device=device)
        puma_ans = torch.zeros(BATCH_SIZE, dtype=torch.long, device=device)  # answer start per slot
        puma_stage = torch.zeros(BATCH_SIZE, dtype=torch.long, device=device)
        sample_pool = list(range(N))

        def _puma_refresh(indices):
            nonlocal sample_pool
            for bi in indices:
                if not sample_pool:
                    sample_pool = list(range(N))
                    random.shuffle(sample_pool)
                si = sample_pool.pop()
                puma_x0[bi] = train_ids[si]
                puma_z[bi] = train_ids[si].clone()
                a_s = train_ans[si].item()
                puma_ans[bi] = a_s
                for j in range(ANS_LEN):
                    p = a_s + j
                    if p < T:
                        puma_z[bi, p] = mask_id
                puma_stage[bi] = 0

        def _chain_advance(logits, K_cur):
            nonlocal puma_stage
            B_buf = puma_z.shape[0]
            ans_pos = puma_ans.unsqueeze(1) + torch.arange(ANS_LEN, device=device)
            ans_pos = ans_pos.clamp(max=T-1)
            batch_idx = torch.arange(B_buf, device=device).unsqueeze(1).expand_as(ans_pos)

            is_masked = (puma_z[batch_idx, ans_pos] == mask_id)
            if not is_masked.any():
                _puma_refresh(list(range(B_buf)))
                return

            # Confidence for position selection
            lp = logits[batch_idx, ans_pos]
            lp[:, :, mask_id] = -float('inf')
            confs = F.softmax(lp, dim=-1).max(dim=-1).values
            confs[~is_masked] = -float('inf')

            n_masked = is_masked.sum(dim=1).float()
            n_reveal = (n_masked / max(K_cur, 1)).ceil().long().clamp(min=1)

            if mask_type == 'puma':
                ranked = confs.argsort(dim=1, descending=True)
                rank_of_pos = torch.zeros_like(ranked)
                rank_of_pos.scatter_(1, ranked, torch.arange(ANS_LEN, device=device).expand(B_buf, -1))
                reveal = (rank_of_pos < n_reveal.unsqueeze(1)) | (confs > PUMA_TAU)
                reveal = reveal & is_masked
            else:
                # oracle: reveal by some fixed order (not implemented for sudoku)
                reveal = is_masked  # fallback

            reveal_abs = ans_pos[reveal]
            batch_reveal = batch_idx[reveal]
            puma_z[batch_reveal, reveal_abs] = puma_x0[batch_reveal, reveal_abs]
            puma_stage += 1

            still_masked = (puma_z[batch_idx, ans_pos] == mask_id).any(dim=1)
            done = (~still_masked) | (puma_stage >= K_cur)
            if done.any():
                _puma_refresh(done.nonzero(as_tuple=True)[0].tolist())

        _puma_refresh(list(range(BATCH_SIZE)))

    def _do_eval(epoch):
        nonlocal best_loss, best_state
        probe = probe_per_position(
            model, tokenizer, test_data, objective, max_len, device)
        dynamics['checkpoints'].append({
            'epoch': epoch, 'iter': it, 'token_gradients': tg, **probe})
        pl = probe['overall_loss']
        if pl < best_loss and epoch > 0:
            best_loss = pl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    def _do_gen_eval(epoch):
        """Lightweight generation eval with decode order analysis."""
        test_strings = [d['string'] for d in test_data[:GEN_EVAL_N]]
        test_metas = [d['meta'] for d in test_data[:GEN_EVAL_N]]
        r = final_evaluate(model, tokenizer, test_data[:GEN_EVAL_N],
                           'diffusion', decode_policy='confidence')
        entry = {'epoch': epoch, 'gen_acc': r['accuracy']}
        oa = r.get('decode_order_analysis')
        if oa:
            entry['depth_vs_rank'] = oa.get('depth_vs_rank')
            entry['difficulty_concordance'] = oa.get('difficulty_concordance')
            entry['given_mean_rank'] = oa.get('given_mean_rank')
            entry['blank_mean_rank'] = oa.get('blank_mean_rank')
        dynamics['gen_checkpoints'].append(entry)

    # ── Main training loop ──
    if uses_streaming:
        _puma_refresh(list(range(BATCH_SIZE)))

    puma_K_cur = PUMA_K_START

    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_loss_t = torch.tensor(0.0, device=device)
        epoch_tg = torch.tensor(0, dtype=torch.long, device=device)
        epoch_n = 0

        if uses_streaming:
            puma_K_cur = PUMA_K_START + int((PUMA_K_END - PUMA_K_START)
                                             * epoch / MAX_EPOCHS)
            for _ in range(bpe):
                for pg in optimizer.param_groups:
                    pg['lr'] = get_lr(it)
                m = (puma_z == mask_id)
                if m.sum() == 0:
                    _puma_refresh(list(range(BATCH_SIZE)))
                    m = (puma_z == mask_id)
                logits = model(puma_z)
                loss = F.cross_entropy(logits[m], puma_x0[m])
                epoch_tg += m.sum()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                _chain_advance(logits.detach(), puma_K_cur)
                epoch_loss_t += loss.detach()
                epoch_n += 1
                it += 1
        else:
            perm = torch.randperm(N, device=device)
            for bi in range(bpe):
                for pg in optimizer.param_groups:
                    pg['lr'] = get_lr(it)
                idx = perm[bi*BATCH_SIZE : min((bi+1)*BATCH_SIZE, N)]
                ids = train_ids[idx]
                ans_starts = train_ans[idx]
                B, T_batch = ids.shape

                if objective == 'ar':
                    logits = model(ids[:, :-1])
                    targets = ids[:, 1:]
                    pos = torch.arange(T_batch-1, device=device).unsqueeze(0)
                    lm = pos >= (ans_starts.unsqueeze(1) - 1)
                    lm = lm & (targets != pad_id)
                    if lm.sum() == 0: it += 1; continue
                    loss = F.cross_entropy(logits[lm], targets[lm])
                    epoch_tg += lm.sum()
                else:
                    ans_pos = ans_starts.unsqueeze(1) + torch.arange(ANS_LEN, device=device)
                    ans_pos = ans_pos.clamp(max=T_batch-1)
                    batch_idx = torch.arange(B, device=device).unsqueeze(1).expand_as(ans_pos)

                    if mask_type == 'random':
                        pos_all = torch.arange(T_batch, device=device).unsqueeze(0)
                        ans_mask = pos_all >= ans_starts.unsqueeze(1)
                        t_ratio = torch.rand(B, device=device)
                        m_probs = t_ratio.unsqueeze(1) * ans_mask.float()
                        m = torch.bernoulli(m_probs).bool()
                        no_m = ~(m.any(dim=1))
                        if no_m.any():
                            rand_j = torch.randint(ANS_LEN, (no_m.sum(),), device=device)
                            fix_pos = ans_pos[no_m].gather(1, rand_j.unsqueeze(1)).squeeze(1)
                            m[no_m, fix_pos] = True
                    elif mask_type == 'confidence':
                        xm_probe = ids.clone()
                        xm_probe[batch_idx, ans_pos] = mask_id
                        model.eval()
                        with torch.no_grad():
                            logits_probe = model(xm_probe)
                        model.train()
                        lp = logits_probe[batch_idx, ans_pos]
                        lp[:, :, mask_id] = -float('inf')
                        confs = F.softmax(lp, dim=-1).max(dim=-1).values
                        ranked = confs.argsort(dim=1)
                        t_ratio = torch.rand(B, device=device)
                        nm = (t_ratio * ANS_LEN).ceil().long().clamp(min=1)
                        rank_of_pos = torch.zeros_like(ranked)
                        rank_of_pos.scatter_(1, ranked, torch.arange(ANS_LEN, device=device).expand(B, -1))
                        to_mask = rank_of_pos < nm.unsqueeze(1)
                        m = torch.zeros(B, T_batch, dtype=torch.bool, device=device)
                        m[batch_idx, ans_pos] = to_mask

                    xm = ids.clone(); xm[m] = mask_id
                    logits = model(xm)
                    if m.sum() == 0: it += 1; continue
                    loss = F.cross_entropy(logits[m], ids[m])
                    epoch_tg += m.sum()

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                epoch_loss_t += loss.detach()
                epoch_n += 1
                it += 1

        tg += epoch_tg.item()
        avg_loss = epoch_loss_t.item() / max(epoch_n, 1)

        if epoch % LOG_EVERY == 0:
            dynamics['train_loss'].append((epoch, avg_loss))
            print(f"    ep {epoch:4d}/{MAX_EPOCHS} | loss {avg_loss:.4f} | "
                  f"lr {get_lr(it):.1e} | tg {tg:,} | {time.time()-t0:.0f}s")

        # Eval schedule (front-loaded)
        do_eval = False
        if epoch % EVAL_EVERY == 0 and epoch < MAX_EPOCHS:
            do_eval = True
        elif epoch < MAX_EPOCHS * 0.1 and epoch % max(EVAL_EVERY // 5, 1) == 0:
            do_eval = True
        elif epoch < MAX_EPOCHS * 0.3 and epoch % max(EVAL_EVERY // 2, 1) == 0:
            do_eval = True
        if do_eval:
            model.eval(); _do_eval(epoch); model.train()

        # Gen eval (less frequent)
        if objective == 'diffusion' and epoch % GEN_EVAL_EVERY == 0 and epoch > 0:
            model.eval(); _do_gen_eval(epoch); model.train()

    print(f"    Done {MAX_EPOCHS} epochs (best probe loss: {best_loss:.4f})")
    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    _do_eval(MAX_EPOCHS)
    return model, dynamics


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Final Evaluation (generation-based)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def final_evaluate(model, tokenizer, test_data, objective,
                   decode_policy='confidence', batch_size=32, device=None):
    if device is None: device = DEVICE
    mask_id = tokenizer.special_ids['mask']
    pad_id = tokenizer.special_ids['pad']
    model.eval()

    strings = [d['string'] for d in test_data]
    metas = [d['meta'] for d in test_data]
    results = []
    all_orders = []

    for st in range(0, len(test_data), batch_size):
        batch = test_data[st:st+batch_size]
        B = len(batch)
        batch_strings = [d['string'] for d in batch]
        penc = [tokenizer.encode(s.split('=')[0] + '=') for s in batch_strings]
        pm = max(len(p) for p in penc)
        pids = torch.full((B, pm), pad_id, dtype=torch.long)
        for i, e in enumerate(penc):
            pids[i, :len(e)] = torch.tensor(e)

        with torch.no_grad():
            if objective == 'ar':
                gen = generate_ar(model, pids, ANS_LEN, device)
                pred_ids = gen[:, pm:pm+ANS_LEN]
                bo = None
            else:
                policy = 'confidence' if decode_policy == 'confidence' else decode_policy
                gen, _, info = generate_diffusion(
                    model, pids, ANS_LEN, mask_id,
                    policy=policy, greedy=True, device=device)
                pred_ids = gen[:, pm:pm+ANS_LEN]
                bo = info.get('orders')
        if bo is not None:
            all_orders.append(bo)

        for i in range(B):
            ps = tokenizer.decode(pred_ids[i].cpu().tolist())
            gs = batch[i]['string'].split('=')[1]
            meta_i = batch[i]['meta']
            pc = [ps[j] == gs[j] if j < len(ps) else False for j in range(len(gs))]
            results.append({
                'correct': ps == gs,
                'pos_correct': pc,
                'meta': meta_i,
                'n_blanks': batch[i]['n_blanks'],
            })

    n = len(results)
    acc = sum(r['correct'] for r in results) / max(n, 1)
    pos_acc = [sum(r['pos_correct'][j] for r in results) / max(n, 1)
               for j in range(ANS_LEN)]

    # Per-category accuracy
    cat_correct = defaultdict(list)
    for r in results:
        for j in range(ANS_LEN):
            cat = _depth_to_cat(r['meta'][j])
            cat_correct[cat].append(r['pos_correct'][j])
    cat_acc = {cat: sum(v)/len(v) for cat, v in cat_correct.items() if v}

    # By n_blanks
    blanks_acc = defaultdict(list)
    for r in results:
        blanks_acc[r['n_blanks']].append(r['correct'])
    by_blanks = {nb: (sum(v)/len(v), len(v)) for nb, v in sorted(blanks_acc.items())}

    # Decode order analysis
    oa = None
    if all_orders:
        oc = torch.cat(all_orders, dim=0)
        pl = len(tokenizer.encode(test_data[0]['string'].split('=')[0] + '='))
        oa = _analyse_orders(oc, pl, [r['meta'] for r in results])

    return {
        'accuracy': acc, 'n_samples': n, 'position_accuracy': pos_acc,
        'category_accuracy': cat_acc, 'by_blanks': by_blanks,
        'decode_order_analysis': oa,
    }


def _analyse_orders(decode_orders, prefix_len, metas):
    """Decode order analysis for Sudoku.

    Computes:
      depth_vs_rank: {depth_cat: mean_decode_rank}
      difficulty_concordance: fraction of (easy, hard) pairs where easy first
      given_mean_rank: mean decode rank of given cells
      blank_mean_rank: mean decode rank of blank cells
      n_cands_vs_rank: {n_candidates: mean_rank}
    """
    N, S = decode_orders.shape

    # Build rop [N, ANS_LEN]: decode step for each position
    rel_orders = decode_orders - prefix_len
    rop = torch.full((N, ANS_LEN), float('nan'))
    valid = (rel_orders >= 0) & (rel_orders < ANS_LEN)
    for s in range(S):
        v = valid[:, s]
        if v.any():
            positions = rel_orders[v, s].long()
            rop[v.nonzero(as_tuple=True)[0], positions] = s

    valid_mask = ~rop.isnan().any(dim=1)
    rop_valid = rop[valid_mask]
    n_valid = rop_valid.shape[0]

    if n_valid == 0:
        return {'depth_vs_rank': {}, 'difficulty_concordance': None}

    # Per-cell category
    cat_ids = torch.zeros(N, ANS_LEN, dtype=torch.long)
    n_cands_t = torch.zeros(N, ANS_LEN, dtype=torch.long)
    is_blank = torch.zeros(N, ANS_LEN, dtype=torch.bool)
    for si in range(N):
        for j in range(ANS_LEN):
            m = metas[si][j]
            cat_ids[si, j] = DEPTH_CAT_TO_ID[_depth_to_cat(m)]
            n_cands_t[si, j] = m['n_candidates']
            is_blank[si, j] = not m['is_given']

    cat_v = cat_ids[valid_mask].reshape(-1)
    rank_v = rop_valid.reshape(-1)
    nc_v = n_cands_t[valid_mask].reshape(-1)
    blank_v = is_blank[valid_mask].reshape(-1)

    # depth_vs_rank
    depth_vs_rank = {}
    for ci, cname in enumerate(DEPTH_CATS):
        mask = (cat_v == ci)
        if mask.any():
            depth_vs_rank[cname] = rank_v[mask].mean().item()

    # given vs blank mean rank
    given_mask = ~blank_v
    blank_mask = blank_v
    given_mean_rank = rank_v[given_mask].mean().item() if given_mask.any() else None
    blank_mean_rank = rank_v[blank_mask].mean().item() if blank_mask.any() else None

    # n_candidates vs rank (blank cells only)
    n_cands_vs_rank = {}
    for nc_val in nc_v[blank_mask].unique().tolist():
        m = blank_mask & (nc_v == nc_val)
        if m.any():
            n_cands_vs_rank[int(nc_val)] = rank_v[m].mean().item()

    # Difficulty concordance (blank cells only)
    # For each sample, for each pair of blank cells (i, j) where depth_i < depth_j:
    # concordant if rank_i < rank_j (easier cell decoded first)
    conc_total = 0
    conc_correct = 0
    for si in range(n_valid):
        orig_si = valid_mask.nonzero(as_tuple=True)[0][si].item()
        meta = metas[orig_si]
        ranks = rop_valid[si]
        blank_js = [j for j in range(ANS_LEN) if not meta[j]['is_given']]
        # Sort by prop_depth for comparison
        for i_idx in range(len(blank_js)):
            for j_idx in range(i_idx + 1, len(blank_js)):
                ji = blank_js[i_idx]
                jj = blank_js[j_idx]
                di = meta[ji]['prop_depth']
                dj = meta[jj]['prop_depth']
                # Map -1 to large number for ordering
                di_eff = di if di >= 0 else 100
                dj_eff = dj if dj >= 0 else 100
                if di_eff < dj_eff:
                    conc_total += 1
                    if ranks[ji] < ranks[jj]:
                        conc_correct += 1
                elif dj_eff < di_eff:
                    conc_total += 1
                    if ranks[jj] < ranks[ji]:
                        conc_correct += 1
    difficulty_concordance = conc_correct / max(conc_total, 1)

    return {
        'depth_vs_rank': depth_vs_rank,
        'difficulty_concordance': difficulty_concordance,
        'given_mean_rank': given_mean_rank,
        'blank_mean_rank': blank_mean_rank,
        'n_cands_vs_rank': n_cands_vs_rank,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Visualization
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DEPTH_COLORS = {
    'given': '#95a5a6', 'depth_0': '#2ecc71', 'depth_1': '#3498db',
    'depth_2': '#e67e22', 'depth_3plus': '#e74c3c', 'depth_hard': '#8e44ad',
}

def _fk(obj, mt='', dp=''):
    if obj == 'ar':
        return 'ar'
    return f"{obj}_{mt}_{dp}"

def _ck(obj, mt='', dp=''):
    if obj == 'ar':
        return 'ar'
    return f"diff-{mt[:3]}-{dp[:3]}"


def make_figures(all_dyn, all_final):
    figs = {}

    conds = []
    if RUN_AR:
        conds.append(('ar', ''))
    for mt in MASK_TYPES:
        conds.append(('diffusion', mt))
    nc = len(conds)

    # ── Fig 1: Per-depth-category accuracy over training ──
    if nc > 0:
        fig, axes = plt.subplots(1, nc, figsize=(7*nc, 5), squeeze=False)
        axes = axes[0]
        for ai, (obj, mt) in enumerate(conds):
            key = _fk(obj, mt, 'confidence')
            dyn = all_dyn.get(key)
            if not dyn: continue
            ax = axes[ai]
            cps = dyn['checkpoints']
            xs = [c['epoch'] for c in cps]
            for cat in DEPTH_CATS:
                ys = []
                for c in cps:
                    dc = c.get('depth_context', {})
                    entry = dc.get(cat, {})
                    ys.append(entry.get('mean_acc', float('nan')))
                if any(not math.isnan(y) for y in ys):
                    ax.plot(xs, ys, '-', color=DEPTH_COLORS.get(cat, '#333'),
                            label=cat, lw=1.5, alpha=0.8)
            ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy')
            ax.set_ylim(-0.05, 1.05)
            ax.set_title(_ck(obj, mt, 'confidence'))
            ax.legend(fontsize=7); ax.grid(alpha=0.3)
        fig.suptitle('Accuracy by Difficulty Category', fontsize=13, y=1.02)
        fig.tight_layout()
        figs['depth_acc_evolution'] = fig

    # ── Fig 2: Per-depth-category confidence over training ──
    if nc > 0:
        fig, axes = plt.subplots(1, nc, figsize=(7*nc, 5), squeeze=False)
        axes = axes[0]
        for ai, (obj, mt) in enumerate(conds):
            key = _fk(obj, mt, 'confidence')
            dyn = all_dyn.get(key)
            if not dyn: continue
            ax = axes[ai]
            cps = dyn['checkpoints']
            xs = [c['epoch'] for c in cps]
            for cat in DEPTH_CATS:
                ys = []
                for c in cps:
                    dc = c.get('depth_context', {})
                    entry = dc.get(cat, {})
                    ys.append(entry.get('mean_conf', float('nan')))
                if any(not math.isnan(y) for y in ys):
                    ax.plot(xs, ys, '-', color=DEPTH_COLORS.get(cat, '#333'),
                            label=cat, lw=1.5, alpha=0.8)
            ax.set_xlabel('Epoch'); ax.set_ylabel('Confidence')
            ax.set_ylim(-0.05, 1.05)
            ax.set_title(_ck(obj, mt, 'confidence'))
            ax.legend(fontsize=7); ax.grid(alpha=0.3)
        fig.suptitle('Confidence by Difficulty Category', fontsize=13, y=1.02)
        fig.tight_layout()
        figs['depth_conf_evolution'] = fig

    # ── Fig 3: Depth vs decode rank (bar chart, final) ──
    rank_data = {}
    for obj, mt in conds:
        if obj == 'ar': continue
        key = _fk(obj, mt, 'confidence')
        r = all_final.get(key)
        if r and r.get('decode_order_analysis'):
            rank_data[_ck(obj, mt, 'confidence')] = r['decode_order_analysis'].get('depth_vs_rank', {})

    if rank_data:
        fig, axes = plt.subplots(1, len(rank_data), figsize=(7*len(rank_data), 5), squeeze=False)
        axes = axes[0]
        for ai, (label, dvr) in enumerate(rank_data.items()):
            ax = axes[ai]
            cats = [c for c in DEPTH_CATS if c in dvr]
            vals = [dvr[c] for c in cats]
            colors = [DEPTH_COLORS.get(c, '#333') for c in cats]
            ax.bar(range(len(cats)), vals, color=colors, tick_label=cats)
            ax.set_ylabel('Mean Decode Rank')
            ax.set_title(label)
            ax.grid(alpha=0.3, axis='y')
            for i, v in enumerate(vals):
                ax.text(i, v + 0.5, f'{v:.1f}', ha='center', fontsize=8)
        fig.suptitle('Decode Rank by Cell Difficulty', fontsize=13, y=1.02)
        fig.tight_layout()
        figs['depth_vs_rank'] = fig

    # ── Fig 4: Gen accuracy + difficulty concordance over training ──
    gen_keys = [(obj, mt) for obj, mt in conds if obj == 'diffusion']
    if gen_keys:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for obj, mt in gen_keys:
            key = _fk(obj, mt, 'confidence')
            dyn = all_dyn.get(key)
            if not dyn: continue
            gcs = dyn.get('gen_checkpoints', [])
            xs = [g['epoch'] for g in gcs]
            accs = [g.get('gen_acc', 0) for g in gcs]
            concs = [g.get('difficulty_concordance', 0.5) for g in gcs]
            label = _ck(obj, mt, 'confidence')
            axes[0].plot(xs, accs, '-o', ms=2, label=label)
            axes[1].plot(xs, concs, '-o', ms=2, label=label)
        axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Gen Accuracy')
        axes[0].set_ylim(-0.05, 1.05); axes[0].legend(); axes[0].grid(alpha=0.3)
        axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Difficulty Concordance')
        axes[1].axhline(0.5, color='gray', ls='--', alpha=0.5, label='random')
        axes[1].legend(); axes[1].grid(alpha=0.3)
        fig.suptitle('Generation Performance Over Training', fontsize=13, y=1.02)
        fig.tight_layout()
        figs['gen_dynamics'] = fig

    return figs


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run(tag=''):
    exp_name = f"{EXP_NAME}_{tag}" if tag else EXP_NAME
    print(f"\n{'='*70}")
    print(f"  {exp_name}")
    print(f"  ANS_LEN={ANS_LEN}, blanks={MIN_BLANKS}-{MAX_BLANKS}")
    print(f"  N_TRAIN={N_TRAIN}, N_TEST={N_TEST}")
    print(f"  {N_LAYER}L/{N_EMBD}D/{N_HEAD}H, epochs={MAX_EPOCHS}")
    print(f"  masks={MASK_TYPES}, decode={DECODE_POLICIES}")
    print(f"  PUMA: tau={PUMA_TAU}, K={PUMA_K_START}→{PUMA_K_END}")
    print(f"{'='*70}")

    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    tok = build_tok()

    # ── Generate data ──
    print("\n  Generating training data...")
    t0 = time.time()
    train_data = gen_sudoku_data(N_TRAIN, seed=SEED)
    print(f"  Train: {len(train_data)} puzzles in {time.time()-t0:.1f}s")

    print("  Generating test data...")
    test_data = gen_sudoku_data(N_TEST, seed=9000)
    print(f"  Test:  {len(test_data)} puzzles")

    # Diagnostics
    depth_dist = defaultdict(int)
    blanks_dist = defaultdict(int)
    for d in train_data:
        blanks_dist[d['n_blanks']] += 1
        for j in range(81):
            cat = _depth_to_cat(d['meta'][j])
            depth_dist[cat] += 1
    print(f"  Train depth dist: {dict(sorted(depth_dist.items()))}")
    blanks_summary = sorted(blanks_dist.items())
    print(f"  Train blanks range: {blanks_summary[0][0]}-{blanks_summary[-1][0]}")

    sample = train_data[0]['string']
    max_len = max(len(tok.encode(d['string'])) for d in train_data)
    print(f"  max_len={max_len}, sample={sample[:30]}...={sample.split('=')[1][:15]}...")

    all_dyn = {}
    all_final = {}

    # ── AR ──
    if RUN_AR:
        key = _fk('ar')
        print(f"\n{'━'*60}\n▶ {key}\n{'━'*60}")
        m, d = train_with_dynamics('ar', tok, train_data, test_data, max_len)
        all_dyn[key] = d
        r = final_evaluate(m, tok, test_data, 'ar')
        all_final[key] = r
        print(f"  Final gen acc: {r['accuracy']:.4f}")
        del m; torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── Diffusion ──
    for mt in MASK_TYPES:
        kb = _fk('diffusion', mt, 'confidence')
        print(f"\n{'━'*60}\n▶ {kb}\n{'━'*60}")
        m, d = train_with_dynamics('diffusion', tok, train_data, test_data,
                                   max_len, mask_type=mt)
        all_dyn[kb] = d

        for dp in DECODE_POLICIES:
            key = _fk('diffusion', mt, dp)
            print(f"\n  Final eval: {key}")
            r = final_evaluate(m, tok, test_data, 'diffusion', decode_policy=dp)
            all_final[key] = r
            print(f"  Acc: {r['accuracy']:.4f}")
            oa = r.get('decode_order_analysis')
            if oa:
                dvr = oa.get('depth_vs_rank', {})
                dc = oa.get('difficulty_concordance')
                dc = oa.get('difficulty_concordance')
                if dc is not None:
                    print(f"    Difficulty concordance: {dc:.4f}")
                gmr = oa.get('given_mean_rank')
                bmr = oa.get('blank_mean_rank')
                gmr_s = f'{gmr:.1f}' if gmr is not None else 'n/a'
                bmr_s = f'{bmr:.1f}' if bmr is not None else 'n/a'
                print(f"    Given rank: {gmr_s}, Blank rank: {bmr_s}")
                if dvr:
                    parts = [f"{cat}={v:.1f}" for cat, v in dvr.items()]
                    print(f"    Depth→rank: {' '.join(parts)}")
                ncr = oa.get('n_cands_vs_rank', {})
                if ncr:
                    parts = [f"nc={k}→{v:.1f}" for k, v in sorted(ncr.items())]
                    print(f"    N_cands→rank: {' '.join(parts)}")
            ca = r.get('category_accuracy', {})
            if ca:
                parts = [f"{cat}={v:.3f}" for cat, v in ca.items()]
                print(f"    Cat accuracy: {' '.join(parts)}")

        # ── Constraint cascade ──
        print(f"  Constraint cascade analysis...")
        cascade = analyse_constraint_cascade(
            m, tok, test_data, max_len, n_samples=50, device=DEVICE)
        all_final[_fk('diffusion', mt, 'cascade')] = cascade
        print(f"    {cascade['summary']}")
        by_rev = cascade.get('by_revealed_cat', {})
        for cat, info in sorted(by_rev.items()):
            sg = info['same_group_delta']
            dg = info['diff_group_delta']
            print(f"    reveal={cat}: same={sg:+.4f} diff={dg:+.4f}")

        del m; torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── Figures ──
    print(f"\n{'='*70}\n  Generating figures...\n{'='*70}")
    figs = make_figures(all_dyn, all_final)

    # ── Save ──
    sd = {'config': {
        'ANS_LEN': ANS_LEN, 'N_TRAIN': N_TRAIN, 'N_TEST': N_TEST,
        'MIN_BLANKS': MIN_BLANKS, 'MAX_BLANKS': MAX_BLANKS,
        'MAX_EPOCHS': MAX_EPOCHS, 'BATCH_SIZE': BATCH_SIZE,
        'N_LAYER': N_LAYER, 'N_HEAD': N_HEAD, 'N_EMBD': N_EMBD,
        'MASK_TYPES': MASK_TYPES, 'DECODE_POLICIES': DECODE_POLICIES,
        'RUN_AR': RUN_AR, 'tag': tag,
    }}
    for k, d_val in all_dyn.items():
        sd[f'dyn_{k}'] = {'checkpoints': d_val['checkpoints'],
                          'gen_checkpoints': d_val.get('gen_checkpoints', []),
                          'train_loss': d_val['train_loss']}
    for k, r in all_final.items():
        sr = {kk: vv for kk, vv in r.items() if kk != 'decode_order_analysis'}
        oa = r.get('decode_order_analysis')
        if oa:
            sr['decode_order'] = oa
        sd[f'final_{k}'] = sr
    save_results(exp_name, sd, figures=figs)

    # ── Summary ──
    print(f"\n{'='*70}\n  SUMMARY\n{'='*70}")
    for key, r in all_final.items():
        if 'accuracy' in r:
            print(f"  {key:<40s} acc={r['accuracy']:.4f}")
            ca = r.get('category_accuracy', {})
            if ca:
                for cat in DEPTH_CATS:
                    if cat in ca:
                        print(f"    {cat:<15s} {ca[cat]:.4f}")

    return all_dyn, all_final


if __name__ == '__main__':
    args = parse_args()
    seeds = args.seeds if args.seeds else [SEED]
    if len(seeds) == 1:
        globals()['SEED'] = seeds[0]
        run(tag=args.tag)
    else:
        for si, seed in enumerate(seeds):
            globals()['SEED'] = seed
            seed_tag = f"{args.tag}_s{seed}" if args.tag else f"s{seed}"
            print(f"\n{'#'*70}\n# Seed {seed} ({si+1}/{len(seeds)})\n{'#'*70}")
            run(tag=seed_tag)
