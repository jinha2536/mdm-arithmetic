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
MIN_BLANKS = 45
MAX_BLANKS = 58

BATCH_SIZE = 64
MAX_EPOCHS = 3000
EVAL_EVERY = 50
LOG_EVERY = 20
GEN_EVAL_EVERY = 500
GEN_EVAL_N = 50
THRESHOLD = 0.95

MASK_TYPES = ['random', 'puma']
DECODE_POLICIES = ['confidence']

N_LAYER = 2
N_HEAD = 2
N_EMBD = 128
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
# Sudoku Core: Bitmask Solver, Generator, Depth
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ── Precomputed topology ──
_rows = [[(r, c) for c in range(9)] for r in range(9)]
_cols = [[(r, c) for r in range(9)] for c in range(9)]
_boxes = [[(br+dr, bc+dc) for dr in range(3) for dc in range(3)]
          for br in range(0, 9, 3) for bc in range(0, 9, 3)]
ALL_GROUPS = _rows + _cols + _boxes

def _peers_rc(r, c):
    br, bc = (r // 3) * 3, (c // 3) * 3
    ps = set()
    for i in range(9):
        ps.add((r, i)); ps.add((i, c))
    for dr in range(3):
        for dc in range(3):
            ps.add((br + dr, bc + dc))
    ps.discard((r, c))
    return ps

PEERS = {(r, c): _peers_rc(r, c) for r in range(9) for c in range(9)}

PEERS_FLAT = [None] * 81
UNITS_FLAT = [None] * 81
for _r in range(9):
    for _c in range(9):
        _i = _r * 9 + _c
        _units = [u for u in ALL_GROUPS if (_r, _c) in u]
        UNITS_FLAT[_i] = [[rr * 9 + cc for rr, cc in u] for u in _units]
        _ps = set()
        for u in _units:
            for rr, cc in u:
                _ps.add(rr * 9 + cc)
        _ps.discard(_i)
        PEERS_FLAT[_i] = list(_ps)

ALL_9 = 0x1FF
VAL_BIT = [0] + [1 << (v - 1) for v in range(1, 10)]
BIT_VAL = {1 << i: i + 1 for i in range(9)}
POPCOUNT = [bin(i).count('1') for i in range(512)]
LOWEST_BIT = [0] * 512
for _i in range(1, 512):
    LOWEST_BIT[_i] = _i & (-_i)


# ── Bitmask constraint propagation solver ──

def _bm_init(grid_flat):
    cands = [ALL_9] * 81
    for i in range(81):
        if grid_flat[i] != 0:
            if not _bm_assign(cands, i, grid_flat[i]):
                return None
    return cands

def _bm_assign(cands, i, val):
    others = cands[i] & ~VAL_BIT[val]
    while others:
        ob = LOWEST_BIT[others]
        if not _bm_elim(cands, i, BIT_VAL[ob]):
            return False
        others &= ~ob
    return True

def _bm_elim(cands, i, val):
    bit = VAL_BIT[val]
    if not (cands[i] & bit):
        return True
    cands[i] &= ~bit
    c = cands[i]
    if c == 0:
        return False
    if POPCOUNT[c] == 1:
        for p in PEERS_FLAT[i]:
            if not _bm_elim(cands, p, BIT_VAL[c]):
                return False
    for unit in UNITS_FLAT[i]:
        places = [j for j in unit if cands[j] & bit]
        n = len(places)
        if n == 0:
            return False
        if n == 1:
            if not _bm_assign(cands, places[0], val):
                return False
    return True

def _bm_solved(cands):
    for i in range(81):
        if POPCOUNT[cands[i]] != 1:
            return False
    return True

def _bm_search(cands, solutions, max_solutions):
    if len(solutions) >= max_solutions:
        return
    best_i, best_n = -1, 10
    for i in range(81):
        n = POPCOUNT[cands[i]]
        if 1 < n < best_n:
            best_i = i
            best_n = n
    if best_i == -1:
        solutions.append([BIT_VAL[cands[i]] for i in range(81)])
        return
    c = cands[best_i]
    while c:
        bit = LOWEST_BIT[c]
        cp = list(cands)
        if _bm_assign(cp, best_i, BIT_VAL[bit]):
            _bm_search(cp, solutions, max_solutions)
        c &= ~bit

def _is_unique(grid_flat):
    cands = _bm_init(grid_flat)
    if cands is None:
        return False
    if _bm_solved(cands):
        return True
    solutions = []
    _bm_search(cands, solutions, max_solutions=2)
    return len(solutions) == 1


# ── Grid generation (flat arrays) ──

def _generate_grid(rng):
    grid = [0] * 81
    base = list(range(1, 10))
    rng.shuffle(base)
    shifts = [0, 3, 6, 1, 4, 7, 2, 5, 8]
    for r in range(9):
        for c in range(9):
            grid[r * 9 + c] = base[(c + shifts[r]) % 9]
    for _ in range(30):
        t = rng.randint(0, 4)
        if t == 0:
            band = rng.randint(0, 2)
            r1, r2 = rng.sample(range(band*3, band*3+3), 2)
            for c in range(9):
                grid[r1*9+c], grid[r2*9+c] = grid[r2*9+c], grid[r1*9+c]
        elif t == 1:
            stack = rng.randint(0, 2)
            c1, c2 = rng.sample(range(stack*3, stack*3+3), 2)
            for r in range(9):
                grid[r*9+c1], grid[r*9+c2] = grid[r*9+c2], grid[r*9+c1]
        elif t == 2:
            b1, b2 = rng.sample(range(3), 2)
            for i in range(3):
                for c in range(9):
                    grid[(b1*3+i)*9+c], grid[(b2*3+i)*9+c] = grid[(b2*3+i)*9+c], grid[(b1*3+i)*9+c]
        elif t == 3:
            s1, s2 = rng.sample(range(3), 2)
            for r in range(9):
                for i in range(3):
                    grid[r*9+s1*3+i], grid[r*9+s2*3+i] = grid[r*9+s2*3+i], grid[r*9+s1*3+i]
        else:
            perm = list(range(1, 10))
            rng.shuffle(perm)
            grid = [perm[v - 1] for v in grid]
    return grid


def _make_puzzle(sol, n_blanks, rng):
    puzzle = list(sol)
    cells = list(range(81))
    rng.shuffle(cells)
    removed = 0
    for i in cells:
        if removed >= n_blanks:
            break
        val = puzzle[i]
        puzzle[i] = 0
        if _is_unique(puzzle):
            removed += 1
        else:
            puzzle[i] = val
    return puzzle if removed >= n_blanks else None


# ── Depth computation ──

def compute_prop_depth(puzzle_flat):
    grid = list(puzzle_flat)
    blanks = {i for i in range(81) if grid[i] == 0}
    depths = {}
    current_depth = 0
    def _cands(i):
        if grid[i] != 0: return set()
        used = {grid[p] for p in PEERS_FLAT[i]} - {0}
        return set(range(1, 10)) - used
    while True:
        newly = []
        for i in blanks - set(depths.keys()):
            c = _cands(i)
            if len(c) == 1:
                grid[i] = c.pop()
                depths[i] = current_depth
                newly.append(i)
        for group_cells in ALL_GROUPS:
            gi = [r*9+c for r, c in group_cells]
            unfilled = [i for i in gi if grid[i] == 0 and i not in depths]
            if not unfilled: continue
            for val in range(1, 10):
                if any(grid[i] == val for i in gi): continue
                possible = [i for i in unfilled if val in _cands(i)]
                if len(possible) == 1:
                    i = possible[0]
                    if i not in depths:
                        grid[i] = val
                        depths[i] = current_depth
                        newly.append(i)
        if not newly: break
        current_depth += 1
    for i in blanks:
        if i not in depths:
            depths[i] = -1
    return depths


def compute_n_candidates(puzzle_flat):
    result = {}
    for i in range(81):
        if puzzle_flat[i] == 0:
            used = {puzzle_flat[p] for p in PEERS_FLAT[i]} - {0}
            result[i] = 9 - len(used)
    return result


def compute_n_cands_after_cp(puzzle_flat):
    """For depth_hard cells: how many candidates remain after full CP?
    This gives a gradient within depth_hard:
      n_cands_cp=2 → almost determined, one guess away
      n_cands_cp=5 → very uncertain, deep reasoning needed
    Returns dict: flat_idx → n_candidates_after_cp (only for hard cells).
    """
    cands = _bm_init(puzzle_flat)
    if cands is None:
        return {}
    result = {}
    for i in range(81):
        if puzzle_flat[i] == 0 and POPCOUNT[cands[i]] > 1:
            result[i] = POPCOUNT[cands[i]]
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Data Format & Generation (multiprocessing + disk cache)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _flat_idx_to_rc(idx):
    return idx // 9, idx % 9

def _rc_to_flat_idx(r, c):
    return r * 9 + c

def _parse_sudoku(s):
    parts = s.split('=')
    return parts[0], parts[1]


def _gen_one_puzzle(args):
    """Worker function for multiprocessing."""
    seed, min_blanks, max_blanks = args
    rng = random.Random(seed)
    sol = _generate_grid(rng)
    nb = rng.randint(min_blanks, max_blanks)
    puzzle = _make_puzzle(sol, nb, rng)
    if puzzle is None:
        return None
    n_blanks = sum(1 for v in puzzle if v == 0)
    depths = compute_prop_depth(puzzle)
    n_cands = compute_n_candidates(puzzle)
    n_cands_cp = compute_n_cands_after_cp(puzzle)
    puzzle_str = ''.join(str(v) for v in puzzle)
    sol_str = ''.join(str(v) for v in sol)
    meta = {}
    for i in range(81):
        is_given = puzzle[i] != 0
        meta[i] = {
            'is_given': is_given,
            'prop_depth': -99 if is_given else depths.get(i, -1),
            'n_candidates': 0 if is_given else n_cands.get(i, 0),
            'n_cands_after_cp': 0 if is_given else n_cands_cp.get(i, 0),
            'row': i // 9, 'col': i % 9,
            'box': (i // 9 // 3) * 3 + (i % 9) // 3,
        }
    return {'string': f"{puzzle_str}={sol_str}", 'meta': meta, 'n_blanks': n_blanks}


def gen_sudoku_data(n, seed, min_blanks=None, max_blanks=None):
    """Generate n puzzles with multiprocessing + disk caching."""
    if min_blanks is None: min_blanks = MIN_BLANKS
    if max_blanks is None: max_blanks = MAX_BLANKS

    # Disk cache
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.sudoku_cache')
    cache_key = f"n{n}_s{seed}_bl{min_blanks}-{max_blanks}"
    cache_path = os.path.join(cache_dir, f"{cache_key}.json")
    if os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                data = json.load(f)
            for d in data:
                d['meta'] = {int(k): v for k, v in d['meta'].items()}
            print(f"    Loaded {len(data)} puzzles from cache")
            return data
        except Exception:
            pass

    from concurrent.futures import ProcessPoolExecutor
    n_workers = min(os.cpu_count() or 1, 16)
    n_attempt = int(n * 1.5) + 200
    args_list = [(seed * 100000 + i, min_blanks, max_blanks) for i in range(n_attempt)]

    results = []
    t0 = time.time()
    if n_workers > 1 and n > 50:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            cs = max(n_attempt // (n_workers * 4), 1)
            for r in pool.map(_gen_one_puzzle, args_list, chunksize=cs):
                if r is not None:
                    results.append(r)
                if len(results) >= n:
                    break
    else:
        for a in args_list:
            r = _gen_one_puzzle(a)
            if r is not None:
                results.append(r)
            if len(results) >= n:
                break

    elapsed = time.time() - t0
    if len(results) < n:
        print(f"    WARNING: generated {len(results)}/{n}")
    results = results[:n]
    print(f"    Generated {len(results)} puzzles in {elapsed:.1f}s "
          f"({elapsed/max(len(results),1)*1000:.0f}ms each, {n_workers} workers)")

    try:
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump(results, f)
        print(f"    Cached to {cache_path}")
    except Exception as e:
        print(f"    Cache write failed: {e}")

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
                       max_len, device=None,
                       test_blank_masks=None, test_cat_ids=None,
                       ids_all=None, ans_all=None):
    """Fully-masked probe with difficulty-category tracking.
    Accepts precomputed tensors to avoid Python loops in hot path.
    """
    if device is None: device = DEVICE
    model.eval()
    mask_id = tokenizer.special_ids['mask']
    N_test = len(test_data)

    # Use precomputed encoded data or encode on the fly
    if ids_all is None or ans_all is None:
        strings = [d['string'] for d in test_data]
        ids_all, ans_all = encode_samples(strings, tokenizer, max_len)
        ids_all, ans_all = ids_all.to(device), ans_all.to(device)

    # Use precomputed or build on the fly
    if test_blank_masks is None or test_cat_ids is None:
        test_blank_masks_l = torch.zeros(N_test, ANS_LEN, dtype=torch.bool, device=device)
        test_cat_ids_l = torch.zeros(N_test, ANS_LEN, dtype=torch.long, device=device)
        for si, d in enumerate(test_data):
            puzzle_str = d['string'].split('=')[0]
            for j in range(ANS_LEN):
                if puzzle_str[j] == '0':
                    test_blank_masks_l[si, j] = True
                test_cat_ids_l[si, j] = DEPTH_CAT_TO_ID[_depth_to_cat(d['meta'][j])]
        blank_masks_t = test_blank_masks_l
        cat_ids = test_cat_ids_l
    else:
        blank_masks_t = test_blank_masks[:N_test]
        cat_ids = test_cat_ids[:N_test]

    # Accumulators
    total_loss = torch.zeros(ANS_LEN, device=device)
    total_correct = torch.zeros(ANS_LEN, device=device)
    total_conf = torch.zeros(ANS_LEN, device=device)
    total_n = torch.zeros(ANS_LEN, device=device)

    # Per-category accumulators (on GPU for speed)
    n_cats = len(DEPTH_CATS)
    cat_conf_acc = torch.zeros(n_cats, device=device)
    cat_correct_acc = torch.zeros(n_cats, device=device)
    cat_count_acc = torch.zeros(n_cats, dtype=torch.long, device=device)

    BS = 128  # probe batch size
    for st in range(0, N_test, BS):
        en = min(st + BS, N_test)
        ids = ids_all[st:en]
        ans = ans_all[st:en]
        B = ids.shape[0]
        T = ids.shape[1]

        ans_pos = ans.unsqueeze(1) + torch.arange(ANS_LEN, device=device)
        ans_pos = ans_pos.clamp(max=T-1)
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand_as(ans_pos)
        bl = blank_masks_t[st:en]  # [B, ANS_LEN]

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
            w = valid.float()
        else:
            # Mask only blank positions
            xm = ids.clone()
            xm[batch_idx[bl], ans_pos[bl]] = mask_id
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
            w = bl.float()

        # Vectorized accumulation (no per-position loop)
        total_loss += (losses * w).sum(dim=0)
        total_correct += (corrects * w).sum(dim=0)
        total_conf += (confs * w).sum(dim=0)
        total_n += w.sum(dim=0)

        # Per-category accumulation (vectorized scatter)
        cat_batch = cat_ids[st:en]  # [B, ANS_LEN]
        w_flat = w.reshape(-1)
        valid_flat = w_flat > 0
        cat_flat = cat_batch.reshape(-1)
        confs_flat = confs.reshape(-1)
        corrects_flat = corrects.reshape(-1)
        if valid_flat.any():
            vc = cat_flat[valid_flat]
            cat_conf_acc.scatter_add_(0, vc, confs_flat[valid_flat])
            cat_correct_acc.scatter_add_(0, vc, corrects_flat[valid_flat])
            cat_count_acc.scatter_add_(0, vc, torch.ones_like(vc, dtype=torch.long))

    N_safe = total_n.clamp(min=1)
    pos_loss = (total_loss / N_safe).tolist()
    pos_acc = (total_correct / N_safe).tolist()
    pos_conf = (total_conf / N_safe).tolist()
    total_n_sum = max(total_n.sum().item(), 1)
    overall_loss = total_loss.sum().item() / total_n_sum
    overall_acc = total_correct.sum().item() / total_n_sum

    # Per-category summary
    depth_context = {}
    for ci, cname in enumerate(DEPTH_CATS):
        n = cat_count_acc[ci].item()
        if n > 0:
            depth_context[cname] = {
                'mean_conf': cat_conf_acc[ci].item() / n,
                'mean_acc': cat_correct_acc[ci].item() / n,
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

    # ── Digit-level constraint tracking ──
    # When cell X is revealed as value V:
    #   same-group cells: P(V) should DECREASE (V is now forbidden)
    #   diff-group cells: P(V) should NOT change
    # This is a direct test of constraint-consistent inference.
    digit_elim_same = []    # ΔP(revealed_digit) for same-group cells
    digit_elim_diff = []    # ΔP(revealed_digit) for diff-group cells
    # Also: does P(correct answer) increase for same-group cells?
    gold_boost_same = []    # ΔP(gold_digit) for same-group cells
    gold_boost_diff = []    # ΔP(gold_digit) for diff-group cells

    for si, sample in enumerate(samples):
        s = sample['string']
        meta = sample['meta']

        puzzle_str, sol_str = _parse_sudoku(s)
        prefix = puzzle_str + '='
        penc = tokenizer.encode(prefix)
        sol_enc = tokenizer.encode(sol_str)
        T_pre = len(penc)

        # Build input: prefix + solution (given cells filled, blanks masked)
        x = torch.tensor(penc + sol_enc, dtype=torch.long, device=device).unsqueeze(0)
        unmasked = torch.ones(T_pre + ANS_LEN, dtype=torch.bool, device=device)
        blank_js = set()
        for j in range(ANS_LEN):
            if puzzle_str[j] == '0':
                x[0, T_pre + j] = mask_id
                unmasked[T_pre + j] = False
                blank_js.add(j)

        # Precompute cell peer sets (flat index → set of flat indices)
        peer_flat = {}
        for j in range(81):
            r, c = _flat_idx_to_rc(j)
            peer_flat[j] = {_rc_to_flat_idx(pr, pc) for pr, pc in PEERS[(r, c)]}

        # Initial confidence and probs (only for blank positions)
        logits = model(x)
        prev_conf = torch.zeros(ANS_LEN, device=device)
        prev_probs = {}  # j → prob vector (for digit-level tracking)
        sol_enc = tokenizer.encode(sol_str)
        for j in blank_js:
            cl = logits[0, T_pre + j].clone()
            cl[mask_id] = -float('inf')
            p = F.softmax(cl, dim=-1)
            prev_conf[j] = p.max()
            prev_probs[j] = p.clone()

        # Step-by-step decode (confidence policy, blank positions only)
        for step in range(len(blank_js)):
            logits = model(x)

            # Find highest-confidence masked (blank) position
            best_conf, best_j = -1.0, -1
            for j in blank_js:
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
            revealed_tok = cl.argmax()
            x[0, pos] = revealed_tok
            unmasked[pos] = True
            # The digit value that was revealed (token ID)
            revealed_digit_id = revealed_tok.item()

            # New confidences and probs
            logits_new = model(x)
            new_conf = torch.zeros(ANS_LEN, device=device)
            new_probs = {}
            for j in blank_js:
                if unmasked[T_pre + j]:
                    new_conf[j] = 1.0
                    continue
                cl2 = logits_new[0, T_pre + j].clone()
                cl2[mask_id] = -float('inf')
                p = F.softmax(cl2, dim=-1)
                new_conf[j] = p.max()
                new_probs[j] = p

            # Classify revealed cell
            rev_cat = _depth_to_cat(meta[best_j])
            rev_peers = peer_flat[best_j]

            # Record Δconf AND digit-level changes for remaining cells
            for j in blank_js:
                if unmasked[T_pre + j] or j == best_j:
                    continue
                delta = (new_conf[j] - prev_conf[j]).item()
                obs_cat = _depth_to_cat(meta[j])

                # Digit-level: ΔP(revealed_digit) — should be negative for same-group
                if j in prev_probs and j in new_probs:
                    dp_revealed = (new_probs[j][revealed_digit_id]
                                   - prev_probs[j][revealed_digit_id]).item()
                    # ΔP(gold) — P(correct answer) change
                    gold_id = sol_enc[j]
                    dp_gold = (new_probs[j][gold_id]
                               - prev_probs[j][gold_id]).item()
                else:
                    dp_revealed = 0.0
                    dp_gold = 0.0

                if j in rev_peers:
                    delta_same_group.append(delta)
                    delta_by_cat[rev_cat]['same'].append(delta)
                    digit_elim_same.append(dp_revealed)
                    gold_boost_same.append(dp_gold)
                else:
                    delta_diff_group.append(delta)
                    delta_by_cat[rev_cat]['diff'].append(delta)
                    digit_elim_diff.append(dp_revealed)
                    gold_boost_diff.append(dp_gold)

                delta_obs_by_cat[obs_cat].append(delta)

            prev_conf = new_conf
            prev_probs = new_probs

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
        # Digit-level constraint consistency
        'digit_elim_same': _mean(digit_elim_same),   # should be negative
        'digit_elim_diff': _mean(digit_elim_diff),   # should be ~0
        'gold_boost_same': _mean(gold_boost_same),    # should be positive
        'gold_boost_diff': _mean(gold_boost_diff),    # should be ~0
        'n_digit_same': len(digit_elim_same),
        'n_digit_diff': len(digit_elim_diff),
    }
    sg = result['same_group_delta']
    dg = result['diff_group_delta']
    if dg != 0:
        ratio_str = f"ratio={sg/dg:.2f}x"
    else:
        ratio_str = "ratio=inf"
    de_s = result['digit_elim_same']
    de_d = result['digit_elim_diff']
    gb_s = result['gold_boost_same']
    gb_d = result['gold_boost_diff']
    result['summary'] = (
        f"Constraint cascade: same_group={sg:+.4f} diff_group={dg:+.4f} {ratio_str}\n"
        f"    Digit elimination: same_group ΔP(revealed)={de_s:+.4f}, "
        f"diff_group ΔP(revealed)={de_d:+.4f}\n"
        f"    Gold probability:  same_group ΔP(gold)={gb_s:+.4f}, "
        f"diff_group ΔP(gold)={gb_d:+.4f}"
    )
    return result


@torch.no_grad()
def probe_selective_masking(model, tokenizer, test_data, max_len, device=None):
    """Test parallel reasoning ability by comparing accuracy under different masking.

    Conditions:
      all_blank:   all blank cells masked (standard probe)
      hard_only:   only depth_hard cells masked, depth_0/1/2 revealed as ground truth
      easy_only:   only depth_0 cells masked, harder cells revealed

    If hard_only acc ≈ all_blank acc for hard cells → model doesn't rely on
    sequential propagation through easy cells (parallel reasoning).
    If hard_only acc >> all_blank acc for hard cells → model needs easy cells
    as stepping stones (sequential reasoning).
    """
    if device is None: device = DEVICE
    model.eval()
    mask_id = tokenizer.special_ids['mask']

    strings = [d['string'] for d in test_data]
    metas = [d['meta'] for d in test_data]
    ids_all, ans_all = encode_samples(strings, tokenizer, max_len)
    ids_all, ans_all = ids_all.to(device), ans_all.to(device)
    N_test = len(test_data)

    conditions = {
        'all_blank': lambda m: not m['is_given'],
        'hard_only': lambda m: m['prop_depth'] == -1,
        'easy_only': lambda m: m['prop_depth'] == 0 and not m['is_given'],
    }

    results = {}
    for cond_name, mask_fn in conditions.items():
        cat_correct = defaultdict(list)

        for st in range(0, N_test, 64):
            en = min(st + 64, N_test)
            ids = ids_all[st:en]
            ans = ans_all[st:en]
            B = ids.shape[0]
            T = ids.shape[1]

            ans_pos = ans.unsqueeze(1) + torch.arange(ANS_LEN, device=device)
            ans_pos = ans_pos.clamp(max=T-1)
            batch_idx = torch.arange(B, device=device).unsqueeze(1).expand_as(ans_pos)

            mask_cells = torch.zeros(B, ANS_LEN, dtype=torch.bool, device=device)
            for bi in range(B):
                si = st + bi
                for j in range(ANS_LEN):
                    if mask_fn(metas[si][j]):
                        mask_cells[bi, j] = True

            xm = ids.clone()
            xm[batch_idx[mask_cells], ans_pos[mask_cells]] = mask_id
            logits = model(xm)
            ans_logits = logits[batch_idx, ans_pos]
            tgt_ids = ids[batch_idx, ans_pos]
            cl = ans_logits.clone()
            cl[:, :, mask_id] = -float('inf')
            preds = cl.argmax(dim=-1)
            corrects = (preds == tgt_ids)

            for bi in range(B):
                si = st + bi
                for j in range(ANS_LEN):
                    if mask_cells[bi, j]:
                        cat = _depth_to_cat(metas[si][j])
                        cat_correct[cat].append(corrects[bi, j].item())

        results[cond_name] = {
            cat: sum(v)/len(v) for cat, v in cat_correct.items() if v
        }

    return results


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

    # Precompute blank masks: [N, ANS_LEN] — True = blank (needs prediction)
    # Only blank positions receive masking during training, so ALL gradient
    # signal goes to constraint solving rather than trivial "copy given" tasks.
    blank_masks = torch.zeros(N, ANS_LEN, dtype=torch.bool, device=device)
    for si, d in enumerate(train_data):
        puzzle_str = d['string'].split('=')[0]
        for j in range(ANS_LEN):
            blank_masks[si, j] = (puzzle_str[j] == '0')
    n_blank_total = blank_masks.sum().item()
    n_given_total = N * ANS_LEN - n_blank_total
    print(f"    Blank-only masking: {n_blank_total} blank positions, "
          f"{n_given_total} given (skipped), "
          f"avg {n_blank_total/N:.1f} blanks/puzzle")

    # Precompute reusable tensors
    T = train_ids.shape[1]
    _arange_ans = torch.arange(ANS_LEN, device=device)

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
                # Only mask blank positions
                for j in range(ANS_LEN):
                    p = a_s + j
                    if p < T and blank_masks[si, j]:
                        puma_z[bi, p] = mask_id
                puma_stage[bi] = 0

        def _chain_advance(logits, K_cur):
            nonlocal puma_stage
            B_buf = puma_z.shape[0]
            ans_pos = puma_ans.unsqueeze(1) + _arange_ans
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
                rank_of_pos.scatter_(1, ranked, _arange_ans.expand(B_buf, -1))
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

    # Precompute test data tensors (once, reused by every eval)
    test_strings = [d['string'] for d in test_data]
    test_ids_all, test_ans_all = encode_samples(test_strings, tokenizer, max_len)
    test_ids_all = test_ids_all.to(device)
    test_ans_all = test_ans_all.to(device)
    N_test = len(test_data)
    test_blank_masks = torch.zeros(N_test, ANS_LEN, dtype=torch.bool, device=device)
    test_cat_ids = torch.zeros(N_test, ANS_LEN, dtype=torch.long, device=device)
    for si, d in enumerate(test_data):
        puzzle_str = d['string'].split('=')[0]
        for j in range(ANS_LEN):
            if puzzle_str[j] == '0':
                test_blank_masks[si, j] = True
            test_cat_ids[si, j] = DEPTH_CAT_TO_ID[_depth_to_cat(d['meta'][j])]

    def _do_eval(epoch):
        nonlocal best_loss, best_state
        t_eval = time.time()
        probe = probe_per_position(
            model, tokenizer, test_data, objective, max_len, device,
            test_blank_masks=test_blank_masks, test_cat_ids=test_cat_ids,
            ids_all=test_ids_all, ans_all=test_ans_all)
        dynamics['checkpoints'].append({
            'epoch': epoch, 'iter': it, 'token_gradients': tg, **probe})
        pl = probe['overall_loss']
        if pl < best_loss and epoch > 0:
            best_loss = pl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        # Log depth context
        dc = probe.get('depth_context', {})
        dc_parts = []
        for cat in DEPTH_CATS:
            if cat in dc and cat != 'given':
                dc_parts.append(f"{cat}={dc[cat]['mean_acc']:.3f}")
        dc_str = ' '.join(dc_parts) if dc_parts else ''
        print(f"    [eval ep {epoch}] loss={pl:.4f} acc={probe['overall_acc']:.4f} "
              f"{dc_str} | {time.time()-t_eval:.0f}s")

    def _do_gen_eval(epoch):
        """Lightweight generation eval with decode order analysis."""
        t_gen = time.time()
        try:
            r = final_evaluate(model, tokenizer, test_data[:GEN_EVAL_N],
                               'diffusion', decode_policy='confidence', batch_size=16)
            entry = {'epoch': epoch, 'gen_acc': r['accuracy'],
                     'blank_cell_acc': r.get('blank_cell_acc', 0)}
            oa = r.get('decode_order_analysis')
            if oa:
                entry['depth_vs_rank'] = oa.get('depth_vs_rank')
                entry['difficulty_concordance'] = oa.get('difficulty_concordance')
                entry['spearman_depth_rank'] = oa.get('spearman_depth_rank')
                entry['spearman_ncands_rank'] = oa.get('spearman_ncands_rank')
                entry['given_mean_rank'] = oa.get('given_mean_rank')
                entry['blank_mean_rank'] = oa.get('blank_mean_rank')
                entry['n_cands_vs_rank'] = oa.get('n_cands_vs_rank')
            dynamics['gen_checkpoints'].append(entry)
            dc_str = ''
            if oa:
                conc = entry.get('difficulty_concordance', 0)
                sp_d = entry.get('spearman_depth_rank')
                sp_nc = entry.get('spearman_ncands_rank')
                dc_str = f"conc={conc:.3f}"
                if sp_d is not None: dc_str += f" ρ_depth={sp_d:.3f}"
                if sp_nc is not None: dc_str += f" ρ_ncands={sp_nc:.3f}"
            bl_acc = r.get('blank_cell_acc', 0)
            print(f"    [gen ep {epoch}] acc={r['accuracy']:.4f} blank={bl_acc:.4f} "
                  f"{dc_str} | {time.time()-t_gen:.0f}s")
        except Exception as e:
            print(f"    [gen ep {epoch}] FAILED: {e}")

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
                    ans_pos = ans_starts.unsqueeze(1) + _arange_ans
                    ans_pos = ans_pos.clamp(max=T_batch-1)
                    batch_idx = torch.arange(B, device=device).unsqueeze(1).expand_as(ans_pos)

                    if mask_type == 'random':
                        bl = blank_masks[idx]  # [B, ANS_LEN]
                        t_ratio = torch.rand(B, device=device)
                        m_probs = torch.zeros(B, T_batch, dtype=torch.float, device=device)
                        m_probs[batch_idx, ans_pos] = t_ratio.unsqueeze(1) * bl.float()
                        m = torch.bernoulli(m_probs).bool()
                        # Vectorized no-mask fallback: pick random blank per sample
                        no_m = ~(m.any(dim=1))
                        if no_m.any():
                            n_no = no_m.sum()
                            bl_no = bl[no_m]  # [n_no, ANS_LEN]
                            # Sample one blank per row using Gumbel trick
                            rand_scores = torch.rand_like(bl_no.float())
                            rand_scores[~bl_no] = -1.0  # exclude non-blank
                            chosen_j = rand_scores.argmax(dim=1)  # [n_no]
                            chosen_abs = ans_pos[no_m].gather(1, chosen_j.unsqueeze(1)).squeeze(1)
                            m[no_m.nonzero(as_tuple=True)[0], chosen_abs] = True
                    elif mask_type == 'confidence':
                        bl = blank_masks[idx]  # [B, ANS_LEN]
                        xm_probe = ids.clone()
                        # Only mask blank positions for probe
                        blank_pos_mask = torch.zeros(B, T_batch, dtype=torch.bool, device=device)
                        blank_pos_mask[batch_idx, ans_pos] = bl
                        xm_probe[blank_pos_mask] = mask_id
                        model.eval()
                        with torch.no_grad():
                            logits_probe = model(xm_probe)
                        model.train()
                        lp = logits_probe[batch_idx, ans_pos]
                        lp[:, :, mask_id] = -float('inf')
                        confs = F.softmax(lp, dim=-1).max(dim=-1).values
                        confs[~bl] = float('inf')  # given positions → never mask
                        ranked = confs.argsort(dim=1)  # ascending: lowest conf first
                        n_blanks_per = bl.sum(dim=1).float()
                        t_ratio = torch.rand(B, device=device)
                        nm = (t_ratio * n_blanks_per).ceil().long().clamp(min=1)
                        rank_of_pos = torch.zeros_like(ranked)
                        rank_of_pos.scatter_(1, ranked, _arange_ans.expand(B, -1))
                        to_mask = (rank_of_pos < nm.unsqueeze(1)) & bl
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

    # Per-category accuracy (blank categories only for meaningful comparison)
    cat_correct = defaultdict(list)
    for r in results:
        for j in range(ANS_LEN):
            cat = _depth_to_cat(r['meta'][j])
            if cat != 'given':
                cat_correct[cat].append(r['pos_correct'][j])
    cat_acc = {cat: sum(v)/len(v) for cat, v in cat_correct.items() if v}

    # Overall blank-only accuracy (per-cell, not per-puzzle)
    blank_correct_total = 0
    blank_total = 0
    for r in results:
        for j in range(ANS_LEN):
            if not r['meta'][j]['is_given']:
                blank_total += 1
                blank_correct_total += r['pos_correct'][j]
    blank_cell_acc = blank_correct_total / max(blank_total, 1)

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
        'accuracy': acc, 'blank_cell_acc': blank_cell_acc,
        'n_samples': n, 'position_accuracy': pos_acc,
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
    n_cands_cp_t = torch.zeros(N, ANS_LEN, dtype=torch.long)
    is_blank = torch.zeros(N, ANS_LEN, dtype=torch.bool)
    for si in range(N):
        for j in range(ANS_LEN):
            m = metas[si][j]
            cat_ids[si, j] = DEPTH_CAT_TO_ID[_depth_to_cat(m)]
            n_cands_t[si, j] = m['n_candidates']
            n_cands_cp_t[si, j] = m.get('n_cands_after_cp', 0)
            is_blank[si, j] = not m['is_given']

    cat_v = cat_ids[valid_mask].reshape(-1)
    rank_v = rop_valid.reshape(-1)
    nc_v = n_cands_t[valid_mask].reshape(-1)
    nc_cp_v = n_cands_cp_t[valid_mask].reshape(-1)
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

    # n_cands_after_cp vs rank (depth_hard cells only — gradient within hard)
    hard_mask = blank_mask & (cat_v == DEPTH_CAT_TO_ID['depth_hard'])
    n_cands_cp_vs_rank = {}
    for nc_val in nc_cp_v[hard_mask].unique().tolist():
        if nc_val == 0: continue
        m = hard_mask & (nc_cp_v == nc_val)
        if m.any():
            n_cands_cp_vs_rank[int(nc_val)] = rank_v[m].mean().item()

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

    # ── Spearman rank correlations (blank cells only, per-sample then averaged) ──
    # More robust than concordance: directly measures monotonic relationship
    # between difficulty measures and decode order.
    spearman_depth_rank = []      # ρ(prop_depth, decode_rank)
    spearman_ncands_rank = []     # ρ(n_candidates, decode_rank)
    for si in range(n_valid):
        orig_si = valid_mask.nonzero(as_tuple=True)[0][si].item()
        meta = metas[orig_si]
        ranks = rop_valid[si]
        # Collect blank cell data for this sample
        b_depths, b_ncands, b_ranks = [], [], []
        for j in range(ANS_LEN):
            if not meta[j]['is_given']:
                d = meta[j]['prop_depth']
                b_depths.append(d if d >= 0 else 100)  # -1 → hardest
                b_ncands.append(meta[j]['n_candidates'])
                b_ranks.append(ranks[j].item())
        if len(b_depths) < 5:
            continue  # too few blanks for meaningful correlation

        def _spearman(x, y):
            """Spearman ρ between two lists."""
            n = len(x)
            if n < 3: return None
            # Rank transform
            def _rank(vals):
                order = sorted(range(n), key=lambda i: vals[i])
                r = [0.0] * n
                for rank_val, idx in enumerate(order):
                    r[idx] = rank_val
                return r
            rx, ry = _rank(x), _rank(y)
            mx = sum(rx) / n
            my = sum(ry) / n
            cov = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
            sx = sum((rx[i] - mx)**2 for i in range(n)) ** 0.5
            sy = sum((ry[i] - my)**2 for i in range(n)) ** 0.5
            if sx == 0 or sy == 0: return 0.0
            return cov / (sx * sy)

        rho_d = _spearman(b_depths, b_ranks)
        rho_nc = _spearman(b_ncands, b_ranks)
        if rho_d is not None:
            spearman_depth_rank.append(rho_d)
        if rho_nc is not None:
            spearman_ncands_rank.append(rho_nc)

    avg_spearman_depth = (sum(spearman_depth_rank) / len(spearman_depth_rank)
                          if spearman_depth_rank else None)
    avg_spearman_ncands = (sum(spearman_ncands_rank) / len(spearman_ncands_rank)
                           if spearman_ncands_rank else None)

    return {
        'depth_vs_rank': depth_vs_rank,
        'difficulty_concordance': difficulty_concordance,
        'spearman_depth_rank': avg_spearman_depth,
        'spearman_ncands_rank': avg_spearman_ncands,
        'given_mean_rank': given_mean_rank,
        'blank_mean_rank': blank_mean_rank,
        'n_cands_vs_rank': n_cands_vs_rank,
        'n_cands_cp_vs_rank': n_cands_cp_vs_rank,
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

    # OOD test sets (distribution-shifted)
    print("  Generating OOD test data (harder)...")
    test_hard = gen_sudoku_data(max(N_TEST // 2, 100), seed=9100,
                                 min_blanks=56, max_blanks=64)
    print(f"  Test-hard: {len(test_hard)} puzzles (56-64 blanks)")

    print("  Generating OOD test data (easier)...")
    test_easy = gen_sudoku_data(max(N_TEST // 2, 100), seed=9200,
                                 min_blanks=30, max_blanks=40)
    print(f"  Test-easy: {len(test_easy)} puzzles (30-40 blanks)")

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
            print(f"  Acc: {r['accuracy']:.4f}  Blank cell acc: {r.get('blank_cell_acc', 0):.4f}")
            oa = r.get('decode_order_analysis')
            if oa:
                dvr = oa.get('depth_vs_rank', {})
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
                # Spearman correlations
                sp_d = oa.get('spearman_depth_rank')
                sp_nc = oa.get('spearman_ncands_rank')
                sp_parts = []
                if sp_d is not None: sp_parts.append(f"ρ(depth,rank)={sp_d:.3f}")
                if sp_nc is not None: sp_parts.append(f"ρ(n_cands,rank)={sp_nc:.3f}")
                if sp_parts:
                    print(f"    Spearman: {' '.join(sp_parts)}")
                nc_cp_r = oa.get('n_cands_cp_vs_rank', {})
                if nc_cp_r:
                    parts = [f"nc_cp={k}→{v:.1f}" for k, v in sorted(nc_cp_r.items())]
                    print(f"    Hard cells CP-cands→rank: {' '.join(parts)}")
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

        # ── Selective masking probe (parallel reasoning test) ──
        print(f"  Selective masking probe...")
        sel = probe_selective_masking(m, tok, test_data, max_len, device=DEVICE)
        all_final[_fk('diffusion', mt, 'selective')] = sel
        for cond, accs in sel.items():
            parts = [f"{cat}={v:.3f}" for cat, v in accs.items()]
            print(f"    {cond}: {' '.join(parts)}")

        # ── OOD evaluation (distribution-shifted test sets) ──
        for ood_name, ood_data in [('harder', test_hard), ('easier', test_easy)]:
            if not ood_data:
                continue
            print(f"  OOD eval ({ood_name}, {len(ood_data)} puzzles)...")
            r_ood = final_evaluate(m, tok, ood_data, 'diffusion',
                                   decode_policy='confidence', batch_size=16)
            all_final[_fk('diffusion', mt, f'ood_{ood_name}')] = r_ood
            bl_acc = r_ood.get('blank_cell_acc', 0)
            print(f"    acc={r_ood['accuracy']:.4f} blank_cell={bl_acc:.4f}")
            ca = r_ood.get('category_accuracy', {})
            if ca:
                parts = [f"{cat}={v:.3f}" for cat, v in ca.items()]
                print(f"    Cat: {' '.join(parts)}")
            oa = r_ood.get('decode_order_analysis')
            if oa:
                sp_d = oa.get('spearman_depth_rank')
                dc = oa.get('difficulty_concordance')
                if sp_d is not None and dc is not None:
                    print(f"    conc={dc:.3f} ρ_depth={sp_d:.3f}")

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
            bl_acc = r.get('blank_cell_acc', 0)
            print(f"  {key:<40s} acc={r['accuracy']:.4f}  blank_cell={bl_acc:.4f}")
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
