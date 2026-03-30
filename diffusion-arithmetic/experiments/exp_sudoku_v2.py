"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Sudoku v2 — PUMA vs Random on Difficulty-Stratified Data
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Core question: Does PUMA's exploration failure on rare hard patterns
  mirror the addition carry-in rarity effect?

  Data: HuggingFace sapientinc/sudoku-extreme (tdoku backtrack ratings)
        or local CSV / self-generated fallback
  Training: easy-dominant (configurable easy:hard ratio)
  Eval: rating-stratified accuracy × multiple decode policies
  Decode: confidence | n_cands | n_cands_cp | random
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import sys, os, time, math, json, random, copy
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
from core.model import Transformer
from core.train_utils import mount_drive, save_results, encode_samples, DEVICE

EXP_NAME = 'exp_sudoku_v2'

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Config
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ANS_LEN = 81

# Data (easy-dominant by default)
N_EASY = 45000          # rating = 0 (CP-solvable)
N_HARD = 5000           # rating >= HARD_THRESHOLD
N_TEST_PER_TIER = 500
HARD_THRESHOLD = 10     # tdoku backtracks
DATA_SOURCE = 'hf'      # 'hf', 'csv', 'generate'
CSV_PATH = ''

# Model (PUMA paper Sudoku config)
N_LAYER = 8; N_HEAD = 8; N_EMBD = 256
DROPOUT = 0.0; POS_ENC = 'absolute'

# Training
MAX_ITERS = 60000; BATCH_SIZE = 64
LR = 3e-4; MIN_LR = 1e-5; WARMUP_ITERS = 1000
GRAD_CLIP = 1.0; WEIGHT_DECAY = 0.01
EMA_DECAY = 0.9999
EVAL_EVERY = 2000; LOG_EVERY = 500; GEN_EVAL_EVERY = 5000; GEN_EVAL_N = 100

MASK_TYPES = ['random', 'puma']
DECODE_POLICIES = ['confidence', 'n_cands', 'random']
PUMA_TAU = 0.9; PUMA_K = 8
SEED = 42

# Rating tiers for analysis
RATING_TIERS = {'easy': (0, 0), 'medium': (1, 9), 'hard': (10, 99), 'extreme': (100, 99999)}


def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data-source', default=None, choices=['hf', 'csv', 'generate'])
    p.add_argument('--csv-path', default=None)
    p.add_argument('--n-easy', type=int, default=None)
    p.add_argument('--n-hard', type=int, default=None)
    p.add_argument('--hard-threshold', type=int, default=None)
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
    p.add_argument('--tag', type=str, default='')
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--seeds', nargs='+', type=int, default=None)
    args = p.parse_args()
    g = globals()
    for a, gl in {'data_source': 'DATA_SOURCE', 'csv_path': 'CSV_PATH',
                   'n_easy': 'N_EASY', 'n_hard': 'N_HARD', 'hard_threshold': 'HARD_THRESHOLD',
                   'n_test': 'N_TEST_PER_TIER', 'max_iters': 'MAX_ITERS',
                   'batch_size': 'BATCH_SIZE', 'n_layer': 'N_LAYER', 'n_head': 'N_HEAD',
                   'n_embd': 'N_EMBD', 'dropout': 'DROPOUT', 'lr': 'LR',
                   'puma_k': 'PUMA_K', 'puma_tau': 'PUMA_TAU', 'seed': 'SEED'}.items():
        v = getattr(args, a); g[gl] = v if v is not None else g[gl]
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

DEPTH_CATS = ['given', 'depth_0', 'depth_1', 'depth_2', 'depth_3plus', 'depth_hard']
DEPTH_CAT_TO_ID = {n: i for i, n in enumerate(DEPTH_CATS)}

def _depth_to_cat(meta):
    if meta['is_given']: return 'given'
    d = meta['prop_depth']
    if d == 0: return 'depth_0'
    if d == 1: return 'depth_1'
    if d == 2: return 'depth_2'
    if d >= 3: return 'depth_3plus'
    return 'depth_hard'


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
    meta = {}
    for i in range(81):
        is_given = puzzle_flat[i] != 0
        meta[i] = {
            'is_given': is_given,
            'prop_depth': -99 if is_given else depths.get(i, -1),
            'n_candidates': 0 if is_given else n_cands.get(i, 0),
            'n_cands_after_cp': 0 if is_given else n_cands_cp.get(i, 0),
        }
    # Normalize puzzle string to use '0' for blanks
    pstr = ''.join(str(v) for v in puzzle_flat)
    return {'string': f"{pstr}={sol_str}", 'meta': meta, 'n_blanks': n_blanks}


def _compute_meta_worker(args):
    """Worker for parallel meta computation."""
    puzzle_str, sol_str, rating = args
    d = _compute_puzzle_meta(puzzle_str, sol_str)
    d['rating'] = rating
    return d


def load_hf_data(n_easy, n_hard, n_test, hard_threshold, seed=42, cache_dir='.sudoku_cache'):
    """Load from HuggingFace sapientinc/sudoku-extreme."""
    print("  Loading HuggingFace sudoku-extreme dataset...")
    from datasets import load_dataset
    ds = load_dataset('sapientinc/sudoku-extreme', cache_dir=cache_dir)

    rng = random.Random(seed)
    train_raw = ds['train']
    test_raw = ds['test']

    # Convert to lists for sampling
    def _sample_by_rating(data, lo, hi, n, rng_inst):
        indices = [i for i in range(len(data)) if lo <= data[i]['rating'] <= hi]
        if len(indices) > n:
            indices = rng_inst.sample(indices, n)
        return [(data[i]['puzzle'], data[i]['solution'], data[i]['rating']) for i in indices]

    # Train: easy-dominant mix
    easy_train = _sample_by_rating(train_raw, 0, 0, n_easy, rng)
    hard_train = _sample_by_rating(train_raw, hard_threshold, 999999, n_hard, rng)
    train_tuples = easy_train + hard_train
    rng.shuffle(train_tuples)

    # Test: separate tiers
    rng2 = random.Random(seed + 1000)
    test_tiers = {}
    for tier_name, (lo, hi) in RATING_TIERS.items():
        samples = _sample_by_rating(test_raw, lo, hi, n_test, rng2)
        if not samples:  # try train set if test doesn't have this tier
            samples = _sample_by_rating(train_raw, lo, hi, n_test, rng2)
        test_tiers[tier_name] = samples

    print(f"  Train: {len(easy_train)} easy + {len(hard_train)} hard = {len(train_tuples)}")
    for tn, ts in test_tiers.items():
        print(f"  Test/{tn}: {len(ts)}")

    return train_tuples, test_tiers


def load_csv_data(csv_path, n_easy, n_hard, n_test, hard_threshold, seed=42):
    """Load from local CSV (puzzle,solution,rating or puzzle,solution,difficulty)."""
    import csv
    print(f"  Loading CSV: {csv_path}")
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            p = row.get('puzzle', row.get('quizzes', ''))
            s = row.get('solution', row.get('solutions', ''))
            r = int(row.get('rating', row.get('difficulty', 0)))
            if len(p) == 81 and len(s) == 81:
                rows.append((p, s, r))
    print(f"  Loaded {len(rows)} puzzles from CSV")
    rng = random.Random(seed)
    rng.shuffle(rows)
    easy = [r for r in rows if r[2] <= 0][:n_easy]
    hard = [r for r in rows if r[2] >= hard_threshold][:n_hard]
    train_tuples = easy + hard
    rng.shuffle(train_tuples)
    # Simple test split from remainder
    used = set(id(t) for t in train_tuples)
    rest = [r for r in rows if id(r) not in used]
    test_tiers = {}
    for tn, (lo, hi) in RATING_TIERS.items():
        test_tiers[tn] = [r for r in rest if lo <= r[2] <= hi][:n_test]
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


def prepare_datasets(source=None, seed=42):
    """Load data and compute per-puzzle metadata in parallel."""
    source = source or DATA_SOURCE
    if source == 'hf':
        train_tuples, test_tiers = load_hf_data(
            N_EASY, N_HARD, N_TEST_PER_TIER, HARD_THRESHOLD, seed)
    elif source == 'csv':
        train_tuples, test_tiers = load_csv_data(
            CSV_PATH, N_EASY, N_HARD, N_TEST_PER_TIER, HARD_THRESHOLD, seed)
    else:
        raw = gen_fallback_data(N_EASY + N_HARD, seed)
        train_tuples = raw
        test_raw = gen_fallback_data(N_TEST_PER_TIER * 2, seed + 5000)
        test_tiers = {'all': test_raw}

    # Compute metadata in parallel
    print("  Computing puzzle metadata...")
    t0 = time.time()
    workers = min(os.cpu_count() or 1, 16)
    all_tuples = train_tuples + [t for ts in test_tiers.values() for t in ts]

    with ProcessPoolExecutor(max_workers=workers) as pool:
        all_data = list(pool.map(_compute_meta_worker, all_tuples,
                                  chunksize=max(1, len(all_tuples) // (workers*4))))

    n_train = len(train_tuples)
    train_data = all_data[:n_train]
    idx = n_train
    test_data = {}
    for tn, ts in test_tiers.items():
        test_data[tn] = all_data[idx:idx+len(ts)]
        idx += len(ts)

    print(f"  Metadata computed in {time.time()-t0:.1f}s")

    # Diagnostics
    for name, data in [('train', train_data)] + list(test_data.items()):
        if not data: continue
        ratings = [d.get('rating', 0) for d in data]
        depth_dist = defaultdict(int)
        for d in data:
            for j in range(81):
                depth_dist[_depth_to_cat(d['meta'][j])] += 1
        print(f"  [{name}] n={len(data)} rating: "
              f"min={min(ratings)} med={sorted(ratings)[len(ratings)//2]} max={max(ratings)} "
              f"depth: {dict(sorted(depth_dist.items()))}")

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

def train(mask_type, tokenizer, train_data, test_data_dict, max_len, device=None):
    """Iteration-based training with EMA. Returns (model, ema_state, dynamics)."""
    if device is None: device = DEVICE
    strings = [d['string'] for d in train_data]
    train_ids, train_ans = encode_samples(strings, tokenizer, max_len)
    train_ids, train_ans = train_ids.to(device), train_ans.to(device)
    N, T = train_ids.shape
    mask_id = tokenizer.special_ids['mask']
    pad_id = tokenizer.special_ids['pad']

    # Blank masks [N, ANS_LEN]
    blank_masks = torch.zeros(N, ANS_LEN, dtype=torch.bool, device=device)
    for si, d in enumerate(train_data):
        ps = d['string'].split('=')[0]
        for j in range(ANS_LEN):
            if ps[j] == '0': blank_masks[si, j] = True

    _arange = torch.arange(ANS_LEN, device=device)
    model = Transformer(vocab_size=len(tokenizer), block_size=max_len+8,
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

    # Pick one test tier for periodic eval (prefer 'hard' if available)
    eval_tier = 'hard' if 'hard' in test_data_dict else list(test_data_dict.keys())[0]
    eval_data = test_data_dict[eval_tier]

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
            idx_t = torch.tensor(indices, device=device)
            n = len(indices)
            if buf_ptr + n > len(buf_pool):
                buf_pool = torch.randperm(N); buf_ptr = 0
            si = buf_pool[buf_ptr:buf_ptr+n].to(device); buf_ptr += n
            buf_x0[idx_t] = train_ids[si]
            buf_z[idx_t] = train_ids[si].clone()
            buf_ans[idx_t] = train_ans[si]
            buf_stage[idx_t] = 0
            # Mask blank positions
            ap = (buf_ans[idx_t].unsqueeze(1) + _arange).clamp(max=T-1)
            bii = idx_t.unsqueeze(1).expand_as(ap)
            bl = blank_masks[si]  # [n, ANS_LEN]
            buf_z[bii[bl], ap[bl]] = mask_id

        def _advance(logits):
            nonlocal buf_stage
            B_buf = BATCH_SIZE
            ap = (buf_ans.unsqueeze(1) + _arange).clamp(max=T-1)
            bi = torch.arange(B_buf, device=device).unsqueeze(1).expand_as(ap)
            is_m = (buf_z[bi, ap] == mask_id)
            if not is_m.any(): _refresh(list(range(B_buf))); return
            lp = logits[bi, ap].clone(); lp[:, :, mask_id] = -float('inf')
            confs = F.softmax(lp, dim=-1).max(dim=-1).values
            confs[~is_m] = -float('inf')
            nm = is_m.sum(dim=1).float()
            nr = (nm / max(PUMA_K, 1)).ceil().long().clamp(min=1)
            ranked = confs.argsort(dim=1, descending=True)
            rop = torch.zeros_like(ranked)
            rop.scatter_(1, ranked, _arange.expand(B_buf, -1))
            reveal = ((rop < nr.unsqueeze(1)) | (confs > PUMA_TAU)) & is_m
            buf_z[bi[reveal], ap[reveal]] = buf_x0[bi[reveal], ap[reveal]]
            buf_stage += 1
            done = (~(buf_z[bi, ap] == mask_id).any(dim=1)) | (buf_stage >= PUMA_K)
            if done.any(): _refresh(done.nonzero(as_tuple=True)[0].tolist())

        _refresh(list(range(BATCH_SIZE)))

    # ── Training loop ──
    perm = torch.randperm(N, device=device); perm_ptr = 0

    def _next_batch():
        nonlocal perm, perm_ptr
        if perm_ptr + BATCH_SIZE > N:
            perm = torch.randperm(N, device=device); perm_ptr = 0
        idx = perm[perm_ptr:perm_ptr+BATCH_SIZE]; perm_ptr += BATCH_SIZE
        return idx

    for it in range(1, MAX_ITERS + 1):
        for pg in optimizer.param_groups: pg['lr'] = get_lr(it)

        if uses_streaming:
            m = (buf_z == mask_id)
            if m.sum() == 0: _refresh(list(range(BATCH_SIZE))); m = (buf_z == mask_id)
            logits = model(buf_z)
            loss = F.cross_entropy(logits[m], buf_x0[m])
            tg += m.sum().item()
        else:
            idx = _next_batch()
            ids = train_ids[idx]; ans_starts = train_ans[idx]
            B_b = ids.shape[0]
            ap = (ans_starts.unsqueeze(1) + _arange).clamp(max=T-1)
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

        if uses_streaming: _advance(logits.detach())

        # Logging
        if it % LOG_EVERY == 0:
            dynamics['train_loss'].append((it, loss.item()))
            print(f"    it {it:6d}/{MAX_ITERS} | loss {loss.item():.4f} | "
                  f"lr {get_lr(it):.1e} | tg {tg:,} | {time.time()-t0:.0f}s")

        # Eval
        if it % EVAL_EVERY == 0 or (it <= MAX_ITERS * 0.1 and it % max(EVAL_EVERY//5, 1) == 0):
            # Swap in EMA for eval
            orig_state = {k: v.clone() for k, v in model.state_dict().items()}
            model.load_state_dict(ema_state)
            model.eval()
            probe = probe_per_cell(model, tokenizer, eval_data, max_len, device)
            dynamics['checkpoints'].append({'iter': it, 'tg': tg, **probe})
            dc = probe.get('depth_context', {})
            parts = [f"{c}={dc[c]['mean_acc']:.3f}" for c in DEPTH_CATS if c in dc and c != 'given']
            print(f"    [eval it {it}] loss={probe['overall_loss']:.4f} "
                  f"acc={probe['overall_acc']:.4f} {' '.join(parts)}")
            if probe['overall_loss'] < best_loss:
                best_loss = probe['overall_loss']
                best_ema = {k: v.clone() for k, v in ema_state.items()}
            model.load_state_dict(orig_state)
            model.train()

    # Load best EMA
    if best_ema:
        model.load_state_dict({k: v.to(device) for k, v in best_ema.items()})
    model.eval()
    print(f"  Done {MAX_ITERS} iters (best probe loss: {best_loss:.4f})")
    return model, ema_state, dynamics


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Generation (multi-policy decode)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@torch.no_grad()
def generate_blanks(model, tokenizer, test_data, decode_policy='confidence',
                    batch_size=32, device=None):
    """Decode blank cells with configurable policy.
    Policies: 'confidence' (model max prob), 'n_cands' (fewest initial candidates first),
              'n_cands_cp' (fewest CP candidates first), 'random'.
    """
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

        # Pre-fill givens, mask blanks
        x = ids.clone()
        blank_m = torch.zeros(B, ANS_LEN, dtype=torch.bool, device=device)
        for i in range(B):
            ps = batch[i]['string'].split('=')[0]
            for j in range(ANS_LEN):
                if ps[j] == '0': x[i, ans_starts[i]+j] = mask_id; blank_m[i,j] = True
        n_blanks = blank_m.sum(dim=1)
        max_steps = n_blanks.max().item()

        # Precompute static decode order for n_cands / n_cands_cp / random
        static_order = None
        if decode_policy in ('n_cands', 'n_cands_cp', 'random'):
            static_order = torch.full((B, ANS_LEN), 9999, dtype=torch.long, device=device)
            for i in range(B):
                meta = batch[i]['meta']
                blank_js = [j for j in range(ANS_LEN) if not meta[j]['is_given']]
                if decode_policy == 'random':
                    random.shuffle(blank_js)
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
            else:
                # Use static order: pick the cell with lowest static_order that is still masked
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
            }

    return {'accuracy': acc, 'blank_cell_acc': bl_acc, 'n': n,
            'category_accuracy': cat_acc, 'rating_accuracy': rating_acc}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Analysis
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def analyse_rating_correlation(data):
    """Correlate our depth/n_cands metrics with external tdoku rating."""
    features, ratings = [], []
    for d in data:
        r = d.get('rating', 0)
        meta = d['meta']
        n_hard = sum(1 for j in range(81) if meta[j]['prop_depth'] == -1)
        blanks = [j for j in range(81) if not meta[j]['is_given']]
        if not blanks: continue
        n_cands_cp = [meta[j]['n_cands_after_cp'] for j in blanks if meta[j]['n_cands_after_cp'] > 0]
        max_depth = max((meta[j]['prop_depth'] for j in blanks if meta[j]['prop_depth'] >= 0), default=0)
        features.append({
            'n_depth_hard': n_hard,
            'n_blanks': len(blanks),
            'mean_n_cands_cp': sum(n_cands_cp)/len(n_cands_cp) if n_cands_cp else 0,
            'max_depth': max_depth,
            'frac_hard': n_hard / max(len(blanks), 1),
        })
        ratings.append(r)

    if len(ratings) < 10: return {}

    def _corr(xs, ys):
        n = len(xs); mx, my = sum(xs)/n, sum(ys)/n
        c = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
        sx = sum((x-mx)**2 for x in xs)**0.5
        sy = sum((y-my)**2 for y in ys)**0.5
        return c/(sx*sy) if sx > 0 and sy > 0 else 0.0

    correlations = {}
    for feat_name in ['n_depth_hard', 'n_blanks', 'mean_n_cands_cp', 'max_depth', 'frac_hard']:
        xs = [f[feat_name] for f in features]
        correlations[feat_name] = _corr(xs, ratings)

    # Map rating ranges to SE-equivalent tiers (conceptual)
    se_mapping = {
        'rating=0 → SE<4 (singles only)': sum(1 for r in ratings if r == 0),
        'rating=1-9 → SE 4-7 (moderate search)': sum(1 for r in ratings if 1 <= r <= 9),
        'rating=10-99 → SE 7-10 (hard techniques)': sum(1 for r in ratings if 10 <= r <= 99),
        'rating=100+ → SE 10+ (extreme)': sum(1 for r in ratings if r >= 100),
    }

    return {'correlations': correlations, 'se_mapping': se_mapping,
            'features': features, 'ratings': ratings}


@torch.no_grad()
def probe_selective_masking(model, tokenizer, test_data, max_len, device=None):
    """Compare accuracy with different masking: all_blank vs hard_only vs easy_only."""
    if device is None: device = DEVICE
    model.eval()
    mask_id = tokenizer.special_ids['mask']
    strings = [d['string'] for d in test_data]
    metas = [d['meta'] for d in test_data]
    ids_all, ans_all = encode_samples(strings, tokenizer, max_len)
    ids_all, ans_all = ids_all.to(device), ans_all.to(device)
    N = len(test_data); _ar = torch.arange(ANS_LEN, device=device)

    conditions = {
        'all_blank': lambda m: not m['is_given'],
        'hard_only': lambda m: m['prop_depth'] == -1,
        'easy_only': lambda m: m['prop_depth'] == 0 and not m['is_given'],
    }
    results = {}
    for cn, fn in conditions.items():
        cat_correct = defaultdict(list)
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
                    if mc[b, j]:
                        cat_correct[_depth_to_cat(metas[st+b][j])].append(corrects[b,j].item())
        results[cn] = {c: sum(v)/len(v) for c, v in cat_correct.items() if v}
    return results


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

    return figs


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Run
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run(tag=''):
    exp_name = f"{EXP_NAME}_{tag}" if tag else EXP_NAME
    print(f"\n{'='*70}")
    print(f"  {exp_name}")
    print(f"  Model: {N_LAYER}L/{N_EMBD}D/{N_HEAD}H  Data: {N_EASY}easy+{N_HARD}hard")
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

    # ── Rating correlation analysis ──
    print(f"\n{'━'*60}\n  Rating Correlation Analysis\n{'━'*60}")
    all_data_for_corr = train_data + [d for ds in test_data.values() for d in ds]
    corr = analyse_rating_correlation(all_data_for_corr)
    if corr.get('correlations'):
        for fn, r in corr['correlations'].items():
            print(f"    corr({fn}, tdoku_rating) = {r:.3f}")
    if corr.get('se_mapping'):
        for desc, n in corr['se_mapping'].items():
            print(f"    {desc}: n={n}")

    # ── Merge all test tiers for eval ──
    # Keep separate by tier for rating-stratified eval
    all_test = [d for ds in test_data.values() for d in ds]

    all_results = {}
    all_dyn = {}

    for mt in MASK_TYPES:
        print(f"\n{'━'*60}\n▶ Training: {mt}\n{'━'*60}")
        model, ema, dyn = train(mt, tok, train_data, test_data, max_len, device=DEVICE)
        all_dyn[mt] = dyn

        # ── Multi-policy eval ──
        for dp in DECODE_POLICIES:
            key = f'{mt}_{dp}'
            print(f"\n  Eval: {key}")
            r = evaluate(model, tok, all_test, decode_policy=dp, batch_size=32, device=DEVICE)
            all_results[key] = r
            print(f"    acc={r['accuracy']:.4f}  blank_cell={r['blank_cell_acc']:.4f}")
            ca = r.get('category_accuracy', {})
            if ca:
                parts = [f"{c}={v:.3f}" for c, v in ca.items()]
                print(f"    Cat: {' '.join(parts)}")
            ra = r.get('rating_accuracy', {})
            for tn, info in ra.items():
                print(f"    {tn}: exact={info['exact']:.4f} cell={info['cell']:.4f} (n={info['n']})")

        # ── Selective masking ──
        print(f"  Selective masking probe...")
        sel = probe_selective_masking(model, tok, all_test[:200], max_len, device=DEVICE)
        all_results[f'{mt}_selective'] = sel
        for cn, accs in sel.items():
            parts = [f"{c}={v:.3f}" for c, v in accs.items()]
            print(f"    {cn}: {' '.join(parts)}")

        # ── PUMA coverage (puma model only) ──
        if mt == 'puma':
            print(f"  PUMA coverage simulation...")
            hard_test = test_data.get('hard', test_data.get('all', all_test))[:100]
            cov = simulate_puma_coverage(model, tok, hard_test, max_len, device=DEVICE)
            all_results[f'{mt}_coverage'] = cov
            for cn, info in cov.items():
                print(f"    {cn}: coverage={info['coverage']:.3f} (n={info['n']})")

        del model; torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── Figures ──
    print(f"\n{'='*70}\n  Generating figures...\n{'='*70}")
    figs = make_figures(all_results, all_dyn, corr)

    # ── Summary table ──
    print(f"\n{'='*70}\n  SUMMARY\n{'='*70}")
    header = f"  {'Condition':<25}"
    for tn in RATING_TIERS:
        has = any(tn in all_results.get(f'{mt}_{dp}', {}).get('rating_accuracy', {})
                  for mt in MASK_TYPES for dp in DECODE_POLICIES)
        if has: header += f" {tn:>10}"
    print(header)
    print(f"  {'─'*70}")
    for mt in MASK_TYPES:
        for dp in DECODE_POLICIES:
            key = f'{mt}_{dp}'
            r = all_results.get(key)
            if not r: continue
            row = f"  {key:<25}"
            for tn in RATING_TIERS:
                ra = r.get('rating_accuracy', {}).get(tn)
                if ra: row += f" {ra['exact']:>10.4f}"
            print(row)

    # ── Save ──
    sd = {'config': {k: globals()[k] for k in ['N_EASY', 'N_HARD', 'HARD_THRESHOLD',
           'N_LAYER', 'N_EMBD', 'N_HEAD', 'MAX_ITERS', 'BATCH_SIZE',
           'MASK_TYPES', 'DECODE_POLICIES', 'PUMA_K', 'PUMA_TAU']}}
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
