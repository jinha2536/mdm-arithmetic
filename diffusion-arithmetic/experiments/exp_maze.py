"""Maze path labelling on a GRID_N x GRID_N logical maze (rendered as a
(2 GRID_N + 1)^2 wall/corridor grid).

The prompt is the maze with start (S) and goal (E); the model labels every
corridor cell as on-path (1) or off-path (0). Only corridor cells are masked
during training and decoding. Training schemes: random / PAPL / PUMA masking.
Decoding (one cell per forward pass): confidence, dead-end filling (the
polynomial-time maze solver, used as the dependency-respecting reference order)
and uniform random. Test mazes are stratified by the longest corridor on the
start-to-goal path (CORRIDOR_SWEEP, N_PER_BUCKET mazes per stratum).
"""
import sys, os, time, random
from collections import defaultdict, deque
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                if '__file__' in dir() else '.')
from core.tokenizer import CharTokenizer
from core.train_utils import (
    prepare_results_dir, save_results, save_checkpoint, encode_samples,
    train_diffusion, puma_k_step, DEVICE,
)

EXP_NAME = 'exp_maze'

# Config
GRID_N = 10                 # logical size; grid is (2*10+1)^2 = 21 x 21 = 441 cells
GRID_H = 2 * GRID_N + 1
GRID_W = GRID_H
CELL_N = GRID_H * GRID_W
ANS_LEN = CELL_N            # the solution grid has the same size as the puzzle

N_TRAIN = 10000; BATCH_SIZE = 256
N_TEST = 5000               # held-out mazes for the fully-masked probe
N_PER_BUCKET = 300          # mazes per corridor-length stratum
MAX_ITERS = 50000; EVAL_EVERY = 5000; LOG_EVERY = 1000

MASK_TYPES = ['random', 'papl', 'puma']
DECODE_POLICIES = ['confidence', 'dead_end_filling', 'random']

N_LAYER = 3; N_HEAD = 3; N_EMBD = 192; DROPOUT = 0.1
LR = 3e-4; MIN_LR = 1e-5; WARMUP_ITERS = 2000; GRAD_CLIP = 1.0
WEIGHT_DECAY = 0.01; EMA_DECAY = 0.9999

PUMA_TAU = 0.9
# PUMA K schedule K_START -> K_END (~10 cells per step at K_END); K_EVERY=None
# ramps over the first 1/3 of training
PUMA_K_START = 10; PUMA_K_END = 40; PUMA_K_STEP = 3; PUMA_K_EVERY = None
PAPL_TAU = 1.0; PAPL_ALPHA = 5.0
SEED = 42
NO_AMP = True   # bf16 autocast off (fp32 training)

# minimum longest-corridor lengths of the test strata
CORRIDOR_SWEEP = [2, 4, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60]


def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--n-train', type=int); p.add_argument('--n-test', type=int)
    p.add_argument('--max-iters', type=int); p.add_argument('--batch-size', type=int)
    p.add_argument('--eval-every', type=int)
    p.add_argument('--n-layer', type=int); p.add_argument('--n-head', type=int)
    p.add_argument('--n-embd', type=int); p.add_argument('--dropout', type=float)
    p.add_argument('--lr', type=float)
    p.add_argument('--puma-tau', type=float)
    p.add_argument('--puma-k-start', type=int); p.add_argument('--puma-k-end', type=int)
    p.add_argument('--puma-k-step', type=int); p.add_argument('--puma-k-every', type=int)
    p.add_argument('--papl-tau', type=float); p.add_argument('--papl-alpha', type=float)
    p.add_argument('--masks', nargs='+'); p.add_argument('--decode', nargs='+')
    p.add_argument('--no-amp', action='store_true')
    p.add_argument('--tag', type=str, default=''); p.add_argument('--seed', type=int)
    p.add_argument('--seeds', nargs='+', type=int)
    args, _ = p.parse_known_args()
    g = globals()
    for a, gl in {'n_train': 'N_TRAIN', 'n_test': 'N_TEST', 'max_iters': 'MAX_ITERS',
                   'batch_size': 'BATCH_SIZE', 'eval_every': 'EVAL_EVERY',
                   'n_layer': 'N_LAYER', 'n_head': 'N_HEAD', 'n_embd': 'N_EMBD',
                   'dropout': 'DROPOUT', 'lr': 'LR', 'puma_tau': 'PUMA_TAU',
                   'puma_k_start': 'PUMA_K_START', 'puma_k_end': 'PUMA_K_END',
                   'puma_k_step': 'PUMA_K_STEP', 'puma_k_every': 'PUMA_K_EVERY',
                   'papl_tau': 'PAPL_TAU', 'papl_alpha': 'PAPL_ALPHA',
                   'seed': 'SEED'}.items():
        v = getattr(args, a, None)
        if v is not None: g[gl] = v
    if args.no_amp: g['NO_AMP'] = True
    if args.masks: g['MASK_TYPES'] = args.masks
    if args.decode: g['DECODE_POLICIES'] = args.decode
    return args


# Maze generation

def _neighbors_2(r, c, H, W):
    """Passage neighbors 2 steps away (for DFS carving)."""
    for dr, dc in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
        nr, nc = r + dr, c + dc
        if 0 < nr < H and 0 < nc < W:
            yield nr, nc, dr, dc


def gen_maze_dfs(grid_n, rng, straightness_bias=0.0):
    """Perfect maze by randomized DFS carving on a (2 grid_n + 1)^2 grid.
    straightness_bias > 0 prefers continuing in the same direction (longer
    corridors). Start = top-left passage cell, end = bottom-right passage cell.
    Returns (grid, start, end): flat list of '#'/'.' and two cell indices."""
    H = W = 2 * grid_n + 1
    grid = ['#'] * (H * W)

    def _set(r, c, v): grid[r * W + c] = v

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


def _open_neighbors(grid, ci, H, W):
    """Count open (non-wall) neighbors of cell ci."""
    r, c = ci // W, ci % W
    count = 0
    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < H and 0 <= nc < W and grid[nr * W + nc] != '#':
            count += 1
    return count


def max_corridor_len(grid, path, start, end, H, W):
    """Longest corridor on the start-to-goal path: maximal run of consecutive
    path cells other than start/end with fewer than 3 open neighbours (cells
    with >= 3 open neighbours are branching points)."""
    best, run = 0, 0
    for ci in path:
        if ci != start and ci != end and _open_neighbors(grid, ci, H, W) < 3:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best


def compute_dead_end_filling_order(grid, start, end, H, W):
    """Dead-end filling order over the open cells: degree-1 cells (other than
    start/end) are removed iteratively and ranked by removal order; the
    remaining cells (the backbone) are ranked afterwards by BFS from start.

    Returns:
        order: dict cell_index -> rank (lower = decoded first)
        n_fillable: number of dead-end-fillable cells; cells with
                    rank >= n_fillable form the backbone
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

    # remaining cells = backbone, ordered by BFS from start
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


# Data formatting

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


def _make_entry(grid, start, end, with_order=False):
    """{'string': 'puzzle=solution', 'max_corridor_len'} (+ 'de_filling_order')."""
    H, W = GRID_H, GRID_W
    path = find_path_bfs(grid, start, end, H, W)
    if not path:
        return None
    puzzle_str, sol_str = _maze_to_strings(grid, set(path), start, end)
    entry = {'string': f"{puzzle_str}={sol_str}",
             'max_corridor_len': max_corridor_len(grid, path, start, end, H, W)}
    if with_order:
        entry['de_filling_order'], _ = compute_dead_end_filling_order(grid, start, end, H, W)
    return entry


def build_tok():
    return CharTokenizer(list('#.01SE='), {'mask': 'M', 'pad': 'P'})


# Data generation

def gen_data(n, seed):
    """Mazes from plain randomized DFS (training set / probe set)."""
    rng = random.Random(seed)
    data = []
    for _ in range(int(n * 1.5)):
        if len(data) >= n: break
        grid, si, ei = gen_maze_dfs(GRID_N, rng, 0.0)
        entry = _make_entry(grid, si, ei)
        if entry: data.append(entry)
    return data[:n]


def gen_min_corridor_test(n, seed, min_corridor):
    """Rejection-sample mazes whose longest corridor has length >= min_corridor
    (DFS with a straightness bias that grows with min_corridor)."""
    rng = random.Random(seed)
    data = []
    bias = min(0.95, 0.3 + min_corridor * 0.03)
    for _ in range(n * 50):
        if len(data) >= n: break
        grid, si, ei = gen_maze_dfs(GRID_N, rng, straightness_bias=bias)
        entry = _make_entry(grid, si, ei, with_order=True)
        if entry and entry['max_corridor_len'] >= min_corridor:
            data.append(entry)
    if len(data) < n:
        print(f"  WARNING: corridor>={min_corridor}: {len(data)}/{n}")
    return data[:n]


def build_test_suite(seed=None):
    """probe: N_TEST DFS mazes (fully-masked probe during training);
    constructed['corridor_{L}']: N_PER_BUCKET mazes with longest corridor >= L."""
    if seed is None: seed = SEED + 1000
    suite = {'probe': gen_data(N_TEST, seed), 'constructed': {}}
    for L in CORRIDOR_SWEEP:
        ent = gen_min_corridor_test(N_PER_BUCKET, seed=seed + 300 + L, min_corridor=L)
        if ent:
            suite['constructed'][f'corridor_{L}'] = ent
    print(f"  Test suite: probe {len(suite['probe'])}, " +
          ', '.join(f"{k} {len(v)}" for k, v in suite['constructed'].items()))
    return suite


# Evaluation

@torch.no_grad()
def probe_per_cell(model, tokenizer, test_data, max_len, device=None):
    """Fully-masked probe: every corridor cell masked, one forward pass; mean NLL
    and argmax accuracy (mask token excluded) over the corridor cells."""
    if device is None: device = DEVICE
    model.eval()
    mask_id = tokenizer.special_ids['mask']

    strings = [d['string'] for d in test_data]
    ids_all, ans_all = encode_samples(strings, tokenizer, max_len)
    ids_all, ans_all = ids_all.to(device), ans_all.to(device)
    N = len(test_data)

    dot_id = tokenizer.encode('.')[0]
    _arange = torch.arange(ANS_LEN, device=device)
    blank_masks = (ids_all[:, :ANS_LEN] == dot_id).to(device)

    total_loss = torch.tensor(0.0, device=device)
    total_n = torch.tensor(0, dtype=torch.long, device=device)
    total_correct = 0.0

    for st in range(0, N, 64):
        en = min(st + 64, N)
        ids, ans = ids_all[st:en], ans_all[st:en]
        B, T = ids.shape
        ans_pos = (ans.unsqueeze(1) + _arange).clamp(max=T - 1)
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


@torch.no_grad()
def generate_blanks(model, tokenizer, test_data, decode_policy='confidence',
                    batch_size=32, device=None):
    """Decode the corridor cells one per forward pass. Position selection: max
    softmax probability among masked cells ('confidence'), the dead-end filling
    rank ('dead_end_filling') or a uniformly random order ('random'); the token
    is always the argmax. Returns per-maze exact-match flags."""
    if device is None: device = DEVICE
    mask_id = tokenizer.special_ids['mask']
    pad_id = tokenizer.special_ids['pad']
    dot_id = tokenizer.encode('.')[0]
    eq_id = tokenizer.encode('=')[0]
    model.eval(); correct = []
    _ar = torch.arange(ANS_LEN, device=device)

    for st in range(0, len(test_data), batch_size):
        batch = test_data[st:st + batch_size]; B = len(batch)
        full_enc = [tokenizer.encode(d['string']) for d in batch]
        ml = max(len(e) for e in full_enc)
        ids = torch.full((B, ml), pad_id, dtype=torch.long, device=device)
        for i, e in enumerate(full_enc):
            ids[i, :len(e)] = torch.tensor(e, device=device)

        ans_starts = (ids == eq_id).long().argmax(dim=1) + 1
        ap = (ans_starts.unsqueeze(1) + _ar).clamp(max=ml - 1)
        bi = torch.arange(B, device=device).unsqueeze(1).expand_as(ap)

        blank_m = (ids[:, :ANS_LEN] == dot_id)
        x = ids.clone()
        x[bi[blank_m], ap[blank_m]] = mask_id

        # fixed decode order for the non-confidence policies
        static_order = None
        if decode_policy in ('dead_end_filling', 'random'):
            static_order = torch.full((B, ANS_LEN), 9999, dtype=torch.long, device=device)
            for i in range(B):
                blank_js = blank_m[i].nonzero(as_tuple=True)[0].tolist()
                if decode_policy == 'random':
                    random.shuffle(blank_js)
                    for rank, j in enumerate(blank_js): static_order[i, j] = rank
                else:
                    oracle = batch[i]['de_filling_order']
                    for j in blank_js: static_order[i, j] = oracle.get(j, 9999)

        for step in range(int(blank_m.sum(dim=1).max().item())):
            is_m = blank_m & (x[bi, ap] == mask_id)
            if not is_m.any(): break
            logits = model(x)
            al = logits[bi, ap].clone(); al[:, :, mask_id] = -float('inf')
            probs = F.softmax(al, dim=-1)
            confs = probs.max(dim=-1).values; preds = probs.argmax(dim=-1)
            confs[~is_m] = -float('inf')

            if decode_policy == 'confidence':
                ranked = confs.argsort(dim=1, descending=True)
            else:
                rank_vals = torch.where(is_m, static_order,
                                        torch.tensor(9999, dtype=torch.long, device=device))
                ranked = rank_vals.argsort(dim=1)
            rop = torch.zeros_like(ranked)
            rop.scatter_(1, ranked, _ar.expand(B, -1))
            reveal = (rop < 1) & is_m           # one cell per maze per step
            x[bi[reveal], ap[reveal]] = preds[reveal]

        pos_correct = (x[bi, ap] == ids[bi, ap])
        correct.extend((pos_correct | ~blank_m).all(dim=1).tolist())
    return correct


def gen_accuracy(model, tokenizer, entries, decode_policy, device=None):
    ok = generate_blanks(model, tokenizer, entries, decode_policy=decode_policy, device=device)
    return {'accuracy': sum(ok) / max(len(ok), 1), 'n': len(ok)}


# Training

def train_model(mask_type, tokenizer, train_data, suite, max_len, device=None):
    """train_diffusion with the corridor cells as the maskable positions and the
    fully-masked probe on suite['probe'] as eval callback."""
    if device is None: device = DEVICE
    max_iters = MAX_ITERS

    strings = [d['string'] for d in train_data]
    train_ids, train_ans = encode_samples(strings, tokenizer, max_len)
    train_ids, train_ans = train_ids.to(device), train_ans.to(device)

    # only '.' cells are maskable
    dot_id = tokenizer.encode('.')[0]
    blank_masks = (train_ids[:, :ANS_LEN] == dot_id)
    probe_entries = suite['probe']

    k_sched = None
    if mask_type == 'puma':
        avg_blanks = blank_masks.sum(dim=1).float().mean().item()
        n_increments = max(1, (PUMA_K_END - PUMA_K_START) // PUMA_K_STEP)
        if PUMA_K_EVERY is not None:
            k_every = PUMA_K_EVERY
        else:
            k_every = max(1000, (max_iters // 3) // n_increments)
        k_sched = puma_k_step(PUMA_K_START, PUMA_K_END, PUMA_K_STEP, k_every)
        final_k = k_sched(max_iters)
        print(f"  PUMA K: {PUMA_K_START} -> {final_k} (+{PUMA_K_STEP} every {k_every//1000}k, "
              f"avg blanks={avg_blanks:.0f}, ~{avg_blanks/final_k:.1f} cells/step)")

    def eval_fn(model, it, tg):
        probe = probe_per_cell(model, tokenizer, probe_entries, max_len, device)
        print(f"    [eval it {it}] loss={probe['overall_loss']:.4f} "
              f"acc={probe['overall_acc']:.4f}")
        return probe

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


# Run

def run(tag=''):
    exp_name = f"{EXP_NAME}_{tag}" if tag else EXP_NAME
    prepare_results_dir()
    print(f"\n{'='*70}\n  {exp_name}\n{'='*70}")
    print(f"  Grid: {GRID_H}x{GRID_W} ({CELL_N} cells)")
    print(f"  Model: L={N_LAYER} H={N_HEAD} E={N_EMBD}")
    print(f"  Masks: {MASK_TYPES}, Decode: {DECODE_POLICIES}")

    torch.manual_seed(SEED); random.seed(SEED)
    tok = build_tok()
    max_len = 2 * CELL_N + 2  # puzzle + '=' + solution + margin

    t0 = time.time()
    train_data = gen_data(N_TRAIN, seed=SEED)
    print(f"  Train: {len(train_data)} mazes in {time.time()-t0:.1f}s")
    suite = build_test_suite(seed=SEED + 1000)

    all_dyn = {}; all_final = {}
    for mt in MASK_TYPES:
        print(f"\n{'='*60}\n  Training: {mt}\n{'='*60}")
        m, dyn = train_model(mt, tok, train_data, suite, max_len, device=DEVICE)
        all_dyn[f'dyn_{mt}'] = dyn
        save_checkpoint(exp_name, {k: v.cpu().clone() for k, v in m.state_dict().items()}, tag=mt)

        print("  Corridor strata...")
        for L in CORRIDOR_SWEEP:
            key = f'corridor_{L}'
            if key not in suite['constructed']: continue
            for dp in DECODE_POLICIES:
                r = gen_accuracy(m, tok, suite['constructed'][key], dp, device=DEVICE)
                all_final[f'{mt}_corridor_sweep_{L}_{dp}'] = {**r, 'min_corridor': L}
                print(f"    corridor>={L:2d} {dp}: {r['accuracy']:.4f}")

        del m; torch.cuda.empty_cache() if torch.cuda.is_available() else None

    sd = {'config': {k: globals()[k] for k in [
        'GRID_N', 'GRID_H', 'GRID_W', 'CELL_N', 'N_TRAIN', 'N_TEST', 'N_PER_BUCKET',
        'MAX_ITERS', 'BATCH_SIZE', 'N_LAYER', 'N_HEAD', 'N_EMBD', 'MASK_TYPES',
        'DECODE_POLICIES', 'PUMA_K_START', 'PUMA_K_END', 'PUMA_K_STEP', 'PUMA_K_EVERY',
        'PUMA_TAU', 'PAPL_TAU', 'PAPL_ALPHA', 'CORRIDOR_SWEEP', 'SEED']}}
    for k, v in all_dyn.items():
        sd[k] = {'checkpoints': v['checkpoints'], 'train_loss': v['train_loss']}
    for k, v in all_final.items():
        sd[f'final_{k}'] = v
    save_results(exp_name, sd)

    print(f"\n{'='*70}\n  SUMMARY (exact match)\n{'='*70}")
    print(f"\n  {'Test':<40s}", end='')
    for mt in MASK_TYPES: print(f" {mt:>14s}", end='')
    print()
    for dp in DECODE_POLICIES:
        for L in CORRIDOR_SWEEP:
            accs = [all_final.get(f'{mt}_corridor_sweep_{L}_{dp}', {}).get('accuracy')
                    for mt in MASK_TYPES]
            if any(a is not None for a in accs):
                print(f"  {'corridor>='+str(L)+'_'+dp:<40s}", end='')
                for a in accs:
                    print(f" {a:>14.4f}" if a is not None else f" {'N/A':>14s}", end='')
                print()
    return all_dyn, all_final


if __name__ == '__main__':
    args = parse_args()
    seeds = args.seeds if args.seeds else [SEED]
    for si, seed in enumerate(seeds):
        globals()['SEED'] = seed
        t = (f"{args.tag}_s{seed}" if args.tag else f"s{seed}") if len(seeds) > 1 else args.tag
        if len(seeds) > 1:
            print(f"\n{'#'*70}\n# Seed {seed} ({si+1}/{len(seeds)})\n{'#'*70}")
        run(tag=t)
