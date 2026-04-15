"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ListOps — Hierarchical Dependency Learning + PUMA Coverage Deficit
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Task:     Evaluate nested prefix operations (MAX, MIN, MED, SM)
            and output the evaluation trace (inner→outer).
  Training: random vs puma masking (iteration-based, EMA)
  Decode:   confidence (model) vs l2r (oracle=inner→outer) vs random
  Analyses: depth dependency, nesting rarity × accuracy, PUMA coverage,
            corner cases, confidence cascade
  Continuation: random→puma, puma→random (representation persistence)

  Rarity axis: nesting depth (deep trees are rare in DEPTH_DECAY distribution)
  Dependency:  hierarchical — each operator depends on its sub-expression results
  Oracle:      l2r = inner-to-outer evaluation order (= trace order)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import sys, os, time, math, json, random, statistics
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
from core.train_utils import (
    mount_drive, save_results, save_checkpoint, encode_samples,
    train_diffusion, puma_k_fixed, puma_k_linear, puma_k_step,
    generate_diffusion, DEVICE,
)

EXP_NAME = 'exp_listops'

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Config
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tree generation
MAX_DEPTH = 5           # max nesting depth (root=depth 0)
MIN_ARGS = 2            # min arguments per operator
MAX_ARGS = 4            # max arguments per operator
EXPAND_PROB = 0.6       # P(argument becomes sub-expression) during generation
OPS = ['X', 'N', 'D', 'S']     # MAX, MIN, MED, SM
OP_NAMES = {'X': 'MAX', 'N': 'MIN', 'D': 'MED', 'S': 'SM'}

# Sequence format
MAX_ANS_LEN = 20        # fixed answer length (trace + PAD)
MAX_SEQ_LEN = 200       # total sequence length cap

# Distribution
DEPTH_DECAY = 0.8       # training: P(target_depth=d) ∝ DEPTH_DECAY^(d-1)

# Training
N_TRAIN = 20000; N_TEST = 1000; BATCH_SIZE = 256
MAX_ITERS = 400000; EVAL_EVERY = 5000; LOG_EVERY = 1000
GEN_EVAL_EVERY = 10000; GEN_EVAL_N = 500
MASK_TYPES = ['random', 'puma']
DECODE_POLICIES = ['confidence', 'l2r', 'random']   # l2r = oracle (inner→outer)
N_LAYER = 3; N_HEAD = 3; N_EMBD = 192; DROPOUT = 0.1; POS_ENC = 'absolute'
LR = 1e-3; MIN_LR = 1e-4; WARMUP_ITERS = 2000; GRAD_CLIP = 1.0
WEIGHT_DECAY = 0.1; EMA_DECAY = 0.9999
PUMA_TAU = 0.9; PUMA_K = 8
PUMA_K_START = None; PUMA_K_END = None
PUMA_K_STEP = 3; PUMA_K_EVERY = None
SEED = 42
NO_AMP = False
PATIENCE = 50000
CONTINUATION_ITERS = 10000


def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--max-depth', type=int); p.add_argument('--max-args', type=int)
    p.add_argument('--min-args', type=int); p.add_argument('--expand-prob', type=float)
    p.add_argument('--max-ans-len', type=int); p.add_argument('--depth-decay', type=float)
    p.add_argument('--n-train', type=int); p.add_argument('--n-test', type=int)
    p.add_argument('--max-iters', type=int); p.add_argument('--batch-size', type=int)
    p.add_argument('--eval-every', type=int); p.add_argument('--gen-eval-every', type=int)
    p.add_argument('--n-layer', type=int); p.add_argument('--n-head', type=int)
    p.add_argument('--n-embd', type=int); p.add_argument('--dropout', type=float)
    p.add_argument('--lr', type=float); p.add_argument('--weight-decay', type=float)
    p.add_argument('--patience', type=int)
    p.add_argument('--puma-tau', type=float); p.add_argument('--puma-k', type=int)
    p.add_argument('--puma-k-start', type=int); p.add_argument('--puma-k-end', type=int)
    p.add_argument('--puma-k-step', type=int); p.add_argument('--puma-k-every', type=int)
    p.add_argument('--masks', nargs='+'); p.add_argument('--decode', nargs='+')
    p.add_argument('--continuation-iters', type=int)
    p.add_argument('--no-continuation', action='store_true')
    p.add_argument('--no-amp', action='store_true')
    p.add_argument('--tag', type=str, default=''); p.add_argument('--seed', type=int)
    p.add_argument('--seeds', nargs='+', type=int)
    try:
        args, _ = p.parse_known_args()
    except SystemExit:
        args, _ = p.parse_known_args([])
    g = globals()
    for a, gl in {
        'max_depth': 'MAX_DEPTH', 'max_args': 'MAX_ARGS', 'min_args': 'MIN_ARGS',
        'expand_prob': 'EXPAND_PROB', 'max_ans_len': 'MAX_ANS_LEN',
        'depth_decay': 'DEPTH_DECAY',
        'n_train': 'N_TRAIN', 'n_test': 'N_TEST', 'max_iters': 'MAX_ITERS',
        'batch_size': 'BATCH_SIZE', 'eval_every': 'EVAL_EVERY',
        'gen_eval_every': 'GEN_EVAL_EVERY',
        'n_layer': 'N_LAYER', 'n_head': 'N_HEAD', 'n_embd': 'N_EMBD',
        'dropout': 'DROPOUT', 'lr': 'LR', 'weight_decay': 'WEIGHT_DECAY',
        'patience': 'PATIENCE', 'puma_tau': 'PUMA_TAU', 'puma_k': 'PUMA_K',
        'puma_k_start': 'PUMA_K_START', 'puma_k_end': 'PUMA_K_END',
        'puma_k_step': 'PUMA_K_STEP', 'puma_k_every': 'PUMA_K_EVERY',
        'seed': 'SEED', 'no_amp': 'NO_AMP', 'continuation_iters': 'CONTINUATION_ITERS',
    }.items():
        v = getattr(args, a, None)
        if v is not None:
            g[gl] = v
    if args.masks:
        g['MASK_TYPES'] = args.masks
    if args.decode:
        g['DECODE_POLICIES'] = args.decode
    return args


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tree data structure & operations
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tree node: int (literal 0-9) or dict {'op': str, 'args': list, 'depth': int}

def _eval_op(op, values):
    """Evaluate operator on a list of integer values → single digit 0-9."""
    if op == 'X':
        return max(values)
    elif op == 'N':
        return min(values)
    elif op == 'D':
        s = sorted(values)
        return s[len(s) // 2]       # upper-middle for even, true middle for odd
    elif op == 'S':
        return sum(values) % 10
    raise ValueError(f"Unknown op: {op}")


def _tree_to_str(node):
    """Convert tree to bracketed prefix string."""
    if isinstance(node, int):
        return str(node)
    parts = [node['op']] + [_tree_to_str(a) for a in node['args']]
    return '[' + ' '.join(parts) + ']'


def _evaluate(node):
    """
    Post-order evaluation. Returns (result, trace).
    trace: list of dicts, one per operator, in evaluation order (inner→outer).
    Each trace entry includes children_indices for DAG analysis.
    """
    if isinstance(node, int):
        return node, []
    child_results = []
    trace = []
    children_indices = []   # trace indices of direct sub-expression children
    for arg in node['args']:
        r, t = _evaluate(arg)
        child_results.append(r)
        if isinstance(arg, dict):
            # Sub-expression's root is at end of its sub-trace
            children_indices.append(len(trace) + len(t) - 1)
        trace.extend(t)
    result = _eval_op(node['op'], child_results)
    n_subexpr = sum(1 for a in node['args'] if isinstance(a, dict))
    trace.append({
        'result': result,
        'op': node['op'],
        'depth': node['depth'],
        'n_args': len(node['args']),
        'n_subexpr': n_subexpr,
        'children_indices': children_indices,
        'all_child_values': child_results,
    })
    return result, trace


def _tree_depth(node):
    """Max depth of the tree (root=0)."""
    if isinstance(node, int):
        return -1
    return max((0,) + tuple(1 + _tree_depth(a) for a in node['args'] if isinstance(a, dict)))


def _tree_n_ops(node):
    """Count operator nodes in tree."""
    if isinstance(node, int):
        return 0
    return 1 + sum(_tree_n_ops(a) for a in node['args'])


def _max_chain_depth(node, current=0):
    """Longest single-path nesting chain (like carry chain length)."""
    if isinstance(node, int):
        return current
    sub_chains = [_max_chain_depth(a, current + 1)
                  for a in node['args'] if isinstance(a, dict)]
    return max(sub_chains) if sub_chains else current


def _critical_chain_from_trace(trace):
    """
    Find the longest root→leaf dependency chain in the trace DAG.
    Returns (chain_length, chain_indices).
    This is the true sequential dependency length — the ListOps analog
    of carry chain length in addition.
    """
    n = len(trace)
    if n == 0:
        return 0, []
    # Build reverse DAG: for each node, what's the longest path to a leaf?
    longest = [1] * n  # each node is at least length 1
    best_child = [-1] * n
    # Process in trace order (leaves first, root last)
    for i in range(n):
        for ci in trace[i].get('children_indices', []):
            if ci < n and longest[ci] + 1 > longest[i]:
                longest[i] = longest[ci] + 1
                best_child[i] = ci
    # Root is last in trace
    root = n - 1
    chain = [root]
    cur = root
    while best_child[cur] >= 0:
        cur = best_child[cur]
        chain.append(cur)
    return longest[root], chain


def _build_dependency_pairs(trace):
    """
    Extract all (parent_idx, child_idx) dependency pairs from trace.
    A decode order violates dependency if parent is decoded before child.
    """
    pairs = []
    for i, t in enumerate(trace):
        for ci in t.get('children_indices', []):
            pairs.append((i, ci))  # i depends on ci
    return pairs


def _count_independent_groups(trace):
    """
    Count groups of mutually independent trace positions at each depth level.
    Returns dict: depth → count of positions at that depth.
    """
    depth_groups = defaultdict(int)
    for t in trace:
        depth_groups[t['depth']] += 1
    return dict(depth_groups)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tree generation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _gen_tree(rng, max_depth, current_depth=0, must_reach_max=False):
    """
    Generate a random ListOps tree.
    If must_reach_max=True, guarantees at least one path reaches max_depth.
    """
    op = rng.choice(OPS)
    n_args = rng.randint(MIN_ARGS, MAX_ARGS)

    if current_depth >= max_depth - 1:
        # Leaf level: all literals
        return {'op': op, 'args': [rng.randint(0, 9) for _ in range(n_args)],
                'depth': current_depth}

    args = []
    deep_placed = False

    for i in range(n_args):
        # Must we still place at least one deep child?
        need_force = must_reach_max and not deep_placed and i == n_args - 1

        if need_force:
            args.append(_gen_tree(rng, max_depth, current_depth + 1,
                                  must_reach_max=True))
            deep_placed = True
        elif rng.random() < EXPAND_PROB:
            child_must = must_reach_max and not deep_placed
            args.append(_gen_tree(rng, max_depth, current_depth + 1,
                                  must_reach_max=child_must))
            deep_placed = True
        else:
            args.append(rng.randint(0, 9))

    return {'op': op, 'args': args, 'depth': current_depth}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Dependency context classification
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _dep_context(trace_entry):
    """
    Classify a trace position's dependency context.
    Analogous to g/k/p in addition:
      independent  — all args are literals (no sub-expression dependency)
      shallow_dep  — has sub-expr children, all of which are independent
      deep_dep     — has sub-expr children that themselves have dependencies
      root         — the outermost operator (last in trace)
    """
    ns = trace_entry['n_subexpr']
    if ns == 0:
        return 'independent'
    # Check if children are themselves deep (have sub-expressions)
    # We approximate: if depth > 1 and has sub-expressions, it's deep_dep
    if trace_entry['depth'] == 0:
        return 'root'
    if trace_entry['depth'] >= 2 and ns > 0:
        return 'deep_dep'
    return 'shallow_dep'


DEP_CONTEXTS = ['independent', 'shallow_dep', 'deep_dep', 'root']
DEP_TO_ID = {n: i for i, n in enumerate(DEP_CONTEXTS)}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Data formatting & tokenizer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Input chars: 0-9, X N D S, [ ] (space), =
# Answer: 0-9 (trace digits) + P (pad/eos)
# Special: M (mask), P (pad)

EOS_CHAR = '$'  # marks end of trace in answer
# Rainbow padding: cyclic distinct tokens after EOS to break PAD dominance
# During generation, all answer positions are MASK → if uniform PAD,
# model learns "P is always the right answer" and outputs P everywhere.
# Rainbow uses distinct tokens so no single token dominates.
RAINBOW_CHARS = 'abcdefghijklmnop'  # 16 distinct pad tokens
INPUT_PAD = '#'  # for input sequence padding only (never appears in data)

def _rainbow_pad(trace_str):
    """Pad trace to MAX_ANS_LEN with EOS + cyclic rainbow tokens."""
    remaining = MAX_ANS_LEN - len(trace_str)
    if remaining <= 0:
        return trace_str[:MAX_ANS_LEN]
    # First char after trace = EOS, rest = rainbow
    pad = EOS_CHAR
    for i in range(remaining - 1):
        pad += RAINBOW_CHARS[i % len(RAINBOW_CHARS)]
    return trace_str + pad


def build_tok():
    chars = list('0123456789XNDS[] =') + [EOS_CHAR] + list(RAINBOW_CHARS)
    # deduplicate while preserving order
    seen = set()
    unique = []
    for c in chars:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return CharTokenizer(unique, {'mask': 'M', 'pad': INPUT_PAD})


def _format_sample(tree):
    """
    Format tree as 'expression=trace_padded'.
    Returns (formatted_string, metadata) or None if trace too long.
    """
    expr = _tree_to_str(tree)
    result, trace = _evaluate(tree)
    trace_str = ''.join(str(t['result']) for t in trace)

    if len(trace_str) > MAX_ANS_LEN:
        return None, None
    padded_trace = _rainbow_pad(trace_str)
    sample = f"{expr}={padded_trace}"
    if len(sample) > MAX_SEQ_LEN:
        return None, None

    # Metadata for analysis
    crit_len, crit_chain = _critical_chain_from_trace(trace)
    dep_pairs = _build_dependency_pairs(trace)
    meta = {
        'tree_depth': _tree_depth(tree) + 1,   # 1-indexed: depth 1 = flat
        'n_ops': len(trace),
        'trace_len': len(trace_str),
        'max_chain': _max_chain_depth(tree),
        'critical_chain_len': crit_len,
        'critical_chain': crit_chain,
        'dep_pairs': dep_pairs,             # (parent_idx, child_idx) pairs
        'trace_info': trace,                # per-trace-position metadata
        'dep_contexts': [_dep_context(t) for t in trace],
        'depths': [t['depth'] for t in trace],
        'ops': [t['op'] for t in trace],
        'children_indices': [t.get('children_indices', []) for t in trace],
    }
    return sample, meta


def get_answer(s):
    """Get answer string (padded trace) from formatted sample."""
    return s.split('=', 1)[1]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Data generation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _sample_depth(rng):
    """Sample target depth from DEPTH_DECAY geometric distribution."""
    d = 1
    while d < MAX_DEPTH and rng.random() < DEPTH_DECAY:
        d += 1
    return d


def gen_train_data(n, seed):
    """Training data with DEPTH_DECAY-controlled depth distribution."""
    rng = random.Random(seed)
    data, metas = [], []
    attempts = 0
    while len(data) < n and attempts < n * 50:
        attempts += 1
        target_d = _sample_depth(rng)
        tree = _gen_tree(rng, max_depth=target_d, must_reach_max=True)
        s, m = _format_sample(tree)
        if s is not None:
            data.append(s)
            metas.append(m)
    if len(data) < n:
        print(f"  WARNING: gen_train_data: {len(data)}/{n}")
    return data[:n], metas[:n]


def gen_test_data(n, seed):
    """Test data: uniform across depths for fair per-depth evaluation."""
    rng = random.Random(seed)
    per_depth = max(1, n // MAX_DEPTH)
    data, metas = [], []
    for target_d in range(1, MAX_DEPTH + 1):
        count = 0
        for _ in range(per_depth * 100):
            tree = _gen_tree(rng, max_depth=target_d, must_reach_max=True)
            s, m = _format_sample(tree)
            if s is not None:
                data.append(s)
                metas.append(m)
                count += 1
            if count >= per_depth:
                break
    rng2 = random.Random(seed + 7)
    combined = list(zip(data, metas))
    rng2.shuffle(combined)
    data, metas = zip(*combined) if combined else ([], [])
    return list(data[:n]), list(metas[:n])


def gen_min_depth_test(n, seed, min_depth):
    """Constructive generation with guaranteed min nesting depth."""
    rng = random.Random(seed)
    data, metas = [], []
    for _ in range(n * 100):
        tree = _gen_tree(rng, max_depth=min_depth, must_reach_max=True)
        s, m = _format_sample(tree)
        if s is not None and m['tree_depth'] >= min_depth:
            data.append(s)
            metas.append(m)
        if len(data) >= n:
            break
    if len(data) < n:
        print(f"    WARNING: gen_min_depth_test(d>={min_depth}): {len(data)}/{n}")
    return data[:n], metas[:n]


def gen_corner_case_test(n, seed, category='linear_chain'):
    """
    Corner cases:
      linear_chain: purely linear nesting (each op has exactly 1 sub-expr child)
                    → maximum sequential dependency, analog of full_propagate
      max_branch:   every op has multiple sub-expression children
                    → maximum parallelism, minimum sequential dependency
      flat:         depth 1 only (no nesting) → baseline
      deep_narrow:  deep linear chain with minimal args
    """
    rng = random.Random(seed)
    data, metas = [], []

    def _make_linear(depth):
        if depth <= 0:
            return rng.randint(0, 9)
        op = rng.choice(OPS)
        n_lit = rng.randint(MIN_ARGS - 1, MAX_ARGS - 1)
        args = [rng.randint(0, 9) for _ in range(n_lit)]
        pos = rng.randint(0, len(args))
        args.insert(pos, _make_linear(depth - 1))
        return {'op': op, 'args': args, 'depth': 0}

    def _fix_depth(node, d=0):
        if isinstance(node, int):
            return node
        node['depth'] = d
        node['args'] = [_fix_depth(a, d + 1) for a in node['args']]
        return node

    def _make_max_branch(depth):
        """Max branching, but cap depth so trace fits MAX_ANS_LEN."""
        if depth <= 0:
            return rng.randint(0, 9)
        op = rng.choice(OPS)
        args = [_make_max_branch(depth - 1) for _ in range(MIN_ARGS)]
        return {'op': op, 'args': args, 'depth': 0}

    # For max_branch, find max feasible depth: MIN_ARGS^d + ... <= MAX_ANS_LEN
    mb_depth = 1
    while sum(MIN_ARGS ** i for i in range(mb_depth + 2)) <= MAX_ANS_LEN:
        mb_depth += 1

    def _make_deep_narrow(depth):
        """Deep chain: each op has exactly MIN_ARGS args, one is sub-expr."""
        if depth <= 0:
            return rng.randint(0, 9)
        op = rng.choice(OPS)
        args = [rng.randint(0, 9) for _ in range(MIN_ARGS - 1)]
        args.append(_make_deep_narrow(depth - 1))
        return {'op': op, 'args': args, 'depth': 0}

    for _ in range(n * 100):
        if len(data) >= n:
            break
        if category == 'linear_chain':
            tree = _fix_depth(_make_linear(MAX_DEPTH - 1))
        elif category == 'max_branch':
            tree = _fix_depth(_make_max_branch(mb_depth))
        elif category == 'flat':
            tree = _gen_tree(rng, max_depth=1, must_reach_max=True)
        elif category == 'deep_narrow':
            tree = _fix_depth(_make_deep_narrow(MAX_DEPTH - 1))
        else:
            continue
        s, m = _format_sample(tree)
        if s is not None:
            data.append(s)
            metas.append(m)

    if len(data) < n:
        print(f"    WARNING: corner/{category}: {len(data)}/{n}")
    return data[:n], metas[:n]


def gen_depth_ood_test(n, seed, target_depth):
    """
    Generate trees at a specific depth, potentially OOD (> MAX_DEPTH).
    Trees are narrow to keep trace_len <= MAX_ANS_LEN.
    """
    rng = random.Random(seed)
    data, metas = [], []

    def _make_narrow(depth):
        """One sub-expr child + literals, to keep trace small."""
        if depth <= 0:
            return rng.randint(0, 9)
        op = rng.choice(OPS)
        n_lit = rng.randint(MIN_ARGS - 1, MAX_ARGS - 1)
        # Sometimes add extra sub-expressions at shallow depths
        extra_sub = 1 if depth > 1 and rng.random() < 0.3 else 0
        args = [rng.randint(0, 9) for _ in range(n_lit)]
        # Place the deep child
        pos = rng.randint(0, len(args))
        args.insert(pos, _make_narrow(depth - 1))
        # Maybe add extra shallow sub-expression
        for _ in range(extra_sub):
            shallow_d = rng.randint(0, max(0, depth - 2))
            pos2 = rng.randint(0, len(args))
            args.insert(pos2, _make_narrow(shallow_d))
        return {'op': op, 'args': args, 'depth': 0}

    def _fix_depth(node, d=0):
        if isinstance(node, int):
            return node
        node['depth'] = d
        node['args'] = [_fix_depth(a, d + 1) for a in node['args']]
        return node

    for _ in range(n * 200):
        if len(data) >= n:
            break
        tree = _fix_depth(_make_narrow(target_depth))
        s, m = _format_sample(tree)
        if s is not None and m['tree_depth'] >= target_depth:
            data.append(s)
            metas.append(m)

    if len(data) < n:
        print(f"    WARNING: gen_depth_ood(d={target_depth}): {len(data)}/{n}")
    return data[:n], metas[:n]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Probes & analyses
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _pos_labels():
    return [f't{j}' for j in range(MAX_ANS_LEN)]


@torch.no_grad()
def probe_per_position(model, tokenizer, test_samples, test_metas, max_len,
                       device=None):
    """Fully-masked probe: per-position loss/acc/conf + dependency context."""
    if device is None:
        device = DEVICE
    model.eval()
    mask_id = tokenizer.special_ids['mask']
    ids_all, ans_all = encode_samples(test_samples, tokenizer, max_len)
    ids_all, ans_all = ids_all.to(device), ans_all.to(device)

    # Pre-compute per-sample, per-position dependency context
    dep_ids = torch.zeros(len(test_samples), MAX_ANS_LEN, dtype=torch.long, device=device)
    is_trace = torch.zeros(len(test_samples), MAX_ANS_LEN, dtype=torch.bool, device=device)
    depth_ids = torch.zeros(len(test_samples), MAX_ANS_LEN, dtype=torch.long, device=device)
    for i, m in enumerate(test_metas):
        tl = m['trace_len']
        is_trace[i, :tl] = True
        for j, dc in enumerate(m['dep_contexts'][:tl]):
            dep_ids[i, j] = DEP_TO_ID.get(dc, 0)
        for j, d in enumerate(m['depths'][:tl]):
            depth_ids[i, j] = d

    L = torch.zeros(MAX_ANS_LEN, device=device)
    C = torch.zeros(MAX_ANS_LEN, device=device)
    CF = torch.zeros(MAX_ANS_LEN, device=device)
    N = torch.zeros(MAX_ANS_LEN, device=device)
    _arange = torch.arange(MAX_ANS_LEN, device=device)

    # Per-dep-context accumulators
    dep_conf_sum = defaultdict(float)
    dep_acc_sum = defaultdict(float)
    dep_count = defaultdict(int)

    # Per-depth accumulators
    depth_acc_sum = defaultdict(float)
    depth_count = defaultdict(int)

    for st in range(0, len(test_samples), 128):
        en = min(st + 128, len(test_samples))
        ids, ans = ids_all[st:en], ans_all[st:en]
        B, T = ids.shape
        ans_pos = (ans.unsqueeze(1) + _arange).clamp(max=T - 1)
        bi = torch.arange(B, device=device).unsqueeze(1).expand_as(ans_pos)

        xm = ids.clone()
        xm[bi, ans_pos] = mask_id
        logits = model(xm)
        al = logits[bi, ans_pos]
        tgt = ids[bi, ans_pos]
        lp = F.log_softmax(al, dim=-1)
        losses = -lp.gather(2, tgt.unsqueeze(2)).squeeze(2)
        cl = al.clone()
        cl[:, :, mask_id] = -float('inf')
        probs = F.softmax(cl, dim=-1)
        confs = probs.max(dim=-1).values
        preds = probs.argmax(dim=-1)
        corrects = (preds == tgt).float()

        L += losses.sum(dim=0)
        C += corrects.sum(dim=0)
        CF += confs.sum(dim=0)
        N += B

        # Dep context
        dep_b = dep_ids[st:en].reshape(-1)
        it_b = is_trace[st:en].reshape(-1)
        cf_flat = confs.reshape(-1)
        co_flat = corrects.reshape(-1)
        for di, dn in enumerate(DEP_CONTEXTS):
            mask = (dep_b == di) & it_b
            if mask.any():
                dep_conf_sum[dn] += cf_flat[mask].sum().item()
                dep_acc_sum[dn] += co_flat[mask].sum().item()
                dep_count[dn] += mask.sum().item()

        # Per-depth
        dd_b = depth_ids[st:en].reshape(-1)
        for d in range(MAX_DEPTH + 1):
            mask = (dd_b == d) & it_b
            if mask.any():
                depth_acc_sum[d] += co_flat[mask].sum().item()
                depth_count[d] += mask.sum().item()

    s = N.clamp(1)
    pos_conf = (CF / s).cpu().tolist()
    result = {
        'pos_loss': (L / s).cpu().tolist(),
        'pos_acc': (C / s).cpu().tolist(),
        'pos_conf': pos_conf,
        'overall_loss': (L.sum() / s.sum()).item(),
        'overall_acc': (C.sum() / s.sum()).item(),
    }
    if dep_count:
        result['dep_context'] = {
            ctx: {'conf': dep_conf_sum[ctx] / n, 'acc': dep_acc_sum[ctx] / n, 'n': n}
            for ctx, n in dep_count.items() if n > 0
        }
    if depth_count:
        result['depth_acc'] = {
            d: {'acc': depth_acc_sum[d] / n, 'n': n}
            for d, n in depth_count.items() if n > 0
        }

    # Confidence concordance (higher positions = outer = should be decoded later by oracle)
    cr = sorted(range(MAX_ANS_LEN), key=lambda j: pos_conf[j], reverse=True)
    conc = 0
    n_p = MAX_ANS_LEN * (MAX_ANS_LEN - 1) // 2
    for i in range(MAX_ANS_LEN):
        for j in range(i + 1, MAX_ANS_LEN):
            conc += int(cr.index(j) < cr.index(i))
    result['conf_concordance'] = conc / n_p if n_p > 0 else 0
    result['conf_spread'] = max(pos_conf) - min(pos_conf)
    return result


@torch.no_grad()
def gen_eval_with_stats(model, tokenizer, test_samples, test_metas, max_len,
                        decode_policy='confidence', device=None):
    """Per-sample generation with tree stats, error positions, and decode order."""
    if device is None:
        device = DEVICE
    mask_id = tokenizer.special_ids['mask']
    pad_id = tokenizer.special_ids['pad']
    model.eval()
    out = [None] * len(test_samples)

    # Group by prefix length to avoid padding between '=' and MASK
    groups = {}
    for idx, s in enumerate(test_samples):
        prefix = s.split('=')[0] + '='
        pl = len(tokenizer.encode(prefix))
        groups.setdefault(pl, []).append(idx)

    for pl, indices in groups.items():
        for bstart in range(0, len(indices), 128):
            bind = indices[bstart:bstart + 128]
            B = len(bind)
            batch_s = [test_samples[i] for i in bind]
            batch_m = [test_metas[i] for i in bind]
            gold_full = [get_answer(s) for s in batch_s]

            penc = [tokenizer.encode(s.split('=')[0] + '=') for s in batch_s]
            # All same length (pl), no padding between = and MASK
            pids = torch.tensor(penc, dtype=torch.long)

            gen, _, info = generate_diffusion(
                model, pids, MAX_ANS_LEN, mask_id,
                policy=decode_policy, greedy=True,
                pad_to=max_len, pad_id=pad_id, device=device)
            pred_ids = gen[:, pl:pl + MAX_ANS_LEN]
            batch_orders = info.get('orders')

            for bi in range(B):
                pred_str = tokenizer.decode(pred_ids[bi].cpu().tolist())
                gold_str = gold_full[bi]
                meta = batch_m[bi]
                tl = meta['trace_len']

                pred_trace = pred_str[:tl]
                gold_trace = gold_str[:tl]
                trace_correct = pred_trace == gold_trace
                full_correct = pred_str == gold_str

                pc = [pred_str[j] == gold_str[j] if j < len(pred_str) else False
                      for j in range(len(gold_str))]
                errs = [j for j in range(len(gold_str))
                        if j >= len(pred_str) or pred_str[j] != gold_str[j]]

                decode_order_ans = None
                dag_violations = 0
                dag_total_deps = 0
                if batch_orders is not None:
                    ans_start = pl
                    raw_order = batch_orders[bi].tolist()
                    decode_step = {}
                    for step, abs_pos in enumerate(raw_order):
                        ans_pos = abs_pos - ans_start
                        if 0 <= ans_pos < MAX_ANS_LEN:
                            decode_step[ans_pos] = step
                    decode_order_ans = decode_step

                    dep_pairs = meta.get('dep_pairs', [])
                    for parent_idx, child_idx in dep_pairs:
                        if parent_idx in decode_step and child_idx in decode_step:
                            dag_total_deps += 1
                            if decode_step[parent_idx] < decode_step[child_idx]:
                                dag_violations += 1

                out[bind[bi]] = {
                    'correct': full_correct,
                    'trace_correct': trace_correct,
                    'pos_correct': pc,
                    'error_positions': errs,
                    'tree_depth': meta['tree_depth'],
                    'n_ops': meta['n_ops'],
                    'trace_len': tl,
                    'max_chain': meta['max_chain'],
                    'critical_chain_len': meta['critical_chain_len'],
                    'dep_contexts': meta['dep_contexts'],
                    'depths': meta['depths'],
                    'ops': meta['ops'],
                    'children_indices': meta['children_indices'],
                    'dep_pairs': meta.get('dep_pairs', []),
                    'dag_violations': dag_violations,
                    'dag_total_deps': dag_total_deps,
                    'decode_order': decode_order_ans,
                }
    return out


def stratify_results(per_sample):
    """Stratify by tree depth and operator count."""
    def _depth_bin(d):
        return f'd={d}'

    def _nops_bin(n):
        if n <= 3:
            return 'ops=1-3'
        if n <= 7:
            return 'ops=4-7'
        if n <= 12:
            return 'ops=8-12'
        return 'ops=13+'

    strata = {
        'tree_depth': lambda r: _depth_bin(r['tree_depth']),
        'n_ops': lambda r: _nops_bin(r['n_ops']),
    }
    out = {}
    for name, fn in strata.items():
        bk = defaultdict(list)
        for r in per_sample:
            bk[fn(r)].append(r['trace_correct'])
        out[name] = {k: {'acc': sum(v) / len(v), 'n': len(v)}
                     for k, v in sorted(bk.items())}
    return out


def analyse_depth_rarity(per_sample, test_metas):
    """
    Per-position: depth base rate × conditional accuracy.
    Analog of carry_rarity in addition.
    """
    N = len(per_sample)
    per_pos = []

    for j in range(MAX_ANS_LEN):
        deep, shallow = [], []  # deep = depth >= 2
        for i in range(N):
            if j >= len(per_sample[i]['pos_correct']):
                continue
            c = per_sample[i]['pos_correct'][j]
            depths = test_metas[i]['depths']
            if j < len(depths) and depths[j] >= 2:
                deep.append(c)
            else:
                shallow.append(c)
        br = len(deep) / max(len(deep) + len(shallow), 1)
        a_deep = sum(deep) / len(deep) if deep else None
        a_shallow = sum(shallow) / len(shallow) if shallow else None
        gap = (a_shallow - a_deep) if a_shallow is not None and a_deep is not None else None
        per_pos.append({
            'position': j, 'base_rate': br,
            'acc_deep': a_deep, 'acc_shallow': a_shallow, 'acc_gap': gap,
        })

    # Correlation
    valid = [(p['base_rate'], p['acc_gap']) for p in per_pos if p['acc_gap'] is not None]
    corr = None
    if len(valid) >= 3:
        brs, gs = [v[0] for v in valid], [v[1] for v in valid]
        mb, mg = sum(brs) / len(brs), sum(gs) / len(gs)
        c = sum((b - mb) * (g - mg) for b, g in zip(brs, gs))
        sb = sum((b - mb) ** 2 for b in brs) ** 0.5
        sg = sum((g - mg) ** 2 for g in gs) ** 0.5
        corr = c / (sb * sg) if sb > 0 and sg > 0 else 0.0

    return {'per_position': per_pos, 'corr': corr}


@torch.no_grad()
def simulate_puma_coverage(model, tokenizer, test_samples, test_metas, max_len,
                           K=None, tau=None, n_samples=200, device=None):
    """Measure PUMA chain coverage × depth condition."""
    if device is None:
        device = DEVICE
    if K is None:
        K = PUMA_K_END or PUMA_K
    if tau is None:
        tau = PUMA_TAU
    model.eval()
    mask_id = tokenizer.special_ids['mask']

    N = min(len(test_samples), n_samples)
    # Per-position: coverage for deep vs shallow trace positions
    deep_s = torch.zeros(MAX_ANS_LEN)
    deep_n = torch.zeros(MAX_ANS_LEN, dtype=torch.long)
    shallow_s = torch.zeros(MAX_ANS_LEN)
    shallow_n = torch.zeros(MAX_ANS_LEN, dtype=torch.long)

    for si in range(N):
        s = test_samples[si]
        meta = test_metas[si]
        depths = meta['depths']
        prefix = s.split('=')[0] + '='
        answer = get_answer(s)
        penc = tokenizer.encode(prefix)
        aenc = tokenizer.encode(answer)
        T_pre = len(penc)
        # Pad to max_len for consistent attention patterns
        pad_id = tokenizer.special_ids['pad']
        seq = penc + [mask_id] * MAX_ANS_LEN
        if len(seq) < max_len:
            seq = seq + [pad_id] * (max_len - len(seq))
        x = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)
        x0 = torch.tensor(aenc[:MAX_ANS_LEN], dtype=torch.long, device=device)
        is_m = torch.ones(MAX_ANS_LEN, dtype=torch.bool, device=device)
        steps_m = torch.zeros(MAX_ANS_LEN)
        total = 0

        for step in range(K):
            if not is_m.any():
                break
            total += 1
            steps_m += is_m.cpu().float()
            logits = model(x)
            nm = is_m.sum().item()
            nr = max(1, int(math.ceil(nm / max(K - step, 1))))
            confs = torch.full((MAX_ANS_LEN,), -float('inf'), device=device)
            for j in range(MAX_ANS_LEN):
                if is_m[j]:
                    cl = logits[0, T_pre + j].clone()
                    cl[mask_id] = -float('inf')
                    confs[j] = F.softmax(cl, dim=-1).max()
            ranked = confs.argsort(descending=True)
            reveal = torch.zeros(MAX_ANS_LEN, dtype=torch.bool, device=device)
            for ri in range(MAX_ANS_LEN):
                j = ranked[ri].item()
                if not is_m[j]:
                    continue
                if reveal.sum() < nr or confs[j] > tau:
                    reveal[j] = True
            for j in range(MAX_ANS_LEN):
                if reveal[j]:
                    x[0, T_pre + j] = x0[j]
                    is_m[j] = False

        if total == 0:
            continue
        frac = steps_m / total
        for j in range(MAX_ANS_LEN):
            is_deep = j < len(depths) and depths[j] >= 2
            if is_deep:
                deep_s[j] += frac[j]
                deep_n[j] += 1
            else:
                shallow_s[j] += frac[j]
                shallow_n[j] += 1

    per_pos = []
    for j in range(MAX_ANS_LEN):
        nd, ns = deep_n[j].item(), shallow_n[j].item()
        cv_d = deep_s[j].item() / nd if nd > 0 else None
        cv_s = shallow_s[j].item() / ns if ns > 0 else None
        deficit = (cv_s - cv_d) if cv_s is not None and cv_d is not None else None
        br = nd / max(nd + ns, 1)
        per_pos.append({
            'position': j, 'base_rate': br,
            'cov_deep': cv_d, 'cov_shallow': cv_s, 'deficit': deficit,
        })

    valid = [(p['base_rate'], p['deficit']) for p in per_pos if p['deficit'] is not None]
    corr = None
    if len(valid) >= 3:
        brs, ds = [v[0] for v in valid], [v[1] for v in valid]
        mb, md_v = sum(brs) / len(brs), sum(ds) / len(ds)
        c = sum((b - mb) * (d - md_v) for b, d in zip(brs, ds))
        sb = sum((b - mb) ** 2 for b in brs) ** 0.5
        sd = sum((d - md_v) ** 2 for d in ds) ** 0.5
        corr = c / (sb * sd) if sb > 0 and sd > 0 else 0.0

    return {'per_position': per_pos, 'corr': corr}


def analyse_error_localization(per_sample):
    """Where in the trace do errors occur? By dependency context."""
    cats = defaultdict(int)
    total_errs = 0
    for r in per_sample:
        if r['trace_correct']:
            continue
        for j in r['error_positions']:
            if j >= r['trace_len']:
                cats['pad_error'] += 1
            elif j < len(r['dep_contexts']):
                cats[r['dep_contexts'][j]] += 1
            total_errs += 1
    if total_errs == 0:
        return {'total_errors': 0}
    return {'total_errors': total_errs,
            **{k: v / total_errs for k, v in cats.items()}}


def analyse_dag_violations(per_sample):
    """
    DAG-aware decode order analysis.
    Measures: violation rate, correlation between violations and errors.
    A violation = parent decoded before any of its children.
    """
    total_deps = 0
    total_violations = 0
    # Correlation: does higher violation rate → more errors?
    viol_correct = []
    viol_wrong = []
    # Per critical-chain-length
    by_crit = defaultdict(lambda: {'violations': 0, 'total_deps': 0,
                                    'correct': 0, 'total': 0})

    for r in per_sample:
        v = r.get('dag_violations', 0)
        td = r.get('dag_total_deps', 0)
        total_deps += td
        total_violations += v
        viol_rate = v / td if td > 0 else 0

        if r['trace_correct']:
            viol_correct.append(viol_rate)
        else:
            viol_wrong.append(viol_rate)

        cl = r.get('critical_chain_len', 1)
        by_crit[cl]['violations'] += v
        by_crit[cl]['total_deps'] += td
        by_crit[cl]['correct'] += int(r['trace_correct'])
        by_crit[cl]['total'] += 1

    result = {
        'total_deps': total_deps,
        'total_violations': total_violations,
        'violation_rate': total_violations / max(total_deps, 1),
        'mean_viol_rate_correct': (sum(viol_correct) / len(viol_correct)
                                   if viol_correct else None),
        'mean_viol_rate_wrong': (sum(viol_wrong) / len(viol_wrong)
                                 if viol_wrong else None),
    }

    # Per critical chain length
    by_crit_out = {}
    for cl in sorted(by_crit):
        b = by_crit[cl]
        by_crit_out[cl] = {
            'violation_rate': b['violations'] / max(b['total_deps'], 1),
            'accuracy': b['correct'] / max(b['total'], 1),
            'n': b['total'],
        }
    result['by_critical_chain'] = by_crit_out
    return result


def analyse_operator_error_propagation(per_sample):
    """
    Per-operator analysis: when a child trace position is wrong,
    how often does the parent also get wrong?

    Also: operator sensitivity — some ops (SM) are fully sensitive to
    child errors while others (MAX/MIN) may be partially tolerant.
    """
    op_stats = defaultdict(lambda: {
        'child_err_parent_err': 0,      # child wrong, parent wrong
        'child_err_parent_ok': 0,       # child wrong, parent still correct
        'child_ok_parent_err': 0,       # all children ok, parent wrong
        'child_ok_parent_ok': 0,        # all children ok, parent ok
        'total': 0,
    })

    for r in per_sample:
        pc = r['pos_correct']
        ci_list = r.get('children_indices', [])
        ops = r.get('ops', [])
        tl = r['trace_len']

        for j in range(tl):
            if j >= len(ops) or j >= len(ci_list):
                continue
            op = ops[j]
            children = ci_list[j]
            parent_ok = pc[j] if j < len(pc) else False

            if children:
                any_child_err = any(
                    not pc[ci] if ci < len(pc) else True
                    for ci in children if ci < tl
                )
                if any_child_err and not parent_ok:
                    op_stats[op]['child_err_parent_err'] += 1
                elif any_child_err and parent_ok:
                    op_stats[op]['child_err_parent_ok'] += 1
                elif not any_child_err and not parent_ok:
                    op_stats[op]['child_ok_parent_err'] += 1
                else:
                    op_stats[op]['child_ok_parent_ok'] += 1
            else:
                # No sub-expression children (leaf operator)
                if parent_ok:
                    op_stats[op]['child_ok_parent_ok'] += 1
                else:
                    op_stats[op]['child_ok_parent_err'] += 1
            op_stats[op]['total'] += 1

    result = {}
    for op in sorted(op_stats):
        s = op_stats[op]
        t_child_err = s['child_err_parent_err'] + s['child_err_parent_ok']
        t_child_ok = s['child_ok_parent_err'] + s['child_ok_parent_ok']
        result[op] = {
            'propagation_rate': (s['child_err_parent_err'] / max(t_child_err, 1)
                                 if t_child_err > 0 else None),
            'tolerance_rate': (s['child_err_parent_ok'] / max(t_child_err, 1)
                               if t_child_err > 0 else None),
            'independent_error_rate': (s['child_ok_parent_err'] / max(t_child_ok, 1)
                                       if t_child_ok > 0 else None),
            'n_with_child_err': t_child_err,
            'n_with_child_ok': t_child_ok,
            'total': s['total'],
        }
    return result


def analyse_critical_chain_sweep(per_sample):
    """
    Accuracy stratified by critical chain length.
    This is the true sequential dependency analog of carry chain length.
    """
    by_cl = defaultdict(list)
    for r in per_sample:
        cl = r.get('critical_chain_len', 1)
        by_cl[cl].append(r['trace_correct'])
    return {cl: {'acc': sum(v) / len(v), 'n': len(v)}
            for cl, v in sorted(by_cl.items())}


@torch.no_grad()
def analyse_confidence_calibration(model, tokenizer, test_samples, test_metas,
                                   max_len, device=None):
    """Per depth-bin: mean confidence for correct vs wrong predictions."""
    if device is None:
        device = DEVICE
    ps = gen_eval_with_stats(model, tokenizer, test_samples, test_metas, max_len,
                             decode_policy='confidence', device=device)
    mask_id = tokenizer.special_ids['mask']
    ids_all, ans_all = encode_samples(test_samples, tokenizer, max_len)
    ids_all, ans_all = ids_all.to(device), ans_all.to(device)
    _arange = torch.arange(MAX_ANS_LEN, device=device)

    confs_per = []
    for st in range(0, len(test_samples), 128):
        en = min(st + 128, len(test_samples))
        ids, ans = ids_all[st:en], ans_all[st:en]
        B = ids.shape[0]
        ans_pos = (ans.unsqueeze(1) + _arange).clamp(max=ids.shape[1] - 1)
        bi = torch.arange(B, device=device).unsqueeze(1).expand_as(ans_pos)
        xm = ids.clone()
        xm[bi, ans_pos] = mask_id
        logits = model(xm)
        al = logits[bi, ans_pos]
        cl = al.clone()
        cl[:, :, mask_id] = -float('inf')
        confs_per.extend(
            F.softmax(cl, dim=-1).max(dim=-1).values.mean(dim=1).cpu().tolist())

    def _d_bin(d):
        if d <= 1:
            return 'd<=1'
        if d <= 3:
            return 'd=2-3'
        return 'd>=4'

    bins = defaultdict(lambda: {'correct': [], 'wrong': []})
    for i, r in enumerate(ps):
        bn = _d_bin(r['tree_depth'])
        (bins[bn]['correct'] if r['trace_correct'] else bins[bn]['wrong']).append(
            confs_per[i] if i < len(confs_per) else 0.5)

    result = {}
    for bn, d in sorted(bins.items()):
        mc = sum(d['correct']) / len(d['correct']) if d['correct'] else None
        mw = sum(d['wrong']) / len(d['wrong']) if d['wrong'] else None
        oc = (sum(1 for c in d['wrong'] if c > 0.8) / len(d['wrong'])
              if d['wrong'] else None)
        result[bn] = {
            'mean_conf_correct': mc, 'mean_conf_wrong': mw,
            'overconf_rate': oc,
            'n_correct': len(d['correct']), 'n_wrong': len(d['wrong']),
        }
    return result


@torch.no_grad()
def _quick_gen(model, tokenizer, test_samples, test_metas, max_len, decode_policy,
               n=None, device=None):
    """Quick generation accuracy check with trace-only and full accuracy."""
    if device is None:
        device = DEVICE
    mask_id = tokenizer.special_ids['mask']
    pad_id = tokenizer.special_ids['pad']
    subset_s = test_samples[:n] if n else test_samples
    subset_m = test_metas[:n] if n else test_metas
    results = [None] * len(subset_s)
    examples = []

    # Group by prefix length
    groups = {}
    for idx, s in enumerate(subset_s):
        prefix = s.split('=')[0] + '='
        pl = len(tokenizer.encode(prefix))
        groups.setdefault(pl, []).append(idx)

    for pl, indices in groups.items():
        for bstart in range(0, len(indices), 128):
            bind = indices[bstart:bstart + 128]
            B = len(bind)
            batch = [subset_s[i] for i in bind]
            batch_m = [subset_m[i] for i in bind]

            penc = [tokenizer.encode(s.split('=')[0] + '=') for s in batch]
            pids = torch.tensor(penc, dtype=torch.long)

            gen, _, _ = generate_diffusion(model, pids, MAX_ANS_LEN, mask_id,
                                           policy=decode_policy, greedy=True,
                                           pad_to=max_len, pad_id=pad_id,
                                           device=device)
            pred = gen[:, pl:pl + MAX_ANS_LEN]
            for bi in range(B):
                ps = tokenizer.decode(pred[bi].cpu().tolist())
                gs = get_answer(batch[bi])
                tl = batch_m[bi]['trace_len']
                full_ok = (ps == gs)
                trace_ok = (ps[:tl] == gs[:tl])
                results[bind[bi]] = {'correct': full_ok, 'trace_correct': trace_ok}
                if len(examples) < 5:
                    examples.append({
                        'input': batch[bi].split('=')[0],
                        'gold': gs, 'pred': ps,
                        'trace_len': tl, 'trace_ok': trace_ok, 'full_ok': full_ok,
                    })
    n_total = max(len(results), 1)
    return {
        'accuracy': sum(r['correct'] for r in results if r) / n_total,
        'trace_accuracy': sum(r['trace_correct'] for r in results if r) / n_total,
        'n': len(results),
        'examples': examples,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Training wrapper
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def train_model(mask_type, tokenizer, train_samples, test_samples, test_metas,
                max_len, max_iters=None, init_state=None, device=None):
    """Wrapper around train_diffusion for ListOps experiment."""
    if device is None:
        device = DEVICE
    if max_iters is None:
        max_iters = MAX_ITERS

    train_ids, train_ans = encode_samples(train_samples, tokenizer, max_len)
    train_ids, train_ans = train_ids.to(device), train_ans.to(device)

    # PUMA K schedule
    k_sched = None
    if mask_type == 'puma':
        if PUMA_K_START is not None and PUMA_K_END is not None:
            k_step = PUMA_K_STEP or 3
            if PUMA_K_EVERY is not None:
                k_every = PUMA_K_EVERY
            else:
                n_inc = max(1, (PUMA_K_END - PUMA_K_START) // k_step)
                k_every = max(1000, (max_iters // 3) // n_inc)
            k_sched = puma_k_step(PUMA_K_START, PUMA_K_END, k_step, k_every)
            final_k = k_sched(max_iters)
            print(f"  PUMA K: step {PUMA_K_START}→{final_k} "
                  f"(+{k_step} every {k_every // 1000}k, cap={PUMA_K_END})")
        else:
            k_sched = puma_k_fixed(PUMA_K)
            print(f"  PUMA K: fixed {PUMA_K}")

    # Eval callback
    def eval_fn(model, it, tg):
        probe = probe_per_position(model, tokenizer, test_samples, test_metas,
                                   max_len, device)
        dc = probe.get('dep_context', {})
        parts = [f"{c}={dc[c]['acc']:.2f}" for c in DEP_CONTEXTS if c in dc]
        print(f"    [eval it {it}] loss={probe['overall_loss']:.4f} "
              f"acc={probe['overall_acc']:.4f} {' '.join(parts)}")
        if it > 0 and it % GEN_EVAL_EVERY == 0:
            r = _quick_gen(model, tokenizer, test_samples, test_metas, max_len,
                           'confidence', device=device)
            print(f"      [gen] full={r['accuracy']:.3f} trace={r['trace_accuracy']:.3f}")
            for ex in r.get('examples', [])[:3]:
                print(f"        in={ex['input'][:50]} gold={ex['gold'][:ex['trace_len']]}|{ex['gold'][ex['trace_len']:ex['trace_len']+3]}.. pred={ex['pred'][:ex['trace_len']]}|{ex['pred'][ex['trace_len']:ex['trace_len']+3]}.. {'✓' if ex['trace_ok'] else '✗'}")
            probe['gen_acc_full'] = r['accuracy']
            probe['gen_acc_trace'] = r['trace_accuracy']
        return probe

    model, dynamics = train_diffusion(
        train_ids=train_ids, train_ans=train_ans, ans_len=MAX_ANS_LEN,
        tokenizer=tokenizer,
        mask_type=mask_type, blank_masks=None,
        puma_tau=PUMA_TAU,
        puma_k_schedule=k_sched,
        n_layer=N_LAYER, n_head=N_HEAD, n_embd=N_EMBD,
        dropout=DROPOUT, pos_enc=POS_ENC,
        max_iters=max_iters, batch_size=BATCH_SIZE,
        lr=LR, min_lr=MIN_LR, warmup_iters=WARMUP_ITERS,
        grad_clip=GRAD_CLIP, weight_decay=WEIGHT_DECAY, ema_decay=EMA_DECAY,
        eval_fn=eval_fn, eval_every=EVAL_EVERY, log_every=LOG_EVERY,
        patience=PATIENCE,
        init_state=init_state, device=device,
        use_amp=False if NO_AMP else None,
    )
    return model, dynamics


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Figures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def make_figures(all_dyn, all_final):
    figs = {}
    labels = _pos_labels()
    cmap = plt.cm.coolwarm

    def pc(j):
        return cmap(1.0 - j / (MAX_ANS_LEN - 1))

    # Fig 1: Per-position accuracy over training
    nc = len(MASK_TYPES)
    fig, axes = plt.subplots(1, nc, figsize=(6 * nc, 5), squeeze=False)
    axes = axes[0]
    for ai, mt in enumerate(MASK_TYPES):
        dyn = all_dyn.get(mt)
        if not dyn:
            continue
        ax = axes[ai]
        xs = [c['iter'] for c in dyn['checkpoints']]
        for j in range(MAX_ANS_LEN):
            ys = [c['pos_acc'][j] for c in dyn['checkpoints'] if 'pos_acc' in c]
            if ys:
                ax.plot(xs[:len(ys)], ys, '-', color=pc(j), label=labels[j], lw=1.2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Accuracy')
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(mt)
        ax.legend(fontsize=4, ncol=4)
        ax.grid(alpha=0.3)
    fig.suptitle('Per-Position Probe Accuracy', y=1.02)
    fig.tight_layout()
    figs['pos_acc'] = fig

    # Fig 2: Depth sweep comparison
    sweep_depths = list(range(1, MAX_DEPTH + 1))
    for dp in DECODE_POLICIES:
        fig, ax = plt.subplots(figsize=(10, 5))
        for mt, col, mk in [('random', '#3498db', 'o'), ('puma', '#8e44ad', 's')]:
            accs = []
            for d in sweep_depths:
                r = all_final.get(f'{mt}_depth_sweep_{d}_{dp}')
                accs.append(r['accuracy'] if r else None)
            valid = [(d, a) for d, a in zip(sweep_depths, accs) if a is not None]
            if valid:
                ax.plot([v[0] for v in valid], [v[1] for v in valid],
                        f'-{mk}', color=col, label=mt, lw=2, markersize=8)
        ax.set_xlabel('Min tree depth')
        ax.set_ylabel('Accuracy (trace)')
        ax.set_title(f'Depth Sweep — {dp} decode')
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        figs[f'depth_sweep_{dp}'] = fig

    # Fig 3: Depth rarity × accuracy gap
    fig, ax = plt.subplots(figsize=(8, 5))
    for mt, color, mk in [('random', '#3498db', 'o'), ('puma', '#8e44ad', 's')]:
        r = all_final.get(f'{mt}_depth_rarity')
        if not r:
            continue
        brs = [p['base_rate'] for p in r['per_position'] if p['acc_gap'] is not None]
        gaps = [p['acc_gap'] for p in r['per_position'] if p['acc_gap'] is not None]
        ax.scatter(brs, gaps, c=color, marker=mk, s=50, alpha=0.7,
                   label=f"{mt} (r={r['corr']:.2f})" if r['corr'] else mt)
    ax.axhline(0, color='gray', ls=':')
    ax.set_xlabel('Deep-position base rate')
    ax.set_ylabel('acc(shallow) - acc(deep)')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    figs['depth_rarity'] = fig

    return figs


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Run
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run(tag=''):
    exp_name = f"{EXP_NAME}_{tag}" if tag else EXP_NAME
    mount_drive()
    torch.manual_seed(SEED)
    random.seed(SEED)
    tok = build_tok()

    print(f"\n{'=' * 70}")
    print(f"  ListOps MAX_DEPTH={MAX_DEPTH} MAX_ANS_LEN={MAX_ANS_LEN} "
          f"DEPTH_DECAY={DEPTH_DECAY}")
    print(f"  masks={MASK_TYPES} | decode={DECODE_POLICIES}")
    print(f"  N_TRAIN={N_TRAIN} N_TEST={N_TEST} MAX_ITERS={MAX_ITERS}")
    print(f"  arch: {N_LAYER}L/{N_HEAD}H/{N_EMBD}D | expand_prob={EXPAND_PROB}")
    print(f"{'=' * 70}\n")

    train_data, train_metas = gen_train_data(N_TRAIN, seed=SEED)
    test_data, test_metas = gen_test_data(N_TEST, seed=SEED + 1000)
    max_len = max(len(tok.encode(s)) for s in train_data + test_data)

    # Print distribution info
    train_depths = [m['tree_depth'] for m in train_metas]
    train_traces = [m['trace_len'] for m in train_metas]
    test_depths = [m['tree_depth'] for m in test_metas]
    print(f"  Train depth dist: {dict(sorted(defaultdict(int, {d: train_depths.count(d) for d in set(train_depths)}).items()))}")
    print(f"  Train trace len: mean={sum(train_traces)/len(train_traces):.1f}, "
          f"max={max(train_traces)}, min={min(train_traces)}")
    print(f"  Test depth dist:  {dict(sorted(defaultdict(int, {d: test_depths.count(d) for d in set(test_depths)}).items()))}")
    print(f"  max_len={max_len}, vocab={tok.vocab_size}")

    all_dyn = {}
    all_final = {}
    saved_states = {}

    # ── Main training ──
    for mt in MASK_TYPES:
        print(f"\n{'━' * 60}\n▶ {mt}\n{'━' * 60}")
        m, d = train_model(mt, tok, train_data, test_data, test_metas, max_len)
        all_dyn[mt] = d
        saved_states[mt] = {k: v.cpu().clone() for k, v in m.state_dict().items()}
        save_checkpoint(exp_name, saved_states[mt], tag=mt)

        # Standard eval
        for dp in DECODE_POLICIES:
            ps = gen_eval_with_stats(m, tok, test_data, test_metas, max_len,
                                     decode_policy=dp, device=DEVICE)
            acc = sum(r['trace_correct'] for r in ps) / len(ps)
            key = f'{mt}_standard_{dp}'
            all_final[key] = {
                'accuracy': acc, 'n': len(ps),
                'stratified': stratify_results(ps),
            }
            print(f"    standard {dp}: {acc:.4f}")

        # Corner cases
        for cat in ['linear_chain', 'max_branch', 'flat', 'deep_narrow']:
            cc, cc_m = gen_corner_case_test(N_TEST, seed=6000, category=cat)
            if not cc:
                continue
            for dp in DECODE_POLICIES:
                ps = gen_eval_with_stats(m, tok, cc, cc_m, max_len,
                                         decode_policy=dp, device=DEVICE)
                acc = sum(r['trace_correct'] for r in ps) / len(ps)
                all_final[f'{mt}_corner_{cat}_{dp}'] = {'accuracy': acc, 'n': len(cc)}
                print(f"    corner/{cat} {dp}: {acc:.4f}")

        # Depth sweep (in-distribution)
        print(f"  Depth sweep (in-distribution)...")
        for min_d in range(1, MAX_DEPTH + 1):
            cc, cc_m = gen_min_depth_test(500, seed=6500 + min_d, min_depth=min_d)
            if not cc:
                continue
            for dp in DECODE_POLICIES:
                ps = gen_eval_with_stats(m, tok, cc, cc_m, max_len,
                                         decode_policy=dp, device=DEVICE)
                acc = sum(r['trace_correct'] for r in ps) / len(ps)
                all_final[f'{mt}_depth_sweep_{min_d}_{dp}'] = {
                    'accuracy': acc, 'n': len(cc),
                }
                print(f"    depth>={min_d} {dp}: {acc:.4f}")

        # Depth OOD (beyond training distribution)
        print(f"  Depth OOD...")
        for ood_d in [MAX_DEPTH + 1, MAX_DEPTH + 2]:
            cc, cc_m = gen_depth_ood_test(300, seed=7500 + ood_d,
                                           target_depth=ood_d)
            if not cc:
                continue
            for dp in DECODE_POLICIES:
                ps = gen_eval_with_stats(m, tok, cc, cc_m, max_len,
                                         decode_policy=dp, device=DEVICE)
                acc = sum(r['trace_correct'] for r in ps) / len(ps)
                all_final[f'{mt}_ood_depth_{ood_d}_{dp}'] = {
                    'accuracy': acc, 'n': len(cc),
                }
                print(f"    OOD depth={ood_d} {dp}: {acc:.4f}")

        # DAG-aware decode analysis (confidence only — the interesting case)
        print(f"  DAG analysis...")
        ps_conf = gen_eval_with_stats(m, tok, test_data, test_metas, max_len,
                                       decode_policy='confidence', device=DEVICE)
        dag = analyse_dag_violations(ps_conf)
        all_final[f'{mt}_dag_violations'] = dag
        print(f"    DAG violation rate: {dag['violation_rate']:.3f} "
              f"(correct: {dag['mean_viol_rate_correct']:.3f}, "
              f"wrong: {dag['mean_viol_rate_wrong']:.3f})"
              if dag['mean_viol_rate_correct'] is not None and
                 dag['mean_viol_rate_wrong'] is not None
              else f"    DAG violation rate: {dag['violation_rate']:.3f}")

        # Compare DAG violations across decode policies
        for dp in DECODE_POLICIES:
            if dp == 'confidence':
                continue  # already done
            ps_dp = gen_eval_with_stats(m, tok, test_data, test_metas, max_len,
                                         decode_policy=dp, device=DEVICE)
            dag_dp = analyse_dag_violations(ps_dp)
            all_final[f'{mt}_dag_violations_{dp}'] = dag_dp
            print(f"    DAG violation ({dp}): {dag_dp['violation_rate']:.3f}")

        # Operator error propagation
        op_prop = analyse_operator_error_propagation(ps_conf)
        all_final[f'{mt}_op_propagation'] = op_prop
        for op, stats in op_prop.items():
            pr = stats['propagation_rate']
            tr = stats['tolerance_rate']
            print(f"    Op {OP_NAMES.get(op, op)}: propagation={pr:.3f} "
                  f"tolerance={tr:.3f}"
                  if pr is not None and tr is not None
                  else f"    Op {OP_NAMES.get(op, op)}: no child errors")

        # Critical chain sweep
        crit_sweep = analyse_critical_chain_sweep(ps_conf)
        all_final[f'{mt}_critical_chain'] = crit_sweep
        for cl, v in crit_sweep.items():
            print(f"    critical_chain={cl}: acc={v['acc']:.4f} (n={v['n']})")

        # Error localization
        for min_d in [2, 3, 4]:
            if min_d > MAX_DEPTH:
                continue
            cc, cc_m = gen_min_depth_test(500, seed=6500 + min_d, min_depth=min_d)
            if not cc:
                continue
            ps = gen_eval_with_stats(m, tok, cc, cc_m, max_len,
                                     decode_policy='confidence', device=DEVICE)
            el = analyse_error_localization(ps)
            all_final[f'{mt}_error_loc_{min_d}'] = el

        # Depth rarity
        rarity = analyse_depth_rarity(ps_conf, test_metas)
        all_final[f'{mt}_depth_rarity'] = rarity
        print(f"    Rarity corr: {rarity['corr']:.3f}"
              if rarity['corr'] else "    Rarity corr: N/A")

        # PUMA coverage (puma only)
        if mt == 'puma':
            cov = simulate_puma_coverage(m, tok, test_data, test_metas, max_len,
                                         device=DEVICE)
            all_final[f'{mt}_coverage'] = cov
            print(f"    Coverage corr: {cov['corr']:.3f}"
                  if cov['corr'] else "    Coverage corr: N/A")

        # Confidence calibration
        cal = analyse_confidence_calibration(m, tok, test_data, test_metas, max_len,
                                             device=DEVICE)
        all_final[f'{mt}_calibration'] = cal

        del m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
            print(f"\n{'━' * 60}\n▶ Continuation: {label} "
                  f"({CONTINUATION_ITERS} iters)\n{'━' * 60}")
            m, d = train_model(tgt, tok, train_data, test_data, test_metas,
                               max_len, max_iters=CONTINUATION_ITERS,
                               init_state=saved_states[src])
            all_dyn[label] = d

            for dp in DECODE_POLICIES:
                ps = gen_eval_with_stats(m, tok, test_data, test_metas, max_len,
                                         decode_policy=dp, device=DEVICE)
                acc = sum(r['trace_correct'] for r in ps) / len(ps)
                all_final[f'{label}_standard_{dp}'] = {'accuracy': acc, 'n': len(ps)}
                print(f"    standard {dp}: {acc:.4f}")

            # Depth sweep for continuation
            for min_d in range(1, MAX_DEPTH + 1):
                cc, cc_m = gen_min_depth_test(500, seed=6500 + min_d, min_depth=min_d)
                if not cc:
                    continue
                for dp in DECODE_POLICIES:
                    ps = gen_eval_with_stats(m, tok, cc, cc_m, max_len,
                                             decode_policy=dp, device=DEVICE)
                    acc = sum(r['trace_correct'] for r in ps) / len(ps)
                    all_final[f'{label}_depth_sweep_{min_d}_{dp}'] = {
                        'accuracy': acc, 'n': len(cc),
                    }
                    print(f"    depth>={min_d} {dp}: {acc:.4f}")

            del m
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ── Figures & save ──
    figs = make_figures(all_dyn, all_final)
    sd = {
        'config': {k: globals()[k] for k in [
            'MAX_DEPTH', 'MIN_ARGS', 'MAX_ARGS', 'EXPAND_PROB',
            'MAX_ANS_LEN', 'DEPTH_DECAY',
            'N_TRAIN', 'N_TEST', 'MAX_ITERS', 'BATCH_SIZE',
            'N_LAYER', 'N_HEAD', 'N_EMBD', 'MASK_TYPES', 'DECODE_POLICIES',
        ]},
    }
    for k, v in all_dyn.items():
        sd[f'dyn_{k}'] = {'checkpoints': v['checkpoints'],
                          'train_loss': v['train_loss']}
    for k, v in all_final.items():
        sd[f'final_{k}'] = v
    save_results(exp_name, sd, figures=figs)

    # Summary
    print(f"\n{'=' * 70}\n  SUMMARY\n{'=' * 70}")
    all_conditions = list(MASK_TYPES)
    if not getattr(args, 'no_continuation', False):
        for src, tgt in [('random', 'puma'), ('puma', 'random')]:
            if f'{src}_to_{tgt}' in all_dyn:
                all_conditions.append(f'{src}_to_{tgt}')

    print(f"\n  {'Test':<35s}", end='')
    for mt in all_conditions:
        print(f" {mt:>14s}", end='')
    print()

    for dp in DECODE_POLICIES:
        tests = ['standard', 'corner_linear_chain', 'corner_max_branch',
                 'corner_flat', 'corner_deep_narrow']
        for tt in tests:
            accs = [all_final.get(f'{mt}_{tt}_{dp}', {}).get('accuracy')
                    for mt in all_conditions]
            if any(a is not None for a in accs):
                print(f"  {tt + '_' + dp:<35s}", end='')
                for a in accs:
                    print(f" {a:>14.4f}" if a is not None else f" {'N/A':>14s}", end='')
                print()
        for min_d in range(1, MAX_DEPTH + 1):
            accs = [all_final.get(f'{mt}_depth_sweep_{min_d}_{dp}', {}).get('accuracy')
                    for mt in all_conditions]
            if any(a is not None for a in accs):
                print(f"  {'depth>=' + str(min_d) + '_' + dp:<35s}", end='')
                for a in accs:
                    print(f" {a:>14.4f}" if a is not None else f" {'N/A':>14s}", end='')
                print()
        # OOD depths
        for ood_d in [MAX_DEPTH + 1, MAX_DEPTH + 2]:
            accs = [all_final.get(f'{mt}_ood_depth_{ood_d}_{dp}', {}).get('accuracy')
                    for mt in all_conditions]
            if any(a is not None for a in accs):
                print(f"  {'OOD_d=' + str(ood_d) + '_' + dp:<35s}", end='')
                for a in accs:
                    print(f" {a:>14.4f}" if a is not None else f" {'N/A':>14s}", end='')
                print()

    # DAG violation summary
    print(f"\n  {'DAG violations':<35s}", end='')
    for mt in all_conditions:
        r = all_final.get(f'{mt}_dag_violations', {})
        vr = r.get('violation_rate')
        print(f" {vr:>14.3f}" if vr is not None else f" {'N/A':>14s}", end='')
    print()

    return all_dyn, all_final


if __name__ == '__main__':
    args = parse_args()
    seeds = args.seeds if args.seeds else [SEED]
    for si, seed in enumerate(seeds):
        globals()['SEED'] = seed
        t = f"{args.tag}_s{seed}" if args.tag and len(seeds) > 1 else args.tag
        if len(seeds) > 1:
            print(f"\n{'#' * 70}\n# Seed {seed} ({si + 1}/{len(seeds)})\n{'#' * 70}")
        run(tag=t)
