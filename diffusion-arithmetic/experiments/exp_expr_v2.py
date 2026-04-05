"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Expression Evaluation v2 — Tree Depth + PUMA Coverage Deficit
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Task: evaluate arithmetic expressions → fixed-width result
    Input:  "(3+5)*(7-2)=?????"  (expression + masked answer)
    Output: "00040"              (zero-padded result)

  This is the FORWARD direction of Countdown, ensuring:
    - Solution uniqueness (expression → unique result)
    - Fixed-length output suitable for MDM
    - Tree depth as natural difficulty axis

  Dependency analog:
    Addition carry chain  ←→  Expression tree depth
    g/k position          ←→  Leaf operand (directly computable)
    p position            ←→  Internal node (requires sub-tree evaluation)
    LSB oracle order      ←→  Post-order traversal (leaves first)

  Training: random vs puma
  Decode:   confidence | postorder_oracle | random
  Analyses: depth stratification, intermediate carry × accuracy,
            PUMA coverage, error localization, depth sweep
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import sys, os, time, math, json, random
from collections import defaultdict
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

EXP_NAME = 'exp_expr_v2'

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Config
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ANS_LEN = 6           # zero-padded result width (supports up to 999999)
MAX_DEPTH = 5         # max expression tree depth for training
N_OPERANDS = 6        # number of operands per expression (Countdown-style)
OPERAND_MAX = 25      # max operand value

N_TRAIN = 50000; N_TEST = 1000; BATCH_SIZE = 200
MAX_EPOCHS = 5000; EVAL_EVERY = 100; LOG_EVERY = 50
GEN_EVAL_EVERY = 200; GEN_EVAL_N = 500

MASK_TYPES = ['random', 'puma']
DECODE_POLICIES = ['confidence', 'postorder_oracle', 'random']

N_LAYER = 2; N_HEAD = 2; N_EMBD = 128; DROPOUT = 0.2; POS_ENC = 'absolute'
LR = 1e-3; MIN_LR = 1e-4; WARMUP_EPOCHS = 10; GRAD_CLIP = 1.0
PUMA_TAU = 0.9; PUMA_K_START = 2; PUMA_K_END = ANS_LEN
SEED = 42

DEPTH_SWEEP = [1, 2, 3, 4, 5]


def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--ans-len', type=int); p.add_argument('--max-depth', type=int)
    p.add_argument('--n-operands', type=int); p.add_argument('--operand-max', type=int)
    p.add_argument('--n-train', type=int); p.add_argument('--n-test', type=int)
    p.add_argument('--epochs', type=int); p.add_argument('--batch-size', type=int)
    p.add_argument('--eval-every', type=int); p.add_argument('--gen-eval-every', type=int)
    p.add_argument('--n-layer', type=int); p.add_argument('--n-head', type=int)
    p.add_argument('--n-embd', type=int); p.add_argument('--dropout', type=float)
    p.add_argument('--puma-tau', type=float)
    p.add_argument('--masks', nargs='+'); p.add_argument('--decode', nargs='+')
    p.add_argument('--tag', type=str, default=''); p.add_argument('--seed', type=int)
    p.add_argument('--seeds', nargs='+', type=int)
    try:
        args, _ = p.parse_known_args()
    except SystemExit:
        args, _ = p.parse_known_args([])
    g = globals()
    for a, gl in {'ans_len': 'ANS_LEN', 'max_depth': 'MAX_DEPTH',
                   'n_operands': 'N_OPERANDS', 'operand_max': 'OPERAND_MAX',
                   'n_train': 'N_TRAIN', 'n_test': 'N_TEST', 'epochs': 'MAX_EPOCHS',
                   'batch_size': 'BATCH_SIZE', 'eval_every': 'EVAL_EVERY',
                   'gen_eval_every': 'GEN_EVAL_EVERY', 'n_layer': 'N_LAYER',
                   'n_head': 'N_HEAD', 'n_embd': 'N_EMBD', 'dropout': 'DROPOUT',
                   'puma_tau': 'PUMA_TAU', 'seed': 'SEED'}.items():
        v = getattr(args, a, None)
        if v is not None: g[gl] = v
    if args.masks: g['MASK_TYPES'] = args.masks
    if args.decode: g['DECODE_POLICIES'] = args.decode
    return args


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Expression Tree
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

OPS = ['+', '-', '*']  # no division to avoid fractions

class ExprNode:
    """Binary expression tree node."""
    __slots__ = ['op', 'val', 'left', 'right', 'depth', 'result']

    def __init__(self, op=None, val=None, left=None, right=None):
        self.op = op; self.val = val; self.left = left; self.right = right
        self.depth = 0; self.result = None

    def is_leaf(self):
        return self.op is None

    def evaluate(self):
        if self.is_leaf():
            self.result = self.val; self.depth = 0; return self.val
        lv = self.left.evaluate(); rv = self.right.evaluate()
        if self.op == '+': self.result = lv + rv
        elif self.op == '-': self.result = lv - rv
        elif self.op == '*': self.result = lv * rv
        self.depth = 1 + max(self.left.depth, self.right.depth)
        return self.result

    def to_string(self):
        if self.is_leaf(): return str(self.val)
        ls = self.left.to_string(); rs = self.right.to_string()
        return f"({ls}{self.op}{rs})"

    def max_depth(self):
        if self.is_leaf(): return 0
        return 1 + max(self.left.max_depth(), self.right.max_depth())

    def n_internal(self):
        """Number of internal (operator) nodes = number of operations."""
        if self.is_leaf(): return 0
        return 1 + self.left.n_internal() + self.right.n_internal()

    def all_intermediates(self):
        """Collect all intermediate results and their depths."""
        results = []
        if not self.is_leaf():
            results.append({'result': self.result, 'depth': self.depth, 'op': self.op})
            results.extend(self.left.all_intermediates())
            results.extend(self.right.all_intermediates())
        return results


def gen_random_tree(n_ops, rng, operand_max=None):
    """Generate random binary expression tree with exactly n_ops operations.
    n_ops operands-1 operations means n_ops+1 leaf nodes."""
    if operand_max is None: operand_max = OPERAND_MAX

    if n_ops == 0:
        return ExprNode(val=rng.randint(1, operand_max))

    # Split n_ops-1 remaining operations between left and right
    left_ops = rng.randint(0, n_ops - 1)
    right_ops = n_ops - 1 - left_ops
    op = rng.choice(OPS)
    left = gen_random_tree(left_ops, rng, operand_max)
    right = gen_random_tree(right_ops, rng, operand_max)
    return ExprNode(op=op, left=left, right=right)


def gen_deep_tree(target_depth, rng, operand_max=None):
    """Generate a tree that achieves exactly target_depth.
    One branch goes to target_depth-1, other is a random smaller tree."""
    if operand_max is None: operand_max = OPERAND_MAX

    if target_depth == 0:
        return ExprNode(val=rng.randint(1, operand_max))

    op = rng.choice(OPS)
    # One child must have depth target_depth-1
    deep_child = gen_deep_tree(target_depth - 1, rng, operand_max)
    # Other child is shallower (0 to target_depth-2)
    shallow_depth = rng.randint(0, max(0, target_depth - 2))
    shallow_child = gen_deep_tree(shallow_depth, rng, operand_max)

    if rng.random() < 0.5:
        return ExprNode(op=op, left=deep_child, right=shallow_child)
    else:
        return ExprNode(op=op, left=shallow_child, right=deep_child)


def gen_chain_tree(chain_len, rng, operand_max=None):
    """Generate a pure left-chain tree (= carry chain analog).
    Depth = chain_len, every internal node has a leaf as right child."""
    if operand_max is None: operand_max = OPERAND_MAX

    if chain_len == 0:
        return ExprNode(val=rng.randint(1, operand_max))

    op = rng.choice(OPS)
    left = gen_chain_tree(chain_len - 1, rng, operand_max)
    right = ExprNode(val=rng.randint(1, operand_max))
    return ExprNode(op=op, left=left, right=right)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Data Formatting
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _format_sample(tree):
    """Format: 'expression=zero_padded_result'"""
    expr_str = tree.to_string()
    result = tree.evaluate()
    if result < 0:
        # Handle negative: sign + zero-padded magnitude
        res_str = '-' + str(abs(result)).zfill(ANS_LEN - 1)
    else:
        res_str = str(result).zfill(ANS_LEN)
    # Truncate if too long
    if len(res_str) > ANS_LEN:
        return None  # overflow, skip
    return f"{expr_str}={res_str}"


def _parse_expression(s):
    """Extract expression and answer from formatted string."""
    eq_pos = s.rindex('=')
    return s[:eq_pos], s[eq_pos + 1:]


def _tree_stats(tree):
    """Compute statistics about the expression tree."""
    tree.evaluate()
    intermediates = tree.all_intermediates()
    max_intermediate = max((abs(im['result']) for im in intermediates), default=0)
    return {
        'depth': tree.max_depth(),
        'n_ops': tree.n_internal(),
        'result': tree.result,
        'max_intermediate': max_intermediate,
        'intermediates': intermediates,
        'is_chain': _is_chain(tree),
    }


def _is_chain(tree):
    """Check if tree is a pure chain (every internal node has a leaf child)."""
    if tree.is_leaf(): return True
    if not tree.left.is_leaf() and not tree.right.is_leaf(): return False
    if tree.left.is_leaf(): return _is_chain(tree.right)
    return _is_chain(tree.left)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Difficulty Classification
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DEPTH_CATS = ['depth_1', 'depth_2', 'depth_3', 'depth_4', 'depth_5plus']
DEPTH_CAT_TO_ID = {n: i for i, n in enumerate(DEPTH_CATS)}

def _depth_to_cat(depth):
    if depth <= 1: return 'depth_1'
    if depth == 2: return 'depth_2'
    if depth == 3: return 'depth_3'
    if depth == 4: return 'depth_4'
    return 'depth_5plus'


# Each answer digit's difficulty depends on how many carries propagate through it
# from intermediate computations. We track this via:
#   - digit position (MSB vs LSB)
#   - whether the digit requires carry from intermediate multiplications
#   (multiplication generates the most "deep" carries)

def _digit_difficulty(tree, ans_len):
    """Per-digit difficulty classification based on tree structure.
    Returns list of context names for each answer digit."""
    tree.evaluate()
    result = tree.result
    depth = tree.max_depth()
    n_muls = sum(1 for im in tree.all_intermediates() if im['op'] == '*')

    contexts = []
    for d in range(ans_len):
        # MSB digits are harder (require full tree evaluation)
        # LSB digits are easier (determined by last operation's low bits)
        digit_pos = ans_len - 1 - d  # 0=LSB, ans_len-1=MSB
        if digit_pos == 0:
            ctx = 'lsb'  # easiest: last digit
        elif depth <= 1:
            ctx = 'shallow_digit'
        elif n_muls > 0 and digit_pos >= ans_len // 2:
            ctx = 'mul_msb'  # hardest: MSB after multiplication chains
        elif depth >= 3:
            ctx = 'deep_digit'
        else:
            ctx = 'mid_digit'
        contexts.append(ctx)
    return contexts


DIGIT_CONTEXTS = ['lsb', 'shallow_digit', 'mid_digit', 'deep_digit', 'mul_msb']
DIGIT_CTX_TO_ID = {n: i for i, n in enumerate(DIGIT_CONTEXTS)}


def build_tok():
    return CharTokenizer(list('0123456789+-*()='), {'mask': 'M', 'pad': 'P'})


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Data Generation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def gen_data(n, seed, max_depth=None, depth_balanced=True):
    """Generate expression evaluation data.
    depth_balanced=True: equal samples per depth level (like carry-balanced addition)
    depth_balanced=False: natural distribution (shallow-heavy)
    """
    if max_depth is None: max_depth = MAX_DEPTH
    rng = random.Random(seed)
    seen = set(); data = []

    if depth_balanced:
        per_depth = max(1, n // max_depth)
        for target_d in range(1, max_depth + 1):
            count = 0
            for _ in range(per_depth * 50):
                if count >= per_depth: break
                n_ops = rng.randint(target_d, min(target_d + 2, N_OPERANDS - 1))
                tree = gen_deep_tree(target_d, rng)
                s = _format_sample(tree)
                if s and s not in seen:
                    seen.add(s)
                    tree.evaluate()
                    data.append({
                        'string': s, 'tree_stats': _tree_stats(tree),
                        'digit_contexts': _digit_difficulty(tree, ANS_LEN)
                    })
                    count += 1
    else:
        for _ in range(n * 10):
            if len(data) >= n: break
            n_ops = rng.randint(1, N_OPERANDS - 1)
            tree = gen_random_tree(n_ops, rng)
            s = _format_sample(tree)
            if s and s not in seen:
                seen.add(s)
                tree.evaluate()
                data.append({
                    'string': s, 'tree_stats': _tree_stats(tree),
                    'digit_contexts': _digit_difficulty(tree, ANS_LEN)
                })

    rng.shuffle(data)
    if len(data) < n: print(f"  WARNING: gen_data: {len(data)}/{n}")
    return data[:n]


def gen_depth_test(n, seed, min_depth):
    """Generate test set with tree depth >= min_depth."""
    rng = random.Random(seed); data = []; seen = set()
    for _ in range(n * 100):
        if len(data) >= n: break
        tree = gen_deep_tree(min_depth, rng)
        s = _format_sample(tree)
        if s and s not in seen and tree.max_depth() >= min_depth:
            seen.add(s); tree.evaluate()
            data.append({
                'string': s, 'tree_stats': _tree_stats(tree),
                'digit_contexts': _digit_difficulty(tree, ANS_LEN)
            })
    if len(data) < n: print(f"  WARNING: depth>={min_depth}: {len(data)}/{n}")
    return data[:n]


def gen_chain_test(n, seed, chain_len):
    """Generate pure chain expressions (= full_propagate analog)."""
    rng = random.Random(seed); data = []; seen = set()
    for _ in range(n * 50):
        if len(data) >= n: break
        tree = gen_chain_tree(chain_len, rng)
        s = _format_sample(tree)
        if s and s not in seen:
            seen.add(s); tree.evaluate()
            data.append({
                'string': s, 'tree_stats': _tree_stats(tree),
                'digit_contexts': _digit_difficulty(tree, ANS_LEN)
            })
    if len(data) < n: print(f"  WARNING: chain={chain_len}: {len(data)}/{n}")
    return data[:n]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Probe
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@torch.no_grad()
def probe_per_position(model, tokenizer, test_data, max_len, device=None):
    """Fully-masked probe with per-depth-category tracking."""
    if device is None: device = DEVICE
    model.eval(); mask_id = tokenizer.special_ids['mask']
    strings = [d['string'] for d in test_data]
    ids_all, ans_all = encode_samples(strings, tokenizer, max_len)
    ids_all, ans_all = ids_all.to(device), ans_all.to(device)
    N = len(test_data); _arange = torch.arange(ANS_LEN, device=device)

    # Per-depth category tracking
    n_cats = len(DEPTH_CATS)
    cat_conf = torch.zeros(n_cats, device=device)
    cat_correct = torch.zeros(n_cats, device=device)
    cat_count = torch.zeros(n_cats, dtype=torch.long, device=device)
    total_loss = torch.tensor(0.0, device=device)
    total_n = torch.tensor(0, dtype=torch.long, device=device)

    # Build depth category tensor
    cat_ids = torch.zeros(N, dtype=torch.long, device=device)
    for si, d in enumerate(test_data):
        depth = d['tree_stats']['depth']
        cat_ids[si] = DEPTH_CAT_TO_ID[_depth_to_cat(depth)]

    for st in range(0, N, 128):
        en = min(st + 128, N)
        ids, ans = ids_all[st:en], ans_all[st:en]; B = ids.shape[0]; T = ids.shape[1]
        ans_pos = (ans.unsqueeze(1) + _arange).clamp(max=T - 1)
        bi = torch.arange(B, device=device).unsqueeze(1).expand_as(ans_pos)

        xm = ids.clone(); xm[bi, ans_pos] = mask_id
        logits = model(xm); al = logits[bi, ans_pos]; tgt = ids[bi, ans_pos]
        lp = F.log_softmax(al, dim=-1)
        losses = -lp.gather(2, tgt.unsqueeze(2)).squeeze(2)
        cl = al.clone(); cl[:, :, mask_id] = -float('inf')
        probs = F.softmax(cl, dim=-1)
        confs = probs.max(dim=-1).values
        corrects = (probs.argmax(dim=-1) == tgt).float()

        total_loss += losses.sum(); total_n += B * ANS_LEN

        # Per-sample depth category (broadcast to all positions)
        cb = cat_ids[st:en]
        # Each sample's depth category applies to all its ANS_LEN positions
        for ci in range(n_cats):
            mask_ci = (cb == ci)
            if mask_ci.any():
                cat_conf[ci] += confs[mask_ci].sum()
                cat_correct[ci] += corrects[mask_ci].sum()
                cat_count[ci] += mask_ci.sum() * ANS_LEN

    overall_loss = (total_loss / total_n.clamp(1)).item()
    overall_acc = cat_correct.sum().item() / cat_count.sum().clamp(1).item()
    dep_context = {}
    for ci, cn in enumerate(DEPTH_CATS):
        nc = cat_count[ci].item()
        if nc > 0:
            dep_context[cn] = {'mean_conf': cat_conf[ci].item() / nc,
                                'mean_acc': cat_correct[ci].item() / nc, 'n': nc}
    return {'overall_loss': overall_loss, 'overall_acc': overall_acc, 'dep_context': dep_context}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Generation (multi-reveal)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@torch.no_grad()
def gen_eval(model, tokenizer, test_data, max_len, decode_policy='confidence',
             n_decode_steps=None, device=None):
    """Generate answers with multi-reveal decode."""
    if device is None: device = DEVICE
    if n_decode_steps is None: n_decode_steps = max(ANS_LEN, PUMA_K_END)
    mask_id = tokenizer.special_ids['mask']
    pad_id = tokenizer.special_ids['pad']
    model.eval(); results = []; _ar = torch.arange(ANS_LEN, device=device)

    for st in range(0, len(test_data), 128):
        batch = test_data[st:min(st + 128, len(test_data))]; B = len(batch)
        # Encode prefix (expression + '=')
        penc = [tokenizer.encode(d['string'].rsplit('=', 1)[0] + '=') for d in batch]
        pm = max(len(p) for p in penc)
        pids = torch.full((B, pm), pad_id, dtype=torch.long)
        for i, e in enumerate(penc): pids[i, :len(e)] = torch.tensor(e)
        pids = pids.to(device)

        # Append masked answer
        x = torch.cat([pids, torch.full((B, ANS_LEN), mask_id, dtype=torch.long, device=device)], dim=1)
        ans_start = pm  # all same since we padded prefix
        ap = ans_start + _ar  # [ANS_LEN]
        bi = torch.arange(B, device=device).unsqueeze(1).expand(B, ANS_LEN)

        # Gold answer
        gold_enc = [tokenizer.encode(d['string'].rsplit('=', 1)[1][:ANS_LEN]) for d in batch]
        gold = torch.full((B, ANS_LEN), pad_id, dtype=torch.long, device=device)
        for i, e in enumerate(gold_enc):
            gold[i, :len(e)] = torch.tensor(e, device=device)

        # Multi-reveal decode
        is_m = torch.ones(B, ANS_LEN, dtype=torch.bool, device=device)
        for step in range(n_decode_steps):
            if not is_m.any(): break
            logits = model(x)
            al = logits[:, ap, :].clone(); al[:, :, mask_id] = -float('inf')
            probs = F.softmax(al, dim=-1)
            confs = probs.max(dim=-1).values; preds = probs.argmax(dim=-1)
            confs[~is_m] = -float('inf')

            nm = is_m.sum(dim=1).float()
            K_rem = max(n_decode_steps - step, 1)
            nr = (nm / K_rem).ceil().long().clamp(min=1)

            if decode_policy == 'confidence':
                ranked = confs.argsort(dim=1, descending=True)
            elif decode_policy == 'postorder_oracle':
                # Oracle: reveal LSB first (right to left), like addition LSB
                static_rank = torch.arange(ANS_LEN - 1, -1, -1, device=device).expand(B, -1)
                rank_vals = torch.where(is_m, static_rank, torch.tensor(-1, device=device))
                ranked = rank_vals.argsort(dim=1, descending=True)
            else:  # random
                ranked = torch.randperm(ANS_LEN, device=device).expand(B, -1)

            rop = torch.zeros(B, ANS_LEN, dtype=torch.long, device=device)
            rop.scatter_(1, ranked, _ar.expand(B, -1))
            reveal = (rop < nr.unsqueeze(1)) & is_m

            for i in range(B):
                for j in reveal[i].nonzero(as_tuple=True)[0]:
                    x[i, ap[j]] = preds[i, j]
            is_m = is_m & ~reveal

        # Collect results
        pred_at = x[:, ap]
        pos_correct = (pred_at == gold)
        sample_correct = pos_correct.all(dim=1)

        for i in range(B):
            errs = (~pos_correct[i]).nonzero(as_tuple=True)[0].tolist()
            results.append({
                'correct': sample_correct[i].item(),
                'pos_correct': pos_correct[i].tolist(),
                'error_positions': errs,
                'tree_stats': batch[i].get('tree_stats', {}),
            })
    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Analysis
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def stratify_by_depth(per_sample):
    """Stratify accuracy by tree depth."""
    bk = defaultdict(list)
    for r in per_sample:
        d = r['tree_stats'].get('depth', 0)
        bk[_depth_to_cat(d)].append(r['correct'])
    return {k: {'acc': sum(v) / len(v), 'n': len(v)} for k, v in sorted(bk.items())}


def stratify_by_chain(per_sample):
    """Stratify by chain vs non-chain."""
    bk = defaultdict(list)
    for r in per_sample:
        is_chain = r['tree_stats'].get('is_chain', False)
        bk['chain' if is_chain else 'branched'].append(r['correct'])
    return {k: {'acc': sum(v) / len(v), 'n': len(v)} for k, v in sorted(bk.items())}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Training
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def train_model(mask_type, tokenizer, train_data, test_data, max_len, device=None):
    if device is None: device = DEVICE
    strings = [d['string'] for d in train_data]
    train_ids, train_ans = encode_samples(strings, tokenizer, max_len)
    train_ids, train_ans = train_ids.to(device), train_ans.to(device)
    N, T = train_ids.shape; bpe = (N + BATCH_SIZE - 1) // BATCH_SIZE
    total_iters = MAX_EPOCHS * bpe
    mask_id = tokenizer.special_ids['mask']; pad_id = tokenizer.special_ids['pad']
    _arange = torch.arange(ANS_LEN, device=device)

    model = Transformer(vocab_size=len(tokenizer), block_size=max_len + 8,
                        n_layer=N_LAYER, n_head=N_HEAD, n_embd=N_EMBD,
                        dropout=DROPOUT, is_causal=False, pos_enc=POS_ENC).to(device)
    print(f"  [{mask_type}] params={model.n_params:,}, {bpe} batches/epoch, T={T}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.99), weight_decay=0.1)
    warmup_iters = WARMUP_EPOCHS * bpe
    def get_lr(it):
        if it < warmup_iters: return LR * it / max(warmup_iters, 1)
        ratio = (it - warmup_iters) / max(total_iters - warmup_iters, 1)
        return MIN_LR + 0.5 * (LR - MIN_LR) * (1 + math.cos(math.pi * min(ratio, 1.0)))

    dynamics = {'checkpoints': [], 'train_loss': []}
    best_loss, best_state = float('inf'), None; it = 0; tg = 0; t0 = time.time()

    # PUMA streaming buffer
    uses_streaming = (mask_type == 'puma')
    if uses_streaming:
        puma_x0 = torch.zeros(BATCH_SIZE, T, dtype=torch.long, device=device)
        puma_z = torch.zeros(BATCH_SIZE, T, dtype=torch.long, device=device)
        puma_ans = torch.zeros(BATCH_SIZE, dtype=torch.long, device=device)
        puma_stage = torch.zeros(BATCH_SIZE, dtype=torch.long, device=device)
        pool = torch.randperm(N); pool_ptr = 0

        def _refresh(indices):
            nonlocal pool_ptr, pool
            idx_t = torch.tensor(indices, device=device); n = len(indices)
            if pool_ptr + n > len(pool): pool = torch.randperm(N); pool_ptr = 0
            si = pool[pool_ptr:pool_ptr + n].to(device); pool_ptr += n
            puma_x0[idx_t] = train_ids[si]; puma_z[idx_t] = train_ids[si].clone()
            puma_ans[idx_t] = train_ans[si]; puma_stage[idx_t] = 0
            ap = (puma_ans[idx_t].unsqueeze(1) + _arange).clamp(max=T - 1)
            bii = idx_t.unsqueeze(1).expand_as(ap)
            puma_z[bii, ap] = mask_id

        def _advance(logits, K_cur):
            nonlocal puma_stage
            B = BATCH_SIZE
            ap = (puma_ans.unsqueeze(1) + _arange).clamp(max=T - 1)
            bi = torch.arange(B, device=device).unsqueeze(1).expand_as(ap)
            is_m = (puma_z[bi, ap] == mask_id)
            if not is_m.any(): _refresh(list(range(B))); return
            nm = is_m.sum(dim=1).float()
            K_rem = (K_cur - puma_stage).clamp(min=1)
            nr = (nm / K_rem.float()).ceil().long().clamp(min=1)
            lp = logits[bi, ap].clone(); lp[:, :, mask_id] = -float('inf')
            confs = F.softmax(lp, dim=-1).max(dim=-1).values; confs[~is_m] = -float('inf')
            ranked = confs.argsort(dim=1, descending=True)
            rop = torch.zeros_like(ranked); rop.scatter_(1, ranked, _arange.expand(B, -1))
            reveal = ((rop < nr.unsqueeze(1)) | (confs > PUMA_TAU)) & is_m
            puma_z[bi[reveal], ap[reveal]] = puma_x0[bi[reveal], ap[reveal]]
            puma_stage += 1
            done = (~(puma_z[bi, ap] == mask_id).any(dim=1)) | (puma_stage >= K_cur)
            if done.any(): _refresh(done.nonzero(as_tuple=True)[0].tolist())

        _refresh(list(range(BATCH_SIZE)))

    def _do_eval(epoch):
        nonlocal best_loss, best_state
        model.eval()
        probe = probe_per_position(model, tokenizer, test_data, max_len, device)
        dynamics['checkpoints'].append({'epoch': epoch, 'iter': it, 'tg': tg, **probe})
        if probe['overall_loss'] < best_loss and epoch > 0:
            best_loss = probe['overall_loss']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        dc = probe.get('dep_context', {})
        parts = [f"{c}={dc[c]['mean_acc']:.2f}" for c in DEPTH_CATS if c in dc]
        print(f"    [eval ep {epoch}] loss={probe['overall_loss']:.4f} acc={probe['overall_acc']:.4f} "
              + ' '.join(parts) + f" | {time.time()-t0:.0f}s")
        model.train()

    model.eval(); _do_eval(0); model.train()

    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_loss = torch.tensor(0.0, device=device); epoch_n = 0
        K_cur = PUMA_K_START + int((PUMA_K_END - PUMA_K_START) * epoch / MAX_EPOCHS) if uses_streaming else 0

        if uses_streaming:
            for _ in range(bpe):
                for pg in optimizer.param_groups: pg['lr'] = get_lr(it)
                m = (puma_z == mask_id)
                if m.sum() == 0: _refresh(list(range(BATCH_SIZE))); m = (puma_z == mask_id)
                logits = model(puma_z)
                loss = F.cross_entropy(logits[m], puma_x0[m]); tg += m.sum().item()
                optimizer.zero_grad(set_to_none=True); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP); optimizer.step()
                _advance(logits.detach(), K_cur)
                epoch_loss += loss.detach(); epoch_n += 1; it += 1
        else:
            perm = torch.randperm(N, device=device)
            for bi_idx in range(bpe):
                for pg in optimizer.param_groups: pg['lr'] = get_lr(it)
                idx = perm[bi_idx * BATCH_SIZE:min((bi_idx + 1) * BATCH_SIZE, N)]
                ids = train_ids[idx]; ans_s = train_ans[idx]; B_b = ids.shape[0]
                ap = (ans_s.unsqueeze(1) + _arange).clamp(max=T - 1)
                bii = torch.arange(B_b, device=device).unsqueeze(1).expand_as(ap)
                ans_mask = torch.zeros(B_b, T, dtype=torch.bool, device=device)
                for j in range(ANS_LEN): ans_mask[range(B_b), ap[:, j]] = True
                t_r = torch.rand(B_b, device=device)
                m_probs = t_r.unsqueeze(1) * ans_mask.float()
                m = torch.bernoulli(m_probs).bool()
                no_m = ~m.any(dim=1)
                if no_m.any():
                    rj = torch.randint(ANS_LEN, (no_m.sum(),), device=device)
                    fp = ap[no_m].gather(1, rj.unsqueeze(1)).squeeze(1)
                    m[no_m.nonzero(as_tuple=True)[0], fp] = True
                xm = ids.clone(); xm[m] = mask_id; logits = model(xm)
                if m.sum() == 0: it += 1; continue
                loss = F.cross_entropy(logits[m], ids[m]); tg += m.sum().item()
                optimizer.zero_grad(set_to_none=True); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP); optimizer.step()
                epoch_loss += loss.detach(); epoch_n += 1; it += 1

        tg_now = tg
        if epoch % LOG_EVERY == 0:
            dynamics['train_loss'].append((epoch, epoch_loss.item() / max(epoch_n, 1)))
            print(f"    ep {epoch:4d}/{MAX_EPOCHS} | loss {epoch_loss.item()/max(epoch_n,1):.4f} | "
                  f"lr {get_lr(it):.1e} | tg {tg:,} | {time.time()-t0:.0f}s")
        do_eval = (epoch % EVAL_EVERY == 0) or \
                  (epoch < MAX_EPOCHS * 0.1 and epoch % max(EVAL_EVERY // 5, 1) == 0)
        if do_eval and epoch < MAX_EPOCHS: _do_eval(epoch)

    if best_state: model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval(); _do_eval(MAX_EPOCHS)
    return model, dynamics


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Run
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run(tag=''):
    exp_name = f"{EXP_NAME}_{tag}" if tag else EXP_NAME
    print(f"\n{'='*70}\n  {exp_name}\n{'='*70}")
    print(f"  ANS_LEN={ANS_LEN}, MAX_DEPTH={MAX_DEPTH}, N_OPERANDS={N_OPERANDS}")
    print(f"  Model: L={N_LAYER} H={N_HEAD} E={N_EMBD}")
    print(f"  Masks: {MASK_TYPES}, Decode: {DECODE_POLICIES}")

    torch.manual_seed(SEED); random.seed(SEED)
    tok = build_tok()

    # Generate data
    print(f"\n  Generating {N_TRAIN} train expressions...")
    t0 = time.time()
    train_data = gen_data(N_TRAIN, seed=SEED, depth_balanced=True)
    max_len = max(len(tok.encode(d['string'])) for d in train_data) + 2
    print(f"  Train: {len(train_data)}, max_len={max_len}, {time.time()-t0:.1f}s")

    # Profile depth distribution
    depth_dist = defaultdict(int)
    for d in train_data: depth_dist[_depth_to_cat(d['tree_stats']['depth'])] += 1
    print(f"  Train depth dist: {dict(sorted(depth_dist.items()))}")

    print(f"\n  Generating {N_TEST} test expressions...")
    test_data = gen_data(N_TEST, seed=SEED + 1000, depth_balanced=True)
    print(f"  Test: {len(test_data)}")

    all_dyn = {}; all_final = {}

    for mt in MASK_TYPES:
        print(f"\n{'━'*60}\n  Training: {mt}\n{'━'*60}")
        m, dyn = train_model(mt, tok, train_data, test_data, max_len, device=DEVICE)
        all_dyn[f'dyn_{mt}'] = dyn

        # Standard eval
        for dp in DECODE_POLICIES:
            ps = gen_eval(m, tok, test_data, max_len, decode_policy=dp, device=DEVICE)
            acc = sum(r['correct'] for r in ps) / len(ps)
            depth_strat = stratify_by_depth(ps)
            chain_strat = stratify_by_chain(ps)
            all_final[f'{mt}_standard_{dp}'] = {
                'accuracy': acc, 'n': len(ps),
                'by_depth': depth_strat, 'by_chain': chain_strat}
            print(f"  {mt} {dp}: {acc:.4f}")
            for k, v in depth_strat.items(): print(f"    {k}: {v['acc']:.3f} ({v['n']})")

        # Depth sweep
        print(f"  Depth sweep...")
        for min_d in DEPTH_SWEEP:
            if min_d > MAX_DEPTH: continue
            dd = gen_depth_test(min(500, N_TEST), seed=SEED + 6500 + min_d, min_depth=min_d)
            if not dd: continue
            for dp in DECODE_POLICIES:
                ps = gen_eval(m, tok, dd, max_len, decode_policy=dp, device=DEVICE)
                acc = sum(r['correct'] for r in ps) / len(ps)
                all_final[f'{mt}_depth_sweep_{min_d}_{dp}'] = {
                    'accuracy': acc, 'n': len(dd), 'min_depth': min_d}
                print(f"    depth>={min_d} {dp}: {acc:.4f} (n={len(dd)})")

        # Chain test (pure chain = full_propagate analog)
        for cl in [2, 3, 4, 5]:
            if cl > MAX_DEPTH: continue
            cc = gen_chain_test(min(500, N_TEST), seed=SEED + 7000 + cl, chain_len=cl)
            if not cc: continue
            for dp in DECODE_POLICIES:
                ps = gen_eval(m, tok, cc, max_len, decode_policy=dp, device=DEVICE)
                acc = sum(r['correct'] for r in ps) / len(ps)
                all_final[f'{mt}_chain_{cl}_{dp}'] = {
                    'accuracy': acc, 'n': len(cc), 'chain_len': cl}
                print(f"    chain={cl} {dp}: {acc:.4f} (n={len(cc)})")

        del m; torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Save
    sd = {'config': {k: globals()[k] for k in [
        'ANS_LEN', 'MAX_DEPTH', 'N_OPERANDS', 'N_TRAIN', 'N_TEST', 'MAX_EPOCHS',
        'BATCH_SIZE', 'N_LAYER', 'N_HEAD', 'N_EMBD', 'MASK_TYPES', 'DECODE_POLICIES']}}
    for k, v in all_dyn.items():
        sd[k] = {'checkpoints': v['checkpoints'], 'train_loss': v['train_loss']}
    for k, v in all_final.items(): sd[f'final_{k}'] = v
    save_results(exp_name, sd)

    # Summary
    print(f"\n{'='*70}\n  SUMMARY\n{'='*70}")
    print(f"\n  {'Test':<35s}", end='')
    for mt in MASK_TYPES: print(f" {mt:>14s}", end='')
    print()
    for dp in DECODE_POLICIES:
        # Standard
        keys = [f'{mt}_standard_{dp}' for mt in MASK_TYPES]
        accs = [all_final.get(k, {}).get('accuracy') for k in keys]
        print(f"  {'standard_'+dp:<35s}", end='')
        for a in accs: print(f" {a:>14.4f}" if a is not None else f" {'N/A':>14s}", end='')
        print()
        # Depth sweep
        for min_d in DEPTH_SWEEP:
            keys = [f'{mt}_depth_sweep_{min_d}_{dp}' for mt in MASK_TYPES]
            accs = [all_final.get(k, {}).get('accuracy') for k in keys]
            if any(a is not None for a in accs):
                print(f"  {'depth>='+str(min_d)+'_'+dp:<35s}", end='')
                for a in accs: print(f" {a:>14.4f}" if a is not None else f" {'N/A':>14s}", end='')
                print()
        # Chain
        for cl in [2, 3, 4, 5]:
            keys = [f'{mt}_chain_{cl}_{dp}' for mt in MASK_TYPES]
            accs = [all_final.get(k, {}).get('accuracy') for k in keys]
            if any(a is not None for a in accs):
                print(f"  {'chain='+str(cl)+'_'+dp:<35s}", end='')
                for a in accs: print(f" {a:>14.4f}" if a is not None else f" {'N/A':>14s}", end='')
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
