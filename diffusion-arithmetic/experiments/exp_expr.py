"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Expression Evaluation — Tree Depth + PUMA Coverage Deficit
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Task: evaluate arithmetic expressions → fixed-width result
    Input:  "(3+5)*(7-2)=??????"  → "000040"  (zero-padded)

  Dependency analog:
    carry chain length  ←→  tree depth (deep nesting → large intermediates → carry)
    g/k position        ←→  leaf-only digit (computable from partial tree)
    p position          ←→  interior-node digit (requires full tree evaluation)
    LSB oracle order    ←→  postorder traversal (LSB first within each sub-expr)

  Training: random vs puma
  Decode:   confidence | postorder_oracle | random
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import sys, os, time, math, json, random as pyrandom
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                if '__file__' in dir() else '.')
from core.tokenizer import CharTokenizer
from core.model import Transformer
from core.train_utils import (
    mount_drive, save_results, generate_diffusion,
    encode_samples, DEVICE,
)

EXP_NAME = 'exp_expr'

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Config
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ANS_LEN = 6           # supports results up to 999999
MAX_DEPTH = 5         # max expression tree depth for training data
MAX_OPERAND = 25      # max single operand value
OPS = ['+', '-', '*'] # operators
EXPR_PAD_LEN = 64     # pad expression+= to this length, answer appended after

N_TRAIN = 20000; N_TEST = 1000; BATCH_SIZE = 200
MAX_EPOCHS = 5000; EVAL_EVERY = 100; LOG_EVERY = 50
GEN_EVAL_EVERY = 200; GEN_EVAL_N = 500

MASK_TYPES = ['random', 'puma']
DECODE_POLICIES = ['confidence', 'postorder', 'random']

N_LAYER = 3; N_HEAD = 3; N_EMBD = 192; DROPOUT = 0.1; POS_ENC = 'absolute'
LR = 1e-3; MIN_LR = 1e-4; WARMUP_EPOCHS = 10; GRAD_CLIP = 1.0
PUMA_TAU = 0.9; PUMA_K_START = 2; PUMA_K_END = ANS_LEN
SEED = 42
DEPTH_SWEEP = [1, 2, 3, 4, 5, 6, 7]


def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--ans-len', type=int); p.add_argument('--max-depth', type=int)
    p.add_argument('--max-operand', type=int); p.add_argument('--expr-pad-len', type=int)
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
    for a, gl in {'ans_len': 'ANS_LEN', 'max_depth': 'MAX_DEPTH', 'max_operand': 'MAX_OPERAND',
                   'expr_pad_len': 'EXPR_PAD_LEN', 'n_train': 'N_TRAIN', 'n_test': 'N_TEST',
                   'epochs': 'MAX_EPOCHS', 'batch_size': 'BATCH_SIZE', 'eval_every': 'EVAL_EVERY',
                   'gen_eval_every': 'GEN_EVAL_EVERY', 'n_layer': 'N_LAYER', 'n_head': 'N_HEAD',
                   'n_embd': 'N_EMBD', 'dropout': 'DROPOUT', 'puma_tau': 'PUMA_TAU',
                   'seed': 'SEED'}.items():
        v = getattr(args, a, None)
        if v is not None: g[gl] = v
    if args.masks: g['MASK_TYPES'] = args.masks
    if args.decode: g['DECODE_POLICIES'] = args.decode
    return args


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Expression Tree
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class ExprNode:
    __slots__ = ['op', 'left', 'right', 'value', 'depth']
    def __init__(self, value=None, op=None, left=None, right=None):
        self.op = op; self.left = left; self.right = right; self.value = value
        if op: self.depth = max(left.depth, right.depth) + 1
        else: self.depth = 0

    def evaluate(self):
        if self.op is None: return self.value
        l, r = self.left.evaluate(), self.right.evaluate()
        if l is None or r is None: return None
        if self.op == '+': return l + r
        if self.op == '-': return l - r
        if self.op == '*': return l * r
        return None

    def to_string(self):
        if self.op is None: return str(self.value)
        ls, rs = self.left.to_string(), self.right.to_string()
        return f"({ls}{self.op}{rs})"

    def n_ops(self):
        if self.op is None: return 0
        return 1 + self.left.n_ops() + self.right.n_ops()

    def is_left_chain(self):
        """True if tree is a pure left-chain (worst case for sequential dependency)."""
        if self.op is None: return True
        return self.right.op is None and self.left.is_left_chain()


def _random_tree(rng, max_depth, max_val, ops, target_depth=None):
    """Generate a random expression tree.
    If target_depth is set, ensure tree reaches exactly that depth.
    """
    if max_depth <= 0 or (target_depth is None and rng.random() < 0.3):
        return ExprNode(value=rng.randint(1, max_val))
    op = rng.choice(ops)
    if target_depth is not None and target_depth > 0:
        # Force one branch to reach target_depth-1
        side = rng.choice(['left', 'right'])
        if side == 'left':
            left = _random_tree(rng, max_depth - 1, max_val, ops, target_depth - 1)
            right = _random_tree(rng, max(0, max_depth - 1), max_val, ops, None)
        else:
            left = _random_tree(rng, max(0, max_depth - 1), max_val, ops, None)
            right = _random_tree(rng, max_depth - 1, max_val, ops, target_depth - 1)
    else:
        left = _random_tree(rng, max_depth - 1, max_val, ops, None)
        right = _random_tree(rng, max_depth - 1, max_val, ops, None)
    return ExprNode(op=op, left=left, right=right)


def _left_chain_tree(rng, depth, max_val, ops):
    """Generate a pure left-chain tree of given depth (extreme case)."""
    node = ExprNode(value=rng.randint(1, max_val))
    for _ in range(depth):
        op = rng.choice(ops)
        right = ExprNode(value=rng.randint(1, max_val))
        node = ExprNode(op=op, left=node, right=right)
    return node


def _tree_stats(tree):
    """Stats about tree structure."""
    return {
        'depth': tree.depth,
        'n_ops': tree.n_ops(),
        'is_left_chain': tree.is_left_chain(),
    }


def _count_answer_carries(result_val):
    """Count carry propagation in the multi-digit result (analog to carry chain).
    Returns the number of digit positions where a carry-in occurs during addition
    of the result digits — approximated by looking at digit transitions.
    """
    if result_val < 0:
        return _count_answer_carries(-result_val)
    s = str(result_val)
    # For expression eval, the "carry chain" is inherent in how the result
    # was computed. We approximate it by tree depth (already tracked).
    return len(s)  # number of significant digits as proxy


def _digit_dependency(tree):
    """Classify each answer digit's dependency level.
    MSB depends on the full tree magnitude → high dependency.
    LSB depends only on the last operation modulo 10 → lower dependency.
    Returns list of dep labels for each of ANS_LEN positions (MSB first).
    """
    depth = tree.depth; n_ops = tree.n_ops()
    deps = []
    for j in range(ANS_LEN):
        # MSB (j=0) depends on full tree; LSB (j=ANS_LEN-1) depends less
        rel_pos = j / max(ANS_LEN - 1, 1)  # 0=MSB, 1=LSB
        if depth <= 1:
            deps.append('shallow')
        elif rel_pos < 0.3:
            deps.append('deep_msb')  # top digits depend on full tree
        elif rel_pos > 0.7:
            deps.append('lsb')       # bottom digits partially independent
        else:
            deps.append('mid')
    return deps


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Data generation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _format_sample(tree, pad_len, ans_len):
    """Format: pad(expression=, pad_len) + zero_pad(result, ans_len)"""
    expr_s = tree.to_string() + '='
    result = tree.evaluate()
    if result is None or result < 0 or result >= 10**ans_len:
        return None, None, None
    expr_padded = expr_s.ljust(pad_len, '#')  # '#' = padding
    ans_s = str(result).zfill(ans_len)
    return expr_padded + ans_s, tree, result


def gen_data(n, seed, max_depth=None, natural=True):
    """Generate training data.
    natural=True: depth follows natural distribution (mostly shallow).
    natural=False: balanced across depths.
    """
    if max_depth is None: max_depth = MAX_DEPTH
    rng = pyrandom.Random(seed)
    results = []; seen = set(); attempts = 0
    while len(results) < n and attempts < n * 100:
        attempts += 1
        if natural:
            tree = _random_tree(rng, max_depth, MAX_OPERAND, OPS)
        else:
            # Balanced: pick target depth uniformly
            td = rng.randint(1, max_depth)
            tree = _random_tree(rng, max_depth, MAX_OPERAND, OPS, target_depth=td)
        s, t, v = _format_sample(tree, EXPR_PAD_LEN, ANS_LEN)
        if s is None or s in seen: continue
        seen.add(s)
        results.append({'string': s, 'tree': t, 'result': v,
                        'stats': _tree_stats(t), 'dep_labels': _digit_dependency(t)})
    if len(results) < n:
        print(f"  WARNING: gen_data got {len(results)}/{n}")
    return results


def gen_depth_test(n, seed, min_depth):
    """Constructive: generate expressions with tree depth >= min_depth."""
    rng = pyrandom.Random(seed); results = []; seen = set()
    for _ in range(n * 200):
        if len(results) >= n: break
        td = rng.randint(min_depth, min_depth + 2)
        tree = _random_tree(rng, td + 1, MAX_OPERAND, OPS, target_depth=td)
        s, t, v = _format_sample(tree, EXPR_PAD_LEN, ANS_LEN)
        if s is None or s in seen or t.depth < min_depth: continue
        seen.add(s); results.append({'string': s, 'tree': t, 'result': v,
                                     'stats': _tree_stats(t), 'dep_labels': _digit_dependency(t)})
    if len(results) < n: print(f"  WARNING: depth>={min_depth}: {len(results)}/{n}")
    return results


def gen_chain_test(n, seed, chain_depth):
    """Constructive: pure left-chain trees of given depth (extreme case).
    Analog to full_propagate in addition.
    """
    rng = pyrandom.Random(seed); results = []; seen = set()
    for _ in range(n * 100):
        if len(results) >= n: break
        tree = _left_chain_tree(rng, chain_depth, MAX_OPERAND, OPS)
        s, t, v = _format_sample(tree, EXPR_PAD_LEN, ANS_LEN)
        if s is None or s in seen: continue
        seen.add(s); results.append({'string': s, 'tree': t, 'result': v,
                                     'stats': _tree_stats(t), 'dep_labels': _digit_dependency(t)})
    if len(results) < n: print(f"  WARNING: chain={chain_depth}: {len(results)}/{n}")
    return results


def gen_mul_heavy_test(n, seed, min_depth=3):
    """Expressions with deep multiplication → large intermediate values → many carries."""
    rng = pyrandom.Random(seed); results = []; seen = set()
    for _ in range(n * 200):
        if len(results) >= n: break
        tree = _random_tree(rng, min_depth + 2, MAX_OPERAND, ['*', '+'], target_depth=min_depth)
        s, t, v = _format_sample(tree, EXPR_PAD_LEN, ANS_LEN)
        if s is None or s in seen or t.depth < min_depth: continue
        seen.add(s); results.append({'string': s, 'tree': t, 'result': v,
                                     'stats': _tree_stats(t), 'dep_labels': _digit_dependency(t)})
    if len(results) < n: print(f"  WARNING: mul_heavy: {len(results)}/{n}")
    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tokenizer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def build_tok():
    chars = list('0123456789+-*()=#')
    return CharTokenizer(chars, {'mask': 'M', 'pad': 'P'})


def get_answer(s):
    return s[EXPR_PAD_LEN:]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Probe & Analysis
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@torch.no_grad()
def probe_per_position(model, tokenizer, test_data, max_len, device=None):
    """Fully-masked probe: per-position loss/acc/conf."""
    if device is None: device = DEVICE
    model.eval(); mask_id = tokenizer.special_ids['mask']
    strings = [d['string'] for d in test_data]
    ids_all, ans_all = encode_samples(strings, tokenizer, max_len)
    ids_all, ans_all = ids_all.to(device), ans_all.to(device)

    L = torch.zeros(ANS_LEN, device=device)
    C = torch.zeros(ANS_LEN, device=device)
    CF = torch.zeros(ANS_LEN, device=device)
    N = torch.zeros(ANS_LEN, device=device)
    _arange = torch.arange(ANS_LEN, device=device)

    # Per-dep-label tracking
    dep_labels_all = [d['dep_labels'] for d in test_data]
    dep_names = ['shallow', 'lsb', 'mid', 'deep_msb']
    dep_to_id = {n: i for i, n in enumerate(dep_names)}
    dep_conf_sum = defaultdict(float); dep_acc_sum = defaultdict(float); dep_count = defaultdict(int)

    for st in range(0, len(test_data), 128):
        en = min(st + 128, len(test_data))
        ids, ans = ids_all[st:en], ans_all[st:en]; B = ids.shape[0]; T = ids.shape[1]
        ans_pos = (ans.unsqueeze(1) + _arange).clamp(max=T-1)
        bi = torch.arange(B, device=device).unsqueeze(1).expand_as(ans_pos)

        xm = ids.clone(); xm[bi, ans_pos] = mask_id
        logits = model(xm); al = logits[bi, ans_pos]; tgt = ids[bi, ans_pos]
        lp = F.log_softmax(al, dim=-1)
        losses = -lp.gather(2, tgt.unsqueeze(2)).squeeze(2)
        cl = al.clone(); cl[:, :, mask_id] = -float('inf')
        probs = F.softmax(cl, dim=-1)
        confs = probs.max(dim=-1).values; preds = probs.argmax(dim=-1)
        corrects = (preds == tgt).float()

        for j in range(ANS_LEN):
            L[j] += losses[:, j].sum(); C[j] += corrects[:, j].sum()
            CF[j] += confs[:, j].sum(); N[j] += B

        # dep label tracking
        for i in range(B):
            for j in range(ANS_LEN):
                dl = dep_labels_all[st + i][j]
                dep_conf_sum[dl] += confs[i, j].item()
                dep_acc_sum[dl] += corrects[i, j].item()
                dep_count[dl] += 1

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
    # Confidence concordance (higher conf for easier positions = good)
    cr = sorted(range(ANS_LEN), key=lambda j: pos_conf[j], reverse=True)
    conc = 0; n_p = ANS_LEN * (ANS_LEN - 1) // 2
    for i in range(ANS_LEN):
        for j in range(i + 1, ANS_LEN):
            conc += int(cr.index(j) < cr.index(i))
    result['conf_concordance'] = conc / max(n_p, 1)
    return result


@torch.no_grad()
def gen_eval_with_stats(model, tokenizer, test_data, max_len,
                        decode_policy='confidence', device=None):
    """Per-sample generation evaluation."""
    if device is None: device = DEVICE
    mask_id = tokenizer.special_ids['mask']; pad_id = tokenizer.special_ids['pad']
    model.eval(); out = []

    for st in range(0, len(test_data), 128):
        batch = test_data[st:min(st + 128, len(test_data))]; B = len(batch)
        # Encode prefix (expression=)
        prefixes = [d['string'][:EXPR_PAD_LEN] for d in batch]
        penc = [tokenizer.encode(p) for p in prefixes]
        pm = max(len(p) for p in penc)
        pids = torch.full((B, pm), pad_id, dtype=torch.long)
        for i, e in enumerate(penc): pids[i, :len(e)] = torch.tensor(e)

        # Decode policy
        if decode_policy == 'postorder':
            policy = 'r2l'  # MSB is position 0, LSB is last → r2l reveals LSB first
        elif decode_policy == 'random':
            policy = 'random'
        else:
            policy = 'confidence'

        gen, _, info = generate_diffusion(model, pids, ANS_LEN, mask_id,
                                          policy=policy, greedy=True, device=device)
        pred_ids = gen[:, pm:pm + ANS_LEN]

        for i in range(B):
            ps = tokenizer.decode(pred_ids[i].cpu().tolist())
            gs = get_answer(batch[i]['string'])
            pc = [ps[j] == gs[j] if j < len(ps) else False for j in range(len(gs))]
            errs = [j for j in range(len(gs)) if j >= len(ps) or ps[j] != gs[j]]
            out.append({
                'correct': ps == gs,
                'pos_correct': pc,
                'error_positions': errs,
                'stats': batch[i]['stats'],
                'dep_labels': batch[i]['dep_labels'],
            })
    return out


def stratify_results(per_sample):
    """Stratify by depth."""
    def _depth_bin(d):
        if d <= 1: return 'd=0-1'
        if d <= 2: return 'd=2'
        if d <= 3: return 'd=3'
        if d <= 4: return 'd=4'
        return 'd=5+'
    strata = {
        'depth': lambda st: _depth_bin(st['depth']),
        'chain': lambda st: 'chain' if st.get('is_left_chain') else 'tree',
    }
    out = {}
    for name, fn in strata.items():
        bk = defaultdict(list)
        for r in per_sample: bk[fn(r['stats'])].append(r['correct'])
        out[name] = {k: {'acc': sum(v)/len(v), 'n': len(v)} for k, v in sorted(bk.items())}
    return out


@torch.no_grad()
def simulate_puma_coverage(model, tokenizer, test_data, max_len,
                           K=None, tau=None, n_samples=200, device=None):
    """PUMA coverage simulation per dependency label."""
    if device is None: device = DEVICE
    if K is None: K = PUMA_K_END
    if tau is None: tau = PUMA_TAU
    model.eval(); mask_id = tokenizer.special_ids['mask']
    N = min(len(test_data), n_samples)
    _ar = torch.arange(ANS_LEN, device=device)

    dep_names = ['shallow', 'lsb', 'mid', 'deep_msb']
    cov_sum = defaultdict(float); cov_n = defaultdict(int)

    strings = [d['string'] for d in test_data[:N]]
    ids_all, ans_all = encode_samples(strings, tokenizer, max_len)
    ids_all, ans_all = ids_all.to(device), ans_all.to(device)

    for si in range(N):
        ids = ids_all[si:si+1]; ans_s = ans_all[si].item(); T = ids.shape[1]
        ap = torch.arange(ans_s, ans_s + ANS_LEN, device=device).clamp(max=T-1)
        x = ids.clone(); x0 = ids[0, ap].clone()
        x[0, ap] = mask_id
        is_m = torch.ones(ANS_LEN, dtype=torch.bool, device=device)
        steps_m = torch.zeros(ANS_LEN); total = 0

        for step in range(K):
            if not is_m.any(): break
            total += 1; steps_m += is_m.cpu().float()
            logits = model(x)
            confs = torch.full((ANS_LEN,), -float('inf'), device=device)
            for j in range(ANS_LEN):
                if is_m[j]:
                    cl = logits[0, ap[j]].clone(); cl[mask_id] = -float('inf')
                    confs[j] = F.softmax(cl, dim=-1).max()
            nm = is_m.sum().item()
            nr = max(1, int(math.ceil(nm / max(K - step, 1))))
            ranked = confs.argsort(descending=True)
            reveal = torch.zeros(ANS_LEN, dtype=torch.bool, device=device)
            cnt = 0
            for ri in range(ANS_LEN):
                j = ranked[ri].item()
                if not is_m[j]: continue
                if cnt < nr or confs[j] > tau:
                    reveal[j] = True; cnt += 1
            for j in range(ANS_LEN):
                if reveal[j]: x[0, ap[j]] = x0[j]; is_m[j] = False

        if total == 0: continue
        frac = steps_m / total
        dls = test_data[si]['dep_labels']
        for j in range(ANS_LEN):
            dl = dls[j]
            cov_sum[dl] += frac[j].item(); cov_n[dl] += 1

    per_dep = {}
    for dn in dep_names:
        if cov_n[dn] > 0:
            per_dep[dn] = {'mean_coverage': cov_sum[dn] / cov_n[dn], 'n': cov_n[dn]}
    return per_dep


def analyse_error_localization(per_sample):
    cats = defaultdict(int); total = 0
    for r in per_sample:
        if r['correct']: continue
        for j in r['error_positions']:
            if j < len(r['dep_labels']):
                cats[r['dep_labels'][j]] += 1; total += 1
    if total == 0: return {'total_errors': 0}
    result = {'total_errors': total}
    for k, v in cats.items(): result[k] = v / total
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Training (follows addition code pattern exactly)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def train_model(tokenizer, train_data, test_data, max_len,
                mask_type='random', device=None):
    if device is None: device = DEVICE
    strings_train = [d['string'] for d in train_data]
    train_ids, train_ans = encode_samples(strings_train, tokenizer, max_len)
    train_ids, train_ans = train_ids.to(device), train_ans.to(device)
    N, T = train_ids.shape; bpe = (N + BATCH_SIZE - 1) // BATCH_SIZE
    total_iters = MAX_EPOCHS * bpe
    mask_id = tokenizer.special_ids['mask']; pad_id = tokenizer.special_ids['pad']

    model = Transformer(vocab_size=len(tokenizer), block_size=max_len + 8,
                        n_layer=N_LAYER, n_head=N_HEAD, n_embd=N_EMBD,
                        dropout=DROPOUT, is_causal=False, pos_enc=POS_ENC).to(device)
    print(f"  [{mask_type}] params={model.n_params:,}, {bpe} batches/epoch, {MAX_EPOCHS} epochs")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.99), weight_decay=0.1)
    warmup_iters = WARMUP_EPOCHS * bpe

    def get_lr(it):
        if it < warmup_iters: return LR * it / max(warmup_iters, 1)
        ratio = (it - warmup_iters) / max(total_iters - warmup_iters, 1)
        return MIN_LR + 0.5 * (LR - MIN_LR) * (1 + math.cos(math.pi * min(ratio, 1.0)))

    dynamics = {'checkpoints': [], 'train_loss': []}
    best_loss, best_state = float('inf'), None; it = 0; tg = 0; t0 = time.time()
    _arange = torch.arange(ANS_LEN, device=device)

    # ── PUMA streaming buffer (identical to addition code) ──
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
            ap = (puma_ans[idx_t].unsqueeze(1) + _arange).clamp(max=T-1)
            bii = idx_t.unsqueeze(1).expand_as(ap)
            puma_z[bii, ap] = mask_id

        def _advance(logits, K_cur):
            nonlocal puma_stage
            B = BATCH_SIZE
            ap = (puma_ans.unsqueeze(1) + _arange).clamp(max=T-1)
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
        probe = probe_per_position(model, tokenizer, test_data, max_len, device)
        dynamics['checkpoints'].append({'epoch': epoch, 'iter': it, 'tg': tg, **probe})
        if probe['overall_loss'] < best_loss and epoch > 0:
            best_loss = probe['overall_loss']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        dc = probe.get('dep_context', {})
        parts = [f"{c}={dc[c]['acc']:.2f}" for c in ['shallow', 'lsb', 'mid', 'deep_msb'] if c in dc]
        print(f"    [eval ep {epoch}] loss={probe['overall_loss']:.4f} acc={probe['overall_acc']:.4f} "
              + ' '.join(parts) + f" | {time.time()-t0:.0f}s")

    model.eval(); _do_eval(0); model.train()

    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_loss = torch.tensor(0.0, device=device); epoch_n = 0
        K_cur = PUMA_K_START + int((PUMA_K_END - PUMA_K_START) * epoch / MAX_EPOCHS) if uses_streaming else 0

        if uses_streaming:
            for _ in range(bpe):
                for pg in optimizer.param_groups: pg['lr'] = get_lr(it)
                m = (puma_z == mask_id)
                if m.sum() == 0: _refresh(list(range(BATCH_SIZE))); m = (puma_z == mask_id)
                logits = model(puma_z); loss = F.cross_entropy(logits[m], puma_x0[m])
                tg += m.sum().item()
                optimizer.zero_grad(set_to_none=True); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP); optimizer.step()
                _advance(logits.detach(), K_cur)
                epoch_loss += loss.detach(); epoch_n += 1; it += 1
        else:
            perm = torch.randperm(N, device=device)
            for bi in range(bpe):
                for pg in optimizer.param_groups: pg['lr'] = get_lr(it)
                idx = perm[bi*BATCH_SIZE:min((bi+1)*BATCH_SIZE, N)]
                ids = train_ids[idx]; ans_s = train_ans[idx]; B_b = ids.shape[0]
                ap = (ans_s.unsqueeze(1) + _arange).clamp(max=T-1)
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

        if epoch % LOG_EVERY == 0:
            dynamics['train_loss'].append((epoch, epoch_loss.item() / max(epoch_n, 1)))
            print(f"    ep {epoch:4d}/{MAX_EPOCHS} | loss {epoch_loss.item()/max(epoch_n,1):.4f} | "
                  f"lr {get_lr(it):.1e} | tg {tg:,} | {time.time()-t0:.0f}s")
        do_eval = (epoch % EVAL_EVERY == 0) or \
                  (epoch < MAX_EPOCHS * 0.1 and epoch % max(EVAL_EVERY // 5, 1) == 0)
        if do_eval and epoch < MAX_EPOCHS:
            model.eval(); _do_eval(epoch); model.train()

    if best_state: model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval(); _do_eval(MAX_EPOCHS)
    return model, dynamics


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Figures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def make_figures(all_dyn, all_final):
    figs = {}; COLORS = {'random': '#3498db', 'puma': '#8e44ad'}

    # Fig 1: Depth sweep comparison
    fig, axes = plt.subplots(1, len(DECODE_POLICIES), figsize=(6 * len(DECODE_POLICIES), 5), squeeze=False)
    axes = axes[0]
    for di, dp in enumerate(DECODE_POLICIES):
        ax = axes[di]
        for mt, col in [('random', '#3498db'), ('puma', '#8e44ad')]:
            xs, ys = [], []
            for d in DEPTH_SWEEP:
                k = f'depth_sweep_{d}_{mt}_{dp}'
                if k in all_final:
                    xs.append(d); ys.append(all_final[k]['accuracy'])
            if xs: ax.plot(xs, ys, 'o-', color=col, label=mt, lw=2)
        ax.set_xlabel('Min Tree Depth'); ax.set_ylabel('Accuracy')
        ax.set_title(f'{dp} decode'); ax.legend(); ax.grid(alpha=0.3)
    fig.suptitle('Depth Sweep: Random vs PUMA', y=1.02); fig.tight_layout()
    figs['depth_sweep'] = fig

    # Fig 2: Training dynamics
    nc = len(MASK_TYPES)
    fig, axes = plt.subplots(1, nc, figsize=(6 * nc, 5), squeeze=False); axes = axes[0]
    for ai, mt in enumerate(MASK_TYPES):
        dyn = all_dyn.get(mt)
        if not dyn: continue
        ax = axes[ai]; cps = dyn['checkpoints']
        xs = [c['epoch'] for c in cps]
        for j in range(ANS_LEN):
            ax.plot(xs, [c['pos_acc'][j] for c in cps], '-', lw=0.8, label=f'p{j}')
        ax.set_xlabel('Epoch'); ax.set_ylabel('Acc'); ax.set_title(mt)
        ax.legend(fontsize=5, ncol=3); ax.grid(alpha=0.3)
    fig.suptitle('Per-Position Accuracy', y=1.02); fig.tight_layout()
    figs['pos_acc'] = fig

    # Fig 3: Standard/heavy/corner comparison bars
    test_types = ['standard', 'mul_heavy', 'chain_3', 'chain_5', 'chain_7']
    for dp in DECODE_POLICIES:
        fig, ax = plt.subplots(figsize=(12, 5))
        for mi, mt in enumerate(MASK_TYPES):
            accs, lbls = [], []
            for tt in test_types:
                k = f'{tt}_{mt}_{dp}'
                r = all_final.get(k)
                if r: accs.append(r['accuracy']); lbls.append(tt)
            if accs:
                x = range(len(lbls)); w = 0.35; off = -w/2 if mi == 0 else w/2
                ax.bar([i + off for i in x], accs, w, label=mt,
                       color=COLORS.get(mt, 'gray'), alpha=0.8)
        if lbls: ax.set_xticks(range(len(lbls))); ax.set_xticklabels(lbls, fontsize=8)
        ax.set_ylabel('Accuracy'); ax.set_title(f'{dp}'); ax.legend(); ax.grid(alpha=0.3, axis='y')
        fig.tight_layout(); figs[f'test_types_{dp}'] = fig

    return figs


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Run
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def run(tag=''):
    exp_name = f"{EXP_NAME}{'_'+tag if tag else ''}"
    torch.manual_seed(SEED); pyrandom.seed(SEED)
    tok = build_tok()
    max_len = EXPR_PAD_LEN + ANS_LEN + 2
    print(f"\n{'='*70}\n  {exp_name} | ANS_LEN={ANS_LEN} MAX_DEPTH={MAX_DEPTH}\n{'='*70}")

    # Data
    train_data = gen_data(N_TRAIN, SEED, natural=True)
    test_data = gen_data(N_TEST, SEED + 1000, natural=True)
    print(f"  Train: {len(train_data)}, Test: {len(test_data)}")
    depth_dist = defaultdict(int)
    for d in train_data: depth_dist[d['stats']['depth']] += 1
    print(f"  Train depth dist: {dict(sorted(depth_dist.items()))}")

    all_dyn, all_final = {}, {}

    for mt in MASK_TYPES:
        print(f"\n{'━'*60}\n  Training: {mt}\n{'━'*60}")
        m, dyn = train_model(tok, train_data, test_data, max_len, mask_type=mt, device=DEVICE)
        all_dyn[mt] = dyn

        # Evaluate with all decode policies
        for dp in DECODE_POLICIES:
            # Standard test
            ps = gen_eval_with_stats(m, tok, test_data, max_len, decode_policy=dp, device=DEVICE)
            acc = sum(r['correct'] for r in ps) / len(ps)
            strat = stratify_results(ps)
            all_final[f'standard_{mt}_{dp}'] = {'accuracy': acc, 'n': len(ps), 'stratified': strat}
            print(f"    standard {dp}: {acc:.4f}")
            for sk, sv in strat.items():
                for bk, bv in sv.items():
                    print(f"      {sk}/{bk}: {bv['acc']:.4f} (n={bv['n']})")

            # Mul-heavy test
            mh = gen_mul_heavy_test(N_TEST, SEED + 2000)
            if mh:
                ps = gen_eval_with_stats(m, tok, mh, max_len, decode_policy=dp, device=DEVICE)
                acc = sum(r['correct'] for r in ps) / len(ps)
                all_final[f'mul_heavy_{mt}_{dp}'] = {'accuracy': acc, 'n': len(ps)}
                print(f"    mul_heavy {dp}: {acc:.4f}")

            # Chain tests (extreme left-chain)
            for cd in [3, 5, 7]:
                cc = gen_chain_test(500, SEED + 3000 + cd, cd)
                if cc:
                    ps = gen_eval_with_stats(m, tok, cc, max_len, decode_policy=dp, device=DEVICE)
                    acc = sum(r['correct'] for r in ps) / len(ps)
                    all_final[f'chain_{cd}_{mt}_{dp}'] = {'accuracy': acc, 'n': len(ps)}
                    print(f"    chain={cd} {dp}: {acc:.4f}")

            # Depth sweep
            for d in DEPTH_SWEEP:
                dt = gen_depth_test(500, SEED + 4000 + d, d)
                if dt:
                    ps = gen_eval_with_stats(m, tok, dt, max_len, decode_policy=dp, device=DEVICE)
                    acc = sum(r['correct'] for r in ps) / len(ps)
                    all_final[f'depth_sweep_{d}_{mt}_{dp}'] = {'accuracy': acc, 'n': len(ps)}
                    print(f"    depth>={d} {dp}: {acc:.4f}")

        # PUMA coverage
        if mt == 'puma':
            cov = simulate_puma_coverage(m, tok, test_data, max_len, device=DEVICE)
            all_final[f'coverage_{mt}'] = cov
            for dn, dv in cov.items():
                print(f"    coverage/{dn}: {dv['mean_coverage']:.3f} (n={dv['n']})")

        # Error localization
        ps_conf = gen_eval_with_stats(m, tok, test_data, max_len, decode_policy='confidence', device=DEVICE)
        el = analyse_error_localization(ps_conf)
        all_final[f'error_loc_{mt}'] = el
        if el['total_errors'] > 0:
            parts = [f"{k}={v:.2f}" for k, v in el.items() if k != 'total_errors' and isinstance(v, float)]
            print(f"    errors: {el['total_errors']} — {' '.join(parts)}")

        del m; torch.cuda.empty_cache() if torch.cuda.is_available() else None

    figs = make_figures(all_dyn, all_final)

    # Save
    sd = {'config': {k: globals()[k] for k in [
        'ANS_LEN', 'MAX_DEPTH', 'MAX_OPERAND', 'EXPR_PAD_LEN',
        'N_TRAIN', 'N_TEST', 'MAX_EPOCHS', 'BATCH_SIZE',
        'N_LAYER', 'N_HEAD', 'N_EMBD', 'MASK_TYPES', 'DECODE_POLICIES']}}
    for k, v in all_dyn.items():
        sd[f'dyn_{k}'] = {'checkpoints': v['checkpoints'], 'train_loss': v['train_loss']}
    for k, v in all_final.items():
        sd[f'final_{k}'] = v
    save_results(exp_name, sd, figures=figs)

    # Summary
    print(f"\n{'='*70}\n  SUMMARY\n{'='*70}")
    for dp in DECODE_POLICIES:
        print(f"\n  ── {dp} ──")
        print(f"  {'Test':<30s}", end='')
        for mt in MASK_TYPES: print(f" {mt:>14s}", end='')
        print()
        for tt in ['standard', 'mul_heavy'] + [f'chain_{d}' for d in [3, 5, 7]]:
            accs = [all_final.get(f'{tt}_{mt}_{dp}', {}).get('accuracy') for mt in MASK_TYPES]
            if any(a is not None for a in accs):
                print(f"  {tt:<30s}", end='')
                for a in accs: print(f" {a:>14.4f}" if a is not None else f" {'N/A':>14s}", end='')
                print()
        for d in DEPTH_SWEEP:
            accs = [all_final.get(f'depth_sweep_{d}_{mt}_{dp}', {}).get('accuracy') for mt in MASK_TYPES]
            if any(a is not None for a in accs):
                print(f"  {'depth>='+str(d):<30s}", end='')
                for a in accs: print(f" {a:>14.4f}" if a is not None else f" {'N/A':>14s}", end='')
                print()

    return all_dyn, all_final


if __name__ == '__main__':
    args = parse_args()
    seeds = args.seeds if args.seeds else [SEED]
    for si, seed in enumerate(seeds):
        globals()['SEED'] = seed
        t = f"{args.tag}_s{seed}" if args.tag and len(seeds) > 1 else args.tag
        if len(seeds) > 1: print(f"\n{'#'*70}\n# Seed {seed} ({si+1}/{len(seeds)})\n{'#'*70}")
        run(tag=t)
