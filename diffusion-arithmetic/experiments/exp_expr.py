"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Expression Evaluation — Tree Depth + PUMA Coverage Deficit
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Task: evaluate arithmetic expressions → fixed-width result
    Input:  "(3+5)*(7-2)=??????"  → "000040"  (zero-padded)

  Dependency structure:
    ALL output digits depend equally on the full tree evaluation.
    The difficulty axis is tree depth (across samples), NOT digit
    position (within a sample).  This differs from addition where
    each digit position has a distinct carry-chain dependency.

  Dependency analog:
    carry chain length  ←→  tree depth
    full_propagate      ←→  pure left-chain tree
    chain sweep         ←→  depth sweep

  Training: random vs puma
  Decode:   confidence | lsb | random
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
ANS_LEN = 6; MAX_DEPTH = 5; MAX_OPERAND = 25
OPS = ['+', '-', '*']; NONNEG_OPS = ['+', '*']
EXPR_PAD_LEN = 64

N_TRAIN = 20000; N_TEST = 1000; BATCH_SIZE = 200
MAX_EPOCHS = 5000; EVAL_EVERY = 100; LOG_EVERY = 50
GEN_EVAL_EVERY = 200; GEN_EVAL_N = 500

MASK_TYPES = ['random', 'puma']
DECODE_POLICIES = ['confidence', 'lsb', 'random']

N_LAYER = 3; N_HEAD = 3; N_EMBD = 192; DROPOUT = 0.1; POS_ENC = 'absolute'
LR = 1e-3; MIN_LR = 1e-4; WARMUP_EPOCHS = 10; GRAD_CLIP = 1.0
PUMA_TAU = 0.9; PUMA_K_START = 2; PUMA_K_END = ANS_LEN
SEED = 42; DEPTH_SWEEP = [1, 2, 3, 4, 5, 6, 7]


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
    try: args, _ = p.parse_known_args()
    except SystemExit: args, _ = p.parse_known_args([])
    g = globals()
    for a, gl in {'ans_len':'ANS_LEN','max_depth':'MAX_DEPTH','max_operand':'MAX_OPERAND',
                   'expr_pad_len':'EXPR_PAD_LEN','n_train':'N_TRAIN','n_test':'N_TEST',
                   'epochs':'MAX_EPOCHS','batch_size':'BATCH_SIZE','eval_every':'EVAL_EVERY',
                   'gen_eval_every':'GEN_EVAL_EVERY','n_layer':'N_LAYER','n_head':'N_HEAD',
                   'n_embd':'N_EMBD','dropout':'DROPOUT','puma_tau':'PUMA_TAU','seed':'SEED'}.items():
        v = getattr(args, a, None)
        if v is not None: g[gl] = v
    if args.masks: g['MASK_TYPES'] = args.masks
    if args.decode: g['DECODE_POLICIES'] = args.decode
    if getattr(args, 'ans_len', None) is not None: g['PUMA_K_END'] = g['ANS_LEN']
    return args


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Expression Tree
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class ExprNode:
    __slots__ = ['op', 'left', 'right', 'value', 'depth']
    def __init__(self, value=None, op=None, left=None, right=None):
        self.op = op; self.left = left; self.right = right; self.value = value
        self.depth = (max(left.depth, right.depth) + 1) if op else 0
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
        return f"({self.left.to_string()}{self.op}{self.right.to_string()})"
    def n_ops(self):
        if self.op is None: return 0
        return 1 + self.left.n_ops() + self.right.n_ops()
    def is_left_chain(self):
        if self.op is None: return True
        return self.right.op is None and self.left.is_left_chain()

def _random_tree(rng, max_depth, max_val, ops, target_depth=None):
    if max_depth <= 0 or (target_depth is None and rng.random() < 0.3):
        return ExprNode(value=rng.randint(1, max_val))
    op = rng.choice(ops)
    if target_depth is not None and target_depth > 0:
        side = rng.choice(['left', 'right'])
        forced = _random_tree(rng, max_depth-1, max_val, ops, target_depth-1)
        free = _random_tree(rng, max(0, max_depth-1), max_val, ops, None)
        left, right = (forced, free) if side == 'left' else (free, forced)
    else:
        left = _random_tree(rng, max_depth-1, max_val, ops, None)
        right = _random_tree(rng, max_depth-1, max_val, ops, None)
    return ExprNode(op=op, left=left, right=right)

def _left_chain_tree(rng, depth, max_val, ops):
    node = ExprNode(value=rng.randint(1, max_val))
    for _ in range(depth):
        node = ExprNode(op=rng.choice(ops), left=node,
                        right=ExprNode(value=rng.randint(1, max_val)))
    return node

def _tree_stats(tree):
    return {'depth': tree.depth, 'n_ops': tree.n_ops(), 'is_left_chain': tree.is_left_chain()}

def _depth_category(depth):
    """Uniform per-sample label. All digits in same expression share this
    because all depend equally on the full tree evaluation."""
    if depth <= 1: return 'depth_0_1'
    if depth <= 3: return 'depth_2_3'
    return 'depth_4_plus'


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Data generation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _format_sample(tree, pad_len, ans_len):
    expr_s = tree.to_string() + '='
    result = tree.evaluate()
    if result is None or result < 0 or result >= 10**ans_len: return None, None
    if len(expr_s) > pad_len: return None, None
    return expr_s.ljust(pad_len, '#') + str(result).zfill(ans_len), result

def gen_data(n, seed, max_depth=None, natural=True):
    if max_depth is None: max_depth = MAX_DEPTH
    rng = pyrandom.Random(seed); results = []; seen = set()
    for _ in range(n * 100):
        if len(results) >= n: break
        if natural: tree = _random_tree(rng, max_depth, MAX_OPERAND, OPS)
        else: tree = _random_tree(rng, max_depth, MAX_OPERAND, OPS, target_depth=rng.randint(1, max_depth))
        s, v = _format_sample(tree, EXPR_PAD_LEN, ANS_LEN)
        if s is None or s in seen: continue
        seen.add(s); stats = _tree_stats(tree)
        results.append({'string': s, 'result': v, 'stats': stats,
                        'dep_cat': _depth_category(stats['depth'])})
    if len(results) < n: print(f"  WARNING: gen_data got {len(results)}/{n}")
    return results

def _deep_max_val(depth):
    """Reduce operand range for deep trees to avoid overflow rejection."""
    return min(MAX_OPERAND, max(3, int((10**ANS_LEN) ** (1.0 / max(depth, 1)))))

def gen_depth_test(n, seed, min_depth):
    rng = pyrandom.Random(seed); results = []; seen = set()
    mv = _deep_max_val(min_depth)
    for _ in range(n * 500):
        if len(results) >= n: break
        td = rng.randint(min_depth, min_depth + 2)
        tree = _random_tree(rng, td+1, mv, NONNEG_OPS, target_depth=td)
        s, v = _format_sample(tree, EXPR_PAD_LEN, ANS_LEN)
        if s is None or s in seen or tree.depth < min_depth: continue
        seen.add(s); stats = _tree_stats(tree)
        results.append({'string': s, 'result': v, 'stats': stats,
                        'dep_cat': _depth_category(stats['depth'])})
    if len(results) < n: print(f"  WARNING: depth>={min_depth}: {len(results)}/{n}")
    return results

def gen_chain_test(n, seed, chain_depth):
    rng = pyrandom.Random(seed); results = []; seen = set()
    mv = _deep_max_val(chain_depth)
    for _ in range(n * 200):
        if len(results) >= n: break
        tree = _left_chain_tree(rng, chain_depth, mv, NONNEG_OPS)
        s, v = _format_sample(tree, EXPR_PAD_LEN, ANS_LEN)
        if s is None or s in seen: continue
        seen.add(s); stats = _tree_stats(tree)
        results.append({'string': s, 'result': v, 'stats': stats,
                        'dep_cat': _depth_category(stats['depth'])})
    if len(results) < n: print(f"  WARNING: chain={chain_depth}: {len(results)}/{n}")
    return results

def gen_mul_heavy_test(n, seed, min_depth=3):
    rng = pyrandom.Random(seed); results = []; seen = set()
    mv = _deep_max_val(min_depth + 2)
    for _ in range(n * 500):
        if len(results) >= n: break
        tree = _random_tree(rng, min_depth+2, mv, ['*', '+'], target_depth=min_depth)
        s, v = _format_sample(tree, EXPR_PAD_LEN, ANS_LEN)
        if s is None or s in seen or tree.depth < min_depth: continue
        seen.add(s); stats = _tree_stats(tree)
        results.append({'string': s, 'result': v, 'stats': stats,
                        'dep_cat': _depth_category(stats['depth'])})
    if len(results) < n: print(f"  WARNING: mul_heavy: {len(results)}/{n}")
    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tokenizer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def build_tok():
    return CharTokenizer(list('0123456789+-*()=#'), {'mask': 'M', 'pad': 'P'})
def get_answer(s): return s[EXPR_PAD_LEN:]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Probe & Analysis (vectorized dep-cat tracking)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@torch.no_grad()
def probe_per_position(model, tokenizer, test_data, max_len, device=None):
    if device is None: device = DEVICE
    model.eval(); mask_id = tokenizer.special_ids['mask']
    strings = [d['string'] for d in test_data]
    ids_all, ans_all = encode_samples(strings, tokenizer, max_len)
    ids_all, ans_all = ids_all.to(device), ans_all.to(device)
    L = torch.zeros(ANS_LEN, device=device); C = torch.zeros(ANS_LEN, device=device)
    CF = torch.zeros(ANS_LEN, device=device); N_cnt = torch.zeros(ANS_LEN, device=device)
    _arange = torch.arange(ANS_LEN, device=device)
    dep_cat_names = ['depth_0_1', 'depth_2_3', 'depth_4_plus']
    dci = {n: i for i, n in enumerate(dep_cat_names)}; nc = len(dep_cat_names)
    sample_cats = torch.tensor([dci[d['dep_cat']] for d in test_data], dtype=torch.long, device=device)
    cat_conf = torch.zeros(nc, device=device); cat_acc = torch.zeros(nc, device=device)
    cat_n = torch.zeros(nc, dtype=torch.long, device=device)

    for st in range(0, len(test_data), 128):
        en = min(st+128, len(test_data)); B = en - st
        ids, ans = ids_all[st:en], ans_all[st:en]; T = ids.shape[1]
        ap = (ans.unsqueeze(1) + _arange).clamp(max=T-1)
        bi = torch.arange(B, device=device).unsqueeze(1).expand_as(ap)
        xm = ids.clone(); xm[bi, ap] = mask_id
        logits = model(xm); al = logits[bi, ap]; tgt = ids[bi, ap]
        lp = F.log_softmax(al, dim=-1)
        losses = -lp.gather(2, tgt.unsqueeze(2)).squeeze(2)
        cl = al.clone(); cl[:, :, mask_id] = -float('inf')
        probs = F.softmax(cl, dim=-1)
        confs = probs.max(dim=-1).values; corrects = (probs.argmax(dim=-1) == tgt).float()
        L += losses.sum(0); C += corrects.sum(0); CF += confs.sum(0); N_cnt += B
        bc = sample_cats[st:en]
        for ci in range(nc):
            m = (bc == ci)
            if m.any(): cat_conf[ci] += confs[m].mean(1).sum(); cat_acc[ci] += corrects[m].mean(1).sum(); cat_n[ci] += m.sum()

    s = N_cnt.clamp(1)
    result = {'pos_loss': (L/s).cpu().tolist(), 'pos_acc': (C/s).cpu().tolist(),
              'pos_conf': (CF/s).cpu().tolist(),
              'overall_loss': (L.sum()/s.sum()).item(), 'overall_acc': (C.sum()/s.sum()).item()}
    dc = {}
    for ci, cn in enumerate(dep_cat_names):
        n = cat_n[ci].item()
        if n > 0: dc[cn] = {'conf': cat_conf[ci].item()/n, 'acc': cat_acc[ci].item()/n, 'n': n}
    if dc: result['dep_context'] = dc
    return result


@torch.no_grad()
def gen_eval_with_stats(model, tokenizer, test_data, max_len,
                        decode_policy='confidence', device=None):
    if device is None: device = DEVICE
    mask_id = tokenizer.special_ids['mask']; pad_id = tokenizer.special_ids['pad']
    model.eval(); out = []
    for st in range(0, len(test_data), 128):
        batch = test_data[st:min(st+128, len(test_data))]; B = len(batch)
        prefixes = [d['string'][:EXPR_PAD_LEN] for d in batch]
        penc = [tokenizer.encode(p) for p in prefixes]; pm = max(len(p) for p in penc)
        pids = torch.full((B, pm), pad_id, dtype=torch.long)
        for i, e in enumerate(penc): pids[i, :len(e)] = torch.tensor(e)
        policy = {'lsb': 'r2l', 'random': 'random'}.get(decode_policy, 'confidence')
        gen, _, _ = generate_diffusion(model, pids, ANS_LEN, mask_id,
                                       policy=policy, greedy=True, device=device)
        pred_ids = gen[:, pm:pm+ANS_LEN]
        for i in range(B):
            ps = tokenizer.decode(pred_ids[i].cpu().tolist()); gs = get_answer(batch[i]['string'])
            pc = [ps[j]==gs[j] if j<len(ps) else False for j in range(len(gs))]
            errs = [j for j in range(len(gs)) if j>=len(ps) or ps[j]!=gs[j]]
            out.append({'correct': ps==gs, 'pos_correct': pc, 'error_positions': errs,
                        'stats': batch[i]['stats'], 'dep_cat': batch[i]['dep_cat']})
    return out


def stratify_results(per_sample):
    def _db(d):
        if d<=1: return 'd=0-1'
        if d<=2: return 'd=2'
        if d<=3: return 'd=3'
        if d<=4: return 'd=4'
        return 'd=5+'
    strata = {'depth': lambda st: _db(st['depth']),
              'chain': lambda st: 'chain' if st.get('is_left_chain') else 'tree'}
    out = {}
    for name, fn in strata.items():
        bk = defaultdict(list)
        for r in per_sample: bk[fn(r['stats'])].append(r['correct'])
        out[name] = {k: {'acc': sum(v)/len(v), 'n': len(v)} for k, v in sorted(bk.items())}
    return out


@torch.no_grad()
def simulate_puma_coverage(model, tokenizer, test_data, max_len,
                           K=None, tau=None, n_samples=200, batch_size=32, device=None):
    """PUMA coverage — BATCHED by dep_cat."""
    if device is None: device = DEVICE
    if K is None: K = PUMA_K_END
    if tau is None: tau = PUMA_TAU
    model.eval(); mask_id = tokenizer.special_ids['mask']
    N = min(len(test_data), n_samples); _ar = torch.arange(ANS_LEN, device=device)
    cov_sum = defaultdict(float); cov_n = defaultdict(int)
    strings = [d['string'] for d in test_data[:N]]
    ids_all, ans_all = encode_samples(strings, tokenizer, max_len)
    ids_all, ans_all = ids_all.to(device), ans_all.to(device)

    for st in range(0, N, batch_size):
        en = min(st+batch_size, N); B = en-st
        ids = ids_all[st:en]; ans_s = ans_all[st:en]; T = ids.shape[1]
        ap = (ans_s.unsqueeze(1) + _ar).clamp(max=T-1)
        bi = torch.arange(B, device=device).unsqueeze(1).expand_as(ap)
        x = ids.clone(); x0 = ids[bi, ap].clone(); x[bi, ap] = mask_id
        is_m = torch.ones(B, ANS_LEN, dtype=torch.bool, device=device)
        steps_m = torch.zeros(B, ANS_LEN); total = 0
        for step in range(K):
            if not is_m.any(): break
            total += 1; steps_m += is_m.cpu().float()
            logits = model(x); al = logits[bi, ap].clone(); al[:,:,mask_id] = -float('inf')
            confs = F.softmax(al, dim=-1).max(dim=-1).values; confs[~is_m] = -float('inf')
            nm = is_m.sum(dim=1).float(); nr = (nm/max(K-step,1)).ceil().long().clamp(min=1)
            ranked = confs.argsort(dim=1, descending=True)
            rop = torch.zeros_like(ranked); rop.scatter_(1, ranked, _ar.expand(B,-1))
            reveal = ((rop < nr.unsqueeze(1)) | (confs > tau)) & is_m
            x[bi[reveal], ap[reveal]] = x0[reveal]; is_m = is_m & ~reveal
        if total == 0: continue
        frac = steps_m / total
        for i in range(B):
            dc = test_data[st+i]['dep_cat']
            cov_sum[dc] += frac[i].mean().item(); cov_n[dc] += 1
    return {dn: {'mean_coverage': cov_sum[dn]/cov_n[dn], 'n': cov_n[dn]}
            for dn in ['depth_0_1','depth_2_3','depth_4_plus'] if cov_n[dn] > 0}


def analyse_error_localization(per_sample):
    cats = defaultdict(int); total = 0
    for r in per_sample:
        if r['correct']: continue
        ne = len(r['error_positions'])
        if ne > 0: cats[r['dep_cat']] += ne; total += ne
    if total == 0: return {'total_errors': 0}
    return {'total_errors': total, **{k: v/total for k, v in cats.items()}}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Training
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def train_model(tokenizer, train_data, test_data, max_len,
                mask_type='random', device=None):
    if device is None: device = DEVICE
    train_ids, train_ans = encode_samples([d['string'] for d in train_data], tokenizer, max_len)
    train_ids, train_ans = train_ids.to(device), train_ans.to(device)
    N, T = train_ids.shape; bpe = (N+BATCH_SIZE-1)//BATCH_SIZE; total_iters = MAX_EPOCHS*bpe
    mask_id = tokenizer.special_ids['mask']
    model = Transformer(vocab_size=len(tokenizer), block_size=max_len+8, n_layer=N_LAYER,
                        n_head=N_HEAD, n_embd=N_EMBD, dropout=DROPOUT,
                        is_causal=False, pos_enc=POS_ENC).to(device)
    print(f"  [{mask_type}] params={model.n_params:,}, {bpe} batches/epoch, {MAX_EPOCHS} epochs")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9,0.99), weight_decay=0.1)
    warmup_iters = WARMUP_EPOCHS * bpe
    def get_lr(it):
        if it < warmup_iters: return LR * it / max(warmup_iters, 1)
        ratio = (it-warmup_iters)/max(total_iters-warmup_iters, 1)
        return MIN_LR + 0.5*(LR-MIN_LR)*(1+math.cos(math.pi*min(ratio,1.0)))
    dynamics = {'checkpoints': [], 'train_loss': []}
    best_loss, best_state = float('inf'), None; it = 0; tg = 0; t0 = time.time()
    _arange = torch.arange(ANS_LEN, device=device)

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
            if pool_ptr+n > len(pool): pool = torch.randperm(N); pool_ptr = 0
            si = pool[pool_ptr:pool_ptr+n].to(device); pool_ptr += n
            puma_x0[idx_t] = train_ids[si]; puma_z[idx_t] = train_ids[si].clone()
            puma_ans[idx_t] = train_ans[si]; puma_stage[idx_t] = 0
            ap = (puma_ans[idx_t].unsqueeze(1)+_arange).clamp(max=T-1)
            puma_z[idx_t.unsqueeze(1).expand_as(ap), ap] = mask_id
        def _advance(logits, K_cur):
            nonlocal puma_stage
            B = BATCH_SIZE; ap = (puma_ans.unsqueeze(1)+_arange).clamp(max=T-1)
            bi = torch.arange(B, device=device).unsqueeze(1).expand_as(ap)
            is_m = (puma_z[bi, ap] == mask_id)
            if not is_m.any(): _refresh(list(range(B))); return
            nm = is_m.sum(dim=1).float(); K_rem = (K_cur-puma_stage).clamp(min=1)
            nr = (nm/K_rem.float()).ceil().long().clamp(min=1)
            lp = logits[bi, ap].clone(); lp[:,:,mask_id] = -float('inf')
            confs = F.softmax(lp, dim=-1).max(dim=-1).values; confs[~is_m] = -float('inf')
            ranked = confs.argsort(dim=1, descending=True)
            rop = torch.zeros_like(ranked); rop.scatter_(1, ranked, _arange.expand(B,-1))
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
        parts = [f"{c}={dc[c]['acc']:.2f}" for c in ['depth_0_1','depth_2_3','depth_4_plus'] if c in dc]
        print(f"    [eval ep {epoch}] loss={probe['overall_loss']:.4f} acc={probe['overall_acc']:.4f} "
              + ' '.join(parts) + f" | {time.time()-t0:.0f}s")

    model.eval(); _do_eval(0); model.train()
    for epoch in range(1, MAX_EPOCHS+1):
        epoch_loss = torch.tensor(0.0, device=device); epoch_n = 0
        K_cur = PUMA_K_START + int((PUMA_K_END-PUMA_K_START)*epoch/MAX_EPOCHS) if uses_streaming else 0
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
            for bi_idx in range(bpe):
                for pg in optimizer.param_groups: pg['lr'] = get_lr(it)
                idx = perm[bi_idx*BATCH_SIZE:min((bi_idx+1)*BATCH_SIZE, N)]
                ids = train_ids[idx]; ans_s = train_ans[idx]; B_b = ids.shape[0]
                ap = (ans_s.unsqueeze(1)+_arange).clamp(max=T-1)
                ans_mask = torch.zeros(B_b, T, dtype=torch.bool, device=device)
                ans_mask.scatter_(1, ap, True)
                t_r = torch.rand(B_b, device=device)
                m = torch.bernoulli(t_r.unsqueeze(1) * ans_mask.float()).bool()
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
            dynamics['train_loss'].append((epoch, epoch_loss.item()/max(epoch_n,1)))
            print(f"    ep {epoch:4d}/{MAX_EPOCHS} | loss {epoch_loss.item()/max(epoch_n,1):.4f} | "
                  f"lr {get_lr(it):.1e} | tg {tg:,} | {time.time()-t0:.0f}s")
        do_eval = (epoch % EVAL_EVERY == 0) or (epoch < MAX_EPOCHS*0.1 and epoch % max(EVAL_EVERY//5,1) == 0)
        if do_eval and epoch < MAX_EPOCHS: model.eval(); _do_eval(epoch); model.train()
    if best_state: model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval(); _do_eval(MAX_EPOCHS)
    return model, dynamics


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Figures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def make_figures(all_dyn, all_final):
    figs = {}; COLORS = {'random': '#3498db', 'puma': '#8e44ad'}
    fig, axes = plt.subplots(1, len(DECODE_POLICIES), figsize=(6*len(DECODE_POLICIES),5), squeeze=False)
    for di, dp in enumerate(DECODE_POLICIES):
        ax = axes[0][di]
        for mt, col in COLORS.items():
            xs, ys = [], []
            for d in DEPTH_SWEEP:
                k = f'depth_sweep_{d}_{mt}_{dp}'
                if k in all_final: xs.append(d); ys.append(all_final[k]['accuracy'])
            if xs: ax.plot(xs, ys, 'o-', color=col, label=mt, lw=2)
        ax.set_xlabel('Min Tree Depth'); ax.set_ylabel('Accuracy')
        ax.set_title(f'{dp} decode'); ax.legend(); ax.grid(alpha=0.3)
    fig.suptitle('Depth Sweep', y=1.02); fig.tight_layout(); figs['depth_sweep'] = fig
    test_types = ['standard', 'mul_heavy', 'chain_3', 'chain_5', 'chain_7']
    for dp in DECODE_POLICIES:
        fig, ax = plt.subplots(figsize=(12,5))
        for mi, mt in enumerate(MASK_TYPES):
            accs, lbls = [], []
            for tt in test_types:
                r = all_final.get(f'{tt}_{mt}_{dp}')
                if r: accs.append(r['accuracy']); lbls.append(tt)
            if accs:
                w = 0.35; off = -w/2 if mi==0 else w/2
                ax.bar([i+off for i in range(len(lbls))], accs, w, label=mt, color=COLORS.get(mt), alpha=0.8)
        if lbls: ax.set_xticks(range(len(lbls))); ax.set_xticklabels(lbls, fontsize=8)
        ax.set_ylabel('Accuracy'); ax.set_title(dp); ax.legend(); ax.grid(alpha=0.3, axis='y')
        fig.tight_layout(); figs[f'test_types_{dp}'] = fig
    return figs


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Run
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def run(tag=''):
    exp_name = f"{EXP_NAME}{'_'+tag if tag else ''}"
    torch.manual_seed(SEED); pyrandom.seed(SEED)
    tok = build_tok(); max_len = EXPR_PAD_LEN + ANS_LEN + 2
    print(f"\n{'='*70}\n  {exp_name} | ANS_LEN={ANS_LEN} MAX_DEPTH={MAX_DEPTH}\n{'='*70}")
    train_data = gen_data(N_TRAIN, SEED, natural=True)
    test_data = gen_data(N_TEST, SEED+1000, natural=True)
    print(f"  Train: {len(train_data)}, Test: {len(test_data)}")
    dd = defaultdict(int)
    for d in train_data: dd[d['stats']['depth']] += 1
    print(f"  Depth dist: {dict(sorted(dd.items()))}")
    all_dyn, all_final = {}, {}
    for mt in MASK_TYPES:
        print(f"\n{'━'*60}\n  Training: {mt}\n{'━'*60}")
        m, dyn = train_model(tok, train_data, test_data, max_len, mask_type=mt, device=DEVICE)
        all_dyn[mt] = dyn
        for dp in DECODE_POLICIES:
            ps = gen_eval_with_stats(m, tok, test_data, max_len, decode_policy=dp, device=DEVICE)
            acc = sum(r['correct'] for r in ps)/len(ps); strat = stratify_results(ps)
            all_final[f'standard_{mt}_{dp}'] = {'accuracy': acc, 'n': len(ps), 'stratified': strat}
            print(f"    standard {dp}: {acc:.4f}")
            for sk, sv in strat.items():
                for bk, bv in sv.items(): print(f"      {sk}/{bk}: {bv['acc']:.4f} (n={bv['n']})")
            mh = gen_mul_heavy_test(N_TEST, SEED+2000)
            if mh:
                ps2 = gen_eval_with_stats(m, tok, mh, max_len, decode_policy=dp, device=DEVICE)
                all_final[f'mul_heavy_{mt}_{dp}'] = {'accuracy': sum(r['correct'] for r in ps2)/len(ps2), 'n': len(ps2)}
                print(f"    mul_heavy {dp}: {all_final[f'mul_heavy_{mt}_{dp}']['accuracy']:.4f}")
            for cd in [3, 5, 7]:
                cc = gen_chain_test(500, SEED+3000+cd, cd)
                if cc:
                    ps2 = gen_eval_with_stats(m, tok, cc, max_len, decode_policy=dp, device=DEVICE)
                    all_final[f'chain_{cd}_{mt}_{dp}'] = {'accuracy': sum(r['correct'] for r in ps2)/len(ps2), 'n': len(ps2)}
                    print(f"    chain={cd} {dp}: {all_final[f'chain_{cd}_{mt}_{dp}']['accuracy']:.4f}")
            for d in DEPTH_SWEEP:
                dt = gen_depth_test(500, SEED+4000+d, d)
                if dt:
                    ps2 = gen_eval_with_stats(m, tok, dt, max_len, decode_policy=dp, device=DEVICE)
                    all_final[f'depth_sweep_{d}_{mt}_{dp}'] = {'accuracy': sum(r['correct'] for r in ps2)/len(ps2), 'n': len(ps2)}
                    print(f"    depth>={d} {dp}: {all_final[f'depth_sweep_{d}_{mt}_{dp}']['accuracy']:.4f}")
        if mt == 'puma':
            cov = simulate_puma_coverage(m, tok, test_data, max_len, device=DEVICE)
            all_final[f'coverage_{mt}'] = cov
            for dn, dv in cov.items(): print(f"    coverage/{dn}: {dv['mean_coverage']:.3f} (n={dv['n']})")
        ps_conf = gen_eval_with_stats(m, tok, test_data, max_len, decode_policy='confidence', device=DEVICE)
        el = analyse_error_localization(ps_conf); all_final[f'error_loc_{mt}'] = el
        if el['total_errors'] > 0:
            parts = [f"{k}={v:.2f}" for k,v in el.items() if k!='total_errors' and isinstance(v, float)]
            print(f"    errors: {el['total_errors']} — {' '.join(parts)}")
        del m; torch.cuda.empty_cache() if torch.cuda.is_available() else None
    figs = make_figures(all_dyn, all_final)
    sd = {'config': {k: globals()[k] for k in ['ANS_LEN','MAX_DEPTH','MAX_OPERAND','EXPR_PAD_LEN',
          'N_TRAIN','N_TEST','MAX_EPOCHS','BATCH_SIZE','N_LAYER','N_HEAD','N_EMBD','MASK_TYPES','DECODE_POLICIES']}}
    for k, v in all_dyn.items(): sd[f'dyn_{k}'] = {'checkpoints': v['checkpoints'], 'train_loss': v['train_loss']}
    for k, v in all_final.items(): sd[f'final_{k}'] = v
    save_results(exp_name, sd, figures=figs)
    print(f"\n{'='*70}\n  SUMMARY\n{'='*70}")
    for dp in DECODE_POLICIES:
        print(f"\n  ── {dp} ──"); print(f"  {'Test':<30s}", end='')
        for mt in MASK_TYPES: print(f" {mt:>14s}", end='')
        print()
        for tt in ['standard','mul_heavy']+[f'chain_{d}' for d in [3,5,7]]:
            accs = [all_final.get(f'{tt}_{mt}_{dp}',{}).get('accuracy') for mt in MASK_TYPES]
            if any(a is not None for a in accs):
                print(f"  {tt:<30s}", end='')
                for a in accs: print(f" {a:>14.4f}" if a is not None else f" {'N/A':>14s}", end='')
                print()
        for d in DEPTH_SWEEP:
            accs = [all_final.get(f'depth_sweep_{d}_{mt}_{dp}',{}).get('accuracy') for mt in MASK_TYPES]
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
        t = f"{args.tag}_s{seed}" if args.tag and len(seeds)>1 else args.tag
        if len(seeds)>1: print(f"\n{'#'*70}\n# Seed {seed} ({si+1}/{len(seeds)})\n{'#'*70}")
        run(tag=t)
