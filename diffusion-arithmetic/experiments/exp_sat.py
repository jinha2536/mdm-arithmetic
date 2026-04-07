"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Boolean Satisfiability — BCP Depth + PUMA Coverage Deficit
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Task: find a satisfying assignment for a planted-solution 3-SAT
    Input:  "+a-c+f-b+d+e...=????????????????"  (compact, no parens)
    Output: "0110101001110101"  (binary assignment)

  Encoding: each literal = 2 chars (+a or -c), clause = 6 chars (3 lits)
    Clauses concatenated directly; model learns 6-char periodicity.
    Repeated-literal trick for unit/binary clauses:
      Unit:   "+a+a+a"  (forces a)
      Binary: "-a+b-a"  (a=1 → forces b)

  Dependency: BCP depth ←→ carry chain length
  Training: random vs puma
  Decode: confidence | bcp_oracle | random
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

EXP_NAME = 'exp_sat'

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Config — compact encoding: 6 chars/clause (no parens)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
N_VARS = 16; CLAUSE_K = 3
CLAUSE_RATIO = 2.5          # α = clauses/vars (< 4.27 phase transition)
N_CLAUSES = int(N_VARS * CLAUSE_RATIO)
ANS_LEN = N_VARS
VAR_NAMES = [chr(ord('a') + i) for i in range(26)]
CLAUSE_CHAR_LEN = CLAUSE_K * 2   # "+a-c+f" = 6 (no parentheses)
INPUT_LEN = N_CLAUSES * CLAUSE_CHAR_LEN

N_TRAIN = 20000; N_TEST = 1000; BATCH_SIZE = 128
MAX_EPOCHS = 5000; EVAL_EVERY = 100; LOG_EVERY = 50

MASK_TYPES = ['random', 'puma']
DECODE_POLICIES = ['confidence', 'bcp_oracle', 'random']

N_LAYER = 3; N_HEAD = 4; N_EMBD = 256; DROPOUT = 0.1; POS_ENC = 'absolute'
LR = 1e-3; MIN_LR = 1e-4; WARMUP_EPOCHS = 10; GRAD_CLIP = 1.0
PUMA_TAU = 0.9; PUMA_K_START = 2; PUMA_K_END = ANS_LEN
SEED = 42; BCP_DEPTH_SWEEP = [0, 1, 2, 3, 4, 6, 8, 10, 12, 14, 16]


def _update_derived():
    """Recompute derived constants after config changes."""
    globals()['N_CLAUSES'] = int(N_VARS * CLAUSE_RATIO)
    globals()['ANS_LEN'] = N_VARS
    globals()['INPUT_LEN'] = globals()['N_CLAUSES'] * CLAUSE_CHAR_LEN
    globals()['PUMA_K_END'] = N_VARS


def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--n-vars', type=int); p.add_argument('--n-clauses', type=int)
    p.add_argument('--clause-ratio', type=float)
    p.add_argument('--n-train', type=int); p.add_argument('--n-test', type=int)
    p.add_argument('--epochs', type=int); p.add_argument('--batch-size', type=int)
    p.add_argument('--eval-every', type=int)
    p.add_argument('--n-layer', type=int); p.add_argument('--n-head', type=int)
    p.add_argument('--n-embd', type=int); p.add_argument('--dropout', type=float)
    p.add_argument('--puma-tau', type=float)
    p.add_argument('--masks', nargs='+'); p.add_argument('--decode', nargs='+')
    p.add_argument('--tag', type=str, default=''); p.add_argument('--seed', type=int)
    p.add_argument('--seeds', nargs='+', type=int)
    try: args, _ = p.parse_known_args()
    except SystemExit: args, _ = p.parse_known_args([])
    g = globals()
    for a, gl in {'n_train':'N_TRAIN','n_test':'N_TEST','epochs':'MAX_EPOCHS',
                   'batch_size':'BATCH_SIZE','eval_every':'EVAL_EVERY',
                   'n_layer':'N_LAYER','n_head':'N_HEAD','n_embd':'N_EMBD',
                   'dropout':'DROPOUT','puma_tau':'PUMA_TAU','seed':'SEED',
                   'clause_ratio':'CLAUSE_RATIO'}.items():
        v = getattr(args, a, None)
        if v is not None: g[gl] = v
    if getattr(args, 'n_vars', None) is not None: g['N_VARS'] = args.n_vars
    if getattr(args, 'n_clauses', None) is not None: g['N_CLAUSES'] = args.n_clauses
    _update_derived()
    if getattr(args, 'n_clauses', None) is not None:
        g['N_CLAUSES'] = args.n_clauses  # override ratio-derived value
        g['INPUT_LEN'] = args.n_clauses * CLAUSE_CHAR_LEN
    if args.masks: g['MASK_TYPES'] = args.masks
    if args.decode: g['DECODE_POLICIES'] = args.decode
    return args


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SAT Instance Generation (compact encoding, no parens)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _literal_str(v, p): return ('+' if p else '-') + VAR_NAMES[v]
def _clause_str(lits): return ''.join(_literal_str(v, p) for v, p in lits)
def _assignment_str(sigma, n): return ''.join(str(sigma[i]) for i in range(n))

def _check_clause(clause, sigma):
    for var, pos in clause:
        if (pos and sigma[var] == 1) or (not pos and sigma[var] == 0): return True
    return False

def _run_bcp(clauses, n_vars, sigma):
    """BFS-level BCP: collect ALL newly-forced variables per round."""
    assigned = {}; depth = {}; order = []; cd = 0
    while True:
        new = {}
        for cl in clauses:
            ua = []; sat = False; seen = set()
            for v, p in cl:
                if v in assigned:
                    if (p and assigned[v]==1) or (not p and assigned[v]==0): sat = True; break
                elif v not in seen: ua.append((v, p)); seen.add(v)
            if sat or len(ua) != 1: continue
            v, p = ua[0]
            if v not in assigned and v not in new: new[v] = 1 if p else 0
        if not new: break
        for v, val in new.items():
            assigned[v] = val; depth[v] = cd; order.append((v, cd))
        cd += 1
    for v in range(n_vars):
        if v not in assigned:
            assigned[v] = sigma[v]; depth[v] = cd; order.append((v, cd))
    return depth, order


def gen_sat_instance(rng, n_vars, n_clauses, max_chain_depth=None, mode='natural'):
    sigma = [rng.randint(0, 1) for _ in range(n_vars)]
    clauses = []; chain_set = set()
    if mode == 'chain' and max_chain_depth is not None:
        cl = min(max_chain_depth, n_vars)
        cv = rng.sample(range(n_vars), cl); chain_set = set(cv)
        clauses.append([(cv[0], sigma[cv[0]]==1)]*3)
        for ci in range(1, cl):
            prev, curr = cv[ci-1], cv[ci]
            clauses.append([(prev, sigma[prev]==0), (curr, sigma[curr]==1),
                            (prev, sigma[prev]==0)])
    non_chain = [v for v in range(n_vars) if v not in chain_set]
    att = 0
    while len(clauses) < n_clauses and att < n_clauses * 100:
        att += 1
        if len(non_chain) >= CLAUSE_K:
            vs = rng.sample(non_chain, CLAUSE_K)
        elif non_chain:
            vs = [rng.choice(non_chain) for _ in range(CLAUSE_K)]
        else:
            v = rng.choice(range(n_vars))
            clauses.append([(v, True), (v, False), (v, True)]); continue
        lits = [(v, rng.choice([True, False])) for v in vs]
        if _check_clause(lits, sigma): clauses.append(lits)
    clauses = clauses[:n_clauses]; rng.shuffle(clauses)
    bcp_depth, bcp_order = _run_bcp(clauses, n_vars, sigma)
    input_s = ''.join(_clause_str(cl) for cl in clauses)
    full_s = input_s + '=' + _assignment_str(sigma, n_vars)
    return {'string': full_s, 'sigma': sigma, 'clauses': clauses,
            'bcp_depth': bcp_depth, 'bcp_order': bcp_order,
            'max_bcp_depth': max(bcp_depth.values()) if bcp_depth else 0}


def _dep_label(bcp_d, max_d):
    if bcp_d == 0: return 'seed'
    if bcp_d <= 2: return 'shallow'
    if bcp_d <= max(max_d // 2, 3): return 'mid'
    return 'deep'


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Data generation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def gen_data(n, seed):
    rng = pyrandom.Random(seed); results = []; seen = set()
    weights = [math.exp(-0.3*d) for d in range(N_VARS+1)]
    tw = sum(weights); cdf = []; c = 0
    for w in weights: c += w/tw; cdf.append(c)
    for _ in range(n * 20):
        if len(results) >= n: break
        r = rng.random(); td = 0
        for d, cp in enumerate(cdf):
            if r <= cp: td = d; break
        if td == 0: d = gen_sat_instance(rng, N_VARS, N_CLAUSES, mode='natural')
        else: d = gen_sat_instance(rng, N_VARS, N_CLAUSES, max_chain_depth=td, mode='chain')
        if d['string'] in seen: continue
        seen.add(d['string']); md = d['max_bcp_depth']
        d['dep_labels'] = [_dep_label(d['bcp_depth'].get(i, md), max(md,1)) for i in range(N_VARS)]
        results.append(d)
    if len(results) < n: print(f"  WARNING: gen_data got {len(results)}/{n}")
    return results

def gen_chain_test(n, seed, chain_depth):
    rng = pyrandom.Random(seed); results = []; seen = set()
    for _ in range(n*50):
        if len(results) >= n: break
        d = gen_sat_instance(rng, N_VARS, N_CLAUSES, max_chain_depth=chain_depth, mode='chain')
        if d['string'] in seen: continue
        seen.add(d['string']); md = d['max_bcp_depth']
        d['dep_labels'] = [_dep_label(d['bcp_depth'].get(i, md), max(md,1)) for i in range(N_VARS)]
        results.append(d)
    if len(results) < n: print(f"  WARNING: chain={chain_depth}: {len(results)}/{n}")
    return results

def gen_min_depth_test(n, seed, min_depth):
    rng = pyrandom.Random(seed); results = []; seen = set()
    for _ in range(n*100):
        if len(results) >= n: break
        d = gen_sat_instance(rng, N_VARS, N_CLAUSES, max_chain_depth=min_depth, mode='chain')
        if d['max_bcp_depth'] < min_depth or d['string'] in seen: continue
        seen.add(d['string']); md = d['max_bcp_depth']
        d['dep_labels'] = [_dep_label(d['bcp_depth'].get(i, md), max(md,1)) for i in range(N_VARS)]
        results.append(d)
    if len(results) < n: print(f"  WARNING: depth>={min_depth}: {len(results)}/{n}")
    return results

def gen_full_chain_test(n, seed): return gen_chain_test(n, seed, N_VARS)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tokenizer (no parens needed)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def build_tok():
    return CharTokenizer(list('01+-=#') + VAR_NAMES[:N_VARS], {'mask': 'M', 'pad': 'P'})
def get_answer(s): return s.split('=')[1]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BCP Oracle Decode (vectorized, O(max_depth) forward passes)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@torch.no_grad()
def _decode_bcp_oracle(model, prefix_ids, ans_len, mask_id, bcp_depths_batch, device):
    B = prefix_ids.shape[0]; pm = prefix_ids.shape[1]
    full = torch.cat([prefix_ids.to(device),
                      torch.full((B, ans_len), mask_id, dtype=torch.long, device=device)], dim=1)
    max_depth = max(max(d.values()) for d in bcp_depths_batch) + 1
    _ar = torch.arange(ans_len, device=device)
    # Precompute BCP depth tensor (B, ans_len)
    bcp_t = torch.tensor([[bcp_depths_batch[i].get(j, max_depth)
                           for j in range(ans_len)] for i in range(B)],
                          dtype=torch.long, device=device)
    ap = torch.arange(pm, pm + ans_len, device=device).unsqueeze(0).expand(B, -1)
    bi = torch.arange(B, device=device).unsqueeze(1).expand_as(ap)

    for depth in range(max_depth):
        reveal = (bcp_t == depth)  # (B, ans_len)
        if not reveal.any(): continue
        logits = model(full)
        al = logits[bi, ap]  # (B, ans_len, V)
        al[:, :, mask_id] = -float('inf')
        preds = al.argmax(dim=-1)  # (B, ans_len)
        full[bi[reveal], ap[reveal]] = preds[reveal]
    return full


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Probe & Analysis (vectorized)
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
    dep_names = ['seed','shallow','mid','deep']
    dci = {n: i for i, n in enumerate(dep_names)}; nc = len(dep_names)
    dep_ids = torch.tensor([[dci.get(dl,0) for dl in d['dep_labels']] for d in test_data],
                            dtype=torch.long, device=device)
    cat_conf = torch.zeros(nc, device=device); cat_acc = torch.zeros(nc, device=device)
    cat_n = torch.zeros(nc, dtype=torch.long, device=device)
    for st in range(0, len(test_data), 128):
        en = min(st+128, len(test_data)); B = en-st
        ids, ans = ids_all[st:en], ans_all[st:en]; T = ids.shape[1]
        ap = (ans.unsqueeze(1)+_arange).clamp(max=T-1)
        bi = torch.arange(B, device=device).unsqueeze(1).expand_as(ap)
        xm = ids.clone(); xm[bi, ap] = mask_id
        logits = model(xm); al = logits[bi, ap]; tgt = ids[bi, ap]
        losses = -F.log_softmax(al, dim=-1).gather(2, tgt.unsqueeze(2)).squeeze(2)
        cl = al.clone(); cl[:,:,mask_id] = -float('inf')
        probs = F.softmax(cl, dim=-1)
        confs = probs.max(dim=-1).values; corrects = (probs.argmax(dim=-1)==tgt).float()
        L += losses.sum(0); C += corrects.sum(0); CF += confs.sum(0); N_cnt += B
        bd = dep_ids[st:en]
        for ci in range(nc):
            m = (bd==ci)
            if m.any(): cat_conf[ci] += confs[m].sum(); cat_acc[ci] += corrects[m].sum(); cat_n[ci] += m.sum()
    s = N_cnt.clamp(1)
    result = {'pos_loss': (L/s).cpu().tolist(), 'pos_acc': (C/s).cpu().tolist(),
              'pos_conf': (CF/s).cpu().tolist(),
              'overall_loss': (L.sum()/s.sum()).item(), 'overall_acc': (C.sum()/s.sum()).item()}
    dc = {}
    for ci, cn in enumerate(dep_names):
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
        prefixes = [d['string'].split('=')[0]+'=' for d in batch]
        penc = [tokenizer.encode(p) for p in prefixes]; pm = max(len(p) for p in penc)
        pids = torch.full((B, pm), pad_id, dtype=torch.long)
        for i, e in enumerate(penc): pids[i, :len(e)] = torch.tensor(e)
        if decode_policy == 'bcp_oracle':
            gen = _decode_bcp_oracle(model, pids, ANS_LEN, mask_id,
                                     [d['bcp_depth'] for d in batch], device)
        else:
            policy = 'random' if decode_policy == 'random' else 'confidence'
            gen, _, _ = generate_diffusion(model, pids, ANS_LEN, mask_id,
                                           policy=policy, greedy=True, device=device)
        pred_ids = gen[:, pm:pm+ANS_LEN]
        for i in range(B):
            ps = tokenizer.decode(pred_ids[i].cpu().tolist())
            gs = get_answer(batch[i]['string'])
            pc = [ps[j]==gs[j] if j<len(ps) else False for j in range(len(gs))]
            errs = [j for j in range(len(gs)) if j>=len(ps) or ps[j]!=gs[j]]
            all_sat = False
            if len(ps)==N_VARS and all(c in '01' for c in ps):
                all_sat = all(_check_clause(cl, [int(c) for c in ps]) for cl in batch[i]['clauses'])
            out.append({'correct': ps==gs, 'pos_correct': pc, 'error_positions': errs,
                        'satisfies_formula': all_sat, 'max_bcp_depth': batch[i]['max_bcp_depth'],
                        'bcp_depth': batch[i]['bcp_depth'], 'dep_labels': batch[i]['dep_labels']})
    return out


def stratify_results(per_sample):
    def _db(d):
        if d<=1: return 'bcp=0-1'
        if d<=3: return 'bcp=2-3'
        if d<=6: return 'bcp=4-6'
        if d<=10: return 'bcp=7-10'
        return 'bcp=11+'
    bk = defaultdict(list)
    for r in per_sample: bk[_db(r['max_bcp_depth'])].append(r['correct'])
    return {'max_bcp_depth': {k: {'acc': sum(v)/len(v), 'n': len(v)} for k, v in sorted(bk.items())}}

def analyse_per_depth_accuracy(per_sample):
    dc = defaultdict(list)
    for r in per_sample:
        for j in range(ANS_LEN):
            d = r['bcp_depth'].get(j, 0) if isinstance(r['bcp_depth'], dict) else 0
            dc[d].append(r['pos_correct'][j] if j < len(r['pos_correct']) else False)
    return {d: {'acc': sum(cs)/len(cs), 'n': len(cs)} for d, cs in sorted(dc.items())}

@torch.no_grad()
def simulate_puma_coverage(model, tokenizer, test_data, max_len,
                           K=None, tau=None, n_samples=200, batch_size=32, device=None):
    if device is None: device = DEVICE
    if K is None: K = PUMA_K_END
    if tau is None: tau = PUMA_TAU
    model.eval(); mask_id = tokenizer.special_ids['mask']
    N = min(len(test_data), n_samples); _ar = torch.arange(ANS_LEN, device=device)
    dep_names = ['seed','shallow','mid','deep']
    dci = {n: i for i, n in enumerate(dep_names)}; nc = len(dep_names)
    cov_sum = torch.zeros(nc); cov_n = torch.zeros(nc, dtype=torch.long)
    strings = [d['string'] for d in test_data[:N]]
    ids_all, ans_all = encode_samples(strings, tokenizer, max_len)
    ids_all, ans_all = ids_all.to(device), ans_all.to(device)
    dep_ids = torch.tensor([[dci.get(dl,0) for dl in d['dep_labels']] for d in test_data[:N]],
                            dtype=torch.long, device=device)
    for st in range(0, N, batch_size):
        en = min(st+batch_size, N); B = en-st
        ids = ids_all[st:en]; ans_s = ans_all[st:en]; T = ids.shape[1]
        ap = (ans_s.unsqueeze(1)+_ar).clamp(max=T-1)
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
        frac = steps_m / total; bd = dep_ids[st:en]
        for ci in range(nc):
            m = (bd==ci)
            if m.any(): cov_sum[ci] += frac.cpu()[m.cpu()].sum(); cov_n[ci] += m.sum().item()
    return {dn: {'mean_coverage': cov_sum[ci].item()/cov_n[ci].item(), 'n': cov_n[ci].item()}
            for ci, dn in enumerate(dep_names) if cov_n[ci] > 0}

def analyse_error_localization(per_sample):
    cats = defaultdict(int); total = 0
    for r in per_sample:
        if r['correct']: continue
        for j in r['error_positions']:
            if j < len(r['dep_labels']): cats[r['dep_labels'][j]] += 1; total += 1
    if total == 0: return {'total_errors': 0}
    return {'total_errors': total, **{k: v/total for k, v in cats.items()}}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Training
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def train_model(tokenizer, train_data, test_data, max_len,
                mask_type='random', device=None):
    if device is None: device = DEVICE
    assert N_EMBD % N_HEAD == 0, f"N_EMBD({N_EMBD}) must be divisible by N_HEAD({N_HEAD})"
    train_ids, train_ans = encode_samples([d['string'] for d in train_data], tokenizer, max_len)
    train_ids, train_ans = train_ids.to(device), train_ans.to(device)
    N, T = train_ids.shape; bpe = (N+BATCH_SIZE-1)//BATCH_SIZE; total_iters = MAX_EPOCHS*bpe
    mask_id = tokenizer.special_ids['mask']
    model = Transformer(vocab_size=len(tokenizer), block_size=max_len+8, n_layer=N_LAYER,
                        n_head=N_HEAD, n_embd=N_EMBD, dropout=DROPOUT,
                        is_causal=False, pos_enc=POS_ENC).to(device)
    print(f"  [{mask_type}] params={model.n_params:,}, {bpe} batches/ep, {MAX_EPOCHS} ep, T={T}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9,0.99), weight_decay=0.1)
    warmup_iters = WARMUP_EPOCHS * bpe
    def get_lr(it):
        if it < warmup_iters: return LR*it/max(warmup_iters,1)
        ratio = (it-warmup_iters)/max(total_iters-warmup_iters,1)
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
        parts = [f"{c}={dc[c]['acc']:.2f}" for c in ['seed','shallow','mid','deep'] if c in dc]
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
                m = torch.bernoulli(t_r.unsqueeze(1)*ans_mask.float()).bool()
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
            for d in BCP_DEPTH_SWEEP:
                k = f'depth_sweep_{d}_{mt}_{dp}'
                if k in all_final: xs.append(d); ys.append(all_final[k]['accuracy'])
            if xs: ax.plot(xs, ys, 'o-', color=col, label=mt, lw=2)
        ax.set_xlabel('Min BCP Depth'); ax.set_ylabel('Accuracy')
        ax.set_title(f'{dp} decode'); ax.legend(); ax.grid(alpha=0.3)
    fig.suptitle('BCP Depth Sweep', y=1.02); fig.tight_layout(); figs['depth_sweep'] = fig
    return figs


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Run
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def run(tag=''):
    exp_name = f"{EXP_NAME}{'_'+tag if tag else ''}"
    torch.manual_seed(SEED); pyrandom.seed(SEED)
    tok = build_tok(); max_len = INPUT_LEN + 1 + ANS_LEN + 2
    print(f"\n{'='*70}\n  {exp_name} | N_VARS={N_VARS} N_CL={N_CLAUSES} α={CLAUSE_RATIO} "
          f"INPUT_LEN={INPUT_LEN} max_len={max_len}\n{'='*70}")
    train_data = gen_data(N_TRAIN, SEED); test_data = gen_data(N_TEST, SEED+1000)
    print(f"  Train: {len(train_data)}, Test: {len(test_data)}")
    dd = defaultdict(int)
    for d in train_data: dd[d['max_bcp_depth']] += 1
    print(f"  BCP depth dist: {dict(sorted(dd.items()))}")
    all_dyn, all_final = {}, {}
    for mt in MASK_TYPES:
        print(f"\n{'━'*60}\n  Training: {mt}\n{'━'*60}")
        m, dyn = train_model(tok, train_data, test_data, max_len, mask_type=mt, device=DEVICE)
        all_dyn[mt] = dyn
        for dp in DECODE_POLICIES:
            ps = gen_eval_with_stats(m, tok, test_data, max_len, decode_policy=dp, device=DEVICE)
            acc = sum(r['correct'] for r in ps)/len(ps)
            sat = sum(r['satisfies_formula'] for r in ps)/len(ps)
            strat = stratify_results(ps)
            all_final[f'standard_{mt}_{dp}'] = {'accuracy': acc, 'sat_rate': sat, 'n': len(ps), 'stratified': strat}
            print(f"    standard {dp}: acc={acc:.4f} sat={sat:.4f}")
            for cd in [4, 8, 12, 16]:
                if cd > N_VARS: continue
                cc = gen_chain_test(500, SEED+3000+cd, cd)
                if cc:
                    ps2 = gen_eval_with_stats(m, tok, cc, max_len, decode_policy=dp, device=DEVICE)
                    a2 = sum(r['correct'] for r in ps2)/len(ps2)
                    s2 = sum(r['satisfies_formula'] for r in ps2)/len(ps2)
                    all_final[f'chain_{cd}_{mt}_{dp}'] = {'accuracy': a2, 'sat_rate': s2, 'n': len(ps2)}
                    print(f"    chain={cd} {dp}: acc={a2:.4f} sat={s2:.4f}")
            fc = gen_full_chain_test(500, SEED+5000)
            if fc:
                ps2 = gen_eval_with_stats(m, tok, fc, max_len, decode_policy=dp, device=DEVICE)
                a2 = sum(r['correct'] for r in ps2)/len(ps2)
                s2 = sum(r['satisfies_formula'] for r in ps2)/len(ps2)
                all_final[f'full_chain_{mt}_{dp}'] = {'accuracy': a2, 'sat_rate': s2, 'n': len(ps2)}
                print(f"    full_chain {dp}: acc={a2:.4f} sat={s2:.4f}")
            for d in BCP_DEPTH_SWEEP:
                if d > N_VARS: continue
                dt = gen_min_depth_test(500, SEED+4000+d, d)
                if dt:
                    ps2 = gen_eval_with_stats(m, tok, dt, max_len, decode_policy=dp, device=DEVICE)
                    all_final[f'depth_sweep_{d}_{mt}_{dp}'] = {
                        'accuracy': sum(r['correct'] for r in ps2)/len(ps2), 'n': len(ps2)}
                    print(f"    depth>={d} {dp}: {all_final[f'depth_sweep_{d}_{mt}_{dp}']['accuracy']:.4f}")
        ps_conf = gen_eval_with_stats(m, tok, test_data, max_len, decode_policy='confidence', device=DEVICE)
        pda = analyse_per_depth_accuracy(ps_conf); all_final[f'per_depth_acc_{mt}'] = pda
        for d, info in sorted(pda.items()): print(f"    bcp_depth={d}: acc={info['acc']:.4f} (n={info['n']})")
        if mt == 'puma':
            cov = simulate_puma_coverage(m, tok, test_data, max_len, device=DEVICE)
            all_final[f'coverage_{mt}'] = cov
            for dn, dv in cov.items(): print(f"    cov/{dn}: {dv['mean_coverage']:.3f} (n={dv['n']})")
        el = analyse_error_localization(ps_conf); all_final[f'error_loc_{mt}'] = el
        if el['total_errors'] > 0:
            print(f"    errors: {el['total_errors']} — " +
                  ' '.join(f"{k}={v:.2f}" for k,v in el.items() if k!='total_errors' and isinstance(v,float)))
        del m; torch.cuda.empty_cache() if torch.cuda.is_available() else None
    figs = make_figures(all_dyn, all_final)
    sd = {'config': {k: globals()[k] for k in ['N_VARS','ANS_LEN','N_CLAUSES','CLAUSE_K','CLAUSE_RATIO',
          'INPUT_LEN','N_TRAIN','N_TEST','MAX_EPOCHS','BATCH_SIZE','N_LAYER','N_HEAD','N_EMBD',
          'MASK_TYPES','DECODE_POLICIES']}}
    for k, v in all_dyn.items(): sd[f'dyn_{k}'] = {'checkpoints': v['checkpoints'], 'train_loss': v['train_loss']}
    for k, v in all_final.items(): sd[f'final_{k}'] = v
    save_results(exp_name, sd, figures=figs)
    print(f"\n{'='*70}\n  SUMMARY\n{'='*70}")
    for dp in DECODE_POLICIES:
        print(f"\n  ── {dp} ──"); print(f"  {'Test':<28s}  {'random':>14s}  {'puma':>14s}")
        for tt in ['standard']+[f'chain_{d}' for d in [4,8,12,16] if d<=N_VARS]+['full_chain']:
            vals = [all_final.get(f'{tt}_{mt}_{dp}') for mt in MASK_TYPES]
            if any(v is not None for v in vals):
                print(f"  {tt:<28s}", end='')
                for v in vals:
                    if v: print(f"  {v['accuracy']:.4f}/{v.get('sat_rate',0):.4f}", end='')
                    else: print(f"  {'N/A':>14s}", end='')
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
