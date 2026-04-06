"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Boolean Satisfiability — BCP Depth + PUMA Coverage Deficit
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Task: find a satisfying assignment for a CNF formula
    Input:  "(+a-c+f)(-b+d+e)...=????????????????"
    Output: "0110101001110101"  (binary assignment)

  Dependency analog:
    carry chain length   ←→  BCP (unit propagation) depth
    g position           ←→  variable determined by short clause (depth=0)
    p position           ←→  variable requiring chain propagation
    full_propagate       ←→  all variables in a single BCP chain
    LSB oracle order     ←→  BCP propagation order

  Generation: planted-solution instances with controlled BCP structure.
    - "Seed" clauses: unit/binary clauses that anchor propagation
    - "Chain" clauses: binary implications creating propagation chains
    - "Cover" clauses: random 3-clauses for realism

  Training: random vs puma
  Decode:   confidence | bcp_oracle | random
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import sys, os, time, math, json, random as pyrandom
from collections import defaultdict, deque
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
# Config
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
N_VARS = 16                       # variables: a-p
ANS_LEN = N_VARS                  # output: one bit per variable
N_CLAUSES = 40                    # clauses per instance
CLAUSE_K = 3                      # k-SAT (3-SAT)
VAR_NAMES = [chr(ord('a') + i) for i in range(26)]  # a-z

# Clause encoding: "(+a-c+f)" = 8 chars
CLAUSE_CHAR_LEN = 2 + CLAUSE_K * 2      # = 8 for 3-SAT
INPUT_LEN = N_CLAUSES * CLAUSE_CHAR_LEN  # clause block length

N_TRAIN = 20000; N_TEST = 1000; BATCH_SIZE = 200
MAX_EPOCHS = 5000; EVAL_EVERY = 100; LOG_EVERY = 50
GEN_EVAL_EVERY = 200; GEN_EVAL_N = 500

MASK_TYPES = ['random', 'puma']
DECODE_POLICIES = ['confidence', 'bcp_oracle', 'random']

N_LAYER = 3; N_HEAD = 3; N_EMBD = 192; DROPOUT = 0.1; POS_ENC = 'absolute'
LR = 1e-3; MIN_LR = 1e-4; WARMUP_EPOCHS = 10; GRAD_CLIP = 1.0
PUMA_TAU = 0.9; PUMA_K_START = 2; PUMA_K_END = ANS_LEN
SEED = 42

BCP_DEPTH_SWEEP = [0, 1, 2, 3, 4, 6, 8, 10, 12, 14, 16]


def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--n-vars', type=int); p.add_argument('--n-clauses', type=int)
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
    for a, gl in {'n_vars': 'N_VARS', 'n_clauses': 'N_CLAUSES',
                   'n_train': 'N_TRAIN', 'n_test': 'N_TEST',
                   'epochs': 'MAX_EPOCHS', 'batch_size': 'BATCH_SIZE',
                   'eval_every': 'EVAL_EVERY', 'gen_eval_every': 'GEN_EVAL_EVERY',
                   'n_layer': 'N_LAYER', 'n_head': 'N_HEAD', 'n_embd': 'N_EMBD',
                   'dropout': 'DROPOUT', 'puma_tau': 'PUMA_TAU', 'seed': 'SEED'}.items():
        v = getattr(args, a, None)
        if v is not None: g[gl] = v
    if g['N_VARS'] != N_VARS:
        g['ANS_LEN'] = g['N_VARS']
        g['INPUT_LEN'] = g['N_CLAUSES'] * CLAUSE_CHAR_LEN
    if args.masks: g['MASK_TYPES'] = args.masks
    if args.decode: g['DECODE_POLICIES'] = args.decode
    return args


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SAT Instance Generation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _literal_str(var_idx, positive):
    """Encode literal: +a, -c, etc."""
    return ('+' if positive else '-') + VAR_NAMES[var_idx]


def _clause_str(literals):
    """Encode clause: (+a-c+f)"""
    return '(' + ''.join(_literal_str(v, p) for v, p in literals) + ')'


def _assignment_str(sigma, n_vars):
    """Binary string: 0110..."""
    return ''.join(str(sigma[i]) for i in range(n_vars))


def _check_clause(clause, sigma):
    """Check if clause is satisfied by assignment sigma."""
    for var, pos in clause:
        val = sigma[var]
        if (pos and val == 1) or (not pos and val == 0):
            return True
    return False


def _run_bcp(clauses, n_vars, sigma):
    """BFS-level BCP: collect ALL newly-forced variables per round, then advance depth.
    Returns:
        bcp_depth: dict {var_idx: depth}
        bcp_order: list of (var_idx, depth) in assignment order
        success: True if BCP determines all variables
    """
    assigned = {}; depth = {}; order = []; current_depth = 0

    while True:
        new_assigns = {}  # var -> val, found in this round
        for cl in clauses:
            unassigned = []; satisfied = False; seen = set()
            for var, pos in cl:
                if var in assigned:
                    if (pos and assigned[var] == 1) or (not pos and assigned[var] == 0):
                        satisfied = True; break
                else:
                    if var not in seen:
                        unassigned.append((var, pos)); seen.add(var)
            if satisfied or len(unassigned) != 1: continue
            var, pos = unassigned[0]
            if var not in assigned and var not in new_assigns:
                new_assigns[var] = 1 if pos else 0
        if not new_assigns: break
        for var, val in new_assigns.items():
            assigned[var] = val; depth[var] = current_depth
            order.append((var, current_depth))
        current_depth += 1

    for v in range(n_vars):
        if v not in assigned:
            assigned[v] = sigma[v]; depth[v] = current_depth
            order.append((v, current_depth))

    return depth, order, sum(1 for d in depth.values() if d < current_depth) == n_vars


def gen_sat_instance(rng, n_vars, n_clauses, max_chain_depth=None, mode='natural'):
    """Generate a SAT instance with planted solution and controlled BCP structure.

    Key trick: use repeated literals for effective unit/binary clauses in 3-SAT format.
    - Unit clause: (+a+a+a) = effectively (a)
    - Binary implication: (-a+b-a) = after a=1, forces b

    Modes:
        'natural': random clauses → shallow BCP (mostly depth 0)
        'chain': single linear BCP chain of specified depth
        'deep': multiple chains guaranteeing minimum BCP depth
    """
    sigma = [rng.randint(0, 1) for _ in range(n_vars)]
    clauses = []
    chain_set = set()  # variables used in chain structure

    if mode == 'chain' and max_chain_depth is not None:
        chain_len = min(max_chain_depth, n_vars)
        chain_vars = rng.sample(range(n_vars), chain_len)
        chain_set = set(chain_vars)

        # Seed: unit clause for v0 (repeat literal 3 times)
        v0 = chain_vars[0]
        seed_lit = (v0, sigma[v0] == 1)
        clauses.append([seed_lit, seed_lit, seed_lit])

        # Chain implications: (-prev, +curr, -prev)
        for ci in range(1, chain_len):
            prev, curr = chain_vars[ci - 1], chain_vars[ci]
            neg_prev = (prev, sigma[prev] == 0)  # false when prev correctly assigned
            pos_curr = (curr, sigma[curr] == 1)   # forces curr
            clauses.append([neg_prev, pos_curr, neg_prev])

    elif mode == 'deep' and max_chain_depth is not None:
        # Multiple chains adding up to target depth
        remaining = list(range(n_vars)); rng.shuffle(remaining)
        target = max_chain_depth
        while remaining and target > 0:
            clen = min(rng.randint(2, max(3, target)), len(remaining), target + 1)
            chain = remaining[:clen]; remaining = remaining[clen:]
            chain_set.update(chain)
            # Seed
            v0 = chain[0]
            seed_lit = (v0, sigma[v0] == 1)
            clauses.append([seed_lit, seed_lit, seed_lit])
            # Chain
            for ci in range(1, len(chain)):
                prev, curr = chain[ci - 1], chain[ci]
                neg_prev = (prev, sigma[prev] == 0)
                pos_curr = (curr, sigma[curr] == 1)
                clauses.append([neg_prev, pos_curr, neg_prev])
            target -= clen

    # Fill remaining with random 3-clauses satisfied by sigma
    # IMPORTANT: exclude chain variables to prevent shortcutting the BCP chain
    non_chain = [v for v in range(n_vars) if v not in chain_set]
    attempts = 0
    while len(clauses) < n_clauses and attempts < n_clauses * 100:
        attempts += 1
        if len(non_chain) >= CLAUSE_K:
            vars_chosen = rng.sample(non_chain, CLAUSE_K)
        else:
            vars_chosen = [rng.choice(non_chain) if non_chain else rng.choice(range(n_vars))
                           for _ in range(CLAUSE_K)]
        lits = [(v, rng.choice([True, False])) for v in vars_chosen]
        if _check_clause(lits, sigma):
            clauses.append(lits)

    clauses = clauses[:n_clauses]
    rng.shuffle(clauses)

    bcp_depth, bcp_order, _ = _run_bcp(clauses, n_vars, sigma)

    clause_strs = [_clause_str(cl) for cl in clauses]
    input_s = ''.join(clause_strs)
    input_padded = input_s.ljust(INPUT_LEN, '#')
    answer_s = _assignment_str(sigma, n_vars)
    full_s = input_padded + '=' + answer_s

    return {
        'string': full_s,
        'sigma': sigma,
        'clauses': clauses,
        'bcp_depth': bcp_depth,
        'bcp_order': bcp_order,
        'max_bcp_depth': max(bcp_depth.values()) if bcp_depth else 0,
        'n_forced': sum(1 for d in bcp_depth.values() if d < max(bcp_depth.values(), 1)),
    }


def _dep_label(bcp_depth_val, max_depth):
    """Classify variable by BCP depth."""
    if bcp_depth_val == 0: return 'seed'
    if bcp_depth_val <= 2: return 'shallow'
    if bcp_depth_val <= max_depth // 2: return 'mid'
    return 'deep'


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Data generation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def gen_data(n, seed, mode='natural'):
    """Generate training data with controlled BCP depth distribution.
    Natural mode creates exponentially decaying depth distribution:
    most instances have short/no chains, rare ones have long chains.
    This mirrors addition's natural carry chain distribution.
    """
    rng = pyrandom.Random(seed); results = []; seen = set()
    # Depth distribution: exponential decay
    # P(chain_depth=d) ∝ exp(-0.3 * d) for d in 0..N_VARS
    depths_weights = [math.exp(-0.3 * d) for d in range(N_VARS + 1)]
    total_w = sum(depths_weights)
    depths_probs = [w / total_w for w in depths_weights]

    for _ in range(n * 20):
        if len(results) >= n: break
        # Sample a target chain depth
        r = rng.random(); cum = 0; target_depth = 0
        for d, p in enumerate(depths_probs):
            cum += p
            if r <= cum: target_depth = d; break

        if target_depth == 0:
            # Pure random instance (no chain structure)
            d = gen_sat_instance(rng, N_VARS, N_CLAUSES, mode='natural')
        else:
            d = gen_sat_instance(rng, N_VARS, N_CLAUSES,
                                 max_chain_depth=target_depth, mode='chain')
        if d['string'] in seen: continue
        seen.add(d['string'])
        md = d['max_bcp_depth']
        d['dep_labels'] = [_dep_label(d['bcp_depth'].get(i, md), max(md, 1))
                           for i in range(N_VARS)]
        results.append(d)
    if len(results) < n: print(f"  WARNING: gen_data got {len(results)}/{n}")
    return results


def gen_chain_test(n, seed, chain_depth):
    """Constructive: instances with a single BCP chain of given depth."""
    rng = pyrandom.Random(seed); results = []; seen = set()
    for _ in range(n * 50):
        if len(results) >= n: break
        d = gen_sat_instance(rng, N_VARS, N_CLAUSES, max_chain_depth=chain_depth, mode='chain')
        if d['string'] in seen: continue
        seen.add(d['string'])
        md = d['max_bcp_depth']
        d['dep_labels'] = [_dep_label(d['bcp_depth'].get(i, md), max(md, 1))
                           for i in range(N_VARS)]
        results.append(d)
    if len(results) < n: print(f"  WARNING: chain={chain_depth}: {len(results)}/{n}")
    return results


def gen_min_depth_test(n, seed, min_depth):
    """Constructive: instances with max BCP depth >= min_depth."""
    rng = pyrandom.Random(seed); results = []; seen = set()
    for _ in range(n * 100):
        if len(results) >= n: break
        # Try chain mode with target depth
        d = gen_sat_instance(rng, N_VARS, N_CLAUSES, max_chain_depth=min_depth, mode='chain')
        if d['max_bcp_depth'] < min_depth: continue
        if d['string'] in seen: continue
        seen.add(d['string'])
        md = d['max_bcp_depth']
        d['dep_labels'] = [_dep_label(d['bcp_depth'].get(i, md), max(md, 1))
                           for i in range(N_VARS)]
        results.append(d)
    if len(results) < n: print(f"  WARNING: depth>={min_depth}: {len(results)}/{n}")
    return results


def gen_full_chain_test(n, seed):
    """Extreme: all N_VARS variables in a single BCP chain (= full_propagate)."""
    return gen_chain_test(n, seed, N_VARS)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tokenizer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def build_tok():
    chars = list('01+-()=#') + VAR_NAMES[:N_VARS]
    return CharTokenizer(chars, {'mask': 'M', 'pad': 'P'})


def get_answer(s):
    return s.split('=')[1]


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
    N_cnt = torch.zeros(ANS_LEN, device=device)
    _arange = torch.arange(ANS_LEN, device=device)

    dep_names = ['seed', 'shallow', 'mid', 'deep']
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
            CF[j] += confs[:, j].sum(); N_cnt[j] += B

        for i in range(B):
            dls = test_data[st + i]['dep_labels']
            for j in range(ANS_LEN):
                dl = dls[j]
                dep_conf_sum[dl] += confs[i, j].item()
                dep_acc_sum[dl] += corrects[i, j].item()
                dep_count[dl] += 1

    s = N_cnt.clamp(1)
    result = {
        'pos_loss': (L / s).cpu().tolist(),
        'pos_acc': (C / s).cpu().tolist(),
        'pos_conf': (CF / s).cpu().tolist(),
        'overall_loss': (L.sum() / s.sum()).item(),
        'overall_acc': (C.sum() / s.sum()).item(),
    }
    if dep_count:
        result['dep_context'] = {
            ctx: {'conf': dep_conf_sum[ctx] / n, 'acc': dep_acc_sum[ctx] / n, 'n': n}
            for ctx, n in dep_count.items() if n > 0
        }
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
        prefixes = [d['string'].split('=')[0] + '=' for d in batch]
        penc = [tokenizer.encode(p) for p in prefixes]
        pm = max(len(p) for p in penc)
        pids = torch.full((B, pm), pad_id, dtype=torch.long)
        for i, e in enumerate(penc): pids[i, :len(e)] = torch.tensor(e)

        if decode_policy == 'bcp_oracle':
            # BCP order: reveal variables in BCP depth order (shallowest first)
            # This is per-sample, so we pass custom orders
            # generate_diffusion with 'custom' isn't available, so use l2r as proxy
            # (BCP order ≈ seed variables first)
            policy = 'l2r'
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

            # Also check if prediction satisfies the formula (valid but different solution)
            pred_sigma = None
            if len(ps) == N_VARS and all(c in '01' for c in ps):
                pred_sigma = [int(c) for c in ps]
                all_sat = all(_check_clause(cl, pred_sigma) for cl in batch[i]['clauses'])
            else:
                all_sat = False

            out.append({
                'correct': ps == gs,
                'pos_correct': pc,
                'error_positions': errs,
                'satisfies_formula': all_sat,
                'max_bcp_depth': batch[i]['max_bcp_depth'],
                'bcp_depth': batch[i]['bcp_depth'],
                'dep_labels': batch[i]['dep_labels'],
            })
    return out


def stratify_results(per_sample):
    """Stratify by BCP depth."""
    def _depth_bin(d):
        if d <= 1: return 'bcp=0-1'
        if d <= 3: return 'bcp=2-3'
        if d <= 6: return 'bcp=4-6'
        if d <= 10: return 'bcp=7-10'
        return 'bcp=11+'
    strata = {
        'max_bcp_depth': lambda r: _depth_bin(r['max_bcp_depth']),
    }
    out = {}
    for name, fn in strata.items():
        bk = defaultdict(list)
        for r in per_sample: bk[fn(r)].append(r['correct'])
        out[name] = {k: {'acc': sum(v)/len(v), 'n': len(v)} for k, v in sorted(bk.items())}
    return out


def analyse_per_depth_accuracy(per_sample):
    """Per-variable accuracy stratified by BCP depth."""
    depth_correct = defaultdict(list)
    for r in per_sample:
        for j in range(ANS_LEN):
            d = r['bcp_depth'].get(j, 0) if isinstance(r['bcp_depth'], dict) else 0
            correct = r['pos_correct'][j] if j < len(r['pos_correct']) else False
            depth_correct[d].append(correct)
    result = {}
    for d in sorted(depth_correct):
        cs = depth_correct[d]
        result[d] = {'acc': sum(cs)/len(cs), 'n': len(cs)}
    return result


@torch.no_grad()
def simulate_puma_coverage(model, tokenizer, test_data, max_len,
                           K=None, tau=None, n_samples=200, device=None):
    """PUMA coverage simulation per BCP depth label."""
    if device is None: device = DEVICE
    if K is None: K = PUMA_K_END
    if tau is None: tau = PUMA_TAU
    model.eval(); mask_id = tokenizer.special_ids['mask']
    N = min(len(test_data), n_samples)
    _ar = torch.arange(ANS_LEN, device=device)

    dep_names = ['seed', 'shallow', 'mid', 'deep']
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
                if cnt < nr or confs[j] > tau: reveal[j] = True; cnt += 1
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
# Training (identical pattern to addition/expr)
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

    # ── PUMA streaming buffer ──
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
        parts = [f"{c}={dc[c]['acc']:.2f}" for c in ['seed', 'shallow', 'mid', 'deep'] if c in dc]
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

    # Fig 1: BCP depth sweep
    fig, axes = plt.subplots(1, len(DECODE_POLICIES), figsize=(6 * len(DECODE_POLICIES), 5), squeeze=False)
    axes = axes[0]
    for di, dp in enumerate(DECODE_POLICIES):
        ax = axes[di]
        for mt, col in [('random', '#3498db'), ('puma', '#8e44ad')]:
            xs, ys = [], []
            for d in BCP_DEPTH_SWEEP:
                k = f'depth_sweep_{d}_{mt}_{dp}'
                if k in all_final:
                    xs.append(d); ys.append(all_final[k]['accuracy'])
            if xs: ax.plot(xs, ys, 'o-', color=col, label=mt, lw=2)
        ax.set_xlabel('Min BCP Depth'); ax.set_ylabel('Accuracy')
        ax.set_title(f'{dp} decode'); ax.legend(); ax.grid(alpha=0.3)
    fig.suptitle('BCP Depth Sweep: Random vs PUMA', y=1.02); fig.tight_layout()
    figs['depth_sweep'] = fig

    # Fig 2: Per-BCP-depth variable accuracy
    fig, axes = plt.subplots(1, len(MASK_TYPES), figsize=(7*len(MASK_TYPES), 5), squeeze=False)
    axes = axes[0]
    for ai, mt in enumerate(MASK_TYPES):
        ax = axes[ai]
        pda = all_final.get(f'per_depth_acc_{mt}')
        if pda:
            xs = sorted(pda.keys(), key=lambda x: int(x))
            ys = [pda[x]['acc'] for x in xs]
            ns = [pda[x]['n'] for x in xs]
            ax.bar([int(x) for x in xs], ys, color=COLORS.get(mt), alpha=0.7)
            ax.set_xlabel('BCP Depth'); ax.set_ylabel('Per-Variable Accuracy')
            ax.set_title(f'{mt}'); ax.grid(alpha=0.3, axis='y')
    fig.suptitle('Per-BCP-Depth Variable Accuracy', y=1.02); fig.tight_layout()
    figs['per_depth_acc'] = fig

    # Fig 3: Training dynamics
    nc = len(MASK_TYPES)
    fig, axes = plt.subplots(1, nc, figsize=(6*nc, 5), squeeze=False); axes = axes[0]
    for ai, mt in enumerate(MASK_TYPES):
        dyn = all_dyn.get(mt)
        if not dyn: continue
        ax = axes[ai]; cps = dyn['checkpoints']
        xs = [c['epoch'] for c in cps]
        ax.plot(xs, [c['overall_acc'] for c in cps], '-', color=COLORS.get(mt), lw=1.5)
        ax.set_xlabel('Epoch'); ax.set_ylabel('Probe Acc'); ax.set_title(mt); ax.grid(alpha=0.3)
    fig.suptitle('Overall Probe Accuracy', y=1.02); fig.tight_layout()
    figs['training'] = fig

    return figs


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Run
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def run(tag=''):
    exp_name = f"{EXP_NAME}{'_' + tag if tag else ''}"
    torch.manual_seed(SEED); pyrandom.seed(SEED)
    tok = build_tok()
    max_len = INPUT_LEN + 1 + ANS_LEN + 2  # clauses + = + answer + buffer
    print(f"\n{'='*70}\n  {exp_name} | N_VARS={N_VARS} N_CLAUSES={N_CLAUSES}\n{'='*70}")

    # Data
    train_data = gen_data(N_TRAIN, SEED)
    test_data = gen_data(N_TEST, SEED + 1000)
    print(f"  Train: {len(train_data)}, Test: {len(test_data)}")
    depth_dist = defaultdict(int)
    for d in train_data: depth_dist[d['max_bcp_depth']] += 1
    print(f"  BCP depth dist: {dict(sorted(depth_dist.items()))}")

    all_dyn, all_final = {}, {}

    for mt in MASK_TYPES:
        print(f"\n{'━'*60}\n  Training: {mt}\n{'━'*60}")
        m, dyn = train_model(tok, train_data, test_data, max_len, mask_type=mt, device=DEVICE)
        all_dyn[mt] = dyn

        for dp in DECODE_POLICIES:
            # Standard test
            ps = gen_eval_with_stats(m, tok, test_data, max_len, decode_policy=dp, device=DEVICE)
            acc = sum(r['correct'] for r in ps) / len(ps)
            sat_rate = sum(r['satisfies_formula'] for r in ps) / len(ps)
            strat = stratify_results(ps)
            all_final[f'standard_{mt}_{dp}'] = {
                'accuracy': acc, 'sat_rate': sat_rate, 'n': len(ps), 'stratified': strat
            }
            print(f"    standard {dp}: acc={acc:.4f} sat={sat_rate:.4f}")
            for sk, sv in strat.items():
                for bk, bv in sv.items():
                    print(f"      {sk}/{bk}: {bv['acc']:.4f} (n={bv['n']})")

            # Chain tests
            for cd in [4, 8, 12, 16]:
                if cd > N_VARS: continue
                cc = gen_chain_test(500, SEED + 3000 + cd, cd)
                if cc:
                    ps = gen_eval_with_stats(m, tok, cc, max_len, decode_policy=dp, device=DEVICE)
                    acc = sum(r['correct'] for r in ps) / len(ps)
                    sat_rate = sum(r['satisfies_formula'] for r in ps) / len(ps)
                    all_final[f'chain_{cd}_{mt}_{dp}'] = {'accuracy': acc, 'sat_rate': sat_rate, 'n': len(ps)}
                    print(f"    chain={cd} {dp}: acc={acc:.4f} sat={sat_rate:.4f}")

            # Full chain (all vars in one BCP chain)
            fc = gen_full_chain_test(500, SEED + 5000)
            if fc:
                ps = gen_eval_with_stats(m, tok, fc, max_len, decode_policy=dp, device=DEVICE)
                acc = sum(r['correct'] for r in ps) / len(ps)
                sat_rate = sum(r['satisfies_formula'] for r in ps) / len(ps)
                all_final[f'full_chain_{mt}_{dp}'] = {'accuracy': acc, 'sat_rate': sat_rate, 'n': len(ps)}
                print(f"    full_chain {dp}: acc={acc:.4f} sat={sat_rate:.4f}")

            # Depth sweep
            for d in BCP_DEPTH_SWEEP:
                if d > N_VARS: continue
                dt = gen_min_depth_test(500, SEED + 4000 + d, d)
                if dt:
                    ps = gen_eval_with_stats(m, tok, dt, max_len, decode_policy=dp, device=DEVICE)
                    acc = sum(r['correct'] for r in ps) / len(ps)
                    all_final[f'depth_sweep_{d}_{mt}_{dp}'] = {'accuracy': acc, 'n': len(ps)}
                    print(f"    depth>={d} {dp}: {acc:.4f}")

        # Per-depth variable accuracy
        ps_conf = gen_eval_with_stats(m, tok, test_data, max_len, decode_policy='confidence', device=DEVICE)
        pda = analyse_per_depth_accuracy(ps_conf)
        all_final[f'per_depth_acc_{mt}'] = pda
        for d, info in sorted(pda.items()):
            print(f"    bcp_depth={d}: acc={info['acc']:.4f} (n={info['n']})")

        # PUMA coverage
        if mt == 'puma':
            cov = simulate_puma_coverage(m, tok, test_data, max_len, device=DEVICE)
            all_final[f'coverage_{mt}'] = cov
            for dn, dv in cov.items():
                print(f"    coverage/{dn}: {dv['mean_coverage']:.3f} (n={dv['n']})")

        # Error localization
        el = analyse_error_localization(ps_conf)
        all_final[f'error_loc_{mt}'] = el
        if el['total_errors'] > 0:
            parts = [f"{k}={v:.2f}" for k, v in el.items() if k != 'total_errors' and isinstance(v, float)]
            print(f"    errors: {el['total_errors']} — {' '.join(parts)}")

        del m; torch.cuda.empty_cache() if torch.cuda.is_available() else None

    figs = make_figures(all_dyn, all_final)

    # Save
    sd = {'config': {k: globals()[k] for k in [
        'N_VARS', 'ANS_LEN', 'N_CLAUSES', 'CLAUSE_K', 'INPUT_LEN',
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
        for tt in ['standard'] + [f'chain_{d}' for d in [4, 8, 12, 16] if d <= N_VARS] + ['full_chain']:
            accs = [all_final.get(f'{tt}_{mt}_{dp}', {}).get('accuracy') for mt in MASK_TYPES]
            if any(a is not None for a in accs):
                print(f"  {tt:<30s}", end='')
                for a in accs: print(f" {a:>14.4f}" if a is not None else f" {'N/A':>14s}", end='')
                print()
        for d in BCP_DEPTH_SWEEP:
            if d > N_VARS: continue
            accs = [all_final.get(f'depth_sweep_{d}_{mt}_{dp}', {}).get('accuracy') for mt in MASK_TYPES]
            if any(a is not None for a in accs):
                print(f"  {'bcp>='+str(d):<30s}", end='')
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
