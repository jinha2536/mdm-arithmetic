"""ListOps: nested prefix expressions over MAX / MIN / MED / SUM-MOD-10.

The model outputs the post-order evaluation trace (one digit per sub-expression,
inner to outer), rainbow-padded to MAX_ANS_LEN. Training schemes: random / PAPL
/ PUMA masking. Decoding (one position per forward pass): confidence, layered
post-order (children before parents, ties broken by confidence) and uniform
random. Deep trees are made rare in training through DEPTH_DECAY; test trees
are stratified by depth (N_PER_DEPTH trees of each depth 1..MAX_DEPTH).
Accuracy = exact match of the trace (the positions before the padding).
"""
import sys, os, random
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                if '__file__' in dir() else '.')
from core.tokenizer import CharTokenizer
from core.train_utils import (
    prepare_results_dir, save_results, save_checkpoint, encode_samples,
    train_diffusion, puma_k_step, generate_diffusion, DEVICE,
)

EXP_NAME = 'exp_listops'

# Config
# tree generation
MAX_DEPTH = 5           # max nesting depth (root=depth 0)
MIN_ARGS = 2            # min arguments per operator
MAX_ARGS = 4            # max arguments per operator
EXPAND_PROB = 0.6       # P(argument becomes sub-expression) during generation
OPS = ['X', 'N', 'D', 'S']     # MAX, MIN, MED, SM

# sequence format
MAX_ANS_LEN = 20        # fixed answer length (trace + padding)
MAX_SEQ_LEN = 200       # total sequence length cap

DEPTH_DECAY = 0.5       # training: P(target_depth=d) ~ DEPTH_DECAY^(d-1)

N_TRAIN = 200000; BATCH_SIZE = 256
N_TEST = 5000           # held-out trees for the fully-masked probe
N_PER_DEPTH = 500       # test trees per depth stratum
MAX_ITERS = 300000; EVAL_EVERY = 5000; LOG_EVERY = 1000
MASK_TYPES = ['random', 'papl', 'puma']
DECODE_POLICIES = ['confidence', 'layered_oracle', 'random']
N_LAYER = 8; N_HEAD = 8; N_EMBD = 384; DROPOUT = 0.1
LR = 3e-4; MIN_LR = 1e-5; WARMUP_ITERS = 2000; GRAD_CLIP = 1.0
WEIGHT_DECAY = 0.01; EMA_DECAY = 0.9999
PUMA_TAU = 0.9
PUMA_K_START = 2; PUMA_K_END = 10   # ans_len=20: 10 -> 2 tokens per step
PUMA_K_STEP = 2; PUMA_K_EVERY = None  # None = ramp over the first 1/3 of training
PAPL_TAU = 1.0; PAPL_ALPHA = 5.0
SEED = 42
NO_AMP = True   # bf16 autocast off (fp32 training)


def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--depth-decay', type=float)
    p.add_argument('--n-train', type=int); p.add_argument('--n-test', type=int)
    p.add_argument('--max-iters', type=int); p.add_argument('--batch-size', type=int)
    p.add_argument('--eval-every', type=int)
    p.add_argument('--n-layer', type=int); p.add_argument('--n-head', type=int)
    p.add_argument('--n-embd', type=int); p.add_argument('--dropout', type=float)
    p.add_argument('--lr', type=float); p.add_argument('--weight-decay', type=float)
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
    for a, gl in {
        'depth_decay': 'DEPTH_DECAY',
        'n_train': 'N_TRAIN', 'n_test': 'N_TEST', 'max_iters': 'MAX_ITERS',
        'batch_size': 'BATCH_SIZE', 'eval_every': 'EVAL_EVERY',
        'n_layer': 'N_LAYER', 'n_head': 'N_HEAD', 'n_embd': 'N_EMBD',
        'dropout': 'DROPOUT', 'lr': 'LR', 'weight_decay': 'WEIGHT_DECAY',
        'puma_tau': 'PUMA_TAU',
        'puma_k_start': 'PUMA_K_START', 'puma_k_end': 'PUMA_K_END',
        'puma_k_step': 'PUMA_K_STEP', 'puma_k_every': 'PUMA_K_EVERY',
        'papl_tau': 'PAPL_TAU', 'papl_alpha': 'PAPL_ALPHA', 'seed': 'SEED',
    }.items():
        v = getattr(args, a, None)
        if v is not None:
            g[gl] = v
    if args.no_amp:
        g['NO_AMP'] = True
    if args.masks:
        g['MASK_TYPES'] = args.masks
    if args.decode:
        g['DECODE_POLICIES'] = args.decode
    return args


# Trees
# node: int (literal 0-9) or dict {'op': str, 'args': list, 'depth': int}

def _eval_op(op, values):
    """Evaluate operator on a list of integer values -> single digit 0-9."""
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
    """Post-order evaluation. Returns (result, trace); trace is a list of dicts,
    one per operator in evaluation order (inner to outer), with children_indices
    into the merged trace (sub-trace indices are shifted by the parent's offset).
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
            offset = len(trace)  # where t will start in the merged trace
            for entry in t:
                entry['children_indices'] = [ci + offset for ci in entry['children_indices']]
            # sub-expression root is the last entry of its sub-trace
            children_indices.append(offset + len(t) - 1)
        trace.extend(t)
    result = _eval_op(node['op'], child_results)
    trace.append({'result': result, 'children_indices': children_indices})
    return result, trace


def _tree_depth(node):
    """Max depth of the tree (root=0)."""
    if isinstance(node, int):
        return -1
    return max((0,) + tuple(1 + _tree_depth(a) for a in node['args'] if isinstance(a, dict)))


def _gen_tree(rng, max_depth, current_depth=0, must_reach_max=False):
    """Random ListOps tree; with must_reach_max=True at least one path reaches max_depth."""
    op = rng.choice(OPS)
    n_args = rng.randint(MIN_ARGS, MAX_ARGS)

    if current_depth >= max_depth - 1:
        # leaf level: all literals
        return {'op': op, 'args': [rng.randint(0, 9) for _ in range(n_args)],
                'depth': current_depth}

    args = []
    deep_placed = False

    for i in range(n_args):
        # still need to place at least one deep child?
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


# Format & tokenizer
# input chars: 0-9, X N D S, [ ] (space), =; answer: trace digits, then EOS and
# a cyclic pattern of distinct pad tokens (rainbow padding)

EOS_CHAR = '$'
RAINBOW_CHARS = 'abcdefghijklmnop'  # 16 distinct pad tokens
INPUT_PAD = '#'  # input sequence padding only (never appears in data)

def _rainbow_pad(trace_str):
    """Pad trace to MAX_ANS_LEN with EOS + cyclic rainbow tokens."""
    remaining = MAX_ANS_LEN - len(trace_str)
    if remaining <= 0:
        return trace_str[:MAX_ANS_LEN]
    pad = EOS_CHAR
    for i in range(remaining - 1):
        pad += RAINBOW_CHARS[i % len(RAINBOW_CHARS)]
    return trace_str + pad


def build_tok():
    chars = list('0123456789XNDS[] =') + [EOS_CHAR] + list(RAINBOW_CHARS)
    seen = set()
    unique = []
    for c in chars:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return CharTokenizer(unique, {'mask': 'M', 'pad': INPUT_PAD})


def _format_sample(tree):
    """'expression=padded_trace' and its metadata, or (None, None) if the trace
    or the sequence is too long."""
    expr = _tree_to_str(tree)
    result, trace = _evaluate(tree)
    trace_str = ''.join(str(t['result']) for t in trace)

    if len(trace_str) > MAX_ANS_LEN:
        return None, None
    sample = f"{expr}={_rainbow_pad(trace_str)}"
    if len(sample) > MAX_SEQ_LEN:
        return None, None
    meta = {
        'tree_depth': _tree_depth(tree) + 1,   # 1-indexed: depth 1 = flat
        'trace_len': len(trace_str),
        'children_indices': [t['children_indices'] for t in trace],
    }
    return sample, meta


def get_answer(s):
    """Answer string (padded trace) of a formatted sample."""
    return s.split('=', 1)[1]


# Data generation

def _sample_depth(rng):
    """Sample a target depth from the DEPTH_DECAY geometric distribution."""
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
    """Probe set: n trees, uniform across depths 1..MAX_DEPTH."""
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


def gen_depth_test(n, seed, depth):
    """n trees of depth `depth`."""
    rng = random.Random(seed)
    data, metas = [], []
    for _ in range(n * 100):
        tree = _gen_tree(rng, max_depth=depth, must_reach_max=True)
        s, m = _format_sample(tree)
        if s is not None and m['tree_depth'] >= depth:
            data.append(s)
            metas.append(m)
        if len(data) >= n:
            break
    if len(data) < n:
        print(f"    info: gen_depth_test(d={depth}): {len(data)}/{n}")
    return data[:n], metas[:n]


# Evaluation

@torch.no_grad()
def probe_per_position(model, tokenizer, test_samples, max_len, device=None):
    """Fully-masked probe: every answer position masked, one forward pass; mean
    NLL and argmax accuracy (mask token excluded) over the answer positions."""
    if device is None:
        device = DEVICE
    model.eval()
    mask_id = tokenizer.special_ids['mask']
    ids_all, ans_all = encode_samples(test_samples, tokenizer, max_len)
    ids_all, ans_all = ids_all.to(device), ans_all.to(device)

    L = torch.zeros(MAX_ANS_LEN, device=device)
    C = torch.zeros(MAX_ANS_LEN, device=device)
    N = torch.zeros(MAX_ANS_LEN, device=device)
    _arange = torch.arange(MAX_ANS_LEN, device=device)

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
        corrects = (cl.argmax(dim=-1) == tgt).float()

        L += losses.sum(dim=0)
        C += corrects.sum(dim=0)
        N += B

    s = N.clamp(1)
    return {'overall_loss': (L.sum() / s.sum()).item(),
            'overall_acc': (C.sum() / s.sum()).item()}


def _layered_ranks(meta):
    """Layered post-order rank of each trace position: leaves 0, otherwise
    1 + max rank of the children."""
    tl = min(meta.get('trace_len', 0), MAX_ANS_LEN)
    ci_list = meta.get('children_indices', [])
    ranks = [0] * tl
    for j in range(tl):
        if j < len(ci_list) and ci_list[j]:
            valid = [c for c in ci_list[j] if 0 <= c < j]
            if valid:
                ranks[j] = 1 + max(ranks[c] for c in valid)
    return ranks


@torch.no_grad()
def gen_eval(model, tokenizer, test_samples, test_metas, max_len,
             decode_policy='confidence', device=None):
    """Greedy decoding of the answer region; returns per-sample exact match of
    the trace. Samples are batched by prompt length."""
    if device is None:
        device = DEVICE
    mask_id = tokenizer.special_ids['mask']
    pad_id = tokenizer.special_ids['pad']
    model.eval()
    out = [None] * len(test_samples)

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

            penc = [tokenizer.encode(s.split('=')[0] + '=') for s in batch_s]
            pids = torch.tensor(penc, dtype=torch.long)

            r_rank = None
            if decode_policy == 'layered_oracle':
                # positions past the trace (padding) get rank MAX_ANS_LEN: decoded last
                r_rank = torch.full((B, MAX_ANS_LEN), MAX_ANS_LEN, dtype=torch.long)
                for bi, m in enumerate(batch_m):
                    for j, r in enumerate(_layered_ranks(m)):
                        r_rank[bi, j] = r

            gen, _, _ = generate_diffusion(
                model, pids, MAX_ANS_LEN, mask_id,
                policy=decode_policy, reasoning_rank=r_rank,
                pad_to=max_len, pad_id=pad_id, device=device)
            pred_ids = gen[:, pl:pl + MAX_ANS_LEN]

            for bi in range(B):
                pred_str = tokenizer.decode(pred_ids[bi].cpu().tolist())
                tl = batch_m[bi]['trace_len']
                out[bind[bi]] = pred_str[:tl] == get_answer(batch_s[bi])[:tl]
    return out


# Training

def train_model(mask_type, tokenizer, train_samples, probe_samples, max_len, device=None):
    """train_diffusion with the fully-masked probe on probe_samples as eval callback."""
    if device is None:
        device = DEVICE
    max_iters = MAX_ITERS

    train_ids, train_ans = encode_samples(train_samples, tokenizer, max_len)
    train_ids, train_ans = train_ids.to(device), train_ans.to(device)

    k_sched = None
    if mask_type == 'puma':
        k_step = PUMA_K_STEP or 3
        if PUMA_K_EVERY is not None:
            k_every = PUMA_K_EVERY
        else:
            n_inc = max(1, (PUMA_K_END - PUMA_K_START) // k_step)
            k_every = max(1000, (max_iters // 3) // n_inc)
        k_sched = puma_k_step(PUMA_K_START, PUMA_K_END, k_step, k_every)
        print(f"  PUMA K: step {PUMA_K_START}->{k_sched(max_iters)} "
              f"(+{k_step} every {k_every // 1000}k, cap={PUMA_K_END})")

    def eval_fn(model, it, tg):
        probe = probe_per_position(model, tokenizer, probe_samples, max_len, device)
        print(f"    [eval it {it}] loss={probe['overall_loss']:.4f} "
              f"acc={probe['overall_acc']:.4f}")
        return probe

    return train_diffusion(
        train_ids=train_ids, train_ans=train_ans, ans_len=MAX_ANS_LEN,
        tokenizer=tokenizer,
        mask_type=mask_type, blank_masks=None,
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
    torch.manual_seed(SEED)
    random.seed(SEED)
    tok = build_tok()

    print(f"\n{'=' * 70}")
    print(f"  ListOps MAX_DEPTH={MAX_DEPTH} MAX_ANS_LEN={MAX_ANS_LEN} "
          f"DEPTH_DECAY={DEPTH_DECAY}")
    print(f"  masks={MASK_TYPES} | decode={DECODE_POLICIES}")
    print(f"  N_TRAIN={N_TRAIN} MAX_ITERS={MAX_ITERS} | arch: {N_LAYER}L/{N_HEAD}H/{N_EMBD}D")
    print(f"{'=' * 70}\n")

    train_data, train_metas = gen_train_data(N_TRAIN, seed=SEED)
    # position embeddings sized to MAX_SEQ_LEN (deep test trees can exceed the
    # training-set maximum)
    max_len = MAX_SEQ_LEN
    train_depths = [m['tree_depth'] for m in train_metas]
    print(f"  Train depth dist: {dict(sorted((d, train_depths.count(d)) for d in set(train_depths)))}")

    probe_samples, _ = gen_test_data(N_TEST, seed=SEED + 1000)
    depth_sets = {d: gen_depth_test(N_PER_DEPTH, seed=6500 + d, depth=d)
                  for d in range(1, MAX_DEPTH + 1)}

    all_dyn = {}
    all_final = {}
    for mt in MASK_TYPES:
        print(f"\n{'=' * 60}\n{mt}\n{'=' * 60}")
        m, d = train_model(mt, tok, train_data, probe_samples, max_len)
        all_dyn[mt] = d
        save_checkpoint(exp_name, {k: v.cpu().clone() for k, v in m.state_dict().items()}, tag=mt)

        print("  Depth strata...")
        for depth, (cc, cc_m) in depth_sets.items():
            if not cc:
                continue
            for dp in DECODE_POLICIES:
                ok = gen_eval(m, tok, cc, cc_m, max_len, decode_policy=dp, device=DEVICE)
                acc = sum(ok) / len(ok)
                all_final[f'{mt}_depth_{depth}_{dp}'] = {'accuracy': acc, 'n': len(ok)}
                print(f"    depth={depth} {dp}: {acc:.4f}")

        del m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    sd = {
        'config': {k: globals()[k] for k in [
            'MAX_DEPTH', 'MIN_ARGS', 'MAX_ARGS', 'EXPAND_PROB',
            'MAX_ANS_LEN', 'DEPTH_DECAY', 'N_TRAIN', 'N_TEST', 'N_PER_DEPTH',
            'MAX_ITERS', 'BATCH_SIZE', 'N_LAYER', 'N_HEAD', 'N_EMBD',
            'MASK_TYPES', 'DECODE_POLICIES', 'PUMA_K_START', 'PUMA_K_END',
            'PAPL_TAU', 'PAPL_ALPHA', 'SEED',
        ]},
    }
    for k, v in all_dyn.items():
        sd[f'dyn_{k}'] = {'checkpoints': v['checkpoints'], 'train_loss': v['train_loss']}
    for k, v in all_final.items():
        sd[f'final_{k}'] = v
    save_results(exp_name, sd)

    print(f"\n{'=' * 70}\n  SUMMARY (trace exact match)\n{'=' * 70}")
    print(f"\n  {'Test':<35s}", end='')
    for mt in MASK_TYPES:
        print(f" {mt:>14s}", end='')
    print()
    for dp in DECODE_POLICIES:
        for depth in range(1, MAX_DEPTH + 1):
            accs = [all_final.get(f'{mt}_depth_{depth}_{dp}', {}).get('accuracy')
                    for mt in MASK_TYPES]
            if any(a is not None for a in accs):
                print(f"  {'depth=' + str(depth) + '_' + dp:<35s}", end='')
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
            print(f"\n{'#' * 70}\n# Seed {seed} ({si + 1}/{len(seeds)})\n{'#' * 70}")
        run(tag=t)
