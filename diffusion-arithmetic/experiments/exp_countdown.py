"""Countdown (CD4): combine four numbers with +, -, *, / to reach a target,
written as a three-step equation chain.

  prompt  "86,28,13,31,96"          (four inputs, target)
  answer  "86+28=114,31-13=18,114-18=96"

Instances are stratified by solution multiplicity m (number of distinct equation
chains reaching the target, enumerated by DFS); m in [1, 3] puzzles are
sub-sampled in training (LOW_MULT_SAMPLE_RATE). Decoding (one position per
forward pass): confidence, step-sequential (complete step i before step i+1)
and uniform random; accuracy = exact match of the (padded) answer. Selective
reveal: a random half of the gold answer tokens is revealed and the remaining
answer tokens are predicted in a single forward pass (token accuracy).
"""
import sys, os, time, json, random, itertools
from collections import defaultdict, Counter
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                if '__file__' in dir() else '.')
from core.tokenizer import CharTokenizer
from core.train_utils import (
    prepare_results_dir, save_results, save_checkpoint,
    train_diffusion, puma_k_step, generate_diffusion, DEVICE,
)


def encode_countdown_samples(samples, tokenizer, max_len=None):
    """encode_samples variant using '|' separator."""
    encoded = [tokenizer.encode(s) for s in samples]
    if max_len is None:
        max_len = max(len(e) for e in encoded)
    pad_id = tokenizer.special_ids['pad']
    ids = torch.full((len(encoded), max_len), pad_id, dtype=torch.long)
    ans_starts = torch.zeros(len(encoded), dtype=torch.long)
    sep_char = SEP_CHAR
    for i, enc in enumerate(encoded):
        L = min(len(enc), max_len)
        ids[i, :L] = torch.tensor(enc[:L])
        ans_starts[i] = samples[i].index(sep_char) + 1
    return ids, ans_starts

EXP_NAME = 'exp_countdown'

# Config
MAX_ANS_LEN = 40
MAX_SEQ_LEN = 60

N_TRAIN = None; N_TEST = 1000; BATCH_SIZE = 256
# keep 10% of the m=1-3 training puzzles (the stratum then makes up ~5% of the
# training set; the test set keeps the natural distribution)
LOW_MULT_SAMPLE_RATE = 0.10
MAX_ITERS = 200000; EVAL_EVERY = 5000; LOG_EVERY = 1000
MASK_TYPES = ['random', 'papl', 'puma']
# step_seq: complete step i (plan, then calc) before step i+1
DECODE_POLICIES = ['confidence', 'step_seq', 'random']

N_LAYER = 12; N_HEAD = 12; N_EMBD = 384; DROPOUT = 0.0
LR = 3e-4; MIN_LR = 1e-5; WARMUP_ITERS = 1000; GRAD_CLIP = 1.0
WEIGHT_DECAY = 0.01; EMA_DECAY = 0.9999
PUMA_TAU = 0.9
PUMA_K_START = 4; PUMA_K_END = 20   # ans_len=40: 10 -> 2 tokens per step
PUMA_K_STEP = 3; PUMA_K_EVERY = None  # None = ramp over the first 1/3 of training
PAPL_TAU = 1.0; PAPL_ALPHA = 5.0
SEED = 42
NO_AMP = True   # bf16 autocast off (fp32 training)

SELECTIVE_REVEAL_FRAC = 0.5

DATA_DIR = 'experiments/data'
TRAIN_FILE = 'cd4_train.jsonl'
TEST_FILE = 'cd4_test.jsonl'


def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', type=str)
    p.add_argument('--train-file', type=str); p.add_argument('--test-file', type=str)
    p.add_argument('--n-train', type=int); p.add_argument('--n-test', type=int)
    p.add_argument('--low-mult-sample-rate', type=float, default=None,
                   help='fraction of the m=1-3 training puzzles that is kept (default 0.10)')
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
        'data_dir': 'DATA_DIR', 'train_file': 'TRAIN_FILE', 'test_file': 'TEST_FILE',
        'n_train': 'N_TRAIN', 'n_test': 'N_TEST', 'max_iters': 'MAX_ITERS',
        'batch_size': 'BATCH_SIZE', 'eval_every': 'EVAL_EVERY',
        'n_layer': 'N_LAYER', 'n_head': 'N_HEAD', 'n_embd': 'N_EMBD',
        'dropout': 'DROPOUT', 'lr': 'LR', 'weight_decay': 'WEIGHT_DECAY',
        'puma_tau': 'PUMA_TAU',
        'puma_k_start': 'PUMA_K_START', 'puma_k_end': 'PUMA_K_END',
        'puma_k_step': 'PUMA_K_STEP', 'puma_k_every': 'PUMA_K_EVERY',
        'papl_tau': 'PAPL_TAU', 'papl_alpha': 'PAPL_ALPHA',
        'low_mult_sample_rate': 'LOW_MULT_SAMPLE_RATE', 'seed': 'SEED',
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


# Position types of the answer: plan (operands and operators, left of '='),
# calc (result digits, right of '='), sep ('=' and ',')
POS_PLAN = 'plan'
POS_CALC = 'calc'
POS_SEP = 'sep'


def classify_output_positions(output_str):
    types = []
    steps = output_str.split(',')
    for si, step in enumerate(steps):
        eq_pos = step.find('=')
        if eq_pos < 0:
            types.extend([POS_PLAN] * len(step))
        else:
            types.extend([POS_PLAN] * eq_pos)
            types.append(POS_SEP)
            types.extend([POS_CALC] * (len(step) - eq_pos - 1))
        if si < len(steps) - 1:
            types.append(POS_SEP)
    return types


# Solution multiplicity

def _combine_nums_values(a, b):
    a, b = int(a), int(b)
    out = [a + b, a * b]
    if a <= b:
        out.append(b - a)
        if a != 0 and b % a == 0:
            out.append(b // a)
    else:
        out.append(a - b)
        if b != 0 and a % b == 0:
            out.append(a // b)
    return out


def count_solutions(nums, target, limit=None):
    """Number of solution paths reaching the target (exhaustive DFS over pairs)."""
    def _rec(remaining):
        if len(remaining) == 1:
            return 1 if remaining[0] == target else 0
        total = 0
        for i, j in itertools.combinations(range(len(remaining)), 2):
            rem = [remaining[k] for k in range(len(remaining)) if k != i and k != j]
            for r in _combine_nums_values(remaining[i], remaining[j]):
                total += _rec(rem + [r])
                if limit is not None and total > limit:
                    return total
        return total
    return _rec(list(nums))


def _mult_bin(m):
    """Multiplicity bins (fewer solutions = more constrained instance)."""
    if m <= 3:
        return 'm=1-3'
    if m <= 10:
        return 'm=4-10'
    return 'm=11+'


MULT_BINS = ['m=1-3', 'm=4-10', 'm=11+']


def build_oracle_order_step_seq(output_str):
    """Step-sequential order: step 1 plan -> '=' -> calc, then ',' and step 2, ..."""
    types = classify_output_positions(output_str)
    order = []
    steps_raw = output_str.split(',')
    for si, step_str in enumerate(steps_raw):
        start = sum(len(s) + 1 for s in steps_raw[:si])
        step_len = len(step_str)
        step_range = list(range(start, start + step_len))
        step_types = types[start:start + step_len]
        plan_pos = [p for p, t in zip(step_range, step_types) if t == POS_PLAN]
        sep_pos = [p for p, t in zip(step_range, step_types) if t == POS_SEP]
        calc_pos = [p for p, t in zip(step_range, step_types) if t == POS_CALC]
        order.extend(plan_pos + sep_pos + calc_pos)
        if si < len(steps_raw) - 1:
            comma_pos = start + step_len
            order.append(comma_pos)
    return order


# Format & tokenizer
SEP_CHAR = '|'
EOS_CHAR = '$'
RAINBOW_CHARS = 'abcdefghijklmnop'
INPUT_PAD = '#'


def _rainbow_pad(output_str):
    remaining = MAX_ANS_LEN - len(output_str)
    if remaining <= 0:
        return output_str[:MAX_ANS_LEN]
    pad = EOS_CHAR
    for i in range(remaining - 1):
        pad += RAINBOW_CHARS[i % len(RAINBOW_CHARS)]
    return output_str + pad


def build_tok():
    chars = list('0123456789+-*/=,') + [SEP_CHAR, EOS_CHAR] + list(RAINBOW_CHARS)
    seen = set()
    unique = []
    for c in chars:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return CharTokenizer(unique, {'mask': 'M', 'pad': INPUT_PAD})


def _format_sample(input_str, output_str, compute_mult=False):
    sample = f"{input_str}{SEP_CHAR}{_rainbow_pad(output_str)}"
    if len(sample) > MAX_SEQ_LEN:
        return None, None

    nums_s = input_str.split(',')
    target = nums_s[-1] if nums_s else ''
    input_nums = [int(x) for x in nums_s[:-1]] if len(nums_s) > 1 else []
    meta = {'output_len': len(output_str), 'output_str': output_str}

    if compute_mult and input_nums and target:
        try:
            mult = count_solutions(input_nums, int(target))
            meta['solution_mult'] = mult
            meta['mult_bin'] = _mult_bin(mult)
        except (ValueError, TypeError):
            meta['solution_mult'] = -1
            meta['mult_bin'] = 'unknown'
    return sample, meta


def get_answer(s):
    return s.split(SEP_CHAR, 1)[1]


# Data loading

def load_jsonl(filepath, max_n=None):
    data = []
    with open(filepath) as f:
        for line in f:
            d = json.loads(line.strip())
            data.append(d)
            if max_n and len(data) >= max_n:
                break
    return data


def load_and_format(filepath, max_n=None, seed=42, compute_mult=False,
                    low_mult_sample_rate=1.0):
    """Load a jsonl file and format the samples; returns (samples, metas).

    compute_mult=True adds solution_mult / mult_bin to each meta.
    low_mult_sample_rate < 1.0 keeps only that fraction of the m=1-3 puzzles.
    """
    raw = load_jsonl(filepath, max_n)
    rng = random.Random(seed)
    rng.shuffle(raw)

    samples, metas = [], []
    skipped = 0
    low_mult_dropped = 0
    for d in raw:
        s, m = _format_sample(d['input'], d['output'], compute_mult=compute_mult)
        if s is None:
            skipped += 1
            continue
        if (low_mult_sample_rate < 1.0 and compute_mult
                and m.get('mult_bin') == 'm=1-3'
                and rng.random() >= low_mult_sample_rate):
            low_mult_dropped += 1
            continue
        samples.append(s)
        metas.append(m)
    if skipped:
        print(f"  Skipped {skipped} samples (too long)")
    if low_mult_dropped:
        print(f"  Sub-sampled m=1-3: dropped {low_mult_dropped}, kept "
              f"{sum(1 for m in metas if m.get('mult_bin') == 'm=1-3')} "
              f"(rate={low_mult_sample_rate})")
    return samples, metas


# Probe

@torch.no_grad()
def probe_per_position(model, tokenizer, test_samples, max_len, device=None):
    """Fully-masked probe: every answer position masked, one forward pass; mean
    NLL and argmax accuracy over the answer positions."""
    if device is None:
        device = DEVICE
    mask_id = tokenizer.special_ids['mask']
    pad_id = tokenizer.special_ids['pad']
    model.eval()

    ids_all, ans_all = encode_countdown_samples(test_samples, tokenizer, max_len)
    ids_all, ans_all = ids_all.to(device), ans_all.to(device)

    total_loss = 0.0
    total_count = 0
    total_correct = 0

    BS = 256
    for start in range(0, len(ids_all), BS):
        end = min(start + BS, len(ids_all))
        ids = ids_all[start:end]
        ans = ans_all[start:end]
        B = ids.shape[0]
        inp = ids.clone()
        _arange = torch.arange(MAX_ANS_LEN, device=ids.device)
        ans_pos = (ans.unsqueeze(1) + _arange).clamp(max=ids.shape[1] - 1)
        bi = torch.arange(B, device=ids.device).unsqueeze(1).expand_as(ans_pos)
        inp[bi, ans_pos] = mask_id
        logits = model(inp)

        for i in range(B):
            ans_start = ans[i].item()
            for j in range(MAX_ANS_LEN):
                pos = ans_start + j
                if pos >= ids.shape[1]:
                    break
                target = ids[i, pos].item()
                if target == pad_id:
                    continue
                total_correct += int(logits[i, pos].argmax().item() == target)
                loss = F.cross_entropy(logits[i, pos].unsqueeze(0),
                                       ids[i, pos].unsqueeze(0))
                total_loss += loss.item()
                total_count += 1

    return {'overall_loss': total_loss / max(total_count, 1),
            'overall_acc': total_correct / max(total_count, 1)}


# Decoding

@torch.no_grad()
def _generate_step_seq(model, prefix_ids, oracle_orders, n_tokens, mask_id,
                       pad_to=None, pad_id=None, device=None):
    """Greedy generation with a per-sample fixed order.
    oracle_orders[b]: answer-region indices in decode order (missing indices
    are appended in ascending order)."""
    if device is None:
        device = DEVICE
    model.eval()
    B = prefix_ids.shape[0]
    T_pre = prefix_ids.shape[1]
    T = T_pre + n_tokens

    x = torch.full((B, T), mask_id, dtype=torch.long, device=device)
    x[:, :T_pre] = prefix_ids.to(device)

    if pad_to is not None and pad_to > T:
        assert pad_id is not None
        pad_block = torch.full((B, pad_to - T), pad_id, dtype=torch.long, device=device)
        x = torch.cat([x, pad_block], dim=1)

    full_orders = []
    for b in range(B):
        seen = set(oracle_orders[b])
        ext = list(oracle_orders[b]) + [i for i in range(n_tokens) if i not in seen]
        full_orders.append(ext[:n_tokens])

    for t in range(n_tokens):
        logits = model(x)
        logits[:, :, mask_id] = -float('inf')
        pos = torch.tensor([T_pre + full_orders[b][t] for b in range(B)],
                           dtype=torch.long, device=device)
        batch_arange = torch.arange(B, device=device)
        tok = logits[batch_arange, pos].argmax(-1)
        x[batch_arange, pos] = tok
    return x


@torch.no_grad()
def gen_eval(model, tokenizer, test_samples, test_metas, max_len,
             decode_policy='confidence', device=None):
    """Exact-match accuracy of the padded answer, overall and by multiplicity
    bin. Samples are batched by prompt length."""
    if device is None:
        device = DEVICE
    mask_id = tokenizer.special_ids['mask']
    pad_id = tokenizer.special_ids['pad']
    model.eval()

    results = []
    groups = {}
    for idx, s in enumerate(test_samples):
        prefix = s.split(SEP_CHAR, 1)[0] + SEP_CHAR
        pl = len(tokenizer.encode(prefix))
        groups.setdefault(pl, []).append(idx)

    for pl, indices in groups.items():
        for bstart in range(0, len(indices), 128):
            bind = indices[bstart:bstart + 128]
            batch_s = [test_samples[i] for i in bind]
            batch_m = [test_metas[i] for i in bind]

            penc = [tokenizer.encode(s.split(SEP_CHAR, 1)[0] + SEP_CHAR)
                    for s in batch_s]
            pids = torch.tensor(penc, dtype=torch.long)

            if decode_policy == 'step_seq':
                oracle_orders = [build_oracle_order_step_seq(m['output_str'])
                                 for m in batch_m]
                gen = _generate_step_seq(
                    model, pids, oracle_orders, MAX_ANS_LEN, mask_id,
                    pad_to=max_len, pad_id=pad_id, device=device)
            else:
                gen, _, _ = generate_diffusion(
                    model, pids, MAX_ANS_LEN, mask_id, policy=decode_policy,
                    pad_to=max_len, pad_id=pad_id, device=device)
            pred_ids = gen[:, pl:pl + MAX_ANS_LEN]

            for bi in range(len(bind)):
                pred_str = tokenizer.decode(pred_ids[bi].cpu().tolist())
                results.append((pred_str == get_answer(batch_s[bi]),
                                batch_m[bi].get('mult_bin', 'unknown')))

    n_total = len(results)
    agg = {'accuracy': sum(ok for ok, _ in results) / max(n_total, 1), 'n': n_total}
    for mb in MULT_BINS:
        subset = [ok for ok, b in results if b == mb]
        if subset:
            agg[f'acc_{mb}'] = sum(subset) / len(subset)
            agg[f'n_{mb}'] = len(subset)
    return agg


@torch.no_grad()
def selective_reveal_eval(model, tokenizer, test_samples, test_metas, max_len,
                          reveal_frac=0.5, seed=0, device=None):
    """Reveal a random `reveal_frac` of the gold answer tokens (excluding the
    padding) and predict the remaining answer tokens in a single forward pass;
    token accuracy over the unrevealed answer tokens, overall and by
    multiplicity bin."""
    if device is None:
        device = DEVICE
    rng = random.Random(seed)
    mask_id = tokenizer.special_ids['mask']
    pad_id = tokenizer.special_ids['pad']
    model.eval()

    total_correct = 0; total = 0
    by_mult = defaultdict(lambda: [0, 0])

    groups = {}
    for idx, s in enumerate(test_samples):
        prefix = s.split(SEP_CHAR, 1)[0] + SEP_CHAR
        pl = len(tokenizer.encode(prefix))
        groups.setdefault(pl, []).append(idx)

    for pl, indices in groups.items():
        for bstart in range(0, len(indices), 128):
            bind = indices[bstart:bstart + 128]
            B = len(bind)
            batch_s = [test_samples[i] for i in bind]
            batch_m = [test_metas[i] for i in bind]

            seq_list = []; reveal_sets = []; aenc_list = []
            for s, m in zip(batch_s, batch_m):
                prefix = s.split(SEP_CHAR, 1)[0] + SEP_CHAR
                penc = tokenizer.encode(prefix)
                aenc = tokenizer.encode(get_answer(s))
                out_len = m.get('output_len', 0)
                n_reveal = int(out_len * reveal_frac)
                reveal_pos = set(rng.sample(range(out_len), n_reveal)) if out_len > 0 else set()
                seq = list(penc)
                for j in range(MAX_ANS_LEN):
                    seq.append(aenc[j] if j in reveal_pos else mask_id)
                if len(seq) < max_len:
                    seq = seq + [pad_id] * (max_len - len(seq))
                else:
                    seq = seq[:max_len]
                seq_list.append(seq)
                reveal_sets.append(reveal_pos)
                aenc_list.append(aenc)

            x = torch.tensor(seq_list, dtype=torch.long, device=device)
            logits = model(x)
            logits[:, :, mask_id] = -float('inf')
            preds = logits.argmax(dim=-1)

            for bi in range(B):
                meta = batch_m[bi]
                out_len = meta.get('output_len', 0)
                mb = meta.get('mult_bin', 'unknown')
                aenc = aenc_list[bi]
                reveal_pos = reveal_sets[bi]
                for j in range(out_len):
                    if j in reveal_pos:
                        continue
                    t_pos = pl + j
                    if t_pos >= preds.shape[1]:
                        break
                    ok = (preds[bi, t_pos].item() == aenc[j])
                    total_correct += int(ok); total += 1
                    by_mult[mb][0] += int(ok); by_mult[mb][1] += 1

    return {
        'reveal_frac': reveal_frac,
        'accuracy': total_correct / max(total, 1),
        'n_predictions': total,
        'by_mult_bin': {mb: {'acc': c / max(n, 1), 'n': n}
                        for mb, (c, n) in by_mult.items() if n > 0},
    }


# Training

def train_model(mask_type, tokenizer, train_samples, probe_samples, max_len, device=None):
    """train_diffusion with the fully-masked probe on probe_samples as eval callback."""
    if device is None:
        device = DEVICE
    max_iters = MAX_ITERS

    train_ids, train_ans = encode_countdown_samples(train_samples, tokenizer, max_len)
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
    torch.manual_seed(SEED)
    random.seed(SEED)
    prepare_results_dir()

    exp_name = f"{EXP_NAME}_{tag}" if tag else EXP_NAME
    print(f"\n{'='*70}")
    print(f"  {exp_name}")
    print(f"  Model: {N_LAYER}L/{N_EMBD}D/{N_HEAD}H, ANS_LEN={MAX_ANS_LEN}")
    print(f"  Masks: {MASK_TYPES}, Decode: {DECODE_POLICIES}")
    print(f"{'='*70}")

    train_path = os.path.join(DATA_DIR, TRAIN_FILE)
    test_path = os.path.join(DATA_DIR, TEST_FILE)

    t0 = time.time()
    # m=1-3 puzzles sub-sampled at LOW_MULT_SAMPLE_RATE
    train_samples, train_metas = load_and_format(
        train_path, max_n=N_TRAIN, seed=SEED, compute_mult=True,
        low_mult_sample_rate=LOW_MULT_SAMPLE_RATE)
    print(f"  {len(train_samples)} training samples ({time.time()-t0:.1f}s)")
    # test set at the natural distribution
    test_samples, test_metas = load_and_format(
        test_path, max_n=N_TEST, seed=SEED + 1, compute_mult=True,
        low_mult_sample_rate=1.0)
    print(f"  {len(test_samples)} test samples")
    for name, metas in [('train', train_metas), ('test', test_metas)]:
        cnt = Counter(m.get('mult_bin', 'unknown') for m in metas)
        print(f"  Mult bin ({name}): " +
              ', '.join(f"{mb} {cnt.get(mb, 0)} ({100*cnt.get(mb, 0)/max(len(metas), 1):.1f}%)"
                        for mb in MULT_BINS))

    tokenizer = build_tok()
    max_len = MAX_SEQ_LEN

    all_dyn, all_final = {}, {}
    for mask_type in MASK_TYPES:
        print(f"\n{'-'*60}\n  Training: {mask_type}\n{'-'*60}")
        # the fully-masked probe runs on the test set
        model, dynamics = train_model(mask_type, tokenizer, train_samples, test_samples,
                                      max_len, device=DEVICE)
        all_dyn[mask_type] = dynamics
        save_checkpoint(exp_name, {k: v.cpu().clone() for k, v in model.state_dict().items()},
                        tag=mask_type)

        for dp in DECODE_POLICIES:
            r = gen_eval(model, tokenizer, test_samples, test_metas, max_len, dp, device=DEVICE)
            all_final[f"{mask_type}_{dp}"] = r
            print(f"  {dp}: exact={r['accuracy']:.3f} | " +
                  ' '.join(f"{mb}={r[f'acc_{mb}']:.3f}(n={r[f'n_{mb}']})"
                           for mb in MULT_BINS if f'acc_{mb}' in r))

        frac = SELECTIVE_REVEAL_FRAC
        r = selective_reveal_eval(model, tokenizer, test_samples, test_metas, max_len,
                                  reveal_frac=frac, seed=SEED + int(frac * 100), device=DEVICE)
        all_final[f"{mask_type}_selective_{int(frac*100)}"] = r
        print(f"  selective reveal {int(frac*100)}%: token acc={r['accuracy']:.3f} | " +
              ' '.join(f"{mb}={v['acc']:.3f}" for mb, v in sorted(r['by_mult_bin'].items())))

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n{'='*70}\n  SUMMARY (exact match; selective reveal: token accuracy)\n{'='*70}")
    print(f"  {'Test':<28s}" + ''.join(f" {mt:>12s}" for mt in MASK_TYPES))
    for dp in DECODE_POLICIES:
        for test_name in ['accuracy', 'acc_m=1-3', 'acc_m=4-10', 'acc_m=11+']:
            vals = [all_final.get(f'{mt}_{dp}', {}).get(test_name) for mt in MASK_TYPES]
            if any(v is not None for v in vals):
                print(f"  {dp + ' ' + test_name:<28s}" +
                      ''.join(f" {v:>12.4f}" if v is not None else f" {'N/A':>12s}" for v in vals))
    pct = int(SELECTIVE_REVEAL_FRAC * 100)
    for mb in ['overall'] + MULT_BINS:
        vals = []
        for mt in MASK_TYPES:
            r = all_final.get(f'{mt}_selective_{pct}', {})
            vals.append(r.get('accuracy') if mb == 'overall' else r.get('by_mult_bin', {}).get(mb, {}).get('acc'))
        print(f"  {f'reveal {pct}% ' + mb:<28s}" +
              ''.join(f" {v:>12.4f}" if v is not None else f" {'N/A':>12s}" for v in vals))

    sd = {'config': {k: globals()[k] for k in [
        'MAX_ANS_LEN', 'MAX_SEQ_LEN', 'N_LAYER', 'N_HEAD', 'N_EMBD', 'N_TRAIN', 'N_TEST',
        'LOW_MULT_SAMPLE_RATE', 'MASK_TYPES', 'DECODE_POLICIES', 'MAX_ITERS', 'BATCH_SIZE',
        'PUMA_K_START', 'PUMA_K_END', 'PAPL_TAU', 'PAPL_ALPHA', 'SEED',
        'SELECTIVE_REVEAL_FRAC']}}
    for k, v in all_dyn.items():
        sd[f'dyn_{k}'] = {'checkpoints': v['checkpoints'], 'train_loss': v['train_loss']}
    for k, v in all_final.items():
        sd[f'final_{k}'] = v
    save_results(exp_name, sd)
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
