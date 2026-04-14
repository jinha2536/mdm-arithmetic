"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Countdown (CD4) — Planning Depth + PUMA Coverage Deficit
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Task:     Given 5 numbers (last = target), produce a 3-step equation
            chain reaching the target using +, -, *, /.
  Example:  input  = "86,28,13,31,96"
            output = "86+28=114,31-13=18,114-18=96"

  Position types in output:
    Plan:  operands & operators (left side of '=')  — requires planning
    Calc:  result digits (right side of '=')         — locally computable
    Sep:   '=' and ','                              — structural

  Rarity axis:     chain depth (how many steps reuse previous results)
  Dependency:      plan positions require multi-step lookahead;
                   calc positions are stepping stones
  Oracle:          step_sequential = complete step 1 (plan→calc) → step 2 → ...
  Corner case:     Game of 24 (fixed target, rare planning patterns)

  Key question: Does PUMA's curriculum (unmask calc first, then plan)
                align with the correct reasoning order?
                → Concordance measurement is central.

  Training:  iter-based with EMA (following ListOps pattern)
  Decode:    confidence | step_seq (oracle) | calc_first | random
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import sys, os, time, math, json, random, re, statistics
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
    mount_drive, save_results, save_checkpoint,
    train_diffusion, puma_k_fixed, puma_k_linear, puma_k_step,
    generate_diffusion, DEVICE,
)


def encode_countdown_samples(samples, tokenizer, max_len=None):
    """encode_samples variant that uses '|' separator instead of '='."""
    encoded = [tokenizer.encode(s) for s in samples]
    if max_len is None:
        max_len = max(len(e) for e in encoded)
    pad_id = tokenizer.special_ids['pad']
    ids = torch.full((len(encoded), max_len), pad_id, dtype=torch.long)
    ans_starts = torch.zeros(len(encoded), dtype=torch.long)
    sep_char = SEP_CHAR  # '|'
    for i, enc in enumerate(encoded):
        L = min(len(enc), max_len)
        ids[i, :L] = torch.tensor(enc[:L])
        ans_starts[i] = samples[i].index(sep_char) + 1
    return ids, ans_starts

EXP_NAME = 'exp_countdown'

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Config
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MAX_ANS_LEN = 40        # padded answer length (max output ~34, pad to 40)
MAX_SEQ_LEN = 60        # total sequence cap (input ~14 + | + ans 40 ≈ 55)

# Training
N_TRAIN = None          # None = use all available data
N_TEST = 1000; BATCH_SIZE = 256
MAX_ITERS = 400000; EVAL_EVERY = 5000; LOG_EVERY = 1000
GEN_EVAL_EVERY = 10000; GEN_EVAL_N = 500
MASK_TYPES = ['random', 'puma']
DECODE_POLICIES = ['confidence', 'l2r', 'random']  # l2r = step_seq oracle (data in step order)

# Model (He et al. tiny config)
N_LAYER = 3; N_HEAD = 12; N_EMBD = 384; DROPOUT = 0.1; POS_ENC = 'absolute'
LR = 1e-3; MIN_LR = 1e-4; WARMUP_ITERS = 2000; GRAD_CLIP = 1.0
WEIGHT_DECAY = 0.1; EMA_DECAY = 0.9999
PUMA_TAU = 0.9; PUMA_K = 8
PUMA_K_START = None; PUMA_K_END = None
PUMA_K_STEP = 3; PUMA_K_EVERY = None
SEED = 42
NO_AMP = False
PATIENCE = 50000
CONTINUATION_ITERS = 10000

# Data paths
DATA_DIR = 'experiments/data'
TRAIN_FILE = 'cd4_train.jsonl'
TEST_FILE = 'cd4_test.jsonl'
CORNER_FILE = 'cd4_tot24.jsonl'     # Game of 24


def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', type=str)
    p.add_argument('--train-file', type=str); p.add_argument('--test-file', type=str)
    p.add_argument('--corner-file', type=str)
    p.add_argument('--max-ans-len', type=int)
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
        'data_dir': 'DATA_DIR', 'train_file': 'TRAIN_FILE',
        'test_file': 'TEST_FILE', 'corner_file': 'CORNER_FILE',
        'max_ans_len': 'MAX_ANS_LEN',
        'n_train': 'N_TRAIN', 'n_test': 'N_TEST', 'max_iters': 'MAX_ITERS',
        'batch_size': 'BATCH_SIZE', 'eval_every': 'EVAL_EVERY',
        'gen_eval_every': 'GEN_EVAL_EVERY',
        'n_layer': 'N_LAYER', 'n_head': 'N_HEAD', 'n_embd': 'N_EMBD',
        'dropout': 'DROPOUT', 'lr': 'LR', 'weight_decay': 'WEIGHT_DECAY',
        'patience': 'PATIENCE', 'puma_tau': 'PUMA_TAU', 'puma_k': 'PUMA_K',
        'puma_k_start': 'PUMA_K_START', 'puma_k_end': 'PUMA_K_END',
        'puma_k_step': 'PUMA_K_STEP', 'puma_k_every': 'PUMA_K_EVERY',
        'seed': 'SEED', 'no_amp': 'NO_AMP',
        'continuation_iters': 'CONTINUATION_ITERS',
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
# Position type classification
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Types: plan (operands/operators), calc (results), sep (=,)

POS_PLAN = 'plan'
POS_CALC = 'calc'
POS_SEP = 'sep'
POS_PAD = 'pad'
POS_TYPES = [POS_PLAN, POS_CALC, POS_SEP, POS_PAD]
POS_TO_ID = {n: i for i, n in enumerate(POS_TYPES)}


def classify_output_positions(output_str):
    """Classify each character in output as plan/calc/sep.

    Output format: "86+28=114,31-13=18,114-18=96"
    Plan: left side of '=' (operands + operator)
    Calc: right side of '=' (result digits)
    Sep:  '=' and ','

    Returns list of POS_* for each character.
    """
    types = []
    steps = output_str.split(',')
    for si, step in enumerate(steps):
        eq_pos = step.find('=')
        if eq_pos < 0:
            # Malformed — treat all as plan
            types.extend([POS_PLAN] * len(step))
        else:
            # Left of '=' = plan
            types.extend([POS_PLAN] * eq_pos)
            # '=' itself = sep
            types.append(POS_SEP)
            # Right of '=' = calc
            types.extend([POS_CALC] * (len(step) - eq_pos - 1))
        # ',' between steps = sep
        if si < len(steps) - 1:
            types.append(POS_SEP)
    return types


def classify_chain_depth(output_str):
    """Compute chain depth: how many steps reuse a previous result.

    Returns (chain_depth, is_full_chain, step_details).
    chain_depth: 0 = fully parallel, len(steps)-1 = fully sequential
    """
    steps = output_str.split(',')
    results = []
    reuse_count = 0
    step_details = []
    for i, step in enumerate(steps):
        eq_pos = step.find('=')
        if eq_pos < 0:
            step_details.append({'step': i, 'reuses_prev': False})
            continue
        left = step[:eq_pos]
        right = step[eq_pos + 1:]
        reuses = any(r in left for r in results)
        if reuses:
            reuse_count += 1
        step_details.append({
            'step': i, 'left': left, 'right': right,
            'reuses_prev': reuses,
        })
        results.append(right)
    is_full_chain = (reuse_count == len(steps) - 1) if len(steps) > 1 else True
    return reuse_count, is_full_chain, step_details


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Oracle decode orders
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_oracle_order_step_seq(output_str):
    """Step-sequential oracle: complete step 1 (plan→calc), then step 2, etc.

    Within each step: plan positions first, then sep (=), then calc.
    This is the natural reasoning order: decide what to compute,
    then compute it.

    Returns list of answer-region indices in decode order.
    """
    types = classify_output_positions(output_str)
    n = len(types)

    # Group positions by step
    steps_pos = []  # list of lists of (pos, type)
    cur_step = []
    step_idx = 0
    pos = 0
    steps_raw = output_str.split(',')
    for si, step_str in enumerate(steps_raw):
        step_positions = list(range(pos, pos + len(step_str)))
        steps_pos.append(step_positions)
        pos += len(step_str)
        if si < len(steps_raw) - 1:
            pos += 1  # comma

    order = []
    pos = 0
    for si, step_str in enumerate(steps_raw):
        # Positions for this step
        start = sum(len(s) + 1 for s in steps_raw[:si])  # +1 for commas
        step_len = len(step_str)
        step_range = list(range(start, start + step_len))
        step_types = types[start:start + step_len]

        # Plan first, then sep, then calc
        plan_pos = [p for p, t in zip(step_range, step_types) if t == POS_PLAN]
        sep_pos = [p for p, t in zip(step_range, step_types) if t == POS_SEP]
        calc_pos = [p for p, t in zip(step_range, step_types) if t == POS_CALC]
        order.extend(plan_pos + sep_pos + calc_pos)

        # Comma after step (if not last)
        if si < len(steps_raw) - 1:
            comma_pos = start + step_len
            order.append(comma_pos)

    return order


def build_oracle_order_calc_first(output_str):
    """Calc-first oracle: all calc/sep positions first, then all plan positions.

    This mimics what PUMA's confidence order would naturally do.
    High concordance with PUMA = stepping stone hypothesis confirmed.
    """
    types = classify_output_positions(output_str)
    calc_sep = [i for i, t in enumerate(types) if t in (POS_CALC, POS_SEP)]
    plan = [i for i, t in enumerate(types) if t == POS_PLAN]
    return calc_sep + plan


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Data formatting & tokenizer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SEP_CHAR = '|'          # separates input from output
EOS_CHAR = '$'          # marks end of output
RAINBOW_CHARS = 'abcdefghijklmnop'  # 16 distinct pad tokens
INPUT_PAD = '#'         # pad for sequences shorter than MAX_SEQ_LEN


def _rainbow_pad(output_str):
    """Pad output to MAX_ANS_LEN with EOS + cyclic rainbow tokens."""
    remaining = MAX_ANS_LEN - len(output_str)
    if remaining <= 0:
        return output_str[:MAX_ANS_LEN]
    pad = EOS_CHAR
    for i in range(remaining - 1):
        pad += RAINBOW_CHARS[i % len(RAINBOW_CHARS)]
    return output_str + pad


def build_tok():
    """Build tokenizer with all chars that appear in data."""
    chars = list('0123456789+-*/=,') + [SEP_CHAR, EOS_CHAR] + list(RAINBOW_CHARS)
    seen = set()
    unique = []
    for c in chars:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return CharTokenizer(unique, {'mask': 'M', 'pad': INPUT_PAD})


def _format_sample(input_str, output_str):
    """Format as 'input|output_rainbow_padded'.

    Returns (formatted_string, metadata) or (None, None) if too long.
    """
    padded_output = _rainbow_pad(output_str)
    sample = f"{input_str}{SEP_CHAR}{padded_output}"
    if len(sample) > MAX_SEQ_LEN:
        return None, None

    # Position types for output region
    pos_types = classify_output_positions(output_str)
    # Extend with PAD types for rainbow padding
    pos_types.extend([POS_PAD] * (MAX_ANS_LEN - len(output_str)))

    # Chain analysis
    chain_depth, is_full_chain, step_details = classify_chain_depth(output_str)

    # Parse target
    nums = input_str.split(',')
    target = nums[-1] if nums else ''

    meta = {
        'output_len': len(output_str),
        'chain_depth': chain_depth,
        'is_full_chain': is_full_chain,
        'n_steps': len(output_str.split(',')),
        'step_details': step_details,
        'pos_types': pos_types,             # per-answer-position type
        'target': target,
        'input_str': input_str,
        'output_str': output_str,
    }
    return sample, meta


def get_answer(s):
    """Get answer string (padded output) from formatted sample."""
    return s.split(SEP_CHAR, 1)[1]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Data loading
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_jsonl(filepath, max_n=None):
    """Load data from jsonl file."""
    data = []
    with open(filepath) as f:
        for line in f:
            d = json.loads(line.strip())
            data.append(d)
            if max_n and len(data) >= max_n:
                break
    return data


def load_and_format(filepath, max_n=None, seed=42):
    """Load jsonl, format samples, return (samples, metas)."""
    raw = load_jsonl(filepath, max_n)
    rng = random.Random(seed)
    rng.shuffle(raw)

    samples, metas = [], []
    skipped = 0
    for d in raw:
        s, m = _format_sample(d['input'], d['output'])
        if s is not None:
            samples.append(s)
            metas.append(m)
        else:
            skipped += 1
    if skipped:
        print(f"  Skipped {skipped} samples (too long)")
    return samples, metas


def load_corner_cases(filepath, seed=42):
    """Load Game of 24 corner cases.

    These have output='=24', meaning the model must generate
    the full equation chain. We format them with masked output.
    """
    raw = load_jsonl(filepath)
    samples, metas = [], []
    for d in raw:
        # For Game of 24, output is just "=24" — not a full equation
        # We need the model to generate the equations
        # Format: input|MASK...  (all answer positions masked)
        meta = {
            'output_len': 0,    # unknown — model must generate
            'chain_depth': -1,
            'is_full_chain': False,
            'n_steps': -1,
            'target': '24',
            'input_str': d['input'],
            'output_str': d['output'],
            'is_corner_case': True,
        }
        # For generation eval, we only need the input prefix
        samples.append(d['input'])
        metas.append(meta)
    return samples, metas


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Evaluation: equation validation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def check_equation(eq_str):
    """Check if a sub-equation like '86+28=114' is arithmetically correct."""
    try:
        left, right = eq_str.split('=')
        left_match = re.match(r'^(\d+)([+\-*/])(\d+)$', left)
        if not left_match:
            return False
        a, op, b = left_match.group(1), left_match.group(2), left_match.group(3)
        a, b = int(a), int(b)
        if op == '+':
            result = a + b
        elif op == '-':
            result = a - b
        elif op == '*':
            result = a * b
        elif op == '/':
            if b == 0:
                return False
            if a % b != 0:
                return False
            result = a // b
        else:
            return False
        return result == int(right)
    except:
        return False


def validate_countdown(pred_output, target):
    """Validate a predicted output string for Countdown.

    Returns dict with: valid (bool), n_correct_eqs, n_total_eqs,
    reaches_target, equation_details.
    """
    # Strip EOS and rainbow padding
    clean = pred_output.split(EOS_CHAR)[0] if EOS_CHAR in pred_output else pred_output
    # Also strip any rainbow chars
    for c in RAINBOW_CHARS:
        clean = clean.replace(c, '')

    steps = clean.split(',')
    n_correct = 0
    details = []
    all_correct = True
    for step in steps:
        ok = check_equation(step)
        details.append({'eq': step, 'correct': ok})
        if ok:
            n_correct += 1
        else:
            all_correct = False

    # Check target
    reaches_target = False
    if steps and '=' in steps[-1]:
        final_result = steps[-1].split('=')[-1]
        reaches_target = (final_result == str(target))

    return {
        'valid': all_correct and reaches_target,
        'n_correct_eqs': n_correct,
        'n_total_eqs': len(steps),
        'reaches_target': reaches_target,
        'all_eqs_correct': all_correct,
        'details': details,
        'clean_output': clean,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Probe: per-position classification accuracy
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@torch.no_grad()
def probe_per_position(model, tokenizer, test_samples, test_metas,
                       max_len, device=None):
    """Evaluate per-position accuracy stratified by position type."""
    if device is None:
        device = DEVICE
    mask_id = tokenizer.special_ids['mask']
    pad_id = tokenizer.special_ids['pad']
    model.eval()

    ids_all, ans_all = encode_countdown_samples(test_samples, tokenizer, max_len)
    ids_all, ans_all = ids_all.to(device), ans_all.to(device)

    total_loss = 0.0
    total_count = 0
    type_correct = defaultdict(int)
    type_total = defaultdict(int)
    chain_correct = defaultdict(int)
    chain_total = defaultdict(int)

    BS = 256
    for start in range(0, len(ids_all), BS):
        end = min(start + BS, len(ids_all))
        ids = ids_all[start:end]
        ans = ans_all[start:end]
        B = ids.shape[0]

        # Create masked input: mask answer region
        sep_id = tokenizer.char_to_id[SEP_CHAR]
        inp = ids.clone()
        for i in range(B):
            sep_pos = (ids[i] == sep_id).nonzero(as_tuple=True)[0]
            if len(sep_pos) > 0:
                ans_start = sep_pos[0].item() + 1
                inp[i, ans_start:ans_start + MAX_ANS_LEN] = mask_id

        logits = model(inp)

        for i in range(B):
            sep_pos = (ids[i] == sep_id).nonzero(as_tuple=True)[0]
            if len(sep_pos) == 0:
                continue
            ans_start = sep_pos[0].item() + 1
            meta = test_metas[start + i]
            pos_types = meta.get('pos_types', [])
            chain_depth = meta.get('chain_depth', 0)

            for j in range(min(MAX_ANS_LEN, len(pos_types))):
                pos = ans_start + j
                if pos >= ids.shape[1]:
                    break
                target = ids[i, pos].item()
                if target == pad_id:
                    continue
                pred = logits[i, pos].argmax().item()
                correct = (pred == target)

                ptype = pos_types[j] if j < len(pos_types) else POS_PAD
                type_correct[ptype] += int(correct)
                type_total[ptype] += 1

                # Chain depth stratification
                cd_key = f"chain_{chain_depth}"
                chain_correct[cd_key] += int(correct)
                chain_total[cd_key] += 1

                # Loss
                loss = F.cross_entropy(logits[i, pos].unsqueeze(0),
                                       ids[i, pos].unsqueeze(0))
                total_loss += loss.item()
                total_count += 1

    result = {
        'overall_loss': total_loss / max(total_count, 1),
        'overall_acc': sum(type_correct.values()) / max(sum(type_total.values()), 1),
    }
    # Per-type accuracy
    for ptype in POS_TYPES:
        if type_total[ptype] > 0:
            result[f'acc_{ptype}'] = type_correct[ptype] / type_total[ptype]
            result[f'n_{ptype}'] = type_total[ptype]
    # Per-chain-depth accuracy
    for key in sorted(chain_total.keys()):
        if chain_total[key] > 0:
            result[f'acc_{key}'] = chain_correct[key] / chain_total[key]
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Generation evaluation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@torch.no_grad()
def gen_eval(model, tokenizer, test_samples, test_metas, max_len,
             decode_policy='confidence', n=None, device=None):
    """Full generation evaluation with validation."""
    if device is None:
        device = DEVICE
    if n is not None:
        test_samples = test_samples[:n]
        test_metas = test_metas[:n]

    mask_id = tokenizer.special_ids['mask']
    pad_id = tokenizer.special_ids['pad']
    sep_id = tokenizer.char_to_id[SEP_CHAR]
    model.eval()

    results = []

    # Group by prefix length (ListOps pattern)
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

            # Encode prefixes
            penc = [tokenizer.encode(s.split(SEP_CHAR, 1)[0] + SEP_CHAR)
                    for s in batch_s]
            pids = torch.tensor(penc, dtype=torch.long)

            # Build oracle order if needed
            gen, _, info = generate_diffusion(
                model, pids, MAX_ANS_LEN, mask_id,
                policy=decode_policy, greedy=True,
                pad_to=max_len, pad_id=pad_id, device=device)

            pred_ids = gen[:, pl:pl + MAX_ANS_LEN]

            for bi in range(B):
                meta = batch_m[bi]
                pred_str = tokenizer.decode(pred_ids[bi].cpu().tolist())
                gold_str = get_answer(batch_s[bi])

                # Exact match
                exact = (pred_str == gold_str)

                # Validate as countdown
                target = meta.get('target', '')
                validation = validate_countdown(pred_str, target)

                # Per-position-type accuracy
                pos_types = meta.get('pos_types', [])
                type_corr = defaultdict(int)
                type_tot = defaultdict(int)
                for j in range(min(len(pred_str), len(gold_str), len(pos_types))):
                    ptype = pos_types[j]
                    type_tot[ptype] += 1
                    if pred_str[j] == gold_str[j]:
                        type_corr[ptype] += 1

                # Decode order for concordance
                decode_order = info.get('orders')
                concordance_step_seq = None
                concordance_calc_first = None
                if decode_order is not None:
                    ans_start = pl
                    raw_order = decode_order[bi].tolist() if hasattr(decode_order[bi], 'tolist') else decode_order[bi]
                    decode_step = {}
                    for step, abs_pos in enumerate(raw_order):
                        ans_pos = abs_pos - ans_start
                        if 0 <= ans_pos < MAX_ANS_LEN:
                            decode_step[ans_pos] = step

                    # Concordance with step_seq oracle
                    oracle_ss = build_oracle_order_step_seq(meta['output_str'])
                    concordance_step_seq = _rank_concordance(
                        decode_step, oracle_ss, MAX_ANS_LEN)

                    # Concordance with calc_first oracle
                    oracle_cf = build_oracle_order_calc_first(meta['output_str'])
                    concordance_calc_first = _rank_concordance(
                        decode_step, oracle_cf, MAX_ANS_LEN)

                results.append({
                    'exact': exact,
                    'valid': validation['valid'],
                    'all_eqs_correct': validation['all_eqs_correct'],
                    'reaches_target': validation['reaches_target'],
                    'n_correct_eqs': validation['n_correct_eqs'],
                    'chain_depth': meta.get('chain_depth', -1),
                    'is_full_chain': meta.get('is_full_chain', False),
                    'type_acc': {t: type_corr[t] / type_tot[t]
                                 for t in type_tot if type_tot[t] > 0},
                    'concordance_l2r': concordance_step_seq,
                    'concordance_calc_first': concordance_calc_first,
                    'pred': pred_str[:40],
                    'gold': gold_str[:40],
                })

    # Aggregate
    n_total = len(results)
    agg = {
        'accuracy': sum(r['exact'] for r in results) / max(n_total, 1),
        'valid_rate': sum(r['valid'] for r in results) / max(n_total, 1),
        'eq_correct_rate': sum(r['all_eqs_correct'] for r in results) / max(n_total, 1),
        'target_rate': sum(r['reaches_target'] for r in results) / max(n_total, 1),
        'n': n_total,
    }

    # Per-type accuracy
    for ptype in [POS_PLAN, POS_CALC, POS_SEP]:
        vals = [r['type_acc'].get(ptype, None) for r in results
                if ptype in r['type_acc']]
        if vals:
            agg[f'acc_{ptype}'] = sum(vals) / len(vals)

    # Per-chain-depth accuracy
    for cd in sorted(set(r['chain_depth'] for r in results)):
        subset = [r for r in results if r['chain_depth'] == cd]
        if subset:
            agg[f'acc_chain_{cd}'] = sum(r['exact'] for r in subset) / len(subset)
            agg[f'valid_chain_{cd}'] = sum(r['valid'] for r in subset) / len(subset)
            agg[f'n_chain_{cd}'] = len(subset)

    # Full chain vs mixed
    chain_sub = [r for r in results if r['is_full_chain']]
    mixed_sub = [r for r in results if not r['is_full_chain']]
    if chain_sub:
        agg['acc_full_chain'] = sum(r['exact'] for r in chain_sub) / len(chain_sub)
        agg['valid_full_chain'] = sum(r['valid'] for r in chain_sub) / len(chain_sub)
        agg['n_full_chain'] = len(chain_sub)
    if mixed_sub:
        agg['acc_mixed'] = sum(r['exact'] for r in mixed_sub) / len(mixed_sub)
        agg['valid_mixed'] = sum(r['valid'] for r in mixed_sub) / len(mixed_sub)
        agg['n_mixed'] = len(mixed_sub)

    # Concordance
    conc_ss = [r['concordance_l2r'] for r in results
               if r['concordance_l2r'] is not None]
    conc_cf = [r['concordance_calc_first'] for r in results
               if r['concordance_calc_first'] is not None]
    if conc_ss:
        agg['concordance_l2r'] = sum(conc_ss) / len(conc_ss)
    if conc_cf:
        agg['concordance_calc_first'] = sum(conc_cf) / len(conc_cf)

    # Example outputs
    agg['examples'] = results[:5]
    return agg


def _rank_concordance(decode_step_map, oracle_order, n):
    """Compute concordance (Kendall's tau-like) between decode order and oracle.

    decode_step_map: {ans_pos: step_number} from confidence decode
    oracle_order: list of ans_pos in oracle decode order
    Returns concordance in [0, 1], where 1 = perfect agreement.
    """
    # Build rank arrays
    oracle_rank = {}
    for rank, pos in enumerate(oracle_order):
        if pos < n:
            oracle_rank[pos] = rank

    concordant = 0
    discordant = 0
    positions = [p for p in range(n) if p in decode_step_map and p in oracle_rank]
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            pi, pj = positions[i], positions[j]
            d_diff = decode_step_map[pi] - decode_step_map[pj]
            o_diff = oracle_rank[pi] - oracle_rank[pj]
            if d_diff * o_diff > 0:
                concordant += 1
            elif d_diff * o_diff < 0:
                discordant += 1
    total = concordant + discordant
    if total == 0:
        return 0.5
    return concordant / total


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Training wrapper
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def train_model(mask_type, tokenizer, train_samples, test_samples, test_metas,
                max_len, max_iters=None, init_state=None, device=None):
    """Wrapper around train_diffusion for Countdown experiment."""
    if device is None:
        device = DEVICE
    if max_iters is None:
        max_iters = MAX_ITERS

    train_ids, train_ans = encode_countdown_samples(train_samples, tokenizer, max_len)
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
        parts = []
        for pt in [POS_PLAN, POS_CALC, POS_SEP]:
            key = f'acc_{pt}'
            if key in probe:
                parts.append(f"{pt}={probe[key]:.3f}")
        print(f"    [eval it {it}] loss={probe['overall_loss']:.4f} "
              f"acc={probe['overall_acc']:.4f} {' '.join(parts)}")

        if it > 0 and it % GEN_EVAL_EVERY == 0:
            r = gen_eval(model, tokenizer, test_samples, test_metas,
                         max_len, 'confidence', n=GEN_EVAL_N, device=device)
            print(f"      [gen] exact={r['accuracy']:.3f} valid={r['valid_rate']:.3f} "
                  f"eqs_ok={r['eq_correct_rate']:.3f}")
            if 'concordance_calc_first' in r:
                print(f"      [conc] step_seq={r.get('concordance_l2r', '?'):.3f} "
                      f"calc_first={r.get('concordance_calc_first', '?'):.3f}")
            for ex in r.get('examples', [])[:3]:
                print(f"        gold={ex['gold'][:35]}  pred={ex['pred'][:35]}  "
                      f"{'✓' if ex['exact'] else '✗'}")
            probe['gen_accuracy'] = r['accuracy']
            probe['gen_valid'] = r['valid_rate']
            probe['gen_concordance_ss'] = r.get('concordance_l2r')
            probe['gen_concordance_cf'] = r.get('concordance_calc_first')
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
# Main run
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run(tag=''):
    torch.manual_seed(SEED)
    random.seed(SEED)

    exp_name = f"{EXP_NAME}_{tag}" if tag else EXP_NAME
    print(f"\n{'='*70}")
    print(f"  {exp_name}")
    print(f"  Model: {N_LAYER}L/{N_EMBD}D/{N_HEAD}H, ANS_LEN={MAX_ANS_LEN}")
    print(f"  Masks: {MASK_TYPES}, Decode: {DECODE_POLICIES}")
    print(f"{'='*70}")

    # Load data
    train_path = os.path.join(DATA_DIR, TRAIN_FILE)
    test_path = os.path.join(DATA_DIR, TEST_FILE)
    corner_path = os.path.join(DATA_DIR, CORNER_FILE)

    print(f"\nLoading train data from {train_path}...")
    train_samples, train_metas = load_and_format(train_path, max_n=N_TRAIN, seed=SEED)
    print(f"  {len(train_samples)} training samples")

    print(f"Loading test data from {test_path}...")
    test_samples, test_metas = load_and_format(test_path, max_n=N_TEST, seed=SEED + 1)
    print(f"  {len(test_samples)} test samples")

    # Data stats
    chain_dist = defaultdict(int)
    for m in train_metas:
        chain_dist[m['chain_depth']] += 1
    print(f"  Chain depth distribution (train): {dict(sorted(chain_dist.items()))}")

    full_chain_pct = sum(1 for m in train_metas if m['is_full_chain']) / len(train_metas)
    print(f"  Full chain: {full_chain_pct*100:.1f}%")

    # Corner cases
    corner_samples, corner_metas = None, None
    if os.path.exists(corner_path):
        corner_samples, corner_metas = load_corner_cases(corner_path)
        print(f"  Corner cases (Game of 24): {len(corner_samples)} samples")

    # Tokenizer
    tokenizer = build_tok()
    max_len = MAX_SEQ_LEN
    print(f"  Vocab size: {tokenizer.vocab_size}, Max seq len: {max_len}")

    # Train
    all_dyn, all_final = {}, {}
    models = {}

    for mask_type in MASK_TYPES:
        print(f"\n{'─'*60}")
        print(f"  Training: {mask_type}")
        print(f"{'─'*60}")

        model, dynamics = train_model(
            mask_type, tokenizer, train_samples, test_samples, test_metas,
            max_len, device=DEVICE)
        models[mask_type] = model
        all_dyn[mask_type] = dynamics

        # Final evaluation across all decode policies
        for dp in DECODE_POLICIES:
            print(f"\n  Evaluating: {mask_type} × {dp}")
            r = gen_eval(model, tokenizer, test_samples, test_metas,
                         max_len, dp, device=DEVICE)
            key = f"{mask_type}_{dp}"
            all_final[key] = r
            print(f"    exact={r['accuracy']:.3f} valid={r['valid_rate']:.3f}")
            print(f"    plan={r.get('acc_plan', '?'):.3f} "
                  f"calc={r.get('acc_calc', '?'):.3f}")
            if 'acc_full_chain' in r:
                print(f"    full_chain={r['acc_full_chain']:.3f} "
                      f"mixed={r.get('acc_mixed', '?'):.3f}")
            if 'concordance_l2r' in r:
                print(f"    concordance: step_seq={r['concordance_l2r']:.3f} "
                      f"calc_first={r.get('concordance_calc_first', '?'):.3f}")

    # Continuation training (PUMA → random)
    args = parse_args()
    if not getattr(args, 'no_continuation', False) and 'puma' in models:
        print(f"\n{'─'*60}")
        print(f"  Continuation: PUMA → random ({CONTINUATION_ITERS} iters)")
        print(f"{'─'*60}")
        puma_state = models['puma'].state_dict()
        cont_model, cont_dyn = train_model(
            'random', tokenizer, train_samples, test_samples, test_metas,
            max_len, max_iters=CONTINUATION_ITERS,
            init_state=puma_state, device=DEVICE)
        for dp in DECODE_POLICIES[:2]:  # confidence + step_seq
            r = gen_eval(cont_model, tokenizer, test_samples, test_metas,
                         max_len, dp, device=DEVICE)
            key = f"cont_puma2random_{dp}"
            all_final[key] = r
            print(f"    {key}: exact={r['accuracy']:.3f} valid={r['valid_rate']:.3f}")

    # Summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    for dp in DECODE_POLICIES:
        print(f"\n  ── {dp} ──")
        print(f"  {'Test':<25s}", end='')
        for mt in MASK_TYPES:
            print(f" {mt:>12s}", end='')
        print()
        for test_name in ['accuracy', 'valid_rate', 'acc_plan', 'acc_calc',
                          'acc_full_chain', 'acc_mixed',
                          'concordance_l2r', 'concordance_calc_first']:
            vals = [all_final.get(f'{mt}_{dp}', {}).get(test_name) for mt in MASK_TYPES]
            if any(v is not None for v in vals):
                print(f"  {test_name:<25s}", end='')
                for v in vals:
                    print(f" {v:>12.4f}" if v is not None else f" {'N/A':>12s}", end='')
                print()

    # Save
    sd = {'config': {k: globals()[k] for k in [
        'MAX_ANS_LEN', 'MAX_SEQ_LEN', 'N_LAYER', 'N_HEAD', 'N_EMBD',
        'MASK_TYPES', 'DECODE_POLICIES', 'MAX_ITERS', 'BATCH_SIZE',
        'PUMA_K', 'SEED']}}
    for k, v in all_dyn.items():
        sd[f'dyn_{k}'] = v
    for k, v in all_final.items():
        sd[f'final_{k}'] = v
    save_results(exp_name, sd)
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
