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

  Difficulty axis (NEW): solution_multiplicity — # of valid solutions per
                         instance (DFS-enumerated). Low mult = constrained
                         instance, high mult = many valid paths.
                         This replaces/supplements chain_depth which is
                         actually a DFS-generation artifact proxy for mult.

  Oracle:          step_seq = complete step 1 (plan→calc) → step 2 → ...
                   (now available as an actual decode policy.)

  Corner case:     bottom-multiplicity OOD instances (mult=1 & long output)
                   — theory-motivated: instances where conditioning pattern
                   PUMA was trained on is least likely to cover.

  Additional analyses (NEW):
    • selective_reveal: reveal X% of gold tokens, single-pass predict rest
      → context-sensitivity test (zebra-style)
    • simulate_puma_coverage: per-position-type mask persistence
      → coverage-paradox diagnostic (listops-style)
    • step propagation: calc_i → plan_{i+1} error cascade
      → independent-error rate (listops op-propagation analog)
    • strict validation: multiset consistency check → cheat rate

  Training:  iter-based with EMA
  Decode:    confidence | step_seq (oracle) | l2r | random
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import sys, os, time, math, json, random, re, statistics, itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                if '__file__' in dir() else '.')
from core.tokenizer import CharTokenizer
from core.train_utils import (
    mount_drive, save_results, save_checkpoint,
    train_diffusion, puma_k_fixed, puma_k_linear, puma_k_step,
    generate_diffusion, DEVICE,
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

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Config
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MAX_ANS_LEN = 40
MAX_SEQ_LEN = 60

N_TRAIN = None; N_TEST = 1000; BATCH_SIZE = 256
MAX_ITERS = 400000; EVAL_EVERY = 5000; LOG_EVERY = 1000
GEN_EVAL_EVERY = 10000; GEN_EVAL_N = 500
MASK_TYPES = ['random', 'puma']
# step_seq = structural oracle (complete step i before step i+1)
# l2r = token-level left-to-right (partial oracle for chain_depth=1 samples)
DECODE_POLICIES = ['confidence', 'step_seq', 'l2r', 'random']

N_LAYER = 3; N_HEAD = 12; N_EMBD = 384; DROPOUT = 0.0; POS_ENC = 'absolute'
LR = 3e-4; MIN_LR = 1e-5; WARMUP_ITERS = 1000; GRAD_CLIP = 1.0
WEIGHT_DECAY = 0.01; EMA_DECAY = 0.9999
PUMA_TAU = 0.9; PUMA_K = 8
PUMA_K_START = None; PUMA_K_END = None
PUMA_K_STEP = 3; PUMA_K_EVERY = None
SEED = 42
NO_AMP = False
PATIENCE = 50000
CONTINUATION_ITERS = 10000

# NEW
SELECTIVE_REVEAL_FRACS = [0.25, 0.50, 0.75]
CORNER_OOD_N = 100

DATA_DIR = 'experiments/data'
TRAIN_FILE = 'cd4_train.jsonl'
TEST_FILE = 'cd4_test.jsonl'


def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', type=str)
    p.add_argument('--train-file', type=str); p.add_argument('--test-file', type=str)
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
    p.add_argument('--corner-ood-n', type=int)
    p.add_argument('--skip-selective', action='store_true')
    p.add_argument('--skip-coverage', action='store_true')
    p.add_argument('--skip-ood', action='store_true')
    p.add_argument('--tag', type=str, default=''); p.add_argument('--seed', type=int)
    p.add_argument('--seeds', nargs='+', type=int)
    try:
        args, _ = p.parse_known_args()
    except SystemExit:
        args, _ = p.parse_known_args([])
    g = globals()
    for a, gl in {
        'data_dir': 'DATA_DIR', 'train_file': 'TRAIN_FILE',
        'test_file': 'TEST_FILE',
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
        'corner_ood_n': 'CORNER_OOD_N',
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
POS_PLAN = 'plan'
POS_CALC = 'calc'
POS_SEP = 'sep'
POS_PAD = 'pad'
POS_TYPES = [POS_PLAN, POS_CALC, POS_SEP, POS_PAD]
POS_TO_ID = {n: i for i, n in enumerate(POS_TYPES)}


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


def classify_chain_depth(output_str):
    """Chain depth: # steps that reuse a previous result.
    NOTE: DFS-generation artifact, not intrinsic difficulty; kept for
    legacy compatibility. Use solution_mult for primary stratification.
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
# NEW: Solution multiplicity (intrinsic difficulty)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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
    """Exhaustive DFS: count valid solution paths reaching target.
    For start_size=4, <1ms per instance.
    """
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
    if m <= 1:
        return 'm=1'
    if m <= 5:
        return 'm=2-5'
    if m <= 20:
        return 'm=6-20'
    return 'm=21+'


MULT_BINS = ['m=1', 'm=2-5', 'm=6-20', 'm=21+']


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Oracle decode orders
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_oracle_order_step_seq(output_str):
    """Step-sequential oracle: step1 plan→sep→calc, then step2, ..."""
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


def build_oracle_order_calc_first(output_str):
    """Calc-first oracle: all calc/sep first, then all plan."""
    types = classify_output_positions(output_str)
    calc_sep = [i for i, t in enumerate(types) if t in (POS_CALC, POS_SEP)]
    plan = [i for i, t in enumerate(types) if t == POS_PLAN]
    return calc_sep + plan


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Data formatting & tokenizer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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
    padded_output = _rainbow_pad(output_str)
    sample = f"{input_str}{SEP_CHAR}{padded_output}"
    if len(sample) > MAX_SEQ_LEN:
        return None, None

    pos_types = classify_output_positions(output_str)
    pos_types.extend([POS_PAD] * (MAX_ANS_LEN - len(output_str)))

    chain_depth, is_full_chain, step_details = classify_chain_depth(output_str)

    nums_s = input_str.split(',')
    target = nums_s[-1] if nums_s else ''
    input_nums = [int(x) for x in nums_s[:-1]] if len(nums_s) > 1 else []

    meta = {
        'output_len': len(output_str),
        'chain_depth': chain_depth,
        'is_full_chain': is_full_chain,
        'n_steps': len(output_str.split(',')),
        'step_details': step_details,
        'pos_types': pos_types,
        'target': target,
        'input_str': input_str,
        'output_str': output_str,
        'input_nums': input_nums,
    }

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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Data loading
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_jsonl(filepath, max_n=None):
    data = []
    with open(filepath) as f:
        for line in f:
            d = json.loads(line.strip())
            data.append(d)
            if max_n and len(data) >= max_n:
                break
    return data


def load_and_format(filepath, max_n=None, seed=42, compute_mult=False):
    """Load jsonl, format samples, return (samples, metas).

    If `compute_mult=True`, each meta gets solution_mult + mult_bin.
    """
    raw = load_jsonl(filepath, max_n)
    rng = random.Random(seed)
    rng.shuffle(raw)

    samples, metas = [], []
    skipped = 0
    for d in raw:
        s, m = _format_sample(d['input'], d['output'], compute_mult=compute_mult)
        if s is not None:
            samples.append(s)
            metas.append(m)
        else:
            skipped += 1
    if skipped:
        print(f"  Skipped {skipped} samples (too long)")
    return samples, metas


def select_ood_corner_cases(test_samples, test_metas, n=CORNER_OOD_N):
    """Bottom-multiplicity, long-output instances as OOD corner.

    Theory: PUMA's training manifold is narrow; OOD = instances whose
    conditioning patterns are least covered by training. Low mult =
    constrained solution space = pattern underrepresented in training.
    """
    scored = []
    for i, m in enumerate(test_metas):
        mult = m.get('solution_mult', -1)
        if mult < 0:
            continue
        scored.append((mult, -m.get('output_len', 0), i))
    scored.sort()
    selected_idx = [i for _, _, i in scored[:n]]
    return ([test_samples[i] for i in selected_idx],
            [test_metas[i] for i in selected_idx])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Equation validation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def check_equation(eq_str):
    try:
        parts = eq_str.split('=')
        if len(parts) != 2:
            return False
        left, right = parts[0], parts[1]
        m = re.match(r'^(\d+)([+\-*/])(\d+)$', left)
        if not m:
            return False
        a, op, b = m.group(1), m.group(2), m.group(3)
        a, b = int(a), int(b)
        if op == '+':
            result = a + b
        elif op == '-':
            result = a - b
        elif op == '*':
            result = a * b
        elif op == '/':
            if b == 0 or a % b != 0:
                return False
            result = a // b
        else:
            return False
        return result == int(right)
    except:
        return False


def validate_countdown(pred_output, target, input_nums=None, strict=False):
    """Validate output. `strict=True` adds multiset-consistent number usage
    check: each equation's operands must come from pool = (inputs - used +
    previous results).
    """
    clean = pred_output.split(EOS_CHAR)[0] if EOS_CHAR in pred_output else pred_output
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

    reaches_target = False
    if steps and '=' in steps[-1]:
        final_result = steps[-1].split('=')[-1]
        reaches_target = (final_result == str(target))

    out = {
        'valid': all_correct and reaches_target,
        'n_correct_eqs': n_correct,
        'n_total_eqs': len(steps),
        'reaches_target': reaches_target,
        'all_eqs_correct': all_correct,
        'clean_output': clean,
    }

    if strict and input_nums is not None:
        pool = Counter(input_nums)
        strict_ok = True
        for step in steps:
            # Malformed predictions may contain 0 or 2+ '=' signs.
            # Reject any step that isn't of the exact form 'a OP b = c'.
            parts = step.split('=')
            if len(parts) != 2:
                strict_ok = False; break
            left, right = parts[0], parts[1]
            m = re.match(r'^(\d+)([+\-*/])(\d+)$', left)
            if not m:
                strict_ok = False; break
            try:
                a = int(m.group(1)); b = int(m.group(3))
                r_val = int(right)
            except (ValueError, TypeError):
                strict_ok = False; break
            if pool[a] == 0:
                strict_ok = False; break
            pool[a] -= 1
            if pool[b] == 0:
                strict_ok = False; break
            pool[b] -= 1
            pool[r_val] += 1
        out['strict_valid'] = strict_ok and all_correct and reaches_target
        out['cheat'] = out['valid'] and not out['strict_valid']
    else:
        out['strict_valid'] = out['valid']
        out['cheat'] = False
    return out


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Probe
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@torch.no_grad()
def probe_per_position(model, tokenizer, test_samples, test_metas,
                       max_len, device=None):
    if device is None:
        device = DEVICE
    mask_id = tokenizer.special_ids['mask']
    pad_id = tokenizer.special_ids['pad']
    model.eval()

    ids_all, ans_all = encode_countdown_samples(test_samples, tokenizer, max_len)
    ids_all, ans_all = ids_all.to(device), ans_all.to(device)

    total_loss = 0.0
    total_count = 0
    type_correct = defaultdict(int); type_total = defaultdict(int)
    chain_correct = defaultdict(int); chain_total = defaultdict(int)
    mult_correct = defaultdict(int); mult_total = defaultdict(int)

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
            meta = test_metas[start + i]
            pos_types = meta.get('pos_types', [])
            chain_depth = meta.get('chain_depth', 0)
            mb = meta.get('mult_bin')

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

                cd_key = f"chain_{chain_depth}"
                chain_correct[cd_key] += int(correct)
                chain_total[cd_key] += 1

                if mb:
                    mult_correct[mb] += int(correct)
                    mult_total[mb] += 1

                loss = F.cross_entropy(logits[i, pos].unsqueeze(0),
                                       ids[i, pos].unsqueeze(0))
                total_loss += loss.item()
                total_count += 1

    result = {
        'overall_loss': total_loss / max(total_count, 1),
        'overall_acc': sum(type_correct.values()) / max(sum(type_total.values()), 1),
    }
    for ptype in POS_TYPES:
        if type_total[ptype] > 0:
            result[f'acc_{ptype}'] = type_correct[ptype] / type_total[ptype]
            result[f'n_{ptype}'] = type_total[ptype]
    for key in sorted(chain_total.keys()):
        if chain_total[key] > 0:
            result[f'acc_{key}'] = chain_correct[key] / chain_total[key]
    for mb in MULT_BINS:
        if mult_total[mb] > 0:
            result[f'acc_{mb}'] = mult_correct[mb] / mult_total[mb]
            result[f'n_{mb}'] = mult_total[mb]
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NEW: step_seq oracle generation (extends generate_diffusion's policies)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@torch.no_grad()
def _generate_step_seq(model, prefix_ids, oracle_orders, n_tokens, mask_id,
                       pad_to=None, pad_id=None, device=None):
    """Greedy generation with per-sample oracle order.
    oracle_orders[b] = list of answer-region indices in decode order.
    Missing indices are appended in ascending order.
    """
    if device is None:
        device = DEVICE
    model.eval()
    B = prefix_ids.shape[0]
    T_pre = prefix_ids.shape[1]
    T_ans = n_tokens
    T = T_pre + T_ans

    x = torch.full((B, T), mask_id, dtype=torch.long, device=device)
    x[:, :T_pre] = prefix_ids.to(device)

    if pad_to is not None and pad_to > T:
        assert pad_id is not None
        n_pad = pad_to - T
        pad_block = torch.full((B, n_pad), pad_id, dtype=torch.long, device=device)
        x = torch.cat([x, pad_block], dim=1)

    # Full per-sample decode order (pad missing positions with ascending)
    full_orders = []
    for b in range(B):
        seen = set(oracle_orders[b])
        ext = list(oracle_orders[b]) + [i for i in range(n_tokens) if i not in seen]
        full_orders.append(ext[:n_tokens])

    orders_record = []
    for t in range(n_tokens):
        logits = model(x)
        logits[:, :, mask_id] = -float('inf')
        pos = torch.tensor([T_pre + full_orders[b][t] for b in range(B)],
                           dtype=torch.long, device=device)
        batch_arange = torch.arange(B, device=device)
        lp_at_pos = logits[batch_arange, pos]
        tok = lp_at_pos.argmax(-1)
        x[batch_arange, pos] = tok
        orders_record.append(pos)

    orders_t = torch.stack(orders_record, dim=1).cpu()
    return x, {'orders': orders_t}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Generation evaluation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@torch.no_grad()
def gen_eval(model, tokenizer, test_samples, test_metas, max_len,
             decode_policy='confidence', n=None, device=None):
    if device is None:
        device = DEVICE
    if n is not None:
        test_samples = test_samples[:n]
        test_metas = test_metas[:n]

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
            B = len(bind)
            batch_s = [test_samples[i] for i in bind]
            batch_m = [test_metas[i] for i in bind]

            penc = [tokenizer.encode(s.split(SEP_CHAR, 1)[0] + SEP_CHAR)
                    for s in batch_s]
            pids = torch.tensor(penc, dtype=torch.long)

            if decode_policy == 'step_seq':
                oracle_orders = [build_oracle_order_step_seq(m['output_str'])
                                 for m in batch_m]
                gen, info = _generate_step_seq(
                    model, pids, oracle_orders, MAX_ANS_LEN, mask_id,
                    pad_to=max_len, pad_id=pad_id, device=device)
            else:
                gen, _, info = generate_diffusion(
                    model, pids, MAX_ANS_LEN, mask_id,
                    policy=decode_policy, greedy=True,
                    pad_to=max_len, pad_id=pad_id, device=device)

            pred_ids = gen[:, pl:pl + MAX_ANS_LEN]

            for bi in range(B):
                meta = batch_m[bi]
                pred_str = tokenizer.decode(pred_ids[bi].cpu().tolist())
                gold_str = get_answer(batch_s[bi])

                exact = (pred_str == gold_str)
                target = meta.get('target', '')
                input_nums = meta.get('input_nums', [])
                validation = validate_countdown(
                    pred_str, target, input_nums=input_nums, strict=True)

                pos_types = meta.get('pos_types', [])
                type_corr = defaultdict(int); type_tot = defaultdict(int)
                for j in range(min(len(pred_str), len(gold_str), len(pos_types))):
                    ptype = pos_types[j]
                    type_tot[ptype] += 1
                    if pred_str[j] == gold_str[j]:
                        type_corr[ptype] += 1

                decode_order = info.get('orders')
                concordance_step_seq = None
                concordance_calc_first = None
                if decode_order is not None:
                    ans_start = pl
                    raw_order = (decode_order[bi].tolist()
                                 if hasattr(decode_order[bi], 'tolist')
                                 else decode_order[bi])
                    decode_step = {}
                    for step, abs_pos in enumerate(raw_order):
                        ans_pos = abs_pos - ans_start
                        if 0 <= ans_pos < MAX_ANS_LEN:
                            decode_step[ans_pos] = step
                    oracle_ss = build_oracle_order_step_seq(meta['output_str'])
                    concordance_step_seq = _rank_concordance(
                        decode_step, oracle_ss, MAX_ANS_LEN)
                    oracle_cf = build_oracle_order_calc_first(meta['output_str'])
                    concordance_calc_first = _rank_concordance(
                        decode_step, oracle_cf, MAX_ANS_LEN)

                results.append({
                    'exact': exact,
                    'valid': validation['valid'],
                    'strict_valid': validation['strict_valid'],
                    'cheat': validation['cheat'],
                    'all_eqs_correct': validation['all_eqs_correct'],
                    'reaches_target': validation['reaches_target'],
                    'n_correct_eqs': validation['n_correct_eqs'],
                    'chain_depth': meta.get('chain_depth', -1),
                    'is_full_chain': meta.get('is_full_chain', False),
                    'mult_bin': meta.get('mult_bin', 'unknown'),
                    'solution_mult': meta.get('solution_mult', -1),
                    'type_acc': {t: type_corr[t] / type_tot[t]
                                 for t in type_tot if type_tot[t] > 0},
                    'concordance_step_seq': concordance_step_seq,
                    'concordance_calc_first': concordance_calc_first,
                    'pred': pred_str[:40],
                    'gold': gold_str[:40],
                })

    n_total = len(results)
    agg = {
        'accuracy': sum(r['exact'] for r in results) / max(n_total, 1),
        'valid_rate': sum(r['valid'] for r in results) / max(n_total, 1),
        'strict_valid_rate': sum(r['strict_valid'] for r in results) / max(n_total, 1),
        'cheat_rate': sum(r['cheat'] for r in results) / max(n_total, 1),
        'eq_correct_rate': sum(r['all_eqs_correct'] for r in results) / max(n_total, 1),
        'target_rate': sum(r['reaches_target'] for r in results) / max(n_total, 1),
        'n': n_total,
    }

    for ptype in [POS_PLAN, POS_CALC, POS_SEP]:
        vals = [r['type_acc'].get(ptype) for r in results if ptype in r['type_acc']]
        if vals:
            agg[f'acc_{ptype}'] = sum(vals) / len(vals)

    for cd in sorted(set(r['chain_depth'] for r in results)):
        subset = [r for r in results if r['chain_depth'] == cd]
        if subset:
            agg[f'acc_chain_{cd}'] = sum(r['exact'] for r in subset) / len(subset)
            agg[f'n_chain_{cd}'] = len(subset)

    for mb in MULT_BINS:
        subset = [r for r in results if r['mult_bin'] == mb]
        if subset:
            agg[f'acc_{mb}'] = sum(r['exact'] for r in subset) / len(subset)
            agg[f'valid_{mb}'] = sum(r['valid'] for r in subset) / len(subset)
            agg[f'strict_valid_{mb}'] = sum(r['strict_valid'] for r in subset) / len(subset)
            agg[f'cheat_{mb}'] = sum(r['cheat'] for r in subset) / len(subset)
            agg[f'n_{mb}'] = len(subset)

    chain_sub = [r for r in results if r['is_full_chain']]
    mixed_sub = [r for r in results if not r['is_full_chain']]
    if chain_sub:
        agg['acc_full_chain'] = sum(r['exact'] for r in chain_sub) / len(chain_sub)
        agg['n_full_chain'] = len(chain_sub)
    if mixed_sub:
        agg['acc_mixed'] = sum(r['exact'] for r in mixed_sub) / len(mixed_sub)
        agg['n_mixed'] = len(mixed_sub)

    conc_ss = [r['concordance_step_seq'] for r in results
               if r['concordance_step_seq'] is not None]
    conc_cf = [r['concordance_calc_first'] for r in results
               if r['concordance_calc_first'] is not None]
    if conc_ss:
        agg['concordance_step_seq'] = sum(conc_ss) / len(conc_ss)
    if conc_cf:
        agg['concordance_calc_first'] = sum(conc_cf) / len(conc_cf)

    agg['examples'] = results[:5]
    agg['per_sample'] = [{'pred': r['pred'], 'gold': r['gold']}
                          for r in results]  # for propagation analysis
    return agg


def _rank_concordance(decode_step_map, oracle_order, n):
    oracle_rank = {}
    for rank, pos in enumerate(oracle_order):
        if pos < n:
            oracle_rank[pos] = rank
    concordant = 0; discordant = 0
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
# NEW: Selective reveal (zebra-style context sensitivity test)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@torch.no_grad()
def selective_reveal_eval(model, tokenizer, test_samples, test_metas, max_len,
                          reveal_frac=0.5, seed=0, device=None):
    """Reveal `reveal_frac` of gold tokens randomly; single-pass predict
    the rest. Accuracy stratified by position type & mult bin.

    A context-sensitive model should improve as reveal_frac grows.
    A context-insensitive model shows flat accuracy (zebra PUMA signature).
    """
    if device is None:
        device = DEVICE
    rng = random.Random(seed)
    mask_id = tokenizer.special_ids['mask']
    pad_id = tokenizer.special_ids['pad']
    model.eval()

    total_correct = 0; total = 0
    by_ptype = defaultdict(lambda: [0, 0])
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
                ans = get_answer(s)
                penc = tokenizer.encode(prefix)
                aenc = tokenizer.encode(ans)
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
                ptypes = meta.get('pos_types', [])
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
                    t = ptypes[j] if j < len(ptypes) else POS_PAD
                    by_ptype[t][0] += int(ok); by_ptype[t][1] += 1
                    by_mult[mb][0] += int(ok); by_mult[mb][1] += 1

    result = {
        'reveal_frac': reveal_frac,
        'accuracy': total_correct / max(total, 1),
        'n_predictions': total,
        'by_pos_type': {t: {'acc': c / max(n, 1), 'n': n}
                        for t, (c, n) in by_ptype.items() if n > 0},
        'by_mult_bin': {mb: {'acc': c / max(n, 1), 'n': n}
                        for mb, (c, n) in by_mult.items() if n > 0},
    }
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NEW: PUMA coverage simulation (listops-style)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@torch.no_grad()
def simulate_puma_coverage(model, tokenizer, test_samples, test_metas, max_len,
                           K=None, tau=None, n_samples=200, device=None):
    """Simulate PUMA teacher-forced chain; measure per-position-type mask
    persistence (fraction of steps each position remains masked).

    cov[t] ≈ training-time gradient signal to positions of type t.
    Expected paradox: cov[plan] > cov[calc] BUT acc[plan] < acc[calc].
    """
    if device is None:
        device = DEVICE
    if K is None:
        K = PUMA_K_END or PUMA_K
    if tau is None:
        tau = PUMA_TAU
    model.eval()
    mask_id = tokenizer.special_ids['mask']
    pad_id = tokenizer.special_ids['pad']

    cov_by_type = defaultdict(lambda: [0.0, 0])

    N = min(len(test_samples), n_samples)
    for si in range(N):
        s = test_samples[si]
        meta = test_metas[si]
        ptypes = meta.get('pos_types', [POS_PAD] * MAX_ANS_LEN)
        prefix = s.split(SEP_CHAR, 1)[0] + SEP_CHAR
        ans = get_answer(s)
        penc = tokenizer.encode(prefix)
        aenc = tokenizer.encode(ans)
        T_pre = len(penc)

        seq = penc + [mask_id] * MAX_ANS_LEN
        if len(seq) < max_len:
            seq = seq + [pad_id] * (max_len - len(seq))
        x = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)
        x0 = torch.tensor(aenc[:MAX_ANS_LEN], dtype=torch.long, device=device)
        is_m = torch.ones(MAX_ANS_LEN, dtype=torch.bool, device=device)
        steps_masked = torch.zeros(MAX_ANS_LEN)
        total_steps = 0

        for step in range(K):
            if not is_m.any():
                break
            total_steps += 1
            steps_masked += is_m.cpu().float()
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

        if total_steps == 0:
            continue
        frac = steps_masked / total_steps
        for j in range(MAX_ANS_LEN):
            t = ptypes[j] if j < len(ptypes) else POS_PAD
            cov_by_type[t][0] += frac[j].item()
            cov_by_type[t][1] += 1

    per_type = {}
    for t, (s, n) in cov_by_type.items():
        if n > 0:
            per_type[t] = s / n

    calc_cov = per_type.get(POS_CALC)
    deficit = {}
    if calc_cov is not None:
        for t, v in per_type.items():
            deficit[t] = v - calc_cov

    return {
        'per_type': per_type,
        'deficit_vs_calc': deficit,
        'K': K, 'tau': tau, 'n_samples': N,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NEW: Step propagation (listops op_propagation analog)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def analyse_step_propagation(per_sample_results, test_metas):
    """Step-level error cascade on full-chain instances.

    propagation_rate       = P(plan_{i+1} err | calc_i err)
    independent_error_rate = P(plan_{i+1} err | calc_i ok)
    """
    ce_pe = 0; ce_po = 0; co_pe = 0; co_po = 0

    for r, m in zip(per_sample_results, test_metas):
        if not m.get('is_full_chain'):
            continue
        pred = r['pred']; gold = r['gold']
        out_str = m['output_str']
        steps = out_str.split(',')
        step_bounds = []
        cur = 0
        for si, step in enumerate(steps):
            step_bounds.append((cur, cur + len(step)))
            cur += len(step)
            if si < len(steps) - 1:
                cur += 1

        step_calc_ok = [None] * len(steps)
        step_plan_ok = [None] * len(steps)
        for si, (a, b) in enumerate(step_bounds):
            eq = steps[si].find('=')
            if eq < 0:
                continue
            plan_range = range(a, a + eq)
            calc_range = range(a + eq + 1, b)
            p_ok = all(
                k < len(pred) and k < len(gold) and pred[k] == gold[k]
                for k in plan_range)
            c_ok = all(
                k < len(pred) and k < len(gold) and pred[k] == gold[k]
                for k in calc_range)
            step_plan_ok[si] = p_ok
            step_calc_ok[si] = c_ok

        for si in range(len(steps) - 1):
            if step_calc_ok[si] is None or step_plan_ok[si + 1] is None:
                continue
            c_ok = step_calc_ok[si]
            p_ok = step_plan_ok[si + 1]
            if not c_ok and not p_ok: ce_pe += 1
            elif not c_ok and p_ok:   ce_po += 1
            elif c_ok and not p_ok:   co_pe += 1
            else:                     co_po += 1

    total_child_err = ce_pe + ce_po
    total_child_ok = co_pe + co_po
    return {
        'n_transitions_child_err': total_child_err,
        'n_transitions_child_ok': total_child_ok,
        'propagation_rate': (ce_pe / total_child_err
                             if total_child_err > 0 else None),
        'tolerance_rate': (ce_po / total_child_err
                           if total_child_err > 0 else None),
        'independent_error_rate': (co_pe / total_child_ok
                                    if total_child_ok > 0 else None),
        'clean_pass_rate': (co_po / total_child_ok
                            if total_child_ok > 0 else None),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Training wrapper
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def train_model(mask_type, tokenizer, train_samples, test_samples, test_metas,
                max_len, max_iters=None, init_state=None, device=None):
    if device is None:
        device = DEVICE
    if max_iters is None:
        max_iters = MAX_ITERS

    train_ids, train_ans = encode_countdown_samples(train_samples, tokenizer, max_len)
    train_ids, train_ans = train_ids.to(device), train_ans.to(device)

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
            print(f"      [gen] exact={r['accuracy']:.3f} "
                  f"valid={r['valid_rate']:.3f} "
                  f"strict={r['strict_valid_rate']:.3f} "
                  f"cheat={r['cheat_rate']:.3f}")
            for mb in MULT_BINS:
                k = f'acc_{mb}'
                if k in r:
                    print(f"        {mb}: acc={r[k]:.3f} (n={r.get(f'n_{mb}', 0)})")
            probe['gen_accuracy'] = r['accuracy']
            probe['gen_valid'] = r['valid_rate']
            probe['gen_strict_valid'] = r['strict_valid_rate']
            probe['gen_cheat'] = r['cheat_rate']
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
# Serialization safety
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _to_serializable(obj):
    """Recursively convert tensors/sets to JSON-safe types."""
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if torch.is_tensor(obj):
        return obj.item() if obj.numel() == 1 else obj.cpu().tolist()
    if isinstance(obj, set):
        return list(obj)
    return obj


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

    args = parse_args()

    train_path = os.path.join(DATA_DIR, TRAIN_FILE)
    test_path = os.path.join(DATA_DIR, TEST_FILE)

    print(f"\nLoading train data from {train_path}...")
    train_samples, train_metas = load_and_format(
        train_path, max_n=N_TRAIN, seed=SEED, compute_mult=False)
    print(f"  {len(train_samples)} training samples")

    print(f"Loading test data from {test_path}...")
    t0 = time.time()
    test_samples, test_metas = load_and_format(
        test_path, max_n=N_TEST, seed=SEED + 1, compute_mult=True)
    print(f"  {len(test_samples)} test samples "
          f"(mult computed in {time.time()-t0:.1f}s)")

    chain_dist = Counter(m['chain_depth'] for m in train_metas)
    print(f"  Chain depth (train): {dict(sorted(chain_dist.items()))}")
    chain_test = Counter(m['chain_depth'] for m in test_metas)
    print(f"  Chain depth (test):  {dict(sorted(chain_test.items()))}")
    mult_test = Counter(m['mult_bin'] for m in test_metas)
    print(f"  Mult bin (test):     "
          f"{[(mb, mult_test.get(mb, 0)) for mb in MULT_BINS]}")

    ood_samples, ood_metas = [], []
    if not getattr(args, 'skip_ood', False):
        ood_samples, ood_metas = select_ood_corner_cases(
            test_samples, test_metas, n=CORNER_OOD_N)
        print(f"  OOD corner (low-mult tail): {len(ood_samples)} instances "
              f"— {Counter(m['mult_bin'] for m in ood_metas)}")

    tokenizer = build_tok()
    max_len = MAX_SEQ_LEN
    print(f"  Vocab size: {tokenizer.vocab_size}, Max seq len: {max_len}")

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

        # ─── Main eval across decode policies ───
        for dp in DECODE_POLICIES:
            print(f"\n  Evaluating: {mask_type} × {dp}")
            r = gen_eval(model, tokenizer, test_samples, test_metas,
                         max_len, dp, device=DEVICE)
            # Strip per_sample to keep saved JSON compact; we'll keep it
            # only for confidence (used in propagation) and drop elsewhere.
            per_sample = r.pop('per_sample', None)
            all_final[f"{mask_type}_{dp}"] = r
            print(f"    exact={r['accuracy']:.3f} "
                  f"valid={r['valid_rate']:.3f} "
                  f"strict={r['strict_valid_rate']:.3f} "
                  f"cheat={r['cheat_rate']:.3f}")
            for mb in MULT_BINS:
                k = f'acc_{mb}'
                if k in r:
                    print(f"    {mb}: acc={r[k]:.3f} "
                          f"strict={r.get(f'strict_valid_{mb}', 0):.3f} "
                          f"(n={r.get(f'n_{mb}', 0)})")
            if 'concordance_step_seq' in r:
                print(f"    concordance: step_seq={r['concordance_step_seq']:.3f} "
                      f"calc_first={r.get('concordance_calc_first', 0):.3f}")

            # Save per_sample only from confidence decode → used in propagation
            if dp == 'confidence' and per_sample is not None:
                prop = analyse_step_propagation(per_sample, test_metas)
                all_final[f"{mask_type}_step_propagation"] = prop
                ind = prop.get('independent_error_rate')
                prop_r = prop.get('propagation_rate')
                print(f"    step propagation: "
                      f"indep_err={ind:.3f}" if ind is not None else "indep_err=N/A",
                      f"propagation={prop_r:.3f}" if prop_r is not None else "propagation=N/A")

        # ─── NEW: Selective reveal ───
        if not getattr(args, 'skip_selective', False):
            print(f"\n  Selective reveal ({mask_type}):")
            for frac in SELECTIVE_REVEAL_FRACS:
                r = selective_reveal_eval(
                    model, tokenizer, test_samples, test_metas,
                    max_len, reveal_frac=frac, seed=SEED + int(frac * 100),
                    device=DEVICE)
                all_final[f"{mask_type}_selective_{int(frac*100)}"] = r
                parts = [f"acc={r['accuracy']:.3f}"]
                for pt in [POS_PLAN, POS_CALC, POS_SEP]:
                    if pt in r['by_pos_type']:
                        parts.append(f"{pt}={r['by_pos_type'][pt]['acc']:.3f}")
                print(f"    reveal={int(frac*100)}%: {' '.join(parts)}")

        # ─── NEW: PUMA coverage simulation ───
        if not getattr(args, 'skip_coverage', False):
            print(f"\n  PUMA coverage sim ({mask_type}):")
            cov = simulate_puma_coverage(
                model, tokenizer, test_samples, test_metas, max_len,
                n_samples=200, device=DEVICE)
            all_final[f"{mask_type}_coverage"] = cov
            for t, v in cov.get('per_type', {}).items():
                dv = cov.get('deficit_vs_calc', {}).get(t, 0)
                print(f"    cov[{t}]={v:.3f} (vs calc: {dv:+.3f})")

        # ─── NEW: OOD corner case eval ───
        if ood_samples:
            print(f"\n  OOD corner case (low-mult tail, n={len(ood_samples)}): {mask_type}")
            r_ood = gen_eval(model, tokenizer, ood_samples, ood_metas,
                             max_len, 'confidence', device=DEVICE)
            r_ood.pop('per_sample', None)
            all_final[f"{mask_type}_ood_corner"] = r_ood
            print(f"    exact={r_ood['accuracy']:.3f} "
                  f"valid={r_ood['valid_rate']:.3f} "
                  f"strict={r_ood['strict_valid_rate']:.3f} "
                  f"cheat={r_ood['cheat_rate']:.3f}")
            for ex in r_ood.get('examples', [])[:3]:
                print(f"      gold={ex['gold'][:35]}  pred={ex['pred'][:35]} "
                      f"{'✓' if ex['exact'] else '✗'}")

    # ─── Continuation training ───
    if not getattr(args, 'no_continuation', False) and 'puma' in models:
        print(f"\n{'─'*60}")
        print(f"  Continuation: PUMA → random ({CONTINUATION_ITERS} iters)")
        print(f"{'─'*60}")
        puma_state = models['puma'].state_dict()
        cont_model, cont_dyn = train_model(
            'random', tokenizer, train_samples, test_samples, test_metas,
            max_len, max_iters=CONTINUATION_ITERS,
            init_state=puma_state, device=DEVICE)
        for dp in ['confidence', 'step_seq']:
            r = gen_eval(cont_model, tokenizer, test_samples, test_metas,
                         max_len, dp, device=DEVICE)
            r.pop('per_sample', None)
            all_final[f"cont_puma2random_{dp}"] = r
            print(f"    cont_puma2random_{dp}: exact={r['accuracy']:.3f} "
                  f"valid={r['valid_rate']:.3f} "
                  f"strict={r['strict_valid_rate']:.3f}")

    # ─── Summary ───
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    for dp in DECODE_POLICIES:
        print(f"\n  ── {dp} ──")
        print(f"  {'Test':<28s}", end='')
        for mt in MASK_TYPES:
            print(f" {mt:>12s}", end='')
        print()
        for test_name in ['accuracy', 'valid_rate', 'strict_valid_rate', 'cheat_rate',
                          'acc_plan', 'acc_calc',
                          'acc_m=1', 'acc_m=2-5', 'acc_m=6-20', 'acc_m=21+',
                          'concordance_step_seq', 'concordance_calc_first']:
            vals = [all_final.get(f'{mt}_{dp}', {}).get(test_name) for mt in MASK_TYPES]
            if any(v is not None for v in vals):
                print(f"  {test_name:<28s}", end='')
                for v in vals:
                    print(f" {v:>12.4f}" if v is not None else f" {'N/A':>12s}", end='')
                print()

    print(f"\n  ── Selective reveal (overall accuracy) ──")
    print(f"  {'reveal_frac':<28s}", end='')
    for mt in MASK_TYPES:
        print(f" {mt:>12s}", end='')
    print()
    for frac in SELECTIVE_REVEAL_FRACS:
        pct = int(frac * 100)
        print(f"  {f'{pct}%':<28s}", end='')
        for mt in MASK_TYPES:
            r = all_final.get(f'{mt}_selective_{pct}', {})
            v = r.get('accuracy')
            print(f" {v:>12.4f}" if v is not None else f" {'N/A':>12s}", end='')
        print()

    print(f"\n  ── PUMA coverage (per-type mean) ──")
    print(f"  {'pos_type':<28s}", end='')
    for mt in MASK_TYPES:
        print(f" {mt:>12s}", end='')
    print()
    for pt in [POS_PLAN, POS_CALC, POS_SEP, POS_PAD]:
        print(f"  {pt:<28s}", end='')
        for mt in MASK_TYPES:
            v = all_final.get(f'{mt}_coverage', {}).get('per_type', {}).get(pt)
            print(f" {v:>12.4f}" if v is not None else f" {'N/A':>12s}", end='')
        print()

    print(f"\n  ── Step propagation ──")
    for name in ['independent_error_rate', 'propagation_rate']:
        print(f"  {name:<28s}", end='')
        for mt in MASK_TYPES:
            v = all_final.get(f'{mt}_step_propagation', {}).get(name)
            print(f" {v:>12.4f}" if v is not None else f" {'N/A':>12s}", end='')
        print()

    print(f"\n  ── OOD corner (bottom-multiplicity) ──")
    for name in ['accuracy', 'valid_rate', 'strict_valid_rate', 'cheat_rate']:
        print(f"  {name:<28s}", end='')
        for mt in MASK_TYPES:
            v = all_final.get(f'{mt}_ood_corner', {}).get(name)
            print(f" {v:>12.4f}" if v is not None else f" {'N/A':>12s}", end='')
        print()

    # ─── Save ───
    sd = {'config': {k: globals()[k] for k in [
        'MAX_ANS_LEN', 'MAX_SEQ_LEN', 'N_LAYER', 'N_HEAD', 'N_EMBD',
        'MASK_TYPES', 'DECODE_POLICIES', 'MAX_ITERS', 'BATCH_SIZE',
        'PUMA_K', 'SEED', 'CORNER_OOD_N', 'SELECTIVE_REVEAL_FRACS']}}
    for k, v in all_dyn.items():
        sd[f'dyn_{k}'] = _to_serializable(v)
    for k, v in all_final.items():
        sd[f'final_{k}'] = _to_serializable(v)

    try:
        json.dumps(sd)
    except Exception as e:
        print(f"\n  WARNING: JSON serialization failed: {e}")
        for k, v in sd.items():
            try:
                json.dumps(v)
            except Exception as ee:
                print(f"    BAD KEY: {k} → {ee}")

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
