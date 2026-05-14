"""Wrong-commit profile analysis for Countdown.

Mirrors the structure of addition_decode_analysis.py / maze_decode_analysis.py.

Core hypothesis (Countdown analog of chain-MSB shortcut in addition):
  Confidence-based decoding commits to first-step operand choices using
  proximity-to-target as a local proxy, before validating that the chosen
  operands can complete a chain reaching the target. PUMA's confidence-aligned
  training entrenches this premature-operand-commit pattern, producing chains
  that arithmetic-validate locally but fail globally (target not reached, or
  intermediate operands violate the input pool).

What this script measures, per (method, stratum) checkpoint:
  C1  Per-instance side-by-side decoding (confidence vs step_seq oracle vs random).
  C2  Failure categorization: only_oracle / only_conf / neither / both_correct,
      stratified by multiplicity bin (m=1-3, m=4-10, m=11+).
  C3  Position-type of first wrong commit (PLAN / SEP / CALC) and step index
      (1, 2, 3). Hypothesis: PUMA failures concentrate at PLAN positions
      of step 1.
  C4  Arithmetic-consistency vs target-reachability:
       - arithmetic_valid: does each emitted "a OP b = c" satisfy the arithmetic?
       - operand_pool_valid: do operands come from inputs ∪ prior intermediates?
       - target_reached: does the final result equal the target?
      Hypothesis: PUMA's wrong commits have HIGH arithmetic_valid + LOW
      target_reached. Random's wrong commits have lower arithmetic_valid
      (random arithmetic mistakes) but similar target_reached.
  C5  Confidence calibration on wrong commits: top-1 probability distribution
      of correct-commit vs wrong-commit. (Addition analog: 0.997 on wrong
      commits, gold ranked second.)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

if '__file__' in dir():
    _here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(_here))
    sys.path.insert(0, _here)
else:
    sys.path.insert(0, '.')

from core.train_utils import DEVICE  # type: ignore
from exp_countdown import (  # type: ignore
    MAX_ANS_LEN, MAX_SEQ_LEN, MULT_BINS, _mult_bin,
    SEP_CHAR, EOS_CHAR, INPUT_PAD,
    classify_output_positions, POS_PLAN, POS_CALC, POS_SEP, POS_PAD,
    build_tok, count_solutions,
    build_oracle_order_step_seq,
    _generate_step_seq,
)
from core.train_utils import generate_diffusion  # type: ignore

METHODS = ["random", "papl", "puma"]


# ── Model loading (mirrors addition_decode_analysis) ───────────────────────
def load_model(ckpt_path, device, n_head_override=None):
    sd = torch.load(ckpt_path, map_location=device, weights_only=True)
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    if any(k.startswith("module.") for k in sd):
        sd = {k.removeprefix("module."): v for k, v in sd.items()}
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    vocab_size, n_embd = sd["wte.weight"].shape
    has_wpe = "wpe.weight" in sd
    block_size = sd["wpe.weight"].shape[0] if has_wpe else 512
    n_layer = 0
    while f"blocks.{n_layer}.attn.c_attn.weight" in sd:
        n_layer += 1
    n_head = n_head_override if n_head_override is not None else 12
    pos_enc = "absolute" if has_wpe else "rope"
    print(f"  inferred arch: vocab={vocab_size} n_embd={n_embd} "
          f"block_size={block_size} n_layer={n_layer} n_head={n_head}")
    from core.model import Transformer  # type: ignore
    model = Transformer(
        vocab_size=vocab_size, block_size=block_size,
        n_layer=n_layer, n_head=n_head, n_embd=n_embd,
        dropout=0.0, is_causal=False, pos_enc=pos_enc,
    )
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    return model


# ── Equation chain parsing ─────────────────────────────────────────────────
EQ_PATTERN = re.compile(r'^(-?\d+)\s*([+\-*/])\s*(-?\d+)\s*=\s*(-?\d+)$')


def _eval_op(a, op, b):
    """Evaluate a single arithmetic operation. Returns int or None on failure."""
    try:
        a, b = int(a), int(b)
    except (ValueError, TypeError):
        return None
    if op == '+':
        return a + b
    if op == '-':
        return a - b
    if op == '*':
        return a * b
    if op == '/':
        if b == 0 or a % b != 0:
            return None
        return a // b
    return None


def parse_chain(output_str):
    """Parse 'a+b=c,d-e=f,...' into list of (a, op, b, c) tuples.

    Returns (steps, parse_ok) where parse_ok is True only if every step matches
    the canonical 'a OP b = c' regex.
    """
    raw = output_str.split(EOS_CHAR)[0].strip(',')
    raw = raw.strip()
    if not raw:
        return [], False
    steps = []
    parse_ok = True
    for step_str in raw.split(','):
        m = EQ_PATTERN.match(step_str.strip())
        if m is None:
            parse_ok = False
            steps.append(None)
        else:
            a, op, b, c = m.groups()
            steps.append((int(a), op, int(b), int(c)))
    return steps, parse_ok


def chain_validity_profile(output_str, inputs, target):
    """Score the emitted chain on four axes:
       - parse_ok: all steps match 'a OP b = c' regex
       - arithmetic_valid: each step's c equals the actual OP(a, b)
       - operand_pool_valid: operands at each step are drawn from
            (inputs ∪ {results of all prior steps}), and each input used at
            most once across the chain
       - target_reached: final step's c equals target
    """
    steps, parse_ok = parse_chain(output_str)
    if not parse_ok or not steps:
        return {
            'parse_ok': False,
            'arithmetic_valid': False,
            'operand_pool_valid': False,
            'target_reached': False,
            'n_steps': len(steps),
        }
    arithmetic_valid = True
    operand_pool_valid = True
    available = list(map(int, inputs))   # mutable pool (consumed as used)
    intermediates_seen = []
    for step in steps:
        if step is None:
            arithmetic_valid = False
            operand_pool_valid = False
            continue
        a, op, b, c = step
        # Arithmetic
        actual = _eval_op(a, op, b)
        if actual is None or actual != c:
            arithmetic_valid = False
        # Operand pool: a and b must each be present in available pool
        # (countdown allows reusing produced intermediates, not original inputs)
        local_pool = available + intermediates_seen
        consumed = {}
        for v in (a, b):
            consumed[v] = consumed.get(v, 0) + 1
        ok = True
        for v, k in consumed.items():
            if local_pool.count(v) < k:
                ok = False
                break
        if not ok:
            operand_pool_valid = False
        else:
            # Greedy consumption: prefer to consume an input first if
            # both an input and an intermediate could supply the operand.
            for v in (a, b):
                if v in available:
                    available.remove(v)
                elif v in intermediates_seen:
                    intermediates_seen.remove(v)
        intermediates_seen.append(c)
    target_reached = (steps[-1] is not None and steps[-1][3] == int(target))
    return {
        'parse_ok': True,
        'arithmetic_valid': arithmetic_valid,
        'operand_pool_valid': operand_pool_valid,
        'target_reached': target_reached,
        'n_steps': len(steps),
    }


# ── Test bucket builder ────────────────────────────────────────────────────
def build_test_buckets(test_entries, bucket_axis='mult'):
    """Group test entries into multiplicity buckets for stratified analysis."""
    buckets = {bn: [] for bn in MULT_BINS}
    for e in test_entries:
        if bucket_axis == 'mult':
            bn = _mult_bin(e.get('multiplicity', 0))
        else:
            bn = 'overall'
        if bn in buckets:
            buckets[bn].append(e)
    return buckets


# ── C1  Per-instance side-by-side decoding ─────────────────────────────────
@torch.no_grad()
def c1_per_instance(model, tokenizer, entries, device=None, batch_size=64):
    """For each entry, run confidence + step_seq + random decode, collect
    (correct?, output_str) triples for each policy.
    """
    device = device or DEVICE
    mask_id = tokenizer.special_ids['mask']
    pad_id = tokenizer.special_ids['pad']
    results = []
    for st in range(0, len(entries), batch_size):
        batch = entries[st:min(st + batch_size, len(entries))]
        B = len(batch)
        # Prompt = input_str + SEP_CHAR
        prompts = [e['input_str'] + SEP_CHAR for e in batch]
        penc = [tokenizer.encode(p) for p in prompts]
        pm = max(len(p) for p in penc)
        pids = torch.full((B, pm), pad_id, dtype=torch.long)
        for i, e in enumerate(penc):
            pids[i, :len(e)] = torch.tensor(e)
        pids = pids.to(device)

        per_policy = {}
        for policy in ('confidence', 'random'):
            gen, _, _ = generate_diffusion(model, pids, MAX_ANS_LEN, mask_id,
                                            policy=policy, greedy=True,
                                            pad_to=MAX_SEQ_LEN, pad_id=pad_id,
                                            device=device)
            pred_ids = gen[:, pm:pm + MAX_ANS_LEN]
            per_policy[policy] = [tokenizer.decode(pred_ids[i].cpu().tolist())
                                  for i in range(B)]
        # Step-sequential oracle
        oracle_orders = [build_oracle_order_step_seq(e['output_str']) for e in batch]
        gen_ss, _ = _generate_step_seq(model, pids, oracle_orders, MAX_ANS_LEN,
                                       mask_id, pad_to=MAX_SEQ_LEN, pad_id=pad_id,
                                       device=device)
        pred_ids_ss = gen_ss[:, pm:pm + MAX_ANS_LEN]
        per_policy['step_seq'] = [tokenizer.decode(pred_ids_ss[i].cpu().tolist())
                                  for i in range(B)]

        for i in range(B):
            e = batch[i]
            gold = e['output_str']
            rec = {
                'inputs': e.get('inputs'),
                'target': e.get('target'),
                'multiplicity': e.get('multiplicity'),
                'gold_chain': gold,
                'preds': {p: per_policy[p][i].split(EOS_CHAR)[0].strip(',').strip()
                          for p in per_policy},
            }
            for p in per_policy:
                pred_strip = rec['preds'][p]
                gold_strip = gold.split(EOS_CHAR)[0].strip(',').strip()
                rec[f'{p}_correct'] = (pred_strip == gold_strip)
                rec[f'{p}_validity'] = chain_validity_profile(
                    rec['preds'][p], e.get('inputs', []), e.get('target', 0))
            results.append(rec)
    return results


# ── C2  Failure categorization, stratified by multiplicity ─────────────────
def c2_categorize(per_instance):
    """Split into {both_correct, only_oracle, only_conf, neither} for the
       confidence vs step_seq comparison, broken down by multiplicity bin.
    """
    cats = defaultdict(lambda: defaultdict(int))
    for r in per_instance:
        bn = _mult_bin(r.get('multiplicity', 0))
        c = r.get('confidence_correct', False)
        o = r.get('step_seq_correct', False)
        if c and o:
            cats[bn]['both_correct'] += 1
        elif (not c) and o:
            cats[bn]['only_oracle'] += 1
        elif c and (not o):
            cats[bn]['only_conf'] += 1
        else:
            cats[bn]['neither'] += 1
        cats[bn]['n'] += 1
    return {bn: dict(d) for bn, d in cats.items()}


# ── C3  Failure dissection: position type & step of first wrong commit ────
@torch.no_grad()
def c3_failure_dissection(model, tokenizer, entries, max_examples=80,
                          device=None):
    """For each entry, run confidence decode with full trace.  For wrong
    instances, identify the first answer-region position that diverges from
    gold, and classify by (position type, step index, what was committed).
    """
    device = device or DEVICE
    mask_id = tokenizer.special_ids['mask']
    pad_id = tokenizer.special_ids['pad']

    bucket_examples = defaultdict(list)
    summary = {bn: defaultdict(int) for bn in MULT_BINS}
    # Aggregate: where do failures land?
    pos_type_dist = {bn: defaultdict(int) for bn in MULT_BINS}
    step_idx_dist = {bn: defaultdict(int) for bn in MULT_BINS}

    for entry in entries:
        prompt = entry['input_str'] + SEP_CHAR
        pe = tokenizer.encode(prompt)
        pm = len(pe)
        pids = torch.tensor([pe], dtype=torch.long, device=device)
        # Gold answer chars (untokenized: same chars as positions)
        gold_str = entry['output_str']
        bn = _mult_bin(entry.get('multiplicity', 0))

        # Manual confidence-greedy decode capturing per-stage commits.
        T = pm + MAX_ANS_LEN
        x = torch.full((1, T), mask_id, dtype=torch.long, device=device)
        x[:, :pm] = pids
        unmasked = torch.zeros(1, T, dtype=torch.bool, device=device)
        unmasked[:, :pm] = True

        trace = []   # list of {stage, ans_offset, committed, top1_prob, gold_char}
        for stage in range(MAX_ANS_LEN):
            logits = model(x)
            logits[:, :, mask_id] = -float('inf')
            max_logit = logits.max(dim=-1).values
            max_logit[unmasked] = -float('inf')
            if not torch.isfinite(max_logit).any():
                break
            pos = max_logit.argmax(-1)
            p = pos[0].item()
            if unmasked[0, p]:
                break
            probs = F.softmax(logits[0, p], dim=-1)
            top1 = probs.argmax().item()
            top1_prob = probs[top1].item()
            ans_offset = p - pm
            committed_char = tokenizer.decode([top1])
            gold_char = (gold_str[ans_offset] if ans_offset < len(gold_str)
                         else EOS_CHAR)
            trace.append({
                'stage': stage,
                'ans_offset': ans_offset,
                'committed': committed_char,
                'gold': gold_char,
                'top1_prob': top1_prob,
                'is_correct': (committed_char == gold_char),
            })
            x[0, p] = top1
            unmasked[0, p] = True

        # Build emitted output string (in ans-offset order)
        emitted = [None] * MAX_ANS_LEN
        for t in trace:
            if 0 <= t['ans_offset'] < MAX_ANS_LEN:
                emitted[t['ans_offset']] = t['committed']
        pred_str = ''.join(c if c is not None else '?' for c in emitted)
        pred_strip = pred_str.split(EOS_CHAR)[0].strip(',').strip()
        gold_strip = gold_str.split(EOS_CHAR)[0].strip(',').strip()
        if pred_strip == gold_strip:
            summary[bn]['both_correct_or_conf_correct'] += 1
            continue
        summary[bn]['wrong'] += 1

        # First wrong ans_offset (by position, not by stage)
        first_wrong_offset = None
        for off in range(min(len(gold_str), MAX_ANS_LEN)):
            if emitted[off] != gold_str[off]:
                first_wrong_offset = off
                break
        if first_wrong_offset is None:
            continue

        # Classify by position type
        pos_types = classify_output_positions(gold_str)
        if first_wrong_offset < len(pos_types):
            ptype = pos_types[first_wrong_offset]
        else:
            ptype = POS_PAD
        # Which step is it?
        steps_raw = gold_str.split(EOS_CHAR)[0].split(',')
        cumlen = 0
        step_idx = -1
        for si, sstr in enumerate(steps_raw):
            if first_wrong_offset < cumlen + len(sstr):
                step_idx = si
                break
            cumlen += len(sstr) + 1   # +1 for the comma
        pos_type_dist[bn][ptype] += 1
        step_idx_dist[bn][f'step_{step_idx + 1}'] += 1

        # Find the stage at which the wrong-offset position was committed
        wrong_commit_trace = next(
            (t for t in trace if t['ans_offset'] == first_wrong_offset), None)

        record = {
            'inputs': entry.get('inputs'),
            'target': entry.get('target'),
            'multiplicity': entry.get('multiplicity'),
            'gold': gold_strip,
            'pred': pred_strip,
            'first_wrong_offset': first_wrong_offset,
            'first_wrong_pos_type': ptype,
            'first_wrong_step': step_idx + 1,
            'commit_stage_of_wrong': (wrong_commit_trace['stage']
                                      if wrong_commit_trace else None),
            'commit_prob_of_wrong': (wrong_commit_trace['top1_prob']
                                     if wrong_commit_trace else None),
            'validity': chain_validity_profile(
                pred_str, entry.get('inputs', []), entry.get('target', 0)),
        }
        if len(bucket_examples[bn]) < max_examples:
            bucket_examples[bn].append(record)

    return {
        'summary': {bn: dict(d) for bn, d in summary.items()},
        'pos_type_dist': {bn: dict(d) for bn, d in pos_type_dist.items()},
        'step_idx_dist': {bn: dict(d) for bn, d in step_idx_dist.items()},
        'examples': {bn: examples for bn, examples in bucket_examples.items()},
    }


# ── C4  Validity profile aggregate (from C1 results) ───────────────────────
def c4_validity_aggregate(per_instance_results):
    """Aggregate chain-validity profiles for wrong instances of each method.

    Key question: are PUMA's wrong chains arithmetic-valid (just don't reach
    target) or arithmetic-invalid (broken)?
    """
    agg = defaultdict(lambda: defaultdict(int))
    for r in per_instance_results:
        bn = _mult_bin(r.get('multiplicity', 0))
        for policy in ('confidence', 'step_seq', 'random'):
            v = r.get(f'{policy}_validity', {})
            correct = r.get(f'{policy}_correct', False)
            key_prefix = f'{policy}_{bn}'
            agg[key_prefix]['n'] += 1
            if correct:
                agg[key_prefix]['correct'] += 1
                continue
            agg[key_prefix]['wrong'] += 1
            if v.get('parse_ok'):
                agg[key_prefix]['wrong_parse_ok'] += 1
                if v.get('arithmetic_valid'):
                    agg[key_prefix]['wrong_arith_valid'] += 1
                if v.get('operand_pool_valid'):
                    agg[key_prefix]['wrong_pool_valid'] += 1
                if v.get('target_reached'):
                    agg[key_prefix]['wrong_target_reached'] += 1
                # Most diagnostic combination: arithmetic-valid, pool-valid,
                # but target NOT reached.  This is the "fluent dead-end chain"
                # — the model wrote a mathematically coherent equation chain
                # that simply doesn't compute to the requested target.
                if (v.get('arithmetic_valid') and v.get('operand_pool_valid')
                        and not v.get('target_reached')):
                    agg[key_prefix]['wrong_fluent_deadend'] += 1
            else:
                agg[key_prefix]['wrong_parse_fail'] += 1
    return {k: dict(d) for k, d in agg.items()}


# ── C5  Confidence calibration (correct vs wrong) ──────────────────────────
@torch.no_grad()
def c5_calibration(model, tokenizer, entries, device=None, max_examples=200):
    """At each commit, log top-1 probability and whether the commit is correct.
    Aggregates separately for PLAN, SEP, CALC positions.
    """
    device = device or DEVICE
    mask_id = tokenizer.special_ids['mask']
    pad_id = tokenizer.special_ids['pad']
    by_pt = defaultdict(lambda: {'correct': [], 'wrong': []})
    by_pt_bn = defaultdict(lambda: defaultdict(lambda: {'correct': [], 'wrong': []}))

    for ent_i, entry in enumerate(entries[:max_examples]):
        prompt = entry['input_str'] + SEP_CHAR
        pe = tokenizer.encode(prompt)
        pm = len(pe)
        T = pm + MAX_ANS_LEN
        pids = torch.tensor([pe], dtype=torch.long, device=device)
        x = torch.full((1, T), mask_id, dtype=torch.long, device=device)
        x[:, :pm] = pids
        unmasked = torch.zeros(1, T, dtype=torch.bool, device=device)
        unmasked[:, :pm] = True

        gold_str = entry['output_str']
        pos_types = classify_output_positions(gold_str)
        bn = _mult_bin(entry.get('multiplicity', 0))

        for stage in range(MAX_ANS_LEN):
            logits = model(x)
            logits[:, :, mask_id] = -float('inf')
            max_logit = logits.max(dim=-1).values
            max_logit[unmasked] = -float('inf')
            if not torch.isfinite(max_logit).any():
                break
            pos = max_logit.argmax(-1)
            p = pos[0].item()
            if unmasked[0, p]:
                break
            probs = F.softmax(logits[0, p], dim=-1)
            top1 = probs.argmax().item()
            top1_prob = probs[top1].item()
            ans_offset = p - pm
            committed_char = tokenizer.decode([top1])
            if ans_offset < len(gold_str):
                gold_char = gold_str[ans_offset]
                ptype = (pos_types[ans_offset] if ans_offset < len(pos_types)
                         else POS_PAD)
                correct = (committed_char == gold_char)
                bucket = by_pt[ptype]['correct' if correct else 'wrong']
                bucket.append(top1_prob)
                bn_bucket = by_pt_bn[bn][ptype]['correct' if correct else 'wrong']
                bn_bucket.append(top1_prob)
            x[0, p] = top1
            unmasked[0, p] = True

    def _summary(lst):
        if not lst:
            return None
        return {'n': len(lst), 'mean': sum(lst) / len(lst),
                'p10': sorted(lst)[max(0, len(lst) // 10 - 1)],
                'p50': sorted(lst)[len(lst) // 2],
                'p90': sorted(lst)[min(len(lst) - 1, 9 * len(lst) // 10)]}

    return {
        'by_position_type': {
            pt: {'correct': _summary(d['correct']),
                 'wrong': _summary(d['wrong'])} for pt, d in by_pt.items()},
        'by_pt_x_mult': {
            bn: {pt: {'correct': _summary(d['correct']),
                      'wrong': _summary(d['wrong'])} for pt, d in by_pt_dict.items()}
            for bn, by_pt_dict in by_pt_bn.items()},
    }


# ── CLI driver ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", type=Path, required=True,
                    help="Directory containing checkpoint_random.pt, "
                         "checkpoint_papl.pt, checkpoint_puma.pt")
    ap.add_argument("--test_data", type=Path, required=True,
                    help="JSONL test file (must contain input_str, output_str, "
                         "inputs, target, multiplicity)")
    ap.add_argument("--n_per_bucket", type=int, default=300,
                    help="Number of test instances per multiplicity bucket")
    ap.add_argument("--out_dir", type=Path, default=Path("./decode_analysis_countdown"))
    ap.add_argument("--n_head", type=int, default=12)
    ap.add_argument("--device", default=str(DEVICE))
    ap.add_argument("--analyses", nargs="+", default=["c1", "c2", "c3", "c4", "c5"])
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    tokenizer = build_tok()

    # Load test entries
    test_entries = []
    with open(args.test_data) as f:
        for line in f:
            d = json.loads(line)
            test_entries.append(d)
    print(f"Loaded {len(test_entries)} test entries")

    buckets = build_test_buckets(test_entries, bucket_axis='mult')
    sampled = []
    for bn in MULT_BINS:
        sampled.extend(buckets[bn][:args.n_per_bucket])
    print(f"Sampled {len(sampled)} entries across {len(buckets)} buckets")

    out_all = {}
    for method in METHODS:
        ckpt = args.ckpt_dir / f"checkpoint_{method}.pt"
        if not ckpt.exists():
            print(f"[skip {method}] no checkpoint at {ckpt}")
            continue
        print(f"\n▶ {method} ({ckpt})")
        model = load_model(ckpt, device, n_head_override=args.n_head)
        out = {}
        per_instance = None
        if "c1" in args.analyses or "c2" in args.analyses or "c4" in args.analyses:
            print("  [c1] per-instance side-by-side decode...")
            per_instance = c1_per_instance(model, tokenizer, sampled, device=device)
            out['c1_per_instance'] = per_instance
        if "c2" in args.analyses and per_instance is not None:
            print("  [c2] failure categorization...")
            out['c2_categorize'] = c2_categorize(per_instance)
        if "c3" in args.analyses:
            print("  [c3] failure dissection...")
            out['c3_failure_dissection'] = c3_failure_dissection(
                model, tokenizer, sampled, device=device)
        if "c4" in args.analyses and per_instance is not None:
            print("  [c4] validity aggregate...")
            out['c4_validity'] = c4_validity_aggregate(per_instance)
        if "c5" in args.analyses:
            print("  [c5] calibration probe...")
            out['c5_calibration'] = c5_calibration(
                model, tokenizer, sampled, device=device)
        out_all[method] = out
        with open(args.out_dir / f"{method}.json", "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"  saved {args.out_dir / f'{method}.json'}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Compact summary across methods
    summary = {}
    for method, out in out_all.items():
        s = {}
        if 'c2_categorize' in out:
            s['categorize'] = out['c2_categorize']
        if 'c3_failure_dissection' in out:
            s['pos_type_dist'] = out['c3_failure_dissection']['pos_type_dist']
            s['step_idx_dist'] = out['c3_failure_dissection']['step_idx_dist']
        if 'c4_validity' in out:
            s['validity'] = out['c4_validity']
        if 'c5_calibration' in out:
            s['calibration_by_pt'] = out['c5_calibration']['by_position_type']
        summary[method] = s
    with open(args.out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved summary: {args.out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
