"""Wrong-commit profile analysis for ListOps.

Mirrors addition_decode_analysis.py / maze_decode_analysis.py.

Core hypothesis (ListOps analog of chain-MSB shortcut in addition):
  Confidence-based decoding commits to an OUTER expression's trace value
  using only that expression's *directly-visible* leaf children, ignoring
  sub-expression children that are still masked. The model behaves as if
  the sub-expression children evaluate to a value compatible with the
  operator and visible leaves -- a "leaf-only" approximation of the outer
  computation that is often correct (because sub-expressions tend to
  return values within the leaf range) but fails on cases where the
  sub-expression result is the dominant operand.

What this script measures, per (method) checkpoint:
  L1  Per-instance side-by-side decoding (confidence vs layered_oracle vs random).
  L2  Failure categorization {both_correct, only_oracle, only_conf, neither},
      stratified by tree depth.
  L3  Failure dissection: for each wrong instance, identify the first
      mismatched trace position and (a) which sub-expression it represents,
      (b) the model's committed digit, (c) the true sub-expression value,
      (d) the "leaf-only" heuristic value for the same sub-expression.
  L4  Shortcut-match rate: among wrong commits, what fraction equal the
      leaf-only heuristic prediction? High rate = confirms confidence shortcut.
  L5  Calibration: top-1 probability on correct vs wrong commits, stratified
      by node depth (leaf vs intermediate vs root).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

import torch
import torch.nn.functional as F

if '__file__' in dir():
    _here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(_here))
    sys.path.insert(0, _here)
else:
    sys.path.insert(0, '.')

from core.train_utils import DEVICE, generate_diffusion  # type: ignore
from exp_listops import (  # type: ignore
    MAX_ANS_LEN, MAX_SEQ_LEN, build_tok,
)

METHODS = ["random", "papl", "puma"]


# ── Model loading ──────────────────────────────────────────────────────────
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
    n_head = n_head_override if n_head_override is not None else 3
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


# ── Operator evaluation ────────────────────────────────────────────────────
OPS = {
    'X': lambda xs: max(xs),
    'N': lambda xs: min(xs),
    'D': lambda xs: sorted(xs)[len(xs) // 2],   # MEDIAN
    'S': lambda xs: sum(xs) % 10,
}


def apply_op(op_ch, operands):
    """Apply ListOps operator to a list of integer operands."""
    if op_ch not in OPS:
        return None
    return OPS[op_ch](list(operands))


def leaf_only_value(op_ch, children_resolved, children_unresolved_default=None):
    """Compute the 'leaf-only' heuristic value for an expression with operator
    op_ch.  children_resolved is the list of operand values that the model
    actually sees (visible leaves + already-decoded sub-expressions); children
    that are still masked are skipped (or, optionally, replaced by a default).

    Returns the operator applied to ONLY the resolved children; if no
    resolved children, returns None.
    """
    if not children_resolved:
        if children_unresolved_default is not None:
            return apply_op(op_ch, [children_unresolved_default])
        return None
    return apply_op(op_ch, children_resolved)


# ── Meta extraction: parse the prompt to identify sub-expression structure ─
def parse_prompt_tree(prompt_str):
    """Parse the bracketed expression in the prompt into a tree.

    Returns a list of sub-expression nodes in post-order, each as a dict with
    fields:
       'op'      : operator character (X/N/D/S)
       'children': list of either int (leaf value) or sub-expr index
                   (referring to an earlier-evaluated sub-expr in post-order)
    The returned list's length == the number of trace digits the model emits.
    """
    s = prompt_str.replace(' ', '').replace('=', '')
    pos = 0
    n = len(s)
    nodes = []   # post-order list

    def parse_expr():
        nonlocal pos
        assert s[pos] == '['
        pos += 1   # consume '['
        op = s[pos]
        pos += 1
        children = []
        while pos < n and s[pos] != ']':
            if s[pos] == '[':
                child_idx = parse_expr()
                children.append(('sub', child_idx))
            elif s[pos].isdigit():
                children.append(('leaf', int(s[pos])))
                pos += 1
            else:
                pos += 1   # skip whitespace/spaces just in case
        assert pos < n and s[pos] == ']'
        pos += 1
        nodes.append({'op': op, 'children': children})
        return len(nodes) - 1

    if '[' in s:
        parse_expr()
    return nodes


def node_depth(nodes, idx):
    """Depth of node idx in the parse tree (0 = leaf-only / shallowest)."""
    node = nodes[idx]
    if not any(t == 'sub' for t, _ in node['children']):
        return 1
    return 1 + max(node_depth(nodes, ci) for t, ci in node['children']
                   if t == 'sub')


# ── L1  Per-instance side-by-side decoding ─────────────────────────────────
@torch.no_grad()
def l1_per_instance(model, tokenizer, entries, device=None, batch_size=64):
    """For each entry, run confidence/layered_oracle/random and collect
    (correct?, output_str) tuples.

    Entries must provide:
       'string'           full sequence (with = and trace)
       'prompt'           the bracketed expression up to (and including) '='
       'children_indices' per-trace-position list of child indices (used to
                          build the layered oracle reasoning_rank).
       'trace_len'        number of valid trace digits
    """
    device = device or DEVICE
    mask_id = tokenizer.special_ids['mask']
    pad_id = tokenizer.special_ids['pad']
    results = []
    for st in range(0, len(entries), batch_size):
        batch = entries[st:min(st + batch_size, len(entries))]
        B = len(batch)
        penc = [tokenizer.encode(e['prompt']) for e in batch]
        pm = max(len(p) for p in penc)
        pids = torch.full((B, pm), pad_id, dtype=torch.long)
        for i, e in enumerate(penc):
            pids[i, :len(e)] = torch.tensor(e)
        pids = pids.to(device)

        # Build layered reasoning rank for oracle
        r_rank = torch.full((B, MAX_ANS_LEN), MAX_ANS_LEN, dtype=torch.long)
        for bi, e in enumerate(batch):
            tl = min(e.get('trace_len', 0), MAX_ANS_LEN)
            ci_list = e.get('children_indices', [])
            ranks = [0] * tl
            for j in range(tl):
                if j < len(ci_list) and ci_list[j]:
                    valid = [c for c in ci_list[j] if 0 <= c < j]
                    if valid:
                        ranks[j] = 1 + max(ranks[c] for c in valid)
            for j in range(tl):
                r_rank[bi, j] = ranks[j]

        per_policy = {}
        for policy in ('confidence', 'random'):
            gen, _, _ = generate_diffusion(model, pids, MAX_ANS_LEN, mask_id,
                                            policy=policy, greedy=True,
                                            pad_to=MAX_SEQ_LEN, pad_id=pad_id,
                                            device=device)
            pred_ids = gen[:, pm:pm + MAX_ANS_LEN]
            per_policy[policy] = [tokenizer.decode(pred_ids[i].cpu().tolist())
                                  for i in range(B)]
        gen_o, _, _ = generate_diffusion(model, pids, MAX_ANS_LEN, mask_id,
                                          policy='layered_oracle', greedy=True,
                                          reasoning_rank=r_rank,
                                          pad_to=MAX_SEQ_LEN, pad_id=pad_id,
                                          device=device)
        pred_ids_o = gen_o[:, pm:pm + MAX_ANS_LEN]
        per_policy['layered_oracle'] = [tokenizer.decode(pred_ids_o[i].cpu().tolist())
                                        for i in range(B)]

        for i in range(B):
            e = batch[i]
            tl = e['trace_len']
            gold_trace = e['gold_trace'][:tl]   # list of digit chars/ints
            rec = {
                'prompt': e['prompt'],
                'depth': e.get('depth'),
                'gold_trace': gold_trace,
                'trace_len': tl,
                'preds': {},
            }
            for p in per_policy:
                pred = per_policy[p][i][:tl]
                rec['preds'][p] = pred
                # Position-by-position correctness
                pos_ok = [pred[k] == str(gold_trace[k]) if k < len(pred) else False
                          for k in range(tl)]
                rec[f'{p}_pos_correct'] = pos_ok
                rec[f'{p}_trace_correct'] = all(pos_ok)
            results.append(rec)
    return results


# ── L2  Categorization, by depth ───────────────────────────────────────────
def l2_categorize(per_instance):
    """Categorize each instance as {both, only_oracle, only_conf, neither}
    based on full-trace correctness, stratified by depth bin.
    """
    cats = defaultdict(lambda: defaultdict(int))
    for r in per_instance:
        d = r.get('depth', 0)
        dbin = f"d={d}"
        c = r.get('confidence_trace_correct', False)
        o = r.get('layered_oracle_trace_correct', False)
        if c and o:
            cats[dbin]['both_correct'] += 1
        elif (not c) and o:
            cats[dbin]['only_oracle'] += 1
        elif c and (not o):
            cats[dbin]['only_conf'] += 1
        else:
            cats[dbin]['neither'] += 1
        cats[dbin]['n'] += 1
    return {dbin: dict(d) for dbin, d in cats.items()}


# ── L3 / L4  Failure dissection with leaf-only heuristic probe ──────────────
def l3_l4_leaf_only_probe(per_instance):
    """For each wrong confidence-decoded instance:
       1. Find the first trace position that diverges from gold.
       2. Identify the corresponding sub-expression node (using
          parse_prompt_tree on the prompt).
       3. Compute the "leaf-only" heuristic value:
             apply operator to only the leaf children (skip sub-expr children).
       4. Compare model's wrong commit to (a) gold value, (b) leaf-only heuristic.
       5. Aggregate by node depth (leaf-only nodes vs intermediate vs root).

    A high shortcut-match rate -- model's wrong commit equals the leaf-only
    heuristic -- supports the confidence shortcut hypothesis.
    """
    summary = defaultdict(lambda: {
        'n_wrong': 0,
        'n_shortcut_match': 0,
        'n_neither': 0,
        'n_off_by_one': 0,
    })
    examples = defaultdict(list)
    by_node_role = defaultdict(lambda: defaultdict(int))

    for r in per_instance:
        if r.get('confidence_trace_correct', True):
            continue
        prompt = r['prompt']
        gold_trace = r['gold_trace']
        pred = r['preds']['confidence']
        nodes = parse_prompt_tree(prompt)
        if not nodes or len(nodes) != len(gold_trace):
            continue   # parse mismatch -- skip
        dbin = f"d={r.get('depth', '?')}"
        # First mismatch
        first_wrong = None
        for k in range(min(len(gold_trace), len(pred))):
            if pred[k] != str(gold_trace[k]):
                first_wrong = k
                break
        if first_wrong is None:
            continue
        node = nodes[first_wrong]
        op = node['op']
        # Resolved (visible) children: leaves + sub-expressions whose value
        # has already been correctly committed in earlier trace positions.
        leaf_vals = [v for tp, v in node['children'] if tp == 'leaf']
        sub_ids = [v for tp, v in node['children'] if tp == 'sub']
        # NOTE: when first_wrong is the *first* trace position diverging, sub-expr
        # children are at indices < first_wrong, so model has already seen their
        # commits.  Use model's committed values (= true if they were correct
        # before first_wrong).
        sub_vals_from_pred = []
        sub_vals_true = []
        for si in sub_ids:
            if si < len(pred) and pred[si].isdigit():
                sub_vals_from_pred.append(int(pred[si]))
            sub_vals_true.append(int(gold_trace[si]))

        true_val = int(gold_trace[first_wrong])
        try:
            pred_val = int(pred[first_wrong])
        except ValueError:
            pred_val = None
        # Leaf-only heuristic: operator applied to leaf operands only,
        # ignoring sub-expression children.  Captures the "confidence
        # shortcut": predict the outer using only visible leaves.
        leaf_only = leaf_only_value(op, leaf_vals)
        # Full-resolved-with-pred: operator using leaves + model's sub-expr
        # predictions (which were already committed before first_wrong).
        full_pred = (apply_op(op, leaf_vals + sub_vals_from_pred)
                     if (leaf_vals or sub_vals_from_pred) else None)
        # True value: matches operator applied to leaves + true sub-vals
        # (sanity check)
        full_true = apply_op(op, leaf_vals + sub_vals_true) if node['children'] else None
        is_root = (first_wrong == len(nodes) - 1)
        node_kind = ('leaf_only' if not sub_ids
                     else ('root' if is_root else 'intermediate'))

        summary[dbin]['n_wrong'] += 1
        by_node_role[dbin][node_kind] += 1
        if pred_val is not None and leaf_only is not None and pred_val == leaf_only:
            summary[dbin]['n_shortcut_match'] += 1
            by_node_role[dbin][f'{node_kind}_shortcut'] += 1
        elif pred_val is not None and abs(pred_val - true_val) == 1:
            summary[dbin]['n_off_by_one'] += 1
        else:
            summary[dbin]['n_neither'] += 1

        if len(examples[dbin]) < 30:
            examples[dbin].append({
                'prompt': prompt,
                'gold_trace': gold_trace,
                'pred_trace': pred,
                'first_wrong_idx': first_wrong,
                'node_op': op,
                'node_kind': node_kind,
                'leaf_children': leaf_vals,
                'sub_children_true': sub_vals_true,
                'sub_children_pred': sub_vals_from_pred,
                'true_val': true_val,
                'pred_val': pred_val,
                'leaf_only_heuristic': leaf_only,
                'full_pred_heuristic': full_pred,
                'full_true_heuristic': full_true,
            })

    return {
        'summary': {dbin: dict(d) for dbin, d in summary.items()},
        'by_node_role': {dbin: dict(d) for dbin, d in by_node_role.items()},
        'examples': {dbin: ex for dbin, ex in examples.items()},
    }


# ── L5  Calibration ────────────────────────────────────────────────────────
@torch.no_grad()
def l5_calibration(model, tokenizer, entries, device=None, max_examples=400):
    """Top-1 probability of correct vs wrong commits, stratified by node-depth
    of the trace position (leaf-only / intermediate / root).
    """
    device = device or DEVICE
    mask_id = tokenizer.special_ids['mask']
    pad_id = tokenizer.special_ids['pad']
    by_kind = defaultdict(lambda: {'correct': [], 'wrong': []})

    for entry in entries[:max_examples]:
        prompt = entry['prompt']
        pe = tokenizer.encode(prompt)
        pm = len(pe)
        T = pm + MAX_ANS_LEN
        pids = torch.tensor([pe], dtype=torch.long, device=device)
        x = torch.full((1, T), mask_id, dtype=torch.long, device=device)
        x[:, :pm] = pids
        unmasked = torch.zeros(1, T, dtype=torch.bool, device=device)
        unmasked[:, :pm] = True

        nodes = parse_prompt_tree(prompt)
        gold_trace = entry['gold_trace']
        tl = entry['trace_len']

        for stage in range(min(tl, MAX_ANS_LEN)):
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
            if 0 <= ans_offset < tl and ans_offset < len(nodes):
                node = nodes[ans_offset]
                has_sub = any(t == 'sub' for t, _ in node['children'])
                is_root = (ans_offset == len(nodes) - 1)
                kind = ('leaf_only' if not has_sub
                        else ('root' if is_root else 'intermediate'))
                committed_char = tokenizer.decode([top1])
                correct = (committed_char == str(gold_trace[ans_offset]))
                by_kind[kind]['correct' if correct else 'wrong'].append(top1_prob)
            x[0, p] = top1
            unmasked[0, p] = True

    def _summary(lst):
        if not lst:
            return None
        s = sorted(lst)
        return {'n': len(lst), 'mean': sum(lst) / len(lst),
                'median': median(lst),
                'p10': s[max(0, len(s) // 10 - 1)],
                'p90': s[min(len(s) - 1, 9 * len(s) // 10)]}

    return {kind: {'correct': _summary(d['correct']),
                   'wrong': _summary(d['wrong'])} for kind, d in by_kind.items()}


# ── CLI driver ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", type=Path, required=True)
    ap.add_argument("--test_data", type=Path, required=True,
                    help="JSONL test file (entries with prompt, gold_trace, "
                         "trace_len, depth, children_indices)")
    ap.add_argument("--n_per_depth", type=int, default=500)
    ap.add_argument("--depths", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--out_dir", type=Path, default=Path("./decode_analysis_listops"))
    ap.add_argument("--n_head", type=int, default=3)
    ap.add_argument("--device", default=str(DEVICE))
    ap.add_argument("--analyses", nargs="+", default=["l1", "l2", "l3l4", "l5"])
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    tokenizer = build_tok()

    test_entries = []
    with open(args.test_data) as f:
        for line in f:
            test_entries.append(json.loads(line))
    # Bucket by depth
    bucketed = defaultdict(list)
    for e in test_entries:
        bucketed[e.get('depth', 0)].append(e)
    sampled = []
    for d in args.depths:
        sampled.extend(bucketed.get(d, [])[:args.n_per_depth])
    print(f"Sampled {len(sampled)} entries across depths {args.depths}")

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
        if "l1" in args.analyses or "l2" in args.analyses or "l3l4" in args.analyses:
            print("  [l1] per-instance side-by-side decode...")
            per_instance = l1_per_instance(model, tokenizer, sampled, device=device)
            out['l1_per_instance'] = per_instance
        if "l2" in args.analyses and per_instance is not None:
            print("  [l2] failure categorization...")
            out['l2_categorize'] = l2_categorize(per_instance)
        if "l3l4" in args.analyses and per_instance is not None:
            print("  [l3l4] leaf-only shortcut probe...")
            out['l3_l4_shortcut'] = l3_l4_leaf_only_probe(per_instance)
        if "l5" in args.analyses:
            print("  [l5] calibration...")
            out['l5_calibration'] = l5_calibration(
                model, tokenizer, sampled, device=device)
        out_all[method] = out
        with open(args.out_dir / f"{method}.json", "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"  saved {args.out_dir / f'{method}.json'}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = {}
    for method, out in out_all.items():
        s = {}
        if 'l2_categorize' in out:
            s['categorize'] = out['l2_categorize']
        if 'l3_l4_shortcut' in out:
            s['shortcut'] = out['l3_l4_shortcut']['summary']
            s['by_node_role'] = out['l3_l4_shortcut']['by_node_role']
        if 'l5_calibration' in out:
            s['calibration'] = out['l5_calibration']
        summary[method] = s
    with open(args.out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved summary: {args.out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
