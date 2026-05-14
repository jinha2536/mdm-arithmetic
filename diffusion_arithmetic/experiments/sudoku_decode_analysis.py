"""Wrong-commit profile analysis for Sudoku.

Mirrors addition_decode_analysis.py / maze_decode_analysis.py, but inverted:
sudoku is the *success case* for PUMA, so the analysis isolates what makes
PUMA's wrong commits qualitatively different from Random's, and tests the
unified mechanism hypothesis across all five tasks:

  Unified mechanism: PUMA's confidence-aligned training produces *locally
  consistent* commits.  Whether this helps or hurts depends on whether
  local consistency entails global correctness.

  - In addition / maze / listops / countdown: local consistency does NOT
    imply global correctness, so PUMA produces locally-coherent but
    globally-wrong commits more often than Random.
  - In sudoku: local consistency (row/col/box constraints) is precisely
    what defines a valid solution, so PUMA's locally-coherent commits
    are usually globally correct.  Random's commits violate row/col/box
    constraints far more often, even when individual cell accuracies are
    not far apart.

Analyses:
  S1  Per-instance side-by-side decode (confidence / random / oracle_technique /
      n_cands_cp), collecting board strings and per-cell correctness.
  S2  Local-constraint-violation rate of WRONG cells: among cells the model
      predicted wrongly, what fraction violate at least one row/col/box
      constraint with the surrounding committed values?  Hypothesis:
      Random's wrong cells violate constraints much more often than PUMA's.
  S3  Constraint-loss aggregated by technique level (TL0-TL4) and rating tier.
      Complements the existing reeval's metric aggregates with stratification.
  S4  Calibration of top-1 probability on correct vs wrong commits, stratified
      by technique level.  Hypothesis: PUMA's confidence tracks technique
      level (high conf on TL0, low on TL4).
  S5  Failure cascade: when a single cell is wrong, how many of its row/col/
      box partners are also wrong?  Spatial clustering metric.
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

from core.train_utils import DEVICE  # type: ignore
import exp_sudoku
from exp_sudoku import (  # type: ignore
    ANS_LEN, build_tok, evaluate,
)

METHODS = ["random", "papl", "puma"]
DIGITS = set('123456789')


# ── Model loading ──────────────────────────────────────────────────────────
def load_model(ckpt_path, device, n_head=8, n_embd=256, n_layer=8,
               block_size=200):
    sd = torch.load(ckpt_path, map_location=device, weights_only=True)
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    if any(k.startswith("module.") for k in sd):
        sd = {k.removeprefix("module."): v for k, v in sd.items()}
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    vocab_size, n_embd_inf = sd["wte.weight"].shape
    has_wpe = "wpe.weight" in sd
    block_size_inf = sd["wpe.weight"].shape[0] if has_wpe else block_size
    n_layer_inf = 0
    while f"blocks.{n_layer_inf}.attn.c_attn.weight" in sd:
        n_layer_inf += 1
    pos_enc = "absolute" if has_wpe else "rope"
    print(f"  inferred arch: vocab={vocab_size} n_embd={n_embd_inf} "
          f"block_size={block_size_inf} n_layer={n_layer_inf} n_head={n_head}")
    from core.model import Transformer  # type: ignore
    model = Transformer(
        vocab_size=vocab_size, block_size=block_size_inf,
        n_layer=n_layer_inf, n_head=n_head, n_embd=n_embd_inf,
        dropout=0.0, is_causal=False, pos_enc=pos_enc,
    )
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    return model


# ── Sudoku helpers ─────────────────────────────────────────────────────────
def cell_partners(idx):
    """Return the set of cell indices that share row, col, or 3x3 box with idx."""
    r, c = idx // 9, idx % 9
    br, bc = (r // 3) * 3, (c // 3) * 3
    partners = set()
    for j in range(9):
        partners.add(r * 9 + j)
        partners.add(j * 9 + c)
    for dr in range(3):
        for dc in range(3):
            partners.add((br + dr) * 9 + (bc + dc))
    partners.discard(idx)
    return partners


# Precompute partners table once
_PARTNERS = [cell_partners(i) for i in range(81)]


def cell_violates_constraint(board, idx, val):
    """Does placing val at idx violate any row/col/box partner with a
    *committed digit*? Empty (non-digit) partner cells don't trigger
    a violation.
    """
    if val not in DIGITS:
        return True
    for p in _PARTNERS[idx]:
        if board[p] == val:
            return True
    return False


def board_constraint_violations(board):
    """Count of (row/col/box) constraint pairs violated by a board string."""
    n_violations = 0
    for i in range(81):
        if board[i] not in DIGITS:
            continue
        for p in _PARTNERS[i]:
            if p > i and board[p] == board[i]:
                n_violations += 1
    return n_violations


def is_locally_consistent_commit(board_after, idx):
    """Was the value at idx consistent with surrounding row/col/box partners
    at the time of commit (using the board state with all committed values)?
    """
    val = board_after[idx]
    if val not in DIGITS:
        return False
    for p in _PARTNERS[idx]:
        if board_after[p] in DIGITS and board_after[p] == val:
            return False
    return True


# ── S1  Per-instance decode + wrong cell tagging ───────────────────────────
@torch.no_grad()
def s1_per_instance(model, tokenizer, test_data, decode_policies,
                    batch_size=32, device=None):
    """For each test puzzle, run each decode policy and collect:
       - predicted board (81-char string)
       - true (gold) board
       - blank mask
       - per-cell correctness
       - per-cell metadata (technique_level, is_given)
    """
    device = device or DEVICE
    results_by_policy = {}
    for policy in decode_policies:
        # Use the experiment's own evaluate function but capture per-cell info.
        r = evaluate(model, tokenizer, test_data, decode_policy=policy,
                     batch_size=batch_size, device=device)
        results_by_policy[policy] = r
    return results_by_policy


# Lightweight per-instance reconstructor (when r contains 'predictions')
def per_cell_records(test_data, eval_result, tl_meta=None):
    """Given an evaluate() result dict that includes 'predictions' (list of
    81-char strings, one per test puzzle), build per-cell wrong/correct
    records for blank cells.

    Returns list of dicts keyed by puzzle index, each containing 'cells':
       [{'idx', 'pred', 'gold', 'correct', 'tl', 'rating_tier'} ...]
    """
    preds = eval_result.get('predictions') or eval_result.get('boards')
    if not preds:
        return []
    out = []
    for pi, d in enumerate(test_data):
        if pi >= len(preds):
            break
        gold = d['solution'] if 'solution' in d else d.get('answer', '')
        puzzle = d['puzzle'] if 'puzzle' in d else d.get('prompt', '')
        pred = preds[pi]
        cells = []
        for idx in range(81):
            is_given = (puzzle[idx] in DIGITS) if idx < len(puzzle) else False
            if is_given:
                continue
            g = gold[idx] if idx < len(gold) else '?'
            p = pred[idx] if idx < len(pred) else '?'
            tl = None
            if tl_meta and pi < len(tl_meta):
                tl_arr = tl_meta[pi]
                if isinstance(tl_arr, (list, tuple)) and idx < len(tl_arr):
                    tl = tl_arr[idx]
            cells.append({
                'idx': idx,
                'pred': p,
                'gold': g,
                'correct': (p == g),
                'tl': tl,
            })
        out.append({'idx': pi, 'cells': cells, 'pred_board': pred,
                    'gold_board': gold, 'puzzle': puzzle,
                    'rating': d.get('rating')})
    return out


# ── S2  Local constraint violation rate of wrong cells ─────────────────────
def s2_wrong_cell_constraint_rate(per_cell_records_list):
    """Among wrong blank cells, what fraction VIOLATE a row/col/box
    constraint with surrounding committed cells in the same board?

    Hypothesis (testing the unified mechanism):
       Random's wrong cells: high violation rate
       PUMA's wrong cells:   low violation rate (locally consistent)
    """
    n_wrong = 0
    n_wrong_violating = 0
    n_wrong_locally_consistent = 0
    by_tl = defaultdict(lambda: {'n_wrong': 0, 'n_violating': 0,
                                  'n_locally_consistent': 0})
    for puz in per_cell_records_list:
        board = puz['pred_board']
        for c in puz['cells']:
            if c['correct']:
                continue
            n_wrong += 1
            tl = c.get('tl')
            tl_key = f"tl_{tl}" if tl is not None else "tl_unknown"
            by_tl[tl_key]['n_wrong'] += 1
            if cell_violates_constraint(board, c['idx'], c['pred']):
                n_wrong_violating += 1
                by_tl[tl_key]['n_violating'] += 1
            else:
                n_wrong_locally_consistent += 1
                by_tl[tl_key]['n_locally_consistent'] += 1
    return {
        'n_wrong': n_wrong,
        'n_wrong_violating': n_wrong_violating,
        'n_wrong_locally_consistent': n_wrong_locally_consistent,
        'frac_violating': (n_wrong_violating / n_wrong) if n_wrong else None,
        'frac_locally_consistent': (n_wrong_locally_consistent / n_wrong) if n_wrong else None,
        'by_tl': {k: dict(v) for k, v in by_tl.items()},
    }


# ── S3  Constraint loss stratified by TL ───────────────────────────────────
def s3_constraint_loss_by_tl(per_cell_records_list):
    """Total constraint violations on the final board, stratified by the
    dominant TL level of the puzzle's blank cells (proxy for difficulty).
    """
    by_tl_bucket = defaultdict(lambda: {'n_puzzles': 0, 'sum_violations': 0,
                                         'n_zero_violation': 0})
    for puz in per_cell_records_list:
        tls = [c.get('tl') for c in puz['cells'] if c.get('tl') is not None]
        if not tls:
            bucket = 'unknown'
        else:
            tl4_count = sum(1 for t in tls if t == 4)
            tl4_frac = tl4_count / len(tls)
            if tl4_frac >= 0.95:
                bucket = 'tl4_geq_0.95'
            elif tl4_frac >= 0.5:
                bucket = 'tl4_0.5-0.95'
            elif tl4_frac >= 0.1:
                bucket = 'tl4_0.1-0.5'
            else:
                bucket = 'tl4_lt_0.1'
        v = board_constraint_violations(puz['pred_board'])
        by_tl_bucket[bucket]['n_puzzles'] += 1
        by_tl_bucket[bucket]['sum_violations'] += v
        if v == 0:
            by_tl_bucket[bucket]['n_zero_violation'] += 1
    out = {}
    for b, d in by_tl_bucket.items():
        if d['n_puzzles'] == 0:
            continue
        out[b] = {
            'n_puzzles': d['n_puzzles'],
            'mean_violations': d['sum_violations'] / d['n_puzzles'],
            'frac_zero_violation': d['n_zero_violation'] / d['n_puzzles'],
        }
    return out


# ── S4  Calibration by technique level ─────────────────────────────────────
@torch.no_grad()
def s4_calibration_by_tl(model, tokenizer, test_data, device=None,
                         max_puzzles=200):
    """Confidence at commit time for correct vs wrong cells, stratified by TL.

    This requires manual decoding (the evaluate function doesn't surface
    confidence values).  We do confidence-greedy decode, recording the
    top-1 probability of each committed cell.
    """
    device = device or DEVICE
    mask_id = tokenizer.special_ids['mask']
    pad_id = tokenizer.special_ids['pad']
    by_tl = defaultdict(lambda: {'correct': [], 'wrong': []})

    for pi, d in enumerate(test_data[:max_puzzles]):
        s = d['string'] if 'string' in d else d.get('prompt', '')
        enc = tokenizer.encode(s)
        ml = len(enc)
        x = torch.tensor([enc], dtype=torch.long, device=device)
        # Find answer region start (just after '=')
        eq_id = tokenizer.encode('=')[0]
        eq_pos = (x[0] == eq_id).nonzero(as_tuple=True)[0]
        if len(eq_pos) == 0:
            continue
        ans_start = eq_pos[-1].item() + 1
        # Mask blanks (positions where puzzle has '.')
        meta = d.get('meta', [])
        gold = d.get('solution') or d.get('answer', '')
        puzzle_str = d.get('puzzle') or s.split('=')[0]
        blank_idx = [j for j in range(81) if j < len(puzzle_str)
                     and puzzle_str[j] not in DIGITS]
        for j in blank_idx:
            if ans_start + j < x.shape[1]:
                x[0, ans_start + j] = mask_id

        unmasked = (x != mask_id)
        for step in range(len(blank_idx)):
            logits = model(x)
            logits[:, :, mask_id] = -float('inf')
            cl = logits[0, ans_start:ans_start + 81].clone()
            still = (~unmasked[0, ans_start:ans_start + 81])
            scores = cl.max(dim=-1).values
            scores[~still] = -float('inf')
            if not torch.isfinite(scores).any():
                break
            best_j = int(scores.argmax().item())
            probs = F.softmax(cl[best_j], dim=-1)
            top1 = int(probs.argmax().item())
            top1_prob = float(probs[top1].item())
            committed_char = tokenizer.decode([top1])
            gold_char = gold[best_j] if best_j < len(gold) else '?'
            correct = (committed_char == gold_char)
            tl = None
            if meta and best_j < len(meta):
                tl = meta[best_j].get('technique_level')
            key = f"tl_{tl}" if tl is not None else "tl_unknown"
            by_tl[key]['correct' if correct else 'wrong'].append(top1_prob)
            x[0, ans_start + best_j] = top1
            unmasked[0, ans_start + best_j] = True

    def _summary(lst):
        if not lst:
            return None
        return {'n': len(lst), 'mean': sum(lst) / len(lst),
                'median': median(lst),
                'p10': sorted(lst)[max(0, len(lst) // 10 - 1)],
                'p90': sorted(lst)[min(len(lst) - 1, 9 * len(lst) // 10)]}

    return {k: {'correct': _summary(v['correct']),
                'wrong': _summary(v['wrong'])} for k, v in by_tl.items()}


# ── S5  Spatial cascade: cluster size of wrong cells ───────────────────────
def s5_failure_cascade(per_cell_records_list):
    """For each puzzle with any wrong cells, measure:
       - n_wrong: number of wrong cells
       - n_wrong_with_wrong_partner: wrong cells whose row/col/box partner
         is ALSO wrong
       - mean_wrong_partners_per_wrong: average number of wrong partners
         per wrong cell
    Then summarize across puzzles.

    Hypothesis: PUMA's wrong cells are more isolated (few wrong partners
    per wrong cell, i.e., independent errors), while Random's wrong cells
    cluster (one wrong commit propagates to violate many partners).
    """
    summary = {
        'n_puzzles': 0,
        'n_puzzles_any_wrong': 0,
        'mean_wrong_per_puzzle': 0.0,
        'mean_clustering_score': 0.0,
        'n_isolated_wrong': 0,
        'n_clustered_wrong': 0,
    }
    total_clustering = 0.0
    total_wrong_cells_overall = 0
    n_puz_with_wrong = 0
    for puz in per_cell_records_list:
        wrong_idxs = {c['idx'] for c in puz['cells'] if not c['correct']}
        if not wrong_idxs:
            summary['n_puzzles'] += 1
            continue
        summary['n_puzzles'] += 1
        n_puz_with_wrong += 1
        total_wrong_cells_overall += len(wrong_idxs)
        cluster_counts = []
        for wi in wrong_idxs:
            n_wrong_partners = sum(1 for p in _PARTNERS[wi] if p in wrong_idxs)
            cluster_counts.append(n_wrong_partners)
            if n_wrong_partners == 0:
                summary['n_isolated_wrong'] += 1
            else:
                summary['n_clustered_wrong'] += 1
        total_clustering += (sum(cluster_counts) / max(len(cluster_counts), 1))
    summary['n_puzzles_any_wrong'] = n_puz_with_wrong
    summary['mean_wrong_per_puzzle'] = (
        total_wrong_cells_overall / max(n_puz_with_wrong, 1))
    summary['mean_clustering_score'] = (
        total_clustering / max(n_puz_with_wrong, 1))
    return summary


# ── CLI driver ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", type=Path, required=True)
    ap.add_argument("--test_data", type=Path, required=True,
                    help="JSON or pickled test data (with meta, solution, puzzle)")
    ap.add_argument("--max_puzzles", type=int, default=1000)
    ap.add_argument("--decode_policies", nargs="+",
                    default=["confidence", "random", "oracle_technique",
                             "n_cands_cp"])
    ap.add_argument("--out_dir", type=Path, default=Path("./decode_analysis_sudoku"))
    ap.add_argument("--device", default=str(DEVICE))
    ap.add_argument("--analyses", nargs="+",
                    default=["s2", "s3", "s4", "s5"])
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    tokenizer = build_tok()

    # Load test data (assumes JSON list / JSONL)
    if str(args.test_data).endswith('.jsonl'):
        test_data = []
        with open(args.test_data) as f:
            for line in f:
                test_data.append(json.loads(line))
    else:
        with open(args.test_data) as f:
            test_data = json.load(f)
    test_data = test_data[:args.max_puzzles]
    print(f"Loaded {len(test_data)} test puzzles")

    # Extract TL meta for each puzzle (used by per_cell_records)
    tl_meta = []
    for d in test_data:
        meta = d.get('meta', [])
        if isinstance(meta, list):
            tl_meta.append([m.get('technique_level') if isinstance(m, dict)
                            else None for m in meta])
        else:
            tl_meta.append(None)

    out_all = {}
    for method in METHODS:
        ckpt = args.ckpt_dir / f"checkpoint_{method}.pt"
        if not ckpt.exists():
            print(f"[skip {method}] no checkpoint at {ckpt}")
            continue
        print(f"\n▶ {method} ({ckpt})")
        model = load_model(ckpt, device)
        out = {}
        # Run evaluate() for each decode policy
        per_policy_records = {}
        for policy in args.decode_policies:
            print(f"  evaluating with {policy}...")
            r = evaluate(model, tokenizer, test_data, decode_policy=policy,
                         batch_size=32, device=device)
            recs = per_cell_records(test_data, r, tl_meta=tl_meta)
            per_policy_records[policy] = recs
            out[f'eval_{policy}'] = {
                'cell_accuracy': r.get('blank_cell_acc'),
                'accuracy': r.get('accuracy'),
            }
            if "s2" in args.analyses:
                out[f's2_violation_rate_{policy}'] = (
                    s2_wrong_cell_constraint_rate(recs))
            if "s3" in args.analyses:
                out[f's3_constraint_loss_by_tl_{policy}'] = (
                    s3_constraint_loss_by_tl(recs))
            if "s5" in args.analyses:
                out[f's5_cascade_{policy}'] = s5_failure_cascade(recs)
        if "s4" in args.analyses:
            print("  [s4] calibration-by-TL probe (confidence decode)...")
            out['s4_calibration_by_tl'] = s4_calibration_by_tl(
                model, tokenizer, test_data, device=device)
        out_all[method] = out
        with open(args.out_dir / f"{method}.json", "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"  saved {args.out_dir / f'{method}.json'}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Cross-method summary table: the headline number is frac_locally_consistent
    # under confidence decode for each method.
    summary = {}
    for method, out in out_all.items():
        s = {}
        for k, v in out.items():
            if k.startswith('s2_violation_rate_') or k.startswith('s5_cascade_'):
                s[k] = v
            elif k == 's4_calibration_by_tl':
                s[k] = v
            elif k.startswith('eval_'):
                s[k] = v
        summary[method] = s
    with open(args.out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved summary: {args.out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
