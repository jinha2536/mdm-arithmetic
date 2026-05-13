"""Re-evaluate trained sudoku checkpoints with constraint-loss metric."""
import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

PROJECT_ROOT = '.'
sys.path.insert(0, PROJECT_ROOT)

import torch

import exp_sudoku
from exp_sudoku import (
    build_tok, evaluate, filter_by_tl4_frac,
    _compute_meta_worker, _make_train_entry,
    ANS_LEN,
)
from core.model import Transformer  # type: ignore


# ── Architecture (from results_sudoku.json config) ─────────────────────────
N_LAYER = 8
N_HEAD = 8
N_EMBD = 256
POS_ENC = 'absolute'
DROPOUT = 0.0
SEED = 42

# ── Hardcoded rating tiers ─────────────────────────────────────────────────
RATING_TIERS = {
    'easy':      (0, 0),
    'medium':    (1, 21),
    'hard':      (22, 41),
    'very_hard': (42, 68),
    'extreme':   (69, 105),
    'top1pct':   (106, 465),
}
N_TEST_PER_TIER_DICT = {
    'easy': 1000, 'medium': 1000, 'hard': 500,
    'very_hard': 200, 'extreme': 100,
}
DEFAULT_N_TEST_PER_TIER = 500

DECODE_POLICIES = ('confidence', 'random', 'n_cands_cp', 'oracle_technique')


VALID_DIGITS = set('123456789')

def constraint_loss(board_str, n=9):
    """Compute soft constraint loss for a 9x9 sudoku board string."""
    s = board_str.ljust(n * n, '.')[: n * n]
    block = int(round(n ** 0.5))
    units = []
    for r in range(n):
        units.append([s[r * n + c] for c in range(n)])         # rows
    for c in range(n):
        units.append([s[r * n + c] for r in range(n)])         # cols
    for br in range(block):
        for bc in range(block):
            units.append([s[(br * block + dr) * n + (bc * block + dc)]
                          for dr in range(block) for dc in range(block)])
    total = 0.0
    for u in units:
        unique_valid = len(set(u) & VALID_DIGITS)
        total += 1 - unique_valid / n
    return total


def evaluate_with_constraint(model, tok, test_data, decode_policy, batch_size, device):
    """Wrap exp_sudoku.evaluate, intercepting tokenizer.decode to capture the predicted 81-char board for each test instance, then aggregate..."""
    captured = []
    orig_decode = tok.decode

    def wrapped_decode(ids):
        out = orig_decode(ids)

        if isinstance(out, str) and len(out) == ANS_LEN:
            captured.append(out)
        return out

    tok.decode = wrapped_decode
    try:
        r = evaluate(model, tok, test_data, decode_policy=decode_policy,
                     batch_size=batch_size, device=device)
    finally:
        tok.decode = orig_decode

    n = r['n']
    if len(captured) != n:
        print(f'    [warn] captured {len(captured)} predicted boards but '
              f'n_results={n}; constraint loss may be incomplete')

    if captured:
        losses = [constraint_loss(b) for b in captured]
        r['constraint_loss_mean'] = sum(losses) / len(losses)
        r['constraint_loss_max'] = max(losses)
        r['constraint_zero_frac'] = sum(1 for x in losses if x == 0) / len(losses)
    else:
        r['constraint_loss_mean'] = None
        r['constraint_loss_max'] = None
        r['constraint_zero_frac'] = None
    return r


# ── Test-only data loader ──────────────────────────────────────────────────
def load_test_only(n_test_per_tier_dict, seed=42, cache_dir='.sudoku_cache'):
    print('  Loading HuggingFace sudoku-extreme...')
    from datasets import load_dataset
    ds = load_dataset('sapientinc/sudoku-extreme', cache_dir=cache_dir)
    test_raw = ds['test']
    train_raw = ds['train']

    rng2 = random.Random(seed + 1000)

    def _sample_by_rating(data, lo, hi, n, rng_inst):
        ratings = data['rating']
        indices = [i for i, r in enumerate(ratings) if lo <= r <= hi]
        if len(indices) > n:
            indices = rng_inst.sample(indices, n)
        if not indices:
            return []
        rows = data.select(indices)
        return [(rows[i]['question'], rows[i]['answer'], rows[i]['rating'],
                 rows[i].get('source', ''))
                for i in range(len(rows))]

    test_tiers = {}
    for tier_name, (lo, hi) in RATING_TIERS.items():
        n_tier = n_test_per_tier_dict.get(tier_name, DEFAULT_N_TEST_PER_TIER)
        samples = _sample_by_rating(test_raw, lo, hi, n_tier, rng2)
        if not samples:
            print(f'    {tier_name}: empty in test split, falling back to train')
            samples = _sample_by_rating(train_raw, lo, hi, n_tier, rng2)
        test_tiers[tier_name] = samples
        print(f'    {tier_name} [{lo},{hi}]: {len(samples)} puzzles')

    test_tuples = [t for ts in test_tiers.values() for t in ts]
    print(f'  Computing metadata for {len(test_tuples)} test puzzles...')
    t0 = time.time()
    workers = min(os.cpu_count() or 1, 16)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        test_entries = list(pool.map(_compute_meta_worker, test_tuples,
                                     chunksize=max(1, len(test_tuples) // (workers * 4))))
    print(f'  Metadata computed in {time.time()-t0:.1f}s')

    test_data = {}
    idx = 0
    for tn, ts in test_tiers.items():
        test_data[tn] = test_entries[idx:idx + len(ts)]
        idx += len(ts)
    return test_data


def load_model(ckpt_path, vocab_size, device):
    sd = torch.load(ckpt_path, map_location=device)
    if isinstance(sd, dict) and 'model_state_dict' in sd:
        sd = sd['model_state_dict']
    elif isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
    sd = {k.replace('module.', '', 1): v for k, v in sd.items()}

    block_size = sd['wpe.weight'].shape[0]

    model = Transformer(
        vocab_size=vocab_size, block_size=block_size,
        n_layer=N_LAYER, n_head=N_HEAD, n_embd=N_EMBD,
        dropout=DROPOUT, is_causal=False, pos_enc=POS_ENC,
    ).to(device)
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        '--checkpoint_dir',
        default='/path/to/results/exp_sudoku',
        type=Path,
    )
    ap.add_argument('--out', type=Path, default=Path('sudoku_reeval_results.json'))
    ap.add_argument('--methods', nargs='+', default=['random', 'papl', 'puma'])
    ap.add_argument('--device', default=None)
    args = ap.parse_args()

    device = torch.device(args.device or
                          ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f'Device: {device}')

    print('\n[1/4] Loading test data (no train)...')
    t0 = time.time()
    test_tiers = load_test_only(N_TEST_PER_TIER_DICT, seed=SEED)
    all_test = [d for ts in test_tiers.values() for d in ts]
    print(f'  Total test puzzles: {len(all_test)} ({time.time()-t0:.1f}s)')

    print('\n[2/4] Filtering strata...')
    tl4_subset = filter_by_tl4_frac(all_test, 0.95)
    print(f'  TL4 frac >= 0.95: {len(tl4_subset)} puzzles')

    lo, hi = RATING_TIERS['top1pct']
    top1pct_subset = [d for d in all_test if lo <= d.get('rating', 0) <= hi]
    print(f'  rating = top1pct ({lo}-{hi}): {len(top1pct_subset)} puzzles')

    tok = build_tok()

    print('\n[3/4] Running evaluations...')
    out = {
        'config': {
            'rating_tiers': RATING_TIERS,
            'n_test_per_tier_dict': N_TEST_PER_TIER_DICT,
            'seed': SEED,
            'overall_n': len(all_test),
            'tl4_subset_n': len(tl4_subset),
            'top1pct_subset_n': len(top1pct_subset),
        },
        'metric_definitions': {
            'constraint_loss': (
                'Soft constraint loss summed over 27 units (9 rows + 9 cols + '
                '9 blocks). For each unit u: loss_u = 1 - |unique_valid_digits(u)| / 9. '
                'Range [0, 24]: 0 = perfectly valid board, 24 = all identical digits. '
                'Invalid tokens are excluded from unique counts, so they penalize at '
                'least as much as duplicates.'
            ),
            'constraint_zero_frac': (
                'Fraction of generated boards with constraint_loss == 0, i.e., '
                'fully sudoku-valid. Note: a valid board is not necessarily the '
                'correct solution to the puzzle; this counts structural validity only.'
            ),
        },
        'results': {},
    }

    strata = {
        'overall': all_test,
        'tl4_geq_0.95': tl4_subset,
        'rating_top1pct': top1pct_subset,
    }

    for method in args.methods:
        ckpt_path = args.checkpoint_dir / f'checkpoint_{method}.pt'
        if not ckpt_path.exists():
            print(f'  [skip] {ckpt_path} not found')
            continue
        print(f'\n  === {method} ({ckpt_path.name}) ===')
        model = load_model(ckpt_path, len(tok), device)

        for stratum_name, subset in strata.items():
            if not subset:
                print(f'    [{stratum_name}] empty subset, skipping')
                continue
            for decode in DECODE_POLICIES:
                t0 = time.time()
                r = evaluate_with_constraint(
                    model, tok, subset, decode_policy=decode,
                    batch_size=32, device=device)
                dt = time.time() - t0
                key = f'{method}_{stratum_name}_{decode}'
                out['results'][key] = {
                    'blank_cell_acc': r['blank_cell_acc'],
                    'accuracy': r['accuracy'],
                    'constraint_loss_mean': r['constraint_loss_mean'],
                    'constraint_loss_max': r['constraint_loss_max'],
                    'constraint_zero_frac': r['constraint_zero_frac'],
                    'n': r['n'],
                    'time_seconds': round(dt, 1),
                }
                cl = r['constraint_loss_mean']
                cl_str = f'{cl:.3f}' if cl is not None else 'N/A'
                zf = r['constraint_zero_frac']
                zf_str = f'{zf:.3f}' if zf is not None else 'N/A'
                print(f'    [{stratum_name}] {decode:17s}: '
                      f'cell={r["blank_cell_acc"]:.4f} '
                      f'puzzle={r["accuracy"]:.4f} '
                      f'closs={cl_str} '
                      f'zero={zf_str} '
                      f'(n={r["n"]}, {dt:.1f}s)')

        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    print(f'\n[4/4] Writing results to {args.out}')
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)

    # ── Summary tables ──
    print('\n=== Cell-level accuracy (blank cells) ===')
    print(f'{"stratum":>20} {"decode":>17} {"random":>8} {"papl":>8} {"puma":>8}')
    for stratum_name in strata:
        for decode in DECODE_POLICIES:
            row = [stratum_name, decode]
            for m in ['random', 'papl', 'puma']:
                key = f'{m}_{stratum_name}_{decode}'
                v = out['results'].get(key, {}).get('blank_cell_acc')
                row.append(f'{v:.4f}' if v is not None else '   N/A ')
            print(f'{row[0]:>20} {row[1]:>17} {row[2]:>8} {row[3]:>8} {row[4]:>8}')

    print('\n=== Mean constraint loss (lower is better; 0 = valid board) ===')
    print(f'{"stratum":>20} {"decode":>17} {"random":>8} {"papl":>8} {"puma":>8}')
    for stratum_name in strata:
        for decode in DECODE_POLICIES:
            row = [stratum_name, decode]
            for m in ['random', 'papl', 'puma']:
                key = f'{m}_{stratum_name}_{decode}'
                v = out['results'].get(key, {}).get('constraint_loss_mean')
                row.append(f'{v:.3f}' if v is not None else '  N/A')
            print(f'{row[0]:>20} {row[1]:>17} {row[2]:>8} {row[3]:>8} {row[4]:>8}')

    print('\n=== Fraction of structurally-valid boards (constraint_zero_frac) ===')
    print(f'{"stratum":>20} {"decode":>17} {"random":>8} {"papl":>8} {"puma":>8}')
    for stratum_name in strata:
        for decode in DECODE_POLICIES:
            row = [stratum_name, decode]
            for m in ['random', 'papl', 'puma']:
                key = f'{m}_{stratum_name}_{decode}'
                v = out['results'].get(key, {}).get('constraint_zero_frac')
                row.append(f'{v:.3f}' if v is not None else '  N/A')
            print(f'{row[0]:>20} {row[1]:>17} {row[2]:>8} {row[3]:>8} {row[4]:>8}')


if __name__ == '__main__':
    main()
