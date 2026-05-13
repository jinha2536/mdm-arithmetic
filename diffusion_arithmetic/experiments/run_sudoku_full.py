"""run_sudoku_full.py — train 3 seeds + analyze + aggregate, one session.

Usage in Colab (after %cd .../experiments):
    %run run_sudoku_full.py

Approximate wall time: ~24-30hr total (8-10hr × 3 seeds) on a Colab A100;
sudoku is 6.4M params with 300k iterations and batch 256.  Resumable across
sessions — seeds with all 3 method checkpoints already present are skipped.
"""
import json
import os
import sys
import time
import traceback
from pathlib import Path
from statistics import mean, pstdev

import torch

# ─── Cell 1: setup ─────────────────────────────────────────────────────────
_THIS = Path(__file__).resolve() if '__file__' in dir() else Path(os.getcwd())
EXP_DIR = _THIS.parent if _THIS.is_file() else _THIS
if EXP_DIR.name != 'experiments':
    EXP_DIR = Path(os.getcwd()).resolve()
REPO_ROOT = EXP_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(EXP_DIR))
print(f"REPO_ROOT = {REPO_ROOT}\nEXP_DIR   = {EXP_DIR}")

try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
except (ImportError, ModuleNotFoundError):
    print("[info] not in Colab — Drive not mounted")

from core.train_utils import DRIVE_BASE  # noqa: E402

CKPT_BASE = Path(DRIVE_BASE)
DRIVE_OUT = Path(DRIVE_BASE) / 'analysis_3seed'
DRIVE_OUT.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 43, 44]
METHODS = ['random', 'papl', 'puma']
DOMAIN = 'sudoku'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"CKPT_BASE = {CKPT_BASE}\nDRIVE_OUT = {DRIVE_OUT}")
print(f"device = {DEVICE}, domain = {DOMAIN}, seeds = {SEEDS}")


def aggregate_across_seeds(seed_dicts):
    if not seed_dicts:
        return None
    seed_dicts = [d for d in seed_dicts if d is not None]
    if not seed_dicts:
        return None
    if all(isinstance(d, (int, float)) and not isinstance(d, bool)
           for d in seed_dicts):
        vals = list(seed_dicts)
        return {'mean': mean(vals),
                'std': pstdev(vals) if len(vals) > 1 else 0.0,
                'per_seed': vals, 'n': len(vals)}
    if all(isinstance(d, list) for d in seed_dicts):
        lengths = {len(d) for d in seed_dicts}
        if len(lengths) == 1 and all(
                all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in d)
                for d in seed_dicts):
            L = lengths.pop()
            return [aggregate_across_seeds([d[i] for d in seed_dicts])
                    for i in range(L)]
        return {'per_seed': seed_dicts}
    if all(isinstance(d, dict) for d in seed_dicts):
        keys = set()
        for d in seed_dicts:
            keys.update(d.keys())
        return {k: aggregate_across_seeds(
                    [d.get(k) for d in seed_dicts if isinstance(d, dict) and k in d])
                for k in keys}
    return {'per_seed': list(seed_dicts)}


def find_seed_dir(seed):
    for c in [CKPT_BASE / f'exp_{DOMAIN}_s{seed}',
              CKPT_BASE / f'{DOMAIN}_s{seed}']:
        if c.exists() and any(c.glob('checkpoint_*.pt')):
            return c
    if CKPT_BASE.exists():
        for d in CKPT_BASE.iterdir():
            if (d.is_dir() and DOMAIN in d.name.lower() and f's{seed}' in d.name
                    and any(d.glob('checkpoint_*.pt'))):
                return d
    return None


def seed_has_all_checkpoints(seed):
    d = find_seed_dir(seed)
    if d is None:
        return False
    return all((d / f'checkpoint_{m}.pt').exists() for m in METHODS)


# ─── Cell 2: train 3 seeds ─────────────────────────────────────────────────
print(f"\n{'═' * 70}\n  TRAIN: {DOMAIN}\n{'═' * 70}")
import exp_sudoku as ex  # noqa: E402

ex.TRAIN_ONLY = True

for seed in SEEDS:
    if seed_has_all_checkpoints(seed):
        print(f"\n[seed {seed}] all 3 method checkpoints exist, skipping training")
        continue
    print(f"\n[seed {seed}] training ({DOMAIN}, TRAIN_ONLY=True)...")
    ex.SEED = seed
    t0 = time.time()
    try:
        ex.run(tag=f's{seed}')
    except Exception as e:
        print(f"  ERROR during training seed {seed}: {e}")
        traceback.print_exc()
        continue
    print(f"  ✓ seed {seed} training done in {(time.time() - t0) / 60:.1f} min")


# ─── Cell 3: analyze 3 seeds ───────────────────────────────────────────────
print(f"\n{'═' * 70}\n  ANALYZE: {DOMAIN}\n{'═' * 70}")
import sudoku_decode_analysis as sda  # noqa: E402

tok = ex.build_tok()

# Mirror the experiment's test loader.
test_path = Path(ex.DATA_DIR) / ex.TEST_FILE
test_data = []
if str(test_path).endswith('.jsonl'):
    with open(test_path) as f:
        for line in f:
            test_data.append(json.loads(line))
else:
    with open(test_path) as f:
        test_data = json.load(f)
test_data = test_data[:500]
print(f"loaded {len(test_data)} test puzzles from {test_path}")

# Per-cell TL metadata (needed for stratifying violation rates by technique level)
tl_meta = []
for d in test_data:
    meta = d.get('meta', [])
    if isinstance(meta, list):
        tl_meta.append([m.get('technique_level') if isinstance(m, dict) else None
                        for m in meta])
    else:
        tl_meta.append(None)

decode_policies = ['confidence', 'random', 'oracle_technique', 'n_cands_cp']


def analyze_one_seed(ckpt_dir):
    out = {}
    for method in METHODS:
        ckpt = ckpt_dir / f'checkpoint_{method}.pt'
        if not ckpt.exists():
            print(f"  [skip {method}] {ckpt} not found")
            continue
        t0 = time.time()
        model = sda.load_model(ckpt, DEVICE)
        method_out = {}
        for policy in decode_policies:
            print(f"    {method} × {policy}...", end=' ', flush=True)
            r = ex.evaluate(model, tok, test_data, decode_policy=policy,
                            batch_size=32, device=DEVICE)
            recs = sda.per_cell_records(test_data, r, tl_meta=tl_meta)
            method_out[f'eval_{policy}'] = {
                'cell_accuracy': r.get('blank_cell_acc'),
                'accuracy': r.get('accuracy'),
            }
            method_out[f'violation_rate_{policy}'] = (
                sda.s2_wrong_cell_constraint_rate(recs))
            method_out[f'constraint_loss_by_tl_{policy}'] = (
                sda.s3_constraint_loss_by_tl(recs))
            method_out[f'cascade_{policy}'] = sda.s5_failure_cascade(recs)
            ca = r.get('blank_cell_acc')
            print(f"acc={ca:.4f}" if ca is not None else "done")
        method_out['calibration_by_tl'] = sda.s4_calibration_by_tl(
            model, tok, test_data, device=DEVICE, max_puzzles=200)
        out[method] = method_out
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  ✓ {method} analysis done in {time.time() - t0:.1f}s")
    return out


per_seed_results = []
for seed in SEEDS:
    ckpt_dir = find_seed_dir(seed)
    if ckpt_dir is None:
        print(f"\n[seed {seed}] no checkpoint dir found, skipping analysis")
        continue
    print(f"\n[seed {seed}] analyzing {ckpt_dir}")
    try:
        res = analyze_one_seed(ckpt_dir)
    except Exception as e:
        print(f"  ERROR during analysis seed {seed}: {e}")
        traceback.print_exc()
        continue
    if not res:
        continue
    per_seed_results.append({'seed': seed, **res})
    out_path = DRIVE_OUT / f'{DOMAIN}_s{seed}_analysis.json'
    with open(out_path, 'w') as f:
        json.dump(res, f, indent=2, default=str)
    print(f"  saved {out_path}")


# ─── Cell 4: aggregate ─────────────────────────────────────────────────────
if not per_seed_results:
    print(f"\nNo seeds completed for {DOMAIN}; nothing to aggregate.")
else:
    print(f"\n{'═' * 70}\n  AGGREGATE: {DOMAIN} ({len(per_seed_results)} seeds)"
          f"\n{'═' * 70}")
    aggregated = {}
    for method in METHODS:
        per_method = [r.get(method) for r in per_seed_results if method in r]
        per_method = [m for m in per_method if m]
        if per_method:
            aggregated[method] = aggregate_across_seeds(per_method)
    aggregated['_meta'] = {
        'domain': DOMAIN,
        'seeds_used': [r['seed'] for r in per_seed_results],
        'n_seeds': len(per_seed_results),
        'methods': [m for m in METHODS if m in aggregated],
    }
    agg_path = DRIVE_OUT / f'{DOMAIN}_3seed_average.json'
    with open(agg_path, 'w') as f:
        json.dump(aggregated, f, indent=2, default=str)
    print(f"✓ saved {agg_path}")

    # ─── Cell 5: summary ───────────────────────────────────────────────────
    def _fmt(stat):
        if not isinstance(stat, dict) or 'mean' not in stat:
            return str(stat)
        return f"{stat['mean']:.4f} ± {stat['std']:.4f}"

    print(f"\n{'─' * 70}\n  SUMMARY\n{'─' * 70}")
    for method in METHODS:
        if method not in aggregated:
            continue
        m = aggregated[method]
        print(f"\n  ── {method} ──")
        for policy in decode_policies:
            ev = m.get(f'eval_{policy}', {})
            vr = m.get(f'violation_rate_{policy}', {})
            print(f"    {policy:18s}: cell_acc={_fmt(ev.get('cell_accuracy'))}, "
                  f"frac_violating={_fmt(vr.get('frac_violating'))}, "
                  f"frac_loc_consistent={_fmt(vr.get('frac_locally_consistent'))}")

print(f"\nDone. Output: {DRIVE_OUT}")
