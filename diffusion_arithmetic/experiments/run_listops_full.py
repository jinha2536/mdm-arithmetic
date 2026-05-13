"""run_listops_full.py — train 3 seeds + analyze + aggregate, one session.

Usage in Colab (after %cd .../experiments):
    %run run_listops_full.py

What this does, end-to-end:
    1. Detect repo root, mount Drive, inherit DRIVE_BASE from core.train_utils
    2. Train ListOps with seeds 42/43/44 in --train-only mode
       (skips already-completed seeds — resumable)
    3. Run wrong-commit analysis on each seed's checkpoints
    4. Aggregate mean ± std across seeds
    5. Save per-seed JSON + 3seed_average JSON to Drive

Approximate wall time: ~18hr total (6hr × 3 seeds), all on a single Colab A100.
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
DOMAIN = 'listops'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"CKPT_BASE = {CKPT_BASE}\nDRIVE_OUT = {DRIVE_OUT}")
print(f"device = {DEVICE}, domain = {DOMAIN}, seeds = {SEEDS}")


# ─── Helper: aggregate mean ± std across seeds at every numeric leaf ──────
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
    """Locate <DRIVE_BASE>/exp_listops_s<seed>/ with checkpoints."""
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
import exp_listops as ex  # noqa: E402

# Toggle module-level flag so run() skips eval/continuation/post-loop work.
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
import listops_decode_analysis as lda  # noqa: E402

# Build a shared test set (same seed across all training seeds) so variance is
# measured purely over training, not over test sampling.
tok = ex.build_tok()
test_entries = []
for d in [1, 2, 3, 4, 5]:
    test_entries.extend(ex.gen_random_listops_test(300, depth=d, seed=7000 + d))
print(f"test set: {len(test_entries)} entries across depths 1-5")


def analyze_one_seed(ckpt_dir):
    out = {}
    per_inst_cache = None
    for method in METHODS:
        ckpt = ckpt_dir / f'checkpoint_{method}.pt'
        if not ckpt.exists():
            print(f"  [skip {method}] {ckpt} not found")
            continue
        t0 = time.time()
        model = lda.load_model(ckpt, DEVICE, n_head_override=3)
        per_instance = lda.l1_per_instance(model, tok, test_entries, device=DEVICE)
        analyses = {
            'categorize': lda.l2_categorize(per_instance),
            'shortcut_probe': lda.l3_l4_leaf_only_probe(per_instance),
            'calibration': lda.l5_calibration(model, tok, test_entries, device=DEVICE),
        }
        analyses['shortcut_probe'].pop('examples', None)  # drop bulky lists
        out[method] = analyses
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
        for dbin in ['d=1', 'd=2', 'd=3', 'd=4', 'd=5']:
            cat = m.get('categorize', {}).get(dbin, {})
            if cat:
                print(f"    {dbin}: both={_fmt(cat.get('both_correct'))}, "
                      f"only_oracle={_fmt(cat.get('only_oracle'))}, "
                      f"only_conf={_fmt(cat.get('only_conf'))}, "
                      f"neither={_fmt(cat.get('neither'))}")
        sc = m.get('shortcut_probe', {}).get('summary', {})
        for dbin, v in (sc.items() if isinstance(sc, dict) else []):
            if not isinstance(v, dict):
                continue
            n_w = v.get('n_wrong')
            n_s = v.get('n_shortcut_match')
            if (isinstance(n_w, dict) and n_w.get('mean', 0) > 0
                    and isinstance(n_s, dict)):
                rate = n_s['mean'] / n_w['mean']
                print(f"    shortcut/{dbin}: n_wrong={_fmt(n_w)}, "
                      f"match_rate≈{rate:.3f}")

print(f"\nDone. Output: {DRIVE_OUT}")
