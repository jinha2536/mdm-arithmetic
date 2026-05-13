"""run_countdown_full.py — train 3 seeds + analyze + aggregate, one session.

Usage in Colab (after %cd .../experiments):
    %run run_countdown_full.py

Approximate wall time: ~30hr total (10hr × 3 seeds), countdown is the largest
model (21M params, 12 layers). On Colab A100 may need multiple sessions if
each one limits to ~12hr; the training loop is resumable across runs (skips
seeds whose 3 checkpoints already exist).
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
DOMAIN = 'countdown'
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
import exp_countdown as ex  # noqa: E402

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
import countdown_decode_analysis as cda  # noqa: E402

tok = ex.build_tok()

# Mirror the experiment's test loader so we hit the same test set the paper
# reports on.
test_path = os.path.join(ex.DATA_DIR, ex.TEST_FILE)
test_entries = []
with open(test_path) as f:
    for line in f:
        test_entries.append(json.loads(line))
print(f"loaded {len(test_entries)} test entries from {test_path}")

buckets = cda.build_test_buckets(test_entries, bucket_axis='mult')
sampled = []
for bn in ex.MULT_BINS:
    sampled.extend(buckets.get(bn, [])[:300])
print(f"sampled {len(sampled)} across multiplicity buckets")


def analyze_one_seed(ckpt_dir):
    out = {}
    for method in METHODS:
        ckpt = ckpt_dir / f'checkpoint_{method}.pt'
        if not ckpt.exists():
            print(f"  [skip {method}] {ckpt} not found")
            continue
        t0 = time.time()
        model = cda.load_model(ckpt, DEVICE, n_head_override=12)
        per_instance = cda.c1_per_instance(model, tok, sampled, device=DEVICE)
        analyses = {
            'categorize': cda.c2_categorize(per_instance),
            'failure_dissection': cda.c3_failure_dissection(
                model, tok, sampled, device=DEVICE),
            'validity': cda.c4_validity_aggregate(per_instance),
            'calibration': cda.c5_calibration(model, tok, sampled, device=DEVICE),
        }
        analyses['failure_dissection'].pop('examples', None)
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
        cat = m.get('categorize', {})
        for mb in ['m=1-3', 'm=4-10', 'm=11+']:
            v = cat.get(mb, {})
            if v:
                print(f"    {mb}: only_oracle={_fmt(v.get('only_oracle'))}, "
                      f"only_conf={_fmt(v.get('only_conf'))}, "
                      f"both={_fmt(v.get('both_correct'))}, "
                      f"neither={_fmt(v.get('neither'))}")
        pt = m.get('failure_dissection', {}).get('pos_type_dist', {})
        for mb, v in (pt.items() if isinstance(pt, dict) else []):
            if isinstance(v, dict):
                pieces = ', '.join(
                    f"{k}={_fmt(vv)}" for k, vv in v.items()
                    if isinstance(vv, dict) and 'mean' in vv)
                if pieces:
                    print(f"    pos_type/{mb}: {pieces}")
        # Validity profile aggregate — the "fluent dead-end chain" rate is the
        # key signature for confidence-shortcut Countdown failures.
        val = m.get('validity', {})
        for k, v in (val.items() if isinstance(val, dict) else []):
            if not isinstance(v, dict):
                continue
            n_wrong = v.get('wrong')
            n_fluent = v.get('wrong_fluent_deadend')
            if (isinstance(n_wrong, dict) and n_wrong.get('mean', 0) > 0
                    and isinstance(n_fluent, dict)):
                rate = n_fluent['mean'] / n_wrong['mean']
                print(f"    validity/{k}: wrong={_fmt(n_wrong)}, "
                      f"fluent_deadend≈{rate:.3f}")

print(f"\nDone. Output: {DRIVE_OUT}")
