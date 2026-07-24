"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Multi-seed aggregation for maze / countdown (rebuttal error bars)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Reads all results_seed{N}.json files in an experiment directory (the
per-seed files written by the patched exp_maze.py / exp_countdown.py),
collects every accuracy-type scalar under the final_* keys, and reports
n / mean / std / min / max per metric across seeds. Optionally emits
LaTeX-ready "mean ± std" cells.

Usage (Colab, after mounting Drive):
    %run experiments/aggregate_multiseed.py \
        /content/drive/MyDrive/diffusion-arithmetic-results/exp_maze_ms1

    # Only the paper-table keys:
    %run experiments/aggregate_multiseed.py <dir> --keys 'backbone_sweep|standard'
    %run experiments/aggregate_multiseed.py <dir> --keys 'acc_m=|selective_50'

    # LaTeX cells:
    %run experiments/aggregate_multiseed.py <dir> --keys backbone_sweep --latex

Fields collected per final_* entry: 'accuracy' and any numeric field whose
name starts with 'acc' (e.g. countdown's acc_m=1, acc_m=21+, acc_plan).
Use --fields to add more (e.g. --fields valid_rate strict_valid_rate).
"""
import argparse, glob, json, math, os, re, sys


def _std(vals):
    n = len(vals)
    if n < 2:
        return 0.0
    m = sum(vals) / n
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1))


def collect(path, extra_fields):
    """results_seed{N}.json → {metric_key: scalar}"""
    with open(path) as f:
        d = json.load(f)
    out = {}
    for k, v in d.items():
        if not k.startswith('final_') or not isinstance(v, dict):
            continue
        base = k[len('final_'):]
        for fk, fv in v.items():
            take = (fk == 'accuracy' or fk.startswith('acc')
                    or fk in extra_fields)
            if take and isinstance(fv, (int, float)) and not isinstance(fv, bool):
                name = base if fk == 'accuracy' else f'{base}.{fk}'
                out[name] = float(fv)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('exp_dir', help='Directory containing results_seed*.json')
    p.add_argument('--keys', type=str, default=None,
                   help='Regex filter on metric names')
    p.add_argument('--fields', nargs='+', default=[],
                   help="Extra scalar fields to collect (e.g. valid_rate)")
    p.add_argument('--latex', action='store_true',
                   help='Also print LaTeX "mean ± std" cells')
    p.add_argument('--decimals', type=int, default=3)
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.exp_dir, 'results_seed*.json')))
    if not files:
        sys.exit(f"No results_seed*.json found in {args.exp_dir}")

    per_seed = {}
    for fp in files:
        m = re.search(r'results_seed(\d+)', os.path.basename(fp))
        seed = int(m.group(1)) if m else fp
        per_seed[seed] = collect(fp, set(args.fields))
        print(f"  loaded seed {seed}: {len(per_seed[seed])} metrics  ({fp})")

    seeds = sorted(per_seed)
    all_keys = sorted({k for d in per_seed.values() for k in d})
    if args.keys:
        rx = re.compile(args.keys)
        all_keys = [k for k in all_keys if rx.search(k)]

    dec = args.decimals
    w = max((len(k) for k in all_keys), default=10)
    seed_hdr = ' '.join(f"s{str(s):>6s}" for s in seeds)
    print(f"\n{'metric':<{w}s}  n  {'mean':>7s} {'std':>7s} {'min':>7s} {'max':>7s}   {seed_hdr}")
    print('─' * (w + 42 + 8 * len(seeds)))
    rows = []
    for k in all_keys:
        vals = [per_seed[s][k] for s in seeds if k in per_seed[s]]
        if not vals:
            continue
        m, sd = sum(vals) / len(vals), _std(vals)
        cells = ' '.join(
            f"{per_seed[s][k]:>7.{dec}f}" if k in per_seed[s] else f"{'—':>7s}"
            for s in seeds)
        print(f"{k:<{w}s}  {len(vals)}  {m:>7.{dec}f} {sd:>7.{dec}f} "
              f"{min(vals):>7.{dec}f} {max(vals):>7.{dec}f}   {cells}")
        rows.append((k, len(vals), m, sd))

    if args.latex:
        print("\n% LaTeX cells (mean ± std, n per row in comment)")
        for k, n, m, sd in rows:
            print(f"{k}: ${m:.{dec}f} \\pm {sd:.{dec}f}$  % n={n}")


if __name__ == '__main__':
    main()
