"""Compact dashboard for addition decode-analysis JSONs.

Aggregates analysis_seed{N}_{method}_iter{NNNNNN}.json files across multiple
experiment directories (one per training condition) and prints four pivot
tables that map directly to the paper's main claims:

    T1  conf_acc vs lsb_acc per chain bucket  →  trajectory narrowing (B)
    T2  Kendall τ vs LSB order per chain      →  PUMA reveal order deviation
    T3  top-1 confidence on wrong commits at chain-MSB g/k cells
                                              →  calibration view of 405/406 mechanism
    T4  only_lsb share among conf failures    →  recoverable via decode swap

Each table is a flat printout: rows = (condition, method, iter), columns = chain.
Output is plain text aligned for terminal / notebook reading. No plots.

Usage:
    %run summarize_decode.py \\
        --analysis_dirs DRIVE/exp_addition_result42/analysis \\
                        DRIVE/exp_addition_result137/analysis \\
                        DRIVE/exp_addition_result2027/analysis \\
                        DRIVE/exp_addition_fresh/analysis \\
                        DRIVE/exp_addition_largeN/analysis \\
        --labels result42 result137 result2027 fresh largeN

Filters:
    --methods random puma         # subset of methods to include
    --iters 100000 200000 300000  # subset of training iters
    --chains 8 16 24 28           # subset of chain buckets
    --metrics t1 t3               # subset of tables to print
"""
import argparse
import json
import re
from pathlib import Path
from collections import defaultdict


_FNAME_RE = re.compile(r'^analysis_seed(\d+)_([a-z_]+)_iter(\d+)\.json$')
# Legacy format: analysis_{method}.json (no seed/iter in name).
# Used by addition_decode_analysis.py --legacy-single. We tag these with
# seed=0 and iter=0 so they sort first and are visually distinct.
_FNAME_RE_LEGACY = re.compile(r'^analysis_([a-z_]+)\.json$')


def _discover(d: Path):
    """Walk dir, yield (seed, method, iter, path) for each parseable file.

    Recognizes both the new multi-checkpoint format
    (analysis_seed{N}_{method}_iter{NNNNNN}.json) and the legacy
    single-checkpoint format (analysis_{method}.json, mapped to seed=0
    iter=0).
    """
    out = []
    seen_legacy_methods = set()
    for f in sorted(d.glob('analysis_*.json')):
        m = _FNAME_RE.match(f.name)
        if m:
            out.append((int(m[1]), m[2], int(m[3]), f))
            continue
        m = _FNAME_RE_LEGACY.match(f.name)
        if m and m[1] not in ('summary',):
            # Avoid double-counting if both legacy and new exist for same method
            if m[1] in seen_legacy_methods:
                continue
            seen_legacy_methods.add(m[1])
            out.append((0, m[1], 0, f))
    return out


def _load_all(dirs, labels, methods=None, iters=None):
    """Return list of (label, seed, method, iter, dict).

    Each entry is one analysis JSON, tagged by its parent dir's label.
    """
    rows = []
    for label, d in zip(labels, dirs):
        d = Path(d)
        if not d.exists():
            print(f"[warn] {d} does not exist, skipping label={label}")
            continue
        files = _discover(d)
        for seed, mt, it, path in files:
            if methods and mt not in methods: continue
            if iters and it not in iters:     continue
            try:
                with open(path) as f:
                    data = json.load(f)
            except Exception as e:
                print(f"[warn] cannot read {path}: {e}")
                continue
            rows.append((label, seed, mt, it, data))
    return rows


def _chain_key(k):
    """analysis dict uses 'chain_4', 'chain_8', etc."""
    return f"chain_{k}"


def _safe_div(a, b):
    return float(a) / float(b) if b else float('nan')


def _fmt(v, w=6, prec=3):
    """Format float or NaN consistently to fixed width."""
    if v is None or (isinstance(v, float) and (v != v)):
        return f"{'--':>{w}}"
    if abs(v) >= 100:
        return f"{v:>{w}.0f}"
    if abs(v) >= 10:
        return f"{v:>{w}.1f}"
    return f"{v:>{w}.{prec}f}"


def _sort_key(label, seed, mt, it, label_order):
    """Sort: by condition order, then method order, then iter, then seed."""
    method_order = {'random': 0, 'papl': 1, 'puma': 2}
    li = label_order.index(label) if label in label_order else 99
    mi = method_order.get(mt, 99)
    return (li, mi, it, seed)


# ─────────────────────────────────────────────────────────────────────────────
# Table extractors
# ─────────────────────────────────────────────────────────────────────────────

def _t1_acc(d, chain):
    """conf_acc and lsb_acc from a1.cross for a chain bucket.
    Returns (conf_acc, lsb_acc, n) or (nan, nan, 0) if missing.
    """
    a1 = d.get('a1', {})
    bucket = a1.get(_chain_key(chain))
    if not bucket:
        return float('nan'), float('nan'), 0
    cr = bucket.get('cross', {})
    bc = cr.get('both_correct', 0)
    ol = cr.get('only_lsb', 0)
    oc = cr.get('only_conf', 0)
    n  = bucket.get('n', bc + ol + oc + cr.get('neither', 0))
    return _safe_div(bc + oc, n), _safe_div(bc + ol, n), n


def _t1_lsb_share(d, chain):
    """only_lsb / (only_lsb + only_conf): the share of conf failures that
    LSB ordering would have recovered. Higher = more trajectory narrowing.
    """
    a1 = d.get('a1', {})
    bucket = a1.get(_chain_key(chain))
    if not bucket: return float('nan')
    cr = bucket.get('cross', {})
    ol = cr.get('only_lsb', 0)
    oc = cr.get('only_conf', 0)
    return _safe_div(ol, ol + oc)


def _t2_tau(d, chain):
    """Kendall τ of PUMA-style reveal order vs LSB order, per chain bucket."""
    a2 = d.get('a2', {})
    bucket = a2.get(_chain_key(chain))
    if not bucket: return float('nan')
    return bucket.get('overall_mean_tau', float('nan'))


def _t3_top1_wrong(d, role='g'):
    """Mean top-1 confidence on WRONG commits at chain-MSB g/k cells,
    aggregated over the '>=29' chain range in a4.
    """
    a4 = d.get('a4', {})
    bucket = a4.get('>=29')
    if not bucket: return float('nan'), 0
    role_d = bucket.get(role, {})
    nw = role_d.get('n_wrong', 0)
    mc = role_d.get('mean_conf_wrong')
    return (mc if mc is not None else float('nan'), nw)


# ─────────────────────────────────────────────────────────────────────────────
# Printers
# ─────────────────────────────────────────────────────────────────────────────

def _print_header(title, subtitle=None):
    bar = '─' * 78
    print(f"\n{bar}\n {title}")
    if subtitle:
        print(f"   {subtitle}")
    print(bar)


def _print_t1(rows, chains, label_order):
    """conf_acc vs lsb_acc per chain.

    For each (label, method, iter, seed) we print two rows: conf and lsb.
    Side-by-side per chain so the gap (= trajectory narrowing) is visible.
    """
    _print_header(
        "T1  conf_acc / lsb_acc  per chain bucket",
        "Each row pair = same model. If lsb > conf, trajectory narrowing.")
    rows_sorted = sorted(rows, key=lambda r: _sort_key(*r[:4], label_order))
    head = f"{'condition':>11}  {'seed':>4}  {'method':>6}  {'iter':>7}  {'pol':>4} "
    head += "".join(f"{f'≥{c}':>8}" for c in chains)
    print(head)
    print('─' * len(head))
    last_key = None
    for label, seed, mt, it, data in rows_sorted:
        key = (label, seed, mt, it)
        if last_key is not None and last_key != key:
            pass  # no separator; pairs are tight
        confs, lsbs = [], []
        for c in chains:
            ca, la, _ = _t1_acc(data, c)
            confs.append(ca); lsbs.append(la)
        base = f"{label:>11}  {seed:>4}  {mt:>6}  {it:>7}  "
        print(base + f"{'conf':>4} " + "".join(_fmt(x, w=8) for x in confs))
        print(base + f"{'lsb':>4} "  + "".join(_fmt(x, w=8) for x in lsbs))
        last_key = key


def _print_t2(rows, chains, label_order):
    """Kendall τ vs LSB order per chain."""
    _print_header(
        "T2  Kendall τ of PUMA-style reveal order vs LSB order",
        "Lower τ = farther from LSB-first. (Most relevant for PUMA.)")
    rows_sorted = sorted(rows, key=lambda r: _sort_key(*r[:4], label_order))
    head = f"{'condition':>11}  {'seed':>4}  {'method':>6}  {'iter':>7} "
    head += "".join(f"{f'≥{c}':>8}" for c in chains)
    print(head)
    print('─' * len(head))
    for label, seed, mt, it, data in rows_sorted:
        vals = [_t2_tau(data, c) for c in chains]
        print(f"{label:>11}  {seed:>4}  {mt:>6}  {it:>7} "
              + "".join(_fmt(v, w=8) for v in vals))


def _print_t3(rows, label_order):
    """top-1 conf on wrong commits at chain-MSB g/k cells (a4 >=29)."""
    _print_header(
        "T3  Top-1 confidence on WRONG commits at chain-MSB g/k cells",
        "Aggregated over a4 chain bucket '>=29'. n_wrong shown for sanity.")
    rows_sorted = sorted(rows, key=lambda r: _sort_key(*r[:4], label_order))
    head = (f"{'condition':>11}  {'seed':>4}  {'method':>6}  {'iter':>7} "
            f"{'g.conf':>8}{'g.n_w':>8}  {'k.conf':>8}{'k.n_w':>8}")
    print(head)
    print('─' * len(head))
    for label, seed, mt, it, data in rows_sorted:
        g_conf, g_nw = _t3_top1_wrong(data, 'g')
        k_conf, k_nw = _t3_top1_wrong(data, 'k')
        print(f"{label:>11}  {seed:>4}  {mt:>6}  {it:>7} "
              + f"{_fmt(g_conf, w=8)}{g_nw:>8d}  "
              + f"{_fmt(k_conf, w=8)}{k_nw:>8d}")


def _print_t4(rows, chains, label_order):
    """only_lsb share of conf failures per chain."""
    _print_header(
        "T4  only_lsb / (only_lsb + only_conf)  per chain",
        "Share of conf failures that LSB order would have recovered. "
        "High = recoverable via decode swap (B); low + low acc = (A).")
    rows_sorted = sorted(rows, key=lambda r: _sort_key(*r[:4], label_order))
    head = f"{'condition':>11}  {'seed':>4}  {'method':>6}  {'iter':>7} "
    head += "".join(f"{f'≥{c}':>8}" for c in chains)
    print(head)
    print('─' * len(head))
    for label, seed, mt, it, data in rows_sorted:
        vals = [_t1_lsb_share(data, c) for c in chains]
        print(f"{label:>11}  {seed:>4}  {mt:>6}  {it:>7} "
              + "".join(_fmt(v, w=8) for v in vals))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--analysis_dirs', nargs='+', required=True, type=Path)
    ap.add_argument('--labels', nargs='+', default=None,
                    help='Display label per analysis dir (default: dir basename)')
    ap.add_argument('--methods', nargs='+', default=None,
                    help="Restrict to these methods (random/papl/puma)")
    ap.add_argument('--iters', nargs='+', type=int, default=None,
                    help="Restrict to these iter milestones")
    ap.add_argument('--chains', nargs='+', type=int,
                    default=[8, 16, 20, 24, 28, 30, 32],
                    help="Chain buckets to show in chain-stratified tables")
    ap.add_argument('--metrics', nargs='+', default=['t1', 't2', 't3', 't4'],
                    choices=['t1', 't2', 't3', 't4'],
                    help="Which tables to print (default: all four)")
    args = ap.parse_args()

    labels = (args.labels if args.labels
              else [d.parent.name if d.name == 'analysis' else d.name
                    for d in args.analysis_dirs])
    if len(labels) != len(args.analysis_dirs):
        raise SystemExit(f"--labels count {len(labels)} != "
                         f"--analysis_dirs count {len(args.analysis_dirs)}")

    rows = _load_all(args.analysis_dirs, labels,
                     methods=args.methods, iters=args.iters)
    if not rows:
        print("No analysis files found after filtering.")
        return

    print(f"\nLoaded {len(rows)} analysis files across "
          f"{len(set(r[0] for r in rows))} conditions.")
    print(f"  conditions: {sorted(set(r[0] for r in rows))}")
    print(f"  seeds:      {sorted(set(r[1] for r in rows))}")
    print(f"  methods:    {sorted(set(r[2] for r in rows))}")
    print(f"  iters:      {sorted(set(r[3] for r in rows))}")
    print(f"  chains:     {args.chains}")

    if 't1' in args.metrics: _print_t1(rows, args.chains, labels)
    if 't2' in args.metrics: _print_t2(rows, args.chains, labels)
    if 't3' in args.metrics: _print_t3(rows, labels)
    if 't4' in args.metrics: _print_t4(rows, args.chains, labels)


if __name__ == '__main__':
    main()
