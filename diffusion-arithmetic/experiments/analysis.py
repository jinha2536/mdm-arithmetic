"""Paper-ready figure generation from experiment results.

Usage:
    python analysis.py --results results.json --domain addition --out figures/
    python analysis.py --results-dir results/ --out figures/          # batch mode

Loads a domain's results.json (3-way: random / papl / puma) and emits PDF figures
suitable for direct LaTeX inclusion. Figures target NeurIPS column widths and
use consistent typography + color palette across domains.

Design choices:
  - pdf.fonttype=42 → TrueType embedded (editable in Illustrator, accepted by publishers)
  - serif family (Times-like) matching LaTeX body text
  - Color palette: accessible + consistent mapping
      Random = navy   #1F3A5F  (neutral baseline)
      PAPL   = orange #E67E22  (loss-level intervention)
      PUMA   = teal   #16A085  (mask-level intervention)
  - Stratum gradient uses plasma (perceptually uniform, low → high difficulty)
  - NeurIPS column widths: 3.25in (one-column), 6.75in (two-column)
  - Grid at alpha=0.25, tick labels 8pt, axis labels 9pt, title 10pt
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ─────────────────────────────────────────────────────────────────────
# Global style
# ─────────────────────────────────────────────────────────────────────

COLORS = {
    'random': '#1F3A5F',
    'papl':   '#E67E22',
    'puma':   '#16A085',
}
LINESTYLES = {'random': '-', 'papl': '-', 'puma': '-'}
MARKERS = {'random': 'o', 'papl': 's', 'puma': '^'}

METHOD_ORDER = ['random', 'papl', 'puma']
METHOD_LABELS = {'random': 'Random', 'papl': 'PAPL', 'puma': 'PUMA'}


def apply_paper_style():
    """Set matplotlib rcParams for paper-ready output."""
    mpl.rcParams.update({
        'pdf.fonttype': 42,            # TrueType embedding
        'ps.fonttype': 42,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'axes.labelsize': 9,
        'axes.titlesize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'figure.titlesize': 10,
        'axes.linewidth': 0.6,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'grid.linewidth': 0.4,
        'grid.alpha': 0.25,
        'lines.linewidth': 1.3,
        'lines.markersize': 4,
        'legend.frameon': False,
        'legend.handlelength': 1.8,
        'legend.borderpad': 0.2,
        'legend.columnspacing': 1.2,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


# NeurIPS column widths (inches)
W_SINGLE = 3.25
W_DOUBLE = 6.75


# ─────────────────────────────────────────────────────────────────────
# Domain config: axis names, sweep keys, reasoning-order name
# ─────────────────────────────────────────────────────────────────────

DOMAIN_SPEC = {
    'addition': {
        'sweep_prefix': 'chain_sweep',
        'sweep_label': 'carry chain length ≥ L',
        'sweep_xaxis': 'chain length L',
        'decode_policies': ['confidence', 'lsb'],
        'reasoning_label': 'r2l (LSB-first)',
        'extreme_strata_key': 'chain_16plus',
    },
    'maze': {
        'sweep_prefix': 'backbone_sweep',
        'sweep_label': 'backbone length ≥ L',
        'sweep_xaxis': 'backbone length L',
        'decode_policies': ['confidence', 'dead_end_filling'],
        'reasoning_label': 'dead-end filling',
        'extreme_strata_key': 'bb_24plus',
    },
    'listops': {
        'sweep_prefix': 'critical_sweep',
        'sweep_label': 'critical chain length ≥ L',
        'sweep_xaxis': 'critical chain L',
        'decode_policies': ['confidence', 'layered_oracle'],
        'reasoning_label': 'layered post-order',
        'extreme_strata_key': 's_chain_3plus',
    },
    'zebra': {
        'sweep_prefix': 'size_sweep',           # routed to _zebra_size_sweep
        'sweep_label': 'puzzle size (n_houses)',
        'sweep_xaxis': 'n_houses',
        'decode_policies': ['confidence', 'layered_oracle'],
        'reasoning_label': 'solver order',
        'extreme_strata_key': 'size_6',
    },
    'countdown': {
        'sweep_prefix': None,                   # no per-L sweep — uses overall metrics
        'sweep_label': 'decode policy',
        'sweep_xaxis': 'decode',
        'decode_policies': ['confidence', 'step_seq'],
        'reasoning_label': 'step-sequential',
        'extreme_strata_key': None,
    },
    'sudoku': {
        'sweep_prefix': 'rating_tier',          # routed to _sudoku_rating_sweep
        'sweep_label': 'rating tier',
        'sweep_xaxis': 'tier',
        'decode_policies': ['confidence'],      # main axis: rating across tiers
        'reasoning_label': 'oracle solver',
        'extreme_strata_key': 'extreme',
    },
}


# ─────────────────────────────────────────────────────────────────────
# Results parsing
# ─────────────────────────────────────────────────────────────────────

def load_results(path):
    with open(path) as f:
        return json.load(f)


def sweep_values(data, method, sweep_prefix, decode):
    """Extract sweep sequence as [(L, accuracy, n), ...] sorted by L.

    Standard format: data[f'final_{method}_{sweep_prefix}_{L}_{decode}']['accuracy'].
    Domains with non-standard formats (zebra, sudoku, countdown) are routed
    through domain-specific extractors below.
    """
    # Domain-specific dispatch
    if sweep_prefix == 'size_sweep':
        return _zebra_size_sweep(data, method, decode)
    if sweep_prefix == 'tl4frac_sweep':
        return _sudoku_tl4_sweep(data, method, decode)
    if sweep_prefix == 'rating_tier':
        return _sudoku_rating_sweep(data, method, decode)
    if sweep_prefix == 'technique_acc':
        return _sudoku_technique_sweep(data, method, decode)

    # Default: final_<mt>_<sweep>_<L>_<dp> pattern
    rx = re.compile(rf'^final_{method}_{sweep_prefix}_(.+)_{decode}$')
    out = []
    for k, v in data.items():
        m = rx.match(k)
        if not m or not isinstance(v, dict) or 'accuracy' not in v:
            continue
        key = m.group(1)
        try:
            L = float(key) if '.' in key else int(key)
        except ValueError:
            L = key
        out.append((L, v['accuracy'], v.get('n', None)))
    try:
        out.sort(key=lambda x: float(x[0]))
    except (TypeError, ValueError):
        out.sort(key=lambda x: str(x[0]))
    return out


def _zebra_size_sweep(data, method, decode):
    """Zebra stores per-size accuracy as fields acc_size_3..acc_size_6
    inside data[f'final_{method}_{decode}']."""
    final = data.get(f'final_{method}_{decode}', {})
    if not isinstance(final, dict):
        return []
    out = []
    for n in [3, 4, 5, 6]:
        v = final.get(f'acc_size_{n}')
        if v is not None:
            out.append((n, v, None))
    return out


def _sudoku_tl4_sweep(data, method, decode):
    """Sudoku tl4frac sweep is stored as result_<mt>_tl4frac_<frac>_<decode>."""
    rx = re.compile(rf'^result_{method}_tl4frac_([0-9.]+)_{decode}$')
    out = []
    for k, v in data.items():
        m = rx.match(k)
        if not m or not isinstance(v, dict):
            continue
        try:
            L = float(m.group(1))
        except ValueError:
            continue
        acc = v.get('accuracy', v.get('blank_cell_acc'))
        if acc is not None:
            out.append((L, acc, v.get('n')))
    out.sort(key=lambda x: float(x[0]))
    return out


def _sudoku_rating_sweep(data, method, decode):
    """Sudoku rating-tier accuracy is nested in result_<mt>_<decode>['rating_accuracy']."""
    final = data.get(f'result_{method}_{decode}', {})
    if not isinstance(final, dict):
        return []
    ra = final.get('rating_accuracy', {})
    out = []
    tier_order = {'easy': 0, 'medium': 1, 'hard': 2,
                  'very_hard': 3, 'extreme': 4, 'top1pct': 5}
    for tname, info in ra.items():
        if not isinstance(info, dict):
            continue
        order = tier_order.get(tname, 99)
        out.append((order, info.get('exact', info.get('cell', 0.0)),
                    info.get('n', None)))
    out.sort(key=lambda x: x[0])
    name_lookup = {v: k for k, v in tier_order.items()}
    return [(name_lookup.get(o, str(o)), a, n) for o, a, n in out]


def _sudoku_technique_sweep(data, method, decode):
    """Sudoku per-technique-level accuracy from technique_accuracy dict."""
    final = data.get(f'result_{method}_{decode}', {})
    if not isinstance(final, dict):
        return []
    ta = final.get('technique_accuracy', {})
    out = []
    # Sort by technique level number (tl_0_giveup, tl_1_naked, ..., tl_4_search)
    for tname, acc in ta.items():
        if isinstance(tname, str) and tname.startswith('tl_'):
            try:
                lvl = int(tname.split('_')[1])
            except (ValueError, IndexError):
                lvl = 99
            out.append((lvl, acc, None))
    out.sort(key=lambda x: x[0])
    return out


def dyn_gen_acc_trajectory(data, method, policy='confidence'):
    """Extract (iters, gen_accs) from dyn_<method>.checkpoints."""
    dyn = data.get(f'dyn_{method}', {})
    cps = dyn.get('checkpoints', [])
    xs, ys = [], []
    key = f'gen_acc_{policy}'
    for c in cps:
        if key in c and c[key] is not None:
            xs.append(c['iter']); ys.append(c[key])
        elif 'gen_acc' in c and policy == 'confidence':
            xs.append(c['iter']); ys.append(c['gen_acc'])
    return xs, ys


def stratified_loss_trajectory(data, method):
    """Extract stratified training loss: returns (iters, per_stratum_losses[S,T], stratum_names)."""
    dyn = data.get(f'dyn_{method}', {})
    sl = dyn.get('stratified_loss', [])
    names = dyn.get('stratum_names', [])
    if not sl:
        return None, None, names
    iters = np.array([e['iter'] for e in sl])
    S = len(names) if names else len(sl[0].get('per_stratum_loss', []))
    losses = np.full((S, len(iters)), np.nan)
    for t, e in enumerate(sl):
        pl = e.get('per_stratum_loss', [])
        pn = e.get('per_stratum_n', [])
        for si in range(min(S, len(pl))):
            if pn[si] > 0 and pl[si] is not None:
                losses[si, t] = pl[si]
    return iters, losses, names


def reveal_tau_trajectory(data, method):
    """Extract (iters, q50, q25, q75, per_stratum dict of trajectories)."""
    dyn = data.get(f'dyn_{method}', {})
    cps = [c for c in dyn.get('checkpoints', []) if 'reveal_tau' in c]
    if not cps:
        return None
    iters = np.array([c['iter'] for c in cps])
    rt = [c['reveal_tau'] for c in cps]
    q50 = np.array([r['q50'] for r in rt])
    q25 = np.array([r['q25'] for r in rt])
    q75 = np.array([r['q75'] for r in rt])
    per_stratum = {}
    for r, it in zip(rt, iters):
        for sn, d in r.get('per_stratum', {}).items():
            per_stratum.setdefault(sn, {'iter': [], 'q50': [], 'n': []})
            per_stratum[sn]['iter'].append(it)
            per_stratum[sn]['q50'].append(d['q50'])
            per_stratum[sn]['n'].append(d.get('n', 0))
    return {'iter': iters, 'q50': q50, 'q25': q25, 'q75': q75,
            'per_stratum': per_stratum}


# ─────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────

def fig_stratum_accuracy(data, domain, outdir, decode=None):
    """Fig 1: Chain/backbone/etc sweep — 3 methods × decode policy.

    For each decode policy, plot accuracy vs sweep axis, three lines (random/papl/puma).
    Layout: single row of subplots, one per decode policy.

    Returns None if domain has no per-L sweep (e.g., countdown).
    """
    spec = DOMAIN_SPEC[domain]
    if spec['sweep_prefix'] is None:
        return None  # No sweep for this domain
    policies = [decode] if decode else spec['decode_policies']
    policies = [p for p in policies if any(
        sweep_values(data, mt, spec['sweep_prefix'], p)
        for mt in METHOD_ORDER)]
    if not policies:
        return None
    ncols = len(policies)
    fig, axes = plt.subplots(1, ncols,
                             figsize=(max(W_SINGLE, 2.3 * ncols), 2.4),
                             sharey=True, squeeze=False)
    axes = axes[0]
    for i, pol in enumerate(policies):
        ax = axes[i]
        for mt in METHOD_ORDER:
            vals = sweep_values(data, mt, spec['sweep_prefix'], pol)
            if not vals:
                continue
            xs = [v[0] for v in vals]
            ys = [v[1] for v in vals]
            ax.plot(xs, ys, LINESTYLES[mt], color=COLORS[mt], lw=1.4,
                    marker=MARKERS[mt], markersize=3.5, markeredgewidth=0,
                    label=METHOD_LABELS[mt])
        ax.set_xlabel(spec['sweep_xaxis'])
        if i == 0:
            ax.set_ylabel('accuracy')
        ax.set_ylim(-0.03, 1.05)
        ax.set_title(f'decode: {pol}')
        ax.grid(alpha=0.25)
        if i == 0:
            ax.legend(loc='lower left', fontsize=7)
    fig.tight_layout()
    path = Path(outdir) / f'{domain}_stratum_accuracy.pdf'
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    return path


def fig_stratified_loss(data, domain, outdir):
    """Fig 2: Per-stratum training loss trajectory, 3-panel (one per mask_type)."""
    strata_data = {}
    for mt in METHOD_ORDER:
        iters, losses, names = stratified_loss_trajectory(data, mt)
        if iters is not None and losses is not None:
            strata_data[mt] = (iters, losses, names)
    if not strata_data:
        return None

    n = len(strata_data)
    fig, axes = plt.subplots(1, n, figsize=(W_DOUBLE * n / 3, 2.5),
                             sharey=True, squeeze=False)
    axes = axes[0]
    names = next(iter(strata_data.values()))[2]
    S = len(names)
    s_cmap = plt.cm.plasma
    stratum_colors = [s_cmap(si / max(S - 1, 1)) for si in range(S)]

    # Global y-limits for comparability
    y_min, y_max = 1e10, 0
    for _, (_, losses, _) in strata_data.items():
        valid = losses[~np.isnan(losses)]
        if len(valid):
            y_min = min(y_min, valid.min())
            y_max = max(y_max, valid.max())
    y_min = max(y_min * 0.5, 1e-5)
    y_max = y_max * 2

    for ai, mt in enumerate([m for m in METHOD_ORDER if m in strata_data]):
        ax = axes[ai]
        iters, losses, _ = strata_data[mt]
        for si in range(S):
            y = losses[si]
            mask = ~np.isnan(y)
            if mask.sum() < 2:
                continue
            ax.plot(iters[mask], y[mask], '-', color=stratum_colors[si],
                    lw=1.3, label=names[si] if ai == 0 else None)
        ax.set_yscale('log')
        ax.set_xlabel('training iteration')
        ax.set_ylim(y_min, y_max)
        if ai == 0:
            ax.set_ylabel('per-stratum loss')
        ax.set_title(METHOD_LABELS[mt], color=COLORS[mt], fontweight='bold')
        ax.grid(alpha=0.25, which='both')
    # Single legend on the right for stratum names
    if axes[0].has_data():
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='center right',
                   bbox_to_anchor=(1.0, 0.5), fontsize=7,
                   title='stratum', title_fontsize=7)
    fig.tight_layout(rect=(0, 0, 0.92, 1))
    path = Path(outdir) / f'{domain}_stratified_loss.pdf'
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    return path


def fig_reveal_tau(data, domain, outdir):
    """Fig 3: Kendall τ trajectory, 3-panel (one per mask_type)."""
    spec = DOMAIN_SPEC[domain]
    tau_data = {}
    for mt in METHOD_ORDER:
        rt = reveal_tau_trajectory(data, mt)
        if rt is not None:
            tau_data[mt] = rt
    if not tau_data:
        return None

    n = len(tau_data)
    fig, axes = plt.subplots(1, n, figsize=(W_DOUBLE * n / 3, 2.5),
                             sharey=True, squeeze=False)
    axes = axes[0]
    # Collect stratum names across all
    all_strata = []
    for _, rt in tau_data.items():
        for sn in rt['per_stratum']:
            if sn not in all_strata:
                all_strata.append(sn)
    S = len(all_strata)
    s_cmap = plt.cm.plasma
    stratum_colors = {sn: s_cmap(si / max(S - 1, 1))
                      for si, sn in enumerate(all_strata)}

    for ai, mt in enumerate([m for m in METHOD_ORDER if m in tau_data]):
        ax = axes[ai]
        rt = tau_data[mt]
        # Overall IQR
        ax.fill_between(rt['iter'], rt['q25'], rt['q75'],
                        alpha=0.10, color='#555555')
        ax.plot(rt['iter'], rt['q50'], '--', color='#333333', lw=0.9,
                alpha=0.6, label='overall' if ai == 0 else None)
        # Per-stratum
        for sn, d in rt['per_stratum'].items():
            if len(d['iter']) < 2:
                continue
            ax.plot(d['iter'], d['q50'], '-o',
                    color=stratum_colors[sn], lw=1.2, markersize=3,
                    markeredgewidth=0,
                    label=sn if ai == 0 else None)
        ax.axhline(0, color='gray', ls=':', lw=0.5, alpha=0.6)
        ax.axhline(1, color='#2ca02c', ls='--', lw=0.5, alpha=0.4)
        ax.axhline(-1, color='#d62728', ls='--', lw=0.5, alpha=0.4)
        ax.set_xlabel('training iteration')
        if ai == 0:
            ax.set_ylabel(f"Kendall τ vs {spec['reasoning_label']}")
        ax.set_title(METHOD_LABELS[mt], color=COLORS[mt], fontweight='bold')
        ax.set_ylim(-1.05, 1.05)
        ax.grid(alpha=0.25)

    if axes[0].has_data():
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='center right',
                   bbox_to_anchor=(1.0, 0.5), fontsize=7,
                   title='stratum', title_fontsize=7)
    fig.tight_layout(rect=(0, 0, 0.92, 1))
    path = Path(outdir) / f'{domain}_reveal_tau.pdf'
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    return path


def fig_grokking_trajectory(data, domain, outdir, policy='confidence'):
    """Fig 4: Gen accuracy vs training iter, all methods on one panel.

    Shows the grokking-speed spectrum: PUMA early, Random mid, PAPL late.
    """
    fig, ax = plt.subplots(figsize=(W_SINGLE, 2.3))
    any_data = False
    for mt in METHOD_ORDER:
        xs, ys = dyn_gen_acc_trajectory(data, mt, policy)
        if not xs:
            continue
        any_data = True
        ax.plot(xs, ys, LINESTYLES[mt], color=COLORS[mt], lw=1.5,
                marker=MARKERS[mt], markersize=3, markeredgewidth=0,
                label=METHOD_LABELS[mt])
    if not any_data:
        plt.close(fig); return None
    ax.set_xlabel('training iteration')
    ax.set_ylabel(f'gen. accuracy ({policy})')
    ax.set_ylim(-0.03, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(loc='lower right')
    fig.tight_layout()
    path = Path(outdir) / f'{domain}_grokking.pdf'
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    return path


def fig_summary_bar(data, domain, outdir, decode='confidence'):
    """Fig 5: Catching summary — extreme-stratum accuracy bar, 3 methods.

    A single-bar-per-method panel at the hardest sweep point, making the
    random-advantage-on-rare-patterns claim visually immediate.
    Returns None if domain has no per-L sweep.
    """
    spec = DOMAIN_SPEC[domain]
    if spec['sweep_prefix'] is None:
        return None
    # Take the hardest available sweep value per method
    bars = {}
    for mt in METHOD_ORDER:
        vals = sweep_values(data, mt, spec['sweep_prefix'], decode)
        if not vals:
            continue
        # Hardest = highest L value (last after sort)
        hardest_L, acc, n = vals[-1]
        bars[mt] = (hardest_L, acc, n)
    if not bars:
        return None
    fig, ax = plt.subplots(figsize=(W_SINGLE, 2.1))
    methods = [m for m in METHOD_ORDER if m in bars]
    xs = np.arange(len(methods))
    accs = [bars[m][1] for m in methods]
    colors = [COLORS[m] for m in methods]
    b = ax.bar(xs, accs, color=colors, width=0.6, edgecolor='none')
    for rect, m in zip(b, methods):
        acc = bars[m][1]
        ax.text(rect.get_x() + rect.get_width() / 2, acc + 0.02,
                f'{acc:.2f}', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels([METHOD_LABELS[m] for m in methods])
    L = bars[methods[0]][0]
    ax.set_ylabel('accuracy')
    ax.set_title(f'{spec["sweep_label"]}={L} (decode={decode})')
    ax.set_ylim(0, 1.1)
    ax.grid(alpha=0.25, axis='y')
    fig.tight_layout()
    path = Path(outdir) / f'{domain}_summary_bar.pdf'
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    return path


# ─────────────────────────────────────────────────────────────────────
# Domain-specific catching figures
# ─────────────────────────────────────────────────────────────────────

def fig_addition_failure(data, outdir):
    """Addition: per-position accuracy on a carry-heavy test, 3 methods.

    The catching figure for addition: rightmost digits (LSB) all correct,
    mid positions (where long carries propagate) all wrong — visually shows
    PAPL/PUMA's failure on the carry-chain interior.
    """
    # Look up probe_per_position final eval — stored in 'final_<mt>_position_acc'
    # or in last dyn checkpoint's pos_acc.
    per_method = {}
    for mt in METHOD_ORDER:
        dyn = data.get(f'dyn_{mt}', {})
        cps = dyn.get('checkpoints', [])
        if not cps:
            continue
        # Prefer last checkpoint with pos_acc
        for c in reversed(cps):
            if 'pos_acc' in c and c['pos_acc']:
                per_method[mt] = c['pos_acc']
                break
    if not per_method:
        return None

    fig, ax = plt.subplots(figsize=(W_DOUBLE, 2.2))
    for mt, pa in per_method.items():
        xs = list(range(len(pa)))
        ax.plot(xs, pa, LINESTYLES[mt], color=COLORS[mt], lw=1.3,
                marker=MARKERS[mt], markersize=2.5, markeredgewidth=0,
                label=METHOD_LABELS[mt])
    ax.set_xlabel('answer position (0 = MSB, L-1 = LSB)')
    ax.set_ylabel('per-position probe accuracy')
    ax.set_ylim(-0.03, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(loc='lower right')
    fig.tight_layout()
    path = Path(outdir) / 'addition_position_failure.pdf'
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    return path


# ─────────────────────────────────────────────────────────────────────
# Main driver
# ─────────────────────────────────────────────────────────────────────

def run_domain(results_path, domain, outdir):
    print(f"\n=== {domain}  |  {results_path} ===")
    data = load_results(results_path)
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    produced = []
    for fn, name in [
        (lambda: fig_stratum_accuracy(data, domain, outdir), 'stratum_accuracy'),
        (lambda: fig_stratified_loss(data, domain, outdir), 'stratified_loss'),
        (lambda: fig_reveal_tau(data, domain, outdir), 'reveal_tau'),
        (lambda: fig_grokking_trajectory(data, domain, outdir), 'grokking'),
        (lambda: fig_summary_bar(data, domain, outdir), 'summary_bar'),
    ]:
        try:
            p = fn()
            if p:
                print(f"  ✓ {name}: {p}")
                produced.append(p)
        except Exception as e:
            print(f"  ✗ {name} failed: {e}")
    # Domain-specific
    if domain == 'addition':
        try:
            p = fig_addition_failure(data, outdir)
            if p:
                print(f"  ✓ position_failure: {p}")
                produced.append(p)
        except Exception as e:
            print(f"  ✗ position_failure failed: {e}")
    return produced


def infer_domain(filename):
    for d in DOMAIN_SPEC:
        if d in filename.lower():
            return d
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', type=str, help='Path to single results.json')
    p.add_argument('--domain', type=str,
                   choices=list(DOMAIN_SPEC), help='Domain for --results file')
    p.add_argument('--results-dir', type=str,
                   help='Directory containing results_<domain>.json files (batch mode)')
    p.add_argument('--out', type=str, default='figures',
                   help='Output directory for PDFs')
    args = p.parse_args()

    apply_paper_style()

    if args.results:
        dom = args.domain or infer_domain(args.results)
        if not dom:
            raise SystemExit("Could not infer domain; use --domain")
        run_domain(args.results, dom, args.out)
    elif args.results_dir:
        rdir = Path(args.results_dir)
        produced_all = []
        for p_ in sorted(rdir.glob('*.json')):
            dom = args.domain or infer_domain(p_.name)
            if not dom:
                print(f"  (skip: {p_.name} — no domain inferred)")
                continue
            ps = run_domain(p_, dom, args.out)
            produced_all.extend(ps)
        print(f"\nDone. {len(produced_all)} PDFs → {args.out}")
    else:
        p.print_help()


if __name__ == '__main__':
    main()
