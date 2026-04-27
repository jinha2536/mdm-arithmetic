# Diffusion-arithmetic: 3-way comparison (random / PAPL / PUMA)

## Paper claim (primary)

**"Confidence-aligned training (PUMA, PAPL) misses rare/extreme reasoning
patterns; random masking's uniformity preserves them."**

Sudoku is included as a contrast case where dense-symmetric constraints make
high-confidence cells useful for context propagation, narrowing the gap. This
sharpens rather than weakens the main claim — the failure mode is specific to
sparse-directional dependency structures.

## Architecture

- **Random**: uniform random masking (baseline, no intervention)
- **PAPL** (loss-level intervention; Peng et al. 2025, arXiv:2509.23405):
  uniform masking + softmax-normalized planner weights
  `weight_i = (1/(L-k))(1 + α·w_i)` with `w_i ∝ exp((1/τ) log P(x_0^i|x_k))`.
  Defaults τ=1, α=1. Direction: easy/confident positions get up-weighted.
- **PUMA** (mask-level intervention; Kim et al. 2026, arXiv:2602.10314):
  confidence-guided forward masking with K-schedule curriculum. Direction:
  hard positions stay masked longer → more training signal on hard positions.

PAPL and PUMA work in opposite directions but both are confidence-aligned.
Random sits between, allocating capacity uniformly.

## Domain status (all 6 paper-ready)

| Domain    | 3-way | PAPL | --no-patience | Reveal-τ 3-way | Sweep                   | Decode policies                              |
|-----------|:-----:|:----:|:-------------:|:--------------:|-------------------------|----------------------------------------------|
| addition  |   ✓   |  ✓   |       ✓       |       ✓        | chain_sweep             | confidence, lsb (oracle)                     |
| maze      |   ✓   |  ✓   |       ✓       |       ✓        | backbone_sweep          | confidence, dead_end_filling, random         |
| listops   |   ✓   |  ✓   |       ✓       |       ✓        | critical_sweep          | confidence, layered_oracle, random           |
| zebra     |   ✓   |  ✓   |       ✓       |       ✓        | size_sweep (per cell)   | confidence, layered_oracle, random           |
| countdown |   ✓   |  ✓   |       ✓       |    n/a (—)     | corner_* cases          | confidence, step_seq (oracle), random        |
| sudoku    |   ✓   |  ✓   |       ✓       |    n/a (—)     | rating_tier (5 tiers)   | confidence, oracle_solver, oracle_technique  |

### Notes

- **layered_oracle** (listops, zebra): cell-level partial-order oracle —
  positions with smaller reasoning rank decode first; ties (same-layer
  siblings, same-cell row/col/val) broken by confidence.
- **Sudoku rating tiers**: rebalanced so hardest tier is genuinely rare
  (extreme ~4%, top1pct ~1%; boundaries at p80/p95/p99).
- **Sudoku solver-confidence disagreement (GPT Experiment A)**: Type 1/2/3/4
  cell classification quantifies whether confidence early-decodes
  Transformer-friendly global patterns (Type 2, expected high in
  dense-symmetric tasks) vs true sequential bottlenecks (Type 4, expected
  high in sparse-directional tasks).

## Run commands (Colab; all use --no-patience for fair comparison)

### Quick (1-2 hours)

```bash
# Addition (capacity-reachable)
%run experiments/exp_addition.py \
  --nd 32 --n-train 20000 --n-test 10000 \
  --max-iters 300000 --batch-size 256 \
  --n-layer 2 --n-head 2 --n-embd 128 \
  --papl-tau 1.0 --papl-alpha 1.0 \
  --no-patience --no-amp --tag 3way_v2

# Countdown
%run experiments/exp_countdown.py \
  --max-iters 200000 --batch-size 256 \
  --papl-tau 1.0 --papl-alpha 1.0 \
  --no-patience --no-amp --tag 3way_v1
```

### Long (8-10 hours; parallel sessions)

```bash
# Maze (capacity-reachable)
%run experiments/exp_maze.py \
  --grid-n 15 --n-train 50000 --n-test 5000 \
  --max-iters 150000 --batch-size 128 \
  --n-layer 3 --n-head 4 --n-embd 192 \
  --papl-tau 1.0 --papl-alpha 1.0 \
  --no-patience --no-amp --tag 3way_v1

# ListOps (capacity-limited — sparse-directional)
%run experiments/exp_listops.py \
  --max-depth 5 --depth-decay 0.8 \
  --n-train 20000 --max-iters 300000 --batch-size 256 \
  --n-layer 3 --n-head 3 --n-embd 192 \
  --papl-tau 1.0 --papl-alpha 1.0 \
  --no-patience --no-amp --tag 3way_v2

# Zebra (capacity-limited — instance-specific solver order)
%run experiments/exp_zebra.py \
  --max-iters 300000 --batch-size 64 \
  --papl-tau 1.0 --papl-alpha 1.0 \
  --no-patience --no-amp --tag 3way_v2

# Sudoku (contrast case — dense-symmetric)
%run experiments/exp_sudoku.py \
  --max-iters 400000 --batch-size 256 \
  --papl-tau 1.0 --papl-alpha 1.0 \
  --no-patience --no-amp --tag 3way_v1
```

## Figure generation

```bash
python analysis.py --results results_<domain>.json --domain <domain> --out figures/
# or batch
python analysis.py --results-dir results/ --out figures/
```

Generates per domain (PDF, NeurIPS-compliant):
- `<domain>_stratum_accuracy.pdf` — sweep accuracy, 3-way × decode policies
- `<domain>_stratified_loss.pdf` — per-stratum training loss
- `<domain>_reveal_tau.pdf` — Kendall τ vs canonical reasoning order
- `<domain>_grokking.pdf` — gen accuracy over training iter
- `<domain>_summary_bar.pdf` — extreme-stratum bar chart (catching figure)
- `addition_position_failure.pdf` — per-position probe accuracy (addition only)

## Color palette

- Random = navy `#1F3A5F`
- PAPL = orange `#E67E22`
- PUMA = teal `#16A085`
- Stratum gradient: `plasma` colormap

## Sensitivity sweeps

```bash
--papl-alpha 1.0   # paper default
--papl-alpha 2.0
--papl-alpha 5.0   # paper-recommended max

--puma-k-start 3 --puma-k-end 16   # addition default
--puma-k-start 6 --puma-k-end 24   # stronger confidence bias
```
