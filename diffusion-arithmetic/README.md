# PUMA Coverage Deficit Experiments

Investigating whether PUMA (Progressive UnMasking) creates systematic coverage
deficits on rare/extreme reasoning cases compared to random masking in Masked
Diffusion Models (MDMs).

## Structure

```
core/
  model.py        — Shared Transformer (bidirectional, RoPE/absolute PE, SDPA/FlashAttention)
  train_utils.py  — Unified training loop (EMA, PUMA streaming, AMP bfloat16, early stopping)
  tokenizer.py    — Character-level tokenizer

experiments/
  exp_addition.py — Carry chain dependency (positional linear chain)
  exp_sudoku.py   — Constraint propagation (grid forcing chain)
  exp_maze.py     — Spatial path dependency (corridor traversal)
  exp_listops.py  — Hierarchical nesting dependency (tree evaluation)
```

## Domains & Rarity Axes

| Domain   | Dependency Structure      | Rarity Axis              | Oracle Decode     |
|----------|---------------------------|--------------------------|-------------------|
| Addition | Positional linear chain   | Carry chain length       | LSB (r2l)         |
| Sudoku   | Grid constraint propagation | Technique level        | oracle_technique  |
| Maze     | Spatial sequential path   | Corridor length          | BFS-from-start    |
| ListOps  | Hierarchical tree nesting | Nesting depth            | Inner→outer (l2r) |

## PUMA K Schedule (step-based)

| Task     | ANS_LEN | K_start → K_end | K_step | Reveal/step (start→end) |
|----------|---------|-----------------|--------|-------------------------|
| Addition | 33      | 3 → 16          | 2      | 11 → 2                  |
| ListOps  | 20      | 2 → 10          | 2      | 10 → 2                  |
| Sudoku   | 81      | 5 → 40          | 5      | 16 → 2                  |
| Maze     | 441     | 10 → 40         | 5      | 44 → 11                 |

## Quick Start (Colab A100)

```python
# Addition (ND=32, ~2h)
%run experiments/exp_addition.py --puma-k-start 3 --puma-k-end 16 --puma-k-step 2

# ListOps (depth=5, ~4h)
%run experiments/exp_listops.py --max-depth 5 --n-layer 8 --n-head 4 --n-embd 256 \
    --n-train 100000 --depth-decay 0.1 --puma-k-start 2 --puma-k-end 10 --puma-k-step 2

# Sudoku (~6h)
%run experiments/exp_sudoku.py --puma-k-start 5 --puma-k-end 40 --puma-k-step 5

# Maze (grid=10, ~8h)
%run experiments/exp_maze.py --grid-n 10 --puma-k-start 10 --puma-k-end 40 --puma-k-step 5
```

## Optimizations

- **AMP (bfloat16)**: Auto-enabled on A100, ~2x training speedup
- **FlashAttention**: Via PyTorch SDPA, auto-dispatched on supported hardware
- **Vectorized probes**: Per-position loss/acc computed without Python loops
- **Prefix-length grouping** (ListOps): Avoids positional mismatch during generation
- **Rainbow padding** (ListOps): Breaks PAD token dominance in variable-length traces
