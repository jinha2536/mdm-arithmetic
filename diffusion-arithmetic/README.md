# Coverage Deficit in Progressive Unmasking

Investigating how PUMA's confidence-based training creates systematic blind spots
in Masked Diffusion Models on structured reasoning tasks.

## Quick Start (Colab)

```python
from google.colab import drive
drive.mount('/content/drive')
!git clone https://github.com/<repo>/diffusion-arithmetic.git
%cd diffusion-arithmetic
!pip install -r requirements.txt

# Run experiments:
%run experiments/exp_addition.py --nd 16 --max-iters 100000
%run experiments/exp_maze.py --grid-n 7
%run experiments/exp_sudoku.py
```

Results auto-save to `My Drive/diffusion-arithmetic-results/`.

## Experiments

| Experiment | Dependency Structure | Rarity Axis | Coverage Deficit Signal |
|------------|---------------------|-------------|------------------------|
| `exp_addition` | Sequential carry chains | Carry chain length | Crossover at long chains |
| `exp_maze` | Corridor paths | Corridor length | Same reversal pattern |
| `exp_sudoku` | Parallel constraints | Technique level | Hidden under confidence decode |

## Architecture

All experiments share the same `core/` infrastructure:

```
core/
  model.py          # Transformer (RoPE / absolute PE, causal / bidirectional)
  tokenizer.py      # Character-level tokenizer
  train_utils.py    # Unified iteration-based training with EMA, PUMA streaming
experiments/
  exp_addition.py   # Addition carry chains
  exp_maze.py       # Maze corridor paths
  exp_sudoku.py     # Sudoku technique levels
```

### Unified Training (`train_diffusion`)

All experiments use a single training loop in `train_utils.py`:
- **Iteration-based** with cosine LR schedule + warmup
- **EMA** (exponential moving average) for stable evaluation
- **PUMA streaming** with configurable K schedule (linear, step, fixed)
- **Continuation training**: load checkpoint from one mask type, continue with another
- **Eval callbacks**: experiment-specific probes called at regular intervals

### Masking Strategies

| Strategy | Training | Key Property |
|----------|----------|--------------|
| `random` | Uniform random mask ratio per sample | Equal coverage across all positions |
| `puma` | Progressive unmasking (confidence-based chain) | High-confidence positions revealed first → rare conditions under-covered |

### Continuation Training

Tests whether PUMA's coverage deficit is baked into representations:

```python
# Train with PUMA → continue with random masking
model = train(mask_type='puma', ...)
save_checkpoint(model.state_dict())
model_cont = train(mask_type='random', init_state=puma_state, max_iters=20000)
```

If the deficit persists after random fine-tuning, the bias is in learned representations,
not just training dynamics.

## Key Analyses

Each experiment includes:
- **Dependency stratification**: accuracy by structural difficulty (chain length / corridor length / technique level)
- **Carry/corridor rarity**: per-position base rate × conditional accuracy gap
- **PUMA coverage simulation**: how many training masks each position receives under PUMA
- **Error localization**: where in the dependency structure errors concentrate
- **Counterfactual analysis** (addition): minimal pairs differing only in carry-in
- **Confidence calibration**: overconfidence on rare patterns

## Design Decisions

1. **Greedy decoding for both AR and diffusion** — eliminates sampling noise as confound
2. **EMA for all experiments** — stable evaluation, standard practice
3. **Iteration-based training** — fair comparison across mask types (PUMA streaming requires iteration-level control)
4. **Multiple decode policies** — confidence decode can mask PUMA's weakness; oracle decode exposes it
