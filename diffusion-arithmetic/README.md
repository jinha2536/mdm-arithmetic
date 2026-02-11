# Discrete Diffusion vs AR: Arithmetic Reasoning & Decoding Analysis

Three independent experiment modules comparing autoregressive and masked diffusion
language models on arithmetic reasoning tasks.

## Quick Start (Colab)

```python
from google.colab import drive
drive.mount('/content/drive')
!git clone https://github.com/<repo>/diffusion-arithmetic.git
%cd diffusion-arithmetic
!pip install -r requirements.txt

# Run any module independently:
%run experiments/exp_addition.py
%run experiments/exp_tree.py
%run experiments/exp_toy_distribution.py
```

Results auto-save to `My Drive/diffusion-arithmetic-results/`.

## Modules

| Module | What it tests | Key output |
|--------|--------------|------------|
| `exp_addition` | Addition: carry chain reasoning | AR vs diffusion × format × pos_enc |
| `exp_tree` | Tree expressions: parallel computation | Depth generalisation + scratchpad analysis |
| `exp_toy_distribution` | Decoding policy mechanisms | TV distance, mode coverage, greedy vs sampling |

## Project Structure

```
core/
  model.py          # Transformer with RoPE / absolute / none
  tokenizer.py      # Character-level tokenizer
  train_utils.py    # Convergence training, greedy generation, Drive save
experiments/
  exp_addition.py   # Module 1
  exp_tree.py       # Module 2
  exp_toy_distribution.py  # Module 3
```

## Design Decisions & Vulnerability Mitigations

### 1. Fair Decoding: Greedy (argmax) for Both
AR uses greedy by default. Diffusion now also uses argmax at each position
selection step. This eliminates sampling noise as a confound when comparing
exact match accuracy.

### 2. Convergence-Based Training
Instead of fixed iteration counts (which give AR more effective supervision
per step due to teacher forcing), training uses early stopping with patience.
Both objectives train until their loss plateaus, ensuring fair comparison.

### 3. RoPE for Length/Depth Generalisation
Absolute position embeddings break at unseen positions, confounding
"length generalisation failure" with "position encoding failure".
RoPE (Rotary Position Embedding) is tested as an alternative, isolating
the true effect of AR vs diffusion on generalisation.

### 4. Scratchpad Decode Order Analysis
For diffusion with scratchpad format, the actual decode order is tracked:
does the model fill scratchpad (intermediate) positions before the final
answer? This reveals whether diffusion's scratchpad mechanism differs
fundamentally from AR's sequential chain-of-thought.

### 5. Module 3 Framing
The toy distribution experiment analyses diffusion's sampling *mechanism*
in isolation. Its results explain how policies behave on different
dependency structures but should NOT be used to directly predict
arithmetic/tree performance (different dependency types).

### 6. Greedy vs Sampling in Module 3
Module 3 evaluates both greedy and sampling modes. Greedy shows deterministic
quality; sampling shows distributional coverage. The comparison reveals whether
greedy decoding (used in modules 1-2 for fairness) introduces systematic bias.

## Architecture

Matches NanoGPT from "Teaching Arithmetic to Small Transformers" (Lee et al.):
- 6 layers, 6 heads, 384 embedding dim
- ~10.6M parameters
- Weight tying (embedding = output projection)
- Causal mask (AR) / no mask (diffusion) — same backbone

## Citation

```bibtex
@inproceedings{lee2024teaching,
  title={Teaching Arithmetic to Small Transformers},
  author={Lee, Nayoung and Sreenivasan, Kartik and Lee, Jason D
          and Lee, Kangwook and Papailiopoulos, Dimitris},
  booktitle={ICLR},
  year={2024}
}
```
