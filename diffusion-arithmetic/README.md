# Code for "The Confidence Shortcut"

Training and evaluation code for the masked diffusion experiments in the paper:
character-level masked diffusion models on multi-digit addition, maze path
labelling, ListOps, Countdown and Sudoku, trained with three masking schemes
(random, PAPL, PUMA) and decoded under confidence ordering, a task-specific
reasoning order and uniform-random ordering.

## Layout

```
core/
  model.py                    bidirectional Transformer denoiser
  tokenizer.py                character tokenizer with mask / pad specials
  train_utils.py              training loop (random / PAPL / PUMA masking, EMA),
                              one-token-per-step decoding, result I/O
experiments/
  exp_addition.py             32-digit addition
  exp_maze.py                 maze path labelling on 21 x 21 grids
  exp_listops.py              ListOps evaluation traces
  exp_countdown.py            Countdown (CD4)
  exp_sudoku.py               Sudoku (sudoku-extreme)
  addition_decode_analysis.py failure analysis of confidence decoding (addition)
  remasking_analysis.py       remasking and stochastic decoders (addition)
  data/cd4_test.jsonl         Countdown test set (1,000 puzzles)
```

## Setup

```
pip install -r requirements.txt
```

Run everything from the repository root. Results, checkpoints and figures are
written to `$MDM_RESULTS_DIR/<exp_name>/` (default `./results`).

## Experiments

Each `exp_*.py` trains one model per masking scheme (`random`, `papl`, `puma`),
evaluates it under every decoding policy in `DECODE_POLICIES`, and writes
`results.json` and `checkpoint_<scheme>.pt`. The output directory is
`exp_<task>_<tag>` (`exp_<task>` without `--tag`), with a `_s<seed>` suffix
per seed when several seeds are given. The defaults in the config block at the top of each script are the
settings used in the paper; the main ones can be overridden from the command
line (`--max-iters`, `--n-layer`, `--puma-k-end`, `--papl-alpha`, `--masks`,
`--decode`, `--seeds`, ...; see `--help`).

```
python experiments/exp_addition.py  --tag main --seeds 41 42 43
python experiments/exp_maze.py      --tag main --seeds 41 42 43
python experiments/exp_listops.py   --tag main --seeds 41 42 43
python experiments/exp_countdown.py --tag main --seeds 41 42 43
python experiments/exp_sudoku.py    --tag main --seeds 41 42 43
```

| task      | model (L / H / d) | iters | batch | LR   | PUMA K (start -> end) | PAPL alpha |
|-----------|-------------------|-------|-------|------|-----------------------|------------|
| addition  | 2 / 2 / 128       | 300k  | 256   | 1e-3 | 3 -> 16               | 1          |
| maze      | 3 / 3 / 192       | 50k   | 256   | 3e-4 | 10 -> 40              | 5          |
| ListOps   | 8 / 8 / 384       | 300k  | 256   | 3e-4 | 2 -> 10               | 5          |
| Countdown | 12 / 12 / 384     | 200k  | 256   | 3e-4 | 4 -> 20               | 5          |
| Sudoku    | 8 / 8 / 256       | 300k  | 256   | 3e-4 | 8 -> 40               | 5          |

Dropout, weight decay, warmup and the final learning rate are in the config
blocks. All runs use AdamW (betas 0.9, 0.99) with linear warmup and cosine
decay, fp32 (no mixed precision), EMA 0.9999, and the full iteration budget
(no early stopping). The fully-masked probe loss of the EMA weights (every
answer cell masked, one forward pass, mean NLL) is computed every `EVAL_EVERY`
iterations (every `EVAL_EVERY / 5` during the first 10% of training), and the
EMA state with the lowest probe loss is the one evaluated and saved as
`checkpoint_<scheme>.pt`. The probe set is the natural test set
for addition, a separate set of 5,000 held-out instances for maze and ListOps,
the test set for Countdown, and the hard rating tier of the test set for
Sudoku.

Decoding commits one answer position per forward pass and always takes the
argmax token. The confidence policy picks the masked position with the largest
maximum logit (addition, ListOps, Countdown) or the largest top-1 probability
(maze, Sudoku).

What each script reports:

* **exp_addition.py**: exact match on the natural test set (10,000 instances)
  and on carry-chain strata (longest carry chain >= k for
  k in {2, 3, 4, 6, 8, 12, 16, 20, 24, 28}, 500 instances each) under
  confidence and LSB-first decoding; add `--decode confidence lsb random` for
  uniform-random decoding. Every `GEN_EVAL_EVERY` (10k) iterations the EMA
  weights are also decoded on the first 500 natural test instances
  (confidence) and on the chain >= 28 stratum (confidence and LSB-first); the
  confidence curves are plotted in `acc_trajectory.png`, and the EMA weights
  at each of these iterations are saved as
  `checkpoint_seed<seed>_<scheme>_iter<iter>.pt`. `--train-only`
  trains all schemes from one shared initialisation with a per-scheme
  training seed and saves only the initial state, these snapshots, the
  dynamics (`results_seed<seed>_train.json`) and the figure.
* **exp_maze.py**: exact match on corridor strata (longest corridor on the
  start-to-goal path >= k, 300 mazes each) under confidence, dead-end-filling
  and random decoding.
* **exp_listops.py**: exact match of the evaluation trace on 500 trees of each
  depth 1-5 under confidence, layered post-order and random decoding. Deep
  trees are made rare in training (depth decay 0.5).
* **exp_countdown.py**: exact match on the test set, overall and by solution
  multiplicity (m in [1, 3], [4, 10], >= 11), under confidence,
  step-sequential and random decoding; and selective reveal (half of the gold
  answer tokens revealed, the rest predicted in one forward pass; token
  accuracy). Only 10% of the m in [1, 3] training puzzles are kept.
* **exp_sudoku.py**: blank-cell accuracy and exact match overall, per rating
  tier, and on the first 200 test puzzles whose blank cells are >= 95%
  technique level 4, under confidence, solver-order, technique-order and
  random decoding. The harder rating tiers are sub-sampled in training
  (exponential decay 0.01).

### Data

* Addition, maze and ListOps generate their data from the seed.
* Countdown reads `experiments/data/cd4_train.jsonl` and
  `experiments/data/cd4_test.jsonl`, one
  `{"input": "86,28,13,31,96", "output": "86+28=114,31-13=18,114-18=96"}` per
  line. The test file is included; the training file is the CD4 training split
  of the Countdown data of Ye et al. (2025) cited in the paper, in the same
  format (`--data-dir`, `--train-file` change the paths).
* Sudoku downloads `sapientinc/sudoku-extreme` from the HuggingFace hub.

## Addition analyses

Both scripts read the EMA snapshots written by `exp_addition.py`.

```
# failure analysis of confidence decoding (first wrong commit: position, role,
# confidence, ranking at that commit); writes <checkpoint_dir>/analysis/*.json
python experiments/addition_decode_analysis.py \
    --checkpoint_dir results/exp_addition_main_s42 --iters 300000

# remasking decoders (leave-one-out, random re-noising) and stochastic decoding
python experiments/remasking_analysis.py --mode all \
    --checkpoint-dir results/exp_addition_main_s42 --seeds 42 --iter 300000 \
    --out results/remasking_seed42.json
```
