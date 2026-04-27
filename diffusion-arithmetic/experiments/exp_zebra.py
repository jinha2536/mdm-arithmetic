"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Zebra (Einstein Puzzles) — Parallel Constraints + PUMA Coverage
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Task:     Given symbolic clues about N houses and M attributes,
            fill the N×M solution grid in solver order.
  Format:   [clue tokens ...] ANSWER [row col val row col val ...]
  Example:  "immediate-left LHS c 2 0 RHS c 1 2 CLUE_END ... ANSWER 2 0 0 1 1 2 ..."

  Solution tokens come in triplets: (row, col, value).
  The dataset provides these in solver order (easy cells first),
  so L2R decode = oracle decode — a key advantage over Sudoku.

  Position types within each triplet:
    loc:  row & col positions (which cell to fill — structural)
    val:  value position (what to fill — requires constraint reasoning)

  Rarity axis:     puzzle size N (3→6) and solver step (early=easy, late=hard)
  Dependency:      parallel constraint web — each cell constrained by clues
                   involving potentially any other cell
  Oracle:          L2R = solver order (provided by dataset)
  Latent tokens:   highly effective (He et al., 2026: SIDM→MDM = +25 pts)

  Key question:    Does PUMA's confidence ordering align with solver order?
                   → High concordance predicted → PUMA should win.

  Training:  iter-based with EMA (following ListOps/Countdown pattern)
  Decode:    confidence | l2r (oracle=solver order) | random
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import sys, os, time, math, json, random, pickle, statistics
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                if '__file__' in dir() else '.')
from core.train_utils import (
    mount_drive, save_results, save_checkpoint,
    train_diffusion, puma_k_fixed, puma_k_linear, puma_k_step,
    generate_diffusion, simulate_reveal_trajectory, compute_reveal_vs_order_tau,
    DEVICE,
)

EXP_NAME = 'exp_zebra'

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Config
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MAX_N = 6                   # max houses
MAX_M1 = 6                  # max attributes + 1 (first row = house numbers)
MAX_ANS_TOKENS = 120        # max answer tokens (6*6=36 cells × 3 tokens + pad)
MAX_SEQ_LEN = 600           # total sequence length cap (same as Shah et al.)

# Training
N_TRAIN = None              # None = use all
N_TEST = 2000; BATCH_SIZE = 64
# Per-bucket count for constructed test subsets (puzzle size, solver-difficulty)
N_PER_BUCKET = 500
MAX_ITERS = 500000; EVAL_EVERY = 5000; LOG_EVERY = 1000
GEN_EVAL_EVERY = 20000; GEN_EVAL_N = 500
MASK_TYPES = ['random', 'papl', 'puma']  # spectrum of confidence-alignment intervention
DECODE_POLICIES = ['confidence', 'layered_oracle', 'random']
# layered_oracle = cell-level solver order with row/col/val triplet ties broken
# by confidence. This is the true oracle (vs strict l2r which forces a spurious
# order within each triplet). Replaces the previous 'l2r' which was approximately
# but not exactly oracle.

# Model (He et al. tiny: 3L/384D/12H, ~6M params)
N_LAYER = 3; N_HEAD = 12; N_EMBD = 384; DROPOUT = 0.0; POS_ENC = 'absolute'
LR = 3e-4; MIN_LR = 1e-5; WARMUP_ITERS = 1000; GRAD_CLIP = 1.0
WEIGHT_DECAY = 0.01; EMA_DECAY = 0.9999
PUMA_TAU = 0.9; PUMA_K = 10  # fixed K (unused when K_START is set)
# K range chosen for reveal-per-step alignment. Zebra ans_len=120 (rainbow pad
# included): K=12 → 10 tokens/step, K=60 → 2 tokens/step.
PUMA_K_START = 12; PUMA_K_END = 60
PUMA_K_STEP = 3; PUMA_K_EVERY = None
# PAPL (Peng et al. 2025, arXiv:2509.23405): paper defaults τ=1, α=1.
PAPL_TAU = 1.0; PAPL_ALPHA = 1.0
SEED = 42
NO_AMP = False
PATIENCE = 80000
CONTINUATION_ITERS = 20000

# Data
DATA_DIR = 'experiments/data'
TRAIN_FILE = 'zebra-train-data.pkl'
TEST_FILE = 'zebra-test-data.pkl'

# Rainbow padding — reduced from 16 → 7 per empirical observation (Dueun):
# small word counts (5~7) are sufficient for LLaDA-scale vocabularies;
# more yields little improvement. With 7 rainbow words, each pad position
# token repeats ~13×/sample in 3×3 puzzles (93 pads / 7), enough for the
# model to learn the padding pattern without overwhelming actual cell training.
N_RAINBOW = 7
RAINBOW_WORDS = [f'R{i}' for i in range(N_RAINBOW)]
EOS_WORD = 'EOS'
MASK_WORD = '[MASK]'
PAD_WORD = '[PAD]'
ANSWER_WORD = 'ANSWER'

# Reveal trajectory / reveal-τ diagnostic (PUMA only, extreme-puzzle subset)
# Reasoning order for zebra = solver order (item[2] from data). τ measures
# whether PUMA's confidence-induced reveal order aligns with the deduction
# order a symbolic solver would follow.
REVEAL_K_DEFAULT = 16         # overridden to PUMA K_final at runtime
REVEAL_TAU_MIN_SIZE = 5       # track puzzles with n_houses >= this (harder instances)
REVEAL_TAU_N_TRACKED = 100
REVEAL_TAU_EVERY = 20000
REVEAL_TAU_K_THRESHOLD_FRAC = 0.7


def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', type=str); p.add_argument('--train-file', type=str)
    p.add_argument('--test-file', type=str)
    p.add_argument('--max-ans-tokens', type=int)
    p.add_argument('--n-train', type=int); p.add_argument('--n-test', type=int)
    p.add_argument('--max-iters', type=int); p.add_argument('--batch-size', type=int)
    p.add_argument('--eval-every', type=int); p.add_argument('--gen-eval-every', type=int)
    p.add_argument('--n-layer', type=int); p.add_argument('--n-head', type=int)
    p.add_argument('--n-embd', type=int); p.add_argument('--dropout', type=float)
    p.add_argument('--lr', type=float); p.add_argument('--weight-decay', type=float)
    p.add_argument('--patience', type=int)
    p.add_argument('--no-patience', action='store_true',
                   help='Disable early stopping for fair comparison across mask types')
    p.add_argument('--puma-tau', type=float); p.add_argument('--puma-k', type=int)
    p.add_argument('--puma-k-start', type=int); p.add_argument('--puma-k-end', type=int)
    p.add_argument('--puma-k-step', type=int); p.add_argument('--puma-k-every', type=int)
    p.add_argument('--papl-tau', type=float); p.add_argument('--papl-alpha', type=float)
    p.add_argument('--masks', nargs='+'); p.add_argument('--decode', nargs='+')
    p.add_argument('--continuation-iters', type=int)
    p.add_argument('--no-continuation', action='store_true')
    p.add_argument('--no-amp', action='store_true')
    p.add_argument('--tag', type=str, default=''); p.add_argument('--seed', type=int)
    p.add_argument('--seeds', nargs='+', type=int)
    try:
        args, _ = p.parse_known_args()
    except SystemExit:
        args, _ = p.parse_known_args([])
    g = globals()
    for a, gl in {
        'data_dir': 'DATA_DIR', 'train_file': 'TRAIN_FILE', 'test_file': 'TEST_FILE',
        'max_ans_tokens': 'MAX_ANS_TOKENS',
        'n_train': 'N_TRAIN', 'n_test': 'N_TEST', 'max_iters': 'MAX_ITERS',
        'batch_size': 'BATCH_SIZE', 'eval_every': 'EVAL_EVERY',
        'gen_eval_every': 'GEN_EVAL_EVERY',
        'n_layer': 'N_LAYER', 'n_head': 'N_HEAD', 'n_embd': 'N_EMBD',
        'dropout': 'DROPOUT', 'lr': 'LR', 'weight_decay': 'WEIGHT_DECAY',
        'patience': 'PATIENCE', 'puma_tau': 'PUMA_TAU', 'puma_k': 'PUMA_K',
        'puma_k_start': 'PUMA_K_START', 'puma_k_end': 'PUMA_K_END',
        'puma_k_step': 'PUMA_K_STEP', 'puma_k_every': 'PUMA_K_EVERY',
        'papl_tau': 'PAPL_TAU', 'papl_alpha': 'PAPL_ALPHA',
        'seed': 'SEED', 'no_amp': 'NO_AMP',
        'continuation_iters': 'CONTINUATION_ITERS',
    }.items():
        v = getattr(args, a, None)
        if v is not None:
            g[gl] = v
    if args.masks:
        g['MASK_TYPES'] = args.masks
    if args.decode:
        g['DECODE_POLICIES'] = args.decode
    if getattr(args, 'no_patience', False):
        g['PATIENCE'] = None
    return args


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Word-level tokenizer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class WordTokenizer:
    """Word-level tokenizer with same interface as CharTokenizer.

    encode: list[str] → list[int]
    decode: list[int] → list[str]
    """
    def __init__(self, vocab_words, mask_word=MASK_WORD, pad_word=PAD_WORD):
        self.mask_word = mask_word
        self.pad_word = pad_word
        # Reserve 0=mask, 1=pad
        self.word_to_id = {mask_word: 0, pad_word: 1}
        for i, w in enumerate(vocab_words):
            if w not in self.word_to_id:
                self.word_to_id[w] = len(self.word_to_id)
        self.id_to_word = {v: k for k, v in self.word_to_id.items()}
        self.vocab_size = len(self.word_to_id)
        self.special_ids = {'mask': 0, 'pad': 1}
        # Alias for compatibility with code expecting char_to_id
        self.char_to_id = self.word_to_id

    def __len__(self):
        return self.vocab_size

    def encode(self, word_list):
        """Encode list of word strings → list of int IDs."""
        if isinstance(word_list, str):
            word_list = word_list.split()
        return [self.word_to_id.get(w, self.special_ids['pad']) for w in word_list]

    def decode(self, id_list):
        """Decode list of int IDs → list of word strings."""
        return [self.id_to_word.get(i, '?') for i in id_list]

    def decode_str(self, id_list):
        """Decode to space-joined string."""
        return ' '.join(self.decode(id_list))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Data loading & formatting
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_zebra_pickle(filepath, max_n=None, seed=42):
    """Load Zebra pickle data.

    Each sample is [token_list, solution_grid, solver_order_or_info].
    Returns: list of dicts with parsed fields.
    """
    with open(filepath, 'rb') as f:
        raw = pickle.load(f)

    rng = random.Random(seed)
    indices = list(range(len(raw)))
    rng.shuffle(indices)
    if max_n:
        indices = indices[:max_n]

    samples = []
    for idx in indices:
        item = raw[idx]
        tokens = list(item[0])              # word token sequence
        solution_grid = np.array(item[1])   # (M+1, N) array
        # Third element: solver order as list of [row, col] pairs or similar
        solver_info = item[2] if len(item) > 2 else None

        # Find ANSWER token position
        ans_pos = None
        for i, t in enumerate(tokens):
            if t == ANSWER_WORD:
                ans_pos = i
                break
        if ans_pos is None:
            continue

        clue_tokens = tokens[:ans_pos + 1]          # including ANSWER
        solution_tokens = tokens[ans_pos + 1:]       # after ANSWER

        # Parse puzzle dimensions from solution grid
        n_houses = solution_grid.shape[1]
        n_attrs = solution_grid.shape[0] - 1  # first row = house numbers

        # Parse solution triplets
        n_cells = len(solution_tokens) // 3
        triplets = []
        for ci in range(n_cells):
            row_tok = solution_tokens[ci * 3]
            col_tok = solution_tokens[ci * 3 + 1]
            val_tok = solution_tokens[ci * 3 + 2]
            triplets.append((row_tok, col_tok, val_tok))

        # Solver order (if available)
        solver_order = None
        if solver_info is not None:
            if isinstance(solver_info, (list, np.ndarray)):
                solver_order = [tuple(x) for x in solver_info]

        samples.append({
            'tokens': tokens,
            'clue_tokens': clue_tokens,
            'solution_tokens': solution_tokens,
            'solution_grid': solution_grid,
            'n_houses': n_houses,
            'n_attrs': n_attrs,
            'n_cells': n_cells,
            'triplets': triplets,
            'solver_order': solver_order,
            'ans_pos': ans_pos,
        })
    return samples


def build_tokenizer(samples):
    """Build WordTokenizer from data vocabulary."""
    vocab = set()
    for s in samples:
        vocab.update(s['tokens'])
    # Add special tokens
    vocab.discard(MASK_WORD)
    vocab.discard(PAD_WORD)
    vocab_sorted = sorted(vocab)
    # Add rainbow + EOS
    extra = [EOS_WORD] + RAINBOW_WORDS
    for e in extra:
        if e not in vocab_sorted:
            vocab_sorted.append(e)
    return WordTokenizer(vocab_sorted)


def _rainbow_pad_words(word_list, target_len):
    """Pad word list to target_len with EOS + rainbow words."""
    remaining = target_len - len(word_list)
    if remaining <= 0:
        return word_list[:target_len]
    pad = [EOS_WORD]
    for i in range(remaining - 1):
        pad.append(RAINBOW_WORDS[i % len(RAINBOW_WORDS)])
    return word_list + pad


def format_samples(raw_samples, max_ans_tokens=None):
    """Format raw samples into (token_id_ready_lists, metas).

    Returns:
        formatted: list of word-token lists (clue + ANSWER + solution_padded)
        metas: list of metadata dicts.

    Note on solver order:
        The dataset's solution_tokens are already laid out in solver order.
        Each cell's 3 tokens (row, col, val) share the same solver step rank.
        We store `reasoning_rank[j]` = floor(j/3) for j < n_sol, and a high
        dummy rank (max_ans_tokens) for padding — downstream reveal-τ uses
        blank_masks to exclude padding anyway, but the dummy fill keeps
        tensor shapes regular.
    """
    if max_ans_tokens is None:
        max_ans_tokens = MAX_ANS_TOKENS

    formatted, metas = [], []
    skipped = 0
    for s in raw_samples:
        sol_padded = _rainbow_pad_words(s['solution_tokens'], max_ans_tokens)
        full_seq = s['clue_tokens'] + sol_padded
        if len(full_seq) > MAX_SEQ_LEN:
            skipped += 1
            continue

        # Position types for answer region: loc (row/col) vs val
        n_sol = len(s['solution_tokens'])
        pos_types = []
        reasoning_rank = []
        for ci in range(max_ans_tokens):
            if ci < n_sol:
                pos_in_triplet = ci % 3  # 0=row, 1=col, 2=val
                pos_types.append('loc' if pos_in_triplet < 2 else 'val')
                reasoning_rank.append(ci // 3)  # cell index (= solver step)
            else:
                pos_types.append('pad')
                reasoning_rank.append(max_ans_tokens)  # dummy high rank

        # Solver step for each answer token (which cell, in order).
        # Same as reasoning_rank but with -1 for padding.
        solver_steps = [(ci // 3) if ci < n_sol else -1
                        for ci in range(max_ans_tokens)]

        meta = {
            'n_houses': s['n_houses'],
            'n_attrs': s['n_attrs'],
            'n_cells': s['n_cells'],
            'n_sol_tokens': n_sol,
            'solution_grid': s['solution_grid'],
            'solver_order': s['solver_order'],
            'pos_types': pos_types,             # 'loc' | 'val' | 'pad'
            'solver_steps': solver_steps,        # cell index per token, -1 for pad
            'reasoning_rank': reasoning_rank,    # [max_ans_tokens] — solver step rank per answer pos
            'triplets': s['triplets'],
            'clue_len': len(s['clue_tokens']),
        }
        formatted.append(full_seq)
        metas.append(meta)

    if skipped:
        print(f"  Skipped {skipped} samples (too long)")
    return formatted, metas


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Encoding: word lists → tensor (replaces encode_samples)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def encode_zebra_samples(formatted_samples, tokenizer, max_len):
    """Encode word-token lists to padded tensors.

    Returns:
        ids: (N, max_len) int64 tensor
        ans_starts: (N,) int64 tensor — index of first answer token

    Note on padding philosophy:
        The answer region (length MAX_ANS_TOKENS, fixed) is entirely part of
        the diffusion target. Rainbow padding fills the region after the
        actual solution with diverse tokens (EOS, R0, R1, ..., RAINBOW_WORDS[-1])
        so that padding does not concentrate probability mass on a single
        learnable-by-position constant. This matches the PUMA paper's
        treatment where EOS is identified with PAD and the whole fixed-length
        answer region participates in masking/restoration.
    """
    pad_id = tokenizer.special_ids['pad']
    N = len(formatted_samples)
    ids = torch.full((N, max_len), pad_id, dtype=torch.long)
    ans_starts = torch.zeros(N, dtype=torch.long)

    for i, word_list in enumerate(formatted_samples):
        encoded = tokenizer.encode(word_list)
        L = min(len(encoded), max_len)
        ids[i, :L] = torch.tensor(encoded[:L], dtype=torch.long)
        # Find ANSWER token position — answer starts right after it
        answer_id = tokenizer.word_to_id.get(ANSWER_WORD, -1)
        for j in range(L):
            if ids[i, j].item() == answer_id:
                ans_starts[i] = j + 1
                break
    return ids, ans_starts


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Unified test suite — all analyses slice from here
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _bucket_from_items(formatted, metas, tokenizer, max_len):
    """Package a (formatted, metas) pair with encoded ids for batched analysis."""
    if not formatted:
        return {'formatted': [], 'metas': [],
                'ids': torch.empty(0, max_len, dtype=torch.long),
                'ans_starts': torch.empty(0, dtype=torch.long), 'n': 0}
    ids, ans = encode_zebra_samples(formatted, tokenizer, max_len)
    return {
        'formatted': formatted, 'metas': metas,
        'ids': ids, 'ans_starts': ans, 'n': len(formatted),
    }


def build_test_suite(formatted_all, metas_all, tokenizer, max_len):
    """Unified test suite for zebra: slices from the full test set.

    Structure:
        suite['natural']:                 all test samples (up to N_TEST)
        suite['constructed']['size_N']:   puzzles with n_houses == N
        suite['constructed']['hard']:     largest puzzle size (n_houses=MAX_N)
                                          or difficulty-filtered if available
    All slicing is index-based (no regeneration), preserving the same samples
    across analyses for cross-referencing.
    """
    n_full = min(len(formatted_all), N_TEST)
    suite = {
        'natural': _bucket_from_items(
            formatted_all[:n_full], metas_all[:n_full], tokenizer, max_len),
        'constructed': {},
    }

    # Per-size buckets
    for size in [3, 4, 5, 6]:
        idx = [i for i, m in enumerate(metas_all[:n_full])
               if m['n_houses'] == size]
        if not idx:
            continue
        idx = idx[:N_PER_BUCKET]
        suite['constructed'][f'size_{size}'] = _bucket_from_items(
            [formatted_all[i] for i in idx],
            [metas_all[i] for i in idx],
            tokenizer, max_len)

    # Hardest bucket: largest puzzles (primary extreme axis for zebra)
    for size in [MAX_N, MAX_N - 1]:
        key = f'size_{size}'
        if key in suite['constructed'] and suite['constructed'][key]['n'] >= 50:
            suite['constructed']['hard'] = suite['constructed'][key]
            break

    print(f"  Zebra test suite built:")
    print(f"    natural: {suite['natural']['n']}")
    for k, v in suite['constructed'].items():
        print(f"    constructed/{k}: {v['n']}")
    return suite


def filter_natural(suite, pred):
    """Filter natural bucket by predicate on meta; return same-shape bucket."""
    nat = suite['natural']
    idx = [i for i, m in enumerate(nat['metas']) if pred(m)]
    if not idx:
        return {'formatted': [], 'metas': [],
                'ids': torch.empty(0, nat['ids'].shape[1], dtype=torch.long),
                'ans_starts': torch.empty(0, dtype=torch.long), 'n': 0}
    return {
        'formatted': [nat['formatted'][i] for i in idx],
        'metas': [nat['metas'][i] for i in idx],
        'ids': nat['ids'][idx],
        'ans_starts': nat['ans_starts'][idx],
        'n': len(idx),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Training strata — by puzzle size (proxy for difficulty)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STRATUM_BOUNDS_SIZE = [(3, 4), (4, 5), (5, 6), (6, 7)]
STRATUM_NAMES = ['size_3', 'size_4', 'size_5', 'size_6']


def _size_to_stratum(n_houses):
    for i, (lo, hi) in enumerate(STRATUM_BOUNDS_SIZE):
        if lo <= n_houses < hi:
            return i
    return len(STRATUM_BOUNDS_SIZE) - 1


def build_training_strata(train_metas):
    """Per-sample stratum id by puzzle size (n_houses)."""
    strata = [_size_to_stratum(m['n_houses']) for m in train_metas]
    counts = [strata.count(i) for i in range(len(STRATUM_NAMES))]
    return torch.tensor(strata, dtype=torch.long), STRATUM_NAMES, counts


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Validation: check puzzle correctness
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def validate_zebra(pred_words, meta, tokenizer):
    """Validate predicted solution against ground truth.

    pred_words: list of word strings for the answer region
    Returns dict with detailed metrics.
    """
    n_sol = meta['n_sol_tokens']
    n_cells = meta['n_cells']
    grid = meta['solution_grid']  # (M+1, N)
    n_houses = meta['n_houses']
    n_attrs = meta['n_attrs']

    # Trim to actual solution length (before EOS/rainbow)
    pred_sol = pred_words[:n_sol]
    gold_triplets = meta['triplets']

    # Per-triplet check
    cell_correct = 0
    loc_correct = 0
    val_correct = 0
    loc_total = 0
    val_total = 0
    per_cell = []

    for ci in range(min(n_cells, len(pred_sol) // 3)):
        pi = ci * 3
        if pi + 2 >= len(pred_sol):
            break
        p_row, p_col, p_val = pred_sol[pi], pred_sol[pi + 1], pred_sol[pi + 2]
        g_row, g_col, g_val = gold_triplets[ci]

        row_ok = (p_row == g_row)
        col_ok = (p_col == g_col)
        val_ok = (p_val == g_val)
        cell_ok = row_ok and col_ok and val_ok

        loc_total += 2
        val_total += 1
        loc_correct += int(row_ok) + int(col_ok)
        val_correct += int(val_ok)
        cell_correct += int(cell_ok)

        per_cell.append({
            'cell_idx': ci,
            'correct': cell_ok,
            'row_ok': row_ok, 'col_ok': col_ok, 'val_ok': val_ok,
        })

    # Full puzzle correctness
    full_correct = (cell_correct == n_cells) if n_cells > 0 else False

    # Check against solution grid for structural validity
    model_grid = np.zeros_like(grid)
    grid_valid = True
    for ci in range(min(n_cells, len(pred_sol) // 3)):
        pi = ci * 3
        if pi + 2 >= len(pred_sol):
            grid_valid = False
            break
        try:
            r = int(pred_sol[pi])
            c = int(pred_sol[pi + 1])
            v = int(pred_sol[pi + 2])
            if 0 <= r < n_attrs and 0 <= c < n_houses and 0 <= v < n_houses:
                model_grid[r + 1, c] = v
            else:
                grid_valid = False
        except (ValueError, IndexError):
            grid_valid = False

    return {
        'full_correct': full_correct,
        'cell_correct': cell_correct,
        'n_cells': n_cells,
        'cell_accuracy': cell_correct / max(n_cells, 1),
        'loc_accuracy': loc_correct / max(loc_total, 1),
        'val_accuracy': val_correct / max(val_total, 1),
        'grid_valid': grid_valid,
        'per_cell': per_cell,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Probe: per-position classification (vectorized)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@torch.no_grad()
def probe_per_position(model, tokenizer, formatted_samples, metas,
                       max_len, device=None):
    """Vectorized per-position accuracy probe, stratified by type & solver step."""
    if device is None:
        device = DEVICE
    mask_id = tokenizer.special_ids['mask']
    pad_id = tokenizer.special_ids['pad']
    model.eval()

    ids_all, ans_all = encode_zebra_samples(formatted_samples, tokenizer, max_len)
    ids_all, ans_all = ids_all.to(device), ans_all.to(device)
    N = len(formatted_samples)

    # Accumulators
    type_acc = defaultdict(lambda: [0, 0])  # type → [correct, total]
    size_acc = defaultdict(lambda: [0, 0])  # puzzle_size → [correct, total]
    step_acc = defaultdict(lambda: [0, 0])  # solver_step_bin → [correct, total]
    total_loss = 0.0
    total_count = 0

    _arange = torch.arange(MAX_ANS_TOKENS, device=device)

    for st in range(0, N, BATCH_SIZE):
        en = min(st + BATCH_SIZE, N)
        ids = ids_all[st:en]
        ans = ans_all[st:en]
        B, T = ids.shape

        # Build answer position indices: (B, MAX_ANS_TOKENS)
        ans_pos = (ans.unsqueeze(1) + _arange).clamp(max=T - 1)
        bi = torch.arange(B, device=device).unsqueeze(1).expand_as(ans_pos)

        # Mask answer region
        xm = ids.clone()
        xm[bi, ans_pos] = mask_id

        logits = model(xm)

        # Extract answer logits and targets: (B, MAX_ANS_TOKENS, V)
        ans_logits = logits[bi, ans_pos]
        ans_targets = ids[bi, ans_pos]

        # Predictions
        ans_preds = ans_logits.argmax(dim=-1)  # (B, MAX_ANS_TOKENS)
        correct_mask = (ans_preds == ans_targets)  # (B, MAX_ANS_TOKENS)
        valid_mask = (ans_targets != pad_id)       # (B, MAX_ANS_TOKENS)

        # Loss (vectorized)
        flat_logits = ans_logits.reshape(-1, ans_logits.shape[-1])
        flat_targets = ans_targets.reshape(-1)
        flat_valid = valid_mask.reshape(-1)
        if flat_valid.any():
            losses = F.cross_entropy(flat_logits[flat_valid],
                                     flat_targets[flat_valid], reduction='sum')
            total_loss += losses.item()
            total_count += flat_valid.sum().item()

        # Per-sample stratification (needs metadata — CPU loop but over B only)
        correct_cpu = correct_mask.cpu()
        valid_cpu = valid_mask.cpu()
        for bi_idx in range(B):
            si = st + bi_idx
            meta = metas[si]
            pos_types = meta['pos_types']
            solver_steps = meta['solver_steps']
            n_houses = meta['n_houses']

            for j in range(min(MAX_ANS_TOKENS, len(pos_types))):
                if not valid_cpu[bi_idx, j].item():
                    continue
                c = correct_cpu[bi_idx, j].item()

                # By position type
                ptype = pos_types[j]
                type_acc[ptype][0] += c
                type_acc[ptype][1] += 1

                # By puzzle size
                size_acc[n_houses][0] += c
                size_acc[n_houses][1] += 1

                # By solver step (bin into early/mid/late thirds)
                cell_idx = solver_steps[j]
                if cell_idx >= 0:
                    n_cells = meta['n_cells']
                    if cell_idx < n_cells // 3:
                        step_bin = 'early'
                    elif cell_idx < 2 * n_cells // 3:
                        step_bin = 'mid'
                    else:
                        step_bin = 'late'
                    step_acc[step_bin][0] += c
                    step_acc[step_bin][1] += 1

    result = {
        'overall_loss': total_loss / max(total_count, 1),
        'overall_acc': sum(v[0] for v in type_acc.values()) /
                       max(sum(v[1] for v in type_acc.values()), 1),
    }
    for k, (c, t) in type_acc.items():
        result[f'acc_{k}'] = c / max(t, 1)
    for k, (c, t) in size_acc.items():
        result[f'acc_size_{k}'] = c / max(t, 1)
    for k, (c, t) in step_acc.items():
        result[f'acc_step_{k}'] = c / max(t, 1)
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Reveal trajectory analysis — PUMA-specific failure diagnostic
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@torch.no_grad()
def analyse_reveal_patterns(model, tokenizer, bucket, max_len,
                             K=REVEAL_K_DEFAULT, tau=PUMA_TAU, device=None):
    """Analyze PUMA reveal patterns on extreme puzzles.

    Aggregates over answer-region positions:
      (a) still-masked fraction per PUMA stage, by solver-step bin (early/mid/late)
      (b) per-solver-step mean reveal stage
      (c) never-revealed fraction per answer position
      (d) premature-reveal rate: fraction of (example, cell) where the cell is
          revealed BEFORE its solver step is reached.

    The premature-reveal rate is the central zebra diagnostic: solver orders
    are instance-specific, so a model that reveals cells before the solver
    would have deduced them is committing on insufficient evidence.
    """
    if device is None:
        device = DEVICE
    if bucket['n'] == 0:
        return {'n': 0}

    # blank_masks = all True for zebra (full answer region is diffusion target)
    N = bucket['n']
    blank_masks = torch.ones(N, MAX_ANS_TOKENS, dtype=torch.bool)

    traj = simulate_reveal_trajectory(
        model, tokenizer, bucket['ids'], bucket['ans_starts'], MAX_ANS_TOKENS,
        blank_masks=blank_masks, K=K, tau=tau, device=device)

    rs = traj['reveal_stage']           # [N, MAX_ANS_TOKENS]
    smm = traj['still_masked_start']    # [N, K+1, MAX_ANS_TOKENS]

    # (a) still-masked per stage, binned by solver-step third (early/mid/late)
    bins_acc = {'early': [], 'mid': [], 'late': [], 'pad': []}
    for i, meta in enumerate(bucket['metas']):
        n_cells = meta['n_cells']
        solver_steps = meta['solver_steps']
        for j in range(MAX_ANS_TOKENS):
            cell_idx = solver_steps[j]
            if cell_idx < 0:
                bins_acc['pad'].append(smm[i, :K, j].float())
            elif cell_idx < n_cells // 3:
                bins_acc['early'].append(smm[i, :K, j].float())
            elif cell_idx < 2 * n_cells // 3:
                bins_acc['mid'].append(smm[i, :K, j].float())
            else:
                bins_acc['late'].append(smm[i, :K, j].float())
    by_step_bin = {}
    for b, xs in bins_acc.items():
        if xs:
            by_step_bin[b] = {
                'still_masked_per_stage': torch.stack(xs).mean(dim=0).tolist(),
                'n_positions': len(xs),
            }

    # (b) per-solver-step mean reveal stage, indexed by solver step rank
    # (x = solver step rank, y = avg PUMA stage at which that cell gets revealed)
    step_reveals = defaultdict(list)   # solver_step → list of reveal stages
    for i, meta in enumerate(bucket['metas']):
        solver_steps = meta['solver_steps']
        for j in range(MAX_ANS_TOKENS):
            cs = solver_steps[j]
            if cs >= 0:
                step_reveals[cs].append(float(rs[i, j]))
    solver_to_reveal = [
        {'solver_step': k, 'mean_reveal_stage': sum(v) / len(v), 'n': len(v)}
        for k, v in sorted(step_reveals.items())
    ]

    # (c) never-revealed per position
    never = (rs >= K).float().mean(dim=0).tolist()

    # (d) premature-reveal rate — fraction of cells revealed before solver
    # would have deduced them. For each (example, cell), premature if the
    # reveal stage of any cell token is earlier than expected given solver step.
    # Normalize expected stage as solver_step / n_cells * K.
    premature_per_bin = {'early': [0, 0], 'mid': [0, 0], 'late': [0, 0]}
    for i, meta in enumerate(bucket['metas']):
        n_cells = meta['n_cells']
        solver_steps = meta['solver_steps']
        for j in range(MAX_ANS_TOKENS):
            cs = solver_steps[j]
            if cs < 0:
                continue
            expected_stage = cs / max(n_cells - 1, 1) * (K - 1)
            actual_stage = float(rs[i, j])
            # Premature: decoded noticeably before expected (threshold: 1 stage early)
            is_premature = actual_stage < expected_stage - 1
            if cs < n_cells // 3: bin_name = 'early'
            elif cs < 2 * n_cells // 3: bin_name = 'mid'
            else: bin_name = 'late'
            premature_per_bin[bin_name][0] += int(is_premature)
            premature_per_bin[bin_name][1] += 1
    premature_rate = {
        b: {'rate': c / max(t, 1), 'n': t}
        for b, (c, t) in premature_per_bin.items() if t > 0
    }

    return {
        'n': N, 'K': K, 'tau': tau,
        'by_solver_step_bin': by_step_bin,
        'solver_to_reveal': solver_to_reveal,
        'never_revealed': never,
        'premature_rate': premature_rate,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Generation evaluation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@torch.no_grad()
def gen_eval(model, tokenizer, formatted_samples, metas, max_len,
             decode_policy='confidence', n=None, device=None):
    """Full generation evaluation with validation and concordance."""
    if device is None:
        device = DEVICE
    if n is not None:
        formatted_samples = formatted_samples[:n]
        metas = metas[:n]

    mask_id = tokenizer.special_ids['mask']
    pad_id = tokenizer.special_ids['pad']
    answer_id = tokenizer.word_to_id[ANSWER_WORD]
    model.eval()

    results = []

    # Group by clue length (prefix length) for efficient batching
    groups = {}
    for idx, (sample, meta) in enumerate(zip(formatted_samples, metas)):
        cl = meta['clue_len']
        groups.setdefault(cl, []).append(idx)

    for cl, indices in groups.items():
        for bstart in range(0, len(indices), 64):
            bind = indices[bstart:bstart + 64]
            B = len(bind)

            # Encode prefix (clue tokens including ANSWER)
            batch_samples = [formatted_samples[i] for i in bind]
            batch_metas = [metas[i] for i in bind]
            prefix_lists = [s[:cl] for s in batch_samples]

            # Encode to tensor
            max_prefix_len = max(len(p) for p in prefix_lists)
            pids = torch.full((B, max_prefix_len), pad_id, dtype=torch.long)
            for bi, plist in enumerate(prefix_lists):
                enc = tokenizer.encode(plist)
                pids[bi, :len(enc)] = torch.tensor(enc, dtype=torch.long)

            gen, _, info = generate_diffusion(
                model, pids, MAX_ANS_TOKENS, mask_id,
                policy=decode_policy, greedy=True,
                reasoning_rank=(
                    torch.tensor([m['reasoning_rank'] for m in batch_metas],
                                 dtype=torch.long)
                    if decode_policy == 'layered_oracle' else None),
                pad_to=max_len, pad_id=pad_id, device=device)

            pred_ans_ids = gen[:, max_prefix_len:max_prefix_len + MAX_ANS_TOKENS]

            for bi in range(B):
                meta = batch_metas[bi]
                pred_words = tokenizer.decode(pred_ans_ids[bi].cpu().tolist())
                gold_words = formatted_samples[bind[bi]][cl:]

                # Validate
                val = validate_zebra(pred_words, meta, tokenizer)

                # Per-token exact match
                n_sol = meta['n_sol_tokens']
                token_match = sum(1 for j in range(min(n_sol, len(pred_words)))
                                  if pred_words[j] == gold_words[j])

                # Per-solver-step cell accuracy (early/mid/late)
                n_cells = meta['n_cells']
                step_thirds = {}  # 'early'/'mid'/'late' → [correct, total]
                for ci, pc in enumerate(val.get('per_cell', [])):
                    if ci < n_cells // 3:
                        b = 'early'
                    elif ci < 2 * n_cells // 3:
                        b = 'mid'
                    else:
                        b = 'late'
                    step_thirds.setdefault(b, [0, 0])
                    step_thirds[b][0] += int(pc['correct'])
                    step_thirds[b][1] += 1

                # Concordance with solver order (= L2R)
                concordance = None
                if info.get('orders') is not None:
                    raw_order = info['orders'][bi]
                    if hasattr(raw_order, 'tolist'):
                        raw_order = raw_order.tolist()
                    concordance = _concordance_with_l2r(
                        raw_order, max_prefix_len, MAX_ANS_TOKENS)

                results.append({
                    'full_correct': val['full_correct'],
                    'cell_accuracy': val['cell_accuracy'],
                    'loc_accuracy': val['loc_accuracy'],
                    'val_accuracy': val['val_accuracy'],
                    'token_accuracy': token_match / max(n_sol, 1),
                    'n_houses': meta['n_houses'],
                    'n_cells': meta['n_cells'],
                    'concordance_l2r': concordance,
                    'step_thirds': step_thirds,
                })

    # Aggregate
    n_total = len(results)
    if n_total == 0:
        return {'accuracy': 0, 'n': 0}

    agg = {
        'accuracy': sum(r['full_correct'] for r in results) / n_total,
        'cell_accuracy': sum(r['cell_accuracy'] for r in results) / n_total,
        'loc_accuracy': sum(r['loc_accuracy'] for r in results) / n_total,
        'val_accuracy': sum(r['val_accuracy'] for r in results) / n_total,
        'token_accuracy': sum(r['token_accuracy'] for r in results) / n_total,
        'n': n_total,
    }

    # Per puzzle size
    for n_h in sorted(set(r['n_houses'] for r in results)):
        sub = [r for r in results if r['n_houses'] == n_h]
        agg[f'acc_size_{n_h}'] = sum(r['full_correct'] for r in sub) / len(sub)
        agg[f'cell_acc_size_{n_h}'] = sum(r['cell_accuracy'] for r in sub) / len(sub)
        agg[f'n_size_{n_h}'] = len(sub)

    # Concordance
    conc = [r['concordance_l2r'] for r in results if r['concordance_l2r'] is not None]
    if conc:
        agg['concordance_l2r'] = sum(conc) / len(conc)

    # Per-solver-step accuracy (early/mid/late — key for paper Figure 9)
    for step_bin in ['early', 'mid', 'late']:
        corr_sum, total_sum = 0, 0
        for r in results:
            st = r.get('step_thirds', {}).get(step_bin)
            if st:
                corr_sum += st[0]
                total_sum += st[1]
        if total_sum > 0:
            agg[f'cell_acc_step_{step_bin}'] = corr_sum / total_sum

    # Examples
    agg['examples'] = results[:5]
    return agg


def _concordance_with_l2r(decode_order, prefix_len, ans_len):
    """Compute concordance between model's decode order and L2R (solver) order.

    L2R oracle order = 0, 1, 2, ..., ans_len-1 (since data is already in solver order).
    decode_order: list of absolute positions in the order they were decoded.
    Returns concordance in [0, 1].
    """
    # Extract answer positions in decode order
    ans_decode_rank = {}
    for step, abs_pos in enumerate(decode_order):
        ans_pos = abs_pos - prefix_len
        if 0 <= ans_pos < ans_len:
            ans_decode_rank[ans_pos] = step

    # L2R oracle: position 0 decoded first, 1 second, etc.
    # So oracle rank of position j = j
    concordant = 0
    discordant = 0
    positions = sorted(ans_decode_rank.keys())
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            pi, pj = positions[i], positions[j]
            # Oracle: pi < pj (since L2R)
            # Model: compare decode ranks
            d_diff = ans_decode_rank[pi] - ans_decode_rank[pj]
            o_diff = pi - pj  # always negative since pi < pj
            if d_diff * o_diff > 0:
                concordant += 1
            elif d_diff * o_diff < 0:
                discordant += 1
    total = concordant + discordant
    return concordant / total if total > 0 else 0.5


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Selective masking experiment
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@torch.no_grad()
def eval_selective_mask(model, tokenizer, formatted_samples, metas, max_len,
                        reveal_fraction=0.5, device=None):
    """Reveal early solver steps (easy cells) as GT, mask only late steps.

    Tests whether PUMA's advantage grows when stepping stones are given.
    """
    if device is None:
        device = DEVICE
    mask_id = tokenizer.special_ids['mask']
    pad_id = tokenizer.special_ids['pad']
    model.eval()

    ids_all, ans_all = encode_zebra_samples(formatted_samples, tokenizer, max_len)
    ids_all, ans_all = ids_all.to(device), ans_all.to(device)

    _arange = torch.arange(MAX_ANS_TOKENS, device=device)
    correct_total = 0
    valid_total = 0

    for st in range(0, len(formatted_samples), BATCH_SIZE):
        en = min(st + BATCH_SIZE, len(formatted_samples))
        ids = ids_all[st:en]
        ans = ans_all[st:en]
        B, T = ids.shape

        ans_pos = (ans.unsqueeze(1) + _arange).clamp(max=T - 1)
        bi = torch.arange(B, device=device).unsqueeze(1).expand_as(ans_pos)

        # Selectively mask: reveal first reveal_fraction of answer tokens
        xm = ids.clone()
        for bi_idx in range(B):
            meta = metas[st + bi_idx]
            n_sol = meta['n_sol_tokens']
            n_reveal = int(n_sol * reveal_fraction)
            # Round to triplet boundary
            n_reveal = (n_reveal // 3) * 3
            # Mask only positions after n_reveal
            for j in range(MAX_ANS_TOKENS):
                abs_pos = ans_pos[bi_idx, j].item()
                if j >= n_reveal:
                    xm[bi_idx, abs_pos] = mask_id

        logits = model(xm)
        ans_logits = logits[bi, ans_pos]
        ans_preds = ans_logits.argmax(dim=-1)
        ans_targets = ids[bi, ans_pos]

        for bi_idx in range(B):
            meta = metas[st + bi_idx]
            n_sol = meta['n_sol_tokens']
            n_reveal = (int(n_sol * reveal_fraction) // 3) * 3
            # Check correctness only on masked (late) positions
            for j in range(n_reveal, n_sol):
                valid_total += 1
                if ans_preds[bi_idx, j].item() == ans_targets[bi_idx, j].item():
                    correct_total += 1

    return {
        'reveal_fraction': reveal_fraction,
        'masked_accuracy': correct_total / max(valid_total, 1),
        'n_masked_tokens': valid_total,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Training wrapper
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def train_model(mask_type, tokenizer, train_formatted, train_metas,
                test_formatted, test_metas, max_len,
                max_iters=None, init_state=None, device=None):
    """Wrapper around train_diffusion for Zebra experiment.

    Adds training-side diagnostics:
      - per-puzzle-size stratum training-loss trajectory (both mask types)
      - reveal-vs-solver-order Kendall τ on a tracked extreme-puzzle subset
        (PUMA only, activated once K_cur reaches ~70% of K_final)
    """
    if device is None:
        device = DEVICE
    if max_iters is None:
        max_iters = MAX_ITERS

    train_ids, train_ans = encode_zebra_samples(train_formatted, tokenizer, max_len)
    train_ids, train_ans = train_ids.to(device), train_ans.to(device)

    # Training strata (puzzle size buckets)
    sample_strata, stratum_names, stratum_counts = build_training_strata(train_metas)
    print(f"  Training strata counts: " +
          ', '.join(f"{n}={c}" for n, c in zip(stratum_names, stratum_counts)))

    # ── Reveal-τ tracked subset (PUMA only) ──
    # Pick tracked samples from the LARGEST puzzle size present in train set.
    # Reasoning order = solver order, which is exactly the answer-position
    # order since solution tokens are laid out in solver order by the dataset.
    # Each cell's 3 tokens share the same reasoning rank (cell index = j // 3).
    reveal_tracked_ids = None
    reveal_tracked_ans = None
    reveal_reasoning_order = None
    reveal_blanks = None
    reveal_tracked_strata = None
    # Build tracked subset for ALL mask_types (diagnostic measures confidence-greedy
    # reveal order regardless of how model was trained). Per-stratum tracked
    # indices — one trajectory per puzzle size.
    per_stratum_cap = max(REVEAL_TAU_N_TRACKED // len(STRATUM_NAMES), 10)
    tracked_by_stratum = {sn: [] for sn in STRATUM_NAMES}
    for i, m in enumerate(train_metas):
        si = _size_to_stratum(m['n_houses'])
        sn = STRATUM_NAMES[si]
        if len(tracked_by_stratum[sn]) < per_stratum_cap:
            tracked_by_stratum[sn].append((i, si))
    tracked_flat = [t for v in tracked_by_stratum.values() for t in v]
    strata_counts = {sn: len(v) for sn, v in tracked_by_stratum.items()}
    if sum(strata_counts.values()) >= 30:
        tracked = [t[0] for t in tracked_flat]
        tracked_strata = [t[1] for t in tracked_flat]
        reveal_tracked_ids = train_ids[tracked]
        reveal_tracked_ans = train_ans[tracked]
        reveal_tracked_strata = torch.tensor(tracked_strata, dtype=torch.long)
        N_tr = len(tracked)
        ro = torch.full((N_tr, MAX_ANS_TOKENS), MAX_ANS_TOKENS, dtype=torch.long)
        bm = torch.zeros(N_tr, MAX_ANS_TOKENS, dtype=torch.bool)
        for i, idx in enumerate(tracked):
            meta = train_metas[idx]
            rr = meta['reasoning_rank']
            for j in range(MAX_ANS_TOKENS):
                ro[i, j] = rr[j]
            # Only include actual solution tokens (not rainbow pad).
            n_sol = meta['n_sol_tokens']
            bm[i, :min(n_sol, MAX_ANS_TOKENS)] = True
        reveal_reasoning_order = ro
        reveal_blanks = bm
        print(f"  Reveal-τ tracking (stratified, [{mask_type}]): total={N_tr}, by stratum: " +
              ', '.join(f"{sn}={c}" for sn, c in strata_counts.items() if c > 0))
    else:
        print(f"  Reveal-τ tracking: SKIPPED "
              f"(total tracked < 30: {strata_counts})")

    # PUMA K schedule
    k_sched = None
    if mask_type == 'puma':
        if PUMA_K_START is not None and PUMA_K_END is not None:
            k_step = PUMA_K_STEP or 3
            if PUMA_K_EVERY is not None:
                k_every = PUMA_K_EVERY
            else:
                n_inc = max(1, (PUMA_K_END - PUMA_K_START) // k_step)
                k_every = max(1000, (max_iters // 3) // n_inc)
            k_sched = puma_k_step(PUMA_K_START, PUMA_K_END, k_step, k_every)
            final_k = k_sched(max_iters)
            print(f"  PUMA K: step {PUMA_K_START}→{final_k} "
                  f"(+{k_step} every {k_every // 1000}k, cap={PUMA_K_END})")
        else:
            k_sched = puma_k_fixed(PUMA_K)
            print(f"  PUMA K: fixed {PUMA_K}")
    # Diagnostic K: PUMA uses live k_sched, random/PAPL use PUMA_K_END as fixed K.
    K_final_for_tau = k_sched(max_iters) if k_sched else (PUMA_K_END or PUMA_K)

    def eval_fn(model, it, tg):
        probe = probe_per_position(model, tokenizer, test_formatted, test_metas,
                                   max_len, device)
        parts = [f"{k.replace('acc_','')}={v:.3f}"
                 for k, v in probe.items() if k.startswith('acc_') and not k.startswith('acc_size')]
        print(f"    [eval it {it}] loss={probe['overall_loss']:.4f} "
              f"acc={probe['overall_acc']:.4f} {' '.join(parts[:4])}")

        if it > 0 and it % GEN_EVAL_EVERY == 0:
            r = gen_eval(model, tokenizer, test_formatted, test_metas,
                         max_len, 'confidence', n=GEN_EVAL_N, device=device)
            print(f"      [gen] full={r['accuracy']:.3f} cell={r['cell_accuracy']:.3f} "
                  f"tok={r['token_accuracy']:.3f}")
            if 'concordance_l2r' in r:
                print(f"      [conc] L2R={r['concordance_l2r']:.3f}")
            for n_h in [3, 4, 5, 6]:
                k = f'acc_size_{n_h}'
                if k in r:
                    print(f"      size={n_h}: {r[k]:.3f} (n={r.get(f'n_size_{n_h}', '?')})")
            probe['gen_accuracy'] = r['accuracy']
            probe['gen_cell_accuracy'] = r['cell_accuracy']
            probe['gen_concordance_l2r'] = r.get('concordance_l2r')

        # Reveal-τ (all mask_types, per-type K and eligibility) — stratified by puzzle size
        if (reveal_tracked_ids is not None and it > 0
                and it % REVEAL_TAU_EVERY == 0 and K_final_for_tau is not None):
            if mask_type == 'puma' and k_sched is not None:
                K_cur = k_sched(it)
                eligible = K_cur >= K_final_for_tau * REVEAL_TAU_K_THRESHOLD_FRAC
            else:
                K_cur = K_final_for_tau
                eligible = it >= max(max_iters // 4, 30000)
            if eligible:
                traj = simulate_reveal_trajectory(
                    model, tokenizer, reveal_tracked_ids, reveal_tracked_ans,
                    MAX_ANS_TOKENS,
                    blank_masks=reveal_blanks, K=K_cur, tau=PUMA_TAU,
                    batch_size=32, device=device)
                taus = compute_reveal_vs_order_tau(
                    traj['reveal_stage'], reveal_reasoning_order, reveal_blanks)
                valid_mask = ~np.isnan(taus)
                stratum_np = reveal_tracked_strata.cpu().numpy()
                per_stratum = {}
                for si, sn in enumerate(STRATUM_NAMES):
                    m = valid_mask & (stratum_np == si)
                    vs = taus[m]
                    if len(vs) >= 3:
                        per_stratum[sn] = {
                            'n': int(len(vs)),
                            'mean': float(vs.mean()),
                            'q25': float(np.percentile(vs, 25)),
                            'q50': float(np.percentile(vs, 50)),
                            'q75': float(np.percentile(vs, 75)),
                        }
                valid = taus[valid_mask]
                if len(valid) > 0:
                    probe['reveal_tau'] = {
                        'K_cur': K_cur, 'n': int(len(valid)),
                        'mean': float(valid.mean()),
                        'q25': float(np.percentile(valid, 25)),
                        'q50': float(np.percentile(valid, 50)),
                        'q75': float(np.percentile(valid, 75)),
                        'min': float(valid.min()), 'max': float(valid.max()),
                        'per_stratum': per_stratum,
                    }
                    strata_str = ' | '.join(
                        f"{sn}={d['q50']:+.2f}(n{d['n']})"
                        for sn, d in per_stratum.items())
                    print(f"      [reveal-τ] K={K_cur} overall={probe['reveal_tau']['q50']:+.3f} "
                          f"| {strata_str}")
        return probe

    model, dynamics = train_diffusion(
        train_ids=train_ids, train_ans=train_ans, ans_len=MAX_ANS_TOKENS,
        tokenizer=tokenizer,
        mask_type=mask_type, blank_masks=None,
        puma_tau=PUMA_TAU,
        puma_k_schedule=k_sched,
        papl_tau=PAPL_TAU, papl_alpha=PAPL_ALPHA,
        n_layer=N_LAYER, n_head=N_HEAD, n_embd=N_EMBD,
        dropout=DROPOUT, pos_enc=POS_ENC,
        max_iters=max_iters, batch_size=BATCH_SIZE,
        lr=LR, min_lr=MIN_LR, warmup_iters=WARMUP_ITERS,
        grad_clip=GRAD_CLIP, weight_decay=WEIGHT_DECAY, ema_decay=EMA_DECAY,
        eval_fn=eval_fn, eval_every=EVAL_EVERY, log_every=LOG_EVERY,
        patience=PATIENCE,
        sample_strata=sample_strata, stratum_names=stratum_names,
        init_state=init_state, device=device,
        use_amp=False if NO_AMP else None,
    )
    return model, dynamics


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Figures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def make_figures(all_dyn, all_final):
    """Generate zebra-specific figures.

    Fig 1: Accuracy × puzzle size (per decode policy) — primary extreme axis
    Fig 2: Stratum loss trajectory — training-side diagnostic
    Fig 3: Reveal heatmap on hard bucket, by solver-step bin × PUMA stage
    Fig 4: Reveal-vs-solver-order Kendall τ trajectory (PUMA only)
    Fig 5: Premature-reveal rate by solver-step bin
    """
    figs = {}
    COLORS = {'random': '#3498db', 'puma': '#8e44ad'}

    # Fig 1: Accuracy × puzzle size (primary extreme axis)
    for dp in DECODE_POLICIES:
        fig, ax = plt.subplots(figsize=(9, 5))
        sizes = [3, 4, 5, 6]
        for mt, col, mk in [('random', COLORS['random'], 'o'),
                             ('puma', COLORS['puma'], 's')]:
            r = all_final.get(f'{mt}_{dp}')
            if r is None: continue
            xs, ys, cs = [], [], []
            for sz in sizes:
                acc_key = f'acc_size_{sz}'
                cell_key = f'cell_acc_size_{sz}'
                if acc_key in r:
                    xs.append(sz); ys.append(r[acc_key])
                    cs.append(r.get(cell_key))
            if xs:
                ax.plot(xs, ys, f'-{mk}', color=col, label=f'{mt} (full puzzle)',
                        lw=2, markersize=10)
                # Cell-level accuracy as dashed line on same axes
                if all(c is not None for c in cs):
                    ax.plot(xs, cs, f'--{mk}', color=col, alpha=0.5,
                            label=f'{mt} (cell)', lw=1.5, markersize=8)
        ax.set_xlabel('Puzzle size (n_houses)')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'Accuracy × puzzle size — {dp} decode')
        ax.set_xticks(sizes); ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
        fig.tight_layout(); figs[f'size_sweep_{dp}'] = fig

    # Fig 2: Stratum loss trajectory — training-side
    mts_with_strat = [mt for mt in MASK_TYPES
                      if all_dyn.get(mt, {}).get('stratified_loss')]
    if mts_with_strat:
        nm = len(mts_with_strat)
        fig, axes = plt.subplots(1, nm, figsize=(7 * nm, 5), squeeze=False)
        axes = axes[0]
        s_cmap = plt.cm.viridis
        for ai, mt in enumerate(mts_with_strat):
            dyn = all_dyn[mt]
            sl = dyn['stratified_loss']
            names = dyn.get('stratum_names', [])
            S = len(names) if names else (len(sl[0]['per_stratum_loss']) if sl else 0)
            ax = axes[ai]
            xs = [e['iter'] for e in sl]
            for si in range(S):
                ys = [e['per_stratum_loss'][si] if e['per_stratum_n'][si] > 0 else None
                      for e in sl]
                label = names[si] if si < len(names) else f's{si}'
                valid = [(x, y) for x, y in zip(xs, ys) if y is not None]
                if valid:
                    ax.plot([p[0] for p in valid], [p[1] for p in valid],
                            '-', color=s_cmap(si / max(S - 1, 1)), label=label, lw=1.5)
            ax.set_xlabel('Iteration'); ax.set_ylabel('Masked-token loss (training)')
            ax.set_title(f'{mt}: stratum loss trajectory'); ax.set_yscale('log')
            ax.legend(fontsize=8); ax.grid(alpha=0.3)
        fig.suptitle('Training-time loss by puzzle-size stratum', y=1.02)
        fig.tight_layout(); figs['stratum_loss'] = fig

    # Fig 3: Reveal heatmap on hard bucket, by solver-step bin × PUMA stage
    # For each mask type, aggregate still-masked fraction per (bin, stage) cell.
    mts_with_rev = [mt for mt in MASK_TYPES
                    if f'{mt}_reveal_hard' in all_final
                    and all_final[f'{mt}_reveal_hard'].get('by_solver_step_bin')]
    if mts_with_rev:
        nm = len(mts_with_rev)
        fig, axes = plt.subplots(1, nm, figsize=(6 * nm, 4), squeeze=False)
        axes = axes[0]
        bins_order = ['early', 'mid', 'late', 'pad']
        for ai, mt in enumerate(mts_with_rev):
            rev = all_final[f'{mt}_reveal_hard']
            bsb = rev['by_solver_step_bin']
            K = rev.get('K', REVEAL_K_DEFAULT)
            rows = [b for b in bins_order if b in bsb]
            if not rows: continue
            mat = np.array([bsb[b]['still_masked_per_stage'] for b in rows])
            ax = axes[ai]
            im = ax.imshow(mat, aspect='auto', origin='lower', cmap='Reds',
                           vmin=0, vmax=1, interpolation='nearest')
            ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows)
            ax.set_xlabel(f'PUMA stage (K={K})')
            ax.set_ylabel('Solver-step bin')
            ax.set_title(f'{mt}: still-masked (hard bucket, N={rev["n"]})')
            plt.colorbar(im, ax=ax, label='still masked')
        fig.suptitle('Reveal trajectory — hard bucket (largest puzzle size)', y=1.02)
        fig.tight_layout(); figs['reveal_hard'] = fig

    # Fig 4: Reveal-vs-solver-order Kendall τ trajectory (PUMA only)
    # Fig 4: Reveal-vs-solver-order Kendall τ trajectory (PUMA only, stratified).
    # Central training-time diagnostic for zebra. Paper claim:
    #   size_3 (easy)  → τ trending toward +1 (solver order learned)
    #   size_6 (hard)  → τ staying near 0 (solver order unlearnable, the 0% case)
    # When both are on one figure, the gap directly visualizes "PUMA fails
    # at the extreme case" — the reviewer sees the failure class.
    puma_dyn = all_dyn.get('puma', {})
    tau_pts = [c for c in puma_dyn.get('checkpoints', []) if 'reveal_tau' in c]
    if tau_pts:
        fig, ax = plt.subplots(figsize=(10, 5))
        # Overall IQR shaded
        xs = [c['iter'] for c in tau_pts]
        mids = [c['reveal_tau']['q50'] for c in tau_pts]
        q25s = [c['reveal_tau']['q25'] for c in tau_pts]
        q75s = [c['reveal_tau']['q75'] for c in tau_pts]
        ax.fill_between(xs, q25s, q75s, alpha=0.2, color=COLORS['puma'], label='overall IQR')
        ax.plot(xs, mids, '--', color='#5e2d6e', lw=1.5, label='overall median')
        # Per-stratum (puzzle size) trajectories
        s_cmap = plt.cm.plasma
        for si, sn in enumerate(STRATUM_NAMES):
            xs_s, mids_s = [], []
            for c in tau_pts:
                ps = c['reveal_tau'].get('per_stratum', {}).get(sn)
                if ps is not None:
                    xs_s.append(c['iter']); mids_s.append(ps['q50'])
            if xs_s:
                ax.plot(xs_s, mids_s, '-o',
                        color=s_cmap(si / max(len(STRATUM_NAMES) - 1, 1)),
                        label=sn, lw=2, markersize=5)
        ax.axhline(0, color='gray', ls=':', alpha=0.7, label='τ=0')
        ax.axhline(1, color='green', ls='--', alpha=0.4, label='τ=+1 (solver-aligned)')
        ax.axhline(-1, color='red', ls='--', alpha=0.4, label='τ=−1 (reversed)')
        ax.set_xlabel('Training iteration')
        ax.set_ylabel('Kendall τ (reveal vs solver order)')
        ax.set_title('PUMA reveal-order alignment with solver order, stratified by puzzle size')
        ax.set_ylim(-1.05, 1.05); ax.legend(loc='best', fontsize=8); ax.grid(alpha=0.3)
        fig.tight_layout(); figs['reveal_tau_stratified'] = fig

    # Fig 5: Premature-reveal rate by solver-step bin
    if mts_with_rev:
        fig, ax = plt.subplots(figsize=(8, 5))
        bins_order = ['early', 'mid', 'late']
        x = np.arange(len(bins_order))
        width = 0.35
        for mi, mt in enumerate(mts_with_rev):
            rev = all_final[f'{mt}_reveal_hard']
            pr = rev.get('premature_rate', {})
            rates = [pr.get(b, {}).get('rate', 0) for b in bins_order]
            offset = -width / 2 + mi * width if len(mts_with_rev) == 2 else 0
            ax.bar(x + offset, rates, width, label=mt,
                   color=COLORS.get(mt, 'gray'), alpha=0.8)
        ax.set_xticks(x); ax.set_xticklabels(bins_order)
        ax.set_xlabel('Solver-step bin')
        ax.set_ylabel('Premature-reveal rate')
        ax.set_title('Fraction of cells revealed before expected solver stage (hard bucket)')
        ax.legend(); ax.grid(alpha=0.3, axis='y'); ax.set_ylim(0, 1.05)
        fig.tight_layout(); figs['premature_reveal_rate'] = fig

    return figs


def run(tag=''):
    torch.manual_seed(SEED)
    random.seed(SEED)

    exp_name = f"{EXP_NAME}_{tag}" if tag else EXP_NAME
    print(f"\n{'='*70}")
    print(f"  {exp_name}")
    print(f"  Model: {N_LAYER}L/{N_EMBD}D/{N_HEAD}H, ANS_TOKENS={MAX_ANS_TOKENS}")
    print(f"  Masks: {MASK_TYPES}, Decode: {DECODE_POLICIES}")
    print(f"{'='*70}")

    # Load data
    train_path = os.path.join(DATA_DIR, TRAIN_FILE)
    test_path = os.path.join(DATA_DIR, TEST_FILE)

    print(f"\nLoading train data from {train_path}...")
    train_raw = load_zebra_pickle(train_path, max_n=N_TRAIN, seed=SEED)
    print(f"  {len(train_raw)} raw samples loaded")

    print(f"Loading test data from {test_path}...")
    test_raw = load_zebra_pickle(test_path, max_n=N_TEST, seed=SEED + 1)
    print(f"  {len(test_raw)} raw test samples loaded")

    # Build tokenizer from combined vocab
    tokenizer = build_tokenizer(train_raw + test_raw)
    print(f"  Vocab size: {tokenizer.vocab_size}")

    # Format samples
    print("Formatting samples...")
    train_formatted, train_metas = format_samples(train_raw)
    test_formatted, test_metas = format_samples(test_raw)
    print(f"  Train: {len(train_formatted)}, Test: {len(test_formatted)}")

    # Data statistics
    size_dist = defaultdict(int)
    for m in train_metas:
        size_dist[m['n_houses']] += 1
    print(f"  Puzzle size distribution (train): {dict(sorted(size_dist.items()))}")

    sol_lens = [m['n_sol_tokens'] for m in train_metas]
    print(f"  Solution tokens: min={min(sol_lens)}, max={max(sol_lens)}, "
          f"mean={sum(sol_lens)/len(sol_lens):.0f}")

    max_len = MAX_SEQ_LEN

    # Unified test suite
    print("Building test suite...")
    suite = build_test_suite(test_formatted, test_metas, tokenizer, max_len)

    # Train
    all_dyn, all_final = {}, {}
    models = {}

    for mask_type in MASK_TYPES:
        print(f"\n{'─'*60}")
        print(f"  Training: {mask_type}")
        print(f"{'─'*60}")

        model, dynamics = train_model(
            mask_type, tokenizer, train_formatted, train_metas,
            test_formatted, test_metas,
            max_len, device=DEVICE)
        models[mask_type] = model
        all_dyn[mask_type] = dynamics

        # Final evaluation — use full test set for headline metrics
        for dp in DECODE_POLICIES:
            print(f"\n  Evaluating: {mask_type} × {dp}")
            r = gen_eval(model, tokenizer, test_formatted, test_metas,
                         max_len, dp, device=DEVICE)
            key = f"{mask_type}_{dp}"
            all_final[key] = r
            print(f"    full={r['accuracy']:.3f} cell={r['cell_accuracy']:.3f} "
                  f"loc={r['loc_accuracy']:.3f} val={r['val_accuracy']:.3f}")
            if 'concordance_l2r' in r:
                print(f"    concordance_l2r={r['concordance_l2r']:.3f}")
            step_parts = [f"{b}={r[f'cell_acc_step_{b}']:.3f}"
                          for b in ['early', 'mid', 'late']
                          if f'cell_acc_step_{b}' in r]
            if step_parts:
                print(f"    solver step: {' '.join(step_parts)}")

        # Selective masking experiment
        for frac in [0.25, 0.50, 0.75]:
            sm = eval_selective_mask(model, tokenizer, test_formatted, test_metas,
                                     max_len, reveal_fraction=frac, device=DEVICE)
            key = f"{mask_type}_selective_{int(frac*100)}"
            all_final[key] = sm
            print(f"    selective reveal={frac:.0%}: "
                  f"masked_acc={sm['masked_accuracy']:.3f}")

        # Reveal trajectory analysis on hard bucket — PUMA failure mechanism
        hard_bucket = suite['constructed'].get('hard')
        if hard_bucket and hard_bucket['n'] > 0:
            K_for_reveal = PUMA_K_END if PUMA_K_END else PUMA_K
            # Match actual final K if schedule was step
            if PUMA_K_START is not None and PUMA_K_END is not None:
                K_for_reveal = PUMA_K_END
            rev = analyse_reveal_patterns(
                model, tokenizer, hard_bucket, max_len,
                K=K_for_reveal, tau=PUMA_TAU, device=DEVICE)
            all_final[f'{mask_type}_reveal_hard'] = rev
            if 'premature_rate' in rev:
                parts = [f"{b}={rev['premature_rate'][b]['rate']:.3f}"
                         for b in ['early', 'mid', 'late']
                         if b in rev['premature_rate']]
                print(f"    premature_rate (hard): {' '.join(parts)}")

    # Continuation: PUMA → random
    args = parse_args()
    if not getattr(args, 'no_continuation', False) and 'puma' in models:
        print(f"\n{'─'*60}")
        print(f"  Continuation: PUMA → random ({CONTINUATION_ITERS} iters)")
        print(f"{'─'*60}")
        puma_state = models['puma'].state_dict()
        cont_model, cont_dyn = train_model(
            'random', tokenizer, train_formatted, train_metas,
            test_formatted, test_metas,
            max_len, max_iters=CONTINUATION_ITERS,
            init_state=puma_state, device=DEVICE)
        all_dyn['cont_puma2random'] = cont_dyn
        for dp in DECODE_POLICIES[:2]:
            r = gen_eval(cont_model, tokenizer, test_formatted, test_metas,
                         max_len, dp, device=DEVICE)
            key = f"cont_puma2random_{dp}"
            all_final[key] = r
            print(f"    {key}: full={r['accuracy']:.3f} cell={r['cell_accuracy']:.3f}")

    # Summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    for dp in DECODE_POLICIES:
        print(f"\n  ── {dp} ──")
        print(f"  {'Metric':<25s}", end='')
        for mt in MASK_TYPES:
            print(f" {mt:>12s}", end='')
        print()
        for metric in ['accuracy', 'cell_accuracy', 'loc_accuracy', 'val_accuracy',
                        'concordance_l2r',
                        'cell_acc_step_early', 'cell_acc_step_mid', 'cell_acc_step_late',
                        'acc_size_3', 'acc_size_4', 'acc_size_5', 'acc_size_6']:
            vals = [all_final.get(f'{mt}_{dp}', {}).get(metric) for mt in MASK_TYPES]
            if any(v is not None for v in vals):
                print(f"  {metric:<25s}", end='')
                for v in vals:
                    print(f" {v:>12.4f}" if v is not None else f" {'N/A':>12s}", end='')
                print()

    # Figures
    figs = make_figures(all_dyn, all_final)

    # Save
    sd = {'config': {k: globals()[k] for k in [
        'MAX_ANS_TOKENS', 'MAX_SEQ_LEN', 'N_LAYER', 'N_HEAD', 'N_EMBD',
        'MASK_TYPES', 'DECODE_POLICIES', 'MAX_ITERS', 'BATCH_SIZE',
        'PUMA_K', 'N_RAINBOW', 'SEED']}}
    for k, v in all_dyn.items():
        sd[f'dyn_{k}'] = {
            'checkpoints': v.get('checkpoints', []),
            'train_loss': v.get('train_loss', []),
            'stratified_loss': v.get('stratified_loss', []),
            'stratum_names': v.get('stratum_names', []),
        }
    for k, v in all_final.items():
        if isinstance(v, dict):
            v = {kk: vv for kk, vv in v.items()
                 if not isinstance(vv, (np.ndarray, torch.Tensor))}
        sd[f'final_{k}'] = v
    save_results(exp_name, sd, figures=figs)
    return all_dyn, all_final


if __name__ == '__main__':
    args = parse_args()
    seeds = args.seeds if args.seeds else [SEED]
    for si, seed in enumerate(seeds):
        globals()['SEED'] = seed
        t = f"{args.tag}_s{seed}" if args.tag and len(seeds) > 1 else args.tag
        if len(seeds) > 1:
            print(f"\n{'#'*70}\n# Seed {seed} ({si+1}/{len(seeds)})\n{'#'*70}")
        run(tag=t)
