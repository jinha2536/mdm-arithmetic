"""
Shared training, generation, evaluation, Drive persistence.

Key design:
  1. Iteration-based training with EMA (unified across all experiments)
  2. PUMA streaming with configurable K schedule
  3. Continuation training support (load checkpoint → train with different mask type)
  4. Greedy decoding for fair comparison
"""
import os, time, math, json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')

from core.model import Transformer
from core.tokenizer import CharTokenizer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Google Drive persistence
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DRIVE_BASE = os.environ.get(
    'DIFFUSION_ARITHMETIC_DRIVE',
    '/content/drive/MyDrive/diffusion-arithmetic-results',
)


def mount_drive():
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
        os.makedirs(DRIVE_BASE, exist_ok=True)
        print(f"✓ Drive mounted → {DRIVE_BASE}")
        return True
    except Exception:
        os.makedirs('./results', exist_ok=True)
        print("  Drive unavailable → ./results/")
        return False


def get_save_dir(exp_name):
    base = DRIVE_BASE if os.path.exists('/content/drive/MyDrive') else './results'
    d = os.path.join(base, exp_name)
    os.makedirs(d, exist_ok=True)
    return d


def save_results(exp_name, results, figures=None, model=None, tag=''):
    d = get_save_dir(exp_name)
    sfx = f'_{tag}' if tag else ''
    with open(os.path.join(d, f'results{sfx}.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    if figures:
        for name, fig in figures.items():
            fig.savefig(os.path.join(d, f'{name}{sfx}.png'),
                        dpi=150, bbox_inches='tight')
    if model is not None:
        torch.save(model.state_dict(), os.path.join(d, f'model{sfx}.pt'))
    print(f"  💾 Saved to {d}")


def save_checkpoint(exp_name, state_dict, tag=''):
    """Save model checkpoint for continuation training."""
    d = get_save_dir(exp_name)
    sfx = f'_{tag}' if tag else ''
    path = os.path.join(d, f'checkpoint{sfx}.pt')
    torch.save(state_dict, path)
    print(f"  💾 Checkpoint → {path}")
    return path


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Data encoding
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def encode_samples(samples, tokenizer, max_len=None):
    encoded = [tokenizer.encode(s) for s in samples]
    if max_len is None:
        max_len = max(len(e) for e in encoded)
    pad_id = tokenizer.special_ids['pad']
    ids = torch.full((len(encoded), max_len), pad_id, dtype=torch.long)
    ans_starts = torch.zeros(len(encoded), dtype=torch.long)
    for i, enc in enumerate(encoded):
        L = min(len(enc), max_len)
        ids[i, :L] = torch.tensor(enc[:L])
        ans_starts[i] = samples[i].index('=') + 1
    return ids, ans_starts


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fresh-data source (chunked lazy generation for addition)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class _FreshAdditionSource:
    """Chunked-lazy data source for fresh-data training mode on addition.

    Pre-allocates a single pinned-memory chunk of `chunk_size` tokenized
    samples and refills it in-place when exhausted. Generation is numpy-
    vectorized (no Python per-sample loop). Inline regeneration is rare
    (every chunk_size/batch_size iters) and fast (~200ms for chunk_size=500k);
    no background threading is used, which keeps the lifecycle simple.

    Safety notes on CPU-GPU transfer:
      * Pinned source + non_blocking=True returns a GPU-side tensor whose
        copy is in flight on the default stream. PyTorch synchronizes any
        subsequent op that depends on it (forward pass, gather into PUMA
        buffer), so by the time `next()` is called again, prior transfers
        from this chunk have already been consumed and synchronized.
      * Therefore overwriting the pinned chunk in-place when ptr reaches
        chunk_size is safe — no prior batch's data is still pending on GPU.

    Determinism:
      `seed` controls the entire stream. chunk_idx is mixed in to derive
      per-chunk numpy RNG states, so the i-th sample produced is a pure
      function of (seed, total_samples_consumed_so_far).

    Distribution:
      Natural ND-digit addition: a, b ∈ [10**(ND-1), 10**ND - 1], uniform.
      Matches gen_data_natural() in exp_addition.py byte-for-byte (apart
      from the unused pad columns after the answer).
    """

    def __init__(self, nd, max_len, total_samples, tokenizer, seed,
                 chunk_size=500_000, ans_len=None):
        self.nd = nd
        self.ans_len = ans_len if ans_len is not None else nd + 1
        self.max_len = max_len
        self.total_samples = int(total_samples)
        self.tokenizer = tokenizer
        self.chunk_size = min(int(chunk_size), self.total_samples)
        self.seed = int(seed)

        # Token-id lookup. Char tokenizer maps digit '0'..'9' to consecutive
        # ids matching `list('0123456789+=')` insertion order, but we don't
        # rely on that — go through stoi.
        self._digit_ids = np.array(
            [tokenizer.stoi[c] for c in '0123456789'], dtype=np.int64)
        self._plus_id = tokenizer.stoi['+']
        self._eq_id   = tokenizer.stoi['=']
        self._pad_id  = tokenizer.special_ids['pad']
        # Layout: a (nd) | '+' (1) | b (nd) | '=' (1) | c (ans_len)
        self._ans_start = 2 * nd + 2

        # Pre-allocate pinned chunk (positions after the equation stay as pad)
        # pin_memory() requires CUDA. On CPU-only systems (testing), skip
        # pinning — transfer to CPU is then just a tensor share.
        _pin = torch.cuda.is_available()
        self._cur_ids = torch.full(
            (self.chunk_size, max_len), self._pad_id, dtype=torch.long)
        self._cur_ans = torch.full(
            (self.chunk_size,), self._ans_start, dtype=torch.long)
        if _pin:
            self._cur_ids = self._cur_ids.pin_memory()
            self._cur_ans = self._cur_ans.pin_memory()
        self._ptr = 0
        self._chunk_idx = 0
        self.consumed = 0

        # Reusable numpy scratch for the chunk (avoids realloc each refill)
        self._np_ids_scratch = np.full(
            (self.chunk_size, max_len), self._pad_id, dtype=np.int64)

        # Initial fill
        self._gen_chunk()

    def _gen_chunk(self):
        """Generate one chunk of fresh samples into self._cur_ids in-place.

        Uses a fresh numpy default_rng seeded from (self.seed, chunk_idx)
        so each chunk is deterministic regardless of consumption order.
        """
        cs, nd = self.chunk_size, self.nd
        rng = np.random.default_rng(self.seed + self._chunk_idx * 10007 + 1)

        # ND-digit operands: leading digit ∈ [1,9], rest ∈ [0,9].
        # Matches the natural-distribution lo=10**(ND-1) constraint in
        # exp_addition.py:gen_data_natural.
        a_lead = rng.integers(1, 10, size=cs, dtype=np.int64)
        a_rest = rng.integers(0, 10, size=(cs, nd - 1), dtype=np.int64)
        a_digits = np.concatenate([a_lead[:, None], a_rest], axis=1)
        b_lead = rng.integers(1, 10, size=cs, dtype=np.int64)
        b_rest = rng.integers(0, 10, size=(cs, nd - 1), dtype=np.int64)
        b_digits = np.concatenate([b_lead[:, None], b_rest], axis=1)

        # Compute c = a + b digit-wise with carry (right to left in the
        # digit array, which is MSB-first). ans_len = nd + 1 to carry the
        # potential overflow into a leading digit.
        c_digits = np.zeros((cs, nd + 1), dtype=np.int64)
        carry = np.zeros(cs, dtype=np.int64)
        for i in range(nd - 1, -1, -1):
            s = a_digits[:, i] + b_digits[:, i] + carry
            c_digits[:, i + 1] = s % 10
            carry = s // 10
        c_digits[:, 0] = carry

        # Build flat token-id array. Layout repeats above:
        # cols [0..nd):     a digits → digit_ids
        # col  nd:          '+'
        # cols [nd+1..2nd+1):  b digits
        # col  2nd+1:       '='
        # cols [2nd+2..2nd+2+ans_len):  c digits
        # cols beyond:      pad (left from initial pad fill)
        ids_np = self._np_ids_scratch
        ids_np[:, 0:nd]                        = self._digit_ids[a_digits]
        ids_np[:, nd]                          = self._plus_id
        ids_np[:, nd + 1:2 * nd + 1]           = self._digit_ids[b_digits]
        ids_np[:, 2 * nd + 1]                  = self._eq_id
        ids_np[:, 2 * nd + 2:2 * nd + 2 + nd + 1] = self._digit_ids[c_digits]

        # Copy numpy chunk into pinned tensor. Use the underlying torch
        # tensor view via from_numpy → copy_; this is one memcpy.
        self._cur_ids.copy_(torch.from_numpy(ids_np), non_blocking=False)
        # ans_starts is constant for this experiment layout; pre-filled in __init__.

        self._ptr = 0
        self._chunk_idx += 1

    def next(self, batch_size, device):
        """Return (ids, ans_starts) of shape (batch_size, max_len) and
        (batch_size,) respectively, both on `device`. Refills the chunk
        in-place if exhausted.
        """
        if self._ptr + batch_size > self.chunk_size:
            self._gen_chunk()
        slc = slice(self._ptr, self._ptr + batch_size)
        # non_blocking transfer from pinned memory; PyTorch will sync any
        # downstream op that uses the returned tensor.
        ids = self._cur_ids[slc].to(device, non_blocking=True)
        ans = self._cur_ans[slc].to(device, non_blocking=True)
        self._ptr += batch_size
        self.consumed += batch_size
        return ids, ans

    def __repr__(self):
        return (f"_FreshAdditionSource(nd={self.nd}, total={self.total_samples}, "
                f"chunk={self.chunk_size}, consumed={self.consumed}, "
                f"chunk_idx={self._chunk_idx})")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PUMA K schedule helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def puma_k_linear(k_start, k_end, max_iters):
    """Linear ramp from k_start to k_end over training."""
    def schedule(it):
        frac = min(it / max(max_iters, 1), 1.0)
        return int(k_start + (k_end - k_start) * frac)
    return schedule


def puma_k_step(k_start, k_end, k_step, k_every):
    """Step function: +k_step every k_every iters, capped at k_end."""
    def schedule(it):
        n_steps = it // k_every
        return min(k_start + n_steps * k_step, k_end)
    return schedule


def puma_k_fixed(k):
    """Fixed K throughout training."""
    return lambda it: k


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Unified diffusion training
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def train_diffusion(
    # Data (already on device)
    train_ids,          # [N, T]
    train_ans,          # [N] answer start positions
    ans_len,            # int
    tokenizer,
    # Masking
    mask_type='random',     # 'random', 'papl', or 'puma'
    blank_masks=None,       # [N, ans_len] bool — None = all positions maskable
    # PUMA
    puma_tau=0.9,
    puma_k_schedule=None,   # callable(it) → K; required if mask_type='puma'
    # PAPL (Planner-Aware Path Learning; Peng et al. 2025, arXiv:2509.23405)
    # PAPL uses uniform random masking (like 'random') and reweights per-token
    # loss by softmax-normalized planner weights (paper Eq. 7):
    #     w^i ∝ exp((1/τ) log P(x_0^i | x_k)), softmax over masked positions
    #     weight_i = (1/(L-k)) · (1 + α·w^i)
    #     loss = Σ_{i: x_k^i=mask} weight_i · NLL_i
    # The planner weights w^i are detached — no gradient through them.
    # Direction: positions where the model is already confident on the correct
    # token get additional weight, concentrating capacity on easy patterns.
    # α=0 recovers vanilla random masking; paper defaults τ=1, α=1.
    papl_tau=1.0,
    papl_alpha=1.0,
    # Architecture
    n_layer=4, n_head=4, n_embd=128, dropout=0.1, pos_enc='absolute',
    # Training
    max_iters=200000,
    batch_size=128,
    lr=3e-4, min_lr=1e-5,
    warmup_iters=2000,
    grad_clip=1.0,
    weight_decay=0.01,
    ema_decay=0.9999,
    # Eval callbacks
    eval_fn=None,           # fn(model, it, tg) → dict with 'overall_loss'
    eval_every=5000,
    log_every=1000,
    # Early stopping
    patience=None,          # stop if no improvement for this many iters; None=disabled
    # Continuation
    init_state=None,        # state_dict to initialize model from
    # Stratum-level training loss logging (training-side diagnostic)
    sample_strata=None,     # [N] long tensor — per-sample stratum id (None = no logging)
    stratum_names=None,     # list[str] for readable labels; len determines n_strata
    # Device
    device=None,
    # AMP
    use_amp=None,           # None=auto, True=force, False=disable
    # ── NEW: reproducibility & multi-checkpoint ──
    # seed_train: applied via torch.manual_seed AFTER model init/load. This
    # decouples training randomness (mask sampling, batch order) from model
    # init randomness. Pass distinct seeds per (run-seed, method) so methods
    # within a seed share init but have isolated training RNGs.
    seed_train=None,
    # ckpt_schedule: list of iter values at which to save the EMA state via
    # ckpt_save_fn(state_dict_cpu, it). Set to None to disable per-iter saves.
    ckpt_schedule=None,
    ckpt_save_fn=None,
    # data_source: optional override of where (ids, ans_starts) batches come
    # from. Default (None) uses the pre-loaded train_ids/train_ans on GPU.
    # When provided, must implement `next(batch_size, device) → (ids, ans)`.
    # Used by fresh-data mode where samples are generated on demand and the
    # full dataset is too large to hold in GPU memory.
    data_source=None,
):
    """
    Unified iteration-based diffusion training with EMA.

    Returns: (model, dynamics)
        model: best EMA weights loaded, eval mode
        dynamics: {'checkpoints': [...], 'train_loss': [...]}
    """
    if device is None:
        device = DEVICE

    N, T = train_ids.shape
    mask_id = tokenizer.special_ids['mask']
    _arange = torch.arange(ans_len, device=device)

    if blank_masks is None:
        blank_masks = torch.ones(N, ans_len, dtype=torch.bool, device=device)
    else:
        blank_masks = blank_masks.to(device)

    # ── Build model ──
    model = Transformer(
        vocab_size=len(tokenizer), block_size=T + 8,
        n_layer=n_layer, n_head=n_head, n_embd=n_embd,
        dropout=dropout, is_causal=False, pos_enc=pos_enc,
    ).to(device)

    if init_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in init_state.items()})
        print(f"  [{mask_type}] Loaded init checkpoint")

    # Apply training seed AFTER model init/load: weights are deterministic
    # (either from init_state or seeded model construction upstream), and we
    # want subsequent RNG (mask sampling, batch perm, PUMA buffer) to be
    # deterministic per (run-seed, method). Model init RNG state from the
    # Transformer() call is consumed but doesn't affect us here.
    if seed_train is not None:
        torch.manual_seed(seed_train)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(seed_train)
        print(f"  [{mask_type}] Training RNG seeded with {seed_train}")

    ema_state = {k: v.clone() for k, v in model.state_dict().items()}
    print(f"  [{mask_type}] params={model.n_params:,}, N={N}, T={T}, "
          f"ans_len={ans_len}, max_iters={max_iters}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=weight_decay)

    def get_lr(it):
        if it < warmup_iters:
            return lr * it / max(warmup_iters, 1)
        ratio = (it - warmup_iters) / max(max_iters - warmup_iters, 1)
        return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * min(ratio, 1.0)))

    dynamics = {'checkpoints': [], 'train_loss': []}
    best_loss, best_ema = float('inf'), None
    best_iter = 0
    stopped_early = False
    t0 = time.time()
    tg = 0

    # ── Early stopping: compute min iter before activation ──
    uses_streaming = (mask_type == 'puma')
    if patience is not None and uses_streaming and puma_k_schedule is not None:
        # Don't early-stop before K schedule reaches ~90% of final K
        _final_k = puma_k_schedule(max_iters)
        _target_k = int(_final_k * 0.9)
        _min_es_iter = max_iters * 2 // 3  # fallback
        for _probe_it in range(0, max_iters, 1000):
            if puma_k_schedule(_probe_it) >= _target_k:
                _min_es_iter = _probe_it; break
        es_min_iter = max(_min_es_iter, warmup_iters) + patience
        print(f"  Early stopping: patience={patience}, active after it {es_min_iter} "
              f"(K≥{_target_k}/{_final_k})")
    elif patience is not None:
        es_min_iter = warmup_iters + patience
        print(f"  Early stopping: patience={patience}, active after it {es_min_iter}")
    else:
        es_min_iter = max_iters + 1  # effectively disabled

    # ── PUMA streaming buffer ──
    if uses_streaming:
        assert puma_k_schedule is not None, "puma_k_schedule required for puma"
        buf_z = torch.zeros(batch_size, T, dtype=torch.long, device=device)
        buf_x0 = torch.zeros(batch_size, T, dtype=torch.long, device=device)
        buf_ans = torch.zeros(batch_size, dtype=torch.long, device=device)
        buf_stage = torch.zeros(batch_size, dtype=torch.long, device=device)
        buf_orig_idx = torch.zeros(batch_size, dtype=torch.long, device=device)
        # buf_pool indexing only used in fixed-pool mode (data_source is None)
        buf_pool = torch.randperm(N) if data_source is None else None
        buf_ptr = 0

        def _refresh(indices):
            nonlocal buf_ptr, buf_pool
            idx_t = torch.tensor(indices, device=device)
            n = len(indices)
            if data_source is None:
                # Fixed-pool mode: index train_ids by buf_pool
                if buf_ptr + n > len(buf_pool):
                    buf_pool = torch.randperm(N); buf_ptr = 0
                si = buf_pool[buf_ptr:buf_ptr + n].to(device); buf_ptr += n
                fresh_ids = train_ids[si]
                fresh_ans = train_ans[si]
                fresh_blanks = blank_masks[si]
                buf_orig_idx[idx_t] = si  # used by stratum logging
            else:
                # Fresh-data mode: pull a fresh batch from the data source.
                # No persistent dataset → no stratum logging (orig_idx unused
                # in that branch by design).
                fresh_ids, fresh_ans = data_source.next(n, device)
                # blank_masks is shape (N, ans_len); in fresh-data we assume
                # full-blank answer (no per-sample blank pattern) — matches
                # how addition's blank_masks is constructed (all True).
                fresh_blanks = torch.ones(n, ans_len, dtype=torch.bool, device=device)
            buf_x0[idx_t] = fresh_ids
            buf_z[idx_t] = fresh_ids.clone()
            buf_ans[idx_t] = fresh_ans
            buf_stage[idx_t] = 0
            ap = (buf_ans[idx_t].unsqueeze(1) + _arange).clamp(max=T - 1)
            bii = idx_t.unsqueeze(1).expand_as(ap)
            buf_z[bii[fresh_blanks], ap[fresh_blanks]] = mask_id

        def _advance(logits, K_cur):
            nonlocal buf_stage
            B_buf = batch_size
            ap = (buf_ans.unsqueeze(1) + _arange).clamp(max=T - 1)
            bi = torch.arange(B_buf, device=device).unsqueeze(1).expand_as(ap)
            is_m = (buf_z[bi, ap] == mask_id)
            if not is_m.any():
                _refresh(list(range(B_buf))); return
            lp = logits[bi, ap].clone()
            lp[:, :, mask_id] = -float('inf')
            confs = F.softmax(lp, dim=-1).max(dim=-1).values
            confs[~is_m] = -float('inf')
            nm = is_m.sum(dim=1).float()
            K_rem = (K_cur - buf_stage).clamp(min=1).float()
            nr = (nm / K_rem).ceil().long().clamp(min=1)
            ranked = confs.argsort(dim=1, descending=True)
            rop = torch.zeros_like(ranked)
            rop.scatter_(1, ranked, _arange.expand(B_buf, -1))
            reveal = ((rop < nr.unsqueeze(1)) | (confs > puma_tau)) & is_m
            buf_z[bi[reveal], ap[reveal]] = buf_x0[bi[reveal], ap[reveal]]
            buf_stage += 1
            done = (~(buf_z[bi, ap] == mask_id).any(dim=1)) | (buf_stage >= K_cur)
            if done.any():
                _refresh(done.nonzero(as_tuple=True)[0].tolist())

        _refresh(list(range(batch_size)))

    # ── Random masking batch iterator ──
    # `_next_batch` returns (ids, ans_starts, idx_or_None). Both fixed-pool
    # and fresh-data modes flow through here; only the source differs.
    # idx is None in fresh mode (no persistent per-sample identity) — used
    # to disable stratum logging in that branch.
    perm = torch.randperm(N, device=device) if data_source is None else None
    perm_ptr = 0

    def _next_batch():
        nonlocal perm, perm_ptr
        if data_source is None:
            if perm_ptr + batch_size > N:
                perm = torch.randperm(N, device=device); perm_ptr = 0
            idx = perm[perm_ptr:perm_ptr + batch_size]; perm_ptr += batch_size
            return train_ids[idx], train_ans[idx], idx
        else:
            ids, ans = data_source.next(batch_size, device)
            return ids, ans, None

    # ── Eval helper ──
    def _do_eval(it_num):
        nonlocal best_loss, best_ema, best_iter
        if eval_fn is None:
            return False  # no early stop signal
        # Swap model params ↔ EMA params in-place (no full state_dict copy)
        with torch.no_grad():
            for name, param in model.named_parameters():
                tmp = param.data.clone()
                param.data.copy_(ema_state[name])
                ema_state[name].copy_(tmp)
        model.eval()
        result = eval_fn(model, it_num, tg)
        dynamics['checkpoints'].append({'iter': it_num, 'tg': tg, **(result or {})})
        if result and 'overall_loss' in result:
            if result['overall_loss'] < best_loss:
                best_loss = result['overall_loss']
                best_iter = it_num
                # During swap, model holds EMA weights — save them
                best_ema = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        # Swap back: EMA params → ema_state, original → model
        with torch.no_grad():
            for name, param in model.named_parameters():
                tmp = param.data.clone()
                param.data.copy_(ema_state[name])
                ema_state[name].copy_(tmp)
        model.train()
        # Early stopping check
        if patience is not None and it_num >= es_min_iter:
            if it_num - best_iter >= patience:
                return True  # signal to stop
        return False

    # ── Stratum loss logging setup ──
    # Stratum logging assumes a persistent per-sample stratum id indexed by
    # the dataset position; this is incompatible with fresh-data mode where
    # samples have no persistent identity. Force-disable in that case.
    log_strata = (sample_strata is not None) and (data_source is None)
    if log_strata:
        sample_strata = sample_strata.to(device).long()
        assert sample_strata.shape == (N,), f"sample_strata must be [N={N}], got {sample_strata.shape}"
        S = len(stratum_names) if stratum_names else int(sample_strata.max().item()) + 1
        strat_loss_sum = torch.zeros(S, device=device)
        strat_token_count = torch.zeros(S, dtype=torch.long, device=device)
        dynamics['stratified_loss'] = []   # [{'iter', 'per_stratum_loss'[S], 'per_stratum_n'[S]}]
        dynamics['stratum_names'] = stratum_names or [f's{i}' for i in range(S)]
        print(f"  [{mask_type}] Stratum loss logging: {S} strata "
              f"({dynamics['stratum_names']})")

    model.eval()
    _do_eval(0)
    model.train()

    # ── AMP setup (bfloat16 on A100 — no scaler needed) ──
    if use_amp is None:
        use_amp = (device.type == 'cuda' and torch.cuda.is_bf16_supported())
    amp_dtype = torch.bfloat16 if use_amp else torch.float32
    ctx = torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp)
    if use_amp:
        print(f"  [{mask_type}] AMP bfloat16 enabled")

    # ── Multi-checkpoint schedule ──
    # Save EMA snapshot at each milestone iter. Saved state is in CPU memory
    # to avoid GPU OOM during long runs; callback decides where to persist it.
    ckpt_schedule_set = set(ckpt_schedule or [])
    if ckpt_schedule_set:
        print(f"  [{mask_type}] Ckpt schedule: {sorted(ckpt_schedule_set)}")

    # ── Training loop ──
    for it in range(1, max_iters + 1):
        for pg in optimizer.param_groups:
            pg['lr'] = get_lr(it)

        K_cur = puma_k_schedule(it) if uses_streaming else 0

        if uses_streaming:
            m = (buf_z == mask_id)
            if m.sum() == 0:
                _refresh(list(range(batch_size)))
                m = (buf_z == mask_id)
            with ctx:
                logits = model(buf_z)
                # Per-sample-mean formulation (matches 'random'/'papl' modes).
                # For PUMA's streaming buffer, each row has its own #masked and
                # the per-sample mean equals weighted token-level NLL inside sample.
                log_probs = F.log_softmax(logits.float(), dim=-1)
                tlp = log_probs.gather(-1, buf_x0.unsqueeze(-1)).squeeze(-1)
                nll = -tlp
                n_masked = m.sum(dim=-1).clamp_min(1).float()
                per_sample = (nll * m.float()).sum(dim=-1) / n_masked
                loss = per_sample.mean()
            tg += m.sum().item()
        else:
            ids, ans_starts, idx = _next_batch()
            B_b = ids.shape[0]
            ap = (ans_starts.unsqueeze(1) + _arange).clamp(max=T - 1)
            bi = torch.arange(B_b, device=device).unsqueeze(1).expand_as(ap)
            # In fresh-data mode there is no persistent blank_masks indexed by
            # sample id; the answer region is always fully blank (matches the
            # addition experiment's default of torch.ones blank_masks).
            if idx is not None:
                bl = blank_masks[idx]
            else:
                bl = torch.ones(B_b, ans_len, dtype=torch.bool, device=device)
            t_ratio = torch.rand(B_b, device=device)
            m_probs = torch.zeros(B_b, T, dtype=torch.float, device=device)
            m_probs[bi, ap] = t_ratio.unsqueeze(1) * bl.float()
            m = torch.bernoulli(m_probs).bool()
            no_m = ~m.any(dim=1)
            if no_m.any():
                rs = torch.rand_like(bl[no_m].float())
                rs[~bl[no_m]] = -1.0
                cj = rs.argmax(dim=1)
                ca = ap[no_m].gather(1, cj.unsqueeze(1)).squeeze(1)
                m[no_m.nonzero(as_tuple=True)[0], ca] = True
            xm = ids.clone()
            xm[m] = mask_id
            with ctx:
                logits = model(xm)
                if m.sum() == 0:
                    continue
                # Per-sample-mean formulation: each sample's loss is the mean
                # NLL over its masked positions, then averaged over batch.
                # PAPL paper Eq. 7 with α=0 reduces to this (= standard DLM ELBO).
                log_probs = F.log_softmax(logits.float(), dim=-1)
                tlp = log_probs.gather(-1, ids.unsqueeze(-1)).squeeze(-1)  # [B,T]
                nll = -tlp                                                  # [B,T]
                n_masked = m.sum(dim=-1).clamp_min(1).float()               # [B]
                if mask_type == 'papl':
                    # PAPL loss (Peng et al. 2025, paper Eq. 7):
                    #   L = -E_{x_0,k,x_k} Σ_{i: x_k^i=mask}
                    #         (1/(L-k)) (1 + α·w^i) log P(x_0^i | x_k)
                    #   where w^i ∝ exp((1/τ) log P(x_0^i | x_k)),
                    #         softmax over masked positions, detached.
                    # Direction: confident-on-correct masked positions get
                    # additional weight beyond the base 1/(L-k). w^i is detached
                    # — no gradient through the planner-weight branch.
                    # The paper's github repo provides a simpler educational
                    # form (weight = 1 + α·P(correct), no softmax/τ); we use
                    # the formal Eq. 7 version with paper-default τ=1, α=1.
                    det = (tlp.detach() / papl_tau).masked_fill(~m, float('-inf'))
                    w_papl = F.softmax(det, dim=-1)                         # [B,T]
                    base_w = (1.0 / n_masked).unsqueeze(-1)                 # [B,1]
                    weights = base_w * (1.0 + papl_alpha * w_papl)          # [B,T]
                    per_sample = (weights * nll * m.float()).sum(dim=-1)
                    loss = per_sample.mean()
                else:
                    # 'random': per-sample mean of masked NLL, batch mean
                    per_sample = (nll * m.float()).sum(dim=-1) / n_masked
                    loss = per_sample.mean()
            tg += m.sum().item()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        with torch.no_grad():
            for name, param in model.named_parameters():
                ema_state[name].lerp_(param.data, 1 - ema_decay)

        # ── Multi-checkpoint save (EMA snapshot at milestones) ──
        if ckpt_save_fn is not None and it in ckpt_schedule_set:
            with torch.no_grad():
                ema_cpu = {k: v.detach().cpu().clone() for k, v in ema_state.items()}
            ckpt_save_fn(ema_cpu, it)

        # Stratum-level training loss accumulation (per-token, scattered by sample stratum)
        if log_strata and m.sum() > 0:
            with torch.no_grad():
                if uses_streaming:
                    tgt_tok = buf_x0[m]
                    row_idx = m.nonzero(as_tuple=True)[0]    # [K_masked]
                    sample_strata_b = sample_strata[buf_orig_idx]  # [B]
                else:
                    tgt_tok = ids[m]
                    row_idx = m.nonzero(as_tuple=True)[0]
                    sample_strata_b = sample_strata[idx]     # [B]
                token_losses = F.cross_entropy(
                    logits[m].float(), tgt_tok, reduction='none')  # [K_masked]
                token_strata = sample_strata_b[row_idx]            # [K_masked]
                strat_loss_sum.scatter_add_(0, token_strata, token_losses)
                strat_token_count.scatter_add_(0, token_strata,
                                                torch.ones_like(token_strata))

        if uses_streaming:
            _advance(logits.detach(), K_cur)

        if it % log_every == 0:
            dynamics['train_loss'].append((it, loss.item()))
            if log_strata:
                denom = strat_token_count.clamp(min=1).float()
                per_strat = (strat_loss_sum / denom).cpu().tolist()
                per_n = strat_token_count.cpu().tolist()
                dynamics['stratified_loss'].append({
                    'iter': it,
                    'per_stratum_loss': per_strat,
                    'per_stratum_n': per_n,
                })
                strat_loss_sum.zero_()
                strat_token_count.zero_()
            print(f"    it {it:6d}/{max_iters} | loss {loss.item():.4f} | "
                  f"lr {get_lr(it):.1e} | tg {tg:,} | {time.time() - t0:.0f}s")

        if it % eval_every == 0 or \
           (it <= max_iters * 0.1 and it % max(eval_every // 5, 1) == 0):
            should_stop = _do_eval(it)
            model.train()
            if should_stop:
                print(f"    ✓ Early stop at it {it} (best={best_iter}, "
                      f"patience={patience}, loss={best_loss:.4f})")
                stopped_early = True
                break

    # ── Load best EMA ──
    if best_ema:
        model.load_state_dict({k: v.to(device) for k, v in best_ema.items()})
    elif ema_state:
        model.load_state_dict(ema_state)
    model.eval()
    if not stopped_early:
        _do_eval(max_iters)
    status = f"early stopped at {best_iter}" if stopped_early else f"full {max_iters} iters"
    print(f"  ✓ Done ({status}, best probe loss: {best_loss:.4f}, "
          f"{time.time() - t0:.0f}s)")
    return model, dynamics


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Generation: AR (greedy)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@torch.no_grad()
def generate_ar(model, prefix_ids, n_tokens, device=None):
    if device is None:
        device = DEVICE
    model.eval()
    x = prefix_ids.to(device)
    for _ in range(n_tokens):
        logits = model(x)[:, -1, :]
        x = torch.cat([x, logits.argmax(-1, keepdim=True)], dim=1)
    return x


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Generation: Diffusion (greedy, order tracking)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@torch.no_grad()
def generate_diffusion(model, prefix_ids, n_tokens, mask_id,
                       policy='confidence', greedy=True,
                       parallel_k=1, pad_to=None, pad_id=None,
                       reasoning_rank=None,
                       device=None):
    """
    Masked diffusion generation with decode order tracking.

    Args:
        policy: 'confidence', 'l2r', 'r2l', 'random', 'layered_oracle'
        greedy: argmax (True) or sample (False)
        pad_to: pad to training-length for consistent bidirectional attention
        reasoning_rank: [B, n_tokens] long tensor of per-position partial-order
            ranks (smaller = decode earlier). Used by 'layered_oracle' policy:
            among masked answer positions, select those with minimum rank,
            then break ties by confidence. Positions with rank >= n_tokens
            are treated as "not in the answer" (e.g., rainbow padding past
            trace_len) and decoded last in arbitrary order.
    Returns: (sequences, log_probs, decode_info)
    """
    if device is None:
        device = DEVICE
    model.eval()
    B = prefix_ids.shape[0]
    T_pre = prefix_ids.shape[1]
    T_ans = n_tokens
    T = T_pre + T_ans

    x = torch.full((B, T), mask_id, dtype=torch.long, device=device)
    x[:, :T_pre] = prefix_ids.to(device)
    unmasked = torch.zeros(B, T, dtype=torch.bool, device=device)
    unmasked[:, :T_pre] = True

    if pad_to is not None and pad_to > T:
        assert pad_id is not None
        n_pad = pad_to - T
        pad_block = torch.full((B, n_pad), pad_id, dtype=torch.long, device=device)
        x = torch.cat([x, pad_block], dim=1)
        pad_mask = torch.ones(B, n_pad, dtype=torch.bool, device=device)
        unmasked = torch.cat([unmasked, pad_mask], dim=1)
        T_total = pad_to
    else:
        T_total = T

    # Build full-sequence-aligned reasoning rank for layered_oracle (ans region only)
    if policy == 'layered_oracle':
        assert reasoning_rank is not None, "layered_oracle requires reasoning_rank"
        # reasoning_rank: [B, n_tokens] — extend to [B, T_total] with +inf outside ans
        full_rank = torch.full((B, T_total), float('inf'), device=device)
        full_rank[:, T_pre:T_pre + n_tokens] = reasoning_rank.to(device).float()

    scores = torch.zeros(B, device=device)
    orders = []

    for t in range(n_tokens):
        logits = model(x)
        # Exclude mask token from all selections (in-place, safe under no_grad)
        logits[:, :, mask_id] = -float('inf')

        if policy == 'confidence':
            max_logit = logits.max(dim=-1).values
            max_logit[unmasked] = -float('inf')
            pos = max_logit.argmax(-1)
        elif policy == 'l2r':
            pos = torch.full((B,), T_pre + t, dtype=torch.long, device=device)
        elif policy == 'r2l':
            pos = torch.full((B,), T_pre + n_tokens - 1 - t, dtype=torch.long, device=device)
        elif policy == 'random':
            rand_scores = torch.rand(B, T_total, device=device)
            rand_scores[unmasked] = -float('inf')
            pos = rand_scores.argmax(-1)
        elif policy == 'layered_oracle':
            # Among masked positions, find those with minimum rank per sample;
            # break ties by confidence (max-logit). Two-stage selection.
            max_logit = logits.max(dim=-1).values            # [B, T_total]
            # Mask-out ineligible: already-unmasked or rank=inf (outside ans)
            mask_inel = unmasked | (full_rank == float('inf'))
            # Compute min rank per row over eligible positions
            rank_eff = full_rank.clone()
            rank_eff[mask_inel] = float('inf')
            min_rank = rank_eff.min(dim=-1, keepdim=True).values  # [B, 1]
            # Eligible at min rank
            elig_min = (rank_eff == min_rank) & ~mask_inel
            # Score: confidence among elig_min, else -inf
            score = max_logit.clone()
            score[~elig_min] = -float('inf')
            pos = score.argmax(-1)
        else:
            raise ValueError(f"Unknown policy: {policy}")

        batch_arange = torch.arange(B, device=device)
        lp_at_pos = logits[batch_arange, pos]
        if greedy:
            tok = lp_at_pos.argmax(-1)
        else:
            tok = torch.multinomial(F.softmax(lp_at_pos, dim=-1), 1).squeeze(-1)
        scores += F.log_softmax(lp_at_pos, dim=-1)[batch_arange, tok]
        x[batch_arange, pos] = tok
        unmasked[batch_arange, pos] = True
        orders.append(pos)

    orders_t = torch.stack(orders, dim=1).cpu() if orders else None
    return x, scores.cpu(), {'n_steps': len(orders), 'orders': orders_t}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Evaluation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def evaluate(model, tokenizer, test_samples, objective,
             get_answer_fn, get_full_answer_fn=None,
             policy='confidence', greedy=True,
             batch_size=128, device=None):
    """Exact match evaluation with variable-length answer support."""
    if device is None:
        device = DEVICE
    if get_full_answer_fn is None:
        get_full_answer_fn = lambda s: s.split('=', 1)[1]

    mask_id = tokenizer.special_ids['mask']
    model.eval()
    correct, total = 0, 0
    examples = []
    all_orders = {}
    per_sample = []

    groups = {}
    for idx, s in enumerate(test_samples):
        full = get_full_answer_fn(s)
        al = len(tokenizer.encode(full))
        groups.setdefault(al, []).append((idx, s))

    for ans_len, items in groups.items():
        samples_in_group = [s for _, s in items]
        for start in range(0, len(samples_in_group), batch_size):
            batch = samples_in_group[start:start + batch_size]
            B = len(batch)
            prefixes = [s.split('=')[0] + '=' for s in batch]
            gold_full = [get_full_answer_fn(s) for s in batch]
            gold_final = [get_answer_fn(s) for s in batch]

            penc = [tokenizer.encode(p) for p in prefixes]
            pmax = max(len(p) for p in penc)
            pids = torch.full((B, pmax), tokenizer.special_ids['pad'], dtype=torch.long)
            for i, e in enumerate(penc):
                pids[i, :len(e)] = torch.tensor(e)

            if objective == 'ar':
                gen = generate_ar(model, pids, ans_len, device)
                pred_ids = gen[:, pmax:pmax + ans_len]
                batch_orders = None
            else:
                gen, _, info = generate_diffusion(
                    model, pids, ans_len, mask_id,
                    policy=policy, greedy=greedy, device=device)
                pred_ids = gen[:, pmax:pmax + ans_len]
                batch_orders = info.get('orders')

            if batch_orders is not None:
                all_orders.setdefault(ans_len, []).append(batch_orders)

            for i in range(B):
                pred_str = tokenizer.decode(pred_ids[i].cpu().tolist())
                pred_final = pred_str
                ok = pred_final == gold_final[i]
                correct += int(ok)
                total += 1
                per_sample.append(ok)
                if len(examples) < 10:
                    examples.append({
                        'prefix': prefixes[i], 'pred': pred_final,
                        'gold': gold_final[i], 'correct': ok})

    result = {
        'exact_match': correct / max(total, 1),
        'n_correct': correct, 'n_total': total,
        'examples': examples,
        'per_sample_correct': per_sample,
    }
    if all_orders:
        biggest_group = max(all_orders.keys(), key=lambda k: sum(
            o.shape[0] for o in all_orders[k]))
        result['decode_orders'] = torch.cat(all_orders[biggest_group], dim=0)
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PUMA reveal trajectory — shared across all domains
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@torch.no_grad()
def simulate_reveal_trajectory(
    model, tokenizer, test_ids, ans_starts, ans_len,
    blank_masks=None, K=16, tau=0.9, batch_size=64, device=None):
    """
    Run PUMA-style forward process on a batch of examples, tracking for each
    (example, position) at which stage the position was first revealed.

    This is the shared diagnostic for all domains: it shows exactly which
    positions PUMA's confidence ordering defers (or never reaches) during
    training.

    Args:
        test_ids:    [N, T] encoded sequences (prefix + gold answer + pad)
        ans_starts:  [N] answer-region start positions
        ans_len:     int, length of answer region
        blank_masks: [N, ans_len] bool, True = maskable position
                     (None = all positions maskable)
        K:           number of PUMA stages
        tau:         confidence threshold for immediate reveal (PUMA paper)

    Returns dict:
        reveal_stage:        [N, ans_len] int, stage at which each position was
                             first revealed (0..K-1; K = never revealed)
        still_masked_start:  [N, K+1, ans_len] bool — mask state at start of each stage.
                             [:, 0, :] = initial blank_masks; [:, K, :] = final
        n_revealed_per_stage:[N, K] int — how many maskable positions revealed at stage k
    """
    if device is None:
        device = DEVICE
    model.eval()
    mask_id = tokenizer.special_ids['mask']
    N, T = test_ids.shape

    # Defaults
    if blank_masks is None:
        blank_masks = torch.ones(N, ans_len, dtype=torch.bool)
    blank_masks = blank_masks.to(device)
    test_ids = test_ids.to(device)
    ans_starts = ans_starts.to(device)

    _arange = torch.arange(ans_len, device=device)

    reveal_stage = torch.full((N, ans_len), K, dtype=torch.long, device=device)
    still_masked_start = torch.zeros(N, K + 1, ans_len, dtype=torch.bool, device=device)
    still_masked_start[:, 0, :] = blank_masks.clone()
    n_revealed_per_stage = torch.zeros(N, K, dtype=torch.long, device=device)

    for st in range(0, N, batch_size):
        en = min(st + batch_size, N)
        B = en - st
        ids_b = test_ids[st:en].clone()     # [B, T]
        ans_b = ans_starts[st:en]            # [B]
        bl_b = blank_masks[st:en].clone()    # [B, ans_len]

        # Absolute positions in the sequence
        ap = (ans_b.unsqueeze(1) + _arange).clamp(max=T - 1)  # [B, ans_len]
        bi = torch.arange(B, device=device).unsqueeze(1).expand_as(ap)

        # is_m[b, j] = True iff answer position j in example b is still masked
        is_m = bl_b.clone()
        # Initialize the masked region
        ids_b[bi[is_m], ap[is_m]] = mask_id

        for k in range(K):
            still_masked_start[st:en, k, :] = is_m

            if not is_m.any():
                # All revealed; later stages stay fully revealed
                for k2 in range(k + 1, K + 1):
                    still_masked_start[st:en, k2, :] = False
                break

            # Forward pass
            logits = model(ids_b)                # [B, T, V]
            lp = logits[bi, ap].clone()          # [B, ans_len, V]
            lp[:, :, mask_id] = -float('inf')
            confs = F.softmax(lp, dim=-1).max(dim=-1).values  # [B, ans_len]
            confs = torch.where(is_m, confs, torch.full_like(confs, -float('inf')))

            # PUMA streaming logic (matches train_diffusion _advance)
            nm = is_m.sum(dim=1).float()                      # [B]
            K_rem = torch.full((B,), K - k, dtype=torch.float, device=device)
            nr = (nm / K_rem).ceil().long().clamp(min=1)      # [B]

            ranked = confs.argsort(dim=1, descending=True)    # [B, ans_len]
            rop = torch.zeros_like(ranked)
            rop.scatter_(1, ranked, _arange.expand(B, -1))

            reveal = ((rop < nr.unsqueeze(1)) | (confs > tau)) & is_m
            n_revealed_per_stage[st:en, k] = reveal.sum(dim=1)

            # Teacher-force: reveal gold tokens at these positions
            gold_ids = test_ids[st:en, :]     # [B, T]
            reveal_b, reveal_j = reveal.nonzero(as_tuple=True)
            if reveal_b.numel() > 0:
                reveal_abs = ap[reveal_b, reveal_j]
                ids_b[reveal_b, reveal_abs] = gold_ids[reveal_b, reveal_abs]
                reveal_stage[st + reveal_b, reveal_j] = k
                is_m[reveal_b, reveal_j] = False

        # Record final state
        still_masked_start[st:en, K, :] = is_m

    return {
        'reveal_stage': reveal_stage.cpu(),
        'still_masked_start': still_masked_start.cpu(),
        'n_revealed_per_stage': n_revealed_per_stage.cpu(),
        'K': K, 'tau': tau, 'N': N,
    }


def compute_reveal_vs_order_tau(reveal_stage, reasoning_order, blank_masks=None):
    """Per-example rank correlation between PUMA's reveal order and a canonical
    reasoning order. Used as training-time diagnostic: measures whether the
    model's confidence-induced unmasking order matches the task's logical
    deduction order.

    Handles ties robustly via a fallback chain:
      1. Kendall τ-b (scipy default, ties-aware)
      2. Kendall τ-c (for categorical/heavily-tied data)
      3. Spearman ρ (if both τ variants fail — e.g. all-tied edge case)
      4. NaN only as last resort
    The ties edge case is real: PUMA reveals many positions in a single stage
    on easy examples, inflating ties. In earlier runs, scipy's default
    kendalltau returned NaN for such cases, discarding valid signal.

    Args:
        reveal_stage:    tensor/array [N, L] — stage at which each position
                         was first revealed (low = earlier)
        reasoning_order: tensor/array [N, L] — canonical rank per position
                         (low = earlier in reasoning order)
        blank_masks:     optional bool [N, L] — maskable positions only.
                         If None, all L positions included.

    Returns: numpy array [N] — per-example correlation (NaN only if too few
             maskable positions or if all positions have identical rank in
             either reveal_stage or reasoning_order).
    """
    import numpy as np
    from scipy.stats import kendalltau, spearmanr

    rs = reveal_stage.cpu().numpy() if hasattr(reveal_stage, 'cpu') else np.asarray(reveal_stage)
    ro = reasoning_order.cpu().numpy() if hasattr(reasoning_order, 'cpu') else np.asarray(reasoning_order)
    if blank_masks is not None:
        bm = blank_masks.cpu().numpy() if hasattr(blank_masks, 'cpu') else np.asarray(blank_masks)
        bm = bm.astype(bool)
    else:
        bm = np.ones_like(rs, dtype=bool)

    N = rs.shape[0]
    taus = np.full(N, np.nan, dtype=np.float32)
    for i in range(N):
        m = bm[i]
        if m.sum() < 2:
            continue
        x = rs[i][m]
        y = ro[i][m]
        # If either side is constant (all same), correlation is undefined;
        # skip early rather than firing scipy warnings.
        if np.unique(x).size < 2 or np.unique(y).size < 2:
            continue

        # Primary: Kendall τ-b (ties adjustment)
        t, _ = kendalltau(x, y, variant='b')
        if t is not None and not np.isnan(t):
            taus[i] = float(t); continue

        # Fallback 1: Kendall τ-c (for categorical with ties)
        t, _ = kendalltau(x, y, variant='c')
        if t is not None and not np.isnan(t):
            taus[i] = float(t); continue

        # Fallback 2: Spearman ρ (average ranks, robust to ties)
        rho, _ = spearmanr(x, y)
        if rho is not None and not np.isnan(rho):
            taus[i] = float(rho); continue

        # Last resort: leave NaN (truly degenerate)
    return taus

