"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Experiment 3 — Toy Distribution: Decoding Policy Analysis
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Colab:  %run experiments/exp_toy_distribution.py

  Train diffusion on toy distributions with KNOWN ground truth p(x).
  Compare decoding policies via exact distributional metrics.

  4 distributions (V=4, L=8 → 65536 sequences, fully enumerable):
    A: Near-independent  (no dependency)
    B: 2nd-order Markov  (strong local dependency)
    C: Global sum        (long-range coupling)
    D: Markov + Global   (local + global)

  3 policy families:
    Sequential (7): confidence, low_entropy, high_entropy, margin,
                    random, l2r, r2l  (all NFE=8)
    Adaptive threshold (3): τ=0.5, 0.7, 0.9 — decode all positions
                    above threshold simultaneously (variable NFE)
    Jacobi iteration (3): max_iter=5, 10, 20 — predict all at once,
                    iterate to fixed point (variable NFE)

  Key question: can adaptive/jacobi match sequential quality with
  fewer forward passes?  NFE vs TV scatter shows the answer.

  Decoding: sampling (not greedy — distributional metrics require
  stochasticity). Decode order tracked for first 10K samples only
  (speed optimization).
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                if '__file__' in dir() else '.')

import time, math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('Agg')
from itertools import product as iter_product
from scipy.stats import spearmanr

from core.model import Transformer
from core.train_utils import mount_drive, save_results, DEVICE

EXP_NAME = 'exp_toy_distribution'

# ── Config ──────────────────────────────────────────
V = 4            # vocab {0,1,2,3}
L = 8            # length → V^L = 65,536 possible sequences
MASK_ID = V      # = 4
TOTAL_V = V + 1  # = 5

N_TRAIN   = 200_000
N_SAMPLES = 500_000   # per policy — need >> V^L for reliable TV/KL
                       # TV noise ≈ sqrt(V^L / 2πN) ≈ 0.14 at 500K
BOOTSTRAP_K = 20       # bootstrap resamples for CI
MAX_EPOCHS = 60
PATIENCE_EPOCHS = 8
BATCH_SIZE = 1024      # larger batch for faster generation
LR = 3e-4

D_MODEL = 192; N_HEADS = 4; N_LAYERS = 4

# ── Distributions ───────────────────────────────────

class ToyDistribution:
    def __init__(self):
        self._full = None

    def log_prob(self, x):
        raise NotImplementedError

    def sample(self, n):
        all_seqs, probs = self.full_distribution()
        return all_seqs[torch.multinomial(probs, n, replacement=True)]

    def full_distribution(self):
        if self._full is not None:
            return self._full
        all_seqs = torch.tensor(list(iter_product(range(V), repeat=L)),
                                dtype=torch.long)
        lp = self.log_prob(all_seqs)
        lp = lp - torch.logsumexp(lp, dim=0)
        self._full = (all_seqs, torch.exp(lp))
        return self._full

    def entropy(self):
        _, p = self.full_distribution()
        return -(p * (p + 1e-30).log()).sum().item() / np.log(2)

    def optimal_masked_loss(self, n_eval=50_000):
        """
        Compute the theoretically optimal cross-entropy for a perfect
        masked predictor. This is the Bayes-optimal loss — no model of
        any size can beat this.

        For each (x, mask_config), the optimal predictor outputs
        p(x_i | x_{unmasked}) for each masked position i.
        We Monte-Carlo estimate this over random x and random masks.
        """
        all_seqs, p_true = self.full_distribution()
        samples = self.sample(n_eval)
        total_loss, total_tokens = 0.0, 0

        # For tractability, compute exact conditionals
        # p(x_i=v | x_unmasked) = sum_{x': x'_unmasked=x_unmasked, x'_i=v} p(x')
        #                        / sum_{x': x'_unmasked=x_unmasked} p(x')
        # This is expensive for large V^L, so we use a sampling approach:
        # sample mask configs, then for each, compute the conditional.

        # Faster approach: for random mask ratios, estimate expected CE
        rng = np.random.RandomState(0)
        for trial in range(min(2000, n_eval)):
            x = samples[trial]
            t = rng.uniform(0.1, 0.9)
            mask = torch.bernoulli(torch.full((L,), t)).bool()
            if not mask.any():
                mask[rng.randint(L)] = True

            unmasked_vals = x.clone()
            unmasked_vals[mask] = -1  # sentinel

            # Find all sequences matching unmasked positions
            match = torch.ones(len(all_seqs), dtype=torch.bool)
            for pos in range(L):
                if not mask[pos]:
                    match = match & (all_seqs[:, pos] == x[pos])

            if match.sum() == 0:
                continue

            cond_probs = p_true[match]
            cond_probs = cond_probs / cond_probs.sum()
            matching = all_seqs[match]

            for pos in mask.nonzero(as_tuple=True)[0]:
                true_val = x[pos].item()
                # p(x_pos = true_val | context)
                p_cond = cond_probs[matching[:, pos] == true_val].sum().item()
                if p_cond > 1e-10:
                    total_loss -= np.log(p_cond)
                else:
                    total_loss += 20  # cap
                total_tokens += 1

        return total_loss / max(total_tokens, 1)


class NearIndependent(ToyDistribution):
    """
    p(x) = prod p_i(x_i). Position-wise bias, no dependency.
    alpha controls peakedness: lower → more peaked marginals.
      alpha=3.0: near-uniform (H ≈ 1.9 bits/pos → cond_acc ~0.39)
      alpha=0.5: sparse/peaked (H ≈ 1.0 bits/pos → cond_acc ~0.65)
    """
    def __init__(self, alpha=0.5, seed=42):
        super().__init__()
        rng = np.random.RandomState(seed)
        self.pos_log_probs = torch.tensor(
            np.log(rng.dirichlet([alpha] * V, size=L) + 1e-30),
            dtype=torch.float32)

    def log_prob(self, x):
        x = x.reshape(-1, L)
        return sum(self.pos_log_probs[i][x[:, i]] for i in range(L))


class MarkovChain(ToyDistribution):
    """
    2nd-order Markov chain with peaked transition matrices.
    sparsity controls how peaked: lower → more deterministic transitions.
    """
    def __init__(self, order=2, sparsity=0.1, seed=42):
        super().__init__()
        self.k = order
        rng = np.random.RandomState(seed)
        # Build transition tensor: (V^order, V) log-probs
        n_ctx = V ** order
        self.trans_logp = torch.zeros(n_ctx, V, dtype=torch.float32)
        for idx, ctx in enumerate(iter_product(range(V), repeat=order)):
            alpha = rng.uniform(sparsity * 0.5, sparsity, size=V)
            alpha[rng.randint(V)] += 3.0
            probs = rng.dirichlet(alpha)
            self.trans_logp[idx] = torch.tensor(np.log(probs + 1e-30),
                                                 dtype=torch.float32)
        self.init_lp = np.log(1.0 / n_ctx)
        # Powers for context encoding: ctx_idx = sum(x[t-k+i] * V^i)
        self._ctx_powers = V ** torch.arange(order)

    def _ctx_index(self, x_ctx):
        """x_ctx: (..., order) -> (...) integer index into trans table."""
        return (x_ctx * self._ctx_powers).sum(-1)

    def log_prob(self, x):
        x = x.reshape(-1, L); n = x.shape[0]
        lp = torch.full((n,), self.init_lp, dtype=torch.float32)
        for t in range(self.k, L):
            ctx_idx = self._ctx_index(x[:, t-self.k:t])  # (n,)
            lp += self.trans_logp[ctx_idx, x[:, t]]       # (n,)
        return lp


class GlobalSumConstraint(ToyDistribution):
    """
    p(x) ∝ exp(-β|Σx_i - τ|²) × ∏ p_i(x_i).
    beta controls coupling strength: higher → stronger global dependency.
    """
    def __init__(self, beta=2.0, alpha=0.5, seed=42):
        super().__init__()
        self.beta, self.target = beta, (V-1)*L/2.0
        rng = np.random.RandomState(seed)
        self.base_lp = torch.tensor(
            np.log(rng.dirichlet([alpha]*V, size=L) + 1e-30), dtype=torch.float32)

    def log_prob(self, x):
        x = x.reshape(-1, L)
        lp = sum(self.base_lp[i][x[:, i]] for i in range(L))
        lp += -self.beta * (x.float().sum(-1) - self.target) ** 2
        return lp


class MarkovPlusGlobal(ToyDistribution):
    """Markov local + global sum constraint. Both strong."""
    def __init__(self, beta=1.5, order=2, sparsity=0.1, seed=42):
        super().__init__()
        self.markov = MarkovChain(order, sparsity, seed)
        self.beta, self.target = beta, (V-1)*L/2.0

    def log_prob(self, x):
        x = x.reshape(-1, L)
        lp = self.markov.log_prob(x)
        lp += -self.beta * (x.float().sum(-1) - self.target) ** 2
        return lp


class HardMarkov(ToyDistribution):
    """
    Nearly deterministic 2nd-order Markov chain.
    Each context maps to exactly one token with probability p_peak,
    the remaining tokens share (1 - p_peak) uniformly.

    l2r is provably optimal: given correct context, conditional
    entropy per step ≈ H(Bernoulli(p_peak)) ≈ 0.12 bits.
    Any other ordering must marginalise over uncertain context,
    giving much higher conditional entropy.

    We also store the deterministic map for analytic comparison.
    """
    def __init__(self, order=2, p_peak=0.97, seed=42):
        super().__init__()
        self.k = order
        self.p_peak = p_peak
        rng = np.random.RandomState(seed)
        n_ctx = V ** order
        # Deterministic map: context → preferred token
        self.preferred = rng.randint(0, V, size=n_ctx)
        # Build transition log probs
        p_other = (1.0 - p_peak) / (V - 1)
        self.trans_logp = torch.full(
            (n_ctx, V), np.log(p_other + 1e-30), dtype=torch.float32)
        for c in range(n_ctx):
            self.trans_logp[c, self.preferred[c]] = np.log(p_peak)
        # Uniform initial (first `order` tokens)
        self.init_lp = np.log(1.0 / n_ctx)
        self._ctx_powers = V ** torch.arange(order)

    def _ctx_index(self, x_ctx):
        return (x_ctx * self._ctx_powers).sum(-1)

    def log_prob(self, x):
        x = x.reshape(-1, L); n = x.shape[0]
        lp = torch.full((n,), self.init_lp, dtype=torch.float32)
        for t in range(self.k, L):
            ctx_idx = self._ctx_index(x[:, t-self.k:t])
            lp += self.trans_logp[ctx_idx, x[:, t]]
        return lp

    def theoretical_l2r_entropy(self):
        """Conditional entropy per step for l2r decoding."""
        p_other = (1.0 - self.p_peak) / (V - 1)
        h = -(self.p_peak * np.log2(self.p_peak)
              + (V - 1) * p_other * np.log2(p_other + 1e-30))
        return h

    def expected_accuracy_l2r(self):
        """Expected per-token accuracy for l2r greedy decoding."""
        return self.p_peak


# ── Train ───────────────────────────────────────────

def train_toy_model(dist, print_every=5):
    print(f"  Generating {N_TRAIN} samples...")
    train_data = dist.sample(N_TRAIN)

    model = Transformer(
        vocab_size=TOTAL_V, block_size=L + 4,
        n_layer=N_LAYERS, n_head=N_HEADS, n_embd=D_MODEL,
        dropout=0.1, is_causal=False, pos_enc='absolute',
    ).to(DEVICE)

    loader = DataLoader(train_data, batch_size=BATCH_SIZE,
                        shuffle=True, drop_last=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS * len(loader))

    history = {'loss': [], 'cond_acc': []}
    best_loss, wait = float('inf'), 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        epoch_loss, n_b = 0, 0
        for batch in loader:
            batch = batch.to(DEVICE); B, T = batch.shape
            t = torch.rand(B, device=DEVICE)
            mask = torch.bernoulli(t.unsqueeze(1).expand(B, T)).bool()
            no_m = ~mask.any(dim=1)
            if no_m.any():
                mask[no_m, torch.randint(0, T, (no_m.sum(),), device=DEVICE)] = True
            x_m = batch.clone(); x_m[mask] = MASK_ID
            logits = model(x_m)
            loss = F.cross_entropy(logits[mask], batch[mask])
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step()
            epoch_loss += loss.item(); n_b += 1

        avg_loss = epoch_loss / n_b
        history['loss'].append(avg_loss)

        model.eval()
        with torch.no_grad():
            x = dist.sample(2000).to(DEVICE)
            t = torch.rand(2000, device=DEVICE) * 0.8 + 0.1
            m = torch.bernoulli(t.unsqueeze(1).expand(2000, L)).bool()
            no_m = ~m.any(dim=1)
            if no_m.any():
                m[no_m, torch.randint(0, L, (no_m.sum(),), device=DEVICE)] = True
            x_m = x.clone(); x_m[m] = MASK_ID
            logits = model(x_m)
            acc = ((logits.argmax(-1) == x) & m).sum().float() / m.sum().float()
        history['cond_acc'].append(acc.item())

        if epoch % print_every == 0 or epoch == 1:
            print(f"    epoch {epoch:3d} | loss {avg_loss:.4f} | "
                  f"cond_acc {acc.item():.4f}")

        # Convergence check
        if avg_loss < best_loss - 1e-4:
            best_loss, wait = avg_loss, 0
        else:
            wait += 1
            if wait >= PATIENCE_EPOCHS:
                print(f"    ✓ Converged at epoch {epoch}")
                break

    model.eval()
    return model, history


# ── Decode ──────────────────────────────────────────

N_ORDER_SAMPLES = 10_000  # only track decode order for this many


def _sanitize_decode_output(x, label=''):
    """Clamp any leaked MASK tokens. Returns sanitized tensor."""
    result = x.cpu() if x.is_cuda else x
    n_mask = (result >= V).sum().item()
    if n_mask > 0:
        print(f"    WARNING {label}: {n_mask} tokens >= V, clamping to V-1")
        result = result.clamp(max=V-1)
    return result


@torch.no_grad()
def decode_sequential(model, n, policy='confidence', track_order=False):
    """Sequential decode. track_order=True records position chosen at each step."""
    x = torch.full((n, L), MASK_ID, dtype=torch.long, device=DEVICE)
    unmasked = torch.zeros(n, L, dtype=torch.bool, device=DEVICE)
    scores = torch.zeros(n, device=DEVICE)
    orders = torch.zeros(n, L, dtype=torch.long, device=DEVICE) if track_order else None

    for t in range(L):
        logits = model(x)
        logits[:, :, MASK_ID] = -float('inf')
        probs = F.softmax(logits, dim=-1)

        if policy == 'confidence':
            sc = probs.max(-1).values.clone(); sc[unmasked] = -1
            pos = sc.argmax(-1)
        elif policy == 'low_entropy':
            e = -(probs*(probs+1e-10).log()).sum(-1); e[unmasked] = 1e9
            pos = e.argmin(-1)
        elif policy == 'high_entropy':
            e = -(probs*(probs+1e-10).log()).sum(-1); e[unmasked] = -1e9
            pos = e.argmax(-1)
        elif policy == 'margin':
            t2 = probs.topk(2, dim=-1).values
            mg = t2[:,:,0]-t2[:,:,1]; mg[unmasked] = -1; pos = mg.argmax(-1)
        elif policy == 'random':
            # Assign random scores to unmasked positions, -inf to masked
            rand_sc = torch.rand(n, L, device=DEVICE)
            rand_sc[unmasked] = -1
            pos = rand_sc.argmax(-1)
        elif policy == 'l2r':
            pos = torch.full((n,), t, dtype=torch.long, device=DEVICE)
        elif policy == 'r2l':
            pos = torch.full((n,), L-1-t, dtype=torch.long, device=DEVICE)
        elif policy == 'entropy_adj':
            # Zero-cost proxy: confidence penalised by entropy
            # High confidence + low entropy = genuinely easy (decode first)
            # High confidence + high entropy = misleading (defer)
            max_p = probs.max(-1).values.clone()
            ent = -(probs*(probs+1e-10).log()).sum(-1)
            sc = max_p * (1.0 - ent / math.log(V))  # ∈ [0, 1]
            sc[unmasked] = -1
            pos = sc.argmax(-1)
        elif policy == 'single_perturb':
            # 1-extra-forward-pass variance estimate
            # Fill all masked positions with samples, re-forward,
            # compare → positions that change are unstable
            max_p_base = probs.max(-1).values.clone()  # (n, L)
            x_pert = x.clone()
            masked = ~unmasked
            for j in range(L):
                m_j = masked[:, j]
                if m_j.any():
                    toks = torch.multinomial(probs[m_j, j], 1).squeeze(-1)
                    x_pert[m_j, j] = toks
            logits_pert = model(x_pert)
            logits_pert[:, :, MASK_ID] = -float('inf')
            max_p_pert = F.softmax(logits_pert, dim=-1).max(-1).values
            # Instability = |confidence_base - confidence_perturbed|
            instab = (max_p_base - max_p_pert).abs()
            instab[unmasked] = float('inf')
            pos = instab.argmin(-1)  # most stable = least change
        else:
            raise ValueError(policy)

        if track_order:
            orders[:, t] = pos

        lp = logits[torch.arange(n, device=DEVICE), pos]
        tok = torch.multinomial(F.softmax(lp, dim=-1), 1).squeeze(-1)
        scores += F.log_softmax(lp, dim=-1)[torch.arange(n, device=DEVICE), tok]
        x[torch.arange(n, device=DEVICE), pos] = tok
        unmasked[torch.arange(n, device=DEVICE), pos] = True

    return _sanitize_decode_output(x, 'decode_seq'), scores.cpu(), (orders.cpu() if track_order else None)


@torch.no_grad()
def decode_low_var_oracle(model, n, dist, track_order=True):
    """
    Oracle policy: lowest epistemic variance position first.
    OPTIMISED: groups samples by unique decoded context,
    computes variance once per group instead of per sample.
    At step 0 all samples share one context → 1 computation.
    At step k, at most V^k unique contexts.
    """
    all_seqs, p_true = dist.full_distribution()
    all_seqs_np = all_seqs.numpy()
    p_np = p_true.numpy()

    x = torch.full((n, L), MASK_ID, dtype=torch.long, device=DEVICE)
    unmasked = torch.zeros(n, L, dtype=torch.bool, device=DEVICE)
    scores = torch.zeros(n, device=DEVICE)
    orders = torch.zeros(n, L, dtype=torch.long, device=DEVICE) if track_order else None

    # Track decoded values per sample
    decoded = np.full((n, L), -1, dtype=np.int64)

    def _oracle_best_pos(obs_mask, obs_vals):
        """Compute lowest-variance position for a single context."""
        masked_pos = np.where(~obs_mask)[0]
        if len(masked_pos) <= 1:
            return int(masked_pos[0]) if len(masked_pos) == 1 else 0

        # Filter sequences consistent with observed values
        match = np.ones(len(p_np), dtype=bool)
        for j in np.where(obs_mask)[0]:
            match &= (all_seqs_np[:, j] == obs_vals[j])
        p_ctx = p_np[match]
        if p_ctx.sum() < 1e-20:
            return int(masked_pos[0])
        seqs_ctx = all_seqs_np[match]
        p_ctx_norm = p_ctx / p_ctx.sum()

        # Marginals
        marginals = {}
        for t in masked_pos:
            m = np.zeros(V)
            for v in range(V):
                m[v] = p_ctx_norm[seqs_ctx[:, t] == v].sum()
            marginals[t] = m

        # Epistemic variance
        best_t, best_var = masked_pos[0], float('inf')
        for t in masked_pos:
            total_var = 0.0
            n_other = 0
            for j in masked_pos:
                if j == t:
                    continue
                for w in range(V):
                    cond_mask = seqs_ctx[:, j] == w
                    p_jw = p_ctx_norm[cond_mask].sum()
                    if p_jw < 1e-15:
                        continue
                    cond_t = np.zeros(V)
                    for v in range(V):
                        cond_t[v] = p_ctx_norm[
                            cond_mask & (seqs_ctx[:, t] == v)].sum()
                    cond_t /= p_jw
                    total_var += p_jw * ((cond_t - marginals[t]) ** 2).sum()
                n_other += 1
            evar = total_var / max(n_other, 1)
            if evar < best_var:
                best_var = evar
                best_t = t
        return int(best_t)

    for step in range(L):
        logits = model(x)
        logits[:, :, MASK_ID] = -float('inf')

        # Group samples by unique context (decoded positions + values)
        # Key MUST include WHICH positions are decoded AND their values,
        # otherwise samples with different positions but same values collide.
        context_groups = {}  # key → list of sample indices
        for b in range(n):
            obs_mask = decoded[b] >= 0
            key = tuple((j, int(decoded[b, j]))
                        for j in range(L) if obs_mask[j])
            if key not in context_groups:
                context_groups[key] = []
            context_groups[key].append(b)

        # Compute oracle order ONCE per unique context
        best_pos = np.zeros(n, dtype=np.int64)
        for key, indices in context_groups.items():
            rep = indices[0]  # representative sample
            obs_mask = decoded[rep] >= 0
            bp = _oracle_best_pos(obs_mask, decoded[rep])
            # Safety: ensure bp is actually a still-masked position
            if obs_mask[bp]:
                # Fallback: pick first still-masked position
                remaining = np.where(~obs_mask)[0]
                bp = int(remaining[0]) if len(remaining) > 0 else 0
            for b in indices:
                best_pos[b] = bp

        # Safety: if oracle picked an already-decoded position for
        # any sample (shouldn't happen with correct key), fallback
        for b in range(n):
            if decoded[b, best_pos[b]] >= 0:
                candidates = np.where(decoded[b] < 0)[0]
                if len(candidates) > 0:
                    best_pos[b] = candidates[0]

        pos = torch.tensor(best_pos, dtype=torch.long, device=DEVICE)
        if track_order:
            orders[:, step] = pos

        lp = logits[torch.arange(n, device=DEVICE), pos]
        tok = torch.multinomial(F.softmax(lp, dim=-1), 1).squeeze(-1)
        scores += F.log_softmax(lp, dim=-1)[
            torch.arange(n, device=DEVICE), tok]
        x[torch.arange(n, device=DEVICE), pos] = tok
        unmasked[torch.arange(n, device=DEVICE), pos] = True
        decoded[np.arange(n), best_pos] = tok.cpu().numpy()

    return _sanitize_decode_output(x, 'low_var_oracle'), scores.cpu(), (orders.cpu() if track_order else None)


@torch.no_grad()
def decode_low_var_model(model, n, track_order=True):
    """
    Model-based epistemic variance policy.
    OPTIMISED: uses small sub-batch (32) for perturbation variance
    estimation, then applies determined order to full batch.
    """
    N_PERT = min(n, 32)  # only need small batch for variance estimation

    x = torch.full((n, L), MASK_ID, dtype=torch.long, device=DEVICE)
    unmasked = torch.zeros(n, L, dtype=torch.bool, device=DEVICE)
    scores = torch.zeros(n, device=DEVICE)
    orders = torch.zeros(n, L, dtype=torch.long, device=DEVICE) if track_order else None

    for step in range(L):
        logits_base = model(x)
        logits_base[:, :, MASK_ID] = -float('inf')
        probs_base = F.softmax(logits_base, dim=-1)

        # Use sample 0's mask to determine positions (batch shares order)
        b0_unmasked = unmasked[0]
        masked_pos = (~b0_unmasked).nonzero(as_tuple=True)[0].cpu().tolist()

        if len(masked_pos) <= 1:
            pos = torch.tensor(masked_pos[0] if masked_pos else 0,
                               device=DEVICE).expand(n)
        else:
            # Small sub-batch for perturbation variance estimation
            x_small = x[:N_PERT]
            probs_small = probs_base[:N_PERT]

            evar = torch.zeros(len(masked_pos), device=DEVICE)
            for i, t in enumerate(masked_pos):
                var_accum = torch.zeros(N_PERT, device=DEVICE)
                n_other = 0
                for j in masked_pos:
                    if j == t:
                        continue
                    p_j = probs_small[0, j, :V]
                    for w in range(V):
                        if p_j[w] < 1e-6:
                            continue
                        x_pert = x_small.clone()
                        x_pert[:, j] = w
                        logits_pert = model(x_pert)
                        logits_pert[:, :, MASK_ID] = -float('inf')
                        probs_pert = F.softmax(logits_pert, dim=-1)
                        diff = probs_pert[:, t, :V] - probs_small[:, t, :V]
                        var_accum += p_j[w] * (diff ** 2).sum(-1)
                    n_other += 1
                evar[i] = var_accum.mean() / max(n_other, 1)

            best_idx = evar.argmin().item()
            pos = torch.tensor(masked_pos[best_idx],
                               device=DEVICE).expand(n)

        if track_order:
            orders[:, step] = pos

        lp = logits_base[torch.arange(n, device=DEVICE), pos]
        tok = torch.multinomial(F.softmax(lp, dim=-1), 1).squeeze(-1)
        scores += F.log_softmax(lp, dim=-1)[torch.arange(n, device=DEVICE), tok]
        x[torch.arange(n, device=DEVICE), pos] = tok
        unmasked[torch.arange(n, device=DEVICE), pos] = True

    return _sanitize_decode_output(x, 'low_var_model'), scores.cpu(), (orders.cpu() if track_order else None)


@torch.no_grad()
def decode_adaptive_threshold(model, n, tau=0.7):
    """
    Adaptive parallel: at each step, decode ALL positions whose max
    softmax probability ≥ τ.  If none qualify, fall back to the single
    most confident position (so it always makes progress).
    """
    x = torch.full((n, L), MASK_ID, dtype=torch.long, device=DEVICE)
    unmasked = torch.zeros(n, L, dtype=torch.bool, device=DEVICE)
    scores = torch.zeros(n, device=DEVICE)
    n_steps = 0
    arange_n = torch.arange(n, device=DEVICE)
    arange_L = torch.arange(L, device=DEVICE)

    while not unmasked.all():
        logits = model(x)
        logits[:, :, MASK_ID] = -float('inf')
        probs = F.softmax(logits, dim=-1)          # (n, L, V) — computed ONCE
        conf = probs.max(-1).values                 # (n, L)
        conf[unmasked] = -1

        # Which positions are above threshold?
        above_tau = (conf >= tau) & (~unmasked)
        # Fallback: samples with no position above tau → pick most confident
        has_any = above_tau.any(dim=1)
        if not has_any.all():
            best_pos = conf.argmax(dim=1)
            no_above = ~has_any
            above_tau[arange_n[no_above], best_pos[no_above]] = True

        # Sample tokens ONLY for selected positions (via masked multinomial)
        # Flatten selected positions, sample, scatter back
        sel_idx = above_tau.nonzero(as_tuple=False)          # (M, 2) — [sample, pos]
        if sel_idx.shape[0] == 0:
            break
        sel_logits = logits[sel_idx[:, 0], sel_idx[:, 1]]    # (M, V)
        sel_probs = F.softmax(sel_logits, dim=-1)             # reuse would need gather
        sel_tok = torch.multinomial(sel_probs, 1).squeeze(-1) # (M,)
        sel_lp = F.log_softmax(sel_logits, dim=-1)[
            torch.arange(sel_idx.shape[0], device=DEVICE), sel_tok]  # (M,)

        # Scatter back
        x[sel_idx[:, 0], sel_idx[:, 1]] = sel_tok
        unmasked[sel_idx[:, 0], sel_idx[:, 1]] = True
        scores.scatter_add_(0, sel_idx[:, 0], sel_lp)
        n_steps += 1

    return _sanitize_decode_output(x, 'parallel'), scores.cpu(), n_steps


@torch.no_grad()
def decode_jacobi(model, n, max_iter=20):
    """
    Jacobi-style iteration: predict ALL positions at once,
    then re-predict with updated context, repeat until convergence
    (no tokens change) or max_iter reached.
    """
    # Initialize: predict all positions from all-MASK input
    x = torch.full((n, L), MASK_ID, dtype=torch.long, device=DEVICE)
    logits = model(x)
    logits[:, :, MASK_ID] = -float('inf')
    # Vectorized sampling: reshape (n, L, V) → (n*L, V), sample, reshape back
    flat_probs = F.softmax(logits.view(-1, logits.size(-1)), dim=-1)
    x = torch.multinomial(flat_probs, 1).squeeze(-1).view(n, L)

    n_steps = 1
    active = torch.ones(n, dtype=torch.bool, device=DEVICE)  # per-sample convergence

    for it in range(max_iter - 1):
        x_prev = x.clone()

        # Only compute for active (non-converged) samples
        if not active.any():
            break
        active_idx = active.nonzero(as_tuple=True)[0]
        x_active = x[active_idx]

        logits_a = model(x_active)
        logits_a[:, :, MASK_ID] = -float('inf')
        flat_probs_a = F.softmax(logits_a.view(-1, logits_a.size(-1)), dim=-1)
        new_tok = torch.multinomial(flat_probs_a, 1).squeeze(-1).view(-1, L)
        x[active_idx] = new_tok

        n_steps += 1
        # Mark converged samples as inactive
        converged = (x[active_idx] == x_prev[active_idx]).all(dim=1)
        active[active_idx[converged]] = False

    # Compute final scores (all samples)
    logits = model(x)
    logits[:, :, MASK_ID] = -float('inf')
    lp = F.log_softmax(logits, dim=-1)  # (n, L, V)
    scores = lp[
        torch.arange(n, device=DEVICE).unsqueeze(1).expand(n, L),
        torch.arange(L, device=DEVICE).unsqueeze(0).expand(n, L),
        x
    ].sum(dim=1)  # (n,)

    return _sanitize_decode_output(x, 'parallel'), scores.cpu(), n_steps


# ── Metrics ─────────────────────────────────────────

def _seq_to_idx(seqs):
    """
    Encode sequences of shape (n, L) with values in [0, V) into
    unique integer indices in [0, V^L).  Fully vectorized.
    """
    if isinstance(seqs, torch.Tensor):
        seqs = seqs.numpy() if seqs.device.type == 'cpu' else seqs.cpu().numpy()
    # powers[i] = V^i
    powers = V ** np.arange(L)
    return (seqs * powers).sum(axis=1).astype(np.int64)


def compute_metrics(samples, scores, dist):
    """
    Compute distributional metrics: TV, forward KL, reverse KL, JS,
    mode coverage, path score correlation.

    Metrics:
      tv:         ½ Σ|p - q|          (symmetric, bounded [0,1])
      kl_forward: KL(p ‖ q) = Σ p log(p/q)   (mode-seeking; ∞ if q=0 where p>0)
      kl_reverse: KL(q ‖ p) = Σ q log(q/p)   (mean-seeking; penalises q-mass off p)
      js:         ½ KL(p‖m) + ½ KL(q‖m), m=(p+q)/2  (symmetric, bounded)

    Per-sample: true log-probability for each generated sample,
    enabling direct comparison with model path scores.
    """
    all_seqs, p_true = dist.full_distribution()
    K = len(all_seqs)
    n = len(samples)
    p_np = p_true.numpy()

    # Safety: clamp any leaked MASK_ID tokens to 0
    if (samples >= V).any():
        n_bad = (samples >= V).sum().item()
        print(f"    WARNING: {n_bad} MASK_ID tokens in samples, clamping")
        samples = samples.clamp(max=V-1)

    all_idx = _seq_to_idx(all_seqs)
    sample_idx = _seq_to_idx(samples)

    # Empirical distribution
    raw_counts = np.bincount(sample_idx, minlength=V**L)
    q_emp = raw_counts[all_idx] / n

    # ── TV ──
    tv = np.sum(np.abs(p_np - q_emp)) / 2

    # ── Forward KL: KL(p ‖ q) ──
    # Capped: where q=0 and p>0, use log(p/eps)
    eps = 1e-10
    q_safe = np.maximum(q_emp, eps)
    mask_p = p_np > eps
    kl_forward = np.sum(p_np[mask_p] * np.log(p_np[mask_p] / q_safe[mask_p]))

    # ── Reverse KL: KL(q ‖ p) ──
    p_safe = np.maximum(p_np, eps)
    mask_q = q_emp > eps
    kl_reverse = np.sum(q_emp[mask_q] * np.log(q_emp[mask_q] / p_safe[mask_q]))

    # ── Jensen-Shannon divergence ──
    m = (p_np + q_emp) / 2
    m_safe = np.maximum(m, eps)
    js = 0.5 * np.sum(p_np[mask_p] * np.log(p_np[mask_p] / m_safe[mask_p])) \
       + 0.5 * np.sum(q_emp[mask_q] * np.log(q_emp[mask_q] / m_safe[mask_q]))

    # ── Mode coverage (top 100) ──
    top_idx = p_true.topk(min(100, K)).indices.numpy()
    coverage = np.mean([q_emp[i] > 0 for i in top_idx])

    # ── Per-sample true log probability ──
    true_lp = dist.log_prob(samples).numpy()
    scores_np = scores.numpy()
    valid = np.isfinite(scores_np) & np.isfinite(true_lp)
    sr = spearmanr(scores_np[valid], true_lp[valid])[0] \
        if valid.sum() > 10 else 0

    # ── Bootstrap CI for KL forward ──
    kl_boots = np.empty(BOOTSTRAP_K)
    for b in range(BOOTSTRAP_K):
        idx = np.random.choice(n, size=n, replace=True)
        boot_counts = np.bincount(sample_idx[idx], minlength=V**L)
        q_boot = boot_counts[all_idx] / n
        q_boot_safe = np.maximum(q_boot, eps)
        kl_boots[b] = np.sum(
            p_np[mask_p] * np.log(p_np[mask_p] / q_boot_safe[mask_p]))

    # ── Ideal baseline ──
    ideal_kls = np.empty(5)
    ideal_tvs = np.empty(5)
    for t in range(5):
        ideal_samples = dist.sample(n)
        ideal_idx = _seq_to_idx(ideal_samples)
        ideal_counts = np.bincount(ideal_idx, minlength=V**L)
        q_ideal = ideal_counts[all_idx] / n
        ideal_tvs[t] = np.sum(np.abs(p_np - q_ideal)) / 2
        q_ideal_safe = np.maximum(q_ideal, eps)
        ideal_kls[t] = np.sum(
            p_np[mask_p] * np.log(p_np[mask_p] / q_ideal_safe[mask_p]))

    return {
        'tv': float(tv),
        'tv_baseline': float(ideal_tvs.mean()),
        'kl_forward': float(kl_forward),
        'kl_forward_ci_lo': float(np.percentile(kl_boots, 2.5)),
        'kl_forward_ci_hi': float(np.percentile(kl_boots, 97.5)),
        'kl_forward_baseline': float(ideal_kls.mean()),
        'kl_reverse': float(kl_reverse),
        'js': float(js),
        'mode_coverage': float(coverage),
        'spearman_r': float(sr),
        'mean_true_lp': float(true_lp.mean()),
        'std_true_lp': float(true_lp.std()),
        'n_unique': int(len(np.unique(sample_idx))),
        'support_coverage': float(len(np.unique(sample_idx)) / K),
    }


# ── Mutual information matrix ──────────────────────

def compute_mi(dist):
    all_s, probs = dist.full_distribution()
    MI = np.zeros((L, L))
    marginals = []
    for i in range(L):
        m = np.zeros(V)
        for v in range(V):
            m[v] = probs[all_s[:, i] == v].sum().item()
        marginals.append(m)
    for i in range(L):
        for j in range(i+1, L):
            joint = np.zeros((V, V))
            for vi in range(V):
                for vj in range(V):
                    joint[vi, vj] = probs[
                        (all_s[:,i]==vi) & (all_s[:,j]==vj)].sum().item()
            mi = sum(joint[vi,vj] * np.log(
                    joint[vi,vj]/(marginals[i][vi]*marginals[j][vj]+1e-30)+1e-30)
                for vi in range(V) for vj in range(V) if joint[vi,vj]>1e-30)
            MI[i,j] = MI[j,i] = mi
    return MI


def compute_mean_decode_order(orders_list):
    """
    From decode orders (n_samples × L), compute mean rank for each position.
    orders[i, t] = position decoded at step t for sample i.
    Returns: array of shape (L,) where entry j = mean step at which position j was decoded.
    Fully vectorized using scatter.
    """
    all_orders = torch.cat(orders_list, dim=0)  # (N, L)
    N = all_orders.shape[0]
    # Convert from "step→position" to "position→step"
    # pos_ranks[b, pos] = step  where all_orders[b, step] = pos
    pos_ranks = torch.zeros(N, L)
    steps = torch.arange(L).unsqueeze(0).expand(N, -1).float()  # (N, L)
    pos_ranks.scatter_(1, all_orders.long(), steps)
    return pos_ranks.mean(dim=0).numpy()  # (L,)


def compute_mi_order_alignment(mean_ranks, MI):
    """
    Spearman correlation between "total MI of position" and "decode rank".
    High-MI positions should ideally be decoded first (lower rank).
    Returns negative correlation = MI-aligned (good).
    """
    total_mi = MI.sum(axis=1)  # total MI with all other positions
    rho, p = spearmanr(total_mi, mean_ranks)
    return {'rho': float(rho), 'p_value': float(p),
            'total_mi': total_mi.tolist(), 'mean_ranks': mean_ranks.tolist()}


def compute_detailed_analysis(samples, dist):
    """
    Per-sample and per-position analysis from the SAME samples
    used for aggregate metrics (no separate generation).

    Args:
        samples: (N, L) tensor — the same 500K decoded sequences
        dist: ToyDistribution — ground truth

    Returns dict with:
        per_position_marginal_kl:  KL(p_true_marginal ‖ q_empirical_marginal) per pos
        per_position_mode_acc:     fraction where empirical mode matches true mode
        true_lp_histogram:         binned distribution of log p_true(x) for samples
        true_lp_percentiles:       [5, 25, 50, 75, 95] percentiles
        context_analysis:          (for Markov dists) per-context accuracy
    """
    all_seqs, p_true = dist.full_distribution()
    p_np = p_true.numpy()
    if isinstance(samples, torch.Tensor):
        S = samples.numpy()
    else:
        S = np.array(samples)
    N = len(S)
    eps = 1e-10

    # ── 1. Per-position marginal KL ──
    # True marginals from full distribution
    true_marginals = []
    for i in range(L):
        m = np.zeros(V)
        for v in range(V):
            m[v] = p_np[all_seqs[:, i].numpy() == v].sum()
        true_marginals.append(m)

    # Empirical marginals from samples
    emp_marginals = []
    for i in range(L):
        m = np.bincount(S[:, i], minlength=V).astype(float) / N
        emp_marginals.append(m)

    pos_marginal_kl = []
    for i in range(L):
        p_m = true_marginals[i]
        q_m = np.maximum(emp_marginals[i], eps)
        kl = np.sum(p_m * np.log(p_m / q_m) * (p_m > eps))
        pos_marginal_kl.append(float(kl))

    # ── 2. Per-position mode accuracy ──
    # Does the most common token at each position match the true mode?
    pos_mode_acc = []
    for i in range(L):
        true_mode = int(np.argmax(true_marginals[i]))
        emp_mode = int(np.argmax(emp_marginals[i]))
        pos_mode_acc.append(true_mode == emp_mode)

    # ── 3. True log-probability distribution of generated samples ──
    true_lp = dist.log_prob(torch.tensor(S)).numpy()
    percentiles = np.percentile(true_lp, [5, 25, 50, 75, 95]).tolist()

    # Histogram for plotting
    hist_counts, hist_edges = np.histogram(true_lp, bins=50)
    hist_centers = ((hist_edges[:-1] + hist_edges[1:]) / 2).tolist()
    hist_counts = hist_counts.tolist()

    # Compare with true expected log-prob: E_{x~p}[log p(x)]
    all_lp = dist.log_prob(all_seqs).numpy()
    true_expected_lp = float((p_np * all_lp).sum())

    result = {
        'per_position_marginal_kl': pos_marginal_kl,
        'per_position_mode_correct': pos_mode_acc,
        'true_marginals': [m.tolist() for m in true_marginals],
        'emp_marginals': [m.tolist() for m in emp_marginals],
        'true_lp_percentiles': percentiles,
        'true_lp_mean': float(true_lp.mean()),
        'true_lp_std': float(true_lp.std()),
        'true_expected_lp': true_expected_lp,
        'lp_gap': float(true_lp.mean() - true_expected_lp),
        'hist_centers': hist_centers,
        'hist_counts': hist_counts,
    }

    # ── 4. Context-conditional analysis for Markov distributions ──
    if isinstance(dist, (MarkovChain, MarkovPlusGlobal, HardMarkov)):
        markov = dist if isinstance(dist, (MarkovChain, HardMarkov)) \
            else dist.markov
        order = markov.k

        # For each context, compute empirical vs true conditional
        n_ctx = V ** order
        ctx_analysis = {}
        ctx_counts = np.zeros((n_ctx, V))  # counts of x_t given context
        ctx_total = np.zeros(n_ctx)

        for t in range(order, L):
            # Compute context index for all samples
            ctx_idx = np.zeros(N, dtype=np.int64)
            for o in range(order):
                ctx_idx += S[:, t - order + o] * (V ** o)
            for c in range(n_ctx):
                mask = ctx_idx == c
                if mask.any():
                    for v in range(V):
                        ctx_counts[c, v] += (S[mask, t] == v).sum()
                    ctx_total[c] += mask.sum()

        # Per-context KL and accuracy
        ctx_kl = []
        ctx_acc = []
        ctx_details = []
        for c in range(n_ctx):
            if ctx_total[c] < 10:
                continue
            emp_cond = ctx_counts[c] / max(ctx_total[c], 1)
            true_cond = np.exp(markov.trans_logp[c].numpy())
            kl = float(np.sum(
                true_cond * np.log(true_cond / np.maximum(emp_cond, eps))
                * (true_cond > eps)))
            true_mode = int(np.argmax(true_cond))
            emp_mode = int(np.argmax(emp_cond))
            acc = float(emp_cond[true_mode])

            ctx_kl.append(kl)
            ctx_acc.append(acc)
            ctx_details.append({
                'ctx': c, 'kl': kl,
                'true_mode': true_mode, 'emp_mode': emp_mode,
                'true_mode_prob': float(true_cond[true_mode]),
                'emp_mode_prob': acc,
                'n_obs': int(ctx_total[c]),
            })

        # Sort by KL to find worst contexts
        ctx_details.sort(key=lambda x: -x['kl'])

        result['context_analysis'] = {
            'mean_ctx_kl': float(np.mean(ctx_kl)) if ctx_kl else 0,
            'max_ctx_kl': float(np.max(ctx_kl)) if ctx_kl else 0,
            'mean_ctx_mode_acc': float(np.mean(ctx_acc)) if ctx_acc else 0,
            'n_contexts_observed': len(ctx_kl),
            'worst_contexts': ctx_details[:10],
            'best_contexts': ctx_details[-5:] if len(ctx_details) > 5
                else [],
        }

        # HardMarkov-specific: deterministic map compliance
        if isinstance(dist, HardMarkov):
            compliant = 0
            total = 0
            per_pos_compliance = []
            for t in range(order, L):
                ctx_idx = np.zeros(N, dtype=np.int64)
                for o in range(order):
                    ctx_idx += S[:, t - order + o] * (V ** o)
                preferred = dist.preferred[ctx_idx]
                match = (S[:, t] == preferred)
                per_pos_compliance.append(float(match.mean()))
                compliant += match.sum()
                total += N

            result['hard_markov'] = {
                'overall_compliance': float(compliant / max(total, 1)),
                'per_position_compliance': per_pos_compliance,
                'expected_compliance': dist.p_peak,
            }

    return result


@torch.no_grad()
def compute_initial_confidence_profile(model):
    """
    Forward pass with ALL positions masked → model's "prior" confidence.

    Returns per-position: max_prob, entropy, predicted_mode.
    This reveals what confidence policy "sees" at step 0:
    if a position has peaked marginal → high confidence even without context.
    """
    x = torch.full((1, L), MASK_ID, dtype=torch.long, device=DEVICE)
    logits = model(x)
    logits[:, :, MASK_ID] = -float('inf')
    probs = F.softmax(logits, dim=-1).squeeze(0)   # (L, V)

    max_prob = probs.max(-1).values.cpu().numpy()   # confidence score
    entropy = -(probs * (probs + 1e-10).log()).sum(-1).cpu().numpy()
    mode = probs.argmax(-1).cpu().numpy()

    # Rank: which position would confidence decode first?
    confidence_rank = np.argsort(-max_prob)  # positions sorted by confidence
    return {
        'max_prob': max_prob.tolist(),
        'entropy': entropy.tolist(),
        'mode': mode.tolist(),
        'confidence_order': confidence_rank.tolist(),  # position decoded 1st, 2nd, ...
    }


def compute_true_conditional_entropy_profile(dist):
    """
    For each position t, compute:
    - H(X_t) = marginal entropy (decode without any context)
    - H(X_t | X_{<t}) = average conditional entropy in l2r order

    The gap H(X_t) - H(X_t | X_{<t}) measures how much context helps.
    If confidence tracks H(X_t), it may choose "peaked marginal" positions
    first, even when H(X_t | context) is much lower elsewhere.
    """
    all_seqs, p_true = dist.full_distribution()
    p_np = p_true.numpy()
    S = all_seqs.numpy()
    eps = 1e-30

    # Marginal entropy per position
    marginal_H = []
    for i in range(L):
        m = np.zeros(V)
        for v in range(V):
            m[v] = p_np[S[:, i] == v].sum()
        h = -sum(m[v] * np.log(m[v] + eps) for v in range(V) if m[v] > eps)
        marginal_H.append(float(h))

    # Conditional entropy H(X_t | X_0, ..., X_{t-1}) in l2r order
    # = E_{x_{<t}} [ H(X_t | x_{<t}) ]
    cond_H_l2r = [marginal_H[0]]  # position 0 has no context
    for t in range(1, L):
        # Group sequences by their prefix x_0..x_{t-1}
        # For each prefix, compute conditional entropy of x_t
        # Weight by prefix probability
        prefix_to_probs = {}
        for idx in range(len(S)):
            prefix = tuple(S[idx, :t])
            if prefix not in prefix_to_probs:
                prefix_to_probs[prefix] = np.zeros(V)
            prefix_to_probs[prefix][S[idx, t]] += p_np[idx]

        total_h = 0.0
        for prefix, cond_dist in prefix_to_probs.items():
            p_prefix = cond_dist.sum()
            if p_prefix < eps:
                continue
            cond_norm = cond_dist / p_prefix
            h = -sum(cond_norm[v] * np.log(cond_norm[v] + eps)
                     for v in range(V) if cond_norm[v] > eps)
            total_h += p_prefix * h
        cond_H_l2r.append(float(total_h))

    # Also compute for r2l order
    cond_H_r2l = [marginal_H[L - 1]]
    for step in range(1, L):
        t = L - 1 - step  # position being decoded
        # Context = positions t+1, t+2, ..., L-1
        ctx_positions = list(range(t + 1, L))
        prefix_to_probs = {}
        for idx in range(len(S)):
            ctx = tuple(S[idx, p] for p in ctx_positions)
            if ctx not in prefix_to_probs:
                prefix_to_probs[ctx] = np.zeros(V)
            prefix_to_probs[ctx][S[idx, t]] += p_np[idx]

        total_h = 0.0
        for ctx, cond_dist in prefix_to_probs.items():
            p_ctx = cond_dist.sum()
            if p_ctx < eps:
                continue
            cond_norm = cond_dist / p_ctx
            h = -sum(cond_norm[v] * np.log(cond_norm[v] + eps)
                     for v in range(V) if cond_norm[v] > eps)
            total_h += p_ctx * h
        cond_H_r2l.append(float(total_h))
    cond_H_r2l.reverse()  # now indexed by position

    # Context benefit: how much does l2r context reduce entropy?
    context_benefit = [marginal_H[t] - cond_H_l2r[t] for t in range(L)]

    return {
        'marginal_H': marginal_H,
        'cond_H_l2r': cond_H_l2r,
        'cond_H_r2l': cond_H_r2l,
        'context_benefit_l2r': context_benefit,
    }


def compute_stepwise_cond_entropy(dist, orders_list, samples):
    """
    For each decode policy's tracked orders, compute the TRUE conditional
    entropy at each decode step.

    Step t: position p=order[t] is being decoded.
    Context: positions {order[0], ..., order[t-1]} with their actual values.
    H_step[t] = E_samples[ H(X_p | X_{context} = x_{context}) ]

    For l2r, H_step should be low and flat (good context every step).
    For confidence, H_step may start low (peaked marginals) but spike
    at later steps where context is fragmentary.
    """
    all_seqs, p_true = dist.full_distribution()
    p_np = p_true.numpy()
    S_all = all_seqs.numpy()
    eps = 1e-30

    if not orders_list:
        return {}

    orders = torch.cat(orders_list, dim=0)     # (N_track, L)
    # Use corresponding samples (first N_track from 500K)
    N_track = orders.shape[0]
    samp = samples[:N_track].numpy() if isinstance(samples, torch.Tensor) \
        else np.array(samples[:N_track])

    step_entropies = np.zeros(L)
    step_counts = np.zeros(L)

    # For efficiency: precompute lookup table seq→index
    seq_to_idx = {}
    for idx in range(len(S_all)):
        seq_to_idx[tuple(S_all[idx])] = idx

    # Process in mini-batches for memory
    n_analysis = min(N_track, 2000)  # limit for computational cost
    for i in range(n_analysis):
        order = orders[i].numpy()
        sample = samp[i]

        for step in range(L):
            pos = order[step]
            if step == 0:
                # No context — use marginal
                marg = np.zeros(V)
                for v in range(V):
                    marg[v] = p_np[S_all[:, pos] == v].sum()
                h = -sum(marg[v] * np.log(marg[v] + eps)
                         for v in range(V) if marg[v] > eps)
            else:
                # Context: positions order[0..step-1] with sample values
                ctx_positions = order[:step]
                ctx_values = sample[ctx_positions]

                # Filter sequences matching context
                mask = np.ones(len(S_all), dtype=bool)
                for cp, cv in zip(ctx_positions, ctx_values):
                    mask &= (S_all[:, cp] == cv)

                matching_p = p_np[mask]
                if matching_p.sum() < eps:
                    h = np.log(V)  # uniform fallback
                else:
                    cond = np.zeros(V)
                    for v in range(V):
                        cond[v] = p_np[mask & (S_all[:, pos] == v)].sum()
                    cond /= cond.sum() + eps
                    h = -sum(cond[v] * np.log(cond[v] + eps)
                             for v in range(V) if cond[v] > eps)

            step_entropies[step] += h
            step_counts[step] += 1

    mean_H = (step_entropies / np.maximum(step_counts, 1)).tolist()
    return {'step_cond_entropy': mean_H, 'n_samples_analysed': n_analysis}


def compute_error_cascade(samples, dist):
    """
    From the SAME 500K samples, measure error correlation between
    adjacent positions. For Markov distributions, confidence policy
    may cause correlated errors (wrong context → wrong next token).

    Returns:
      p_err[t]: P(error at position t)
      p_err_given_prev_err[t]: P(error at t | error at t-1)
      cascade_ratio[t]: p_err_given_prev_err[t] / p_err[t]
        > 1 means errors cascade, = 1 means independent
    """
    all_seqs, p_true = dist.full_distribution()
    p_np = p_true.numpy()
    S_all = all_seqs.numpy()

    if isinstance(samples, torch.Tensor):
        S = samples.numpy()
    else:
        S = np.array(samples)
    N = len(S)

    # "Correct" = matches the mode of conditional distribution
    # For position t, the mode depends on context.
    # For tractability, use marginal mode (policy-independent baseline)
    # AND Markov conditional mode (if applicable)

    # 1. Marginal mode error
    marginal_mode = np.zeros(L, dtype=np.int64)
    for i in range(L):
        m = np.zeros(V)
        for v in range(V):
            m[v] = p_np[S_all[:, i] == v].sum()
        marginal_mode[i] = m.argmax()

    marg_err = (S != marginal_mode[None, :])  # (N, L)
    p_err_marginal = marg_err.mean(axis=0).tolist()

    result = {'p_err_marginal': p_err_marginal}

    # 2. Conditional mode error (for Markov distributions)
    if isinstance(dist, (MarkovChain, MarkovPlusGlobal, HardMarkov)):
        markov = dist if isinstance(dist, (MarkovChain, HardMarkov)) \
            else dist.markov
        order = markov.k
        trans = np.exp(markov.trans_logp.numpy())  # (n_ctx, V)

        # For each position t >= order, check if sample matches
        # conditional mode given actual context
        cond_err = np.zeros((N, L), dtype=bool)
        # First `order` positions: use marginal mode
        for t in range(order):
            cond_err[:, t] = S[:, t] != marginal_mode[t]

        for t in range(order, L):
            ctx_idx = np.zeros(N, dtype=np.int64)
            for o in range(order):
                ctx_idx += S[:, t - order + o] * (V ** o)
            cond_mode = trans[ctx_idx].argmax(axis=1)
            cond_err[:, t] = S[:, t] != cond_mode

        p_err_cond = cond_err.mean(axis=0).tolist()

        # Cascade: P(err at t | err at t-1)
        cascade_ratio = []
        p_err_given_prev = []
        for t in range(1, L):
            prev_err_mask = cond_err[:, t - 1]
            n_prev_err = prev_err_mask.sum()
            if n_prev_err > 0:
                p_e_given = cond_err[prev_err_mask, t].mean()
            else:
                p_e_given = 0.0
            p_e = cond_err[:, t].mean()
            p_err_given_prev.append(float(p_e_given))
            ratio = float(p_e_given / (p_e + 1e-10))
            cascade_ratio.append(ratio)

        result['p_err_conditional'] = p_err_cond
        result['p_err_given_prev_err'] = p_err_given_prev
        result['cascade_ratio'] = cascade_ratio

    return result

def run():
    print("=" * 70)
    print("  EXP 3: Toy Distribution — Decoding Policy Analysis")
    print("=" * 70)
    mount_drive()
    torch.manual_seed(42); np.random.seed(42)

    distributions = {
        'A_Independent':   NearIndependent(alpha=0.5, seed=42),
        'B_Markov2':       MarkovChain(order=2, sparsity=0.1, seed=42),
        'C_GlobalSum':     GlobalSumConstraint(beta=2.0, alpha=0.5, seed=42),
        'D_Markov+Global': MarkovPlusGlobal(beta=1.5, order=2, sparsity=0.1, seed=42),
        'E_HardMarkov':    HardMarkov(order=2, p_peak=0.97, seed=42),
    }
    for name, dist in distributions.items():
        h = dist.entropy()
        extra = ''
        if hasattr(dist, 'theoretical_l2r_entropy'):
            h_l2r = dist.theoretical_l2r_entropy()
            p_acc = dist.expected_accuracy_l2r()
            extra = f"  H_l2r/step={h_l2r:.3f}bits  greedy_acc={p_acc:.2%}"
        print(f"  {name}: H={h:.2f} bits{extra}")

    print(f"\n  Sampling stats: V^L={V**L:,}, N_SAMPLES={N_SAMPLES:,}, "
          f"ratio={N_SAMPLES/V**L:.1f}x")
    print(f"  Expected TV noise ≈ {np.sqrt(V**L / (2*np.pi*N_SAMPLES)):.4f}")
    print(f"  Bootstrap resamples: {BOOTSTRAP_K}")

    # Compute theoretical optimal loss for each distribution
    print("\n  Computing Bayes-optimal masked loss (may take ~1min)...")
    for name, dist in distributions.items():
        opt_loss = dist.optimal_masked_loss(n_eval=2000)
        uniform_loss = np.log(V)
        print(f"    {name}: optimal_CE={opt_loss:.4f}, "
              f"uniform_CE={uniform_loss:.4f}, "
              f"gap={uniform_loss - opt_loss:.4f}")

    # MI matrices
    fig_mi, axes = plt.subplots(1, len(distributions), figsize=(5 * len(distributions), 4))
    for idx, (name, dist) in enumerate(distributions.items()):
        MI = compute_mi(dist)
        im = axes[idx].imshow(MI, cmap='YlOrRd', vmin=0)
        axes[idx].set_title(name, fontsize=10)
        plt.colorbar(im, ax=axes[idx], shrink=0.8)
    fig_mi.suptitle('Mutual Information I(X_i; X_j)', fontsize=13, y=1.02)
    fig_mi.tight_layout()

    # Train
    models, train_hists = {}, {}
    for name, dist in distributions.items():
        print(f"\n▶ Training: {name}")
        model, hist = train_toy_model(dist)
        models[name] = model
        train_hists[name] = hist

    # ── Confidence diagnostic: what does the model "see" at step 0? ──
    print("\n  Computing confidence diagnostics & true conditional entropy...")
    confidence_profiles = {}
    cond_entropy_profiles = {}
    for name, dist in distributions.items():
        model = models[name]
        confidence_profiles[name] = compute_initial_confidence_profile(model)
        cond_entropy_profiles[name] = compute_true_conditional_entropy_profile(dist)

        cp = confidence_profiles[name]
        ce = cond_entropy_profiles[name]
        print(f"    {name}:")
        print(f"      model conf order: {cp['confidence_order']}")
        print(f"      marginal H:       "
              + ' '.join(f"p{i}={h:.3f}" for i, h in enumerate(ce['marginal_H'])))
        print(f"      cond H (l2r):     "
              + ' '.join(f"p{i}={h:.3f}" for i, h in enumerate(ce['cond_H_l2r'])))
        print(f"      context benefit:  "
              + ' '.join(f"p{i}={b:.3f}" for i, b in enumerate(ce['context_benefit_l2r'])))

    # Evaluate policies (sampling only — greedy is deterministic, TV/KL meaningless)
    seq_policies = ['confidence', 'low_entropy', 'high_entropy', 'margin',
                    'random', 'l2r', 'r2l',
                    'entropy_adj', 'single_perturb']
    # New parallel configs: adaptive threshold + jacobi iteration
    adaptive_taus = [0.5, 0.7, 0.9]
    jacobi_iters  = [5, 10, 20]

    all_results = {}
    all_orders = {}  # dist_name → policy → list of order tensors
    all_detailed = {}  # dist_name → policy → detailed analysis
    saved_samples_for_stepwise = {}  # dist → pol → first N_ORDER samples

    # Policies to run detailed per-sample analysis on (same 500K samples)
    DETAILED_POLICIES = {'confidence', 'l2r', 'r2l', 'random',
                         'entropy_adj', 'single_perturb'}

    for dist_name, dist in distributions.items():
        model = models[dist_name]
        print(f"\n▶ Evaluating: {dist_name}")
        all_results[dist_name] = {}
        all_orders[dist_name] = {}
        all_detailed[dist_name] = {}
        saved_samples_for_stepwise[dist_name] = {}

        # ── Sequential policies ──
        for pol in seq_policies:
            t0 = time.time()
            samples, scores, orders_list = [], [], []
            n_generated = 0
            for start in range(0, N_SAMPLES, BATCH_SIZE):
                bs = min(BATCH_SIZE, N_SAMPLES - start)
                need_order = (n_generated < N_ORDER_SAMPLES)
                s, sc, ords = decode_sequential(model, bs, pol,
                                                track_order=need_order)
                samples.append(s); scores.append(sc)
                if ords is not None:
                    orders_list.append(ords)
                n_generated += bs
            samples = torch.cat(samples); scores = torch.cat(scores)
            m = compute_metrics(samples, scores, dist)
            m['nfe'] = 2*L if pol == 'single_perturb' else L
            all_results[dist_name][pol] = m
            all_orders[dist_name][pol] = orders_list

            # Detailed analysis on the SAME samples
            if pol in DETAILED_POLICIES:
                det = compute_detailed_analysis(samples, dist)
                # Error cascade from SAME samples
                det['error_cascade'] = compute_error_cascade(samples, dist)
                all_detailed[dist_name][pol] = det

                # Save subset for stepwise conditional entropy
                if orders_list:
                    n_tracked = sum(o.shape[0] for o in orders_list)
                    saved_samples_for_stepwise[dist_name][pol] = \
                        samples[:n_tracked].clone()

                # Print key details
                mkl = det['per_position_marginal_kl']
                print(f"    {pol:<20} KL_f={m['kl_forward']:.4f} "
                      f"KL_r={m['kl_reverse']:.4f} "
                      f"JS={m['js']:.4f} "
                      f"lp_gap={det['lp_gap']:+.3f}")
                print(f"      pos_marginal_KL: "
                      + ' '.join(f"p{i}={k:.4f}" for i, k in enumerate(mkl)))
                # Error cascade summary
                ec = det['error_cascade']
                if 'cascade_ratio' in ec:
                    cr = ec['cascade_ratio']
                    print(f"      cascade ratio: "
                          + ' '.join(f"t{i+1}={r:.2f}" for i, r in enumerate(cr)))
                if 'context_analysis' in det:
                    ca = det['context_analysis']
                    print(f"      ctx: mean_KL={ca['mean_ctx_kl']:.4f} "
                          f"max_KL={ca['max_ctx_kl']:.4f} "
                          f"mode_acc={ca['mean_ctx_mode_acc']:.3f}")
                if 'hard_markov' in det:
                    hm = det['hard_markov']
                    print(f"      compliance: {hm['overall_compliance']:.4f} "
                          f"(expected: {hm['expected_compliance']:.4f})")
                    ppc = hm['per_position_compliance']
                    print(f"      per_pos: "
                          + ' '.join(f"p{i+2}={c:.3f}"
                                     for i, c in enumerate(ppc)))
            else:
                print(f"    {pol:<20} KL_f={m['kl_forward']:.4f} "
                      f"KL_r={m['kl_reverse']:.4f} "
                      f"JS={m['js']:.4f} "
                      f"cov={m['support_coverage']:.2%} "
                      f"({time.time()-t0:.1f}s)")

        # ── Adaptive threshold ──
        for tau in adaptive_taus:
            key = f"adaptive_τ{tau}"
            t0 = time.time()
            samples, scores, total_steps = [], [], 0
            n_batches = 0
            for start in range(0, N_SAMPLES, BATCH_SIZE):
                bs = min(BATCH_SIZE, N_SAMPLES - start)
                s, sc, ns = decode_adaptive_threshold(model, bs, tau=tau)
                samples.append(s); scores.append(sc)
                total_steps += ns; n_batches += 1
            samples = torch.cat(samples); scores = torch.cat(scores)
            m = compute_metrics(samples, scores, dist)
            m['nfe'] = total_steps / max(n_batches, 1)
            all_results[dist_name][key] = m
            print(f"    {key:<20} KL_f={m['kl_forward']:.4f} "
                  f"KL_r={m['kl_reverse']:.4f} "
                  f"NFE={m['nfe']:.1f} ({time.time()-t0:.1f}s)")

        # ── Jacobi iteration ──
        for mi in jacobi_iters:
            key = f"jacobi_i{mi}"
            t0 = time.time()
            samples, scores, total_steps = [], [], 0
            n_batches = 0
            for start in range(0, N_SAMPLES, BATCH_SIZE):
                bs = min(BATCH_SIZE, N_SAMPLES - start)
                s, sc, ns = decode_jacobi(model, bs, max_iter=mi)
                samples.append(s); scores.append(sc)
                total_steps += ns; n_batches += 1
            samples = torch.cat(samples); scores = torch.cat(scores)
            m = compute_metrics(samples, scores, dist)
            m['nfe'] = total_steps / max(n_batches, 1)
            all_results[dist_name][key] = m
            print(f"    {key:<20} KL_f={m['kl_forward']:.4f} "
                  f"KL_r={m['kl_reverse']:.4f} "
                  f"NFE={m['nfe']:.1f} ({time.time()-t0:.1f}s)")

        # ── Low-variance policies (epistemic uncertainty) ──
        # Oracle: uses ground truth distribution for exact variance
        print(f"\n    -- Epistemic Variance Policies --")
        # Use fewer samples for oracle (expensive per-sample computation)
        N_VAR_SAMPLES = min(N_SAMPLES, 2_000)  # reduced from 10K — oracle+model too slow
        t0 = time.time()
        samples, scores, orders_list = [], [], []
        for start in range(0, N_VAR_SAMPLES, BATCH_SIZE):
            bs = min(BATCH_SIZE, N_VAR_SAMPLES - start)
            s, sc, ords = decode_low_var_oracle(model, bs, dist,
                                                 track_order=True)
            samples.append(s); scores.append(sc)
            if ords is not None:
                orders_list.append(ords)
        samples = torch.cat(samples); scores = torch.cat(scores)
        m = compute_metrics(samples, scores, dist)
        m['nfe'] = L
        key = 'low_var_oracle'
        all_results[dist_name][key] = m
        all_orders[dist_name][key] = orders_list
        # Save samples for stepwise analysis
        saved_samples_for_stepwise[dist_name][key] = samples.clone()
        # Decode order analysis
        if orders_list:
            mean_ranks = compute_mean_decode_order(orders_list)
            order_str = ' '.join(f"p{i}={r:.1f}" for i, r in enumerate(mean_ranks))
        else:
            order_str = 'N/A'
        print(f"    {key:<20} KL_f={m['kl_forward']:.4f} "
              f"KL_r={m['kl_reverse']:.4f} "
              f"JS={m['js']:.4f} ({time.time()-t0:.1f}s)")
        print(f"      decode order: {order_str}")

        # Model-based: uses perturbation forward passes
        t0 = time.time()
        samples, scores, orders_list = [], [], []
        for start in range(0, N_VAR_SAMPLES, BATCH_SIZE):
            bs = min(BATCH_SIZE, N_VAR_SAMPLES - start)
            s, sc, ords = decode_low_var_model(model, bs,
                                                track_order=True)
            samples.append(s); scores.append(sc)
            if ords is not None:
                orders_list.append(ords)
        samples = torch.cat(samples); scores = torch.cat(scores)
        m = compute_metrics(samples, scores, dist)
        m['nfe'] = L
        key = 'low_var_model'
        all_results[dist_name][key] = m
        all_orders[dist_name][key] = orders_list
        saved_samples_for_stepwise[dist_name][key] = samples.clone()
        if orders_list:
            mean_ranks = compute_mean_decode_order(orders_list)
            order_str = ' '.join(f"p{i}={r:.1f}" for i, r in enumerate(mean_ranks))
        else:
            order_str = 'N/A'
        print(f"    {key:<20} KL_f={m['kl_forward']:.4f} "
              f"KL_r={m['kl_reverse']:.4f} "
              f"JS={m['js']:.4f} ({time.time()-t0:.1f}s)")
        print(f"      decode order: {order_str}")

    # ── Stepwise conditional entropy: l2r vs confidence vs low_var ──
    print("\n  Computing stepwise conditional entropy (may take a few min)...")
    all_stepwise = {}
    STEPWISE_POLICIES = {'confidence', 'l2r', 'low_var_oracle', 'low_var_model',
                         'entropy_adj', 'single_perturb'}
    for dist_name, dist in distributions.items():
        all_stepwise[dist_name] = {}
        for pol in STEPWISE_POLICIES:
            ords = all_orders.get(dist_name, {}).get(pol, [])
            samps = saved_samples_for_stepwise.get(dist_name, {}).get(pol)
            if ords and samps is not None:
                sw = compute_stepwise_cond_entropy(dist, ords, samps)
                all_stepwise[dist_name][pol] = sw
                hh = sw['step_cond_entropy']
                print(f"    {dist_name} / {pol}: "
                      + ' '.join(f"s{i}={h:.3f}" for i, h in enumerate(hh)))
                if pol in all_detailed.get(dist_name, {}):
                    all_detailed[dist_name][pol]['stepwise_cond_entropy'] = hh
                # Store in JSON
                if pol not in all_detailed.get(dist_name, {}):
                    if dist_name not in all_detailed:
                        all_detailed[dist_name] = {}
                    all_detailed[dist_name][pol] = {}
                all_detailed[dist_name].setdefault(pol, {})
                all_detailed[dist_name][pol]['stepwise_cond_entropy'] = hh

    # ── Compute MI matrices and alignment scores ──
    mi_matrices = {}
    mi_alignments = {}
    # Include low_var policies in MI alignment
    all_ordered_policies = (seq_policies
                            + ['low_var_oracle', 'low_var_model'])
    for dist_name, dist in distributions.items():
        MI = compute_mi(dist)
        mi_matrices[dist_name] = MI
        mi_alignments[dist_name] = {}
        for pol in all_ordered_policies:
            if pol in all_orders[dist_name] and all_orders[dist_name][pol]:
                mean_ranks = compute_mean_decode_order(all_orders[dist_name][pol])
                alignment = compute_mi_order_alignment(mean_ranks, MI)
                mi_alignments[dist_name][pol] = alignment

    # ── Collect all policy names ──
    all_pol_names = (seq_policies
                     + [f"adaptive_τ{t}" for t in adaptive_taus]
                     + [f"jacobi_i{m}" for m in jacobi_iters]
                     + ['low_var_oracle', 'low_var_model'])

    # ── Visualisation ──
    figs = {'mi_matrices': fig_mi}
    dist_names = list(distributions.keys())

    # Training curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for name, h in train_hists.items():
        axes[0].plot(h['loss'], label=name, alpha=0.8)
        axes[1].plot(h['cond_acc'], label=name, alpha=0.8)
    axes[0].set_title('Training Loss'); axes[0].set_xlabel('Epoch')
    axes[1].set_title('Conditional Accuracy'); axes[1].set_xlabel('Epoch')
    for ax in axes: ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.tight_layout(); figs['training_curves'] = fig

    # Print baseline KL
    print("\n  Baseline KL_forward (from sampling p_true directly):")
    for dn in dist_names:
        bl = all_results[dn][seq_policies[0]]['kl_forward_baseline']
        print(f"    {dn}: {bl:.4f}")

    # ── FIG 1: KL forward heatmap (ALL policies) ──
    fig, ax = plt.subplots(figsize=(11, 8))
    data_kl = np.full((len(all_pol_names), len(dist_names)), np.nan)
    for j, d in enumerate(dist_names):
        for i, p in enumerate(all_pol_names):
            if p in all_results.get(d, {}):
                data_kl[i, j] = all_results[d][p]['kl_forward']
    im = ax.imshow(data_kl, cmap='RdYlGn_r', aspect='auto')
    ax.set_xticks(range(len(dist_names)))
    ax.set_xticklabels(dist_names, fontsize=8, rotation=20)
    ax.set_yticks(range(len(all_pol_names)))
    ax.set_yticklabels(all_pol_names, fontsize=8)
    for i in range(data_kl.shape[0]):
        for j in range(data_kl.shape[1]):
            if not np.isnan(data_kl[i, j]):
                ax.text(j, i, f"{data_kl[i,j]:.3f}", ha='center',
                        va='center', fontsize=7,
                        color='white' if data_kl[i,j] > np.nanmedian(
                            data_kl[~np.isnan(data_kl)]) else 'black')
    ax.axhline(y=len(seq_policies)-0.5, color='blue', lw=1.5, ls='--')
    ax.axhline(y=len(seq_policies)+len(adaptive_taus)-0.5,
               color='orange', lw=1.5, ls='--')
    plt.colorbar(im, ax=ax, shrink=0.7)
    ax.set_title('KL(p‖q) Forward: Sequential vs Adaptive vs Jacobi (↓ better)',
                 fontsize=12)
    fig.tight_layout(); figs['kl_forward_heatmap'] = fig

    # ── FIG 2: NFE vs KL scatter (Pareto frontier) ──
    fig, axes = plt.subplots(1, len(dist_names), figsize=(5*len(dist_names), 4.5))
    if len(dist_names) == 1: axes = [axes]
    for col, dn in enumerate(dist_names):
        ax = axes[col]
        dr = all_results.get(dn, {})
        bl = dr[seq_policies[0]]['kl_forward_baseline']
        ax.axhline(y=bl, color='gray', ls=':', alpha=0.5, label='baseline')

        # Sequential: all at NFE=L
        for p in seq_policies:
            if p in dr:
                c = '#2ecc71' if p == 'confidence' else '#95a5a6'
                ax.scatter(dr[p]['nfe'], dr[p]['kl_forward'],
                           marker='x', s=50, color=c, zorder=5)
                ax.annotate(p, (dr[p]['nfe'], dr[p]['kl_forward']),
                            fontsize=5,
                            textcoords="offset points", xytext=(3, 3))

        # Adaptive threshold
        a_nfes = [dr[f"adaptive_τ{t}"]['nfe'] for t in adaptive_taus
                  if f"adaptive_τ{t}" in dr]
        a_kls  = [dr[f"adaptive_τ{t}"]['kl_forward']
                  for t in adaptive_taus
                  if f"adaptive_τ{t}" in dr]
        if a_nfes:
            ax.plot(a_nfes, a_kls, '-o', color='#e74c3c',
                    label='adaptive', markersize=6, zorder=10)
            for t, nf, kl in zip(adaptive_taus, a_nfes, a_kls):
                ax.annotate(f'τ={t}', (nf, kl), fontsize=5,
                            textcoords="offset points", xytext=(4, 4))

        # Jacobi
        j_nfes = [dr[f"jacobi_i{m}"]['nfe'] for m in jacobi_iters
                  if f"jacobi_i{m}" in dr]
        j_kls  = [dr[f"jacobi_i{m}"]['kl_forward']
                  for m in jacobi_iters
                  if f"jacobi_i{m}" in dr]
        if j_nfes:
            ax.plot(j_nfes, j_kls, '-s', color='#3498db',
                    label='jacobi', markersize=6, zorder=10)
            for m, nf, kl in zip(jacobi_iters, j_nfes, j_kls):
                ax.annotate(f'i={m}', (nf, kl), fontsize=5,
                            textcoords="offset points", xytext=(4, 4))

        ax.set_title(dn, fontsize=10)
        ax.set_xlabel('NFE (forward passes)')
        ax.set_ylabel('KL(p‖q)')
        ax.grid(alpha=0.3)
        if col == 0: ax.legend(fontsize=7)
    fig.suptitle('Speed vs Quality: NFE vs KL(p‖q)', fontsize=13, y=1.02)
    fig.tight_layout(); figs['nfe_vs_kl'] = fig

    # ── FIG 3: Decode order heatmap (confidence only, per distribution) ──
    adaptive_pols = [p for p in seq_policies if p not in ('l2r', 'r2l')]
    fig, axes = plt.subplots(len(dist_names), len(adaptive_pols),
                             figsize=(3.5*len(adaptive_pols),
                                      2.5*len(dist_names)))
    if len(dist_names) == 1: axes = [axes]
    for row, dn in enumerate(dist_names):
        for col_idx, pol in enumerate(adaptive_pols):
            ax = axes[row][col_idx] if len(dist_names) > 1 else axes[col_idx]
            if (pol in all_orders[dn] and all_orders[dn][pol]):
                mr = compute_mean_decode_order(all_orders[dn][pol])
                ax.bar(range(L), mr, color='#3498db', alpha=0.85)
            ax.set_xticks(range(L))
            ax.set_title(f'{pol}' if row == 0 else '', fontsize=8)
            if col_idx == 0:
                ax.set_ylabel(dn.split('_')[-1], fontsize=8)
            ax.tick_params(labelsize=6)
            ax.grid(axis='y', alpha=0.3)
    fig.suptitle('Decode Order: Mean Step per Position (lower = earlier)',
                 fontsize=11, y=1.02)
    fig.tight_layout(); figs['decode_order'] = fig

    # ── FIG 4: MI-Order Alignment ──
    fig, ax = plt.subplots(figsize=(10, 5))
    align_data = np.full((len(seq_policies), len(dist_names)), np.nan)
    for j, dn in enumerate(dist_names):
        for i, pol in enumerate(seq_policies):
            if pol in mi_alignments.get(dn, {}):
                align_data[i, j] = mi_alignments[dn][pol]['rho']
    im = ax.imshow(align_data, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    ax.set_xticks(range(len(dist_names)))
    ax.set_xticklabels(dist_names, fontsize=8, rotation=20)
    ax.set_yticks(range(len(seq_policies)))
    ax.set_yticklabels(seq_policies, fontsize=9)
    for i in range(align_data.shape[0]):
        for j in range(align_data.shape[1]):
            if not np.isnan(align_data[i, j]):
                ax.text(j, i, f"{align_data[i,j]:.2f}", ha='center',
                        va='center', fontsize=8)
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title('MI-Order Alignment: ρ(total_MI, decode_rank)\n'
                 'Negative = high-MI positions decoded first (good)', fontsize=11)
    fig.tight_layout(); figs['mi_alignment'] = fig

    # ── FIG 5: Policy Ranking Table ──
    fig, ax = plt.subplots(figsize=(11, 7))
    rank_data = np.full((len(all_pol_names), len(dist_names)), np.nan)
    for j, dn in enumerate(dist_names):
        kl_vals = [(p, all_results[dn].get(p, {}).get('kl_forward', 99))
                    for p in all_pol_names]
        kl_vals.sort(key=lambda x: x[1])
        for rank, (p, _) in enumerate(kl_vals):
            i = all_pol_names.index(p)
            rank_data[i, j] = rank + 1
    im = ax.imshow(rank_data, cmap='YlGn_r', aspect='auto',
                   vmin=1, vmax=len(all_pol_names))
    ax.set_xticks(range(len(dist_names)))
    ax.set_xticklabels(dist_names, fontsize=8, rotation=20)
    ax.set_yticks(range(len(all_pol_names)))
    ax.set_yticklabels(all_pol_names, fontsize=8)
    for i in range(rank_data.shape[0]):
        for j in range(rank_data.shape[1]):
            if not np.isnan(rank_data[i, j]):
                ax.text(j, i, f"{int(rank_data[i,j])}", ha='center',
                        va='center', fontsize=9, fontweight='bold')
    ax.axhline(y=len(seq_policies)-0.5, color='blue', lw=1.5, ls='--')
    ax.axhline(y=len(seq_policies)+len(adaptive_taus)-0.5,
               color='orange', lw=1.5, ls='--')
    plt.colorbar(im, ax=ax, shrink=0.7, label='Rank (1=best)')
    ax.set_title('Policy Ranking by KL(p‖q) (1=best)\n'
                 'Blue line: seq|adaptive, Orange: adaptive|jacobi',
                 fontsize=11)
    fig.tight_layout(); figs['policy_ranking'] = fig

    # ── FIG 6: Spearman correlation heatmap ──
    corr_data = np.full((len(all_pol_names), len(dist_names)), np.nan)
    for j, d in enumerate(dist_names):
        for i, p in enumerate(all_pol_names):
            if p in all_results.get(d, {}):
                corr_data[i, j] = all_results[d][p]['spearman_r']
    fig, ax = plt.subplots(figsize=(11, 7))
    im = ax.imshow(corr_data, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)
    ax.set_xticks(range(len(dist_names)))
    ax.set_xticklabels(dist_names, fontsize=8, rotation=20)
    ax.set_yticks(range(len(all_pol_names)))
    ax.set_yticklabels(all_pol_names, fontsize=8)
    for i in range(corr_data.shape[0]):
        for j in range(corr_data.shape[1]):
            if not np.isnan(corr_data[i, j]):
                ax.text(j, i, f"{corr_data[i,j]:.2f}", ha='center',
                        va='center', fontsize=7)
    plt.colorbar(im, ax=ax, shrink=0.7)
    ax.set_title('Spearman Corr(path_score, log p_true)', fontsize=12)
    fig.tight_layout(); figs['correlation_heatmap'] = fig

    # ── FIG 7: Per-position marginal KL (detailed analysis) ──
    det_pols = sorted(DETAILED_POLICIES & set(seq_policies))
    if all_detailed:
        fig, axes = plt.subplots(1, len(dist_names),
                                  figsize=(5*len(dist_names), 4))
        if len(dist_names) == 1: axes = [axes]
        pol_colors = {'confidence': '#2ecc71', 'l2r': '#3498db',
                      'r2l': '#e74c3c', 'random': '#95a5a6'}
        for col, dn in enumerate(dist_names):
            ax = axes[col]
            for pol in det_pols:
                det = all_detailed.get(dn, {}).get(pol)
                if det is None or 'per_position_marginal_kl' not in det:
                    continue
                mkl = det['per_position_marginal_kl']
                ax.plot(range(L), mkl, '-o', markersize=4,
                        color=pol_colors.get(pol, '#888'),
                        label=pol, alpha=0.8)
            ax.set_xlabel('Position')
            ax.set_ylabel('KL(p_marginal ‖ q_marginal)')
            ax.set_title(dn, fontsize=9)
            ax.grid(alpha=0.3)
            if col == 0: ax.legend(fontsize=7)
        fig.suptitle('Per-Position Marginal KL by Policy',
                     fontsize=12, y=1.02)
        fig.tight_layout(); figs['pos_marginal_kl'] = fig

    # ── FIG 8: True log-prob histogram (confidence vs l2r) ──
    if all_detailed:
        fig, axes = plt.subplots(1, len(dist_names),
                                  figsize=(5*len(dist_names), 4))
        if len(dist_names) == 1: axes = [axes]
        for col, dn in enumerate(dist_names):
            ax = axes[col]
            for pol in ['confidence', 'l2r', 'r2l']:
                det = all_detailed.get(dn, {}).get(pol)
                if det is None or 'hist_centers' not in det:
                    continue
                ax.plot(det['hist_centers'], det['hist_counts'],
                        color=pol_colors.get(pol, '#888'),
                        label=f"{pol} (μ={det['true_lp_mean']:.1f})",
                        alpha=0.7)
            # Mark true expected log-prob
            det0 = next((v for v in all_detailed.get(dn, {}).values()
                         if 'true_expected_lp' in v), None)
            if det0:
                ax.axvline(det0['true_expected_lp'], color='black',
                           ls=':', alpha=0.5, label='E_p[log p]')
            ax.set_xlabel('log p_true(x)')
            ax.set_ylabel('Count')
            ax.set_title(dn, fontsize=9)
            ax.legend(fontsize=6); ax.grid(alpha=0.3)
        fig.suptitle('Distribution of True Probability of Generated Samples',
                     fontsize=12, y=1.02)
        fig.tight_layout(); figs['true_lp_histogram'] = fig

    # ── FIG 9: Context-conditional accuracy (Markov dists) ──
    markov_dists = [dn for dn in dist_names
                    if any(all_detailed.get(dn, {}).get(p, {})
                           .get('context_analysis')
                           for p in det_pols)]
    if markov_dists:
        fig, axes = plt.subplots(len(det_pols), len(markov_dists),
                                  figsize=(5*len(markov_dists),
                                           3.5*len(det_pols)),
                                  squeeze=False)
        for col, dn in enumerate(markov_dists):
            for row, pol in enumerate(det_pols):
                ax = axes[row, col]
                det = all_detailed.get(dn, {}).get(pol, {})
                ca = det.get('context_analysis')
                if ca and ca['worst_contexts']:
                    ctxs = ca['worst_contexts']
                    kls = [c['kl'] for c in ctxs]
                    labels = [f"c{c['ctx']}" for c in ctxs]
                    colors = ['#e74c3c' if c['true_mode'] != c['emp_mode']
                              else '#2ecc71' for c in ctxs]
                    ax.barh(range(len(kls)), kls, color=colors, alpha=0.8)
                    ax.set_yticks(range(len(kls)))
                    ax.set_yticklabels(labels, fontsize=7)
                    ax.set_xlabel('KL')
                ax.set_title(f'{pol} / {dn}', fontsize=8)
                ax.grid(alpha=0.3, axis='x')
        fig.suptitle('Worst Contexts by KL '
                     '(red=wrong mode, green=correct mode)',
                     fontsize=11, y=1.02)
        fig.tight_layout(); figs['context_kl'] = fig

    # ── FIG 10: HardMarkov compliance per position ──
    hm_name = 'E_HardMarkov'
    if hm_name in all_detailed:
        fig, ax = plt.subplots(figsize=(8, 5))
        for pol in det_pols:
            det = all_detailed.get(hm_name, {}).get(pol, {})
            hm_d = det.get('hard_markov')
            if hm_d:
                ppc = hm_d['per_position_compliance']
                ax.plot(range(2, 2 + len(ppc)), ppc, '-o',
                        markersize=5,
                        color=pol_colors.get(pol, '#888'),
                        label=f"{pol} "
                              f"(avg={hm_d['overall_compliance']:.3f})")
        ax.axhline(y=0.97, color='black', ls=':', alpha=0.5,
                   label='expected (0.97)')
        ax.set_xlabel('Position (t)')
        ax.set_ylabel('Compliance Rate')
        ax.set_ylim(0.9, 1.005)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        ax.set_title('HardMarkov: Preferred Token Compliance by Position')
        fig.tight_layout(); figs['hard_markov_compliance'] = fig

    # ── FIG 11: Confidence vs True Conditional Entropy ──
    # Key figure: shows WHY confidence policy fails
    if confidence_profiles and cond_entropy_profiles:
        markov_dists = [dn for dn in dist_names
                        if dn in ('B_Markov2', 'E_HardMarkov',
                                  'D_Markov+Global')]
        if markov_dists:
            fig, axes = plt.subplots(1, len(markov_dists),
                                      figsize=(6*len(markov_dists), 5))
            if len(markov_dists) == 1: axes = [axes]
            for col, dn in enumerate(markov_dists):
                ax = axes[col]
                cp = confidence_profiles[dn]
                ce = cond_entropy_profiles[dn]
                positions = range(L)

                # Model's initial confidence (step 0, all masked)
                ax.bar(positions, cp['max_prob'], alpha=0.4,
                       color='#e74c3c', label='Model confidence (step 0)')
                # True marginal entropy (normalized to [0,1])
                max_h = max(ce['marginal_H'])
                norm_mH = [h / max_h for h in ce['marginal_H']]
                ax.plot(positions, norm_mH, 's-', color='#7f8c8d',
                        label='Marginal H (norm)', markersize=5)
                # True conditional entropy in l2r (normalized)
                norm_cH = [h / max_h for h in ce['cond_H_l2r']]
                ax.plot(positions, norm_cH, 'o-', color='#2ecc71',
                        label='Cond H|l2r context (norm)', markersize=5)
                # Mark confidence decode order (1st, 2nd, ...)
                order = cp['confidence_order']
                for rank, pos in enumerate(order[:3]):
                    ax.annotate(f'{rank+1}st', (pos, cp['max_prob'][pos]),
                                fontsize=8, ha='center', va='bottom',
                                color='#e74c3c', fontweight='bold')
                ax.set_xlabel('Position')
                ax.set_ylabel('Score')
                ax.set_title(f'{dn}', fontsize=10)
                ax.legend(fontsize=7); ax.grid(alpha=0.3)
            fig.suptitle('Why Confidence Fails: Model Confidence vs '
                         'True Information Content',
                         fontsize=12, y=1.02)
            fig.tight_layout(); figs['confidence_diagnostic'] = fig

    # ── FIG 12: Stepwise Conditional Entropy (l2r vs confidence vs low_var) ──
    if all_stepwise:
        n_dists_with_sw = sum(1 for dn in dist_names
                              if dn in all_stepwise
                              and all_stepwise[dn])
        if n_dists_with_sw > 0:
            sw_dists = [dn for dn in dist_names
                        if dn in all_stepwise and all_stepwise[dn]]
            fig, axes = plt.subplots(1, len(sw_dists),
                                      figsize=(5*len(sw_dists), 4))
            if len(sw_dists) == 1: axes = [axes]
            sw_pol_colors = {
                'l2r': '#27ae60', 'confidence': '#3498db',
                'low_var_oracle': '#8e44ad', 'low_var_model': '#e67e22',
                'entropy_adj': '#e74c3c', 'single_perturb': '#1abc9c',
            }
            for col, dn in enumerate(sw_dists):
                ax = axes[col]
                for pol in ['l2r', 'confidence',
                            'low_var_oracle', 'low_var_model',
                            'entropy_adj', 'single_perturb']:
                    sw = all_stepwise.get(dn, {}).get(pol)
                    if sw:
                        hh = sw['step_cond_entropy']
                        ax.plot(range(L), hh, '-o', markersize=5,
                                color=sw_pol_colors.get(pol, '#888'),
                                label=f'{pol} (mean={np.mean(hh):.3f})')
                # Uniform entropy baseline
                ax.axhline(y=np.log(V), color='gray', ls=':',
                           alpha=0.3, label=f'H(uniform)={np.log(V):.2f}')
                ax.set_xlabel('Decode Step')
                ax.set_ylabel('True H(X_pos | decoded context)')
                ax.set_title(dn, fontsize=9)
                ax.legend(fontsize=7); ax.grid(alpha=0.3)
            fig.suptitle('Stepwise Conditional Entropy: '
                         'l2r vs Confidence vs Stability Policies',
                         fontsize=12, y=1.02)
            fig.tight_layout(); figs['stepwise_cond_entropy'] = fig

    # ── FIG 13: Error Cascade (Markov distributions) ──
    cascade_dists = [dn for dn in dist_names
                     if any(all_detailed.get(dn, {}).get(p, {})
                            .get('error_cascade', {}).get('cascade_ratio')
                            for p in det_pols)]
    if cascade_dists:
        fig, axes = plt.subplots(1, len(cascade_dists),
                                  figsize=(6*len(cascade_dists), 4))
        if len(cascade_dists) == 1: axes = [axes]
        for col, dn in enumerate(cascade_dists):
            ax = axes[col]
            for pol in det_pols:
                ec = all_detailed.get(dn, {}).get(pol, {}).get(
                    'error_cascade', {})
                cr = ec.get('cascade_ratio')
                if cr:
                    ax.plot(range(1, L), cr, '-o', markersize=4,
                            color=pol_colors.get(pol, '#888'),
                            label=f'{pol} (mean={np.mean(cr):.2f})')
            ax.axhline(y=1.0, color='black', ls=':', alpha=0.5,
                       label='no cascade (ratio=1)')
            ax.set_xlabel('Position t')
            ax.set_ylabel('P(err_t | err_{t-1}) / P(err_t)')
            ax.set_title(dn, fontsize=9)
            ax.legend(fontsize=7); ax.grid(alpha=0.3)
        fig.suptitle('Error Cascade: Do Errors Propagate to Neighbors?',
                     fontsize=12, y=1.02)
        fig.tight_layout(); figs['error_cascade'] = fig

    # JSON-safe results
    json_results = {}
    for d in all_results:
        json_results[d] = {}
        for p in all_results[d]:
            json_results[d][p] = {
                k: float(v) for k, v in all_results[d][p].items()}
    json_results['mi_alignments'] = {}
    for dn in mi_alignments:
        json_results['mi_alignments'][dn] = {}
        for pol in mi_alignments[dn]:
            a = mi_alignments[dn][pol]
            json_results['mi_alignments'][dn][pol] = {
                'rho': a['rho'], 'p_value': a['p_value'],
                'mean_ranks': a['mean_ranks'],  # NEW: full position ranks
            }
    # Confidence diagnostic profiles
    json_results['confidence_profiles'] = {}
    for dn in confidence_profiles:
        json_results['confidence_profiles'][dn] = confidence_profiles[dn]
    json_results['cond_entropy_profiles'] = {}
    for dn in cond_entropy_profiles:
        json_results['cond_entropy_profiles'][dn] = cond_entropy_profiles[dn]
    # Include detailed analysis (serializable parts only)
    json_results['detailed'] = {}
    for dn in all_detailed:
        json_results['detailed'][dn] = {}
        for pol in all_detailed[dn]:
            det = all_detailed[dn][pol]
            entry = {}
            # Full detailed fields (only present for DETAILED_POLICIES)
            if 'per_position_marginal_kl' in det:
                entry['per_position_marginal_kl'] = det['per_position_marginal_kl']
                entry['per_position_mode_correct'] = det['per_position_mode_correct']
                entry['true_lp_percentiles'] = det['true_lp_percentiles']
                entry['true_lp_mean'] = det['true_lp_mean']
                entry['true_expected_lp'] = det['true_expected_lp']
                entry['lp_gap'] = det['lp_gap']
            json_results['detailed'][dn][pol] = entry
            if 'context_analysis' in det:
                ca = det['context_analysis']
                json_results['detailed'][dn][pol]['context'] = {
                    'mean_ctx_kl': ca['mean_ctx_kl'],
                    'max_ctx_kl': ca['max_ctx_kl'],
                    'mean_ctx_mode_acc': ca['mean_ctx_mode_acc'],
                }
            if 'hard_markov' in det:
                json_results['detailed'][dn][pol]['hard_markov'] = \
                    det['hard_markov']
            # NEW: error cascade
            if 'error_cascade' in det:
                ec = det['error_cascade']
                json_results['detailed'][dn][pol]['error_cascade'] = {
                    'p_err_marginal': ec['p_err_marginal'],
                }
                if 'cascade_ratio' in ec:
                    json_results['detailed'][dn][pol]['error_cascade'].update({
                        'p_err_conditional': ec['p_err_conditional'],
                        'p_err_given_prev_err': ec['p_err_given_prev_err'],
                        'cascade_ratio': ec['cascade_ratio'],
                    })
            # NEW: stepwise conditional entropy
            if 'stepwise_cond_entropy' in det:
                json_results['detailed'][dn][pol]['stepwise_cond_entropy'] = \
                    det['stepwise_cond_entropy']

    save_results(EXP_NAME, json_results, figures=figs)

    # ── Summary ──
    print("\n" + "=" * 70)
    print("SUMMARY")
    print(f"  (N_SAMPLES={N_SAMPLES:,}, V^L={V**L:,}, "
          f"bootstrap_k={BOOTSTRAP_K})")
    print("=" * 70)

    # Combined table: KL forward + reverse + NFE
    header = f"{'Policy':<22} {'NFE':>5}"
    for dn in dist_names:
        header += f"  {'KL_f':>8} {'KL_r':>8}"
    print(f"\n{header}")
    print("-" * len(header))
    for pol in all_pol_names:
        nfe = all_results[dist_names[0]].get(pol, {}).get('nfe',
                                                           float('nan'))
        row = f"{pol:<22} {nfe:>5.1f}"
        for dn in dist_names:
            kl_f = all_results[dn].get(pol, {}).get('kl_forward',
                                                     float('nan'))
            kl_r = all_results[dn].get(pol, {}).get('kl_reverse',
                                                     float('nan'))
            row += f"  {kl_f:>8.4f} {kl_r:>8.4f}"
        print(row)

    # MI alignment summary
    print(f"\nMI-Order Alignment (ρ, negative = MI-aligned):")
    for dn in dist_names:
        best_pol = min(mi_alignments.get(dn, {}),
                       key=lambda p: mi_alignments[dn][p]['rho'],
                       default='N/A')
        if best_pol != 'N/A':
            a = mi_alignments[dn][best_pol]
            print(f"  {dn}: best={best_pol} ρ={a['rho']:+.3f}")

    # NFE efficiency summary
    print(f"\nNFE Efficiency (per distribution):")
    for dn in dist_names:
        conf_kl = all_results[dn].get('confidence', {}).get(
            'kl_forward', 99)
        print(f"  {dn} (confidence baseline: KL_f={conf_kl:.4f}, "
              f"NFE=8):")
        for key in ([f"adaptive_τ{t}" for t in adaptive_taus]
                    + [f"jacobi_i{m}" for m in jacobi_iters]):
            if key in all_results[dn]:
                r = all_results[dn][key]
                delta = r['kl_forward'] - conf_kl
                speedup = L / r['nfe'] if r['nfe'] > 0 else 0
                print(f"    {key:<22} NFE={r['nfe']:.1f} "
                      f"({speedup:.1f}× faster) "
                      f"ΔKL={delta:+.4f}")

    # ── HardMarkov focus analysis ──
    hm_name = 'E_HardMarkov'
    if hm_name in all_results:
        print(f"\n{'='*60}")
        print(f"  HARD MARKOV FOCUS ANALYSIS")
        print(f"{'='*60}")
        hm = distributions[hm_name]
        print(f"  Theoretical l2r entropy/step: "
              f"{hm.theoretical_l2r_entropy():.4f} bits")
        print(f"  Expected l2r greedy accuracy: "
              f"{hm.expected_accuracy_l2r():.2%}")

        # Check if confidence discovers l2r
        hm_r = all_results[hm_name]
        if 'confidence' in hm_r and 'l2r' in hm_r:
            conf_kl = hm_r['confidence']['kl_forward']
            l2r_kl = hm_r['l2r']['kl_forward']
            r2l_kl = hm_r['r2l']['kl_forward']
            rand_kl = hm_r.get('random', {}).get('kl_forward', float('nan'))
            print(f"\n  Policy KL(p‖q):")
            print(f"    l2r:        {l2r_kl:.4f}  (optimal)")
            print(f"    confidence: {conf_kl:.4f}  "
                  f"(gap={conf_kl - l2r_kl:+.4f})")
            print(f"    random:     {rand_kl:.4f}")
            print(f"    r2l:        {r2l_kl:.4f}  (worst)")

        # Decode order alignment
        if hm_name in mi_alignments and 'confidence' in mi_alignments[hm_name]:
            a = mi_alignments[hm_name]['confidence']
            print(f"\n  Confidence decode order MI alignment: "
                  f"ρ={a['rho']:+.3f}")
            mean_ranks = a['mean_ranks']
            print(f"  Mean decode rank by position: "
                  + ' '.join(f"p{i}={r:.1f}"
                             for i, r in enumerate(mean_ranks)))
            is_l2r = all(mean_ranks[i] < mean_ranks[i+1]
                         for i in range(len(mean_ranks)-1))
            print(f"  Strictly l2r order: {'YES' if is_l2r else 'NO'}")

        # Detailed per-sample analysis
        for pol in ['confidence', 'l2r', 'r2l']:
            det = all_detailed.get(hm_name, {}).get(pol)
            if det and 'hard_markov' in det:
                hmd = det['hard_markov']
                print(f"\n  {pol} compliance: "
                      f"{hmd['overall_compliance']:.4f} "
                      f"(expected: {hmd['expected_compliance']:.4f})")
                ppc = hmd['per_position_compliance']
                print(f"    per-position: "
                      + ' '.join(f"p{i+2}={c:.4f}"
                                 for i, c in enumerate(ppc)))
            if det and 'context_analysis' in det:
                ca = det['context_analysis']
                print(f"    context KL: mean={ca['mean_ctx_kl']:.4f} "
                      f"max={ca['max_ctx_kl']:.4f} "
                      f"mode_acc={ca['mean_ctx_mode_acc']:.3f}")
                # Show worst contexts
                worst = ca['worst_contexts'][:3]
                if worst:
                    print(f"    worst contexts:")
                    for w in worst:
                        print(f"      ctx={w['ctx']:>2}: "
                              f"true_mode={w['true_mode']} "
                              f"emp_mode={w['emp_mode']} "
                              f"KL={w['kl']:.4f} "
                              f"p_true={w['true_mode_prob']:.3f} "
                              f"p_emp={w['emp_mode_prob']:.3f} "
                              f"(n={w['n_obs']})")

    # ── Detailed analysis summary table ──
    if all_detailed:
        print(f"\n{'='*60}")
        print(f"  DETAILED ANALYSIS SUMMARY")
        print(f"  (from same {N_SAMPLES:,} samples as aggregate metrics)")
        print(f"{'='*60}")
        for dn in dist_names:
            dd = all_detailed.get(dn, {})
            if not dd:
                continue
            print(f"\n  {dn}:")
            for pol in det_pols:
                det = dd.get(pol)
                if det is None or 'per_position_marginal_kl' not in det:
                    continue
                mkl_max = max(det['per_position_marginal_kl'])
                mkl_mean = np.mean(det['per_position_marginal_kl'])
                mode_ok = sum(det['per_position_mode_correct'])
                print(f"    {pol:<15} "
                      f"margKL_mean={mkl_mean:.5f} "
                      f"margKL_max={mkl_max:.5f} "
                      f"modes_ok={mode_ok}/{L} "
                      f"lp_gap={det['lp_gap']:+.3f}")

    # ── Confidence failure diagnostic ──
    print(f"\n{'='*60}")
    print(f"  WHY CONFIDENCE FAILS: DIAGNOSTIC SUMMARY")
    print(f"{'='*60}")
    for dn in dist_names:
        cp = confidence_profiles.get(dn, {})
        ce = cond_entropy_profiles.get(dn, {})
        if not cp or not ce:
            continue
        print(f"\n  {dn}:")
        print(f"    Confidence decode order (step 0): {cp['confidence_order']}")
        print(f"    Model conf:  "
              + ' '.join(f"p{i}={c:.3f}" for i, c in enumerate(cp['max_prob'])))
        print(f"    Marginal H:  "
              + ' '.join(f"p{i}={h:.3f}" for i, h in enumerate(ce['marginal_H'])))
        print(f"    Cond H|l2r:  "
              + ' '.join(f"p{i}={h:.3f}" for i, h in enumerate(ce['cond_H_l2r'])))
        print(f"    Ctx benefit: "
              + ' '.join(f"p{i}={b:.3f}" for i, b in enumerate(ce['context_benefit_l2r'])))

        # Key insight: correlation between model confidence and conditional entropy
        rho_marg, _ = spearmanr(cp['max_prob'], ce['marginal_H'])
        rho_cond, _ = spearmanr(cp['max_prob'], ce['cond_H_l2r'])
        print(f"    Corr(conf, marginal_H): ρ={rho_marg:+.3f}  "
              f"(negative = conf tracks peaked marginals)")
        print(f"    Corr(conf, cond_H|l2r): ρ={rho_cond:+.3f}  "
              f"(negative = conf tracks easy conditionals)")
        # Ideal: confidence should correlate with cond_H, not marginal_H

        # Stepwise summary
        sw_conf = all_stepwise.get(dn, {}).get('confidence', {})
        sw_l2r = all_stepwise.get(dn, {}).get('l2r', {})
        if sw_conf and sw_l2r:
            h_conf = sw_conf['step_cond_entropy']
            h_l2r = sw_l2r['step_cond_entropy']
            print(f"    Stepwise H: confidence avg={np.mean(h_conf):.4f}  "
                  f"l2r avg={np.mean(h_l2r):.4f}  "
                  f"gap={np.mean(h_conf)-np.mean(h_l2r):+.4f}")

    plt.show()
    return all_results


if __name__ == '__main__':
    run()
