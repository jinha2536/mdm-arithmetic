"""Shared training loop (iteration-based, EMA weights, PUMA streaming buffer,
PAPL loss), greedy diffusion decoding and result persistence used by all
experiment scripts."""
import os, time, math, json
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.model import Transformer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Result persistence

RESULTS_DIR = os.environ.get('MDM_RESULTS_DIR', './results')


def prepare_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"  results dir: {RESULTS_DIR}")
    return RESULTS_DIR


def get_save_dir(exp_name):
    d = os.path.join(RESULTS_DIR, exp_name)
    os.makedirs(d, exist_ok=True)
    return d


def save_results(exp_name, results, figures=None, tag=''):
    d = get_save_dir(exp_name)
    sfx = f'_{tag}' if tag else ''
    with open(os.path.join(d, f'results{sfx}.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    if figures:
        for name, fig in figures.items():
            fig.savefig(os.path.join(d, f'{name}{sfx}.png'),
                        dpi=150, bbox_inches='tight')
    print(f"  Saved to {d}")


def save_checkpoint(exp_name, state_dict, tag=''):
    """Save a model state_dict as checkpoint_{tag}.pt."""
    d = get_save_dir(exp_name)
    sfx = f'_{tag}' if tag else ''
    path = os.path.join(d, f'checkpoint{sfx}.pt')
    torch.save(state_dict, path)
    print(f"  Checkpoint: {path}")
    return path


# Data encoding

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


# PUMA K schedule

def puma_k_step(k_start, k_end, k_step, k_every):
    """Step function: +k_step every k_every iters, capped at k_end."""
    def schedule(it):
        n_steps = it // k_every
        return min(k_start + n_steps * k_step, k_end)
    return schedule


# Training

def train_diffusion(
    # data (already on device)
    train_ids,          # [N, T]
    train_ans,          # [N] answer start positions
    ans_len,            # int
    tokenizer,
    # masking scheme
    mask_type='random',     # 'random', 'papl' or 'puma'
    blank_masks=None,       # [N, ans_len] bool; None = all answer positions maskable
    # PUMA
    puma_tau=0.9,
    puma_k_schedule=None,   # callable(it) -> K; required if mask_type='puma'
    # PAPL (Peng et al. 2025): uniform random masking with per-token loss weights
    #   w_i ~ softmax_{masked}((1/tau) log p(x_0^i | x_k)),  weight_i = (1/|M|)(1 + alpha w_i)
    # (w detached; alpha=0 recovers plain random masking)
    papl_tau=1.0,
    papl_alpha=1.0,
    # architecture
    n_layer=4, n_head=4, n_embd=128, dropout=0.1,
    # optimisation
    max_iters=200000,
    batch_size=128,
    lr=3e-4, min_lr=1e-5,
    warmup_iters=2000,
    grad_clip=1.0,
    weight_decay=0.01,
    ema_decay=0.9999,
    # evaluation callback: fn(model, it, tg) -> dict with 'overall_loss'
    eval_fn=None,
    eval_every=5000,
    log_every=1000,
    init_state=None,        # state_dict to initialise the model from
    device=None,
    use_amp=None,           # None=auto (bf16 on CUDA), True=force, False=disable
    seed_train=None,        # seeds the training RNG after model init/load (mask sampling, batch order)
    ckpt_schedule=None,     # iters at which ckpt_save_fn(ema_state_cpu, it) is called
    ckpt_save_fn=None,
):
    """Iteration-based masked-diffusion training with EMA weights.

    eval_fn is called on the EMA weights every eval_every iterations (and more
    often during the first 10% of training); the EMA state with the lowest
    returned 'overall_loss' is loaded into the model at the end.

    Returns: (model, dynamics)
        model: selected EMA weights, eval mode
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

    model = Transformer(
        vocab_size=len(tokenizer), block_size=T + 8,
        n_layer=n_layer, n_head=n_head, n_embd=n_embd, dropout=dropout,
    ).to(device)

    if init_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in init_state.items()})
        print(f"  [{mask_type}] Loaded init checkpoint")

    # seed the training RNG after model construction so that (run seed, method)
    # determines mask sampling / batch order independently of the init
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
    t0 = time.time()
    tg = 0

    # PUMA streaming buffer: one teacher-forced confidence-ordered chain per
    # batch slot, advanced by one stage per gradient step
    uses_streaming = (mask_type == 'puma')
    if uses_streaming:
        assert puma_k_schedule is not None, "puma_k_schedule required for puma"
        buf_z = torch.zeros(batch_size, T, dtype=torch.long, device=device)
        buf_x0 = torch.zeros(batch_size, T, dtype=torch.long, device=device)
        buf_ans = torch.zeros(batch_size, dtype=torch.long, device=device)
        buf_stage = torch.zeros(batch_size, dtype=torch.long, device=device)
        buf_pool = torch.randperm(N)
        buf_ptr = 0

        def _refresh(indices):
            nonlocal buf_ptr, buf_pool
            idx_t = torch.tensor(indices, device=device)
            n = len(indices)
            if buf_ptr + n > len(buf_pool):
                buf_pool = torch.randperm(N); buf_ptr = 0
            si = buf_pool[buf_ptr:buf_ptr + n].to(device); buf_ptr += n
            fresh_ids = train_ids[si]
            fresh_ans = train_ans[si]
            fresh_blanks = blank_masks[si]
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

    # random-masking batch iterator (the permutation is drawn for every scheme,
    # so all schemes consume the RNG identically up to this point)
    perm = torch.randperm(N, device=device)
    perm_ptr = 0

    def _next_batch():
        nonlocal perm, perm_ptr
        if perm_ptr + batch_size > N:
            perm = torch.randperm(N, device=device); perm_ptr = 0
        idx = perm[perm_ptr:perm_ptr + batch_size]; perm_ptr += batch_size
        return train_ids[idx], train_ans[idx], idx

    def _do_eval(it_num):
        nonlocal best_loss, best_ema
        if eval_fn is None:
            return
        # swap model params <-> EMA params in place
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
                best_ema = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        with torch.no_grad():
            for name, param in model.named_parameters():
                tmp = param.data.clone()
                param.data.copy_(ema_state[name])
                ema_state[name].copy_(tmp)
        model.train()

    model.eval()
    _do_eval(0)
    model.train()

    # AMP (bfloat16 autocast, no grad scaler)
    if use_amp is None:
        use_amp = (device.type == 'cuda' and torch.cuda.is_bf16_supported())
    amp_dtype = torch.bfloat16 if use_amp else torch.float32
    ctx = torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp)
    if use_amp:
        print(f"  [{mask_type}] AMP bfloat16 enabled")

    # EMA snapshot schedule (state is moved to CPU before the callback)
    ckpt_schedule_set = set(ckpt_schedule or [])
    if ckpt_schedule_set:
        print(f"  [{mask_type}] Ckpt schedule: {sorted(ckpt_schedule_set)}")

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
                # per-sample mean of masked-token NLL, then batch mean (as for 'random')
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
            bl = blank_masks[idx]
            # masking rate t ~ U(0, 1) per sample, each maskable position masked w.p. t
            t_ratio = torch.rand(B_b, device=device)
            m_probs = torch.zeros(B_b, T, dtype=torch.float, device=device)
            m_probs[bi, ap] = t_ratio.unsqueeze(1) * bl.float()
            m = torch.bernoulli(m_probs).bool()
            no_m = ~m.any(dim=1)
            if no_m.any():
                # at least one masked position per sample
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
                # per-sample mean of masked-token NLL, then batch mean
                log_probs = F.log_softmax(logits.float(), dim=-1)
                tlp = log_probs.gather(-1, ids.unsqueeze(-1)).squeeze(-1)  # [B,T]
                nll = -tlp                                                  # [B,T]
                n_masked = m.sum(dim=-1).clamp_min(1).float()               # [B]
                if mask_type == 'papl':
                    # PAPL: weight_i = (1/|M|) (1 + alpha * softmax_M(log p_i / tau)), w detached
                    det = (tlp.detach() / papl_tau).masked_fill(~m, float('-inf'))
                    w_papl = F.softmax(det, dim=-1)                         # [B,T]
                    base_w = (1.0 / n_masked).unsqueeze(-1)                 # [B,1]
                    weights = base_w * (1.0 + papl_alpha * w_papl)          # [B,T]
                    per_sample = (weights * nll * m.float()).sum(dim=-1)
                    loss = per_sample.mean()
                else:
                    per_sample = (nll * m.float()).sum(dim=-1) / n_masked
                    loss = per_sample.mean()
            tg += m.sum().item()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        with torch.no_grad():
            # remove_duplicate=False: wte.weight and lm_head.weight are tied, and
            # both ema_state entries must be updated
            for name, param in model.named_parameters(remove_duplicate=False):
                ema_state[name].lerp_(param.data, 1 - ema_decay)

        if ckpt_save_fn is not None and it in ckpt_schedule_set:
            with torch.no_grad():
                ema_cpu = {k: v.detach().cpu().clone() for k, v in ema_state.items()}
            ckpt_save_fn(ema_cpu, it)

        if uses_streaming:
            _advance(logits.detach(), K_cur)

        if it % log_every == 0:
            dynamics['train_loss'].append((it, loss.item()))
            print(f"    it {it:6d}/{max_iters} | loss {loss.item():.4f} | "
                  f"lr {get_lr(it):.1e} | tg {tg:,} | {time.time() - t0:.0f}s")

        if it % eval_every == 0 or \
           (it <= max_iters * 0.1 and it % max(eval_every // 5, 1) == 0):
            _do_eval(it)
            model.train()

    # load the selected EMA state
    if best_ema:
        model.load_state_dict({k: v.to(device) for k, v in best_ema.items()})
    elif ema_state:
        model.load_state_dict(ema_state)
    model.eval()
    _do_eval(max_iters)
    print(f"  Done (full {max_iters} iters, best probe loss: {best_loss:.4f}, "
          f"{time.time() - t0:.0f}s)")
    return model, dynamics


# Decoding

@torch.no_grad()
def generate_diffusion(model, prefix_ids, n_tokens, mask_id,
                       policy='confidence', pad_to=None, pad_id=None,
                       reasoning_rank=None, device=None):
    """Greedy masked-diffusion decoding, one token per forward pass.

    policy:
        'confidence'     masked position with the largest max-logit
        'r2l'            right to left (LSB-first for addition)
        'random'         uniformly random masked position
        'layered_oracle' masked position of minimum reasoning_rank, ties broken
                         by max-logit
    The committed token is always the argmax (the mask token is excluded).

    pad_to: pad the sequence to this length (training length) with pad_id.
    reasoning_rank: [B, n_tokens] long, per-position partial-order rank (smaller
        = earlier), used by 'layered_oracle'.
    Returns: (sequences, log_probs, {'n_steps', 'orders'})
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

    if policy == 'layered_oracle':
        assert reasoning_rank is not None, "layered_oracle requires reasoning_rank"
        # rank over the full sequence: +inf outside the answer region
        full_rank = torch.full((B, T_total), float('inf'), device=device)
        full_rank[:, T_pre:T_pre + n_tokens] = reasoning_rank.to(device).float()

    scores = torch.zeros(B, device=device)
    orders = []

    for t in range(n_tokens):
        logits = model(x)
        logits[:, :, mask_id] = -float('inf')   # never select the mask token

        if policy == 'confidence':
            max_logit = logits.max(dim=-1).values
            max_logit[unmasked] = -float('inf')
            pos = max_logit.argmax(-1)
        elif policy == 'r2l':
            pos = torch.full((B,), T_pre + n_tokens - 1 - t, dtype=torch.long, device=device)
        elif policy == 'random':
            rand_scores = torch.rand(B, T_total, device=device)
            rand_scores[unmasked] = -float('inf')
            pos = rand_scores.argmax(-1)
        elif policy == 'layered_oracle':
            # masked positions of minimum rank; ties broken by max logit
            max_logit = logits.max(dim=-1).values            # [B, T_total]
            mask_inel = unmasked | (full_rank == float('inf'))
            rank_eff = full_rank.clone()
            rank_eff[mask_inel] = float('inf')
            min_rank = rank_eff.min(dim=-1, keepdim=True).values  # [B, 1]
            elig_min = (rank_eff == min_rank) & ~mask_inel
            score = max_logit.clone()
            score[~elig_min] = -float('inf')
            pos = score.argmax(-1)
        else:
            raise ValueError(f"Unknown policy: {policy}")

        batch_arange = torch.arange(B, device=device)
        lp_at_pos = logits[batch_arange, pos]
        tok = lp_at_pos.argmax(-1)
        scores += F.log_softmax(lp_at_pos, dim=-1)[batch_arange, tok]
        x[batch_arange, pos] = tok
        unmasked[batch_arange, pos] = True
        orders.append(pos)

    orders_t = torch.stack(orders, dim=1).cpu() if orders else None
    return x, scores.cpu(), {'n_steps': len(orders), 'orders': orders_t}
