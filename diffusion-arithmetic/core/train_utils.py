"""
Shared training, generation, evaluation, Drive persistence.

Key design decisions (v2):
  1. Greedy decoding (argmax) for BOTH AR and diffusion — fair comparison.
  2. Convergence-based training: early stopping on training loss plateau.
  3. Diffusion decode order tracking for scratchpad analysis.
  4. RoPE support via pos_enc parameter.
"""
import os, time, math, json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')

from core.model import Transformer
from core.tokenizer import CharTokenizer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Google Drive persistence
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DRIVE_BASE = '/content/drive/MyDrive/diffusion-arithmetic-results'


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
# Convergence-based training
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def train_model(
    objective,        # 'ar' or 'diffusion'
    tokenizer,
    train_samples,
    max_len=None,
    # architecture
    n_layer=6, n_head=6, n_embd=384, dropout=0.2,
    pos_enc='absolute',   # 'absolute', 'rope', 'none'
    # training
    batch_size=256, lr=1e-3,
    max_iters=15000,      # hard upper bound
    warmup_iters=100, min_lr=1e-4, grad_clip=1.0,
    # convergence
    patience=2000,        # early stop if no improvement for this many iters
    min_delta=1e-4,       # minimum loss decrease to count as improvement
    # logging
    log_interval=500,
    device=None,
):
    """
    Train until convergence or max_iters, whichever comes first.
    Returns (model, history, max_len, converged_iter).
    """
    if device is None:
        device = DEVICE

    train_ids, train_ans = encode_samples(train_samples, tokenizer, max_len)
    if max_len is None:
        max_len = train_ids.shape[1]

    loader = DataLoader(
        TensorDataset(train_ids, train_ans),
        batch_size=batch_size, shuffle=True, drop_last=True)

    mask_id = tokenizer.special_ids['mask']
    pad_id = tokenizer.special_ids['pad']
    is_causal = (objective == 'ar')

    model = Transformer(
        vocab_size=len(tokenizer), block_size=max_len + 8,
        n_layer=n_layer, n_head=n_head, n_embd=n_embd,
        dropout=dropout, is_causal=is_causal, pos_enc=pos_enc,
    ).to(device)

    print(f"  [{objective}|{pos_enc}] params={model.n_params:,}, "
          f"seq_len={max_len}, vocab={len(tokenizer)}, "
          f"max_iters={max_iters}, patience={patience}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=0.1)

    def get_lr(it):
        if it < warmup_iters:
            return lr * it / max(warmup_iters, 1)
        ratio = (it - warmup_iters) / max(max_iters - warmup_iters, 1)
        return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * min(ratio, 1.0)))

    history = {'loss': [], 'iter': []}
    best_loss, best_iter = float('inf'), 0
    best_state = None
    model.train()
    it = 0
    t0 = time.time()

    while it < max_iters:
        for batch in loader:
            if it >= max_iters:
                break

            for pg in optimizer.param_groups:
                pg['lr'] = get_lr(it)

            ids = batch[0].to(device)
            ans_starts = batch[1].to(device)
            B, T = ids.shape

            if objective == 'ar':
                logits = model(ids[:, :-1])
                targets = ids[:, 1:]
                pos = torch.arange(T - 1, device=device).unsqueeze(0)
                # answer-only loss: positions >= (ans_start - 1) in shifted view
                loss_mask = pos >= (ans_starts.unsqueeze(1) - 1)
                # also exclude PAD targets
                loss_mask = loss_mask & (targets != pad_id)
                if loss_mask.sum() == 0:
                    it += 1; continue
                loss = F.cross_entropy(logits[loss_mask], targets[loss_mask])

            else:  # diffusion
                pos = torch.arange(T, device=device).unsqueeze(0)
                ans_mask = (pos >= ans_starts.unsqueeze(1)) & (ids != pad_id)
                # random mask ratio per sample
                t_ratio = torch.rand(B, device=device)
                m_probs = t_ratio.unsqueeze(1) * ans_mask.float()
                m = torch.bernoulli(m_probs).bool()
                # ensure >= 1 masked per sample
                no_m = ~(m.any(dim=1))
                for b in no_m.nonzero(as_tuple=True)[0]:
                    valid = ans_mask[b].nonzero(as_tuple=True)[0]
                    if len(valid) > 0:
                        m[b, valid[torch.randint(len(valid), (1,))]] = True
                x_m = ids.clone()
                x_m[m] = mask_id
                logits = model(x_m)
                loss = F.cross_entropy(logits[m], ids[m])

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            cur_loss = loss.item()
            if it % log_interval == 0:
                elapsed = time.time() - t0
                print(f"    it {it:5d} | loss {cur_loss:.4f} | "
                      f"lr {get_lr(it):.1e} | {elapsed:.0f}s")
                history['loss'].append(cur_loss)
                history['iter'].append(it)

            # convergence check (smoothed)
            if cur_loss < best_loss - min_delta:
                best_loss = cur_loss
                best_iter = it
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            elif it - best_iter >= patience and it > warmup_iters + patience:
                print(f"    ✓ Converged at it {it} (best={best_iter}, "
                      f"loss={best_loss:.4f})")
                if best_state:
                    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
                model.eval()
                return model, history, max_len, best_iter

            it += 1

    print(f"    ✓ Reached max_iters={max_iters} (best_loss={best_loss:.4f} "
          f"at it {best_iter})")
    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    return model, history, max_len, best_iter


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Generation: AR (greedy by default)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@torch.no_grad()
def generate_ar(model, prefix_ids, n_tokens, device=None):
    """Greedy (argmax) AR generation."""
    if device is None:
        device = DEVICE
    model.eval()
    x = prefix_ids.to(device)
    for _ in range(n_tokens):
        logits = model(x)[:, -1, :]
        x = torch.cat([x, logits.argmax(-1, keepdim=True)], dim=1)
    return x


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Generation: Diffusion (greedy by default, order tracking)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@torch.no_grad()
def generate_diffusion(model, prefix_ids, n_tokens, mask_id,
                       policy='confidence', greedy=True,
                       parallel_k=1, device=None):
    """
    Masked diffusion generation.

    Args:
        greedy: if True, always pick argmax token (fair with AR).
                if False, sample from softmax.
        policy: position selection strategy.
        parallel_k: tokens per step for parallel_* policies.

    Returns: (sequences, log_probs, decode_info)
        decode_info['orders']: (B, n_tokens) position indices in decode order
                               — useful for analysing scratchpad behaviour.
    """
    if device is None:
        device = DEVICE
    model.eval()
    B = prefix_ids.shape[0]
    T_pre = prefix_ids.shape[1]
    T = T_pre + n_tokens

    x = torch.full((B, T), mask_id, dtype=torch.long, device=device)
    x[:, :T_pre] = prefix_ids.to(device)
    unmasked = torch.zeros(B, T, dtype=torch.bool, device=device)
    unmasked[:, :T_pre] = True

    scores = torch.zeros(B, device=device)
    orders = []
    n_steps = 0

    def _pick_token(logits_pos):
        """Greedy or sample from a single position's logits."""
        if greedy:
            tok = logits_pos.argmax(-1)
        else:
            tok = torch.multinomial(F.softmax(logits_pos, dim=-1), 1).squeeze(-1)
        lp = F.log_softmax(logits_pos, dim=-1)
        return tok, lp[torch.arange(tok.shape[0], device=device), tok]

    # ── Parallel ──
    if policy.startswith('parallel'):
        while not unmasked[:, T_pre:].all():
            logits = model(x)
            probs = F.softmax(logits, dim=-1)
            step_positions = torch.full((B,), -1, dtype=torch.long)  # track first
            for b in range(B):
                mp = (~unmasked[b]).nonzero(as_tuple=True)[0]
                if len(mp) == 0:
                    continue
                k = min(parallel_k, len(mp))
                if policy == 'parallel_random':
                    sel = mp[torch.randperm(len(mp), device=device)[:k]]
                elif policy == 'parallel_confidence':
                    sel = mp[probs[b, mp].max(-1).values.topk(k).indices]
                elif policy == 'parallel_low_dep':
                    ln = F.normalize(logits[b, mp], dim=-1)
                    conf = probs[b, mp].max(-1).values
                    chosen = [conf.argmax().item()]
                    for _ in range(k - 1):
                        sims = (ln @ ln[chosen].T).max(-1).values
                        for c in chosen:
                            sims[c] = float('inf')
                        chosen.append(sims.argmin().item())
                    sel = mp[torch.tensor(chosen, device=device)]
                else:
                    raise ValueError(policy)
                for pos in sel:
                    if greedy:
                        tok = logits[b, pos].argmax()
                    else:
                        tok = torch.multinomial(
                            F.softmax(logits[b, pos].unsqueeze(0), dim=-1), 1).item()
                        tok = torch.tensor(tok, device=device)
                    scores[b] += F.log_softmax(logits[b, pos], dim=-1)[tok]
                    x[b, pos] = tok
                    unmasked[b, pos] = True
            n_steps += 1
        return x, scores.cpu(), {'n_steps': n_steps, 'orders': None}

    # ── Sequential ──
    for t in range(n_tokens):
        logits = model(x)
        probs = F.softmax(logits, dim=-1)

        if policy == 'confidence':
            sc = probs.max(-1).values.clone(); sc[unmasked] = -1; pos = sc.argmax(-1)
        elif policy == 'low_entropy':
            e = -(probs * (probs + 1e-10).log()).sum(-1)
            e[unmasked] = 1e9; pos = e.argmin(-1)
        elif policy == 'high_entropy':
            e = -(probs * (probs + 1e-10).log()).sum(-1)
            e[unmasked] = -1e9; pos = e.argmax(-1)
        elif policy == 'margin':
            t2 = probs.topk(2, dim=-1).values
            mg = t2[:, :, 0] - t2[:, :, 1]
            mg[unmasked] = -1; pos = mg.argmax(-1)
        elif policy == 'random':
            pos = torch.zeros(B, dtype=torch.long, device=device)
            for b in range(B):
                mp = (~unmasked[b]).nonzero(as_tuple=True)[0]
                pos[b] = mp[torch.randint(len(mp), (1,))]
        elif policy == 'l2r':
            pos = torch.full((B,), T_pre + t, dtype=torch.long, device=device)
        elif policy == 'r2l':
            pos = torch.full((B,), T_pre + n_tokens - 1 - t, dtype=torch.long, device=device)
        else:
            raise ValueError(f"Unknown policy: {policy}")

        lp_at_pos = logits[torch.arange(B, device=device), pos]
        if greedy:
            tok = lp_at_pos.argmax(-1)
        else:
            tok = torch.multinomial(F.softmax(lp_at_pos, dim=-1), 1).squeeze(-1)
        scores += F.log_softmax(lp_at_pos, dim=-1)[torch.arange(B, device=device), tok]
        x[torch.arange(B, device=device), pos] = tok
        unmasked[torch.arange(B, device=device), pos] = True
        orders.append(pos.cpu())
        n_steps += 1

    return x, scores.cpu(), {
        'n_steps': n_steps,
        'orders': torch.stack(orders, dim=1) if orders else None,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Evaluation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def evaluate(model, tokenizer, test_samples, objective,
             get_answer_fn, get_full_answer_fn=None,
             policy='confidence', greedy=True,
             parallel_k=1, batch_size=128, device=None):
    """
    Exact match evaluation.

    Args:
        get_answer_fn: sample_str → final answer string
        get_full_answer_fn: sample_str → full string after '='
    Returns dict with 'exact_match', 'n_correct', 'n_total',
        'examples', 'decode_orders' (for diffusion sequential).
    """
    if device is None:
        device = DEVICE
    if get_full_answer_fn is None:
        get_full_answer_fn = lambda s: s.split('=', 1)[1]

    mask_id = tokenizer.special_ids['mask']
    model.eval()
    correct, total = 0, 0
    examples = []
    all_orders = []

    for start in range(0, len(test_samples), batch_size):
        batch = test_samples[start:start + batch_size]
        B = len(batch)

        prefixes = [s.split('=')[0] + '=' for s in batch]
        gold_full = [get_full_answer_fn(s) for s in batch]
        gold_final = [get_answer_fn(s) for s in batch]

        penc = [tokenizer.encode(p) for p in prefixes]
        pmax = max(len(p) for p in penc)
        pids = torch.full((B, pmax), tokenizer.special_ids['pad'], dtype=torch.long)
        for i, e in enumerate(penc):
            pids[i, :len(e)] = torch.tensor(e)

        ans_len = len(tokenizer.encode(gold_full[0]))

        if objective == 'ar':
            gen = generate_ar(model, pids, ans_len, device)
            pred_ids = gen[:, pmax:pmax + ans_len]
            batch_orders = None
        else:
            gen, _, info = generate_diffusion(
                model, pids, ans_len, mask_id,
                policy=policy, greedy=greedy,
                parallel_k=parallel_k, device=device)
            pred_ids = gen[:, pmax:pmax + ans_len]
            batch_orders = info.get('orders')  # (B, ans_len) or None

        if batch_orders is not None:
            all_orders.append(batch_orders)

        for i in range(B):
            pred_str = tokenizer.decode(pred_ids[i].cpu().tolist())
            # extract final answer for scratchpad formats
            if '>>' in gold_full[i]:
                pred_final = pred_str.split('>>')[-1] if '>>' in pred_str \
                    else pred_str[-len(gold_final[i]):]
            elif '=>' in gold_full[i]:
                pred_final = pred_str.split('=>')[-1] if '=>' in pred_str \
                    else pred_str[-len(gold_final[i]):]
            else:
                pred_final = pred_str
            ok = pred_final == gold_final[i]
            correct += int(ok)
            total += 1
            if len(examples) < 10:
                examples.append({
                    'prefix': prefixes[i], 'pred': pred_final,
                    'gold': gold_final[i], 'correct': ok})

    result = {
        'exact_match': correct / max(total, 1),
        'n_correct': correct, 'n_total': total,
        'examples': examples,
    }
    if all_orders:
        result['decode_orders'] = torch.cat(all_orders, dim=0)
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Scratchpad decode order analysis
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def analyse_decode_order(decode_orders, ans_start, scratchpad_end,
                         total_len):
    """
    Analyse whether diffusion fills scratchpad positions before
    final answer positions.

    Args:
        decode_orders: (N, ans_len) tensor of absolute positions
        ans_start: position where answer starts (after '=')
        scratchpad_end: position where '>>' or '=>' starts
                        (i.e. scratchpad occupies [ans_start, scratchpad_end))
        total_len: total sequence length
    Returns:
        dict with 'scratchpad_first_ratio', 'avg_scratchpad_rank',
        'avg_final_rank', 'position_heatmap' (ans_len,) of mean rank.
    """
    if decode_orders is None or len(decode_orders) == 0:
        return {}

    N, S = decode_orders.shape
    # positions relative to ans_start
    rel = decode_orders - ans_start

    scratchpad_len = scratchpad_end - ans_start
    final_len = total_len - scratchpad_end

    # For each sample, compute mean rank of scratchpad vs final positions
    sp_ranks, fn_ranks = [], []
    sp_first_count = 0
    for i in range(N):
        order = decode_orders[i].tolist()
        sp_r, fn_r = [], []
        for rank, pos in enumerate(order):
            if ans_start <= pos < scratchpad_end:
                sp_r.append(rank)
            elif pos >= scratchpad_end:
                fn_r.append(rank)
        if sp_r and fn_r:
            sp_ranks.append(sum(sp_r) / len(sp_r))
            fn_ranks.append(sum(fn_r) / len(fn_r))
            if max(sp_r) < min(fn_r):
                sp_first_count += 1

    n_valid = max(len(sp_ranks), 1)
    # Position heatmap: average decode rank per answer position
    heatmap = torch.zeros(S, dtype=torch.float)
    for step in range(S):
        heatmap[step] = (decode_orders[:, step] - ans_start).float().mean()

    return {
        'scratchpad_first_ratio': sp_first_count / n_valid,
        'avg_scratchpad_rank': sum(sp_ranks) / n_valid if sp_ranks else -1,
        'avg_final_rank': sum(fn_ranks) / n_valid if fn_ranks else -1,
        'n_samples': N,
    }
