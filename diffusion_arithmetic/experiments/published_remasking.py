"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Published remasking methods at their published default settings
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Companion to remasking_analysis.py (which provides the class CEILING: exact
leave-one-out audit after full decode). This file evaluates the methods AS
PUBLISHED — remasking DURING generation, within the standard NFE budget,
with the authors' own default hyperparameters:

  remdm_cap   ReMDM cap schedule, η = 0.008
              (kuleshov-group/remdm scripts/remdm-cap.sh)
  remdm_conf  ReMDM confidence-weighted schedule, η = 0.008; per-token
              remask probability ∝ softmax(−ψ) with ψ = current-state
              probability of the committed token (per the official
              diffusion.py; note the PRISM paper describes ReMDM-conf as
              a frozen commit-time score — the official code uses the
              current state, which we follow)
  remdm_loop  ReMDM loop schedule, η = 0.02, t_on = 0.55, t_off = 0.05,
              α_on = 0.9 (scripts/remdm-loop.sh; same values used in the
              published LLaDA-8B port)
  prism       PRISM [Kim et al.], implemented faithfully: a linear
              per-token-quality head on the backbone's final hidden state,
              fine-tuned with the PRISM loss (Algorithm 1) at their Sudoku
              defaults (k = 4, n_y = 1, nucleus p = 1.0, AdamW lr 3e-4,
              batch 256); inference via their Algorithm 3 (preallocated
              |T| = K = 4 remasks per eligible step, l_on = 0). Backbone
              frozen = their sanctioned adapter-only mode (the λ·MDM
              regularizer is then vacuous and omitted).

All decoders run from scratch (fully masked answer) at the SAME NFE as the
base confidence decode (T = L forward passes), deterministic greedy tokens
and confidence unmask selection, matching the paper's decoding regime.
Buckets, checkpoints, and canonical-failure identification are shared with
remasking_analysis.py, so tables are directly comparable.

Usage (Colab):
  %run experiments/published_remasking.py \\
      --checkpoint-dir <drive>/exp_addition_3way_v1 \\
      --ckpt-template 'checkpoint_{method}.pt' \\
      --seeds 41 --methods random papl puma \\
      --out <drive>/published_remasking_3way_v1.json
"""
from __future__ import annotations

import argparse, json, math, os, sys, time

import torch
import torch.nn as nn
import torch.nn.functional as F

if '__file__' in dir():
    _here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(_here))
    sys.path.insert(0, _here)
else:
    sys.path.insert(0, '.')

import exp_addition as A
from addition_decode_analysis import load_model
from remasking_analysis import (set_nd, build_bucket, make_meta,
                                base_decode_with_audits, audit_and_observe,
                                confidence_fill, _jsonable, DETECT_TAUS)

# ── Published defaults (verbatim from the official repos/papers) ──
REMDM_CAP_ETA = 0.008          # scripts/remdm-cap.sh
REMDM_LOOP_ETA = 0.02          # scripts/remdm-loop.sh
REMDM_LOOP_T_ON = 0.55
REMDM_LOOP_T_OFF = 0.05
REMDM_LOOP_ALPHA_ON = 0.9
PRISM_K = 4                    # PRISM Table 5, Sudoku column
PRISM_NY = 1
PRISM_L_ON = 0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint-dir', type=str, required=True)
    p.add_argument('--ckpt-template', type=str,
                   default='checkpoint_seed{seed}_{method}_iter{iter:06d}.pt')
    p.add_argument('--seeds', nargs='+', type=int, default=[41])
    p.add_argument('--methods', nargs='+', default=['random', 'papl', 'puma'])
    p.add_argument('--iter', type=int, default=300000)
    p.add_argument('--nd', type=int, default=32)
    p.add_argument('--n-head', type=int, default=None)
    p.add_argument('--chain-sweep', nargs='+', type=int,
                   default=[4, 8, 12, 16, 20, 24, 28])
    p.add_argument('--n-per-bucket', type=int, default=300)
    p.add_argument('--natural-n', type=int, default=500)
    p.add_argument('--test-seed', type=int, default=1042)
    p.add_argument('--decoders', nargs='+',
                   default=['remdm_cap', 'remdm_conf', 'remdm_loop', 'prism'],
                   choices=['remdm_cap', 'remdm_conf', 'remdm_loop', 'prism'])
    # PRISM fine-tuning (Table 5 Sudoku defaults; scale-adjusted iteration count)
    p.add_argument('--prism-iters', type=int, default=2000)
    p.add_argument('--prism-batch', type=int, default=256)
    p.add_argument('--prism-lr', type=float, default=3e-4)
    p.add_argument('--prism-n-train', type=int, default=20000)
    p.add_argument('--prism-data-seed', type=int, default=7)
    p.add_argument('--out', type=str, default='published_remasking.json')
    p.add_argument('--device', type=str, default=None)
    return p.parse_args()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ReMDM discrete port (during-generation, equal NFE)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Linear masking schedule α(t) = 1 − t over T = L steps (one net commit per
# step, matching the paper's one-token-per-step regime). At step i (diffusion
# time t = 1 − i/T → s = 1 − (i+1)/T):
#   1. each committed token is remasked with probability σ (schedule-specific)
#   2. tokens are unmasked by greedy confidence until the unmasked count
#      reaches the schedule target round(L·(1−s)); the final step unmasks all.

@torch.no_grad()
def decode_remdm(model, x0, ans_start, AL, mask_id, variant, rng, device):
    x = x0.clone()
    B = x.shape[0]
    ar = torch.arange(B, device=device)
    masked = torch.ones(B, AL, dtype=torch.bool, device=device)
    T = AL
    n_remask_total = 0
    loop_target = min(AL - 1, int(round(AL * REMDM_LOOP_ALPHA_ON)))
    for i in range(T):
        t = 1.0 - i / T
        s = 1.0 - (i + 1) / T
        in_loop = (variant == 'remdm_loop'
                   and REMDM_LOOP_T_OFF < t <= REMDM_LOOP_T_ON)
        # ── remask phase ──
        committed = ~masked
        n_comm = committed.sum()
        if n_comm > 0:
            if variant == 'remdm_loop':
                if in_loop:
                    a = REMDM_LOOP_ALPHA_ON
                    sigma_max = min(1.0, (1.0 - a) / a)
                    sigma = min(REMDM_LOOP_ETA, sigma_max)
                else:
                    sigma = 0.0
            else:  # cap / conf share the cap σ with η = 0.008
                alpha_t, alpha_s = max(1.0 - t, 1e-8), 1.0 - s
                sigma_max = min(1.0, (1.0 - alpha_s) / alpha_t)
                sigma = min(REMDM_CAP_ETA, sigma_max)
            if sigma > 0:
                if variant == 'remdm_conf':
                    # per-token σ_ℓ ∝ softmax(−ψ), ψ = current-state prob of
                    # the committed token (official diffusion.py semantics)
                    logits = model(x)
                    al = logits[:, ans_start:ans_start + AL, :]
                    probs = F.softmax(al, dim=-1)
                    psi = probs.gather(
                        -1, x[:, ans_start:ans_start + AL].unsqueeze(-1)
                    ).squeeze(-1)
                    w = torch.where(committed, -psi,
                                    torch.full_like(psi, -float('inf')))
                    w = F.softmax(w, dim=-1)
                    p_rm = (w * sigma * committed.sum(1, keepdim=True)
                            ).clamp(0, 1)
                else:
                    p_rm = torch.where(
                        committed, torch.full_like(masked, sigma,
                                                   dtype=torch.float),
                        torch.zeros(B, AL, device=device))
                u = torch.rand(B, AL, generator=rng).to(device)
                rm = (u < p_rm) & committed
                if in_loop:
                    pass  # loop phase: remask freely; refill below to target
                if rm.any():
                    pos = rm.nonzero(as_tuple=False)
                    x[pos[:, 0], ans_start + pos[:, 1]] = mask_id
                    masked |= rm
                    n_remask_total += int(rm.sum())
        # ── unmask phase (greedy confidence up to schedule target) ──
        if in_loop:
            target = loop_target
        elif i == T - 1:
            target = AL
        else:
            target = int(round(AL * (1.0 - s)))
        need = target - int((~masked).sum(1).min())  # per-row via loop below
        # per-row refill to target
        deficit = (target - (~masked).sum(1)).clamp(min=0)
        max_d = int(deficit.max())
        for _ in range(max_d):
            todo = (deficit > 0) & masked.any(1)
            if not todo.any():
                break
            logits = model(x)
            al = logits[:, ans_start:ans_start + AL, :].clone()
            al[:, :, mask_id] = -float('inf')
            ml = al.max(-1).values
            ml[~masked] = -float('inf')
            pos = ml.argmax(-1)
            tok = al[ar, pos].argmax(-1)
            sel = todo & masked[ar, pos]
            x[ar[sel], ans_start + pos[sel]] = tok[sel]
            masked[ar[sel], pos[sel]] = False
            deficit = (target - (~masked).sum(1)).clamp(min=0)
    # safety: fill any stragglers
    if masked.any():
        confidence_fill(model, x, masked.clone(), ans_start, AL, mask_id)
    return x[:, ans_start:ans_start + AL], n_remask_total


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PRISM (faithful implementation)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class PrismWrapper(nn.Module):
    """Backbone + linear per-token-quality head on the final hidden state
    (ln_f output, captured via forward hook — no backbone modification)."""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        n_embd = backbone.ln_f.normalized_shape[0]
        self.q_head = nn.Linear(n_embd, 1)
        self._h = None
        backbone.ln_f.register_forward_hook(
            lambda m, inp, out: setattr(self, '_h', out))

    def forward(self, x):
        logits = self.backbone(x)
        q = torch.sigmoid(self.q_head(self._h)).squeeze(-1)
        return logits, q


def prism_finetune(backbone, tok, ans_start, AL, mask_id, args, device,
                   seed_ft):
    """Algorithm 1, adapter-only mode: backbone frozen, train the head with
    L(θ) = mean_{i∈S} BCE(1[x^i = y^i], g_i(y)); |S| = k = 4, y^i sampled
    from f(·|z) at temperature 1 (nucleus p = 1.0). λ·MDM term omitted
    (backbone frozen ⇒ vacuous)."""
    torch.manual_seed(seed_ft)
    data = A.gen_data_natural(args.prism_n_train, seed=args.prism_data_seed)
    enc = torch.tensor([tok.encode(s) for s in data], dtype=torch.long,
                       device=device)
    gold_ans = enc[:, ans_start:ans_start + AL]
    wrapper = PrismWrapper(backbone).to(device)
    for p_ in wrapper.backbone.parameters():
        p_.requires_grad_(False)
    wrapper.backbone.eval()
    opt = torch.optim.AdamW(wrapper.q_head.parameters(), lr=args.prism_lr)
    gen = torch.Generator().manual_seed(seed_ft)
    B = args.prism_batch
    N = enc.shape[0]
    losses = []
    for it in range(args.prism_iters):
        idx = torch.randint(0, N, (B,), generator=gen).to(device)
        xb = enc[idx]
        gb = gold_ans[idx]
        n = torch.randint(0, AL + 1, (B,), generator=gen).to(device)
        u = torch.rand(B, AL, generator=gen).to(device)
        thr = u.sort(dim=1).values.gather(1, (n.clamp(max=AL - 1))
                                          .unsqueeze(1))
        mask = (u <= thr) & (n > 0).unsqueeze(1)          # n masked per row
        z = xb.clone()
        z[:, ans_start:ans_start + AL][mask] = mask_id
        with torch.no_grad():
            logits = wrapper.backbone(z)[:, ans_start:ans_start + AL, :]
            logits[:, :, mask_id] = -float('inf')
            # sample y^i for k random masked positions per row
            r = torch.rand(B, AL, generator=gen).to(device)
            r[~mask] = 2.0
            order = r.argsort(dim=1)
            sel = torch.zeros(B, AL, dtype=torch.bool, device=device)
            kk = torch.minimum(mask.sum(1),
                               torch.full_like(mask.sum(1), PRISM_K))
            for b in range(B):
                if kk[b] > 0:
                    sel[b, order[b, :kk[b]]] = True
            gum = -torch.log(-torch.log(
                torch.rand(B, AL, logits.shape[-1], generator=gen)
                .clamp_(1e-20, 1 - 1e-20))).to(device)
            ydraw = (logits + gum).argmax(-1)
        y = z.clone()
        ya = y[:, ans_start:ans_start + AL]
        ya[sel] = ydraw[sel]
        y[:, ans_start:ans_start + AL] = ya
        label = (ydraw == gb).float()
        _, q = wrapper(y)
        qa = q[:, ans_start:ans_start + AL]
        if sel.any():
            loss = F.binary_cross_entropy(qa[sel].clamp(1e-6, 1 - 1e-6),
                                          label[sel])
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss))
        if (it + 1) % 500 == 0:
            print(f"    [prism ft] it {it+1}: BCE "
                  f"{sum(losses[-200:])/max(1,len(losses[-200:])):.4f}")
    wrapper.eval()
    return wrapper, losses


@torch.no_grad()
def decode_prism(wrapper, x0, ans_start, AL, mask_id, device):
    """Algorithm 3, preallocated fixed-K: at each step, if K ≤ |M| < L−K
    (and step ≥ l_on = 0): remask the K committed cells with lowest quality
    scores and unmask ⌈L/N⌉ + K = 1 + K cells; else unmask 1. Greedy tokens,
    confidence unmask selection. N = L steps."""
    x = x0.clone()
    B = x.shape[0]
    ar = torch.arange(B, device=device)
    masked = torch.ones(B, AL, dtype=torch.bool, device=device)
    n_remask_total = 0
    T = AL
    for i in range(T):
        logits, q = wrapper(x)
        al = logits[:, ans_start:ans_start + AL, :].clone()
        al[:, :, mask_id] = -float('inf')
        qa = q[:, ans_start:ans_start + AL].clone()
        M = masked.sum(1)
        elig = (M >= PRISM_K) & (M < AL - PRISM_K) & (i >= PRISM_L_ON)
        # remask K lowest-quality committed cells on eligible rows
        if elig.any():
            qa_c = torch.where(~masked, qa, torch.full_like(qa, float('inf')))
            thr = qa_c.topk(PRISM_K, dim=1, largest=False).values[:, -1:]
            rm = (qa_c <= thr) & ~masked & elig.unsqueeze(1)
            if rm.any():
                pos = rm.nonzero(as_tuple=False)
                x[pos[:, 0], ans_start + pos[:, 1]] = mask_id
                masked |= rm
                n_remask_total += int(rm.sum())
        # unmask 1 + K (eligible) or 1
        n_un = torch.where(elig, torch.full_like(M, 1 + PRISM_K),
                           torch.ones_like(M))
        n_un = torch.minimum(n_un, masked.sum(1))
        max_u = int(n_un.max())
        for u_ in range(max_u):
            todo = (n_un > 0) & masked.any(1)
            if not todo.any():
                break
            logits = wrapper.backbone(x)
            al = logits[:, ans_start:ans_start + AL, :].clone()
            al[:, :, mask_id] = -float('inf')
            ml = al.max(-1).values
            ml[~masked] = -float('inf')
            pos = ml.argmax(-1)
            tok = al[ar, pos].argmax(-1)
            sel = todo & masked[ar, pos]
            x[ar[sel], ans_start + pos[sel]] = tok[sel]
            masked[ar[sel], pos[sel]] = False
            n_un = n_un - sel.long()
    if masked.any():
        confidence_fill(wrapper.backbone, x, masked.clone(), ans_start, AL,
                        mask_id)
    return x[:, ans_start:ans_start + AL], n_remask_total


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Orchestration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run(args):
    device = torch.device(args.device) if args.device else (
        torch.device('cuda') if torch.cuda.is_available()
        else torch.device('cpu'))
    nd, AL = set_nd(args.nd)
    if args.n_head is not None:
        import addition_decode_analysis as ADA
        ADA.N_HEAD_OVERRIDE = args.n_head
    tok = A.build_tok()
    mask_id = tok.special_ids['mask']

    buckets = {}
    if args.natural_n > 0:
        buckets['natural'] = A.gen_data_natural(args.natural_n,
                                                seed=args.test_seed)
    for cl in args.chain_sweep:
        sp = A.gen_min_chain_test(args.n_per_bucket,
                                  seed=args.test_seed + 500 + cl,
                                  min_chain=cl)
        if sp:
            buckets[f'chain>={cl}'] = sp
    enc_buckets = {}
    for name, samples in buckets.items():
        x0, gold, ans_start = build_bucket(tok, samples, device)
        metas = [make_meta(s, nd, AL) for s in samples]
        enc_buckets[name] = (x0, gold, ans_start, metas)
        print(f"  bucket {name}: n={len(samples)}")

    results = {'config': _jsonable(vars(args)),
               'defaults': {'remdm_cap_eta': REMDM_CAP_ETA,
                            'remdm_loop': [REMDM_LOOP_ETA, REMDM_LOOP_T_ON,
                                           REMDM_LOOP_T_OFF,
                                           REMDM_LOOP_ALPHA_ON],
                            'prism_K': PRISM_K}}
    for seed in args.seeds:
        for method in args.methods:
            ck = args.ckpt_template.format(seed=seed, method=method,
                                           iter=args.iter)
            path = os.path.join(args.checkpoint_dir, ck)
            print(f"\n{'━'*66}\n  seed {seed} | {method} | {path}\n{'━'*66}")
            if not os.path.exists(path):
                print("  !! checkpoint missing, skipping")
                continue
            model = load_model(path, device)
            model.eval()
            wrapper = None
            if 'prism' in args.decoders:
                print("  PRISM fine-tuning (head-only, Algorithm 1)...")
                wrapper, _ = prism_finetune(
                    model, tok,
                    enc_buckets[next(iter(enc_buckets))][2], AL, mask_id,
                    args, device, seed_ft=seed * 1000 + 17)
            key = f'seed{seed}_{method}'
            results[key] = {}
            for name, (x0, gold, ans_start, metas) in enc_buckets.items():
                t0 = time.time()
                base = base_decode_with_audits(model, x0, gold, ans_start,
                                               AL, mask_id, 0, 4096)
                obs = audit_and_observe(base, gold, metas, tok, DETECT_TAUS)
                recs = obs['records']
                dep_b = torch.tensor([r['b'] for r in recs], dtype=torch.long,
                                     device=device) if recs else None
                dep_c = (torch.tensor(
                    [int((~base['pos_correct'][r['b']]).nonzero(
                        as_tuple=True)[0][0]) for r in recs],
                    dtype=torch.long, device=device) if recs else None)
                entry = {'base_acc': float(base['correct'].float().mean()),
                         'n_trap': len(recs), 'decoders': {}}
                for dec in args.decoders:
                    rng = torch.Generator().manual_seed(
                        50_000 + seed * 131 + sum(ord(c) for c in name)
                        + sum(ord(c) for c in dec))
                    if dec == 'prism':
                        pred, n_rm = decode_prism(wrapper, x0, ans_start, AL,
                                                  mask_id, device)
                    else:
                        pred, n_rm = decode_remdm(model, x0, ans_start, AL,
                                                  mask_id, dec, rng, device)
                    ok = (pred == gold).all(1)
                    d = {'acc': float(ok.float().mean()),
                         'n_remask_total': n_rm,
                         'remask_per_seq': n_rm / x0.shape[0]}
                    if recs:
                        d['trap_dep_correct'] = float(
                            (pred[dep_b, dep_c] == gold[dep_b, dep_c])
                            .float().mean())
                        d['trap_full_rescue'] = float(ok[dep_b].float().mean())
                    entry['decoders'][dec] = d
                results[key][name] = entry
                dstr = ' '.join(f"{k}={v['acc']:.3f}"
                                for k, v in entry['decoders'].items())
                print(f"  {name:>12s}  base={entry['base_acc']:.3f}  {dstr}"
                      f"  ({time.time()-t0:.0f}s)")
            del model, wrapper
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\n{'='*70}\n  TABLE R1b — published methods, default settings"
          f"\n{'='*70}")
    for key in [k for k in results if k.startswith('seed')]:
        print(f"\n── {key} ──")
        rows = results[key]
        cols = list(next(iter(rows.values()))['decoders'].keys())
        print(f"  {'bucket':>12s} {'base':>7s}" +
              ''.join(f" {c:>11s}" for c in cols))
        for name, e in rows.items():
            print(f"  {name:>12s} {e['base_acc']:>7.3f}" +
                  ''.join(f" {e['decoders'][c]['acc']:>11.3f}"
                          for c in cols))
    with open(args.out, 'w') as f:
        json.dump(_jsonable(results), f, indent=1)
    print(f"\n  💾 {args.out}")
    return results


if __name__ == '__main__':
    run(parse_args())
