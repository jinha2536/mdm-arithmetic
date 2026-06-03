"""Comprehensive analysis of confidence vs LSB-oracle decoding on addition."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

import os

import sys
if '__file__' in dir():
    _here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(_here))  # repo root
    sys.path.insert(0, _here)                    # experiments dir
else:
    sys.path.insert(0, '.')
from core.train_utils import (  # type: ignore
    encode_samples, generate_diffusion, simulate_reveal_trajectory, DEVICE,
)
from exp_addition import (  # type: ignore
    ND, ANS_LEN, build_tok,
    _annotate_sample, _bucket_from_samples,
    _fmt_plain, _pad, _parse_operands, _chain_stats, _max_chain_len,
    gen_min_chain_test, gen_corner_case_test,
)


# ── Configuration ────────────────────────────────────────────────────────────
METHODS = ["random", "papl", "puma"]
TOTAL_LEN = 2 * ND + 2 + ANS_LEN
N_HEAD_OVERRIDE = None   # set by --n_head CLI flag

# Math-position to string-position mapping (MSB-first text layout, math d=0=LSB)
def a_str_pos(d): return ND - 1 - d
def b_str_pos(d): return 2 * ND + 1 - d
def c_str_pos(d, ans_start=0): return ans_start + (ANS_LEN - 1 - d)


def load_model(ckpt_path, device):
    """Load checkpoint saved as a state_dict."""
    sd = torch.load(ckpt_path, map_location=device, weights_only=True)

    # Some checkpoints may wrap state_dict under a key
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    # Strip common prefixes
    if any(k.startswith("module.") for k in sd):
        sd = {k.removeprefix("module."): v for k, v in sd.items()}
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}

    # Infer architecture from shapes
    vocab_size, n_embd = sd["wte.weight"].shape
    has_wpe = "wpe.weight" in sd
    block_size = sd["wpe.weight"].shape[0] if has_wpe else 512
    n_layer = 0
    while f"blocks.{n_layer}.attn.c_attn.weight" in sd:
        n_layer += 1

    n_head = N_HEAD_OVERRIDE if N_HEAD_OVERRIDE is not None else 2
    pos_enc = "absolute" if has_wpe else "rope"

    print(f"  inferred arch: vocab={vocab_size} n_embd={n_embd} "
          f"block_size={block_size} n_layer={n_layer} n_head={n_head} "
          f"pos_enc={pos_enc}")

    # Use the project's own Transformer class to guarantee architecture match
    from core.model import Transformer  # type: ignore
    model = Transformer(
        vocab_size=vocab_size, block_size=block_size,
        n_layer=n_layer, n_head=n_head, n_embd=n_embd,
        dropout=0.0, is_causal=False, pos_enc=pos_enc,
    )
    missing, unexpected = model.load_state_dict(sd, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"state_dict mismatch:\n  missing: {missing}\n  unexpected: {unexpected}"
        )
    # ── WORKAROUND for tied-weight EMA bug ──
    # Older training code saved wte.weight (the EMA-updated, trained one)
    # and lm_head.weight (still at init due to named_parameters()
    # deduplicating tied params during EMA update) as separate tensors.
    # load_state_dict writes both into the same underlying memory and the
    # alphabetically-later key (lm_head.weight) wins, leaving the model
    # with an UNTRAINED output head. Force the trained wte.weight to win.
    if 'wte.weight' in sd:
        model.wte.weight.data.copy_(sd['wte.weight'].to(device))
    model.to(device).eval()
    return model


# ── Decode helper ────────────────────────────────────────────────────────────
@torch.no_grad()
def _decode_samples(model, tokenizer, samples, policy, device=None):
    """Wrap generate_diffusion: take list of full-sequence sample strings, extract the prefix (everything up to and including '='), run diffusion..."""
    device = device or DEVICE
    pad_id = tokenizer.special_ids['pad']
    mask_id = tokenizer.special_ids['mask']

    B = len(samples)
    if B == 0:
        return torch.empty(0, ANS_LEN, dtype=torch.long), \
               torch.empty(0, ANS_LEN, dtype=torch.long)

    penc = [tokenizer.encode(s.split("=")[0] + "=") for s in samples]
    pm = max(len(p) for p in penc)
    pids = torch.full((B, pm), pad_id, dtype=torch.long)
    for i, e in enumerate(penc):
        pids[i, : len(e)] = torch.tensor(e)
    pids = pids.to(device)

    # Gold answer ids
    ans_strs = [s.split("=")[1] for s in samples]
    gold = torch.full((B, ANS_LEN), pad_id, dtype=torch.long)
    for i, ans in enumerate(ans_strs):
        ids = tokenizer.encode(ans)
        gold[i, : len(ids)] = torch.tensor(ids)

    gen, _, _ = generate_diffusion(model, pids, ANS_LEN, mask_id,
                                   policy=policy, greedy=True, device=device)
    pred = gen[:, pm : pm + ANS_LEN].cpu()
    return pred, gold


#  A1  Per-instance correctness comparison
@torch.no_grad()
def a1_per_instance(model, tokenizer, bucket, device=None):
    """For each instance, record correctness under confidence vs LSB-oracle (LSB = r2l for plain MSB-first text layout)."""
    samples = bucket["samples"]
    metas = bucket["metas"]
    B = len(samples)
    if B == 0:
        return {"cross": {}, "per_chain": {}, "examples": {}, "n": 0}

    pred_conf, gold = _decode_samples(model, tokenizer, samples, "confidence", device)
    pred_lsb, _ = _decode_samples(model, tokenizer, samples, "r2l", device)

    cross = {"both_correct": 0, "only_lsb": 0, "only_conf": 0, "neither": 0}
    per_chain = defaultdict(lambda: dict(cross))
    failure_examples = {"only_lsb": [], "only_conf": [], "neither": []}

    for i in range(B):
        c_correct = bool(torch.equal(pred_conf[i], gold[i]))
        l_correct = bool(torch.equal(pred_lsb[i], gold[i]))
        if c_correct and l_correct:
            cat = "both_correct"
        elif l_correct and not c_correct:
            cat = "only_lsb"
        elif c_correct and not l_correct:
            cat = "only_conf"
        else:
            cat = "neither"
        cross[cat] += 1
        chain = metas[i]["chain_stats"]["max_chain_len"]
        per_chain[chain][cat] += 1
        if cat in failure_examples and len(failure_examples[cat]) < 20:
            failure_examples[cat].append({
                "a": metas[i]["a"], "b": metas[i]["b"],
                "max_chain": chain,
                "gkp": metas[i]["chain_stats"]["gkp"],
                "conf_pred": pred_conf[i].tolist(),
                "lsb_pred": pred_lsb[i].tolist(),
                "gold": gold[i].tolist(),
            })

    return {
        "cross": cross,
        "per_chain": {k: dict(v) for k, v in sorted(per_chain.items())},
        "examples": failure_examples,
        "n": B,
    }


#  A2  Reveal-order vs LSB-order divergence (Kendall tau)
@torch.no_grad()
def a2_kendall_tau(model, tokenizer, bucket, K=16, tau=0.9, device=None):
    """Kendall-tau between confidence-greedy reveal order (per instance) and LSB order (math digit position 0..ANS_LEN-1)."""
    device = device or DEVICE
    traj = simulate_reveal_trajectory(
        model, tokenizer, bucket["ids"].to(device),
        bucket["ans_starts"].to(device), ANS_LEN,
        blank_masks=None, K=K, tau=tau, device=device)

    reveal_stage = traj["reveal_stage"]  # [N, ANS_LEN]

    N = reveal_stage.shape[0]
    metas = bucket["metas"]
    by_chain = defaultdict(list)
    for i in range(N):
        # math-digit reveal order: sort positions by reveal_stage ascending
        # (ties broken by digit index)
        rs = reveal_stage[i].cpu().numpy()
        # math digit d at answer-position j = ANS_LEN - 1 - j
        # pairs: (d, reveal_stage)
        pairs = [(ANS_LEN - 1 - j, rs[j]) for j in range(ANS_LEN)]
        pairs.sort(key=lambda x: (x[1], x[0]))   # by stage then by d (LSB tie)
        actual_order = [p[0] for p in pairs]

        # Kendall tau vs LSB order [0, 1, 2, ..., ANS_LEN-1]
        tau_val = _kendall_tau(actual_order, list(range(ANS_LEN)))
        chain = metas[i]["chain_stats"]["max_chain_len"]
        by_chain[chain].append(tau_val)

    return {
        "by_chain": {
            k: {"mean_tau": sum(v) / len(v), "n": len(v),
                "min": min(v), "max": max(v)}
            for k, v in sorted(by_chain.items())
        },
        "overall_mean_tau": sum(t for vs in by_chain.values() for t in vs)
                          / max(sum(len(vs) for vs in by_chain.values()), 1),
        "K": K,
    }


def _kendall_tau(perm_a, perm_b):
    """Normalized Kendall tau-b for two equal-length permutations."""
    n = len(perm_a)
    pos_a = {x: i for i, x in enumerate(perm_a)}
    pos_b = {x: i for i, x in enumerate(perm_b)}
    concordant = discordant = 0
    keys = sorted(set(perm_a) & set(perm_b))
    for i, x in enumerate(keys):
        for y in keys[i + 1:]:
            da = pos_a[x] - pos_a[y]
            db = pos_b[x] - pos_b[y]
            if da * db > 0:
                concordant += 1
            elif da * db < 0:
                discordant += 1
    total = concordant + discordant
    return (concordant - discordant) / total if total > 0 else 0.0


#  A3  Per-stage commit-correctness during confidence decode
@torch.no_grad()
def a3_stage_correctness(model, tokenizer, bucket, K=16, tau=0.9, device=None):
    """At each confidence-greedy reveal stage, log whether the committed token equals the ground-truth, stratified by position role..."""
    device = device or DEVICE
    mask_id = tokenizer.special_ids["mask"]

    ids = bucket["ids"].to(device)
    ans_starts = bucket["ans_starts"].to(device)
    metas = bucket["metas"]
    N = ids.shape[0]

    # Mask all answer positions
    inp = ids.clone()
    for i in range(N):
        a0 = ans_starts[i].item()
        inp[i, a0:a0 + ANS_LEN] = mask_id

    # Confidence-greedy decode in K stages
    correct_by_stage_role = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    # role -> stage -> [n_correct, n_total]

    tokens_per_stage = max(1, ANS_LEN // K)
    for stage in range(K):
        logits = model(inp)
        # For each instance, find masked positions in answer region and pick
        # top-tokens_per_stage by max-prob.
        for i in range(N):
            a0 = ans_starts[i].item()
            slc = slice(a0, a0 + ANS_LEN)
            masked = (inp[i, slc] == mask_id).nonzero(as_tuple=True)[0]
            if masked.numel() == 0:
                continue
            probs = logits[i, slc][masked].softmax(-1)
            confs, preds = probs.max(-1)
            n_reveal = min(tokens_per_stage, masked.numel())
            top_idx = confs.topk(n_reveal).indices
            for k_idx in top_idx:
                pos_in_ans = masked[k_idx].item()       # answer-region offset
                pred_tok = preds[k_idx].item()
                gold_tok = ids[i, a0 + pos_in_ans].item()
                math_d = ANS_LEN - 1 - pos_in_ans
                # Look up role for this math digit
                dep_ctx = metas[i].get("dep_ctx", [])
                # dep_ctx is indexed by answer-position offset (matches j in
                # _gkp_at_answer_pos). So role at pos_in_ans:
                role = dep_ctx[pos_in_ans] if pos_in_ans < len(dep_ctx) else "?"
                correct_by_stage_role[role][stage][1] += 1
                correct_by_stage_role[role][stage][0] += int(pred_tok == gold_tok)
                inp[i, a0 + pos_in_ans] = pred_tok

    out = {}
    for role, stages in correct_by_stage_role.items():
        out[role] = {
            f"stage_{s}": {"n_correct": v[0], "n_total": v[1],
                          "acc": v[0] / max(v[1], 1)}
            for s, v in sorted(stages.items())
        }
    return out


#  A4  Confidence calibration on long chains
@torch.no_grad()
def a4_calibration(model, tokenizer, bucket, K=16, tau=0.9, device=None):
    """During confidence-greedy decode, log (confidence, correct?) for every committed token, stratified by chain length and position role."""
    device = device or DEVICE
    mask_id = tokenizer.special_ids["mask"]

    ids = bucket["ids"].to(device)
    ans_starts = bucket["ans_starts"].to(device)
    metas = bucket["metas"]
    N = ids.shape[0]

    inp = ids.clone()
    for i in range(N):
        a0 = ans_starts[i].item()
        inp[i, a0:a0 + ANS_LEN] = mask_id

    # Aggregate: chain_bin -> role -> [conf_correct_list, conf_wrong_list]
    agg = defaultdict(lambda: defaultdict(lambda: {"correct": [], "wrong": []}))

    tokens_per_stage = max(1, ANS_LEN // K)
    for stage in range(K):
        logits = model(inp)
        for i in range(N):
            a0 = ans_starts[i].item()
            slc = slice(a0, a0 + ANS_LEN)
            masked = (inp[i, slc] == mask_id).nonzero(as_tuple=True)[0]
            if masked.numel() == 0:
                continue
            probs = logits[i, slc][masked].softmax(-1)
            confs, preds = probs.max(-1)
            n_reveal = min(tokens_per_stage, masked.numel())
            top_idx = confs.topk(n_reveal).indices
            chain = metas[i]["chain_stats"]["max_chain_len"]
            chain_bin = (
                "<=4" if chain <= 4 else
                "5-12" if chain <= 12 else
                "13-20" if chain <= 20 else
                "21-28" if chain <= 28 else
                ">=29"
            )
            for k_idx in top_idx:
                pos_in_ans = masked[k_idx].item()
                pred_tok = preds[k_idx].item()
                gold_tok = ids[i, a0 + pos_in_ans].item()
                conf = confs[k_idx].item()
                dep_ctx = metas[i].get("dep_ctx", [])
                role = dep_ctx[pos_in_ans] if pos_in_ans < len(dep_ctx) else "?"
                key = "correct" if pred_tok == gold_tok else "wrong"
                agg[chain_bin][role][key].append(conf)
                inp[i, a0 + pos_in_ans] = pred_tok

    out = {}
    for chain_bin, roles in agg.items():
        out[chain_bin] = {}
        for role, d in roles.items():
            cs = d["correct"]; ws = d["wrong"]
            out[chain_bin][role] = {
                "n_correct": len(cs), "n_wrong": len(ws),
                "mean_conf_correct": sum(cs) / len(cs) if cs else None,
                "mean_conf_wrong":   sum(ws) / len(ws) if ws else None,
            }
    return out


#  A5  Adversarial sum-9 slice (deterministic)
def construct_sum9_chain(chain_start, chain_len, n, seed):
    """Build n instances where math positions [chain_start, chain_start+chain_len) are exactly sum-9 (propagate), the position immediately below..."""
    import random
    rng = random.Random(seed)
    samples = []
    seen = set()
    attempts = 0
    while len(samples) < n and attempts < n * 200:
        attempts += 1
        a_d = [0] * ND
        b_d = [0] * ND
        # Generate event below chain
        if chain_start >= 1:
            ga = rng.randint(1, 8)
            gb = rng.randint(10 - ga, 9)
            a_d[chain_start - 1] = ga
            b_d[chain_start - 1] = gb
        # Sum-9 chain
        for d in range(chain_start, chain_start + chain_len):
            if d >= ND:
                break
            ad = rng.randint(0, 9)
            a_d[d] = ad
            b_d[d] = 9 - ad
        # Kill events elsewhere
        for d in range(ND):
            if d == chain_start - 1:
                continue
            if chain_start <= d < chain_start + chain_len:
                continue
            ad = rng.randint(0, 9)
            bd = rng.randint(0, max(0, 8 - ad))
            a_d[d] = ad; b_d[d] = bd
        # MSB nonzero
        if a_d[ND - 1] == 0:
            a_d[ND - 1] = rng.randint(1, 8)
            b_d[ND - 1] = rng.randint(0, max(0, 8 - a_d[ND - 1]))
        if b_d[ND - 1] == 0:
            b_d[ND - 1] = rng.randint(1, max(1, 8 - a_d[ND - 1]))

        a = sum(d * 10**i for i, d in enumerate(a_d))
        b = sum(d * 10**i for i, d in enumerate(b_d))
        if (a, b) in seen:
            continue
        seen.add((a, b))
        # Verify chain length matches (debug guard)
        st = _chain_stats(a, b)
        if st["max_chain_len"] >= chain_len:
            samples.append(_fmt_plain(a, b))
    return samples


@torch.no_grad()
def a5_adversarial_slice(model, tokenizer, n_per_bucket=300, device=None):
    """Sweep chain_len in {4, 8, 12, 16, 20, 24, 28, 32}; evaluate confidence, lsb (=r2l), and uniform-random decoding on deterministically..."""
    chain_lens = [4, 8, 12, 16, 20, 24, 28, 32]

    out = {}
    for k in chain_lens:
        if k > ND:
            continue
        chain_start = max(1, (ND - k) // 2)
        samples = construct_sum9_chain(chain_start, k, n_per_bucket,
                                       seed=2026 + k)
        if not samples:
            continue
        for decode in ["confidence", "r2l", "random"]:
            pred, gold = _decode_samples(model, tokenizer, samples, decode, device)
            correct = sum(int(torch.equal(pred[i], gold[i]))
                          for i in range(pred.shape[0]))
            label = "lsb" if decode == "r2l" else decode
            out[f"k{k}_{label}"] = {
                "accuracy": correct / max(pred.shape[0], 1),
                "n": pred.shape[0],
                "chain_len_exact": k,
                "chain_start": chain_start,
            }
    return out


#  A6  Lookahead-window probe
@torch.no_grad()
def a6_lookahead_probe(model, tokenizer, n_per_bucket=300, device=None):
    """For each (chain_len k, window w), construct chain instances and measure one-shot prediction accuracy at chain interior digits when only..."""
    device = device or DEVICE
    mask_id = tokenizer.special_ids["mask"]

    chain_lens = [4, 8, 12, 16, 20, 24, 28, 30, 32]
    windows = [2, 4, 6, 8, 12, 16, 20, 24, 28]
    out = {}

    for k in chain_lens:
        if k > ND:
            continue
        # Place the chain so it fits in [chain_start, chain_start + k - 1] ⊂ [0, ND-1]
        chain_start = max(1, (ND - k) // 2) if k < ND else 0
        samples = construct_sum9_chain(chain_start, k, n_per_bucket, seed=3000 + k)
        if not samples:
            continue
        sample_strs = samples
        ids_all, ans_all = encode_samples(sample_strs, tokenizer, TOTAL_LEN)
        ids_all = ids_all.to(device); ans_all = ans_all.to(device)
        B = ids_all.shape[0]

        for w in windows:
            if w > ND:
                continue
            # Window anchored at chain MSB end, math positions [chain_start+k-w, chain_start+k-1]
            window_lo = chain_start + k - w
            window_hi = chain_start + k - 1
            inp = ids_all.clone()
            for i in range(B):
                # Mask operands outside window
                for d in range(ND):
                    if window_lo <= d <= window_hi:
                        continue
                    inp[i, a_str_pos(d)] = mask_id
                    inp[i, b_str_pos(d)] = mask_id
                # Mask all answer
                a0 = ans_all[i].item()
                inp[i, a0:a0 + ANS_LEN] = mask_id

            logits = model(inp)
            # Eval at carry-out position (math d = chain_start + k)
            target_d = chain_start + k
            if target_d >= ANS_LEN:
                continue
            correct = 0
            for i in range(B):
                a0 = ans_all[i].item()
                pos = a0 + (ANS_LEN - 1 - target_d)
                pred = logits[i, pos].argmax().item()
                gold = ids_all[i, pos].item()
                correct += int(pred == gold)
            out[f"k{k}_w{w}"] = {
                "accuracy": correct / max(B, 1),
                "n": B,
                "target_math_d": target_d,
                "window_lo": window_lo, "window_hi": window_hi,
            }

    return out


@torch.no_grad()
def a8_failure_dissection(model, tokenizer, bucket, max_examples=50, device=None):
    """For each instance, run BOTH confidence-greedy and LSB-oracle decode, logging the full per-stage trace: which position was committed at each..."""
    device = device or DEVICE
    pad_id = tokenizer.special_ids["pad"]
    mask_id = tokenizer.special_ids["mask"]

    samples = bucket["samples"]
    metas = bucket["metas"]
    B = len(samples)
    if B == 0:
        return {"only_lsb_details": [], "summary": {}}

    # Encode prefixes
    penc = [tokenizer.encode(s.split("=")[0] + "=") for s in samples]
    pm = max(len(p) for p in penc)
    pids = torch.full((B, pm), pad_id, dtype=torch.long)
    for i, e in enumerate(penc):
        pids[i, : len(e)] = torch.tensor(e)
    pids = pids.to(device)

    # Gold answer
    ans_strs = [s.split("=")[1] for s in samples]
    gold_ans = torch.full((B, ANS_LEN), pad_id, dtype=torch.long)
    for i, ans in enumerate(ans_strs):
        ids = tokenizer.encode(ans)
        gold_ans[i, : len(ids)] = torch.tensor(ids)
    gold_ans = gold_ans.to(device)

    # Trace both decode policies side by side, fully manual loop so we can
    # capture per-stage confidence distributions.
    def _trace_decode(policy):
        """Returns list of dicts (one per example) with per-stage trace."""
        T_pre = pids.shape[1]
        T = T_pre + ANS_LEN
        x = torch.full((B, T), mask_id, dtype=torch.long, device=device)
        x[:, :T_pre] = pids
        unmasked = torch.zeros(B, T, dtype=torch.bool, device=device)
        unmasked[:, :T_pre] = True

        # Per-example trace storage
        traces = [[] for _ in range(B)]

        for stage in range(ANS_LEN):
            logits = model(x)
            logits[:, :, mask_id] = -float("inf")

            if policy == "confidence":
                max_logit = logits.max(dim=-1).values
                max_logit[unmasked] = -float("inf")
                pos = max_logit.argmax(-1)
            elif policy == "r2l":
                pos = torch.full((B,), T_pre + ANS_LEN - 1 - stage,
                                 dtype=torch.long, device=device)
            else:
                raise ValueError(policy)

            # Per-example commit details
            for i in range(B):
                p = pos[i].item()
                if unmasked[i, p]:
                    continue  # already revealed (shouldn't happen)
                # Probability distribution at this position
                probs = F.softmax(logits[i, p], dim=-1)
                top2 = probs.topk(2)
                top1_prob = top2.values[0].item()
                top2_prob = top2.values[1].item()
                top1_tok = top2.indices[0].item()

                # Gold token at this position
                ans_offset = p - T_pre  # answer-region offset
                gold_tok = gold_ans[i, ans_offset].item()
                gold_prob = probs[gold_tok].item()

                # Math digit (LSB=0)
                math_d = ANS_LEN - 1 - ans_offset
                # Role (from dep_ctx, which is indexed by ans-region offset)
                dep_ctx = metas[i].get("dep_ctx", [])
                role = dep_ctx[ans_offset] if ans_offset < len(dep_ctx) else "?"

                traces[i].append({
                    "stage": stage,
                    "ans_offset": ans_offset,
                    "math_d": math_d,
                    "role": role,
                    "committed_tok": top1_tok,
                    "gold_tok": gold_tok,
                    "is_correct": (top1_tok == gold_tok),
                    "top1_prob": top1_prob,
                    "top2_prob": top2_prob,
                    "gold_prob": gold_prob,
                    "margin": top1_prob - top2_prob,
                })

            # Commit the predicted token at chosen positions
            B_ar = torch.arange(B, device=device)
            top_pred = logits[B_ar, pos].argmax(-1)
            x[B_ar, pos] = top_pred
            unmasked[B_ar, pos] = True

        return traces, x[:, T_pre:].cpu()

    print("    [a8] running confidence decode trace...")
    conf_traces, conf_preds = _trace_decode("confidence")
    print("    [a8] running r2l (LSB) decode trace...")
    lsb_traces, lsb_preds = _trace_decode("r2l")
    gold_cpu = gold_ans.cpu()

    # Categorize and assemble output
    only_lsb_examples = []
    only_conf_examples = []
    neither_examples = []
    summary = {"both_correct": 0, "only_lsb": 0, "only_conf": 0, "neither": 0}

    for i in range(B):
        c_correct = bool(torch.equal(conf_preds[i], gold_cpu[i]))
        l_correct = bool(torch.equal(lsb_preds[i], gold_cpu[i]))
        if c_correct and l_correct:
            cat = "both_correct"
        elif l_correct and not c_correct:
            cat = "only_lsb"
        elif c_correct and not l_correct:
            cat = "only_conf"
        else:
            cat = "neither"
        summary[cat] += 1

        if cat == "both_correct":
            continue

        # Identify divergence: first answer-region offset where conf_pred != gold
        conf_wrong_offsets = [
            j for j in range(ANS_LEN)
            if conf_preds[i, j].item() != gold_cpu[i, j].item()
        ]
        if not conf_wrong_offsets:
            continue
        first_wrong = conf_wrong_offsets[0]

        # Find conf-trace entry that committed this offset
        conf_commit = next((t for t in conf_traces[i]
                            if t["ans_offset"] == first_wrong), None)
        # Find LSB-trace entry for same offset
        lsb_commit = next((t for t in lsb_traces[i]
                           if t["ans_offset"] == first_wrong), None)

        # What did conf decode reveal *before* this stage?
        conf_stage_of_wrong = conf_commit["stage"] if conf_commit else None
        if conf_stage_of_wrong is not None:
            preceding = [
                {"stage": t["stage"], "ans_offset": t["ans_offset"],
                 "math_d": t["math_d"], "role": t["role"],
                 "tok": t["committed_tok"], "gold": t["gold_tok"],
                 "correct": t["is_correct"], "top1_prob": t["top1_prob"]}
                for t in conf_traces[i] if t["stage"] < conf_stage_of_wrong
            ]
        else:
            preceding = []

        record = {
            "instance_idx": i,
            "a": metas[i]["a"], "b": metas[i]["b"],
            "max_chain": metas[i]["chain_stats"]["max_chain_len"],
            "gkp": metas[i]["chain_stats"]["gkp"],
            "all_conf_wrong_offsets": conf_wrong_offsets,
            "first_wrong_offset": first_wrong,
            "first_wrong_math_d": ANS_LEN - 1 - first_wrong,
            "first_wrong_role": (metas[i].get("dep_ctx", [])[first_wrong]
                                 if first_wrong < len(metas[i].get("dep_ctx", []))
                                 else "?"),
            "conf_commit": conf_commit,
            "lsb_commit": lsb_commit,
            "conf_preceding_reveals": preceding,
            "n_conf_wrong_total": len(conf_wrong_offsets),
        }

        target = (only_lsb_examples if cat == "only_lsb"
                  else only_conf_examples if cat == "only_conf"
                  else neither_examples)
        if len(target) < max_examples:
            target.append(record)

    return {
        "summary": summary,
        "only_lsb_examples": only_lsb_examples,
        "only_conf_examples": only_conf_examples,
        "neither_examples": neither_examples,
        "n_total": B,
    }


#  A9  Confidence-ranking margin diagnostic (extends A8 with cross-position info)
@torch.no_grad()
def a9_ranking_margin(model, tokenizer, bucket, max_examples=50, device=None):
    """At each confidence-greedy reveal stage, log not only the chosen position's top1_prob, but also the runner-up's top1_prob (the next-best..."""
    device = device or DEVICE
    pad_id = tokenizer.special_ids["pad"]
    mask_id = tokenizer.special_ids["mask"]

    samples = bucket["samples"]
    metas = bucket["metas"]
    B = len(samples)
    if B == 0:
        return {"records": [], "summary": {}}

    penc = [tokenizer.encode(s.split("=")[0] + "=") for s in samples]
    pm = max(len(p) for p in penc)
    pids = torch.full((B, pm), pad_id, dtype=torch.long)
    for i, e in enumerate(penc):
        pids[i, : len(e)] = torch.tensor(e)
    pids = pids.to(device)

    ans_strs = [s.split("=")[1] for s in samples]
    gold_ans = torch.full((B, ANS_LEN), pad_id, dtype=torch.long)
    for i, ans in enumerate(ans_strs):
        ids = tokenizer.encode(ans)
        gold_ans[i, : len(ids)] = torch.tensor(ids)
    gold_ans = gold_ans.to(device)

    T_pre = pids.shape[1]
    T = T_pre + ANS_LEN

    x = torch.full((B, T), mask_id, dtype=torch.long, device=device)
    x[:, :T_pre] = pids
    unmasked = torch.zeros(B, T, dtype=torch.bool, device=device)
    unmasked[:, :T_pre] = True

    traces = [[] for _ in range(B)]
    stage_of_first_wrong = [None] * B
    full_ranking_at_wrong = [None] * B
    final_pred = torch.full((B, ANS_LEN), -1, dtype=torch.long, device=device)

    for stage in range(ANS_LEN):
        logits = model(x)
        logits[:, :, mask_id] = -float("inf")

        max_logit, top1_tok = logits.max(dim=-1)  # both [B, T]
        # For per-stage diagnostics we still want top1_prob as a human-readable
        # number ,  compute it but don't use for ranking.
        probs = F.softmax(logits, dim=-1)  # [B, T, V]
        top1_prob_all = probs.max(dim=-1).values  # [B, T]

        # For each instance, find the masked positions and rank them by max_logit
        for i in range(B):
            ans_slc = slice(T_pre, T_pre + ANS_LEN)
            still_masked_in_ans = (~unmasked[i, ans_slc]).nonzero(as_tuple=True)[0]
            if still_masked_in_ans.numel() == 0:
                continue

            masked_abs = T_pre + still_masked_in_ans
            scores = max_logit[i, masked_abs]  # [n_masked] ,  use logits for ranking

            sorted_scores, sorted_idx = scores.sort(descending=True)
            chosen_local = sorted_idx[0].item()
            chosen_ans_offset = still_masked_in_ans[chosen_local].item()
            chosen_top1 = top1_prob_all[i, T_pre + chosen_ans_offset].item()
            chosen_logit = sorted_scores[0].item()
            chosen_math_d = ANS_LEN - 1 - chosen_ans_offset
            dep_ctx = metas[i].get("dep_ctx", [])
            chosen_role = dep_ctx[chosen_ans_offset] if chosen_ans_offset < len(dep_ctx) else "?"

            runner_top1 = None; runner_math_d = None; runner_role = None; runner_logit = None
            if sorted_scores.numel() > 1:
                runner_local = sorted_idx[1].item()
                runner_ans_offset = still_masked_in_ans[runner_local].item()
                runner_top1 = top1_prob_all[i, T_pre + runner_ans_offset].item()
                runner_logit = sorted_scores[1].item()
                runner_math_d = ANS_LEN - 1 - runner_ans_offset
                runner_role = dep_ctx[runner_ans_offset] if runner_ans_offset < len(dep_ctx) else "?"

            # Margin in BOTH spaces ,  logit margin is the "true" decision margin,
            # prob margin is the human-readable size.
            ranking_margin_prob = (chosen_top1 - runner_top1) if runner_top1 is not None else None
            ranking_margin_logit = (chosen_logit - runner_logit) if runner_logit is not None else None

            # Was this commit wrong?
            chosen_pred_tok = top1_tok[i, T_pre + chosen_ans_offset].item()
            chosen_gold_tok = gold_ans[i, chosen_ans_offset].item()
            is_wrong = (chosen_pred_tok != chosen_gold_tok)

            traces[i].append({
                "stage": stage,
                "chosen_ans_offset": chosen_ans_offset,
                "chosen_math_d": chosen_math_d,
                "chosen_role": chosen_role,
                "chosen_top1": chosen_top1,
                "chosen_logit": chosen_logit,
                "chosen_correct": (not is_wrong),
                "runner_math_d": runner_math_d,
                "runner_role": runner_role,
                "runner_top1": runner_top1,
                "runner_logit": runner_logit,
                "ranking_margin_prob":  ranking_margin_prob,
                "ranking_margin_logit": ranking_margin_logit,
            })

            # If this is the first wrong commit, dump the full top-10 ranking
            if is_wrong and stage_of_first_wrong[i] is None:
                stage_of_first_wrong[i] = stage
                top_k = min(10, sorted_scores.numel())
                ranking = []
                for r in range(top_k):
                    local = sorted_idx[r].item()
                    ans_off = still_masked_in_ans[local].item()
                    md = ANS_LEN - 1 - ans_off
                    rl = dep_ctx[ans_off] if ans_off < len(dep_ctx) else "?"
                    ranking.append({
                        "rank": r,
                        "math_d": md,
                        "ans_offset": ans_off,
                        "role": rl,
                        "top1_prob": top1_prob_all[i, T_pre + ans_off].item(),
                        "max_logit":  sorted_scores[r].item(),
                        "would_be_correct": (
                            top1_tok[i, T_pre + ans_off].item()
                            == gold_ans[i, ans_off].item()
                        ),
                    })
                full_ranking_at_wrong[i] = ranking

            # Commit
            x[i, T_pre + chosen_ans_offset] = chosen_pred_tok
            unmasked[i, T_pre + chosen_ans_offset] = True
            final_pred[i, chosen_ans_offset] = chosen_pred_tok

    # Aggregate output
    summary = {"both_correct": 0, "only_lsb_or_neither_or_only_conf": 0}
    records = []  # only the interesting (wrong-commit-occurred) instances

    for i in range(B):
        all_correct = bool(torch.equal(final_pred[i].cpu(), gold_ans[i].cpu()))
        if all_correct:
            summary["both_correct"] += 1
            continue
        summary["only_lsb_or_neither_or_only_conf"] += 1
        if len(records) >= max_examples:
            continue

        rec = {
            "instance_idx": i,
            "a": metas[i]["a"], "b": metas[i]["b"],
            "max_chain": metas[i]["chain_stats"]["max_chain_len"],
            "stage_of_first_wrong": stage_of_first_wrong[i],
            "full_ranking_at_wrong_stage": full_ranking_at_wrong[i],
            "preceding_trace": traces[i][: stage_of_first_wrong[i] + 1]
                              if stage_of_first_wrong[i] is not None else [],
        }
        records.append(rec)

    all_margin_logit = []
    all_margin_prob = []
    for i in range(B):
        for t in traces[i]:
            if t.get("ranking_margin_logit") is None: continue
            all_margin_logit.append(t["ranking_margin_logit"])
            all_margin_prob.append(t["ranking_margin_prob"])

    wrong_margins_logit = []
    wrong_margins_prob = []
    for i in range(B):
        if stage_of_first_wrong[i] is None: continue
        s = stage_of_first_wrong[i]
        if s < len(traces[i]) and traces[i][s].get("ranking_margin_logit") is not None:
            wrong_margins_logit.append(traces[i][s]["ranking_margin_logit"])
            wrong_margins_prob.append(traces[i][s]["ranking_margin_prob"])

    def _stats(xs):
        if not xs: return {"mean": None, "min": None, "max": None}
        return {"mean": sum(xs)/len(xs), "min": min(xs), "max": max(xs)}

    return {
        "summary": summary,
        "records": records,
        "n_total": B,
        "all_stage_n":   len(all_margin_logit),
        "wrong_stage_n": len(wrong_margins_logit),
        "all_stage_margin_logit": {
            **_stats(all_margin_logit),
            "frac_lt_0.01": (sum(m < 0.01 for m in all_margin_logit) / len(all_margin_logit)) if all_margin_logit else None,
            "frac_lt_0.1":  (sum(m < 0.1  for m in all_margin_logit) / len(all_margin_logit)) if all_margin_logit else None,
            "frac_lt_1.0":  (sum(m < 1.0  for m in all_margin_logit) / len(all_margin_logit)) if all_margin_logit else None,
        },
        "all_stage_margin_prob": {
            **_stats(all_margin_prob),
            "frac_lt_1e-4": (sum(m < 1e-4 for m in all_margin_prob) / len(all_margin_prob)) if all_margin_prob else None,
            "frac_lt_1e-3": (sum(m < 1e-3 for m in all_margin_prob) / len(all_margin_prob)) if all_margin_prob else None,
        },
        "wrong_stage_margin_logit": _stats(wrong_margins_logit),
        "wrong_stage_margin_prob":  _stats(wrong_margins_prob),
    }


#  Driver
ALL_ANALYSES = {
    "a1": "Per-instance correctness comparison",
    "a2": "Reveal-order Kendall-tau vs LSB",
    "a3": "Per-stage commit correctness",
    "a4": "Confidence calibration",
    "a5": "Adversarial sum-9 slice",
    "a6": "Lookahead-window probe",
    "a8": "Confidence-failure dissection (full per-example trace)",
    "a9": "Confidence-ranking margin diagnostic (chosen vs runner-up)",
}


import re
_CKPT_RE = re.compile(
    r'^checkpoint_seed(?P<seed>\d+)_(?P<method>[a-z_]+)_iter(?P<it>\d+)\.pt$'
)


def _discover_checkpoints(ckpt_dir: Path):
    """Walk a checkpoint directory and group ckpt files by (seed, method, iter).

    Returns a sorted list of dicts: {seed, method, iter, path}.
    Files not matching the seed{N}_{method}_iter{NNNNNN} convention are ignored.
    """
    out = []
    for f in sorted(ckpt_dir.glob('checkpoint_seed*_iter*.pt')):
        m = _CKPT_RE.match(f.name)
        if not m:
            continue
        out.append({
            'seed': int(m['seed']),
            'method': m['method'],
            'iter': int(m['it']),
            'path': f,
        })
    out.sort(key=lambda d: (d['seed'], d['method'], d['iter']))
    return out


def _filter_ckpts(ckpts, seed=None, methods=None, iters=None):
    out = ckpts
    if seed is not None:
        out = [c for c in out if c['seed'] == seed]
    if methods:
        s = set(methods)
        out = [c for c in out if c['method'] in s]
    if iters:
        s = set(iters)
        out = [c for c in out if c['iter'] in s]
    return out


def _run_analyses(model, tokenizer, chain_buckets, selected, args, device):
    """Run the selected analyses against one model. Returns the results dict."""
    results = {"n_per_bucket": args.n_per_bucket}

    if "a1" in selected:
        print("    A1: per-instance correctness comparison")
        results["a1"] = {f"chain_{k}": a1_per_instance(model, tokenizer, b, device=device)
                         for k, b in chain_buckets.items()}
    if "a2" in selected:
        print("    A2: Kendall tau vs LSB order")
        results["a2"] = {f"chain_{k}": a2_kendall_tau(model, tokenizer, b,
                                                      K=args.K_puma, tau=args.tau, device=device)
                         for k, b in chain_buckets.items()}
    if "a3" in selected:
        print("    A3: per-stage commit correctness")
        results["a3"] = {f"chain_{k}": a3_stage_correctness(model, tokenizer, b,
                                                            K=args.K_decode, tau=args.tau, device=device)
                         for k, b in chain_buckets.items()}
    if "a4" in selected:
        print("    A4: confidence calibration")
        all_samples = [s for b in chain_buckets.values() for s in b["samples"]]
        big_bucket = _bucket_from_samples(all_samples, tokenizer, TOTAL_LEN)
        results["a4"] = a4_calibration(model, tokenizer, big_bucket,
                                       K=args.K_decode, tau=args.tau, device=device)
    if "a5" in selected:
        print("    A5: adversarial sum-9 slice")
        results["a5"] = a5_adversarial_slice(model, tokenizer,
                                             n_per_bucket=args.n_per_bucket, device=device)
    if "a6" in selected:
        print("    A6: lookahead-window probe")
        results["a6"] = a6_lookahead_probe(model, tokenizer,
                                           n_per_bucket=args.n_per_bucket, device=device)
    if "a8" in selected:
        print("    A8: confidence-failure dissection")
        results["a8"] = {f"chain_{k}": a8_failure_dissection(model, tokenizer, b,
                                                              max_examples=50, device=device)
                         for k, b in chain_buckets.items()}
    if "a9" in selected:
        print("    A9: confidence-ranking margin diagnostic")
        results["a9"] = {f"chain_{k}": a9_ranking_margin(model, tokenizer, b,
                                                          max_examples=50, device=device)
                         for k, b in chain_buckets.items()}
    return results


def _process_one_dir(checkpoint_dir, out_dir, args, tokenizer, chain_buckets,
                     selected, device):
    """Run analyses on all (filtered) checkpoints in one directory.

    Returns the number of analyses written. Skips files that already exist
    in out_dir (resume-friendly).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  Checkpoint dir: {checkpoint_dir}")
    print(f"  Out dir:        {out_dir}")

    if args.legacy_single:
        print(f"  [legacy-single mode]")
        n_written = 0
        for method in METHODS:
            ckpt = checkpoint_dir / f"checkpoint_{method}.pt"
            if not ckpt.exists():
                print(f"    [skip] {ckpt.name} not found"); continue
            out_path = out_dir / f"analysis_{method}.json"
            if out_path.exists() and not args.force:
                print(f"    [skip] {out_path.name} already exists"); continue
            print(f"    === {method} ===")
            model = load_model(ckpt, device)
            results = {"method": method, **_run_analyses(
                model, tokenizer, chain_buckets, selected, args, device)}
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"    wrote {out_path.name}")
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            n_written += 1
        return n_written

    # Multi-checkpoint mode
    all_ckpts = _discover_checkpoints(checkpoint_dir)
    if not all_ckpts:
        print(f"    [no ckpts] No checkpoint_seed*_iter*.pt in {checkpoint_dir}")
        return 0

    ckpts = _filter_ckpts(all_ckpts, seed=args.seed,
                           methods=args.methods, iters=args.iters)
    print(f"    Discovered {len(all_ckpts)} ckpts, "
          f"{len(ckpts)} after filtering")
    if not ckpts:
        return 0

    seeds_seen = sorted(set(c['seed'] for c in ckpts))
    methods_seen = sorted(set(c['method'] for c in ckpts))
    iters_seen = sorted(set(c['iter'] for c in ckpts))
    print(f"    seeds:   {seeds_seen}")
    print(f"    methods: {methods_seen}")
    print(f"    iters:   {iters_seen}")

    n_written = 0
    for ci, c in enumerate(ckpts):
        out_path = out_dir / (
            f"analysis_seed{c['seed']}_{c['method']}_iter{c['iter']:06d}.json")
        if out_path.exists() and not args.force:
            print(f"    [{ci+1}/{len(ckpts)}] skip (exists): {out_path.name}")
            continue
        print(f"    [{ci+1}/{len(ckpts)}] seed={c['seed']} {c['method']} iter={c['iter']}")
        model = load_model(c['path'], device)
        results = {
            "seed": c['seed'], "method": c['method'], "iter": c['iter'],
            **_run_analyses(model, tokenizer, chain_buckets, selected, args, device),
        }
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        n_written += 1
    return n_written


def _discover_experiment_dirs(root):
    """Find subdirs of `root` that contain at least one ckpt-pattern file.

    A "ckpt-pattern" file is either `checkpoint_seed*_iter*.pt`
    (multi-checkpoint format) or `checkpoint_{random,papl,puma}.pt`
    (legacy format). Returns a sorted list of Path objects.
    """
    out = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        has_multi = any(d.glob('checkpoint_seed*_iter*.pt'))
        has_legacy = any((d / f'checkpoint_{m}.pt').exists() for m in METHODS)
        if has_multi or has_legacy:
            out.append(d)
    return out


def main():
    ap = argparse.ArgumentParser()
    # Either --checkpoint_dir (single experiment) OR --checkpoint_root (parent
    # containing multiple experiment dirs). Exactly one of the two is required.
    ap.add_argument("--checkpoint_dir", type=Path, default=None,
                    help="Single experiment directory containing ckpt files")
    ap.add_argument("--checkpoint_root", type=Path, default=None,
                    help="Parent directory containing multiple experiment "
                         "subdirs. Each subdir with ckpt files is analyzed "
                         "sequentially; analyses go to <subdir>/analysis/.")
    ap.add_argument("--out_dir", default=None, type=Path,
                    help="Output dir for analyses. Default: "
                         "<checkpoint_dir>/analysis or <subdir>/analysis "
                         "when using --checkpoint_root.")
    ap.add_argument("--analyses", default="all",
                    help="Comma-separated subset of " + ",".join(ALL_ANALYSES))
    ap.add_argument("--n_per_bucket", default=300, type=int)
    ap.add_argument("--K_decode", default=ANS_LEN, type=int,
                    help="Stages for inference-time decode-trace analyses "
                         "(A3, A4). Default ANS_LEN = one token per stage.")
    ap.add_argument("--K_puma", default=16, type=int,
                    help="Stages for PUMA-style reveal trajectory simulation (A2).")
    ap.add_argument("--tau", default=0.9, type=float)
    ap.add_argument("--n_head", default=None, type=int,
                    help="Override n_head when loading checkpoints.")
    ap.add_argument("--seed", type=int, default=None,
                    help="Filter to checkpoints from this seed only")
    ap.add_argument("--methods", nargs='+', default=None,
                    help="Filter to these methods (e.g. --methods random puma)")
    ap.add_argument("--iters", nargs='+', type=int, default=None,
                    help="Filter to these iters (e.g. --iters 10000 100000 300000)")
    ap.add_argument("--legacy-single", action='store_true',
                    help="Use legacy mode: load checkpoint_{method}.pt (one per method, no iter)")
    ap.add_argument("--force", action='store_true',
                    help="Overwrite existing analysis JSONs (default: skip if exists)")
    args = ap.parse_args()

    if (args.checkpoint_dir is None) == (args.checkpoint_root is None):
        ap.error("Specify exactly one of --checkpoint_dir or --checkpoint_root")

    if args.n_head is not None:
        global N_HEAD_OVERRIDE
        N_HEAD_OVERRIDE = args.n_head
    selected = (list(ALL_ANALYSES) if args.analyses == "all"
                else args.analyses.split(","))

    tokenizer = build_tok()
    device = DEVICE
    print(f"Device: {device}")
    print(f"Selected analyses: {selected}")

    # Test buckets are built once and reused across all experiment dirs —
    # they're driven only by ND, n_per_bucket, and seeded RNG, all of which
    # are constant within an invocation. Saves substantial time when looping.
    print("\nBuilding test buckets (chain sweep, including extreme tail)...")
    chain_buckets = {}
    for k in [4, 8, 12, 16, 20, 24, 28, 30, 32]:
        if k > ND: continue
        sp = gen_min_chain_test(args.n_per_bucket, seed=5000 + k, min_chain=k)
        if sp:
            chain_buckets[k] = _bucket_from_samples(sp, tokenizer, TOTAL_LEN)
            print(f"  chain>={k}: {chain_buckets[k]['n']} samples")

    # Determine list of (checkpoint_dir, out_dir) pairs to process.
    if args.checkpoint_root is not None:
        exp_dirs = _discover_experiment_dirs(args.checkpoint_root)
        if not exp_dirs:
            print(f"\n[error] No experiment subdirs found under {args.checkpoint_root}")
            return
        print(f"\nDiscovered {len(exp_dirs)} experiment dirs under "
              f"{args.checkpoint_root}:")
        for d in exp_dirs:
            print(f"  {d.name}")
        # When using checkpoint_root, default out_dir is <subdir>/analysis
        # unless user provided an explicit override (in which case all
        # analyses go into that single dir, prefixed by exp name).
        targets = []
        for d in exp_dirs:
            if args.out_dir is not None:
                # explicit override: nest by exp name to avoid collisions
                od = args.out_dir / d.name
            else:
                od = d / 'analysis'
            targets.append((d, od))
    else:
        od = args.out_dir or (args.checkpoint_dir / 'analysis')
        targets = [(args.checkpoint_dir, od)]

    # Process each (checkpoint_dir, out_dir) pair sequentially.
    total_written = 0
    for ti, (cdir, odir) in enumerate(targets):
        print(f"\n{'='*70}\n  [{ti+1}/{len(targets)}] {cdir.name}\n{'='*70}")
        n = _process_one_dir(cdir, odir, args, tokenizer, chain_buckets,
                              selected, device)
        total_written += n
        print(f"  wrote {n} analyses to {odir.name}/")

    print(f"\n{'='*70}")
    print(f"  ALL DONE. {total_written} analyses written across {len(targets)} dirs.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
