"""Failure analyses of confidence decoding on addition checkpoints, run on
carry-chain strata (chain >= k for k in 4..32, --n_per_bucket instances each,
generated with seed 5000 + k):

  a1  per-instance correctness under confidence vs LSB-first decoding
  a3  commit correctness at each confidence-greedy stage, by position role
  a4  confidence of correct vs wrong commits, by chain length and position role
  a8  failure traces: first wrong commit (position, role, committed vs gold
      digit, top-1 probability) and the commits preceding it
  a9  ranking at the first wrong commit (chosen vs runner-up positions and the
      top-10 candidates)

Usage (from the repository root):
    python experiments/addition_decode_analysis.py --checkpoint_dir results/exp_addition
Reads checkpoint_seed{S}_{method}_iter{N}.pt (or checkpoint_{method}.pt with
--legacy-single) and writes one analysis_*.json per checkpoint to
<checkpoint_dir>/analysis/."""
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
from core.train_utils import generate_diffusion, DEVICE  # type: ignore
from exp_addition import (  # type: ignore
    ND, ANS_LEN, build_tok, _bucket_from_samples, gen_min_chain_test,
)


# Configuration
METHODS = ["random", "papl", "puma"]
TOTAL_LEN = 2 * ND + 2 + ANS_LEN
N_HEAD_OVERRIDE = None   # set by --n_head CLI flag


def load_model(ckpt_path, device):
    """Load checkpoint saved as a state_dict."""
    sd = torch.load(ckpt_path, map_location=device, weights_only=True)

    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    if any(k.startswith("module.") for k in sd):
        sd = {k.removeprefix("module."): v for k, v in sd.items()}
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}

    # architecture inferred from tensor shapes (n_head is not recoverable)
    vocab_size, n_embd = sd["wte.weight"].shape
    block_size = sd["wpe.weight"].shape[0]
    n_layer = 0
    while f"blocks.{n_layer}.attn.c_attn.weight" in sd:
        n_layer += 1

    n_head = N_HEAD_OVERRIDE if N_HEAD_OVERRIDE is not None else 2

    print(f"  inferred arch: vocab={vocab_size} n_embd={n_embd} "
          f"block_size={block_size} n_layer={n_layer} n_head={n_head}")

    from core.model import Transformer  # type: ignore
    model = Transformer(
        vocab_size=vocab_size, block_size=block_size,
        n_layer=n_layer, n_head=n_head, n_embd=n_embd, dropout=0.0,
    )
    missing, unexpected = model.load_state_dict(sd, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"state_dict mismatch:\n  missing: {missing}\n  unexpected: {unexpected}"
        )
    # tied-weight guard: wte.weight and lm_head.weight share storage; if they
    # differ in the checkpoint, keep wte.weight (load_state_dict writes lm_head last)
    if 'wte.weight' in sd:
        model.wte.weight.data.copy_(sd['wte.weight'].to(device))
    model.to(device).eval()
    return model


# Decode helper
@torch.no_grad()
def _decode_samples(model, tokenizer, samples, policy, device=None):
    """Decode the answers of full-sequence sample strings with generate_diffusion;
    returns (pred, gold) answer-token tensors."""
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

    ans_strs = [s.split("=")[1] for s in samples]
    gold = torch.full((B, ANS_LEN), pad_id, dtype=torch.long)
    for i, ans in enumerate(ans_strs):
        ids = tokenizer.encode(ans)
        gold[i, : len(ids)] = torch.tensor(ids)

    gen, _, _ = generate_diffusion(model, pids, ANS_LEN, mask_id,
                                   policy=policy, device=device)
    pred = gen[:, pm : pm + ANS_LEN].cpu()
    return pred, gold


#  A1  Per-instance correctness comparison
@torch.no_grad()
def a1_per_instance(model, tokenizer, bucket, device=None):
    """Per-instance correctness under confidence vs LSB-first (r2l) decoding."""
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


#  A3  Per-stage commit-correctness during confidence decode
@torch.no_grad()
def a3_stage_correctness(model, tokenizer, bucket, K=16, tau=0.9, device=None):
    """Commit correctness at each confidence-greedy reveal stage, by position role."""
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

    # confidence-greedy decode in K stages
    correct_by_stage_role = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    # role -> stage -> [n_correct, n_total]

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
            for k_idx in top_idx:
                pos_in_ans = masked[k_idx].item()       # answer-region offset
                pred_tok = preds[k_idx].item()
                gold_tok = ids[i, a0 + pos_in_ans].item()
                dep_ctx = metas[i].get("dep_ctx", [])
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
    """(confidence, correct) of every committed token during confidence-greedy
    decoding, by chain length and position role."""
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


@torch.no_grad()
def a8_failure_dissection(model, tokenizer, bucket, max_examples=50, device=None):
    """Per-stage traces of confidence-greedy and LSB-first decoding for the
    instances where the two disagree (first wrong commit, its confidence, and the
    commits preceding it)."""
    device = device or DEVICE
    pad_id = tokenizer.special_ids["pad"]
    mask_id = tokenizer.special_ids["mask"]

    samples = bucket["samples"]
    metas = bucket["metas"]
    B = len(samples)
    if B == 0:
        return {"only_lsb_details": [], "summary": {}}

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

    def _trace_decode(policy):
        """Returns list of dicts (one per example) with per-stage trace."""
        T_pre = pids.shape[1]
        T = T_pre + ANS_LEN
        x = torch.full((B, T), mask_id, dtype=torch.long, device=device)
        x[:, :T_pre] = pids
        unmasked = torch.zeros(B, T, dtype=torch.bool, device=device)
        unmasked[:, :T_pre] = True

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

            for i in range(B):
                p = pos[i].item()
                if unmasked[i, p]:
                    continue  # already revealed (shouldn't happen)
                probs = F.softmax(logits[i, p], dim=-1)
                top2 = probs.topk(2)
                top1_prob = top2.values[0].item()
                top2_prob = top2.values[1].item()
                top1_tok = top2.indices[0].item()

                ans_offset = p - T_pre  # answer-region offset
                gold_tok = gold_ans[i, ans_offset].item()
                gold_prob = probs[gold_tok].item()

                math_d = ANS_LEN - 1 - ans_offset
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

        # first answer position where the confidence decode is wrong
        conf_wrong_offsets = [
            j for j in range(ANS_LEN)
            if conf_preds[i, j].item() != gold_cpu[i, j].item()
        ]
        if not conf_wrong_offsets:
            continue
        first_wrong = conf_wrong_offsets[0]

        conf_commit = next((t for t in conf_traces[i]
                            if t["ans_offset"] == first_wrong), None)
        lsb_commit = next((t for t in lsb_traces[i]
                           if t["ans_offset"] == first_wrong), None)

        # commits preceding the first wrong one
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


#  A9  Confidence-ranking margin diagnostic (chosen vs runner-up position)
@torch.no_grad()
def a9_ranking_margin(model, tokenizer, bucket, max_examples=50, device=None):
    """At each confidence-greedy stage, record the chosen position and the
    runner-up position with their max logits / top-1 probabilities (the ranking
    margin), and the full top-10 ranking at the first wrong commit."""
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
        # top1_prob is recorded for readability; ranking uses the max logit
        probs = F.softmax(logits, dim=-1)  # [B, T, V]
        top1_prob_all = probs.max(dim=-1).values  # [B, T]

        for i in range(B):
            ans_slc = slice(T_pre, T_pre + ANS_LEN)
            still_masked_in_ans = (~unmasked[i, ans_slc]).nonzero(as_tuple=True)[0]
            if still_masked_in_ans.numel() == 0:
                continue

            masked_abs = T_pre + still_masked_in_ans
            scores = max_logit[i, masked_abs]  # [n_masked], ranked by max logit

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

            # margin in logit space (the decision margin) and in probability space
            ranking_margin_prob = (chosen_top1 - runner_top1) if runner_top1 is not None else None
            ranking_margin_logit = (chosen_logit - runner_logit) if runner_logit is not None else None

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

            # full top-10 ranking at the first wrong commit
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

            x[i, T_pre + chosen_ans_offset] = chosen_pred_tok
            unmasked[i, T_pre + chosen_ans_offset] = True
            final_pred[i, chosen_ans_offset] = chosen_pred_tok

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
    "a3": "Per-stage commit correctness",
    "a4": "Confidence calibration",
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
        print("  [legacy-single mode]")
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

    # multi-checkpoint mode
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
                    help="load checkpoint_{method}.pt (the selected EMA state of each method) "
                         "instead of the iteration snapshots")
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

    # test buckets are built once and shared by all checkpoints
    print("\nBuilding test buckets (chain sweep, including extreme tail)...")
    chain_buckets = {}
    for k in [4, 8, 12, 16, 20, 24, 28, 30, 32]:
        if k > ND: continue
        sp = gen_min_chain_test(args.n_per_bucket, seed=5000 + k, min_chain=k)
        if sp:
            chain_buckets[k] = _bucket_from_samples(sp, tokenizer, TOTAL_LEN)
            print(f"  chain>={k}: {chain_buckets[k]['n']} samples")

    if args.checkpoint_root is not None:
        exp_dirs = _discover_experiment_dirs(args.checkpoint_root)
        if not exp_dirs:
            print(f"\n[error] No experiment subdirs found under {args.checkpoint_root}")
            return
        print(f"\nDiscovered {len(exp_dirs)} experiment dirs under "
              f"{args.checkpoint_root}:")
        for d in exp_dirs:
            print(f"  {d.name}")
        targets = []
        for d in exp_dirs:
            if args.out_dir is not None:
                od = args.out_dir / d.name
            else:
                od = d / 'analysis'
            targets.append((d, od))
    else:
        od = args.out_dir or (args.checkpoint_dir / 'analysis')
        targets = [(args.checkpoint_dir, od)]

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
