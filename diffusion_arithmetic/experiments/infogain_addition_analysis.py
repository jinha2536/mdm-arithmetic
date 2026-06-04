"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Info-Gain Sampler on Addition  —  Confidence-Shortcut Probe
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Implements the full Info-Gain (IG) Sampler of Yang et al. (arXiv:2602.18176)
  and applies it to the trained addition MDM, to test whether replacing the
  greedy confidence selection rule with an entropy-reduction objective
  recovers the LSB-first (carry-chain) decoding order that the confidence
  shortcut violates.

  IG Sampler (paper §3.2):
    state uncertainty   H(z_t)   = mean marginal entropy over masked positions
    information gain     IG(a;z) = H(z_t) - H(z_{t-1})       (Eq. 4)
    immediate cost       C(a;z)  = sum_{l in A} H^(l)(z_t)   (Eq. 16)
    objective            J_IG    = IG - C                    (Eq. 5)
    selection            a* = argmax_{a in candidate set} J_IG

  Candidate set (paper §3.2.2 Action Sampler):
    token sampling   : v_l ~ p_theta (temperature tau_token)
    position sampling: l  ~ softmax(phi(l,z) / tau_pos)
  We use K=1 (one position per step) to match the addition decode loop, so a
  candidate action is a single (position, token) pair. With K=1, C(a;z) is the
  marginal entropy at the chosen position — i.e. exactly the "Entropy" baseline's
  per-position score — and IG is the entropy reduction it induces elsewhere.

  This module produces a per-stage trace in the SAME format as
  addition_decode_analysis.a8/a9 so the existing chain-MSB mechanism analyses
  apply unchanged.

  Two analyses:
    (B1) LSB-first alignment   : Kendall tau-b between IG reveal order and the
                                 canonical LSB-first order, per policy
                                 (confidence / IG / r2l-oracle).
    (B2) chain-MSB dissection  : at the first wrong commit, what digit / carry
                                 role does each policy commit to, and with what
                                 top-1 / top-2 / gold probabilities. Directly
                                 comparable to the 405/406 confidence finding.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import math
import torch
import torch.nn.functional as F
import numpy as np


# ─────────────────────────────────────────────────────────────────
# Info-Gain Sampler (K=1, full candidate-set evaluation)
# ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def infogain_decode_trace(model, pids, ans_len, mask_id, gold_ans, metas,
                          N_cand=8, tau_token=0.7, tau_pos=0.1,
                          gamma=None, commit='greedy', device=None):
    """Decode the answer region with the Info-Gain Sampler, returning a
    per-example per-stage trace.

    Args:
        model     : trained MDM. model(x) -> logits [B, T, V].
        pids      : [B, T_pre] prefix ids (already padded/encoded).
        ans_len   : number of answer tokens to decode (ANS_LEN).
        mask_id   : mask token id.
        gold_ans  : [B, ans_len] gold answer token ids (for trace bookkeeping).
        metas     : list of dicts (len B) with 'dep_ctx' (per ans-offset role).
        N_cand    : candidate-set size N (paper uses 8).
        tau_token : token sampling temperature (paper 0.7).
        tau_pos   : position sampling temperature (paper 0.1).
        gamma     : high-confidence bypass threshold (paper §3.2.3, latency
                    optimization). DEFAULT None = bypass DISABLED. In addition,
                    the chain-MSB shortcut cell has top-1 prob ~0.997, so any
                    gamma below that would force-commit exactly the shortcut
                    cell BEFORE the IG objective is evaluated — collapsing IG to
                    confidence at precisely the position we want to study. Keep
                    None to measure the IG objective itself; set e.g. 0.8 only to
                    reproduce the paper's latency-optimized sampler.
        commit    : 'greedy' (default) commits argmax token at the IG-selected
                    position — apples-to-apples with confidence decode (a8).
                    'sample' commits the temperature-sampled token (paper's
                    stochastic sampler); makes correctness noisy vs greedy a8.

    Returns: list (len B) of per-stage trace dicts. Same schema as
             addition_decode_analysis._trace_decode plus 'selected_by_bypass'.
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    B = pids.shape[0]
    T_pre = pids.shape[1]
    T = T_pre + ans_len

    x = torch.full((B, T), mask_id, dtype=torch.long, device=device)
    x[:, :T_pre] = pids.to(device)
    unmasked = torch.zeros(B, T, dtype=torch.bool, device=device)
    unmasked[:, :T_pre] = True

    traces = [[] for _ in range(B)]
    B_ar = torch.arange(B, device=device)

    def _state_entropy(logits, umask):
        """Mean marginal entropy over masked (not-yet-revealed) positions, per row.
        logits: [B,T,V] with mask token already -inf'd. umask: [B,T] bool unmasked.
        Returns [B] mean entropy; masked-count guarded against zero."""
        logp = F.log_softmax(logits, dim=-1)            # [B,T,V]
        p = logp.exp()
        ent = -(p * logp).sum(-1)                        # [B,T] per-position entropy
        ent = ent.masked_fill(umask, 0.0)
        n_masked = (~umask).sum(-1).clamp(min=1)         # [B]
        return ent.sum(-1) / n_masked, ent               # mean-H [B], per-pos H [B,T]

    for stage in range(ans_len):
        logits = model(x)
        logits[:, :, mask_id] = -float('inf')

        # State uncertainty H(z_t) and per-position entropy at current state.
        H_zt, perpos_H = _state_entropy(logits, unmasked)   # [B], [B,T]

        max_prob = F.softmax(logits, dim=-1).max(dim=-1).values  # [B,T] top-1 prob
        # eligible = masked positions only
        eligible = ~unmasked                                      # [B,T]

        # ── High-confidence bypass (paper §3.2.3) ────────────────
        # DISABLED when gamma is None (default). When enabled, force-commits the
        # single most confident masked position — in addition this is the
        # shortcut cell, so it collapses IG to confidence. See docstring.
        mp = max_prob.clone()
        mp[~eligible] = -float('inf')
        best_conf_pos = mp.argmax(-1)                             # [B]
        if gamma is None:
            bypass = torch.zeros(B, dtype=torch.bool, device=device)
        else:
            bypass = mp[B_ar, best_conf_pos] >= gamma             # [B] bool

        # ── Candidate position sampling (softmax over confidence / tau_pos) ──
        # phi(l,z) = top-1 prob (confidence). Sample N positions per row.
        phi = max_prob.clone()
        phi[~eligible] = -float('inf')
        pos_logits = phi / max(tau_pos, 1e-6)                     # [B,T]
        pos_probs = F.softmax(pos_logits, dim=-1)                 # [B,T]
        # guard: rows with a single eligible pos still sample fine
        cand_pos = torch.multinomial(pos_probs, N_cand, replacement=True)  # [B,N]

        # ── Candidate token sampling (temperature tau_token) at each cand pos ──
        # gather logits at candidate positions, sample one token each.
        cand_logits = logits[B_ar.unsqueeze(1), cand_pos]        # [B,N,V]
        tok_probs = F.softmax(cand_logits / max(tau_token, 1e-6), dim=-1)
        flat = tok_probs.reshape(B * N_cand, -1)
        cand_tok = torch.multinomial(flat, 1).reshape(B, N_cand)  # [B,N]

        # ── Evaluate J_IG = IG - C for every candidate via batched re-forward ──
        # Build [B*N, T] candidate states (apply each (pos,tok) to a copy of x).
        x_rep = x.unsqueeze(1).expand(B, N_cand, T).reshape(B * N_cand, T).clone()
        um_rep = unmasked.unsqueeze(1).expand(B, N_cand, T).reshape(B * N_cand, T).clone()
        flat_pos = cand_pos.reshape(-1)                           # [B*N]
        flat_tok = cand_tok.reshape(-1)                           # [B*N]
        bn_ar = torch.arange(B * N_cand, device=device)
        x_rep[bn_ar, flat_pos] = flat_tok
        um_rep[bn_ar, flat_pos] = True

        logits_next = model(x_rep)
        logits_next[:, :, mask_id] = -float('inf')
        H_next, _ = _state_entropy(logits_next, um_rep)          # [B*N]
        H_next = H_next.reshape(B, N_cand)                        # [B,N]

        # Immediate cost C = per-position entropy at chosen pos (current state).
        C = perpos_H[B_ar.unsqueeze(1), cand_pos]                # [B,N]
        IG = H_zt.unsqueeze(1) - H_next                          # [B,N]
        J = IG - C                                               # [B,N] objective

        best_cand = J.argmax(-1)                                 # [B]
        sel_pos = cand_pos[B_ar, best_cand]                      # [B] IG-selected position
        sampled_tok = cand_tok[B_ar, best_cand]                  # [B] temperature-sampled token

        # Commit token at the IG-selected position.
        # 'greedy' (default): argmax at that position — matches a8 confidence
        #   decode, so correctness differences reflect POSITION ORDER, not
        #   sampling noise. 'sample': the temperature-sampled token (paper).
        greedy_tok_at_sel = logits[B_ar, sel_pos].argmax(-1)     # [B]
        if commit == 'greedy':
            sel_tok = greedy_tok_at_sel
        else:
            sel_tok = sampled_tok

        # Apply bypass override where triggered (no-op when gamma is None).
        sel_pos = torch.where(bypass, best_conf_pos, sel_pos)
        bypass_tok = logits[B_ar, best_conf_pos].argmax(-1)
        sel_tok = torch.where(bypass, bypass_tok, sel_tok)

        # ── Record trace (greedy top-1 reporting, mirrors a8) ───
        for i in range(B):
            p = sel_pos[i].item()
            if unmasked[i, p]:
                continue
            probs = F.softmax(logits[i, p], dim=-1)
            top2 = probs.topk(2)
            top1_prob = top2.values[0].item()
            top2_prob = top2.values[1].item()
            top1_tok = top2.indices[0].item()
            ans_offset = p - T_pre
            gold_tok = gold_ans[i, ans_offset].item()
            gold_prob = probs[gold_tok].item()
            math_d = ans_len - 1 - ans_offset
            dep_ctx = metas[i].get('dep_ctx', [])
            role = dep_ctx[ans_offset] if ans_offset < len(dep_ctx) else '?'
            # committed token: IG sampler commits sel_tok (may be sampled),
            # but for shortcut analysis we report greedy top-1 like a8 AND the
            # actually-committed token so both views are available.
            committed = sel_tok[i].item()
            traces[i].append({
                'stage': stage,
                'ans_offset': ans_offset,
                'math_d': math_d,
                'role': role,
                'committed_tok': committed,
                'greedy_top1_tok': top1_tok,
                'gold_tok': gold_tok,
                'is_correct': (committed == gold_tok),
                'top1_prob': top1_prob,
                'top2_prob': top2_prob,
                'gold_prob': gold_prob,
                'margin': top1_prob - top2_prob,
                'selected_by_bypass': bool(bypass[i].item()),
                'J_ig': float(J[i, best_cand[i]].item()),
                'state_entropy': float(H_zt[i].item()),
            })

        # Commit.
        x[B_ar, sel_pos] = sel_tok
        unmasked[B_ar, sel_pos] = True

    return traces


# ─────────────────────────────────────────────────────────────────
# (B3) bypass dissection  —  WHERE does the high-confidence bypass fire?
# ─────────────────────────────────────────────────────────────────
def b3_bypass_dissection(traces, ans_len):
    """Cross-analyse bypass-committed vs IG-committed positions. Only
    meaningful when the IG sampler was run with gamma != None.

    Tests the hypothesis: bypass fires on EASY cells (k/g with determined
    carry) early, clearing them as context, so IG is left to act only on a
    narrowed, less-broken sub-problem. If true:
      - bypassed positions concentrate on k/g (and chain-OUTSIDE), not p-interior
      - bypassed positions are committed EARLIER (lower stage) on average
      - bypassed commits are nearly always correct; IG-objective commits carry
        the error mass.

    Returns per-policy-run aggregates."""
    by_role = {}        # role -> count among bypassed commits
    ig_role = {}        # role -> count among IG-objective commits
    by_stage, ig_stage = [], []
    by_correct = by_total = 0
    ig_correct = ig_total = 0
    by_top1, ig_top1 = [], []

    for tr in traces:
        for s in tr:
            is_by = s.get('selected_by_bypass', False)
            role = s.get('role', '?')
            if is_by:
                by_role[role] = by_role.get(role, 0) + 1
                by_stage.append(s['stage'])
                by_total += 1
                by_correct += int(s['is_correct'])
                by_top1.append(s['top1_prob'])
            else:
                ig_role[role] = ig_role.get(role, 0) + 1
                ig_stage.append(s['stage'])
                ig_total += 1
                ig_correct += int(s['is_correct'])
                ig_top1.append(s['top1_prob'])

    def _m(xs): return float(np.mean(xs)) if xs else float('nan')
    return {
        'n_bypass_commits': by_total,
        'n_ig_commits': ig_total,
        'bypass_frac_of_all': float(by_total / (by_total + ig_total))
                              if (by_total + ig_total) else float('nan'),
        'bypass_role_dist': by_role,
        'ig_role_dist': ig_role,
        'bypass_mean_stage': _m(by_stage),    # lower => fired earlier
        'ig_mean_stage': _m(ig_stage),
        'bypass_acc': float(by_correct / by_total) if by_total else float('nan'),
        'ig_acc': float(ig_correct / ig_total) if ig_total else float('nan'),
        'bypass_mean_top1': _m(by_top1),
        'ig_mean_top1': _m(ig_top1),
    }


# ─────────────────────────────────────────────────────────────────
# (B1) LSB-first alignment
# ─────────────────────────────────────────────────────────────────
def reveal_order_from_trace(trace, ans_len):
    """Map a per-stage trace to a reveal-stage array indexed by answer offset.
    reveal_stage[ans_offset] = stage at which that offset was committed."""
    rs = np.full(ans_len, ans_len, dtype=np.float32)
    for step in trace:
        rs[step['ans_offset']] = step['stage']
    return rs


def lsb_first_rank(ans_len):
    """Canonical LSB-first reasoning order indexed by ANSWER OFFSET.
    LSB = math digit 0 = answer offset (ans_len-1) should be revealed first.
    Rank by math digit ascending => rank[offset] = math_d = ans_len-1-offset."""
    offsets = np.arange(ans_len)
    math_d = ans_len - 1 - offsets        # LSB-first: smaller math_d = earlier
    return math_d.astype(np.float32)


def b1_lsb_alignment(traces, ans_len):
    """Kendall tau-b per example between reveal order and LSB-first order.
    Reuses the same ties-robust fallback as train_utils.compute_reveal_vs_order_tau."""
    from scipy.stats import kendalltau, spearmanr
    canonical = lsb_first_rank(ans_len)
    taus = []
    for tr in traces:
        rs = reveal_order_from_trace(tr, ans_len)
        x, y = rs, canonical
        if np.unique(x).size < 2 or np.unique(y).size < 2:
            continue
        t, _ = kendalltau(x, y, variant='b')
        if t is None or np.isnan(t):
            t, _ = kendalltau(x, y, variant='c')
        if t is None or np.isnan(t):
            t, _ = spearmanr(x, y)
        if t is not None and not np.isnan(t):
            taus.append(float(t))
    taus = np.asarray(taus, dtype=np.float32)
    return {
        'mean_tau': float(taus.mean()) if len(taus) else float('nan'),
        'median_tau': float(np.median(taus)) if len(taus) else float('nan'),
        'std_tau': float(taus.std()) if len(taus) else float('nan'),
        'n': int(len(taus)),
        'frac_tau_gt_0.5': float((taus > 0.5).mean()) if len(taus) else float('nan'),
    }


# ─────────────────────────────────────────────────────────────────
# (B2) chain-MSB dissection at first wrong commit
# ─────────────────────────────────────────────────────────────────
def b2_first_wrong_dissection(traces):
    """For each failing example, find the first wrong commit (by stage order)
    and characterise it: math digit, carry role, top-1/top-2/gold prob, whether
    gold was top-2. Aggregates role distribution at the first wrong commit.

    Mirrors the confidence-decode finding (e.g. wrong commits concentrate at
    the chain-MSB g/k cell, gold sits at top-2)."""
    role_counts = {}
    gold_in_top2 = 0
    n_wrong = 0
    n_bypass_at_fw = 0
    top1_probs, gold_probs, margins, math_ds = [], [], [], []
    examples = []

    for tr in traces:
        # order trace by stage to find the first wrong commit
        ordered = sorted(tr, key=lambda s: s['stage'])
        fw = next((s for s in ordered if not s['is_correct']), None)
        if fw is None:
            continue
        n_wrong += 1
        if fw.get('selected_by_bypass', False):
            n_bypass_at_fw += 1
        role = fw['role']
        role_counts[role] = role_counts.get(role, 0) + 1
        # gold-in-top2 test: gold_prob >= top2_prob means gold ranked >= 2nd
        if fw['gold_prob'] >= fw['top2_prob'] - 1e-9:
            gold_in_top2 += 1
        top1_probs.append(fw['top1_prob'])
        gold_probs.append(fw['gold_prob'])
        margins.append(fw['margin'])
        math_ds.append(fw['math_d'])
        if len(examples) < 8:
            examples.append({k: fw[k] for k in
                             ('stage', 'math_d', 'role', 'top1_prob',
                              'top2_prob', 'gold_prob', 'selected_by_bypass')})

    def _m(xs): return float(np.mean(xs)) if xs else float('nan')
    return {
        'n_wrong': n_wrong,
        'role_counts': role_counts,
        'gold_in_top2_frac': float(gold_in_top2 / n_wrong) if n_wrong else float('nan'),
        'mean_top1_prob': _m(top1_probs),
        'mean_gold_prob': _m(gold_probs),
        'mean_margin': _m(margins),
        'mean_first_wrong_math_d': _m(math_ds),
        # NOW over ALL wrong commits, not the truncated example list:
        'frac_bypass_at_first_wrong':
            float(n_bypass_at_fw / n_wrong) if n_wrong else float('nan'),
        'examples': examples,
    }


# ─────────────────────────────────────────────────────────────────
# End-to-end runner  —  Colab entry point
# ─────────────────────────────────────────────────────────────────
def a10_infogain(model, tokenizer, bucket, ANS_LEN, max_examples=300,
                 N_cand=8, tau_token=0.7, tau_pos=0.1, gamma=None,
                 commit='greedy', device=None):
    """Dispatcher-friendly wrapper matching the a8/a9 call signature used by
    addition_decode_analysis._run_analyses. Caps the bucket to max_examples
    (IG runs N candidate forwards per step, so it is ~N x slower than greedy).
    Returns the per-policy B1/B2 dict from run_infogain_addition."""
    samples = bucket.get('samples', [])
    if max_examples is not None and len(samples) > max_examples:
        capped = {'samples': samples[:max_examples],
                  'metas': bucket['metas'][:max_examples],
                  'n': max_examples}
    else:
        capped = bucket
    return run_infogain_addition(
        model, tokenizer, capped, ANS_LEN,
        policies=('confidence', 'infogain', 'random', 'r2l'),
        N_cand=N_cand, tau_token=tau_token, tau_pos=tau_pos,
        gamma=gamma, commit=commit, device=device)


def run_infogain_addition(model, tokenizer, bucket, ANS_LEN,
                          policies=('confidence', 'infogain', 'random', 'r2l'),
                          N_cand=8, tau_token=0.7, tau_pos=0.1, gamma=None,
                          commit='greedy', seed=0, device=None):
    """Run IG (and baseline confidence / random / r2l-oracle for contrast) on
    one bucket, returning B1 (LSB alignment) and B2 (first-wrong dissection)
    per policy.

    Reproduces the a8_failure_dissection setup (prefix/gold encoding) so it can
    be called directly on any bucket from the experiment suite, e.g.
    suite['constructed']['chain_24'].

    Args:
        model, tokenizer : the trained addition MDM and its CharTokenizer.
        bucket           : dict with 'samples' (list of "a+b=c" strings) and
                           'metas' (list of dicts with 'dep_ctx').
        ANS_LEN          : answer length (ND + 1).
        policies         : which decoders to run. 'infogain' = IG sampler;
                           'confidence' = greedy top-1; 'random' = uniform over
                           masked positions (KEY CONTROL — does IG's reveal order
                           differ from random?); 'r2l' = LSB-first oracle.
        seed             : fixes torch RNG so the stochastic 'random' policy and
                           the IG candidate sampler are reproducible.
    Returns: {policy: {'b1_lsb_alignment': {...}, 'b2_first_wrong': {...}}}
    """
    import torch
    import torch.nn.functional as F
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    torch.manual_seed(seed)

    pad_id = tokenizer.special_ids['pad']
    mask_id = tokenizer.special_ids['mask']
    samples = bucket['samples']
    metas = bucket['metas']
    B = len(samples)
    if B == 0:
        return {}

    # prefix encode (mirrors a8)
    penc = [tokenizer.encode(s.split('=')[0] + '=') for s in samples]
    pm = max(len(p) for p in penc)
    pids = torch.full((B, pm), pad_id, dtype=torch.long)
    for i, e in enumerate(penc):
        pids[i, :len(e)] = torch.tensor(e)
    pids = pids.to(device)

    # gold answer encode (mirrors a8)
    ans_strs = [s.split('=')[1] for s in samples]
    gold_ans = torch.full((B, ANS_LEN), pad_id, dtype=torch.long)
    for i, ans in enumerate(ans_strs):
        ids = tokenizer.encode(ans)
        gold_ans[i, :len(ids)] = torch.tensor(ids)
    gold_ans = gold_ans.to(device)

    @torch.no_grad()
    def _baseline_trace(policy):
        """confidence-greedy or r2l-oracle, same trace schema as IG (for B1/B2)."""
        T_pre = pids.shape[1]
        T = T_pre + ANS_LEN
        x = torch.full((B, T), mask_id, dtype=torch.long, device=device)
        x[:, :T_pre] = pids
        um = torch.zeros(B, T, dtype=torch.bool, device=device)
        um[:, :T_pre] = True
        traces = [[] for _ in range(B)]
        B_ar = torch.arange(B, device=device)
        for stage in range(ANS_LEN):
            logits = model(x)
            logits[:, :, mask_id] = -float('inf')
            if policy == 'confidence':
                # top-1 softmax probability (LLaDA/Dream convention), matching
                # the IG sampler's max_prob and the patched generate_diffusion.
                ml = F.softmax(logits, dim=-1).max(dim=-1).values
                ml[um] = -float('inf')
                pos = ml.argmax(-1)
            elif policy == 'r2l':
                pos = torch.full((B,), T_pre + ANS_LEN - 1 - stage,
                                 dtype=torch.long, device=device)
            elif policy == 'random':
                # uniform random over masked positions (matches train_utils
                # generate_diffusion random policy). This is the key control:
                # if IG's reveal-order tau matches random's, the IG candidate
                # sampler is effectively random, not measuring IG. If IG differs
                # significantly from random, IG is doing something real.
                eligible = (~um).float()                 # [B,T] 1.0 at masked
                pos = torch.multinomial(eligible, 1).squeeze(-1)  # [B]
            else:
                raise ValueError(policy)
            for i in range(B):
                p = pos[i].item()
                if um[i, p]:
                    continue
                probs = F.softmax(logits[i, p], dim=-1)
                top2 = probs.topk(2)
                ans_offset = p - T_pre
                gold_tok = gold_ans[i, ans_offset].item()
                committed = top2.indices[0].item()
                dep_ctx = metas[i].get('dep_ctx', [])
                traces[i].append({
                    'stage': stage, 'ans_offset': ans_offset,
                    'math_d': ANS_LEN - 1 - ans_offset,
                    'role': dep_ctx[ans_offset] if ans_offset < len(dep_ctx) else '?',
                    'committed_tok': committed, 'gold_tok': gold_tok,
                    'is_correct': committed == gold_tok,
                    'top1_prob': top2.values[0].item(),
                    'top2_prob': top2.values[1].item(),
                    'gold_prob': probs[gold_tok].item(),
                    'margin': top2.values[0].item() - top2.values[1].item(),
                    'selected_by_bypass': False,
                })
            B_ar2 = torch.arange(B, device=device)
            x[B_ar2, pos] = logits[B_ar2, pos].argmax(-1)
            um[B_ar2, pos] = True
        return traces

    out = {}
    for pol in policies:
        if pol == 'infogain':
            traces = infogain_decode_trace(
                model, pids, ANS_LEN, mask_id, gold_ans, metas,
                N_cand=N_cand, tau_token=tau_token, tau_pos=tau_pos,
                gamma=gamma, commit=commit, device=device)
        else:
            traces = _baseline_trace(pol)
        out[pol] = {
            'b1_lsb_alignment': b1_lsb_alignment(traces, ANS_LEN),
            'b2_first_wrong': b2_first_wrong_dissection(traces),
        }
        if pol == 'infogain':
            out[pol]['b3_bypass'] = b3_bypass_dissection(traces, ANS_LEN)
    return out


# ─────────────────────────────────────────────────────────────────
# Bypass OFF vs ON  —  side-by-side over a chain sweep
# ─────────────────────────────────────────────────────────────────
def compare_bypass(model, tokenizer, chain_buckets, ANS_LEN,
                   gamma_on=0.8, N_cand=8, tau_token=0.7, tau_pos=0.1,
                   commit='greedy', seed=0, device=None, print_table=True):
    """Run the IG sampler with bypass OFF (gamma=None) and ON (gamma=gamma_on)
    on the same chain_buckets, and lay the key metrics next to each other.

    chain_buckets : {k: bucket} as built in addition_decode_analysis.main()
                    (gen_min_chain_test + _bucket_from_samples).

    For each chain stratum reports, for IG under both settings:
      tau     : LSB-first reveal-order Kendall tau (B1)
      wrong   : # first-wrong examples (B2)
      by_frac : fraction of ALL commits taken by the bypass (0.0 when OFF)
      by_acc  : accuracy of bypass-committed cells (nan when OFF)
      ig_acc  : accuracy of IG-objective-committed cells
    plus confidence (greedy) tau/wrong as the reference line — confidence is
    invariant to gamma, so it is shown once.

    Returns {k: {'off': <run dict>, 'on': <run dict>}}.
    """
    results = {}
    for k, b in chain_buckets.items():
        off = run_infogain_addition(
            model, tokenizer, b, ANS_LEN,
            policies=('confidence', 'infogain'),
            gamma=None, N_cand=N_cand, tau_token=tau_token, tau_pos=tau_pos,
            commit=commit, seed=seed, device=device)
        on = run_infogain_addition(
            model, tokenizer, b, ANS_LEN,
            policies=('infogain',),
            gamma=gamma_on, N_cand=N_cand, tau_token=tau_token, tau_pos=tau_pos,
            commit=commit, seed=seed, device=device)
        results[k] = {'off': off, 'on': on}

    if print_table:
        for k in sorted(results):
            off, on = results[k]['off'], results[k]['on']
            c_b1 = off['confidence']['b1_lsb_alignment']
            c_b2 = off['confidence']['b2_first_wrong']
            o_b1 = off['infogain']['b1_lsb_alignment']
            o_b2 = off['infogain']['b2_first_wrong']
            n_b1 = on['infogain']['b1_lsb_alignment']
            n_b2 = on['infogain']['b2_first_wrong']
            n_b3 = on['infogain']['b3_bypass']
            print(f"\nchain>={k}")
            print(f"  confidence       τ={c_b1['mean_tau']:+.3f}  wrong={c_b2['n_wrong']}")
            print(f"  infogain  bypass-OFF  τ={o_b1['mean_tau']:+.3f}  wrong={o_b2['n_wrong']}")
            print(f"  infogain  bypass-ON   τ={n_b1['mean_tau']:+.3f}  wrong={n_b2['n_wrong']}"
                  f"  by_frac={n_b3['bypass_frac_of_all']:.2f}"
                  f"  by_acc={n_b3['bypass_acc']:.3f}  ig_acc={n_b3['ig_acc']:.3f}")
            print(f"      bypass roles={n_b3['bypass_role_dist']}  "
                  f"stage(by={n_b3['bypass_mean_stage']:.1f} vs ig={n_b3['ig_mean_stage']:.1f})")
    return results


# ─────────────────────────────────────────────────────────────────
# Self-test on a tiny random model (interface + shape validation only)
# ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    torch.manual_seed(0)
    V, T_pre, ANS = 16, 5, 9
    MASK = V - 1

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = torch.nn.Embedding(V, 32)
            self.lin = torch.nn.Linear(32, V)
        def forward(self, x):
            return self.lin(self.emb(x))

    model = Toy().eval()
    B = 4
    pids = torch.randint(0, V - 1, (B, T_pre))
    gold = torch.randint(0, V - 1, (B, ANS))
    metas = [{'dep_ctx': ['g' if d % 2 else 'k' for d in range(ANS)]} for _ in range(B)]

    traces = infogain_decode_trace(model, pids, ANS, MASK, gold, metas,
                                   N_cand=8, device='cpu')
    assert len(traces) == B
    assert all(len(t) == ANS for t in traces), [len(t) for t in traces]
    # each offset revealed exactly once
    for t in traces:
        offs = sorted(s['ans_offset'] for s in t)
        assert offs == list(range(ANS)), offs

    b1 = b1_lsb_alignment(traces, ANS)
    b2 = b2_first_wrong_dissection(traces)
    print("self-test OK")
    print("  B1 lsb-alignment:", {k: b1[k] for k in ('mean_tau', 'n', 'frac_tau_gt_0.5')})
    print("  B2 first-wrong  :", {k: b2[k] for k in ('n_wrong', 'role_counts', 'gold_in_top2_frac')})
