"""32-digit addition: training of the three masking schemes, exact-match
accuracy on the natural test set and on carry-chain strata under confidence /
LSB-first (/ uniform-random) decoding, and the training-dynamics figure.

Dynamics saved in results.json under dyn_{method}:
  train_loss                          batch loss of the online weights, every LOG_EVERY
  checkpoints[k]['overall_loss']      fully-masked probe NLL of the EMA weights on the
                                      natural test set (used to select the EMA state)
  checkpoints[k]['gen_acc_confidence']
                                      exact-match accuracy of the EMA weights under
                                      confidence decoding on natural[:GEN_EVAL_N],
                                      every GEN_EVAL_EVERY
  checkpoints[k]['gen_acc_{bucket}_{policy}']
                                      same for TAIL_GEN_BUCKETS x TAIL_GEN_POLICIES
EMA snapshots checkpoint_seed{SEED}_{method}_iter{it:06d}.pt are written on the
same grid (CKPT_ITERS = 'gen_eval'); addition_decode_analysis.py and
remasking_analysis.py read them.

Entry points:
  run()           train every scheme, evaluate, save results.json, checkpoints
                  and acc_trajectory.png
  run_training()  (--train-only) train every scheme from one shared initial
                  state with a per-scheme training seed; save the EMA snapshots
                  and the dynamics only
"""
import sys, os, random
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                if '__file__' in dir() else '.')
from core.tokenizer import CharTokenizer
from core.model import Transformer
from core.train_utils import (
    prepare_results_dir, save_results, save_checkpoint, encode_samples,
    train_diffusion, puma_k_step, generate_diffusion, DEVICE,
)

EXP_NAME = 'exp_addition'

# Config
ND = 32; ANS_LEN = ND + 1

N_TRAIN = 20000; N_TEST = 10000; BATCH_SIZE = 256
N_PER_BUCKET = 500        # instances per carry-chain stratum
MAX_ITERS = 300000; EVAL_EVERY = 5000; LOG_EVERY = 1000
GEN_EVAL_EVERY = 10000; GEN_EVAL_N = 500
# carry-chain strata evaluated at every gen eval (dynamics key gen_acc_{bucket}_{policy})
TAIL_GEN_BUCKETS = ['chain_28']
TAIL_GEN_POLICIES = ['confidence', 'lsb']
MASK_TYPES = ['random', 'papl', 'puma']
DECODE_POLICIES = ['confidence', 'lsb']     # add 'random' for uniform-random decoding
N_LAYER = 2; N_HEAD = 2; N_EMBD = 128; DROPOUT = 0.1
LR = 1e-3; MIN_LR = 1e-4; WARMUP_ITERS = 2000; GRAD_CLIP = 1.0
WEIGHT_DECAY = 0.1; EMA_DECAY = 0.9999
PUMA_TAU = 0.9
PUMA_K_START = 3; PUMA_K_END = 16
PUMA_K_STEP = 3; PUMA_K_EVERY = None  # None = ramp over the first 1/3 of training
PAPL_TAU = 1.0; PAPL_ALPHA = 1.0      # addition uses alpha=1 (alpha=5 for the other domains)
SEED = 42
NO_AMP = True   # bf16 autocast off: reduced mantissa precision hurts exact digit prediction

# EMA snapshot schedule: 'gen_eval' = every GEN_EVAL_EVERY up to MAX_ITERS,
# an explicit list of iterations, or [] (disabled)
CKPT_ITERS = 'gen_eval'

FIG_COLORS = {'random': '#3b76b8', 'papl': '#d8443d', 'puma': '#e89537'}
FIG_LABELS = {'random': 'Random', 'papl': 'PAPL', 'puma': 'PUMA'}
TAIL_FIG_BUCKET = 'chain_28'   # stratum overlaid in the accuracy-trajectory figure
ACC_FIG_SMOOTH = 1             # centred moving-average window in points (1 = off)


def resolve_ckpt_iters(max_iters=None):
    """CKPT_ITERS as a sorted list of iterations in (0, max_iters]; for
    'gen_eval' the final iteration is always included."""
    if max_iters is None: max_iters = MAX_ITERS
    if CKPT_ITERS == 'gen_eval':
        its = list(range(GEN_EVAL_EVERY, max_iters + 1, GEN_EVAL_EVERY))
        if max_iters not in its: its.append(max_iters)
    elif not CKPT_ITERS:
        its = []
    else:
        its = list(CKPT_ITERS)
    return sorted(set(int(i) for i in its if 0 < int(i) <= max_iters))


def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--n-train', type=int); p.add_argument('--n-test', type=int)
    p.add_argument('--max-iters', type=int); p.add_argument('--batch-size', type=int)
    p.add_argument('--eval-every', type=int); p.add_argument('--gen-eval-every', type=int)
    p.add_argument('--n-layer', type=int); p.add_argument('--n-head', type=int)
    p.add_argument('--n-embd', type=int); p.add_argument('--dropout', type=float)
    p.add_argument('--lr', type=float); p.add_argument('--weight-decay', type=float)
    p.add_argument('--puma-tau', type=float)
    p.add_argument('--puma-k-start', type=int); p.add_argument('--puma-k-end', type=int)
    p.add_argument('--puma-k-step', type=int); p.add_argument('--puma-k-every', type=int)
    p.add_argument('--papl-tau', type=float); p.add_argument('--papl-alpha', type=float)
    p.add_argument('--masks', nargs='+'); p.add_argument('--decode', nargs='+')
    p.add_argument('--ckpt-iters', type=str,
                   help="EMA snapshot schedule: 'gen_eval' (default), 'none', "
                        "or comma-separated iterations")
    p.add_argument('--train-only', action='store_true',
                   help='run_training(): shared init, per-scheme training seed, '
                        'snapshots and dynamics only')
    p.add_argument('--no-amp', action='store_true')
    p.add_argument('--tag', type=str, default=''); p.add_argument('--seed', type=int)
    p.add_argument('--seeds', nargs='+', type=int)
    args, _ = p.parse_known_args()
    g = globals()
    for a, gl in {'n_train': 'N_TRAIN', 'n_test': 'N_TEST', 'max_iters': 'MAX_ITERS',
                   'batch_size': 'BATCH_SIZE', 'eval_every': 'EVAL_EVERY',
                   'gen_eval_every': 'GEN_EVAL_EVERY', 'n_layer': 'N_LAYER',
                   'n_head': 'N_HEAD', 'n_embd': 'N_EMBD', 'dropout': 'DROPOUT',
                   'lr': 'LR', 'weight_decay': 'WEIGHT_DECAY', 'puma_tau': 'PUMA_TAU',
                   'puma_k_start': 'PUMA_K_START', 'puma_k_end': 'PUMA_K_END',
                   'puma_k_step': 'PUMA_K_STEP', 'puma_k_every': 'PUMA_K_EVERY',
                   'papl_tau': 'PAPL_TAU', 'papl_alpha': 'PAPL_ALPHA',
                   'seed': 'SEED'}.items():
        v = getattr(args, a, None)
        if v is not None: g[gl] = v
    if args.no_amp: g['NO_AMP'] = True
    if args.masks: g['MASK_TYPES'] = args.masks
    if args.decode: g['DECODE_POLICIES'] = args.decode
    if getattr(args, 'ckpt_iters', None) is not None:
        s = args.ckpt_iters.strip().lower()
        if s in ('none', 'off', ''):
            g['CKPT_ITERS'] = []
        elif s == 'gen_eval':
            g['CKPT_ITERS'] = 'gen_eval'
        else:
            g['CKPT_ITERS'] = sorted(int(x) for x in s.split(',') if x.strip())
    return args


# Data helpers
def _pad(n, w): return str(n).zfill(w)
def _fmt_plain(a, b): return f"{_pad(a,ND)}+{_pad(b,ND)}={_pad(a+b,ANS_LEN)}"
def get_answer(s): return s.split('=')[1]
def _parse_operands(s):
    parts = s.split('=')[0].split('+'); return int(parts[0]), int(parts[1])

def _dependency_context_at_pos(a, b):
    """Role of every answer position: g, k, p_above_g, p_above_k, p_above_p,
    p_bottom (p at the least significant digit) or carry_out."""
    a_s, b_s = _pad(a, ND), _pad(b, ND)
    gkp = []
    for i in range(ND - 1, -1, -1):
        s = int(a_s[i]) + int(b_s[i])
        gkp.append('g' if s >= 10 else ('p' if s == 9 else 'k'))
    dep = ['?'] * ND
    for d in range(ND):
        if gkp[d] in ('g', 'k'): dep[d] = gkp[d]
        elif d == 0: dep[d] = 'p_bottom'
        elif gkp[d-1] == 'g': dep[d] = 'p_above_g'
        elif gkp[d-1] == 'k': dep[d] = 'p_above_k'
        else: dep[d] = 'p_above_p'
    out = ['?'] * ANS_LEN
    out[0] = 'carry_out'
    for j in range(ND): out[ND - j] = dep[j]
    return out

def _chain_stats(a, b):
    """Carry-chain statistics; gkp is indexed from the least significant digit."""
    a_s, b_s = _pad(a, ND), _pad(b, ND)
    gkp = []
    for i in range(ND - 1, -1, -1):
        s = int(a_s[i]) + int(b_s[i])
        gkp.append('g' if s >= 10 else ('p' if s == 9 else 'k'))
    carry = 0
    for i in range(ND - 1, -1, -1):
        s = int(a_s[i]) + int(b_s[i]) + carry; carry = s // 10
    chains, rs, rl = [], None, 0
    for d in range(ND):
        if gkp[d] == 'p':
            if rs is None: rs, rl = d, 1
            else: rl += 1
        else:
            if rs is not None: chains.append(rl)
            rs, rl = None, 0
    if rs is not None: chains.append(rl)
    reaches_msb = (gkp[ND-1] == 'p')
    msb_cl = 0
    if reaches_msb:
        d = ND-1
        while d >= 0 and gkp[d] == 'p': msb_cl += 1; d -= 1
    return {'max_chain_len': max(chains, default=0), 'n_propagate': sum(1 for g in gkp if g == 'p'),
            'chain_reaches_msb': reaches_msb, 'msb_carry_out': bool(carry),
            'msb_chain_len': msb_cl, 'gkp': gkp}

def _max_chain_len(a, b): return _chain_stats(a, b)['max_chain_len']

def build_tok():
    return CharTokenizer(list('0123456789+='), {'mask': 'M', 'pad': 'P'})


# Data generation
def gen_data_natural(n, seed):
    """Operands sampled uniformly among ND-digit numbers."""
    rng = random.Random(seed)
    lo, hi = 10**(ND-1), 10**ND - 1
    return [_fmt_plain(rng.randint(lo, hi), rng.randint(lo, hi)) for _ in range(n)]

def gen_min_chain_test(n, seed, min_chain):
    """Instances whose longest carry chain (run of s_d = 9) has length >= min_chain."""
    rng = random.Random(seed); results = []; seen = set()
    for _ in range(n * 50):
        if len(results) >= n: break
        max_start = ND - min_chain
        if max_start < 0: break
        chain_start = rng.randint(0, max_start)
        a_digits = [0] * ND; b_digits = [0] * ND
        for d in range(ND):
            if chain_start <= d < chain_start + min_chain:
                a_d = rng.randint(0, 9); b_d = 9 - a_d
            else:
                a_d = rng.randint(0, 9); b_d = rng.randint(0, 9)
                while a_d + b_d == 9: b_d = rng.randint(0, 9)
            a_digits[d] = a_d; b_digits[d] = b_d
        if a_digits[ND-1] == 0: a_digits[ND-1] = rng.randint(1, 9)
        if b_digits[ND-1] == 0: b_digits[ND-1] = rng.randint(1, 9)
        if chain_start <= ND-1 < chain_start + min_chain:
            a_digits[ND-1] = rng.randint(1, 4); b_digits[ND-1] = 9 - a_digits[ND-1]
        a_str = ''.join(str(d) for d in reversed(a_digits))
        b_str = ''.join(str(d) for d in reversed(b_digits))
        a, b = int(a_str), int(b_str)
        if (a, b) in seen: continue
        seen.add((a, b))
        if _max_chain_len(a, b) >= min_chain:
            results.append(_fmt_plain(a, b))
    return results[:n]


# Test suite
def _annotate_sample(s):
    """Structural annotations used by addition_decode_analysis.py."""
    a, b = _parse_operands(s)
    return {'a': a, 'b': b, 'chain_stats': _chain_stats(a, b),
            'dep_ctx': _dependency_context_at_pos(a, b)}


def _bucket_from_samples(samples, tokenizer, max_len):
    """Package a sample list with encoded ids and annotations."""
    metas = [_annotate_sample(s) for s in samples]
    ids, ans = encode_samples(samples, tokenizer, max_len)
    return {'samples': samples, 'metas': metas, 'ids': ids, 'ans_starts': ans,
            'n': len(samples)}


def build_test_suite(seed=None):
    """natural: N_TEST instances of the training distribution;
    constructed['chain_{k}']: N_PER_BUCKET instances with a carry chain >= k."""
    if seed is None: seed = SEED + 1000
    suite = {'natural': gen_data_natural(N_TEST, seed), 'constructed': {}}
    sweep_lengths = [2, 3, 4, 6, 8, 12]
    if ND >= 24: sweep_lengths += [16, 20]
    if ND >= 32: sweep_lengths += [24, 28]
    sweep_lengths = [cl for cl in sweep_lengths if cl <= ND]
    for min_cl in sweep_lengths:
        sp = gen_min_chain_test(N_PER_BUCKET, seed=seed + 500 + min_cl, min_chain=min_cl)
        if sp:
            suite['constructed'][f'chain_{min_cl}'] = sp
    print(f"  Test suite: natural {len(suite['natural'])}, " +
          ', '.join(f"{k} {len(v)}" for k, v in suite['constructed'].items()))
    return suite


# Evaluation

@torch.no_grad()
def probe_loss(model, tokenizer, test_samples, max_len, device=None):
    """Fully-masked probe: every answer cell masked, one forward pass; mean NLL
    and argmax accuracy (mask token excluded) over all answer cells."""
    if device is None: device = DEVICE
    model.eval(); mask_id = tokenizer.special_ids['mask']
    ids_all, ans_all = encode_samples(test_samples, tokenizer, max_len)
    ids_all, ans_all = ids_all.to(device), ans_all.to(device)
    L = torch.zeros(ANS_LEN, device=device); C = torch.zeros(ANS_LEN, device=device)
    N = torch.zeros(ANS_LEN, device=device)
    _arange = torch.arange(ANS_LEN, device=device)
    for st in range(0, len(test_samples), 128):
        en = min(st+128, len(test_samples))
        ids, ans = ids_all[st:en], ans_all[st:en]; B, T = ids.shape
        ans_pos = (ans.unsqueeze(1) + _arange).clamp(max=T-1)
        bi = torch.arange(B, device=device).unsqueeze(1).expand_as(ans_pos)
        xm = ids.clone(); xm[bi, ans_pos] = mask_id
        logits = model(xm); al = logits[bi, ans_pos]; tgt = ids[bi, ans_pos]
        lp = F.log_softmax(al, dim=-1)
        losses = -lp.gather(2, tgt.unsqueeze(2)).squeeze(2)
        cl = al.clone(); cl[:, :, mask_id] = -float('inf')
        corrects = (cl.argmax(dim=-1) == tgt).float()
        L += losses.sum(dim=0)
        C += corrects.sum(dim=0)
        N += B
    s = N.clamp(1)
    return {'overall_loss': (L.sum()/s.sum()).item(), 'overall_acc': (C.sum()/s.sum()).item()}


@torch.no_grad()
def gen_accuracy(model, tokenizer, test_samples, decode_policy='confidence', device=None):
    """Exact-match accuracy of greedy decoding ('confidence', 'lsb' or 'random')."""
    if device is None: device = DEVICE
    mask_id = tokenizer.special_ids['mask']; pad_id = tokenizer.special_ids['pad']
    model.eval(); n_correct = 0
    policy = 'r2l' if decode_policy == 'lsb' else decode_policy   # LSB is rightmost
    for st in range(0, len(test_samples), 128):
        batch = test_samples[st:st+128]; B = len(batch)
        penc = [tokenizer.encode(s.split('=')[0]+'=') for s in batch]
        pm = max(len(p) for p in penc)
        pids = torch.full((B, pm), pad_id, dtype=torch.long)
        for i, e in enumerate(penc): pids[i, :len(e)] = torch.tensor(e)
        gen, _, _ = generate_diffusion(model, pids, ANS_LEN, mask_id,
                                       policy=policy, device=device)
        pred = gen[:, pm:pm+ANS_LEN]
        for i in range(B):
            n_correct += int(tokenizer.decode(pred[i].cpu().tolist()) == get_answer(batch[i]))
    return {'accuracy': n_correct / max(len(test_samples), 1), 'n': len(test_samples)}


# Training

def train_model(mask_type, tokenizer, train_samples, suite, max_len,
                init_state=None, device=None,
                seed_train=None, ckpt_schedule=None, ckpt_save_fn=None):
    """train_diffusion with the addition eval callback (EMA weights):
        every eval:           fully-masked probe on the natural test set
        every GEN_EVAL_EVERY: gen_acc_confidence on natural[:GEN_EVAL_N] and
                              gen_acc_{bucket}_{policy} on TAIL_GEN_BUCKETS
    """
    if device is None: device = DEVICE
    max_iters = MAX_ITERS
    train_ids, train_ans = encode_samples(train_samples, tokenizer, max_len)
    train_ids, train_ans = train_ids.to(device), train_ans.to(device)
    natural_samples = suite['natural']

    tail_buckets = [(bk, suite['constructed'][bk]) for bk in TAIL_GEN_BUCKETS
                    if suite['constructed'].get(bk)]

    k_sched = None
    if mask_type == 'puma':
        k_step = PUMA_K_STEP or 3
        if PUMA_K_EVERY is not None:
            k_every = PUMA_K_EVERY
        else:
            n_increments = max(1, (PUMA_K_END - PUMA_K_START) // k_step)
            k_every = max(1000, (max_iters // 3) // n_increments)
        k_sched = puma_k_step(PUMA_K_START, PUMA_K_END, k_step, k_every)
        print(f"  PUMA K: step {PUMA_K_START}->{k_sched(max_iters)} "
              f"(+{k_step} every {k_every//1000}k, cap={PUMA_K_END})")

    def eval_fn(model, it, tg):
        probe = probe_loss(model, tokenizer, natural_samples, max_len, device)
        print(f"    [eval it {it}] loss={probe['overall_loss']:.4f} "
              f"acc={probe['overall_acc']:.4f}")
        if it > 0 and it % GEN_EVAL_EVERY == 0:
            r = gen_accuracy(model, tokenizer, natural_samples[:GEN_EVAL_N], 'confidence',
                             device=device)
            probe['gen_acc_confidence'] = r['accuracy']
            probe['gen_n'] = r['n']
            print(f"      [gen] natural confidence={r['accuracy']:.3f}")
            for bk, sp in tail_buckets:
                for dp in TAIL_GEN_POLICIES:
                    rt = gen_accuracy(model, tokenizer, sp, dp, device=device)
                    probe[f'gen_acc_{bk}_{dp}'] = rt['accuracy']
                    print(f"      [gen] {bk} {dp}={rt['accuracy']:.3f} (n={rt['n']})")
        return probe

    return train_diffusion(
        train_ids=train_ids, train_ans=train_ans, ans_len=ANS_LEN, tokenizer=tokenizer,
        mask_type=mask_type, blank_masks=None,
        puma_tau=PUMA_TAU, puma_k_schedule=k_sched,
        papl_tau=PAPL_TAU, papl_alpha=PAPL_ALPHA,
        n_layer=N_LAYER, n_head=N_HEAD, n_embd=N_EMBD, dropout=DROPOUT,
        max_iters=max_iters, batch_size=BATCH_SIZE,
        lr=LR, min_lr=MIN_LR, warmup_iters=WARMUP_ITERS,
        grad_clip=GRAD_CLIP, weight_decay=WEIGHT_DECAY, ema_decay=EMA_DECAY,
        eval_fn=eval_fn, eval_every=EVAL_EVERY, log_every=LOG_EVERY,
        init_state=init_state, device=device,
        use_amp=False if NO_AMP else None,
        seed_train=seed_train, ckpt_schedule=ckpt_schedule, ckpt_save_fn=ckpt_save_fn,
    )


def _make_ckpt_saver(exp_name, mask_type, seed, tags_out):
    """ckpt_save_fn writing checkpoint_seed{seed}_{method}_iter{it:06d}.pt."""
    def _save(state_cpu, it, _mt=mask_type, _seed=seed, _tags=tags_out):
        t = f'seed{_seed}_{_mt}_iter{it:06d}'
        save_checkpoint(exp_name, state_cpu, tag=t)
        _tags.append(t)
    return _save


def figure_basis(ckpt_iters):
    """Metadata describing how the dynamics curves were computed."""
    return {
        'loss_curve': f'dyn_*.train_loss: online-weight batch loss every LOG_EVERY={LOG_EVERY}',
        'acc_curve': (f"dyn_*.checkpoints[*].gen_acc_confidence: EMA weights, confidence "
                      f"decode, natural[:GEN_EVAL_N={GEN_EVAL_N}] of suite seed {SEED + 1000}, "
                      f"every GEN_EVAL_EVERY={GEN_EVAL_EVERY}"),
        'tail_curves': {f'gen_acc_{bk}_{dp}': f"suite['constructed']['{bk}'] (n=N_PER_BUCKET), {dp} decode"
                        for bk in TAIL_GEN_BUCKETS for dp in TAIL_GEN_POLICIES},
        'ckpt_iters': list(ckpt_iters),
        'ckpt_note': 'EMA snapshot at iter it == weights evaluated at it (saved right after the EMA update of step it)',
    }


# Training-dynamics figure

def _smooth(ys, w):
    """Centred moving average with edge padding; w <= 1 returns ys unchanged."""
    if not w or w <= 1:
        return list(ys)
    import numpy as np
    arr = np.asarray(ys, dtype=float)
    pad = w // 2
    arr = np.pad(arr, (pad, w - 1 - pad), mode='edge')
    return np.convolve(arr, np.ones(w) / w, mode='valid').tolist()


def _fig_acc_trajectory(all_dyn, bucket=None, smooth=None):
    """Single panel: for every scheme, EMA generation accuracy under confidence
    decoding on natural[:GEN_EVAL_N] (solid) and on the `bucket` stratum
    (dashed with markers), at the GEN_EVAL_EVERY grid. Only the first entry per
    iteration is used (the post-training entry at max_iters, which re-evaluates
    the selected EMA state, is skipped)."""
    if bucket is None: bucket = TAIL_FIG_BUCKET
    if smooth is None: smooth = ACC_FIG_SMOOTH
    key_tail = f'gen_acc_{bucket}_confidence'
    mts = [mt for mt in ['random', 'papl', 'puma'] if mt in all_dyn]
    if not mts:
        return None
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.set_facecolor('#eeeeee'); ax.grid(color='white', lw=1.0); ax.set_axisbelow(True)
    for sp in ax.spines.values(): sp.set_visible(False)
    have_tail = False
    for mt in mts:
        col = FIG_COLORS.get(mt, '#444444')
        seen = set(); pts = []
        for c in all_dyn[mt].get('checkpoints', []):
            if 'gen_acc_confidence' not in c or c['iter'] in seen: continue
            seen.add(c['iter']); pts.append(c)
        if not pts:
            continue
        ax.plot([c['iter'] for c in pts],
                _smooth([c['gen_acc_confidence'] for c in pts], smooth),
                '-', color=col, lw=2.0)
        tail = [c for c in pts if key_tail in c]
        if tail:
            have_tail = True
            ax.plot([c['iter'] for c in tail],
                    _smooth([c[key_tail] for c in tail], smooth),
                    '--', marker='o', ms=3.5, color=col, lw=1.6)
    ax.set_ylim(-0.02, 1.02); ax.set_xlim(left=0)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{int(round(v / 1000))}k' if v else '0'))
    ax.set_xlabel('Training iteration')
    ax.set_ylabel('Exact-match accuracy (confidence decode)')
    handles = [Line2D([0], [0], color=FIG_COLORS.get(mt, '#444444'), lw=2.0,
                      label=FIG_LABELS.get(mt, mt)) for mt in mts]
    handles.append(Line2D([0], [0], color='#444444', lw=2.0, label='natural'))
    if have_tail:
        handles.append(Line2D([0], [0], color='#444444', lw=1.6, ls='--', marker='o',
                              ms=3.5, label=bucket.replace('chain_', 'chain ≥ ')))
    ax.legend(handles=handles, frameon=False, fontsize=8, ncol=len(handles),
              loc='lower center', bbox_to_anchor=(0.5, 1.0), handlelength=2.2)
    fig.tight_layout()
    return fig


# Run

def _build_shared_init(tok, max_len, seed):
    """Deterministic init state (CPU state_dict) for the given seed; shared by
    all schemes within one seed in run_training()."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    init_model = Transformer(
        vocab_size=len(tok), block_size=max_len + 8,
        n_layer=N_LAYER, n_head=N_HEAD, n_embd=N_EMBD, dropout=DROPOUT,
    )
    return {k: v.cpu().clone() for k, v in init_model.state_dict().items()}


def _dyn_record(d):
    return {'checkpoints': d['checkpoints'], 'train_loss': d['train_loss'],
            'ckpt_tags': d.get('ckpt_tags', [])}


def run_training(tag=''):
    """Training only: one shared init state per SEED; scheme i is trained from
    it with training seed SEED * 100 + i. Writes to {RESULTS_DIR}/{exp_name}/:
        checkpoint_init_seed{SEED}.pt
        checkpoint_seed{SEED}_{method}_iter{iter:06d}.pt
        results_seed{SEED}_train.json, acc_trajectory_seed{SEED}_train.png
    """
    exp_name = f'{EXP_NAME}_{tag}' if tag else EXP_NAME
    prepare_results_dir()
    torch.manual_seed(SEED); random.seed(SEED)
    tok = build_tok()
    ckpt_iters = resolve_ckpt_iters(MAX_ITERS)
    print(f"\n{'='*70}\n  Addition (training only) | seed={SEED} | masks={MASK_TYPES}\n"
          f"  {len(ckpt_iters)} EMA snapshots per scheme | exp_name: {exp_name}\n{'='*70}\n")

    train_data = gen_data_natural(N_TRAIN, seed=SEED)
    max_len = max(len(tok.encode(s)) for s in train_data)
    suite = build_test_suite(seed=SEED + 1000)

    shared_init = _build_shared_init(tok, max_len, SEED)
    init_tag = f'init_seed{SEED}'
    save_checkpoint(exp_name, shared_init, tag=init_tag)

    all_dyn, method_ckpt_tags, method_seeds = {}, {}, {}
    for mi, mt in enumerate(MASK_TYPES):
        method_seed = SEED * 100 + mi
        method_seeds[mt] = method_seed
        print(f"\n=== {mt}  (seed={SEED}, train_seed={method_seed}) ===")
        ckpt_tags = []
        _save = _make_ckpt_saver(exp_name, mt, SEED, ckpt_tags)
        m, d = train_model(mt, tok, train_data, suite, max_len,
                           init_state=shared_init, seed_train=method_seed,
                           ckpt_schedule=ckpt_iters if ckpt_iters else None,
                           ckpt_save_fn=_save if ckpt_iters else None)
        d['ckpt_tags'] = list(ckpt_tags)
        all_dyn[mt] = d
        method_ckpt_tags[mt] = ckpt_tags
        del m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    config_keys = ['ND', 'ANS_LEN', 'N_TRAIN', 'N_TEST', 'N_PER_BUCKET', 'MAX_ITERS',
                   'BATCH_SIZE', 'N_LAYER', 'N_HEAD', 'N_EMBD', 'DROPOUT', 'MASK_TYPES',
                   'SEED', 'CKPT_ITERS', 'LOG_EVERY', 'EVAL_EVERY', 'GEN_EVAL_EVERY',
                   'GEN_EVAL_N', 'TAIL_GEN_BUCKETS', 'TAIL_GEN_POLICIES', 'PUMA_TAU',
                   'PUMA_K_START', 'PUMA_K_END', 'PUMA_K_STEP', 'PUMA_K_EVERY',
                   'PAPL_TAU', 'PAPL_ALPHA', 'NO_AMP', 'LR', 'MIN_LR', 'WARMUP_ITERS',
                   'GRAD_CLIP', 'WEIGHT_DECAY', 'EMA_DECAY']
    sd = {'config': {k: globals()[k] for k in config_keys},
          'seed': SEED, 'train_seeds': method_seeds, 'init_ckpt_tag': init_tag,
          'method_ckpt_tags': method_ckpt_tags, 'figure_basis': figure_basis(ckpt_iters)}
    for mt, d in all_dyn.items():
        sd[f'dyn_{mt}'] = _dyn_record(d)
    figs = {}
    f0 = _fig_acc_trajectory(all_dyn)
    if f0 is not None: figs['acc_trajectory'] = f0
    save_results(exp_name, sd, figures=figs, tag=f'seed{SEED}_train')
    return all_dyn


def run(tag=''):
    exp_name = f'{EXP_NAME}_{tag}' if tag else EXP_NAME
    prepare_results_dir()
    torch.manual_seed(SEED); random.seed(SEED)
    tok = build_tok()
    ckpt_iters = resolve_ckpt_iters(MAX_ITERS)

    print(f"\n{'='*70}")
    print(f"  Addition ND={ND} | masks={MASK_TYPES} | decode={DECODE_POLICIES}")
    print(f"  N_TRAIN={N_TRAIN} N_TEST={N_TEST} MAX_ITERS={MAX_ITERS}")
    print(f"  arch: {N_LAYER}L/{N_HEAD}H/{N_EMBD}D | {len(ckpt_iters)} EMA snapshots per scheme")
    print(f"{'='*70}\n")

    train_data = gen_data_natural(N_TRAIN, seed=SEED)
    max_len = max(len(tok.encode(s)) for s in train_data)
    suite = build_test_suite(seed=SEED + 1000)
    sweep_keys = sorted(suite['constructed'], key=lambda k: int(k.split('_')[1]))

    all_dyn = {}; all_final = {}; method_ckpt_tags = {}
    for mt in MASK_TYPES:
        print(f"\n=== {mt} ===")
        ckpt_tags = []
        _save = _make_ckpt_saver(exp_name, mt, SEED, ckpt_tags)
        m, d = train_model(mt, tok, train_data, suite, max_len,
                           ckpt_schedule=ckpt_iters if ckpt_iters else None,
                           ckpt_save_fn=_save if ckpt_iters else None)
        d['ckpt_tags'] = list(ckpt_tags)
        all_dyn[mt] = d
        method_ckpt_tags[mt] = ckpt_tags
        save_checkpoint(exp_name, {k: v.cpu().clone() for k, v in m.state_dict().items()}, tag=mt)

        for dp in DECODE_POLICIES:
            r = gen_accuracy(m, tok, suite['natural'], dp, device=DEVICE)
            all_final[f'{mt}_standard_{dp}'] = r
            print(f"    standard {dp}: {r['accuracy']:.4f}  (n={r['n']})")
        print("  Carry-chain strata...")
        for key in sweep_keys:
            min_cl = int(key.split('_')[1])
            for dp in DECODE_POLICIES:
                r = gen_accuracy(m, tok, suite['constructed'][key], dp, device=DEVICE)
                all_final[f'{mt}_chain_sweep_{min_cl}_{dp}'] = r
                print(f"    chain>={min_cl:2d} {dp}: {r['accuracy']:.4f}")

        del m; torch.cuda.empty_cache() if torch.cuda.is_available() else None

    figs = {}
    f0 = _fig_acc_trajectory(all_dyn)
    if f0 is not None: figs['acc_trajectory'] = f0
    sd = {'config': {k: globals()[k] for k in
           ['ND', 'ANS_LEN', 'N_TRAIN', 'N_TEST', 'N_PER_BUCKET', 'MAX_ITERS',
            'BATCH_SIZE', 'N_LAYER', 'N_HEAD', 'N_EMBD', 'MASK_TYPES', 'DECODE_POLICIES',
            'SEED', 'LOG_EVERY', 'EVAL_EVERY', 'GEN_EVAL_EVERY', 'GEN_EVAL_N',
            'TAIL_GEN_BUCKETS', 'TAIL_GEN_POLICIES', 'CKPT_ITERS']},
          'figure_basis': figure_basis(ckpt_iters),
          'method_ckpt_tags': method_ckpt_tags}
    for k, v in all_dyn.items():
        sd[f'dyn_{k}'] = _dyn_record(v)
    for k, v in all_final.items():
        sd[f'final_{k}'] = v
    save_results(exp_name, sd, figures=figs)

    print(f"\n{'='*70}\n  SUMMARY (exact match)\n{'='*70}")
    print(f"\n  {'Test':<35s}", end='')
    for mt in MASK_TYPES: print(f" {mt:>14s}", end='')
    print()
    for dp in DECODE_POLICIES:
        rows = [('standard_' + dp, f'standard_{dp}')] + \
               [(f'chain>={k.split("_")[1]}_{dp}', f'chain_sweep_{k.split("_")[1]}_{dp}')
                for k in sweep_keys]
        for label, key in rows:
            accs = [all_final.get(f'{mt}_{key}', {}).get('accuracy') for mt in MASK_TYPES]
            print(f"  {label:<35s}", end='')
            for a in accs: print(f" {a:>14.4f}" if a is not None else f" {'N/A':>14s}", end='')
            print()
    return all_dyn, all_final


if __name__ == '__main__':
    args = parse_args()
    seeds = args.seeds if args.seeds else [SEED]
    for si, seed in enumerate(seeds):
        globals()['SEED'] = seed
        t = (f"{args.tag}_s{seed}" if args.tag else f"s{seed}") if len(seeds) > 1 else args.tag
        if len(seeds) > 1:
            print(f"\n{'#'*70}\n# Seed {seed} ({si+1}/{len(seeds)})\n{'#'*70}")
        if args.train_only:
            run_training(tag=t)
        else:
            run(tag=t)
