"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Experiment 2 — Tree Expression Evaluation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Colab:  %run experiments/exp_tree.py

  Trees test parallel computation ability:
    ((3+5)*(2+7))=072  →  level-wise reduce, inherently parallel

  Key difference from addition:
    - No sequential carry chain
    - Scratchpad shows LEVEL-wise intermediates (parallel structure)
    - Scratchpad output is longer → diffusion decode order matters more

  Controls:
    - Same greedy decoding, convergence-based training
    - RoPE tested for depth generalisation
    - Full scratchpad decode order analysis
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                if '__file__' in dir() else '.')

import random, torch
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('Agg')
import numpy as np

from core.tokenizer import CharTokenizer
from core.train_utils import (
    mount_drive, save_results, train_model, evaluate,
    analyse_decode_order, DEVICE,
)

EXP_NAME = 'exp_tree'

# ── Config ──────────────────────────────────────────
N_TRAIN      = 10_000
N_TEST       = 2_000
MAX_ITERS    = 15_000
PATIENCE     = 2_000
MOD          = 1000
DEPTH_TRAIN  = [2, 3]
DEPTH_OOD    = [4, 5]
LEAVES_TRAIN = list(range(8))
LEAVES_TEST  = list(range(10))
POS_ENCS     = ['absolute', 'rope']
FORMATS      = ['plain', 'scratchpad']

# ── Tree generation ─────────────────────────────────

class TreeNode:
    def __init__(self, val=None, op=None, left=None, right=None):
        self.val, self.op, self.left, self.right = val, op, left, right

    def is_leaf(self): return self.val is not None

    def evaluate(self, mod=MOD):
        if self.is_leaf(): return self.val % mod
        l, r = self.left.evaluate(mod), self.right.evaluate(mod)
        return ((l + r) if self.op == '+' else (l * r)) % mod

    def to_string(self):
        if self.is_leaf(): return str(self.val)
        return f"({self.left.to_string()}{self.op}{self.right.to_string()})"

    def level_values(self, mod=MOD):
        result = {}
        self._collect(result, 0, mod)
        if not result: return {}
        mx = max(result.keys())
        return {mx - d + 1: vs for d, vs in result.items()}

    def _collect(self, result, depth, mod):
        if self.is_leaf(): return
        self.left._collect(result, depth + 1, mod)
        self.right._collect(result, depth + 1, mod)
        result.setdefault(depth, []).append(self.evaluate(mod))


def gen_tree(depth, leaves, rng):
    if depth == 0: return TreeNode(val=rng.choice(leaves))
    return TreeNode(op=rng.choice(['+', '*']),
                    left=gen_tree(depth - 1, leaves, rng),
                    right=gen_tree(depth - 1, leaves, rng))


def tree_to_plain(t):
    return f"{t.to_string()}={str(t.evaluate()).zfill(3)}"


def tree_to_scratchpad(t):
    expr = t.to_string()
    ans = str(t.evaluate()).zfill(3)
    levels = t.level_values()
    parts = [f"[L{lv}:{','.join(str(v) for v in vs)}]"
             for lv, vs in sorted(levels.items())]
    return f"{expr}={''.join(parts)}=>{ans}"


def gen_tree_data(depths, n, leaves, fmt, seed=42):
    rng = random.Random(seed)
    fn = tree_to_plain if fmt == 'plain' else tree_to_scratchpad
    seen, out = set(), []
    for _ in range(n * 5):
        if len(out) >= n: break
        t = gen_tree(rng.choice(depths), leaves, rng)
        expr = t.to_string()
        if expr in seen: continue
        seen.add(expr)
        out.append(fn(t))
    return out[:n]


def get_tree_answer(s, fmt):
    if fmt == 'scratchpad': return s.split('=>')[-1]
    return s.split('=')[-1]

def get_tree_full(s):
    return s.split('=', 1)[1]

def build_splits(fmt):
    return {
        'train':          gen_tree_data(DEPTH_TRAIN, N_TRAIN, LEAVES_TRAIN, fmt, 42),
        'test_id':        gen_tree_data(DEPTH_TRAIN, N_TEST,  LEAVES_TRAIN, fmt, 1042),
        'test_ood_digit': gen_tree_data(DEPTH_TRAIN, N_TEST,  LEAVES_TEST,  fmt, 2042),
        # Depth OOD with IN-distribution digits → isolates depth effect
        'test_ood_depth': gen_tree_data(DEPTH_OOD,   N_TEST,  LEAVES_TRAIN, fmt, 3042),
        # Both OOD (hardest)
        'test_ood_both':  gen_tree_data(DEPTH_OOD,   N_TEST,  LEAVES_TEST,  fmt, 4042),
    }

TEST_SPLITS = ['test_id', 'test_ood_digit', 'test_ood_depth', 'test_ood_both']

def build_tok(fmt):
    chars = list('0123456789()+*=')
    if fmt == 'scratchpad': chars.extend(['[', ']', 'L', ':', ',', '>'])
    return CharTokenizer(chars, {'mask': 'M', 'pad': 'P'})


# ── Main ────────────────────────────────────────────

def run():
    print("=" * 70)
    print("  EXP 2: Tree Expression — AR vs Diffusion")
    print("=" * 70)
    mount_drive()
    torch.manual_seed(42)

    all_results = {}
    all_histories = {}
    convergence_iters = {}
    scratchpad_analysis = {}

    for pos_enc in POS_ENCS:
        for objective in ['ar', 'diffusion']:
            for fmt in FORMATS:
                key = f"{objective}_{fmt}_{pos_enc}"
                print(f"\n▶ {key}")

                splits = build_splits(fmt)
                tok = build_tok(fmt)
                print(f"  ex: {splits['train'][0]}")

                all_s = [s for v in splits.values() for s in v]
                max_len = max(len(tok.encode(s)) for s in all_s) + 1

                model, hist, ml, conv_it = train_model(
                    objective, tok, splits['train'], max_len=max_len,
                    max_iters=MAX_ITERS, patience=PATIENCE,
                    pos_enc=pos_enc, log_interval=500,
                )
                all_histories[key] = hist
                convergence_iters[key] = conv_it

                get_ans = lambda s, f=fmt: get_tree_answer(s, f)
                all_results[key] = {}

                for sp in TEST_SPLITS:
                    res = evaluate(
                        model, tok, splits[sp], objective, get_ans,
                        get_tree_full, policy='confidence', greedy=True,
                    )
                    all_results[key][sp] = res['exact_match']
                    print(f"    {sp}: {res['exact_match']:.4f}")

                    # Scratchpad order analysis
                    if objective == 'diffusion' and fmt == 'scratchpad' \
                            and sp == 'test_id' and 'decode_orders' in res:
                        ex = splits[sp][0]
                        full = get_tree_full(ex)
                        ans_start_pos = len(tok.encode(ex.split('=')[0])) + 1
                        sp_part = full.split('=>')[0] + '=>'
                        sp_end = ans_start_pos + len(tok.encode(sp_part))
                        total = len(tok.encode(ex))
                        sa = analyse_decode_order(
                            res['decode_orders'], ans_start_pos, sp_end, total)
                        scratchpad_analysis[key] = sa
                        print(f"    scratchpad_first: "
                              f"{sa.get('scratchpad_first_ratio', 'N/A')}")

                save_results(EXP_NAME, all_results, model=model, tag=key)

    all_results['convergence_iters'] = convergence_iters
    all_results['scratchpad_analysis'] = {
        k: {kk: float(vv) if isinstance(vv, (int, float)) else vv
            for kk, vv in v.items()}
        for k, v in scratchpad_analysis.items()
    }

    # ── Visualisation ──
    figs = {}
    configs = [k for k in all_results
               if isinstance(all_results[k], dict) and 'test_id' in all_results[k]]

    # Training curves
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for pe_idx, pe in enumerate(POS_ENCS):
        ax = axes[pe_idx]
        for k, h in all_histories.items():
            if k.endswith(pe):
                ax.plot(h['iter'], h['loss'], label=k.replace(f'_{pe}', ''),
                        alpha=0.8)
        ax.set_xlabel('Iteration'); ax.set_ylabel('Loss')
        ax.set_title(f'Training Loss ({pe})'); ax.legend(fontsize=6)
        ax.grid(alpha=0.3)
    fig.tight_layout(); figs['training_curves'] = fig

    # Accuracy
    split_order = TEST_SPLITS
    labels = ['ID (d2-3)', 'OOD Digit', 'OOD Depth (d4-5, 0-7)', 'OOD Both']
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    for idx, (sp, lb) in enumerate(zip(split_order, labels)):
        ax = axes[idx]
        vals = [all_results[k].get(sp, 0) for k in configs]
        colors = ['#e74c3c' if 'ar' in k else '#3498db' for k in configs]
        hatches = ['///' if 'rope' in k else '' for k in configs]
        bars = ax.bar(range(len(configs)), vals, color=colors, alpha=0.85)
        for bar, h in zip(bars, hatches):
            bar.set_hatch(h)
        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels(configs, fontsize=5, rotation=45, ha='right')
        ax.set_ylabel('Exact Match'); ax.set_title(lb); ax.set_ylim(0, 1.05)
        ax.grid(axis='y', alpha=0.3)
    fig.suptitle('Exp 2: Tree Expression (hatched = RoPE)', fontsize=13, y=1.02)
    fig.tight_layout(); figs['accuracy'] = fig

    # RoPE vs Absolute on depth OOD
    fig, ax = plt.subplots(figsize=(8, 6))
    for fmt in FORMATS:
        for obj in ['ar', 'diffusion']:
            abs_k = f"{obj}_{fmt}_absolute"
            rope_k = f"{obj}_{fmt}_rope"
            a = all_results.get(abs_k, {}).get('test_ood_depth', 0)
            r = all_results.get(rope_k, {}).get('test_ood_depth', 0)
            ax.scatter(a, r, s=80, marker='o' if obj == 'ar' else 's',
                       label=f"{obj}_{fmt}")
            ax.annotate(f"{obj}_{fmt}", (a, r), fontsize=6,
                        textcoords="offset points", xytext=(3, 3))
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.set_xlabel('Absolute PE (depth OOD)')
    ax.set_ylabel('RoPE (depth OOD)')
    ax.set_title('Depth Generalisation: RoPE vs Absolute')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.tight_layout(); figs['rope_vs_abs'] = fig

    save_results(EXP_NAME, all_results, figures=figs)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Config':<35} {'ID':>8} {'OOD-d':>8} {'OOD-D':>8} {'Both':>8} {'conv':>8}")
    print("-" * 75)
    for k in configs:
        r = all_results[k]
        print(f"{k:<35} {r.get('test_id',0):>8.4f} "
              f"{r.get('test_ood_digit',0):>8.4f} "
              f"{r.get('test_ood_depth',0):>8.4f} "
              f"{r.get('test_ood_both',0):>8.4f} "
              f"{convergence_iters.get(k,'?'):>8}")

    if scratchpad_analysis:
        print("\nScratchpad Decode Order (diffusion):")
        for k, v in scratchpad_analysis.items():
            print(f"  {k}: sp_first={v.get('scratchpad_first_ratio','N/A'):.3f}")

    plt.show()
    return all_results


if __name__ == '__main__':
    run()
