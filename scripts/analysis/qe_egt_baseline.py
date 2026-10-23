#!/usr/bin/env python3
"""EGT-style traversal (Chang et al., CVPR 2019) under the Track V-QE online budget.

rEGT verifies the whole database k-NN graph offline and traverses it with an explore/exploit rule. Here
only query-time checks are allowed (B per query), so the rule becomes: verify, among available
candidates, the unlocked neighbour whose best verified parent has the most inliers if that count is
>= t (exploit); otherwise the most similar available candidate (explore). t = inf is plain
similarity order. t is chosen on GLDv2 validation (mean of plain and distractor pools, the same split
the policies are selected on); benchmarks are evaluated once.
Usage: python scripts/analysis/qe_egt_baseline.py --qe_pre DIR [DIR ...] --out results/qe/egt.json
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_qe_policy import load_pool, sub_env                    # noqa: E402
from training.qe_env import QEEnv                                  # noqa: E402

T_GRID = [35, 50, 100, 200, 400, float('inf')]


@torch.no_grad()
def egt_rollout(env, t):
    idx = torch.arange(env.Q, device=env.d['sim'].device)
    verified = torch.zeros(env.Q, env.C, dtype=torch.bool, device=idx.device)
    inl = env.d['inliers'].float()
    for _ in range(env.B):
        avail = env.available(idx, verified)
        if not avail.any():
            break
        vm = (verified & env.match).float() * inl                         # inliers of verified matches
        parent = (vm[:, :, None] * env.d['unlock'].float()).amax(1)         # best verified parent per candidate
        score = torch.where((env.d['hop'] > 0) & (parent >= t), 1e3 + parent, env.d['sim'] * 100)
        score = score.masked_fill(~avail, float('-inf'))
        has = avail.any(1)
        a = score.argmax(1)
        step = torch.zeros_like(verified); step[idx, a] = True
        verified = verified | (step & has[:, None])
    return env.reward(idx, verified)


def mean_ap(d, B, t, thr, chunk=512):
    env = QEEnv(d, B, thr)
    order = torch.argsort(d['valid'].sum(1))
    out = torch.empty(env.Q, device=d['sim'].device)
    for s in range(0, env.Q, chunk):
        ids = order[s:s + chunk]
        out[ids] = egt_rollout(sub_env(env, ids), t)
    return float(out.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--qe_pre', nargs='+', required=True)
    ap.add_argument('--budgets', type=int, nargs='+', default=[20, 40])
    ap.add_argument('--threshold', type=int, default=35)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    res = {}
    for pre in a.qe_pre:
        name = Path(pre).name
        val = {}
        for p in ('gldv2c_val', 'gldv2c_val_dist'):
            d, z = load_pool(Path(pre) / f'{p}.npz', dev)
            lm = z['labels'][z['queries']]
            va = torch.tensor(np.where((lm * 2654435761 % 3) == 0)[0], device=dev)  # policy validation split
            val[p] = {k: v[va] for k, v in d.items()}
        res[name] = {}
        for B in a.budgets:
            sel = {t: float(np.mean([mean_ap(val[p], B, t, a.threshold) for p in val])) for t in T_GRID}
            t_best = max(sel, key=sel.get)
            r = {'val_by_t': {str(k): v for k, v in sel.items()}, 't': str(t_best)}
            for pool in ('roxford5k', 'rparis6k', 'roxford5k_r1m', 'rparis6k_r1m'):
                for prot in ('medium', 'hard'):
                    d, _ = load_pool(Path(pre) / f'{pool}.npz', dev, prot)
                    r[f'{pool}_{prot}'] = mean_ap(d, B, t_best, a.threshold)
            res[name][f'B{B}'] = r
            print(name, B, 't =', t_best, json.dumps({k: round(100 * v, 2) for k, v in r.items() if isinstance(v, float)}), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
