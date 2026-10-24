#!/usr/bin/env python3
"""Summarise qe_policy_vqe.py outputs: mean AP per scheduler (+ verified-QE re-query) and paired
bootstrap over queries of per-query differences averaged over descriptor seeds.
Usage: python scripts/analysis/qe_policy_vqe_summary.py results/qe/policy_vqe/newresnet_seed{42,1,2,3,4}.json
"""
import json, sys

import numpy as np

runs = [json.load(open(f)) for f in sys.argv[1:]]
rng = np.random.default_rng(0)
pools = list(runs[0]['pools'])
methods = list(runs[0]['pools'][pools[0]])
print('descriptors:', [r['desc'] for r in runs])
print(f"{'method':11s} " + ' '.join(f"{p[:13] + (' M' if pr == 'medium' else ' H'):>16s}" for p in pools for pr in ('medium', 'hard')))
for m in methods:
    print(f"{m:11s} " + ' '.join(f"{100 * np.nanmean([np.nanmean(r['pools'][p][m][pr]) for r in runs]):16.2f}"
                                 for p in pools for pr in ('medium', 'hard')))
print('matches per query:', {m: round(float(np.mean([r['pools'][pools[0]][m].get('n_matched', np.nan) for r in runs])), 2)
                             for m in methods if m != 'none'})
for a, b in (('rl', 'shortlist'), ('rl', 'value'), ('rl', 'classifier'), ('value', 'shortlist'), ('rl', 'top')):
    if a not in methods or b not in methods:
        continue
    for p in pools:
        for pr in ('medium', 'hard'):
            diffs = np.stack([100 * (np.array(r['pools'][p][a][pr]) - np.array(r['pools'][p][b][pr])) for r in runs])
            d = np.nanmean(diffs, 0); d = d[~np.isnan(d)]
            bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)])
            lo, hi = np.percentile(bs, [2.5, 97.5])
            seeds_pos = int((np.nanmean(diffs, 1) > 0).sum())
            print(f"  {a}-{b:10s} {p + ' ' + pr:22s} {d.mean():+6.2f} [{lo:+.2f}, {hi:+.2f}] {'*' if lo > 0 or hi < 0 else ' '}  seeds+ {seeds_pos}/{len(runs)}")
