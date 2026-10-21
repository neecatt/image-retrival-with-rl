#!/usr/bin/env python3
"""Track V over several descriptor seeds: per-seed selection on validation, pooled comparison.

--seed name=dir1,dir2,... gives each seed's run dirs (each a configuration; a dir may hold only some
methods). Per seed and budget, every method's configuration is chosen on its validation selection score
(results.json res[method]['selection']); no benchmark number is used. Then, per pool, RL - baseline
per-query differences are averaged over seeds (the queries are the same in every seed) and
bootstrapped over queries; the per-seed mean differences and a one-sample t over seeds are reported too.
Usage: python scripts/analysis/trackv_seeds.py --seed s42=dirA,dirB --seed s1=... [--out out.json]
"""
import argparse, json
from pathlib import Path

import numpy as np
from scipy import stats

ap = argparse.ArgumentParser()
ap.add_argument('--seed', action='append', required=True)
ap.add_argument('--methods', nargs='+', default=['classifier', 'value', 'rl'])
ap.add_argument('--ref', default='rl')
ap.add_argument('--budgets', type=int, nargs='+', default=[5, 10, 20])
ap.add_argument('--n', type=int, default=10000)
ap.add_argument('--out')
a = ap.parse_args()
rng = np.random.default_rng(0)
HEUR = ['none', 'top', 'shortlist', 'all_K']          # whichever the run type has
seeds = {}
for spec in a.seed:
    name, dirs = spec.split('=', 1)
    seeds[name] = [(Path(d).name, json.loads((Path(d) / 'results.json').read_text()), np.load(Path(d) / 'per_query.npz'))
                   for d in dirs.split(',')]
first = seeds[next(iter(seeds))][0][1]
pools = list(first['pools'])
out = {}
for B in a.budgets:
    per = {}                       # seed -> method -> (dir name, means, per-query dict)
    for s, runs in seeds.items():
        per[s] = {}
        for m in a.methods + [h for h in HEUR if any(h in r[f'B{B}'] for _, r, _ in runs)]:
            cand = [(r[f'B{B}'][m].get('selection', 0.0), n, r, z) for n, r, z in runs if m in r[f'B{B}']]
            sc, n, r, z = max(cand, key=lambda t: t[0])
            per[s][m] = (n, r[f'B{B}'][m], z)
    print(f"\n=== B={B}  selected per seed: " + '; '.join(
        f"{s}: " + ','.join(f"{m[0]}={per[s][m][0].split('_', 3)[-1]}" for m in a.methods) for s in seeds))
    shown = [h for h in HEUR if h != 'all_K' and all(h in per[s] for s in seeds)] + a.methods + \
            [h for h in HEUR if h == 'all_K' and all(h in per[s] for s in seeds)]
    means = {m: {p: float(np.mean([per[s][m][1][p] for s in seeds])) for p in pools} for m in shown}
    print(f"{'mean over seeds':14s} " + ' '.join(f"{p[:20]:>20s}" for p in pools))
    for m in means:
        print(f"{m:14s} " + ' '.join(f"{100 * means[m][p]:20.2f}" for p in pools))
    out[f'B{B}'] = {'means': means, 'selected': {s: {m: per[s][m][0] for m in a.methods} for s in seeds}, 'cmp': {}}
    for m in a.methods:
        if m == a.ref:
            continue
        for p in pools:
            if p == 'gld_train':
                continue
            diffs = np.stack([100 * (per[s][a.ref][2][f'B{B}|{a.ref}|{p}'] - per[s][m][2][f'B{B}|{m}|{p}']) for s in seeds])
            d = diffs.mean(0)
            bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(a.n)])
            lo, hi = np.percentile(bs, [2.5, 97.5])
            seed_means = diffs.mean(1)
            t = stats.ttest_1samp(seed_means, 0.0)
            out[f'B{B}']['cmp'][f'{a.ref}-{m} {p}'] = {'mean': float(d.mean()), 'ci95': [float(lo), float(hi)],
                                                       'per_seed': [round(float(x), 3) for x in seed_means],
                                                       'seeds_positive': int((seed_means > 0).sum()), 't': float(t.statistic),
                                                       'p_seeds': float(t.pvalue)}
            flag = '*' if lo > 0 or hi < 0 else ' '
            print(f"  {a.ref}-{m:10s} {p:24s} {d.mean():+6.2f} [{lo:+.2f}, {hi:+.2f}] {flag}  seeds+ {(seed_means > 0).sum()}/{len(seeds)}  "
                  f"per-seed {' '.join(f'{x:+.2f}' for x in seed_means)}  p={t.pvalue:.3f}")
if a.out:
    Path(a.out).write_text(json.dumps(out, indent=1))
