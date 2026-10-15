#!/usr/bin/env python3
"""Paired bootstrap over queries for Track V (results/trackv/<run>/per_query.npz).
Usage: python scripts/analysis/trackv_bootstrap.py results/trackv/<run> [--ref rl]"""
import argparse, json, sys
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('run'); ap.add_argument('--ref', default='rl'); ap.add_argument('--n', type=int, default=10000)
a = ap.parse_args()
z = np.load(Path(a.run) / 'per_query.npz')
keys = {tuple(k.split('|')) for k in z.files}
rng = np.random.default_rng(0)
out = {}
for B, m, pool in sorted(keys):
    if m == a.ref:
        continue
    ref = z.get(f'{B}|{a.ref}|{pool}')
    if ref is None:
        continue
    d = 100 * (ref - z[f'{B}|{m}|{pool}'])
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(a.n)])
    out[f'{B} {a.ref}-{m} {pool}'] = {'mean': round(float(d.mean()), 2), 'ci95': [round(float(np.percentile(boots, 2.5)), 2),
                                      round(float(np.percentile(boots, 97.5)), 2)], 'queries_better': int((d > 1e-9).sum()),
                                      'queries_worse': int((d < -1e-9).sum()), 'n': len(d)}
for k, v in out.items():
    print(f"{k:45s} {v['mean']:+6.2f}  CI95 [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}]  better/worse {v['queries_better']}/{v['queries_worse']} of {v['n']}")
(Path(a.run) / f'bootstrap_{a.ref}.json').write_text(json.dumps(out, indent=1))
