#!/usr/bin/env python3
"""Track V model selection across runs, then paired bootstrap on the benchmarks.

For each budget and learned method, the configuration (run dir) with the best validation selection
score (results.json: res[method]['selection'], mean gld_val over the training pools) is chosen; no
benchmark number is used. RL is then compared with each selected baseline on per-query APs
(per_query.npz), bootstrap over queries.
Usage: python scripts/analysis/trackv_select.py results/trackv/vd_newresnet_* [--out sel.json]
"""
import argparse, json
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('runs', nargs='+')
ap.add_argument('--methods', nargs='+', default=['classifier', 'value', 'rl'])
ap.add_argument('--ref', default='rl')
ap.add_argument('--n', type=int, default=10000)
ap.add_argument('--out')
a = ap.parse_args()
runs = {Path(r).name: (json.loads((Path(r) / 'results.json').read_text()), np.load(Path(r) / 'per_query.npz'))
        for r in a.runs if (Path(r) / 'results.json').exists()}
rng = np.random.default_rng(0)
first = next(iter(runs.values()))[0]
budgets = sorted(int(k[1:]) for k in first if k.startswith('B') and all(k in r for r, _ in runs.values()))
pools = [p for p in first['pools'] if not p.startswith('gld_val') or True]
out = {}
for B in budgets:
    sel = {}
    for m in a.methods:
        cands = [(r[f'B{B}'][m]['selection'] if 'selection' in r[f'B{B}'][m] else r[f'B{B}'][m]['gld_val'], name)
                 for name, (r, _) in runs.items() if m in r[f'B{B}']]
        sel[m] = max(cands)[1]
    print(f"\n=== B={B}  selected: " + ', '.join(f"{m}={sel[m]}" for m in a.methods))
    rows = {m: runs[sel[m]][0][f'B{B}'][m] for m in a.methods}
    base = runs[sel[a.ref]][0][f'B{B}']
    heur = [m for m in ('none', 'top', 'shortlist', 'all_K') if m in base]   # whatever the run type has
    for m in heur:
        rows[m] = base[m]
    print(f"{'method':12s} " + ' '.join(f"{p[:22]:>22s}" for p in pools))
    for m in [h for h in heur if h != 'all_K'] + a.methods + [h for h in heur if h == 'all_K']:
        print(f"{m:12s} " + ' '.join(f"{100 * rows[m][p]:22.2f}" for p in pools))
    out[f'B{B}'] = {'selected': sel, 'means': {m: {p: rows[m][p] for p in pools} for m in rows}, 'bootstrap': {}}
    for m in a.methods:
        if m == a.ref:
            continue
        for p in pools:
            d = 100 * (runs[sel[a.ref]][1][f'B{B}|{a.ref}|{p}'] - runs[sel[m]][1][f'B{B}|{m}|{p}'])
            bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(a.n)])
            lo, hi = np.percentile(bs, [2.5, 97.5])
            out[f'B{B}']['bootstrap'][f'{a.ref}-{m} {p}'] = [float(d.mean()), float(lo), float(hi)]
            flag = '*' if lo > 0 or hi < 0 else ' '
            print(f"  {a.ref}-{m:10s} {p:24s} {d.mean():+6.2f} [{lo:+.2f}, {hi:+.2f}] {flag}")
if a.out:
    Path(a.out).write_text(json.dumps(out, indent=1))
