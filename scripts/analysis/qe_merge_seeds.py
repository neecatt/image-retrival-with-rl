#!/usr/bin/env python3
"""Merge Track V-QE runs of one configuration that were split over budgets and policy seeds
(train_qe_policy.py --budgets B --seeds s, one job each) into one results.json / per_query.npz, in the
format the selection scripts read: per budget and method, the per-seed results are concatenated and
the means (incl. the validation selection score) recomputed; per-query APs are averaged over seeds.
Usage: python scripts/analysis/qe_merge_seeds.py OUT_DIR RUN_DIR [RUN_DIR ...]
"""
import json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

out, runs = Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]]
merged, perq = None, defaultdict(list)
for r in runs:
    if not (r / 'results.json').exists():
        print('missing', r); continue
    res = json.loads((r / 'results.json').read_text())
    z = np.load(r / 'per_query.npz')
    if merged is None:
        merged = {k: v for k, v in res.items() if not k.startswith('B')}
        merged['merged_from'] = []
    merged['merged_from'].append(str(r))
    for bk in [k for k in res if k.startswith('B')]:
        mb = merged.setdefault(bk, {})
        for m, v in res[bk].items():
            if m.endswith('_seeds') or m.endswith('_curves'):
                mb.setdefault(m, []).extend(v)
            elif m not in mb:
                mb[m] = v                                  # heuristics: identical in every run
    for k in z.files:
        perq[k].append(z[k])
sel = merged['selection']
for bk, mb in merged.items():
    if not bk.startswith('B'):
        continue
    for m in [k[:-6] for k in mb if k.endswith('_seeds')]:
        seeds = mb[m + '_seeds']
        mb[m] = {p: float(np.mean([s[p] for s in seeds])) for p in seeds[0]}
        mb[m]['selection'] = float(np.mean([mb[m][p] for p in sel]))
        mb[m]['n_seeds'] = len(seeds)
out.mkdir(parents=True, exist_ok=True)
(out / 'results.json').write_text(json.dumps(merged, indent=1))
np.savez(out / 'per_query.npz', **{k: np.mean(v, 0) for k, v in perq.items()})
print(out, {bk: {m: v.get('n_seeds') for m, v in mb.items() if isinstance(v, dict) and 'n_seeds' in v}
             for bk, mb in merged.items() if bk.startswith('B')})
