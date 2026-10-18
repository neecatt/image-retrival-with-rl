#!/usr/bin/env python3
"""Track V diagnosis: which shortlist items do the saved policies verify on the benchmarks?

Re-runs the saved policies (policy_B{B}_{method}_seed{s}.pt from train_verify_policy.py) greedily on the
benchmark shortlists of one or more precompute dirs (e.g. without and with R1M) and reports, per method:
benchmark AP, mean rank of verified items, verification hit rate (inliers >= T), fraction of shortlist
positives verified, and the fraction of verified items that are R1M distractors.
Usage: python scripts/analysis/trackv_selection_stats.py --policies results/trackv/<run> \
         --bench name=<precompute dir> [name=<dir> ...] --bench_root data/archives
"""
import argparse, json, pickle, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.verify_policy import AttnPolicy, VerifyEnv, rollout     # noqa: E402
from train_verify_policy import build, subset                          # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--policies', required=True)
ap.add_argument('--bench', nargs='+', required=True, help='name=precompute_dir')
ap.add_argument('--bench_root', default='data/archives')
ap.add_argument('--budgets', type=int, nargs='+', default=[10, 20])
ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
ap.add_argument('--threshold', type=int, default=35)
a = ap.parse_args()
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
out = {}
for spec in a.bench:
    tag, d = spec.split('=', 1)
    for name in ('roxford5k', 'rparis6k'):
        zb = np.load(Path(d) / f'{name}.npz')
        gnd = pickle.load(open(Path(a.bench_root) / name / f'{name}.pkl', 'rb'))['gnd']
        n_db = int(zb['n_db']) if 'n_db' in zb.files else zb['sim_full'].shape[1]
        nq, ndb = zb['sim_full'].shape
        rf = np.zeros((nq, ndb), dtype=np.int8)
        for q, e in enumerate(gnd):
            rf[q, list(e['easy']) + list(e['hard'])] = 1; rf[q, list(e['junk'])] = -1
        pool = build(zb['sim_full'], zb['top_idx'], zb['inliers'], zb['cand_sim'], rf, zb['top_idx'].shape[1], dev)
        is_r1m = torch.tensor(zb['top_idx'] >= n_db, device=dev)
        for B in a.budgets:
            env = VerifyEnv(pool, B, a.threshold)
            idx = torch.arange(env.Q, device=dev)
            scorers = {'top': [lambda f: f[..., 0] * 100]}
            for m in ('classifier', 'value', 'rl'):
                scorers[m] = []
                for s in a.seeds:
                    p = AttnPolicy().to(dev)
                    p.load_state_dict(torch.load(Path(a.policies) / f'policy_B{B}_{m}_seed{s}.pt', map_location=dev))
                    scorers[m].append(p.eval())
            for m, lst in scorers.items():
                st = []
                for sc in lst:
                    with torch.no_grad():
                        r, _, ver = rollout(env, idx, sc, 'greedy')
                    vf = ver.float()
                    rank = torch.arange(env.K, device=dev).float()
                    pos = (pool['rel'] == 1)
                    st.append({'AP': 100 * r.mean().item(),
                               'mean_rank': ((vf * rank).sum(1) / B).mean().item(),
                               'frac_beyond_top20': ((vf * (rank >= 20)).sum(1) / B).mean().item(),
                               'hit_rate': ((ver & env.match).sum(1).float() / B).mean().item(),
                               'pos_verified': ((ver & pos).sum(1).float() / pos.sum(1).clamp(min=1)).mean().item(),
                               'r1m_verified': ((ver & is_r1m).sum(1).float() / B).mean().item()})
                out[f'{tag} {name} B{B} {m}'] = {k: round(float(np.mean([x[k] for x in st])), 3) for k in st[0]}
        out[f'{tag} {name} shortlist'] = {'r1m_frac': round(float(is_r1m.float().mean()), 3),
                                          'match_frac': round(float(env.match.float().mean()), 3),
                                          'top1_sim': round(float(pool['sim'][:, 0].mean()), 3),
                                          'sim_at_100': round(float(pool['sim'][:, -1].mean()), 3)}
for k, v in out.items():
    print(f"{k:42s} {json.dumps(v)}")
(Path(a.policies) / 'selection_stats.json').write_text(json.dumps(out, indent=1))
