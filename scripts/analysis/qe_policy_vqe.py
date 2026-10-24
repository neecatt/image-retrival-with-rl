#!/usr/bin/env python3
"""Does choosing which images to verify help on top of verified query expansion?

For one descriptor: the saved B-check policies (classifier / value / RL, configs already selected on
validation) and the heuristics run greedily in the expansion environment and pick B verifications per
query; every verified match (inliers >= T) then feeds the verified-QE re-query of Q10: matches first by
inliers, everything else re-ranked over the whole gallery with q' = norm(q + sum of matched descriptors)
(alpha = 0, as selected on GLDv2 validation). 'shortlist' = the B most similar shortlist items, i.e.
classic verified QE (sanity check against qe_baselines.py). No new choices are made on any benchmark.
Output: <out>/<desc>.json with mean and per-query AP (official Revisited protocol) per method and pool.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dataset import LongSideResize, RevisitedDataset                     # noqa: E402
from evaluate import create_model_from_checkpoint                               # noqa: E402
from evaluation.revisited_ap import official_ap                                 # noqa: E402
from eval_baselines import embed                                                # noqa: E402
from train_qe_policy import load_pool, sub_env                                  # noqa: E402
from training.qe_env import QEEnv, rollout                                      # noqa: E402
from training.verify_policy import AttnPolicy                                   # noqa: E402

NORM = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def per_query_ap(sim, gnd):
    out = {}
    for prot in ('medium', 'hard'):
        aps = []
        for q, e in enumerate(gnd):
            easy, hard, junk = set(map(int, e['easy'])), set(map(int, e['hard'])), set(map(int, e['junk']))
            pos, jk = (easy | hard, junk) if prot == 'medium' else (hard, junk | easy)
            aps.append(official_ap(np.argsort(-sim[q], kind='stable'), pos, jk) if pos else float('nan'))
        out[prot] = aps
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--desc', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--qe_pre', required=True)
    ap.add_argument('--r1m_emb', required=True)
    ap.add_argument('--policies', nargs='+', required=True, help='method=path/to/policy.pt (several per method allowed)')
    ap.add_argument('--B', type=int, default=20)
    ap.add_argument('--threshold', type=int, default=35)
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=16)
    a = ap.parse_args()
    dev = torch.device('cuda')
    model = create_model_from_checkpoint(a.model, dev).eval()
    tf = T.Compose([LongSideResize(1024), T.ToTensor(), NORM])
    R = F.normalize(torch.from_numpy(np.load(a.r1m_emb).astype(np.float32)).to(dev), dim=1)
    pols = {}
    for spec in a.policies:
        m, p = spec.split('=', 1)
        net = AttnPolicy(n_feat=QEEnv.n_feat).to(dev)
        net.load_state_dict(torch.load(p, map_location=dev)); net.eval()
        pols.setdefault(m, []).append(net)
    sim_sc = lambda f: f[..., 0] * 100
    short_sc = lambda f: torch.where(f[..., 11] > 0, torch.full_like(f[..., 0], -1e4), f[..., 0] * 100)
    res = {'desc': a.desc, 'B': a.B, 'policies': a.policies, 'pools': {}}
    for name in ('roxford5k', 'rparis6k'):
        h = RevisitedDataset(str(Path(a.data_root) / name), name)
        gnd = h.get_ground_truth()
        Q = F.normalize(torch.from_numpy(embed(model, h.get_query_dataset(tf), dev, a.workers)).to(dev), dim=1)
        Gd = F.normalize(torch.from_numpy(embed(model, h.get_database_dataset(tf), dev, a.workers)).to(dev), dim=1)
        for suffix in ('', '_r1m'):
            pool = name + suffix
            G = torch.cat([Gd, R]) if suffix else Gd
            z = np.load(Path(a.qe_pre) / f'{pool}.npz')
            cand, inl = torch.as_tensor(z['cand'], device=dev), torch.as_tensor(z['inliers'], device=dev).float()
            d, _ = load_pool(Path(a.qe_pre) / f'{pool}.npz', dev, 'medium')          # all 70 queries have positives
            assert d['valid'].shape[0] == Q.shape[0]
            env = QEEnv(d, a.B, a.threshold)
            idx = torch.arange(env.Q, device=dev)
            schedulers = {'shortlist': [short_sc], 'top': [sim_sc]}
            schedulers.update(pols)
            base = Q @ G.t()
            out = {'none': per_query_ap(base.cpu().numpy(), gnd)}
            for m, scs in schedulers.items():
                runs = []
                for sc in scs:
                    with torch.no_grad():
                        se = sub_env(env, idx)
                        _, _, ver = rollout(se, torch.arange(se.Q, device=dev), sc, 'greedy')
                    C = ver.shape[1]
                    matched = ver & (inl[:, :C] >= a.threshold) & (cand[:, :C] >= 0)
                    gid = cand[:, :C].clamp(min=0)
                    w = matched.float()
                    q2 = F.normalize(Q + (w[..., None] * G[gid]).sum(1), dim=1)          # alpha = 0
                    s2 = q2 @ G.t()
                    bonus = torch.where(matched, 1e3 + inl[:, :C], torch.full_like(w, float('-inf')))
                    s2.scatter_reduce_(1, gid, bonus, reduce='amax')
                    pq = per_query_ap(s2.cpu().numpy(), gnd)
                    pq['n_verified'] = float(ver.float().sum(1).mean()); pq['n_matched'] = float(matched.float().sum(1).mean())
                    runs.append(pq)
                out[m] = {k: (np.nanmean([r[k] for r in runs], 0).tolist() if k in ('medium', 'hard') else float(np.mean([r[k] for r in runs])))
                          for k in runs[0]}
            res['pools'][pool] = out
            print(pool, json.dumps({m: (round(100 * np.nanmean(v['medium']), 2), round(100 * np.nanmean(v['hard']), 2),
                                        v.get('n_matched')) for m, v in out.items()}), flush=True)
            del G, base
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / f'{a.desc}.json').write_text(json.dumps(res))


if __name__ == '__main__':
    main()
