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


def per_query_ap_order(orders, gnd):
    """Like per_query_ap, from explicit rankings (one gallery-index array per query)."""
    out = {}
    for prot in ('medium', 'hard'):
        aps = []
        for q, e in enumerate(gnd):
            easy, hard, junk = set(map(int, e['easy'])), set(map(int, e['hard'])), set(map(int, e['junk']))
            pos, jk = (easy | hard, junk) if prot == 'medium' else (hard, junk | easy)
            aps.append(official_ap(orders[q], pos, jk) if pos else float('nan'))
        out[prot] = aps
    return out


def sg_rerank(mda, rmda, G, q, sim, front):
    """SuperGlobal reranking (MDescAug + RerankwMDA) of the ranking given by sim for queries q; the
    verified matches in front[i] (gallery ids, by inliers) are then moved back to the top."""
    ranks = torch.argsort(-sim, dim=1).t()                                   # (N, Q), SuperGlobal's layout
    with torch.no_grad():
        rr = rmda(ranks, *mda(G, q, ranks)).t().cpu().numpy()
    orders = []
    for i in range(rr.shape[0]):
        f = np.asarray(front[i], np.int64)
        orders.append(np.concatenate([f, rr[i][~np.isin(rr[i], f)]]) if len(f) else rr[i])
    return orders


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--desc', required=True)
    ap.add_argument('--model', default=None)
    ap.add_argument('--emb_dir', default=None, help='precomputed <bench>_q.npy / <bench>_db.npy instead of --model')
    ap.add_argument('--qe_pre', required=True)
    ap.add_argument('--r1m_emb', required=True)
    ap.add_argument('--policies', nargs='+', required=True, help='method=path/to/policy.pt (several per method allowed)')
    ap.add_argument('--B', type=int, default=20)
    ap.add_argument('--threshold', type=int, default=35)
    ap.add_argument('--out', required=True)
    ap.add_argument('--sg_repo', default=None, help='also report SuperGlobal reranking on top of each method (<method>+sgr)')
    ap.add_argument('--workers', type=int, default=16)
    a = ap.parse_args()
    dev = torch.device('cuda')
    model = create_model_from_checkpoint(a.model, dev).eval() if a.emb_dir is None else None
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
    if a.sg_repo:
        sys.path.insert(0, a.sg_repo)
        from modules.reranking.MDescAug import MDescAug
        from modules.reranking.RerankwMDA import RerankwMDA
        mda, rmda = MDescAug(), RerankwMDA()
    res = {'desc': a.desc, 'B': a.B, 'policies': a.policies, 'pools': {}}
    for name in ('roxford5k', 'rparis6k'):
        h = RevisitedDataset(str(Path(a.data_root) / name), name)
        gnd = h.get_ground_truth()
        if a.emb_dir:
            Q = F.normalize(torch.from_numpy(np.load(Path(a.emb_dir) / f'{name}_q.npy').astype(np.float32)).to(dev), dim=1)
            Gd = F.normalize(torch.from_numpy(np.load(Path(a.emb_dir) / f'{name}_db.npy').astype(np.float32)).to(dev), dim=1)
        else:
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
            if a.sg_repo:
                out['none+sgr'] = per_query_ap_order(sg_rerank(mda, rmda, G, Q, base, [[] for _ in range(Q.shape[0])]), gnd)
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
                    if a.sg_repo:
                        front = []
                        for i in range(Q.shape[0]):
                            mi = torch.nonzero(matched[i]).flatten()
                            o = torch.argsort(-inl[i, mi], stable=True)
                            seen, f = set(), []
                            for g in cand[i, mi][o].tolist():
                                if g not in seen:
                                    seen.add(g); f.append(g)
                            front.append(f)
                        sgr = per_query_ap_order(sg_rerank(mda, rmda, G, q2, s2, front), gnd)
                        pq['sgr_medium'], pq['sgr_hard'] = sgr['medium'], sgr['hard']
                    runs.append(pq)
                out[m] = {k: (np.nanmean([r[k] for r in runs], 0).tolist() if k in ('medium', 'hard') else float(np.mean([r[k] for r in runs])))
                          for k in runs[0] if not k.startswith('sgr_')}
                if a.sg_repo:
                    out[m + '+sgr'] = {pr: np.nanmean([r[f'sgr_{pr}'] for r in runs], 0).tolist() for pr in ('medium', 'hard')}
            res['pools'][pool] = out
            print(pool, json.dumps({m: (round(100 * np.nanmean(v['medium']), 2), round(100 * np.nanmean(v['hard']), 2),
                                        v.get('n_matched')) for m, v in out.items()}), flush=True)
            del G, base
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / f'{a.desc}.json').write_text(json.dumps(res))


if __name__ == '__main__':
    main()
