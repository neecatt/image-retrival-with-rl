#!/usr/bin/env python3
"""Headroom check for verification + query expansion (Track V-QE), before building the environment.

In Track V a check can only reorder the top-K shortlist. With expansion, every verified match
(inliers >= T) also pulls its own k nearest gallery neighbours into the candidate set, so relevant
images below rank K become reachable. This script measures, from labels, embeddings and the existing
shortlist inliers only (no new verification):
  - how many positives lie below the shortlist, and how many of them are among the k nearest
    neighbours of the shortlist's verified matches ("reachable");
  - AP of the global ranking, of verify-all (matches first, by inliers), and an oracle upper bound for
    expansion: verify-all, then every reachable positive (ordered by similarity), then the rest.
The oracle assumes expansion candidates verify perfectly, so it bounds what any expansion policy
could gain; if it is small, expansion cannot give a sequential policy anything to plan for.

Pools: GLDv2-clean val queries (plain pool and pool + GLDv2 distractors; policy-training data) and
ROxford/RParis Medium/Hard (with and without R1M). Selection-free: nothing here trains or selects.
Output: <out>/qe_headroom.json
"""
import argparse, json, pickle, sys, time
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dataset import LongSideResize, RevisitedDataset, get_transforms     # noqa: E402
from data.dataset_sfm120k import load_gldv2c                                    # noqa: E402
from evaluate import create_model_from_checkpoint                               # noqa: E402
from evaluation.revisited_ap import official_ap                                 # noqa: E402
from evaluation.sfm_val import extract_embeddings                               # noqa: E402
from eval_baselines import embed                                                # noqa: E402

NORM = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def neighbours(G, items, k, chunk=1024):
    """k nearest gallery items of each gallery item in `items` (self excluded): dict item -> LongTensor(k)."""
    out = {}
    items = sorted(set(int(i) for i in items))
    for s in range(0, len(items), chunk):
        it = torch.as_tensor(items[s:s + chunk], device=G.device)
        sim = G[it] @ G.t()
        sim[torch.arange(len(it), device=G.device), it] = float('-inf')
        nb = sim.topk(k, dim=1).indices.cpu()
        for i, row in zip(items[s:s + chunk], nb):
            out[i] = row
    return out


def ap_vec(rel):
    """AP of one ranking (1D float tensor of 0/1, junk already removed), trapezoid rule as official_ap."""
    npos = rel.sum()
    if npos == 0:
        return float('nan')
    r = torch.arange(rel.numel(), device=rel.device, dtype=torch.float32)
    j = rel.cumsum(0) - rel
    p0 = torch.where(r > 0, j / r.clamp(min=1), torch.ones_like(r))
    p1 = (j + 1) / (r + 1)
    return float(((p0 + p1) / 2 * rel).sum() / npos)


def reorder(order, front):
    """Move `front` (list of gallery ids, in that order) to the top of `order`, keep the rest in order."""
    if not front:
        return order
    f = torch.as_tensor(front, device=order.device)
    keep = ~torch.isin(order, f)
    return torch.cat([f, order[keep]])


def summarise(rows):
    keys = rows[0].keys()
    return {k: float(np.nanmean([r[k] for r in rows])) for k in keys}


def gld_pool(tag, G, emb_q, qs, labels, z, n_pool, ks, thr, dev):
    lab = torch.as_tensor(labels, device=dev)
    top, inl = z['top_idx'], z['inliers']
    matched = top[inl >= thr]
    nb = neighbours(G, matched, max(ks))
    res = {k: [] for k in ks}
    for i, q in enumerate(qs):
        sim = emb_q[i] @ G.t()
        sim[q] = float('-inf')                                                 # the query itself (junk)
        order = torch.argsort(sim, descending=True)[:-1]
        relf = lambda o: ((o < n_pool) & (lab[o.clamp(max=n_pool - 1)] == lab[q])).float()
        short = set(int(x) for x in top[i])
        srt = np.argsort(-inl[i], kind='stable')
        m = [int(top[i][j]) for j in srt if inl[i][j] >= thr]                # verified matches, by inliers
        base = ap_vec(relf(order))
        va = reorder(order, m)
        ap_va = ap_vec(relf(va))
        pos = set(torch.nonzero((lab == lab[q])).flatten().tolist()) - {int(q)}
        for k in ks:
            E = set()
            for x in m:
                E.update(int(y) for y in nb[x][:k])
            E -= short; E.discard(int(q))
            reach = sorted(E & pos, key=lambda y: -float(sim[y]))
            ap_qe = ap_vec(relf(reorder(order, m + reach)))
            res[k].append({'npos': len(pos), 'pos_in_shortlist': len(pos & short), 'pos_below_shortlist': len(pos - short),
                           'reachable_pos': len(reach), 'expansion_size': len(E), 'matches': len(m),
                           'ap_global': base, 'ap_verify_all': ap_va, 'ap_oracle_qe': ap_qe})
    return {f'k{k}': summarise(v) for k, v in res.items()}


def bench_pool(name, h, Q, G, z, ks, thr):
    top, inl = z['top_idx'], z['inliers']
    nb = neighbours(G, top[inl >= thr], max(ks))
    sim = (Q @ G.t()).cpu().numpy()
    out = {}
    for prot in ('medium', 'hard'):
        res = {k: [] for k in ks}
        for q, e in enumerate(h.cfg['gnd']):
            easy, hard, junk = set(map(int, e['easy'])), set(map(int, e['hard'])), set(map(int, e['junk']))
            pos, jk = (easy | hard, junk) if prot == 'medium' else (hard, junk | easy)
            if not pos:
                continue
            order = np.argsort(-sim[q], kind='stable')
            short = set(int(x) for x in top[q])
            srt = np.argsort(-inl[q], kind='stable')
            m = [int(top[q][j]) for j in srt if inl[q][j] >= thr]
            front = lambda f: np.concatenate([np.asarray(f, dtype=np.int64), order[~np.isin(order, f)]]) if f else order
            base, va = official_ap(order, pos, jk), official_ap(front(m), pos, jk)
            for k in ks:
                E = set()
                for x in m:
                    E.update(int(y) for y in nb[x][:k])
                E -= short
                reach = sorted(E & pos, key=lambda y: -sim[q][y])
                res[k].append({'npos': len(pos), 'pos_in_shortlist': len(pos & short), 'pos_below_shortlist': len(pos - short),
                               'reachable_pos': len(reach), 'expansion_size': len(E), 'matches': len(m),
                               'ap_global': base, 'ap_verify_all': va, 'ap_oracle_qe': official_ap(front(m + reach), pos, jk)})
        out[prot] = {f'k{k}': summarise(v) for k, v in res.items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--plain_pre', required=True)
    ap.add_argument('--dist_pre', required=True)
    ap.add_argument('--r1m_pre', required=True)
    ap.add_argument('--dist_emb', required=True)
    ap.add_argument('--r1m_emb', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--ks', type=int, nargs='+', default=[5, 10, 20])
    ap.add_argument('--threshold', type=int, default=35)
    ap.add_argument('--gld_queries', type=int, default=3000, help='first N of the 10k queries (speed)')
    ap.add_argument('--workers', type=int, default=16)
    a = ap.parse_args()
    dev = torch.device('cuda')
    t0 = time.time()
    model = create_model_from_checkpoint(a.model, dev).eval()
    out = {'model': a.model, 'threshold': a.threshold, 'ks': a.ks}

    # ---- GLDv2 val pool (+ distractors)
    _, val = load_gldv2c(str(Path(a.data_root) / 'gldv2c'), val_transform=get_transforms(512, 'eval'))
    with torch.no_grad():
        emb, _ = extract_embeddings(model, val, dev, batch_size=64, num_workers=a.workers)
    labels = np.asarray(val.labels)
    zp, zd = np.load(Path(a.plain_pre) / 'gldv2c_val.npz'), np.load(Path(a.dist_pre) / 'gldv2c_val.npz')
    assert np.array_equal(zp['labels'], labels) and np.array_equal(zp['queries'], zd['queries'])
    n = min(a.gld_queries, len(zp['queries']))
    qs = zp['queries'][:n]
    D = torch.from_numpy(np.load(a.dist_emb).astype(np.float32)).to(dev)
    sub = lambda z: {'top_idx': z['top_idx'][:n], 'inliers': z['inliers'][:n]}
    out['gld_val'] = gld_pool('plain', emb, emb[torch.as_tensor(qs, device=dev)], qs, labels, sub(zp), len(labels), a.ks, a.threshold, dev)
    print('gld_val', json.dumps(out['gld_val']), flush=True)
    G = torch.cat([emb, D]); del D
    out['gld_val_dist'] = gld_pool('dist', G, emb[torch.as_tensor(qs, device=dev)], qs, labels, sub(zd), len(labels), a.ks, a.threshold, dev)
    print('gld_val_dist', json.dumps(out['gld_val_dist']), flush=True)
    del G, emb; torch.cuda.empty_cache()

    # ---- ROxford / RParis (+ R1M)
    tf = T.Compose([LongSideResize(1024), T.ToTensor(), NORM])
    R = torch.from_numpy(np.load(a.r1m_emb).astype(np.float32)).to(dev)
    for name in ('roxford5k', 'rparis6k'):
        h = RevisitedDataset(str(Path(a.data_root) / name), name)
        Q = torch.from_numpy(embed(model, h.get_query_dataset(tf), dev, a.workers)).to(dev)
        Gd = torch.from_numpy(embed(model, h.get_database_dataset(tf), dev, a.workers)).to(dev)
        out[name] = bench_pool(name, h, Q, Gd, np.load(Path(a.plain_pre) / f'{name}.npz'), a.ks, a.threshold)
        out[name + '_r1m'] = bench_pool(name, h, Q, torch.cat([Gd, R]), np.load(Path(a.r1m_pre) / f'{name}.npz'), a.ks, a.threshold)
        print(name, json.dumps({k: out[k] for k in (name, name + '_r1m')}), flush=True)
    out['minutes'] = round((time.time() - t0) / 60, 1)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / 'qe_headroom.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
