#!/usr/bin/env python3
"""Track V-QE precompute: verification with query expansion.

Extends each query's top-K shortlist (from the Track V precomputes) with expansion candidates: every
verified match (inliers >= T) unlocks its k nearest gallery neighbours; matched expansion candidates
unlock theirs in turn (up to --hops), at most --cap new candidates per query, added breadth-first
(parents by inliers, then neighbour order). Only matches can unlock, so only neighbours of matches are
ever needed; every new (query, candidate) pair is verified here, offline.

Per pool and query it stores the candidate list (shortlist first), the unlock edges, inliers, the
query-candidate and candidate-candidate similarities, and junk-free global ranks of every candidate
and of every positive. With these the environment computes the exact AP of "verified matches first
(by inliers), everything else in global order" for any set of verified candidates, without the full
similarity matrix.

Pools: gldv2c_val (val pool), gldv2c_val_dist (val pool + GLDv2 distractors) for policy training;
roxford5k / rparis6k (database) and *_r1m (database + R1M) for evaluation (Medium and Hard ranks).
Output: <out>/<pool>.npz and precompute_qe_log.json.
"""
import argparse, json, os, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data.dataset import LongSideResize, RevisitedDataset, get_transforms     # noqa: E402
from data.dataset_sfm120k import _gldv2_id_to_path, load_gldv2c                # noqa: E402
from evaluate import create_model_from_checkpoint                               # noqa: E402
from evaluation.sfm_val import extract_embeddings                               # noqa: E402
from eval_baselines import embed                                                # noqa: E402
from training.verifier import Verifier                                          # noqa: E402
from v_precompute_r1m import extract_needed                                     # noqa: E402

NORM = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def knn(G, items, k, chunk=1024):
    """k nearest gallery neighbours (self excluded) of each gallery id in items -> dict id -> list."""
    out, items = {}, sorted(set(int(i) for i in items))
    for s in range(0, len(items), chunk):
        it = torch.as_tensor(items[s:s + chunk], device=G.device)
        sim = G[it] @ G.t()
        sim[torch.arange(len(it), device=G.device), it] = float('-inf')
        for i, row in zip(items[s:s + chunk], sim.topk(k, dim=1).indices.cpu().tolist()):
            out[i] = row
    return out


def expand(G, top, inl, self_ids, verify, k, hops, cap, thr):
    """Breadth-first expansion. top/inl (Q, K); self_ids: gallery id of each query or None.
    verify(list of (query row, gallery id)) -> inliers. Returns per-query lists cand, inl, hop, edges."""
    Q, K = top.shape
    cand = [list(map(int, top[q])) for q in range(Q)]
    cinl = [list(map(int, inl[q])) for q in range(Q)]
    hop = [[0] * K for _ in range(Q)]
    edges = [[] for _ in range(Q)]                     # (parent position, child position)
    pos = [{g: i for i, g in enumerate(cand[q])} for q in range(Q)]
    frontier = [[i for i in np.argsort(-inl[q], kind='stable') if inl[q][i] >= thr] for q in range(Q)]
    for h in range(1, hops + 1):
        nb = knn(G, [cand[q][p] for q in range(Q) for p in frontier[q]], k)
        new = []
        for q in range(Q):
            for p in frontier[q]:
                for y in nb[cand[q][p]]:
                    if self_ids is not None and y == self_ids[q]:
                        continue
                    if y in pos[q]:
                        c = pos[q][y]
                        if c >= K and c != p:
                            edges[q].append((p, c))
                    elif len(cand[q]) < K + cap:
                        pos[q][y] = len(cand[q]); cand[q].append(y); cinl[q].append(-1); hop[q].append(h)
                        edges[q].append((p, pos[q][y])); new.append((q, pos[q][y]))
        print(f"  hop {h}: {len(new)} new pairs to verify", flush=True)
        res = verify([(q, cand[q][c]) for q, c in new]) if new else []
        for (q, c), v in zip(new, res):
            cinl[q][c] = int(v)
        frontier = [[] for _ in range(Q)]
        for q, c in new:
            if cinl[q][c] >= thr:
                frontier[q].append(c)
        for q in range(Q):
            frontier[q].sort(key=lambda c: -cinl[q][c])
    return cand, cinl, hop, edges


def ranks_for(sim_row, junk_mask, cand, pos_ids):
    """Junk-free global ranks (0-based) of candidates (-1 if junk) and sorted ranks of positives."""
    order = torch.argsort(sim_row, descending=True)
    nonjunk = ~junk_mask[order]
    r = torch.full_like(order, -1)
    r[order] = torch.where(nonjunk, nonjunk.long().cumsum(0) - 1, torch.full_like(order, -1))
    c_r = r[torch.as_tensor(cand, device=r.device)].cpu().numpy()
    p_r = np.sort(r[torch.as_tensor(sorted(pos_ids), device=r.device)].cpu().numpy()) if pos_ids else np.zeros(0, np.int64)
    return c_r, p_r


def pack(Q, C, cand, cinl, hop, edges, Gq, G, extra):
    out = {'cand': np.full((Q, C), -1, np.int64), 'inliers': np.full((Q, C), -1, np.int32),
           'hop': np.full((Q, C), -1, np.int8), 'unlock': np.zeros((Q, C, C), bool),
           'sim': np.zeros((Q, C), np.float32), 'cand_sim': np.zeros((Q, C, C), np.float16)}
    for q in range(Q):
        n = len(cand[q])
        out['cand'][q, :n] = cand[q]; out['inliers'][q, :n] = cinl[q]; out['hop'][q, :n] = hop[q]
        for p, c in edges[q]:
            out['unlock'][q, p, c] = True
        g = G[torch.as_tensor(cand[q], device=G.device)]
        out['sim'][q, :n] = (g @ Gq[q]).cpu().numpy()
        out['cand_sim'][q, :n, :n] = (g @ g.t()).half().cpu().numpy()
    out.update(extra)
    return out


def pad(rows, fill, width=None):
    P = width or max([len(r) for r in rows] + [1])
    a = np.full((len(rows), P), fill, np.int64)
    for i, r in enumerate(rows):
        a[i, :len(r)] = r
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', required=True, help='staged gldv2c, gldv2d, roxford5k, rparis6k')
    ap.add_argument('--model', default=None)
    ap.add_argument('--emb_dir', default=None, help='precomputed embeddings instead of --model (gld_pool.npy, '
                    '<bench>_q.npy, <bench>_db.npy; distractors via --dist_emb, R1M via --r1m_emb)')
    ap.add_argument('--plain_pre', required=True)
    ap.add_argument('--dist_pre', required=True)
    ap.add_argument('--r1m_pre', required=True)
    ap.add_argument('--dist_emb', required=True)
    ap.add_argument('--r1m_emb', required=True)
    ap.add_argument('--r1m_dir', default='/home/atuin/b268dc/b268dc21/datasets/revisitop1m')
    ap.add_argument('--out', required=True)
    ap.add_argument('--k', type=int, default=10)
    ap.add_argument('--hops', type=int, default=2)
    ap.add_argument('--cap', type=int, default=150)
    ap.add_argument('--threshold', type=int, default=35)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--max_queries', type=int, default=0, help='smoke test: first N GLDv2 queries only')
    ap.add_argument('--pools', nargs='+', default=['gldv2c_val', 'gldv2c_val_dist', 'roxford5k', 'rparis6k',
                                                   'roxford5k_r1m', 'rparis6k_r1m'])
    a = ap.parse_args()
    dev = torch.device('cuda')
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    root = Path(a.data_root)
    t0 = time.time()
    model = create_model_from_checkpoint(a.model, dev).eval() if a.emb_dir is None else None
    E = Path(a.emb_dir) if a.emb_dir else None
    log = {'model': a.model, 'emb_dir': a.emb_dir, 'k': a.k, 'hops': a.hops, 'cap': a.cap, 'threshold': a.threshold}
    tmp = Path(os.environ['TMPDIR'])

    # ---- GLDv2 val pool (+ distractors)
    if any(p.startswith('gldv2c') for p in a.pools):
        _, val = load_gldv2c(str(root / 'gldv2c'), val_transform=get_transforms(512, 'eval'))
        if E is not None:
            emb = torch.from_numpy(np.load(E / 'gld_pool.npy').astype(np.float32)).to(dev)
        else:
            with torch.no_grad():
                emb, _ = extract_embeddings(model, val, dev, batch_size=64, num_workers=a.workers)
        labels = np.asarray(val.labels); n_pool = len(labels)
        assert emb.shape[0] == n_pool
        pool_paths = [str(Path(val.image_root) / p) for p in val.image_paths]
        import pickle
        d_ids = pickle.load(open(root / 'gldv2d' / 'gldv2d.pkl', 'rb'))['cids']
        d_paths = [str(root / 'gldv2d' / 'jpg' / _gldv2_id_to_path(i)) for i in d_ids]
        lab_t = torch.as_tensor(labels, device=dev)
        for name, pre in (('gldv2c_val', a.plain_pre), ('gldv2c_val_dist', a.dist_pre)):
            if name not in a.pools:
                continue
            t1 = time.time()
            z = dict(np.load(Path(pre) / 'gldv2c_val.npz'))
            if a.max_queries:
                for key in ('queries', 'top_idx', 'inliers'):
                    z[key] = z[key][:a.max_queries]
            qs = z['queries']
            assert np.array_equal(z['labels'], labels)
            if name.endswith('dist'):
                G = torch.cat([emb, torch.from_numpy(np.load(a.dist_emb).astype(np.float32)).to(dev)])
                paths = pool_paths + d_paths
            else:
                G, paths = emb, pool_paths
            v = Verifier('/', paths, dev, max_kpts=1024, resize=512, batch=32, cpu_workers=16, io_threads=8)
            cand, cinl, hop, edges = expand(G, z['top_idx'], z['inliers'], [int(q) for q in qs],
                                            lambda pairs: v.verify([(int(qs[q]), g) for q, g in pairs]),
                                            a.k, a.hops, a.cap, a.threshold)
            v.close()
            Gq = G[torch.as_tensor(qs, device=dev)]
            c_rank, p_rank, c_rel = [], [], []
            junk = torch.zeros(G.shape[0], dtype=torch.bool, device=dev)
            for i, q in enumerate(qs):
                junk[q] = True                                                # the query itself
                pos_ids = (torch.nonzero(lab_t == lab_t[q]).flatten()).tolist()
                pos_ids = [p for p in pos_ids if p != q]
                cr, pr = ranks_for(Gq[i] @ G.t(), junk, cand[i], pos_ids)
                junk[q] = False
                c_rank.append(cr); p_rank.append(pr)
                c = np.asarray(cand[i])
                c_rel.append(np.where(c == q, -1, ((c < n_pool) & (labels[np.minimum(c, n_pool - 1)] == labels[q])).astype(np.int8)))
            C = 100 + a.cap
            o = pack(len(qs), C, cand, cinl, hop, edges, Gq, G,
                     {'queries': qs, 'labels': labels, 'n_short': z['top_idx'].shape[1],
                      'rank': pad(c_rank, -1, C), 'rel': pad(c_rel, -1, C).astype(np.int8),
                      'pos_ranks': pad(p_rank, -1), 'npos': np.asarray([len(p) for p in p_rank])})
            np.savez_compressed(out / f'{name}.npz', **o)
            nexp = np.asarray([len(c) - 100 for c in cand])
            log[name] = {'queries': len(qs), 'expanded_mean': float(nexp.mean()), 'expanded_at_cap': float((nexp >= a.cap).mean()),
                         'expanded_match_ge35': float(np.mean([x >= a.threshold for c in cinl for x in c[100:]] or [0])),
                         'expanded_relevant': float(np.mean([r == 1 for rr in c_rel for r in rr[100:]] or [0])),
                         'minutes': round((time.time() - t1) / 60, 1)}
            print(name, json.dumps(log[name]), flush=True)
            del G; torch.cuda.empty_cache()
        del emb

    # ---- ROxford / RParis (+ R1M)
    tf = T.Compose([LongSideResize(1024), T.ToTensor(), NORM])
    names = [l.strip() for l in open(Path(a.r1m_dir) / 'revisitop1m.txt') if l.strip()]
    R = torch.from_numpy(np.load(a.r1m_emb).astype(np.float32)).to(dev) if any(p.endswith('_r1m') for p in a.pools) else None
    bench = {}
    for name in ('roxford5k', 'rparis6k'):
        if name not in a.pools and name + '_r1m' not in a.pools:
            continue
        h = RevisitedDataset(str(root / name), name)
        if E is not None:
            Qe = torch.from_numpy(np.load(E / f'{name}_q.npy').astype(np.float32)).to(dev)
            Gd = torch.from_numpy(np.load(E / f'{name}_db.npy').astype(np.float32)).to(dev)
        else:
            Qe = torch.from_numpy(embed(model, h.get_query_dataset(tf), dev, a.workers)).to(dev)
            Gd = torch.from_numpy(embed(model, h.get_database_dataset(tf), dev, a.workers)).to(dev)
        crops = []
        (tmp / 'qe_crops' / name).mkdir(parents=True, exist_ok=True)
        for i, (img_name, e) in enumerate(zip(h.cfg['query_images'], h.cfg['gnd'])):
            im = Image.open(h.root_dir / 'jpg' / img_name).convert('RGB')
            if e.get('bbx') is not None:
                im = im.crop(tuple(int(round(x)) for x in e['bbx']))
            p = tmp / 'qe_crops' / name / f'q{i:03d}.jpg'; im.save(p, quality=95); crops.append(str(p))
        bench[name] = (h, Qe, Gd, crops, [str(h.root_dir / 'jpg' / x) for x in h.cfg['db_images']])

    # R1M images any hop could need: neighbours of shortlist matches (hop 1) and of every hop-1
    # candidate (superset of hop 2), extracted from the archives in one pass
    if R is not None:
        want = set()
        for name, (h, Qe, Gd, crops, db_paths) in bench.items():
            if name + '_r1m' not in a.pools:
                continue
            z = np.load(Path(a.r1m_pre) / f'{name}.npz')
            G = torch.cat([Gd, R]); n_db = Gd.shape[0]
            s1 = knn(G, z['top_idx'][z['inliers'] >= a.threshold], a.k)
            # all neighbours of matches, not minus the shortlists: an item in one query's shortlist can
            # be another query's expansion candidate, whose neighbours hop 2 may need
            hop1 = {y for v in s1.values() for y in v}
            s2 = knn(G, hop1, a.k) if a.hops > 1 else {}
            ids = set(z['top_idx'].ravel().tolist()) | hop1 | {y for v in s2.values() for y in v}
            want |= {names[g - n_db] for g in ids if g >= n_db}
            del G
        t1 = time.time()
        archives = [str(Path(a.r1m_dir) / 'raw' / f'revisitop1m.{i}.tar.gz') for i in range(1, 101)]
        with Pool(24) as p:
            got = sum(p.map(extract_needed, [(f, want, str(tmp / 'r1m_qe')) for f in archives]))
        if got != len(want):
            raise RuntimeError(f"extracted {got} of {len(want)} R1M images")
        log['r1m_extracted'] = len(want); log['r1m_extract_minutes'] = round((time.time() - t1) / 60, 1)
        print(f"extracted {len(want)} R1M images in {log['r1m_extract_minutes']} min", flush=True)

    for name, (h, Qe, Gd, crops, db_paths) in bench.items():
        n_db = Gd.shape[0]
        for suffix, pre in (('', a.plain_pre), ('_r1m', a.r1m_pre)):
            pool = name + suffix
            if pool not in a.pools:
                continue
            t1 = time.time()
            z = np.load(Path(pre) / f'{name}.npz')
            G = torch.cat([Gd, R]) if suffix else Gd

            def verify(pairs):
                gal = sorted({g for _, g in pairs})
                paths = crops + [db_paths[g] if g < n_db else str(tmp / 'r1m_qe' / names[g - n_db]) for g in gal]
                gpos = {g: len(crops) + i for i, g in enumerate(gal)}
                missing = [x for x in paths if not Path(x).exists()]
                if missing:
                    raise RuntimeError(f"{len(missing)} images missing before verification, e.g. {missing[:3]}")
                vv = Verifier('/', paths, dev, max_kpts=1024, resize=512, batch=32, cpu_workers=16, io_threads=8)
                res = vv.verify([(q, gpos[g]) for q, g in pairs])
                vv.close()
                return res
            cand, cinl, hop, edges = expand(G, z['top_idx'], z['inliers'], None, verify, a.k, a.hops, a.cap, a.threshold)
            extra = {'n_short': z['top_idx'].shape[1], 'n_db': n_db}
            C = 100 + a.cap
            for prot in ('medium', 'hard'):
                c_rank, p_rank, c_rel = [], [], []
                for q, e in enumerate(h.cfg['gnd']):
                    easy, hard, jk = set(map(int, e['easy'])), set(map(int, e['hard'])), set(map(int, e['junk']))
                    pos_ids, junk_ids = (easy | hard, jk) if prot == 'medium' else (hard, jk | easy)
                    junk = torch.zeros(G.shape[0], dtype=torch.bool, device=dev)
                    if junk_ids:
                        junk[torch.as_tensor(sorted(junk_ids), device=dev)] = True
                    cr, pr = ranks_for(Qe[q] @ G.t(), junk, cand[q], pos_ids)
                    c_rank.append(cr); p_rank.append(pr)
                    c_rel.append(np.asarray([-1 if g in junk_ids else int(g in pos_ids) for g in cand[q]], np.int8))
                extra[f'rank_{prot}'] = pad(c_rank, -1, C)
                extra[f'rel_{prot}'] = pad(c_rel, -1, C).astype(np.int8)
                extra[f'pos_ranks_{prot}'] = pad(p_rank, -1)
                extra[f'npos_{prot}'] = np.asarray([len(p) for p in p_rank])
            o = pack(len(cand), C, cand, cinl, hop, edges, Qe, G, extra)
            np.savez_compressed(out / f'{pool}.npz', **o)
            nexp = np.asarray([len(c) - 100 for c in cand])
            log[pool] = {'queries': len(cand), 'expanded_mean': float(nexp.mean()), 'expanded_at_cap': float((nexp >= a.cap).mean()),
                         'expanded_match_ge35': float(np.mean([x >= a.threshold for c in cinl for x in c[100:]] or [0])),
                         'expanded_relevant_medium': float(np.mean([r == 1 for q in range(len(cand)) for r in extra['rel_medium'][q][100:len(cand[q])]] or [0])),
                         'minutes': round((time.time() - t1) / 60, 1)}
            print(pool, json.dumps(log[pool]), flush=True)
    log['minutes'] = round((time.time() - t0) / 60, 1)
    lf = out / 'precompute_qe_log.json'                         # merge with earlier runs into the same dir
    prev = json.loads(lf.read_text()) if lf.exists() else {}
    prev.setdefault('runs', []).append({k: v for k, v in log.items() if k not in a.pools})
    prev.update({k: v for k, v in log.items() if k in a.pools})
    lf.write_text(json.dumps(prev, indent=2))


if __name__ == '__main__':
    main()
