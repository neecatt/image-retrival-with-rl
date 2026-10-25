#!/usr/bin/env python3
"""Track V / V-QE shortlists for an external descriptor given as precomputed embeddings (e.g. SuperGlobal).

Writes the three formats the pipeline reads, for the same 10,000 GLDv2 val queries as --queries_from:
  <out>/q10k/gldv2c_val.npz      shortlists over the val pool (v_precompute.py format) + roxford5k / rparis6k.npz
  <out>/dist/gldv2c_val.npz      shortlists over val pool + GLDv2 distractors (compact v_precompute_dist.py format)
  <out>/r1m/{roxford5k,rparis6k}.npz   benchmark shortlists over database + R1M (v_precompute_r1m.py format)
Inliers depend only on the image pair and every precompute indexes images the same way (val pool, then
distractors; database, then R1M), so pairs already verified for other descriptors (--cache_from dirs)
are reused and only new pairs are verified (same verifier settings, query crops with round() as before).
Embeddings in --emb_dir: gld_pool.npy, gldv2d.npy, <bench>_q.npy, <bench>_db.npy, r1m.npy (L2-normalised).
"""
import argparse, json, os, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data.dataset import RevisitedDataset                                       # noqa: E402
from data.dataset_sfm120k import _gldv2_id_to_path, load_gldv2c                # noqa: E402
from training.verifier import Verifier                                          # noqa: E402
from training.verify_policy import query_tables                                 # noqa: E402
from v_precompute_dist import tail_tables                                       # noqa: E402
from v_precompute_r1m import extract_needed                                     # noqa: E402

BIG = 10_000_000          # key = query * BIG + gallery id (gallery ids < 2.1M)


class PairCache:
    def __init__(self):
        self.k, self.v = [], []

    def add(self, q, g, inl):
        q, g, inl = np.asarray(q).ravel(), np.asarray(g).ravel(), np.asarray(inl).ravel()
        ok = (g >= 0) & (inl >= 0)
        self.k.append(q[ok].astype(np.int64) * BIG + g[ok]); self.v.append(inl[ok].astype(np.int32))

    def freeze(self):
        k, v = np.concatenate(self.k), np.concatenate(self.v)
        k, first = np.unique(k, return_index=True)
        self.keys, self.vals = k, v[first]
        print(f"  pair cache: {len(k)} pairs", flush=True)

    def get(self, q, g):
        key = np.asarray(q, np.int64) * BIG + np.asarray(g, np.int64)
        pos = np.clip(np.searchsorted(self.keys, key), 0, len(self.keys) - 1)
        hit = self.keys[pos] == key
        return np.where(hit, self.vals[pos], -1)


def harvest(dirs, n_q):
    """Pair caches for GLDv2 (query = val-pool id) and per benchmark (query = query index)."""
    gld, bench = PairCache(), {'roxford5k': PairCache(), 'rparis6k': PairCache()}
    for d in dirs:
        d = Path(d)
        for f in ('gldv2c_val.npz', 'gldv2c_val_dist.npz'):
            if (d / f).exists():
                z = np.load(d / f)
                g = z['cand'] if 'cand' in z.files else z['top_idx']
                gld.add(np.repeat(z['queries'][:, None], g.shape[1], 1), g, z['inliers'])
        for name in bench:
            for f in (f'{name}.npz', f'{name}_r1m.npz'):
                if (d / f).exists():
                    z = np.load(d / f)
                    g = z['cand'] if 'cand' in z.files else z['top_idx']
                    bench[name].add(np.repeat(np.arange(g.shape[0])[:, None], g.shape[1], 1), g, z['inliers'])
    gld.freeze()
    for c in bench.values():
        c.freeze()
    return gld, bench


def fill(cache, qs, top, verify):
    """Inliers for (qs[i], top[i, k]); cached pairs reused, the rest verified."""
    Q, K = top.shape
    inl = cache.get(np.repeat(qs[:, None], K, 1), top).reshape(Q, K)
    miss = np.argwhere(inl < 0)
    print(f"  {len(miss)} of {Q * K} pairs to verify ({100 * (1 - len(miss) / (Q * K)):.1f}% reused)", flush=True)
    if len(miss):
        res = verify([(int(qs[i]), int(top[i, k])) for i, k in miss])
        inl[miss[:, 0], miss[:, 1]] = res
    return inl, len(miss)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--emb_dir', required=True)
    ap.add_argument('--data_root', required=True, help='staged gldv2c, gldv2d, roxford5k, rparis6k')
    ap.add_argument('--queries_from', required=True, help='an existing q10k gldv2c_val.npz (same 10k queries)')
    ap.add_argument('--cache_from', nargs='+', required=True)
    ap.add_argument('--r1m_dir', default='/home/atuin/b268dc/b268dc21/datasets/revisitop1m')
    ap.add_argument('--out', required=True)
    ap.add_argument('--K', type=int, default=100)
    a = ap.parse_args()
    dev = torch.device('cuda')
    E, out, root = Path(a.emb_dir), Path(a.out), Path(a.data_root)
    for sub in ('q10k', 'dist', 'r1m'):
        (out / sub).mkdir(parents=True, exist_ok=True)
    tmp = Path(os.environ['TMPDIR'])
    log = {'emb_dir': str(E), 'K': a.K}
    gcache, bcache = harvest(a.cache_from, None)
    K = a.K

    # ---------------- GLDv2 val pool (+ distractors)
    _, val = load_gldv2c(str(root / 'gldv2c'))
    labels = np.asarray(val.labels); n_pool = len(labels)
    zq = np.load(a.queries_from)
    assert np.array_equal(zq['labels'], labels)
    qs = zq['queries']
    P = torch.from_numpy(np.load(E / 'gld_pool.npy').astype(np.float32)).to(dev)
    D = torch.from_numpy(np.load(E / 'gldv2d.npy').astype(np.float32)).to(dev)
    assert P.shape[0] == n_pool
    import pickle
    d_ids = pickle.load(open(root / 'gldv2d' / 'gldv2d.pkl', 'rb'))['cids']
    paths = [str(Path(val.image_root) / p) for p in val.image_paths] + [str(root / 'gldv2d' / 'jpg' / _gldv2_id_to_path(i)) for i in d_ids]
    v = Verifier('/', paths, dev, max_kpts=1024, resize=512, batch=32, cpu_workers=16, io_threads=8)
    q_t = torch.as_tensor(qs, device=dev)
    lab_t = torch.as_tensor(labels, device=dev)

    # plain pool (v_precompute.py format); parts already written by an earlier run are skipped
    t0 = time.time()
    do_q10k = not (out / 'q10k' / 'gldv2c_val.npz').exists()
    do_dist = not (out / 'dist' / 'gldv2c_val.npz').exists()
    if do_q10k:
        sim = P[q_t] @ P.t(); sim[torch.arange(len(qs), device=dev), q_t] = float('-inf')
        _, ti = sim.topk(K, dim=1)
        top = ti.cpu().numpy()
        cand = torch.einsum('qkd,qld->qkl', P[ti], P[ti]).half().cpu().numpy()
        inl, nv = fill(gcache, qs, top, v.verify)
        np.savez_compressed(out / 'q10k' / 'gldv2c_val.npz', queries=qs, labels=labels, top_idx=top, inliers=inl,
                            sim_full=sim.half().cpu().numpy(), cand_sim=cand)
        log['q10k_gld'] = {'verified': nv, 'minutes': round((time.time() - t0) / 60, 1), 'match_ge35': float((inl >= 35).mean())}
        print('q10k gld', log['q10k_gld'], flush=True)
        del sim

    # pool + distractors (compact format, as v_precompute_dist.py)
    t0 = time.time()
    G = torch.cat([P, D])
    if not do_dist:
        print('dist gld: exists, skipped', flush=True)
    Q = len(qs)
    top = np.zeros((Q, K), np.int64); sim_top = np.zeros((Q, K), np.float32); rel_top = np.zeros((Q, K), np.int8)
    ap_tail = np.zeros(Q, np.float32); npos = np.zeros(Q, np.int64); cand = np.zeros((Q, K, K), np.float16)
    n_bad = 0
    for s in (range(0, Q, 128) if do_dist else []):
        qb = q_t[s:s + 128]
        sm = P[qb] @ G.t(); sm[torch.arange(len(qb), device=dev), qb] = float('-inf')
        ts, ti = sm.topk(K, dim=1)
        cand[s:s + len(qb)] = torch.einsum('qkd,qld->qkl', G[ti], G[ti]).half().cpu().numpy()
        top[s:s + len(qb)] = ti.cpu().numpy(); sim_top[s:s + len(qb)] = ts.cpu().numpy()
        for b in range(len(qb)):
            q, i = int(qb[b]), s + b
            pos = torch.nonzero(lab_t == lab_t[q]).flatten(); pos = pos[pos != q]
            ranks = (sm[b][None, :] > sm[b][pos][:, None]).sum(1).cpu().numpy()
            npos[i] = len(pos)
            rel_top[i] = (top[i] < n_pool) & (labels[np.minimum(top[i], n_pool - 1)] == labels[q])
            ap_tail[i] = tail_tables(ranks, len(pos), K)
            if i < 20:                                                     # exact check on the full ranking
                order = torch.argsort(sm[b], descending=True).cpu().numpy()
                rel = np.zeros(G.shape[0], np.int8); rel[pos.cpu().numpy()] = 1; rel[q] = -1
                t, apt, np_, _ = query_tables(rel[order], K)
                n_bad += not (np.array_equal(t, rel_top[i]) and abs(apt - ap_tail[i]) < 1e-5 and np_ == npos[i])
    if n_bad > 2:
        raise RuntimeError(f"compact tables disagree on {n_bad} of 20 queries")
    if do_dist:
        inl, nv = fill(gcache, qs, top, v.verify)
        np.savez_compressed(out / 'dist' / 'gldv2c_val.npz', queries=qs, labels=labels, top_idx=top, inliers=inl, cand_sim=cand,
                            sim_top=sim_top, rel_top=rel_top, ap_tail=ap_tail, npos=npos, n_pool=n_pool)
        log['dist_gld'] = {'verified': nv, 'minutes': round((time.time() - t0) / 60, 1), 'table_check_bad': n_bad,
                           'distractor_frac': float((top >= n_pool).mean())}
        print('dist gld', log['dist_gld'], flush=True)
    v.close()
    del G, P, D; torch.cuda.empty_cache()

    # ---------------- benchmarks (database) and + R1M
    names = [l.strip() for l in open(Path(a.r1m_dir) / 'revisitop1m.txt') if l.strip()]
    R = torch.from_numpy(np.load(E / 'r1m.npy').astype(np.float32)).to(dev)
    tops = {}
    for name in ('roxford5k', 'rparis6k'):
        Qe = torch.from_numpy(np.load(E / f'{name}_q.npy').astype(np.float32)).to(dev)
        Gd = torch.from_numpy(np.load(E / f'{name}_db.npy').astype(np.float32)).to(dev)
        for tag, G in (('q10k', Gd), ('r1m', torch.cat([Gd, R]))):
            sm = Qe @ G.t(); _, ti = sm.topk(K, dim=1)
            tops[(name, tag)] = (sm.half().cpu().numpy(), ti.cpu().numpy(),
                                 torch.einsum('qkd,qld->qkl', G[ti], G[ti]).half().cpu().numpy(), Gd.shape[0])
    want = set()
    for (name, tag), (_, top, _, n_db) in tops.items():
        miss = bcache[name].get(np.repeat(np.arange(top.shape[0])[:, None], K, 1), top) < 0
        want |= {names[g - n_db] for g in top[miss] if g >= n_db}
    if want:
        archives = [str(Path(a.r1m_dir) / 'raw' / f'revisitop1m.{i}.tar.gz') for i in range(1, 101)]
        with Pool(24) as p:
            got = sum(p.map(extract_needed, [(f, want, str(tmp / 'r1m_emb')) for f in archives]))
        if got != len(want):
            raise RuntimeError(f"extracted {got} of {len(want)} R1M images")
    log['r1m_extracted'] = len(want)
    for name in ('roxford5k', 'rparis6k'):
        h = RevisitedDataset(str(root / name), name)
        crops = []
        (tmp / 'crops' / name).mkdir(parents=True, exist_ok=True)
        for i, (img_name, e) in enumerate(zip(h.cfg['query_images'], h.cfg['gnd'])):
            im = Image.open(h.root_dir / 'jpg' / img_name).convert('RGB')
            if e.get('bbx') is not None:
                im = im.crop(tuple(int(round(x)) for x in e['bbx']))
            p = tmp / 'crops' / name / f'q{i:03d}.jpg'; im.save(p, quality=95); crops.append(str(p))
        db_paths = [str(h.root_dir / 'jpg' / x) for x in h.cfg['db_images']]
        for tag in ('q10k', 'r1m'):
            sim_full, top, cand, n_db = tops[(name, tag)]

            def verify(pairs):
                gal = sorted({g for _, g in pairs})
                pths = crops + [db_paths[g] if g < n_db else str(tmp / 'r1m_emb' / names[g - n_db]) for g in gal]
                missing = [x for x in pths if not Path(x).exists()]
                if missing:
                    raise RuntimeError(f"{len(missing)} images missing, e.g. {missing[:2]}")
                gpos = {g: len(crops) + i for i, g in enumerate(gal)}
                vv = Verifier('/', pths, dev, max_kpts=1024, resize=512, batch=32, cpu_workers=16, io_threads=8)
                r = vv.verify([(q, gpos[g]) for q, g in pairs]); vv.close()
                return r
            inl, nv = fill(bcache[name], np.arange(top.shape[0]), top, verify)
            extra = {'n_db': n_db} if tag == 'r1m' else {}
            np.savez_compressed(out / tag / f'{name}.npz', top_idx=top, inliers=inl, sim_full=sim_full, cand_sim=cand, **extra)
            log[f'{name}_{tag}'] = {'verified': nv, 'match_ge35': float((inl >= 35).mean())}
            print(name, tag, log[f'{name}_{tag}'], flush=True)
    (out / 'precompute_emb_log.json').write_text(json.dumps(log, indent=2))


if __name__ == '__main__':
    main()
