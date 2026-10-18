#!/usr/bin/env python3
"""Track V-D: GLDv2 policy-training shortlists over a large gallery (val pool + GLDv2 distractors).

The GLDv2-clean val queries of an existing v_precompute.py run (--plain_pre, same queries) are ranked
against [the 21,285 val-pool images, the ~910k GLDv2 distractors of scripts/prepare_gldv2_distractors.py],
so policies can be trained on shortlists that look like ROxford/RParis + R1M (about half distractors)
without touching R1M, which stays test-only. Distractors come from landmarks absent from GLDv2-clean,
so they are never relevant to a val query.

Embeddings use the same eval transform as v_precompute.py (512, centre crop). Inliers depend only on the
image pair, so pairs already verified in --plain_pre are reused and only new pairs are verified.

The full similarity matrix (10k x 931k) is not stored. Instead, per query: relevance of the top-K, the AP
contribution of positives below K and the number of positives (exactly what VerifyEnv needs, as in
query_tables), checked against query_tables on the full ranking for the first queries.
Output: <out>/gldv2c_val.npz with queries, labels, top_idx, inliers, cand_sim, sim_top, rel_top, ap_tail,
npos, n_pool (compact format read by train_verify_policy.py), and precompute_dist_log.json.
"""
import argparse, json, pickle, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dataset import get_transforms                                       # noqa: E402
from data.dataset_sfm120k import SfM120kDataset, _gldv2_id_to_path, load_gldv2c   # noqa: E402
from evaluate import create_model_from_checkpoint                             # noqa: E402
from evaluation.sfm_val import extract_embeddings                             # noqa: E402
from training.verifier import Verifier                                        # noqa: E402
from training.verify_policy import query_tables                               # noqa: E402


def tail_tables(pos_ranks, npos, K):
    """AP contribution of positives at ranks >= K (0-based, junk-free ranking) as in query_tables."""
    r = np.sort(pos_ranks)
    pos_in_top = int((r < K).sum())
    ap = 0.0
    for t, rk in enumerate(r[r >= K]):
        j = pos_in_top + t
        ap += (j / rk + (j + 1) / (rk + 1)) / 2
    return ap / max(npos, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', required=True, help='staged gldv2c and gldv2d')
    ap.add_argument('--model', required=True)
    ap.add_argument('--plain_pre', required=True, help='v_precompute.py output of the same model (queries, inliers)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--cache_dir', default='/home/atuin/b268dc/b268dc21/datasets/gldv2d/embeddings')
    ap.add_argument('--tag', required=True, help='name of the distractor embedding cache for this model')
    ap.add_argument('--K', type=int, default=100)
    ap.add_argument('--workers', type=int, default=16)
    a = ap.parse_args()
    dev = torch.device('cuda')
    root, out = Path(a.data_root), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    model = create_model_from_checkpoint(a.model, dev).eval()
    tf = get_transforms(512, 'eval')
    log = {'model': a.model, 'plain_pre': a.plain_pre, 'K': a.K}

    t0 = time.time()
    _, val = load_gldv2c(str(root / 'gldv2c'), val_transform=tf)
    with torch.no_grad():
        emb, _ = extract_embeddings(model, val, dev, batch_size=64, num_workers=a.workers)
    labels = np.asarray(val.labels)
    n_pool = emb.shape[0]
    plain = np.load(Path(a.plain_pre) / 'gldv2c_val.npz')
    qs = plain['queries']
    assert np.array_equal(plain['labels'], labels), "val pool differs from --plain_pre"

    dpk = pickle.load(open(root / 'gldv2d' / 'gldv2d.pkl', 'rb'))
    d_paths = [_gldv2_id_to_path(i) for i in dpk['cids']]
    cache = Path(a.cache_dir) / f'{a.tag}_e512.npy'
    if cache.exists():
        D = torch.from_numpy(np.load(cache).astype(np.float32)).to(dev)
    else:
        dds = SfM120kDataset(str(root / 'gldv2d' / 'jpg'), d_paths, [0] * len(d_paths), transform=tf)
        with torch.no_grad():
            D, _ = extract_embeddings(model, dds, dev, batch_size=64, num_workers=a.workers)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, D.half().cpu().numpy())
    log['embed_minutes'] = round((time.time() - t0) / 60, 1)
    log['distractors'] = int(D.shape[0])
    print(json.dumps(log), flush=True)
    del model; torch.cuda.empty_cache()

    # sanity: the fresh val-pool similarities reproduce the plain run
    q_t = torch.as_tensor(qs, device=dev)
    s_chk = (emb[q_t[:64]] @ emb.t()).half().cpu().numpy().astype(np.float32)
    ref = plain['sim_full'][:64].astype(np.float32)
    ok = np.isfinite(ref)
    log['max_sim_diff_vs_plain'] = float(np.abs(s_chk - ref)[ok].max())
    print('max |sim - plain sim| =', log['max_sim_diff_vs_plain'], flush=True)

    gal = torch.cat([emb, D])
    lab_t = torch.as_tensor(labels, device=dev)
    Q, K = len(qs), a.K
    top_idx = np.zeros((Q, K), dtype=np.int64); sim_top = np.zeros((Q, K), dtype=np.float32)
    rel_top = np.zeros((Q, K), dtype=np.int8); ap_tail = np.zeros(Q, dtype=np.float32)
    npos = np.zeros(Q, dtype=np.int64); cand = np.zeros((Q, K, K), dtype=np.float16)
    n_checked = n_bad = 0
    for s in range(0, Q, 128):
        qb = q_t[s:s + 128]
        sim = emb[qb] @ gal.t()
        sim[torch.arange(len(qb), device=dev), qb] = float('-inf')            # the query itself
        ts, ti = sim.topk(K, dim=1)
        g_top = gal[ti]
        cand[s:s + len(qb)] = torch.einsum('qkd,qld->qkl', g_top, g_top).half().cpu().numpy()
        top_idx[s:s + len(qb)] = ti.cpu().numpy(); sim_top[s:s + len(qb)] = ts.cpu().numpy()
        for b in range(len(qb)):
            q = int(qb[b])
            pos = torch.nonzero(lab_t == lab_t[q]).flatten()
            pos = pos[pos != q]
            ranks = (sim[b][None, :] > sim[b][pos][:, None]).sum(1).cpu().numpy()   # 0-based ranks
            i = s + b
            npos[i] = len(pos)
            rel_top[i] = (top_idx[i] < n_pool) & (labels[np.minimum(top_idx[i], n_pool - 1)] == labels[q])
            ap_tail[i] = tail_tables(ranks, len(pos), K)
            if n_checked < 20:                                             # exact check on the full ranking
                order = torch.argsort(sim[b], descending=True).cpu().numpy()
                rel = np.zeros(gal.shape[0], dtype=np.int8)
                rel[pos.cpu().numpy()] = 1; rel[q] = -1
                t, apt, np_, _ = query_tables(rel[order], K)
                n_bad += not (np.array_equal(t, rel_top[i]) and abs(apt - ap_tail[i]) < 1e-5 and np_ == npos[i])
                n_checked += 1
    if n_bad > 2:                                                          # (exact ties may reorder one item)
        raise RuntimeError(f"compact AP tables disagree with query_tables on {n_bad} of {n_checked} queries")
    log['table_check'] = f"{n_checked - n_bad}/{n_checked} exact"
    log['r1m_like_frac'] = float((top_idx >= n_pool).mean())
    log['rel_in_shortlist'] = float(rel_top.mean())
    print(json.dumps(log), flush=True)

    # inliers: reuse the plain run's pairs, verify the rest
    known = {}
    for i in range(Q):
        for c, v in zip(plain['top_idx'][i], plain['inliers'][i]):
            known[(i, int(c))] = int(v)
    inl = np.full((Q, K), -1, dtype=np.int64)
    todo = []
    for i in range(Q):
        for k in range(K):
            v = known.get((i, int(top_idx[i, k])))
            if v is None:
                todo.append((i, k))
            else:
                inl[i, k] = v
    log['pairs_reused'] = int(Q * K - len(todo)); log['pairs_verified'] = len(todo)
    print(f"verifying {len(todo)} new pairs ({log['pairs_reused']} reused)", flush=True)
    t1 = time.time()
    pool_paths = [str(Path(val.image_root) / p) for p in val.image_paths]
    paths = pool_paths + [str(root / 'gldv2d' / 'jpg' / p) for p in d_paths]
    v = Verifier('/', paths, dev, max_kpts=1024, resize=512, batch=32, cpu_workers=16, io_threads=8)
    res = v.verify([(int(qs[i]), int(top_idx[i, k])) for i, k in todo])
    v.close()
    for (i, k), x in zip(todo, res):
        inl[i, k] = int(x)
    assert (inl >= 0).all()
    log['verify_minutes'] = round((time.time() - t1) / 60, 1)
    log['match_ge35'] = float((inl >= 35).mean())
    log['distractor_match_ge35'] = float((inl[top_idx >= n_pool] >= 35).mean())
    np.savez_compressed(out / 'gldv2c_val.npz', queries=qs, labels=labels, top_idx=top_idx, inliers=inl,
                        cand_sim=cand, sim_top=sim_top, rel_top=rel_top, ap_tail=ap_tail, npos=npos, n_pool=n_pool)
    log['minutes'] = round((time.time() - t0) / 60, 1)
    (out / 'precompute_dist_log.json').write_text(json.dumps(log, indent=2))
    print(json.dumps(log, indent=2), flush=True)


if __name__ == '__main__':
    main()
