#!/usr/bin/env python3
"""Track V-QE: baselines reviewers will ask for, budget curves and timings, per descriptor.

  aqe        alpha query expansion without verification (Radenovic et al.): q' = norm(q + sum_{i<=n} s_i^a g_i),
             full re-search. (n, a) chosen on GLDv2 val (mean of plain pool and pool + distractors).
  vqe        spatially verified query expansion (Chum et al. 2007 style), same budget as the policies:
             verify the B most similar, put matches first (by inliers), re-query with
             q' = norm(q + sum_matches s^a g) for everything else. a chosen on GLDv2 val.
  verify_all expansion environment (training/qe_env.py) with every available candidate verified,
             i.e. the upper bound of the expansion re-ranking rule; plus top-B / shortlist-B curves.
  timing     verifier pairs/s on ROxford pairs, policy step time, R1M search time (first descriptor).
Nothing here uses benchmark numbers for any choice. Output: <out>/<descriptor>.json
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dataset import LongSideResize, RevisitedDataset, get_transforms     # noqa: E402
from data.dataset_sfm120k import load_gldv2c                                    # noqa: E402
from evaluate import create_model_from_checkpoint                               # noqa: E402
from evaluation.revisited_ap import revisited_map                               # noqa: E402
from evaluation.sfm_val import extract_embeddings                               # noqa: E402
from eval_baselines import embed                                                # noqa: E402

NORM = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
AQE_GRID = [(n, al) for n in (1, 2, 3, 5, 10) for al in (0.0, 3.0)]
VQE_GRID = [0.0, 1.0, 3.0]


def gld_map(sim, qs, labels, chunk=64):
    """Mean AP over GLDv2 queries; sim (Q, N) on GPU with the query itself at -inf (junk, ranked last).
    Gallery items beyond len(labels) are distractors (label -1, never relevant; landmark ids are >= 0)."""
    lab = torch.full((sim.shape[1],), -1, dtype=torch.long, device=sim.device)
    lab[:len(labels)] = torch.as_tensor(labels, device=sim.device)
    aps = []
    for s in range(0, sim.shape[0], chunk):
        sb = sim[s:s + chunk]
        order = torch.argsort(sb, dim=1, descending=True)[:, :-1]          # drop the query (last)
        rel = (lab[order] == lab[torch.as_tensor(qs[s:s + chunk], device=sim.device)][:, None]).float()
        npos = rel.sum(1)
        r = torch.arange(rel.shape[1], device=sim.device, dtype=torch.float32)[None]
        j = rel.cumsum(1) - rel
        p0 = torch.where(r > 0, j / r.clamp(min=1), torch.ones_like(r))
        p1 = (j + 1) / (r + 1)
        aps.append(((p0 + p1) / 2 * rel).sum(1) / npos.clamp(min=1))
    return float(torch.cat(aps).mean())


def aqe_query(q, G, base_sim, n, al):
    top_s, top_i = base_sim.topk(n, dim=1)
    w = top_s.clamp(min=0) ** al
    return F.normalize(q + (w[..., None] * G[top_i]).sum(1), dim=1)


def vqe_sim(q, G, base_sim, top_idx, inliers, B, thr, al, self_idx=None):
    """Verified QE with budget B: matches among the B most similar first (by inliers), rest re-queried."""
    ti = torch.as_tensor(top_idx[:, :B], device=G.device)
    inl = torch.as_tensor(inliers[:, :B], device=G.device).float()
    m = inl >= thr
    s = torch.gather(base_sim, 1, ti)
    w = torch.where(m, s.clamp(min=0) ** al, torch.zeros_like(s))
    q2 = F.normalize(q + (w[..., None] * G[ti]).sum(1), dim=1)
    sim = q2 @ G.t()
    if self_idx is not None:
        sim[torch.arange(len(q), device=G.device), self_idx] = float('-inf')
    bonus = torch.where(m, 1e3 + inl, torch.full_like(inl, float('-inf')))
    sim.scatter_(1, ti, torch.maximum(torch.gather(sim, 1, ti), bonus))
    return sim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--desc', required=True, help='name for the output')
    ap.add_argument('--model', required=True)
    ap.add_argument('--plain_pre', required=True)
    ap.add_argument('--dist_pre', required=True)
    ap.add_argument('--r1m_pre', required=True)
    ap.add_argument('--qe_pre', required=True)
    ap.add_argument('--dist_emb', required=True)
    ap.add_argument('--r1m_emb', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--B', type=int, default=20)
    ap.add_argument('--threshold', type=int, default=35)
    ap.add_argument('--gld_queries', type=int, default=3000)
    ap.add_argument('--timing', action='store_true')
    ap.add_argument('--workers', type=int, default=16)
    a = ap.parse_args()
    dev = torch.device('cuda')
    out = {'desc': a.desc, 'model': a.model, 'B': a.B}
    t0 = time.time()
    model = create_model_from_checkpoint(a.model, dev).eval()

    # ---------------- GLDv2 val: choose (n, a) for aQE and a for vQE
    _, val = load_gldv2c(str(Path(a.data_root) / 'gldv2c'), val_transform=get_transforms(512, 'eval'))
    with torch.no_grad():
        emb, _ = extract_embeddings(model, val, dev, batch_size=64, num_workers=a.workers)
    emb = F.normalize(emb.float(), dim=1)
    labels = np.asarray(val.labels)
    D = F.normalize(torch.from_numpy(np.load(a.dist_emb).astype(np.float32)).to(dev), dim=1)
    gl = {}
    for tag, pre, G in (('plain', a.plain_pre, emb), ('dist', a.dist_pre, None)):
        z = np.load(Path(pre) / 'gldv2c_val.npz')
        n = a.gld_queries
        qs, top, inl = z['queries'][:n], z['top_idx'][:n], z['inliers'][:n]
        if G is None:
            G = torch.cat([emb, D])
        q_t = torch.as_tensor(qs, device=dev)
        q = emb[q_t]
        r = {}
        for s in range(0, n, 500):                                    # chunked to bound memory
            qb, sl = q[s:s + 500], slice(s, s + 500)
            base = qb @ G.t(); base[torch.arange(len(qb), device=dev), q_t[sl]] = float('-inf')
            r.setdefault('none', []).append((gld_map(base, qs[sl], labels), len(qb)))
            for nn_, al in AQE_GRID:
                q2 = aqe_query(qb, G, base, nn_, al)
                s2 = q2 @ G.t(); s2[torch.arange(len(qb), device=dev), q_t[sl]] = float('-inf')
                r.setdefault(f'aqe_n{nn_}_a{al:g}', []).append((gld_map(s2, qs[sl], labels), len(qb)))
            for al in VQE_GRID:
                s3 = vqe_sim(qb, G, base, top[sl], inl[sl], a.B, a.threshold, al, self_idx=q_t[sl])
                r.setdefault(f'vqe_a{al:g}', []).append((gld_map(s3, qs[sl], labels), len(qb)))
        gl[tag] = {k: sum(x * w for x, w in v) / sum(w for _, w in v) for k, v in r.items()}
        print(tag, json.dumps({k: round(100 * v, 2) for k, v in gl[tag].items()}), flush=True)
    sel = {k: (gl['plain'][k] + gl['dist'][k]) / 2 for k in gl['plain']}
    best_aqe = max([k for k in sel if k.startswith('aqe')], key=sel.get)
    best_vqe = max([k for k in sel if k.startswith('vqe')], key=sel.get)
    out['gld_val'] = gl; out['selected'] = {'aqe': best_aqe, 'vqe': best_vqe}
    print('selected on GLDv2 val:', best_aqe, best_vqe, flush=True)
    n_sel = int(best_aqe.split('_n')[1].split('_')[0]); a_sel = float(best_aqe.split('_a')[1]); v_sel = float(best_vqe.split('_a')[1])
    del D; torch.cuda.empty_cache()

    # ---------------- benchmarks
    tf = T.Compose([LongSideResize(1024), T.ToTensor(), NORM])
    R = torch.from_numpy(np.load(a.r1m_emb).astype(np.float32)).to(dev)
    bench, timing = {}, {}
    for name in ('roxford5k', 'rparis6k'):
        h = RevisitedDataset(str(Path(a.data_root) / name), name)
        gnd = h.get_ground_truth()
        Q = torch.from_numpy(embed(model, h.get_query_dataset(tf), dev, a.workers)).to(dev)
        Gd = torch.from_numpy(embed(model, h.get_database_dataset(tf), dev, a.workers)).to(dev)
        for suffix, pre in (('', a.plain_pre), ('_r1m', a.r1m_pre)):
            G = torch.cat([Gd, R]) if suffix else Gd
            z = np.load(Path(pre) / f'{name}.npz')
            t1 = time.time(); base = Q @ G.t(); torch.cuda.synchronize(); t_search = time.time() - t1
            res = {'none': revisited_map(base.cpu().numpy(), gnd)}
            res['aqe'] = revisited_map((aqe_query(Q, G, base, n_sel, a_sel) @ G.t()).cpu().numpy(), gnd)
            res['vqe'] = revisited_map(vqe_sim(Q, G, base, z['top_idx'], z['inliers'], a.B, a.threshold, v_sel).cpu().numpy(), gnd)
            bench[name + suffix] = res
            if suffix:
                timing[f'{name}_r1m_search_s_per_query'] = t_search / Q.shape[0]
            print(name + suffix, json.dumps({k: (round(v['mAP_medium'], 2), round(v['mAP_hard'], 2)) for k, v in res.items()}), flush=True)
        del G
    out['bench'] = bench

    # ---------------- expansion environment: verify everything, and heuristic budget curves
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from train_qe_policy import load_pool, evaluate                 # noqa: E402
    from training.qe_env import QEEnv
    curves = {}
    for pool in ('roxford5k', 'rparis6k', 'roxford5k_r1m', 'rparis6k_r1m'):
        for prot in ('medium', 'hard'):
            d, _ = load_pool(Path(a.qe_pre) / f'{pool}.npz', dev, prot)
            C = d['valid'].shape[1]
            c = {}
            for B in (5, 10, 20, 40, 80, C):
                c[f'top_B{B}'] = evaluate(QEEnv(d, B, a.threshold), lambda f: f[..., 0] * 100)
            for B in (5, 10, 20, 40, 80, 100):
                c[f'shortlist_B{B}'] = evaluate(QEEnv(d, B, a.threshold),
                                                lambda f: torch.where(f[..., 11] > 0, torch.full_like(f[..., 0], -1e4), f[..., 0] * 100))
            curves[f'{pool}_{prot}'] = c
        print(pool, json.dumps({k: round(100 * v, 2) for k, v in curves[f'{pool}_medium'].items()}), flush=True)
    out['budget_curves'] = curves

    # ---------------- timings (verifier, policy step)
    if a.timing:
        from training.verifier import Verifier
        from training.verify_policy import AttnPolicy
        h = RevisitedDataset(str(Path(a.data_root) / 'roxford5k'), 'roxford5k')
        z = np.load(Path(a.plain_pre) / 'roxford5k.npz')
        paths = [str(h.root_dir / 'jpg' / x) for x in h.cfg['db_images']]
        v = Verifier('/', paths, dev, max_kpts=1024, resize=512, batch=32, cpu_workers=16, io_threads=8)
        pairs = [(int(z['top_idx'][q, 0]), int(c)) for q in range(z['top_idx'].shape[0]) for c in z['top_idx'][q, 1:31]]
        v.verify(pairs[:64])                                          # warm-up
        t1 = time.time(); v.verify(pairs); timing['verifier_pairs_per_s'] = len(pairs) / (time.time() - t1)
        timing['verifier_note'] = 'pairs among database images (features extracted inside the timing), ALIKED 1024 kpts @512 + LightGlue + MAGSAC, one GPU'
        v.close()
        pol = AttnPolicy(n_feat=QEEnv.n_feat).to(dev).eval()
        f = torch.randn(70, 250, QEEnv.n_feat, device=dev); Cm = torch.rand(70, 250, 250, device=dev)
        with torch.no_grad():
            for _ in range(3):
                pol(f, Cm)
            torch.cuda.synchronize(); t1 = time.time()
            for _ in range(20):
                pol(f, Cm)
            torch.cuda.synchronize()
        timing['policy_step_ms_per_70_queries'] = (time.time() - t1) / 20 * 1000
    out['timing'] = timing
    out['minutes'] = round((time.time() - t0) / 60, 1)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / f'{a.desc}.json').write_text(json.dumps(out, indent=1))
    print(json.dumps(timing), flush=True)


if __name__ == '__main__':
    main()
