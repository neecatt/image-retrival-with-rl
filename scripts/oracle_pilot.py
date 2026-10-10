#!/usr/bin/env python3
"""Phase 2.2 pilot: geometric-verification relevance oracle on a 1% sample of SfM-120k train.

For N random queries: candidates = up to P same-cluster images (positives by label) + the K hardest
different-cluster images from a random pool under a Stage 1 model. Each (query, candidate) pair is
verified with ALIKED (BSD-3) + LightGlue (Apache-2.0) matches and a MAGSAC fundamental-matrix fit;
the inlier count is the graded relevance signal. Reports throughput (to cost the full run) and how
inliers relate to the binary landmark labels and to Stage 1 similarity.
"""
import argparse, json, random, sys, time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dataset import get_transforms                       # noqa: E402
from data.dataset_sfm120k import load_sfm120k                 # noqa: E402
from evaluate import create_model_from_checkpoint             # noqa: E402
from lightglue import ALIKED, LightGlue                       # noqa: E402
from lightglue.utils import load_image, rbd                    # noqa: E402


@torch.no_grad()
def embed(model, ds, idx, device, workers):
    loader = DataLoader(Subset(ds, idx), batch_size=16, num_workers=workers)
    return torch.cat([model(x.to(device)).float() for x, _, _ in loader])


def auc(scores, labels):
    order = np.argsort(scores)
    ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels.astype(bool); n1, n0 = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--queries', type=int, default=200)
    ap.add_argument('--pos', type=int, default=25)
    ap.add_argument('--neg', type=int, default=25)
    ap.add_argument('--pool', type=int, default=5000)
    ap.add_argument('--max_kpts', type=int, default=2048)
    ap.add_argument('--resize', type=int, default=1024)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    dev = torch.device('cuda')

    train, _ = load_sfm120k(str(Path(a.data_root) / 'sfm120k'), train_transform=get_transforms(1024, mode='eval'))
    labels = np.array(train.labels)
    by_label = {}
    for i, l in enumerate(labels):
        by_label.setdefault(int(l), []).append(i)

    queries = rng.sample(range(len(train)), a.queries)
    pool = rng.sample(range(len(train)), a.pool)
    model = create_model_from_checkpoint(a.checkpoint, dev)
    t0 = time.time()
    q_emb = embed(model, train, queries, dev, a.workers)
    p_emb = embed(model, train, pool, dev, a.workers)
    t_embed = time.time() - t0

    pairs = []
    for qi, q in enumerate(queries):
        same = [i for i in by_label[int(labels[q])] if i != q]
        pos = rng.sample(same, min(a.pos, len(same)))
        sims = (q_emb[qi:qi + 1] @ p_emb.t())[0]
        sims[torch.tensor(labels[pool] == labels[q], device=dev)] = -2   # exclude same landmark
        neg = [pool[j] for j in sims.topk(a.neg).indices.tolist()]
        for c in pos:
            pairs.append((q, c, 1))
        for c in neg:
            pairs.append((q, c, 0))
    del model
    torch.cuda.empty_cache()

    extractor = ALIKED(max_num_keypoints=a.max_kpts).eval().to(dev)
    matcher = LightGlue(features='aliked').eval().to(dev)
    root = Path(train.image_root)
    feats = {}
    images = sorted({i for p in pairs for i in p[:2]})
    t0 = time.time()
    with torch.no_grad():
        for i in images:
            img = load_image(root / train.image_paths[i])
            feats[i] = extractor.extract(img.to(dev), resize=a.resize)
    t_feat = time.time() - t0

    # Stage 1 similarity for every pair (re-embed the few candidates outside the pool lazily)
    model = create_model_from_checkpoint(a.checkpoint, dev)
    cand = sorted({c for _, c, _ in pairs})
    c_emb = dict(zip(cand, embed(model, train, cand, dev, a.workers)))
    q_map = dict(zip(queries, q_emb))

    rows = []
    t0 = time.time(); t_ransac = 0.0
    with torch.no_grad():
        for q, c, is_pos in pairs:
            f0, f1 = feats[q], feats[c]
            m = rbd(matcher({'image0': f0, 'image1': f1}))['matches']
            k0 = rbd(f0)['keypoints'][m[:, 0]].cpu().numpy()
            k1 = rbd(f1)['keypoints'][m[:, 1]].cpu().numpy()
            inl = 0
            tr = time.time()
            if len(m) >= 8:
                _, mask = cv2.findFundamentalMat(k0, k1, cv2.USAC_MAGSAC, 1.0, 0.999, 10000)
                inl = int(mask.sum()) if mask is not None else 0
            t_ransac += time.time() - tr
            rows.append({'q': q, 'c': c, 'pos': is_pos, 'matches': int(len(m)), 'inliers': inl,
                         'sim': float(q_map[q] @ c_emb[c])})
    t_match = time.time() - t0

    inl = np.array([r['inliers'] for r in rows]); lab = np.array([r['pos'] for r in rows])
    sim = np.array([r['sim'] for r in rows])
    pos_i, neg_i = inl[lab == 1], inl[lab == 0]
    q_of = lambda x: {p: float(np.percentile(x, p)) for p in (10, 25, 50, 75, 90)}
    from scipy.stats import spearmanr
    summary = {
        'pairs': len(rows), 'queries': len(queries), 'images_featurised': len(images),
        'seconds': {'embed': round(t_embed, 1), 'features': round(t_feat, 1), 'match_total': round(t_match, 1),
                    'ransac': round(t_ransac, 1)},
        'throughput': {'images_per_s': round(len(images) / t_feat, 2), 'pairs_per_s': round(len(rows) / t_match, 2)},
        'inliers_pos_percentiles': q_of(pos_i), 'inliers_neg_percentiles': q_of(neg_i),
        'auc_inliers_vs_label': auc(inl, lab), 'auc_stage1_sim_vs_label': auc(sim, lab),
        'pos_with_lt15_inliers': float((pos_i < 15).mean()), 'neg_with_ge15_inliers': float((neg_i >= 15).mean()),
        'neg_with_ge50_inliers': float((neg_i >= 50).mean()),
        'spearman_inliers_sim_within_pos': float(spearmanr(pos_i, sim[lab == 1]).correlation),
    }
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / 'pairs.json').write_text(json.dumps(rows))
    (Path(a.out) / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
