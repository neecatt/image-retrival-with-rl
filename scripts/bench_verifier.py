#!/usr/bin/env python3
"""Track N1c: throughput and agreement of the batched verifier (training/verifier.py).

On a staged GLDv2 split: verify random same-landmark and random different-landmark pairs with the
batched verifier; report pairs/s and how well inliers separate the two groups (AUC). On a subset,
compare with the per-pair 2048-keypoint oracle used before (Spearman correlation, AUC).
"""
import argparse, json, random, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data.dataset_sfm120k import load_gldv2c, load_gldv2n   # noqa: E402
from training.verifier import Verifier                       # noqa: E402


def auc(pos, neg):
    from scipy.stats import mannwhitneyu
    return float(mannwhitneyu(pos, neg).statistic / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--dataset', default='gldv2c', choices=['gldv2c', 'gldv2n'])
    ap.add_argument('--pairs', type=int, default=5000)
    ap.add_argument('--compare', type=int, default=400)
    ap.add_argument('--max_kpts', type=int, default=1024)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--out', default='results/n1_verifier_bench')
    a = ap.parse_args()
    dev = torch.device('cuda')
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    fn = load_gldv2c if a.dataset == 'gldv2c' else load_gldv2n
    train, _ = fn(str(Path(a.data_root) / a.dataset))
    lab = np.asarray(train.labels)
    by = {}
    for i, l in enumerate(lab):
        by.setdefault(int(l), []).append(i)
    multi = [l for l, v in by.items() if len(v) >= 2]
    rng = random.Random(0)
    pos = [tuple(rng.sample(by[l], 2)) for l in rng.sample(multi, a.pairs)]
    neg = []
    while len(neg) < a.pairs:
        i, j = rng.randrange(len(lab)), rng.randrange(len(lab))
        if lab[i] != lab[j]:
            neg.append((i, j))
    pairs = pos + neg
    rng.shuffle(pairs)
    is_pos = np.array([lab[i] == lab[j] for i, j in pairs])

    v = Verifier(train.image_root, train.image_paths, dev, max_kpts=a.max_kpts, batch=a.batch,
                 cpu_workers=16, io_threads=16)
    v.verify(pairs[:200])                                   # warm-up
    t0 = time.time()
    inl = v.verify(pairs)
    sec = time.time() - t0
    full = np.mean([f[0].shape[0] == a.max_kpts for f in v.feats.values()])
    res = {'dataset': a.dataset, 'max_kpts': a.max_kpts, 'batch': a.batch, 'pairs': len(pairs),
           'seconds': round(sec, 1), 'pairs_per_s': round(len(pairs) / sec, 1),
           'images_with_full_kpts': float(full),
           'auc_same_vs_diff': auc(inl[is_pos], inl[~is_pos]),
           'same_landmark_ge25': float((inl[is_pos] >= 25).mean()), 'diff_landmark_ge25': float((inl[~is_pos] >= 25).mean()),
           'same_median': float(np.median(inl[is_pos])), 'diff_median': float(np.median(inl[~is_pos])),
           'diff_p99': float(np.percentile(inl[~is_pos], 99))}
    print(json.dumps(res, indent=2), flush=True)

    # agreement with the per-pair 2048-keypoint oracle on a subset
    from lightglue import ALIKED, LightGlue
    from lightglue.utils import load_image, rbd
    from oracle_label import ransac_inliers
    ext = ALIKED(max_num_keypoints=2048).eval().to(dev); mat = LightGlue(features='aliked').eval().to(dev)
    sub = list(range(min(a.compare, len(pairs))))
    ref = []
    with torch.no_grad():
        for k in sub:
            i, j = pairs[k]
            f = [ext.extract(load_image(Path(train.image_root) / train.image_paths[x]).to(dev), resize=512) for x in (i, j)]
            m = rbd(mat({'image0': f[0], 'image1': f[1]}))['matches']
            ref.append(ransac_inliers((rbd(f[0])['keypoints'][m[:, 0]].cpu().numpy(), rbd(f[1])['keypoints'][m[:, 1]].cpu().numpy())))
    ref = np.array(ref)
    from scipy.stats import spearmanr
    res['compare'] = {'pairs': len(sub), 'spearman_vs_2048_oracle': float(spearmanr(ref, inl[sub]).correlation),
                      'auc_2048_oracle': auc(ref[is_pos[sub]], ref[~is_pos[sub]]),
                      'auc_batched_same_subset': auc(inl[sub][is_pos[sub]], inl[sub][~is_pos[sub]])}
    print(json.dumps(res['compare'], indent=2))
    (out / f'bench_{a.dataset}_k{a.max_kpts}.json').write_text(json.dumps(res, indent=2))
    v.close()


if __name__ == '__main__':
    main()
