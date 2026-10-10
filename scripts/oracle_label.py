#!/usr/bin/env python3
"""Phase 2.4: graded relevance labels for SfM-120k from geometric verification.

Only same-landmark pairs are verified (other landmarks are grade 0 by construction).
  train: up to Q queries per landmark, each paired with up to P random same-landmark images
  val:   all unordered same-landmark pairs (for graded leave-one-out validation)
Per pair: ALIKED (BSD-3) features at long side 1024, LightGlue (Apache-2.0) matches, MAGSAC
fundamental-matrix inliers. RANSAC runs in a CPU process pool while the GPU matches the next pairs.
Sharded by landmark: shard s handles landmarks with index % num_shards == s.
Output: <out>/<split>_shard<s>.npz with arrays i, j (indices into the split as loaded by
load_sfm120k), matches, inliers.
"""
import argparse, random, sys, time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dataset_sfm120k import load_sfm120k      # noqa: E402
from lightglue import ALIKED, LightGlue            # noqa: E402
from lightglue.utils import load_image, rbd         # noqa: E402


def ransac_inliers(args):
    k0, k1 = args
    if len(k0) < 8:
        return 0
    cv2.setNumThreads(1)
    try:
        _, mask = cv2.findFundamentalMat(k0, k1, cv2.USAC_MAGSAC, 1.0, 0.999, 10000)
    except cv2.error:
        return 0
    return int(mask.sum()) if mask is not None else 0


def pairs_for_landmark(members, split, rng, q_per, p_per):
    if split == 'val':
        return [(a, b) for x, a in enumerate(members) for b in members[x + 1:]]
    queries = rng.sample(members, min(q_per, len(members)))
    out = set()
    for q in queries:
        others = [m for m in members if m != q]
        for c in rng.sample(others, min(p_per, len(others))):
            out.add((min(q, c), max(q, c)))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--split', choices=['train', 'val'], required=True)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--num_shards', type=int, default=1)
    ap.add_argument('--queries_per_landmark', type=int, default=40)
    ap.add_argument('--positives_per_query', type=int, default=25)
    ap.add_argument('--max_kpts', type=int, default=2048)
    ap.add_argument('--resize', type=int, default=1024)
    ap.add_argument('--cpu_workers', type=int, default=24)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    dev = torch.device('cuda')

    train, val = load_sfm120k(str(Path(a.data_root) / 'sfm120k'))
    ds = train if a.split == 'train' else val
    by_label = {}
    for idx, l in enumerate(ds.labels):
        by_label.setdefault(int(l), []).append(idx)
    landmarks = sorted(by_label)[a.shard::a.num_shards]

    extractor = ALIKED(max_num_keypoints=a.max_kpts).eval().to(dev)
    matcher = LightGlue(features='aliked').eval().to(dev)
    root = Path(ds.image_root)
    out_i, out_j, out_m, pending = [], [], [], []
    t0, n_pairs = time.time(), 0
    with Pool(a.cpu_workers) as pool, torch.no_grad():
        for li, lab in enumerate(landmarks):
            rng = random.Random(a.seed * 100003 + lab)
            pairs = pairs_for_landmark(by_label[lab], a.split, rng, a.queries_per_landmark, a.positives_per_query)
            if not pairs:
                continue
            feats = {}
            for idx in sorted({x for p in pairs for x in p}):
                feats[idx] = extractor.extract(load_image(root / ds.image_paths[idx]).to(dev), resize=a.resize)
            jobs = []
            for i, j in pairs:
                m = rbd(matcher({'image0': feats[i], 'image1': feats[j]}))['matches']
                k0 = rbd(feats[i])['keypoints'][m[:, 0]].cpu().numpy()
                k1 = rbd(feats[j])['keypoints'][m[:, 1]].cpu().numpy()
                jobs.append((k0, k1)); out_i.append(i); out_j.append(j); out_m.append(len(m))
            pending.append(pool.map_async(ransac_inliers, jobs, chunksize=16))
            n_pairs += len(pairs)
            if li % 10 == 0:
                el = time.time() - t0
                print(f"[shard {a.shard}] landmark {li + 1}/{len(landmarks)}  pairs {n_pairs}  "
                      f"{n_pairs / el:.1f} pairs/s  elapsed {el / 60:.1f} min", flush=True)
        inliers = [x for p in pending for x in p.get()]

    Path(a.out).mkdir(parents=True, exist_ok=True)
    f = Path(a.out) / f"{a.split}_shard{a.shard}.npz"
    np.savez_compressed(f, i=np.array(out_i), j=np.array(out_j), matches=np.array(out_m), inliers=np.array(inliers))
    print(f"saved {f}: {len(out_i)} pairs in {(time.time() - t0) / 60:.1f} min")


if __name__ == '__main__':
    main()
