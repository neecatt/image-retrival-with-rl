"""
Batched geometric verifier for verifier-in-the-loop training (Track N, docs/RL_RESEARCH_PLAN.md).

Same pipeline as scripts/oracle_label.py (ALIKED features, LightGlue matches, MAGSAC fundamental-
matrix inliers), made fast enough for training rounds:
  - features are extracted once per image and cached on the GPU (fp16 descriptors);
  - images are decoded in a thread pool;
  - pairs whose images both have exactly max_kpts keypoints are matched in batches (LightGlue's
    adaptive depth/width pruning is disabled so batches have a fixed shape); other pairs one by one;
  - RANSAC runs in a CPU process pool.
Inlier counts are not comparable to the 2048-keypoint oracle used for SfM-120k; thresholds for
Track N are set on this verifier's own distribution.
"""

from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from lightglue import ALIKED, LightGlue
from lightglue.utils import numpy_image_to_torch


def _ransac(args):
    import cv2
    k0, k1 = args
    if len(k0) < 8:
        return 0
    cv2.setNumThreads(1)
    try:
        _, mask = cv2.findFundamentalMat(k0, k1, cv2.USAC_MAGSAC, 1.0, 0.999, 10000)
    except cv2.error:
        return 0
    return int(mask.sum()) if mask is not None else 0


class Verifier:
    def __init__(self, image_root: str, rel_paths: Sequence[str], device, max_kpts: int = 1024,
                 resize: int = 512, batch: int = 32, cpu_workers: int = 16, io_threads: int = 16,
                 feature_cache: int = 100_000):
        self.root, self.paths, self.dev = Path(image_root), rel_paths, device
        self.M, self.resize, self.batch, self.cache_cap = int(max_kpts), int(resize), int(batch), int(feature_cache)
        self.extractor = ALIKED(max_num_keypoints=self.M, detection_threshold=1e-6).eval().to(device)
        self.matcher = LightGlue(features='aliked', depth_confidence=-1, width_confidence=-1).eval().to(device)
        self.feats: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self.io = ThreadPoolExecutor(io_threads)
        self.pool = Pool(cpu_workers)

    def close(self):
        self.pool.close(); self.io.shutdown()

    def _load(self, i: int):
        return i, numpy_image_to_torch(np.asarray(Image.open(self.root / self.paths[i]).convert('RGB')))

    @torch.no_grad()
    def _extract(self, idx: List[int]):
        todo = [i for i in dict.fromkeys(idx) if i not in self.feats]
        if len(self.feats) + len(todo) > self.cache_cap:
            self.feats.clear()
        for i, img in self.io.map(self._load, todo):
            f = self.extractor.extract(img.to(self.dev), resize=self.resize)
            self.feats[i] = (f['keypoints'][0], f['descriptors'][0].half(), f['image_size'][0])

    @torch.no_grad()
    def verify(self, pairs: Sequence[Tuple[int, int]]) -> np.ndarray:
        """Inlier count for each (i, j) pair."""
        pairs = list(pairs)
        if not pairs:
            return np.zeros(0, dtype=np.int64)
        self._extract([x for p in pairs for x in p])
        full = [k for k, (i, j) in enumerate(pairs)
                if self.feats[i][0].shape[0] == self.M and self.feats[j][0].shape[0] == self.M]
        full_set = set(full)
        rest = [k for k in range(len(pairs)) if k not in full_set]
        kp_jobs: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        groups = [full[s:s + self.batch] for s in range(0, len(full), self.batch)] + [[k] for k in rest]
        for g in groups:
            f0 = [self.feats[pairs[k][0]] for k in g]
            f1 = [self.feats[pairs[k][1]] for k in g]
            d0 = {'keypoints': torch.stack([f[0] for f in f0]), 'descriptors': torch.stack([f[1] for f in f0]).float(),
                  'image_size': torch.stack([f[2] for f in f0])}
            d1 = {'keypoints': torch.stack([f[0] for f in f1]), 'descriptors': torch.stack([f[1] for f in f1]).float(),
                  'image_size': torch.stack([f[2] for f in f1])}
            m0 = self.matcher({'image0': d0, 'image1': d1})['matches0']           # (B, M), -1 = unmatched
            for b, k in enumerate(g):
                sel = m0[b] > -1
                kp_jobs[k] = (d0['keypoints'][b][sel].cpu().numpy(), d1['keypoints'][b][m0[b][sel]].cpu().numpy())
        inl = self.pool.map(_ransac, [kp_jobs[k] for k in range(len(pairs))], chunksize=16)
        return np.asarray(inl, dtype=np.int64)


class PairCache:
    """Verified inlier counts keyed by unordered image pair, persisted as .npz."""

    def __init__(self, path: str = None):
        self.path, self.d = path, {}
        if path and Path(path).exists():
            z = np.load(path)
            self.d = {(int(a), int(b)): int(c) for a, b, c in zip(z['i'], z['j'], z['inliers'])}

    @staticmethod
    def key(i, j):
        return (i, j) if i < j else (j, i)

    def missing(self, pairs):
        return list(dict.fromkeys(self.key(i, j) for i, j in pairs if i != j and self.key(i, j) not in self.d))

    def add(self, pairs, inliers):
        for p, c in zip(pairs, inliers):
            self.d[self.key(*p)] = int(c)

    def get(self, i, j, default=-1):
        return self.d.get(self.key(i, j), default)

    def __len__(self):
        return len(self.d)

    def save(self):
        if self.path and self.d:
            k = np.array(list(self.d.keys())); v = np.array(list(self.d.values()))
            tmp = str(self.path) + '.tmp.npz'
            np.savez(tmp, i=k[:, 0], j=k[:, 1], inliers=v)
            Path(tmp).replace(self.path)
