"""
Candidate-list batches with graded relevance (Phase 2 of docs/RL_RESEARCH_PLAN.md).

Relevance levels: different landmark = 0. Same landmark, by geometric
verification (scripts/oracle_label.py): inliers < 25 -> 1, 25-99 -> 2,
100-299 -> 3, >= 300 -> 4. Same-landmark images without visible overlap stay
relevant (level 1) because the Revisited benchmarks count other views of a
landmark as positives; verification only orders positives. Same-landmark
pairs that were not verified are unknown and are masked out of rankings.

A batch holds B queries from B distinct landmarks, each with C verified
same-landmark candidates. Every image in the batch is gallery for every query
(except itself): own candidates carry their grades, all other images are other
landmarks (grade 0). Graded RL and the graded control both train on these batches.
"""

import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

GRADE_EDGES = (25, 100, 300)


def inliers_to_grade(inliers: np.ndarray) -> np.ndarray:
    return np.searchsorted(np.array(GRADE_EDGES), inliers, side='right').astype(np.int8)


def load_oracle(oracle_dir: str, split: str) -> Dict[Tuple[int, int], int]:
    """{(i, j): relevance level 1-4} for verified same-landmark pairs, both orders."""
    grades = {}
    files = sorted(Path(oracle_dir).glob(f'{split}_shard*.npz'))
    if not files:
        raise FileNotFoundError(f"no {split}_shard*.npz in {oracle_dir}")
    for f in files:
        d = np.load(f)
        for i, j, g in zip(d['i'].tolist(), d['j'].tolist(), (1 + inliers_to_grade(d['inliers'])).tolist()):
            grades[(i, j)] = g
            grades[(j, i)] = g
    return grades


class GradedListSampler(torch.utils.data.Sampler):
    """Yields flat index lists: [q_1, c_11..c_1C, q_2, c_21..., ...] for B distinct landmarks."""

    def __init__(self, labels: Sequence[int], grades: Dict[Tuple[int, int], int],
                 queries_per_batch: int = 4, candidates_per_query: int = 7,
                 num_batches: int = 1000, seed: int = 0):
        self.B, self.C, self.n, self.seed = queries_per_batch, candidates_per_query, num_batches, seed
        self.epoch = 0
        partners: Dict[int, List[int]] = {}
        for (i, j) in grades:
            partners.setdefault(i, []).append(j)
        self.partners = {q: sorted(v) for q, v in partners.items() if len(v) >= self.C}
        self.by_label: Dict[int, List[int]] = {}
        for q in self.partners:
            self.by_label.setdefault(int(labels[q]), []).append(q)
        self.landmarks = sorted(self.by_label)
        if len(self.landmarks) < self.B:
            raise ValueError("fewer landmarks with enough verified candidates than queries per batch")

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed * 1_000_003 + self.epoch)
        for _ in range(self.n):
            batch = []
            for lab in rng.sample(self.landmarks, self.B):
                q = rng.choice(self.by_label[lab])
                batch.append(q)
                batch.extend(rng.sample(self.partners[q], self.C))
            yield batch

    def __len__(self):
        return self.n


def batch_gains(indices: torch.Tensor, labels: torch.Tensor, grades: Dict[Tuple[int, int], int],
                queries_per_batch: int, candidates_per_query: int):
    """For a flat batch from GradedListSampler return
    q_pos: (B,) positions of queries in the batch,
    gains: (B, N) grade of every batch item for each query,
    valid: (B, N) bool, False for the query itself and unverified same-landmark items."""
    B, C = queries_per_batch, candidates_per_query
    N = indices.numel()
    q_pos = torch.arange(B) * (C + 1)
    idx = indices.tolist()
    lab = labels.tolist()
    gains = torch.zeros(B, N)
    valid = torch.ones(B, N, dtype=torch.bool)
    for b, p in enumerate(q_pos.tolist()):
        q = idx[p]
        for n in range(N):
            if n == p:
                valid[b, n] = False
            elif lab[n] == lab[p]:
                g = grades.get((q, idx[n]))
                if g is None:
                    valid[b, n] = False
                else:
                    gains[b, n] = g
    return q_pos, gains, valid
