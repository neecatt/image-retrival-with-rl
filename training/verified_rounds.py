"""
Round-based training with a geometric verifier in the loop (Track N, docs/RL_RESEARCH_PLAN.md).

Each round: embed the pool with the current model, choose k candidates for a set of queries,
verify the (query, candidate) pairs that are not cached yet, then train on query lists whose
relevance comes from the verifier. Arms differ only in how candidates are chosen and the objective:
  rl      candidates = Plackett-Luce sample (Gumbel top-k) from the top-`shortlist` scores at
          temperature tau; objective = PL policy gradient with AP@k reward
  remine  candidates = greedy top-k of the current model; objective = FastAP
  static  candidates = greedy top-k of the initial model, all verified in round 0 with the same
          total verifier budget; objective = FastAP
Within a batch, a query's relevance to every item is: verified match (inliers >= threshold) if the
pair is in the cache, otherwise unknown. Unknown items with the query's noisy label are masked
(they may be matches); unknown items with another label are negatives. Memory-bank items follow
the same rule. Only queries are anchors.
"""

import random
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from training.pl_policy import ap_at_k, pl_log_prob


@torch.no_grad()
def choose_candidates(q_emb: torch.Tensor, g_emb: torch.Tensor, q_idx: torch.Tensor, k: int,
                      mode: str, tau: float = 0.05, shortlist: int = 100, chunk: int = 1024,
                      seed: int = 0) -> torch.Tensor:
    """(Q, k) candidate gallery indices per query (self excluded). mode: 'greedy' or 'sample'."""
    gen = torch.Generator(device=q_emb.device).manual_seed(seed)
    out = []
    for s in range(0, q_emb.size(0), chunk):
        sim = q_emb[s:s + chunk] @ g_emb.t()
        sim[torch.arange(sim.size(0), device=sim.device), q_idx[s:s + chunk]] = float('-inf')
        top_s, top_i = sim.topk(shortlist, dim=1)
        if mode == 'greedy':
            out.append(top_i[:, :k])
        else:
            u = torch.rand(top_s.shape, generator=gen, device=top_s.device).clamp_(1e-10, 1 - 1e-7)
            pick = (top_s / tau - torch.log(-torch.log(u))).topk(k, dim=1).indices
            out.append(torch.gather(top_i, 1, pick))
    return torch.cat(out)


class ListBatchSampler(torch.utils.data.Sampler):
    """Flat index batches [q, c_1..c_m, q', ...] from verified lists; each query contributes up to
    `pos_per_query` verified positives and fills the rest with its verified negatives."""

    def __init__(self, lists: Dict[int, Tuple[List[int], List[int]]], queries_per_batch: int,
                 cands_per_query: int, pos_per_query: int, num_batches: int, seed: int):
        # need >= 1 verified positive and enough verified candidates for a fixed block size
        self.lists = {q: v for q, v in lists.items() if v[0] and len(v[0]) + len(v[1]) >= cands_per_query}
        self.qs = sorted(self.lists)
        self.B, self.m, self.p, self.n, self.seed = queries_per_batch, cands_per_query, pos_per_query, num_batches, seed
        if len(self.qs) < self.B:
            raise ValueError("too few queries with a verified positive")

    def __iter__(self):
        rng = random.Random(self.seed)
        for _ in range(self.n):
            batch = []
            for q in rng.sample(self.qs, self.B):
                pos, neg = self.lists[q]
                pp = rng.sample(pos, min(self.p, len(pos)))
                nn_ = rng.sample(neg, min(self.m - len(pp), len(neg)))
                rest = [x for x in pos if x not in pp]
                fill = rng.sample(rest, min(self.m - len(pp) - len(nn_), len(rest)))
                batch += [q] + pp + nn_ + fill
            yield batch

    def __len__(self):
        return self.n


def relevance(q_ids: Sequence[int], item_ids: Sequence[int], labels: np.ndarray, cache, threshold: int):
    """rel, valid (Q, N) bool from cached verification; unknown same-label items are masked."""
    Q, N = len(q_ids), len(item_ids)
    rel = np.zeros((Q, N), dtype=bool)
    valid = np.ones((Q, N), dtype=bool)
    lab_items = labels[np.asarray(item_ids)]
    for a, q in enumerate(q_ids):
        for b, it in enumerate(item_ids):
            if it == q:
                valid[a, b] = False
                continue
            c = cache.get(q, it)
            if c >= 0:
                rel[a, b] = c >= threshold
            elif lab_items[b] == labels[q]:
                valid[a, b] = False
    return rel, valid


def pl_loss_masked(q: torch.Tensor, gal: torch.Tensor, rel: torch.Tensor, valid: torch.Tensor,
                   tau: float, num_rankings: int, topk: int):
    """PL policy gradient (AP@k reward, leave-one-out baseline) with explicit relevance/validity."""
    scores = (q @ gal.t() / tau).masked_fill(~valid, float('-inf'))
    relf = (rel & valid).float()
    npos = relf.sum(-1)
    k = min(topk, int(valid.sum(-1).min().item()) - 1)
    R = num_rankings
    with torch.no_grad():
        u = torch.rand(R, *scores.shape, device=q.device).clamp_(1e-10, 1 - 1e-7)
        idx = (scores.detach().unsqueeze(0) - torch.log(-torch.log(u))).topk(k, -1).indices
        reward = ap_at_k(torch.gather(relf.unsqueeze(0).expand(R, -1, -1), -1, idx), npos)
        greedy = ap_at_k(torch.gather(relf, -1, scores.detach().topk(k, -1).indices), npos)
        adv = reward - (reward.sum(0, keepdim=True) - reward) / (R - 1)
        has = npos > 0
    logp = pl_log_prob(scores, idx)
    loss = -(adv * logp)[:, has].mean()
    return loss, {'reward': reward[:, has].mean().item(), 'reward_greedy': greedy[has].mean().item()}


def fastap_masked(q: torch.Tensor, gal: torch.Tensor, rel: torch.Tensor, valid: torch.Tensor,
                  num_bins: int = 10, topk: int = 100):
    """FastAP (Cakir et al. 2019) over the items marked valid, positives = rel."""
    sim = q @ gal.t()
    v, r = valid.float(), (rel & valid).float()
    npos = r.sum(-1)
    d = (2 - 2 * sim).clamp(0, 4)
    centers = torch.linspace(0, 4, num_bins + 1, device=q.device)
    pulse = (1 - (d[..., None] - centers).abs() / (4.0 / num_bins)).clamp(min=0)
    h_pos, h_all = (pulse * (r * v)[..., None]).sum(1), (pulse * v[..., None]).sum(1)
    ap = (h_pos.cumsum(-1) * h_pos / h_all.cumsum(-1).clamp(min=1e-8)).sum(-1) / npos.clamp(min=1)
    has = npos > 0
    with torch.no_grad():
        k = min(topk, int(valid.sum(-1).min().item()) - 1)
        greedy = ap_at_k(torch.gather(r, -1, sim.masked_fill(~valid, -3).topk(k, -1).indices), npos)
    return (1 - ap[has]).mean(), {'reward_greedy': greedy[has].mean().item()}
