"""
Track V: RL for budgeted geometric verification (docs/RL_RESEARCH_PLAN.md).

A fixed descriptor gives each query a top-K shortlist. A policy may verify B of the K items, one at a
time, seeing each outcome before the next choice. After B steps, verified matches (inliers >= T) move
to the top of the list in order of inliers; everything else keeps its global order. Reward = AP of the
final ranking (official revisitop AP; junk ignored).

Reordering inside the top-K does not change the rank of any positive below K, so
AP = AP of the reordered top-K part + a per-query constant for positives below K; both are computed
here from the precomputed relevance of every gallery item (no images needed).

Per-candidate features (static and state-dependent):
  sim to query, rank/K, sim minus top-1 sim, max/mean sim to verified matches, max sim to verified
  non-matches, inliers-weighted sim to matches, fraction of budget used, fraction of matches so far.
"""

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------------------------
# AP bookkeeping
# ---------------------------------------------------------------------------------------------
def query_tables(order_rel: np.ndarray, K: int) -> Tuple[np.ndarray, float, int, int]:
    """From the relevance of the full global ranking of one query (1 pos, 0 neg, -1 junk) return:
    rel of the top-K (K,), the AP contribution of positives below K, the number of positives,
    and the number of non-junk items in the top-K (their ranks are 0..n_k-1 after junk removal)."""
    top = order_rel[:K]
    npos = int((order_rel == 1).sum())
    nonjunk = order_rel[order_rel != -1]
    n_k = int((top != -1).sum())
    pos_in_top = int((top == 1).sum())
    tail = nonjunk[n_k:]
    tail_pos_ranks = np.where(tail == 1)[0] + n_k
    ap_tail = 0.0
    for t, r in enumerate(tail_pos_ranks):
        j = pos_in_top + t
        p0 = j / r if r > 0 else 1.0
        p1 = (j + 1) / (r + 1)
        ap_tail += (p0 + p1) / 2
    return top.astype(np.int8), ap_tail / max(npos, 1), npos, n_k


def ap_top(rel: torch.Tensor, npos: torch.Tensor) -> torch.Tensor:
    """AP contribution of the top-K part. rel (..., K) in final order with 1/0/-1 (junk ignored)."""
    nonjunk = (rel != -1).float()
    pos = (rel == 1).float()
    r = nonjunk.cumsum(-1) - 1                       # rank among non-junk items
    j = pos.cumsum(-1) - pos                         # positives ranked above
    p0 = torch.where(r > 0, j / r.clamp(min=1), torch.ones_like(r))
    p1 = (j + 1) / (r + 1).clamp(min=1)         # r = -1 only at leading junk, where pos = 0
    return ((p0 + p1) / 2 * pos).sum(-1) / npos.clamp(min=1)


def final_order(match: torch.Tensor, inliers: torch.Tensor, verified: torch.Tensor) -> torch.Tensor:
    """Permutation of the K slots: verified matches first by inliers (desc), then the rest in order."""
    K = match.size(-1)
    key = torch.where(verified & match, 1e6 + inliers.float(), -torch.arange(K, device=match.device).float())
    return key.argsort(-1, descending=True)


# ---------------------------------------------------------------------------------------------
# Environment (batched over queries x trajectories)
# ---------------------------------------------------------------------------------------------
class VerifyEnv:
    def __init__(self, data: Dict[str, torch.Tensor], budget: int, threshold: int = 35, hard_mix: float = 0.0):
        # data: sim (Q,K), cand_sim (Q,K,K), inliers (Q,K), rel (Q,K) int, npos (Q,), ap_tail (Q,)
        self.d, self.B, self.T = data, budget, threshold
        # hard_mix > 0 (training only): reward = (1 - w) AP + w AP_hardproxy, where the hard proxy
        # ignores "easy" positives (data['rel_hard'], 'npos_hard', 'ap_tail_hard'), like the Revisited Hard protocol
        self.w = hard_mix if 'rel_hard' in data else 0.0
        self.Q, self.K = data['sim'].shape
        self.match = data['inliers'] >= threshold

    def features(self, idx, verified, outcome):
        """idx (N,) query ids; verified/outcome (N,K) bool. Returns (N,K,F)."""
        sim, C, inl = self.d['sim'][idx], self.d['cand_sim'][idx].float(), self.d['inliers'][idx].float()
        K = self.K
        posm = (verified & outcome).float()
        negm = (verified & ~outcome).float()
        neg_inf = torch.full_like(C, -1.0)
        max_pos = torch.where(posm[:, None, :] > 0, C, neg_inf).amax(-1)
        max_neg = torch.where(negm[:, None, :] > 0, C, neg_inf).amax(-1)
        mean_pos = (C * posm[:, None, :]).sum(-1) / posm.sum(-1, keepdim=True).clamp(min=1)
        w = posm * torch.log1p(inl)
        wpos = (C * w[:, None, :]).sum(-1) / w.sum(-1, keepdim=True).clamp(min=1e-6)
        used = verified.float().sum(-1, keepdim=True) / self.B
        frac_m = posm.sum(-1, keepdim=True) / verified.float().sum(-1, keepdim=True).clamp(min=1)
        rank = torch.arange(K, device=sim.device).float()[None].expand_as(sim) / K
        f = torch.stack([sim, rank, sim - sim[:, :1], max_pos, mean_pos, max_neg, wpos,
                         used.expand_as(sim), frac_m.expand_as(sim), (posm.sum(-1, keepdim=True) > 0).float().expand_as(sim)], -1)
        return f

    def reward(self, idx, verified):
        match, inl, rel = self.match[idx], self.d['inliers'][idx], self.d['rel'][idx]
        perm = final_order(match, inl, verified)
        ap = ap_top(torch.gather(rel, -1, perm), self.d['npos'][idx].float()) + self.d['ap_tail'][idx]
        if self.w > 0:
            rh = torch.gather(self.d['rel_hard'][idx], -1, perm)
            ap_h = ap_top(rh, self.d['npos_hard'][idx].float()) + self.d['ap_tail_hard'][idx]
            ap = (1 - self.w) * ap + self.w * ap_h
        return ap


class Policy(nn.Module):
    def __init__(self, n_feat: int = 10, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_feat, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))

    def forward(self, f):
        return self.net(f).squeeze(-1)


def rollout(env: VerifyEnv, idx: torch.Tensor, scorer, mode: str = 'sample', temperature: float = 1.0):
    """Run B verification steps for queries idx. scorer(features) -> (N,K) logits.
    mode: 'sample' (policy), 'greedy' (argmax). Returns final AP (N,), sum of log-probs (N,)."""
    N, K = idx.numel(), env.K
    dev = env.d['sim'].device
    verified = torch.zeros(N, K, dtype=torch.bool, device=dev)
    match = env.match[idx]
    logp = torch.zeros(N, device=dev)
    for _ in range(env.B):
        f = env.features(idx, verified, match)
        logits = scorer(f) / temperature
        logits = logits.masked_fill(verified, float('-inf'))
        if mode == 'sample':
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()
            logp = logp + dist.log_prob(a)
        else:
            a = logits.argmax(-1)
        verified = verified.clone()
        verified[torch.arange(N, device=dev), a] = True
    return env.reward(idx, verified), logp, verified
