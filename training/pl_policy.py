"""
Plackett–Luce ranking policy (Phase 3 of docs/RL_RESEARCH_PLAN.md).

The action is a ranking, not a perturbed embedding. For query i the policy
scores every gallery item with s_ij = q_i·g_j / τ and draws a top-k ranking
from the Plackett–Luce distribution, i.e. repeatedly picks the next item with
probability softmax over the items not yet placed. Sampling uses the Gumbel
top-k trick (top-k of s + Gumbel noise is an exact PL top-k sample).

log π(ranking) = Σ_{r<k} [ s_{π_r} − log Σ_{j not in π_{<r}} exp s_j ]
is an exact, differentiable function of the scores, so the score-function
gradient flows into query and gallery embeddings without the proxy density
of the Gaussian embedding policy (PG-Rank, Singh & Joachims 2019). The
normaliser for position r splits into the chosen items at positions ≥ r
(reverse logcumsumexp) and all items outside the top-k (one logsumexp), so
every term is computed stably.

Every batch item is a query against all other batch items plus the memory
bank (deterministic means, no gradient). Leave-one-out baselines over the R
sampled rankings of the same query (Kool et al. 2019).
"""

from typing import Dict, Optional, Tuple

import torch


def ap_at_k(rel_ranked: torch.Tensor, npos: torch.Tensor) -> torch.Tensor:
    """AP of a top-k list; positives outside the list count as missed. rel_ranked (..., k), npos broadcastable."""
    ranks = torch.arange(1, rel_ranked.size(-1) + 1, device=rel_ranked.device, dtype=rel_ranked.dtype)
    prec = rel_ranked.cumsum(-1) / ranks
    return (prec * rel_ranked).sum(-1) / npos.clamp(min=1)


def pl_log_prob(scores: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """log PL probability of the top-k lists idx (R, B, k) under scores (B, N); -inf scores are excluded items."""
    R = idx.size(0)
    s = scores.unsqueeze(0).expand(R, -1, -1)
    s_sel = torch.gather(s, -1, idx)                                   # (R,B,k)
    chosen = torch.zeros_like(s, dtype=torch.bool).scatter_(-1, idx, True)
    lse_rest = s.masked_fill(chosen, float('-inf')).logsumexp(-1)       # (R,B)
    tail = s_sel.flip(-1).logcumsumexp(-1).flip(-1)                     # chosen items at positions >= r
    log_z = torch.logaddexp(tail, lse_rest.unsqueeze(-1))
    return (s_sel - log_z).sum(-1)


def pl_policy_loss(emb: torch.Tensor, labels: torch.Tensor, tau: float, num_rankings: int, topk: int,
                   mem_emb: Optional[torch.Tensor] = None, mem_labels: Optional[torch.Tensor] = None
                   ) -> Tuple[torch.Tensor, Dict[str, float]]:
    """REINFORCE loss with AP@k reward for PL rankings of each batch item against the rest (+ memory)."""
    B = emb.size(0)
    gal, glab = emb, labels
    if mem_emb is not None and mem_emb.numel() > 0:
        gal = torch.cat([emb, mem_emb.detach()])
        glab = torch.cat([labels, mem_labels])
    N = gal.size(0)
    self_mask = torch.zeros(B, N, dtype=torch.bool, device=emb.device)
    self_mask[torch.arange(B), torch.arange(B)] = True
    scores = (emb @ gal.t() / tau).masked_fill(self_mask, float('-inf'))
    rel = ((labels[:, None] == glab[None, :]) & ~self_mask).float()
    npos = rel.sum(-1)
    k = min(topk, N - 2)          # leave at least one unplaced item so every normaliser is finite
    R = num_rankings

    with torch.no_grad():
        u = torch.rand(R, B, N, device=emb.device).clamp_(1e-10, 1 - 1e-7)
        idx = (scores.detach().unsqueeze(0) - torch.log(-torch.log(u))).topk(k, -1).indices
        reward = ap_at_k(torch.gather(rel.unsqueeze(0).expand(R, -1, -1), -1, idx), npos)    # (R,B)
        greedy = ap_at_k(torch.gather(rel, -1, scores.detach().topk(k, -1).indices), npos)   # (B,)
        adv = reward - (reward.sum(0, keepdim=True) - reward) / (R - 1)
        has_pos = npos > 0

    logp = pl_log_prob(scores, idx)                                     # (R,B)
    loss = -(adv * logp)[:, has_pos].mean() if has_pos.any() else scores.new_zeros(())
    stats = {'reward': reward[:, has_pos].mean().item(),
             'reward_mean_policy': greedy[has_pos].mean().item(),
             'adv_q_abs': adv[:, has_pos].abs().mean().item(),
             'reward_std_within_query': reward[:, has_pos].std(0).mean().item()}
    return loss, stats
