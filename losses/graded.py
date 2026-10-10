"""
Graded-relevance objectives for Phase 2 of docs/RL_RESEARCH_PLAN.md.

- `ndcg_at_k`: exact nDCG@k with exponential gains (2^g - 1); non-differentiable.
  Used as the RL reward (training/rl_estimators.py reward_fn) and for validation.
- `approx_ndcg_loss`: ApproxNDCG (Qin, Liu & Li, "A general approximation framework
  for direct optimization of information retrieval measures", Inf. Retr. 2010):
  ranks replaced by sums of sigmoids, so nDCG becomes differentiable. This is the
  graded differentiable control that reward tuning has to beat, trained on the
  same graded labels.

Shapes: sim, gains are (..., N); gains are non-negative grades (0 = irrelevant).
"""

import torch


def _discounts(n: int, device, dtype) -> torch.Tensor:
    return 1.0 / torch.log2(torch.arange(2, n + 2, device=device, dtype=dtype))


def ndcg_at_k(sim: torch.Tensor, gains: torch.Tensor, k: int = 10) -> torch.Tensor:
    """Exact nDCG@k per row; rows with no relevant item return 0."""
    k = min(k, sim.size(-1))
    disc = _discounts(k, sim.device, sim.dtype)
    g = torch.pow(2.0, gains.to(sim.dtype)) - 1
    top = sim.topk(k, dim=-1).indices
    dcg = (torch.gather(g, -1, top) * disc).sum(-1)
    ideal = (g.topk(k, dim=-1).values * disc).sum(-1)
    return torch.where(ideal > 0, dcg / ideal.clamp(min=1e-12), torch.zeros_like(dcg))


def approx_ndcg_loss(sim: torch.Tensor, gains: torch.Tensor, tau: float = 0.05) -> torch.Tensor:
    """1 - ApproxNDCG (full list), averaged over rows that have a relevant item."""
    g = torch.pow(2.0, gains.to(sim.dtype)) - 1
    diff = (sim.unsqueeze(-2) - sim.unsqueeze(-1)) / tau            # [..., i, j] = s_j - s_i
    approx_rank = 0.5 + torch.sigmoid(diff).sum(-1)                  # self term contributes 0.5
    dcg = (g / torch.log2(1 + approx_rank)).sum(-1)
    n = sim.size(-1)
    ideal = (g.sort(dim=-1, descending=True).values * _discounts(n, sim.device, sim.dtype)).sum(-1)
    valid = ideal > 0
    ndcg = dcg[valid] / ideal[valid]
    return (1 - ndcg).mean() if valid.any() else sim.sum() * 0
