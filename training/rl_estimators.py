"""
Multi-sample REINFORCE estimator for embedding policies (Phase 1 of
docs/RL_RESEARCH_PLAN.md).

Policy: each query and gallery embedding mean μ (unit norm) defines a
sampler a = normalize(μ + σ ε) with log-density proxy κ μ·a, κ = 1/σ²
(a detached), as in RLTrainer._sample_on_sphere.

Product rollouts (after RELER, arXiv 2610.07731): S query samples per
query and S gallery "bundles" (one sample of every gallery item) give S×S
rewards per query from the precomputed similarities, with no extra
network forward passes. Leave-one-out baselines:
  query sample s:  A_q[i,s] = mean_t R[i,s,t] - mean_{s'≠s} mean_t R[i,s',t]
  gallery bundle t (per query): A_g[i,t] = mean_s R[i,s,t] - mean_{t'≠t} mean_s R[i,s,t']
Gallery items receive gradient through their bundle log-probabilities, so
the gallery is part of the policy rather than environment noise.
Memory-bank items are deterministic context (no samples, no gradient).
"""

from typing import Callable, Dict, Optional

import torch
import torch.nn.functional as F


def batched_ap(sim: torch.Tensor, rel: torch.Tensor) -> torch.Tensor:
    """Non-interpolated AP per row. sim, rel: (..., N). Rows without positives give 0."""
    order = sim.argsort(dim=-1, descending=True)
    r = torch.gather(rel, -1, order)
    hits = r.cumsum(-1)
    ranks = torch.arange(1, r.size(-1) + 1, device=r.device, dtype=r.dtype)
    npos = r.sum(-1)
    return (hits / ranks * r).sum(-1) / npos.clamp(min=1)


def sample_sphere(mean: torch.Tensor, sigma: float, num: int):
    """num samples per row: returns samples (num, *mean.shape) and log-prob proxy (num, rows)."""
    noise = sigma * torch.randn((num,) + tuple(mean.shape), device=mean.device, dtype=mean.dtype)
    samples = F.normalize(mean.unsqueeze(0) + noise, dim=-1)
    log_prob = (samples.detach() * mean.unsqueeze(0)).sum(-1) / sigma ** 2
    return samples, log_prob


def multi_sample_policy_loss(
    q_mean: torch.Tensor, g_mean: torch.Tensor,
    q_labels: torch.Tensor, g_labels: torch.Tensor,
    sigma: float, num_samples: int,
    mem_emb: Optional[torch.Tensor] = None, mem_labels: Optional[torch.Tensor] = None,
    sample_gallery: bool = True, normalize_advantage: bool = False,
    reward_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
):
    """Return (policy loss to minimise, stats). reward_fn(sim, rel) -> per-row reward; default AP."""
    reward_fn = reward_fn or batched_ap
    S = num_samples
    nq, ng = q_mean.size(0), g_mean.size(0)

    q_s, q_logp = sample_sphere(q_mean, sigma, S)                     # (S,nq,d), (S,nq)
    if sample_gallery:
        g_s, g_logp = sample_sphere(g_mean, sigma, S)                 # (S,ng,d), (S,ng)
    else:
        g_s, g_logp = g_mean.detach().unsqueeze(0).expand(S, ng, -1), None

    rel_g = (q_labels[:, None] == g_labels[None, :]).float()           # (nq,ng)
    with torch.no_grad():
        sim = torch.einsum('sqd,tgd->qstg', q_s.detach(), g_s.detach())  # (nq,S,S,ng)
        rel = rel_g[:, None, None, :].expand(nq, S, S, ng)
        if mem_emb is not None and mem_emb.numel() > 0:
            sim_m = torch.einsum('sqd,md->qsm', q_s.detach(), mem_emb)    # (nq,S,M)
            sim = torch.cat([sim, sim_m[:, :, None, :].expand(-1, -1, S, -1)], dim=-1)
            rel_m = (q_labels[:, None] == mem_labels[None, :]).float()
            rel = torch.cat([rel, rel_m[:, None, None, :].expand(-1, S, S, -1)], dim=-1)
        R = reward_fn(sim, rel)                                        # (nq,S,S)
        has_pos = rel[:, 0, 0].sum(-1) > 0

        if S > 1:
            r_q = R.mean(2)                                            # (nq,S) query sample s
            A_q = r_q - (r_q.sum(1, keepdim=True) - r_q) / (S - 1)
            r_g = R.mean(1)                                            # (nq,S) gallery bundle t
            A_g = r_g - (r_g.sum(1, keepdim=True) - r_g) / (S - 1)
        else:
            A_q = A_g = torch.zeros_like(R[:, :, 0])
        if normalize_advantage:
            A_q = A_q / (R.flatten(1).std(1, keepdim=True) + 1e-6)
            A_g = A_g / (R.flatten(1).std(1, keepdim=True) + 1e-6)
        A_q = A_q * has_pos[:, None]
        A_g = A_g * has_pos[:, None]

    # mean over queries and samples; gallery bundle log-prob = sum over items
    loss = -(A_q * q_logp.t()).mean()
    if sample_gallery:
        loss = loss - (A_g * g_logp.sum(1)[None, :]).mean()

    stats = {
        'reward': R.mean().item(),
        'adv_q_abs': A_q.abs().mean().item(),
        'adv_g_abs': A_g.abs().mean().item(),
        'reward_std_within_query': R.flatten(1).std(1).mean().item(),
        'frac_queries_with_signal': (R.flatten(1).std(1) > 1e-6).float().mean().item(),
    }
    return loss, stats
