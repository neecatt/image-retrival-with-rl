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


def softmax_control_loss(emb: torch.Tensor, labels: torch.Tensor, tau: float, topk: int,
                         mem_emb: Optional[torch.Tensor] = None, mem_labels: Optional[torch.Tensor] = None
                         ) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Differentiable control for the PL policy on identical inputs: multi-positive softmax
    cross-entropy (SupCon, Khosla et al. 2020) over the same scores q·g/τ, batch and memory bank.
    It is the maximum-likelihood counterpart of the PL policy (the PL first-place probability of
    each positive), so the comparison isolates reward-driven updates from the bank and τ."""
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
    has_pos = npos > 0
    log_p = scores - scores.logsumexp(-1, keepdim=True)
    per_q = -(log_p.masked_fill(rel == 0, 0) * rel).sum(-1) / npos.clamp(min=1)
    loss = per_q[has_pos].mean() if has_pos.any() else scores.new_zeros(())
    with torch.no_grad():
        k = min(topk, N - 2)
        greedy = ap_at_k(torch.gather(rel, -1, scores.topk(k, -1).indices), npos)
    return loss, {'reward': greedy[has_pos].mean().item(), 'reward_mean_policy': greedy[has_pos].mean().item()}


def _bank_scores(emb, labels, mem_emb, mem_labels):
    """Cosine scores of every batch item against the batch plus the memory bank, with masks."""
    B = emb.size(0)
    gal, glab = emb, labels
    if mem_emb is not None and mem_emb.numel() > 0:
        gal = torch.cat([emb, mem_emb.detach()])
        glab = torch.cat([labels, mem_labels])
    N = gal.size(0)
    self_mask = torch.zeros(B, N, dtype=torch.bool, device=emb.device)
    self_mask[torch.arange(B), torch.arange(B)] = True
    sim = emb @ gal.t()
    rel = ((labels[:, None] == glab[None, :]) & ~self_mask)
    return sim, rel, self_mask


def smoothap_control_loss(emb, labels, tau_sigmoid: float, topk: int, mem_emb=None, mem_labels=None):
    """Differentiable AP control on identical inputs: Smooth-AP (Brown et al., ECCV 2020), ranks
    relaxed with sigmoid((s_j - s_i) / tau_sigmoid), over the batch plus memory bank."""
    sim, rel, self_mask = _bank_scores(emb, labels, mem_emb, mem_labels)
    valid = ~self_mask
    npos = rel.sum(-1)
    has = npos > 0
    sim, rel, valid, npos = sim[has], rel[has], valid[has], npos[has]
    P = int(npos.max().item()) if has.any() else 0
    if P == 0:
        return emb.sum() * 0, {'reward': 0.0, 'reward_mean_policy': 0.0}
    # positives of each query, padded to P (padding masked out)
    pos_idx = torch.where(rel, torch.arange(rel.size(1), device=sim.device)[None].expand_as(rel),
                          torch.full_like(rel, rel.size(1), dtype=torch.long)).sort(-1).values[:, :P]
    pos_ok = pos_idx < rel.size(1)
    pos_idx = pos_idx.clamp(max=rel.size(1) - 1)
    s_pos = torch.gather(sim, 1, pos_idx)                                     # (Q,P)
    d = torch.sigmoid((sim[:, None, :] - s_pos[:, :, None]) / tau_sigmoid)    # (Q,P,N)
    not_self = torch.ones_like(d, dtype=torch.bool).scatter_(2, pos_idx[:, :, None], False)
    rank_all = 1 + (d * (valid[:, None, :] & not_self)).sum(-1)
    rank_pos = 1 + (d * (rel[:, None, :] & not_self)).sum(-1)
    ap = ((rank_pos / rank_all) * pos_ok).sum(-1) / npos
    loss = (1 - ap).mean()
    with torch.no_grad():
        k = min(topk, sim.size(1) - 2)
        greedy = ap_at_k(torch.gather(rel.float(), -1, sim.masked_fill(~valid, -2).topk(k, -1).indices), npos)
    return loss, {'reward': greedy.mean().item(), 'reward_mean_policy': greedy.mean().item()}


def fastap_bank_loss(emb, labels, num_bins: int, topk: int, mem_emb=None, mem_labels=None):
    """FastAP (Cakir et al., CVPR 2019; the Stage 1 loss) over the batch plus memory bank:
    soft histogram of squared distances d = 2 - 2 s in [0, 4] with triangular kernels."""
    sim, rel, self_mask = _bank_scores(emb, labels, mem_emb, mem_labels)
    valid = (~self_mask).float()
    relf = rel.float()
    npos = relf.sum(-1)
    has = npos > 0
    d = (2 - 2 * sim).clamp(0, 4)
    delta = 4.0 / num_bins
    centers = torch.linspace(0, 4, num_bins + 1, device=emb.device)
    pulse = (1 - (d[..., None] - centers).abs() / delta).clamp(min=0)          # (B,N,L+1)
    h_pos = (pulse * (relf * valid)[..., None]).sum(1)                         # (B,L+1)
    h_all = (pulse * valid[..., None]).sum(1)
    H_pos, H_all = h_pos.cumsum(-1), h_all.cumsum(-1)
    ap = (H_pos * h_pos / H_all.clamp(min=1e-8)).sum(-1) / npos.clamp(min=1)
    loss = (1 - ap[has]).mean() if has.any() else emb.sum() * 0
    with torch.no_grad():
        k = min(topk, sim.size(1) - 2)
        greedy = ap_at_k(torch.gather(relf, -1, sim.masked_fill(self_mask, -2).topk(k, -1).indices), npos)[has]
    return loss, {'reward': greedy.mean().item(), 'reward_mean_policy': greedy.mean().item()}
