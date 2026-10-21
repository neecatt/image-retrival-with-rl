"""
Track V-QE: budgeted geometric verification with query expansion.

Candidates of a query = its top-K shortlist (available from the start) + expansion candidates
(v_precompute_qe.py), which become available only when a verified match (inliers >= T) that lists
them among its k nearest neighbours has been verified. A policy verifies B candidates one at a time.
Final ranking: verified matches first (by inliers, descending), then every other gallery item in its
global order; the reward is the official AP of that ranking (trapezoid rule, junk ignored), computed
exactly from junk-free global ranks of the candidates and of all positives.

Unlike Track V (where one step's AP gain is a good guide, so greedy one-step policies are near
optimal), a check here can be worth more than its immediate AP gain: a match can unlock relevant
images that the global ranking put far below the shortlist.
"""
from typing import Dict

import torch


def final_ap(front, inl, rank, pos_ranks, npos):
    """AP of 'front first (by inliers), rest in global order'. All tensors batched over N queries.
    front (N,C) bool: verified matches that are not junk; inl (N,C) inliers; rank (N,C) junk-free global
    rank of each candidate (-1 for junk/padding); pos_ranks (N,P) sorted global ranks of the positives
    (-1 padding); npos (N,)."""
    N, C = front.shape
    f = front.sum(1)                                                             # (N,)
    key = torch.where(front, inl.float(), torch.full_like(inl, -1e9, dtype=torch.float32))
    order = torch.sort(key, dim=1, descending=True, stable=True).indices
    fpos = torch.empty_like(order)
    fpos.scatter_(1, order, torch.arange(C, device=front.device).expand(N, C))   # position within front
    valid_p = pos_ranks >= 0
    r_front = torch.where(front, rank, torch.full_like(rank, -2))                # (N,C), -2 if not in front
    in_front = (pos_ranks[:, :, None] == r_front[:, None, :]) & valid_p[:, :, None]          # (N,P,C)
    is_f = in_front.any(2)
    pos_fpos = (in_front.long() * fpos[:, None, :]).sum(2)
    below = ((r_front[:, None, :] >= 0) & (r_front[:, None, :] < pos_ranks[:, :, None])).sum(2)
    new_r = torch.where(is_f, pos_fpos, f[:, None] + pos_ranks - below)
    new_r = torch.where(valid_p, new_r, torch.full_like(new_r, 1 << 40))
    r = new_r.sort(1).values.float()
    t = torch.arange(r.shape[1], device=r.device, dtype=torch.float32).expand_as(r)
    p0 = torch.where(r > 0, t / r.clamp(min=1), torch.ones_like(r))
    p1 = (t + 1) / (r + 1)
    contrib = torch.where(valid_p.sort(1, descending=True).values, (p0 + p1) / 2, torch.zeros_like(r))
    return contrib.sum(1) / npos.clamp(min=1).float()


class QEEnv:
    """Batched environment over one pool. data keys (tensors on one device): valid (Q,C) bool,
    hop (Q,C), unlock (Q,C,C) bool, inliers (Q,C), sim (Q,C), cand_sim (Q,C,C), rank (Q,C),
    rel (Q,C), pos_ranks (Q,P), npos (Q,)."""
    n_feat = 17

    def __init__(self, data: Dict[str, torch.Tensor], budget: int, threshold: int = 35, c_norm: int = None):
        self.d, self.B, self.T = data, budget, threshold
        self.Q, self.C = data['valid'].shape
        self.c_norm = c_norm or self.C        # fixed position scale, so trimmed padding leaves features unchanged
        self.match = (data['inliers'] >= threshold) & data['valid']

    def available(self, idx, verified):
        vm = verified & self.match[idx]
        unlocked = (vm[:, :, None] & self.d['unlock'][idx]).any(1)
        return ((self.d['hop'][idx] == 0) | unlocked) & self.d['valid'][idx] & ~verified

    def reward(self, idx, verified):
        front = verified & self.match[idx] & (self.d['rank'][idx] >= 0)
        return final_ap(front, self.d['inliers'][idx], self.d['rank'][idx], self.d['pos_ranks'][idx], self.d['npos'][idx])

    def features(self, idx, verified, avail):
        d = self.d
        sim, C, inl = d['sim'][idx], d['cand_sim'][idx].float(), d['inliers'][idx].float()
        valid = d['valid'][idx]
        outcome = self.match[idx]
        posm, negm = (verified & outcome).float(), (verified & ~outcome).float()
        neg1 = torch.full_like(C, -1.0)
        max_pos = torch.where(posm[:, None, :] > 0, C, neg1).amax(-1)
        max_neg = torch.where(negm[:, None, :] > 0, C, neg1).amax(-1)
        mean_pos = (C * posm[:, None, :]).sum(-1) / posm.sum(-1, keepdim=True).clamp(min=1)
        w = posm * torch.log1p(inl.clamp(min=0))
        wpos = (C * w[:, None, :]).sum(-1) / w.sum(-1, keepdim=True).clamp(min=1e-6)
        used = verified.float().sum(-1, keepdim=True) / self.B
        frac_m = posm.sum(-1, keepdim=True) / verified.float().sum(-1, keepdim=True).clamp(min=1)
        vf = valid.float()
        mu = (sim * vf).sum(-1, keepdim=True) / vf.sum(-1, keepdim=True)
        sd = (((sim - mu) ** 2) * vf).sum(-1, keepdim=True).div(vf.sum(-1, keepdim=True)).sqrt().clamp(min=1e-4)
        top1 = sim[:, :1]
        hop = d['hop'][idx].float().clamp(min=0)
        # expansion structure: candidates this one would unlock that are not available yet, and how
        # similar its best not-yet-available child is to the verified matches
        unl = d['unlock'][idx]
        hidden = valid & ~avail & ~verified
        n_children = (unl & hidden[:, None, :]).float().sum(-1)
        n_avail = avail.float().sum(-1, keepdim=True)
        rank_in = torch.arange(self.C, device=sim.device).float()[None].expand_as(sim) / self.c_norm
        f = torch.stack([sim, rank_in, sim - top1, max_pos, mean_pos, max_neg, wpos, used.expand_as(sim),
                         frac_m.expand_as(sim), (posm.sum(-1, keepdim=True) > 0).float().expand_as(sim),
                         (sim - mu) / sd, hop, (hop > 0).float(), torch.log1p(n_children),
                         avail.float(), torch.log1p(n_avail).expand_as(sim) / 5,
                         (1 - used).expand_as(sim)], -1)
        return f


def score(env, scorer, f, idx):
    if getattr(scorer, 'uses_sim', False):
        return scorer(f, env.d['cand_sim'][idx].float(), env.d['valid'][idx])
    return scorer(f)


def rollout(env: QEEnv, idx, scorer, mode='sample', temperature=1.0, record=None):
    """B verification steps; only available candidates can be chosen. Returns AP, sum log-prob, verified.
    record: optional list that receives (verified, available, action, has) per step, for replaying the
    trajectory with gradients one step at a time (policy_gradient_backward)."""
    N, C = idx.numel(), env.C
    dev = env.d['sim'].device
    verified = torch.zeros(N, C, dtype=torch.bool, device=dev)
    logp = torch.zeros(N, device=dev)
    for _ in range(env.B):
        avail = env.available(idx, verified)
        if not avail.any():
            break
        f = env.features(idx, verified, avail)
        logits = score(env, scorer, f, idx) / temperature
        logits = logits.masked_fill(~avail, float('-inf'))
        has = avail.any(1)
        logits = torch.where(has[:, None], logits, torch.zeros_like(logits))        # queries with nothing left
        if mode == 'sample':
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()
            logp = logp + torch.where(has, dist.log_prob(a), torch.zeros_like(logp))
        else:
            a = logits.argmax(-1)
        if record is not None:
            record.append((verified, avail, a, has))
        step = torch.zeros_like(verified)
        step[torch.arange(N, device=dev), a] = True
        verified = verified | (step & has[:, None])
    return env.reward(idx, verified), logp, verified


def policy_gradient_backward(env: QEEnv, idx, scorer, adv, record, chunk=1024):
    """Backpropagate -mean(adv * sum_t log pi(a_t | s_t)) by replaying a recorded trajectory one step
    (and one chunk of rollouts) at a time: the same gradient as differentiating the whole rollout, with
    memory independent of the budget. Returns the loss value."""
    total, N = 0.0, idx.numel()
    for verified, avail, a, has in record:
        for s in range(0, N, chunk):
            sl = slice(s, min(s + chunk, N))
            f = env.features(idx[sl], verified[sl], avail[sl])
            logits = score(env, scorer, f, idx[sl]).masked_fill(~avail[sl], float('-inf'))
            logits = torch.where(has[sl, None], logits, torch.zeros_like(logits))
            lp = torch.distributions.Categorical(logits=logits).log_prob(a[sl])
            lp = torch.where(has[sl], lp, torch.zeros_like(lp))
            loss = -(adv[sl] * lp).sum() / N
            loss.backward()
            total += loss.detach().item()
    return total
