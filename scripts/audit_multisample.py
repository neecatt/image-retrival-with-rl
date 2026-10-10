#!/usr/bin/env python3
"""Phase 1.1 audit: multi-sample product-rollout estimator vs the exact gradient (CPU).

Synthetic retrieval problem: learnable query/gallery embedding tables, labels from a few classes.
Reward = smooth AP (sigmoid ranks), so the exact gradient of J = E[R] is available pathwise.
Compared estimators (all use reward values only, like REINFORCE):
  old_1+1    : current RLTrainer path (1 query sample, independent baseline sample, gallery not in policy)
  S=k q-only : product rollouts, leave-one-out baseline, gallery sampled but not trained
  S=k both   : product rollouts, query and gallery both trained
Reports, per estimator, the cosine of a single estimate with the truth (median over repeats) and the
cosine/relative error of the average of all repeats (bias check).
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.rl_estimators import multi_sample_policy_loss, sample_sphere  # noqa: E402

torch.manual_seed(0)
DT = torch.float64


def smooth_ap(sim, rel, tau=0.05):
    """Differentiable AP: rank via sigmoids. sim, rel: (..., N)."""
    diff = (sim.unsqueeze(-2) - sim.unsqueeze(-1)) / tau          # [..., i, j] = s_j - s_i
    sg = torch.sigmoid(diff)
    eye = torch.eye(sim.size(-1), dtype=sim.dtype)
    sg = sg * (1 - eye)
    rank_all = 1 + sg.sum(-1)
    rank_pos = 1 + (sg * rel.unsqueeze(-2)).sum(-1)
    npos = rel.sum(-1)
    return (rel * rank_pos / rank_all).sum(-1) / npos.clamp(min=1)


def setup(nq=8, ng=12, d=16, classes=4):
    ql = torch.arange(nq) % classes
    gl = torch.arange(ng) % classes
    centers = F.normalize(torch.randn(classes, d, dtype=DT), dim=1)
    Wq = (centers[ql] + 0.9 * torch.randn(nq, d, dtype=DT))
    Wg = (centers[gl] + 0.9 * torch.randn(ng, d, dtype=DT))
    return Wq, Wg, ql, gl


def truth_grad(Wq, Wg, ql, gl, sigma, n=40_000, chunk=4_000):
    gq = torch.zeros_like(Wq); gg = torch.zeros_like(Wg)
    rel = (ql[:, None] == gl[None, :]).to(DT)
    for _ in range(n // chunk):
        wq = Wq.clone().requires_grad_(True); wg = Wg.clone().requires_grad_(True)
        qs = F.normalize(F.normalize(wq, dim=1)[None] + sigma * torch.randn(chunk, *Wq.shape, dtype=DT), dim=-1)
        gs = F.normalize(F.normalize(wg, dim=1)[None] + sigma * torch.randn(chunk, *Wg.shape, dtype=DT), dim=-1)
        sim = torch.einsum('cqd,cgd->cqg', qs, gs)
        r = smooth_ap(sim, rel.expand(chunk, -1, -1)).mean()
        a, b = torch.autograd.grad(r, (wq, wg))
        gq += a; gg += b
    k = n // chunk
    return torch.cat([(gq / k).flatten(), (gg / k).flatten()])


def est_old(Wq, Wg, ql, gl, sigma):
    """Current RLTrainer estimator: 1 query sample, 1 independent baseline sample, query-only gradient."""
    wq = Wq.clone().requires_grad_(True); wg = Wg.clone().requires_grad_(True)
    mq, mg = F.normalize(wq, dim=1), F.normalize(wg, dim=1)
    rel = (ql[:, None] == gl[None, :]).to(DT)
    q1, lp = sample_sphere(mq, sigma, 1)
    g1, _ = sample_sphere(mg, sigma, 1)
    with torch.no_grad():
        q2, _ = sample_sphere(mq, sigma, 1); g2, _ = sample_sphere(mg, sigma, 1)
        adv = smooth_ap(q1[0] @ g1[0].t(), rel) - smooth_ap(q2[0] @ g2[0].t(), rel)
    loss = -(adv * lp[0]).mean()
    a, b = torch.autograd.grad(loss, (wq, wg), allow_unused=True)
    b = torch.zeros_like(Wg) if b is None else b
    return -torch.cat([a.flatten(), b.flatten()])


def est_multi(Wq, Wg, ql, gl, sigma, S, both):
    wq = Wq.clone().requires_grad_(True); wg = Wg.clone().requires_grad_(True)
    loss, _ = multi_sample_policy_loss(F.normalize(wq, dim=1), F.normalize(wg, dim=1), ql, gl,
                                       sigma, S, sample_gallery=True, reward_fn=smooth_ap)
    if not both:  # gallery sampled but not trained: drop its gradient
        a, = torch.autograd.grad(loss, (wq,), retain_graph=False)
        b = torch.zeros_like(Wg)
    else:
        a, b = torch.autograd.grad(loss, (wq, wg))
    return -torch.cat([a.flatten(), b.flatten()])


def summarise(name, ests, truth, n_q):
    E = torch.stack(ests)
    per = F.cosine_similarity(E, truth[None], dim=1)
    mean = E.mean(0)
    # Query-part cosine as well, since old estimator has no gallery gradient
    per_q = F.cosine_similarity(E[:, :n_q], truth[None, :n_q], dim=1)
    print(f"  {name:14s} single-estimate cos: all {per.median():+.3f} | query part {per_q.median():+.3f}"
          f"   mean-of-{len(ests)} cos {F.cosine_similarity(mean, truth, dim=0):+.3f}"
          f"  |mean|/|truth| {mean.norm() / truth.norm():.2f}")


def main():
    Wq, Wg, ql, gl = setup()
    n_q = Wq.numel()
    for sigma in (0.05, 0.15):
        truth = truth_grad(Wq, Wg, ql, gl, sigma)
        print(f"sigma={sigma}  |true grad|={truth.norm():.4f}  (query part {truth[:n_q].norm():.4f}, gallery part {truth[n_q:].norm():.4f})")
        reps = 300
        summarise('old_1+1', [est_old(Wq, Wg, ql, gl, sigma) for _ in range(reps)], truth, n_q)
        for S in (4, 16, 64):
            summarise(f'S={S} q-only', [est_multi(Wq, Wg, ql, gl, sigma, S, False) for _ in range(reps // 3)], truth, n_q)
            summarise(f'S={S} both', [est_multi(Wq, Wg, ql, gl, sigma, S, True) for _ in range(reps // 3)], truth, n_q)


if __name__ == '__main__':
    main()
