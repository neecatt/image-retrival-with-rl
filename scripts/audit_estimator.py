#!/usr/bin/env python3
"""CPU audit of the REINFORCE estimator in training/rl_trainer.py on a tractable problem.

Policy: a ~ normalize(mu + sigma*eps), mu = normalize(w) (same sampler as RLTrainer._sample_on_sphere).
Reward: smooth r(a) = exp(kappa * a.t). Ground truth gradient of J(w)=E[r(a)] comes from the pathwise
(reparameterised) derivative, which is valid because r and the sampler are differentiable.
Compared: (A) estimator exactly as coded (sampled action NOT detached inside log_prob),
          (B) same but sampled action detached (textbook score-function with vMF-like density),
          each with the independent-second-sample baseline used by RLTrainer and with no baseline.
"""
import sys, torch, torch.nn.functional as F
torch.manual_seed(0)
dt = torch.float64
def run(d, sigma, kappa=5.0, N=2_000_000, chunk=200_000):
    t = F.normalize(torch.randn(d, dtype=dt), dim=0)
    w0 = torch.randn(d, dtype=dt)
    def reward(a): return torch.exp(kappa * (a @ t))
    def draw(w, n):
        mu = F.normalize(w, dim=0)
        eps = torch.randn(n, d, dtype=dt)
        return mu, F.normalize(mu + sigma * eps, dim=1)
    grads = {k: torch.zeros(d, dtype=dt) for k in ('truth', 'A_base', 'B_base', 'A_nobase', 'B_nobase')}
    n_done = 0
    for _ in range(N // chunk):
        w = w0.clone().requires_grad_(True)
        # ground truth: pathwise
        mu, a = draw(w, chunk)
        g, = torch.autograd.grad(reward(a).mean(), w)
        grads['truth'] += g * chunk
        for name, detach, base in (('A_base', False, True), ('B_base', True, True),
                                   ('A_nobase', False, False), ('B_nobase', True, False)):
            w = w0.clone().requires_grad_(True)
            mu, a = draw(w, chunk)
            s = a.detach() if detach else a
            logp = (s * mu).sum(1) / sigma**2           # as in _sample_on_sphere
            r = reward(a).detach()
            if base:
                _, a2 = draw(w0, chunk)                  # independent second sample, no grad
                r = r - reward(a2)
            loss = -(r * logp).mean()
            g, = torch.autograd.grad(loss, w)
            grads[name] += (-g) * chunk                  # ascent direction on J
    truth = grads.pop('truth') / N
    print(f"d={d} sigma={sigma} sigma*sqrt(d)={sigma*d**0.5:.3f} |true grad|={truth.norm():.4f}")
    for k, g in grads.items():
        g = g / N
        print(f"  {k:9s} cos(est,true)={F.cosine_similarity(g, truth, dim=0):+.3f}  |est|/|true|={g.norm()/truth.norm():.3f}")
for d, s in ((16, 0.1), (16, 0.05), (64, 0.05), (512, 0.01)):
    run(d, s, N=1_000_000 if d == 512 else 2_000_000)
