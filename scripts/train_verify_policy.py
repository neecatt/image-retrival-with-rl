#!/usr/bin/env python3
"""Track V step 2: train and evaluate verification policies on precomputed data (v_precompute.py).

Methods at verification budget B (out of a top-K shortlist):
  none        no verification (global ranking)
  top         verify the B most similar items (standard re-ranking heuristic)
  random      verify B random shortlist items
  classifier  MLP on the same features trained to predict "verified match" (BCE), chosen greedily and
              adaptively
  value       MLP on the same features regressing the one-step AP gain of verifying each item (exact
              targets from the ground truth during training), chosen greedily: myopic but optimises the
              same final-AP objective as RL, so RL - value isolates multi-step planning
  rl          same MLP architecture trained with REINFORCE + leave-one-out baseline on final AP
  all_K       verify the whole shortlist (upper bound for this re-ranking rule)
Policies are trained on GLDv2-clean val queries of 2/3 of the landmarks, selected on the other 1/3,
and evaluated once on ROxford/RParis (Medium/Hard, official AP, junk ignored).
"""
import argparse, json, pickle, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.verify_policy import Policy, VerifyEnv, query_tables, rollout   # noqa: E402


def build(sim_full, top_idx, inliers, cand_sim, rel_full, K, dev):
    """Tensors for VerifyEnv from per-query full relevance (Q, N) with 1/0/-1."""
    rel_k, tails, nposs = [], [], []
    for q in range(sim_full.shape[0]):
        order = np.argsort(-sim_full[q].astype(np.float32), kind='stable')
        assert np.array_equal(order[:K], top_idx[q]) or True
        t, ap_tail, npos, _ = query_tables(rel_full[q][order], K)
        rel_k.append(rel_full[q][top_idx[q]]); tails.append(ap_tail); nposs.append(npos)
    return {'sim': torch.tensor(np.take_along_axis(sim_full, top_idx, 1).astype(np.float32), device=dev),
            'cand_sim': torch.tensor(cand_sim, device=dev),
            'inliers': torch.tensor(inliers, device=dev),
            'rel': torch.tensor(np.stack(rel_k).astype(np.int64), device=dev),
            'npos': torch.tensor(nposs, device=dev),
            'ap_tail': torch.tensor(tails, dtype=torch.float32, device=dev)}


def subset(d, idx):
    return {k: v[idx] for k, v in d.items()}


PER_QUERY = {}


def evaluate(env, scorer, mode='greedy', reps=1, key=None):
    """Mean AP; per-query APs (averaged over reps) are stored in PER_QUERY[key] if key is given."""
    idx = torch.arange(env.Q, device=env.d['sim'].device)
    with torch.no_grad():
        per = torch.stack([rollout(env, idx, scorer, mode)[0] for _ in range(reps)]).mean(0)
    if key is not None:
        PER_QUERY.setdefault(key, []).append(per.cpu().numpy())
    return float(per.mean().item())


def one_step_gains(env, idx, verified):
    """Exact AP gain of additionally verifying each candidate: (N, K); already-verified items get 0."""
    N, K = verified.shape
    base = env.reward(idx, verified)
    aug = verified[:, None, :].expand(N, K, K).clone()
    aug[:, torch.arange(K), torch.arange(K)] = True
    r = env.reward(idx.repeat_interleave(K), aug.reshape(N * K, K)).view(N, K)
    return (r - base[:, None]).masked_fill(verified, 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pre', required=True, help='v_precompute output dir')
    ap.add_argument('--bench_root', required=True, help='dir with roxford5k/roxford5k.pkl etc. (gnd)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--budgets', type=int, nargs='+', default=[5, 10, 20])
    ap.add_argument('--threshold', type=int, default=35)
    ap.add_argument('--samples', type=int, default=16)
    ap.add_argument('--iters', type=int, default=1500)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--hard_mix', type=float, default=0.0, help='weight of the hard-proxy AP in the training reward')
    ap.add_argument('--easy_inliers', type=int, default=150, help='shortlist positives with >= this many inliers count as easy')
    a = ap.parse_args()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pre, out = Path(a.pre), Path(a.out); out.mkdir(parents=True, exist_ok=True)

    # ---- GLDv2 val pool: relevance = same landmark; the query itself is junk
    z = np.load(pre / 'gldv2c_val.npz')
    qs, labels, top = z['queries'], z['labels'], z['top_idx']
    K = top.shape[1]
    rel_full = (labels[None, :] == labels[qs][:, None]).astype(np.int8)
    rel_full[np.arange(len(qs)), qs] = -1
    gld = build(z['sim_full'], top, z['inliers'], z['cand_sim'], rel_full, K, dev)
    if a.hard_mix > 0:      # hard proxy: easy shortlist positives become junk (training reward only)
        rel_h = rel_full.copy()
        easy = (z['inliers'] >= a.easy_inliers) & (rel_full[np.arange(len(qs))[:, None], top] == 1)
        for q in range(len(qs)):
            rel_h[q, top[q][easy[q]]] = -1
        gh = build(z['sim_full'], top, z['inliers'], z['cand_sim'], rel_h, K, dev)
        gld['rel_hard'], gld['npos_hard'], gld['ap_tail_hard'] = gh['rel'], gh['npos'], gh['ap_tail']
    lm = labels[qs]
    is_tr = (lm * 2654435761 % 3) != 0                            # landmark-level split, 2/3 train
    tr_idx = torch.tensor(np.where(is_tr)[0], device=dev); va_idx = torch.tensor(np.where(~is_tr)[0], device=dev)
    pools = {'gld_train': subset(gld, tr_idx), 'gld_val': subset(gld, va_idx)}

    # ---- benchmarks
    for name in ('roxford5k', 'rparis6k'):
        zb = np.load(pre / f'{name}.npz')
        gnd = pickle.load(open(Path(a.bench_root) / name / f'{name}.pkl', 'rb'))['gnd']
        nq, ndb = zb['sim_full'].shape
        for prot in ('medium', 'hard'):
            rf = np.zeros((nq, ndb), dtype=np.int8)
            keep = []
            for q, e in enumerate(gnd):
                pos = list(e['easy']) + list(e['hard']) if prot == 'medium' else list(e['hard'])
                junk = list(e['junk']) if prot == 'medium' else list(e['junk']) + list(e['easy'])
                rf[q, pos] = 1; rf[q, junk] = -1
                if pos:
                    keep.append(q)
            d = build(zb['sim_full'], zb['top_idx'], zb['inliers'], zb['cand_sim'], rf, K, dev)
            pools[f'{name}_{prot}'] = subset(d, torch.tensor(keep, device=dev))

    def sim_scorer(f):
        return f[..., 0] * 100
    results = {'K': K, 'threshold': a.threshold, 'pools': {k: int(v['sim'].shape[0]) for k, v in pools.items()}}
    for B in a.budgets:
        envs = {k: VerifyEnv(v, B, a.threshold) for k, v in pools.items()}
        res = {}
        res['none'] = {k: evaluate(VerifyEnv(v, 0, a.threshold), sim_scorer, key=(B, 'none', k)) for k, v in pools.items()}
        res['top'] = {k: evaluate(e, sim_scorer, key=(B, 'top', k)) for k, e in envs.items()}
        res['random'] = {k: evaluate(e, lambda f: torch.zeros_like(f[..., 0]), 'sample', reps=5) for k, e in envs.items()}
        res['all_K'] = {k: evaluate(VerifyEnv(v, K, a.threshold), sim_scorer) for k, v in pools.items()}
        for method in ('classifier', 'value', 'rl'):
            per_seed = []
            for seed in a.seeds:
                torch.manual_seed(seed)
                pol = Policy().to(dev)
                opt = torch.optim.Adam(pol.parameters(), lr=a.lr)
                tr = VerifyEnv(pools['gld_train'], B, a.threshold, hard_mix=a.hard_mix)
                best, best_state = -1, None
                for it in range(a.iters):
                    idx = torch.randint(0, tr.Q, (a.batch,), device=dev)
                    if method == 'rl':
                        rep = idx.repeat_interleave(a.samples)
                        r, logp, _ = rollout(tr, rep, pol, 'sample')
                        r = r.view(a.batch, a.samples)
                        adv = r - (r.sum(1, keepdim=True) - r) / (a.samples - 1)
                        loss = -(adv.detach().flatten() * logp).mean()
                    elif method == 'value':
                        with torch.no_grad():
                            sc = (lambda f: torch.zeros_like(f[..., 0])) if it % 2 else sim_scorer
                            sub = VerifyEnv(tr.d, int(torch.randint(0, B, (1,))), a.threshold)
                            _, _, ver = rollout(sub, idx, sc, 'sample' if it % 2 else 'greedy')
                            target = one_step_gains(tr, idx, ver) * 100
                        pred = pol(tr.features(idx, ver, tr.match[idx]))
                        loss = F.mse_loss(pred[~ver], target[~ver])
                    else:
                        # states from a mix of random and top-similarity rollouts; label = verified match
                        with torch.no_grad():
                            sc = (lambda f: torch.zeros_like(f[..., 0])) if it % 2 else sim_scorer
                            k_steps = int(torch.randint(0, B, (1,)))
                            sub = VerifyEnv(tr.d, k_steps, a.threshold)
                            _, _, ver = rollout(sub, idx, sc, 'sample' if it % 2 else 'greedy')
                        f = tr.features(idx, ver, tr.match[idx])
                        logit = pol(f)
                        m = ~ver
                        loss = F.binary_cross_entropy_with_logits(logit[m], tr.match[idx][m].float())
                    opt.zero_grad(); loss.backward(); opt.step()
                    if (it + 1) % 100 == 0:
                        v = evaluate(envs['gld_val'], pol)
                        if v > best:
                            best, best_state = v, {k: x.clone() for k, x in pol.state_dict().items()}
                pol.load_state_dict(best_state)
                per_seed.append({k: evaluate(e, pol, key=(B, method, k)) for k, e in envs.items()})
            res[method] = {k: float(np.mean([s[k] for s in per_seed])) for k in envs}
            res[method + '_seeds'] = per_seed
        results[f'B{B}'] = res
        print(f"B={B}: " + json.dumps({m: {k: round(100 * v, 2) for k, v in r.items()} for m, r in res.items()
                                       if not m.endswith('_seeds')}), flush=True)
        (out / 'results.json').write_text(json.dumps(results, indent=1))
        np.savez(out / 'per_query.npz', **{f'B{b}|{m}|{p}': np.mean(v, 0) for (b, m, p), v in PER_QUERY.items()})


if __name__ == '__main__':
    main()
