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

Options (Track V-D; defaults reproduce the earlier runs):
  --gld_extra tag=dir   more GLDv2 training shortlists of the same queries (e.g. over val pool + GLDv2
                        distractors, v_precompute_dist.py); training samples all pools, selection uses the
                        mean gld_val over pools
  --bench_extra tag=dir more benchmark shortlists (e.g. + R1M), reported as <bench>_<prot>_<tag>
  --feat v2             two similarity-scale-invariant features (all learned methods)
  --rl_init classifier  RL starts from the selected classifier (or value) policy of the same seed (pretrain,
                        then tune with the task reward, as in the original framework); --rl_lr sets its lr.
                        The init itself is scored first, so the selected RL policy is never below it on val
  --dagger              classifier/value also train on states visited by their own current policy
  --cosine              per-step cosine lr decay with 2% warm-up (all methods)
  --sup_iters N         update budget of the supervised baselines (default --iters), to match RL's
                        pretraining + tuning updates
"""
import argparse, json, math, pickle, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.verify_policy import AttnPolicy, Policy, VerifyEnv, query_tables, rollout   # noqa: E402


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


def load_gld(path, dev):
    """GLDv2 val-pool shortlists: full format (v_precompute.py) or compact (v_precompute_dist.py)."""
    z = np.load(path)
    qs, labels, top = z['queries'], z['labels'], z['top_idx']
    if 'rel_top' in z.files:
        d = {'sim': torch.tensor(z['sim_top'], device=dev), 'cand_sim': torch.tensor(z['cand_sim'], device=dev),
             'inliers': torch.tensor(z['inliers'], device=dev),
             'rel': torch.tensor(z['rel_top'].astype(np.int64), device=dev),
             'npos': torch.tensor(z['npos'], device=dev),
             'ap_tail': torch.tensor(z['ap_tail'], dtype=torch.float32, device=dev)}
        return d, qs, labels, top.shape[1], None
    rel_full = (labels[None, :] == labels[qs][:, None]).astype(np.int8)
    rel_full[np.arange(len(qs)), qs] = -1
    return build(z['sim_full'], top, z['inliers'], z['cand_sim'], rel_full, top.shape[1], dev), qs, labels, top.shape[1], (z, rel_full)


def cat_pools(ds):
    return {k: torch.cat([d[k] for d in ds]) for k in ds[0]}


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
    ap.add_argument('--bench_pre', default=None,
                    help='load benchmark shortlists from here instead of --pre (e.g. +R1M, v_precompute_r1m.py)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--budgets', type=int, nargs='+', default=[5, 10, 20])
    ap.add_argument('--threshold', type=int, default=35)
    ap.add_argument('--samples', type=int, default=16)
    ap.add_argument('--iters', type=int, default=1500)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--arch', choices=['mlp', 'attn'], default='mlp', help='policy/value network (same for all learned methods)')
    ap.add_argument('--hard_mix', type=float, default=0.0, help='weight of the hard-proxy AP in the training reward')
    ap.add_argument('--easy_inliers', type=int, default=150, help='shortlist positives with >= this many inliers count as easy')
    ap.add_argument('--gld_extra', nargs='*', default=[], help='tag=dir: more GLDv2 shortlists of the same queries')
    ap.add_argument('--bench_extra', nargs='*', default=[], help='tag=dir: more benchmark shortlists')
    ap.add_argument('--feat', choices=['v1', 'v2'], default='v1')
    ap.add_argument('--rl_init', choices=['none', 'classifier', 'value'], default='none')
    ap.add_argument('--dagger', action='store_true')
    ap.add_argument('--sup_iters', type=int, default=None)
    ap.add_argument('--cosine', action='store_true', help='per-step cosine lr decay (2%% warm-up) for every method')
    ap.add_argument('--rl_lr', type=float, default=None, help='RL learning rate (default --lr); e.g. lower for fine-tuning')
    ap.add_argument('--methods', nargs='+', default=['classifier', 'value', 'rl'])
    a = ap.parse_args()
    a.sup_iters = a.sup_iters or a.iters
    if a.rl_init != 'none' and a.rl_init not in a.methods:
        raise SystemExit('--rl_init needs that method in --methods (trained first)')
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pre, out = Path(a.pre), Path(a.out); out.mkdir(parents=True, exist_ok=True)

    # ---- GLDv2 val pool: relevance = same landmark; the query itself is junk
    gld, qs, labels, K, full = load_gld(pre / 'gldv2c_val.npz', dev)
    if a.hard_mix > 0:      # hard proxy: easy shortlist positives become junk (training reward only)
        z, rel_full = full
        top = z['top_idx']
        rel_h = rel_full.copy()
        easy = (z['inliers'] >= a.easy_inliers) & (rel_full[np.arange(len(qs))[:, None], top] == 1)
        for q in range(len(qs)):
            rel_h[q, top[q][easy[q]]] = -1
        gh = build(z['sim_full'], top, z['inliers'], z['cand_sim'], rel_h, K, dev)
        gld['rel_hard'], gld['npos_hard'], gld['ap_tail_hard'] = gh['rel'], gh['npos'], gh['ap_tail']
    lm = labels[qs]
    is_tr = (lm * 2654435761 % 3) != 0                            # landmark-level split, 2/3 train
    tr_idx = torch.tensor(np.where(is_tr)[0], device=dev); va_idx = torch.tensor(np.where(~is_tr)[0], device=dev)
    pools = {'gld_val': subset(gld, va_idx)}
    train_parts, val_keys = [subset(gld, tr_idx)], ['gld_val']
    for spec in a.gld_extra:
        tag, d = spec.split('=', 1)
        g2, qs2, _, K2, _ = load_gld(Path(d) / 'gldv2c_val.npz', dev)
        assert np.array_equal(qs2, qs) and K2 == K, f"{spec}: different queries or K"
        if a.hard_mix > 0:
            raise SystemExit('--hard_mix is not supported with --gld_extra')
        pools[f'gld_val_{tag}'] = subset(g2, va_idx); val_keys.append(f'gld_val_{tag}')
        train_parts.append(subset(g2, tr_idx))
    train_pool = cat_pools(train_parts) if len(train_parts) > 1 else train_parts[0]

    # ---- benchmarks
    bench_dirs = [('', Path(a.bench_pre or pre))] + [('_' + x.split('=', 1)[0], Path(x.split('=', 1)[1])) for x in a.bench_extra]
    for (suffix, bdir), name in [(b, n) for b in bench_dirs for n in ('roxford5k', 'rparis6k')]:
        zb = np.load(bdir / f'{name}.npz')
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
            pools[f'{name}_{prot}{suffix}'] = subset(d, torch.tensor(keep, device=dev))

    def sim_scorer(f):
        return f[..., 0] * 100
    results = {'K': K, 'threshold': a.threshold, 'pools': {k: int(v['sim'].shape[0]) for k, v in pools.items()},
               'selection': val_keys, 'args': vars(a)}
    E = lambda d, b, **kw: VerifyEnv(d, b, a.threshold, feat=a.feat, **kw)
    zero = lambda f: torch.zeros_like(f[..., 0])
    for B in a.budgets:
        envs = {k: E(v, B) for k, v in pools.items()}
        res = {}
        res['none'] = {k: evaluate(E(v, 0), sim_scorer, key=(B, 'none', k)) for k, v in pools.items()}
        res['top'] = {k: evaluate(e, sim_scorer, key=(B, 'top', k)) for k, e in envs.items()}
        res['random'] = {k: evaluate(e, zero, 'sample', reps=5) for k, e in envs.items()}
        res['all_K'] = {k: evaluate(E(v, K), sim_scorer) for k, v in pools.items()}
        trained = {}                                     # (method, seed) -> selected state
        for method in a.methods:
            per_seed, curves = [], []
            for seed in a.seeds:
                torch.manual_seed(seed)
                tr = E(train_pool, B, hard_mix=a.hard_mix)
                pol = (AttnPolicy(n_feat=tr.n_feat) if a.arch == 'attn' else Policy(n_feat=tr.n_feat)).to(dev)
                if method == 'rl' and a.rl_init != 'none':
                    pol.load_state_dict(trained[(a.rl_init, seed)])
                opt = torch.optim.Adam(pol.parameters(), lr=(a.rl_lr or a.lr) if method == 'rl' else a.lr)
                n_it = a.iters if method == 'rl' else a.sup_iters
                sched = None
                if a.cosine:
                    w = max(1, n_it // 50)
                    sched = torch.optim.lr_scheduler.LambdaLR(
                        opt, lambda t: min(1.0, (t + 1) / w) * 0.5 * (1 + math.cos(math.pi * min(t, n_it) / n_it)))
                score = lambda: float(np.mean([evaluate(envs[k], pol) for k in val_keys]))
                best, best_state, curve = score(), {k: x.clone() for k, x in pol.state_dict().items()}, []
                for it in range(n_it):
                    idx = torch.randint(0, tr.Q, (a.batch,), device=dev)
                    if method == 'rl':
                        rep = idx.repeat_interleave(a.samples)
                        r, logp, _ = rollout(tr, rep, pol, 'sample')
                        r = r.view(a.batch, a.samples)
                        adv = r - (r.sum(1, keepdim=True) - r) / (a.samples - 1)
                        loss = -(adv.detach().flatten() * logp).mean()
                    else:
                        # training states: top-similarity rollouts, random rollouts and (with --dagger) rollouts
                        # of the current policy itself, truncated after a random number of steps
                        with torch.no_grad():
                            kind = it % 3 if a.dagger else it % 2
                            sc, mode = [(sim_scorer, 'greedy'), (zero, 'sample'), (pol, 'sample')][kind]
                            sub = E(tr.d, int(torch.randint(0, B, (1,))))
                            _, _, ver = rollout(sub, idx, sc, mode)
                            if method == 'value':
                                target = one_step_gains(tr, idx, ver) * 100
                        fx = tr.features(idx, ver, tr.match[idx])
                        pred = pol(fx, tr.d['cand_sim'][idx].float()) if a.arch == 'attn' else pol(fx)
                        if method == 'value':
                            loss = F.mse_loss(pred[~ver], target[~ver])
                        else:
                            m = ~ver
                            loss = F.binary_cross_entropy_with_logits(pred[m], tr.match[idx][m].float())
                    opt.zero_grad(); loss.backward(); opt.step()
                    if sched is not None:
                        sched.step()
                    if (it + 1) % (100 if n_it <= 12000 else 400) == 0:
                        v = score(); curve.append(round(100 * v, 3))
                        if v > best:
                            best, best_state = v, {k: x.clone() for k, x in pol.state_dict().items()}
                pol.load_state_dict(best_state)
                trained[(method, seed)] = best_state
                torch.save(best_state, out / f'policy_B{B}_{method}_seed{seed}.pt')   # for later re-evaluation
                per_seed.append({k: evaluate(e, pol, key=(B, method, k)) for k, e in envs.items()})
                curves.append(curve)
            res[method] = {k: float(np.mean([s[k] for s in per_seed])) for k in envs}
            res[method]['selection'] = float(np.mean([res[method][k] for k in val_keys]))
            res[method + '_seeds'] = per_seed
            res[method + '_curves'] = curves
        results[f'B{B}'] = res
        print(f"B={B}: " + json.dumps({m: {k: round(100 * v, 2) for k, v in r.items()} for m, r in res.items()
                                       if not m.endswith(('_seeds', '_curves'))}), flush=True)
        (out / 'results.json').write_text(json.dumps(results, indent=1))
        np.savez(out / 'per_query.npz', **{f'B{b}|{m}|{p}': np.mean(v, 0) for (b, m, p), v in PER_QUERY.items()})


if __name__ == '__main__':
    main()
