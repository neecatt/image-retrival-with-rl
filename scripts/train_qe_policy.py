#!/usr/bin/env python3
"""Track V-QE: train and evaluate verification policies with query expansion (training/qe_env.py).

Methods at budget B (candidates = top-K shortlist + expansion candidates unlocked by verified matches):
  none        no verification
  top         verify the most similar available candidate (standard re-ranking order; expansion
              candidates enter as soon as they are unlocked)
  shortlist   verify the B most similar shortlist items and never expand (Track V's top-B)
  random      random available candidate
  classifier  attention network trained to predict "verified match" (BCE), greedy
  value       same network regressing the exact one-step AP gain of verifying each available
              candidate, greedy (myopic: a check that unlocks relevant images gains nothing now)
  rl          same network, REINFORCE with leave-one-out baseline on the final AP
All learned methods see the same features (including how many hidden candidates a check would
unlock). Policies train on GLDv2 val queries of 2/3 of the landmarks (plain + distractor pools),
are selected on the other 1/3, and are evaluated once on ROxford/RParis (+R1M), Medium and Hard.
"""
import argparse, json, math, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.qe_env import QEEnv, policy_gradient_backward, rollout, score                          # noqa: E402
from training.verify_policy import AttnPolicy                              # noqa: E402


def load_pool(path, dev, prot=None):
    z = np.load(path)
    sfx = f'_{prot}' if prot else ''
    cand = z['cand']
    d = {'valid': torch.tensor(cand >= 0), 'hop': torch.tensor(z['hop'].astype(np.int64)),
         'unlock': torch.tensor(z['unlock']), 'inliers': torch.tensor(z['inliers'].astype(np.int64)),
         'sim': torch.tensor(z['sim']), 'cand_sim': torch.tensor(z['cand_sim']),
         'rank': torch.tensor(z['rank' + sfx]), 'rel': torch.tensor(z['rel' + sfx].astype(np.int64)),
         'pos_ranks': torch.tensor(z['pos_ranks' + sfx]), 'npos': torch.tensor(z['npos' + sfx])}
    keep = None
    if prot:                                                   # queries without positives are skipped
        keep = torch.nonzero(d['npos'] > 0).flatten()
        d = {k: v[keep] for k, v in d.items()}
    return {k: v.to(dev) for k, v in d.items()}, z


def subset(d, idx):
    return {k: v[idx] for k, v in d.items()}


def cat_pools(ds):
    P = max(d['pos_ranks'].shape[1] for d in ds)
    out = {}
    for k in ds[0]:
        xs = [d[k] for d in ds]
        if k == 'pos_ranks':
            xs = [F.pad(x, (0, P - x.shape[1]), value=-1) for x in xs]
        if k in ('rq_s', 'rq_rel', 'rq_S'):                      # re-query subsets differ in size: pad X
            X = max(x.shape[-1] for x in xs)
            fill = {'rq_s': float('-inf'), 'rq_rel': -1, 'rq_S': 0}[k]
            xs = [F.pad(x, (0, X - x.shape[-1]), value=fill) for x in xs]
        out[k] = torch.cat(xs)
    return out


PER_QUERY = {}
CANDIDATE_DIMS = {'valid': 1, 'hop': 1, 'inliers': 1, 'sim': 1, 'rank': 1, 'rel': 1, 'unlock': 2, 'cand_sim': 2}


def trim(d, idx):
    """Queries idx with candidate padding cut to the longest candidate list among them (candidates are
    stored first, padding last) and positive padding cut likewise. Saves the C^2 cost of padding."""
    sub = {k: v[idx] for k, v in d.items()}
    c = int(sub['valid'].sum(1).max())
    p = max(int(sub['npos'].max()), 1)
    for k, nd in CANDIDATE_DIMS.items():
        sub[k] = sub[k][:, :c] if nd == 1 else sub[k][:, :c, :c]
    sub['pos_ranks'] = sub['pos_ranks'][:, :p]
    if 'rq_S' in sub:                                            # re-query data: candidate dimension only
        sub['rq_S'] = sub['rq_S'][:, :c]; sub['rq_cpos'] = sub['rq_cpos'][:, :c]
    return sub


def sub_env(env, idx, budget=None):
    return QEEnv(trim(env.d, idx), env.B if budget is None else budget, env.T, c_norm=env.c_norm)


def evaluate(env, scorer, mode='greedy', reps=1, key=None, chunk=512):
    """Mean AP over the pool; queries are processed in chunks of similar candidate count."""
    dev = env.d['sim'].device
    order = torch.argsort(env.d['valid'].sum(1))
    with torch.no_grad():
        per = []
        for _ in range(reps):
            r = torch.empty(env.Q, device=dev)
            for s in range(0, env.Q, chunk):
                ids = order[s:s + chunk]
                se = sub_env(env, ids)
                r[ids] = rollout(se, torch.arange(len(ids), device=dev), scorer, mode)[0]
            per.append(r)
        per = torch.stack(per).mean(0)
    if key is not None:
        PER_QUERY.setdefault(key, []).append(per.cpu().numpy())
    return float(per.mean())


def one_step_gains(env, idx, verified, avail):
    """Exact AP gain of additionally verifying each available candidate (N, C); 0 elsewhere."""
    N, C = verified.shape
    base = env.reward(idx, verified)
    aug = verified[:, None, :].expand(N, C, C).clone()
    aug[:, torch.arange(C), torch.arange(C)] = True
    r = env.reward(idx.repeat_interleave(C), aug.reshape(N * C, C)).view(N, C)
    return (r - base[:, None]).masked_fill(~avail, 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pre', required=True, help='v_precompute_qe.py output dir')
    ap.add_argument('--out', required=True)
    ap.add_argument('--train_pools', nargs='+', default=['gldv2c_val', 'gldv2c_val_dist'])
    ap.add_argument('--bench_pools', nargs='*', default=['roxford5k', 'rparis6k', 'roxford5k_r1m', 'rparis6k_r1m'])
    ap.add_argument('--requery_reward', action='store_true',
                    help='train and select with the verified-QE re-query AP (training/qe_requery.py); needs '
                         '--data_root (staged gldv2c), --model and --dist_emb')
    ap.add_argument('--data_root'); ap.add_argument('--model'); ap.add_argument('--dist_emb')
    ap.add_argument('--requery_M', type=int, default=1000)
    ap.add_argument('--budgets', type=int, nargs='+', default=[10, 20, 40])
    ap.add_argument('--threshold', type=int, default=35)
    ap.add_argument('--methods', nargs='+', default=['classifier', 'value', 'rl'])
    ap.add_argument('--iters', type=int, default=6000, help='RL updates')
    ap.add_argument('--sup_iters', type=int, default=12000, help='classifier / value updates')
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--samples', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--rl_lr', type=float, default=None)
    ap.add_argument('--rl_init', choices=['none', 'classifier', 'value'], default='none')
    ap.add_argument('--dagger', action='store_true')
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--eval_every', type=int, default=200)
    ap.add_argument('--expand_weight', type=float, default=0.0,
                    help='fraction of training batches drawn from queries whose expansion candidates include at '
                         'least --expand_min relevant images (same for every method; validation is unchanged)')
    ap.add_argument('--expand_min', type=int, default=3)
    a = ap.parse_args()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    pre = Path(a.pre)

    pools, train_parts, val_keys = {}, [], []
    if a.requery_reward:
        from data.dataset import get_transforms
        from data.dataset_sfm120k import load_gldv2c
        from evaluate import create_model_from_checkpoint
        from evaluation.sfm_val import extract_embeddings
        from training.qe_requery import build as build_requery
        net = create_model_from_checkpoint(a.model, dev).eval()
        _, val = load_gldv2c(str(Path(a.data_root) / 'gldv2c'), val_transform=get_transforms(512, 'eval'))
        with torch.no_grad():
            emb, _ = extract_embeddings(net, val, dev, batch_size=64, num_workers=16)
        emb = F.normalize(emb.float(), dim=1); del net
    for name in a.train_pools:
        d, z = load_pool(pre / f'{name}.npz', dev)
        if a.requery_reward:
            assert np.array_equal(np.asarray(val.labels), z['labels'])
            G = torch.cat([emb, F.normalize(torch.from_numpy(np.load(a.dist_emb).astype(np.float32)).to(dev), dim=1)]) \
                if name.endswith('dist') else emb
            d.update(build_requery(G, z['queries'], z['cand'], z['labels'], len(z['labels']), a.requery_M))
            print(name, 're-query subsets: X =', d['rq_s'].shape[1], flush=True)
            del G; torch.cuda.empty_cache()
        lm = z['labels'][z['queries']]
        is_tr = (lm * 2654435761 % 3) != 0                    # landmark-level split, as in Track V
        tr = torch.tensor(np.where(is_tr)[0], device=dev); va = torch.tensor(np.where(~is_tr)[0], device=dev)
        pools[name] = subset(d, va); val_keys.append(name); train_parts.append(subset(d, tr))
    train_pool = cat_pools(train_parts) if len(train_parts) > 1 else train_parts[0]
    for name in a.bench_pools:
        for prot in ('medium', 'hard'):
            pools[f'{name}_{prot}'], _ = load_pool(pre / f'{name}.npz', dev, prot)

    sim_scorer = lambda f: f[..., 0] * 100
    zero = lambda f: torch.zeros_like(f[..., 0])

    class ShortlistOnly:                                      # top-B of the shortlist, never expands
        def __call__(self, f):
            return torch.where(f[..., 11] > 0, torch.full_like(f[..., 0], -1e4), f[..., 0] * 100)

    results = {'threshold': a.threshold, 'pools': {k: int(v['sim'].shape[0]) for k, v in pools.items()},
               'selection': val_keys, 'args': vars(a)}
    for B in a.budgets:
        envs = {k: QEEnv(v, B, a.threshold) for k, v in pools.items()}
        res = {'none': {k: evaluate(QEEnv(v, 0, a.threshold), sim_scorer, key=(B, 'none', k)) for k, v in pools.items()},
               'top': {k: evaluate(e, sim_scorer, key=(B, 'top', k)) for k, e in envs.items()},
               'shortlist': {k: evaluate(e, ShortlistOnly(), key=(B, 'shortlist', k)) for k, e in envs.items()},
               'random': {k: evaluate(e, zero, 'sample', reps=3) for k, e in envs.items()}}
        for m in ('top', 'shortlist', 'none'):
            res[m]['selection'] = float(np.mean([res[m][k] for k in val_keys]))
        trained = {}
        for method in a.methods:
            per_seed, curves = [], []
            for seed in a.seeds:
                torch.manual_seed(seed)
                tr = QEEnv(train_pool, B, a.threshold)
                # batches are drawn from one of 8 equal-size groups of queries with similar candidate
                # counts (uniform over queries overall), so padding can be trimmed per batch
                groups = torch.argsort(tr.d['valid'].sum(1)).tensor_split(8)
                useful = torch.nonzero(((tr.d['rel'] == 1) & (tr.d['hop'] > 0)).sum(1) >= a.expand_min).flatten()
                useful_groups = useful[torch.argsort(tr.d['valid'][useful].sum(1))].tensor_split(4)
                if a.expand_weight > 0 and len(useful) < 4 * a.batch:
                    raise SystemExit(f"only {len(useful)} training queries with >= {a.expand_min} relevant expansion candidates")
                pol = AttnPolicy(n_feat=QEEnv.n_feat).to(dev)
                if method == 'rl' and a.rl_init != 'none':
                    pol.load_state_dict(trained[(a.rl_init, seed)])
                opt = torch.optim.Adam(pol.parameters(), lr=(a.rl_lr or a.lr) if method == 'rl' else a.lr)
                n_it = a.iters if method == 'rl' else a.sup_iters
                w = max(1, n_it // 50)
                sched = torch.optim.lr_scheduler.LambdaLR(
                    opt, lambda t: min(1.0, (t + 1) / w) * 0.5 * (1 + math.cos(math.pi * min(t, n_it) / n_it)))
                sel = lambda: float(np.mean([evaluate(envs[k], pol) for k in val_keys]))
                best, best_state, curve = sel(), {k: x.clone() for k, x in pol.state_dict().items()}, []
                for it in range(n_it):
                    if a.expand_weight > 0 and float(torch.rand(1)) < a.expand_weight:
                        g = useful_groups[int(torch.randint(0, len(useful_groups), (1,)))]
                    else:
                        g = groups[int(torch.randint(0, len(groups), (1,)))]
                    bt = sub_env(tr, g[torch.randint(0, len(g), (a.batch,), device=dev)])
                    idx = torch.arange(a.batch, device=dev)
                    if method == 'rl':
                        rep = idx.repeat_interleave(a.samples)
                        rec = []
                        with torch.no_grad():
                            r, _, _ = rollout(bt, rep, pol, 'sample', record=rec)
                        r = r.view(a.batch, a.samples)
                        adv = (r - (r.sum(1, keepdim=True) - r) / (a.samples - 1)).flatten()
                        opt.zero_grad()
                        policy_gradient_backward(bt, rep, pol, adv, rec)
                        opt.step(); sched.step()
                    else:
                        with torch.no_grad():
                            kind = it % 3 if a.dagger else it % 2
                            sc, mode = [(sim_scorer, 'greedy'), (zero, 'sample'), (pol, 'sample')][kind]
                            _, _, ver = rollout(QEEnv(bt.d, int(torch.randint(0, B, (1,))), a.threshold, c_norm=bt.c_norm),
                                                idx, sc, mode)
                            avail = bt.available(idx, ver)
                            if method == 'value':
                                if a.requery_reward:
                                    from training.qe_requery import requery_gains
                                    target = requery_gains(bt.d, idx, ver & bt.match[idx], bt.d['inliers'][idx],
                                                           avail, bt.match[idx]) * 100
                                else:
                                    target = one_step_gains(bt, idx, ver, avail) * 100
                        fx = bt.features(idx, ver, avail)
                        pred = score(bt, pol, fx, idx)
                        m = avail
                        if not m.any():
                            continue
                        loss = (F.mse_loss(pred[m], target[m]) if method == 'value'
                                else F.binary_cross_entropy_with_logits(pred[m], bt.match[idx][m].float()))
                        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
                    if (it + 1) % a.eval_every == 0:
                        v = sel(); curve.append(round(100 * v, 3))
                        if v > best:
                            best, best_state = v, {k: x.clone() for k, x in pol.state_dict().items()}
                pol.load_state_dict(best_state)
                trained[(method, seed)] = best_state
                torch.save(best_state, out / f'policy_B{B}_{method}_seed{seed}.pt')
                per_seed.append({k: evaluate(e, pol, key=(B, method, k)) for k, e in envs.items()})
                curves.append(curve)
                print(f"B={B} {method} seed {seed}: sel {100 * best:.2f}", flush=True)
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
