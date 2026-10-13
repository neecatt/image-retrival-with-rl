#!/usr/bin/env python3
"""Track N1d: verifier-in-the-loop fine-tuning on the noisy GLDv2 pool (see training/verified_rounds.py).

  --arm rl       PL-sampled candidates, PL policy gradient with AP@k reward
  --arm remine   greedy candidates from the current model each round, FastAP
  --arm static   greedy candidates of the initial model, whole verifier budget spent in round 0, FastAP
The verifier budget (number of verified pairs) is the same for all arms: rounds x queries x k.
Selection: GLDv2-clean val leave-one-out mAP after each round; best and last checkpoints are saved
in EmbeddingNet format (loadable by evaluate.py / eval_baselines.py).
"""
import argparse, copy, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dataset import get_transforms                                    # noqa: E402
from data.dataset_sfm120k import load_gldv2n                                # noqa: E402
from evaluate import create_model_from_checkpoint                           # noqa: E402
from evaluation.sfm_val import evaluate_sfm_val, extract_embeddings         # noqa: E402
from training.verified_rounds import (ListBatchSampler, choose_candidates,  # noqa: E402
                                      fastap_masked, pl_loss_masked, relevance)
from training.verifier import PairCache, Verifier                           # noqa: E402
from utils.helpers import save_checkpoint, set_seed                         # noqa: E402

AUG = {'random_flip': True, 'color_jitter': True, 'brightness': 0.2, 'contrast': 0.2,
       'saturation': 0.2, 'hue': 0.1, 'rotation': 10}


class IdBank:
    """FIFO memory of detached mean embeddings with their pool indices."""

    def __init__(self, cap):
        self.cap, self.emb, self.ids = cap, None, []

    def reset(self):
        self.emb, self.ids = None, []

    def add(self, e, ids):
        self.emb = e if self.emb is None else torch.cat([self.emb, e])[-self.cap:]
        self.ids = (self.ids + list(ids))[-self.cap:]

    def get(self):
        return (self.emb, self.ids) if self.ids else (None, [])


def freeze_bn(model):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)):
            m.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=['rl', 'remine', 'static'], required=True)
    ap.add_argument('--init', required=True)
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--ckpt_dir', required=True)
    ap.add_argument('--rounds', type=int, default=4)
    ap.add_argument('--queries_per_round', type=int, default=10000)
    ap.add_argument('--k', type=int, default=10)
    ap.add_argument('--shortlist', type=int, default=100)
    ap.add_argument('--sample_tau', type=float, default=0.05, help='rl: candidate sampling temperature')
    ap.add_argument('--threshold', type=int, default=35)
    ap.add_argument('--steps_per_round', type=int, default=716)
    ap.add_argument('--queries_per_batch', type=int, default=24)
    ap.add_argument('--cands_per_query', type=int, default=4)
    ap.add_argument('--pos_per_query', type=int, default=2)
    ap.add_argument('--lr', type=float, default=1e-6)
    ap.add_argument('--warmup', type=int, default=100)
    ap.add_argument('--tau', type=float, default=0.05, help='rl: PL policy temperature')
    ap.add_argument('--num_rankings', type=int, default=64)
    ap.add_argument('--topk', type=int, default=100)
    ap.add_argument('--bank', type=int, default=4096)
    ap.add_argument('--image_size', type=int, default=512)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--max_pool', type=int, default=0, help='debug: use only the first N train images')
    a = ap.parse_args()
    set_seed(a.seed)
    dev = torch.device('cuda')
    out, ck = Path(a.out), Path(a.ckpt_dir)
    out.mkdir(parents=True, exist_ok=True); ck.mkdir(parents=True, exist_ok=True)
    (out / 'args.json').write_text(json.dumps(vars(a), indent=2))
    root = str(Path(a.data_root) / 'gldv2n')
    tr_eval, val = load_gldv2n(root, train_transform=get_transforms(a.image_size, 'eval'),
                               val_transform=get_transforms(a.image_size, 'eval'))
    tr_aug, _ = load_gldv2n(root, train_transform=get_transforms(a.image_size, 'train', AUG))
    if a.max_pool:
        for ds in (tr_eval, tr_aug):
            ds.image_paths, ds.labels = ds.image_paths[:a.max_pool], ds.labels[:a.max_pool]
    labels = np.asarray(tr_eval.labels)
    n_pool = len(labels)

    model = create_model_from_checkpoint(a.init, dev)
    init_model = copy.deepcopy(model).eval() if a.arm == 'static' else None
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(a.warmup, 1)))
    cache = PairCache(str(ck / 'pair_cache.npz'))
    verifier = Verifier(tr_eval.image_root, tr_eval.image_paths, dev, max_kpts=1024, resize=512,
                        batch=32, cpu_workers=16, io_threads=16)
    bank = IdBank(a.bank)
    budget_used, history, best = 0, [], float('-inf')
    v0 = evaluate_sfm_val(model, val, dev, num_workers=a.workers)
    history.append({'round': 0, **v0})
    rng = np.random.default_rng(a.seed)
    lists = {}
    step = 0

    for r in range(1, a.rounds + 1):
        t_round = time.time()
        # ---- candidates and verification -------------------------------------------------
        if a.arm != 'static' or r == 1:
            nq = a.queries_per_round * (a.rounds if a.arm == 'static' else 1)
            qs = rng.choice(n_pool, size=min(nq, n_pool), replace=False)
            emb_model = init_model if a.arm == 'static' else model
            g_emb, _ = extract_embeddings(emb_model, tr_eval, dev, batch_size=64, num_workers=a.workers)
            q_idx = torch.as_tensor(qs, device=dev)
            cand = choose_candidates(g_emb[q_idx], g_emb, q_idx, a.k, 'sample' if a.arm == 'rl' else 'greedy',
                                     tau=a.sample_tau, shortlist=a.shortlist, seed=a.seed * 1000 + r).cpu().numpy()
            del g_emb
            pairs = [(int(q), int(c)) for q, row in zip(qs, cand) for c in row]
            new = cache.missing(pairs)
            t_v = time.time()
            if new:
                cache.add(new, verifier.verify(new)); cache.save()
            budget_used += len(new)
            lists = {}
            for q, row in zip(qs, cand):
                q = int(q)
                pos = [int(c) for c in row if cache.get(q, int(c)) >= a.threshold]
                neg = [int(c) for c in row if 0 <= cache.get(q, int(c)) < a.threshold]
                lists[q] = (pos, neg)
            n_pos_q = sum(1 for p, _ in lists.values() if p)
            print(f"round {r}: {len(qs)} queries, {len(new)} new verified pairs ({time.time() - t_v:.0f} s), "
                  f"budget {budget_used}, queries with a verified positive {n_pos_q}", flush=True)
        # ---- training ------------------------------------------------------------------------
        sampler = ListBatchSampler(lists, a.queries_per_batch, a.cands_per_query, a.pos_per_query,
                                   a.steps_per_round, seed=a.seed * 1000 + r)
        loader = torch.utils.data.DataLoader(tr_aug, batch_sampler=sampler, num_workers=a.workers, pin_memory=True)
        model.train(); freeze_bn(model); bank.reset()
        blk = 1 + a.cands_per_query
        sums, nb = {}, 0
        for images, _, idx in loader:
            images = images.to(dev, non_blocking=True)
            ids = idx.tolist()
            emb = model(images).float()
            qpos = list(range(0, len(ids), blk))          # every block is [query, m candidates]
            q_ids = [ids[i] for i in qpos]
            m_emb, m_ids = bank.get()
            gal = torch.cat([emb, m_emb]) if m_ids else emb
            rel, valid = relevance(q_ids, ids + m_ids, labels, cache, a.threshold)
            rel_t, valid_t = torch.as_tensor(rel, device=dev), torch.as_tensor(valid, device=dev)
            if a.arm == 'rl':
                loss, st = pl_loss_masked(emb[qpos], gal, rel_t, valid_t, a.tau, a.num_rankings, a.topk)
            else:
                loss, st = fastap_masked(emb[qpos], gal, rel_t, valid_t, topk=a.topk)
            opt.zero_grad(); loss.backward()
            st['grad_norm'] = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf')).item()
            opt.step(); sched.step()
            bank.add(emb.detach(), ids)
            st['loss'] = loss.item()
            for kk, vv in st.items():
                sums[kk] = sums.get(kk, 0.0) + vv
            nb += 1; step += 1
            if step % 100 == 0:
                print(f"  step {step}: " + ", ".join(f"{kk}={vv:.4g}" for kk, vv in st.items()), flush=True)
        tr = {kk: vv / max(nb, 1) for kk, vv in sums.items()}
        vm = evaluate_sfm_val(model, val, dev, num_workers=a.workers)
        rec = {'round': r, 'budget_used': budget_used, 'minutes': round((time.time() - t_round) / 60, 1),
               **{f'train_{kk}': vv for kk, vv in tr.items()}, **vm}
        history.append(rec)
        print(json.dumps(rec), flush=True)
        state = {'epoch': r, 'model_state_dict': model.state_dict(), 'val_metrics': vm,
                 'config': torch.load(a.init, map_location='cpu', weights_only=False)['config'],
                 'verified_args': vars(a)}
        save_checkpoint(state, str(ck / 'verified_last.pth'))
        if vm['sfm_val_mAP'] > best:
            best = vm['sfm_val_mAP']
            save_checkpoint(state, str(ck / 'verified_best.pth'))
        (out / 'history.json').write_text(json.dumps(history, indent=2))
    verifier.close()
    print(json.dumps(history, indent=2))


if __name__ == '__main__':
    main()
