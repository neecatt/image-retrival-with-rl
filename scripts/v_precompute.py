#!/usr/bin/env python3
"""Track V (RL for budgeted verification), step 1: precompute everything a verification policy needs.

With a fixed descriptor, for each query: the global similarities to the whole gallery, its top-K
shortlist, the pairwise similarities among the shortlist, and the verifier's inlier count for every
(query, shortlist item) pair. Policies are then trained and evaluated without touching images.

Pools:
  gldv2c_val  GLDv2-clean held-out val (21,285 images, 1,500 landmarks); --gld_queries random queries
              against all other val images; relevance = same landmark (clean labels). Used to train
              and select policies (split by landmark later).
  roxford5k / rparis6k  official queries (bounding-box crops) against the database; relevance and junk
              from the Revisited ground truth. Used only for final evaluation.
Output: <out>/<pool>.npz with sim_full (Q, N) fp16, top_idx (Q, K), inliers (Q, K), cand_sim (Q, K, K) fp16,
and labels/queries (GLDv2) or nothing extra (benchmarks: gnd comes from the staged pkl).
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dataset import LongSideResize, RevisitedDataset, get_transforms   # noqa: E402
from data.dataset_sfm120k import load_gldv2c                                  # noqa: E402
from evaluate import create_model_from_checkpoint                             # noqa: E402
from evaluation.sfm_val import extract_embeddings                             # noqa: E402
from training.verifier import Verifier                                        # noqa: E402
import torchvision.transforms as T                                            # noqa: E402

NORM = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


@torch.no_grad()
def embed_each(model, ds, dev, workers):
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=workers, pin_memory=True)
    return F.normalize(torch.cat([model(x.to(dev)).float() for x, *_ in loader]), dim=1)


def shortlist(q_emb, g_emb, K, self_idx=None):
    sim = q_emb @ g_emb.t()
    if self_idx is not None:
        sim[torch.arange(sim.size(0)), self_idx] = float('-inf')
    top_sim, top_idx = sim.topk(K, dim=1)
    g_top = g_emb[top_idx]                                    # (Q, K, d)
    cand_sim = torch.einsum('qkd,qld->qkl', g_top, g_top)
    return sim, top_idx, cand_sim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--K', type=int, default=100)
    ap.add_argument('--gld_queries', type=int, default=3000)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    dev = torch.device('cuda')
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    root = Path(a.data_root)
    model = create_model_from_checkpoint(a.model, dev).eval()
    log = {'model': a.model, 'K': a.K}

    # ---- GLDv2-clean val pool --------------------------------------------------------------
    t0 = time.time()
    _, val = load_gldv2c(str(root / 'gldv2c'), val_transform=get_transforms(512, 'eval'))
    emb, _ = extract_embeddings(model, val, dev, batch_size=64, num_workers=a.workers)
    labels = np.asarray(val.labels)
    cnt = np.bincount(labels)
    rng = np.random.default_rng(a.seed)
    eligible = np.where(cnt[labels] >= 2)[0]
    qs = np.sort(rng.choice(eligible, size=min(a.gld_queries, len(eligible)), replace=False))
    q_t = torch.as_tensor(qs, device=dev)
    sim, top_idx, cand_sim = shortlist(emb[q_t], emb, a.K, self_idx=q_t)
    top = top_idx.cpu().numpy()
    v = Verifier(val.image_root, val.image_paths, dev, max_kpts=1024, resize=512, batch=32,
                 cpu_workers=16, io_threads=8)
    pairs = [(int(q), int(c)) for q, row in zip(qs, top) for c in row]
    inl = v.verify(pairs).reshape(len(qs), a.K)
    v.close()
    np.savez_compressed(out / 'gldv2c_val.npz', queries=qs, labels=labels, top_idx=top, inliers=inl,
                        sim_full=sim.half().cpu().numpy(), cand_sim=cand_sim.half().cpu().numpy())
    log['gldv2c_val'] = {'queries': len(qs), 'pool': len(labels), 'pairs': len(pairs),
                         'minutes': round((time.time() - t0) / 60, 1),
                         'rel_in_shortlist': float((labels[top] == labels[qs][:, None]).mean()),
                         'match_ge35': float((inl >= 35).mean())}
    print(json.dumps(log['gldv2c_val']), flush=True)

    # ---- Revisited benchmarks ----------------------------------------------------------------
    tf = T.Compose([LongSideResize(1024), T.ToTensor(), NORM])
    crop_dir = Path(a.out) / 'query_crops'
    for name in ('roxford5k', 'rparis6k'):
        t0 = time.time()
        h = RevisitedDataset(str(root / name), name)
        qd, gd = h.get_query_dataset(tf), h.get_database_dataset(tf)
        q_emb, g_emb = embed_each(model, qd, dev, a.workers), embed_each(model, gd, dev, a.workers)
        sim, top_idx, cand_sim = shortlist(q_emb, g_emb, a.K)
        top = top_idx.cpu().numpy()
        # cropped query images (as in the protocol) are verified against database images
        (crop_dir / name).mkdir(parents=True, exist_ok=True)
        q_paths = []
        for i, (img_name, entry) in enumerate(zip(h.cfg['query_images'], h.cfg['gnd'])):
            im = Image.open(h.root_dir / 'jpg' / img_name).convert('RGB')
            if entry.get('bbx') is not None:
                im = im.crop(tuple(int(round(x)) for x in entry['bbx']))
            p = crop_dir / name / f'q{i:03d}.jpg'
            im.save(p, quality=95)
            q_paths.append(str(p.resolve()))
        paths = q_paths + [str((h.root_dir / 'jpg' / x).resolve()) for x in h.cfg['db_images']]
        nq = len(q_paths)
        v = Verifier('/', paths, dev, max_kpts=1024, resize=512, batch=32, cpu_workers=16, io_threads=8)
        pairs = [(i, nq + int(c)) for i, row in enumerate(top) for c in row]
        inl = v.verify(pairs).reshape(nq, a.K)
        v.close()
        np.savez_compressed(out / f'{name}.npz', top_idx=top, inliers=inl, sim_full=sim.half().cpu().numpy(),
                            cand_sim=cand_sim.half().cpu().numpy())
        log[name] = {'queries': nq, 'db': len(h.cfg['db_images']), 'pairs': len(pairs),
                     'minutes': round((time.time() - t0) / 60, 1), 'match_ge35': float((inl >= 35).mean())}
        print(json.dumps(log[name]), flush=True)
    (out / 'precompute_log.json').write_text(json.dumps(log, indent=2))


if __name__ == '__main__':
    main()
