#!/usr/bin/env python3
"""A0 of the revised RL plan (docs/RL_RESEARCH_PLAN.md §4b): how much can a verifier in the loop add?

Embed SfM-120k train with a Stage 1 model, take the top-K neighbours of random queries, and
geometrically verify every retrieved pair (ALIKED + LightGlue + MAGSAC, as in oracle_label.py).
Measures, by rank:
  - retrieved "negatives" (different 3D-model cluster) that verify as matches (label noise the
    cluster labels cannot see, and that a verifier-in-the-loop policy could exploit);
  - retrieved positives with no geometric support (grade disagreement).
Outputs <out>/a0_pairs.npz, <out>/a0_summary.json, train embeddings at --emb_file and a contact sheet
of the strongest "negative" matches for visual inspection.
"""
import argparse, json, random, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data.dataset import get_transforms                     # noqa: E402
from data.dataset_sfm120k import load_sfm120k               # noqa: E402
from evaluate import create_model_from_checkpoint           # noqa: E402
from evaluation.sfm_val import extract_embeddings           # noqa: E402
from oracle_label import ransac_inliers                     # noqa: E402
from lightglue import ALIKED, LightGlue                     # noqa: E402
from lightglue.utils import load_image, rbd                 # noqa: E402


def contact_sheet(root, paths, pairs, out, size=256):
    rows = []
    for i, j, inl in pairs:
        ims = [Image.open(root / paths[x]).convert('RGB').resize((size, size)) for x in (i, j)]
        row = Image.new('RGB', (2 * size + 8, size), 'white')
        row.paste(ims[0], (0, 0)); row.paste(ims[1], (size + 8, 0))
        rows.append(row)
    if not rows:
        return
    sheet = Image.new('RGB', (rows[0].width, len(rows) * (size + 8)), 'white')
    for r, row in enumerate(rows):
        sheet.paste(row, (0, r * (size + 8)))
    sheet.save(out, quality=85)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--num_queries', type=int, default=2000)
    ap.add_argument('--topk', type=int, default=20)
    ap.add_argument('--image_size', type=int, default=1024)
    ap.add_argument('--max_kpts', type=int, default=2048)
    ap.add_argument('--resize', type=int, default=1024)
    ap.add_argument('--cpu_workers', type=int, default=24)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--emb_file', required=True, help='cache for train embeddings (keep on HPCVAULT)')
    a = ap.parse_args()
    dev = torch.device('cuda')
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    tf = get_transforms(a.image_size, 'eval')
    train, _ = load_sfm120k(str(Path(a.data_root) / 'sfm120k'), train_transform=tf, val_transform=tf)
    labels = np.asarray(train.labels)
    emb_f = Path(a.emb_file); emb_f.parent.mkdir(parents=True, exist_ok=True)
    if emb_f.exists():
        emb = torch.from_numpy(np.load(emb_f)).to(dev)
    else:
        t0 = time.time()
        model = create_model_from_checkpoint(a.model, dev)
        emb, _ = extract_embeddings(model, train, dev, batch_size=32, num_workers=16)
        np.save(emb_f, emb.cpu().numpy())
        del model
        print(f"embedded {emb.size(0)} train images in {(time.time() - t0) / 60:.1f} min", flush=True)

    rng = random.Random(a.seed)
    queries = sorted(rng.sample(range(len(labels)), a.num_queries))
    q = torch.tensor(queries, device=dev)
    sim = emb[q] @ emb.t()
    sim[torch.arange(len(queries)), q] = float('-inf')
    top_sim, top_idx = sim.topk(a.topk, dim=1)
    top_sim, top_idx = top_sim.cpu().numpy(), top_idx.cpu().numpy()

    extractor = ALIKED(max_num_keypoints=a.max_kpts).eval().to(dev)
    matcher = LightGlue(features='aliked').eval().to(dev)
    root = Path(train.image_root)
    rec_q, rec_n, rec_r, rec_s, rec_m, pending = [], [], [], [], [], []
    t0 = time.time()
    with Pool(a.cpu_workers) as pool, torch.no_grad():
        for n, qi in enumerate(queries):
            fq = extractor.extract(load_image(root / train.image_paths[qi]).to(dev), resize=a.resize)
            jobs = []
            for r, nb in enumerate(top_idx[n].tolist()):
                fn = extractor.extract(load_image(root / train.image_paths[nb]).to(dev), resize=a.resize)
                m = rbd(matcher({'image0': fq, 'image1': fn}))['matches']
                jobs.append((rbd(fq)['keypoints'][m[:, 0]].cpu().numpy(), rbd(fn)['keypoints'][m[:, 1]].cpu().numpy()))
                rec_q.append(qi); rec_n.append(nb); rec_r.append(r); rec_s.append(float(top_sim[n, r])); rec_m.append(len(m))
            pending.append(pool.map_async(ransac_inliers, jobs, chunksize=4))
            if n % 100 == 0:
                el = time.time() - t0
                print(f"query {n + 1}/{len(queries)}  {len(rec_q) / max(el, 1e-9):.1f} pairs/s  {el / 60:.1f} min", flush=True)
        inliers = np.array([x for p in pending for x in p.get()])

    rq, rn, rr = np.array(rec_q), np.array(rec_n), np.array(rec_r)
    same = labels[rq] == labels[rn]
    np.savez_compressed(out / 'a0_pairs.npz', q=rq, n=rn, rank=rr, sim=np.array(rec_s),
                        matches=np.array(rec_m), inliers=inliers, same_label=same)

    def frac(mask):
        return float(mask.mean()) if mask.size else float('nan')
    neg, pos = ~same, same
    summary = {
        'model': a.model, 'num_queries': len(queries), 'topk': a.topk, 'pairs': int(len(rq)),
        'retrieved_negative_frac': frac(neg),
        'neg_verified_ge25': frac(inliers[neg] >= 25), 'neg_verified_ge100': frac(inliers[neg] >= 100),
        'pos_unsupported_lt25': frac(inliers[pos] < 25),
        'disagreement_frac_all_pairs': frac((neg & (inliers >= 100)) | (pos & (inliers < 25))),
        'queries_with_verified_neg_ge100': frac(np.array([((rq == x) & neg & (inliers >= 100)).any() for x in queries])),
        'by_rank': {}, 'seconds': time.time() - t0,
    }
    for lo, hi in [(0, 5), (5, 10), (10, 20)]:
        sel = (rr >= lo) & (rr < hi)
        summary['by_rank'][f'{lo + 1}-{hi}'] = {
            'neg_frac': frac(neg[sel]),
            'neg_verified_ge100': frac(inliers[sel & neg] >= 100),
            'pos_unsupported_lt25': frac(inliers[sel & pos] < 25)}
    (out / 'a0_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    order = np.argsort(-inliers * neg)
    top_neg = [(int(rq[k]), int(rn[k]), int(inliers[k])) for k in order[:16] if neg[k]]
    contact_sheet(root, train.image_paths, top_neg, out / 'a0_top_verified_negatives.jpg')
    mid = [k for k in np.where(neg & (inliers >= 100) & (inliers < 200))[0][:16]]
    contact_sheet(root, train.image_paths, [(int(rq[k]), int(rn[k]), int(inliers[k])) for k in mid],
                  out / 'a0_borderline_verified_negatives.jpg')
    print("verified negatives (q, n, inliers):", top_neg)


if __name__ == '__main__':
    main()
