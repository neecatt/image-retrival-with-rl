#!/usr/bin/env python3
"""Option 2 feasibility (F0): how noisy are full-GLDv2 labels for the GLDv2-clean training landmarks?

For random training landmarks (val landmarks excluded) pick one image that the cleaning dropped
("dropped": in train.csv, not in train_clean.csv) and one kept image ("clean"), and verify each
against up to R other clean images of the same landmark (ALIKED + LightGlue + MAGSAC, long side 512).
A query counts as verified if its best reference has >= 25 (or >= 100) inliers.
If dropped images verify far less often than clean ones, full GLDv2 carries label noise that a
verifier can detect, which is the premise of verifier-in-the-loop RL.

Also builds (once) an index of the 500 raw train tars: id -> (tar, offset, size), saved next to
the tars, so single images can be read without extraction.
Outputs: <out>/f0_pairs.npz, <out>/f0_summary.json, contact sheets of dropped images.
"""
import argparse, io, json, random, sys, tarfile, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from oracle_label import ransac_inliers                 # noqa: E402
from lightglue import ALIKED, LightGlue                 # noqa: E402
from lightglue.utils import numpy_image_to_torch, rbd   # noqa: E402


def index_tar(path):
    ids, offs, sizes = [], [], []
    with tarfile.open(path, 'r:') as t:
        for m in t:
            if m.isreg() and m.name.endswith('.jpg'):
                ids.append(m.name.rsplit('/', 1)[1][:-4]); offs.append(m.offset_data); sizes.append(m.size)
    return path.name, ids, offs, sizes


def build_index(src, workers):
    f = src / 'train_index.npz'
    if f.exists():
        d = np.load(f)
        return {i: (int(t), int(o), int(s)) for i, t, o, s in zip(d['ids'], d['tar'], d['offset'], d['size'])}
    t0 = time.time()
    tars = [src / 'train' / f'images_{n:03d}.tar' for n in range(500)]
    I, T, O, S = [], [], [], []
    with Pool(workers) as p:
        for name, ids, offs, sizes in p.imap_unordered(index_tar, tars):
            n = int(name[7:10])
            I += ids; T += [n] * len(ids); O += offs; S += sizes
    np.savez(f, ids=np.array(I), tar=np.array(T, dtype=np.int16), offset=np.array(O, dtype=np.int64),
             size=np.array(S, dtype=np.int64))
    print(f"indexed {len(I)} images in {(time.time() - t0) / 60:.1f} min -> {f}", flush=True)
    return {i: (t, o, s) for i, t, o, s in zip(I, T, O, S)}


def read_image(src, entry):
    t, o, s = entry
    with open(src / 'train' / f'images_{t:03d}.tar', 'rb') as f:
        f.seek(o)
        return Image.open(io.BytesIO(f.read(s))).convert('RGB')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='/home/vault/b268dc/b268dc21/datasets/gldv2')
    ap.add_argument('--split_pkl', default='/home/vault/b268dc/b268dc21/datasets/gldv2c/gldv2c.pkl')
    ap.add_argument('--out', default='results/f0_gld_noise')
    ap.add_argument('--landmarks', type=int, default=2000)
    ap.add_argument('--refs', type=int, default=3)
    ap.add_argument('--resize', type=int, default=512)
    ap.add_argument('--max_kpts', type=int, default=2048)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device('cuda')

    import pickle
    val_lm = set(pickle.load(open(a.split_pkl, 'rb'))['meta']['val_landmarks'])
    clean = {}
    with open(src / 'metadata' / 'train_clean.csv') as f:
        next(f)
        for line in f:
            lid, ims = line.rstrip('\n').split(',', 1)
            clean[int(lid)] = ims.split()
    dropped, clean_sets = {}, {}
    with open(src / 'metadata' / 'train.csv') as f:
        next(f)
        for line in f:
            iid, rest = line.split(',', 1)
            lid = int(line.rstrip('\n').rsplit(',', 1)[1])
            if lid in clean and lid not in val_lm and iid not in clean_sets.setdefault(lid, set(clean[lid])):
                dropped.setdefault(lid, []).append(iid)
    eligible = sorted(l for l in dropped if len(clean[l]) >= a.refs + 1)
    rng = random.Random(a.seed)
    chosen = rng.sample(eligible, min(a.landmarks, len(eligible)))
    jobs = []   # (kind, landmark, query, refs)
    for l in chosen:
        cl = sorted(clean[l])
        qc = rng.choice(cl)
        refs = rng.sample([x for x in cl if x != qc], a.refs)
        jobs.append(('clean', l, qc, refs))
        jobs.append(('dropped', l, rng.choice(sorted(dropped[l])), refs))
    print(f"{len(eligible)} eligible landmarks; {len(chosen)} sampled; {len(jobs)} queries x {a.refs} refs", flush=True)

    index = build_index(src, a.workers)
    extractor = ALIKED(max_num_keypoints=a.max_kpts).eval().to(dev)
    matcher = LightGlue(features='aliked').eval().to(dev)
    feats = {}

    def feat(i):
        if i not in feats:
            img = numpy_image_to_torch(np.asarray(read_image(src, index[i]))).to(dev)
            feats[i] = extractor.extract(img, resize=a.resize)
        return feats[i]

    rec, pend, t0 = [], [], time.time()
    with Pool(a.workers) as pool, torch.no_grad():
        for n, (kind, l, q, refs) in enumerate(jobs):
            fq, kp = feat(q), []
            for r in refs:
                fr = feat(r)
                m = rbd(matcher({'image0': fq, 'image1': fr}))['matches']
                kp.append((rbd(fq)['keypoints'][m[:, 0]].cpu().numpy(), rbd(fr)['keypoints'][m[:, 1]].cpu().numpy()))
                rec.append((kind, l, q, r))
            pend.append(pool.map_async(ransac_inliers, kp))
            if len(feats) > 20000:
                feats.clear()
            if n % 500 == 0:
                print(f"query {n + 1}/{len(jobs)}  {len(rec) / max(time.time() - t0, 1e-9):.1f} pairs/s", flush=True)
        inl = np.array([x for p in pend for x in p.get()])

    kinds = np.array([r[0] for r in rec]); qs = np.array([r[2] for r in rec])
    np.savez_compressed(out / 'f0_pairs.npz', kind=kinds, landmark=np.array([r[1] for r in rec]),
                        query=qs, ref=np.array([r[3] for r in rec]), inliers=inl)
    best = {}
    for k, q, x in zip(kinds, qs, inl):
        best[(k, q)] = max(best.get((k, q), 0), int(x))
    summ = {'landmarks': len(chosen), 'refs': a.refs, 'resize': a.resize, 'pairs': int(len(inl)),
            'seconds': round(time.time() - t0)}
    for k in ('clean', 'dropped'):
        b = np.array([v for (kk, _), v in best.items() if kk == k])
        summ[k] = {'queries': int(b.size), 'verified_ge25': float((b >= 25).mean()),
                   'verified_ge100': float((b >= 100).mean()), 'median_best_inliers': float(np.median(b))}
    (out / 'f0_summary.json').write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))

    def sheet(items, name, size=200):
        rows = []
        for q, r in items[:16]:
            ims = [read_image(src, index[x]).resize((size, size)) for x in (q, r)]
            row = Image.new('RGB', (2 * size + 6, size), 'white'); row.paste(ims[0], (0, 0)); row.paste(ims[1], (size + 6, 0))
            rows.append(row)
        if rows:
            s = Image.new('RGB', (rows[0].width, len(rows) * (size + 6)), 'white')
            for i, row in enumerate(rows):
                s.paste(row, (0, i * (size + 6)))
            s.save(out / name, quality=85)
    fail = [(q, refs[0]) for kind, l, q, refs in jobs if kind == 'dropped' and best[('dropped', q)] < 25]
    ok = [(q, refs[0]) for kind, l, q, refs in jobs if kind == 'dropped' and best[('dropped', q)] >= 100]
    sheet(fail, 'f0_dropped_unverified.jpg'); sheet(ok, 'f0_dropped_verified.jpg')


if __name__ == '__main__':
    main()
