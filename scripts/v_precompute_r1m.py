#!/usr/bin/env python3
"""Track V with the Revisited 1M distractors: benchmark shortlists over database + R1M.

For ROxford5k / RParis6k: queries (bounding-box crops) and database images are embedded at long side
1024 exactly as in eval_r1m.py; R1M embeddings are read from the eval_r1m.py cache of the same model.
The gallery is [database images, R1M images] (R1M index = n_db + official list index), so the
Revisited ground truth (database indices) stays valid and every distractor is a negative.
For each query: the full similarity row, its top-K shortlist, the pairwise similarities inside the
shortlist, and verifier inliers for every (query, shortlist item) pair. Only the R1M images that occur
in some shortlist are extracted (one parallel pass over the official archives, into TMPDIR).
Output: <out>/{roxford5k,rparis6k}.npz in the format of v_precompute.py, read by
train_verify_policy.py --bench_pre <out>.
"""
import argparse, json, sys, tarfile, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data.dataset import LongSideResize, RevisitedDataset                    # noqa: E402
from eval_baselines import build_model, embed                                # noqa: E402
from training.verifier import Verifier                                       # noqa: E402

NORM = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def extract_needed(job):
    archive, needed, out_dir = job
    n = 0
    with tarfile.open(archive, 'r:gz') as t:
        for m in t:
            if not m.isreg():
                continue
            rel = '/'.join(m.name.split('/')[-2:])
            if rel in needed:
                dst = Path(out_dir) / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(t.extractfile(m).read())
                n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', required=True, help='staged roxford5k / rparis6k')
    ap.add_argument('--model', required=True)
    ap.add_argument('--r1m_emb', required=True, help='eval_r1m.py embedding cache (.npy) of this model')
    ap.add_argument('--r1m_dir', default='/home/atuin/b268dc/b268dc21/datasets/revisitop1m')
    ap.add_argument('--out', required=True)
    ap.add_argument('--K', type=int, default=100)
    ap.add_argument('--workers', type=int, default=16)
    a = ap.parse_args()
    dev = torch.device('cuda')
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    model = build_model(a.model, dev)
    tf = T.Compose([LongSideResize(1024), T.ToTensor(), NORM])
    names = [l.strip() for l in open(Path(a.r1m_dir) / 'revisitop1m.txt') if l.strip()]
    R = torch.from_numpy(np.load(a.r1m_emb).astype(np.float32)).to(dev)
    assert R.shape[0] == len(names), "R1M cache does not match revisitop1m.txt"
    log, per_ds = {'model': a.model, 'r1m_emb': a.r1m_emb, 'K': a.K}, {}

    for name in ('roxford5k', 'rparis6k'):
        h = RevisitedDataset(str(Path(a.data_root) / name), name)
        Q = torch.from_numpy(embed(model, h.get_query_dataset(tf), dev, a.workers)).to(dev)
        G = torch.from_numpy(embed(model, h.get_database_dataset(tf), dev, a.workers)).to(dev)
        gal = torch.cat([G, R])                                             # [database, R1M]
        sim = Q @ gal.t()
        top_s, top_i = sim.topk(a.K, dim=1)
        g_top = gal[top_i]
        cand = torch.einsum('qkd,qld->qkl', g_top, g_top)
        per_ds[name] = dict(h=h, sim=sim.half().cpu().numpy(), top=top_i.cpu().numpy(),
                            cand=cand.half().cpu().numpy(), n_db=G.shape[0])
        log[name] = {'queries': int(Q.shape[0]), 'db': int(G.shape[0]),
                     'r1m_in_shortlists': int((per_ds[name]['top'] >= G.shape[0]).sum())}
        print(name, json.dumps(log[name]), flush=True)
    del R, model; torch.cuda.empty_cache()

    # extract only the R1M images that appear in some shortlist
    needed = sorted({names[i - d['n_db']] for d in per_ds.values() for i in d['top'].ravel() if i >= d['n_db']})
    r1m_tmp = Path(__import__('os').environ['TMPDIR']) / 'r1m_needed'
    t0 = time.time()
    archives = [str(Path(a.r1m_dir) / 'raw' / f'revisitop1m.{i}.tar.gz') for i in range(1, 101)]
    need_set = set(needed)
    with Pool(24) as p:
        got = sum(p.map(extract_needed, [(f, need_set, str(r1m_tmp)) for f in archives]))
    if got != len(needed):
        raise RuntimeError(f"extracted {got} of {len(needed)} needed R1M images")
    log['r1m_needed'] = len(needed); log['extract_minutes'] = round((time.time() - t0) / 60, 1)
    print(f"extracted {got} R1M images in {log['extract_minutes']} min", flush=True)

    crop_dir = r1m_tmp.parent / 'query_crops'
    for name, d in per_ds.items():
        h, n_db, top = d['h'], d['n_db'], d['top']
        q_paths = []
        (crop_dir / name).mkdir(parents=True, exist_ok=True)
        for i, (img_name, entry) in enumerate(zip(h.cfg['query_images'], h.cfg['gnd'])):
            im = Image.open(h.root_dir / 'jpg' / img_name).convert('RGB')
            if entry.get('bbx') is not None:
                im = im.crop(tuple(int(round(x)) for x in entry['bbx']))
            p = crop_dir / name / f'q{i:03d}.jpg'; im.save(p, quality=95); q_paths.append(str(p))
        nq = len(q_paths)
        db_paths = [str(h.root_dir / 'jpg' / x) for x in h.cfg['db_images']]
        uniq = sorted(set(top.ravel().tolist()))
        gal_paths = {g: (db_paths[g] if g < n_db else str(r1m_tmp / names[g - n_db])) for g in uniq}
        paths = q_paths + [gal_paths[g] for g in uniq]
        pos = {g: nq + k for k, g in enumerate(uniq)}
        v = Verifier('/', paths, torch.device('cuda'), max_kpts=1024, resize=512, batch=32,
                     cpu_workers=16, io_threads=8)
        pairs = [(qi, pos[int(g)]) for qi, row in enumerate(top) for g in row]
        inl = v.verify(pairs).reshape(nq, a.K)
        v.close()
        np.savez_compressed(out / f'{name}.npz', top_idx=top, inliers=inl, sim_full=d['sim'], cand_sim=d['cand'],
                            n_db=n_db)
        log[name]['match_ge35'] = float((inl >= 35).mean())
        log[name]['r1m_match_ge35'] = float((inl[top >= n_db] >= 35).mean()) if (top >= n_db).any() else None
        print(name, json.dumps(log[name]), flush=True)
    (out / 'precompute_r1m_log.json').write_text(json.dumps(log, indent=2))


if __name__ == '__main__':
    main()
