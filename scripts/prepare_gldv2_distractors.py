#!/usr/bin/env python3
"""Track V-D: pack a GLDv2 distractor set for training verification policies on large galleries.

Distractors = every full-GLDv2 train image (train.csv) whose landmark has no image in GLDv2-clean
(neither its train nor its val split): 909,836 images of 121,781 landmarks that the Stage 1 descriptors
never saw and that cannot be relevant to a GLDv2-clean val query (different landmark ids). This plays
the role R1M plays for ROxford/RParis, but R1M itself stays test-only. Images are resized to long side
--size and repacked one output shard per raw tar (as prepare_gldv2.py).

Output (<out>/): shards/gldv2d_NNN.tar, gldv2d.pkl {'cids', 'cluster', 'meta'}, prepare_log.json
Run inside a Slurm allocation (CPU-heavy).
"""
import argparse, json, pickle, sys, time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_gldv2 import init, process   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='/home/atuin/b268dc/b268dc21/datasets/gldv2')
    ap.add_argument('--clean_pkl', default='/home/vault/b268dc/b268dc21/datasets/gldv2c/gldv2c.pkl')
    ap.add_argument('--out', default='/home/atuin/b268dc/b268dc21/datasets/gldv2d')
    ap.add_argument('--size', type=int, default=512)
    ap.add_argument('--workers', type=int, default=32)
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    (out / 'shards').mkdir(parents=True, exist_ok=True)

    cpk = pickle.load(open(a.clean_pkl, 'rb'))
    clean_lm = set(cpk['train']['cluster']) | set(cpk['val']['cluster'])
    keep = {}
    with open(src / 'metadata' / 'train.csv') as f:
        next(f)
        for line in f:
            iid = line.split(',', 1)[0]
            lid = int(line.rstrip('\n').rsplit(',', 1)[1])
            if lid not in clean_lm:
                keep[iid] = lid
    print(f"{len(keep)} distractor images of {len(set(keep.values()))} landmarks", flush=True)

    tars = [src / 'train' / f'images_{n:03d}.tar' for n in range(500)]
    missing = [t.name for t in tars if not Path(str(t) + '.ok').exists()]
    if missing:
        raise SystemExit(f"{len(missing)} tars missing or not md5-verified")
    jobs = [(str(t), str(out / 'shards' / t.name.replace('images_', 'gldv2d_')), a.size) for t in tars]
    t0, found, bad_total = time.time(), [], 0
    with Pool(a.workers, initializer=init, initargs=(keep,)) as pool:
        for k, (name, kept, bad, sec) in enumerate(pool.imap_unordered(process, jobs)):
            found += kept; bad_total += bad
            if k % 50 == 0:
                print(f"{k + 1}/500 tars, {len(found)} images, {(time.time() - t0) / 60:.1f} min", flush=True)

    found.sort()
    meta = {'source': 'GLDv2 train.csv images of landmarks absent from GLDv2-clean (train and val)',
            'size': a.size, 'clean_pkl': a.clean_pkl}
    with open(out / 'gldv2d.pkl', 'wb') as f:
        pickle.dump({'cids': found, 'cluster': [keep[i] for i in found], 'meta': meta}, f)
    assert not {keep[i] for i in found} & clean_lm
    log = {'args': vars(a), 'requested': len(keep), 'found': len(found), 'bad': bad_total,
           'landmarks': len({keep[i] for i in found}), 'minutes': round((time.time() - t0) / 60, 1)}
    (out / 'prepare_log.json').write_text(json.dumps(log, indent=1))
    print(json.dumps(log, indent=1))


if __name__ == '__main__':
    main()
