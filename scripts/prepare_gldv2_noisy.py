#!/usr/bin/env python3
"""Track N1a: pack a noisy-label GLDv2 training pool.

Pool = ALL full-GLDv2 train images (train.csv labels, i.e. noisy) of --landmarks random GLDv2-clean
training landmarks (the GLDv2-clean val landmarks are never chosen). The validation split is the
GLDv2-clean val split (clean labels, same images as gldv2c), so selection is identical to Track G.
Images are resized to long side --size and repacked one output shard per raw tar (as prepare_gldv2.py).

Output (<out>/):
  shards/gldv2n_NNN.tar
  gldv2n.pkl   {'train': {'cids', 'cluster', 'in_clean'}, 'val': {'cids', 'cluster'}, 'meta'}
               in_clean marks train images that GLDv2-clean kept (for analysis only; never used in training)
  prepare_log.json
"""
import argparse, json, pickle, random, sys, time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_gldv2 import init, process   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='/home/atuin/b268dc/b268dc21/datasets/gldv2')
    ap.add_argument('--clean_pkl', default='/home/vault/b268dc/b268dc21/datasets/gldv2c/gldv2c.pkl')
    ap.add_argument('--out', default='/home/vault/b268dc/b268dc21/datasets/gldv2n')
    ap.add_argument('--landmarks', type=int, default=20000)
    ap.add_argument('--size', type=int, default=512)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--workers', type=int, default=32)
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    (out / 'shards').mkdir(parents=True, exist_ok=True)

    cpk = pickle.load(open(a.clean_pkl, 'rb'))
    val_lm = set(cpk['meta']['val_landmarks'])
    clean_ids = set(cpk['train']['cids']) | set(cpk['val']['cids'])
    train_lms = sorted(set(cpk['train']['cluster']))
    chosen = set(random.Random(a.seed).sample(train_lms, min(a.landmarks, len(train_lms))))
    assert not chosen & val_lm

    keep = {}   # id -> landmark (noisy train label), plus clean val images
    with open(src / 'metadata' / 'train.csv') as f:
        next(f)
        for line in f:
            iid, rest = line.split(',', 1)
            lid = int(line.rstrip('\n').rsplit(',', 1)[1])
            if lid in chosen:
                keep[iid] = lid
    for i, l in zip(cpk['val']['cids'], cpk['val']['cluster']):
        keep[i] = l
    print(f"{len(chosen)} landmarks; {len(keep)} images to pack (incl. {len(cpk['val']['cids'])} val)", flush=True)

    tars = [src / 'train' / f'images_{n:03d}.tar' for n in range(500)]
    missing = [t.name for t in tars if not Path(str(t) + '.ok').exists()]
    if missing:
        raise SystemExit(f"{len(missing)} tars missing or not md5-verified")
    jobs = [(str(t), str(out / 'shards' / t.name.replace('images_', 'gldv2n_')), a.size) for t in tars]
    t0, found, bad_total = time.time(), set(), 0
    with Pool(a.workers, initializer=init, initargs=(keep,)) as pool:
        for k, (name, kept, bad, sec) in enumerate(pool.imap_unordered(process, jobs)):
            found.update(kept); bad_total += bad
            if k % 50 == 0:
                print(f"{k + 1}/500 tars, {len(found)} images, {(time.time() - t0) / 60:.1f} min", flush=True)

    val_set = set(cpk['val']['cids'])
    tr = sorted((keep[i], i) for i in found if i not in val_set)
    split = {'train': {'cids': [i for _, i in tr], 'cluster': [l for l, _ in tr],
                       'in_clean': [i in clean_ids for _, i in tr]},
             'val': {'cids': list(cpk['val']['cids']), 'cluster': list(cpk['val']['cluster'])},
             'meta': {'source': 'GLDv2 train.csv (noisy) for a random subset of GLDv2-clean train landmarks; '
                                'val = GLDv2-clean val', 'landmarks': sorted(chosen), 'size': a.size,
                      'seed': a.seed, 'val_landmarks': cpk['meta']['val_landmarks']}}
    missing_val = val_set - found
    with open(out / 'gldv2n.pkl', 'wb') as f:
        pickle.dump(split, f)
    n_clean = sum(split['train']['in_clean'])
    log = {'args': vars(a), 'requested': len(keep), 'found': len(found), 'bad': bad_total,
           'train_images': len(tr), 'train_in_clean': n_clean, 'train_dropped_by_cleaning': len(tr) - n_clean,
           'train_landmarks': len(set(split['train']['cluster'])), 'val_images': len(split['val']['cids']),
           'val_missing': len(missing_val), 'minutes': round((time.time() - t0) / 60, 1)}
    (out / 'prepare_log.json').write_text(json.dumps(log, indent=1))
    print(json.dumps(log, indent=1))


if __name__ == '__main__':
    main()
