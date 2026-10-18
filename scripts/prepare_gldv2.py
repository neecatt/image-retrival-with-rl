#!/usr/bin/env python3
"""Track G2: build the packed GLDv2-clean training set used by this project.

Reads the 500 official GLDv2 train tars (md5-verified by fetch_train.sh) and train_clean.csv,
keeps only GLDv2-clean images, resizes them to long side --size (JPEG quality 90) and writes
one uncompressed output shard per input tar, with the official a/b/c/<id>.jpg layout.
Nothing is extracted to shared storage: images go tar -> memory -> tar.

Split: --val_landmarks landmarks with at least 4 clean images are held out as validation,
chosen with a fixed seed; at most --val_cap images per held-out landmark are kept (the rest of
that landmark is dropped, so no landmark appears in both splits).

Output (<out>/):
  shards/gldv2c_NNN.tar      images
  gldv2c.pkl                 {'train': {'cids', 'cluster'}, 'val': {...}, 'meta': {...}}
                             same keys as retrieval-SfM-120k.pkl; cids are GLDv2 image ids
  prepare_log.json           counts per shard, timing, arguments
Run inside a Slurm allocation (CPU-heavy).
"""
import argparse, io, json, os, random, re, tarfile, time, pickle
from multiprocessing import Pool
from pathlib import Path

from PIL import Image

ID_RE = re.compile(r'^([0-9a-f])/([0-9a-f])/([0-9a-f])/([0-9a-f]{16})\.jpg$')
CLEAN = None   # id -> landmark, set in workers via initializer


def init(clean):
    global CLEAN
    CLEAN = clean


def process(job):
    src, dst, size = job
    kept, bad, t0 = [], 0, time.time()
    tmp = dst + '.part'
    with tarfile.open(src, 'r:') as tin, tarfile.open(tmp, 'w:') as tout:
        for m in tin:
            if not m.isreg():
                continue
            g = ID_RE.match(m.name)
            if not g or g.group(4) not in CLEAN:
                continue
            try:
                im = Image.open(io.BytesIO(tin.extractfile(m).read()))
                im.draft('RGB', (size, size))
                im = im.convert('RGB')
                im.thumbnail((size, size), Image.BILINEAR)
                buf = io.BytesIO()
                im.save(buf, 'JPEG', quality=90)
            except Exception:
                bad += 1
                continue
            data = buf.getvalue()
            info = tarfile.TarInfo(m.name)
            info.size, info.mtime = len(data), 0
            tout.addfile(info, io.BytesIO(data))
            kept.append(g.group(4))
    os.replace(tmp, dst)
    return os.path.basename(src), kept, bad, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='/home/atuin/b268dc/b268dc21/datasets/gldv2')
    ap.add_argument('--out', default='/home/vault/b268dc/b268dc21/datasets/gldv2c')
    ap.add_argument('--size', type=int, default=512)
    ap.add_argument('--val_landmarks', type=int, default=1500)
    ap.add_argument('--val_cap', type=int, default=30)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--workers', type=int, default=32)
    ap.add_argument('--limit_tars', type=int, default=0, help='debug: only the first N tars')
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    (out / 'shards').mkdir(parents=True, exist_ok=True)

    clean = {}
    with open(src / 'metadata' / 'train_clean.csv') as f:
        next(f)
        for line in f:
            lid, ims = line.rstrip('\n').split(',', 1)
            for i in ims.split():
                clean[i] = int(lid)
    print(f"clean ids: {len(clean)} over {len(set(clean.values()))} landmarks", flush=True)

    tars = [src / 'train' / f'images_{n:03d}.tar' for n in range(a.limit_tars or 500)]
    missing = [t.name for t in tars if not Path(str(t) + '.ok').exists()]
    if missing:
        raise SystemExit(f"{len(missing)} tars missing or not md5-verified, e.g. {missing[:3]}")
    jobs = [(str(t), str(out / 'shards' / t.name.replace('images_', 'gldv2c_')), a.size) for t in tars]

    t0, found, bad_total, per_shard = time.time(), {}, 0, {}
    with Pool(a.workers, initializer=init, initargs=(clean,)) as pool:
        for k, (name, kept, bad, sec) in enumerate(pool.imap_unordered(process, jobs)):
            for i in kept:
                found[i] = clean[i]
            bad_total += bad
            per_shard[name] = {'kept': len(kept), 'bad': bad, 'seconds': round(sec, 1)}
            if k % 20 == 0:
                print(f"{k + 1}/{len(jobs)} tars, {len(found)} images, {(time.time() - t0) / 60:.1f} min", flush=True)

    by_lm = {}
    for i, lid in found.items():
        by_lm.setdefault(lid, []).append(i)
    for v in by_lm.values():
        v.sort()
    rng = random.Random(a.seed)
    eligible = sorted(l for l, v in by_lm.items() if len(v) >= 4)
    val_lm = set(rng.sample(eligible, min(a.val_landmarks, len(eligible))))
    split = {'train': {'cids': [], 'cluster': []}, 'val': {'cids': [], 'cluster': []}}
    for lid in sorted(by_lm):
        ims = by_lm[lid]
        if lid in val_lm:
            ims = sorted(rng.sample(ims, min(a.val_cap, len(ims))))
            part = split['val']
        else:
            part = split['train']
        part['cids'] += ims
        part['cluster'] += [lid] * len(ims)
    split['meta'] = {'source': 'GLDv2 train_clean.csv', 'size': a.size, 'seed': a.seed,
                     'val_landmarks': sorted(val_lm), 'val_cap': a.val_cap}
    with open(out / 'gldv2c.pkl', 'wb') as f:
        pickle.dump(split, f)
    log = {'args': vars(a), 'clean_ids': len(clean), 'found': len(found), 'bad': bad_total,
           'train_images': len(split['train']['cids']), 'train_landmarks': len(set(split['train']['cluster'])),
           'val_images': len(split['val']['cids']), 'val_landmarks': len(val_lm),
           'minutes': round((time.time() - t0) / 60, 1), 'per_shard': per_shard}
    (out / 'prepare_log.json').write_text(json.dumps(log, indent=1))
    print(json.dumps({k: v for k, v in log.items() if k != 'per_shard'}, indent=1))


if __name__ == '__main__':
    main()
