#!/usr/bin/env python3
"""Pack the Revisited 1M distractors (R1M) for evaluation.

Reads the 100 official archives (revisitop1m.{1..100}.tar.gz, verified by fetch.sh), resizes every
image so its longer side is at most --size (the Revisited protocol evaluates at long side 1024;
smaller images are kept as they are), re-encodes as JPEG (quality 90) and writes one uncompressed
shard per archive. Member names keep the official relative paths (e.g. 744/74498...jpg), so the
official image list revisitop1m.txt still identifies every image. Nothing is extracted to disk.

Output (<out>/): shards/r1m_NNN.tar, manifest.json (per-shard counts, failures, timing).
Run inside a Slurm allocation (CPU-heavy).
"""
import argparse, io, json, os, tarfile, time
from multiprocessing import Pool
from pathlib import Path

from PIL import Image


def process(job):
    src, dst, size = job
    n, bad, names, t0 = 0, [], [], time.time()
    tmp = dst + '.part'
    with tarfile.open(src, 'r:gz') as tin, tarfile.open(tmp, 'w:') as tout:
        for m in tin:
            if not m.isreg() or not m.name.lower().endswith('.jpg'):
                continue
            parts = [p for p in m.name.split('/') if p not in ('', '.')]
            if m.name.startswith('/') or '..' in parts:
                raise ValueError(f"unsafe member {m.name}")
            rel = '/'.join(parts[-2:])                       # official list format: dir/hash.jpg
            try:
                im = Image.open(io.BytesIO(tin.extractfile(m).read()))
                im.draft('RGB', (size, size))
                im = im.convert('RGB')
                if max(im.size) > size:
                    s = size / max(im.size)
                    im = im.resize((max(1, round(im.size[0] * s)), max(1, round(im.size[1] * s))), Image.BILINEAR)
                buf = io.BytesIO(); im.save(buf, 'JPEG', quality=90)
            except Exception as e:
                bad.append(f"{rel}: {e}")
                continue
            data = buf.getvalue()
            info = tarfile.TarInfo(rel); info.size, info.mtime = len(data), 0
            tout.addfile(info, io.BytesIO(data))
            names.append(rel); n += 1
    os.replace(tmp, dst)
    return os.path.basename(src), n, bad, names, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='/home/atuin/b268dc/b268dc21/datasets/revisitop1m')
    ap.add_argument('--out', default='/home/atuin/b268dc/b268dc21/datasets/revisitop1m/r1m1024')
    ap.add_argument('--size', type=int, default=1024)
    ap.add_argument('--workers', type=int, default=32)
    ap.add_argument('--only', type=int, nargs='*', help='debug: archive numbers')
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    (out / 'shards').mkdir(parents=True, exist_ok=True)
    nums = a.only or list(range(1, 101))
    missing = [i for i in nums if not (src / 'raw' / f'revisitop1m.{i}.tar.gz.ok').exists()]
    if missing:
        raise SystemExit(f"archives not downloaded/verified: {missing[:5]} ({len(missing)})")
    jobs = [(str(src / 'raw' / f'revisitop1m.{i}.tar.gz'), str(out / 'shards' / f'r1m_{i:03d}.tar'), a.size) for i in nums]
    official = set(l.strip() for l in open(src / 'revisitop1m.txt') if l.strip())
    t0, man, all_names = time.time(), {}, []
    with Pool(min(a.workers, len(jobs))) as pool:
        for k, (name, n, bad, names, sec) in enumerate(pool.imap_unordered(process, jobs)):
            man[name] = {'images': n, 'failed': bad, 'seconds': round(sec)}
            all_names += names
            print(f"{k + 1}/{len(jobs)} {name}: {n} images, {len(bad)} failed, {sec / 60:.1f} min", flush=True)
    s = set(all_names)
    log = {'size': a.size, 'archives': len(jobs), 'images': len(all_names), 'unique': len(s),
           'failed': sum(len(v['failed']) for v in man.values()),
           'in_official_list': len(s & official), 'official_list': len(official),
           'not_in_list': sorted(s - official)[:20], 'minutes': round((time.time() - t0) / 60, 1), 'per_archive': man}
    (out / ('manifest.json' if not a.only else 'manifest_debug.json')).write_text(json.dumps(log, indent=1))
    print(json.dumps({k: v for k, v in log.items() if k != 'per_archive'}, indent=1))


if __name__ == '__main__':
    main()
