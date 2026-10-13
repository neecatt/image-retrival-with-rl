#!/usr/bin/env python3
"""Extract packed dataset archives into the allocated job's node-local TMPDIR.

Layout produced (matches config dataset.root_dir and the loaders):
    $TMPDIR/data/raw/<name>/jpg/...  plus metadata pickles.
Fails if TMPDIR is unset, not writable, or lacks space/inodes. Archive members are
validated (regular files, no absolute/.. paths) and written manually, never via extractall.
"""
import argparse, json, os, re, shutil, sys, tarfile, time
from pathlib import Path

SRC = Path(os.environ.get('DATASET_ARCHIVES', '/home/vault/b268dc/b268dc21/datasets'))
HASH_RE = re.compile(r'^[0-9a-f]{32}$')
GLD_RE = re.compile(r'^[0-9a-f]/[0-9a-f]/[0-9a-f]/[0-9a-f]{16}\.jpg$')
# name -> (archives, metadata files, extra bytes needed estimate, flatten?)
SPEC = {
    'roxford5k': (['oxbuild_images-v1.tgz'], ['roxford5k.pkl', 'gnd_roxford5k.pkl'], 3 << 30),
    'rparis6k':  (['paris_1-v1.tgz', 'paris_2-v1.tgz'], ['rparis6k.pkl', 'gnd_rparis6k.pkl'], 5 << 30),
    'sfm120k':   (['ims.tar.gz'], ['retrieval-SfM-120k.pkl'], 60 << 30),
    # GLDv2-clean at long side 512 (scripts/prepare_gldv2.py), 500 uncompressed shards
    'gldv2c':    ([f'shards/gldv2c_{n:03d}.tar' for n in range(500)], ['gldv2c.pkl'], 120 << 30),
    # noisy-label GLDv2 pool (scripts/prepare_gldv2_noisy.py)
    'gldv2n':    ([f'shards/gldv2n_{n:03d}.tar' for n in range(500)], ['gldv2n.pkl'], 80 << 30),
}

def die(msg):
    sys.exit(f"ERROR: {msg}")

def extract(archive, jpg, name):
    n = 0
    with tarfile.open(archive, 'r:*') as tar:
        for m in tar:
            if m.isdir():
                continue
            parts = [p for p in m.name.split('/') if p not in ('', '.')]
            if not m.isreg() or m.name.startswith('/') or '..' in parts or not parts:
                die(f"unsafe member {m.name!r} in {archive}")
            if name in ('gldv2c', 'gldv2n'):
                if not GLD_RE.match(m.name):
                    die(f"unexpected gldv2c member {m.name!r}")
                rel = Path(*parts)
            elif name == 'sfm120k':
                if len(parts) != 4 or not HASH_RE.match(parts[-1]):
                    die(f"unexpected sfm120k member {m.name!r}")
                rel = Path(*parts)
            else:  # revisited: flatten landmark subdirs to jpg/<file>.jpg
                if not parts[-1].lower().endswith('.jpg'):
                    die(f"unexpected non-jpg member {m.name!r}")
                rel = Path(parts[-1])
            dst = jpg / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            with open(dst, "wb") as out:
                shutil.copyfileobj(tar.extractfile(m), out)
            n += 1
    return n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('datasets', nargs='+', choices=sorted(SPEC))
    ap.add_argument('--config-in'); ap.add_argument('--config-out')
    a = ap.parse_args()

    tmp = os.environ.get('TMPDIR')
    if not tmp or not os.path.isdir(tmp) or not os.access(tmp, os.W_OK):
        die("TMPDIR is unset or not writable; run inside a Slurm allocation")
    st = os.statvfs(tmp)
    free_b, free_i = st.f_bavail * st.f_frsize, st.f_favail
    need_b = sum(SPEC[d][2] for d in a.datasets)
    print(f"TMPDIR={tmp} free={free_b>>30} GiB, {free_i} inodes; estimated need={need_b>>30} GiB")
    if free_b < need_b or free_i < 2_000_000:
        die("insufficient node-local space or inodes")

    root = Path(tmp) / 'data' / 'raw'
    manifest = {}
    for name in a.datasets:
        archives, metas, _ = SPEC[name]
        for f in archives + metas:
            if not (SRC / name / f).is_file():
                die(f"missing {SRC/name/f}")
        jpg = root / name / 'jpg'
        jpg.mkdir(parents=True, exist_ok=True)
        t0, n = time.time(), 0
        for arc in archives:
            n += extract(SRC / name / arc, jpg, name)
        for f in metas:
            shutil.copy2(SRC / name / f, root / name / f)
        manifest[name] = {'files': n, 'archives': {x: (SRC/name/x).stat().st_size for x in archives},
                          'seconds': round(time.time() - t0, 1)}
        print(f"{name}: extracted {n} files in {manifest[name]['seconds']} s")
    (root / 'stage_manifest.json').write_text(json.dumps(manifest, indent=2))

    if a.config_in:
        txt = Path(a.config_in).read_text()
        new, k = re.subn(r'(?m)^(\s*root_dir:\s*).*$', lambda m: f'{m.group(1)}"{root}"', txt, count=1)
        if k != 1:
            die("root_dir not found in config")
        Path(a.config_out).write_text(new)
        print(f"wrote run config {a.config_out} with root_dir={root}")
    print(f"STAGED_ROOT={root}")

if __name__ == '__main__':
    main()
