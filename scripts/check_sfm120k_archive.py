#!/usr/bin/env python3
"""Stream-list the packed SfM-120k archive (no extraction) and compare it with the metadata pickle.

Checks member path safety, the hash-based path layout expected by data/dataset_sfm120k.py,
and that every train/val image hash in the pickle is present in the archive.
"""
import json, pickle, sys, tarfile, collections

class _Safe(pickle.Unpickler):
    def find_class(self, module, name):
        if module.split('.')[0] in ('numpy', 'builtins', 'collections'):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked {module}.{name}")

def main(archive, pkl, out_json):
    with open(pkl, 'rb') as f:
        db = _Safe(f).load()
    need = {s: set(db[s]['cids']) for s in ('train', 'val')}

    members, unsafe, shapes = set(), [], collections.Counter()
    n_files = 0
    with tarfile.open(archive, 'r:gz') as tar:
        for m in tar:
            if m.isdir():
                continue
            n_files += 1
            parts = m.name.split('/')
            if not m.isreg() or m.name.startswith('/') or '..' in parts:
                unsafe.append(m.name)
                continue
            shapes[(len(parts), parts[-1].endswith('.jpg'))] += 1
            members.add(m.name)
    print('regular files:', n_files, 'unsafe:', len(unsafe), 'shape (depth, has .jpg):', dict(shapes))
    print('sample members:', sorted(members)[:3])

    # Find the prefix that makes archive members match the loader formula
    # jpg/{h[-2:]}/{h[-4:-2]}/{h[-6:-4]}/{h}
    def rel(h):
        return f"{h[-2:]}/{h[-4:-2]}/{h[-6:-4]}/{h}"
    sample = next(iter(need['train']))
    cand = [m for m in members if m.endswith(rel(sample))]
    prefix = cand[0][:-len(rel(sample))] if cand else None
    print('layout prefix before hash path:', repr(prefix))
    result = {'regular_files': n_files, 'unsafe': len(unsafe), 'prefix': prefix}
    for split, hashes in need.items():
        missing = [h for h in hashes if prefix is None or (prefix + rel(h)) not in members]
        result[split] = {'needed': len(hashes), 'missing': len(missing), 'missing_sample': missing[:5]}
        print(split, 'needed', len(hashes), 'missing', len(missing))
    extra = len(members) - sum(len(h) - r['missing'] for h, r in
                               ((need[s], result[s]) for s in need)) if prefix is not None else None
    result['archive_files_not_in_pkl'] = extra
    print('archive files not referenced by train/val:', extra)
    with open(out_json, 'w') as f:
        json.dump(result, f, indent=2)
    return 1 if unsafe or any(result[s]['missing'] for s in need) else 0

if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:4]))
