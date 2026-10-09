#!/usr/bin/env python3
"""Convert official Revisited gnd_<name>.pkl (imlist/qimlist/gnd) into the <name>.pkl
format read by data/dataset.py (db_images/query_images/gnd). Contents are not altered."""
import pickle, sys
from pathlib import Path

class _Safe(pickle.Unpickler):
    def find_class(self, module, name):
        if module.split('.')[0] in ('numpy', 'builtins', 'collections'):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked {module}.{name}")

def convert(dataset_dir, name):
    src = Path(dataset_dir) / f"gnd_{name}.pkl"
    dst = Path(dataset_dir) / f"{name}.pkl"
    with open(src, 'rb') as f:
        d = _Safe(f).load()
    cfg = {'db_images': [str(x) for x in d['imlist']],
           'query_images': [str(x) for x in d['qimlist']],
           'gnd': d['gnd']}
    assert len(cfg['query_images']) == len(cfg['gnd']), "query/gnd length mismatch"
    with open(dst, 'wb') as f:
        pickle.dump(cfg, f, protocol=4)
    print(f"{name}: {len(cfg['db_images'])} db, {len(cfg['query_images'])} queries -> {dst}")

if __name__ == '__main__':
    root = Path(sys.argv[1])
    for name in sys.argv[2:] or ('roxford5k', 'rparis6k'):
        convert(root / name, name)
