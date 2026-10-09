#!/usr/bin/env python3
"""Check every SfM-120k train/val image hash exists at the loader's path under a staged root."""
import pickle, sys
from pathlib import Path
class _Safe(pickle.Unpickler):
    def find_class(self, m, n):
        if m.split('.')[0] in ('numpy', 'builtins', 'collections'):
            return super().find_class(m, n)
        raise pickle.UnpicklingError(m)
root = Path(sys.argv[1]) / 'sfm120k'
db = _Safe(open(root / 'retrieval-SfM-120k.pkl', 'rb')).load()
bad = 0
for split in ('train', 'val'):
    miss = [h for h in db[split]['cids']
            if not (root / 'jpg' / h[-2:] / h[-4:-2] / h[-6:-4] / h).is_file()]
    print(f"{split}: images={len(db[split]['cids'])} missing={len(miss)} {miss[:3]}")
    bad += len(miss)
sys.exit(1 if bad else 0)
