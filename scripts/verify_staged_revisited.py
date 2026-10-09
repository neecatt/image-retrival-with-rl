#!/usr/bin/env python3
"""Check that every image named by the converted Revisited pickles exists under a staged root."""
import pickle, sys
from pathlib import Path
class _Safe(pickle.Unpickler):
    def find_class(self, m, n):
        if m.split('.')[0] in ('numpy', 'builtins', 'collections'):
            return super().find_class(m, n)
        raise pickle.UnpicklingError(m)
root = Path(sys.argv[1]); bad = 0
for name in sys.argv[2:]:
    cfg = _Safe(open(root / name / f'{name}.pkl', 'rb')).load()
    names = [x if x.endswith('.jpg') else x + '.jpg' for x in cfg['db_images'] + cfg['query_images']]
    miss = [x for x in names if not (root / name / 'jpg' / x).is_file()]
    on_disk = sum(1 for _ in (root / name / 'jpg').iterdir())
    print(f"{name}: referenced={len(set(names))} missing={len(miss)} files_on_disk={on_disk} {miss[:3]}")
    bad += len(miss)
sys.exit(1 if bad else 0)
