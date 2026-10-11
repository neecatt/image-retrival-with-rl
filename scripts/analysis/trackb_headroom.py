#!/usr/bin/env python3
# Usage: python scripts/analysis/trackb_headroom.py data/archives/sfm120k/retrieval-SfM-120k.pkl (CPU, reads results/a0_label_noise)
"""Track B headroom: AP of the global top-20 ranking vs after verifier re-ranking, from A0/A0b pairs."""
import sys, numpy as np
sys.path.insert(0, '.')
import pickle
db = pickle.load(open(sys.argv[1], 'rb'))
class D: pass
def mk(split):   # same label mapping as data/dataset_sfm120k.py (all images present when staged)
    cl = db[split]['cluster']; m = {c: i for i, c in enumerate(sorted(set(cl)))}
    d = D(); d.labels = [m[c] for c in cl]; return d
tr, va = mk('train'), mk('val')
for name, ds, f in [('train', tr, 'results/a0_label_noise/a0_pairs.npz'), ('val', va, 'results/a0_label_noise/val/a0_pairs.npz')]:
    lab = np.asarray(ds.labels); cnt = np.bincount(lab)
    d = np.load(f); q, rank, same, inl = d['q'], d['rank'], d['same_label'], d['inliers']
    for T in (25, 100):
        ap_g, ap_r, rec = [], [], []
        for x in np.unique(q):
            m = q == x; o = np.argsort(rank[m]); rel = same[m][o].astype(float); il = inl[m][o]
            npos = cnt[lab[x]] - 1
            if npos == 0: continue
            def ap(r): 
                h = np.cumsum(r); return float((h / np.arange(1, len(r) + 1) * r).sum() / npos)
            ver = il >= T
            new = np.concatenate([np.where(ver)[0][np.argsort(-il[ver], kind='stable')], np.where(~ver)[0]])
            ap_g.append(ap(rel)); ap_r.append(ap(rel[new])); rec.append(rel.sum() / min(npos, 20))
        ap_g, ap_r, rec = map(np.array, (ap_g, ap_r, rec))
        print(f"{name} T={T}: queries {len(ap_g)}  AP@20 global {ap_g.mean():.4f}  after rerank {ap_r.mean():.4f}  "
              f"rerank changes AP in {(np.abs(ap_r - ap_g) > 1e-9).mean():.3f} of queries  "
              f"recall@20 (of min(npos,20)) {rec.mean():.4f}  queries with recall@20<1 {(rec < 1 - 1e-9).mean():.3f}  "
              f"reranked-AP std across queries {ap_r.std():.3f}")
