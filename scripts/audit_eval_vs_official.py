#!/usr/bin/env python3
"""Compare evaluate.py's Revisited scoring with a re-implementation of the official revisitop protocol.

Official reference: filipradenovic/revisitop python/evaluate.py (compute_ap / compute_map). Re-implemented here
(not imported) so it can be audited; hand-checkable synthetic cases run first.

Usage: audit_eval_vs_official.py                       # synthetic checks only
       audit_eval_vs_official.py EMB.npz GND.pkl       # also score saved embeddings both ways
"""
import pickle, sys
import numpy as np

class _Safe(pickle.Unpickler):
    def find_class(self, m, n):
        if m.split('.')[0] in ('numpy', 'builtins', 'collections'):
            return super().find_class(m, n)
        raise pickle.UnpicklingError(m)

def official_ap(ranks, nres):
    """ranks: 0-indexed positions of positives in the junk-removed ranking (revisitop compute_ap)."""
    ap, recall_step = 0.0, 1.0 / nres
    for j in range(len(ranks)):
        r = ranks[j]
        p0 = 1.0 if r == 0 else float(j) / r
        p1 = float(j + 1) / (r + 1)
        ap += (p0 + p1) * recall_step / 2.0
    return ap

def official_query_ap(order, ok, junk):
    ok, junk = set(ok), set(junk)
    kept = [i for i in order if i not in junk]
    pos = [k for k, i in enumerate(kept) if i in ok]
    return official_ap(pos, len(ok)) if len(ok) else None

def project_query_ap(order, pos_set, junk_set):          # evaluate.py compute_ap_with_junk
    n_rel = rank = 0; ap = 0.0
    for idx in order:
        if idx in junk_set: continue
        rank += 1
        if idx in pos_set:
            n_rel += 1; ap += n_rel / rank
    return 0.0 if n_rel == 0 else ap / len(pos_set)

def synthetic():
    order = [10, 11, 12, 13, 14]
    # positives at ranks 0 and 2 (1-indexed 1 and 3): non-interpolated AP = (1 + 2/3)/2
    print("hand case  project =", round(project_query_ap(order, {10, 12}, set()), 4),
          " official =", round(official_query_ap(order, {10, 12}, set()), 4), "(hand: 0.8333 vs 0.7917)")
    # junk ahead of the positive must be ignored by both
    print("junk case  project =", round(project_query_ap(order, {12}, {10, 11}), 4),
          " official =", round(official_query_ap(order, {12}, {10, 11}), 4), "(both 1.0 expected)")
    # perfect ranking
    print("perfect    project =", project_query_ap(order, {10, 11}, set()),
          " official =", official_query_ap(order, {10, 11}, set()), "(both 1.0 expected)")

def score(Q, G, gnd):
    sim = Q @ G.T
    out = {}
    for proto in ('medium', 'hard'):
        ap_p, ap_o = [], []
        for q, e in enumerate(gnd):
            easy, hard, junk = set(e['easy']), set(e['hard']), set(e['junk'])
            ok, jk = (easy | hard, junk) if proto == 'medium' else (hard, junk | easy)
            if not ok: continue
            order = list(np.argsort(-sim[q]))
            ap_p.append(project_query_ap(order, ok, jk))
            ap_o.append(official_query_ap(order, ok, jk))
        out[proto] = (len(ap_p), 100 * np.mean(ap_p), 100 * np.mean(ap_o))
    return out

if __name__ == '__main__':
    synthetic()
    if len(sys.argv) == 3:
        emb = np.load(sys.argv[1]); gnd = _Safe(open(sys.argv[2], 'rb')).load()['gnd']
        for proto, (n, p, o) in score(emb['query'], emb['gallery'], gnd).items():
            print(f"{proto:6s} queries_used={n}  mAP project={p:.2f}  official={o:.2f}  diff={o-p:+.2f} points")
