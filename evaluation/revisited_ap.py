"""
Official Revisited Oxford/Paris scoring (revisitop compute_ap / compute_map).

Positives are 'easy'+'hard' (Medium) or 'hard' (Hard); junk is 'junk'
(Medium) or 'junk'+'easy' (Hard). Junk is removed from the ranking, AP
integrates precision with trapezoids, and queries without positives are
skipped. Matches github.com/filipradenovic/revisitop python/evaluate.py
(checked in scripts/audit_eval_vs_official.py).
"""

from typing import Dict, List

import numpy as np


def official_ap(ranked: np.ndarray, positives: set, junk: set) -> float:
    """AP for one query. `ranked` is the gallery ordering, best first."""
    if not positives:
        return float('nan')
    recall_step = 1.0 / len(positives)
    ap, found, rank = 0.0, 0, 0
    for idx in ranked:
        idx = int(idx)
        if idx in junk:
            continue
        if idx in positives:
            precision_0 = 1.0 if rank == 0 else found / rank
            precision_1 = (found + 1) / (rank + 1)
            ap += (precision_0 + precision_1) * recall_step / 2.0
            found += 1
            if found == len(positives):
                break
        rank += 1
    return ap


def revisited_map(sim: np.ndarray, gnd: List[dict]) -> Dict[str, float]:
    """Medium and Hard mAP in percent from a (queries x gallery) similarity matrix."""
    out = {}
    order = np.argsort(-sim, axis=1)
    for protocol in ('medium', 'hard'):
        aps = []
        for q, entry in enumerate(gnd):
            easy, hard, junk = set(map(int, entry['easy'])), set(map(int, entry['hard'])), set(map(int, entry['junk']))
            if protocol == 'medium':
                pos, jk = easy | hard, junk
            else:
                pos, jk = hard, junk | easy
            if pos:
                aps.append(official_ap(order[q], pos, jk))
        out[f'mAP_{protocol}'] = 100.0 * float(np.mean(aps))
        out[f'queries_{protocol}'] = len(aps)
    return out
