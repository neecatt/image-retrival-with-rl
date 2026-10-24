"""
Verified-QE re-query reward for Track V-QE training (PROGRESS Q11).

Final ranking after B checks: verified matches first (by inliers), every other image by the re-query
score s(q, x) + sum over verified matches m of s(m, x) (the order of norm(q + sum g_m) . g_x, since the
norm is common to all x). Training queries are GLDv2 val images; exact re-ranking of a 931k gallery
per rollout is too costly, so each query keeps a subset X = its top-M images by s(q, .), all its
positives and all its expansion candidates, with s(q, x) and s(c, x) for every candidate c. The reward
is the official AP over X (all positives are in X; negatives outside the top-M are assumed to stay below
the positives, a training-only approximation). Benchmarks are evaluated with the exact full re-query.
"""
from typing import Dict

import torch

from training.verify_policy import ap_top


@torch.no_grad()
def build(G: torch.Tensor, q_ids, cand, labels, n_pool: int, M: int = 1000, chunk: int = 256) -> Dict[str, torch.Tensor]:
    """G (N, d) unit gallery embeddings on GPU (val pool first, then distractors); q_ids (Q,) gallery ids of
    the queries; cand (Q, C) candidate gallery ids (-1 padding); labels (n_pool,) val-pool landmark ids.
    Returns X ids, s_qx, rel_x (1 / 0 / -1 padding), S (Q, C, X) fp16 and cpos (Q, C) (position of each
    candidate in X, -1 for padding)."""
    dev = G.device
    lab = torch.full((G.shape[0],), -1, dtype=torch.long, device=dev)
    lab[:n_pool] = torch.as_tensor(labels, device=dev)
    q_ids = torch.as_tensor(q_ids, device=dev); cand = torch.as_tensor(cand, device=dev)
    Q, C = cand.shape
    rows = []
    for s in range(0, Q, chunk):
        qb = q_ids[s:s + chunk]
        sim = G[qb] @ G.t()
        sim[torch.arange(len(qb), device=dev), qb] = float('-inf')                       # the query itself
        top = sim.topk(M, dim=1).indices
        for b in range(len(qb)):
            q = int(qb[b])
            pos = torch.nonzero(lab == lab[q]).flatten(); pos = pos[pos != q]
            cb = cand[s + b]; cb = cb[cb >= 0]
            rows.append(torch.unique(torch.cat([top[b], pos, cb])))
    X = max(len(r) for r in rows)
    xid = torch.full((Q, X), -1, dtype=torch.long, device=dev)
    for i, r in enumerate(rows):
        xid[i, :len(r)] = r
    valid = xid >= 0
    xg = xid.clamp(min=0)
    s_qx = torch.empty(Q, X, device=dev); S = torch.zeros(Q, C, X, dtype=torch.float16, device=dev)
    for s in range(0, Q, chunk):
        sl = slice(s, s + chunk)
        gx = G[xg[sl]]                                                                    # (b, X, d)
        s_qx[sl] = torch.einsum('bd,bxd->bx', G[q_ids[sl]], gx)
        gc = G[cand[sl].clamp(min=0)]                                                     # (b, C, d)
        S[sl] = torch.einsum('bcd,bxd->bcx', gc, gx).half()
    s_qx = s_qx.masked_fill(~valid, float('-inf'))
    rel = torch.where(valid, (lab[xg] == lab[q_ids][:, None]).to(torch.int8), torch.full_like(xid, -1, dtype=torch.int8))
    # position of each candidate inside X
    cpos = torch.full((Q, C), -1, dtype=torch.long, device=dev)
    for s in range(0, Q, chunk):
        sl = slice(s, s + chunk)
        eq = (cand[sl][:, :, None] == xid[sl][:, None, :]) & (cand[sl][:, :, None] >= 0)
        hit = eq.any(2)
        cpos[sl] = torch.where(hit, eq.float().argmax(2), torch.full_like(cpos[sl], -1))
    npos = (rel == 1).sum(1)
    return {'rq_s': s_qx, 'rq_rel': rel, 'rq_S': S, 'rq_cpos': cpos, 'rq_npos': npos}


def requery_ap(d: Dict[str, torch.Tensor], idx, matched: torch.Tensor, inliers: torch.Tensor, extra=None):
    """AP after the re-query for queries idx; matched (N, C) bool verified matches.
    extra (N, K, X) optional per-row score increments (one-step gains: K alternative additions)."""
    S = d['rq_S'][idx]                                                                   # (N, C, X)
    C = matched.shape[1]
    S = S[:, :C]
    score = d['rq_s'][idx] + torch.bmm(matched.float()[:, None, :], S.float()).squeeze(1)
    bonus = torch.full_like(score, float('-inf'))
    cp = d['rq_cpos'][idx][:, :C]
    val = torch.where(matched & (cp >= 0), 1e6 + inliers[:, :C].float(), torch.full_like(score[:, :C], float('-inf')))
    bonus.scatter_reduce_(1, cp.clamp(min=0), val, reduce='amax')
    key = torch.maximum(score, bonus)
    rel = d['rq_rel'][idx]
    order = key.argsort(1, descending=True)
    return ap_top(torch.gather(rel.long(), 1, order), d['rq_npos'][idx].float())


def requery_gains(d, idx, matched, inliers, avail, match_of):
    """Exact one-step re-query AP gain of verifying each available candidate (N, C); 0 elsewhere.
    match_of (N, C) bool: whether that candidate would verify as a match (a non-match changes nothing)."""
    N, C = matched.shape
    S = d['rq_S'][idx][:, :C].float()                                                   # (N, C, X)
    score = d['rq_s'][idx] + torch.bmm(matched.float()[:, None, :], S).squeeze(1)     # (N, X)
    cp = d['rq_cpos'][idx][:, :C]
    inl = inliers[:, :C].float()
    bonus = torch.full_like(score, float('-inf'))
    val = torch.where(matched & (cp >= 0), 1e6 + inl, torch.full_like(inl, float('-inf')))
    bonus.scatter_reduce_(1, cp.clamp(min=0), val, reduce='amax')
    rel = d['rq_rel'][idx].long(); npos = d['rq_npos'][idx].float()
    base = ap_top(torch.gather(rel, 1, torch.maximum(score, bonus).argsort(1, descending=True)), npos)
    cand_new = avail & match_of & ~matched                                               # only these change the ranking
    out = torch.zeros(N, C, device=matched.device)
    rows, cols = torch.nonzero(cand_new, as_tuple=True)
    for s in range(0, len(rows), 4096):
        r, c = rows[s:s + 4096], cols[s:s + 4096]
        sc = score[r] + S[r, c]                                                          # add candidate c to the query
        bn = bonus[r].clone()
        bn[torch.arange(len(r), device=bn.device), cp[r, c].clamp(min=0)] = torch.where(cp[r, c] >= 0, 1e6 + inl[r, c], bn[torch.arange(len(r), device=bn.device), cp[r, c].clamp(min=0)])
        ap = ap_top(torch.gather(rel[r], 1, torch.maximum(sc, bn).argsort(1, descending=True)), npos[r])
        out[r, c] = ap - base[r]
    return out
