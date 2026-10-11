"""
Held-out checkpoint-selection metric on the SfM-120k (or GLDv2-clean) validation split.

Every validation image is a query against all other validation images
(the query itself is excluded); positives share its 3D-model cluster.
AP uses the same trapezoidal integration as the official Revisited code,
so selection and final benchmark scoring follow one AP definition.

Deterministic: fixed image order, eval transforms, no sampling.
"""

from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


@torch.no_grad()
def extract_embeddings(model, dataset, device, batch_size: int = 16, num_workers: int = 4):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    model.eval()
    vecs, labels = [], []
    for images, lab, _ in tqdm(loader, desc="SfM val embeddings"):
        vecs.append(model(images.to(device, non_blocking=True)).float())
        labels.append(torch.as_tensor(lab, device=device))
    return torch.cat(vecs), torch.cat(labels)


def leave_one_out_map(embeddings: torch.Tensor, labels: torch.Tensor, chunk: int = 512) -> Dict[str, float]:
    """mAP and any-positive success@1 with each item queried against all others."""
    n = embeddings.size(0)
    aps, top1 = [], []
    for start in range(0, n, chunk):
        q = slice(start, min(start + chunk, n))
        sim = embeddings[q] @ embeddings.t()
        rows = torch.arange(sim.size(0), device=sim.device)
        sim[rows, rows + start] = float('-inf')            # drop the query itself
        order = sim.argsort(dim=1, descending=True)[:, :-1]  # self is ranked last
        rel = (labels[order] == labels[q].unsqueeze(1)).float()
        npos = rel.sum(1)
        hits_before = rel.cumsum(1) - rel                  # j: positives ranked above
        rank = torch.arange(rel.size(1), device=rel.device, dtype=torch.float32).expand_as(rel)
        p0 = torch.where(rank == 0, torch.ones_like(rank), hits_before / rank.clamp(min=1))
        p1 = (hits_before + 1) / (rank + 1)
        ap = ((p0 + p1) / 2 * rel).sum(1) / npos.clamp(min=1)
        valid = npos > 0
        aps.append(ap[valid])
        top1.append(rel[valid, 0])
    aps, top1 = torch.cat(aps), torch.cat(top1)
    return {'sfm_val_mAP': aps.mean().item(), 'sfm_val_success@1': top1.mean().item(),
            'sfm_val_queries': int(aps.numel())}


def graded_leave_one_out(embeddings: torch.Tensor, grades: dict, ks=(10, 100), chunk: int = 512) -> Dict[str, float]:
    """Graded nDCG@k with each val image as query against all others (self excluded).
    grades: {(i, j): relevance level 1-4} for same-landmark pairs (data.graded_lists.load_oracle);
    missing pairs are level 0 (val verifies every same-landmark pair, so only other landmarks are missing)."""
    from losses.graded import ndcg_at_k
    n = embeddings.size(0)
    G = torch.zeros(n, n, dtype=torch.float32, device=embeddings.device)
    if grades:
        ij = torch.tensor(list(grades.keys()), device=embeddings.device)
        G[ij[:, 0], ij[:, 1]] = torch.tensor(list(grades.values()), dtype=torch.float32, device=embeddings.device)
    out = {k: [] for k in ks}
    for start in range(0, n, chunk):
        q = slice(start, min(start + chunk, n))
        sim = embeddings[q] @ embeddings.t()
        rows = torch.arange(sim.size(0), device=sim.device)
        sim[rows, rows + start] = float('-inf')
        for k in ks:
            out[k].append(ndcg_at_k(sim, G[q], k))
    res = {}
    for k in ks:
        v = torch.cat(out[k])
        has = G.sum(1) > 0
        res[f'sfm_val_graded_ndcg@{k}'] = v[has].mean().item()
    return res


def evaluate_sfm_val(model, val_dataset, device, batch_size: int = 16, num_workers: int = 4,
                     grades: dict = None) -> Dict[str, float]:
    emb, lab = extract_embeddings(model, val_dataset, device, batch_size, num_workers)
    metrics = leave_one_out_map(emb, lab)
    if grades is not None:
        metrics.update(graded_leave_one_out(emb, grades))
        print("  val (leave-one-out) graded: " + "  ".join(f"{k.split('_')[-1]}={100 * v:.2f}"
                                                    for k, v in metrics.items() if 'graded' in k))
    print(f"  val (leave-one-out): mAP={100 * metrics['sfm_val_mAP']:.2f}  "
          f"success@1={100 * metrics['sfm_val_success@1']:.2f}  "
          f"({metrics['sfm_val_queries']} queries)")
    return metrics
