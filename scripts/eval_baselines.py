#!/usr/bin/env python3
"""Evaluate models on SfM-120k val (selection metric) and ROxford5k/RParis6k (official scoring).

Models: 'imagenet' = off-the-shelf torchvision ResNet50 (IMAGENET1K_V1) + GeM(p=3), 2048-d, no training;
        any other value = path to a project checkpoint (EmbeddingNet, config stored inside).
Revisited images are evaluated at single scale with two resize rules, to quantify the protocol choice:
  short: shorter side = SIZE (current evaluate.py behaviour)
  long:  longer side = SIZE (common practice for the Revisited benchmarks)
Query images are cropped to their bounding box. Writes one JSON with all numbers.
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dataset import LongSideResize, RevisitedDataset, get_transforms  # noqa: E402
from data.dataset_sfm120k import load_sfm120k, load_gldv2c                           # noqa: E402
from evaluate import create_model_from_checkpoint                       # noqa: E402
from evaluation.revisited_ap import revisited_map                       # noqa: E402
from evaluation.sfm_val import evaluate_sfm_val                         # noqa: E402
from models.embedding_net import EmbeddingNet                           # noqa: E402

NORM = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def build_model(spec, device):
    if spec == 'imagenet':
        m = EmbeddingNet(architecture='resnet50', pretrained=True, pooling='gem',
                         embedding_dim=None, normalize_embeddings=True, pooling_kwargs={'p': 3.0})
        return m.to(device).eval()
    return create_model_from_checkpoint(spec, device)


@torch.no_grad()
def embed(model, dataset, device, workers):
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=workers, pin_memory=True)
    vecs = [model(x.to(device, non_blocking=True)).float().cpu() for x, _, _ in loader]
    return F.normalize(torch.cat(vecs), dim=1).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--models', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--size', type=int, default=1024)
    ap.add_argument('--val_size', type=int, default=1024)
    ap.add_argument('--val_set', default='sfm120k', choices=['sfm120k', 'gldv2c'],
                    help='held-out validation set reported next to the benchmarks (key <val_set>_val; sfm_val for sfm120k)')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--datasets', nargs='+', default=['roxford5k', 'rparis6k'])
    ap.add_argument('--resize', nargs='+', default=['short', 'long'], choices=['short', 'long'])
    a = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root = Path(a.data_root)

    loader_fn = load_gldv2c if a.val_set == 'gldv2c' else load_sfm120k
    _, sfm_val = loader_fn(str(root / a.val_set), val_transform=get_transforms(a.val_size, mode='eval'))
    val_key = 'sfm_val' if a.val_set == 'sfm120k' else f'{a.val_set}_val'
    resize_tf = {'short': T.Compose([T.Resize(a.size), T.ToTensor(), NORM]),
                 'long': T.Compose([LongSideResize(a.size), T.ToTensor(), NORM])}

    results = {}
    for spec in a.models:
        t0 = time.time()
        model = build_model(spec, device)
        r = {val_key: evaluate_sfm_val(model, sfm_val, device, num_workers=a.workers)}
        for ds in a.datasets:
            handler = RevisitedDataset(str(root / ds), ds)
            gnd = handler.get_ground_truth()
            for mode in a.resize:
                tf = resize_tf[mode]
                Q = embed(model, handler.get_query_dataset(tf), device, a.workers)
                G = embed(model, handler.get_database_dataset(tf), device, a.workers)
                r[f'{ds}_{mode}{a.size}'] = m = revisited_map(Q @ G.T, gnd)
                print(f"[{spec}] {ds} {mode}{a.size}: Medium {m['mAP_medium']:.2f}  Hard {m['mAP_hard']:.2f}", flush=True)
        r['seconds'] = round(time.time() - t0, 1)
        results[spec] = r
        Path(a.out).write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
