#!/usr/bin/env python3
"""Reproduce SuperGlobal (Shao et al., ICCV 2023) global retrieval and reranking with the public CVNet
weights, as the strong base for budgeted verification (branch feat/superglobal-verification).

Model and reranking modules are imported from the official SuperGlobal repository (MIT licence, cloned
outside this repo, --sg_repo); the CVNet ResNet weights come from the CVNet authors' release (no licence
file in that repository; used for research evaluation only). Preprocessing follows SuperGlobal's
test/dataset.py: OpenCV BGR at full resolution, query bounding-box crop with int() truncation, BGR mean/std
normalisation, 3 internal scales with GeM+, regional GeM and scale-GeM (all on, as in their README command).

Outputs (in --out): <dataset>_q.npy, <dataset>_db.npy (fp16, L2-normalised), optionally r1m.npy, and
results.json with global and SuperGlobal-reranked mAP (official Revisited protocol, Medium / Hard),
with and without R1M.
"""
import argparse, io, json, sys, tarfile, time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dataset import RevisitedDataset                                       # noqa: E402
from evaluation.revisited_ap import official_ap                                 # noqa: E402

MEAN, SD = [0.406, 0.456, 0.485], [0.225, 0.224, 0.229]                          # BGR order, as SuperGlobal


def prep(im_bgr):
    im = im_bgr.astype(np.float32).transpose(2, 0, 1) / 255.0
    for c in range(3):
        im[c] = (im[c] - MEAN[c]) / SD[c]
    return torch.from_numpy(im)


class Bench(Dataset):
    def __init__(self, h, split):
        self.h, self.split = h, split
        self.names = h.cfg['query_images'] if split == 'query' else h.cfg['db_images']

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        im = cv2.imread(str(self.h.root_dir / 'jpg' / self.names[i]))
        if self.split == 'query':
            b = self.h.cfg['gnd'][i]['bbx']
            im = im[int(b[1]):int(b[3]), int(b[0]):int(b[2])]
        return prep(im)


class R1M(IterableDataset):
    def __init__(self, r1m_dir, index):
        self.files = [Path(r1m_dir) / 'raw' / f'revisitop1m.{i}.tar.gz' for i in range(1, 101)]
        self.index = index

    def __iter__(self):
        w = get_worker_info()
        for f in (self.files if w is None else self.files[w.id::w.num_workers]):
            with tarfile.open(f, 'r:gz') as t:
                for m in t:
                    if not m.isreg() or not m.name.lower().endswith('.jpg'):
                        continue
                    data = t.extractfile(m).read()
                    im = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                    trunc = 0
                    if im is None:                       # truncated official JPEG: decode with padding (as eval_r1m.py)
                        ImageFile.LOAD_TRUNCATED_IMAGES = True
                        im = np.asarray(Image.open(io.BytesIO(data)).convert('RGB'))[:, :, ::-1].copy()
                        ImageFile.LOAD_TRUNCATED_IMAGES = False
                        trunc = 1
                    yield prep(im), self.index['/'.join(m.name.split('/')[-2:])], trunc


@torch.no_grad()
def extract(model, loader, dev, amp, n=None, with_index=False):
    feats, idxs, ntr = [], [], 0
    t0 = time.time()
    for k, batch in enumerate(loader):
        x = batch[0] if with_index else batch
        with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
            d = model.extract_global_descriptor(x.to(dev), True, True, True, 3)
        feats.append(F.normalize(d.float().reshape(1, -1), dim=1).cpu())
        if with_index:
            idxs.append(int(batch[1])); ntr += int(batch[2])
        if with_index and k % 50000 == 0:
            print(f"  R1M {k} images, {k / max(time.time() - t0, 1):.1f} img/s", flush=True)
    f = torch.cat(feats)
    if with_index:
        out = torch.zeros(n, f.shape[1])
        out[torch.as_tensor(idxs)] = f
        return out, ntr, len(set(idxs))
    return f


def maps_from_ranks(ranks, gnd):
    """ranks (N, Q) gallery indices best first, as SuperGlobal; official AP, Medium and Hard."""
    out = {}
    for prot in ('medium', 'hard'):
        aps = []
        for q, e in enumerate(gnd):
            easy, hard, junk = set(map(int, e['easy'])), set(map(int, e['hard'])), set(map(int, e['junk']))
            pos, jk = (easy | hard, junk) if prot == 'medium' else (hard, junk | easy)
            if pos:
                aps.append(official_ap(ranks[:, q], pos, jk))
        out[prot] = 100 * float(np.mean(aps))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sg_repo', default='/home/hpc/b268dc/b268dc21/external/SuperGlobal')
    ap.add_argument('--weights', default='/home/vault/b268dc/b268dc21/external_weights/cvnet/pretrained_weights/CVPR2022_CVNet_R101.pyth')
    ap.add_argument('--depth', type=int, default=101)
    ap.add_argument('--data_root', required=True, help='staged roxford5k / rparis6k')
    ap.add_argument('--r1m_dir', default='/home/atuin/b268dc/b268dc21/datasets/revisitop1m')
    ap.add_argument('--r1m', action='store_true', help='also extract / load R1M and evaluate +1M')
    ap.add_argument('--amp', action='store_true', help='fp16 autocast for extraction')
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    dev = torch.device('cuda')
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, a.sg_repo)
    from model.CVNet_Rerank_model import CVNet_Rerank                           # SuperGlobal repo
    from modules.reranking.MDescAug import MDescAug
    from modules.reranking.RerankwMDA import RerankwMDA
    model = CVNet_Rerank(a.depth, 2048, True)
    sd = torch.load(a.weights, map_location='cpu')['model_state']
    missing, unexpected = model.load_state_dict({k.replace('globalmodel', 'encoder_q'): v for k, v in sd.items()}, strict=False)
    enc_missing = [k for k in missing if k.startswith('encoder_q')]
    print(f"weights: {len(sd)} tensors; encoder_q keys missing: {len(enc_missing)} {enc_missing[:5]}; unexpected: {len(unexpected)}", flush=True)
    model = model.to(dev).eval()
    res = {'weights': a.weights, 'depth': a.depth, 'amp': a.amp, 'encoder_missing': enc_missing}

    R = None
    if a.r1m:
        cache = out / 'r1m.npy'
        if cache.exists():
            R = torch.from_numpy(np.load(cache).astype(np.float32))
        else:
            names = [l.strip() for l in open(Path(a.r1m_dir) / 'revisitop1m.txt') if l.strip()]
            loader = DataLoader(R1M(a.r1m_dir, {n: i for i, n in enumerate(names)}), batch_size=None, num_workers=a.workers)
            t0 = time.time()
            R, ntr, nseen = extract(model, loader, dev, a.amp, n=len(names), with_index=True)
            if nseen != len(names):
                raise RuntimeError(f"{len(names) - nseen} R1M images not found")
            np.save(cache, R.half().numpy())
            res['r1m'] = {'images': nseen, 'truncated': ntr, 'minutes': round((time.time() - t0) / 60, 1)}
            print('R1M', res['r1m'], flush=True)
        R = R.to(dev)
    mda, rmda = MDescAug(), RerankwMDA()
    for name in ('roxford5k', 'rparis6k'):
        h = RevisitedDataset(str(Path(a.data_root) / name), name)
        gnd = h.get_ground_truth()
        t0 = time.time()
        Q = extract(model, DataLoader(Bench(h, 'query'), batch_size=None, num_workers=a.workers), dev, a.amp)
        X = extract(model, DataLoader(Bench(h, 'db'), batch_size=None, num_workers=a.workers), dev, a.amp)
        np.save(out / f'{name}_q.npy', Q.half().numpy()); np.save(out / f'{name}_db.npy', X.half().numpy())
        res[f'{name}_extract_minutes'] = round((time.time() - t0) / 60, 1)
        Q, X = Q.to(dev), X.to(dev)
        for suffix, G in (('', X), ('_r1m', torch.cat([X, R]) if R is not None else None)):
            if G is None:
                continue
            ranks = torch.argsort(-(G @ Q.t()), dim=0)
            r = {'global': maps_from_ranks(ranks.cpu().numpy(), gnd)}
            with torch.no_grad():
                a1 = mda(G, Q, ranks)
                rr = rmda(ranks, *a1)
            r['superglobal_rerank'] = maps_from_ranks(rr.cpu().numpy(), gnd)
            res[name + suffix] = r
            print(name + suffix, json.dumps(r), flush=True)
    (out / 'results.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
