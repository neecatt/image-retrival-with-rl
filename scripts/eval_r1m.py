#!/usr/bin/env python3
"""ROxford5k / RParis6k with the Revisited 1M distractors (R1M), official protocol.

Database = the dataset's own database images followed by all R1M images; positives and junk come
from the Revisited ground truth (indices into the database only), so every distractor is a negative.
R1M images are read directly from the verified official archives (revisitop1m.{1..100}.tar.gz on
WORK), in parallel DataLoader workers, without extraction or re-encoding; each image is resized so
its longer side is --size (1024 in the protocol), exactly as the database images.

R1M embeddings are cached per model (fp16, order = official revisitop1m.txt), because they are the
expensive part (~1M forward passes at 1024 px). Results without distractors are reported alongside.
"""
import argparse, hashlib, json, sys, tarfile, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data.dataset import LongSideResize, RevisitedDataset                    # noqa: E402
from evaluation.revisited_ap import revisited_map                            # noqa: E402
from eval_baselines import build_model, embed                                # noqa: E402

NORM = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


class R1MStream(IterableDataset):
    def __init__(self, r1m_dir, size, index):
        self.files = [Path(r1m_dir) / 'raw' / f'revisitop1m.{i}.tar.gz' for i in range(1, 101)]
        self.tf, self.index = T.Compose([LongSideResize(size), T.ToTensor(), NORM]), index

    def __iter__(self):
        w = get_worker_info()
        files = self.files if w is None else self.files[w.id::w.num_workers]
        for f in files:
            with tarfile.open(f, 'r:gz') as t:
                for m in t:
                    if not m.isreg() or not m.name.lower().endswith('.jpg'):
                        continue
                    rel = '/'.join(m.name.split('/')[-2:])
                    img = Image.open(t.extractfile(m)).convert('RGB')
                    yield self.tf(img), self.index[rel]


@torch.no_grad()
def r1m_embeddings(model, a, dev, tag):
    cache = Path(a.cache_dir) / f'{tag}_s{a.size}.npy'
    if cache.exists():
        return np.load(cache).astype(np.float32)
    names = [l.strip() for l in open(Path(a.r1m_dir) / 'revisitop1m.txt') if l.strip()]
    index = {n: i for i, n in enumerate(names)}
    out, seen = None, np.zeros(len(names), dtype=bool)
    loader = DataLoader(R1MStream(a.r1m_dir, a.size, index), batch_size=1, num_workers=a.workers, pin_memory=True)
    t0 = time.time()
    for k, (x, i) in enumerate(loader):
        v = F.normalize(model(x.to(dev, non_blocking=True)).float(), dim=1).cpu().numpy()[0]
        if out is None:
            out = np.zeros((len(names), v.shape[0]), dtype=np.float16)
        out[int(i)] = v; seen[int(i)] = True
        if k % 50000 == 0:
            print(f"  R1M {k}/{len(names)} images, {k / max(time.time() - t0, 1):.0f} img/s", flush=True)
    if not seen.all():
        raise RuntimeError(f"{(~seen).sum()} R1M images listed in revisitop1m.txt were not found in the archives")
    Path(a.cache_dir).mkdir(parents=True, exist_ok=True)
    np.save(cache, out)
    print(f"  R1M embedded in {(time.time() - t0) / 60:.1f} min -> {cache}", flush=True)
    return out.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', required=True, help='staged roxford5k / rparis6k')
    ap.add_argument('--r1m_dir', default='/home/atuin/b268dc/b268dc21/datasets/revisitop1m')
    ap.add_argument('--cache_dir', default='/home/atuin/b268dc/b268dc21/datasets/revisitop1m/embeddings')
    ap.add_argument('--models', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--size', type=int, default=1024)
    ap.add_argument('--workers', type=int, default=16)
    a = ap.parse_args()
    dev = torch.device('cuda')
    tf = T.Compose([LongSideResize(a.size), T.ToTensor(), NORM])
    results = json.loads(Path(a.out).read_text()) if Path(a.out).exists() else {}
    for spec in a.models:
        p = Path(spec)
        tag = (p.parent.parent.name + '_' + p.stem) if p.exists() else spec.replace(':', '_')
        if p.exists():
            tag += '_' + hashlib.sha256(open(p, 'rb').read(1 << 24)).hexdigest()[:10]
        model = build_model(spec, dev)
        dis = r1m_embeddings(model, a, dev, tag)
        r = {'tag': tag}
        for ds in ('roxford5k', 'rparis6k'):
            h = RevisitedDataset(str(Path(a.data_root) / ds), ds)
            gnd = h.get_ground_truth()
            Q = embed(model, h.get_query_dataset(tf), dev, a.workers)
            G = embed(model, h.get_database_dataset(tf), dev, a.workers)
            base = revisited_map(Q @ G.T, gnd)
            full = revisited_map(np.concatenate([Q @ G.T, Q @ dis.T], axis=1), gnd)
            r[ds] = {'without_r1m': base, 'with_r1m': full}
            print(f"{spec} {ds}: M/H {base['mAP_medium']:.2f}/{base['mAP_hard']:.2f} -> +R1M {full['mAP_medium']:.2f}/{full['mAP_hard']:.2f}", flush=True)
        results[spec] = r
        Path(a.out).write_text(json.dumps(results, indent=2))
        del model; torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
