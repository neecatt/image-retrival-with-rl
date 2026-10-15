#!/usr/bin/env python3
"""Save an off-the-shelf pretrained model (no fine-tuning, no projection) as a checkpoint, so the
fine-tuning scripts can start from it exactly like from a Stage 1 checkpoint.
Usage: python scripts/make_stage0_checkpoint.py ARCH OUT.pth   (e.g. dinov2_vitb14_cls)"""
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.embedding_net import EmbeddingNet   # noqa: E402

arch, out = sys.argv[1], sys.argv[2]
cfg = yaml.safe_load(open(Path(__file__).resolve().parents[1] / 'config' / 'config.yaml'))
cfg['model'].update({'architecture': arch, 'pretrained': True, 'embedding_dim': None, 'pooling': 'gem',
                     'gem_power': 3.0, 'normalize_embeddings': True})
m = EmbeddingNet(architecture=arch, pretrained=True, pooling='gem', embedding_dim=None,
                 normalize_embeddings=True, pooling_kwargs={'p': 3.0})
Path(out).parent.mkdir(parents=True, exist_ok=True)
torch.save({'epoch': 0, 'model_state_dict': m.state_dict(), 'config': cfg, 'note': f'off-the-shelf {arch}'}, out)
print('saved', out, sum(p.numel() for p in m.parameters()))
