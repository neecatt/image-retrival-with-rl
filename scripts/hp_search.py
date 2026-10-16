#!/usr/bin/env python3
"""Successive-halving hyperparameter search for Stage 1, selected on held-out validation only.

  sample   draw N configurations from a scrambled Sobol sequence over the search space
  submit   submit one rung (fixed step budget, cosine schedule over the rung) for given config ids
  collect  read each run's selection.json (mean of GLDv2-clean val and SfM-120k val mAP) and rank

Rungs are independent runs with growing budgets (restarts, as in successive halving); the top
fraction of a rung is promoted to the next. No benchmark (ROxford/RParis) number is used anywhere.

Usage:
  python scripts/hp_search.py sample --space dinov2 --n 16 --out results/hpsearch/dinov2.json
  python scripts/hp_search.py submit --configs results/hpsearch/dinov2.json --rung 0 --ids all
  python scripts/hp_search.py collect --configs results/hpsearch/dinov2.json --rung 0
"""
import argparse, json, math, subprocess
from pathlib import Path

import numpy as np
from scipy.stats import qmc

# budget per rung: (epochs, batches per epoch); validation after every epoch
RUNGS = {-1: (1, 50), 0: (2, 750), 1: (3, 1500), 2: (4, 3000)}   # -1: smoke test

SPACES = {
    'dinov2': {
        'fixed': ['model.architecture=dinov2_vitb14_cls', 'model.embedding_dim=null', 'dataset.image_size=448',
                  'dataset.val_image_size=448', 'training.stage1.batch_size=128'],
        'dims': {
            'lr': ('log', 1e-6, 3e-5),
            'weight_decay': ('log', 1e-3, 1e-1),
            'layer_decay': ('lin', 0.55, 1.0),
            'freeze_blocks': ('choice', [0, 2, 4, 6, 8]),
            'loss': ('choice', ['fastap', 'subcenter_arcface']),
            'head_lr': ('log', 3e-4, 3e-3),
            'rrc': ('choice', [0, 0.3]),
        },
    },
    'resnet': {
        'fixed': ['model.architecture=resnet50', 'model.embedding_dim=512', 'dataset.image_size=512',
                  'dataset.val_image_size=512', 'training.stage1.batch_size=128'],
        'dims': {
            'lr': ('log', 2e-5, 5e-4),
            'weight_decay': ('log', 1e-4, 1e-1),
            'loss': ('choice', ['fastap', 'subcenter_arcface']),
            'head_lr': ('log', 3e-4, 3e-3),
            'rrc': ('choice', [0, 0.3]),
        },
    },
}


def decode(space, u):
    cfg = {}
    for (name, spec), x in zip(SPACES[space]['dims'].items(), u):
        if spec[0] == 'log':
            cfg[name] = float(math.exp(math.log(spec[1]) + x * (math.log(spec[2]) - math.log(spec[1]))))
        elif spec[0] == 'lin':
            cfg[name] = float(spec[1] + x * (spec[2] - spec[1]))
        else:
            cfg[name] = spec[1][min(int(x * len(spec[1])), len(spec[1]) - 1)]
    return cfg


def overrides(space, cfg, rung):
    ep, nb = RUNGS[rung]
    o = list(SPACES[space]['fixed']) + [
        'dataset.name=gldv2c', 'training.extra_val_sets=[sfm120k]', 'system.lean_checkpoints=true',
        'system.num_workers=12', 'system.seed=42',
        'training.stage1.optimizer=adamw', 'training.stage1.scheduler=cosine_warmup',
        f"training.stage1.warmup_steps={int(0.05 * ep * nb)}",
        f"training.stage1.epochs={ep}", f"training.stage1.max_batches_per_epoch={nb}",
        f"training.stage1.learning_rate={cfg['lr']:.3g}", f"training.stage1.weight_decay={cfg['weight_decay']:.3g}",
        f"training.stage1.loss={cfg['loss']}", f"training.stage1.head_lr={cfg['head_lr']:.3g}",
        f"augmentation.train.rrc_scale_min={cfg['rrc'] if cfg['rrc'] else 'null'}",
        'augmentation.train.rotation=0']
    if 'layer_decay' in cfg:
        o += [f"model.layer_decay={cfg['layer_decay']:.3f}", f"model.freeze_blocks={cfg['freeze_blocks']}"]
    return o


def run_name(space, rung, cid):
    return f"hp_{space}_r{rung}_c{cid:02d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['sample', 'submit', 'collect'])
    ap.add_argument('--space', default='dinov2')
    ap.add_argument('--n', type=int, default=16)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--configs')
    ap.add_argument('--out')
    ap.add_argument('--rung', type=int, default=0)
    ap.add_argument('--ids', default='all')
    a = ap.parse_args()

    if a.cmd == 'sample':
        dims = len(SPACES[a.space]['dims'])
        u = qmc.Sobol(d=dims, scramble=True, seed=a.seed).random(a.n)
        cfgs = [decode(a.space, row) for row in u]
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({'space': a.space, 'seed': a.seed, 'configs': cfgs}, indent=1))
        for i, c in enumerate(cfgs):
            print(i, c)
        return

    meta = json.loads(Path(a.configs).read_text())
    space, cfgs = meta['space'], meta['configs']
    ids = range(len(cfgs)) if a.ids == 'all' else [int(x) for x in a.ids.split(',')]
    if a.cmd == 'submit':
        for cid in ids:
            name = run_name(space, a.rung, cid)
            cmd = ['sbatch', '--parsable', '--time=08:00:00',
                   '--export=TRAIN_SET=gldv2c,EXTRA_STAGE=sfm120k,SKIP_FINAL_EVAL=1',
                   'scripts/train_stage1_job.sh', name] + overrides(space, cfgs[cid], a.rung)
            jid = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
            print(cid, name, jid, flush=True)
        return

    rows = []
    for cid in ids:
        f = Path('results/runs') / run_name(space, a.rung, cid) / 'selection.json'
        if f.exists():
            d = json.loads(f.read_text())
            rows.append((d['best_selection_score'], cid, d['best_epoch'], d['selection'], cfgs[cid]))
        else:
            rows.append((float('nan'), cid, None, None, cfgs[cid]))
    rows.sort(key=lambda r: -r[0] if r[0] == r[0] else 1e9)
    for score, cid, ep, sel, c in rows:
        s = f"{100 * score:6.2f}" if score == score else "  n/a "
        per = ' '.join(f"{100 * x:.2f}" for x in sel) if sel else ''
        print(f"c{cid:02d} {s} best_ep {ep} [{per}]  {json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in c.items()})}")


if __name__ == '__main__':
    main()
