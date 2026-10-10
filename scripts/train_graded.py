#!/usr/bin/env python3
"""Phase 2 fine-tuning on graded candidate lists, from a Stage 1 checkpoint.

  --mode rl          graded reward tuning (exact nDCG@k reward, multi-sample REINFORCE)
  --mode approxndcg  graded differentiable control (ApproxNDCG) on the same batches

Selection: SfM-120k val graded nDCG@100 (leave-one-out, oracle grades); binary val mAP is logged too.
Checkpoints: <ckpt_dir>/graded_last.pth and graded_best.pth (EmbeddingNet state, loadable by evaluate.py).
"""
import argparse, copy, json, sys, time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dataset import get_transforms                         # noqa: E402
from data.dataset_sfm120k import load_sfm120k                   # noqa: E402
from data.graded_lists import GradedListSampler, load_oracle    # noqa: E402
from evaluate import create_model_from_checkpoint               # noqa: E402
from evaluation.sfm_val import evaluate_sfm_val                 # noqa: E402
from training.graded_trainer import GradedTrainer               # noqa: E402
from utils.helpers import save_checkpoint, set_seed             # noqa: E402

AUG = {'random_flip': True, 'color_jitter': True, 'brightness': 0.2, 'contrast': 0.2,
       'saturation': 0.2, 'hue': 0.1, 'rotation': 10}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['rl', 'approxndcg'], required=True)
    ap.add_argument('--init', required=True)
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--oracle_dir', required=True)
    ap.add_argument('--out', required=True, help='results dir (metrics json)')
    ap.add_argument('--ckpt_dir', required=True)
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--batches_per_epoch', type=int, default=1432)
    ap.add_argument('--queries_per_batch', type=int, default=4)
    ap.add_argument('--candidates_per_query', type=int, default=7)
    ap.add_argument('--lr', type=float, default=2e-6)
    ap.add_argument('--warmup', type=int, default=100)
    ap.add_argument('--consistency_weight', type=float, default=0.0)
    ap.add_argument('--sample_std', type=float, default=0.05)
    ap.add_argument('--num_samples', type=int, default=16)
    ap.add_argument('--ndcg_k', type=int, default=10)
    ap.add_argument('--approx_tau', type=float, default=0.05)
    ap.add_argument('--memory_bank_size', type=int, default=4096)
    ap.add_argument('--image_size', type=int, default=1024)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--select', default='sfm_val_graded_ndcg@100')
    ap.add_argument('--binary_train_grades', action='store_true',
                    help='ablation: every verified same-landmark pair gets level 1 in training (val stays graded)')
    a = ap.parse_args()
    set_seed(a.seed)
    dev = torch.device('cuda')
    out, ck = Path(a.out), Path(a.ckpt_dir)
    out.mkdir(parents=True, exist_ok=True); ck.mkdir(parents=True, exist_ok=True)
    (out / 'args.json').write_text(json.dumps(vars(a), indent=2))

    train, val = load_sfm120k(str(Path(a.data_root) / 'sfm120k'),
                              train_transform=get_transforms(a.image_size, 'train', AUG),
                              val_transform=get_transforms(a.image_size, 'eval'))
    g_train, g_val = load_oracle(a.oracle_dir, 'train'), load_oracle(a.oracle_dir, 'val')
    if a.binary_train_grades:
        g_train = dict.fromkeys(g_train, 1)
    sampler = GradedListSampler(train.labels, g_train, a.queries_per_batch, a.candidates_per_query,
                                a.batches_per_epoch, seed=a.seed)
    print(f"graded lists: {len(sampler.partners)} usable queries over {len(sampler.landmarks)} landmarks; "
          f"{len(g_train) // 2} train / {len(g_val) // 2} val verified pairs")
    loader = torch.utils.data.DataLoader(train, batch_sampler=sampler, num_workers=a.workers,
                                         pin_memory=True, persistent_workers=a.workers > 0)

    model = create_model_from_checkpoint(a.init, dev)
    ref = None
    if a.mode == 'rl' and a.consistency_weight > 0:
        ref = copy.deepcopy(model).eval()
        for p in ref.parameters():
            p.requires_grad = False
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(a.warmup, 1)))
    trainer = GradedTrainer(model, opt, dev, g_train, torch.tensor(train.labels), a.mode,
                            a.queries_per_batch, a.candidates_per_query, reference_model=ref,
                            consistency_weight=a.consistency_weight, sample_std=a.sample_std,
                            num_samples=a.num_samples, ndcg_k=a.ndcg_k, approx_tau=a.approx_tau,
                            memory_bank_size=a.memory_bank_size)

    history, best = [], float('-inf')
    v0 = evaluate_sfm_val(model, val, dev, num_workers=a.workers, grades=g_val)
    history.append({'epoch': 0, **v0})
    for epoch in range(1, a.epochs + 1):
        tr = trainer.train_epoch(loader, epoch, scheduler=sched)
        v = evaluate_sfm_val(model, val, dev, num_workers=a.workers, grades=g_val)
        history.append({'epoch': epoch, **{f'train_{k}': x for k, x in tr.items()}, **v})
        state = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'val_metrics': v,
                 'config': torch.load(a.init, map_location='cpu', weights_only=False)['config'],
                 'graded_args': vars(a)}
        save_checkpoint(state, str(ck / 'graded_last.pth'))
        if v[a.select] > best:
            best = v[a.select]
            save_checkpoint(state, str(ck / 'graded_best.pth'))
            print(f"  new best {a.select} = {100 * best:.2f} at epoch {epoch}")
        (out / 'history.json').write_text(json.dumps(history, indent=2))
    print(json.dumps(history, indent=2))


if __name__ == '__main__':
    main()
