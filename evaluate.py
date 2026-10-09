"""
Evaluation script for image retrieval models.

This script evaluates a trained model on a test set and computes
standard retrieval metrics:
- Mean Average Precision (mAP)
- Recall@K (K=1,5,10)
- Precision@K
- NDCG@K

Usage:
------
# Evaluate a model
python evaluate.py --checkpoint checkpoints/stage2_best.pth --config config/config.yaml

# Evaluate on specific dataset
python evaluate.py --checkpoint checkpoints/model.pth --dataset roxford5k

# Save embeddings for later analysis
python evaluate.py --checkpoint checkpoints/model.pth --save_embeddings
"""

import argparse
import os
import yaml
import torch
import numpy as np
from datetime import datetime
from pathlib import Path

from models.embedding_net import EmbeddingNet
from evaluation.evaluator import RetrievalEvaluator
from data.dataset import get_revisited_benchmark_transforms, RevisitedDataset
from data.dataloader import get_dataloaders
from utils.helpers import get_device, load_checkpoint, save_eval_plots


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate image retrieval model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to config file')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Dataset name (overrides config)')
    parser.add_argument('--save_embeddings', action='store_true',
                        help='Save embeddings to disk')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--stage', type=str, choices=['stage1', 'stage2'],
                        default=None,
                        help='Stage label for evaluation output folders')
    parser.add_argument('--save_eval_plots', action='store_true',
                        help='Save evaluation metric plots (disabled by default)')
    
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def infer_stage_label(checkpoint_path: str, explicit_stage: str | None = None) -> str:
    """Infer stage label from command line or checkpoint name."""
    if explicit_stage is not None:
        return explicit_stage
    stem = Path(checkpoint_path).stem.lower()
    if 'stage1' in stem:
        return 'stage1'
    if 'stage2' in stem:
        return 'stage2'
    return 'eval'


def create_model_from_checkpoint(checkpoint_path, device):
    """
    Create model and load weights from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    print(f"Loading model from {checkpoint_path}...")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get config from checkpoint
    config = checkpoint.get('config', None)
    
    if config is None:
        raise ValueError("Checkpoint does not contain config. Please provide config file.")
    
    # Create model
    model = EmbeddingNet(
        architecture=config['model']['architecture'],
        pretrained=False,  # Will load weights from checkpoint
        pooling=config['model']['pooling'],
        embedding_dim=config['model']['embedding_dim'],
        normalize_embeddings=config['model']['normalize_embeddings'],
        pooling_kwargs={'p': config['model'].get('gem_power', 3.0)}
    )
    
    # Load weights and normalize common wrapper prefixes.
    # Some checkpoints are saved from compiled or wrapped models,
    # which add prefixes like `_orig_mod.` or `module.` to the state_dict keys.
    state_dict = checkpoint['model_state_dict']
    if any(k.startswith('_orig_mod.') for k in state_dict):
        state_dict = {k[len('_orig_mod.'):]: v for k, v in state_dict.items()}
    if any(k.startswith('module.') for k in state_dict):
        state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}

    if any(k.startswith('embedding_net.') for k in state_dict):
        state_dict = {
            k[len('embedding_net.'):]: v
            for k, v in state_dict.items()
            if k.startswith('embedding_net.')
        }

    # Filter out any extra policy/head keys that do not belong to EmbeddingNet.
    model_keys = set(model.state_dict().keys())
    state_dict = {k: v for k, v in state_dict.items() if k in model_keys}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Warning: missing keys when loading checkpoint: {missing}")
    if unexpected:
        print(f"Warning: unexpected keys ignored: {unexpected}")

    model = model.to(device)
    model.eval()
    
    print("Model loaded successfully!")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    if 'best_loss' in checkpoint:
        print(f"  Best loss: {checkpoint['best_loss']:.4f}")
    if 'best_reward' in checkpoint:
        print(f"  Best reward: {checkpoint['best_reward']:.4f}")
    
    return model


def evaluate_revisited(model, dataset_name, config, device, args=None):
    """Evaluate on ROxford5k / RParis6k with revisited benchmark protocols.

    Can be called from the CLI (with args) or programmatically from
    train.py (args=None, no file saving).
    """
    import json
    from tqdm import tqdm

    root_dir_cfg = Path(config['dataset']['root_dir'])
    dataset_root = root_dir_cfg / dataset_name

    eval_size = config['dataset'].get('query_size', 1024)
    print(f"  Eval resolution: shorter edge = {eval_size}px")
    transform = get_revisited_benchmark_transforms(image_size=eval_size)
    dataset_handler = RevisitedDataset(root_dir=str(dataset_root), dataset_name=dataset_name)
    gnd = dataset_handler.get_ground_truth()

    query_dataset  = dataset_handler.get_query_dataset(transform)
    gallery_dataset = dataset_handler.get_database_dataset(transform)

    num_workers = config['system']['num_workers']

    query_loader = torch.utils.data.DataLoader(
        query_dataset, batch_size=1, shuffle=False, num_workers=num_workers)
    gallery_loader = torch.utils.data.DataLoader(
        gallery_dataset, batch_size=1, shuffle=False, num_workers=num_workers)

    print(f"  Query images:   {len(query_dataset)}")
    print(f"  Gallery images: {len(gallery_dataset)}")

    def extract(loader, desc):
        model.eval()
        vecs = []
        with torch.no_grad():
            for imgs, _, _ in tqdm(loader, desc=desc):
                vecs.append(model(imgs.to(device)).cpu())
        return torch.cat(vecs, dim=0).numpy()

    Q = extract(query_loader,   "Query embeddings")
    G = extract(gallery_loader, "Gallery embeddings")

    sim = Q @ G.T

    k_values = [1, 5, 10]

    def compute_ap_with_junk(ranked, pos_set, junk_set):
        # Official revisitop compute_ap: junk removed from the ranking, then
        # trapezoidal integration of precision over recall steps of 1/|pos|.
        # Checked against hand cases in scripts/audit_eval_vs_official.py.
        if not pos_set:
            return 0.0
        recall_step = 1.0 / len(pos_set)
        ap = 0.0
        j = 0  # positives found so far
        rank = 0  # 0-indexed position in the junk-free ranking
        for idx in ranked:
            if idx in junk_set:
                continue
            if idx in pos_set:
                precision_0 = 1.0 if rank == 0 else j / rank
                precision_1 = (j + 1) / (rank + 1)
                ap += (precision_0 + precision_1) * recall_step / 2.0
                j += 1
            rank += 1
        return ap

    ap_medium, ap_hard = [], []
    recall_medium = {k: [] for k in k_values}
    recall_hard = {k: [] for k in k_values}

    for q_idx in range(len(Q)):
        entry = gnd[q_idx] if isinstance(gnd, list) else gnd[q_idx]
        easy = set(entry.get('easy', []))
        hard = set(entry.get('hard', []))
        junk = set(entry.get('junk', []))

        ranked = np.argsort(-sim[q_idx])

        if easy or hard:
            ap_medium.append(compute_ap_with_junk(ranked, easy | hard, junk))
        if hard:
            ap_hard.append(compute_ap_with_junk(ranked, hard, junk | easy))

        ranked_no_junk_medium = [r for r in ranked if r not in junk]
        ranked_no_junk_hard = [r for r in ranked if r not in (junk | easy)]
        pos_medium = easy | hard
        for k in k_values:
            top_m = set(ranked_no_junk_medium[:k])
            recall_medium[k].append(len(top_m & pos_medium) / max(len(pos_medium), 1))
            top_h = set(ranked_no_junk_hard[:k])
            recall_hard[k].append(len(top_h & hard) / max(len(hard), 1) if hard else 0.0)

    metrics = {
        'mAP_medium': float(np.mean(ap_medium)) if ap_medium else 0.0,
        'mAP_hard': float(np.mean(ap_hard)) if ap_hard else 0.0,
    }
    for k in k_values:
        metrics[f'Recall@{k}_medium'] = float(np.mean(recall_medium[k]))
        metrics[f'Recall@{k}_hard'] = float(np.mean(recall_hard[k]))

    print(f"\n{'=' * 50}")
    print(f"  {dataset_name} — Retrieval Performance")
    print(f"{'=' * 50}")
    print(f"  mAP (Medium protocol) ......  {metrics['mAP_medium']:.4f}")
    print(f"  mAP (Hard protocol) ........  {metrics['mAP_hard']:.4f}")
    for k in k_values:
        print(f"  Recall@{k:<3} (Medium) ........  {metrics[f'Recall@{k}_medium']:.4f}")
        print(f"  Recall@{k:<3} (Hard) ..........  {metrics[f'Recall@{k}_hard']:.4f}")
    print(f"{'=' * 50}\n")

    if args is not None:
        out = Path(args.output_dir) / dataset_name
        out.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        with open(out / f'metrics_{ts}.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        with open(out / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        if args.save_embeddings:
            np.savez(out / f'embeddings_{ts}.npz', query=Q, gallery=G)
        print(f"  Results saved to {out}")

    return metrics


def main():
    """Main evaluation function."""
    args = parse_args()
    config = load_config(args.config)
    device = get_device(config['system']['cuda'])

    stage_label = infer_stage_label(args.checkpoint, args.stage)
    args.output_dir = str(Path(args.output_dir) / f'eval_{stage_label}')

    model = create_model_from_checkpoint(args.checkpoint, device)

    # Resolve which datasets to evaluate
    name_cfg = config['dataset']['name']
    if args.dataset:
        datasets = [args.dataset]
    elif isinstance(name_cfg, list):
        datasets = name_cfg
    else:
        datasets = [name_cfg]

    all_metrics = {}
    for ds_name in datasets:
        print(f"\n{'=' * 50}")
        print(f"Evaluating on {ds_name}")
        print(f"{'=' * 50}")
        if ds_name in ('roxford5k', 'rparis6k'):
            all_metrics[ds_name] = evaluate_revisited(model, ds_name, config, device, args)
        else:
            print(f"Dataset {ds_name} not supported yet.")

    print("\n" + "=" * 50)
    print("ALL EVALUATIONS COMPLETED")
    print("=" * 50)
    for ds, m in all_metrics.items():
        print(f"\n{ds}:")
        for k, v in m.items():
            print(f"  {k}: {v:.4f}")

    if all_metrics and args.save_eval_plots:
        ckpt_label = Path(args.checkpoint).stem
        save_eval_plots(all_metrics, args.output_dir, ckpt_label)


if __name__ == '__main__':
    main()

