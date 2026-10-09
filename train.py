"""
Main training script for Image Retrieval with RL.

This script implements two-stage training:

Stage 1: Standard Metric Learning
- Train with triplet loss, FastAP, or other metric learning losses
- Fast convergence, stable training
- Provides good initialization

Stage 2: RL Fine-tuning
- Directly optimize retrieval metrics (mAP, Recall@K)
- Fine-tune using policy gradients (REINFORCE)
- Achieves best performance

Usage:
------
# Full two-stage training
python train.py --config config/config.yaml

# Stage 1 only
python train.py --config config/config.yaml --stage1_only

# Stage 2 only (requires pretrained model)
python train.py --config config/config.yaml --stage2_only --resume checkpoints/stage1_best.pth
"""

import argparse
import copy
import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import OmegaConf
from pathlib import Path
from datetime import datetime
import random

from models.embedding_net import EmbeddingNet
from models.backbone import get_backbone
from losses.metric_losses import get_metric_loss
from losses.rl_rewards import RetrievalReward
from training.trainer import StandardTrainer
from training.rl_trainer import RLTrainer
from evaluation.evaluator import RetrievalEvaluator
from data.dataset import ImageRetrievalDataset, RevisitedDataset, get_transforms
from data.dataset_sfm120k import load_sfm120k
from data.dataloader import get_dataloaders
from data.dataloader import BalancedBatchSampler
from utils.helpers import set_seed, get_device, save_checkpoint, load_checkpoint, save_training_curves
from utils.logger import Logger
from evaluate import evaluate_revisited
from evaluation.sfm_val import evaluate_sfm_val
from data.dataset_sfm120k import SfM120kDataset


def selection_score(model, dataloaders, config, device, fallback: float, logger, tag: str, epoch: int):
    """Checkpoint-selection score (higher is better).

    Default: leave-one-out mAP over the full SfM-120k validation split (held out,
    deterministic). Set training.selection_metric to 'legacy' to fall back to the
    batch-level validation loss/reward, passed in as `fallback`.
    """
    val_loader = dataloaders.get('val')
    use_map = config['training'].get('selection_metric', 'sfm_val_map') == 'sfm_val_map'
    if use_map and val_loader is not None and isinstance(val_loader.dataset, SfM120kDataset):
        m = evaluate_sfm_val(model, val_loader.dataset, device,
                             num_workers=config['system']['num_workers'])
        logger.log_scalar(f'{tag}/sfm_val_mAP', m['sfm_val_mAP'], epoch)
        logger.log_scalar(f'{tag}/sfm_val_success@1', m['sfm_val_success@1'], epoch)
        return m['sfm_val_mAP'], m
    return fallback, {}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train image retrieval model with RL')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to config file')
    parser.add_argument('--stage1_only', action='store_true',
                        help='Only run stage 1 (metric learning)')
    parser.add_argument('--stage2_only', action='store_true',
                        help='Only run stage 2 (RL fine-tuning)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (overrides config)')
    parser.add_argument('--stage1_epochs', type=int, default=None,
                        help='Override stage 1 (metric learning) epochs')
    parser.add_argument('--stage2_epochs', type=int, default=None,
                        help='Override stage 2 (RL fine-tuning) epochs')
    
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(config, device):
    """
    Create embedding network.
    
    Args:
        config: Configuration dictionary
        device: Device to create model on
        
    Returns:
        Embedding network
    """
    model = EmbeddingNet(
        architecture=config['model']['architecture'],
        pretrained=config['model']['pretrained'],
        pooling=config['model']['pooling'],
        embedding_dim=config['model']['embedding_dim'],
        normalize_embeddings=config['model']['normalize_embeddings'],
        pooling_kwargs={'p': config['model'].get('gem_power', 3.0)}
    )
    
    model = model.to(device)
    
    print(f"\nModel created:")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    return model


def _build_labeled_split(dataset_name, root_dir_cfg, seed):
    """
    Build per-label image lists from a single Revisited dataset.

    Returns (image_root, per_label_images) where per_label_images maps
    a globally-unique integer label -> list of image filenames relative to image_root.
    The label integers are offset so that datasets combined later don't collide.
    """
    dataset_root = root_dir_cfg / dataset_name
    if not dataset_root.exists():
        dataset_root = root_dir_cfg

    dataset_handler = RevisitedDataset(root_dir=str(dataset_root), dataset_name=dataset_name)
    cfg = dataset_handler.cfg
    gnd = dataset_handler.get_ground_truth()

    def _to_list(x):
        if x is None:
            return []
        if hasattr(x, "tolist"):
            return x.tolist()
        return list(x) if isinstance(x, (list, tuple)) else [x]

    def _normalize_name(name: str) -> str:
        s = str(name).replace("\\", "/")
        if s.startswith("jpg/"):
            s = s[4:]
        if not (s.lower().endswith(".jpg") or s.lower().endswith(".jpeg") or s.lower().endswith(".png")):
            s = s + ".jpg"
        return s

    per_label_images = {}
    num_queries = len(cfg.get('query_images', []))
    for q_idx in range(num_queries):
        entry = None
        if isinstance(gnd, dict):
            entry = gnd.get(q_idx)
        elif isinstance(gnd, list) and q_idx < len(gnd):
            entry = gnd[q_idx]
        if entry is None:
            continue

        easy = _to_list(entry.get('easy', []))
        hard = _to_list(entry.get('hard', []))
        pos_db_indices = [int(i) for i in (easy + hard) if i is not None]
        if len(pos_db_indices) == 0:
            continue

        imgs = [_normalize_name(cfg['query_images'][q_idx])]
        for db_i in pos_db_indices:
            if 0 <= db_i < len(cfg['db_images']):
                imgs.append(_normalize_name(cfg['db_images'][db_i]))

        if len(imgs) >= 2:
            per_label_images[q_idx] = imgs

    if len(per_label_images) == 0:
        raise ValueError(
            f"Could not build a labeled training set for {dataset_name}. "
            f"Check that the ground-truth pkl is present and has 'easy'/'hard' positives. "
            f"Expected under: {dataset_root}"
        )

    image_root = dataset_root / "jpg"
    return image_root, per_label_images


def create_dataloaders(config, stage='stage1'):
    """
    Create dataloaders for training and validation.

    Supports a single dataset name (string) or a list of dataset names.
    When multiple datasets are given, their labeled splits are merged so
    every batch can contain images from both datasets.

    Args:
        config: Configuration dictionary
        stage: Training stage name ('stage1' or 'stage2')

    Returns:
        Dictionary with 'train' and 'val' dataloaders
    """
    train_transform = get_transforms(
        image_size=config['dataset']['image_size'],
        mode='train',
        augmentation_config=config['augmentation']['train']
    )
    val_img_size = int(
        config['dataset'].get('val_image_size', config['dataset']['image_size'])
    )
    val_transform = get_transforms(
        image_size=val_img_size,
        mode='eval',
        augmentation_config=config['augmentation']['val']
    )
    if val_img_size != int(config['dataset']['image_size']):
        print(
            f"  Val images: {val_img_size}x{val_img_size} "
            f"(train: {config['dataset']['image_size']})"
        )

    dataset_name_cfg = config['dataset']['name']
    # Normalise to a list so the rest of the code is uniform
    if isinstance(dataset_name_cfg, str):
        dataset_names = [dataset_name_cfg]
    else:
        dataset_names = list(dataset_name_cfg)

    root_dir_cfg = Path(config['dataset']['root_dir'])
    seed = int(config['system'].get('seed', 42))
    rng = random.Random(seed)
    batch_size = int(config['training'][stage]['batch_size'])
    images_per_class = max(2, min(4, batch_size))

    revisited = [n for n in dataset_names if n in ('roxford5k', 'rparis6k')]

    def _build_val_loader(val_dataset, val_labels, batch_size, images_per_class):
        """
        Build validation loader with balanced batches when labels are available.

        Sequential validation batches can easily collapse to one class on retrieval
        datasets, which makes metric-learning validation loss and RL rewards
        uninformative. Reusing PK sampling keeps validation batches meaningful.
        """
        if val_labels and len(set(val_labels)) > 1 and len(val_labels) >= batch_size:
            val_sampler = BalancedBatchSampler(
                labels=torch.tensor(val_labels),
                batch_size=batch_size,
                images_per_class=images_per_class
            )
            return torch.utils.data.DataLoader(
                val_dataset,
                batch_sampler=val_sampler,
                num_workers=config['system']['num_workers'],
                pin_memory=config['system']['pin_memory'],
                persistent_workers=config['system']['num_workers'] > 0
            )

        return torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size * 2,
            shuffle=False,
            num_workers=config['system']['num_workers'],
            pin_memory=config['system']['pin_memory']
        )

    if revisited:
        # Merge labeled splits from all requested Revisited datasets.
        # Labels are offset per-dataset so they never collide across datasets.
        merged_train_images, merged_train_labels = [], []
        merged_val_images, merged_val_labels = [], []

        # Each dataset contributes its own image_root; we store (root, filename) pairs
        # and later split into separate ImageRetrievalDataset objects combined via ConcatDataset.
        train_datasets, val_datasets = [], []
        label_offset = 0

        for ds_name in revisited:
            image_root, per_label_images = _build_labeled_split(ds_name, root_dir_cfg, seed)

            all_labels = sorted(per_label_images.keys())
            rng.shuffle(all_labels)
            n_val = max(1, int(0.1 * len(all_labels)))
            val_label_set = set(all_labels[:n_val])

            ds_train_imgs, ds_train_labels = [], []
            ds_val_imgs, ds_val_labels = [], []

            for local_label, imgs in per_label_images.items():
                global_label = local_label + label_offset
                if local_label in val_label_set:
                    ds_val_imgs.extend(imgs)
                    ds_val_labels.extend([global_label] * len(imgs))
                else:
                    ds_train_imgs.extend(imgs)
                    ds_train_labels.extend([global_label] * len(imgs))

            label_offset += max(per_label_images.keys()) + 1

            train_datasets.append(ImageRetrievalDataset(
                root_dir=str(image_root),
                image_list=ds_train_imgs,
                labels=ds_train_labels,
                transform=train_transform,
                mode='train'
            ))
            val_datasets.append(ImageRetrievalDataset(
                root_dir=str(image_root),
                image_list=ds_val_imgs,
                labels=ds_val_labels,
                transform=val_transform,
                mode='train'
            ))

            merged_train_labels.extend(ds_train_labels)
            merged_val_labels.extend(ds_val_labels)

        # Combine datasets from both sources
        if len(train_datasets) == 1:
            train_dataset = train_datasets[0]
            val_dataset = val_datasets[0]
        else:
            train_dataset = torch.utils.data.ConcatDataset(train_datasets)
            val_dataset = torch.utils.data.ConcatDataset(val_datasets)

        print(f"\nCombined training set: {len(train_dataset)} images "
              f"from {len(revisited)} dataset(s): {revisited}")
        print(f"Combined validation set: {len(val_dataset)} images")

        sampler = BalancedBatchSampler(
            labels=torch.tensor(merged_train_labels),
            batch_size=batch_size,
            images_per_class=images_per_class
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_sampler=sampler,
            num_workers=config['system']['num_workers'],
            pin_memory=config['system']['pin_memory'],
            persistent_workers=config['system']['num_workers'] > 0
        )
        val_loader = _build_val_loader(
            val_dataset=val_dataset,
            val_labels=merged_val_labels,
            batch_size=batch_size,
            images_per_class=images_per_class
        )
        return {'train': train_loader, 'val': val_loader}

    # SfM-120k — proper training set for retrieval (separate from eval benchmarks)
    sfm_names = [n for n in dataset_names if n == 'sfm120k']
    if sfm_names:
        sfm_root = root_dir_cfg / 'sfm120k'

        train_ds, val_ds = load_sfm120k(
            root_dir=str(sfm_root),
            train_transform=train_transform,
            val_transform=val_transform,
            seed=seed,
        )

        print(f"\nSfM-120k: {len(train_ds)} train images, {len(val_ds)} val images")

        sampler = BalancedBatchSampler(
            labels=torch.tensor(train_ds.labels),
            batch_size=batch_size,
            images_per_class=images_per_class,
        )
        # Optional cap for pilots/smoke runs; unset for real training
        max_batches = config['training'][stage].get('max_batches_per_epoch')
        if max_batches:
            sampler.num_batches = min(sampler.num_batches, int(max_batches))
            print(f"  max_batches_per_epoch={max_batches}: {sampler.num_batches} train batches/epoch")
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_sampler=sampler,
            num_workers=config['system']['num_workers'],
            pin_memory=config['system']['pin_memory'],
            persistent_workers=config['system']['num_workers'] > 0,
        )
        val_loader = _build_val_loader(
            val_dataset=val_ds,
            val_labels=val_ds.labels,
            batch_size=batch_size,
            images_per_class=images_per_class
        )
        return {'train': train_loader, 'val': val_loader}

    # Fallback for unsupported dataset names
    return get_dataloaders(
        train_dataset=None,
        val_dataset=None,
        batch_size=config['training']['stage1']['batch_size'],
        num_workers=config['system']['num_workers'],
        pin_memory=config['system']['pin_memory']
    )


def train_stage1(model, dataloaders, config, device, logger, start_epoch: int = 1, resume_state=None):
    """
    Stage 1: Standard metric learning training.
    
    Args:
        model: Embedding network
        dataloaders: Dictionary of dataloaders
        config: Configuration
        device: Device
        logger: Logger
        
    Returns:
        Trained model
    """
    print("\n" + "=" * 50)
    print("STAGE 1: METRIC LEARNING")
    print("=" * 50)
    
    stage1_config = config['training']['stage1']
    total_epochs = int(stage1_config['epochs'])
    start_epoch = int(start_epoch)
    if start_epoch < 1:
        start_epoch = 1
    if start_epoch > total_epochs:
        print(f"Stage 1: start_epoch ({start_epoch}) > total_epochs ({total_epochs}), skipping Stage 1.")
        return model
    print(f"Stage 1 will run epochs {start_epoch}..{total_epochs} (total configured: {total_epochs})")
    
    # Create loss function
    loss_fn = get_metric_loss(
        loss_name=stage1_config['loss'],
        margin=stage1_config.get('triplet_margin', 0.2),
        num_bins=stage1_config.get('fastap_num_bins', 10)
    )
    
    # Create optimizer
    if stage1_config['optimizer'] == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=stage1_config['learning_rate'],
            weight_decay=stage1_config['weight_decay']
        )
    elif stage1_config['optimizer'] == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=stage1_config['learning_rate'],
            weight_decay=stage1_config['weight_decay'],
            momentum=0.9
        )
    else:
        raise ValueError(f"Unknown optimizer: {stage1_config['optimizer']}")
    
    # Create scheduler
    if stage1_config['scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=stage1_config['epochs']
        )
    elif stage1_config['scheduler'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=stage1_config['epochs'] // 3,
            gamma=0.1
        )
    else:
        scheduler = None

    # Resuming a Stage 1 run: restore optimizer state, fast-forward the
    # epoch-level scheduler and keep the best validation score so far
    if resume_state is not None and start_epoch > 1:
        if 'optimizer_state_dict' in resume_state:
            optimizer.load_state_dict(resume_state['optimizer_state_dict'])
        if scheduler is not None:
            for _ in range(start_epoch - 1):
                scheduler.step()
        print(f"  Resumed optimizer/scheduler at epoch {start_epoch}, "
              f"best selection score {resume_state.get('best_selection_score')}")

    # Create trainer
    trainer = StandardTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        logger=logger,
        device=device,
        use_amp=config['training']['stage1'].get('use_amp', False)
    )
    
    # Check if train dataloader exists
    train_loader = dataloaders.get('train')
    if train_loader is None:
        raise ValueError(
            "Training dataloader is None. Please ensure you have:\n"
            "1. Downloaded your dataset (e.g., using download_dataset.py)\n"
            "2. Implemented dataset loading in create_dataloaders() function\n"
            "3. Created train_dataset and val_dataset before calling get_dataloaders()\n\n"
            "For ROxford5k/RParis6k datasets, you can use:\n"
            "  from data.dataset import RevisitedDataset\n"
            "  dataset_handler = RevisitedDataset(root_dir, dataset_name)\n"
            "  train_dataset = dataset_handler.get_database_dataset(transform=train_transform)"
        )
    
    # Training loop
    best_loss = float('inf')
    best_score = float('-inf')
    if resume_state is not None and start_epoch > 1 and resume_state.get('best_selection_score') is not None:
        best_score = float(resume_state['best_selection_score'])
    history = {'train_loss': [], 'val_loss': [], 'sfm_val_mAP': []}
    
    for epoch in range(start_epoch, total_epochs + 1):
        # Train
        train_metrics = trainer.train_epoch(
            train_loader,
            epoch=epoch,
            log_interval=config['logging']['log_frequency']
        )
        
        # Validate
        if 'val' in dataloaders:
            val_metrics = trainer.validate(
                dataloaders['val'],
                epoch=epoch
            )
        else:
            val_metrics = {'loss': train_metrics['loss']}

        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        
        # Update scheduler
        if scheduler is not None:
            scheduler.step()
        
        # Save checkpoint (selection on held-out SfM-120k validation)
        best_loss = min(best_loss, val_metrics['loss'])
        score, sel = selection_score(model, dataloaders, config, device,
                                     -val_metrics['loss'], logger, 'stage1', epoch)
        history['sfm_val_mAP'].append(sel.get('sfm_val_mAP', float('nan')))
        is_best = score > best_score
        if is_best:
            best_score = score
        
        # Rolling last (for resume) and best (selected on validation), plus
        # periodic snapshots every save_frequency epochs
        ckpt_dir = config['system']['stage1_checkpoint_dir']
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss,
            'selection_score': score,
            'best_selection_score': best_score,
            'val_metrics': sel,
            'config': config
        }
        save_checkpoint(state, os.path.join(ckpt_dir, 'stage1_last.pth'))
        if is_best:
            save_checkpoint(state, os.path.join(ckpt_dir, 'stage1_best.pth'))
        if epoch % config['system']['save_frequency'] == 0:
            save_checkpoint(state, os.path.join(ckpt_dir, f'stage1_epoch_{epoch}.pth'))
    
    save_training_curves(history, config['system']['stage1_results_dir'], 'stage1')
    print(f"\nStage 1 completed! Best loss: {best_loss:.4f}")
    
    return model


def train_stage2(model, dataloaders, config, device, logger):
    """
    Stage 2: RL fine-tuning.
    
    Args:
        model: Embedding network (from Stage 1)
        dataloaders: Dictionary of dataloaders
        config: Configuration
        device: Device
        logger: Logger
        
    Returns:
        Trained model
    """
    print("\n" + "=" * 50)
    print("STAGE 2: RL FINE-TUNING")
    print("=" * 50)
    
    stage2_config = config['training']['stage2']
    total_epochs = int(stage2_config['epochs'])
    
    # Create reward function
    reward_fn = RetrievalReward(
        metric=stage2_config['reward_metric'],
        k_values=stage2_config['reward_k'],
        baseline_type=stage2_config['baseline'],
        baseline_momentum=stage2_config.get('baseline_momentum', 0.99)
    ).to(device)

    # Keep a frozen copy of the Stage 1 model so RL tuning stays close
    # to the pretrained retrieval representation instead of drifting.
    reference_model = copy.deepcopy(model).to(device)
    reference_model.eval()
    for param in reference_model.parameters():
        param.requires_grad = False
    
    # Create optimizer
    if stage2_config['optimizer'] == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=stage2_config['learning_rate'],
            weight_decay=stage2_config['weight_decay']
        )
    else:
        raise ValueError(f"Unknown optimizer: {stage2_config['optimizer']}")

    # LR warmup then constant (paper uses 1k-4k warmup steps)
    warmup_steps = int(stage2_config.get('warmup_steps', 1000))
    total_steps = total_epochs * len(dataloaders.get('train', []))
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        return 1.0
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Create RL trainer
    trainer = RLTrainer(
        model=model,
        reward_fn=reward_fn,
        optimizer=optimizer,
        logger=logger,
        device=device,
        reference_model=reference_model,
        gradient_clip=stage2_config['gradient_clip'],
        sample_std=stage2_config.get('sample_std', 0.05),
        consistency_weight=stage2_config.get('consistency_weight', 0.0),
        memory_bank_size=stage2_config.get('memory_bank_size', 0),
        use_amp=config['training']['stage2'].get('use_amp', False),
        policy_weight=stage2_config.get('policy_weight', 1.0),
        diagnostics_every=stage2_config.get('diagnostics_every', 0)
    )
    
    # Per-epoch ROxford5k monitoring is diagnostic only and off by default:
    # picking checkpoints on the final test benchmark leaks test information.
    # Model selection uses the SfM-120k validation reward below.
    roxford_root = os.path.join(config['dataset']['root_dir'], 'roxford5k')
    has_roxford = (stage2_config.get('monitor_benchmark', False)
                   and os.path.isfile(os.path.join(roxford_root, 'roxford5k.pkl')))
    if has_roxford:
        print("monitor_benchmark=true — evaluating ROxford5k each epoch "
              "(diagnostic only; do not select checkpoints with it)")
    else:
        print("Benchmark monitoring off — checkpoint selection uses SfM-120k validation")

    # Training loop
    best_reward = 0.0
    best_score = float('-inf')
    best_roxford_map = 0.0
    history = {
        'train_reward': [], 'val_reward': [], 'train_loss': [], 'baseline': [],
        'roxford_mAP_medium': [], 'roxford_mAP_hard': [],
    }

    start_epoch = int(stage2_config.get('start_epoch', 1))
    if start_epoch < 1:
        start_epoch = 1
    if start_epoch > total_epochs:
        print(f"Stage 2: start_epoch ({start_epoch}) > total_epochs ({total_epochs}), skipping Stage 2.")
        return model
    print(f"Stage 2 will run epochs {start_epoch}..{total_epochs} (total configured: {total_epochs})")

    for epoch in range(start_epoch, total_epochs + 1):
        train_metrics = trainer.train_epoch(
            dataloaders.get('train'),
            epoch=epoch,
            log_interval=config['logging']['log_frequency'],
            scheduler=scheduler
        )

        if 'val' in dataloaders:
            val_metrics = trainer.validate(
                dataloaders['val'],
                epoch=epoch
            )
        else:
            val_metrics = {'reward': train_metrics['reward']}

        history['train_reward'].append(train_metrics['reward'])
        history['val_reward'].append(val_metrics['reward'])
        history['train_loss'].append(train_metrics['loss'])
        history['baseline'].append(train_metrics.get('baseline', 0.0))

        # Benchmark eval on ROxford5k
        roxford_map_m = 0.0
        if has_roxford:
            print(f"\n--- ROxford5k benchmark (epoch {epoch}) ---")
            bench = evaluate_revisited(model, 'roxford5k', config, device)
            roxford_map_m = bench['mAP_medium']
            history['roxford_mAP_medium'].append(roxford_map_m)
            history['roxford_mAP_hard'].append(bench['mAP_hard'])

            if roxford_map_m > best_roxford_map:
                best_roxford_map = roxford_map_m
                best_bench_path = os.path.join(
                    config['system']['stage2_checkpoint_dir'],
                    'stage2_best_roxford.pth'
                )
                save_checkpoint({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_roxford_map': best_roxford_map,
                    'config': config
                }, best_bench_path, is_best=False)
                print(f"  New best ROxford5k mAP (Medium): {best_roxford_map:.4f}")

        best_reward = max(best_reward, val_metrics['reward'])
        score, sel = selection_score(model, dataloaders, config, device,
                                     val_metrics['reward'], logger, 'stage2', epoch)
        history.setdefault('sfm_val_mAP', []).append(sel.get('sfm_val_mAP', float('nan')))
        is_best = score > best_score
        if is_best:
            best_score = score

        # Rolling last (for resume) and best (selected on validation), plus
        # periodic snapshots every save_frequency epochs
        ckpt_dir = config['system']['stage2_checkpoint_dir']
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_reward': best_reward,
            'selection_score': score,
            'best_selection_score': best_score,
            'val_metrics': sel,
            'config': config
        }
        save_checkpoint(state, os.path.join(ckpt_dir, 'stage2_last.pth'))
        if is_best:
            save_checkpoint(state, os.path.join(ckpt_dir, 'stage2_best.pth'))
        if epoch % config['system']['save_frequency'] == 0:
            save_checkpoint(state, os.path.join(ckpt_dir, f'stage2_epoch_{epoch}.pth'))

    save_training_curves(history, config['system']['stage2_results_dir'], 'stage2')
    print(f"\nStage 2 completed! Best SfM reward: {best_reward:.4f}")
    if has_roxford:
        print(f"  Best ROxford5k mAP (Medium): {best_roxford_map:.4f}")
    
    return model


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override seed if provided
    if args.seed is not None:
        config['system']['seed'] = args.seed

    # Override stage 1 epochs if provided
    if args.stage1_epochs is not None:
        config['training']['stage1']['epochs'] = args.stage1_epochs
    # Override stage 2 epochs if provided
    if args.stage2_epochs is not None:
        config['training']['stage2']['epochs'] = args.stage2_epochs
    
    # Set random seed
    set_seed(config['system']['seed'])
    
    # Get device
    device = get_device(config['system']['cuda'])
    
    # Create directories
    os.makedirs(config['system']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['system']['log_dir'], exist_ok=True)
    os.makedirs(config['system']['results_dir'], exist_ok=True)

    # Create per-stage timestamped checkpoint folders
    run_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    stage1_checkpoint_dir = os.path.join(config['system']['checkpoint_dir'], f'stage1_{run_ts}')
    stage2_checkpoint_dir = os.path.join(config['system']['checkpoint_dir'], f'stage2_{run_ts}')
    os.makedirs(stage1_checkpoint_dir, exist_ok=True)
    os.makedirs(stage2_checkpoint_dir, exist_ok=True)
    config['system']['stage1_checkpoint_dir'] = stage1_checkpoint_dir
    config['system']['stage2_checkpoint_dir'] = stage2_checkpoint_dir

    # Create per-stage results folders for epoch logs and plots
    stage1_results_dir = os.path.join(config['system']['results_dir'], f'stage1_{run_ts}')
    stage2_results_dir = os.path.join(config['system']['results_dir'], f'stage2_{run_ts}')
    os.makedirs(stage1_results_dir, exist_ok=True)
    os.makedirs(stage2_results_dir, exist_ok=True)
    config['system']['stage1_results_dir'] = stage1_results_dir
    config['system']['stage2_results_dir'] = stage2_results_dir

    print(f"Stage1 checkpoints will be written to: {stage1_checkpoint_dir}")
    print(f"Stage2 checkpoints will be written to: {stage2_checkpoint_dir}")
    print(f"Stage1 results will be written to: {stage1_results_dir}")
    print(f"Stage2 results will be written to: {stage2_results_dir}")
    
    # Create logger
    logger = Logger(
        log_dir=config['system']['log_dir'],
        use_wandb=config['logging']['use_wandb'],
        wandb_project=config['logging'].get('wandb_project'),
        wandb_entity=config['logging'].get('wandb_entity'),
        config=config
    )
    
    # Create model
    model = create_model(config, device)
    
    # Compile model for speedup (PyTorch 2.0+)
    if hasattr(torch, 'compile'):
        model = torch.compile(model)
        print("Model compiled with torch.compile")
    
    # Load checkpoint if resuming
    resume_epoch = 0
    if args.resume is not None:
        print(f"\nLoading checkpoint from {args.resume}")
        checkpoint = load_checkpoint(args.resume, model, device=device)
        resume_epoch = int(checkpoint.get('epoch', 0) or 0)
        print(f"Resumed from epoch {resume_epoch}")
    
    # Create dataloaders
    stage1_dataloaders = None
    stage2_dataloaders = None

    if not args.stage2_only:
        stage1_dataloaders = create_dataloaders(config, stage='stage1')

    if not args.stage1_only:
        stage2_dataloaders = create_dataloaders(config, stage='stage2')
        tsz = int(config['dataset']['image_size'])
        vsz = int(config['dataset'].get('val_image_size', config['dataset']['image_size']))
        print(f"Stage 2 dataloaders: train {tsz}×{tsz}, val {vsz}×{vsz} (same transforms as Stage 1)")

    # Run training stages
    if not args.stage2_only:
        # Stage 1: Metric learning
        start_epoch = resume_epoch + 1 if args.resume is not None else 1
        model = train_stage1(model, stage1_dataloaders, config, device, logger, start_epoch=start_epoch,
                             resume_state=checkpoint if args.resume is not None else None)
    
    if not args.stage1_only:
        # Stage 2: RL fine-tuning. When resuming with --stage2_only, run Stage 2
        # from epoch 1 (--stage2_epochs = number of RL epochs to run, not absolute index).
        if args.resume is not None and args.stage2_only:
            config['training']['stage2']['start_epoch'] = 1
        model = train_stage2(model, stage2_dataloaders, config, device, logger)
    
    # Close logger
    logger.close()
    
    print("\n" + "=" * 50)
    print("TRAINING COMPLETED!")
    print("=" * 50)
    print(f"Checkpoints saved to: {config['system']['checkpoint_dir']}")
    print(f"Logs saved to: {config['system']['log_dir']}")
    print("\nTo evaluate the model, run:")
    print(f"python evaluate.py --checkpoint {config['system']['checkpoint_dir']}/stage2_best.pth")


if __name__ == '__main__':
    main()

