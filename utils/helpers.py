"""
Helper utilities for the image retrieval project.

This module contains general-purpose utility functions including:
- Random seed setting for reproducibility
- Device management (CPU/GPU)
- Metric tracking
- Checkpoint management
"""

import json
import os
import random
from datetime import datetime
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
        
    Why this matters:
        - Ensures reproducible results across different runs
        - Important for comparing different methods fairly
        - Required for scientific rigor in experiments
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Make CuDNN deterministic (slightly slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(cuda: bool = True) -> torch.device:
    """
    Get the appropriate device (CPU or GPU) for computation.
    
    Args:
        cuda: Whether to use GPU if available
        
    Returns:
        torch.device object
        
    Why this matters:
        - Automatic device selection simplifies code
        - GPU acceleration is crucial for training deep models
        - Graceful fallback to CPU if GPU unavailable
    """
    if cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device


class AverageMeter:
    """
    Computes and stores the average and current value of a metric.
    
    This is useful for tracking metrics during training without storing
    all individual values in memory.
    
    Example:
        loss_meter = AverageMeter()
        for batch in dataloader:
            loss = compute_loss(batch)
            loss_meter.update(loss.item(), batch_size)
        print(f"Average loss: {loss_meter.avg}")
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all statistics."""
        self.val = 0  # Current value
        self.avg = 0  # Running average
        self.sum = 0  # Running sum
        self.count = 0  # Number of updates
    
    def update(self, val: float, n: int = 1):
        """
        Update the meter with a new value.
        
        Args:
            val: New value to add
            n: Weight for this value (typically batch size)
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0


def save_checkpoint(
    state: Dict[str, Any],
    filepath: str,
    is_best: bool = False
):
    """
    Save model checkpoint to disk.
    
    Args:
        state: Dictionary containing model state, optimizer state, etc.
        filepath: Path to save checkpoint
        is_best: Whether this is the best model so far
        
    Why this matters:
        - Allows resuming training from interruptions
        - Saves best model for inference
        - Enables model versioning and comparison
        
    Typical state dictionary:
        {
            'epoch': current_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_metric': best_map,
            'config': config
        }
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Write to a temporary file and rename, so an interrupted save never
    # leaves a truncated checkpoint in place of a good one
    def _atomic_save(path):
        tmp = path + '.tmp'
        torch.save(state, tmp)
        os.replace(tmp, path)

    _atomic_save(filepath)
    print(f"Checkpoint saved to {filepath}")

    # If this is the best model, save a separate copy
    if is_best:
        best_filepath = filepath.replace('.pth', '_best.pth')
        _atomic_save(best_filepath)
        print(f"Best model saved to {best_filepath}")


def load_checkpoint(
    filepath: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Load model checkpoint from disk.
    
    Args:
        filepath: Path to checkpoint file
        model: Model to load state into
        optimizer: Optional optimizer to load state into
        device: Device to load model to
        
    Returns:
        Dictionary containing checkpoint information
        
    Why this matters:
        - Resume training from saved checkpoints
        - Load pre-trained models for fine-tuning
        - Deploy trained models for inference
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    
    # Load checkpoint
    checkpoint = torch.load(filepath, map_location=device)
    
    # Load model state into the underlying module, so checkpoints saved with or
    # without torch.compile ("_orig_mod." prefix) load into either kind of model
    state_dict = {k.removeprefix('_orig_mod.'): v for k, v in checkpoint['model_state_dict'].items()}
    getattr(model, '_orig_mod', model).load_state_dict(state_dict)
    print(f"Model loaded from {filepath}")
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Optimizer state loaded")
    
    return checkpoint


def compute_distance_matrix(
    query_embeddings: torch.Tensor,
    gallery_embeddings: torch.Tensor,
    metric: str = 'cosine'
) -> torch.Tensor:
    """
    Compute pairwise distance matrix between queries and gallery.
    
    Args:
        query_embeddings: Query embeddings (N x D)
        gallery_embeddings: Gallery embeddings (M x D)
        metric: Distance metric ('cosine', 'euclidean', 'dot')
        
    Returns:
        Distance matrix (N x M) where smaller values indicate more similar
        
    Why this matters:
        - Core operation for retrieval: finding nearest neighbors
        - Different metrics suit different embedding spaces
        - Efficient batch computation is crucial for large galleries
        
    Note:
        - For cosine similarity, embeddings should be L2-normalized
        - Lower distances = more similar for all metrics
    """
    if metric == 'cosine':
        # Normalize embeddings
        query_norm = torch.nn.functional.normalize(query_embeddings, p=2, dim=1)
        gallery_norm = torch.nn.functional.normalize(gallery_embeddings, p=2, dim=1)
        # Compute cosine similarity, then convert to distance
        similarity = torch.mm(query_norm, gallery_norm.t())
        distance = 1 - similarity
        
    elif metric == 'euclidean':
        # Efficient euclidean distance computation
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a·b
        query_sq = (query_embeddings ** 2).sum(dim=1, keepdim=True)
        gallery_sq = (gallery_embeddings ** 2).sum(dim=1, keepdim=True)
        distance = query_sq + gallery_sq.t() - 2 * torch.mm(query_embeddings, gallery_embeddings.t())
        distance = torch.sqrt(torch.clamp(distance, min=1e-12))
        
    elif metric == 'dot':
        # Dot product similarity (negative for distance)
        distance = -torch.mm(query_embeddings, gallery_embeddings.t())
        
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    return distance


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted string (e.g., "2h 30m 45s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    
    return " ".join(parts)


def save_training_curves(history: Dict[str, List[float]], out_dir: str, stage: str):
    """Save per-epoch training metrics as JSON and PNG plots.

    Args:
        history: metric name -> list of per-epoch values.
        out_dir: directory to write files into.
        stage:   'stage1' or 'stage2'.
    """
    os.makedirs(out_dir, exist_ok=True)
    plots_dir = os.path.join(out_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    json_path = os.path.join(out_dir, f'{stage}_metrics_{ts}.json')
    with open(json_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Metrics saved to {json_path}")

    plot_json_path = os.path.join(plots_dir, f'{stage}_metrics_{ts}.json')
    with open(plot_json_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Metrics copy saved to {plot_json_path}")

    epochs = list(range(1, len(next(iter(history.values()))) + 1))

    if stage == 'stage1':
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, history['train_loss'], 'o-', label='Train Loss')
        ax.plot(epochs, history['val_loss'], 's-', label='Val Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Stage 1 — Metric Learning')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig_path = os.path.join(plots_dir, f'{stage}_loss_{ts}.png')
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"Plot saved to {fig_path}")

    elif stage == 'stage2':
        has_bench = bool(history.get('roxford_mAP_medium'))
        ncols = 3 if has_bench else 2
        fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5))

        ax = axes[0]
        ax.plot(epochs, history['train_reward'], 'o-', label='Train Reward (mAP)')
        ax.plot(epochs, history['val_reward'], 's-', label='Val Reward (mAP)')
        if 'baseline' in history:
            ax.plot(epochs, history['baseline'], '^--', label='Baseline', alpha=0.6)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Reward')
        ax.set_title('Stage 2 — RL Reward')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.plot(epochs, history['train_loss'], 'o-', label='Policy Loss', color='tab:red')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Stage 2 — Policy Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        if has_bench:
            ax = axes[2]
            ax.plot(epochs, history['roxford_mAP_medium'], 'o-', label='Medium', color='tab:green')
            ax.plot(epochs, history['roxford_mAP_hard'], 's-', label='Hard', color='tab:orange')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('mAP')
            ax.set_title('Stage 2 — ROxford5k Benchmark')
            ax.legend()
            ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig_path = os.path.join(plots_dir, f'{stage}_curves_{ts}.png')
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"Plot saved to {fig_path}")


def save_eval_plots(all_metrics: Dict[str, Dict[str, float]], out_dir: str, checkpoint_name: str):
    """Save bar charts comparing evaluation metrics across datasets.

    Args:
        all_metrics: dataset_name -> {metric_name: value}.
        out_dir: base results directory.
        checkpoint_name: label for this checkpoint (used in title/filename).
    """
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    plots_dir = os.path.join(out_dir, 'eval_plots', ts)
    os.makedirs(plots_dir, exist_ok=True)

    json_path = os.path.join(plots_dir, f'eval_{checkpoint_name}_{ts}.json')
    with open(json_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)

    for ds_name, metrics in all_metrics.items():
        names = list(metrics.keys())
        values = list(metrics.values())

        fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.2), 5))
        bars = ax.bar(range(len(names)), values, color='steelblue', edgecolor='black')
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, ha='right')
        ax.set_ylim(0, 1.05)
        ax.set_ylabel('Score')
        ax.set_title(f'{ds_name} — {checkpoint_name}')
        ax.grid(True, axis='y', alpha=0.3)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{v:.4f}', ha='center', va='bottom', fontsize=8)
        fig.tight_layout()
        fig_path = os.path.join(plots_dir, f'eval_{ds_name}_{checkpoint_name}_{ts}.png')
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"Eval plot saved to {fig_path}")

