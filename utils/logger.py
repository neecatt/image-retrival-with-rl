"""
Logging utilities for tracking experiments.

This module provides a unified logging interface that supports:
- Console logging
- TensorBoard logging
- Weights & Biases (wandb) logging
"""

import json
import os
from typing import Dict, Any, Optional
import torch
from torch.utils.tensorboard import SummaryWriter


class Logger:
    """
    Unified logger for experiment tracking.

    Supports multiple backends:
    - Console output
    - TensorBoard for visualization
    - Weights & Biases for experiment management

    Why use this:
        - Centralized logging interface
        - Easy to switch between logging backends
        - Automatic metric tracking and visualization
    """

    def __init__(
        self,
        log_dir: str,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize logger.

        Args:
            log_dir: Directory for TensorBoard logs
            use_wandb: Whether to use Weights & Biases
            wandb_project: W&B project name
            wandb_entity: W&B entity (username or team)
            config: Configuration dictionary to log
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.scalar_json_path = os.path.join(log_dir, 'step_metrics.json')
        self.scalar_history: Dict[str, list[Dict[str, float]]] = {}

        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir)
        print(f"TensorBoard logging to: {log_dir}")
        print(f"Run: tensorboard --logdir {log_dir}")

        # Initialize W&B if requested
        self.use_wandb = use_wandb
        if use_wandb:
            try:
                import wandb
                self.wandb = wandb
                self.wandb.init(
                    project=wandb_project,
                    entity=wandb_entity,
                    config=config,
                    dir=log_dir
                )
                print("Weights & Biases logging enabled")
            except ImportError:
                print("Warning: wandb not installed. Install with: pip install wandb")
                self.use_wandb = False

    def _flush_scalar_history(self):
        """Persist step-level scalar logs so they can be plotted without TensorBoard."""
        with open(self.scalar_json_path, 'w') as f:
            json.dump(self.scalar_history, f, indent=2)

    def log_scalar(self, tag: str, value: float, step: int):
        """
        Log a scalar value.

        Args:
            tag: Name of the metric (e.g., 'train/loss')
            value: Metric value
            step: Training step/epoch

        Example:
            logger.log_scalar('train/loss', 0.5, step=100)
        """
        # Log to TensorBoard
        self.writer.add_scalar(tag, value, step)

        self.scalar_history.setdefault(tag, []).append({
            'step': int(step),
            'value': float(value),
        })
        self._flush_scalar_history()

        # Log to W&B
        if self.use_wandb:
            self.wandb.log({tag: value}, step=step)

    def log_scalars(self, tag: str, value_dict: Dict[str, float], step: int):
        """
        Log multiple related scalars.

        Args:
            tag: Group name (e.g., 'metrics')
            value_dict: Dictionary of metric names and values
            step: Training step/epoch

        Example:
            logger.log_scalars('metrics', {'mAP': 0.7, 'R@1': 0.8}, step=10)
        """
        # Log to TensorBoard
        self.writer.add_scalars(tag, value_dict, step)

        # Log to W&B
        if self.use_wandb:
            wandb_dict = {f"{tag}/{k}": v for k, v in value_dict.items()}
            self.wandb.log(wandb_dict, step=step)

    def log_image(self, tag: str, image: torch.Tensor, step: int):
        """
        Log an image.

        Args:
            tag: Image name
            image: Image tensor (C x H x W)
            step: Training step/epoch
        """
        # Log to TensorBoard
        self.writer.add_image(tag, image, step)

        # Log to W&B
        if self.use_wandb:
            import wandb
            self.wandb.log({tag: wandb.Image(image)}, step=step)

    def log_histogram(self, tag: str, values: torch.Tensor, step: int):
        """
        Log a histogram of values.

        Useful for:
        - Monitoring embedding distributions
        - Tracking gradient magnitudes
        - Visualizing distance distributions

        Args:
            tag: Histogram name
            values: Tensor of values
            step: Training step/epoch
        """
        # Log to TensorBoard
        self.writer.add_histogram(tag, values, step)

        # Log to W&B
        if self.use_wandb:
            import wandb
            self.wandb.log({tag: wandb.Histogram(values.cpu().numpy())}, step=step)

    def log_text(self, tag: str, text: str, step: int):
        """
        Log text information.

        Args:
            tag: Text identifier
            text: Text content
            step: Training step/epoch
        """
        # Log to TensorBoard
        self.writer.add_text(tag, text, step)

        # Log to W&B
        if self.use_wandb:
            self.wandb.log({tag: text}, step=step)

    def close(self):
        """Close all logging backends."""
        self._flush_scalar_history()
        self.writer.close()
        if self.use_wandb:
            self.wandb.finish()
        print("Logger closed")
