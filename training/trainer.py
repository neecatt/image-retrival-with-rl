"""
Standard trainer for metric learning (Stage 1).

This trainer uses traditional metric learning losses (Triplet, FastAP, etc.)
to pre-train the embedding network before RL fine-tuning.

Why two-stage training:
-----------------------
Stage 1 (This module): Metric learning with proxy losses
- Fast convergence
- Stable training
- Good initialization for RL

Stage 2 (RL trainer): Direct metric optimization
- Fine-tunes for task metric
- Achieves better final performance
- Builds on Stage 1 initialization

This two-stage approach is more effective than RL from scratch.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Optional
from tqdm import tqdm
import time

from models.embedding_net import EmbeddingNet
from utils.helpers import AverageMeter, format_time
from utils.logger import Logger


class StandardTrainer:
    """
    Trainer for standard metric learning.
    
    Uses traditional losses like:
    - Triplet Loss
    - Contrastive Loss
    - FastAP Loss
    - Multi-Similarity Loss
    
    Training Loop:
    --------------
    1. Forward pass: compute embeddings
    2. Compute loss: compare embeddings using metric learning loss
    3. Backward pass: compute gradients
    4. Update parameters
    """
    
    def __init__(
        self,
        model: EmbeddingNet,
        loss_fn: nn.Module,
        optimizer: optim.Optimizer,
        logger: Logger,
        device: torch.device,
        gradient_clip: float = 0.0,
        use_amp: bool = False
    ):
        """
        Initialize trainer.
        
        Args:
            model: Embedding network
            loss_fn: Metric learning loss function
            optimizer: Optimizer
            logger: Logger for tracking
            device: Device to train on
            gradient_clip: Max gradient norm (0 = no clipping)
            use_amp: Whether to use automatic mixed precision
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.logger = logger
        self.device = device
        self.gradient_clip = gradient_clip
        self.use_amp = use_amp
        
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        
        self.global_step = 0
    
    def train_epoch(
        self,
        train_loader,
        epoch: int,
        log_interval: int = 10
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            epoch: Current epoch number
            log_interval: How often to log metrics
            
        Returns:
            Dictionary of average metrics
        """
        self.model.train()
        
        loss_meter = AverageMeter()
        start_time = time.time()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} (Metric Learning)")
        
        for batch_idx, (images, labels, _) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            loss = self.train_step(images, labels)
            
            # Update meter
            loss_meter.update(loss.item(), images.size(0))
            
            # Update progress bar
            pbar.set_postfix({'loss': f"{loss_meter.avg:.4f}"})
            
            # Logging
            if batch_idx % log_interval == 0:
                self.logger.log_scalar('train/loss', loss.item(), self.global_step)
            
            self.global_step += 1
        
        elapsed_time = time.time() - start_time
        
        print(f"Epoch {epoch} completed in {format_time(elapsed_time)}")
        print(f"  Average Loss: {loss_meter.avg:.4f}")
        
        return {
            'loss': loss_meter.avg,
            'time': elapsed_time
        }
    
    def train_step(
        self,
        images: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Single training step.
        
        Args:
            images: Batch of images (B, C, H, W)
            labels: Batch of labels (B,)
            
        Returns:
            Loss value
        """
        self.optimizer.zero_grad()
        
        if self.use_amp:
            with torch.cuda.amp.autocast():
                # Forward pass
                embeddings = self.model(images)
                
                # Compute loss
                loss = self.loss_fn(embeddings, labels)
            
            # Backward pass with scaler
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            if self.gradient_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip
                )
            
            # Update parameters
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Forward pass
            embeddings = self.model(images)
            
            # Compute loss
            loss = self.loss_fn(embeddings, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip
                )
            
            # Update parameters
            self.optimizer.step()
        
        return loss
    
    def validate(
        self,
        val_loader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            val_loader: DataLoader for validation
            epoch: Current epoch number
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        loss_meter = AverageMeter()
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch} (Validation)")
            
            for images, labels, _ in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                embeddings = self.model(images)
                
                # Compute loss
                loss = self.loss_fn(embeddings, labels)
                
                # Update meter
                loss_meter.update(loss.item(), images.size(0))
                
                pbar.set_postfix({'loss': f"{loss_meter.avg:.4f}"})
        
        print(f"Validation Loss: {loss_meter.avg:.4f}")
        
        # Log to tensorboard/wandb
        self.logger.log_scalar('val/loss', loss_meter.avg, epoch)
        
        return {'loss': loss_meter.avg}

