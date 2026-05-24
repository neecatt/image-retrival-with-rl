"""
DataLoader creation and management.

This module provides functions to create dataloaders with appropriate
configurations for training and evaluation.
"""

from typing import Dict, Optional
import torch
from torch.utils.data import DataLoader
from .dataset import ImageRetrievalDataset, get_transforms


def get_dataloaders(
    train_dataset: Optional[ImageRetrievalDataset] = None,
    val_dataset: Optional[ImageRetrievalDataset] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Dict[str, DataLoader]:
    """
    Create dataloaders for training and validation.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        
    Returns:
        Dictionary with 'train' and 'val' dataloaders
        
    DataLoader Configuration:
    -------------------------
    
    num_workers:
        - Parallel data loading in background
        - Prevents GPU starvation (waiting for data)
        - Rule of thumb: 4-8 workers per GPU
        - Too many workers can slow down due to overhead
    
    pin_memory:
        - Pre-allocates tensors in page-locked memory
        - Faster transfer from CPU to GPU
        - Uses slightly more RAM
        - Should be True when using GPU
    
    shuffle:
        - Training: True (different order each epoch)
        - Validation: False (consistent order for reproducibility)
    
    drop_last:
        - Training: True (ignore incomplete last batch)
        - Ensures all batches have same size
        - Important for batch normalization
    """
    dataloaders = {}
    
    if train_dataset is not None:
        dataloaders['train'] = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,  # Randomize order each epoch
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,  # Drop incomplete last batch
            persistent_workers=num_workers > 0  # Keep workers alive between epochs
        )
        print(f"Train dataloader: {len(dataloaders['train'])} batches")
    
    if val_dataset is not None:
        dataloaders['val'] = DataLoader(
            val_dataset,
            batch_size=batch_size * 2,  # Larger batch for eval (no gradients)
            shuffle=False,  # Deterministic order
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False  # Use all validation data
        )
        print(f"Val dataloader: {len(dataloaders['val'])} batches")
    
    return dataloaders


class BalancedBatchSampler(torch.utils.data.Sampler):
    """
    Batch sampler that ensures balanced classes within each batch.
    
    Why this is useful:
    -------------------
    For metric learning, we want each batch to contain multiple
    examples from each class. This ensures:
    - Hard triplets can be mined within the batch
    - Positive and negative pairs are available
    - More efficient use of batch for learning
    
    Strategy:
    ---------
    For each batch:
    1. Sample P different classes
    2. Sample K images from each class
    3. Batch size = P * K
    
    Example: P=8 classes, K=4 images -> batch_size=32
    
    This is called "PK sampling" in metric learning literature.
    """
    
    def __init__(
        self,
        labels: torch.Tensor,
        batch_size: int,
        images_per_class: int
    ):
        """
        Initialize balanced batch sampler.
        
        Args:
            labels: Label for each image
            batch_size: Total batch size
            images_per_class: Number of images to sample per class (K)
        """
        self.labels = labels
        self.batch_size = batch_size
        self.images_per_class = images_per_class
        
        # Calculate number of classes per batch (P)
        self.classes_per_batch = batch_size // images_per_class
        
        if batch_size % images_per_class != 0:
            raise ValueError(
                f"batch_size ({batch_size}) must be divisible by "
                f"images_per_class ({images_per_class})"
            )
        
        # Build label-to-indices mapping
        self.label_to_indices = {}
        for idx, label in enumerate(labels):
            label = label.item() if torch.is_tensor(label) else label
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(idx)
        
        # Convert to lists for efficient sampling
        self.labels_list = list(self.label_to_indices.keys())
        self.num_classes = len(self.labels_list)
        
        # Calculate number of batches
        self.num_batches = len(labels) // batch_size
        
        print(f"BalancedBatchSampler: {self.num_classes} classes, "
              f"{self.classes_per_batch} classes per batch, "
              f"{self.images_per_class} images per class")
    
    def __iter__(self):
        """Generate batches."""
        for _ in range(self.num_batches):
            batch = []
            
            # Sample P classes
            sampled_classes = torch.randperm(self.num_classes)[:self.classes_per_batch]
            
            for class_idx in sampled_classes:
                label = self.labels_list[class_idx]
                indices = self.label_to_indices[label]
                
                # Sample K images from this class (with replacement if needed)
                if len(indices) >= self.images_per_class:
                    # Sample without replacement
                    sampled_indices = torch.randperm(len(indices))[:self.images_per_class]
                    batch.extend([indices[i] for i in sampled_indices])
                else:
                    # Sample with replacement if class has fewer than K images
                    sampled_indices = torch.randint(0, len(indices), (self.images_per_class,))
                    batch.extend([indices[i] for i in sampled_indices])
            
            yield batch
    
    def __len__(self):
        """Return number of batches."""
        return self.num_batches

