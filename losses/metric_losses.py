"""
Metric learning loss functions.

These losses train the embedding network to produce embeddings where:
- Similar images (same class) are close together
- Dissimilar images (different classes) are far apart

Common loss functions:
- Triplet Loss: Learn from anchor-positive-negative triplets
- Contrastive Loss: Learn from positive and negative pairs
- FastAP Loss: Directly optimize Average Precision
- ArcFace Loss: Angular margin-based classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np


class TripletLoss(nn.Module):
    """
    Triplet Loss for metric learning.
    
    Concept:
    --------
    For each anchor image, we want:
    - Distance to positive (same class) < Distance to negative (different class)
    
    Loss = max(0, d(anchor, positive) - d(anchor, negative) + margin)
    
    The margin ensures that negatives are not just farther, but
    farther by at least 'margin'.
    
    Triplet Mining:
    ---------------
    Critical for effective training:
    
    1. Easy triplets: d(a,p) + margin < d(a,n)
       - Loss = 0, no gradient
       - Don't help learning
    
    2. Hard triplets: d(a,n) < d(a,p)
       - Negative closer than positive!
       - High loss, strong gradient
    
    3. Semi-hard triplets: d(a,p) < d(a,n) < d(a,p) + margin
       - Negative farther than positive but within margin
       - Moderate loss, useful for learning
    
    Mining Strategy:
    ----------------
    - All triplets: Use every combination (very slow, mostly easy)
    - Hard mining: Only use hardest triplets (can be unstable)
    - Semi-hard mining: Use semi-hard triplets (good balance)
    - Batch hard: For each anchor, use hardest positive and negative in batch
    
    Reference:
    ----------
    "FaceNet: A Unified Embedding for Face Recognition and Clustering"
    Schroff et al., CVPR 2015
    """
    
    def __init__(
        self,
        margin: float = 0.2,
        mining: str = 'batch_hard',
        distance_metric: str = 'euclidean'
    ):
        """
        Initialize triplet loss.
        
        Args:
            margin: Margin value (typically 0.1 to 0.5)
            mining: Mining strategy ('all', 'hard', 'semi_hard', 'batch_hard')
            distance_metric: Distance metric ('euclidean', 'cosine')
        """
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.mining = mining
        self.distance_metric = distance_metric
        
        print(f"TripletLoss: margin={margin}, mining={mining}, distance={distance_metric}")
    
    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute triplet loss.
        
        Args:
            embeddings: Embedding vectors (B, D)
            labels: Class labels (B,)
            
        Returns:
            Loss value
            
        Process:
        --------
        1. Compute pairwise distances
        2. Mine triplets based on strategy
        3. Compute triplet loss
        4. Average over valid triplets
        """
        # Compute pairwise distance matrix
        distance_matrix = self._pairwise_distance(embeddings)
        
        if self.mining == 'batch_hard':
            loss = self._batch_hard_triplet_loss(distance_matrix, labels)
        elif self.mining == 'all':
            loss = self._all_triplet_loss(distance_matrix, labels)
        else:
            raise NotImplementedError(f"Mining strategy {self.mining} not implemented")
        
        return loss
    
    def _pairwise_distance(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise distance matrix.
        
        Args:
            embeddings: Embedding vectors (B, D)
            
        Returns:
            Distance matrix (B, B)
        """
        if self.distance_metric == 'euclidean':
            # Efficient euclidean distance: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a·b
            dot_product = torch.mm(embeddings, embeddings.t())
            squared_norm = torch.diag(dot_product)
            distances = squared_norm.unsqueeze(0) - 2 * dot_product + squared_norm.unsqueeze(1)
            distances = torch.clamp(distances, min=0.0)  # Numerical stability
            distances = torch.sqrt(distances + 1e-16)  # Add epsilon for gradient stability
            
        elif self.distance_metric == 'cosine':
            # Cosine distance = 1 - cosine similarity
            normalized = F.normalize(embeddings, p=2, dim=1)
            similarity = torch.mm(normalized, normalized.t())
            distances = 1 - similarity
            
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
        
        return distances
    
    def _batch_hard_triplet_loss(
        self,
        distance_matrix: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Batch hard triplet loss.
        
        For each anchor:
        - Hardest positive: Most distant sample with same label
        - Hardest negative: Closest sample with different label
        
        This is the most commonly used strategy as it:
        - Is efficient (only one triplet per anchor)
        - Focuses on hardest examples
        - Is relatively stable
        
        Args:
            distance_matrix: Pairwise distances (B, B)
            labels: Class labels (B,)
            
        Returns:
            Loss value
        """
        batch_size = distance_matrix.size(0)
        
        # Create masks for positive and negative pairs
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)  # (B, B)
        labels_not_equal = ~labels_equal
        
        # Mask out self-comparisons (diagonal)
        mask_not_self = ~torch.eye(batch_size, dtype=torch.bool, device=distance_matrix.device)
        
        # Positive mask: same label, not self
        positive_mask = labels_equal & mask_not_self
        
        # Negative mask: different label
        negative_mask = labels_not_equal
        
        # For each anchor, find hardest positive (maximum distance among positives)
        # Set distances to -inf where there's no positive
        positive_distances = distance_matrix.clone()
        positive_distances[~positive_mask] = float('-inf')
        hardest_positive_dist, _ = torch.max(positive_distances, dim=1)
        
        # For each anchor, find hardest negative (minimum distance among negatives)
        # Set distances to +inf where there's no negative
        negative_distances = distance_matrix.clone()
        negative_distances[~negative_mask] = float('inf')
        hardest_negative_dist, _ = torch.min(negative_distances, dim=1)
        
        # Compute triplet loss: max(0, d(a,p) - d(a,n) + margin)
        triplet_loss = F.relu(hardest_positive_dist - hardest_negative_dist + self.margin)
        
        # Only consider anchors that have at least one positive and one negative
        valid_triplets = (hardest_positive_dist > float('-inf')) & (hardest_negative_dist < float('inf'))
        
        if valid_triplets.sum() > 0:
            loss = triplet_loss[valid_triplets].mean()
        else:
            loss = torch.tensor(0.0, device=distance_matrix.device)
        
        return loss
    
    def _all_triplet_loss(
        self,
        distance_matrix: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        All triplet loss (uses all valid triplets).
        
        Warning: This can be very slow for large batches!
        Number of triplets = O(B^3)
        
        Args:
            distance_matrix: Pairwise distances (B, B)
            labels: Class labels (B,)
            
        Returns:
            Loss value
        """
        batch_size = distance_matrix.size(0)
        
        # Expand dimensions for broadcasting
        # anchor_positive_dist: (B, B, 1)
        # anchor_negative_dist: (B, 1, B)
        anchor_positive_dist = distance_matrix.unsqueeze(2)
        anchor_negative_dist = distance_matrix.unsqueeze(1)
        
        # Compute triplet loss for all combinations
        # Shape: (B, B, B) where [i,j,k] is triplet (i=anchor, j=positive, k=negative)
        triplet_loss = anchor_positive_dist - anchor_negative_dist + self.margin
        
        # Create mask for valid triplets
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)  # (B, B)
        i_not_equal_j = ~torch.eye(batch_size, dtype=torch.bool, device=labels.device)
        i_not_equal_k = ~torch.eye(batch_size, dtype=torch.bool, device=labels.device)
        j_not_equal_k = labels.unsqueeze(1) != labels.unsqueeze(2)  # (B, B, B)
        
        valid_labels = (
            labels_equal.unsqueeze(2) &  # i and j have same label
            (~labels_equal).unsqueeze(1) &  # i and k have different labels
            i_not_equal_j.unsqueeze(2) &  # i != j
            i_not_equal_k.unsqueeze(1) &  # i != k
            j_not_equal_k  # j != k
        )
        
        # Only consider triplets with positive loss
        triplet_loss = F.relu(triplet_loss)
        
        # Apply mask and average
        triplet_loss = triplet_loss * valid_labels.float()
        num_valid_triplets = valid_labels.sum()
        
        if num_valid_triplets > 0:
            loss = triplet_loss.sum() / num_valid_triplets
        else:
            loss = torch.tensor(0.0, device=distance_matrix.device)
        
        return loss


def get_metric_loss(
    loss_name: str = 'triplet',
    **kwargs
):
    """
    Factory function to get metric learning loss.
    
    Args:
        loss_name: Name of loss function
        **kwargs: Loss-specific parameters
        
    Returns:
        Loss function instance
        
    Supported losses:
    -----------------
    - 'triplet': Triplet loss (implemented above)
    - 'fastap': FastAP loss (from pytorch-metric-learning)
    - 'multisimilarity': MultiSimilarity loss
    - 'arcface': ArcFace loss
    - 'contrastive': Contrastive loss
    
    For pytorch-metric-learning losses, install:
    pip install pytorch-metric-learning
    """
    loss_name = loss_name.lower()
    
    if loss_name == 'triplet':
        return TripletLoss(**kwargs)
    
    # Try to import pytorch-metric-learning for other losses
    try:
        from pytorch_metric_learning import losses as pml_losses
        
        if loss_name == 'fastap':
            # FastAP: Directly optimizes Average Precision
            # Good for ranking tasks like retrieval
            # Remove margin parameter as FastAPLoss doesn't accept it
            fastap_kwargs = {k: v for k, v in kwargs.items() if k != 'margin'}
            return pml_losses.FastAPLoss(**fastap_kwargs)
        
        elif loss_name == 'multisimilarity':
            # MultiSimilarity: Uses both positive and negative pairs
            # with self-mining of hard examples
            return pml_losses.MultiSimilarityLoss(**kwargs)
        
        elif loss_name == 'arcface':
            # ArcFace: Adds angular margin to softmax
            # Very popular for face recognition
            num_classes = kwargs.pop('num_classes', None)
            embedding_size = kwargs.pop('embedding_size', None)
            if num_classes is None or embedding_size is None:
                raise ValueError("ArcFace requires num_classes and embedding_size")
            return pml_losses.ArcFaceLoss(num_classes, embedding_size, **kwargs)
        
        elif loss_name == 'contrastive':
            # Contrastive: Learn from positive and negative pairs
            return pml_losses.ContrastiveLoss(**kwargs)
        
        elif loss_name == 'ntxent':
            # NT-Xent (Normalized Temperature-scaled Cross Entropy)
            # Used in SimCLR and other contrastive learning methods
            return pml_losses.NTXentLoss(**kwargs)
        
        else:
            raise ValueError(f"Unknown loss: {loss_name}")
    
    except ImportError:
        raise ImportError(
            f"Loss '{loss_name}' requires pytorch-metric-learning. "
            f"Install with: pip install pytorch-metric-learning"
        )

