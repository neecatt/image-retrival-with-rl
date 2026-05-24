"""
Complete embedding network for image retrieval.

This module combines:
1. Backbone (feature extraction)
2. Pooling (spatial aggregation)
3. Projection head (dimensionality reduction, optional)
4. Normalization (L2 normalization)

The resulting network maps images to normalized embedding vectors
that can be compared using cosine similarity or euclidean distance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .backbone import get_backbone
from .pooling import get_pooling_layer


class EmbeddingNet(nn.Module):
    """
    Complete network for image embedding.
    
    Architecture:
    -------------
    Input Image (3 x H x W)
        ↓
    Backbone CNN
        ↓
    Feature Map (C x H' x W')
        ↓
    Global Pooling
        ↓
    Feature Vector (C)
        ↓
    [Optional] Projection Head
        ↓
    Embedding (D)
        ↓
    L2 Normalization
        ↓
    Normalized Embedding (D)
    
    Why each component:
    -------------------
    - Backbone: Extracts rich visual features
    - Pooling: Creates fixed-size representation
    - Projection: Reduces dimension, adds non-linearity
    - Normalization: Enables cosine similarity comparison
    """
    
    def __init__(
        self,
        architecture: str = 'resnet50',
        pretrained: bool = True,
        pooling: str = 'gem',
        embedding_dim: Optional[int] = None,
        normalize_embeddings: bool = True,
        pooling_kwargs: Optional[dict] = None
    ):
        """
        Initialize embedding network.
        
        Args:
            architecture: Backbone architecture
            pretrained: Use pre-trained weights
            pooling: Pooling strategy ('gem', 'avg', 'max')
            embedding_dim: Output embedding dimension
                          If None, use backbone feature dim
            normalize_embeddings: Apply L2 normalization
            pooling_kwargs: Additional arguments for pooling layer
            
        Design Choices:
        ---------------
        embedding_dim:
            - None: Keep full dimension (e.g., 2048 for ResNet50)
            - 512/256/128: Reduce dimension for efficiency
            - Trade-off: larger dim = more info, but slower search
        
        normalize_embeddings:
            - True: Use cosine similarity (standard for retrieval)
            - False: Use euclidean distance
        """
        super(EmbeddingNet, self).__init__()
        
        # Initialize backbone
        self.backbone, self.feature_dim = get_backbone(
            architecture=architecture,
            pretrained=pretrained
        )
        
        # Initialize pooling layer
        if pooling_kwargs is None:
            pooling_kwargs = {}
        self.pooling = get_pooling_layer(pooling, **pooling_kwargs)
        
        # Projection head (optional)
        if embedding_dim is not None and embedding_dim != self.feature_dim:
            # Add projection head: FC + BN + ReLU + FC
            self.projection = nn.Sequential(
                nn.Linear(self.feature_dim, embedding_dim),
                nn.BatchNorm1d(embedding_dim),
                nn.ReLU(inplace=True),
            )
            self.output_dim = embedding_dim
        else:
            self.projection = None
            self.output_dim = self.feature_dim
        
        self.normalize_embeddings = normalize_embeddings
        
        print(f"EmbeddingNet initialized:")
        print(f"  - Architecture: {architecture}")
        print(f"  - Pooling: {pooling}")
        print(f"  - Feature dim: {self.feature_dim}")
        print(f"  - Output dim: {self.output_dim}")
        print(f"  - Normalize: {normalize_embeddings}")
    
    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        """
        Forward pass through network.
        
        Args:
            x: Input images (B, 3, H, W)
            return_features: If True, return intermediate features
            
        Returns:
            Embeddings (B, output_dim)
            If return_features: tuple of (embeddings, feature_maps)
            
        Why return_features:
        --------------------
        Useful for:
        - Visualization (what the model sees)
        - Debugging (check feature quality)
        - Advanced pooling strategies
        """
        # Extract features
        features = self.backbone(x)
        
        # Pool to fixed size
        pooled = self.pooling(features)
        
        # Apply projection if available
        if self.projection is not None:
            embeddings = self.projection(pooled)
        else:
            embeddings = pooled
        
        # Normalize embeddings
        if self.normalize_embeddings:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        if return_features:
            return embeddings, features
        return embeddings
    
    def get_embedding_dim(self) -> int:
        """Get output embedding dimension."""
        return self.output_dim
    
    def freeze_backbone(self):
        """
        Freeze backbone parameters.
        
        Useful for:
        - Fine-tuning only the projection head
        - Preventing overfitting with limited data
        - Faster training
        """
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("Backbone frozen")
    
    def unfreeze_backbone(self):
        """Unfreeze backbone for end-to-end training."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("Backbone unfrozen")


class EmbeddingNetWithPolicy(nn.Module):
    """
    Embedding network with policy head for RL training.
    
    This extends the base embedding network with a policy head that
    outputs action probabilities or value estimates for RL training.
    
    Architecture:
    -------------
    Input Image
        ↓
    Embedding Network → Embeddings (for retrieval)
        ↓
    Policy Head → Action Logits / Value (for RL)
    
    Why separate heads:
    -------------------
    - Embedding head: Optimized for retrieval (cosine similarity)
    - Policy head: Optimized for RL objective (maximize reward)
    - Shared backbone: Efficient, features useful for both tasks
    
    Actions in retrieval context:
    -----------------------------
    Actions can be:
    1. Which images to compare (sample mining strategy)
    2. Adjustments to embeddings (learned post-processing)
    3. Re-ranking decisions
    
    For this implementation, we focus on learning embeddings
    that directly maximize retrieval metrics (mAP, recall@k).
    """
    
    def __init__(
        self,
        base_net: EmbeddingNet,
        policy_type: str = 'actor_critic',
        hidden_dim: int = 256
    ):
        """
        Initialize embedding network with policy head.
        
        Args:
            base_net: Base embedding network
            policy_type: Type of policy ('actor_critic', 'policy_only')
            hidden_dim: Hidden dimension for policy head
        """
        super(EmbeddingNetWithPolicy, self).__init__()
        
        self.embedding_net = base_net
        self.policy_type = policy_type
        
        input_dim = base_net.output_dim
        
        if policy_type == 'actor_critic':
            # Actor head: outputs action distribution
            self.actor = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim),  # Output: adjustment to embedding
                nn.Tanh()  # Bounded output
            )
            
            # Critic head: estimates value function
            self.critic = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)  # Output: value estimate
            )
        
        elif policy_type == 'policy_only':
            # Policy gradient without value function
            self.actor = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim),
                nn.Tanh()
            )
            self.critic = None
        
        else:
            raise ValueError(f"Unknown policy type: {policy_type}")
        
        print(f"Policy head initialized: {policy_type}")
    
    def forward(
        self,
        x: torch.Tensor,
        return_policy: bool = False
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input images
            return_policy: If True, return policy outputs
            
        Returns:
            If return_policy=False: embeddings
            If return_policy=True: (embeddings, action_adjustment, value)
        """
        # Get base embeddings
        embeddings = self.embedding_net(x)
        
        if not return_policy:
            return embeddings
        
        # Compute policy outputs
        action_adjustment = self.actor(embeddings)
        
        if self.critic is not None:
            value = self.critic(embeddings)
        else:
            value = None
        
        return embeddings, action_adjustment, value
    
    def get_adjusted_embeddings(self, x: torch.Tensor, epsilon: float = 0.1) -> torch.Tensor:
        """
        Get embeddings with policy-based adjustment.
        
        Args:
            x: Input images
            epsilon: Scale factor for adjustment
            
        Returns:
            Adjusted embeddings
            
        Why adjust embeddings:
        ----------------------
        The policy learns small adjustments to embeddings that
        improve retrieval performance. This is a form of learned
        post-processing.
        """
        embeddings, action_adjustment, _ = self.forward(x, return_policy=True)
        
        # Apply adjustment
        adjusted = embeddings + epsilon * action_adjustment
        
        # Re-normalize
        adjusted = F.normalize(adjusted, p=2, dim=1)
        
        return adjusted

