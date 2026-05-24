"""
Pooling layers for image retrieval.

Global pooling layers aggregate spatial feature maps into a single
embedding vector. Different pooling strategies have different properties.

Key Pooling Methods:
--------------------
1. Average Pooling: Simple mean across spatial dimensions
2. Max Pooling: Takes maximum activation
3. GeM (Generalized Mean): Learnable interpolation between avg and max

Why pooling matters:
--------------------
- Converts variable-size feature maps to fixed-size embeddings
- Provides spatial invariance (small position changes don't affect embedding)
- Different pooling strategies emphasize different features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeM(nn.Module):
    """
    Generalized Mean Pooling (GeM).
    
    GeM is a learnable pooling layer that generalizes both average and max pooling:
    
    GeM(X) = (1/N * sum(x_i^p))^(1/p)
    
    Where:
    - p = 1: Equivalent to average pooling
    - p → ∞: Approaches max pooling
    - p = 3: Good default (empirically found in retrieval tasks)
    
    Why GeM is better for retrieval:
    --------------------------------
    - Average pooling: Treats all features equally (may dilute important features)
    - Max pooling: Only keeps strongest feature (loses information)
    - GeM: Balances between the two, emphasizes strong features without ignoring others
    
    The parameter p is learned during training, allowing the model to
    automatically find the best pooling strategy for the task.
    
    Reference:
    ----------
    "Fine-tuning CNN Image Retrieval with No Human Annotation"
    Radenović et al., TPAMI 2018
    """
    
    def __init__(self, p: float = 3.0, eps: float = 1e-6, learnable_p: bool = True):
        """
        Initialize GeM pooling.
        
        Args:
            p: Initial power parameter (default 3.0)
            eps: Small constant for numerical stability
            learnable_p: Whether to make p a learnable parameter
            
        Why eps is needed:
        ------------------
        When computing x^p for small x and large p, we can get numerical issues.
        eps ensures we never take 0^p or get NaN in gradients.
        """
        super(GeM, self).__init__()
        
        if learnable_p:
            # Make p a learnable parameter
            self.p = nn.Parameter(torch.ones(1) * p)
        else:
            # Keep p fixed
            self.register_buffer('p', torch.ones(1) * p)
        
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply GeM pooling.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Pooled tensor of shape (batch_size, channels)
            
        Process:
        --------
        1. Apply power: x^p (with eps for stability)
        2. Average over spatial dimensions
        3. Apply inverse power: (...)^(1/p)
        """
        # Clamp to avoid issues with small values
        x = x.clamp(min=self.eps)
        
        # Apply generalized mean
        # (1/N * sum(x^p))^(1/p)
        x_pow = torch.pow(x, self.p)
        pooled = F.adaptive_avg_pool2d(x_pow, (1, 1))
        result = torch.pow(pooled, 1.0 / self.p)
        
        # Remove spatial dimensions
        return result.flatten(1)
    
    def __repr__(self):
        """String representation."""
        return f"{self.__class__.__name__}(p={self.p.item():.4f}, eps={self.eps})"


class GeneralizedMeanPooling(nn.Module):
    """
    Alternative implementation of GeM with slightly different parameterization.
    
    This version computes GeM as:
    mean(x^p)^(1/p)
    
    Both implementations are mathematically equivalent but may have
    slightly different numerical properties.
    """
    
    def __init__(self, norm: float = 3.0, output_size: int = 1, eps: float = 1e-6):
        """
        Initialize GeM pooling.
        
        Args:
            norm: Power parameter (same as p in GeM)
            output_size: Output spatial size (typically 1 for global pooling)
            eps: Small constant for stability
        """
        super(GeneralizedMeanPooling, self).__init__()
        self.norm = nn.Parameter(torch.ones(1) * norm)
        self.output_size = output_size
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply GeM pooling."""
        x = x.clamp(min=self.eps).pow(self.norm)
        x = F.adaptive_avg_pool2d(x, self.output_size).pow(1.0 / self.norm)
        return x.flatten(1)
    
    def __repr__(self):
        return f"{self.__class__.__name__}(norm={self.norm.item():.4f})"


class MAC(nn.Module):
    """
    Maximum Activations of Convolutions (MAC).
    
    Simply takes the maximum activation across spatial dimensions.
    
    Properties:
    -----------
    - Very simple and fast
    - Focus on most discriminative feature
    - Loses spatial information
    - Can be sensitive to noise/outliers
    
    When to use:
    ------------
    - When objects are small relative to image
    - When you want maximum invariance to position
    - As a baseline to compare against more complex pooling
    """
    
    def __init__(self):
        super(MAC, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply max pooling.
        
        Args:
            x: Input tensor (batch_size, channels, height, width)
            
        Returns:
            Pooled tensor (batch_size, channels)
        """
        return F.adaptive_max_pool2d(x, (1, 1)).flatten(1)


class SPoC(nn.Module):
    """
    Sum Pooling of Convolutions (SPoC).
    
    Sums (rather than averages) activations across spatial dimensions.
    
    Properties:
    -----------
    - Similar to average pooling but without normalization
    - Magnitude depends on number of spatial locations
    - Can be useful when combined with normalization layer
    
    Reference:
    ----------
    "Aggregating Deep Convolutional Features for Image Retrieval"
    Babenko & Lempitsky, ICCV 2015
    """
    
    def __init__(self):
        super(SPoC, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply sum pooling.
        
        Args:
            x: Input tensor (batch_size, channels, height, width)
            
        Returns:
            Pooled tensor (batch_size, channels)
        """
        # Sum over spatial dimensions
        return torch.sum(x, dim=(-2, -1))


class RMAC(nn.Module):
    """
    Regional Maximum Activations of Convolutions (R-MAC).
    
    Instead of pooling over the entire feature map, R-MAC:
    1. Divides the feature map into overlapping regions
    2. Applies MAC to each region
    3. Aggregates region features
    
    Why R-MAC:
    ----------
    - Captures spatial layout information
    - More robust to clutter and occlusion
    - Better for images with multiple objects
    - Improves recall for partial matches
    
    Trade-offs:
    -----------
    + More discriminative than global pooling
    + Spatial awareness
    - More computation
    - Larger embedding size (unless aggregated)
    
    Reference:
    ----------
    "Cross-dimensional Weighting for Aggregated Deep Convolutional Features"
    Kalantidis et al., ECCV 2016
    """
    
    def __init__(self, level_n: int = 3):
        """
        Initialize R-MAC.
        
        Args:
            level_n: Number of pyramid levels
                    level_n=3 means regions at 3 different scales
        """
        super(RMAC, self).__init__()
        self.level_n = level_n
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply R-MAC pooling.
        
        Process:
        --------
        1. For each pyramid level:
           - Divide feature map into overlapping regions
           - Apply MAC to each region
        2. Aggregate all regional features
        3. L2-normalize the result
        
        Args:
            x: Input tensor (batch_size, channels, height, width)
            
        Returns:
            Pooled tensor (batch_size, channels)
        """
        batch_size, channels, height, width = x.size()
        
        # Collect features from all regions
        features = []
        
        for level in range(1, self.level_n + 1):
            # Calculate region size for this level
            h_step = height // level
            w_step = width // level
            
            # Extract features from each region
            for i in range(level):
                for j in range(level):
                    # Define region boundaries
                    h_start = i * h_step
                    h_end = min((i + 1) * h_step, height)
                    w_start = j * w_step
                    w_end = min((j + 1) * w_step, width)
                    
                    # Extract region
                    region = x[:, :, h_start:h_end, w_start:w_end]
                    
                    # Apply MAC to region
                    region_feat = F.adaptive_max_pool2d(region, (1, 1)).flatten(1)
                    features.append(region_feat)
        
        # Stack and sum all regional features
        features = torch.stack(features, dim=0).sum(dim=0)
        
        # L2 normalize
        features = F.normalize(features, p=2, dim=1)
        
        return features


def get_pooling_layer(pooling_type: str = 'gem', **kwargs) -> nn.Module:
    """
    Factory function to create pooling layers.
    
    Args:
        pooling_type: Type of pooling ('gem', 'avg', 'max', 'mac', 'spoc', 'rmac')
        **kwargs: Additional arguments for the pooling layer
        
    Returns:
        Pooling layer module
        
    Usage:
    ------
    pooling = get_pooling_layer('gem', p=3.0)
    pooling = get_pooling_layer('rmac', level_n=3)
    """
    pooling_type = pooling_type.lower()
    
    if pooling_type == 'gem':
        return GeM(**kwargs)
    elif pooling_type == 'avg':
        return nn.AdaptiveAvgPool2d((1, 1))
    elif pooling_type == 'max':
        return nn.AdaptiveMaxPool2d((1, 1))
    elif pooling_type == 'mac':
        return MAC()
    elif pooling_type == 'spoc':
        return SPoC()
    elif pooling_type == 'rmac':
        return RMAC(**kwargs)
    else:
        raise ValueError(f"Unknown pooling type: {pooling_type}")

