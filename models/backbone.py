"""
Backbone networks for feature extraction.

This module provides various CNN architectures pre-trained on ImageNet:
- ResNet family (ResNet50, ResNet101, etc.)
- EfficientNet family
- Vision Transformers (ViT)

Why use pre-trained models:
----------------------------
- Transfer learning: Features learned on ImageNet generalize well
- Faster convergence: Start from good initialization
- Better performance: Especially with limited data
- Standard practice in computer vision

The backbone extracts spatial feature maps that are then pooled
into a fixed-size embedding vector.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple, Optional


class ResNetBackbone(nn.Module):
    """
    ResNet backbone for feature extraction.
    
    ResNet (Residual Network) Architecture:
    ---------------------------------------
    - Uses skip connections to enable training very deep networks
    - Standard architecture in computer vision
    - Different depths available: 18, 34, 50, 101, 152 layers
    
    For retrieval, we typically use:
    - ResNet50: Good balance of speed and accuracy
    - ResNet101: Better accuracy, slower
    
    Feature Extraction:
    -------------------
    We remove the final classification layer (fc) and global pooling,
    keeping only the convolutional layers. This gives us:
    - Input: (B, 3, H, W)
    - Output: (B, 2048, H/32, W/32) for ResNet50/101
    
    The spatial feature map preserves some location information that
    can be useful for retrieval (e.g., for R-MAC pooling).
    """
    
    def __init__(
        self,
        architecture: str = 'resnet50',
        pretrained: bool = True,
        freeze_bn: bool = False
    ):
        """
        Initialize ResNet backbone.
        
        Args:
            architecture: ResNet variant ('resnet18', 'resnet50', 'resnet101')
            pretrained: Load ImageNet pre-trained weights
            freeze_bn: Freeze batch normalization layers
            
        Why freeze BatchNorm:
        ---------------------
        When fine-tuning with small batches, BatchNorm statistics
        can be unstable. Freezing BN uses pre-trained statistics.
        """
        super(ResNetBackbone, self).__init__()
        
        # Load pre-trained ResNet
        if architecture == 'resnet18':
            base_model = models.resnet18(pretrained=pretrained)
            self.feature_dim = 512
        elif architecture == 'resnet34':
            base_model = models.resnet34(pretrained=pretrained)
            self.feature_dim = 512
        elif architecture == 'resnet50':
            base_model = models.resnet50(pretrained=pretrained)
            self.feature_dim = 2048
        elif architecture == 'resnet101':
            base_model = models.resnet101(pretrained=pretrained)
            self.feature_dim = 2048
        elif architecture == 'resnet152':
            base_model = models.resnet152(pretrained=pretrained)
            self.feature_dim = 2048
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        # Remove the final FC layer and global pooling
        # Keep: conv1, bn1, relu, maxpool, layer1-4
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        
        if freeze_bn:
            self._freeze_bn()
        
        print(f"Initialized {architecture} backbone (pretrained={pretrained})")
        print(f"Output feature dimension: {self.feature_dim}")
    
    def _freeze_bn(self):
        """
        Freeze batch normalization layers.
        
        Sets BN layers to eval mode and prevents gradient updates.
        """
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through backbone.
        
        Args:
            x: Input images (B, 3, H, W)
            
        Returns:
            Feature maps (B, feature_dim, H', W')
            where H' = H/32, W' = W/32 (due to stride and pooling)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        return x


class EfficientNetBackbone(nn.Module):
    """
    EfficientNet backbone for feature extraction.
    
    EfficientNet:
    -------------
    - More efficient than ResNet (better accuracy/FLOPS trade-off)
    - Compound scaling: jointly scales depth, width, resolution
    - Variants: B0 (smallest) to B7 (largest)
    
    Good choice when:
    ----------------
    - Want better efficiency
    - Limited computational resources
    - Need to deploy on edge devices
    
    For retrieval:
    - EfficientNet-B0: Fast, good for large-scale search
    - EfficientNet-B3/B4: Better accuracy
    """
    
    def __init__(
        self,
        architecture: str = 'efficientnet_b0',
        pretrained: bool = True
    ):
        """
        Initialize EfficientNet backbone.
        
        Args:
            architecture: EfficientNet variant ('efficientnet_b0' to 'efficientnet_b7')
            pretrained: Load ImageNet pre-trained weights
        """
        super(EfficientNetBackbone, self).__init__()
        
        # Load pre-trained EfficientNet
        if architecture == 'efficientnet_b0':
            base_model = models.efficientnet_b0(pretrained=pretrained)
            self.feature_dim = 1280
        elif architecture == 'efficientnet_b1':
            base_model = models.efficientnet_b1(pretrained=pretrained)
            self.feature_dim = 1280
        elif architecture == 'efficientnet_b3':
            base_model = models.efficientnet_b3(pretrained=pretrained)
            self.feature_dim = 1536
        elif architecture == 'efficientnet_b4':
            base_model = models.efficientnet_b4(pretrained=pretrained)
            self.feature_dim = 1792
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        # Extract feature extractor (all layers except classifier)
        self.features = base_model.features
        
        print(f"Initialized {architecture} backbone (pretrained={pretrained})")
        print(f"Output feature dimension: {self.feature_dim}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through backbone.
        
        Args:
            x: Input images (B, 3, H, W)
            
        Returns:
            Feature maps (B, feature_dim, H', W')
        """
        return self.features(x)


def get_backbone(
    architecture: str = 'resnet50',
    pretrained: bool = True,
    **kwargs
) -> Tuple[nn.Module, int]:
    """
    Factory function to create backbone networks.
    
    Args:
        architecture: Backbone architecture name
        pretrained: Load pre-trained weights
        **kwargs: Additional arguments for the backbone
        
    Returns:
        Tuple of (backbone_module, feature_dimension)
        
    Usage:
    ------
    backbone, feat_dim = get_backbone('resnet50', pretrained=True)
    backbone, feat_dim = get_backbone('efficientnet_b0', pretrained=True)
    """
    architecture = architecture.lower()
    
    if 'resnet' in architecture:
        backbone = ResNetBackbone(architecture, pretrained, **kwargs)
        return backbone, backbone.feature_dim
    
    elif 'efficientnet' in architecture:
        backbone = EfficientNetBackbone(architecture, pretrained)
        return backbone, backbone.feature_dim
    
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

