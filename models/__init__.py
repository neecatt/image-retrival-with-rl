"""
Model architectures for image retrieval.
"""

from .backbone import get_backbone
from .embedding_net import EmbeddingNet
from .pooling import GeM, GeneralizedMeanPooling

__all__ = ['get_backbone', 'EmbeddingNet', 'GeM', 'GeneralizedMeanPooling']

