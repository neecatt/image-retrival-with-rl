"""
Dataset classes for image retrieval.

This module handles:
- Loading images from disk
- Applying data augmentation
- Managing query-gallery splits
- Handling ground truth annotations

Supports datasets like:
- ROxford5k, RParis6k (Revisited datasets)
- CUB-200, Stanford Cars (Fine-grained retrieval)
- Custom datasets
"""

import os
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class ImageRetrievalDataset(Dataset):
    """
    Dataset for image retrieval tasks.
    
    This dataset handles both training and evaluation:
    - Training: Returns image pairs/triplets for metric learning
    - Evaluation: Returns individual images with their IDs
    
    Key Concepts:
    -------------
    Query Images: Images we want to find similar images for
    Gallery Images: Database of images to search through
    Ground Truth: Which gallery images are relevant for each query
    
    Why this structure:
    -------------------
    Image retrieval is different from classification:
    - We don't have fixed classes, we learn similarity
    - Training uses pairs/triplets to learn embeddings
    - Evaluation compares query embeddings to gallery embeddings
    """
    
    def __init__(
        self,
        root_dir: str,
        image_list: List[str],
        labels: Optional[List[int]] = None,
        transform: Optional[transforms.Compose] = None,
        mode: str = 'train'
    ):
        """
        Initialize dataset.
        
        Args:
            root_dir: Root directory containing images
            image_list: List of image paths (relative to root_dir)
            labels: List of labels/classes for each image (for training)
            transform: Image transformations to apply
            mode: 'train' or 'eval'
            
        Training Mode:
            - Uses labels to sample pairs/triplets
            - Applies data augmentation
            - Returns multiple views of same image
            
        Eval Mode:
            - Returns single image with ID
            - Minimal augmentation (resize, normalize only)
            - Used for computing embeddings
        """
        self.root_dir = Path(root_dir)
        self.image_list = image_list
        self.labels = labels
        self.transform = transform
        self.mode = mode
        
        # Build label-to-images mapping for efficient sampling
        if labels is not None and mode == 'train':
            self.label_to_indices = self._build_label_mapping()
        
        print(f"Dataset initialized: {len(self.image_list)} images, mode={mode}")
    
    def _build_label_mapping(self) -> Dict[int, List[int]]:
        """
        Build a mapping from labels to image indices.
        
        Why this is needed:
        -------------------
        For triplet mining, we need to quickly find:
        - Positive examples: Images with same label as anchor
        - Negative examples: Images with different label
        
        This mapping allows O(1) lookup instead of O(n) search.
        
        Returns:
            Dictionary mapping label -> list of image indices
        """
        label_to_indices = {}
        for idx, label in enumerate(self.labels):
            if label not in label_to_indices:
                label_to_indices[label] = []
            label_to_indices[label].append(idx)
        
        return label_to_indices
    
    def __len__(self) -> int:
        """Return number of images in dataset."""
        return len(self.image_list)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        """
        Get a single item from the dataset.
        
        Args:
            idx: Index of image to retrieve
            
        Returns:
            Tuple of (image_tensor, label, index)
            
        Process:
        --------
        1. Load image from disk
        2. Convert to RGB (handle grayscale/RGBA)
        3. Apply transformations (resize, augment, normalize)
        4. Return as tensor with metadata
        """
        # Load image
        img_path = self.root_dir / self.image_list[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Apply transformations
        if self.transform is not None:
            image = self.transform(image)
        
        # Get label (or -1 if not available)
        label = self.labels[idx] if self.labels is not None else -1
        
        return image, label, idx
    
    def get_positive_indices(self, idx: int) -> List[int]:
        """
        Get indices of images with same label (positives).
        
        Used for:
        - Sampling positive pairs in contrastive learning
        - Finding positive examples in triplet loss
        - Data augmentation through positive sampling
        
        Args:
            idx: Index of anchor image
            
        Returns:
            List of indices with same label (excluding idx itself)
        """
        if self.labels is None:
            raise ValueError("Labels not available")
        
        label = self.labels[idx]
        positive_indices = [i for i in self.label_to_indices[label] if i != idx]
        
        return positive_indices
    
    def get_negative_indices(self, idx: int) -> List[int]:
        """
        Get indices of images with different label (negatives).
        
        Args:
            idx: Index of anchor image
            
        Returns:
            List of indices with different labels
        """
        if self.labels is None:
            raise ValueError("Labels not available")
        
        label = self.labels[idx]
        negative_indices = [i for i, l in enumerate(self.labels) if l != label]
        
        return negative_indices


class RevisitedDataset:
    """
    Handler for Revisited Oxford/Paris datasets.
    
    These are standard benchmarks for image retrieval:
    - ROxford5k: 4,993 database images + 70 queries
    - RParis6k: 6,322 database images + 70 queries
    
    Dataset Structure:
    ------------------
    - Database images: Gallery to search through
    - Query images: Images to find matches for
    - Ground truth: Easy/Hard/Junk labels for each query
    
    Evaluation Protocol:
    --------------------
    For each query, gallery images are ranked by similarity.
    Ground truth categories:
    - Easy: Clear matches (used in evaluation)
    - Hard: Harder matches (used in harder evaluation)
    - Junk: Ignore (don't count as correct or incorrect)
    
    Reference:
    ----------
    "Revisiting Oxford and Paris: Large-Scale Image Retrieval Benchmarking"
    Radenović et al., CVPR 2018
    """
    
    def __init__(self, root_dir: str, dataset_name: str = 'roxford5k'):
        """
        Initialize Revisited dataset.
        
        Args:
            root_dir: Root directory containing the dataset
            dataset_name: 'roxford5k' or 'rparis6k'
        """
        self.root_dir = Path(root_dir)
        self.dataset_name = dataset_name
        
        # Load dataset configuration
        self.cfg = self._load_config()

        # Normalize image names so ImageRetrievalDataset can open files reliably
        def _normalize_name(name: str) -> str:
            s = str(name).replace("\\", "/")
            if s.startswith("jpg/"):
                s = s[4:]
            if not (s.lower().endswith(".jpg") or s.lower().endswith(".jpeg") or s.lower().endswith(".png")):
                s = s + ".jpg"
            return s

        if 'db_images' in self.cfg:
            self.cfg['db_images'] = [_normalize_name(x) for x in self.cfg['db_images']]
        if 'query_images' in self.cfg:
            self.cfg['query_images'] = [_normalize_name(x) for x in self.cfg['query_images']]
        
        print(f"Loaded {dataset_name}:")
        print(f"  - Database images: {len(self.cfg['db_images'])}")
        print(f"  - Query images: {len(self.cfg['query_images'])}")
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load dataset configuration from disk.
        
        The configuration file (typically .pkl or .mat) contains:
        - List of database image paths
        - List of query image paths
        - Ground truth annotations
        - Query bounding boxes (optional)
        
        Returns:
            Configuration dictionary
        """
        import pickle
        
        # Try to load configuration file
        config_file = self.root_dir / f"{self.dataset_name}.pkl"
        
        if not config_file.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {config_file}\n"
                f"Please download the dataset from:\n"
                f"https://github.com/filipradenovic/revisitop"
            )
        
        with open(config_file, 'rb') as f:
            cfg = pickle.load(f)
        
        return cfg
    
    def get_database_dataset(
        self,
        transform: Optional[transforms.Compose] = None
    ) -> ImageRetrievalDataset:
        """
        Get dataset for database (gallery) images.
        
        Returns:
            ImageRetrievalDataset for gallery images
        """
        return ImageRetrievalDataset(
            root_dir=self.root_dir / 'jpg',
            image_list=self.cfg['db_images'],
            labels=None,  # No labels for evaluation
            transform=transform,
            mode='eval'
        )
    
    def get_query_dataset(
        self,
        transform: Optional[transforms.Compose] = None
    ) -> 'QueryDataset':
        """Get dataset for query images with bounding-box cropping."""
        bbxs = [entry.get('bbx', None) for entry in self.cfg['gnd']]
        return QueryDataset(
            root_dir=self.root_dir / 'jpg',
            image_list=self.cfg['query_images'],
            bbxs=bbxs,
            transform=transform,
        )
    
    def get_ground_truth(self) -> Dict[int, Dict[str, List[int]]]:
        """
        Get ground truth annotations for evaluation.
        
        Returns:
            Dictionary mapping query_idx -> {
                'easy': list of easy positive indices,
                'hard': list of hard positive indices,
                'junk': list of junk indices
            }
        """
        return self.cfg['gnd']


class QueryDataset(Dataset):
    """Query dataset that crops images to their bounding box before transform.

    The revisited benchmark defines a bounding box per query that isolates the
    region of interest (e.g. a building facade). Standard evaluation requires
    cropping to this box before computing the embedding.
    """

    def __init__(self, root_dir, image_list, bbxs, transform=None):
        self.root_dir = Path(root_dir)
        self.image_list = image_list
        self.bbxs = bbxs
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = self.root_dir / self.image_list[idx]
        image = Image.open(img_path).convert('RGB')

        bbx = self.bbxs[idx]
        if bbx is not None:
            x0, y0, x1, y1 = [int(round(c)) for c in bbx]
            image = image.crop((x0, y0, x1, y1))

        if self.transform is not None:
            image = self.transform(image)

        return image, -1, idx


def get_transforms(
    image_size: int = 224,
    mode: str = 'train',
    augmentation_config: Optional[Dict[str, Any]] = None
) -> transforms.Compose:
    """
    Get image transformations for training or evaluation.
    
    Args:
        image_size: Target image size
        mode: 'train' or 'eval'
        augmentation_config: Data augmentation parameters
        
    Returns:
        Composed transformation pipeline
        
    Training Transformations:
    -------------------------
    1. Resize: Scale to slightly larger than target
    2. Random Crop: Crop to target size (adds translation invariance)
    3. Random Flip: Horizontal flip (50% chance)
    4. Color Jitter: Vary brightness/contrast/saturation
    5. Random Rotation: Small rotations for robustness
    6. Normalize: Zero mean, unit variance (ImageNet stats)
    
    Why augmentation matters:
    -------------------------
    - Increases effective dataset size
    - Improves generalization
    - Adds invariance to irrelevant transformations
    - Prevents overfitting
    
    Evaluation Transformations:
    ---------------------------
    1. Resize: Scale to target size
    2. Center Crop: Deterministic crop
    3. Normalize: Same as training
    
    No augmentation in evaluation:
    - Want consistent, reproducible results
    - No randomness in feature extraction
    """
    if augmentation_config is None:
        augmentation_config = {}
    
    # ImageNet normalization (standard for pre-trained models)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet mean (RGB)
        std=[0.229, 0.224, 0.225]     # ImageNet std (RGB)
    )
    
    if mode == 'train':
        # Training augmentations
        transform_list = [
            transforms.Resize(int(image_size * 1.1)),  # Slightly larger
            transforms.RandomCrop(image_size),
        ]
        
        # Optional augmentations
        if augmentation_config.get('random_flip', True):
            transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
        
        if augmentation_config.get('color_jitter', True):
            transform_list.append(transforms.ColorJitter(
                brightness=augmentation_config.get('brightness', 0.2),
                contrast=augmentation_config.get('contrast', 0.2),
                saturation=augmentation_config.get('saturation', 0.2),
                hue=augmentation_config.get('hue', 0.1)
            ))
        
        if augmentation_config.get('rotation', 0) > 0:
            transform_list.append(transforms.RandomRotation(
                augmentation_config['rotation']
            ))
        
        # Convert to tensor and normalize
        transform_list.extend([
            transforms.ToTensor(),
            normalize
        ])
        
    else:  # eval mode — fixed square (required for batched DataLoaders)
        transform_list = [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize
        ]

    return transforms.Compose(transform_list)


class LongSideResize:
    """Resize so the longer image side equals `size`, keeping the aspect ratio (PIL bilinear)."""

    def __init__(self, size: int):
        self.size = size

    def __call__(self, img):
        w, h = img.size
        s = self.size / max(w, h)
        return img.resize((max(1, round(w * s)), max(1, round(h * s))), resample=Image.BILINEAR)


def get_revisited_benchmark_transforms(image_size: int = 1024, resize: str = 'long') -> transforms.Compose:
    """Single-scale Revisited benchmark transform, aspect ratio kept (no square crop).

    resize='long' (default) sets the longer side to image_size, the usual
    protocol for ROxford/RParis; 'short' sets the shorter side (the previous
    behaviour, which scored Stage 1 ROxford Medium ~10 points lower).
    Use with batch_size=1 only — tensors have variable H/W.
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    if resize == 'long':
        resize_tf = LongSideResize(image_size)
    elif resize == 'short':
        resize_tf = transforms.Resize(image_size)
    else:
        raise ValueError(f"Unknown resize mode: {resize}")
    return transforms.Compose([
        resize_tf,
        transforms.ToTensor(),
        normalize,
    ])
