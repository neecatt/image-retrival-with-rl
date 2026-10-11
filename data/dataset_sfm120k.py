"""
SfM-120k dataset loader for image retrieval training.

Actual pkl structure (retrieval-SfM-120k.pkl):
{
    'train': {
        'cids':    [str, ...]   - image hash (32 hex chars), also the filename
        'cluster': [int, ...]   - 3D model / landmark ID per image (class label)
        'qidxs':   [int, ...]   - query indices into cids list
        'pidxs':   [int, ...]   - positive index for each query (same cluster)
    },
    'val': { same keys, 6403 images }
}

File path formula for a given hash h:
    jpg/{h[-2:]}/{h[-4:-2]}/{h[-6:-4]}/{h}      (no .jpg extension)

Example: hash = '03c816ed23cf5bc4a061945d77c70000'
         path = 'jpg/00/00/c7/03c816ed23cf5bc4a061945d77c70000'

Images with the same cluster integer are positives (same 3D landmark).
"""

import pickle
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image


def _hash_to_path(h: str) -> str:
    """Convert image hash to relative file path inside jpg/."""
    return f"{h[-2:]}/{h[-4:-2]}/{h[-6:-4]}/{h}"


def _gldv2_id_to_path(i: str) -> str:
    """GLDv2 image id (16 hex chars) to its official relative path a/b/c/<id>.jpg."""
    return f"{i[0]}/{i[1]}/{i[2]}/{i}.jpg"


class SfM120kDataset(Dataset):
    """
    SfM-120k images with their cluster (landmark) ID as the class label.
    Compatible with BalancedBatchSampler.
    """

    def __init__(
        self,
        image_root: str,
        image_paths: List[str],     # relative paths inside image_root
        labels: List[int],
        transform: Optional[transforms.Compose] = None,
    ):
        self.image_root = Path(image_root)
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

        label_set = sorted(set(labels))
        self.label_to_indices: Dict[int, List[int]] = {l: [] for l in label_set}
        for idx, l in enumerate(labels):
            self.label_to_indices[l].append(idx)

        print(f"SfM120kDataset: {len(self.image_paths)} images, "
              f"{len(label_set)} clusters")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        img_path = self.image_root / self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"SfM-120k sample missing at {img_path}; this should have been filtered on load. "
                "Check your dataset and run download/extract again if necessary."
            ) from e

        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[idx], idx


def load_gldv2c(
    root_dir: str,
    train_transform: Optional[transforms.Compose] = None,
    val_transform: Optional[transforms.Compose] = None,
    seed: int = 42,
) -> Tuple[SfM120kDataset, SfM120kDataset]:
    """GLDv2-clean as packed by scripts/prepare_gldv2.py: <root>/gldv2c.pkl (SfM-120k keys,
    landmark ids as clusters, held-out landmarks as val) and images under <root>/jpg/a/b/c/<id>.jpg."""
    return load_sfm120k(root_dir, train_transform, val_transform, seed,
                        pkl_name='gldv2c.pkl', path_fn=_gldv2_id_to_path)


def load_sfm120k(
    root_dir: str,
    train_transform: Optional[transforms.Compose] = None,
    val_transform: Optional[transforms.Compose] = None,
    seed: int = 42,
    pkl_name: str = 'retrieval-SfM-120k.pkl',
    path_fn=_hash_to_path,
) -> Tuple[SfM120kDataset, SfM120kDataset]:
    """
    Load SfM-120k from disk.

    Uses the pkl's own 'train'/'val' split:
      - train: 91,642 images across thousands of 3D landmark clusters
      - val:    6,403 images

    Args:
        root_dir:        Path to data/raw/sfm120k/
        train_transform: Augmentation transforms for training images
        val_transform:   Eval-mode transforms for validation images
        seed:            Unused (kept for API consistency)

    Returns:
        (train_dataset, val_dataset)
    """
    root = Path(root_dir)
    image_root = root / 'jpg'
    # Prefer partial pkl if you ran partial_extract_sfm120k.py (quota-limited extraction)
    partial_pkl = root / 'retrieval-SfM-120k-partial.pkl'
    pkl_path = partial_pkl if (partial_pkl.exists() and pkl_name == 'retrieval-SfM-120k.pkl') else root / pkl_name

    if not pkl_path.exists():
        raise FileNotFoundError(
            f"SfM-120k metadata not found: {pkl_path}\n"
            f"Download with:\n"
            f"  wget http://cmp.felk.cvut.cz/cnnimageretrieval/data/train/dbs/"
            f"retrieval-SfM-120k.pkl -O {pkl_path}"
        )

    with open(pkl_path, 'rb') as f:
        db = pickle.load(f)

    def _build_dataset(split_data, transform):
        hashes  = split_data['cids']       # hash strings = filenames
        clusters = split_data['cluster']   # integer cluster IDs

        # Map cluster integers to contiguous label space
        unique_clusters = sorted(set(clusters))
        c_to_label = {c: i for i, c in enumerate(unique_clusters)}

        paths = []
        labels = []
        missing_count = 0
        for h, c in zip(hashes, clusters):
            rel_path = path_fn(h)
            absolute_path = image_root / rel_path
            if absolute_path.exists():
                paths.append(rel_path)
                labels.append(c_to_label[c])
            else:
                missing_count += 1

        if missing_count > 0:
            print(
                f"WARNING: {missing_count} missing images were not found under {image_root}" \
                " and were excluded from this split."
            )

        if len(paths) == 0:
            raise RuntimeError(
                f"No images found under {image_root}. "
                f"Check data integrity and the dataset paths."
            )

        return paths, labels

    train_paths, train_labels = _build_dataset(db['train'], train_transform)
    val_paths,   val_labels   = _build_dataset(db['val'],   val_transform)

    train_ds = SfM120kDataset(image_root, train_paths, train_labels, train_transform)
    val_ds   = SfM120kDataset(image_root, val_paths,   val_labels,   val_transform)

    return train_ds, val_ds
