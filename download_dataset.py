#!/usr/bin/env python3
"""
Download and prepare Revisited Oxford/Paris datasets for image retrieval.

This script helps download and organize the ROxford5k and RParis6k datasets
from the revisitop repository.

Usage:
    python download_dataset.py roxford5k
    python download_dataset.py rparis6k
    python download_dataset.py all
"""

import os
import sys
import argparse
import subprocess
import urllib.request
import tarfile
import shutil
from pathlib import Path
import pickle

# Dataset information
REVISITOP_REPO = "https://github.com/filipradenovic/revisitop"
DATA_DIR = Path("./data/raw")

# Download URLs
OXFORD_IMAGES_URL = "https://www.robots.ox.ac.uk/~vgg/data/oxbuildings/oxbuild_images-v1.tgz"
PARIS_IMAGES_URL_1 = "https://www.robots.ox.ac.uk/~vgg/data/parisbuildings/paris_1-v1.tgz"
PARIS_IMAGES_URL_2 = "https://www.robots.ox.ac.uk/~vgg/data/parisbuildings/paris_2-v1.tgz"
GND_BASE_URL = "http://cmp.felk.cvut.cz/revisitop/data/datasets"
SFM120K_IMAGES_URL = "http://cmp.felk.cvut.cz/cnnimageretrieval/data/train/ims/ims.tar.gz"
SFM120K_PKL_URL = "http://cmp.felk.cvut.cz/cnnimageretrieval/data/train/dbs/retrieval-SfM-120k.pkl"


def check_dataset_exists(dataset_name):
    """Check if dataset is already downloaded."""
    dataset_dir = DATA_DIR / dataset_name
    jpg_dir = dataset_dir / "jpg"
    pkl_file = dataset_dir / f"{dataset_name}.pkl"
    gnd_pkl_file = dataset_dir / f"gnd_{dataset_name}.pkl"
    
    # Check for both naming conventions
    pkl_exists = pkl_file.exists() or gnd_pkl_file.exists()
    
    if jpg_dir.exists() and pkl_exists:
        # Check if jpg directory has images
        images = list(jpg_dir.glob("*.jpg"))
        if len(images) > 0:
            print(f"✓ {dataset_name} appears to be already downloaded")
            print(f"  Found {len(images)} images")
            return True
    return False


def download_file(url, dst_file, description=""):
    """Download a file from URL with progress indication."""
    print(f"  Downloading {description}...")
    print(f"    From: {url}")
    print(f"    To: {dst_file}")
    
    try:
        def reporthook(count, block_size, total_size):
            if total_size > 0:
                percent = int(count * block_size * 100 / total_size)
                bar_length = 40
                filled = int(bar_length * count * block_size / total_size)
                bar = '=' * filled + '-' * (bar_length - filled)
                print(f"\r    [{bar}] {percent}%", end='', flush=True)
        
        urllib.request.urlretrieve(url, dst_file, reporthook=reporthook)
        print()  # New line after progress
        return True
    except Exception as e:
        print(f"\n  ✗ Download failed: {e}")
        return False


def extract_tar(tar_file, extract_to):
    """Extract tar.gz file and flatten directory structure."""
    print(f"  Extracting {tar_file.name}...")
    
    try:
        # Create temporary directory
        tmp_dir = extract_to / "tmp"
        tmp_dir.mkdir(exist_ok=True)
        
        # Extract to tmp directory
        with tarfile.open(tar_file, 'r:gz') as tar:
            tar.extractall(path=tmp_dir)
        
        # Move all files from subdirectories to main directory
        for root, dirs, files in os.walk(tmp_dir):
            for file in files:
                src = Path(root) / file
                dst = extract_to / file
                if dst.exists():
                    # Skip if file already exists
                    continue
                shutil.move(str(src), str(dst))
        
        # Remove temporary directory
        shutil.rmtree(tmp_dir)
        
        # Remove tar file
        tar_file.unlink()
        print(f"  ✓ Extracted and cleaned up")
        return True
    except Exception as e:
        print(f"  ✗ Extraction failed: {e}")
        return False


def download_dataset(dataset_name):
    """Download a specific dataset."""
    dataset_dir = DATA_DIR / dataset_name
    jpg_dir = dataset_dir / "jpg"
    
    # Create directories
    jpg_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if images already exist
    existing_images = list(jpg_dir.glob("*.jpg"))
    if len(existing_images) > 0:
        print(f"  ✓ Found {len(existing_images)} existing images, skipping image download")
    else:
        if dataset_name == 'roxford5k':
            # Download Oxford images
            tar_file = jpg_dir / "oxbuild_images-v1.tgz"
            if not tar_file.exists():
                if not download_file(OXFORD_IMAGES_URL, tar_file, "Oxford images"):
                    return False
            if tar_file.exists():
                print(f"  Extracting archive...")
                extract_tar(tar_file, jpg_dir)
        
        elif dataset_name == 'rparis6k':
            # Download Paris images (two parts)
            tar_file_1 = jpg_dir / "paris_1-v1.tgz"
            tar_file_2 = jpg_dir / "paris_2-v1.tgz"
            
            if not tar_file_1.exists():
                if not download_file(PARIS_IMAGES_URL_1, tar_file_1, "Paris images part 1"):
                    return False
            if tar_file_1.exists():
                print(f"  Extracting archive 1...")
                extract_tar(tar_file_1, jpg_dir)
            
            if not tar_file_2.exists():
                if not download_file(PARIS_IMAGES_URL_2, tar_file_2, "Paris images part 2"):
                    return False
            if tar_file_2.exists():
                print(f"  Extracting archive 2...")
                extract_tar(tar_file_2, jpg_dir)
        
        else:
            print(f"✗ Unknown dataset: {dataset_name}")
            return False
    
    # Download ground truth file
    gnd_url = f"{GND_BASE_URL}/{dataset_name}/gnd_{dataset_name}.pkl"
    gnd_file = dataset_dir / f"gnd_{dataset_name}.pkl"
    pkl_file = dataset_dir / f"{dataset_name}.pkl"
    
    if not gnd_file.exists() and not pkl_file.exists():
        print(f"  Downloading ground truth file...")
        if download_file(gnd_url, gnd_file, "ground truth"):
            # Convert ground truth file structure if needed
            try:
                with open(gnd_file, 'rb') as f:
                    gnd_data = pickle.load(f)
                
                # Check if we need to convert the structure
                # The downloaded file might have different keys
                if 'imlist' in gnd_data and 'qimlist' in gnd_data:
                    # Convert to expected format
                    cfg = {
                        'db_images': gnd_data.get('imlist', []),
                        'query_images': gnd_data.get('qimlist', []),
                        'gnd': gnd_data.get('gnd', {})
                    }
                    # Ensure lists are Python lists
                    if hasattr(cfg['db_images'], 'tolist'):
                        cfg['db_images'] = cfg['db_images'].tolist()
                    if hasattr(cfg['query_images'], 'tolist'):
                        cfg['query_images'] = cfg['query_images'].tolist()
                    
                    with open(pkl_file, 'wb') as f:
                        pickle.dump(cfg, f)
                else:
                    # File already in correct format, just copy
                    shutil.copy(gnd_file, pkl_file)
                
                print(f"  ✓ Ground truth downloaded and converted")
            except Exception as e:
                print(f"  ⚠ Error processing ground truth: {e}")
                # Try to just copy it anyway
                if not pkl_file.exists():
                    shutil.copy(gnd_file, pkl_file)
        else:
            print(f"  ⚠ Ground truth download failed, but images are downloaded")
            return False
    else:
        print(f"  ✓ Ground truth file already exists")
        # Ensure both naming conventions exist
        if gnd_file.exists() and not pkl_file.exists():
            try:
                # Try to convert structure
                with open(gnd_file, 'rb') as f:
                    gnd_data = pickle.load(f)
                if 'imlist' in gnd_data and 'qimlist' in gnd_data:
                    cfg = {
                        'db_images': gnd_data.get('imlist', []),
                        'query_images': gnd_data.get('qimlist', []),
                        'gnd': gnd_data.get('gnd', {})
                    }
                    if hasattr(cfg['db_images'], 'tolist'):
                        cfg['db_images'] = cfg['db_images'].tolist()
                    if hasattr(cfg['query_images'], 'tolist'):
                        cfg['query_images'] = cfg['query_images'].tolist()
                    with open(pkl_file, 'wb') as f:
                        pickle.dump(cfg, f)
                else:
                    shutil.copy(gnd_file, pkl_file)
            except:
                shutil.copy(gnd_file, pkl_file)
        elif pkl_file.exists() and not gnd_file.exists():
            shutil.copy(pkl_file, gnd_file)
    
    return True




def download_sfm120k():
    """Download SfM-120k training dataset (~10 GB images + metadata pkl)."""
    dataset_dir = DATA_DIR / 'sfm120k'
    jpg_dir = dataset_dir / 'jpg'
    jpg_dir.mkdir(parents=True, exist_ok=True)

    # Download metadata pkl
    pkl_file = dataset_dir / 'retrieval-SfM-120k.pkl'
    if not pkl_file.exists():
        print("  Downloading SfM-120k metadata...")
        if not download_file(SFM120K_PKL_URL, pkl_file, "SfM-120k metadata pkl"):
            return False
    else:
        print("  ✓ Metadata pkl already exists")

    # Download images
    existing = list(jpg_dir.rglob("*.jpg"))
    if len(existing) > 1000:
        print(f"  ✓ Found {len(existing)} existing images, skipping download")
        return True

    tar_file = dataset_dir / 'ims.tar.gz'
    if not tar_file.exists():
        print("  Downloading SfM-120k images (~10 GB, this will take a while)...")
        if not download_file(SFM120K_IMAGES_URL, tar_file, "SfM-120k images"):
            return False

    if tar_file.exists():
        print("  Extracting images...")
        try:
            import tarfile as tarmod
            with tarmod.open(tar_file, 'r:gz') as tar:
                tar.extractall(path=jpg_dir)
            tar_file.unlink()
            print("  ✓ Extracted and cleaned up")
        except Exception as e:
            print(f"  ✗ Extraction failed: {e}")
            return False

    return True


def convert_mat_to_pkl(mat_file, pkl_file, dataset_name):
    """Convert .mat file to .pkl format."""
    try:
        from scipy.io import loadmat
    except ImportError:
        print("✗ scipy is required to convert .mat files")
        print("  Install with: pip install scipy")
        return False
    
    print(f"Converting {mat_file} to {pkl_file}...")
    try:
        # Load MATLAB file
        mat_data = loadmat(
            str(mat_file),
            struct_as_record=False,
            squeeze_me=True
        )
        
        # Extract relevant information
        # Note: Structure may vary, adjust based on actual file
        cfg = {
            'db_images': mat_data.get('imlist', []),
            'query_images': mat_data.get('qimlist', []),
            'gnd': mat_data.get('gnd', {})
        }
        
        # Ensure lists are Python lists, not numpy arrays
        if hasattr(cfg['db_images'], 'tolist'):
            cfg['db_images'] = cfg['db_images'].tolist()
        if hasattr(cfg['query_images'], 'tolist'):
            cfg['query_images'] = cfg['query_images'].tolist()
        
        # Save as pickle
        with open(pkl_file, 'wb') as f:
            pickle.dump(cfg, f)
        
        print(f"✓ Converted {dataset_name} dataset")
        return True
        
    except Exception as e:
        print(f"✗ Error converting file: {e}")
        print("  You may need to adjust the conversion logic based on")
        print("  the actual structure of your .mat file")
        return False


def verify_dataset(dataset_name):
    """Verify that dataset is properly set up."""
    dataset_dir = DATA_DIR / dataset_name
    jpg_dir = dataset_dir / "jpg"
    pkl_file = dataset_dir / f"{dataset_name}.pkl"
    gnd_pkl_file = dataset_dir / f"gnd_{dataset_name}.pkl"
    
    print(f"\nVerifying {dataset_name}...")
    
    issues = []
    
    if not dataset_dir.exists():
        issues.append(f"Dataset directory not found: {dataset_dir}")
    else:
        if not jpg_dir.exists():
            issues.append(f"Images directory not found: {jpg_dir}")
        else:
            images = list(jpg_dir.glob("*.jpg"))
            if len(images) == 0:
                issues.append(f"No .jpg images found in {jpg_dir}")
            else:
                print(f"  ✓ Found {len(images)} images")
        
        # Check for either naming convention
        config_file = pkl_file if pkl_file.exists() else (gnd_pkl_file if gnd_pkl_file.exists() else None)
        
        if config_file is None:
            issues.append(f"Config file not found: {pkl_file} or {gnd_pkl_file}")
        else:
            try:
                with open(config_file, 'rb') as f:
                    cfg = pickle.load(f)
                print(f"  ✓ Config file loaded: {config_file.name}")
                if 'db_images' in cfg:
                    print(f"  ✓ Database images: {len(cfg['db_images'])}")
                if 'query_images' in cfg:
                    print(f"  ✓ Query images: {len(cfg['query_images'])}")
            except Exception as e:
                issues.append(f"Error loading config file: {e}")
    
    if issues:
        print("\n⚠ Issues found:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print(f"\n✓ {dataset_name} dataset is ready!")
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare Revisited Oxford/Paris datasets"
    )
    parser.add_argument(
        'dataset',
        choices=['roxford5k', 'rparis6k', 'sfm120k', 'all'],
        help='Dataset to download'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify existing dataset'
    )
    parser.add_argument(
        '--convert-mat',
        type=str,
        help='Convert .mat file to .pkl (provide path to .mat file)'
    )
    
    args = parser.parse_args()
    
    # Create data directory
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    datasets = ['roxford5k', 'rparis6k', 'sfm120k'] if args.dataset == 'all' else [args.dataset]
    
    for dataset_name in datasets:
        print(f"\n{'=' * 60}")
        print(f"Processing {dataset_name.upper()}")
        print('=' * 60)
        
        # Handle conversion
        if args.convert_mat:
            dataset_dir = DATA_DIR / dataset_name
            dataset_dir.mkdir(parents=True, exist_ok=True)
            pkl_file = dataset_dir / f"{dataset_name}.pkl"
            convert_mat_to_pkl(args.convert_mat, pkl_file, dataset_name)
            continue
        
        # Handle verification
        if args.verify:
            verify_dataset(dataset_name)
            continue
        
        # Check if already downloaded
        if check_dataset_exists(dataset_name):
            print(f"  Dataset {dataset_name} is already downloaded. Skipping...")
            continue
        
        # Download the dataset
        print(f"\nDownloading {dataset_name}...")
        print("-" * 60)
        if dataset_name == 'sfm120k':
            success = download_sfm120k()
        else:
            success = download_dataset(dataset_name)
        if success:
            print(f"\n✓ Successfully downloaded {dataset_name}!")
        else:
            print(f"\n✗ Failed to download {dataset_name}")
            print("  You may need to download it manually")
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    # Final verification
    all_ok = True
    for dataset_name in datasets:
        if verify_dataset(dataset_name):
            print(f"✓ {dataset_name} is ready!")
        else:
            print(f"✗ {dataset_name} has issues")
            all_ok = False
    
    if all_ok:
        print("\n🎉 All datasets are ready for training!")
        print("\nYou can now run:")
        print("  python train.py --config config/config.yaml")
    else:
        print("\n⚠ Some datasets need attention. Run with --verify to check details:")
        print("  python download_dataset.py roxford5k --verify")


if __name__ == '__main__':
    main()

