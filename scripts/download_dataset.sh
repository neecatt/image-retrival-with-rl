#!/bin/bash
# Download script for Revisited Oxford/Paris datasets
# 
# Usage:
#   ./download_dataset.sh [dataset_name]
# 
# Examples:
#   ./download_dataset.sh roxford5k
#   ./download_dataset.sh rparis6k
#   ./download_dataset.sh all

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
DATA_DIR="./data/raw"
REVISITOP_REPO="https://github.com/filipradenovic/revisitop"

# Dataset URLs (these may need to be updated - check revisitop repository)
# The actual download URLs are typically provided in the revisitop repository
ROXFORD5K_URL="http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/oxbuild_images.tgz"
RPARIS6K_URL="http://www.robots.ox.ac.uk/~vgg/data/parisbuildings/paris_1.tgz"
RPARIS6K_URL2="http://www.robots.ox.ac.uk/~vgg/data/parisbuildings/paris_2.tgz"

echo "=========================================="
echo "Dataset Download Script"
echo "=========================================="
echo ""

# Create data directory
mkdir -p "$DATA_DIR"

# Function to download and extract dataset
download_roxford5k() {
    echo -e "${GREEN}Downloading ROxford5k dataset...${NC}"
    
    DATASET_DIR="$DATA_DIR/roxford5k"
    mkdir -p "$DATASET_DIR"
    
    echo ""
    echo "ROxford5k requires manual download steps:"
    echo "1. Visit: https://github.com/filipradenovic/revisitop"
    echo "2. Download the dataset files"
    echo "3. Extract to: $DATASET_DIR"
    echo ""
    echo "Expected structure:"
    echo "  $DATASET_DIR/"
    echo "    ├── jpg/          (images directory)"
    echo "    └── roxford5k.pkl (dataset configuration)"
    echo ""
    echo "Alternatively, you can use the official download script from revisitop:"
    echo "  git clone $REVISITOP_REPO"
    echo "  cd revisitop"
    echo "  # Follow their download instructions"
    echo ""
    
    # Check if dataset already exists
    if [ -d "$DATASET_DIR/jpg" ] && [ -f "$DATASET_DIR/roxford5k.pkl" ]; then
        echo -e "${GREEN}✓ Dataset appears to be already downloaded${NC}"
        return 0
    fi
    
    echo -e "${YELLOW}⚠ Dataset not found. Please download manually.${NC}"
    return 1
}

download_rparis6k() {
    echo -e "${GREEN}Downloading RParis6k dataset...${NC}"
    
    DATASET_DIR="$DATA_DIR/rparis6k"
    mkdir -p "$DATASET_DIR"
    
    echo ""
    echo "RParis6k requires manual download steps:"
    echo "1. Visit: https://github.com/filipradenovic/revisitop"
    echo "2. Download the dataset files"
    echo "3. Extract to: $DATASET_DIR"
    echo ""
    echo "Expected structure:"
    echo "  $DATASET_DIR/"
    echo "    ├── jpg/          (images directory)"
    echo "    └── rparis6k.pkl  (dataset configuration)"
    echo ""
    echo "Alternatively, you can use the official download script from revisitop:"
    echo "  git clone $REVISITOP_REPO"
    echo "  cd revisitop"
    echo "  # Follow their download instructions"
    echo ""
    
    # Check if dataset already exists
    if [ -d "$DATASET_DIR/jpg" ] && [ -f "$DATASET_DIR/rparis6k.pkl" ]; then
        echo -e "${GREEN}✓ Dataset appears to be already downloaded${NC}"
        return 0
    fi
    
    echo -e "${YELLOW}⚠ Dataset not found. Please download manually.${NC}"
    return 1
}

# Function to download using revisitop repository
download_with_revisitop() {
    echo "Attempting to download using revisitop repository..."
    echo ""
    
    # Check if revisitop is already cloned
    if [ ! -d "revisitop" ]; then
        echo "Cloning revisitop repository..."
        git clone "$REVISITOP_REPO" revisitop || {
            echo -e "${RED}✗ Failed to clone revisitop repository${NC}"
            echo "Please clone manually: git clone $REVISITOP_REPO"
            return 1
        }
    fi
    
    cd revisitop
    
    echo ""
    echo "Please follow the instructions in the revisitop repository to download datasets."
    echo "Typically, you need to:"
    echo "1. Run their download script (if available)"
    echo "2. Or download the .mat files and convert them"
    echo ""
    echo "After downloading, you need to:"
    echo "1. Convert .mat files to .pkl format"
    echo "2. Organize images in jpg/ directory"
    echo "3. Place everything in $DATA_DIR/[dataset_name]/"
    
    cd ..
}

# Function to create a helper script for dataset preparation
create_prep_script() {
    cat > prepare_dataset.py << 'EOF'
"""
Helper script to prepare Revisited Oxford/Paris datasets.

This script helps convert .mat files to .pkl format and organize the dataset.
"""
import os
import pickle
import numpy as np
from scipy.io import loadmat
from pathlib import Path

def convert_mat_to_pkl(mat_file, pkl_file, dataset_name):
    """Convert .mat file to .pkl format."""
    print(f"Loading {mat_file}...")
    mat_data = loadmat(mat_file, struct_as_record=False, squeeze_me=True)
    
    # Extract relevant information
    # The structure may vary, adjust based on actual .mat file structure
    cfg = {
        'db_images': mat_data.get('imlist', []),
        'query_images': mat_data.get('qimlist', []),
        'gnd': mat_data.get('gnd', {})
    }
    
    print(f"Saving to {pkl_file}...")
    with open(pkl_file, 'wb') as f:
        pickle.dump(cfg, f)
    
    print(f"✓ Converted {dataset_name} dataset")

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        print("Usage: python prepare_dataset.py <mat_file> <output_pkl> <dataset_name>")
        sys.exit(1)
    
    convert_mat_to_pkl(sys.argv[1], sys.argv[2], sys.argv[3])
EOF
    echo -e "${GREEN}✓ Created prepare_dataset.py helper script${NC}"
}

# Main download logic
DATASET_NAME="${1:-all}"

case "$DATASET_NAME" in
    roxford5k)
        download_roxford5k
        ;;
    rparis6k)
        download_rparis6k
        ;;
    all)
        echo "Downloading all datasets..."
        download_roxford5k
        echo ""
        download_rparis6k
        ;;
    *)
        echo -e "${RED}Unknown dataset: $DATASET_NAME${NC}"
        echo "Usage: $0 [roxford5k|rparis6k|all]"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Next Steps:"
echo "=========================================="
echo ""
echo "1. Download the datasets from:"
echo "   $REVISITOP_REPO"
echo ""
echo "2. For automated download, you can try:"
echo "   git clone $REVISITOP_REPO"
echo "   cd revisitop"
echo "   # Follow their download instructions"
echo ""
echo "3. Organize the dataset in this structure:"
echo "   $DATA_DIR/roxford5k/"
echo "     ├── jpg/          # All images"
echo "     └── roxford5k.pkl # Dataset config"
echo ""
echo "   $DATA_DIR/rparis6k/"
echo "     ├── jpg/          # All images"
echo "     └── rparis6k.pkl  # Dataset config"
echo ""
echo "4. If you have .mat files, use the helper script:"
create_prep_script
echo "   python prepare_dataset.py <input.mat> <output.pkl> <dataset_name>"
echo ""
echo "5. Verify the dataset:"
echo "   ls -la $DATA_DIR/*/jpg/ | head"
echo "   ls -la $DATA_DIR/*/*.pkl"
echo ""

