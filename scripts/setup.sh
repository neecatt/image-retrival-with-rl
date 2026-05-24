#!/bin/bash

# Setup script for Image Retrieval with RL project

echo "============================================"
echo "Image Retrieval with RL - Setup Script"
echo "============================================"
echo ""

# Create necessary directories
echo "Creating project directories..."
mkdir -p data/raw
mkdir -p data/processed
mkdir -p checkpoints
mkdir -p logs
mkdir -p results

echo "✓ Directories created"
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

required_version="3.8"
if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "⚠ Warning: Python 3.8+ recommended"
fi
echo ""

# Create virtual environment
echo "Creating virtual environment..."
if [ -d "venv" ]; then
    echo "⚠ Virtual environment already exists. Skipping..."
else
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi
echo ""

# Activate virtual environment
echo "To activate the virtual environment, run:"
echo "  source venv/bin/activate  # Linux/Mac"
echo "  venv\\Scripts\\activate   # Windows"
echo ""

# Install dependencies
read -p "Install dependencies now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Installing dependencies..."
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    echo "✓ Dependencies installed"
else
    echo "Skipping dependency installation"
    echo "To install later, run:"
    echo "  source venv/bin/activate"
    echo "  pip install -r requirements.txt"
fi
echo ""

# Download dataset instructions
echo "============================================"
echo "Dataset Setup Instructions"
echo "============================================"
echo ""
echo "To use ROxford5k / RParis6k datasets:"
echo "1. Visit: https://github.com/filipradenovic/revisitop"
echo "2. Download the datasets"
echo "3. Extract to: data/raw/roxford5k/ and data/raw/rparis6k/"
echo ""
echo "Expected structure:"
echo "  data/raw/"
echo "    ├── roxford5k/"
echo "    │   ├── jpg/"
echo "    │   └── roxford5k.pkl"
echo "    └── rparis6k/"
echo "        ├── jpg/"
echo "        └── rparis6k.pkl"
echo ""

echo "============================================"
echo "Setup Complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Download datasets (see instructions above)"
echo "3. Run training: python train.py --config config/config.yaml"
echo ""
echo "For detailed instructions, see README.md"
echo ""

