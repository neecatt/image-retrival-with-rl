#!/bin/bash -l
set -euo pipefail
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:a100:1
#SBATCH --time=08:00:00
#SBATCH --job-name=ir_rl_train
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

# Load environment
module load python/3.12-conda
source activate rl-env

# Set working directory
cd "$SLURM_SUBMIT_DIR"

# Ensure log directory exists
mkdir -p logs

# Recommended environment tuning for PyTorch on shared nodes
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NCCL_DEBUG=INFO

# Run training
srun python3 train.py --config config/config.yaml