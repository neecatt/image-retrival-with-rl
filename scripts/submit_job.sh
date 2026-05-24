#!/bin/bash
#SBATCH --job-name=img-retrieval-rl
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Create log directory if it doesn't exist
mkdir -p logs

echo "============================================"
echo "Job ID:       $SLURM_JOB_ID"
echo "Node:         $SLURMD_NODENAME"
echo "GPUs:         $CUDA_VISIBLE_DEVICES"
echo "Start time:   $(date)"
echo "============================================"

# Load environment
module load python/3.12-conda
conda activate rl-env

cd "$SLURM_SUBMIT_DIR"

# Run training
python train.py --config config/config.yaml "$@"

echo "============================================"∏a
echo "End time: $(date)"
echo "============================================"
 