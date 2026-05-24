#!/bin/bash
#SBATCH --job-name=img-retrieval-stage2
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_stage2_%j.out
#SBATCH --error=logs/slurm_stage2_%j.err

set -euo pipefail

PROJECT_DIR="/home/hpc/iwi5/iwi5354h/Image Retrieval with RL"
CHECKPOINT="checkpoints/stage1_20260405_073610/stage1_epoch_35.pth"
CONFIG="config/config.yaml"

mkdir -p "${PROJECT_DIR}/logs"

echo "============================================"
echo "Job ID:       ${SLURM_JOB_ID:-N/A}"
echo "Node:         ${SLURMD_NODENAME:-N/A}"
echo "GPUs:         ${CUDA_VISIBLE_DEVICES:-N/A}"
echo "Start time:   $(date)"
echo "Checkpoint:   ${CHECKPOINT}"
echo "Config:       ${CONFIG}"
echo "============================================"

module load python/3.12-conda
conda activate rl-env

cd "${PROJECT_DIR}"

python train.py \
  --config "${CONFIG}" \
  --stage2_only \
  --resume "${CHECKPOINT}" \
  --seed 123 \
  "$@"

echo "============================================"
echo "End time:     $(date)"
echo "============================================"
