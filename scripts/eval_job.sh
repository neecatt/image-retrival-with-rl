#!/bin/bash
#SBATCH --job-name=img-retrieval-eval
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err

# Usage:
#   sbatch eval_job.sh checkpoints/stage2_epoch_11_best.pth
#   sbatch eval_job.sh checkpoints/stage1_epoch_10_best.pth

CHECKPOINT=${1:-checkpoints/stage2_epoch_11_best.pth}

mkdir -p logs results

echo "============================================"
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURMD_NODENAME"
echo "Checkpoint: $CHECKPOINT"
echo "Start time: $(date)"
echo "============================================"

module load python/3.12-conda
conda activate rl-env

cd "$SLURM_SUBMIT_DIR"

echo ""
echo "--- Evaluating on roxford5k ---"
python evaluate.py \
    --checkpoint "$CHECKPOINT" \
    --config config/config.yaml \
    --dataset roxford5k \
    --save_embeddings \
    --output_dir results/roxford5k

echo ""
echo "--- Evaluating on rparis6k ---"
python evaluate.py \
    --checkpoint "$CHECKPOINT" \
    --config config/config.yaml \
    --dataset rparis6k \
    --save_embeddings \
    --output_dir results/rparis6k

echo "============================================"
echo "End time: $(date)"
echo "============================================"
