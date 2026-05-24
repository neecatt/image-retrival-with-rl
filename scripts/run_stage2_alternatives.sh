#!/bin/bash
# Run Stage 2 with alternative approaches (map+high consistency, combined reward).
# Start from Stage 1 best. Each approach saves to its own checkpoint subdir.

set -e
cd "$(dirname "$0")/.."
PROJECT="/home/hpc/iwi5/iwi5354h/Image Retrieval with RL"
STAGE1_CKPT="${PROJECT}/checkpoints/stage1_epoch_12_best.pth"

# Ensure Stage 1 checkpoint exists (use woody path if you train from there)
for d in "$PROJECT/checkpoints" "/home/woody/iwi5/iwi5354h/Image Retrieval with RL/checkpoints"; do
  if [ -f "$d/stage1_epoch_12_best.pth" ]; then
    STAGE1_CKPT="$d/stage1_epoch_12_best.pth"
    break
  fi
done
if [ ! -f "$STAGE1_CKPT" ]; then
  echo "Stage 1 checkpoint not found. Set STAGE1_CKPT or add stage1_epoch_12_best.pth to checkpoints/"
  exit 1
fi

echo "Using Stage 1 checkpoint: $STAGE1_CKPT"
echo ""

# Approach 1: mAP reward + consistency_weight 8.0
echo "=== Approach 1: mAP reward + high consistency (8.0) ==="
echo "Checkpoints will be in: checkpoints/stage2_map_high_consistency/"
echo "Run:"
echo "  srun --gres=gpu --time=04:00:00 bash -lc 'module load python/3.12-conda && conda activate rl-env && cd \"$PROJECT\" && python train.py --config config/config_stage2_map_high_consistency.yaml --stage2_only --resume \"$STAGE1_CKPT\" --stage2_epochs 3'"
echo ""

# Approach 2: combined reward (0.5*map + 0.5*recall) + consistency 8.0
echo "=== Approach 2: combined reward + high consistency (8.0) ==="
echo "Checkpoints will be in: checkpoints/stage2_combined/"
echo "Run:"
echo "  srun --gres=gpu --time=04:00:00 bash -lc 'module load python/3.12-conda && conda activate rl-env && cd \"$PROJECT\" && python train.py --config config/config_stage2_combined.yaml --stage2_only --resume \"$STAGE1_CKPT\" --stage2_epochs 3'"
echo ""

# After training, select best epoch per approach (use woody path if checkpoints are there):
echo "After each run, pick best epoch with:"
echo "  CHECKPOINT_DIR=/path/to/checkpoints/stage2_map_high_consistency  ./scripts/select_best_epoch.sh"
echo "  CHECKPOINT_DIR=/path/to/checkpoints/stage2_combined               ./scripts/select_best_epoch.sh"
