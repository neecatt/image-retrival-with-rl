#!/bin/bash
# Evaluate Stage 2 checkpoints (epochs 1,2,3) on roxford5k and rparis6k,
# then pick the best by average mAP_easy.

cd "$(dirname "$0")/.."
CKPT_DIR="${CHECKPOINT_DIR:-/home/woody/iwi5/iwi5354h/Image Retrieval with RL/checkpoints}"
CONFIG="${CONFIG:-config/config.yaml}"

best_epoch=""
best_score="0"

for epoch in 1 2 3; do
  ckpt="${CKPT_DIR}/stage2_epoch_${epoch}.pth"
  if [ ! -f "$ckpt" ]; then
    echo "Skipping epoch $epoch: $ckpt not found"
    continue
  fi

  echo "Evaluating epoch $epoch..."
  roxford_out=$(python evaluate.py --config "$CONFIG" --checkpoint "$ckpt" --dataset roxford5k 2>/dev/null)
  rparis_out=$(python evaluate.py --config "$CONFIG" --checkpoint "$ckpt" --dataset rparis6k 2>/dev/null)

  roxford=$(echo "$roxford_out" | grep "mAP_easy" | head -1 | awk '{print $NF}')
  rparis=$(echo "$rparis_out" | grep "mAP_easy" | head -1 | awk '{print $NF}')

  if [ -z "$roxford" ] || [ -z "$rparis" ]; then
    echo "  Failed to parse metrics"
    continue
  fi

  score=$(python -c "print(($roxford + $rparis) / 2)")
  echo "  roxford5k mAP_easy: $roxford | rparis6k mAP_easy: $rparis | avg: $score"

  is_best=$(python -c "print(1 if $score > $best_score else 0)")
  if [ "$is_best" = "1" ]; then
    best_score=$score
    best_epoch=$epoch
  fi
done

echo ""
echo "=========================================="
echo "Best epoch: $best_epoch (avg mAP_easy: $best_score)"
echo "Checkpoint: ${CKPT_DIR}/stage2_epoch_${best_epoch}.pth"
echo "=========================================="
