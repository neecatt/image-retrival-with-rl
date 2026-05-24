#!/bin/bash
# Start training with TensorBoard visualization
# 
# Usage:
#   ./train_with_tensorboard.sh [training arguments]
# 
# Examples:
#   ./train_with_tensorboard.sh --config config/config.yaml
#   ./train_with_tensorboard.sh --config config/config.yaml --stage1_only
#   ./train_with_tensorboard.sh --config config/config.yaml --stage2_only --resume checkpoints/stage1_best.pth

# Default values
LOG_DIR="./logs"
TENSORBOARD_PORT=6006

# Parse log directory from config if provided, or use default
CONFIG_FILE="config/config.yaml"
ARGS=("$@")
for i in "${!ARGS[@]}"; do
    if [[ "${ARGS[$i]}" == --config ]]; then
        # Next argument is the config file
        if [ $((i+1)) -lt ${#ARGS[@]} ]; then
            CONFIG_FILE="${ARGS[$((i+1))]}"
        fi
        break
    elif [[ "${ARGS[$i]}" == --config=* ]]; then
        # Config file is in the same argument
        CONFIG_FILE=$(echo "${ARGS[$i]}" | cut -d'=' -f2-)
        break
    fi
done

# Try to extract log_dir from config file if it exists
if [ -f "$CONFIG_FILE" ]; then
    EXTRACTED_LOG_DIR=$(grep -A 5 "^system:" "$CONFIG_FILE" | grep "log_dir:" | awk '{print $2}' | tr -d '"' | tr -d "'")
    if [ ! -z "$EXTRACTED_LOG_DIR" ]; then
        LOG_DIR="$EXTRACTED_LOG_DIR"
    fi
fi

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "Starting Training with TensorBoard"
echo "=========================================="
echo "Log directory: $LOG_DIR"
echo "TensorBoard port: $TENSORBOARD_PORT"
echo "Training command: python train.py ${ARGS[@]}"
echo ""
echo "TensorBoard will be available at: http://localhost:$TENSORBOARD_PORT"
echo "Press Ctrl+C to stop both training and TensorBoard"
echo "=========================================="
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Stopping TensorBoard..."
    if [ ! -z "$TENSORBOARD_PID" ]; then
        kill $TENSORBOARD_PID 2>/dev/null
        wait $TENSORBOARD_PID 2>/dev/null
    fi
    echo "Cleanup complete."
    exit 0
}

# Set trap to cleanup on script exit or interrupt
trap cleanup SIGINT SIGTERM EXIT

# Start TensorBoard in the background
echo "Starting TensorBoard..."
tensorboard --logdir "$LOG_DIR" --port "$TENSORBOARD_PORT" --reload_interval 1 > /dev/null 2>&1 &
TENSORBOARD_PID=$!

# Wait a moment for TensorBoard to start
sleep 2

# Check if TensorBoard started successfully
if ! kill -0 $TENSORBOARD_PID 2>/dev/null; then
    echo "Warning: TensorBoard may have failed to start. Continuing with training..."
else
    echo "TensorBoard started successfully (PID: $TENSORBOARD_PID)"
    echo "Open http://localhost:$TENSORBOARD_PORT in your browser to view metrics"
    echo ""
fi

# Start training
echo "Starting training..."
echo "------------------------------------------"
python train.py "${ARGS[@]}"
TRAINING_EXIT_CODE=$?

# Training finished
echo ""
echo "------------------------------------------"
if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully!"
else
    echo "Training exited with code: $TRAINING_EXIT_CODE"
fi

# Cleanup will be handled by trap
exit $TRAINING_EXIT_CODE

