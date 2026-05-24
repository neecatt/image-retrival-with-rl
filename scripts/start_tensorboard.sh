#!/bin/bash
# Start TensorBoard for real-time training visualization

LOG_DIR="${1:-./logs}"
PORT="${2:-6006}"

echo "Starting TensorBoard..."
echo "Log directory: $LOG_DIR"
echo "Port: $PORT"
echo ""
echo "Open your browser to: http://localhost:$PORT"
echo "Press Ctrl+C to stop TensorBoard"
echo ""

tensorboard --logdir "$LOG_DIR" --port "$PORT" --reload_interval 1

