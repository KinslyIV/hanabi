#!/bin/bash
# Run the coordinator on your server
# Usage: ./run_coordinator.sh --gpu-host <laptop-ip> [--auth-token TOKEN] [other options]

set -e

# Check for config file
CONFIG_FILE="distributed_config.toml"
if [ -f "$CONFIG_FILE" ]; then
    echo "Note: Found $CONFIG_FILE - you can also run with: python scripts/run_with_config.py"
fi

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Default values
GPU_HOST=""
GPU_PORT=5555
AUTH_TOKEN=""
CHECKPOINT_DIR="checkpoints"
USE_WANDB=""
NUM_ITERATIONS=1000
GAMES_PER_ITERATION=100
TRAIN_STEPS=500
BATCH_SIZE=256

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu-host)
            GPU_HOST="$2"
            shift 2
            ;;
        --gpu-port)
            GPU_PORT="$2"
            shift 2
            ;;
        --auth-token)
            AUTH_TOKEN="$2"
            shift 2
            ;;
        --checkpoint-dir)
            CHECKPOINT_DIR="$2"
            shift 2
            ;;
        --use-wandb)
            USE_WANDB="--use-wandb"
            shift
            ;;
        --iterations)
            NUM_ITERATIONS="$2"
            shift 2
            ;;
        --games)
            GAMES_PER_ITERATION="$2"
            shift 2
            ;;
        --train-steps)
            TRAIN_STEPS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo ""
            echo "Usage: $0 --gpu-host <laptop-ip> [options]"
            echo ""
            echo "Required:"
            echo "  --gpu-host      Laptop IP address"
            echo ""
            echo "Optional:"
            echo "  --gpu-port      GPU server port (default: 5555)"
            echo "  --auth-token    Authentication token"
            echo "  --checkpoint-dir Checkpoint directory (default: checkpoints)"
            echo "  --use-wandb     Enable WandB logging"
            echo "  --iterations    Number of iterations (default: 1000)"
            echo "  --games         Games per iteration (default: 100)"
            echo "  --train-steps   Training steps per iteration (default: 500)"
            echo "  --batch-size    Batch size (default: 256)"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$GPU_HOST" ]; then
    echo "Error: --gpu-host is required"
    echo "Usage: $0 --gpu-host <laptop-ip> [options]"
    exit 1
fi

# Build command
CMD="python -m rl_hanabi.training.distributed.coordinator"
CMD="$CMD --gpu-host $GPU_HOST"
CMD="$CMD --gpu-port $GPU_PORT"
CMD="$CMD --checkpoint-dir $CHECKPOINT_DIR"
CMD="$CMD --num-iterations $NUM_ITERATIONS"
CMD="$CMD --games-per-iteration $GAMES_PER_ITERATION"
CMD="$CMD --train-steps-per-iteration $TRAIN_STEPS"
CMD="$CMD --batch-size $BATCH_SIZE"

if [ -n "$AUTH_TOKEN" ]; then
    CMD="$CMD --auth-token $AUTH_TOKEN"
fi

if [ -n "$USE_WANDB" ]; then
    CMD="$CMD $USE_WANDB"
fi

echo "Starting Coordinator..."
echo "GPU Server: $GPU_HOST:$GPU_PORT"
echo "Checkpoint Dir: $CHECKPOINT_DIR"
echo ""
echo "Press Ctrl+C to stop (state will be saved)"
echo "========================================"
echo ""

exec $CMD
