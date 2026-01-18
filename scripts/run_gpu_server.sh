#!/bin/bash
# Run the GPU server on your laptop
# Usage: ./run_gpu_server.sh [--port PORT] [--auth-token TOKEN]

set -e

# Default values
PORT=5555
AUTH_TOKEN=""
DEVICE="auto"
CHECKPOINT_DIR="checkpoints"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            PORT="$2"
            shift 2
            ;;
        --auth-token)
            AUTH_TOKEN="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --checkpoint-dir)
            CHECKPOINT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Build command
CMD="python -m rl_hanabi.training.distributed.gpu_server"
CMD="$CMD --host 0.0.0.0"
CMD="$CMD --port $PORT"
CMD="$CMD --device $DEVICE"
CMD="$CMD --checkpoint-dir $CHECKPOINT_DIR"

if [ -n "$AUTH_TOKEN" ]; then
    CMD="$CMD --auth-token $AUTH_TOKEN"
fi

echo "Starting GPU Server..."
echo "Host: 0.0.0.0:$PORT"
echo "Device: $DEVICE"
echo ""

# Get IP addresses for reference
echo "Your IP addresses (share one with the server):"
ip addr show 2>/dev/null | grep 'inet ' | grep -v '127.0.0.1' | awk '{print "  " $2}' || hostname -I

echo ""
echo "Press Ctrl+C to stop"
echo "========================================"
echo ""

exec $CMD
