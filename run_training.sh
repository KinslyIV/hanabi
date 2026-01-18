#!/bin/bash
# Run Hanabi self-play training
# Usage: ./run_training.sh [config_preset]
# Example: ./run_training.sh quick_test

set -e

# Default values
PRESET=${1:-"default"}
WANDB_PROJECT=${WANDB_PROJECT:-"hanabi-selfplay"}

echo "=========================================="
echo "Hanabi Self-Play Training"
echo "=========================================="
echo "Preset: $PRESET"
echo "WandB Project: $WANDB_PROJECT"
echo ""

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "Activated virtual environment"
fi

# Check if wandb is installed
if python -c "import wandb" 2>/dev/null; then
    USE_WANDB="--use-wandb"
    echo "WandB: enabled"
else
    USE_WANDB=""
    echo "WandB: disabled (not installed)"
fi

# Run training based on preset
case $PRESET in
    "quick_test")
        echo "Running quick test configuration..."
        python -m rl_hanabi.training.train \
            --num-iterations 5 \
            --games-per-iteration 20 \
            --train-steps-per-iteration 100 \
            --batch-size 64 \
            --buffer-size 10000 \
            --log-interval 10 \
            --save-interval 5 \
            --wandb-project "$WANDB_PROJECT" \
            --tags "quick-test" \
            $USE_WANDB
        ;;
    
    "two_player")
        echo "Running 2-player configuration..."
        python -m rl_hanabi.training.train \
            --min-players 2 \
            --max-players 2 \
            --games-per-iteration 200 \
            --train-steps-per-iteration 1000 \
            --wandb-project "$WANDB_PROJECT" \
            --tags "two-player" \
            $USE_WANDB
        ;;
    
    "full_hanabi")
        echo "Running full Hanabi configuration..."
        python -m rl_hanabi.training.train \
            --min-colors 5 \
            --max-colors 5 \
            --min-ranks 5 \
            --max-ranks 5 \
            --games-per-iteration 150 \
            --train-steps-per-iteration 750 \
            --wandb-project "$WANDB_PROJECT" \
            --tags "full-hanabi" \
            $USE_WANDB
        ;;
    
    "large_scale")
        echo "Running large scale configuration..."
        python -m rl_hanabi.training.train \
            --num-workers 8 \
            --games-per-iteration 500 \
            --train-steps-per-iteration 2000 \
            --num-iterations 500 \
            --batch-size 512 \
            --buffer-size 500000 \
            --d-model 256 \
            --num-layers 6 \
            --num-heads 8 \
            --learning-rate 3e-4 \
            --wandb-project "$WANDB_PROJECT" \
            --tags "large-scale" \
            $USE_WANDB
        ;;
    
    "debug")
        echo "Running debug configuration..."
        python -m rl_hanabi.training.train \
            --num-workers 1 \
            --num-iterations 2 \
            --games-per-iteration 5 \
            --train-steps-per-iteration 10 \
            --batch-size 8 \
            --buffer-size 1000 \
            --log-interval 1 \
            --save-interval 1
        ;;
    
    "default" | *)
        echo "Running default configuration..."
        python -m rl_hanabi.training.train \
            --wandb-project "$WANDB_PROJECT" \
            $USE_WANDB
        ;;
esac

echo ""
echo "Training complete!"
