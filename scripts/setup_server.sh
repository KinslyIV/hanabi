#!/bin/bash
# Setup script for the server (no GPU) - runs the coordinator
# Run this on your server machine

set -e

echo "==================================="
echo "Hanabi Distributed Training Setup"
echo "Server (Coordinator) Setup"
echo "==================================="

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "Error: Please run this script from the hanabi project root directory"
    exit 1
fi

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu  # CPU-only PyTorch
pip install numpy wandb tomli

# Install the hanabi-learning-environment
echo "Installing hanabi-learning-environment..."
cd hanabi-learning-environment
pip install -e .
cd ..

# Install the main package
echo "Installing rl_hanabi package..."
pip install -e .

# Create directories
echo "Creating directories..."
mkdir -p checkpoints
mkdir -p logs

# Copy example config if no config exists
if [ ! -f "distributed_config.toml" ]; then
    echo "Creating config file from example..."
    cp rl_hanabi/training/distributed_config.example.toml distributed_config.toml
    echo ""
    echo "IMPORTANT: Edit distributed_config.toml and set:"
    echo "  - gpu_host: Your laptop's IP address"
    echo "  - auth_token: A secure shared token"
fi

# Create systemd service file
echo "Creating systemd service file..."
cat > systemd/hanabi-coordinator.service << EOF
[Unit]
Description=Hanabi Distributed Training Coordinator
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
Environment="PATH=$(pwd)/.venv/bin:\$PATH"
ExecStart=$(pwd)/.venv/bin/python -m rl_hanabi.training.distributed.coordinator \\
    --gpu-host \${GPU_HOST} \\
    --gpu-port \${GPU_PORT:-5555} \\
    --auth-token \${AUTH_TOKEN} \\
    --checkpoint-dir checkpoints \\
    --use-wandb
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

echo ""
echo "==================================="
echo "Setup complete!"
echo "==================================="
echo ""
echo "Next steps:"
echo "1. Edit distributed_config.toml with your laptop's IP and auth token"
echo "2. On your laptop, run: ./scripts/setup_gpu_server.sh"
echo "3. Start the GPU server on your laptop"
echo "4. Run the coordinator:"
echo "   source .venv/bin/activate"
echo "   python -m rl_hanabi.training.distributed.coordinator --gpu-host <laptop-ip> --auth-token <token>"
echo ""
echo "Or use the systemd service (after editing /etc/default/hanabi-coordinator):"
echo "   sudo cp systemd/hanabi-coordinator.service /etc/systemd/system/"
echo "   sudo systemctl daemon-reload"
echo "   sudo systemctl enable hanabi-coordinator"
echo "   sudo systemctl start hanabi-coordinator"
