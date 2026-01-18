#!/bin/bash
# Setup script for the laptop (with GPU) - runs the GPU server
# Run this on your laptop

set -e

echo "==================================="
echo "Hanabi Distributed Training Setup"
echo "GPU Server (Laptop) Setup"
echo "==================================="

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "Error: Please run this script from the hanabi project root directory"
    exit 1
fi

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "Python version: $python_version"

# Check for CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "Warning: nvidia-smi not found. Make sure CUDA is properly installed."
fi

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

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio  # Will auto-detect CUDA

# Install other dependencies
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

# Create systemd service file for GPU server
echo "Creating systemd service file..."
mkdir -p systemd

cat > systemd/hanabi-gpu-server.service << EOF
[Unit]
Description=Hanabi GPU Training Server
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
Environment="PATH=$(pwd)/.venv/bin:\$PATH"
ExecStart=$(pwd)/.venv/bin/python -m rl_hanabi.training.distributed.gpu_server \\
    --host 0.0.0.0 \\
    --port \${GPU_PORT:-5555} \\
    --auth-token \${AUTH_TOKEN} \\
    --device auto \\
    --checkpoint-dir checkpoints
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Create environment file
cat > /tmp/hanabi-gpu-server.env << EOF
# Edit these values and copy to /etc/default/hanabi-gpu-server
GPU_PORT=5555
AUTH_TOKEN=your-secret-token-here
EOF

echo ""
echo "==================================="
echo "Setup complete!"
echo "==================================="
echo ""
echo "Quick start - Run the GPU server:"
echo "   source .venv/bin/activate"
echo "   python -m rl_hanabi.training.distributed.gpu_server --host 0.0.0.0 --port 5555 --auth-token <token>"
echo ""
echo "Or use the systemd service:"
echo "   1. Edit /tmp/hanabi-gpu-server.env and copy to /etc/default/hanabi-gpu-server"
echo "   2. sudo cp systemd/hanabi-gpu-server.service /etc/systemd/system/"
echo "   3. sudo systemctl daemon-reload"
echo "   4. sudo systemctl enable hanabi-gpu-server"
echo "   5. sudo systemctl start hanabi-gpu-server"
echo ""
echo "To find your IP address for the server config:"
echo "   ip addr show | grep 'inet ' | grep -v '127.0.0.1'"
echo ""
echo "Make sure to:"
echo "   - Open port 5555 in your firewall: sudo ufw allow 5555/tcp"
echo "   - Use the same auth-token on both server and laptop"
