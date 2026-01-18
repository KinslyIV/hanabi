# Distributed Training Guide

This guide explains how to set up distributed training where your **server** (no GPU) handles game simulation and data collection, while your **laptop** (with GPU) handles the neural network training.

## Architecture Overview

```
┌──────────────────────────────────────┐       ┌──────────────────────────────────┐
│           SERVER (No GPU)            │       │         LAPTOP (With GPU)        │
│                                      │       │                                  │
│  ┌─────────────────────────────┐     │       │    ┌────────────────────────┐    │
│  │       Coordinator           │     │       │    │      GPU Server        │    │
│  │                             │     │       │    │                        │    │
│  │  - Game simulation workers  │     │ TCP   │    │  - Model on GPU        │    │
│  │  - Replay buffer            │────────────────▶│  - Forward/backward    │    │
│  │  - Data collection          │     │       │    │  - Optimizer step      │    │
│  │  - WandB logging            │◀────────────────│  - Checkpoints         │    │
│  │  - Training state           │     │       │    │                        │    │
│  └─────────────────────────────┘     │       │    └────────────────────────┘    │
│                                      │       │                                  │
└──────────────────────────────────────┘       └──────────────────────────────────┘
```

## Features

- **Resilient Connection**: Automatically reconnects if laptop becomes unreachable
- **Pause/Resume**: Training pauses when GPU server disconnects, resumes when it reconnects
- **State Persistence**: Training state is saved to disk, allowing restarts without losing progress
- **WandB Integration**: Optional logging to Weights & Biases
- **Systemd Services**: Run as system services with automatic restart

## Quick Start

### 1. Setup on Laptop (GPU machine)

```bash
# Clone/copy the project
cd hanabi

# Run setup script
./scripts/setup_laptop.sh

# Or manually:
python -m venv .venv
source .venv/bin/activate
pip install torch numpy  # With CUDA support
pip install -e .

# Find your IP address
ip addr show | grep 'inet ' | grep -v '127.0.0.1'
```

### 2. Setup on Server (no GPU)

```bash
# Clone/copy the project
cd hanabi

# Run setup script
./scripts/setup_server.sh

# Or manually:
python -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy wandb
pip install -e .

# Create config file
cp rl_hanabi/training/distributed_config.example.toml distributed_config.toml
# Edit distributed_config.toml with your laptop's IP
```

### 3. Start the GPU Server (on laptop)

```bash
# Using the script
./scripts/run_gpu_server.sh --port 5555 --auth-token mysecrettoken

# Or directly
python -m rl_hanabi.training.distributed.gpu_server \
    --host 0.0.0.0 \
    --port 5555 \
    --auth-token mysecrettoken \
    --device auto
```

### 4. Start the Coordinator (on server)

```bash
# Using the script
./scripts/run_coordinator.sh \
    --gpu-host 192.168.1.100 \
    --auth-token mysecrettoken \
    --use-wandb

# Or with config file
python scripts/run_with_config.py distributed_config.toml --server

# Or directly
python -m rl_hanabi.training.distributed.coordinator \
    --gpu-host 192.168.1.100 \
    --gpu-port 5555 \
    --auth-token mysecrettoken \
    --num-iterations 1000 \
    --games-per-iteration 100 \
    --train-steps-per-iteration 500 \
    --use-wandb
```

## Monitoring

### Check Status

```bash
# Monitor GPU server from anywhere
python scripts/monitor.py --gpu-host 192.168.1.100 --auth-token mysecrettoken

# Continuous monitoring
python scripts/monitor.py --gpu-host 192.168.1.100 --watch

# Check local state only
python scripts/monitor.py --local
```

### Logs

```bash
# If using systemd
journalctl -u hanabi-gpu-server@$USER -f
journalctl -u hanabi-coordinator@$USER -f

# If running manually, logs go to stdout
```

## Running as Systemd Services

### On Laptop (GPU Server)

```bash
# Copy service file
sudo cp systemd/hanabi-gpu-server@.service /etc/systemd/system/

# Create environment file
sudo cp systemd/hanabi-gpu-server.env.example /etc/default/hanabi-gpu-server
sudo nano /etc/default/hanabi-gpu-server  # Edit AUTH_TOKEN

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable hanabi-gpu-server@$USER
sudo systemctl start hanabi-gpu-server@$USER

# Check status
sudo systemctl status hanabi-gpu-server@$USER
```

### On Server (Coordinator)

```bash
# Copy service file
sudo cp systemd/hanabi-coordinator@.service /etc/systemd/system/

# Create environment file
sudo cp systemd/hanabi-coordinator.env.example /etc/default/hanabi-coordinator
sudo nano /etc/default/hanabi-coordinator  # Edit GPU_HOST, AUTH_TOKEN

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable hanabi-coordinator@$USER
sudo systemctl start hanabi-coordinator@$USER

# Check status
sudo systemctl status hanabi-coordinator@$USER
```

## Firewall Configuration

On the laptop, open the GPU server port:

```bash
# UFW (Ubuntu)
sudo ufw allow 5555/tcp

# Firewalld (Fedora, CentOS)
sudo firewall-cmd --add-port=5555/tcp --permanent
sudo firewall-cmd --reload
```

## Configuration Reference

### Network Settings
| Parameter | Default | Description |
|-----------|---------|-------------|
| `gpu_host` | localhost | IP address of the laptop |
| `gpu_port` | 5555 | Port for GPU server |
| `auth_token` | None | Shared authentication token |

### Model Settings (must match on both)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_players` | 5 | Maximum number of players |
| `max_colors` | 5 | Maximum number of colors |
| `max_ranks` | 5 | Maximum number of ranks |
| `max_hand_size` | 5 | Maximum hand size |
| `d_model` | 128 | Transformer dimension |
| `num_heads` | 4 | Attention heads |
| `num_layers` | 4 | Transformer layers |

### Training Settings
| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 256 | Training batch size |
| `learning_rate` | 1e-4 | Initial learning rate |
| `games_per_iteration` | 100 | Games to collect per iteration |
| `train_steps_per_iteration` | 500 | Training steps per iteration |

## Troubleshooting

### Connection Issues

1. **Check IP addresses**: Ensure the server can reach the laptop
   ```bash
   ping <laptop-ip>
   ```

2. **Check firewall**: Ensure port 5555 is open on the laptop

3. **Check auth token**: Must match on both sides

### Training Pauses

The training will automatically pause if:
- Laptop goes to sleep
- Network connection drops
- GPU server crashes

It will automatically resume when the connection is restored.

### State Recovery

Training state is saved in `checkpoints/training_state.json`. The coordinator can be restarted at any time and will resume from where it left off.

### GPU Memory Issues

If you get CUDA out of memory errors, reduce batch size:
```bash
--batch-size 128
```

## Performance Tips

1. **Number of workers**: Set `--num-workers` based on CPU cores (default: auto)
2. **Batch size**: Larger batches = faster training but more memory
3. **Games per iteration**: More games = better data variety
4. **Network latency**: Keep laptop and server on same local network
