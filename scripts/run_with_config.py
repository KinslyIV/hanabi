#!/usr/bin/env python3
"""
Run distributed training with a TOML configuration file.
Usage: python scripts/run_with_config.py [config_file] [--server|--laptop]
"""

import argparse
import subprocess
import sys
from pathlib import Path

try:
    import tomli
except ImportError:
    print("Error: tomli not installed. Run: pip install tomli")
    sys.exit(1)


def load_config(config_path: Path) -> dict:
    """Load TOML configuration file."""
    with open(config_path, 'rb') as f:
        return tomli.load(f)


def run_gpu_server(config: dict) -> None:
    """Run the GPU server with config settings."""
    net = config.get('network', {})
    model = config.get('model', {})
    training = config.get('training', {})
    paths = config.get('paths', {})
    
    cmd = [
        sys.executable, '-m', 'rl_hanabi.training.distributed.gpu_server',
        '--host', '0.0.0.0',
        '--port', str(net.get('gpu_port', 5555)),
        '--device', 'auto',
        '--checkpoint-dir', paths.get('checkpoint_dir', 'checkpoints'),
        # Model config
        '--max-players', str(model.get('max_players', 5)),
        '--max-colors', str(model.get('max_colors', 5)),
        '--max-ranks', str(model.get('max_ranks', 5)),
        '--max-hand-size', str(model.get('max_hand_size', 5)),
        '--num-heads', str(model.get('num_heads', 4)),
        '--num-layers', str(model.get('num_layers', 4)),
        '--d-model', str(model.get('d_model', 128)),
        # Training config
        '--learning-rate', str(training.get('learning_rate', 1e-4)),
        '--weight-decay', str(training.get('weight_decay', 0.01)),
        '--max-grad-norm', str(training.get('max_grad_norm', 1.0)),
        '--color-loss-weight', str(training.get('color_loss_weight', 1.0)),
        '--rank-loss-weight', str(training.get('rank_loss_weight', 1.0)),
        '--action-loss-weight', str(training.get('action_loss_weight', 1.0)),
    ]
    
    if net.get('auth_token'):
        cmd.extend(['--auth-token', net['auth_token']])
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)


def run_coordinator(config: dict) -> None:
    """Run the coordinator with config settings."""
    net = config.get('network', {})
    model = config.get('model', {})
    training = config.get('training', {})
    simulation = config.get('simulation', {})
    game_ranges = config.get('game_ranges', {})
    iteration = config.get('iteration', {})
    logging_cfg = config.get('logging', {})
    paths = config.get('paths', {})
    general = config.get('general', {})
    
    cmd = [
        sys.executable, '-m', 'rl_hanabi.training.distributed.coordinator',
        '--gpu-host', net.get('gpu_host', 'localhost'),
        '--gpu-port', str(net.get('gpu_port', 5555)),
        '--seed', str(general.get('seed', 42)),
        '--checkpoint-dir', paths.get('checkpoint_dir', 'checkpoints'),
        # Model config
        '--max-players', str(model.get('max_players', 5)),
        '--max-colors', str(model.get('max_colors', 5)),
        '--max-ranks', str(model.get('max_ranks', 5)),
        '--max-hand-size', str(model.get('max_hand_size', 5)),
        '--num-heads', str(model.get('num_heads', 4)),
        '--num-layers', str(model.get('num_layers', 4)),
        '--d-model', str(model.get('d_model', 128)),
        # Training config
        '--learning-rate', str(training.get('learning_rate', 1e-4)),
        '--weight-decay', str(training.get('weight_decay', 0.01)),
        '--max-grad-norm', str(training.get('max_grad_norm', 1.0)),
        '--batch-size', str(training.get('batch_size', 256)),
        # Simulation config
        '--temperature', str(simulation.get('temperature', 1.0)),
        '--epsilon', str(simulation.get('epsilon', 0.1)),
        '--epsilon-decay', str(simulation.get('epsilon_decay', 0.01)),
        '--min-epsilon', str(simulation.get('min_epsilon', 0.01)),
        # Game ranges
        '--min-players', str(game_ranges.get('min_players', 2)),
        '--min-colors', str(game_ranges.get('min_colors', 3)),
        '--min-ranks', str(game_ranges.get('min_ranks', 3)),
        # Iteration config
        '--num-iterations', str(iteration.get('num_iterations', 1000)),
        '--games-per-iteration', str(iteration.get('games_per_iteration', 100)),
        '--train-steps-per-iteration', str(iteration.get('train_steps_per_iteration', 500)),
        '--buffer-size', str(iteration.get('buffer_size', 100000)),
        '--num-workers', str(iteration.get('num_workers', 0)),
        # Logging config
        '--log-interval', str(logging_cfg.get('log_interval', 100)),
        '--save-interval', str(logging_cfg.get('save_interval', 10)),
    ]
    
    if net.get('auth_token'):
        cmd.extend(['--auth-token', net['auth_token']])
    
    if paths.get('buffer_dir'):
        cmd.extend(['--buffer-dir', paths['buffer_dir']])
    
    if logging_cfg.get('use_wandb'):
        cmd.append('--use-wandb')
        cmd.extend(['--wandb-project', logging_cfg.get('wandb_project', 'hanabi-distributed')])
        if logging_cfg.get('run_name'):
            cmd.extend(['--run-name', logging_cfg['run_name']])
        if logging_cfg.get('tags'):
            cmd.extend(['--tags', logging_cfg['tags']])
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser(description='Run distributed training with config file')
    parser.add_argument('config', nargs='?', default='distributed_config.toml',
                       help='Path to TOML config file (default: distributed_config.toml)')
    parser.add_argument('--server', action='store_true',
                       help='Run coordinator (server mode)')
    parser.add_argument('--laptop', action='store_true',
                       help='Run GPU server (laptop mode)')
    
    args = parser.parse_args()
    
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        print("Copy the example config:")
        print("  cp rl_hanabi/training/distributed_config.example.toml distributed_config.toml")
        sys.exit(1)
    
    config = load_config(config_path)
    
    if args.laptop:
        run_gpu_server(config)
    elif args.server:
        run_coordinator(config)
    else:
        print("Please specify --server or --laptop mode:")
        print("  --laptop : Run GPU server (on machine with GPU)")
        print("  --server : Run coordinator (on machine without GPU)")
        sys.exit(1)


if __name__ == '__main__':
    main()
