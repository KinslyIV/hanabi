"""
Main training script for Hanabi self-play.
Uses multiprocessing for parallel game simulation.
"""

from __future__ import annotations

import argparse
import os
import random
import time
from collections import defaultdict
from dataclasses import asdict
from multiprocessing import Process, Queue, Event, cpu_count, set_start_method
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import queue

import numpy as np
import torch
import wandb

from rl_hanabi.model.belief_model import ActionDecoder
from rl_hanabi.training.game_simulator import (
    GameSimulator,
    GameConfig,
    GameResult,
    sample_game_config,
)
from rl_hanabi.training.mcts_simulator import (
    MCTSGameSimulator,
    run_mcts_self_play,
)
from rl_hanabi.training.data_collection import (
    ReplayBuffer,
    create_dataloader,
)
from rl_hanabi.training.trainer import (
    HanabiTrainer,
    log_game_metrics,
    init_wandb,
)

# Set multiprocessing start method to spawn for CUDA compatibility
try:
    set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set


def game_worker(
    worker_id: int,
    model_state_dict: Dict,
    model_config: Dict[str, int],
    game_queue: Queue,
    result_queue: Queue,
    stop_event: Event, # type: ignore
    device_str: str,
    simulation_config: Dict[str, Any],
):
    """
    Worker process that runs game simulations.
    
    Args:
        worker_id: Unique identifier for this worker
        model_state_dict: Serialized model weights
        model_config: Model architecture configuration
        game_queue: Queue of game configs to simulate
        result_queue: Queue to put results
        stop_event: Event to signal shutdown
        device_str: Device string (e.g., "cpu" or "cuda:0")
        simulation_config: Configuration for game simulation
    """
    print(f"[Worker {worker_id}] Starting on device {device_str}", flush=True)
    
    # Create model for this worker (always use CPU for workers to avoid CUDA issues)
    device = torch.device("cpu")
    model = ActionDecoder(
        max_num_colors=model_config["max_num_colors"],
        max_num_ranks=model_config["max_num_ranks"],
        max_hand_size=model_config["max_hand_size"],
        max_num_players=model_config["max_num_players"],
        num_heads=model_config.get("num_heads", 4),
        num_layers=model_config.get("num_layers", 4),
        d_model=model_config.get("d_model", 128),
        action_dim=model_config.get("action_dim", 4),
    )
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()
    
    # Check training mode for simulator selection
    training_mode = simulation_config.get("training_mode", "supervised")
    
    if training_mode == "mcts":
        # Use MCTS simulator for AlphaZero-style training
        simulator = MCTSGameSimulator(
            model=model,
            device=device,
            mcts_simulations=simulation_config.get("mcts_simulations", 100),
            c_puct=simulation_config.get("c_puct", 1.4),
            temperature=simulation_config.get("temperature", 1.0),
            temperature_drop_move=simulation_config.get("temperature_drop_move", 30),
            dirichlet_alpha=simulation_config.get("dirichlet_alpha", 0.3),
            dirichlet_weight=simulation_config.get("dirichlet_weight", 0.25),
            top_k_actions=simulation_config.get("top_k_actions", 5),
        )
    else:
        # Use standard simulator for supervised learning
        simulator = GameSimulator(
            model=model,
            device=device,
            temperature=simulation_config.get("temperature", 1.0),
            epsilon=simulation_config.get("epsilon", 0.1),
        )
    
    games_played = 0
    
    while not stop_event.is_set():
        try:
            # Get game config from queue (with timeout to check stop_event)
            game_config = game_queue.get(timeout=1.0)
        except queue.Empty:
            continue
        
        if game_config is None:  # Poison pill
            break
        
        try:
            # Run game simulation
            result = simulator.simulate_game(
                config=game_config,
                collect_all_perspectives=simulation_config.get("collect_all_perspectives", True),
            )
            
            # MCTSGameSimulator returns (GameResult, List[SearchTransition])
            # GameSimulator returns just GameResult
            if isinstance(result, tuple):
                game_result, _search_transitions = result
            else:
                game_result = result
            
            # Put result in queue
            result_queue.put((worker_id, game_result))
            games_played += 1
            
            if games_played % 10 == 0:
                print(f"[Worker {worker_id}] Completed {games_played} games")
                
        except Exception as e:
            print(f"[Worker {worker_id}] Error in game simulation: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"[Worker {worker_id}] Shutting down after {games_played} games")


def collect_results(
    result_queue: Queue,
    buffer: ReplayBuffer,
    stop_event: Event, # type: ignore
    max_results: Optional[int] = None,
) -> Tuple[int, Dict[str, int]]:
    """
    Collect results from worker processes.
    
    Args:
        result_queue: Queue with game results
        buffer: Replay buffer to add transitions to
        stop_event: Event to signal shutdown
        max_results: Maximum number of results to collect
    
    Returns:
        Tuple of (number of results collected, config distribution)
    """
    collected = 0
    config_counts = defaultdict(int)
    
    while not stop_event.is_set():
        if max_results is not None and collected >= max_results:
            break
        
        try:
            worker_id, result = result_queue.get(timeout=0.5)
            
            # Handle case where result might be a tuple (from MCTS mode)
            if isinstance(result, tuple):
                game_result = result[0]
            else:
                game_result = result
            
            buffer.add_game_result(game_result)
            
            # Track config distribution
            config_key = f"p{game_result.game_config['num_players']}_c{game_result.game_config['num_colors']}_r{game_result.game_config['num_ranks']}"
            config_counts[config_key] += 1
            
            collected += 1
            
        except queue.Empty:
            continue
    
    return collected, dict(config_counts)


def run_training(args: argparse.Namespace):
    """Main training loop."""
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Device setup
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Model configuration
    model_config = {
        "max_num_colors": args.max_colors,
        "max_num_ranks": args.max_ranks,
        "max_hand_size": args.max_hand_size,
        "max_num_players": args.max_players,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "d_model": args.d_model,
        "action_dim": 4,
    }
    
    # Create model
    model = ActionDecoder(
        max_num_colors=model_config["max_num_colors"],
        max_num_ranks=model_config["max_num_ranks"],
        max_hand_size=model_config["max_hand_size"],
        max_num_players=model_config["max_num_players"],
        num_heads=model_config["num_heads"],
        num_layers=model_config["num_layers"],
        d_model=model_config["d_model"],
        action_dim=model_config["action_dim"],
    )
    
    # Load checkpoint if specified
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint from {args.checkpoint}")
    
    model.to(device)
    
    # Training configuration
    training_config = {
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "color_loss_weight": args.color_loss_weight,
        "rank_loss_weight": args.rank_loss_weight,
        "action_loss_weight": args.action_loss_weight,
        "value_loss_weight": args.value_loss_weight,
        "policy_loss_weight": args.policy_loss_weight,
        "training_mode": args.training_mode,
        "scheduler_t0": args.scheduler_t0,
        "scheduler_t_mult": 2,
        "min_lr": args.min_lr,
    }
    
    # Simulation configuration
    simulation_config = {
        "temperature": args.temperature,
        "epsilon": args.epsilon,
        "collect_all_perspectives": True,
        "training_mode": args.training_mode,
        # MCTS-specific settings
        "mcts_simulations": args.mcts_simulations,
        "c_puct": args.c_puct,
        "temperature_drop_move": args.temperature_drop_move,
        "dirichlet_alpha": args.dirichlet_alpha,
        "dirichlet_weight": args.dirichlet_weight,
        "top_k_actions": args.top_k_actions,
    }
    
    # Full configuration for WandB
    full_config = {
        **model_config,
        **training_config,
        **simulation_config,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "games_per_iteration": args.games_per_iteration,
        "train_steps_per_iteration": args.train_steps_per_iteration,
        "num_iterations": args.num_iterations,
        "buffer_size": args.buffer_size,
        "min_players": args.min_players,
        "max_players": args.max_players,
        "min_colors": args.min_colors,
        "max_colors": args.max_colors,
        "min_ranks": args.min_ranks,
        "max_ranks": args.max_ranks,
    }
    
    # Initialize WandB
    if args.use_wandb:
        run = init_wandb(
            project_name=args.wandb_project,
            config=full_config,
            run_name=args.run_name,
            tags=args.tags.split(",") if args.tags else None,
        )
        wandb.watch(model, log="all", log_freq=100)
    
    # Create directories
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    buffer_dir = Path(args.buffer_dir) if args.buffer_dir else None
    
    # Create replay buffer
    buffer = ReplayBuffer(
        max_size=args.buffer_size,
        save_dir=buffer_dir,
    )
    
    # Create trainer
    trainer = HanabiTrainer(
        model=model,
        buffer=buffer,
        device=device,
        config=training_config,
        checkpoint_dir=checkpoint_dir,
    )
    
    # Multiprocessing setup
    num_workers = args.num_workers if args.num_workers > 0 else max(1, cpu_count() - 1)
    print(f"Using {num_workers} worker processes for game simulation")
    
    # Determine worker devices
    if torch.cuda.is_available() and args.device != "cpu":
        num_gpus = torch.cuda.device_count()
        worker_devices = [f"cuda:{i % num_gpus}" for i in range(num_workers)]
    else:
        worker_devices = ["cpu"] * num_workers
    
    # Main training loop
    for iteration in range(args.num_iterations):
        print(f"\n{'='*60}")
        print(f"Iteration {iteration + 1}/{args.num_iterations}")
        print(f"{'='*60}")
        
        # === Phase 1: Self-play data collection ===
        print(f"\nPhase 1: Collecting {args.games_per_iteration} games...")
        
        # Create communication queues
        game_queue = Queue()
        result_queue = Queue()
        stop_event = Event()
        
        # Get current model state dict
        model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        
        # Start worker processes
        workers = []
        for worker_id in range(num_workers):
            p = Process(
                target=game_worker,
                args=(
                    worker_id,
                    model_state_dict,
                    model_config,
                    game_queue,
                    result_queue,
                    stop_event,
                    worker_devices[worker_id],
                    simulation_config,
                ),
            )
            p.start()
            workers.append(p)
        
        # Queue game configurations
        for _ in range(args.games_per_iteration):
            config = sample_game_config(
                num_players_range=(args.min_players, args.max_players),
                num_colors_range=(args.min_colors, args.max_colors),
                num_ranks_range=(args.min_ranks, args.max_ranks),
            )
            game_queue.put(config)
        
        # Collect results
        start_time = time.time()
        collected, config_counts = collect_results(
            result_queue=result_queue,
            buffer=buffer,
            stop_event=stop_event,
            max_results=args.games_per_iteration,
        )
        
        # Signal workers to stop
        stop_event.set()
        for _ in range(num_workers):
            game_queue.put(None)  # Poison pills
        
        # Wait for workers to finish
        for p in workers:
            p.join(timeout=5.0)
            if p.is_alive():
                p.terminate()
        
        collection_time = time.time() - start_time
        print(f"Collected {collected} games in {collection_time:.2f}s")
        
        # Log game metrics
        buffer_stats = buffer.get_statistics()
        print(f"Buffer stats: {buffer_stats}")
        
        if args.use_wandb:
            log_game_metrics(buffer_stats, config_counts, use_wandb=True)
            wandb.log({
                "iteration": iteration + 1,
                "collection_time": collection_time,
            })
        
        # === Phase 2: Training ===
        print(f"\nPhase 2: Training for {args.train_steps_per_iteration} steps...")
        
        if len(buffer) < args.batch_size:
            print(f"Not enough data in buffer ({len(buffer)} < {args.batch_size}), skipping training")
            continue
        
        # Create dataloader (use num_workers=0 to avoid pickle issues with spawn)
        dataloader = create_dataloader(
            buffer=buffer,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,  # Single-process loading to avoid pickle issues
            max_num_players=model_config["max_num_players"],
            max_num_colors=model_config["max_num_colors"],
            max_num_ranks=model_config["max_num_ranks"],
            max_hand_size=model_config["max_hand_size"],
        )
        
        train_start = time.time()
        steps_done = 0
        
        for batch_idx, batch in enumerate(dataloader):
            if steps_done >= args.train_steps_per_iteration:
                break
            
            metrics = trainer.train_step(batch)
            steps_done += 1
            
            if steps_done % args.log_interval == 0:
                print(f"  Step {steps_done}/{args.train_steps_per_iteration}, Loss: {metrics['total_loss']:.4f}")
                
                if args.use_wandb:
                    log_dict = {
                        "train/step": trainer.global_step,
                        "train/loss": metrics["total_loss"],
                        "train/color_loss": metrics["color_loss"],
                        "train/rank_loss": metrics["rank_loss"],
                        "train/action_loss": metrics["action_loss"],
                        "train/color_accuracy": metrics["color_accuracy"],
                        "train/rank_accuracy": metrics["rank_accuracy"],
                        "train/action_accuracy": metrics["action_accuracy"],
                        "train/learning_rate": metrics["learning_rate"],
                    }
                    
                    # Add MCTS-specific metrics if in MCTS mode
                    if args.training_mode == "mcts":
                        log_dict.update({
                            "train/policy_loss": metrics.get("policy_loss", 0.0),
                            "train/value_loss": metrics.get("value_loss", 0.0),
                            "train/value_mae": metrics.get("value_mae", 0.0),
                            "train/mean_reward": metrics.get("mean_reward", 0.0),
                            "train/mean_value_pred": metrics.get("mean_value_pred", 0.0),
                        })
                    
                    wandb.log(log_dict)
        
        train_time = time.time() - train_start
        print(f"Training completed in {train_time:.2f}s")
        
        # === Phase 3: Checkpointing ===
        if (iteration + 1) % args.save_interval == 0:
            checkpoint_path = trainer.save_checkpoint(
                filename=f"checkpoint_iter_{iteration + 1}.pt",
                extra_data={
                    "iteration": iteration + 1,
                    "buffer_stats": buffer_stats,
                    "config_counts": config_counts,
                },
            )
            
            if args.use_wandb:
                wandb.save(str(checkpoint_path))
        
        # Save latest checkpoint
        trainer.save_checkpoint(filename="checkpoint_latest.pt")
        
        # Clear game results to save memory (keep transitions)
        buffer.clear_game_results()
        
        # Update exploration parameters (decay epsilon)
        if args.epsilon_decay > 0:
            simulation_config["epsilon"] *= (1 - args.epsilon_decay)
            simulation_config["epsilon"] = max(args.min_epsilon, simulation_config["epsilon"])
            print(f"Epsilon: {simulation_config['epsilon']:.4f}")
    
    print("\nTraining complete!")
    
    # Final checkpoint
    trainer.save_checkpoint(filename="checkpoint_final.pt")
    
    if args.use_wandb:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Hanabi Self-Play Training")
    
    # General settings
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu, cuda, or auto)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory for checkpoints")
    parser.add_argument("--buffer-dir", type=str, default=None, help="Directory for buffer saves")
    
    # Model architecture
    parser.add_argument("--max-players", type=int, default=5, help="Maximum number of players")
    parser.add_argument("--max-colors", type=int, default=5, help="Maximum number of colors")
    parser.add_argument("--max-ranks", type=int, default=5, help="Maximum number of ranks")
    parser.add_argument("--max-hand-size", type=int, default=5, help="Maximum hand size")
    parser.add_argument("--num-heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--d-model", type=int, default=128, help="Model dimension")
    
    # Training parameters
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Max gradient norm")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--scheduler-t0", type=int, default=1000, help="Scheduler T_0")
    parser.add_argument("--min-lr", type=float, default=1e-6, help="Minimum learning rate")
    
    # Loss weights
    parser.add_argument("--color-loss-weight", type=float, default=1.0, help="Color prediction loss weight")
    parser.add_argument("--rank-loss-weight", type=float, default=1.0, help="Rank prediction loss weight")
    parser.add_argument("--action-loss-weight", type=float, default=1.0, help="Action prediction loss weight")
    parser.add_argument("--value-loss-weight", type=float, default=1.0, help="Value prediction loss weight")
    parser.add_argument("--policy-loss-weight", type=float, default=1.0, help="MCTS policy distillation loss weight")
    
    # Training mode
    parser.add_argument("--training-mode", type=str, default="supervised", 
                        choices=["supervised", "mcts"],
                        help="Training mode: 'supervised' (cross-entropy with chosen actions) or 'mcts' (policy distillation from MCTS)")
    
    # Self-play parameters
    parser.add_argument("--num-workers", type=int, default=0, help="Number of worker processes (0 = auto)")
    parser.add_argument("--games-per-iteration", type=int, default=100, help="Games to play per iteration")
    parser.add_argument("--train-steps-per-iteration", type=int, default=500, help="Training steps per iteration")
    parser.add_argument("--num-iterations", type=int, default=100, help="Number of iterations")
    parser.add_argument("--buffer-size", type=int, default=100000, help="Replay buffer size")
    
    # Game configuration ranges
    parser.add_argument("--min-players", type=int, default=2, help="Minimum number of players")
    parser.add_argument("--min-colors", type=int, default=3, help="Minimum number of colors")
    parser.add_argument("--min-ranks", type=int, default=3, help="Minimum number of ranks")
    
    # Exploration parameters
    parser.add_argument("--temperature", type=float, default=1.0, help="Action sampling temperature")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Epsilon for epsilon-greedy")
    parser.add_argument("--epsilon-decay", type=float, default=0.01, help="Epsilon decay per iteration")
    parser.add_argument("--min-epsilon", type=float, default=0.01, help="Minimum epsilon")
    
    # MCTS parameters (used when training_mode=mcts)
    parser.add_argument("--mcts-simulations", type=int, default=100, help="Number of MCTS simulations per move")
    parser.add_argument("--c-puct", type=float, default=1.4, help="PUCT exploration constant")
    parser.add_argument("--temperature-drop-move", type=int, default=30, help="Move number after which temperature drops to 0")
    parser.add_argument("--dirichlet-alpha", type=float, default=0.3, help="Dirichlet noise alpha for root exploration")
    parser.add_argument("--dirichlet-weight", type=float, default=0.25, help="Weight for Dirichlet noise at root")
    parser.add_argument("--top-k-actions", type=int, default=10, help="Number of top actions to expand in MCTS")
    
    # Logging
    parser.add_argument("--log-interval", type=int, default=100, help="Log every N steps")
    parser.add_argument("--save-interval", type=int, default=10, help="Save checkpoint every N iterations")
    
    # WandB
    parser.add_argument("--use-wandb", action="store_true", help="Use Weights & Biases for logging")
    parser.add_argument("--wandb-project", type=str, default="hanabi-selfplay", help="WandB project name")
    parser.add_argument("--run-name", type=str, default=None, help="WandB run name")
    parser.add_argument("--tags", type=str, default=None, help="WandB tags (comma-separated)")
    
    args = parser.parse_args()
    
    run_training(args)


if __name__ == "__main__":
    main()
