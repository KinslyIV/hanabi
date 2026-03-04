"""
Main training script for Hanabi self-play.
Uses multiprocessing for parallel game simulation.
"""

from __future__ import annotations

import argparse
import random
import time
from collections import defaultdict
from multiprocessing import Process, Queue, Event, cpu_count, set_start_method
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import queue

import numpy as np
import torch
import wandb

from rl_hanabi.model.action_decoder import ActionDecoder
from rl_hanabi.model.tokenizer import HLETokenizer, TokenizationConfig
from rl_hanabi.training.game_simulator import (
    GameSimulator,
    sample_game_config,
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
    model_config: Dict[str, Any],
    game_queue: Queue,
    result_queue: Queue,
    stop_event: Event, # type: ignore
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
        simulation_config: Configuration for game simulation
    """
    print(f"[Worker {worker_id}] Starting", flush=True)
    
    # Create model for this worker (always use CPU for workers to avoid CUDA issues)
    device = torch.device("cpu")
    token_config = model_config["token_config"]
    model = ActionDecoder(
        num_colors=model_config["max_num_colors"],
        num_ranks=model_config["max_num_ranks"],
        max_cards=model_config["max_cards"],
        hand_size=model_config["max_hand_size"],
        num_players=model_config["max_num_players"],
        num_heads=model_config.get("num_heads", 4),
        num_layers=model_config.get("num_layers", 4),
        d_model=model_config.get("d_model", 128),
        action_dim=model_config.get("action_dim", 4),
        token_config=token_config,
    )
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()
    
    tokenizer = HLETokenizer(token_config)
    simulator = GameSimulator(
        model=model,
        tokenizer=tokenizer,
        device=device,
        temperature=simulation_config.get("temperature", 1.0),
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
            result = simulator.simulate_game(config=game_config)
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
            buffer.add_game_result(result)
            
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
    
    token_config = TokenizationConfig(
        num_colors=args.max_colors,
        num_ranks=args.max_ranks,
        hand_size=args.max_hand_size,
        max_num_players=args.max_players,
        max_info_tokens=args.max_info_tokens,
        max_life_tokens=args.max_life_tokens,
    )

    # Model configuration
    model_config = {
        "max_num_colors": args.max_colors,
        "max_num_ranks": args.max_ranks,
        "max_hand_size": args.max_hand_size,
        "max_num_players": args.max_players,
        "max_cards": args.max_colors * args.max_ranks,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "d_model": args.d_model,
        "action_dim": 4,
        "token_config": token_config,
    }
    
    # Create model
    model = ActionDecoder(
        num_colors=model_config["max_num_colors"],
        num_ranks=model_config["max_num_ranks"],
        max_cards=model_config["max_cards"],
        hand_size=model_config["max_hand_size"],
        num_players=model_config["max_num_players"],
        num_heads=model_config["num_heads"],
        num_layers=model_config["num_layers"],
        d_model=model_config["d_model"],
        action_dim=model_config["action_dim"],
        token_config=token_config,
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
        "scheduler_t0": args.scheduler_t0,
        "scheduler_t_mult": 2,
        "min_lr": args.min_lr,
    }
    
    # Simulation configuration
    simulation_config = {
        "temperature": args.temperature,
    }
    
    # Full configuration for WandB
    model_config_log = {k: v for k, v in model_config.items() if k != "token_config"}
    full_config = {
        **model_config_log,
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
        "max_info_tokens": args.max_info_tokens,
        "max_life_tokens": args.max_life_tokens,
        "temperature_decay": args.temperature_decay,
        "min_temperature": args.min_temperature,
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
        token_config=token_config,
        checkpoint_dir=checkpoint_dir,
    )
    
    # Multiprocessing setup
    num_workers = args.num_workers if args.num_workers > 0 else max(1, cpu_count() - 1)
    print(f"Using {num_workers} worker processes for game simulation")
    
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
                max_information_tokens=args.max_info_tokens,
                max_life_tokens=args.max_life_tokens,
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
            token_config=token_config,
            shuffle=True,
            num_workers=0,
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
                    wandb.log({
                        "train/step": trainer.global_step,
                        "train/loss": metrics["total_loss"],
                        "train/action_accuracy": metrics["action_accuracy"],
                        "train/mean_reward": metrics["mean_reward"],
                        "train/mean_advantage": metrics["mean_advantage"],
                        "train/learning_rate": metrics["learning_rate"],
                    })
        
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
        
        # Update exploration parameters (decay temperature)
        if args.temperature_decay > 0:
            old_temp = simulation_config["temperature"]
            simulation_config["temperature"] *= (1 - args.temperature_decay)
            simulation_config["temperature"] = max(args.min_temperature, simulation_config["temperature"])
            print(f"Temperature: {old_temp:.4f} -> {simulation_config['temperature']:.4f}")
    
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
    parser.add_argument("--max-info-tokens", type=int, default=8, help="Maximum information tokens")
    parser.add_argument("--max-life-tokens", type=int, default=3, help="Maximum life tokens")
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
    parser.add_argument("--temperature-decay", type=float, default=0.01, help="Temperature decay per iteration")
    parser.add_argument("--min-temperature", type=float, default=0.1, help="Minimum temperature")
    
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
