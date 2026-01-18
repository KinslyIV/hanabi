"""
Distributed Training Coordinator - Runs on the server (no GPU).
Handles game simulation, data collection, and coordination with GPU server.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import pickle
import queue
import random
import signal
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from multiprocessing import Process, Queue, Event, cpu_count
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from rl_hanabi.training.distributed.gpu_client import GPUClient, GPUClientConfig
from rl_hanabi.training.game_simulator import (
    GameSimulator,
    GameConfig,
    GameResult,
    sample_game_config,
)
from rl_hanabi.training.data_collection import ReplayBuffer, create_dataloader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('coordinator')


class TrainingState:
    """Persistent training state that survives restarts."""
    
    def __init__(self, state_file: Path):
        self.state_file = state_file
        self.iteration = 0
        self.total_games = 0
        self.total_train_steps = 0
        self.simulation_config = {}
        self.best_avg_score = 0.0
        self.paused = False
        self.pause_reason = ""
        
        self._load()
    
    def _load(self) -> None:
        """Load state from file."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    self.iteration = data.get('iteration', 0)
                    self.total_games = data.get('total_games', 0)
                    self.total_train_steps = data.get('total_train_steps', 0)
                    self.simulation_config = data.get('simulation_config', {})
                    self.best_avg_score = data.get('best_avg_score', 0.0)
                    logger.info(f"Loaded training state: iteration={self.iteration}, "
                              f"total_games={self.total_games}")
            except Exception as e:
                logger.warning(f"Failed to load training state: {e}")
    
    def save(self) -> None:
        """Save state to file."""
        data = {
            'iteration': self.iteration,
            'total_games': self.total_games,
            'total_train_steps': self.total_train_steps,
            'simulation_config': self.simulation_config,
            'best_avg_score': self.best_avg_score,
        }
        with open(self.state_file, 'w') as f:
            json.dump(data, f, indent=2)


def game_worker(
    worker_id: int,
    model_state_dict: Dict,
    model_config: Dict[str, int],
    game_queue: Queue,
    result_queue: Queue,
    stop_event: Event, # type: ignore
    simulation_config: Dict[str, Any],
):
    """Worker process that runs game simulations on CPU."""
    from rl_hanabi.model.belief_model import ActionDecoder
    
    logger.info(f"[Worker {worker_id}] Starting")
    
    # Create model for this worker (CPU only)
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
    
    # Create simulator
    simulator = GameSimulator(
        model=model,
        device=device,
        temperature=simulation_config.get("temperature", 1.0),
        epsilon=simulation_config.get("epsilon", 0.1),
        max_num_colors=model_config["max_num_colors"],
        max_num_ranks=model_config["max_num_ranks"],
        max_hand_size=model_config["max_hand_size"],
        max_num_players=model_config["max_num_players"],
    )
    
    games_played = 0
    
    while not stop_event.is_set():
        try:
            game_config = game_queue.get(timeout=1.0)
        except queue.Empty:
            continue
        
        if game_config is None:  # Poison pill
            break
        
        try:
            result = simulator.simulate_game(
                config=game_config,
                collect_all_perspectives=simulation_config.get("collect_all_perspectives", True),
            )
            result_queue.put((worker_id, result))
            games_played += 1
            
            if games_played % 10 == 0:
                logger.debug(f"[Worker {worker_id}] Completed {games_played} games")
        except Exception as e:
            logger.error(f"[Worker {worker_id}] Error: {e}")
            continue
    
    logger.info(f"[Worker {worker_id}] Shutting down after {games_played} games")


class DistributedCoordinator:
    """
    Coordinates distributed training between server (data collection) and laptop (GPU training).
    """
    
    def __init__(
        self,
        gpu_client: GPUClient,
        model_config: Dict[str, Any],
        training_config: Dict[str, Any],
        simulation_config: Dict[str, Any],
        buffer: ReplayBuffer,
        state: TrainingState,
        num_workers: int = 0,
        checkpoint_dir: Optional[Path] = None,
        use_wandb: bool = False,
    ):
        self.gpu_client = gpu_client
        self.model_config = model_config
        self.training_config = training_config
        self.simulation_config = simulation_config
        self.buffer = buffer
        self.state = state
        self.num_workers = num_workers if num_workers > 0 else max(1, cpu_count() - 1)
        self.checkpoint_dir = checkpoint_dir
        self.use_wandb = use_wandb and HAS_WANDB
        
        if checkpoint_dir:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Current model weights for workers
        self._current_weights: Optional[Dict] = None
        self._weights_lock = asyncio.Lock()
        
        # Control
        self._shutdown_event = asyncio.Event()
        self._pause_event = asyncio.Event()
        
        # Statistics
        self.collection_times: List[float] = []
        self.training_times: List[float] = []
    
    def _on_gpu_connect(self) -> None:
        """Called when GPU server connects."""
        logger.info("GPU server connected - resuming training")
        self.state.paused = False
        self.state.pause_reason = ""
        self.state.save()
    
    def _on_gpu_disconnect(self, reason: str) -> None:
        """Called when GPU server disconnects."""
        logger.warning(f"GPU server disconnected: {reason}")
        self.state.paused = True
        self.state.pause_reason = f"GPU disconnected: {reason}"
        self.state.save()
        self._pause_event.set()
    
    def _on_gpu_reconnecting(self, attempt: int, delay: float) -> None:
        """Called when attempting reconnection."""
        logger.info(f"Reconnecting to GPU server (attempt {attempt}, delay {delay:.1f}s)")
    
    async def _fetch_model_weights(self) -> Dict:
        """Fetch current model weights from GPU server."""
        async with self._weights_lock:
            self._current_weights = await self.gpu_client.get_weights()
            return self._current_weights
    
    def _collect_games_sync(
        self,
        num_games: int,
        game_config_ranges: Dict[str, Tuple[int, int]],
    ) -> Tuple[int, Dict[str, int], float]:
        """
        Synchronously collect games using worker processes.
        Returns: (num_collected, config_counts, collection_time)
        """
        if self._current_weights is None:
            raise RuntimeError("No model weights available")
        
        game_queue = Queue()
        result_queue = Queue()
        stop_event = Event()
        
        # Start workers
        workers = []
        for worker_id in range(self.num_workers):
            p = Process(
                target=game_worker,
                args=(
                    worker_id,
                    self._current_weights,
                    self.model_config,
                    game_queue,
                    result_queue,
                    stop_event,
                    self.simulation_config,
                ),
            )
            p.start()
            workers.append(p)
        
        # Queue game configurations
        for _ in range(num_games):
            config = sample_game_config(
                num_players_range=game_config_ranges.get("players", (2, 5)),
                num_colors_range=game_config_ranges.get("colors", (3, 5)),
                num_ranks_range=game_config_ranges.get("ranks", (3, 5)),
            )
            game_queue.put(config)
        
        # Collect results
        start_time = time.time()
        collected = 0
        config_counts = defaultdict(int)
        
        while collected < num_games:
            try:
                worker_id, result = result_queue.get(timeout=1.0)
                self.buffer.add_game_result(result)
                
                config_key = f"p{result.game_config['num_players']}_c{result.game_config['num_colors']}_r{result.game_config['num_ranks']}"
                config_counts[config_key] += 1
                collected += 1
                
            except queue.Empty:
                if all(not p.is_alive() for p in workers):
                    break
                continue
        
        # Shutdown workers
        stop_event.set()
        for _ in range(self.num_workers):
            game_queue.put(None)
        
        for p in workers:
            p.join(timeout=5.0)
            if p.is_alive():
                p.terminate()
        
        collection_time = time.time() - start_time
        return collected, dict(config_counts), collection_time
    
    async def _train_steps(
        self,
        num_steps: int,
        batch_size: int,
        log_interval: int = 100,
    ) -> Tuple[List[Dict[str, float]], float]:
        """
        Run training steps on GPU server.
        Returns: (metrics_list, training_time)
        """
        if len(self.buffer) < batch_size:
            logger.warning(f"Not enough data ({len(self.buffer)} < {batch_size})")
            return [], 0.0
        
        dataloader = create_dataloader(
            buffer=self.buffer,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            max_num_players=self.model_config["max_num_players"],
            max_num_colors=self.model_config["max_num_colors"],
            max_num_ranks=self.model_config["max_num_ranks"],
            max_hand_size=self.model_config["max_hand_size"],
        )
        
        all_metrics = []
        start_time = time.time()
        steps_done = 0
        
        for batch_idx, batch in enumerate(dataloader):
            if steps_done >= num_steps:
                break
            
            # Check if paused (GPU disconnected)
            if self.state.paused:
                logger.info("Training paused - waiting for GPU server...")
                await self.gpu_client.wait_for_connection(timeout=None)
                self.state.paused = False
                logger.info("GPU server back - resuming training")
            
            # Convert batch to numpy for serialization
            numpy_batch = {
                k: v.numpy() if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            
            try:
                metrics = await self.gpu_client.train_step(numpy_batch)
                all_metrics.append(metrics)
                steps_done += 1
                self.state.total_train_steps += 1
                
                if steps_done % log_interval == 0:
                    avg_loss = np.mean([m['total_loss'] for m in all_metrics[-log_interval:]])
                    logger.info(f"  Step {steps_done}/{num_steps}, Loss: {avg_loss:.4f}")
                    
                    if self.use_wandb:
                        wandb.log({ # type: ignore
                            "train/step": self.state.total_train_steps,
                            "train/loss": metrics["total_loss"],
                            "train/color_loss": metrics.get("color_loss", 0),
                            "train/rank_loss": metrics.get("rank_loss", 0),
                            "train/action_loss": metrics.get("action_loss", 0),
                            "train/color_accuracy": metrics.get("color_accuracy", 0),
                            "train/rank_accuracy": metrics.get("rank_accuracy", 0),
                            "train/action_accuracy": metrics.get("action_accuracy", 0),
                            "train/learning_rate": metrics.get("learning_rate", 0),
                        })
            
            except Exception as e:
                logger.error(f"Training step failed: {e}")
                # Will retry on next iteration after reconnection
                break
        
        training_time = time.time() - start_time
        return all_metrics, training_time
    
    async def run_iteration(
        self,
        games_per_iteration: int,
        train_steps_per_iteration: int,
        batch_size: int,
        game_config_ranges: Dict[str, Tuple[int, int]],
        save_interval: int = 10,
        log_interval: int = 100,
    ) -> Dict[str, Any]:
        """Run a single training iteration."""
        iteration = self.state.iteration + 1
        logger.info(f"\n{'='*60}")
        logger.info(f"Iteration {iteration}")
        logger.info(f"{'='*60}")
        
        # Phase 1: Get latest model weights
        logger.info("Fetching model weights from GPU server...")
        try:
            await self._fetch_model_weights()
        except Exception as e:
            logger.error(f"Failed to fetch weights: {e}")
            return {"success": False, "error": str(e)}
        
        # Phase 2: Collect games
        logger.info(f"Phase 1: Collecting {games_per_iteration} games...")
        
        # Run collection in executor to not block event loop
        loop = asyncio.get_event_loop()
        collected, config_counts, collection_time = await loop.run_in_executor(
            None,
            self._collect_games_sync,
            games_per_iteration,
            game_config_ranges,
        )
        
        self.state.total_games += collected
        self.collection_times.append(collection_time)
        logger.info(f"Collected {collected} games in {collection_time:.2f}s")
        
        # Log buffer stats
        buffer_stats = self.buffer.get_statistics()
        logger.info(f"Buffer: {buffer_stats['buffer_size']} transitions, "
                   f"avg_score: {buffer_stats.get('avg_score', 0):.2f}")
        
        if self.use_wandb:
            wandb.log({ # type: ignore
                "games/collected": collected,
                "games/total": self.state.total_games,
                "games/collection_time": collection_time,
                "games/buffer_size": buffer_stats['buffer_size'],
                "games/avg_score": buffer_stats.get('avg_score', 0),
                "games/avg_normalized_score": buffer_stats.get('avg_normalized_score', 0),
                "iteration": iteration,
            })
        
        # Phase 3: Training
        logger.info(f"Phase 2: Training for {train_steps_per_iteration} steps...")
        
        metrics_list, training_time = await self._train_steps(
            num_steps=train_steps_per_iteration,
            batch_size=batch_size,
            log_interval=log_interval,
        )
        
        self.training_times.append(training_time)
        
        if metrics_list:
            avg_loss = np.mean([m['total_loss'] for m in metrics_list])
            logger.info(f"Training completed in {training_time:.2f}s, avg_loss: {avg_loss:.4f}")
        else:
            avg_loss = float('inf')
        
        # Phase 4: Save checkpoint
        if iteration % save_interval == 0:
            try:
                filepath = await self.gpu_client.save_checkpoint(f"checkpoint_iter_{iteration}.pt")
                logger.info(f"Saved checkpoint: {filepath}")
            except Exception as e:
                logger.error(f"Failed to save checkpoint: {e}")
        
        # Always save latest
        try:
            await self.gpu_client.save_checkpoint("checkpoint_latest.pt")
        except Exception as e:
            logger.error(f"Failed to save latest checkpoint: {e}")
        
        # Update state
        self.state.iteration = iteration
        
        # Update exploration parameters
        if self.training_config.get("epsilon_decay", 0) > 0:
            old_epsilon = self.simulation_config.get("epsilon", 0.1)
            new_epsilon = old_epsilon * (1 - self.training_config["epsilon_decay"])
            new_epsilon = max(self.training_config.get("min_epsilon", 0.01), new_epsilon)
            self.simulation_config["epsilon"] = new_epsilon
            self.state.simulation_config = self.simulation_config
            logger.info(f"Epsilon: {old_epsilon:.4f} -> {new_epsilon:.4f}")
        
        # Save training state
        self.state.save()
        
        # Clear game results to save memory
        self.buffer.clear_game_results()
        
        return {
            "success": True,
            "iteration": iteration,
            "games_collected": collected,
            "collection_time": collection_time,
            "training_time": training_time,
            "avg_loss": avg_loss if metrics_list else None,
            "buffer_size": len(self.buffer),
        }
    
    async def run(
        self,
        num_iterations: int,
        games_per_iteration: int,
        train_steps_per_iteration: int,
        batch_size: int,
        game_config_ranges: Dict[str, Tuple[int, int]],
        save_interval: int = 10,
        log_interval: int = 100,
    ) -> None:
        """Run the full training loop."""
        
        # Setup GPU client callbacks
        self.gpu_client.config.on_connect = self._on_gpu_connect
        self.gpu_client.config.on_disconnect = self._on_gpu_disconnect
        self.gpu_client.config.on_reconnecting = self._on_gpu_reconnecting
        
        # Connect to GPU server
        logger.info("Connecting to GPU server...")
        await self.gpu_client.connect()
        
        if not self.gpu_client.is_connected:
            logger.info("Waiting for GPU server connection...")
            connected = await self.gpu_client.wait_for_connection(timeout=None)
            if not connected:
                logger.error("Could not connect to GPU server")
                return
        
        logger.info(f"Starting training from iteration {self.state.iteration + 1}")
        
        try:
            while self.state.iteration < num_iterations and not self._shutdown_event.is_set():
                result = await self.run_iteration(
                    games_per_iteration=games_per_iteration,
                    train_steps_per_iteration=train_steps_per_iteration,
                    batch_size=batch_size,
                    game_config_ranges=game_config_ranges,
                    save_interval=save_interval,
                    log_interval=log_interval,
                )
                
                if not result["success"]:
                    logger.warning(f"Iteration failed: {result.get('error')}")
                    # Wait before retry
                    await asyncio.sleep(5)
        
        except asyncio.CancelledError:
            logger.info("Training cancelled")
        
        finally:
            # Final checkpoint
            try:
                await self.gpu_client.save_checkpoint("checkpoint_final.pt")
            except Exception as e:
                logger.error(f"Failed to save final checkpoint: {e}")
            
            await self.gpu_client.disconnect()
        
        logger.info("Training complete!")
    
    async def shutdown(self) -> None:
        """Signal shutdown."""
        logger.info("Shutting down coordinator...")
        self._shutdown_event.set()


async def run_distributed_training(args: argparse.Namespace) -> None:
    """Main async training function."""
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Setup directories
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    state_file = checkpoint_dir / "training_state.json"
    
    # Load or create training state
    state = TrainingState(state_file)
    
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
    
    # Training configuration
    training_config = {
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "epsilon_decay": args.epsilon_decay,
        "min_epsilon": args.min_epsilon,
    }
    
    # Simulation configuration (restore from state if available)
    if state.simulation_config:
        simulation_config = state.simulation_config
    else:
        simulation_config = {
            "temperature": args.temperature,
            "epsilon": args.epsilon,
            "collect_all_perspectives": True,
        }
    
    # Game config ranges
    game_config_ranges = {
        "players": (args.min_players, args.max_players),
        "colors": (args.min_colors, args.max_colors),
        "ranks": (args.min_ranks, args.max_ranks),
    }
    
    # Full config for wandb
    full_config = {
        **model_config,
        **training_config,
        **simulation_config,
        **{f"game_{k}": v for k, v in game_config_ranges.items()},
        "seed": args.seed,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "games_per_iteration": args.games_per_iteration,
        "train_steps_per_iteration": args.train_steps_per_iteration,
        "num_iterations": args.num_iterations,
        "buffer_size": args.buffer_size,
        "gpu_host": args.gpu_host,
        "gpu_port": args.gpu_port,
    }
    
    # Initialize WandB
    if args.use_wandb and HAS_WANDB:
        wandb.init(  # type: ignore
            project=args.wandb_project, 
            name=args.run_name,
            config=full_config,
            tags=args.tags.split(",") if args.tags else ["distributed"],
            resume="allow",
        )
    
    # Create GPU client
    gpu_client_config = GPUClientConfig(
        host=args.gpu_host,
        port=args.gpu_port,
        auth_token=args.auth_token,
        initial_retry_delay=1.0,
        max_retry_delay=60.0,
        connection_timeout=30.0,
        request_timeout=300.0,
        ping_interval=30.0,
    )
    gpu_client = GPUClient(gpu_client_config)
    
    # Create replay buffer
    buffer_dir = Path(args.buffer_dir) if args.buffer_dir else None
    buffer = ReplayBuffer(max_size=args.buffer_size, save_dir=buffer_dir)
    
    # Create coordinator
    coordinator = DistributedCoordinator(
        gpu_client=gpu_client,
        model_config=model_config,
        training_config=training_config,
        simulation_config=simulation_config,
        buffer=buffer,
        state=state,
        num_workers=args.num_workers,
        checkpoint_dir=checkpoint_dir,
        use_wandb=args.use_wandb and HAS_WANDB,
    )
    
    # Setup signal handlers
    loop = asyncio.get_event_loop()
    
    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}")
        loop.create_task(coordinator.shutdown())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run training
    try:
        await coordinator.run(
            num_iterations=args.num_iterations,
            games_per_iteration=args.games_per_iteration,
            train_steps_per_iteration=args.train_steps_per_iteration,
            batch_size=args.batch_size,
            game_config_ranges=game_config_ranges,
            save_interval=args.save_interval,
            log_interval=args.log_interval,
        )
    finally:
        if args.use_wandb and HAS_WANDB:
            wandb.finish() # type: ignore


def main():
    parser = argparse.ArgumentParser(description="Distributed Hanabi Training Coordinator")
    
    # GPU server connection
    parser.add_argument("--gpu-host", type=str, required=True, help="GPU server hostname/IP")
    parser.add_argument("--gpu-port", type=int, default=5555, help="GPU server port")
    parser.add_argument("--auth-token", type=str, default=None, help="Authentication token")
    
    # General settings
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--buffer-dir", type=str, default=None, help="Buffer save directory")
    
    # Model architecture (must match GPU server)
    parser.add_argument("--max-players", type=int, default=5, help="Maximum players")
    parser.add_argument("--max-colors", type=int, default=5, help="Maximum colors")
    parser.add_argument("--max-ranks", type=int, default=5, help="Maximum ranks")
    parser.add_argument("--max-hand-size", type=int, default=5, help="Maximum hand size")
    parser.add_argument("--num-heads", type=int, default=4, help="Attention heads")
    parser.add_argument("--num-layers", type=int, default=4, help="Transformer layers")
    parser.add_argument("--d-model", type=int, default=128, help="Model dimension")
    
    # Training parameters
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Max gradient norm")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    
    # Self-play parameters
    parser.add_argument("--num-workers", type=int, default=0, help="Worker processes (0=auto)")
    parser.add_argument("--games-per-iteration", type=int, default=100, help="Games per iteration")
    parser.add_argument("--train-steps-per-iteration", type=int, default=500, help="Training steps")
    parser.add_argument("--num-iterations", type=int, default=100, help="Total iterations")
    parser.add_argument("--buffer-size", type=int, default=100000, help="Buffer size")
    
    # Game configuration ranges
    parser.add_argument("--min-players", type=int, default=2, help="Minimum players")
    parser.add_argument("--min-colors", type=int, default=3, help="Minimum colors")
    parser.add_argument("--min-ranks", type=int, default=3, help="Minimum ranks")
    
    # Exploration
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Epsilon-greedy")
    parser.add_argument("--epsilon-decay", type=float, default=0.01, help="Epsilon decay")
    parser.add_argument("--min-epsilon", type=float, default=0.01, help="Minimum epsilon")
    
    # Logging
    parser.add_argument("--log-interval", type=int, default=100, help="Log every N steps")
    parser.add_argument("--save-interval", type=int, default=10, help="Save every N iterations")
    
    # WandB
    parser.add_argument("--use-wandb", action="store_true", help="Use WandB")
    parser.add_argument("--wandb-project", type=str, default="hanabi-distributed", help="WandB project")
    parser.add_argument("--run-name", type=str, default=None, help="WandB run name")
    parser.add_argument("--tags", type=str, default=None, help="WandB tags (comma-separated)")
    
    args = parser.parse_args()
    
    asyncio.run(run_distributed_training(args))


if __name__ == "__main__":
    main()
