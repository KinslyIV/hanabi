"""
Main training script for Hanabi self-play.
Uses multiprocessing for parallel game simulation.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader

from rl_hanabi.model.action_decoder import ActionDecoder, ActionDecoderConfig
from rl_hanabi.model.tokenizer import HLETokenizer, TokenizationConfig
from rl_hanabi.training.utils import build_game_config, load_config
from rl_hanabi.training.game_simulator import GameSimulator
from rl_hanabi.training.data_collection import (
    ReplayBuffer,
    GameSequenceDataset,
)
from rl_hanabi.training.trainer import (
    HanabiTrainer,
    log_game_metrics,
    init_wandb,
)


def move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)

def run_training(config: Dict[str, Dict[str, Any]]):
    """Main training loop."""
    
    # Set random seeds
    seed = config["default"].get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Device setup
    device_name = config["default"].get("device", "auto")
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    print(f"Using device: {device}")

    model_cfg = config["model"]
    training_cfg = config["training"]
    selfplay_cfg = config["selfplay"]
    exploration_cfg = config["exploration"]
    logging_cfg = config["logging"]
    wandb_cfg = config["wandb"]
    
    token_config = TokenizationConfig(
        num_colors=model_cfg["max_colors"],
        num_ranks=model_cfg["max_ranks"],
        hand_size=model_cfg["max_hand_size"],
        num_players=model_cfg["max_players"],
        max_info_tokens=model_cfg["max_info_tokens"],
        max_life_tokens=model_cfg["max_life_tokens"],
    )

    decoder_config = ActionDecoderConfig(
        num_colors=model_cfg["max_colors"],
        num_ranks=model_cfg["max_ranks"],
        max_cards=token_config.total_cards,
        hand_size=model_cfg["max_hand_size"],
        num_players=model_cfg["max_players"],
        num_heads=model_cfg["num_heads"],
        num_layers=model_cfg["num_layers"],
        d_model=model_cfg["d_model"],
    )
    
    # Create model
    model = ActionDecoder(
        config=decoder_config,
        token_config=token_config,
    )
    
    # Load checkpoint if specified
    checkpoint_path = config["default"].get("checkpoint")
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint from {checkpoint_path}")
    
    

    
    # Simulation configuration
    simulation_config = {
        "temperature": exploration_cfg["temperature"],
    }
    
    # Full configuration for WandB
    full_config = {
        **config["default"],
        **model_cfg,
        **training_cfg,
        **selfplay_cfg,
        **config["game_config"],
        **exploration_cfg,
        **logging_cfg,
    }
    
    # Initialize WandB
    use_wandb = bool(wandb_cfg.get("enabled", False))
    if use_wandb:
        run = init_wandb(
            project_name=wandb_cfg.get("project", "hanabi-selfplay"),
            config=full_config,
            run_name=wandb_cfg.get("run_name"),
            tags=wandb_cfg.get("tags", "").split(",") if wandb_cfg.get("tags") else None,
        )
        wandb.watch(model, log="all", log_freq=100)
    
    # Create directories
    checkpoint_dir = Path(logging_cfg.get("checkpoint_dir", "checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    buffer_dir = Path(logging_cfg["buffer_dir"]) if logging_cfg.get("buffer_dir") else None
    
    # Create replay buffer
    buffer = ReplayBuffer(
        max_size=selfplay_cfg["buffer_size"],
        save_dir=buffer_dir,
    )
    
    tokenizer = HLETokenizer(token_config)

    # Create trainer
    trainer = HanabiTrainer(
        model=model,
        buffer=buffer,
        device=device,
        config=training_cfg,
        tokenizer=tokenizer,
        checkpoint_dir=checkpoint_dir,
    )

    simulator = GameSimulator(
        model=model,
        tokenizer=tokenizer,
        device=device,
        temperature=simulation_config.get("temperature", 1.0),
    )

    batch_size = training_cfg["batch_size"]
    dataset = GameSequenceDataset(
        buffer=buffer,
        token_config=token_config,
        batch_size=batch_size,
        shuffle_games=True,
        device=device,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=0,
        pin_memory=False,
    )
    
    print("Using single-process mode for game simulation")

    game_config = build_game_config(config)
    
    # Main training loop
    for iteration in range(selfplay_cfg["num_iterations"]):
        print(f"\n{'='*60}")
        print(f"Iteration {iteration + 1}/{selfplay_cfg['num_iterations']}")
        print(f"{'='*60}")
        
        # === Phase 1: Self-play data collection ===
        print(f"\nPhase 1: Collecting {selfplay_cfg['games_per_iteration']} games...")
        
        start_time = time.time()

        if device.type == "cuda":
            model.to("cpu")
            move_optimizer_state(trainer.optimizer, torch.device("cpu"))
            torch.cuda.empty_cache()

        model.eval()
        collected = 0
        for _ in range(selfplay_cfg["games_per_iteration"]):
            result = simulator.simulate_game(config=game_config)
            buffer.add_game_result(result)
            collected += 1
            if _ % 10 == 0:
                print(f"Collected {_} games...")
        model.train()

        simulator.clear_player_models()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            model.to(device)
            move_optimizer_state(trainer.optimizer, device)

        collection_time = time.time() - start_time
        print(f"Collected {collected} games in {collection_time:.2f}s")
        
        # Log game metrics
        buffer_stats = buffer.get_statistics()
        buffer_stats_str = json.dumps(buffer_stats, indent=4)
        print("Buffer Stats: ")
        print(buffer_stats_str)
        
        if use_wandb:
            log_game_metrics(buffer_stats, use_wandb=True)
            wandb.log({
                "iteration": iteration + 1,
                "collection_time": collection_time,
            })
        
        # === Phase 2: Training ===
        print(f"\nPhase 2: Training for {selfplay_cfg['train_steps_per_iteration']} steps...")
        
        num_games = buffer.num_games()
        if num_games == 0:
            print("Not enough games in buffer (0), skipping training")
            continue
        
        train_start = time.time()
        steps_done = 0
        
        for batch_idx, batch in enumerate(dataloader):
            if steps_done >= selfplay_cfg["train_steps_per_iteration"]:
                break
            
            metrics = trainer.train_step(batch)
            steps_done += 1
            
            if steps_done % logging_cfg["log_interval"] == 0:
                print(f"  Step {steps_done}/{selfplay_cfg['train_steps_per_iteration']}, Loss: {metrics['total_loss']:.4f}, "
                      f"Value Loss: {metrics['value_loss']:.4f}, Action Loss: {metrics['action_loss']:.4f}, "
                      f"Mean Reward: {metrics['mean_reward']:.4f}, LR: {metrics['learning_rate']:.6f}")
                
                if use_wandb:
                    wandb.log({
                        "train/step": trainer.global_step,
                        "train/loss": metrics["total_loss"],
                        "train/action_loss": metrics["action_loss"],
                        "train/value_loss": metrics["value_loss"],
                        "train/mean_reward": metrics["mean_reward"],
                        "train/learning_rate": metrics["learning_rate"],
                    })
        
        train_time = time.time() - train_start
        print(f"Training completed in {train_time:.2f}s")


        # === Phase 3: Checkpointing ===
        if (iteration + 1) % logging_cfg["save_interval"] == 0:
            checkpoint_path = trainer.save_checkpoint(
                filename=f"checkpoint_iter_{iteration + 1}.pt",
                extra_data={
                    "iteration": iteration + 1,
                    "buffer_stats": buffer_stats,
                },
            )
            
            if use_wandb:
                wandb.save(str(checkpoint_path))
        
        # Save latest checkpoint
        trainer.save_checkpoint(filename="checkpoint_latest.pt")
        
        # Clear game results to save memory (keep transitions)
        buffer.clear_game_results()
        
        # Update exploration parameters (decay temperature)
        if exploration_cfg["temperature_decay"] > 0:
            old_temp = simulation_config["temperature"]
            simulation_config["temperature"] *= (1 - exploration_cfg["temperature_decay"])
            simulation_config["temperature"] = max(exploration_cfg["min_temperature"], simulation_config["temperature"])
            print(f"Temperature: {old_temp:.4f} -> {simulation_config['temperature']:.4f}")
    
    print("\nTraining complete!")
    
    # Final checkpoint
    trainer.save_checkpoint(filename="checkpoint_final.pt")
    
    if use_wandb:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Hanabi Self-Play Training")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).with_name("config.toml")),
        help="Path to TOML configuration",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        help="Preset name from config.toml (optional)",
    )

    args = parser.parse_args()
    config = load_config(Path(args.config), args.preset)
    run_training(config)


if __name__ == "__main__":
    main()
