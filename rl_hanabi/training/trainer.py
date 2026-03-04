"""
Training loop for Hanabi self-play with tokenized inputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

import wandb

from rl_hanabi.model.action_decoder import ActionDecoder
from rl_hanabi.model.tokenizer import TokenizationConfig
from rl_hanabi.training.data_collection import ReplayBuffer
from rl_hanabi.training.token_utils import build_action_logits_from_tokens


class HanabiTrainer:
    """Trainer for the ActionDecoder using final-score policy gradients."""

    def __init__(
        self,
        model: ActionDecoder,
        buffer: ReplayBuffer,
        device: torch.device,
        config: Dict[str, Any],
        token_config: TokenizationConfig,
        checkpoint_dir: Optional[Path] = None,
    ):
        self.model = model.to(device)
        self.buffer = buffer
        self.device = device
        self.config = config
        self.token_config = token_config
        self.checkpoint_dir = checkpoint_dir

        if checkpoint_dir:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = AdamW(
            model.parameters(),
            lr=config.get("learning_rate", 1e-4),
            weight_decay=config.get("weight_decay", 0.01),
            betas=(0.9, 0.999),
        )

        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config.get("scheduler_t0", 1000),
            T_mult=config.get("scheduler_t_mult", 2),
            eta_min=config.get("min_lr", 1e-6),
        )

        self.global_step = 0
        self.epoch = 0
        self.best_loss = float("inf")

        self.train_metrics: Dict[str, List[float]] = {
            "total_loss": [],
            "action_accuracy": [],
            "mean_reward": [],
            "mean_advantage": [],
        }

    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        tokens = batch["tokens"].to(self.device)
        legal_moves_mask = batch["legal_moves_mask"].to(self.device)
        chosen_action_idx = batch["chosen_action_idx"].to(self.device)
        reward = batch["reward"].to(self.device)
        current_player = batch["current_player"].to(self.device)
        num_players = batch["num_players"].to(self.device)
        num_colors = batch["num_colors"].to(self.device)
        num_ranks = batch["num_ranks"].to(self.device)
        hand_size = batch["hand_size"].to(self.device)

        card_action_logits = self.model(tokens)
        action_logits = build_action_logits_from_tokens(
            card_action_logits=card_action_logits,
            tokens=tokens,
            current_player=current_player,
            num_players=num_players,
            num_colors=num_colors,
            num_ranks=num_ranks,
            hand_size=hand_size,
            token_config=self.token_config,
        )

        masked_logits = action_logits.masked_fill(~legal_moves_mask, -1e9)
        log_probs = F.log_softmax(masked_logits, dim=-1)
        chosen_log_prob = log_probs.gather(1, chosen_action_idx.unsqueeze(1)).squeeze(1)

        baseline = reward.mean()
        advantage = reward - baseline
        total_loss = -(advantage.detach() * chosen_log_prob).mean()

        with torch.no_grad():
            action_preds = masked_logits.argmax(dim=-1)
            action_acc = (action_preds == chosen_action_idx).float().mean()

        metrics = {
            "total_loss": total_loss.item(),
            "action_accuracy": action_acc.item(),
            "mean_reward": reward.mean().item(),
            "mean_advantage": advantage.mean().item(),
        }

        return total_loss, metrics
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform a single training step."""
        self.model.train()
        self.optimizer.zero_grad()
        
        loss, metrics = self.compute_loss(batch)
        
        loss.backward()
        
        # Gradient clipping
        max_grad_norm = self.config.get("max_grad_norm", 1.0)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
        
        self.optimizer.step()
        self.scheduler.step()
        
        self.global_step += 1
        
        # Track metrics
        for key, value in metrics.items():
            self.train_metrics[key].append(value)
        
        metrics["learning_rate"] = self.optimizer.param_groups[0]["lr"]
        metrics["global_step"] = self.global_step
        
        return metrics
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        log_interval: int = 100,
        use_wandb: bool = True,
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_metrics = {key: [] for key in self.train_metrics.keys()}
        
        for batch_idx, batch in enumerate(dataloader):
            metrics = self.train_step(batch)
            
            for key in epoch_metrics:
                if key in metrics:
                    epoch_metrics[key].append(metrics[key])
            
            if batch_idx % log_interval == 0:
                avg_loss = np.mean(epoch_metrics["total_loss"][-log_interval:])
                print(f"  Step {self.global_step}, Batch {batch_idx}, Loss: {avg_loss:.4f}")
                
                if use_wandb:
                    wandb.log({
                        "train/step": self.global_step,
                        "train/loss": metrics["total_loss"],
                        "train/action_accuracy": metrics["action_accuracy"],
                        "train/mean_reward": metrics["mean_reward"],
                        "train/mean_advantage": metrics["mean_advantage"],
                        "train/learning_rate": metrics["learning_rate"],
                    })
        
        # Compute epoch averages
        avg_metrics = {
            f"epoch_{key}": np.mean(values) 
            for key, values in epoch_metrics.items() 
            if values
        }
        
        self.epoch += 1
        avg_metrics["epoch"] = self.epoch # type: ignore
        
        return avg_metrics  # type: ignore
    
    def validate(
        self,
        dataloader: DataLoader,
        use_wandb: bool = True,
    ) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()
        val_metrics = {key: [] for key in self.train_metrics.keys()}
        
        with torch.no_grad():
            for batch in dataloader:
                _, metrics = self.compute_loss(batch)
                for key in val_metrics:
                    if key in metrics:
                        val_metrics[key].append(metrics[key])
        
        # Compute averages
        avg_metrics = {
            f"val_{key}": np.mean(values)
            for key, values in val_metrics.items()
            if values
        }
        
        if use_wandb:
            wandb.log(avg_metrics)
        
        return avg_metrics  # type: ignore
    
    def save_checkpoint(self, filename: str, extra_data: Optional[Dict] = None) -> Path:
        """Save a training checkpoint."""
        if self.checkpoint_dir is None:
            raise ValueError("No checkpoint directory specified")
        
        filepath = self.checkpoint_dir / filename
        
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_loss": self.best_loss,
            "config": self.config,
        }
        
        if extra_data:
            checkpoint.update(extra_data)
        
        torch.save(checkpoint, filepath)
        print(f"Saved checkpoint to {filepath}")
        
        return filepath
    
    def load_checkpoint(self, filepath: Path) -> Dict:
        """Load a training checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.epoch = checkpoint["epoch"]
        self.best_loss = checkpoint.get("best_loss", float('inf'))
        
        print(f"Loaded checkpoint from {filepath}")
        print(f"  Epoch: {self.epoch}, Global Step: {self.global_step}")
        
        return checkpoint


def log_game_metrics(
    buffer_stats: Dict[str, float],
    game_configs_used: Dict[str, int],
    use_wandb: bool = True,
) -> None:
    """Log game statistics to WandB."""
    if not use_wandb:
        return
    
    metrics = {
        "games/buffer_size": buffer_stats.get("buffer_size", 0),
        "games/total_transitions": buffer_stats.get("total_added", 0),
        "games/num_games": buffer_stats.get("num_games", 0),
        "games/avg_score": buffer_stats.get("avg_score", 0),
        "games/max_score": buffer_stats.get("max_score", 0),
        "games/avg_normalized_score": buffer_stats.get("avg_normalized_score", 0),
        "games/avg_turns": buffer_stats.get("avg_turns", 0),
    }
    
    # Log game config distribution
    for config_key, count in game_configs_used.items():
        metrics[f"games/config_{config_key}"] = count
    
    wandb.log(metrics)


def init_wandb(
    project_name: str,
    config: Dict[str, Any],
    run_name: Optional[str] = None,
    tags: Optional[List[str]] = None,
) -> wandb.Run: 
    """Initialize WandB run."""
    run = wandb.init(
        project=project_name,
        name=run_name,
        config=config,
        tags=tags or [],
        save_code=True,
    )
    
    # Log model architecture
    wandb.config.update({
        "model/architecture": "ActionDecoder",
    })
    
    return run
