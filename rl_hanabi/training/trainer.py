"""
Training loop for Hanabi self-play with WandB integration.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

import wandb

from rl_hanabi.model.belief_model import ActionDecoder
from rl_hanabi.training.data_collection import ReplayBuffer, create_dataloader


class HanabiTrainer:
    """
    Trainer for the ActionDecoder model using self-play data.
    Integrates with WandB for experiment tracking.
    """
    
    def __init__(
        self,
        model: ActionDecoder,
        buffer: ReplayBuffer,
        device: torch.device,
        config: Dict[str, Any],
        checkpoint_dir: Optional[Path] = None,
    ):
        self.model = model.to(device)
        self.buffer = buffer
        self.device = device
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        
        if checkpoint_dir:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.get("learning_rate", 1e-4),
            weight_decay=config.get("weight_decay", 0.01),
            betas=(0.9, 0.999),
        )
        
        # Learning rate scheduler
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config.get("scheduler_t0", 1000),
            T_mult=config.get("scheduler_t_mult", 2),
            eta_min=config.get("min_lr", 1e-6),
        )
        
        # Loss weights
        self.color_loss_weight = config.get("color_loss_weight", 1.0)
        self.rank_loss_weight = config.get("rank_loss_weight", 1.0)
        self.action_loss_weight = config.get("action_loss_weight", 1.0)
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        
        # Metrics tracking
        self.train_metrics: Dict[str, List[float]] = {
            "total_loss": [],
            "color_loss": [],
            "rank_loss": [],
            "action_loss": [],
            "color_accuracy": [],
            "rank_accuracy": [],
            "action_accuracy": [],
        }
    
    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute losses for a batch of transitions."""
        
        # Move batch to device
        slot_beliefs = batch["slot_beliefs"].to(self.device)
        affected_mask = batch["affected_mask"].to(self.device)
        move_target_player = batch["move_target_player"].to(self.device)
        acting_player = batch["acting_player"].to(self.device)
        action = batch["action"].to(self.device)
        fireworks = batch["fireworks"].to(self.device)
        discard_pile = batch["discard_pile"].to(self.device)
        true_colors = batch["true_colors"].to(self.device)
        true_ranks = batch["true_ranks"].to(self.device)
        chosen_action_idx = batch["chosen_action_idx"].to(self.device)
        legal_moves_mask = batch["legal_moves_mask"].to(self.device)
        hand_size = batch["hand_size"].to(self.device)
        
        # Forward pass
        color_logits, rank_logits, action_logits = self.model(
            slot_beliefs=slot_beliefs,
            affected_mask=affected_mask,
            move_target_player=move_target_player,
            acting_player=acting_player,
            action=action,
            fireworks=fireworks,
            discard_pile=discard_pile,
        )
        
        B, H, C = color_logits.shape
        _, _, R = rank_logits.shape
        
        # Create valid slot mask (where true colors/ranks are not -1)
        valid_mask = (true_colors >= 0) & (true_ranks >= 0)  # [B, H]
        
        # Color prediction loss
        color_logits_flat = color_logits.view(-1, C)  # [B*H, C]
        true_colors_flat = true_colors.view(-1)  # [B*H]
        valid_mask_flat = valid_mask.view(-1)  # [B*H]
        
        # Mask invalid targets
        true_colors_flat_masked = true_colors_flat.clone()
        true_colors_flat_masked[~valid_mask_flat] = 0  # Placeholder for invalid
        
        color_loss = F.cross_entropy(
            color_logits_flat,
            true_colors_flat_masked,
            reduction='none'
        )
        color_loss = (color_loss * valid_mask_flat.float()).sum() / (valid_mask_flat.sum() + 1e-8)
        
        # Rank prediction loss
        rank_logits_flat = rank_logits.view(-1, R)  # [B*H, R]
        true_ranks_flat = true_ranks.view(-1)  # [B*H]
        
        true_ranks_flat_masked = true_ranks_flat.clone()
        true_ranks_flat_masked[~valid_mask_flat] = 0
        
        rank_loss = F.cross_entropy(
            rank_logits_flat,
            true_ranks_flat_masked,
            reduction='none'
        )
        rank_loss = (rank_loss * valid_mask_flat.float()).sum() / (valid_mask_flat.sum() + 1e-8)
        
        # Action prediction loss
        # Mask illegal actions with large negative values
        action_logits_masked = action_logits.clone()
        action_logits_masked[~legal_moves_mask.bool()] = -1e9
        
        # Ensure chosen_action_idx is within bounds
        max_action_idx = action_logits.size(-1) - 1
        chosen_action_idx_clamped = chosen_action_idx.clamp(0, max_action_idx)
        
        action_loss = F.cross_entropy(
            action_logits_masked,
            chosen_action_idx_clamped,
            reduction='mean'
        )
        
        # Total loss
        total_loss = (
            self.color_loss_weight * color_loss +
            self.rank_loss_weight * rank_loss +
            self.action_loss_weight * action_loss
        )
        
        # Compute accuracies
        with torch.no_grad():
            color_preds = color_logits_flat.argmax(dim=-1)
            color_correct = (color_preds == true_colors_flat) & valid_mask_flat
            color_acc = color_correct.sum().float() / (valid_mask_flat.sum() + 1e-8)
            
            rank_preds = rank_logits_flat.argmax(dim=-1)
            rank_correct = (rank_preds == true_ranks_flat) & valid_mask_flat
            rank_acc = rank_correct.sum().float() / (valid_mask_flat.sum() + 1e-8)
            
            # For action accuracy, use masked logits to get legal predictions
            # but compute accuracy to see if model predicts the chosen legal action
            action_preds = action_logits_masked.argmax(dim=-1)
            action_correct = (action_preds == chosen_action_idx_clamped)
            action_acc = action_correct.float().mean()
        
        metrics = {
            "total_loss": total_loss.item(),
            "color_loss": color_loss.item(),
            "rank_loss": rank_loss.item(),
            "action_loss": action_loss.item(),
            "color_accuracy": color_acc.item(),
            "rank_accuracy": rank_acc.item(),
            "action_accuracy": action_acc.item(),
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
                        "train/color_loss": metrics["color_loss"],
                        "train/rank_loss": metrics["rank_loss"],
                        "train/action_loss": metrics["action_loss"],
                        "train/color_accuracy": metrics["color_accuracy"],
                        "train/rank_accuracy": metrics["rank_accuracy"],
                        "train/action_accuracy": metrics["action_accuracy"],
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
