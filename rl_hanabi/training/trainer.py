"""
Training loop for Hanabi self-play with tokenized inputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import wandb

from rl_hanabi.model import ActionDecoder
from rl_hanabi.model import HLETokenizer
from rl_hanabi.training import ReplayBuffer



class HanabiTrainer:
    """Trainer for the ActionDecoder using final-score policy gradients."""

    def __init__(
        self,
        model: ActionDecoder,
        buffer: ReplayBuffer,
        device: torch.device,
        config: Dict[str, Any],
        tokenizer: HLETokenizer,
        checkpoint_dir: Optional[Path] = None,
    ):
        self.model = model.to(device)
        self.buffer = buffer
        self.device = device
        self.config = config
        self.tokenizer = tokenizer
        self.token_config = tokenizer.config
        self.checkpoint_dir = checkpoint_dir
        self.c = config.get("critic_loss_weight", 1.0)
        self.c_e = config.get("entropy_loss_weight", 0.0)

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
            "value_loss": [],
            "action_loss": [],
            "entropy_loss": [],
            "mean_reward": [],
            "mean_advantage": [],
            "mean_entropy": [],
        }


    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        
        tokens = batch["tokens"]
        legal_moves_mask = batch["legal_moves_mask"]
        chosen_action_idx = batch["chosen_action_idx"]
        advantage = batch["advantage"]
        returns = batch["returns"]
        current_player = batch["current_player"]

        card_action_logits, value = self.model(tokens)
        action_logits = self.tokenizer.action_logits_from_model(card_action_logits, tokens, current_player)

        masked_logits = action_logits.masked_fill(~legal_moves_mask, -1e9)
        log_probs = F.log_softmax(masked_logits, dim=-1)
        probs = log_probs.exp()
        chosen_log_prob = log_probs.gather(1, chosen_action_idx.unsqueeze(1)).squeeze(1)

        value_1d = value.squeeze(-1)
        returns = returns.reshape(value_1d.shape)
        advantage = advantage.reshape(value_1d.shape)
        actor_loss = -(advantage.detach() * chosen_log_prob).mean()
        critic_loss = F.mse_loss(value_1d, returns)
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        entropy_loss = -self.c_e * entropy
        total_loss = actor_loss + self.c * critic_loss + entropy_loss


        metrics = {
            "total_loss": total_loss.item(),
            "action_loss": actor_loss.item(),
            "value_loss": critic_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "mean_reward": returns.mean().item(),
            "mean_advantage": advantage.mean().item(),
            "mean_entropy": entropy.item(),
        }

        return total_loss, metrics
    

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform a single training step."""
        self.optimizer.zero_grad()

        reset_mask = batch.get("reset_mask")
        if reset_mask is not None:
            self.model.reset_state(reset_mask.to(self.device))
        
        loss, metrics = self.compute_loss(batch)
        
        loss.backward()
        
        # Gradient clipping
        # max_grad_norm = self.config.get("max_grad_norm", 1.0)
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
        
        self.optimizer.step()
        self.scheduler.step()
        
        self.global_step += 1
        
        # Track metrics
        for key, value in metrics.items():
            self.train_metrics[key].append(value)
        
        metrics["learning_rate"] = self.optimizer.param_groups[0]["lr"]
        metrics["global_step"] = self.global_step
        
        return metrics
    

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
    
    wandb.log(metrics)


def init_wandb(
    project_name: str,
    config: Dict[str, Any],
    run_name: Optional[str] = None,
    dir_path = None,
    tags: Optional[List[str]] = None,
) -> wandb.Run: 
    """Initialize WandB run."""
    run = wandb.init(
        entity="immatakuete-ostfalia-university-of-applied-sciences",
        project=project_name,
        name=run_name,
        dir=dir_path,
        config=config,
        tags=tags or [],
        save_code=True,
    )
    
    # Log model architecture
    wandb.config.update({
        "model/architecture": "ActionDecoder",
    })
    
    return run
