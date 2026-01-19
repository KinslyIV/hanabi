"""
GPU Server - Runs on the laptop with GPU.
Receives training batches over the network, performs GPU computations, and returns results.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import pickle
import signal
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from rl_hanabi.training.distributed.protocol import TrainingRequest, TrainingResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('gpu_server')


class GPUTrainer:
    """Handles GPU-based training operations."""
    
    def __init__(
        self,
        model_config: Dict[str, Any],
        training_config: Dict[str, Any],
        device: torch.device,
        checkpoint_dir: Optional[Path] = None,
    ):
        self.device = device
        self.model_config = model_config
        self.training_config = training_config
        self.checkpoint_dir = checkpoint_dir
        
        if checkpoint_dir:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Import here to avoid circular imports
        from rl_hanabi.model.belief_model import ActionDecoder
        
        # Create model
        self.model = ActionDecoder(
            max_num_colors=model_config["max_num_colors"],
            max_num_ranks=model_config["max_num_ranks"],
            max_hand_size=model_config["max_hand_size"],
            max_num_players=model_config["max_num_players"],
            num_heads=model_config.get("num_heads", 4),
            num_layers=model_config.get("num_layers", 4),
            d_model=model_config.get("d_model", 128),
            action_dim=model_config.get("action_dim", 4),
        )
        self.model.to(device)
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=training_config.get("learning_rate", 1e-4),
            weight_decay=training_config.get("weight_decay", 0.01),
            betas=(0.9, 0.999),
        )
        
        # Learning rate scheduler
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=training_config.get("scheduler_t0", 1000),
            T_mult=training_config.get("scheduler_t_mult", 1.2),
            eta_min=training_config.get("min_lr", 1e-6),
        )
        
        # Loss weights
        self.color_loss_weight = training_config.get("color_loss_weight", 1.0)
        self.rank_loss_weight = training_config.get("rank_loss_weight", 1.0)
        self.action_loss_weight = training_config.get("action_loss_weight", 1.0)
        self.failed_play_penalty = training_config.get("failed_play_penalty", 2.0)
        self.max_grad_norm = training_config.get("max_grad_norm", 1.0)
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        
        logger.info(f"GPUTrainer initialized on {device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        """Compute losses for a batch of transitions.
        
        Action loss uses reward-weighted cross-entropy:
        - Base reward is the normalized final score of the game
        - Failed play moves receive a heavy penalty
        """
        
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
        reward = batch["reward"].to(self.device)  # [B] normalized final score
        failed_play = batch["failed_play"].to(self.device)  # [B] bool
        
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
        
        # Create valid slot mask
        valid_mask = (true_colors >= 0) & (true_ranks >= 0)
        
        # Color prediction loss
        color_logits_flat = color_logits.view(-1, C)
        true_colors_flat = true_colors.view(-1)
        valid_mask_flat = valid_mask.view(-1)
        
        true_colors_flat_masked = true_colors_flat.clone()
        true_colors_flat_masked[~valid_mask_flat] = 0
        
        color_loss = F.cross_entropy(
            color_logits_flat,
            true_colors_flat_masked,
            reduction='none'
        )
        color_loss = (color_loss * valid_mask_flat.float()).sum() / (valid_mask_flat.sum() + 1e-8)
        
        # Rank prediction loss
        rank_logits_flat = rank_logits.view(-1, R)
        true_ranks_flat = true_ranks.view(-1)
        
        true_ranks_flat_masked = true_ranks_flat.clone()
        true_ranks_flat_masked[~valid_mask_flat] = 0
        
        rank_loss = F.cross_entropy(
            rank_logits_flat,
            true_ranks_flat_masked,
            reduction='none'
        )
        rank_loss = (rank_loss * valid_mask_flat.float()).sum() / (valid_mask_flat.sum() + 1e-8)
        
        # Action loss - Policy Gradient (REINFORCE with baseline)
        # Mask illegal actions
        action_logits_masked = action_logits.clone()
        action_logits_masked[~legal_moves_mask.bool()] = -1e9
        
        max_action_idx = action_logits.size(-1) - 1
        chosen_action_idx_clamped = chosen_action_idx.clamp(0, max_action_idx)
        
        # Get log probability of the chosen action
        log_probs = F.log_softmax(action_logits_masked, dim=-1)
        chosen_log_prob = log_probs.gather(1, chosen_action_idx_clamped.unsqueeze(1)).squeeze(1)
        
        # Compute advantage with baseline (mean reward in batch)
        # This reduces variance in the gradient estimates
        baseline = reward.mean()
        advantage = reward - baseline
        
        # Heavy penalty for failed plays - subtract from advantage
        # Failed plays get negative advantage, discouraging them
        advantage = advantage - self.failed_play_penalty * failed_play.float()
        
        # Policy gradient loss: -log π(a|s) * advantage
        # When advantage > 0: we want to increase log_prob, so minimize -log_prob * pos = negative loss contribution
        # When advantage < 0: we want to decrease log_prob, so minimize -log_prob * neg = positive loss contribution
        # .detach() on advantage to not backprop through the reward/baseline
        action_loss = -(chosen_log_prob * advantage.detach()).mean()
        
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
            
            # Additional metrics
            avg_reward = reward.mean()
            failed_play_rate = failed_play.float().mean()
            avg_advantage = advantage.mean()
        
        metrics = {
            "total_loss": total_loss.item(),
            "color_loss": color_loss.item(),
            "rank_loss": rank_loss.item(),
            "action_loss": action_loss.item(),
            "color_accuracy": color_acc.item(),
            "rank_accuracy": rank_acc.item(),
            "action_accuracy": action_acc.item(),
            "avg_reward": avg_reward.item(),
            "avg_advantage": avg_advantage.item(),
            "failed_play_rate": failed_play_rate.item(),
        }
        
        return total_loss, metrics
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform a single training step."""
        self.model.train()
        self.optimizer.zero_grad()
        
        loss, metrics = self.compute_loss(batch)
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        
        self.optimizer.step()
        self.scheduler.step()
        
        self.global_step += 1
        
        metrics["learning_rate"] = self.optimizer.param_groups[0]["lr"]
        metrics["global_step"] = self.global_step
        
        return metrics
    
    def get_state_dict(self) -> Dict[str, Any]:
        """Get complete state for checkpointing."""
        return {
            "model_state_dict": {k: v.cpu() for k, v in self.model.state_dict().items()},
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_loss": self.best_loss,
            "model_config": self.model_config,
            "training_config": self.training_config,
        }
    
    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load complete state from checkpoint."""
        self.model.load_state_dict(state["model_state_dict"])
        self.model.to(self.device)
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.scheduler.load_state_dict(state["scheduler_state_dict"])
        self.global_step = state["global_step"]
        self.epoch = state.get("epoch", 0)
        self.best_loss = state.get("best_loss", float('inf'))
        logger.info(f"Loaded state at step {self.global_step}")
    
    def get_model_weights(self) -> Dict[str, torch.Tensor]:
        """Get model weights for workers."""
        return {k: v.cpu() for k, v in self.model.state_dict().items()}
    
    def save_checkpoint(self, filename: str) -> Path:
        """Save checkpoint to disk."""
        if self.checkpoint_dir is None:
            raise ValueError("No checkpoint directory specified")
        
        filepath = self.checkpoint_dir / filename
        torch.save(self.get_state_dict(), filepath)
        logger.info(f"Saved checkpoint to {filepath}")
        return filepath
    
    def load_checkpoint(self, filepath: Path) -> None:
        """Load checkpoint from disk."""
        state = torch.load(filepath, map_location=self.device)
        self.load_state_dict(state)
        logger.info(f"Loaded checkpoint from {filepath}")
    
    def load_model_weights_only(self, filepath: Path) -> bool:
        """Load only model weights from checkpoint (for transfer learning/pretrained models).
        
        Returns:
            True if weights were loaded, False if file not found (fresh model used).
        """
        if not filepath.exists():
            logger.warning(f"Pretrained model not found at {filepath}, using fresh model")
            # Reset training state for fresh start
            self.global_step = 0
            self.epoch = 0
            self.best_loss = float('inf')
            return False
        
        try:
            state = torch.load(filepath, map_location=self.device)
            if "model_state_dict" in state:
                self.model.load_state_dict(state["model_state_dict"])
            else:
                # Assume it's just model weights
                self.model.load_state_dict(state)
            self.model.to(self.device)
            # Reset training state for fresh start with pretrained weights
            self.global_step = 0
            self.epoch = 0
            self.best_loss = float('inf')
            logger.info(f"Loaded pretrained model weights from {filepath} (optimizer/scheduler reset)")
            return True
        except Exception as e:
            logger.warning(f"Failed to load pretrained model from {filepath}: {e}, using fresh model")
            self.global_step = 0
            self.epoch = 0
            self.best_loss = float('inf')
            return False
    
    def find_latest_checkpoint(self) -> Optional[Path]:
        """Find the most recent checkpoint in the checkpoint directory."""
        if not self.checkpoint_dir or not self.checkpoint_dir.exists():
            return None
        
        # Look for checkpoint files
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_*.pt"))
        if not checkpoints:
            return None
        
        # Prefer checkpoint_latest.pt if it exists
        latest = self.checkpoint_dir / "checkpoint_latest.pt"
        if latest.exists():
            return latest
        
        # Otherwise find most recent by modification time
        return max(checkpoints, key=lambda p: p.stat().st_mtime)
    
    def auto_resume(self) -> bool:
        """Attempt to auto-resume from the latest checkpoint. Returns True if resumed."""
        latest = self.find_latest_checkpoint()
        if latest:
            try:
                self.load_checkpoint(latest)
                logger.info(f"Auto-resumed from {latest} at step {self.global_step}")
                return True
            except Exception as e:
                logger.warning(f"Failed to auto-resume from {latest}: {e}")
        return False
    
    def reset_model(self, delete_checkpoints: bool = False) -> None:
        """Reset the model to fresh random weights.
        
        Args:
            delete_checkpoints: If True, also delete all checkpoint files.
        """
        # Re-initialize model weights
        from rl_hanabi.model.belief_model import ActionDecoder
        
        self.model = ActionDecoder(
            max_num_colors=self.model_config["max_num_colors"],
            max_num_ranks=self.model_config["max_num_ranks"],
            max_hand_size=self.model_config["max_hand_size"],
            max_num_players=self.model_config["max_num_players"],
            num_heads=self.model_config.get("num_heads", 4),
            num_layers=self.model_config.get("num_layers", 4),
            d_model=self.model_config.get("d_model", 128),
            action_dim=self.model_config.get("action_dim", 4),
        )
        self.model.to(self.device)
        
        # Re-initialize optimizer with new model parameters
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.training_config.get("learning_rate", 1e-4),
            weight_decay=self.training_config.get("weight_decay", 0.01),
            betas=(0.9, 0.999),
        )
        
        # Re-initialize scheduler
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=self.training_config.get("scheduler_t0", 1000),
            T_mult=self.training_config.get("scheduler_t_mult", 2),
            eta_min=self.training_config.get("min_lr", 1e-6),
        )
        
        # Reset training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        
        # Delete checkpoints if requested
        if delete_checkpoints and self.checkpoint_dir and self.checkpoint_dir.exists():
            for checkpoint_file in self.checkpoint_dir.glob("checkpoint_*.pt"):
                try:
                    checkpoint_file.unlink()
                    logger.info(f"Deleted checkpoint: {checkpoint_file}")
                except Exception as e:
                    logger.warning(f"Failed to delete {checkpoint_file}: {e}")
            # Also delete best_model.pt if it exists
            best_model = self.checkpoint_dir / "best_model.pt"
            if best_model.exists():
                try:
                    best_model.unlink()
                    logger.info(f"Deleted best model: {best_model}")
                except Exception as e:
                    logger.warning(f"Failed to delete {best_model}: {e}")
        
        logger.info("Model reset to fresh random weights")


class GPUServer:
    """Async TCP server that handles training requests from the coordinator."""
    
    def __init__(
        self,
        host: str,
        port: int,
        trainer: GPUTrainer,
        auth_token: Optional[str] = None,
    ):
        self.host = host
        self.port = port
        self.trainer = trainer
        self.auth_token = auth_token
        self.server: Optional[asyncio.Server] = None
        self._shutdown_event = asyncio.Event()
        self._active_connections: set = set()
        
        # Statistics
        self.requests_handled = 0
        self.total_training_time = 0.0
        self.start_time = time.time()
    
    async def _send_response(
        self,
        writer: asyncio.StreamWriter,
        response: TrainingResponse,
    ) -> None:
        """Send a response with length prefix."""
        data = pickle.dumps(response)
        length = struct.pack('>I', len(data))
        writer.write(length + data)
        await writer.drain()
    
    async def _recv_request(
        self,
        reader: asyncio.StreamReader,
    ) -> Optional[TrainingRequest]:
        """Receive a request with length prefix."""
        try:
            length_bytes = await asyncio.wait_for(
                reader.readexactly(4),
                timeout=300  # 5 minute timeout
            )
            length = struct.unpack('>I', length_bytes)[0]
            
            data = await asyncio.wait_for(
                reader.readexactly(length),
                timeout=300
            )
            request = pickle.loads(data)
            # Handle dict requests from clients (convert to TrainingRequest)
            if isinstance(request, dict):
                request = TrainingRequest.model_validate(request)
            return request
        except asyncio.TimeoutError:
            logger.warning("Request timeout")
            return None
        except asyncio.IncompleteReadError:
            return None
    
    async def _handle_request(self, request: TrainingRequest) -> TrainingResponse:
        """Process a single request."""
        try:
            if request.request_type == 'ping':
                return TrainingResponse(
                    success=True,
                    response_type='pong',
                    payload={
                        'global_step': self.trainer.global_step,
                        'requests_handled': self.requests_handled,
                        'uptime': time.time() - self.start_time,
                    }
                )
            
            elif request.request_type == 'train_step':

                if request.payload:
                    batch = request.payload['batch']
                else:
                    return TrainingResponse(
                        success=False,
                        response_type='error',
                        error="No batch provided for train_step"
                    )
                # Convert numpy arrays to tensors
                tensor_batch = {
                    k: torch.from_numpy(v) if hasattr(v, 'numpy') or isinstance(v, bytes) else torch.tensor(v)
                    for k, v in batch.items()
                }
                
                start = time.time()
                metrics = self.trainer.train_step(tensor_batch)
                self.total_training_time += time.time() - start
                
                return TrainingResponse(
                    success=True,
                    response_type='train_step_result',
                    payload={'metrics': metrics}
                )
            
            elif request.request_type == 'get_weights':
                weights = self.trainer.get_model_weights()
                return TrainingResponse(
                    success=True,
                    response_type='weights',
                    payload={'weights': weights}
                )
            
            elif request.request_type == 'get_state':
                state = self.trainer.get_state_dict()
                return TrainingResponse(
                    success=True,
                    response_type='state',
                    payload={'state': state}
                )
            
            elif request.request_type == 'set_state':
                if not request.payload or 'state' not in request.payload:
                    return TrainingResponse(
                        success=False,
                        response_type='error',
                        error="No state provided for set_state"
                    )
                state = request.payload['state']
                self.trainer.load_state_dict(state)
                return TrainingResponse(
                    success=True,
                    response_type='state_set',
                )
            
            elif request.request_type == 'save_checkpoint':
                if not request.payload:
                    filename = 'checkpoint.pt'
                else:
                    filename = request.payload.get('filename', 'checkpoint.pt')
                filepath = self.trainer.save_checkpoint(filename)
                return TrainingResponse(
                    success=True,
                    response_type='checkpoint_saved',
                    payload={'filepath': str(filepath)}
                )
            
            elif request.request_type == 'load_checkpoint':
                if not request.payload or 'filepath' not in request.payload:
                    return TrainingResponse(
                        success=False,
                        response_type='error',
                        error="No filepath provided for load_checkpoint"
                    )
                filepath = Path(request.payload['filepath'])
                self.trainer.load_checkpoint(filepath)
                return TrainingResponse(
                    success=True,
                    response_type='checkpoint_loaded',
                )
            
            elif request.request_type == 'load_pretrained_model':
                if not request.payload or 'filepath' not in request.payload:
                    return TrainingResponse(
                        success=False,
                        response_type='error',
                        error="No filepath provided for load_pretrained_model"
                    )
                filepath = Path(request.payload['filepath'])
                loaded = self.trainer.load_model_weights_only(filepath)
                return TrainingResponse(
                    success=True,
                    response_type='pretrained_model_loaded',
                    payload={
                        'global_step': self.trainer.global_step,
                        'loaded': loaded,
                        'message': 'Loaded pretrained weights' if loaded else 'Model not found, initialized fresh model'
                    }
                )
            
            elif request.request_type == 'get_stats':
                return TrainingResponse(
                    success=True,
                    response_type='stats',
                    payload={
                        'global_step': self.trainer.global_step,
                        'requests_handled': self.requests_handled,
                        'total_training_time': self.total_training_time,
                        'uptime': time.time() - self.start_time,
                        'device': str(self.trainer.device),
                    }
                )
            
            elif request.request_type == 'initialize_training':
                # Coordinator tells GPU server how to initialize
                # mode: 'resume' | 'fresh' | 'pretrained'
                mode = request.payload.get('mode', 'resume') if request.payload else 'resume'
                pretrained_path = request.payload.get('pretrained_path') if request.payload else None
                delete_checkpoints = request.payload.get('delete_checkpoints', False) if request.payload else False
                
                if mode == 'fresh':
                    logger.info(f"Initializing fresh model (delete_checkpoints={delete_checkpoints})")
                    self.trainer.reset_model(delete_checkpoints=delete_checkpoints)
                    return TrainingResponse(
                        success=True,
                        response_type='training_initialized',
                        payload={
                            'mode': 'fresh',
                            'global_step': self.trainer.global_step,
                            'message': 'Initialized with fresh random weights'
                        }
                    )
                
                elif mode == 'pretrained':
                    if not pretrained_path:
                        return TrainingResponse(
                            success=False,
                            response_type='error',
                            error="No pretrained_path provided for pretrained mode"
                        )
                    logger.info(f"Initializing from pretrained model: {pretrained_path}")
                    loaded = self.trainer.load_model_weights_only(Path(pretrained_path))
                    return TrainingResponse(
                        success=True,
                        response_type='training_initialized',
                        payload={
                            'mode': 'pretrained',
                            'loaded': loaded,
                            'global_step': self.trainer.global_step,
                            'message': f"Loaded pretrained weights from {pretrained_path}" if loaded else "Pretrained model not found, using fresh weights"
                        }
                    )
                
                else:  # mode == 'resume'
                    logger.info("Attempting to resume from latest checkpoint")
                    resumed = self.trainer.auto_resume()
                    return TrainingResponse(
                        success=True,
                        response_type='training_initialized',
                        payload={
                            'mode': 'resume',
                            'resumed': resumed,
                            'global_step': self.trainer.global_step,
                            'message': f"Resumed from checkpoint at step {self.trainer.global_step}" if resumed else "No checkpoint found, using current model"
                        }
                    )
            
            elif request.request_type == 'stop_training':
                shutdown_server = request.payload.get('shutdown_server', False) if request.payload else False
                logger.info(f"Stop training request received (shutdown_server={shutdown_server})")
                self._shutdown_event.set()
                
                # Save a final checkpoint before stopping
                try:
                    if self.trainer.checkpoint_dir:
                        self.trainer.save_checkpoint('checkpoint_final.pt')
                        logger.info("Saved final checkpoint before shutdown")
                except Exception as e:
                    logger.warning(f"Failed to save final checkpoint: {e}")
                
                return TrainingResponse(
                    success=True,
                    response_type='training_stopped',
                    payload={
                        'message': 'Training stopped and server shutting down' if shutdown_server else 'Training stopped',
                        'shutdown_server': shutdown_server,
                        'final_step': self.trainer.global_step,
                    }
                )
            
            else:
                return TrainingResponse(
                    success=False,
                    response_type='error',
                    error=f"Unknown request type: {request.request_type}"
                )
        
        except Exception as e:
            logger.exception(f"Error handling request: {e}")
            return TrainingResponse(
                success=False,
                response_type='error',
                error=str(e)
            )
    
    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle a single client connection."""
        addr = writer.get_extra_info('peername')
        logger.info(f"New connection from {addr}")
        self._active_connections.add(writer)
        
        try:
            # Authentication
            if self.auth_token:
                auth_request = await self._recv_request(reader)
                if not auth_request or auth_request.request_type != 'auth':
                    logger.warning(f"Auth failed from {addr}: no auth request")
                    await self._send_response(writer, TrainingResponse(
                        success=False,
                        response_type='auth_failed',
                        error='Authentication required'
                    ))
                    return
                if auth_request.payload is None:
                    logger.warning(f"Auth failed from {addr}: no auth payload")
                    await self._send_response(writer, TrainingResponse(
                        success=False,
                        response_type='auth_failed',
                        error='No authentication payload'
                    ))
                    return
                if auth_request.payload.get('token') != self.auth_token: 
                    logger.warning(f"Auth failed from {addr}: invalid token")
                    await self._send_response(writer, TrainingResponse(
                        success=False,
                        response_type='auth_failed',
                        error='Invalid token'
                    ))
                    return
                
                await self._send_response(writer, TrainingResponse(
                    success=True,
                    response_type='auth_success',
                ))
                logger.info(f"Client {addr} authenticated")
            
            # Request handling loop
            while not self._shutdown_event.is_set():
                request = await self._recv_request(reader)
                if request is None:
                    break
                
                response = await self._handle_request(request)
                await self._send_response(writer, response)
                self.requests_handled += 1
        
        except Exception as e:
            logger.exception(f"Error with client {addr}: {e}")
        
        finally:
            self._active_connections.discard(writer)
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
            logger.info(f"Connection closed from {addr}")
    
    async def start(self) -> None:
        """Start the server."""
        self.server = await asyncio.start_server(
            self._handle_client,
            self.host,
            self.port,
        )
        
        addr = self.server.sockets[0].getsockname()
        logger.info(f"GPU Server listening on {addr}")
        
        async with self.server:
            await self.server.serve_forever()
    
    async def stop(self, save_checkpoint: bool = True) -> None:
        """Stop the server gracefully."""
        logger.info("Shutting down GPU Server...")
        self._shutdown_event.set()
        
        # Save final checkpoint
        if save_checkpoint:
            try:
                if self.trainer.checkpoint_dir:
                    self.trainer.save_checkpoint('checkpoint_final.pt')
                    logger.info(f"Saved final checkpoint at step {self.trainer.global_step}")
            except Exception as e:
                logger.error(f"Failed to save final checkpoint: {e}")
        
        # Close all active connections
        for writer in list(self._active_connections):
            writer.close()
        
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        
        logger.info("GPU Server shutdown complete")
    
    async def save_checkpoint_now(self, filename: str = 'checkpoint_signal.pt') -> None:
        """Save a checkpoint immediately (can be triggered by signal)."""
        try:
            if self.trainer.checkpoint_dir:
                self.trainer.save_checkpoint(filename)
                logger.info(f"Signal-triggered checkpoint saved at step {self.trainer.global_step}")
        except Exception as e:
            logger.error(f"Failed to save signal-triggered checkpoint: {e}")


def main():
    parser = argparse.ArgumentParser(description="GPU Training Server")
    
    # Network settings
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5555, help="Port to listen on")
    parser.add_argument("--auth-token", type=str, default=None, help="Authentication token")
    
    # Device settings
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu, cuda, or auto)")
    
    # Model configuration
    parser.add_argument("--max-players", type=int, default=5, help="Maximum number of players")
    parser.add_argument("--max-colors", type=int, default=5, help="Maximum number of colors")
    parser.add_argument("--max-ranks", type=int, default=5, help="Maximum number of ranks")
    parser.add_argument("--max-hand-size", type=int, default=5, help="Maximum hand size")
    parser.add_argument("--num-heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--d-model", type=int, default=128, help="Model dimension")
    
    # Training configuration
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Max gradient norm")
    parser.add_argument("--scheduler-t0", type=int, default=1000, help="Scheduler T_0")
    parser.add_argument("--min-lr", type=float, default=1e-6, help="Minimum learning rate")
    
    # Loss weights
    parser.add_argument("--color-loss-weight", type=float, default=1.0, help="Color loss weight")
    parser.add_argument("--rank-loss-weight", type=float, default=1.0, help="Rank loss weight")
    parser.add_argument("--action-loss-weight", type=float, default=1.0, help="Action loss weight")
    parser.add_argument("--failed-play-penalty", type=float, default=2.0, 
                       help="Penalty multiplier for failed play moves (bombs)")
    
    # Checkpoint settings
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--load-checkpoint", type=str, default=None, 
                       help="Specific checkpoint to load (full state including optimizer)")
    parser.add_argument("--pretrained-model", type=str, default=None,
                       help="Path to pretrained model (loads weights only, resets optimizer/scheduler)")
    parser.add_argument("--auto-resume", action="store_true", default=True,
                       help="Automatically resume from latest checkpoint if available (default: True)")
    parser.add_argument("--no-auto-resume", action="store_false", dest="auto_resume",
                       help="Disable auto-resume, start with fresh model")
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Configuration
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
    
    training_config = {
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "scheduler_t0": args.scheduler_t0,
        "scheduler_t_mult": 2,
        "min_lr": args.min_lr,
        "color_loss_weight": args.color_loss_weight,
        "rank_loss_weight": args.rank_loss_weight,
        "action_loss_weight": args.action_loss_weight,
        "failed_play_penalty": args.failed_play_penalty,
    }
    
    # Create trainer
    trainer = GPUTrainer(
        model_config=model_config,
        training_config=training_config,
        device=device,
        checkpoint_dir=Path(args.checkpoint_dir),
    )
    
    # Checkpoint loading priority:
    # 1. Explicit --load-checkpoint (full state)
    # 2. Explicit --pretrained-model (weights only, reset optimizer)
    # 3. Auto-resume from latest checkpoint (if enabled)
    # 4. Fresh start
    
    if args.load_checkpoint:
        logger.info(f"Loading explicit checkpoint: {args.load_checkpoint}")
        trainer.load_checkpoint(Path(args.load_checkpoint))
    elif args.pretrained_model:
        logger.info(f"Loading pretrained model weights: {args.pretrained_model}")
        trainer.load_model_weights_only(Path(args.pretrained_model))
    elif args.auto_resume:
        if not trainer.auto_resume():
            logger.info("No checkpoint found for auto-resume, starting fresh")
    else:
        logger.info("Auto-resume disabled, starting with fresh model")
    
    # Create and run server
    server = GPUServer(
        host=args.host,
        port=args.port,
        trainer=trainer,
        auth_token=args.auth_token,
    )
    
    # Setup signal handlers
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    shutdown_requested = False
    
    def graceful_shutdown(sig, frame):
        nonlocal shutdown_requested
        sig_name = signal.Signals(sig).name
        if shutdown_requested:
            logger.warning(f"Received {sig_name} again, forcing immediate exit")
            sys.exit(1)
        logger.info(f"Received {sig_name}, initiating graceful shutdown...")
        shutdown_requested = True
        loop.call_soon_threadsafe(lambda: loop.create_task(server.stop(save_checkpoint=True)))
    
    def save_checkpoint_handler(sig, frame):
        sig_name = signal.Signals(sig).name
        logger.info(f"Received {sig_name}, saving checkpoint...")
        loop.call_soon_threadsafe(lambda: loop.create_task(server.save_checkpoint_now()))
    
    signal.signal(signal.SIGINT, graceful_shutdown)
    signal.signal(signal.SIGTERM, graceful_shutdown)
    signal.signal(signal.SIGUSR1, save_checkpoint_handler)  # Save checkpoint without stopping
    
    logger.info("Signal handlers registered:")
    logger.info("  SIGINT/SIGTERM (Ctrl+C): Graceful shutdown with checkpoint")
    logger.info("  SIGUSR1: Save checkpoint without stopping")
    
    try:
        loop.run_until_complete(server.start())
    except KeyboardInterrupt:
        pass
    finally:
        loop.run_until_complete(server.stop())
        loop.close()


if __name__ == "__main__":
    main()
