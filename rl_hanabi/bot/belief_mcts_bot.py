"""
Belief MCTS Bot for Hanabi.

This bot uses the belief-integrated MCTS that combines:
- Neural network policy and value heads
- Belief state tracking
- PUCT-based tree search
"""

import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from rl_hanabi.bot.base_bot import BaseBot
from rl_hanabi.game.game_types import ACTION
from rl_hanabi.game.hle_state import HLEGameState
from rl_hanabi.belief.belief_state import BeliefState
from rl_hanabi.mcts.belief_mcts import BeliefMCTS
from rl_hanabi.model.belief_model import ActionDecoder
from hanabi_learning_environment import pyhanabi


class BeliefMCTSBot(BaseBot):
    """
    A Hanabi bot that uses Belief-Integrated MCTS.
    
    This bot maintains belief states for all players and uses a neural
    network with policy and value heads to guide MCTS search.
    """
    
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        checkpoint_dir: str = "checkpoints",
        time_limit_ms: int = 2000,
        c_puct: float = 1.4,
        temperature: float = 0.5,
        num_simulations: Optional[int] = None,
        device: Optional[str] = None,
    ) -> None:
        """
        Initialize the Belief MCTS Bot.
        
        Args:
            checkpoint_path: Path to model checkpoint (optional)
            checkpoint_dir: Directory to search for checkpoints
            time_limit_ms: MCTS time budget per move
            c_puct: PUCT exploration constant
            temperature: Temperature for action selection
            num_simulations: Fixed number of simulations (overrides time_limit_ms)
            device: Device for inference ("cpu" or "cuda")
        """
        super().__init__(enable_hle=True)
        self.logger = logging.getLogger("rl_hanabi.belief_mcts_bot")
        
        self.time_limit_ms = time_limit_ms
        self.c_puct = c_puct
        self.temperature = temperature
        self.num_simulations = num_simulations
        
        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.logger.info(f"Using device: {self.device}")
        
        # Load model
        self.model: Optional[ActionDecoder] = None
        self._load_model(checkpoint_path, checkpoint_dir)
        
        # Belief states (initialized when game starts)
        self.belief_states: List[BeliefState] = []
        
        # MCTS instance (reused across turns)
        self.mcts: Optional[BeliefMCTS] = None
        
        # Threading
        self._thinking = False
        self._thinking_lock = threading.Lock()
    
    def _find_checkpoint(self, checkpoint_path: Optional[str], checkpoint_dir: str) -> Optional[Path]:
        """Find the best available checkpoint."""
        if checkpoint_path:
            path = Path(checkpoint_path)
            if path.exists():
                return path
        
        checkpoint_dir_path = Path(checkpoint_dir)
        
        # Try checkpoint_latest.pt
        latest = checkpoint_dir_path / "checkpoint_latest.pt"
        if latest.exists():
            return latest
        
        # Find highest numbered checkpoint
        checkpoints = list(checkpoint_dir_path.glob("checkpoint_iter_*.pt"))
        if checkpoints:
            def get_iter_num(p: Path) -> int:
                try:
                    return int(p.stem.split("_")[-1])
                except ValueError:
                    return -1
            checkpoints.sort(key=get_iter_num, reverse=True)
            return checkpoints[0]
        
        # Try checkpoint_final.pt
        final = checkpoint_dir_path / "checkpoint_final.pt"
        if final.exists():
            return final
        
        return None
    
    def _load_model(self, checkpoint_path: Optional[str], checkpoint_dir: str) -> None:
        """Load the model from checkpoint."""
        path = self._find_checkpoint(checkpoint_path, checkpoint_dir)
        
        if path is None:
            self.logger.warning("No checkpoint found! Bot will use random moves.")
            return
        
        self.logger.info(f"Loading checkpoint: {path}")
        
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            
            model_config = checkpoint.get("model_config", {})
            
            self.model = ActionDecoder(
                max_num_colors=model_config.get("max_num_colors", 5),
                max_num_ranks=model_config.get("max_num_ranks", 5),
                max_hand_size=model_config.get("max_hand_size", 5),
                max_num_players=model_config.get("max_num_players", 5),
                num_heads=model_config.get("num_heads", 4),
                num_layers=model_config.get("num_layers", 4),
                d_model=model_config.get("d_model", 128),
                action_dim=4,
            )
            
            # Handle potentially missing value head
            try:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            except RuntimeError:
                self.logger.warning("Some weights not found (likely value head). Initializing randomly.")
                self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            
            self.model.to(self.device)
            self.model.eval()
            
            self.logger.info("Model loaded successfully!")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            self.model = None
    
    def _init_belief_states(self) -> None:
        """Initialize belief states for all players."""
        if not self.state or not self.state.hle_state:
            return
        
        num_players = self.state.num_players
        self.belief_states = [
            BeliefState(self.state.hle_state, player=p)
            for p in range(num_players)
        ]
        
        self.logger.info(f"Initialized belief states for {num_players} players")
    
    def _update_belief_states(self) -> None:
        """Update belief states based on the last move."""
        if not self.state or not self.state.hle_state:
            return
        
        for belief in self.belief_states:
            belief.state = self.state.hle_state
            belief.reinit_belief_state()
            belief.update_from_move(model=self.model)
    
    def _on_game_start(self) -> None:
        """Called when a new game starts."""
        super()._on_game_start()
        self._init_belief_states()
        
        # Initialize MCTS if model is available
        if self.model is not None:
            self.mcts = BeliefMCTS(
                model=self.model,
                device=self.device,
                time_ms=self.time_limit_ms,
                c_puct=self.c_puct,
                num_simulations=self.num_simulations,
                temperature=self.temperature,
            )
    
    def make_move(self) -> None:
        """Start MCTS computation in a background thread."""
        if not self.state or not self.state.hle_state:
            self.logger.error("Cannot make move: state is missing")
            return
        
        # Prevent concurrent runs
        with self._thinking_lock:
            if self._thinking:
                self.logger.warning("Already thinking, ignoring duplicate call")
                return
            self._thinking = True
        
        self.logger.info("Starting Belief MCTS in background thread...")
        
        # Update belief states before search
        self._update_belief_states()
        
        # Copy state for thread
        root_state = self.state.hle_state.copy()
        
        thread = threading.Thread(
            target=self._run_mcts_and_send_move,
            args=(root_state,),
            daemon=True
        )
        thread.start()
    
    def _run_mcts_and_send_move(self, root_state: HLEGameState) -> None:
        """Run MCTS and send the move. Called in background thread."""
        try:
            if self.model is None or self.mcts is None:
                # Fallback to random move
                self.logger.warning("No model loaded, selecting random move")
                legal_moves = root_state.legal_moves()
                if legal_moves:
                    import random
                    move = random.choice(legal_moves)
                    self._send_hanabi_move(move)
                return
            
            # Create belief states for MCTS
            belief_states = [
                BeliefState(root_state, player=p)
                for p in range(root_state.num_players)
            ]
            
            # Update beliefs from game history
            for belief in belief_states:
                belief.update_from_move(model=self.model)
            
            # Initialize and run MCTS
            self.mcts.init_root(root_state, belief_states)
            policy, value = self.mcts.run()
            
            # Select and send move
            action_idx, move = self.mcts.select_action()
            
            self.logger.info(f"Belief MCTS selected: {move}")
            self.logger.info(f"Root value: {value:.3f}, Action prob: {policy[action_idx]:.3f}")
            
            self._send_hanabi_move(move)
            
        except Exception as e:
            self.logger.error(f"MCTS run failed: {e}", exc_info=True)
        finally:
            with self._thinking_lock:
                self._thinking = False
    
    def _send_hanabi_move(self, move: pyhanabi.HanabiMove) -> None:
        """Convert and send a HanabiMove to the server."""
        if not self.state:
            return
        
        self.logger.info(f"Sending move: {move}")
        move_type = move.type()
        payload = {"tableID": self.table_id}
        
        if move_type == pyhanabi.HanabiMoveType.PLAY:
            card_index = move.card_index()
            hand_size = len(self.state.our_hand)
            hanab_index = hand_size - 1 - card_index
            
            if 0 <= hanab_index < hand_size:
                card_order = self.state.our_hand[hanab_index]
                payload["type"] = ACTION.PLAY
                payload["target"] = card_order
                self.send_cmd("action", payload)
            else:
                self.logger.error(f"Invalid card index: {card_index} -> {hanab_index}")
        
        elif move_type == pyhanabi.HanabiMoveType.DISCARD:
            card_index = move.card_index()
            hand_size = len(self.state.our_hand)
            hanab_index = hand_size - 1 - card_index
            
            if 0 <= hanab_index < hand_size:
                card_order = self.state.our_hand[hanab_index]
                payload["type"] = ACTION.DISCARD
                payload["target"] = card_order
                self.send_cmd("action", payload)
            else:
                self.logger.error(f"Invalid card index: {card_index} -> {hanab_index}")
        
        elif move_type == pyhanabi.HanabiMoveType.REVEAL_COLOR:
            target_offset = move.target_offset()
            target_player = (self.state.our_index + target_offset) % self.state.num_players
            color = move.color()
            
            payload["type"] = ACTION.COLOUR
            payload["target"] = target_player
            payload["value"] = color
            self.send_cmd("action", payload)
        
        elif move_type == pyhanabi.HanabiMoveType.REVEAL_RANK:
            target_offset = move.target_offset()
            target_player = (self.state.our_index + target_offset) % self.state.num_players
            rank = move.rank()
            
            payload["type"] = ACTION.RANK
            payload["target"] = target_player
            payload["value"] = rank + 1  # 1-based for hanab.live
            self.send_cmd("action", payload)
    
    def _handle_chat(self, data: Dict[str, Any]) -> None:
        """Handle chat commands."""
        msg: str = data.get("msg", "")
        who: str = data.get("who", "")
        
        if msg.startswith("/version"):
            self.send_pm(who, "belief-mcts-bot v1.0")
            return
        elif msg.startswith("/settings"):
            settings = f"Belief MCTS: time={self.time_limit_ms}ms, c_puct={self.c_puct}, temp={self.temperature}"
            self.send_pm(who, settings)
            return
        elif msg.startswith("/model"):
            if self.model:
                self.send_pm(who, f"Model loaded on {self.device}")
            else:
                self.send_pm(who, "No model loaded (using random)")
            return
        
        super()._handle_chat(data)
