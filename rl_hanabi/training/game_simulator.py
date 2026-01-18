"""
Game simulator for self-play training.
Runs games using the ActionDecoder model to generate training data.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch

from hanabi_learning_environment import pyhanabi
from rl_hanabi.game.hle_state import HLEGameState
from rl_hanabi.belief.belief_state import BeliefState
from rl_hanabi.model.belief_model import ActionDecoder


@dataclass
class GameConfig:
    """Configuration for a Hanabi game."""
    num_players: int = 2
    num_colors: int = 5
    num_ranks: int = 5
    hand_size: int = 5
    max_information_tokens: int = 8
    max_life_tokens: int = 3
    seed: int = -1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "numSuits": self.num_colors,
            "numRanks": self.num_ranks,
            "cardsPerHand": self.hand_size,
            "clueTokens": self.max_information_tokens,
            "strikeTokens": self.max_life_tokens,
            "seed": self.seed,
        }


@dataclass
class Transition:
    """A single transition from the game."""
    # State information
    slot_beliefs: np.ndarray          # [P, H, C+R]
    affected_mask: np.ndarray         # [P, H]
    move_target_player: int           # target player offset
    acting_player: int                # acting player offset
    action: np.ndarray                # [action_dim]
    fireworks: np.ndarray             # [C]
    discard_pile: np.ndarray          # [C*R]
    
    # Targets for training
    true_colors: np.ndarray           # [H] - true colors of observer's hand
    true_ranks: np.ndarray            # [H] - true ranks of observer's hand
    chosen_action_idx: int            # Index of chosen action
    legal_moves_mask: np.ndarray      # [action_space_size] - mask of legal moves
    
    # Metadata
    game_config: Dict[str, int] = field(default_factory=dict)
    reward: float = 0.0               # Immediate reward (e.g., score delta)
    done: bool = False                # Whether game ended


@dataclass
class GameResult:
    """Result of a completed game."""
    transitions: List[Transition]
    final_score: int
    max_possible_score: int
    num_turns: int
    game_config: Dict[str, int]
    

class GameSimulator:
    """
    Simulates Hanabi games for self-play training.
    Uses the ActionDecoder model to select actions.
    """
    
    def __init__(
        self,
        model: ActionDecoder,
        device: torch.device,
        temperature: float = 1.0,
        epsilon: float = 0.1,
        max_num_colors: int = 5,
        max_num_ranks: int = 5,
        max_hand_size: int = 5,
        max_num_players: int = 5,
    ):
        """
        Args:
            model: The ActionDecoder model for action selection
            device: Device to run inference on
            temperature: Temperature for action sampling (higher = more exploration)
            epsilon: Probability of random action (epsilon-greedy)
            max_num_colors: Maximum colors the model expects
            max_num_ranks: Maximum ranks the model expects
            max_hand_size: Maximum hand size the model expects
            max_num_players: Maximum players the model expects
        """
        self.model = model
        self.device = device
        self.temperature = temperature
        self.epsilon = epsilon
        self.max_num_colors = max_num_colors
        self.max_num_ranks = max_num_ranks
        self.max_hand_size = max_hand_size
        self.max_num_players = max_num_players
    
    def _pad_observation(self, all_hands, fireworks, discard_pile, affected_mask, config):
        """Pad observations to match model's expected max dimensions."""
        num_players = config.num_players
        num_colors = config.num_colors
        num_ranks = config.num_ranks
        hand_size = config.hand_size
        
        # Pad slot_beliefs: [P, H, C+R] -> [max_P, max_H, max_C + max_R]
        padded_beliefs = np.zeros(
            (self.max_num_players, self.max_hand_size, self.max_num_colors + self.max_num_ranks),
            dtype=np.float32
        )
        # Copy color beliefs
        padded_beliefs[:num_players, :hand_size, :num_colors] = all_hands[:, :, :num_colors]
        # Copy rank beliefs (shifted to after max_colors)
        padded_beliefs[:num_players, :hand_size, self.max_num_colors:self.max_num_colors + num_ranks] = \
            all_hands[:, :, num_colors:num_colors + num_ranks]
        
        # Pad fireworks: [C] -> [max_C]
        padded_fireworks = np.zeros(self.max_num_colors, dtype=np.float32)
        padded_fireworks[:num_colors] = fireworks
        
        # Pad discard_pile: [C*R] -> [max_C * max_R]
        padded_discard = np.zeros(self.max_num_colors * self.max_num_ranks, dtype=np.float32)
        # Remap indices from original layout to padded layout
        for orig_idx in range(len(discard_pile)):
            if discard_pile[orig_idx] > 0:
                orig_color = orig_idx // num_ranks
                orig_rank = orig_idx % num_ranks
                new_idx = orig_color * self.max_num_ranks + orig_rank
                padded_discard[new_idx] = discard_pile[orig_idx]
        
        # Pad affected_mask: [P, H] -> [max_P, max_H]
        padded_mask = np.zeros((self.max_num_players, self.max_hand_size), dtype=np.float32)
        padded_mask[:num_players, :hand_size] = affected_mask[:num_players, :hand_size]
        
        return padded_beliefs, padded_fireworks, padded_discard, padded_mask
        
    def simulate_game(
        self,
        config: GameConfig,
        collect_all_perspectives: bool = True,
    ) -> GameResult:
        """
        Simulate a complete game and collect transitions.
        
        Args:
            config: Game configuration
            collect_all_perspectives: If True, collect transition from all players' perspectives
        
        Returns:
            GameResult with all transitions and game statistics
        """
        # Create game state
        state = HLEGameState.from_table_options(config.to_dict(), config.num_players)
        
        # Create belief states for all players (without model to avoid dimension issues)
        # The model is only used for action selection, not belief updates during data collection
        belief_states = [
            BeliefState(state, player=p)
            for p in range(config.num_players)
        ]
        
        transitions = []
        num_turns = 0
        prev_score = 0
        
        while not state.is_terminal():
            current_player = state.current_player_index
            
            # Get legal moves
            legal_moves = state.legal_moves()
            legal_moves_mask = state.legal_moves_mask()
            
            if not legal_moves:
                break
            
            # Select action using the model
            action_idx, action_probs = self._select_action(
                belief_states[current_player],
                legal_moves_mask,
                config,
            )
            
            # Get the actual move
            move = state.index_to_move(action_idx)
            move_type = move.type()
            
            # Collect transitions from each observer's perspective
            if collect_all_perspectives:
                observers = range(config.num_players)
            else:
                observers = [current_player]
            
            for observer in observers:
                transition = self._create_transition(
                    belief_states[observer],
                    state,
                    move,
                    action_idx,
                    legal_moves_mask,
                    observer,
                    current_player,
                    config,
                )
                if transition is not None:
                    transitions.append(transition)
            
            # Apply the move
            state.apply_move(move)
            
            # Update belief states
            for bs in belief_states:
                bs.state = state
                bs.update_from_move()
            
            # Calculate reward
            current_score = state.score()
            reward = current_score - prev_score
            prev_score = current_score
            
            # Update last transition with reward
            if transitions:
                transitions[-1].reward = reward
            
            num_turns += 1
        
        # Mark final transitions as done
        for t in transitions[-config.num_players:]:
            t.done = True
        
        return GameResult(
            transitions=transitions,
            final_score=state.score(),
            max_possible_score=state.max_score(),
            num_turns=num_turns,
            game_config={
                "num_players": config.num_players,
                "num_colors": config.num_colors,
                "num_ranks": config.num_ranks,
                "hand_size": config.hand_size,
            },
        )
    
    def _select_action(
        self,
        belief_state: BeliefState,
        legal_moves_mask: np.ndarray,
        config: GameConfig,
    ) -> Tuple[int, np.ndarray]:
        """Select an action using the model with exploration."""
        
        # Epsilon-greedy exploration
        if random.random() < self.epsilon:
            legal_indices = np.where(legal_moves_mask)[0]
            action_idx = random.choice(legal_indices)
            # Create uniform probabilities for legal moves
            probs = np.zeros_like(legal_moves_mask, dtype=np.float32)
            probs[legal_moves_mask] = 1.0 / len(legal_indices)
            return action_idx, probs
        
        # Use model to get action probabilities
        player = belief_state.player
        all_hands, fireworks, discard_pile_one_hot, tokens = belief_state.prepare_belief_obs(player)
        
        # Create dummy action encoding (we're selecting, not updating)
        action_encoding, affected_mask = belief_state.encode_last_action()
        
        # Pad observations to model's expected dimensions
        padded_beliefs, padded_fireworks, padded_discard, padded_mask = self._pad_observation(
            all_hands, fireworks, discard_pile_one_hot, affected_mask, config
        )
        
        # Convert to tensors
        slot_beliefs_tensor = torch.from_numpy(padded_beliefs).float().unsqueeze(0).to(self.device)
        affected_mask_tensor = torch.from_numpy(padded_mask).float().unsqueeze(0).to(self.device)
        action_tensor = torch.from_numpy(action_encoding).float().unsqueeze(0).to(self.device)
        fireworks_tensor = torch.from_numpy(padded_fireworks).float().unsqueeze(0).to(self.device)
        discard_pile_tensor = torch.from_numpy(padded_discard).float().unsqueeze(0).to(self.device)
        target_player_tensor = torch.tensor([0], dtype=torch.long, device=self.device)
        acting_player_tensor = torch.tensor([0], dtype=torch.long, device=self.device)
        
        with torch.no_grad():
            _, _, action_logits = self.model(
                slot_beliefs=slot_beliefs_tensor,
                affected_mask=affected_mask_tensor,
                move_target_player=target_player_tensor,
                acting_player=acting_player_tensor,
                action=action_tensor,
                fireworks=fireworks_tensor,
                discard_pile=discard_pile_tensor,
            )
        
        action_logits = action_logits.squeeze(0).cpu().numpy()
        
        # Mask illegal actions and apply temperature
        action_logits = action_logits[:len(legal_moves_mask)]
        action_logits[~legal_moves_mask] = -float('inf')
        
        # Apply temperature
        if self.temperature > 0:
            action_logits = action_logits / self.temperature
        
        # Softmax to get probabilities
        exp_logits = np.exp(action_logits - np.max(action_logits))
        probs = exp_logits / (exp_logits.sum() + 1e-10)
        
        # Sample action
        action_idx = np.random.choice(len(probs), p=probs)
        
        return action_idx, probs
    
    def _create_transition(
        self,
        belief_state: BeliefState,
        state: HLEGameState,
        move: pyhanabi.HanabiMove,
        action_idx: int,
        legal_moves_mask: np.ndarray,
        observer: int,
        acting_player: int,
        config: GameConfig,
    ) -> Optional[Transition]:
        """Create a transition from an observer's perspective."""
        
        try:
            # Get belief observation
            all_hands, fireworks, discard_pile_one_hot, tokens = belief_state.prepare_belief_obs(observer)
            
            if all_hands is None:
                return None
            
            # Get action encoding based on move type
            move_type = move.type()
            target_offset = 0
            clue_type = 0
            clue_value = 0
            
            if move_type == pyhanabi.HanabiMoveType.REVEAL_COLOR:
                target_offset = move.target_offset()
                clue_type = 0
                clue_value = move.color()
            elif move_type == pyhanabi.HanabiMoveType.REVEAL_RANK:
                target_offset = move.target_offset()
                clue_type = 1
                clue_value = move.rank()
            elif move_type == pyhanabi.HanabiMoveType.PLAY:
                clue_type = 2
                clue_value = move.card_index()
            elif move_type == pyhanabi.HanabiMoveType.DISCARD:
                clue_type = 3
                clue_value = move.card_index()
            
            # Calculate offsets from observer's perspective
            acting_player_offset = (acting_player - observer) % config.num_players
            target_player_offset = target_offset  # Already an offset from acting player
            
            # Encode action
            action_encoding = np.array([
                acting_player_offset,
                target_player_offset,
                clue_type,
                clue_value
            ], dtype=np.float32)
            
            # Create affected mask
            affected_mask = np.zeros((config.num_players, config.hand_size), dtype=np.float32)
            if move_type in (pyhanabi.HanabiMoveType.REVEAL_COLOR, pyhanabi.HanabiMoveType.REVEAL_RANK):
                # Would need to track which cards are affected - simplified here
                target_player = (acting_player + target_offset) % config.num_players
                target_player_offset_from_observer = (target_player - observer) % config.num_players
                # Mark all slots as potentially affected (simplified)
                affected_mask[target_player_offset_from_observer, :] = 1.0
            
            # Get ground truth for observer's hand
            observer_hand = state.get_hands()[observer]
            true_colors = np.array([card.color() for card in observer_hand], dtype=np.int64)
            true_ranks = np.array([card.rank() for card in observer_hand], dtype=np.int64)
            
            # Pad if hand is smaller than max
            if len(true_colors) < config.hand_size:
                pad_size = config.hand_size - len(true_colors)
                true_colors = np.pad(true_colors, (0, pad_size), constant_values=-1)
                true_ranks = np.pad(true_ranks, (0, pad_size), constant_values=-1)
            
            return Transition(
                slot_beliefs=all_hands,
                affected_mask=affected_mask,
                move_target_player=target_player_offset,
                acting_player=acting_player_offset,
                action=action_encoding,
                fireworks=fireworks,
                discard_pile=discard_pile_one_hot,
                true_colors=true_colors,
                true_ranks=true_ranks,
                chosen_action_idx=action_idx,
                legal_moves_mask=legal_moves_mask.copy(),
                game_config={
                    "num_players": config.num_players,
                    "num_colors": config.num_colors,
                    "num_ranks": config.num_ranks,
                    "hand_size": config.hand_size,
                },
            )
        except Exception as e:
            # Log error but don't crash
            print(f"Error creating transition: {e}")
            return None


def sample_game_config(
    num_players_range: Tuple[int, int] = (3, 5),
    num_colors_range: Tuple[int, int] = (3, 5),
    num_ranks_range: Tuple[int, int] = (3, 5),
) -> GameConfig:
    """
    Sample a random game configuration.
    
    Ensures the config is valid: hand_size * num_players <= cards_per_color * num_colors
    Standard Hanabi has approximately (num_ranks * 2) cards per color.
    """
    # Keep trying until we get a valid config
    max_attempts = 100
    for _ in range(max_attempts):
        num_players = random.randint(*num_players_range)
        num_colors = random.randint(*num_colors_range)
        num_ranks = random.randint(*num_ranks_range)
        
        # Hand size depends on number of players
        if num_players <= 3:
            hand_size = 5
        else:
            hand_size = 4
        
        # Estimate cards per color (approximately 2 per rank in standard distribution)
        # In HLE: for 5 ranks, there are 3+2+2+2+1=10 cards per color
        # For smaller ranks, estimate ~2 cards per rank
        cards_per_color = num_ranks * 2
        
        # Check the constraint: hand_size * num_players <= cards_per_color * num_colors
        total_cards_needed = hand_size * num_players
        total_cards_available = cards_per_color * num_colors
        
        if total_cards_needed <= total_cards_available:
            return GameConfig(
                num_players=num_players,
                num_colors=num_colors,
                num_ranks=num_ranks,
                hand_size=hand_size,
                seed=random.randint(0, 2**31 - 1),
            )
    
    # Fallback to a safe default config
    return GameConfig(
        num_players=3,
        num_colors=5,
        num_ranks=5,
        hand_size=5,
        seed=random.randint(0, 2**31 - 1),
    )
