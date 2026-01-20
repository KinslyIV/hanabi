"""
MCTS-based game simulator for self-play training.

This simulator uses BeliefMCTS to generate training data with:
- MCTS policy targets (visit count distributions)
- Value targets (game outcomes)

The collected data can be used to train both the policy and value heads
in an AlphaZero-style training loop.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch

from hanabi_learning_environment import pyhanabi
from rl_hanabi.game.hle_state import HLEGameState
from rl_hanabi.belief.belief_state import BeliefState
from rl_hanabi.model.belief_model import ActionDecoder
from rl_hanabi.mcts.belief_mcts import BeliefMCTS, SearchTransition
from rl_hanabi.training.game_simulator import (
    GameConfig,
    Transition,
    GameResult,
    sample_game_config,
)


class MCTSGameSimulator:
    """
    Simulates Hanabi games using Belief MCTS for data collection.
    
    Unlike the basic GameSimulator that uses the policy network directly,
    this simulator runs MCTS to get improved policy targets for training.
    This enables AlphaZero-style policy improvement.
    
    Can also collect "search transitions" - states explored during MCTS thinking
    that provide additional training signal without actually playing those moves.
    """
    
    def __init__(
        self,
        model: ActionDecoder,
        device: torch.device,
        mcts_simulations: int = 100,
        mcts_time_ms: Optional[int] = None,
        c_puct: float = 1.4,
        temperature: float = 1.0,
        temperature_drop_move: int = 30,  # Drop temperature to 0 after this many moves
        dirichlet_alpha: float = 0.3,
        dirichlet_weight: float = 0.25,
        top_k_actions: int = 10,
        collect_search_transitions: bool = True,  # Collect transitions from MCTS thinking
        min_visits_for_search_transition: int = 5,  # Min visits to collect a search transition
    ):
        """
        Args:
            model: The ActionDecoder model for MCTS evaluation
            device: Device to run inference on
            mcts_simulations: Number of MCTS simulations per move
            mcts_time_ms: Optional time limit for MCTS (overrides simulations)
            c_puct: PUCT exploration constant
            temperature: Temperature for action selection from MCTS policy
            temperature_drop_move: Move number after which temperature drops to 0
            dirichlet_alpha: Dirichlet noise alpha for root exploration
            dirichlet_weight: Weight for Dirichlet noise at root
            top_k_actions: Number of top actions to expand in MCTS
            collect_search_transitions: If True, collect transitions from states explored during MCTS
            min_visits_for_search_transition: Minimum visit count for a node to be collected as a search transition
        """
        self.model = model
        self.device = device
        self.mcts_simulations = mcts_simulations
        self.mcts_time_ms = mcts_time_ms
        self.c_puct = c_puct
        self.temperature = temperature
        self.temperature_drop_move = temperature_drop_move
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_weight = dirichlet_weight
        self.top_k_actions = top_k_actions
        self.collect_search_transitions = collect_search_transitions
        self.min_visits_for_search_transition = min_visits_for_search_transition
        
        # Model dimensions for padding
        self.max_num_colors = model.max_num_colors
        self.max_num_ranks = model.max_num_ranks
        self.max_hand_size = model.max_hand_size
        self.max_num_players = model.max_num_players
    
    def _create_mcts(self, temperature: float) -> BeliefMCTS:
        """Create a new MCTS instance with current settings."""
        return BeliefMCTS(
            model=self.model,
            device=self.device,
            time_ms=self.mcts_time_ms or 0,
            c_puct=self.c_puct,
            num_simulations=self.mcts_simulations if self.mcts_time_ms is None else None,
            temperature=temperature,
            dirichlet_alpha=self.dirichlet_alpha,
            dirichlet_weight=self.dirichlet_weight,
            top_k_actions=self.top_k_actions,
            collect_search_transitions=self.collect_search_transitions,
            min_visits_for_search_transition=self.min_visits_for_search_transition,
        )
    
    def simulate_game(
        self,
        config: GameConfig,
        collect_all_perspectives: bool = True,
    ) -> Tuple[GameResult, List[SearchTransition]]:
        """
        Simulate a complete game using MCTS and collect transitions.
        
        Each transition includes the MCTS policy distribution as a target
        for training the policy network.
        
        Args:
            config: Game configuration
            collect_all_perspectives: If True, collect transition from all players' perspectives
        
        Returns:
            Tuple of (GameResult, search_transitions):
              - GameResult with all game transitions and statistics
              - List of SearchTransition from MCTS thinking (empty if collect_search_transitions=False)
        """
        # Create game state
        state = HLEGameState.from_table_options(config.to_dict(), config.num_players)
        
        # Create belief states for all players
        belief_states = [
            BeliefState(state, player=p)
            for p in range(config.num_players)
        ]
        
        transitions: List[Transition] = []
        all_search_transitions: List[SearchTransition] = []
        num_turns = 0
        
        # Game config dict for search transitions
        game_config_dict = {
            "num_players": config.num_players,
            "num_colors": config.num_colors,
            "num_ranks": config.num_ranks,
            "hand_size": config.hand_size,
        }
        
        while not state.is_terminal():
            current_player = state.current_player_index
            
            # Get legal moves
            legal_moves = state.legal_moves()
            legal_moves_mask = state.legal_moves_mask()
            
            if not legal_moves:
                break
            
            # Determine temperature based on move count
            current_temp = self.temperature if num_turns < self.temperature_drop_move else 0.0
            
            # Create MCTS and run search
            mcts = self._create_mcts(current_temp)
            mcts.init_root(state, belief_states, game_config=game_config_dict)
            mcts_policy, mcts_value = mcts.run()
            
            # Collect search transitions from this MCTS run
            if self.collect_search_transitions:
                all_search_transitions.extend(mcts.get_search_transitions())
            
            # Select action from MCTS policy
            action_idx, move = mcts.select_action()
            
            # Store observers for this turn
            if collect_all_perspectives:
                observers = range(config.num_players)
            else:
                observers = [current_player]
            
            # Store pre-move observations for all observers
            pre_move_hands = {}
            for observer in observers:
                all_hands, fireworks, discard_pile, tokens = belief_states[observer].prepare_belief_obs(observer)
                # Ground truth for observer's hand
                observer_hand = state.get_hands()[observer]
                true_colors = np.array([card.color() for card in observer_hand], dtype=np.int64)
                true_ranks = np.array([card.rank() for card in observer_hand], dtype=np.int64)
                pre_move_hands[observer] = (all_hands, fireworks, discard_pile, tokens, true_colors, true_ranks)
            
            # Track score before move
            score_before = state.score()
            
            # Apply the move
            state.apply_move(move)
            
            # Calculate step reward
            score_after = state.score()
            max_score = state.max_score()
            step_reward = (score_after - score_before) / max_score if max_score > 0 else 0.0
            
            # Get affected indices from move history
            move_type = move.type()
            history_item = state.state.move_history()[-1]
            affected_indices = []
            failed_play = False
            
            if move_type in (pyhanabi.HanabiMoveType.REVEAL_COLOR, pyhanabi.HanabiMoveType.REVEAL_RANK):
                affected_indices = list(history_item.card_info_revealed())
            elif move_type == pyhanabi.HanabiMoveType.PLAY:
                affected_indices = [move.card_index()]
                failed_play = not history_item.scored()
            elif move_type == pyhanabi.HanabiMoveType.DISCARD:
                affected_indices = [move.card_index()]
            
            # Create transitions for each observer
            for observer in observers:
                transition = self._create_transition(
                    pre_move_observation=pre_move_hands[observer],
                    move=move,
                    action_idx=action_idx,
                    legal_moves_mask=legal_moves_mask,
                    mcts_policy=mcts_policy,
                    observer=observer,
                    acting_player=current_player,
                    affected_indices=affected_indices,
                    config=config,
                    failed_play=failed_play,
                    step_reward=step_reward,
                )
                if transition is not None:
                    transitions.append(transition)
            
            # Update belief states after the move
            for bs in belief_states:
                bs.state = state
                bs.update_from_move(model=self.model)
            
            num_turns += 1
        
        # Mark final transitions as done
        if collect_all_perspectives:
            num_final = min(config.num_players, len(transitions))
            for t in transitions[-num_final:]:
                t.done = True
        else:
            if transitions:
                transitions[-1].done = True
        
        # Assign normalized final score as reward for all transitions
        final_score = state.score()
        max_score = state.max_score()
        normalized_reward = final_score / max_score if max_score > 0 else 0.0
        
        for t in transitions:
            t.reward = normalized_reward
        
        game_result = GameResult(
            transitions=transitions,
            final_score=final_score,
            max_possible_score=max_score,
            num_turns=num_turns,
            game_config={
                "num_players": config.num_players,
                "num_colors": config.num_colors,
                "num_ranks": config.num_ranks,
                "hand_size": config.hand_size,
            },
        )
        
        return game_result, all_search_transitions
    
    def _create_transition(
        self,
        pre_move_observation: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        move: pyhanabi.HanabiMove,
        action_idx: int,
        legal_moves_mask: np.ndarray,
        mcts_policy: np.ndarray,
        observer: int,
        acting_player: int,
        affected_indices: List[int],
        config: GameConfig,
        failed_play: bool = False,
        step_reward: float = 0.0,
    ) -> Optional[Transition]:
        """Create a transition with MCTS policy target."""
        try:
            all_hands, fireworks, discard_pile_one_hot, tokens, true_colors, true_ranks = pre_move_observation
            
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
            target_player_index = (acting_player + target_offset) % config.num_players
            target_player_offset = (target_player_index - observer) % config.num_players
            
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
                affected_mask[target_player_offset, affected_indices] = 1.0
            elif move_type in (pyhanabi.HanabiMoveType.PLAY, pyhanabi.HanabiMoveType.DISCARD):
                affected_mask[acting_player_offset, affected_indices] = 1.0
            
            # Pad ground truth if needed
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
                mcts_policy=mcts_policy.copy(),  # Store MCTS policy as target
                game_config={
                    "num_players": config.num_players,
                    "num_colors": config.num_colors,
                    "num_ranks": config.num_ranks,
                    "hand_size": config.hand_size,
                },
                failed_play=failed_play,
                step_reward=step_reward,
            )
        except Exception as e:
            print(f"Error creating MCTS transition: {e}")
            return None


def run_mcts_self_play(
    model: ActionDecoder,
    device: torch.device,
    num_games: int,
    config: Optional[GameConfig] = None,
    mcts_simulations: int = 100,
    c_puct: float = 1.4,
    temperature: float = 1.0,
    collect_all_perspectives: bool = True,
    collect_search_transitions: bool = False,
    min_visits_for_search_transition: int = 2,
    verbose: bool = True,
) -> Tuple[List[GameResult], List[SearchTransition]]:
    """
    Run multiple games of self-play using MCTS.
    
    Args:
        model: The ActionDecoder model
        device: Device for inference
        num_games: Number of games to play
        config: Game configuration (random if None)
        mcts_simulations: MCTS simulations per move
        c_puct: PUCT exploration constant
        temperature: Temperature for action selection
        collect_all_perspectives: Collect data from all player perspectives
        collect_search_transitions: Collect additional transitions from MCTS thinking
        min_visits_for_search_transition: Minimum visit count for search transitions
        verbose: Print progress
    
    Returns:
        Tuple of (List[GameResult], List[SearchTransition]):
          - List of GameResult objects from actual games
          - List of SearchTransition objects from MCTS thinking (empty if disabled)
    """
    simulator = MCTSGameSimulator(
        model=model,
        device=device,
        mcts_simulations=mcts_simulations,
        c_puct=c_puct,
        temperature=temperature,
        collect_search_transitions=collect_search_transitions,
        min_visits_for_search_transition=min_visits_for_search_transition,
    )
    
    results = []
    all_search_transitions: List[SearchTransition] = []
    total_score = 0
    total_max_score = 0
    
    for game_idx in range(num_games):
        game_config = config or sample_game_config()
        result, search_transitions = simulator.simulate_game(game_config, collect_all_perspectives)
        results.append(result)
        all_search_transitions.extend(search_transitions)
        
        total_score += result.final_score
        total_max_score += result.max_possible_score
        
        if verbose and (game_idx + 1) % max(1, num_games // 10) == 0:
            avg_score = total_score / (game_idx + 1)
            avg_normalized = total_score / total_max_score if total_max_score > 0 else 0
            search_info = f", Search transitions: {len(all_search_transitions)}" if collect_search_transitions else ""
            print(f"Game {game_idx + 1}/{num_games}: "
                  f"Score {result.final_score}/{result.max_possible_score}, "
                  f"Avg: {avg_score:.1f}, Normalized: {avg_normalized:.2%}{search_info}")
    
    return results, all_search_transitions
