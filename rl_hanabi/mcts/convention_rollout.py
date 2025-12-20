"""
H-Group Convention-Based Rollout Policy for MCTS.

Implements beginner H-Group conventions:
1. Play Clues - Clue a card that is immediately playable
2. 5 Save - Save 5s on chop with number 5 clue
3. 2 Save - Save 2s on chop with number 2 clue (if only copy visible or critical)
4. Critical Save - Save critical cards on chop
5. Single Card Focus - When clue touches multiple cards, only one is "focused"
6. Clue Interpretation - Non-chop = Play Clue, Chop = could be Play or Save
"""

import random
import numpy as np
from typing import List, Tuple, Optional, Set
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing

from hanabi_learning_environment.pyhanabi import (
    HanabiMove, HanabiMoveType, HanabiCard, HanabiState
)
from rl_hanabi.game.hle_state import HLEGameState


@dataclass
class CardInfo:
    """Information about a card in hand."""
    color: int
    rank: int
    is_clued: bool = False
    known_playable: bool = False
    chop_position: bool = False


class ConventionRolloutPolicy:
    """
    Implements H-Group beginner conventions for rollout policy.
    
    Conventions implemented:
    - Play known playable cards
    - Give play clues for immediately playable cards
    - 5 Save: Save 5s on chop with rank 5 clue
    - 2 Save: Save 2s on chop with rank 2 clue
    - Critical Save: Save last copy of a card on chop
    - Discard oldest unclued card (chop)
    - Avoid discarding clued cards
    - Avoid clues where focus is already clued
    """
    
    def __init__(self, play_weight: float = 10.0, 
                 clue_weight: float = 5.0,
                 save_weight: float = 8.0,
                 discard_weight: float = 1.0):
        """
        Initialize the convention-based rollout policy.
        
        Args:
            play_weight: Weight for playing known playable cards
            clue_weight: Weight for giving play clues
            save_weight: Weight for giving save clues
            discard_weight: Weight for discarding
        """
        self.play_weight = play_weight
        self.clue_weight = clue_weight
        self.save_weight = save_weight
        self.discard_weight = discard_weight
    
    def get_fireworks(self, state: HLEGameState) -> List[int]:
        """Get current fireworks level for each color."""
        return state.fireworks()
    
    def is_playable(self, card: HanabiCard, fireworks: List[int]) -> bool:
        """Check if a card is immediately playable on the fireworks."""
        return card.rank() == fireworks[card.color()]
    
    def is_critical(self, card: HanabiCard, state: HLEGameState) -> bool:
        """Check if a card is critical (last copy remaining)."""
        color, rank = card.color(), card.rank()
        
        # 5s are always critical (only one copy)
        if rank == 4:  # rank is 0-indexed, so 4 = "5"
            return True
        
        # Check discard pile for copies
        discard_pile = state.discard_pile()
        discarded_count = sum(1 for c in discard_pile 
                             if c.color() == color and c.rank() == rank)
        
        # Get total copies of this card type
        total_copies = state.game.num_cards(color, rank)
        
        # Critical if only one copy remains (total - discarded = 1)
        return (total_copies - discarded_count) == 1
    
    def is_already_played(self, card: HanabiCard, fireworks: List[int]) -> bool:
        """Check if a card has already been played on fireworks."""
        return card.rank() < fireworks[card.color()]
    
    def is_card_clued(self, card_knowledge) -> bool:
        """
        Check if a card has been clued (has known color or rank from hints).
        
        Args:
            card_knowledge: HanabiCardKnowledge object for the card
            
        Returns:
            True if the card has received any hint (color or rank), False otherwise
        """
        return card_knowledge.color() is not None or card_knowledge.rank() is not None
    
    def get_clued_status(self, state: HLEGameState, player_idx: int) -> List[bool]:
        """
        Get the clued status for each card in a player's hand.
        
        Args:
            state: The game state
            player_idx: The player index to check
            
        Returns:
            List of booleans indicating if each card is clued
        """
        observation = state.observation_for_player(player_idx)
        card_knowledge = observation.card_knowledge()
        # Player 0 in observation is the observing player
        # For other players (offset 1, 2, ...), index is the offset
        player_knowledge = card_knowledge[0]  # Self knowledge
        return [self.is_card_clued(ck) for ck in player_knowledge]
    
    def get_other_player_clued_status(self, state: HLEGameState, 
                                       current_player: int, 
                                       target_player: int) -> List[bool]:
        """
        Get the clued status for cards in another player's hand from current player's view.
        
        Args:
            state: The game state
            current_player: The observing player index
            target_player: The target player index
            
        Returns:
            List of booleans indicating if each card is clued
        """
        observation = state.observation_for_player(current_player)
        card_knowledge = observation.card_knowledge()
        num_players = state.num_players
        
        # Calculate the offset from current player to target
        target_offset = (target_player - current_player) % num_players
        
        player_knowledge = card_knowledge[target_offset]
        return [self.is_card_clued(ck) for ck in player_knowledge]

    def get_chop_index(self, hand: List[HanabiCard], state: HLEGameState, 
                       player_idx: int) -> int:
        """
        Get the chop position (oldest unclued card) for a player.
        In HLE, cards are ordered oldest to newest (0 = oldest).
        The chop is the oldest unclued card.
        
        Args:
            hand: The player's hand
            state: The game state
            player_idx: The player index
            
        Returns:
            The index of the chop card (oldest unclued), or 0 if all cards are clued
        """
        if not hand:
            return 0
        
        # Get clued status for the player's hand
        try:
            current_player = state.current_player_index
            if player_idx == current_player:
                clued_status = self.get_clued_status(state, player_idx)
            else:
                clued_status = self.get_other_player_clued_status(state, current_player, player_idx)
            
            # Find the oldest unclued card (lowest index that is not clued)
            for i in range(len(hand)):
                if i < len(clued_status) and not clued_status[i]:
                    return i
            
            # If all cards are clued, return the oldest card
            return 0
        except Exception:
            # Fallback: return oldest card
            return 0
    
    def get_playable_cards_in_hand(self, hand: List[HanabiCard], 
                                   fireworks: List[int]) -> List[int]:
        """Get indices of playable cards in a hand."""
        playable = []
        for i, card in enumerate(hand):
            if self.is_playable(card, fireworks):
                playable.append(i)
        return playable
    
    def get_save_candidates(self, hand: List[HanabiCard], state: HLEGameState,
                           fireworks: List[int]) -> Tuple[bool, bool, bool]:
        """
        Check if chop card needs saving.
        Returns: (needs_5_save, needs_2_save, needs_critical_save)
        """
        if not hand:
            return False, False, False
        
        chop_idx = self.get_chop_index(hand, state, -1)
        chop_card = hand[chop_idx]
        
        # Don't save already played cards
        if self.is_already_played(chop_card, fireworks):
            return False, False, False
        
        # Check for 5 save
        needs_5_save = chop_card.rank() == 4  # rank 4 = "5"
        
        # Check for 2 save (rank 1 = "2")
        needs_2_save = chop_card.rank() == 1
        
        # Check for critical save (not 5s, since they're covered by 5 save)
        needs_critical_save = (not needs_5_save and 
                               self.is_critical(chop_card, state))
        
        return needs_5_save, needs_2_save, needs_critical_save
    
    def evaluate_play_clue(self, target_hand: List[HanabiCard], 
                           fireworks: List[int],
                           clue_type: str, 
                           clue_value: int,
                           target_clued_status: Optional[List[bool]] = None) -> float:
        """
        Evaluate the strength of a play clue.
        Returns a score based on how good the clue is.
        
        Args:
            target_hand: The target player's hand
            fireworks: Current fireworks levels
            clue_type: "color" or "rank"
            clue_value: The color/rank value being clued
            target_clued_status: List of booleans indicating which cards are already clued
            
        Returns:
            Score for this clue (0 if bad/redundant, higher for better clues)
        """
        # Find which cards would be touched by this clue
        touched_indices = []
        for i, card in enumerate(target_hand):
            if clue_type == "color" and card.color() == clue_value:
                touched_indices.append(i)
            elif clue_type == "rank" and card.rank() == clue_value:
                touched_indices.append(i)
        
        if not touched_indices:
            return 0.0
        
        # Determine focus: leftmost NEW (unclued) touched card, or leftmost touched if all are clued
        focus_idx = None
        if target_clued_status is not None:
            # Find leftmost unclued touched card
            for idx in sorted(touched_indices):
                if idx < len(target_clued_status) and not target_clued_status[idx]:
                    focus_idx = idx
                    break
            
            # If all touched cards are already clued, this clue has no new focus
            # It could still provide additional information, but with much lower value
            if focus_idx is None:
                return 0.05  # Very low weight for redundant clues
        else:
            focus_idx = min(touched_indices)
        
        focus_card = target_hand[focus_idx]
        
        # High score if focus is playable
        if self.is_playable(focus_card, fireworks):
            return self.clue_weight
        
        return 0.0
    
    def get_move_weights(self, state: HLEGameState) -> np.ndarray:
        """
        Calculate weights for each legal move based on conventions.
        Returns a probability distribution over all moves.
        """
        legal_moves = state.legal_moves()
        weights = np.zeros(state.action_space_size)
        
        if not legal_moves:
            return weights
        
        fireworks = self.get_fireworks(state)
        current_player = state.current_player_index
        hands = state.get_hands()
        info_tokens = state.information_tokens()
        
        # Our hand (current player)
        our_hand = hands[current_player]
        
        for move in legal_moves:
            move_idx = state.move_to_index(move)
            move_type = move.type()
            
            if move_type == HanabiMoveType.PLAY:
                card_idx = move.card_index()
                if card_idx < len(our_hand):
                    card = our_hand[card_idx]
                    if self.is_playable(card, fireworks):
                        weights[move_idx] = self.play_weight
                    else:
                        # Risky play - low weight
                        weights[move_idx] = 0.1
                        
            elif move_type == HanabiMoveType.DISCARD:
                card_idx = move.card_index()
                # Get clued status for our hand
                try:
                    our_clued_status = self.get_clued_status(state, current_player)
                except Exception:
                    our_clued_status = [False] * len(our_hand)
                
                is_card_clued = (card_idx < len(our_clued_status) and 
                                our_clued_status[card_idx])
                
                # Never discard clued cards (very low weight)
                if is_card_clued:
                    weights[move_idx] = 0.01
                # Prefer discarding from chop (oldest unclued card)
                elif card_idx == self.get_chop_index(our_hand, state, current_player):
                    weights[move_idx] = self.discard_weight
                else:
                    # Non-chop discard - very low weight
                    weights[move_idx] = 0.05
                    
            elif move_type in (HanabiMoveType.REVEAL_COLOR, HanabiMoveType.REVEAL_RANK):
                if info_tokens <= 0:
                    continue
                    
                # Get target player
                target_offset = move.target_offset()
                target_player = (current_player + target_offset) % state.num_players
                target_hand = hands[target_player]
                
                if not target_hand:
                    continue
                
                # Get clued status for target player's hand
                try:
                    target_clued_status = self.get_other_player_clued_status(
                        state, current_player, target_player
                    )
                except Exception:
                    target_clued_status = [False] * len(target_hand)
                
                # Check what kind of clue this would be
                if move_type == HanabiMoveType.REVEAL_COLOR:
                    clue_value = move.color()
                    clue_type = "color"
                else:
                    clue_value = move.rank()
                    clue_type = "rank"
                
                # Check for save opportunities
                chop_idx = self.get_chop_index(target_hand, state, target_player)
                if chop_idx < len(target_hand):
                    chop_card = target_hand[chop_idx]
                    
                    # 5 Save - must use rank clue
                    if (move_type == HanabiMoveType.REVEAL_RANK and 
                        clue_value == 4 and  # rank 4 = "5"
                        chop_card.rank() == 4 and
                        not self.is_already_played(chop_card, fireworks)):
                        weights[move_idx] = self.save_weight
                        continue
                    
                    # 2 Save - must use rank clue
                    if (move_type == HanabiMoveType.REVEAL_RANK and 
                        clue_value == 1 and  # rank 1 = "2"
                        chop_card.rank() == 1 and
                        not self.is_already_played(chop_card, fireworks)):
                        weights[move_idx] = self.save_weight * 0.8
                        continue
                    
                    # Critical save - can use color or rank
                    if (self.is_critical(chop_card, state) and
                        not self.is_already_played(chop_card, fireworks)):
                        # Check if clue touches chop
                        if clue_type == "color" and chop_card.color() == clue_value:
                            weights[move_idx] = self.save_weight * 0.9
                            continue
                        elif clue_type == "rank" and chop_card.rank() == clue_value:
                            weights[move_idx] = self.save_weight * 0.9
                            continue
                
                # Evaluate as play clue (now considers already clued cards)
                play_clue_score = self.evaluate_play_clue(
                    target_hand, fireworks, clue_type, clue_value, target_clued_status
                )
                if play_clue_score > 0:
                    weights[move_idx] = play_clue_score
                else:
                    # Non-valuable clue
                    weights[move_idx] = 0.1
        
        # Normalize to probability distribution
        total = np.sum(weights)
        if total > 0:
            weights = weights / total
        else:
            # Fallback: uniform over legal moves
            for move in legal_moves:
                weights[state.move_to_index(move)] = 1.0 / len(legal_moves)
        
        return weights
    
    def select_move(self, state: HLEGameState) -> HanabiMove:
        """
        Select a move based on convention weights.
        Uses weighted random selection.
        """
        weights = self.get_move_weights(state)
        legal_moves = state.legal_moves()
        
        if not legal_moves:
            raise ValueError("No legal moves available")
        
        # Get indices and their weights for legal moves
        legal_indices = [state.move_to_index(m) for m in legal_moves]
        legal_weights = weights[legal_indices]
        
        # Ensure weights sum to 1
        if np.sum(legal_weights) > 0:
            legal_weights = legal_weights / np.sum(legal_weights)
        else:
            legal_weights = np.ones(len(legal_moves)) / len(legal_moves)
        
        # Weighted random selection
        chosen_idx = np.random.choice(len(legal_moves), p=legal_weights)
        return legal_moves[chosen_idx]
    
    def rollout(self, state: HLEGameState, max_depth: int = 50) -> Tuple[float, np.ndarray]:
        """
        Perform a single rollout from the given state.
        
        Args:
            state: The starting state
            max_depth: Maximum number of moves to simulate
            
        Returns:
            Tuple of (normalized_score, move_probability_distribution)
        """
        current_state = state.copy()
        
        # Get initial move probabilities before starting rollout
        initial_probs = self.get_move_weights(current_state)
        
        for _ in range(max_depth):
            if current_state.is_terminal():
                break
            
            move = self.select_move(current_state)
            current_state.apply_move(move)
        
        score = current_state.score() / current_state.max_score()
        return score, initial_probs


def _worker_rollout(args) -> Tuple[float, np.ndarray]:
    """Worker function for parallel rollouts."""
    state_copy, max_depth, play_weight, clue_weight, save_weight, discard_weight = args
    
    policy = ConventionRolloutPolicy(
        play_weight=play_weight,
        clue_weight=clue_weight, 
        save_weight=save_weight,
        discard_weight=discard_weight
    )
    
    return policy.rollout(state_copy, max_depth)


class ParallelConventionRollout:
    """
    Manages parallel execution of convention-based rollouts.
    """
    
    def __init__(self, 
                 num_workers: Optional[int] = None,
                 play_weight: float = 10.0,
                 clue_weight: float = 5.0,
                 save_weight: float = 8.0,
                 discard_weight: float = 1.0):
        """
        Initialize parallel rollout manager.
        
        Args:
            num_workers: Number of parallel workers (default: CPU count)
            play_weight: Weight for play moves
            clue_weight: Weight for clue moves
            save_weight: Weight for save clues
            discard_weight: Weight for discard moves
        """
        self.num_workers = num_workers or multiprocessing.cpu_count()
        self.play_weight = play_weight
        self.clue_weight = clue_weight
        self.save_weight = save_weight
        self.discard_weight = discard_weight
        self.policy = ConventionRolloutPolicy(
            play_weight, clue_weight, save_weight, discard_weight
        )
    
    def run_rollouts(self, state: HLEGameState, 
                     num_rollouts: int,
                     max_depth: int = 50) -> Tuple[float, np.ndarray]:
        """
        Run multiple rollouts and aggregate results.
        
        Note: Process parallelism doesn't work with HLE states (CFFI objects can't be pickled)
        and thread parallelism is slower due to Python's GIL.
        Therefore, we use sequential rollouts which are actually fastest for this use case.
        
        Args:
            state: Starting state
            num_rollouts: Number of rollouts to perform
            max_depth: Maximum depth per rollout
            
        Returns:
            Tuple of (average_score, averaged_probability_distribution)
        """
        if num_rollouts <= 0:
            raise ValueError("num_rollouts must be positive")
        
        # Use sequential rollouts - fastest for HLE states due to:
        # 1. CFFI objects can't be pickled for process parallelism
        # 2. GIL prevents thread parallelism from being effective
        return self.run_rollouts_sequential(state, num_rollouts, max_depth)
    
    def run_rollouts_sequential(self, state: HLEGameState,
                                num_rollouts: int,
                                max_depth: int = 50) -> Tuple[float, np.ndarray]:
        """
        Run multiple rollouts sequentially (for debugging or when parallel overhead is too high).
        """
        scores = []
        probs = []
        
        for _ in range(num_rollouts):
            score, prob = self.policy.rollout(state, max_depth)
            scores.append(score)
            probs.append(prob)
        
        avg_score = float(np.mean(scores))
        avg_probs = np.mean(probs, axis=0)
        
        if np.sum(avg_probs) > 0:
            avg_probs = avg_probs / np.sum(avg_probs)
        
        return avg_score, avg_probs
