
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple
from rl_hanabi.game.hle_state import HLEGameState
from hanabi_learning_environment.pyhanabi import HanabiCardKnowledge, HanabiCard, HanabiMoveType, HanabiHistoryItem

# Standard Hanabi deck composition per color: three 1s, two 2s, two 3s, two 4s, one 5.
# Ranks are 0-indexed (0=1, 1=2, ..., 4=5).
RANK_COUNTS = [3, 2, 2, 2, 1]




class BeliefState:
    """
    The public belief state (V-belief) as described in the BAD paper.
    
    It consists of:
    1. Factorized beliefs for each card in each player's hand.
    2. Knowledge of the remaining deck (implicitly, via card counts).
    """
    
    def __init__(self, state: HLEGameState):
        self.state = state
        self.num_players = state.num_players
        self.num_ranks = state.game.num_ranks()
        self.num_colors = state.game.num_colors()
        self.hand_size = state.game.hand_size()
        self.num_players = state.num_players
        self.belief_state : np.ndarray = self.init_belief_state()

    
    def get_card_index(self, color: int, rank: int) -> int:
        """Maps (color, rank) to a flat index 0-24."""
        return color * self.num_ranks + rank
    
    def get_color_mask(self, color: int) -> np.ndarray:
        """Returns a mask for all cards of a given color."""
        mask = np.zeros(self.num_colors * self.num_ranks, dtype=float)
        mask[color * self.num_ranks:(color + 1) * self.num_ranks] = 1
        return mask
    
    
    def get_rank_mask(self, rank: int) -> np.ndarray:
        """Returns a mask for all cards of a given rank."""
        mask = np.zeros(self.num_colors * self.num_ranks, dtype=float)
        mask[rank::self.num_ranks] = 1
        return mask


    def get_card_color_rank(self, index: int) -> tuple[int, int]:
        """Maps flat index 0-24 to (color, rank)."""
        return index // self.num_ranks, index % self.num_ranks
    

    def init_card_prob(self):
        """Initializes the belief for a specific card slot in a player's hand.  """
        
        probs = self.card_distribution()
        probs = probs / np.sum(probs)
        return probs


    def card_distribution(self) -> np.ndarray:
        """Returns the current distribution over all card types."""
        probs = np.tile(np.array(RANK_COUNTS, dtype=float), self.num_colors)

        # Subtract played cards (fireworks)
        fireworks = np.array(self.state.fireworks(), dtype=float)
        if fireworks.size:
            ranks = np.arange(self.num_ranks, dtype=float)  # shape (num_ranks,)
            mask = ranks < fireworks[:, None]             # shape (num_colors, num_ranks)
            probs -= mask.ravel()

        # Subtract discarded cards
        for card in self.state.discard_pile():
            idx = card.color() * self.num_ranks + card.rank()
            probs[idx] -= 1

        return probs
    

    def card_knowledge_mask(self, player_index: int):
        """
        Applies the card knowledge masks to the belief state for a given player.
        This zeros out impossible cards based on the player's knowledge.
        """
        player_observation = self.state.observation_for_player(player_index)
        card_knowledge = player_observation.card_knowledge()
        player_knowledge = card_knowledge[player_index]
        for card_index in range(self.hand_size):
            
            knowledge : HanabiCardKnowledge = player_knowledge[card_index]
            mask = np.array([knowledge.color_plausible(i) and knowledge.rank_plausible(j) 
                             for i in range(self.num_colors) 
                             for j in range(self.num_ranks)], dtype=bool)
            self.belief_state[player_index, card_index, :] *= mask
            masked_sum = self.belief_state[player_index, card_index, :].sum()
            if masked_sum > 1e-12:
                self.belief_state[player_index, card_index, :] /= masked_sum
            else:
                self.belief_state[player_index, card_index, :] /= masked_sum + 1e-12 

        
    def init_hand_prob(self):
        hand_prob = np.array([self.init_card_prob() for _ in range(self.hand_size)])
        return hand_prob
    
    def init_belief_state(self):
        belief_state = np.array([self.init_hand_prob() for _ in range(self.num_players)])
        assert belief_state.shape == (self.num_players, self.hand_size, self.num_colors * self.num_ranks)
        return belief_state

    def apply_card_count_correction(self):
        """
        Scales all hand beliefs so that the sum of probabilities for each card type
        across all slots does not exceed the available count of that card type.
        
        Uses iterative scaling with numpy for fast computation.
        """
        card_counts = self.card_distribution()  # shape: (num_card_types,)

        # First, apply knowledge masks to ensure impossible cards are zeroed out
        for player in range(self.num_players):
            self.card_knowledge_mask(player)
        
        # Reshape belief_state to (num_players * hand_size, num_card_types)
        flat_beliefs = self.belief_state.reshape(-1, self.num_colors * self.num_ranks)
        
        # Iterative scaling to satisfy constraints
        max_iterations = 10
        for _ in range(max_iterations):
            # Sum probabilities across all slots for each card type
            prob_sums = flat_beliefs.sum(axis=0)  # shape: (num_card_types,)
            
            # Compute scaling factors: min(1, count / sum) for each card type
            # Avoid division by zero
            scale_factors = np.where(
                prob_sums > 1e-12,
                np.minimum(1.0, card_counts / (prob_sums + 1e-12)),
                1.0
            )
            
            # Check if already satisfied
            if np.allclose(scale_factors, 1.0, atol=1e-6):
                break
            
            # Scale beliefs by card type
            flat_beliefs *= scale_factors  # broadcasting: (n_slots, n_types) * (n_types,)
            
            # Re-normalize each slot to sum to 1
            slot_sums = flat_beliefs.sum(axis=1, keepdims=True)
            slot_sums = np.where(slot_sums > 1e-12, slot_sums, slot_sums + 1e-12)  # Avoid division by zero
            flat_beliefs /= slot_sums
        
        # Update belief_state in place
        self.belief_state = flat_beliefs.reshape(self.num_players, self.hand_size, -1)

    def get_last_move(self) -> HanabiHistoryItem | None:
        history = self.state.state.move_history()
        if not history:
            return 
        for item in reversed(history):
            if item.move().move_type() != HanabiMoveType.DEAL:
                return item
            
        return None
            

    def update_from_move(self):
        """
        Updates the belief state based on a Hanabi move.
        
        Args:
            move: The HanabiMove taken.
            revealed_card: If the move is a play or discard, the actual card revealed (color, rank).
                           None for clue moves.
        """
        
        last_history_item = self.get_last_move()
        if last_history_item is None:
            return
        last_move = last_history_item.move()
        player_index = last_history_item.player()

        if last_move is None:
            return
        
        move_type = last_move.type()
        if move_type == HanabiMoveType.REVEAL_COLOR or move_type == HanabiMoveType.REVEAL_RANK:
            target_player = last_move.target_offset()
            clue_type = 0 if move_type == HanabiMoveType.REVEAL_COLOR else 1
            clue_value = last_move.color() if clue_type == 0 else last_move.rank()
            affected_indices = last_history_item.card_info_revealed()

            self.update_from_clue(player_index, clue_type, clue_value, affected_indices, target_player)
            self.model_update(player_index, target_player, clue_type, clue_value, affected_indices)
        
        elif move_type == HanabiMoveType.PLAY or move_type == HanabiMoveType.DISCARD:
            color, rank = last_history_item.color(), last_history_item.rank()
            hand_card_index = last_move.card_index()
            card_idx = self.get_card_index(color, rank)
            self.update_from_draw(player_index)

        self.apply_card_count_correction()


    # --- Deterministic Updates ---

    def update_from_clue(self, 
                        player_index: int, 
                        clue_type: int, 
                        clue_value: int, 
                        affected_indices: List[int],
                        target_player_offset: int):
        """
        Updates the belief state based on a clue action.
        This is a deterministic update on the constraints.
        
        clue_type: 0 for Color, 1 for Rank (matching pyhanabi/game_types)
        clue_value: 0-4 (Color index or Rank index)
        affected_indices: List of card indices in target_player's hand that match the clue.
        """
        
        target_player = (player_index + target_player_offset) % self.num_players
        not_affected_indices = np.setdiff1d(np.arange(self.hand_size), affected_indices)

        if clue_type == 0:  # Color clue
            mask = self.get_color_mask(clue_value)
        else:  # Rank clue
            mask = self.get_rank_mask(clue_value)
        
        # Update unaffected cards: zero out possible cards that match the clue
        inverse_mask = 1.0 - mask
        self.belief_state[target_player, not_affected_indices, :] *= inverse_mask
        masked_sum = self.belief_state[target_player].sum(axis=-1, keepdims=True)
        masked_sum = masked_sum if np.any(masked_sum > 1e-12) else masked_sum + 1e-12  # Avoid division by zero
        self.belief_state[target_player] /= masked_sum

        # Update affected cards: zero out impossible cards
        self.belief_state[target_player, affected_indices, :] *= mask
        masked_sum = self.belief_state[target_player].sum(axis=-1, keepdims=True)
        masked_sum = masked_sum if np.any(masked_sum > 1e-12) else masked_sum + 1e-12  # Avoid division by zero
        self.belief_state[target_player] /= masked_sum 


    def update_from_draw(self, player_index: int):
        """
        Adds a new belief for a drawn card.
        The new card is drawn from the remaining deck.
        """
        self.belief_state[player_index, self.hand_size-1, :] = self.init_card_prob()


    # --- Bayesian Update ---

    def update_from_action(self, 
                        observer: int, 
                        policy_likelihoods: np.ndarray):
        """
        Approximate Bayesian Update as described in the BAD paper.
        
        Equation: B_{t+1}(h) ∝ P(a_t | h) * B_t(h)
        
        In the factorized approximation, we update each card slot independently.
        
        Args:
            belief_state: Current belief state.
            observer: The player observing the action.
            policy_likelihoods: A list of numpy arrays, one for each card slot in the player's hand.
                                Each array has shape (NUM_CARD_TYPES,) and represents 
                                P(action | card_i = x).
                                This assumes the policy's dependence on the hand can be 
                                factorized or approximated per card.
        """
        assert policy_likelihoods.shape == (self.hand_size, self.num_colors * self.num_ranks)
        
        self.belief_state[observer] *= policy_likelihoods


    def rotate_belief_state(self, player_index: int) -> np.ndarray:
        """
        Rotates the belief state so that the next player becomes the observer.
        This is useful for updating beliefs from the perspective of the next player.
        """
        return np.roll(self.belief_state, shift=-player_index, axis=0)
    
    def model_update(self, player_index: int, target_player_off: int, clue_type: int, clue_value: int, affected_indices: List[int]):
        """
        Placeholder for model-based belief update.
        This function can be implemented to use a learned model to refine the belief state.
        """
        target_player = (player_index + target_player_off) % self.num_players
        affected_indices_one_hot = np.zeros(self.hand_size, dtype=bool)
        affected_indices_one_hot[affected_indices] = True
        clue_arr = np.array([target_player_off, clue_type, clue_value], dtype=float)
        prepared_input = self.prepare_belief_obs(player_index)
        final_input = np.concatenate([clue_arr, affected_indices_one_hot.astype(float), prepared_input])
        
        
        
        



    def prepare_belief_obs(self, player_index: int) -> np.ndarray:
        """
        Prepares the belief state as input for a model.
        The shape will be (num_players, hand_size, num_card_types).
        The belief state is rotated so that the observer is first.
        """
        rotated_belief = self.rotate_belief_state(player_index)
        other_hands = self.others_hand_to_indices(player_index)
        fireworks = np.array(self.state.fireworks())
        discard_pile = self.cards_to_indices(self.state.discard_pile())
        info_tokens = self.state.information_tokens()
        life_tokens = self.state.life_tokens()
        model_input = np.concatenate([
            rotated_belief.flatten(),
            other_hands.flatten(),
            fireworks.flatten(),
            discard_pile.flatten(),
            np.array([info_tokens, life_tokens], dtype=float)
        ])
        return model_input


    def others_hand_to_indices(self, player_index: int) -> np.ndarray:
        """
        Converts the player's hand cards to their flat indices.
        """
        observation = self.state.observation_for_player(player_index)
        hands = observation.observed_hands()
        output = []
        for i, hand in enumerate(hands, start=1):

            indices = self.cards_to_indices(hand)
            output.append(indices)
        indices = np.array(output)
        return indices
    
    def cards_to_indices(self, cards: List[HanabiCard]) -> np.ndarray:
        """
        Converts a list of HanabiCard to their flat indices.
        """
        indices = [self.get_card_index(card.color(), card.rank()) for card in cards]
        return np.array(indices)
    
    