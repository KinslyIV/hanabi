import numpy as np
import torch
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from rl_hanabi.game.hle_state import HLEGameState
from rl_hanabi.model.belief_model import ActionDecoder
from hanabi_learning_environment.pyhanabi import (
    HanabiCardKnowledge,
    HanabiCard,
    HanabiMoveType,
    HanabiHistoryItem,
)


# Standard Hanabi deck composition per color: three 1s, two 2s, two 3s, two 4s, one 5.
# Ranks are 0-indexed (0=1, 1=2, ..., 4=5).
RANK_COUNTS = np.array([3, 2, 2, 2, 1], dtype=float)


class BeliefState:
    """
    The public belief state (V-belief).

    """

    def __init__(self, state: HLEGameState, player: int, belief_model: Optional[ActionDecoder] = None):
        self.state = state
        self.player = player
        self.num_players = state.num_players
        self.num_ranks = state.game.num_ranks()
        self.num_colors = state.game.num_colors()
        self.hand_size = state.game.hand_size()
        self.num_players = state.num_players
        self.belief_model = belief_model
        # Initialize belief arrays directly
        init_color = self.init_color_prob()
        init_rank = self.init_rank_prob()
        self.color_belief: np.ndarray = np.tile(init_color, (self.num_players, self.hand_size, 1))
        self.rank_belief: np.ndarray = np.tile(init_rank, (self.num_players, self.hand_size, 1))

    def get_card_index(self, color: int, rank: int) -> int:
        """Maps (color, rank) to a flat index 0-24."""
        return color * self.num_ranks + rank

    def get_card_color_rank(self, index: int) -> tuple[int, int]:
        """Maps flat index 0-24 to (color, rank)."""
        return index // self.num_ranks, index % self.num_ranks

    def init_color_prob(self) -> np.ndarray:
        """Initializes the color probability distribution based on remaining cards."""
        color_counts = self.color_distribution()
        total = color_counts.sum()
        if total > 1e-12:
            return color_counts / total
        return np.ones(self.num_colors, dtype=float) / self.num_colors

    def init_rank_prob(self) -> np.ndarray:
        """Initializes the rank probability distribution based on remaining cards."""
        rank_counts = self.rank_distribution()
        total = rank_counts.sum()
        if total > 1e-12:
            return rank_counts / total
        return np.ones(self.num_ranks, dtype=float) / self.num_ranks

    def color_distribution(self) -> np.ndarray:
        """Returns the count of remaining cards for each color."""
        # Total cards per color: sum of RANK_COUNTS = 10 cards per color
        color_counts = np.ones(self.num_colors, dtype=float) * RANK_COUNTS.sum()

        # Subtract played cards (fireworks)
        fireworks = np.array(self.state.fireworks(), dtype=float)
        if fireworks.size:
            color_counts -= fireworks  # fireworks[i] cards of color i are played

        # Subtract discarded cards
        for card in self.state.discard_pile():
            color_counts[card.color()] -= 1

        return np.maximum(color_counts, 0)

    def rank_distribution(self) -> np.ndarray:
        """Returns the count of remaining cards for each rank."""
        # Total cards per rank: RANK_COUNTS[r] * num_colors
        rank_counts = RANK_COUNTS.copy() * self.num_colors

        # Subtract played cards (fireworks)
        fireworks = np.array(self.state.fireworks(), dtype=float)
        if fireworks.size:
            for color_idx, fw_rank in enumerate(fireworks):
                # Cards of ranks 0 to fw_rank-1 are played for this color
                for r in range(int(fw_rank)):
                    rank_counts[r] -= 1

        # Subtract discarded cards
        for card in self.state.discard_pile():
            rank_counts[card.rank()] -= 1

        return np.maximum(rank_counts, 0)

    def card_knowledge_mask(self, player_index: int):
        """
        Applies the card knowledge masks to the belief state for a given player.
        This zeros out impossible colors/ranks based on the player's knowledge.
        """
        player_observation = self.state.observation_for_player(self.player)
        card_knowledge = player_observation.card_knowledge()
        player_knowledge = card_knowledge[0]

        for card_index in range(self.hand_size):
            knowledge: HanabiCardKnowledge = player_knowledge[card_index]

            # Create color mask
            color_mask = np.array(
                [knowledge.color_plausible(i) for i in range(self.num_colors)],
                dtype=float,
            )
            self.color_belief[player_index, card_index, :] *= color_mask
            color_sum = self.color_belief[player_index, card_index, :].sum()
            if color_sum > 1e-12:
                self.color_belief[player_index, card_index, :] /= color_sum
            else:
                self.color_belief[player_index, card_index, :] = color_mask / (
                    color_mask.sum() + 1e-12
                )

            # Create rank mask
            rank_mask = np.array(
                [knowledge.rank_plausible(j) for j in range(self.num_ranks)], dtype=float
            )
            self.rank_belief[player_index, card_index, :] *= rank_mask
            rank_sum = self.rank_belief[player_index, card_index, :].sum()
            if rank_sum > 1e-12:
                self.rank_belief[player_index, card_index, :] /= rank_sum
            else:
                self.rank_belief[player_index, card_index, :] = rank_mask / (
                    rank_mask.sum() + 1e-12
                )

    def reinit_belief_state(self):
        """Reinitializes factorized belief state with separate color and rank distributions."""
        init_color = self.init_color_prob()
        init_rank = self.init_rank_prob()

        self.color_belief = np.tile(init_color, (self.num_players, self.hand_size, 1))
        self.rank_belief = np.tile(init_rank, (self.num_players, self.hand_size, 1))

        assert self.color_belief.shape == (
            self.num_players,
            self.hand_size,
            self.num_colors,
        )
        assert self.rank_belief.shape == (
            self.num_players,
            self.hand_size,
            self.num_ranks,
        )

    def apply_card_count_correction(self):
        """
        Scales all hand beliefs so that the sum of probabilities for each color/rank
        across all slots does not exceed the available count.

        Uses iterative scaling with numpy for fast computation.
        """
        color_counts = self.color_distribution()  # shape: (num_colors,)
        rank_counts = self.rank_distribution()  # shape: (num_ranks,)

        # First, apply knowledge masks to ensure impossible cards are zeroed out
        for player in range(self.num_players):
            self.card_knowledge_mask(player)

        # Reshape beliefs to (num_players * hand_size, num_colors/ranks)
        flat_color_beliefs = self.color_belief.reshape(-1, self.num_colors)
        flat_rank_beliefs = self.rank_belief.reshape(-1, self.num_ranks)

        # Iterative scaling to satisfy constraints
        max_iterations = 10
        for _ in range(max_iterations):
            converged = True

            # Scale color beliefs
            color_prob_sums = flat_color_beliefs.sum(axis=0)
            color_scale_factors = np.where(
                color_prob_sums > 1e-12,
                np.minimum(1.0, color_counts / (color_prob_sums + 1e-12)),
                1.0,
            )
            if not np.allclose(color_scale_factors, 1.0, atol=1e-6):
                converged = False
                flat_color_beliefs *= color_scale_factors
                color_slot_sums = flat_color_beliefs.sum(axis=1, keepdims=True)
                color_slot_sums = np.where(
                    color_slot_sums > 1e-12, color_slot_sums, color_slot_sums + 1e-12
                )
                flat_color_beliefs /= color_slot_sums

            # Scale rank beliefs
            rank_prob_sums = flat_rank_beliefs.sum(axis=0)
            rank_scale_factors = np.where(
                rank_prob_sums > 1e-12,
                np.minimum(1.0, rank_counts / (rank_prob_sums + 1e-12)),
                1.0,
            )
            if not np.allclose(rank_scale_factors, 1.0, atol=1e-6):
                converged = False
                flat_rank_beliefs *= rank_scale_factors
                rank_slot_sums = flat_rank_beliefs.sum(axis=1, keepdims=True)
                rank_slot_sums = np.where(
                    rank_slot_sums > 1e-12, rank_slot_sums, rank_slot_sums + 1e-12
                )
                flat_rank_beliefs /= rank_slot_sums

            if converged:
                break

        # Update belief states in place
        self.color_belief = flat_color_beliefs.reshape(self.num_players, self.hand_size, -1)
        self.rank_belief = flat_rank_beliefs.reshape(self.num_players, self.hand_size, -1)

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
            target_player_off = last_move.target_offset()
            clue_type = 0 if move_type == HanabiMoveType.REVEAL_COLOR else 1
            clue_value = last_move.color() if clue_type == 0 else last_move.rank()
            affected_indices = last_history_item.card_info_revealed()

            self.update_from_clue(
                player_index,
                clue_type,
                clue_value,
                affected_indices,
                target_player_off,
            )

            self.model_update(
                player_index,
                target_player_off,
                clue_type,
                clue_value,
                affected_indices,
            )

        elif move_type == HanabiMoveType.PLAY or move_type == HanabiMoveType.DISCARD:
            color, rank = last_history_item.color(), last_history_item.rank()
            hand_card_index = last_move.card_index()
            self.update_from_draw(player_index)

        self.apply_card_count_correction()

    # --- Deterministic Updates ---

    def update_from_clue(
        self,
        player_index: int,
        clue_type: int,
        clue_value: int,
        affected_indices: List[int],
        target_player_offset: int,
    ):
        """
        Updates the belief state based on a clue action.
        This is a deterministic update on the constraints.

        clue_type: 0 for Color, 1 for Rank
        clue_value: 0-4 (Color index or Rank index)
        affected_indices: List of card indices in target_player's hand that match the clue.
        """

        target_player = (player_index + target_player_offset) % self.num_players
        not_affected_indices = np.setdiff1d(np.arange(self.hand_size), affected_indices)

        if clue_type == 0:  # Color clue
            # Affected cards: set color to clue_value with probability 1
            self.color_belief[target_player, affected_indices, :] = 0.0
            self.color_belief[target_player, affected_indices, clue_value] = 1.0

            # Unaffected cards: zero out the clued color
            self.color_belief[target_player, not_affected_indices, clue_value] = 0.0
            # Re-normalize
            color_sums = self.color_belief[target_player, not_affected_indices, :].sum(
                axis=-1, keepdims=True
            )
            color_sums = np.where(color_sums > 1e-12, color_sums, color_sums + 1e-12)
            self.color_belief[target_player, not_affected_indices, :] /= color_sums

        else:  # Rank clue
            # Affected cards: set rank to clue_value with probability 1
            self.rank_belief[target_player, affected_indices, :] = 0.0
            self.rank_belief[target_player, affected_indices, clue_value] = 1.0

            # Unaffected cards: zero out the clued rank
            self.rank_belief[target_player, not_affected_indices, clue_value] = 0.0
            # Re-normalize
            rank_sums = self.rank_belief[target_player, not_affected_indices, :].sum(
                axis=-1, keepdims=True
            )
            rank_sums = np.where(rank_sums > 1e-12, rank_sums, rank_sums + 1e-12)
            self.rank_belief[target_player, not_affected_indices, :] /= rank_sums

    def update_from_draw(self, player_index: int):
        """
        Adds a new belief for a drawn card.
        The new card is drawn from the remaining deck.
        """
        self.color_belief[player_index, self.hand_size - 1, :] = self.init_color_prob()
        self.rank_belief[player_index, self.hand_size - 1, :] = self.init_rank_prob()

    # --- Bayesian Update ---

    def update_from_action(
        self,
        observer: int,
        color_likelihoods: np.ndarray,
        rank_likelihoods: np.ndarray,
    ):
        """
        Approximate Bayesian Update as described in the BAD paper.

        Equation: B_{t+1}(h) ∝ P(a_t | h) * B_t(h)

        In the factorized approximation, we update color and rank beliefs independently.

        Args:
            observer: The player observing the action.
            color_likelihoods: shape (hand_size, num_colors) - P(action | color_i = c)
            rank_likelihoods: shape (hand_size, num_ranks) - P(action | rank_i = r)
        """
        assert color_likelihoods.shape == (self.hand_size, self.num_colors)
        assert rank_likelihoods.shape == (self.hand_size, self.num_ranks)

        self.color_belief[observer] *= color_likelihoods
        self.rank_belief[observer] *= rank_likelihoods

        # Re-normalize
        color_sums = self.color_belief[observer].sum(axis=-1, keepdims=True)
        color_sums = np.where(color_sums > 1e-12, color_sums, color_sums + 1e-12)
        self.color_belief[observer] /= color_sums

        rank_sums = self.rank_belief[observer].sum(axis=-1, keepdims=True)
        rank_sums = np.where(rank_sums > 1e-12, rank_sums, rank_sums + 1e-12)
        self.rank_belief[observer] /= rank_sums

    def rotate_belief_state(self, player_index: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rotates the belief state so that the specified player becomes first.
        This is useful for updating beliefs from the perspective of the next player.

        Returns:
            Tuple of (rotated_color_belief, rotated_rank_belief)
        """
        rotated_color = np.roll(self.color_belief, shift=-player_index, axis=0)
        rotated_rank = np.roll(self.rank_belief, shift=-player_index, axis=0)
        return rotated_color, rotated_rank

    def encode_clue_action(
        self,
        move_player_offset: int,
        target_player_offset: int,
        clue_type: int,
        clue_value: int,
    ) -> np.ndarray:
        """
        Encodes a clue action into a vector representation.
        
        Args:
            move_player_offset: Offset of the player who gave the clue
            target_player_offset: Offset of the player who received the clue
            clue_type: 0 for Color, 1 for Rank
            clue_value: The value of the clue (color index or rank index)
        
        Returns:
            Action encoding as numpy array of shape (action_dim,)
        """
        # Simple encoding: [move_player_offset, target_player_offset, clue_type, clue_value]
        # This matches action_dim=4 in ActionDecoder
        return np.array([move_player_offset, target_player_offset, clue_type, clue_value], dtype=np.float32)

    def model_update(
        self,
        move_player_index: int,
        target_player_off: int,
        clue_type: int,
        clue_value: int,
        affected_indices: List[int],
    ):
        """
        Model-based belief update using the ActionDecoder.
        This function uses a learned model to refine the belief state based on observed actions.
        
        Args:
            move_player_index: The index of the player who made the action.
            target_player_off: The offset to the target player who received the clue based on the move player index.
            clue_type: 0 for Color, 1 for Rank
            clue_value: The value of the clue (color index or rank index)
            affected_indices: List of card indices in target player's hand that match the clue.
        """
        # Skip if no belief model is provided
        if self.belief_model is None:
            return
        
        target_player = (move_player_index + target_player_off) % self.num_players
        
        # Run model for each player to update each of their beliefs of their own hand
        for p in range(self.num_players):
            # Prepare observation from player p's perspective
            all_hands, fireworks, discard_pile_one_hot, tokens = self.prepare_belief_obs(p)
            
            # Calculate offsets from observer p's perspective
            move_player_offset = (move_player_index - p) % self.num_players
            target_player_offset = (target_player - p) % self.num_players
            
            # Encode the action
            action_encoding = self.encode_clue_action(
                move_player_offset,
                target_player_offset,
                clue_type,
                clue_value,
            )
            
            # Create affected mask for all players/slots
            affected_mask = np.zeros((self.num_players, self.hand_size), dtype=np.float32)
            affected_mask[target_player_offset, affected_indices] = 1.0
            
            # Convert numpy arrays to torch tensors
            slot_beliefs_tensor = torch.from_numpy(all_hands).float().unsqueeze(0)  # [1, P, H, C+R]
            affected_mask_tensor = torch.from_numpy(affected_mask).float().unsqueeze(0)  # [1, P, H]
            action_tensor = torch.from_numpy(action_encoding).float().unsqueeze(0)  # [1, action_dim]
            fireworks_tensor = torch.from_numpy(fireworks).float().unsqueeze(0)  # [1, C]
            discard_pile_tensor = torch.from_numpy(discard_pile_one_hot).float().unsqueeze(0)  # [1, C*R]
            target_player_tensor = torch.tensor([target_player_offset], dtype=torch.long)
            acting_player_tensor = torch.tensor([move_player_offset], dtype=torch.long)

            # For each affected slot, run the model to get updated beliefs
            with torch.no_grad():
                    
                # Run the belief model
                color_logits, rank_logits = self.belief_model(
                    slot_beliefs=slot_beliefs_tensor,
                    affected_mask=affected_mask_tensor,
                    move_target_player=target_player_tensor,
                    acting_player=acting_player_tensor,
                    action=action_tensor,
                    fireworks=fireworks_tensor,
                    discard_pile=discard_pile_tensor,
                )
                
                # Convert logits to probabilities
                color_probs = color_logits.squeeze(0).numpy()  # [hand_size, num_colors]
                rank_probs = rank_logits.squeeze(0).numpy()  # [hand_size, num_ranks]

                # Update beliefs for observer p about their own hand
                self.update_from_action(
                    observer=p,
                    color_likelihoods=color_probs,
                    rank_likelihoods=rank_probs,)
                
  

    def prepare_belief_obs(self, player_index: int):
        """
        Prepares the belief state as input for a model.
        The belief state is rotated so that the observer is first.

        Returns flattened array containing:
        - Rotated color beliefs: (num_players, hand_size, num_colors)
        - Rotated rank beliefs: (num_players, hand_size, num_ranks)
        - Other hands (as indices)
        - Fireworks
        - Discard pile
        - Info/life tokens
        """

        color_rank_belief = np.concatenate([self.color_belief[player_index], self.rank_belief[player_index]], axis=-1)
        other_hands_obs = self.others_hand_to_observation(player_index)
        all_hands = np.array([color_rank_belief] + other_hands_obs)  # Shape: (num_players, hand_size, num_colors + num_ranks)
        fireworks = np.array(self.state.fireworks())
        discard_pile = self.cards_to_indices(self.state.discard_pile())
        discard_pile_one_hot = np.zeros(self.num_colors * self.num_ranks, dtype=float)
        discard_pile_one_hot[discard_pile] = 1.0
        info_tokens = self.state.information_tokens()
        life_tokens = self.state.life_tokens()
        tokens = np.array([info_tokens, life_tokens], dtype=float)

        return all_hands, fireworks, discard_pile_one_hot, tokens

    def get_joint_probability(
        self,
        player_index: int,
        card_index: int,
        color: int,
        rank: int,
    ) -> float:
        """
        Returns the joint probability P(color, rank) for a specific card slot,
        assuming independence: P(color, rank) = P(color) * P(rank).
        """
        return (
            self.color_belief[player_index, card_index, color]
            * self.rank_belief[player_index, card_index, rank]
        )

    def get_joint_distribution(self, player_index: int, card_index: int) -> np.ndarray:
        """
        Returns the full joint distribution over (color, rank) for a card slot.
        Shape: (num_colors, num_ranks)

        Uses outer product: P(c, r) = P(c) * P(r)
        """
        return np.outer(
            self.color_belief[player_index, card_index],
            self.rank_belief[player_index, card_index],
        )

    def perfect_card_belief(self, HanabiCard: HanabiCard) -> Tuple[np.ndarray, np.ndarray]:
        """
        Converts a HanabiCard into a one-hot belief vector.
        """
        rank_belief = np.zeros(self.num_ranks, dtype=float)
        color_belief = np.zeros(self.num_colors, dtype=float)
        color_belief[HanabiCard.color()] = 1
        rank_belief[HanabiCard.rank()] = 1

        return color_belief, rank_belief

    def others_hand_to_observation(self, player_index: int) -> List[np.ndarray]:
        """
        Converts the player's observation of other's hand into a perfect belief.
        """
        player_observation = self.state.observation_for_player(player_index)
        player_to_mask_offset = (self.player - player_index) % self.num_players
        players_hands = player_observation.observed_hands()
        other_hands = []
        for p in range(1, self.num_players):
            hand_beliefs = []
            if p == player_to_mask_offset:
                # Mask this bot's hand with his own beliefs
                color_belief = self.color_belief[self.player]
                rank_belief = self.rank_belief[self.player]
                other_hands.append(np.concatenate([color_belief, rank_belief], axis=-1))
                continue
            hand_cards = players_hands[p]
            for i, card in enumerate(hand_cards):
                color_belief, rank_belief = self.perfect_card_belief(card)
                hand_beliefs.append(np.concatenate([color_belief, rank_belief]))
            other_hands.append(np.array(hand_beliefs))

        return other_hands

    def cards_to_indices(self, cards: List[HanabiCard]) -> np.ndarray:
        """
        Converts a list of HanabiCard to their flat indices.
        """
        indices = [self.get_card_index(card.color(), card.rank()) for card in cards]
        return np.array(indices)
