from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import List, Optional, Dict, Any
from collections import Counter

from hanabi_learning_environment import pyhanabi
from .game_types import ACTION, CLUE, PerformAction
from .hle_state import HLEGameState


MAX_CLUES = 8


@dataclass
class GameState:
    player_names: List[str]
    our_index: int
    num_players: int
    options: dict
    enable_hle: bool = True

    # Shadow HLE environment (created at game start, advanced with actions)
    hle_state: Optional[HLEGameState] = field(init=False, repr=False)

    # dynamic state (what we mirror from the server)
    turn_count: int = 0
    clue_tokens: int = MAX_CLUES
    strikes: int = 0
    current_player_index: int = 0
    hands: List[List[int]] = field(default_factory=list)  # list of orders per player
    in_progress: bool = False

    def __post_init__(self) -> None:
        if not self.hands:
            self.hands = [[] for _ in range(self.num_players)]
        
        # Initialize current player from options
        self.current_player_index = self.options.get("startingPlayer", 0)

        # Create a new HLE game/state matching the table options if enabled.
        if self.enable_hle:
            self.hle_state = HLEGameState.from_table_options(
                self.options, self.num_players, self.our_index
            )
        else:
            self.hle_state = None

    def __str__(self) -> str:
        return (
            f"GameState(turn={self.turn_count}, clues={self.clue_tokens}, "
            f"strikes={self.strikes}, current_player={self.current_player_index}, "
            f"hands={self.hands}, state_hle=\n{self.hle_state}\n)"
        )

    @property
    def our_hand(self) -> List[int]:
        return self.hands[self.our_index]

    # --- Helpers for mapping orders <-> hand indices ---
    def _hand_index_from_order(self, player_index: int, order: int) -> Optional[int]:
        """Map a hanab.live card order to a slot index in that player's hand."""
        try:
            return self.hands[player_index].index(order)
        except ValueError:
            return None

    # Expose a copy of the HLE state for MCTS
    def hle_state_for_mcts(self) -> HLEGameState:
        if not self.hle_state:
            raise RuntimeError("HLE state is disabled for this GameState")
        return self.hle_state.copy()

    # --- Event application helpers (from server) ---
    def apply_status(self, clues: int) -> None:
        self.clue_tokens = clues
        # HLE manages its own info tokens via moves; we do not override them here.

    def apply_turn(self, num: int, current_player_index: int) -> None:
        self.turn_count = num
        self.current_player_index = current_player_index

    def apply_draw(self, player_index: int, card_dict: Dict[str, Any]) -> None:
        # Local mirror of hand composition; HLE draws when play/discard is applied.
        # Match hanab.live bot semantics: newest card at the front.
        order = card_dict.get("order", -1)
        self.hands[player_index].insert(0, order)

        if self.enable_hle and self.hle_state:
            # If we have card info (color/rank), update the HLE state.
            # The drawn card corresponds to the last filled slot in the HLE hand.
            hand_idx = len(self.hands[player_index]) - 1
            if card_dict:
                self._update_hle_card_from_dict(player_index, hand_idx, card_dict)


    def _update_hle_card_from_dict(self, player_index: int, hle_hand_idx: int, 
                                   card_dict: Dict[str, Any]) -> None:
        if not (self.enable_hle and self.hle_state):
            return

        # Try to find suit/color
        c = card_dict.get("suitIndex")
        if c is None:
            c = card_dict.get("suit")
        if c is None:
            c = card_dict.get("color")
            
        rank = card_dict.get("rank")
        
        if c is not None and rank is not None:
            # If card is unknown (e.g. our own draw), we skip updating HLE.
            if int(rank) == -1 or int(c) == -1:
                return

            # Convert to pyhanabi format
            # hanab.live rank is 1-5
            hle_rank = int(rank) - 1
            hle_color = int(c)
            
            # Validate before creating HanabiCard
            if not (0 <= hle_rank <= 4):
                logging.error(f"Invalid rank for HLE: {rank} -> {hle_rank}")
                return
            if not (0 <= hle_color <= 4): # Assuming 5 colors
                logging.error(f"Invalid color for HLE: {c} -> {hle_color}")
                return

            try:
                card = pyhanabi.HanabiCard(hle_color, hle_rank)
                
                # Map player index to HLE index
                hle_player_idx = (player_index - self.hle_state.player_shift) % self.hle_state.num_players
                
                # Get the hand from HLE
                hle_hand = self.hle_state.state.player_hands()[hle_player_idx]
                
                # Ensure we don't go out of bounds of HLE hand
                if 0 <= hle_hand_idx < len(hle_hand):
                    self.hle_state.state.set_hand_card(hle_player_idx, hle_hand_idx, card)
                else:
                    logging.warning(f"HLE hand index out of bounds: {hle_hand_idx} for hand size {len(hle_hand)}")

            except Exception as e:
                logging.warning(
                    f"Failed to set card in HLE for player {player_index} card {card_dict}: {e}"
                )

    def _remove_order(self, player_index: int, order: int) -> None:
        try:
            self.hands[player_index].remove(order)
        except ValueError:
            pass

    def apply_play(self, order: int, player_index: int, card_dict: Optional[Dict[str, Any]] = None) -> None:
        # Mirror into HLE using the slot index, then update local hands.
        idx = self._hand_index_from_order(player_index, order)
        if self.enable_hle and self.hle_state is not None and idx is not None:
            # idx is index in self.hands (0=Newest). HLE expects 0=Oldest.
            hand_size = len(self.hands[player_index])
            hle_idx = hand_size - 1 - idx
            
            if card_dict:
                self._update_hle_card_from_dict(player_index, hle_idx, card_dict)

            self.hle_state.apply_play_from_hand_index(hle_idx)
        self._remove_order(player_index, order)

    def apply_discard(self, order: int, player_index: int, card_dict: Optional[Dict[str, Any]] = None) -> None:
        idx = self._hand_index_from_order(player_index, order)
        if self.enable_hle and self.hle_state is not None and idx is not None:
            # idx is index in self.hands (0=Newest). HLE expects 0=Oldest.
            hand_size = len(self.hands[player_index])
            hle_idx = hand_size - 1 - idx
            
            if card_dict:
                self._update_hle_card_from_dict(player_index, hle_idx, card_dict)

            self.hle_state.apply_discard_from_hand_index(hle_idx)
        self._remove_order(player_index, order)

    def _fix_hand_for_clue(self, target: int, clue_type: int, value: int, clued_indices: List[int]) -> None:
        if not (self.enable_hle and self.hle_state):
            return

        # Normalize value to pyhanabi standards (0-4 for both)
        if clue_type == CLUE.RANK:
            clue_val_hle = value - 1
        else:
            clue_val_hle = value

        # Map target to HLE index
        hle_target = (target - self.hle_state.player_shift) % self.hle_state.num_players

        # Convert clued_indices (0=Newest) to HLE indices (0=Oldest)
        hand_size = len(self.hands[target])
        hle_clued_indices = set()
        for idx in clued_indices:
            hle_idx = hand_size - 1 - idx
            hle_clued_indices.add(hle_idx)

        # Calculate available cards (Hidden set)
        total_counts = Counter()
        num_colors = self.hle_state.game.num_colors()
        num_ranks = self.hle_state.game.num_ranks()
        
        for c in range(num_colors):
            for r in range(num_ranks):
                # 1:3, 2:2, 3:2, 4:2, 5:1
                count = 0
                if r == 0: count = 3
                elif r == 4: count = 1
                else: count = 2
                total_counts[(c, r)] = count
        
        # Subtract visible
        # Discards
        for card in self.hle_state.state.discard_pile():
            total_counts[(card.color(), card.rank())] -= 1
        
        # Fireworks
        fireworks = self.hle_state.state.fireworks()
        for c, top_r in enumerate(fireworks):
            for r in range(top_r):
                total_counts[(c, r)] -= 1
        
        # Other players' hands
        hands = self.hle_state.state.player_hands()
        for p_idx, hand in enumerate(hands):
            if p_idx != hle_target:
                for card in hand:
                    total_counts[(card.color(), card.rank())] -= 1
        
        # Current target hand in HLE
        target_hand = hands[hle_target]
        new_hand : list[Optional[pyhanabi.HanabiCard]] = [None] * len(target_hand)
        
        # Helper to check if card fits constraint
        def fits(c_idx, r_idx, slot_idx):
            if not self.hle_state:
                return False
            
            # Check positive knowledge using observation
            hle_our_idx = (self.our_index - self.hle_state.player_shift) % self.hle_state.num_players
            obs = self.hle_state.state.observation(hle_our_idx)
            know = obs.card_knowledge()[hle_target][slot_idx]
            
            if know.color() is not None and know.color() != c_idx: return False
            if know.rank() is not None and know.rank() != r_idx: return False
            
            # Check clue constraints
            is_clued = slot_idx in hle_clued_indices
            
            if clue_type == CLUE.COLOUR:
                if is_clued:
                    if c_idx != clue_val_hle: return False
                else:
                    if c_idx == clue_val_hle: return False
            elif clue_type == CLUE.RANK:
                if is_clued:
                    if r_idx != clue_val_hle: return False
                else:
                    if r_idx == clue_val_hle: return False
            
            return True

        # First pass: Try to keep existing cards
        for i, card in enumerate(target_hand):
            c, r = card.color(), card.rank()
            if fits(c, r, i) and total_counts[(c, r)] > 0:
                new_hand[i] = card
                total_counts[(c, r)] -= 1
        
        # Second pass: Fill missing slots
        for i in range(len(target_hand)):
            if new_hand[i] is None:
                # Find a card in total_counts that fits
                found = False
                for (c, r), count in total_counts.items():
                    if count > 0 and fits(c, r, i):
                        # Found one
                        card = pyhanabi.HanabiCard(c, r)
                        new_hand[i] = card 
                        total_counts[(c, r)] -= 1
                        found = True
                        break
                
                if not found:
                    logging.warning(f"Could not find consistent card for slot {i} in clue fix. Keeping original.")
                    new_hand[i] = target_hand[i] # Fallback
                    
        # Apply changes to HLE
        for i, card in enumerate(new_hand):
            # Compare content, not just object identity
            if card is None:
                continue
            if card.color() != target_hand[i].color() or card.rank() != target_hand[i].rank():
                self.hle_state.state.set_hand_card(hle_target, i, card)

    def apply_clue(self, giver: int, target: int, clue_type: int, value: int, 
                   action: Optional[Dict[str, Any]] = None) -> None:
        """Apply a clue action into the HLE shadow state.

        The random baseline bot does not currently use clues, but we still
        mirror them so the HLE state stays as close as possible to the
        website game.
        """
        if not (self.enable_hle and self.hle_state is not None):
            return

        # If we are the target, our hand is random/unknown in HLE.
        # We must ensure it is consistent with the clue we just received.
        if target == self.our_index and action:
            clued_list = action.get("list", [])
            self._fix_hand_for_clue(target, clue_type, value, clued_list)

        if clue_type == CLUE.COLOUR:
            self.hle_state.apply_color_clue(target, value)
        elif clue_type == CLUE.RANK:
            # Hanab.live ranks are 1-based; pyhanabi expects 0-based.
            self.hle_state.apply_rank_clue(target, value - 1)

    # --- Random baseline policy (unchanged) ---
    def random_action(self) -> Optional[PerformAction]:
        import random

        if not self.our_hand:
            return None

        # To avoid illegal moves (e.g., discarding when clues are full),
        # this baseline always plays a random card from our hand.
        target_order = random.choice(self.our_hand)

        return PerformAction(_type=ACTION.PLAY, target=target_order)
