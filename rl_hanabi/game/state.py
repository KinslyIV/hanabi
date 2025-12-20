from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import List, Optional, Dict, Any

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
                self.options, self.num_players
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
    def apply_status(self, clues: int, lives: Optional[int] = None, score: Optional[int] = None) -> None:
        self.clue_tokens = clues
        
        # Sync HLE state to avoid illegal move warnings
        if self.enable_hle and self.hle_state:
            if self.hle_state.state.information_tokens() != clues:
                self.hle_state.state.set_information_tokens(clues)
            
            if lives is not None:
                if self.hle_state.state.life_tokens() != lives:
                    self.hle_state.state.set_life_tokens(lives)

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
        """Update an HLE card from a card dictionary.
        
        This method parses the card dictionary, validates the values, and delegates
        to hle_state.set_hand_card_from_color_rank() for the actual HLE operation.
        """
        if not (self.enable_hle and self.hle_state):
            return
        
        logging.info(f"Updating HLE card for player {player_index} hand index {hle_hand_idx} with {card_dict}")

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
            
            # Validate before calling HLE
            if not (0 <= hle_rank <= 4):
                logging.error(f"Invalid rank for HLE: {rank} -> {hle_rank}")
                return
            if not (0 <= hle_color <= 4):  # Assuming 5 colors
                logging.error(f"Invalid color for HLE: {c} -> {hle_color}")
                return

            # With player_shift=0, HLE player index = website player index
            hle_player_idx = player_index
            
            # Delegate to HLE state for the actual operation
            self.hle_state.set_hand_card_from_color_rank(
                hle_player_idx, hle_hand_idx, hle_color, hle_rank
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

            # Check if this was a failed play (which results in a discard event on the website)
            failed_play = False
            if card_dict and card_dict.get("failed") is True:
                failed_play = True

            if failed_play:
                self.hle_state.apply_play_from_hand_index(hle_idx)
            else:
                self.hle_state.apply_discard_from_hand_index(hle_idx)
        self._remove_order(player_index, order)

    def _fix_hand_for_clue(self, target: int, clue_type: int, value: int, clued_orders: List[int]) -> None:
        """Fix the target player's hand in HLE to be consistent with the received clue.
        
        When we (the bot) receive a clue, the HLE state has random cards in our hand that
        may not match the clue. This method prepares the parameters and delegates to
        hle_state.fix_hand_for_clue() for the actual HLE operations.
        
        This should ALWAYS succeed because we're just shuffling which unknown cards are where,
        and the clue tells us real information about the actual game state.
        """
        if not (self.enable_hle and self.hle_state):
            return

        # Normalize value to pyhanabi standards (0-4 for both)
        if clue_type == CLUE.RANK:
            clue_val_hle = value - 1
        else:
            clue_val_hle = value

        # With player_shift=0, HLE player index = website player index
        hle_target = target

        clue_type_str = "COLOUR" if clue_type == CLUE.COLOUR else "RANK"
        logging.debug(
            f"_fix_hand_for_clue: target={target}, clue={clue_type_str}={value} (hle_val={clue_val_hle}), "
            f"clued_orders={clued_orders}, our_hand_orders={self.hands[target]}"
        )

        # Convert clued_orders (card unique IDs) to hand indices
        # self.hands[target] has orders with 0=Newest, then convert to HLE indices (0=Oldest)
        hand_size = len(self.hands[target])
        hle_clued_indices = set()
        for order in clued_orders:
            # Find the index of this order in the target's hand
            try:
                hand_idx = self.hands[target].index(order)
                # Convert from website index (0=Newest) to HLE index (0=Oldest)
                hle_idx = hand_size - 1 - hand_idx
                hle_clued_indices.add(hle_idx)
            except ValueError:
                logging.warning(f"Clue order {order} not found in target's hand: {self.hands[target]}")

        # Delegate to HLE state for the actual fix
        self.hle_state.fix_hand_for_clue(
            hle_target=hle_target,
            clue_type=clue_type,
            clue_val_hle=clue_val_hle,
            hle_clued_indices=hle_clued_indices,
            clue_type_constant_colour=CLUE.COLOUR,
            clue_type_constant_rank=CLUE.RANK,
        )

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
            self.hle_state.apply_color_clue(target, value, giver)
        elif clue_type == CLUE.RANK:
            # Hanab.live ranks are 1-based; pyhanabi expects 0-based.
            self.hle_state.apply_rank_clue(target, value - 1, giver)

    # --- Random baseline policy (unchanged) ---
    def random_action(self) -> Optional[PerformAction]:
        import random

        if not self.our_hand:
            return None

        # To avoid illegal moves (e.g., discarding when clues are full),
        # this baseline always plays a random card from our hand.
        target_order = random.choice(self.our_hand)

        return PerformAction(_type=ACTION.PLAY, target=target_order)
