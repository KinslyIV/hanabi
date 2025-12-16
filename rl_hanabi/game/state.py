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
            
            # Validate before creating HanabiCard
            if not (0 <= hle_rank <= 4):
                logging.error(f"Invalid rank for HLE: {rank} -> {hle_rank}")
                return
            if not (0 <= hle_color <= 4): # Assuming 5 colors
                logging.error(f"Invalid color for HLE: {c} -> {hle_color}")
                return

            try:
                card = pyhanabi.HanabiCard(hle_color, hle_rank)

                logging.info(f"Setting HLE card for player {player_index} hand index {hle_hand_idx} to {card}")
                
                # With player_shift=0, HLE player index = website player index
                hle_player_idx = player_index
                
                # Get the hand from HLE
                hle_hand = self.hle_state.state.player_hands()[hle_player_idx]

                logging.info(f"HLE player {hle_player_idx} hand before update: {hle_hand}")
                
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
        may not match the clue. This method reassigns cards to ensure consistency.
        
        Algorithm:
        1. Calculate the pool of available cards (deck minus played/discarded/other hands)
        2. The target's current hand cards are considered "returned to pool" for reassignment
        3. Find valid card candidates for each slot based on the clue constraints
        4. Use backtracking to find a consistent assignment
        5. Apply the new assignment to HLE
        
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

        logging.debug(f"HLE clued indices (0=oldest): {hle_clued_indices}")

        # Calculate available cards (the "hidden" pool that could be in our hand)
        total_counts = Counter()
        num_colors = self.hle_state.game.num_colors()
        num_ranks = self.hle_state.game.num_ranks()
        
        all_card_types = []
        for c in range(num_colors):
            for r in range(num_ranks):
                # Get actual card count from game (handles variants correctly)
                count = self.hle_state.game.num_cards(c, r)
                total_counts[(c, r)] = count
                all_card_types.append((c, r))
        
        # Subtract visible
        # Discards
        for card in self.hle_state.state.discard_pile():
            total_counts[(card.color(), card.rank())] -= 1
        
        # Fireworks
        fireworks = self.hle_state.state.fireworks()
        for c, top_r in enumerate(fireworks):
            for r in range(top_r):
                total_counts[(c, r)] -= 1
        
        # Other players' hands (NOT the target)
        hands = self.hle_state.state.player_hands()
        for p_idx, hand in enumerate(hands):
            if p_idx != hle_target:
                for card in hand:
                    total_counts[(card.color(), card.rank())] -= 1
        
        # Current target hand in HLE
        target_hand = hands[hle_target]
        
        logging.debug(f"HLE target hand before fix: {[str(c) for c in target_hand]}")
        logging.debug(f"Available pool (excluding target hand): {dict((k,v) for k,v in total_counts.items() if v > 0)}")
        
        # IMPORTANT: The target hand's current cards are NOT subtracted from total_counts yet.
        # This is intentional - we're treating them as "returned to the pool" so we can
        # reassign consistent cards. However, we need to track them separately to ensure
        # the final assignment doesn't use more copies than exist in the deck.
        
        # Helper to check if card fits constraint
        def fits(c_idx, r_idx, slot_idx):
            if not self.hle_state:
                return False
            
            # We only check constraints from the CURRENT clue, not previous knowledge.
            # Previous knowledge in HLE might be based on incorrect card assignments,
            # so we ignore it and only enforce the new clue constraints.
            
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

        # Find candidates for each slot
        slot_candidates = []
        for i in range(len(target_hand)):
            # Check existing card first
            current_card = target_hand[i]
            current_cr = (current_card.color(), current_card.rank())
            
            possible_types = []
            for (c, r) in all_card_types:
                if total_counts[(c, r)] > 0 and fits(c, r, i):
                    possible_types.append((c, r))
            
            # Sort candidates: put current card first if it's in the list
            if current_cr in possible_types:
                possible_types.remove(current_cr)
                possible_types.insert(0, current_cr)
            
            slot_candidates.append(possible_types)

        # Solve assignment using backtracking
        # Sort slots by number of candidates to fail fast / handle constraints
        sorted_indices = sorted(range(len(target_hand)), key=lambda k: len(slot_candidates[k]))
        
        final_assignment = [None] * len(target_hand)
        
        def solve(idx_in_sorted):
            if idx_in_sorted == len(target_hand):
                return True
            
            original_idx = sorted_indices[idx_in_sorted]
            candidates = slot_candidates[original_idx]
            
            for (c, r) in candidates:
                if total_counts[(c, r)] > 0:
                    total_counts[(c, r)] -= 1
                    final_assignment[original_idx] = pyhanabi.HanabiCard(c, r) # type: ignore
                    
                    if solve(idx_in_sorted + 1):
                        return True
                    
                    # Backtrack
                    total_counts[(c, r)] += 1
                    final_assignment[original_idx] = None
            
            return False

        if solve(0):
            # Apply changes to HLE
            changes_made = 0
            for i, card in enumerate(final_assignment):
                if card is None: continue
                if card.color() != target_hand[i].color() or card.rank() != target_hand[i].rank():
                    self.hle_state.state.set_hand_card(hle_target, i, card)
                    logging.debug(f"Fixed slot {i}: {target_hand[i]} -> {card}")
                    changes_made += 1
            
            # Log final hand
            new_hand = self.hle_state.state.player_hands()[hle_target]
            logging.debug(f"HLE target hand after fix ({changes_made} changes): {[str(c) for c in new_hand]}")
        else:
            # This should never happen if the algorithm is correct
            clue_type_str = "COLOUR" if clue_type == CLUE.COLOUR else "RANK"
            logging.error(
                f"Could not find consistent hand for clue fix! "
                f"Clue: {clue_type_str}={value}, clued_indices={hle_clued_indices}, "
                f"target_hand={[str(c) for c in target_hand]}, "
                f"slot_candidates={slot_candidates}, "
                f"remaining_counts={dict((k,v) for k,v in total_counts.items() if v > 0)}"
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
