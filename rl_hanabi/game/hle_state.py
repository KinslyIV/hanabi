from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import logging

import numpy as np
from hanabi_learning_environment import pyhanabi

logger = logging.getLogger(__name__)

@dataclass
class HLEGameState:
    """Wrapper around pyhanabi.HanabiState used for MCTS simulations.

    This state is a shadow environment that receives the same sequence of
    public moves as the Hanab.live game (play/discard/clues). It is then
    copied and rolled forward inside MCTS.
    """

    game: pyhanabi.HanabiGame
    state: pyhanabi.HanabiState

    # ----- construction -------------------------------------------------

    @classmethod
    def from_table_options(
        cls,
        options: Dict[str, Any],
        num_players: int,
    ) -> "HLEGameState":
        """Create a fresh HanabiGame + initial HanabiState.

        IMPORTANT: We use player_shift=0, meaning HLE player indices equal website player indices.
        HLE player 0 = website player 0, etc.
        
        However, HLE always starts with cur_player() = 0, while website may have startingPlayer != 0.
        This means HLE and website may have different "current players" at the start.
        
        We handle this by applying actions from the website in order. Since every action
        (play/discard/clue) advances the turn in HLE, and the website sends actions in order,
        the HLE state will naturally align with the website state after each action.
        
        The key is that we don't try to "pre-sync" HLE to the website's starting player.
        Instead, we let the action stream do the synchronization.
        """
        starting_player = int(options.get("startingPlayer", 0))

        params = {
            "players": num_players,
            "colors": int(options.get("numSuits", 5)),
            "ranks": int(options.get("numRanks", 5)),
            "hand_size": int(options.get("cardsPerHand", 5)),
            "max_information_tokens": int(options.get("clueTokens", 8)),
            "max_life_tokens": int(options.get("strikeTokens", 3)),
            "seed": int(options.get("seed", -1)),
            "random_start_player": False,
            "observation_type": pyhanabi.AgentObservationType.CARD_KNOWLEDGE,
        }
        game = pyhanabi.HanabiGame(params)
        state = game.new_initial_state()
        
        # Advance through initial chance nodes (dealing cards)
        while state.cur_player() == pyhanabi.CHANCE_PLAYER_ID:
            state.deal_random_card()
        
        # Use player_shift=0: HLE player indices = website player indices
        # Note: HLE starts with cur_player()=0 regardless of website's startingPlayer.
        # This will be aligned after processing the game's action history.
        logger.info(f"HLE initialized: website startingPlayer={starting_player}, HLE cur_player={state.cur_player()}, player_shift=0")
            
        return cls(game=game, state=state)
    
    def __repr__(self) -> str:
        return self.state.__repr__()

    # ----- basic API for MCTS ------------------------------------------

    def copy(self) -> "HLEGameState":
        """Deep copy for use as a root in MCTS."""
        return HLEGameState(
            game=self.game,
            state=self.state.copy()
        )

    @property
    def num_players(self) -> int:
        return self.state.num_players()

    @property
    def clue_tokens(self) -> int:
        return self.state.information_tokens()

    @property
    def strikes(self) -> int:
        return self.state.life_tokens()
    
    def deck_size(self) -> int:
        return self.state.deck_size()
    
    def life_tokens(self) -> int:
        return self.state.life_tokens()
    
    def information_tokens(self) -> int:
        return self.state.information_tokens()
    
    def get_hands(self):
        return self.state.player_hands()

    @property
    def current_player_index(self) -> int:
        # With player_shift=0, HLE player index = website player index
        return self.state.cur_player()
    
    def discard_pile(self) -> List[pyhanabi.HanabiCard]:
        return self.state.discard_pile()

    def is_terminal(self) -> bool:
        return self.state.is_terminal()

    def score(self) -> int:
        sccore = 0
        for f in self.state.fireworks():
            sccore += f
        return sccore

    def legal_moves(self) -> List[pyhanabi.HanabiMove]:
        return self.state.legal_moves()
    
    def fireworks(self) -> List[int]:
        return self.state.fireworks()
    
    # ----- action space helpers -----------------------------------------

    @property
    def action_space_size(self) -> int:
        """
        Total number of possible moves in the game configuration.
        Layout:
        0..H-1: Discard
        H..2H-1: Play
        2H..: Color Clues ( (N-1)*C )
        ... : Rank Clues ( (N-1)*R )
        """
        H = self.game.hand_size()
        N = self.game.num_players()
        C = self.game.num_colors()
        R = self.game.num_ranks()
        return 2 * H + (N - 1) * C + (N - 1) * R

    def move_to_index(self, move: pyhanabi.HanabiMove) -> int:
        """Map a move to a deterministic index in [0, action_space_size)."""
        move_type = move.type()
        H = self.game.hand_size()
        
        if move_type == pyhanabi.HanabiMoveType.DISCARD:
            return move.card_index()
            
        if move_type == pyhanabi.HanabiMoveType.PLAY:
            return H + move.card_index()
            
        # Clues
        N = self.game.num_players()
        C = self.game.num_colors()
        R = self.game.num_ranks()
        offset = move.target_offset()  # 1 to N-1
        
        if move_type == pyhanabi.HanabiMoveType.REVEAL_COLOR:
            base = 2 * H
            return base + (offset - 1) * C + move.color()
            
        if move_type == pyhanabi.HanabiMoveType.REVEAL_RANK:
            base = 2 * H + (N - 1) * C
            return base + (offset - 1) * R + move.rank()
            
        raise ValueError(f"Unknown move type: {move}")

    def index_to_move(self, index: int) -> pyhanabi.HanabiMove:
        """Map a deterministic index back to a move."""
        H = self.game.hand_size()
        
        # Discard
        if index < H:
            return pyhanabi.HanabiMove.get_discard_move(index)
        index -= H
        
        # Play
        if index < H:
            return pyhanabi.HanabiMove.get_play_move(index)
        index -= H
        
        N = self.game.num_players()
        C = self.game.num_colors()
        
        # Reveal Color
        num_color_moves = (N - 1) * C
        if index < num_color_moves:
            offset = (index // C) + 1
            color = index % C
            return pyhanabi.HanabiMove.get_reveal_color_move(offset, color)
        index -= num_color_moves
        
        # Reveal Rank
        R = self.game.num_ranks()
        num_rank_moves = (N - 1) * R
        if index < num_rank_moves:
            offset = (index // R) + 1
            rank = index % R
            return pyhanabi.HanabiMove.get_reveal_rank_move(offset, rank)
            
        raise ValueError(f"Index {index} out of bounds for action space")
    
    def apply_move_by_index(self, index: int) -> None:
        move = self.index_to_move(index)
        self.apply_move(move)

    def max_score(self) -> int:
        """Get the maximum possible score for the current game configuration."""
        return self.game.num_colors() * self.game.num_ranks()

    def legal_moves_mask(self) -> np.ndarray:
        """Get a binary mask of legal moves over the full action space."""
        mask = np.zeros(self.action_space_size, dtype=np.bool)
        for move in self.state.legal_moves():
            mask[self.move_to_index(move)] = True
        return mask

    def apply_move(self, move: pyhanabi.HanabiMove) -> None:
        self.state.apply_move(move)
        self._advance_chance_nodes()

    def _advance_chance_nodes(self) -> None:
        while self.state.cur_player() == pyhanabi.CHANCE_PLAYER_ID:
            self.state.deal_random_card()

    def safe_apply_move(self, move: pyhanabi.HanabiMove) -> None:
        """Safely apply a move, checking for legality to avoid crashes.
        
        If the move is illegal (due to state divergence), we try to recover:
        - If Discard is illegal (e.g. 8 clues), we Play instead (to consume card).
        - If Play is illegal (unlikely), we Discard instead.
        - If Clue is illegal (e.g. 0 clues), we skip it (no card consumed).
        """

        logger.info(f"Current HLE state before move: \n{self.state}\n")
        logger.info(f"Applying move safely: \n{move}\n")
        
        # Check if move is legal in the current pyhanabi state
        is_legal = False
        is_legal = self.state.move_is_legal(move)
        if is_legal:
            self.state.apply_move(move)
            self._advance_chance_nodes()
            return

        logger.warning(f"Illegal move detected in HLE shadow state: {move}. Attempting recovery.")
        
        move_type = move.type()
        
        # Recovery strategy:
        # If we need to consume a card (Play/Discard), we MUST do something that consumes that card index.
        if move_type == pyhanabi.HanabiMoveType.DISCARD:
            # Try to Play instead
            alt_move = pyhanabi.HanabiMove.get_play_move(move.card_index())
            if self._is_legal(alt_move):
                logger.warning(f"Recovering by PLAYING card {move.card_index()} instead of DISCARDING.")
                self.apply_move(alt_move)
                return
                
        elif move_type == pyhanabi.HanabiMoveType.PLAY:
            # Try to Discard instead
            alt_move = pyhanabi.HanabiMove.get_discard_move(move.card_index())
            if self._is_legal(alt_move):
                logger.warning(f"Recovering by DISCARDING card {move.card_index()} instead of PLAYING.")
                self.apply_move(alt_move)
                return

        # If it's a clue, or we couldn't swap Play/Discard, we just skip it.
        # Skipping a Play/Discard is bad because indices will shift, but better than crashing.
        logger.error(f"Could not recover from illegal move {move}. Skipping. State may be desynced.")

    def _is_legal(self, move: pyhanabi.HanabiMove) -> bool:
        for lm in self.state.legal_moves():
            if self.moves_are_equal(move, lm):
                return True
        return False

    @staticmethod
    def moves_are_equal(move1: pyhanabi.HanabiMove, move2: pyhanabi.HanabiMove) -> bool:
        """Check if two HanabiMove objects represent the same action."""
        if move1.type() != move2.type():
            return False
        
        if move1.type() in (pyhanabi.HanabiMoveType.PLAY, pyhanabi.HanabiMoveType.DISCARD):
            return move1.card_index() == move2.card_index() 
            
        if move1.type() == pyhanabi.HanabiMoveType.REVEAL_COLOR:
            return (move1.target_offset() == move2.target_offset()) and (move1.color() == move2.color())
            
        if move1.type() == pyhanabi.HanabiMoveType.REVEAL_RANK:
            return (move1.target_offset() == move2.target_offset()) and (move1.rank() == move2.rank())
            
        if move1.type() == pyhanabi.HanabiMoveType.DEAL:
            # Chance moves are considered equal if they deal the same card to the same player
            # But usually we don't compare chance moves from outside.
            # Just return True if types match for safety, or check details if possible.
            return True

        raise ValueError(f"Unknown move type: {move1}")

    # ----- helpers used by GameState to mirror website actions ---------

    def set_hand_card_from_color_rank(self, hle_player_idx: int, hle_hand_idx: int, 
                                       hle_color: int, hle_rank: int) -> None:
        """Set a card in a player's hand to a specific color and rank.
        
        Args:
            hle_player_idx: The player index in HLE
            hle_hand_idx: The hand index in HLE (0=oldest)
            hle_color: The color index (0-4)
            hle_rank: The rank index (0-4)
        """
        try:
            card = pyhanabi.HanabiCard(hle_color, hle_rank)
            
            logger.info(f"Setting HLE card for player {hle_player_idx} hand index {hle_hand_idx} to {card}")
            
            # Get the hand from HLE
            hle_hand = self.state.player_hands()[hle_player_idx]
            
            logger.info(f"HLE player {hle_player_idx} hand before update: {hle_hand}")
            
            # Ensure we don't go out of bounds of HLE hand
            if 0 <= hle_hand_idx < len(hle_hand):
                self.state.set_hand_card(hle_player_idx, hle_hand_idx, card)
            else:
                logger.warning(f"HLE hand index out of bounds: {hle_hand_idx} for hand size {len(hle_hand)}")
                
        except Exception as e:
            logger.warning(
                f"Failed to set card in HLE for player {hle_player_idx} at index {hle_hand_idx} "
                f"with color={hle_color}, rank={hle_rank}: {e}"
            )

    def apply_play_from_hand_index(self, card_index: int) -> None:
        """Apply a PLAY move for the current player using a hand index."""
        move = pyhanabi.HanabiMove.get_play_move(card_index)
        self.safe_apply_move(move)

    def apply_discard_from_hand_index(self, card_index: int) -> None:
        """Apply a DISCARD move for the current player using a hand index."""
        move = pyhanabi.HanabiMove.get_discard_move(card_index)
        self.safe_apply_move(move)

    def fix_hand_for_clue(
        self,
        hle_target: int,
        clue_type: int,
        clue_val_hle: int,
        hle_clued_indices: set,
        clue_type_constant_colour: int,
        clue_type_constant_rank: int,
    ) -> None:
        """Fix the target player's hand to be consistent with the received clue.
        
        When the bot receives a clue, the HLE state has random cards in the hand that
        may not match the clue. This method reassigns cards to ensure consistency.
        
        Algorithm:
        1. Calculate the pool of available cards (deck minus played/discarded/other hands)
        2. The target's current hand cards are considered "returned to pool" for reassignment
        3. Find valid card candidates for each slot based on the clue constraints
        4. Use backtracking to find a consistent assignment
        5. Apply the new assignment to HLE
        
        Args:
            hle_target: The target player index in HLE
            clue_type: The type of clue (COLOUR or RANK constant from game_types)
            clue_val_hle: The clue value in HLE format (0-4)
            hle_clued_indices: Set of HLE hand indices (0=oldest) that were clued
            clue_type_constant_colour: The CLUE.COLOUR constant value
            clue_type_constant_rank: The CLUE.RANK constant value
        """
        from collections import Counter
        
        clue_type_str = "COLOUR" if clue_type == clue_type_constant_colour else "RANK"
        logger.debug(f"HLE clued indices (0=oldest): {hle_clued_indices}")

        # Calculate available cards (the "hidden" pool that could be in our hand)
        total_counts: Counter = Counter()
        num_colors = self.game.num_colors()
        num_ranks = self.game.num_ranks()
        
        all_card_types = []
        for c in range(num_colors):
            for r in range(num_ranks):
                # Get actual card count from game (handles variants correctly)
                count = self.game.num_cards(c, r)
                total_counts[(c, r)] = count
                all_card_types.append((c, r))
        
        # Subtract visible
        # Discards
        for card in self.state.discard_pile():
            total_counts[(card.color(), card.rank())] -= 1
        
        # Fireworks
        fireworks = self.state.fireworks()
        for c, top_r in enumerate(fireworks):
            for r in range(top_r):
                total_counts[(c, r)] -= 1
        
        # Other players' hands (NOT the target)
        hands = self.state.player_hands()
        for p_idx, hand in enumerate(hands):
            if p_idx != hle_target:
                for card in hand:
                    total_counts[(card.color(), card.rank())] -= 1
        
        # Current target hand in HLE
        target_hand = hands[hle_target]
        
        logger.debug(f"HLE target hand before fix: {[str(c) for c in target_hand]}")
        logger.debug(f"Available pool (excluding target hand): {dict((k,v) for k,v in total_counts.items() if v > 0)}")
        
        # IMPORTANT: The target hand's current cards are NOT subtracted from total_counts yet.
        # This is intentional - we're treating them as "returned to the pool" so we can
        # reassign consistent cards. However, we need to track them separately to ensure
        # the final assignment doesn't use more copies than exist in the deck.
        
        # Helper to check if card fits constraint
        def fits(c_idx: int, r_idx: int, slot_idx: int) -> bool:
            # We only check constraints from the CURRENT clue, not previous knowledge.
            # Previous knowledge in HLE might be based on incorrect card assignments,
            # so we ignore it and only enforce the new clue constraints.
            
            # Check clue constraints
            is_clued = slot_idx in hle_clued_indices
            
            if clue_type == clue_type_constant_colour:
                if is_clued:
                    if c_idx != clue_val_hle: return False
                else:
                    if c_idx == clue_val_hle: return False
            elif clue_type == clue_type_constant_rank:
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
        
        final_assignment: List[Optional[pyhanabi.HanabiCard]] = [None] * len(target_hand)
        
        def solve(idx_in_sorted: int) -> bool:
            if idx_in_sorted == len(target_hand):
                return True
            
            original_idx = sorted_indices[idx_in_sorted]
            candidates = slot_candidates[original_idx]
            
            for (c, r) in candidates:
                if total_counts[(c, r)] > 0:
                    total_counts[(c, r)] -= 1
                    final_assignment[original_idx] = pyhanabi.HanabiCard(c, r)
                    
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
                    self.state.set_hand_card(hle_target, i, card)
                    logger.debug(f"Fixed slot {i}: {target_hand[i]} -> {card}")
                    changes_made += 1
            
            # Log final hand
            new_hand = self.state.player_hands()[hle_target]
            logger.debug(f"HLE target hand after fix ({changes_made} changes): {[str(c) for c in new_hand]}")
        else:
            # This should never happen if the algorithm is correct
            clue_type_str = "COLOUR" if clue_type == clue_type_constant_colour else "RANK"
            logger.error(
                f"Could not find consistent hand for clue fix! "
                f"Clue: {clue_type_str}={clue_val_hle}, clued_indices={hle_clued_indices}, "
                f"target_hand={[str(c) for c in target_hand]}, "
                f"slot_candidates={slot_candidates}, "
                f"remaining_counts={dict((k,v) for k,v in total_counts.items() if v > 0)}"
            )

    def apply_color_clue(self, target_player: int, color_index: int, giver_player: Optional[int] = None) -> None:
        """Apply a color clue to target_player.

        pyhanabi expects a target offset (1 = next player, 2 = player after, ...).
        We convert player indices into the corresponding offset based on the giver.
        
        Note: We use giver_player (from website) to calculate the offset, not HLE's
        cur_player(), because HLE might be desynced from the website due to different
        starting players or other issues.
        """
        # With player_shift=0, HLE player index = website player index
        target_hle = target_player
        
        # Use giver_player to determine who is giving the clue
        if giver_player is not None:
            giver_hle = giver_player
        else:
            giver_hle = self.state.cur_player()
        
        # Check for desync and log it
        current_hle = self.state.cur_player()
        if giver_hle != current_hle:
            logger.warning(f"HLE Desync: Website giver {giver_player} != HLE cur_player {current_hle}. Using giver for offset calculation.")
        
        # Calculate offset from giver to target
        offset = (target_hle - giver_hle) % self.num_players
        
        if offset == 0:
            logger.error(f"Cannot apply clue: Target {target_player} equals giver {giver_player}. Skipping.")
            return

        # We need to apply a clue move in HLE. The move uses offset from HLE's cur_player.
        # If HLE is desynced, we need to use the offset from HLE's cur_player, not from giver.
        # But this could result in cluing the wrong player!
        # 
        # The safest approach: calculate what offset HLE needs to clue the correct target.
        actual_offset = (target_hle - current_hle) % self.num_players
        
        if actual_offset == 0:
            logger.error(f"Cannot apply clue in HLE: Target {target_hle} is HLE cur_player {current_hle}. Skipping.")
            return

        move = pyhanabi.HanabiMove.get_reveal_color_move(actual_offset, color_index)
        self.safe_apply_move(move)

    def apply_rank_clue(self, target_player: int, rank_index: int, giver_player: Optional[int] = None) -> None:
        """Apply a rank clue to target_player.

        pyhanabi expects a target offset (1 = next player, 2 = player after, ...).
        We convert player indices into the corresponding offset based on the giver.
        """
        # With player_shift=0, HLE player index = website player index
        target_hle = target_player
        
        # Use giver_player to determine who is giving the clue
        if giver_player is not None:
            giver_hle = giver_player
        else:
            giver_hle = self.state.cur_player()
        
        # Check for desync and log it
        current_hle = self.state.cur_player()
        if giver_hle != current_hle:
            logger.warning(f"HLE Desync: Website giver {giver_player} != HLE cur_player {current_hle}. Using giver for offset calculation.")
        
        # Calculate offset from giver to target
        offset = (target_hle - giver_hle) % self.num_players
        
        if offset == 0:
            logger.error(f"Cannot apply clue: Target {target_player} equals giver {giver_player}. Skipping.")
            return

        # Calculate what offset HLE needs to clue the correct target.
        actual_offset = (target_hle - current_hle) % self.num_players
        
        if actual_offset == 0:
            logger.error(f"Cannot apply clue in HLE: Target {target_hle} is HLE cur_player {current_hle}. Skipping.")
            return

        move = pyhanabi.HanabiMove.get_reveal_rank_move(actual_offset, rank_index)
        self.safe_apply_move(move)

    # ----- observations -------------------------------------------------

    def observation_for_player(self, player_index: int):
        return self.state.observation(player_index)

