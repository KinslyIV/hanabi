from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .game_types import ACTION, CLUE, PerformAction
from .hle_state import HLEGameState


MAX_CLUES = 8


@dataclass
class GameState:
    player_names: List[str]
    our_index: int
    num_players: int
    options: dict
    # Whether to maintain a shadow HLE (pyhanabi) environment.
    # The random baseline bot disables this to avoid pyhanabi assertion crashes
    # when the mirrored state diverges slightly from hanab.live.
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
        # Create a new HLE game/state matching the table options if enabled.
        if self.enable_hle:
            self.hle_state = HLEGameState.from_table_options(
                self.options, self.num_players, self.our_index
            )
        else:
            self.hle_state = None

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

    def apply_draw(self, order: int, player_index: int) -> None:
        # Local mirror of hand composition; HLE draws when play/discard is applied.
        # Match hanab.live bot semantics: newest card at the front.
        self.hands[player_index].insert(0, order)

    def _remove_order(self, player_index: int, order: int) -> None:
        try:
            self.hands[player_index].remove(order)
        except ValueError:
            pass

    def apply_play(self, order: int, player_index: int) -> None:
        # Mirror into HLE using the slot index, then update local hands.
        idx = self._hand_index_from_order(player_index, order)
        if self.enable_hle and self.hle_state is not None and idx is not None:
            self.hle_state.apply_play_from_hand_index(idx)
        self._remove_order(player_index, order)

    def apply_discard(self, order: int, player_index: int) -> None:
        idx = self._hand_index_from_order(player_index, order)
        if self.enable_hle and self.hle_state is not None and idx is not None:
            self.hle_state.apply_discard_from_hand_index(idx)
        self._remove_order(player_index, order)

    def apply_clue(self, giver: int, target: int, clue_type: int, value: int) -> None:
        """Apply a clue action into the HLE shadow state.

        The random baseline bot does not currently use clues, but we still
        mirror them so the HLE state stays as close as possible to the
        website game.
        """
        if not (self.enable_hle and self.hle_state is not None):
            return

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
