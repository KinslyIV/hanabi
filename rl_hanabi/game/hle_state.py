from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List
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
    our_index: int
    player_shift: int = 0

    # ----- construction -------------------------------------------------

    @classmethod
    def from_table_options(
        cls,
        options: Dict[str, Any],
        num_players: int,
        our_index: int,
    ) -> "HLEGameState":
        """Create a fresh HanabiGame + initial HanabiState.

        The mapping from Hanab.live options to pyhanabi parameters is kept
        minimal for now and can be extended as needed.
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
            "random_start_player": False, # We handle shift manually
            "observation_type": pyhanabi.AgentObservationType.CARD_KNOWLEDGE,
        }
        game = pyhanabi.HanabiGame(params)
        state = game.new_initial_state()
        
        # Advance through initial chance nodes (dealing cards)
        while state.cur_player() == pyhanabi.CHANCE_PLAYER_ID:
            state.deal_random_card()
            
        return cls(game=game, state=state, our_index=our_index, player_shift=starting_player)
    
    def __repr__(self) -> str:
        return self.state.__repr__()

    # ----- basic API for MCTS ------------------------------------------

    def copy(self) -> "HLEGameState":
        """Deep copy for use as a root in MCTS."""
        return HLEGameState(
            game=self.game,
            state=self.state.copy(),
            our_index=self.our_index,
            player_shift=self.player_shift,
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

    @property
    def current_player_index(self) -> int:
        # Map HLE index back to Hanab.live index
        return (self.state.cur_player() + self.player_shift) % self.num_players

    def is_terminal(self) -> bool:
        return self.state.is_terminal()

    def score(self) -> int:
        return self.state.score()

    def legal_moves(self) -> List[pyhanabi.HanabiMove]:
        return self.state.legal_moves()
    
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
        
        # Check if move is legal in the current pyhanabi state
        legal_moves = self.state.legal_moves()
        is_legal = False
        for lm in legal_moves:
            if self.moves_are_equal(move, lm):
                is_legal = True
                break
        
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
                self.state.apply_move(alt_move)
                self._advance_chance_nodes()
                return
                
        elif move_type == pyhanabi.HanabiMoveType.PLAY:
            # Try to Discard instead
            alt_move = pyhanabi.HanabiMove.get_discard_move(move.card_index())
            if self._is_legal(alt_move):
                logger.warning(f"Recovering by DISCARDING card {move.card_index()} instead of PLAYING.")
                self.state.apply_move(alt_move)
                self._advance_chance_nodes()
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

    def apply_play_from_hand_index(self, card_index: int) -> None:
        """Apply a PLAY move for the current player using a hand index."""
        move = pyhanabi.HanabiMove.get_play_move(card_index)
        self.safe_apply_move(move)

    def apply_discard_from_hand_index(self, card_index: int) -> None:
        """Apply a DISCARD move for the current player using a hand index."""
        move = pyhanabi.HanabiMove.get_discard_move(card_index)
        self.safe_apply_move(move)

    def apply_color_clue(self, target_player: int, color_index: int) -> None:
        """Apply a color clue to target_player.

        pyhanabi expects a target offset (0 = current player, 1 = next, ...).
        We convert player indices into the corresponding offset.
        """
        # Map target_player (Hanab.live index) to HLE index
        target_hle = (target_player - self.player_shift) % self.num_players
        current_hle = self.state.cur_player()
        
        offset = (target_hle - current_hle) % self.num_players
        move = pyhanabi.HanabiMove.get_reveal_color_move(offset, color_index)
        self.safe_apply_move(move)

    def apply_rank_clue(self, target_player: int, rank_index: int) -> None:
        """Apply a rank clue to target_player.

        pyhanabi expects a target offset (0 = current player, 1 = next, ...).
        We convert player indices into the corresponding offset.
        """
        # Map target_player (Hanab.live index) to HLE index
        target_hle = (target_player - self.player_shift) % self.num_players
        current_hle = self.state.cur_player()
        
        offset = (target_hle - current_hle) % self.num_players
        move = pyhanabi.HanabiMove.get_reveal_rank_move(offset, rank_index)
        self.safe_apply_move(move)




    # ----- observations -------------------------------------------------

    def observation_for_player(self, player_index: int):
        return self.state.observation(player_index)

    def observation_for_our_player(self):
        return self.observation_for_player(self.our_index)
