from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
from hanabi_learning_environment import pyhanabi


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
        return cls(game=game, state=state, our_index=our_index)
    
    def __repr__(self) -> str:
        return self.state.__repr__()

    # ----- basic API for MCTS ------------------------------------------

    def copy(self) -> "HLEGameState":
        """Deep copy for use as a root in MCTS."""
        return HLEGameState(
            game=self.game,
            state=self.state.copy(),
            our_index=self.our_index,
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
        return self.state.cur_player()

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
        self.state.apply_move(move)

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
            
        raise ValueError(f"Unknown move type: {move1}")

    # ----- helpers used by GameState to mirror website actions ---------

    def apply_play_from_hand_index(self, card_index: int) -> None:
        """Apply a PLAY move for the current player using a hand index."""
        move = pyhanabi.HanabiMove.get_play_move(card_index)
        self.state.apply_move(move)

    def apply_discard_from_hand_index(self, card_index: int) -> None:
        """Apply a DISCARD move for the current player using a hand index."""
        move = pyhanabi.HanabiMove.get_discard_move(card_index)
        self.state.apply_move(move)

    def apply_color_clue(self, target_player: int, color_index: int) -> None:
        """Apply a color clue to target_player.

        pyhanabi expects a target offset (0 = current player, 1 = next, ...).
        We convert player indices into the corresponding offset.
        """
        offset = (target_player - self.current_player_index) % self.num_players
        move = pyhanabi.HanabiMove.get_reveal_color_move(offset, color_index)
        self.state.apply_move(move)

    def apply_rank_clue(self, target_player: int, rank: int) -> None:
        """Apply a rank clue to target_player (rank is 0-based)."""
        offset = (target_player - self.current_player_index) % self.num_players
        move = pyhanabi.HanabiMove.get_reveal_rank_move(offset, rank)
        self.state.apply_move(move)


    # ----- observations -------------------------------------------------

    def observation_for_player(self, player_index: int):
        return self.state.observation(player_index)

    def observation_for_our_player(self):
        return self.observation_for_player(self.our_index)
