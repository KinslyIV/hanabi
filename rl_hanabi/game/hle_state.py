from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

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

    def apply_move(self, move: pyhanabi.HanabiMove) -> None:
        self.state.apply_move(move)

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
