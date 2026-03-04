from __future__ import annotations

from dataclasses import dataclass
import json
from typing import List, Optional, Tuple, Union, TYPE_CHECKING

from hanabi_learning_environment import pyhanabi

CardLike = Union[pyhanabi.HanabiCard, None]


@dataclass(frozen=True)
class TokenizationConfig:
    num_colors: int
    num_ranks: int
    hand_size: int
    max_num_players: int
    max_info_tokens: int
    max_life_tokens: int
    pad_token: int = 0
    masked_card_token: int = 1
    unknown_token: Optional[int] = None

    @classmethod
    def _from_json(cls, data: dict) -> TokenizationConfig:
        return cls(
            num_colors=data["num_colors"],
            num_ranks=data["num_ranks"],
            hand_size=data["hand_size"],
            max_num_players=data["max_num_players"],
            max_info_tokens=data["max_information_tokens"],
            max_life_tokens=data["max_life_tokens"],
            pad_token=data.get("pad_token", 0),
            unknown_token=data.get("unknown_token"),
        )
    
    def _to_json(self, file):
        with open(file, "w") as f:
            json.dump({
                "num_colors": self.num_colors,
                "num_ranks": self.num_ranks,
                "hand_size": self.hand_size,
                "max_num_players": self.max_num_players,
                "max_information_tokens": self.max_info_tokens,
                "max_life_tokens": self.max_life_tokens,
                "pad_token": self.pad_token,
                "unknown_token": self.unknown_token,
            }, f, indent=4)

    @property
    def num_card_tokens(self) -> int:
        return self.num_colors * self.num_ranks
    
    @property
    def total_card_tokens(self) -> int:
        return self.num_card_tokens + 2

    def resolve_unknown_token(self) -> int:
        if self.unknown_token is not None:
            return self.unknown_token
        return self.num_card_tokens + 3

    @property
    def action_space_size(self) -> int:
        H = self.hand_size
        N = self.max_num_players
        C = self.num_colors
        R = self.num_ranks
        return 2 * H + (N - 1) * (C + R)

    @property
    def context_size(self) -> int:
        return 3 + self.num_card_tokens + self.action_space_size 
    



class HLETokenizer:
    """Tokenize HLE cards into integer IDs.

    Token IDs:
      - 0: PAD (empty slot)
      - 1..(C*R): actual cards, ordered by color then rank
    - player/discard/fireworks card tokens are in disjoint ranges
    - action tokens in a separate range
    - life tokens and clue tokens in separate ranges
    """

    def __init__(self, config: TokenizationConfig) -> None:
        self.config = config


    def _card_to_token(self, rank, color):
        if color < 0 or color >= self.config.num_colors:
            raise ValueError(f"color {color} out of bounds")
        if rank < 0 or rank >= self.config.num_ranks:
            raise ValueError(f"rank {rank} out of bounds")
        return 2 + color * self.config.num_ranks + rank

    def card_to_token(self, card: CardLike) -> int:
        if not card:
            raise ValueError("card cannot be None for tokenization")
        
        return self._card_to_token(card.rank(), card.color())

    def token_to_card(self, token: int) -> Optional[Tuple[int, int]]:
        if token == self.config.pad_token:
            return None

        token_index = token - 2
        if token_index < 0 or token_index >= self.config.num_card_tokens:
            raise ValueError(f"token {token} out of bounds for card tokens")
        color = token_index // self.config.num_ranks
        rank = token_index % self.config.num_ranks
        return color, rank

    def tokenize_hand(self, hand: List[CardLike], *, mask: bool = False) -> List[int]:
        if mask:
            tokens = [self.config.masked_card_token for _ in range(len(hand))]
        else:
            tokens = [self.card_to_token(hand[i]) for i in range(len(hand))]
        if len(tokens) < self.config.hand_size:
            tokens.extend([self.config.pad_token] * (self.config.hand_size - len(tokens)))
        return tokens

    def tokenize_fireworks(self, state: "HLEGameState") -> List[int]:
        tokens: List[int] = []
        fireworks = state.state.fireworks()
        for color, top_rank in enumerate(fireworks):
            tokens.append(self._card_to_token(top_rank, color))
        return tokens

    def tokenize_discard_pile(self, state: "HLEGameState") -> List[int]:
        return [self.card_to_token(card) for card in state.state.discard_pile()]

    def tokenize_action_index(self, action_index: int) -> int:
        if action_index < 0 or action_index >= self.config.action_space_size:
            raise ValueError(f"Action index {action_index} out of bounds")
        return action_index

    def token_to_action_index(self, token: int) -> Optional[int]:
        start = 0
        end = start + self.config.action_space_size
        if start <= token < end:
            return token - start
        return None

    def tokenize_move(self, move: pyhanabi.HanabiMove) -> int:
        move_type = move.type()
        H = self.config.hand_size
        if H is None:
            raise ValueError("hand_size is required for action tokenization")

        if move_type == pyhanabi.HanabiMoveType.DISCARD:
            return self.tokenize_action_index(move.card_index())

        if move_type == pyhanabi.HanabiMoveType.PLAY:
            return self.tokenize_action_index(H + move.card_index())

        N = self.config.max_num_players
        C = self.config.num_colors
        R = self.config.num_ranks
        if N is None:
            raise ValueError("num_players is required for action tokenization")

        offset = move.target_offset()
        if move_type == pyhanabi.HanabiMoveType.REVEAL_COLOR:
            base = 2 * H
            return self.tokenize_action_index(base + (offset - 1) * C + move.color())

        if move_type == pyhanabi.HanabiMoveType.REVEAL_RANK:
            base = 2 * H + (N - 1) * C
            return self.tokenize_action_index(base + (offset - 1) * R + move.rank())

        raise ValueError(f"Unknown move type: {move}")


    def tokenize_state_and_action(
        self,
        state: "HLEGameState",
        action: pyhanabi.HanabiMove
    ) -> List[int]:
        tokens: List[int] = []

        tokens.append(state.life_tokens())
        tokens.append(state.information_tokens())
        tokens.append(self.tokenize_move(action))
        tokens.extend(self.tokenize_fireworks(state))

        hands = list(state.state.player_hands())
        for player_idx in range(self.config.max_num_players):
            if player_idx < len(hands):
                is_current = player_idx == state.current_player_index
                tokens.extend(self.tokenize_hand(list(hands[player_idx]), mask=is_current))
            else:
                tokens.extend([self.config.pad_token] * self.config.hand_size)

        tokens.extend(self.tokenize_discard_pile(state))
        
       
        return tokens



if TYPE_CHECKING:
    from rl_hanabi.game.hle_state import HLEGameState
