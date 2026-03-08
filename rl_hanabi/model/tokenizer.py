from __future__ import annotations

from dataclasses import dataclass
import json
from typing import List, Optional, Tuple, Union, TYPE_CHECKING

import torch

from hanabi_learning_environment import pyhanabi

CardLike = Union[pyhanabi.HanabiCard, None]


@dataclass(frozen=True)
class TokenizationConfig:
    num_colors: int
    num_ranks: int
    hand_size: int
    num_players: int
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
            num_players=data["max_num_players"],
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
                "max_num_players": self.num_players,
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


    @property
    def action_space_size(self) -> int:
        # action tokens start at index 1 index 0 is for no action
        H = self.hand_size
        N = self.num_players
        C = self.num_colors
        R = self.num_ranks
        return 2 * H + (N - 1) * (C + R) + 1

    @property
    def context_size(self) -> int:
        return 4 + self.num_colors + self.num_players * self.hand_size + self.max_discard_tokens

    @property
    def cards_per_color(self) -> int:
        # Standard Hanabi distribution: 3, 2, ..., 2, 1
        if self.num_ranks <= 1:
            return 3
        return 3 + 2 * (self.num_ranks - 2) + 1

    @property
    def total_cards(self) -> int:
        return self.num_colors * self.cards_per_color

    @property
    def max_discard_tokens(self) -> int:
        return max(0, self.total_cards)
    



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

    def pad_tokens(self, tokens: List[int]) -> List[int]:
        """Pad tokens to the configured context size."""
        context_size = self.config.context_size
        if len(tokens) > context_size:
            raise ValueError(
                f"Token sequence too long: {len(tokens)} > context_size={context_size}"
            )
        if len(tokens) < context_size:
            tokens = tokens + [self.config.pad_token] * (context_size - len(tokens))
        return tokens

    def action_logits_from_model(
        self,
        card_action_logits: torch.Tensor,
        tokens: torch.Tensor,
        current_player: torch.Tensor,
    ) -> torch.Tensor:
        """Convert per-card action logits to action space logits.
        
        Args:
            card_action_logits: (batch, num_players * hand_size, 4) - logits for [discard, play, hint_color, hint_rank]
            tokens: (batch, context_size) - tokenized game state
            current_player: (batch,) - current player index per batch element
        
        Returns:
            action_logits: (batch, action_space_size) - logits for each action
        """
        # NOTE: This function must stay differentiable from the returned `action_logits`
        # back to `card_action_logits`. Avoid in-place writes into a tensor that does not
        # require grad, as that can sever autograd connectivity.

        device = card_action_logits.device
        batch_size = card_action_logits.size(0)
        H = self.config.hand_size
        N = self.config.num_players
        C = self.config.num_colors
        R = self.config.num_ranks
        hand_start = 4 + C
        hand_end = hand_start + N * H

        # (batch, num_players, hand_size)
        hand_tokens = tokens[:, hand_start:hand_end].view(batch_size, N, H)
        # (batch, num_players, hand_size, 4)
        card_logits_view = card_action_logits.view(batch_size, N, H, 4)

        # Vectorized play/discard: gather current player's hand logits
        cp_idx = current_player.view(batch_size, 1, 1, 1).expand(batch_size, 1, H, 4)
        current_hand_logits = card_logits_view.gather(1, cp_idx).squeeze(1)  # (batch, H, 4)

        discard_logits = current_hand_logits[:, :, 0]  # (batch, H)
        play_logits = current_hand_logits[:, :, 1]  # (batch, H)

        # Hint logits: (batch, (N-1)*(C+R)) corresponding to indices [2H .. end)
        hint_dim = (N - 1) * (C + R)

        offsets = torch.arange(1, N, device=device)  # (N-1,)
        targets = (current_player.view(batch_size, 1) + offsets.view(1, -1)) % N  # (B, N-1)

        gather_idx = targets.view(batch_size, -1, 1).expand(batch_size, N - 1, H)
        target_hand_tokens = hand_tokens.gather(1, gather_idx)  # (B, N-1, H)

        logits_idx = targets.view(batch_size, -1, 1, 1).expand(batch_size, N - 1, H, 4)
        target_logits = card_logits_view.gather(1, logits_idx)  # (B, N-1, H, 4)

        valid = target_hand_tokens >= 2
        color = (target_hand_tokens - 2).clamp(min=0) // R
        rank = (target_hand_tokens - 2).clamp(min=0) % R

        offset_idx = offsets.view(1, -1, 1).expand(batch_size, N - 1, H) - 1  # (B, N-1, H)
        color_idx = offset_idx * C + color
        rank_idx = (N - 1) * C + offset_idx * R + rank

        color_idx = torch.where(valid, color_idx, torch.zeros_like(color_idx))
        rank_idx = torch.where(valid, rank_idx, torch.zeros_like(rank_idx))

        color_vals = torch.where(valid, target_logits[..., 2], torch.full_like(target_logits[..., 2], -1e9))
        rank_vals = torch.where(valid, target_logits[..., 3], torch.full_like(target_logits[..., 3], -1e9))

        color_idx = color_idx.view(batch_size, -1)
        rank_idx = rank_idx.view(batch_size, -1)
        color_vals = color_vals.view(batch_size, -1)
        rank_vals = rank_vals.view(batch_size, -1)

        hint_logits = torch.full((batch_size, hint_dim), -1e9, device=device)
        hint_logits = hint_logits.scatter_reduce(1, color_idx, color_vals, reduce="amax", include_self=True)
        hint_logits = hint_logits.scatter_reduce(1, rank_idx, rank_vals, reduce="amax", include_self=True)

        action_logits = torch.cat([discard_logits, play_logits, hint_logits], dim=1)
        return action_logits

    def _validate_state_config(self, state: "HLEGameState") -> None:
        # State Config und Tokenizer müssen genau dieselbe Configuration haben
        game_config = getattr(state, "game_config", None)
        if game_config is None:
            return

        mismatches = []
        if self.config.num_colors != game_config.num_colors:
            mismatches.append("num_colors")
        if self.config.num_ranks != game_config.num_ranks:
            mismatches.append("num_ranks")
        if self.config.hand_size != game_config.hand_size:
            mismatches.append("hand_size")
        if self.config.num_players != game_config.num_players:
            mismatches.append("max_num_players")
        if self.config.max_info_tokens != game_config.max_information_tokens:
            mismatches.append("max_info_tokens")
        if self.config.max_life_tokens != game_config.max_life_tokens:
            mismatches.append("max_life_tokens")

        if mismatches:
            mismatch_list = ", ".join(mismatches)
            raise ValueError(
                "Tokenizer config does not match game config: "
                f"{mismatch_list}"
            )

    def _card_to_token(self, rank, color):
        # Card tokens start at 2, since 0 is PAD and 1 is MASK
        if rank == -1:
            # This can happen for fireworks where a rank of -1 indicates no card for that color,
            return self.config.pad_token
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
        # 
        if token == self.config.pad_token:
            return None

        # Card tokens start at 2, since 0 is PAD and 1 is MASK
        token_index = token - 2
        if token_index < 0 or token_index >= self.config.num_card_tokens:
            raise ValueError(f"token {token} out of bounds for card tokens")
        color = token_index // self.config.num_ranks
        rank = token_index % self.config.num_ranks
        return color, rank

    def tokenize_hand(self, hand: List[CardLike], *, mask: bool = False) -> List[int]:

        if len(hand) > self.config.hand_size:
            raise ValueError(
                f"hand has {len(hand)} cards, exceeds hand_size {self.config.hand_size}"
            )
        if mask:
            tokens = [self.config.masked_card_token for _ in range(len(hand))]
        else:
            tokens = [self.card_to_token(hand[i]) for i in range(len(hand))]
        # Pads hand to ensure the token list always has length hand_size
        if len(tokens) < self.config.hand_size:
            padded = [self.config.pad_token] * (self.config.hand_size - len(tokens))
            tokens.extend(padded)
        return tokens

    def tokenize_fireworks(self, state: "HLEGameState") -> List[int]:
        self._validate_state_config(state)
        tokens: List[int] = []
        fireworks = state.state.fireworks()
        for color, top_rank in enumerate(fireworks):
            # Here a rank of 0 signifies no card for the color but the normal rank is 0 based 
            # so we need to subtract 1 to get the correct token
            tokens.append(self._card_to_token(top_rank-1, color))
        return tokens

    def tokenize_discard_pile(self, state: "HLEGameState") -> List[int]:
        self._validate_state_config(state)
        return [self.card_to_token(card) for card in state.state.discard_pile()]

    def tokenize_action_index(self, action_index: int) -> int:
        # Action index -1 will give token 0 which is the no action token passed to the model for the first move
        if action_index < -1 or action_index >= self.config.action_space_size:
            raise ValueError(f"Action index {action_index} out of bounds")
        # action tokens start at token 1 token 0 is for no action
        return action_index + 1


    def tokenize_move(self, move: pyhanabi.HanabiMove) -> int:
        move_type = move.type()
        H = self.config.hand_size
        if H is None:
            raise ValueError("hand_size is required for action tokenization")

        if move_type == pyhanabi.HanabiMoveType.DISCARD:
            return self.tokenize_action_index(move.card_index())

        if move_type == pyhanabi.HanabiMoveType.PLAY:
            return self.tokenize_action_index(H + move.card_index())

        N = self.config.num_players
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
        action: Union[pyhanabi.HanabiMove, int],
        current_player: int,
        model_player: int,
    ) -> List[int]:
        self._validate_state_config(state)
        tokens: List[int] = []

        if current_player < 0 or current_player >= self.config.num_players:
            raise ValueError(f"current_player {current_player} out of bounds")

        # Tells the model who made the move 
        # This player must not necessarily be masked 
        # since tzhe model here does not more play for all players but only for one
        tokens.append(current_player)

        tokens.append(state.life_tokens())
        tokens.append(state.information_tokens())
        if isinstance(action, pyhanabi.HanabiMove):
            tokens.append(self.tokenize_move(action))
        else:
            tokens.append(self.tokenize_action_index(action))

        tokens.extend(self.tokenize_fireworks(state))

        hands = list(state.state.player_hands())

        # Mask the player who the model is playing as
        for player_idx in range(self.config.num_players):
            is_current = player_idx == model_player
            tokens.extend(self.tokenize_hand(list(hands[player_idx]), mask=is_current))

        tokens.extend(self.tokenize_discard_pile(state))
        
        if len(tokens) > self.config.context_size:
            discard_len = len(state.state.discard_pile())
            raise ValueError(
                "Token sequence too long: "
                f"len={len(tokens)} > context_size={self.config.context_size}. "
                f"discard={discard_len}, num_players={self.config.num_players}, "
                f"num_colors={self.config.num_colors}, num_ranks={self.config.num_ranks}, "
                f"hand_size={self.config.hand_size}"
            )

        return tokens



if TYPE_CHECKING:
    from rl_hanabi.game.hle_state import HLEGameState
