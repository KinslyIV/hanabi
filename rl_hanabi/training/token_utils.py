from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING

import numpy as np
import torch

from rl_hanabi.model.tokenizer import HLETokenizer, TokenizationConfig


@dataclass(frozen=True)
class GameSizes:
    num_players: int
    num_colors: int
    num_ranks: int
    hand_size: int


class ActionIndexMapper:
    def __init__(self, token_config: TokenizationConfig) -> None:
        self.token_config = token_config

    def to_max_index(self, action_index: int, sizes: GameSizes) -> int:
        H = sizes.hand_size
        N = sizes.num_players
        C = sizes.num_colors
        R = sizes.num_ranks

        max_H = self.token_config.hand_size
        max_N = self.token_config.max_num_players
        max_C = self.token_config.num_colors
        max_R = self.token_config.num_ranks

        if action_index < H:
            return action_index

        if action_index < 2 * H:
            card_idx = action_index - H
            return max_H + card_idx

        color_start = 2 * H
        num_color = (N - 1) * C
        if action_index < color_start + num_color:
            rel_idx = action_index - color_start
            player_offset_minus_1 = rel_idx // C
            color = rel_idx % C
            return 2 * max_H + player_offset_minus_1 * max_C + color

        rank_start = color_start + num_color
        rel_idx = action_index - rank_start
        player_offset_minus_1 = rel_idx // R
        rank = rel_idx % R
        return 2 * max_H + (max_N - 1) * max_C + player_offset_minus_1 * max_R + rank

    def to_actual_index(self, action_index: int, sizes: GameSizes) -> Optional[int]:
        H = sizes.hand_size
        N = sizes.num_players
        C = sizes.num_colors
        R = sizes.num_ranks

        max_H = self.token_config.hand_size
        max_N = self.token_config.max_num_players
        max_C = self.token_config.num_colors
        max_R = self.token_config.num_ranks

        if action_index < max_H:
            if action_index >= H:
                return None
            return action_index

        if action_index < 2 * max_H:
            card_idx = action_index - max_H
            if card_idx >= H:
                return None
            return H + card_idx

        max_color_start = 2 * max_H
        max_color_count = (max_N - 1) * max_C
        if action_index < max_color_start + max_color_count:
            rel_idx = action_index - max_color_start
            player_offset_minus_1 = rel_idx // max_C
            color = rel_idx % max_C
            if player_offset_minus_1 >= (N - 1) or color >= C:
                return None
            return 2 * H + player_offset_minus_1 * C + color

        max_rank_start = max_color_start + max_color_count
        rel_idx = action_index - max_rank_start
        player_offset_minus_1 = rel_idx // max_R
        rank = rel_idx % max_R
        if player_offset_minus_1 >= (N - 1) or rank >= R:
            return None
        return 2 * H + (N - 1) * C + player_offset_minus_1 * R + rank


def pad_tokens(tokens: List[int], token_config: TokenizationConfig) -> List[int]:
    if len(tokens) > token_config.context_size:
        return tokens[: token_config.context_size]
    if len(tokens) < token_config.context_size:
        return tokens + [token_config.pad_token] * (token_config.context_size - len(tokens))
    return tokens


def build_tokens(
    tokenizer: HLETokenizer,
    state: "HLEGameState",
    action_token: int,
) -> List[int]:
    token_config = tokenizer.config
    tokens: List[int] = [
        state.life_tokens(),
        state.information_tokens(),
        action_token,
    ]
    tokens.extend(tokenizer.tokenize_fireworks(state))

    hands = list(state.state.player_hands())
    for player_idx in range(token_config.max_num_players):
        if player_idx < len(hands):
            is_current = player_idx == state.current_player_index
            tokens.extend(tokenizer.tokenize_hand(list(hands[player_idx]), mask=is_current))
        else:
            tokens.extend([token_config.pad_token] * token_config.hand_size)

    discard_tokens = tokenizer.tokenize_discard_pile(state)
    max_discard_tokens = token_config.context_size - len(tokens)
    if max_discard_tokens <= 0:
        return tokens[: token_config.context_size]

    if len(discard_tokens) > max_discard_tokens:
        discard_tokens = discard_tokens[-max_discard_tokens:]
    tokens.extend(discard_tokens)
    return tokens


if TYPE_CHECKING:
    from rl_hanabi.game.hle_state import HLEGameState


def build_legal_moves_mask(
    legal_moves_mask: np.ndarray,
    token_config: TokenizationConfig,
    sizes: GameSizes,
) -> np.ndarray:
    mapper = ActionIndexMapper(token_config)
    max_mask = np.zeros(token_config.action_space_size, dtype=bool)
    for action_idx, is_legal in enumerate(legal_moves_mask):
        if not is_legal:
            continue
        max_idx = mapper.to_max_index(action_idx, sizes)
        max_mask[max_idx] = True
    return max_mask


def build_action_logits_from_tokens(
    card_action_logits: torch.Tensor,
    tokens: torch.Tensor,
    current_player: torch.Tensor,
    num_players: torch.Tensor,
    num_colors: torch.Tensor,
    num_ranks: torch.Tensor,
    hand_size: torch.Tensor,
    token_config: TokenizationConfig,
) -> torch.Tensor:
    device = card_action_logits.device
    batch_size = card_action_logits.size(0)
    max_H = token_config.hand_size
    max_N = token_config.max_num_players
    max_C = token_config.num_colors
    max_R = token_config.num_ranks
    hand_start = 3 + token_config.num_colors
    hand_end = hand_start + max_N * max_H

    action_logits = torch.full(
        (batch_size, token_config.action_space_size),
        -1e9,
        device=device,
    )
    hand_tokens = tokens[:, hand_start:hand_end].view(batch_size, max_N, max_H)

    for b in range(batch_size):
        sizes = GameSizes(
            num_players=int(num_players[b].item()),
            num_colors=int(num_colors[b].item()),
            num_ranks=int(num_ranks[b].item()),
            hand_size=int(hand_size[b].item()),
        )
        current = int(current_player[b].item())

        for slot_idx in range(sizes.hand_size):
            card_slot = current * max_H + slot_idx
            action_logits[b, slot_idx] = card_action_logits[b, card_slot, 0]
            action_logits[b, max_H + slot_idx] = card_action_logits[b, card_slot, 1]

        for offset in range(1, sizes.num_players):
            target_player = (current + offset) % sizes.num_players
            for slot_idx in range(sizes.hand_size):
                token = int(hand_tokens[b, target_player, slot_idx].item())
                if token < 2:
                    continue
                color = (token - 2) // token_config.num_ranks
                rank = (token - 2) % token_config.num_ranks

                card_slot = target_player * max_H + slot_idx

                if color < sizes.num_colors:
                    color_idx = 2 * max_H + (offset - 1) * max_C + color
                    action_logits[b, color_idx] = torch.maximum(
                        action_logits[b, color_idx],
                        card_action_logits[b, card_slot, 2],
                    )

                if rank < sizes.num_ranks:
                    rank_idx = 2 * max_H + (max_N - 1) * max_C + (offset - 1) * max_R + rank
                    action_logits[b, rank_idx] = torch.maximum(
                        action_logits[b, rank_idx],
                        card_action_logits[b, card_slot, 3],
                    )

    return action_logits
