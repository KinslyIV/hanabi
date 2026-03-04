"""
Tests for the token-based ActionDecoder pipeline.
"""

import pytest
import torch

from rl_hanabi.model.action_decoder import ActionDecoder
from rl_hanabi.model.tokenizer import TokenizationConfig
from rl_hanabi.training.token_utils import build_action_logits_from_tokens


def _make_token_config() -> TokenizationConfig:
    return TokenizationConfig(
        num_colors=2,
        num_ranks=2,
        hand_size=2,
        max_num_players=2,
        max_info_tokens=8,
        max_life_tokens=3,
    )


def test_action_decoder_forward_shapes():
    token_config = _make_token_config()
    model = ActionDecoder(
        num_colors=token_config.num_colors,
        num_ranks=token_config.num_ranks,
        max_cards=token_config.num_card_tokens,
        hand_size=token_config.hand_size,
        num_players=token_config.max_num_players,
        num_heads=2,
        num_layers=2,
        d_model=32,
        action_dim=4,
        token_config=token_config,
    )

    batch_size = 3
    tokens = torch.zeros(batch_size, token_config.context_size, dtype=torch.long)
    tokens[:, 0] = 3
    tokens[:, 1] = 8
    tokens[:, 2] = 0
    tokens[:, 3:] = torch.randint(
        low=0,
        high=token_config.total_card_tokens,
        size=(batch_size, token_config.context_size - 3),
    )

    logits = model(tokens)
    assert logits.shape == (batch_size, token_config.max_num_players * token_config.hand_size, 4)


def test_action_logits_mapping():
    token_config = _make_token_config()
    hand_start = 3 + token_config.num_colors
    tokens = torch.zeros(1, token_config.context_size, dtype=torch.long)
    tokens[0, 0] = 3
    tokens[0, 1] = 8
    tokens[0, 2] = 0

    tokens[0, hand_start + 0] = token_config.masked_card_token
    tokens[0, hand_start + 1] = token_config.masked_card_token

    tokens[0, hand_start + 2] = 2 + 1 * token_config.num_ranks + 0
    tokens[0, hand_start + 3] = 2 + 0 * token_config.num_ranks + 1

    card_action_logits = torch.zeros(
        1,
        token_config.max_num_players * token_config.hand_size,
        4,
    )
    card_action_logits[0, 0, 0] = 1.5
    card_action_logits[0, 0, 1] = 2.0
    card_action_logits[0, 2, 2] = 0.7
    card_action_logits[0, 2, 3] = 1.1
    card_action_logits[0, 3, 2] = 1.3
    card_action_logits[0, 3, 3] = -0.2

    action_logits = build_action_logits_from_tokens(
        card_action_logits=card_action_logits,
        tokens=tokens,
        current_player=torch.tensor([0]),
        num_players=torch.tensor([2]),
        num_colors=torch.tensor([2]),
        num_ranks=torch.tensor([2]),
        hand_size=torch.tensor([2]),
        token_config=token_config,
    ).squeeze(0)

    assert action_logits[0].item() == pytest.approx(1.5)
    assert action_logits[2].item() == pytest.approx(2.0)
    assert action_logits[4].item() == pytest.approx(1.3)
    assert action_logits[5].item() == pytest.approx(0.7)
    assert action_logits[6].item() == pytest.approx(1.1)
    assert action_logits[7].item() == pytest.approx(-0.2)
