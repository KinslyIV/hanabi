"""
Game simulator for self-play training using tokenized state/action inputs.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

import numpy as np
import torch

from rl_hanabi.game.hle_state import HLEGameState
from rl_hanabi.model.action_decoder import ActionDecoder
from rl_hanabi.model.tokenizer import HLETokenizer
from rl_hanabi.training.token_utils import (
    ActionIndexMapper,
    GameSizes,
    build_action_logits_from_tokens,
    build_legal_moves_mask,
    build_tokens,
    pad_tokens,
)


@dataclass
class GameConfig:
    num_players: int = 2
    num_colors: int = 5
    num_ranks: int = 5
    hand_size: int = 5
    max_information_tokens: int = 8
    max_life_tokens: int = 3
    seed: int = -1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "numSuits": self.num_colors,
            "numRanks": self.num_ranks,
            "cardsPerHand": self.hand_size,
            "clueTokens": self.max_information_tokens,
            "strikeTokens": self.max_life_tokens,
            "seed": self.seed,
        }


@dataclass
class Transition:
    tokens: List[int]
    legal_moves_mask: np.ndarray
    chosen_action_idx: int
    reward: float
    done: bool
    current_player: int
    game_config: Dict[str, int]


@dataclass
class GameResult:
    transitions: List[Transition]
    final_score: int
    max_possible_score: int
    num_turns: int
    game_config: Dict[str, int]


class GameSimulator:
    def __init__(
        self,
        model: ActionDecoder,
        tokenizer: HLETokenizer,
        device: torch.device,
        temperature: float = 1.0,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.temperature = temperature
        self.mapper = ActionIndexMapper(tokenizer.config)

    def simulate_game(
        self,
        config: GameConfig,
    ) -> GameResult:
        state = HLEGameState.from_table_options(config.to_dict(), config.num_players)
        token_config = self.tokenizer.config
        sizes = GameSizes(
            num_players=config.num_players,
            num_colors=config.num_colors,
            num_ranks=config.num_ranks,
            hand_size=config.hand_size,
        )

        transitions: List[Transition] = []
        num_turns = 0
        previous_action_token = token_config.pad_token

        while not state.is_terminal():
            current_player = state.current_player_index
            legal_moves_mask = state.legal_moves_mask()
            if not legal_moves_mask.any():
                break

            tokens = build_tokens(self.tokenizer, state, previous_action_token)
            token_tensor = torch.tensor(
                pad_tokens(tokens, token_config),
                dtype=torch.long,
                device=self.device,
            ).unsqueeze(0)

            with torch.no_grad():
                card_action_logits = self.model(token_tensor)

            action_logits = build_action_logits_from_tokens(
                card_action_logits=card_action_logits,
                tokens=token_tensor,
                current_player=torch.tensor([current_player], device=self.device),
                num_players=torch.tensor([sizes.num_players], device=self.device),
                num_colors=torch.tensor([sizes.num_colors], device=self.device),
                num_ranks=torch.tensor([sizes.num_ranks], device=self.device),
                hand_size=torch.tensor([sizes.hand_size], device=self.device),
                token_config=token_config,
            ).squeeze(0)

            max_legal_mask = build_legal_moves_mask(legal_moves_mask, token_config, sizes)
            legal_mask_tensor = torch.tensor(max_legal_mask, device=self.device)
            masked_logits = action_logits.masked_fill(~legal_mask_tensor, -1e9)

            if self.temperature > 0:
                masked_logits = masked_logits / self.temperature
                probs = torch.softmax(masked_logits, dim=-1)
                action_idx = int(torch.multinomial(probs, 1).item())
            else:
                action_idx = int(masked_logits.argmax(dim=-1).item())

            actual_action_idx = self.mapper.to_actual_index(action_idx, sizes)
            if actual_action_idx is None:
                legal_indices = np.where(legal_moves_mask)[0]
                actual_action_idx = int(random.choice(legal_indices))
                action_idx = self.mapper.to_max_index(actual_action_idx, sizes)

            move = state.index_to_move(actual_action_idx)
            state.apply_move(move)
            previous_action_token = self.tokenizer.tokenize_move(move)

            transitions.append(
                Transition(
                    tokens=tokens,
                    legal_moves_mask=max_legal_mask.astype(bool),
                    chosen_action_idx=action_idx,
                    reward=0.0,
                    done=False,
                    current_player=current_player,
                    game_config={
                        "num_players": config.num_players,
                        "num_colors": config.num_colors,
                        "num_ranks": config.num_ranks,
                        "hand_size": config.hand_size,
                    },
                )
            )

            num_turns += 1

        if transitions:
            transitions[-1].done = True

        final_score = state.score()
        max_score = state.max_score()
        normalized_reward = final_score / max_score if max_score > 0 else 0.0
        for transition in transitions:
            transition.reward = normalized_reward

        return GameResult(
            transitions=transitions,
            final_score=final_score,
            max_possible_score=max_score,
            num_turns=num_turns,
            game_config={
                "num_players": config.num_players,
                "num_colors": config.num_colors,
                "num_ranks": config.num_ranks,
                "hand_size": config.hand_size,
            },
        )


def sample_game_config(
    num_players_range: Tuple[int, int] = (3, 5),
    num_colors_range: Tuple[int, int] = (3, 5),
    num_ranks_range: Tuple[int, int] = (3, 5),
    max_information_tokens: int = 8,
    max_life_tokens: int = 3,
) -> GameConfig:
    """
    Sample a random game configuration.
    
    Ensures the config is valid: hand_size * num_players <= cards_per_color * num_colors
    Standard Hanabi has approximately (num_ranks * 2) cards per color.
    """
    # Keep trying until we get a valid config
    max_attempts = 100
    for _ in range(max_attempts):
        num_players = random.randint(*num_players_range)
        num_colors = random.randint(*num_colors_range)
        num_ranks = random.randint(*num_ranks_range)
        
        # Hand size depends on number of players
        if num_players <= 3:
            hand_size = 5
        else:
            hand_size = 4
        
        # Estimate cards per color (approximately 2 per rank in standard distribution)
        # In HLE: for 5 ranks, there are 3+2+2+2+1=10 cards per color
        # For smaller ranks, estimate ~2 cards per rank
        cards_per_color = num_ranks * 2
        
        # Check the constraint: hand_size * num_players <= cards_per_color * num_colors
        total_cards_needed = hand_size * num_players
        total_cards_available = cards_per_color * num_colors
        
        if total_cards_needed <= total_cards_available:
            return GameConfig(
                num_players=num_players,
                num_colors=num_colors,
                num_ranks=num_ranks,
                hand_size=hand_size,
                max_information_tokens=max_information_tokens,
                max_life_tokens=max_life_tokens,
                seed=random.randint(0, 2**31 - 1),
            )
    
    # Fallback to a safe default config
    return GameConfig(
        num_players=3,
        num_colors=5,
        num_ranks=5,
        hand_size=5,
        max_information_tokens=max_information_tokens,
        max_life_tokens=max_life_tokens,
        seed=random.randint(0, 2**31 - 1),
    )
