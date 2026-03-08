"""
Game simulator for self-play training using tokenized state/action inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch

from rl_hanabi.game import HLEGameState, GameConfig
from rl_hanabi.model import ActionDecoder
from rl_hanabi.model import HLETokenizer


@dataclass
class Transition:
    tokens: List[int]
    legal_moves_mask: List[int]
    chosen_action_idx: int
    value: float
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

    @torch.no_grad()
    def simulate_game(
        self,
        config: GameConfig,
    ) -> GameResult:
        state = HLEGameState.from_table_options(config)

        # Model device/eval state should be set once in the worker.
        transitions: List[Transition] = []
        num_turns = 0
        previous_action_idx = -1  # No action for the first move

        while not state.is_terminal():
            current_player = state.current_player_index
            legal_moves_mask = state.legal_moves_mask()
            if not legal_moves_mask.any():
                break

            tokens = self.tokenizer.tokenize_state_and_action(state, previous_action_idx)
            token_tensor = torch.tensor(
                self.tokenizer.pad_tokens(tokens),
                dtype=torch.long,
                device=self.device,
            ).unsqueeze(0)

            card_action_logits, value = self.model(token_tensor)

            current_player_tensor = torch.tensor([current_player], device=self.device)
            action_logits = self.tokenizer.action_logits_from_model(
                card_action_logits,
                token_tensor,
                current_player_tensor,
            )

            legal_mask_tensor = torch.as_tensor(legal_moves_mask, device=self.device)
            masked_logits = action_logits.masked_fill(~legal_mask_tensor, -1e9)

            if self.temperature > 0:
                masked_logits = masked_logits / self.temperature
                probs = torch.softmax(masked_logits, dim=-1)
                action_idx = int(torch.multinomial(probs, 1).item())
            else:
                action_idx = int(masked_logits.argmax(dim=-1).item())

            state.apply_move_by_index(action_idx)
            previous_action_idx = action_idx

            transitions.append(
                Transition(
                    tokens=tokens,
                    legal_moves_mask=legal_moves_mask.tolist(),
                    chosen_action_idx=action_idx,
                    value=value.item(),
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

