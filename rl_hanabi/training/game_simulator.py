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
        self._player_models: List[ActionDecoder] = []
        self._player_count = 0

    def _get_player_models(self, num_players: int) -> List[ActionDecoder]:
        if self._player_count != num_players or not self._player_models:
            self._player_models = []
            for _ in range(num_players):
                player_model = ActionDecoder(config=self.model.config, token_config=self.tokenizer.config)
                player_model.to(self.device)
                player_model.eval()
                self._player_models.append(player_model)
            self._player_count = num_players

        base_state = self.model.state_dict()
        for player_model in self._player_models:
            player_model.load_state_dict(base_state)
            player_model.reset_state()

        return self._player_models

    def clear_player_models(self) -> None:
        self._player_models = []
        self._player_count = 0

    @torch.no_grad()
    def simulate_game(
        self,
        config: GameConfig,
    ) -> GameResult:
        state = HLEGameState.from_table_options(config)

        player_models = self._get_player_models(state.num_players)

        # Model device/eval state should be set once in the worker.
        transitions: List[Transition] = []
        num_turns = 0
        previous_action_idx = -1  # No action for the first move
        # action_taken = []

        while not state.is_terminal():
            current_player = state.current_player_index
            legal_moves_mask = state.legal_moves_mask()
            if not legal_moves_mask.any():
                break

            current_player_tensor = torch.tensor([current_player], device=self.device)
            current_player_value = None
            current_player_tokens: List[int] | None = None
            current_player_action_idx: int | None = None

            for player_idx, player_model in enumerate(player_models):
                tokens = self.tokenizer.tokenize_state_and_action(
                    state,
                    previous_action_idx,
                    current_player,
                    player_idx,
                )
                token_tensor = torch.tensor(
                    self.tokenizer.pad_tokens(tokens),
                    dtype=torch.long,
                    device=self.device,
                ).unsqueeze(0)

                card_action_logits, value = player_model(token_tensor)

                if player_idx == current_player:
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
                        current_player_action_idx = int(torch.multinomial(probs, 1).item())
                    else:
                        current_player_action_idx = int(masked_logits.argmax(dim=-1).item())

                    current_player_value = value
                    current_player_tokens = tokens

            if current_player_action_idx is None or current_player_tokens is None:
                break

            action_idx = current_player_action_idx

            state.apply_move_by_index(action_idx)
            previous_action_idx = action_idx

            # DEBUG
            # move = state.index_to_move(action_idx)
            # target = (move.target_offset() + current_player) % state.num_players
            # action_taken.append({"Current Player": current_player,
            #                      "Target player": target,
            #                      "Move": str(move)})
            
            # print(f"{'='*10} Turn {num_turns} - Player {current_player} took action index {action_idx} ({move}) - Target {target} {'='*10}")
            # print(state)

            transitions.append(
                Transition(
                    tokens=current_player_tokens,
                    legal_moves_mask=legal_moves_mask.tolist(),
                    chosen_action_idx=action_idx,
                    value=current_player_value.item() if current_player_value is not None else 0.0,
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
        for transition in transitions:
            transition.reward = final_score

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

