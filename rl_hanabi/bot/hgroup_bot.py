from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from hanabi_learning_environment import pyhanabi

from rl_hanabi.game.hle_state import HLEGameState
from rl_hanabi.hgroup.policy import HGroupPolicy


@dataclass
class HGroupBotPolicy:
    """H-Group convention policy wrapper for HLEGameState."""

    stochastic: bool = False
    temperature: float = 1.0
    _policy: HGroupPolicy = field(default_factory=HGroupPolicy, init=False)

    def select_action_index(self, state: HLEGameState) -> int:
        if not self.stochastic:
            return self._policy.select_action_index(state, stochastic=False)
        weights = self._policy.get_move_weights(state)
        legal_moves = state.legal_moves()
        if not legal_moves:
            raise ValueError("No legal moves available")
        legal_indices = [state.move_to_index(move) for move in legal_moves]
        legal_weights = np.asarray(weights[legal_indices], dtype=np.float64)
        if self.temperature != 1.0 and self.temperature > 0:
            legal_weights = np.power(legal_weights, 1.0 / self.temperature)
        if legal_weights.sum() <= 0:
            return legal_indices[0]
        legal_weights = legal_weights / legal_weights.sum()
        choice = int(np.random.choice(len(legal_indices), p=legal_weights))
        return legal_indices[choice]

    def select_move(self, state: HLEGameState) -> pyhanabi.HanabiMove:
        return state.index_to_move(self.select_action_index(state))

    def apply_action(self, state: HLEGameState) -> int:
        action_idx = self.select_action_index(state)
        state.apply_move_by_index(action_idx)
        return action_idx

    def observe_move(self, state_before: HLEGameState, move: pyhanabi.HanabiMove) -> None:
        self._policy.observe_move(state_before, move)

    def reset(self, state: HLEGameState) -> None:
        self._policy.reset(state)
