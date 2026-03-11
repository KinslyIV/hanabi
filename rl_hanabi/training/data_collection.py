"""
Data collection and dataset classes for token-based Hanabi self-play training.
"""

from __future__ import annotations

import random
import pickle
from collections import deque
from pathlib import Path
from typing import Dict, Iterator, List, Optional
import threading

import numpy as np
import torch
from torch.utils.data import IterableDataset

from rl_hanabi.model import TokenizationConfig
from rl_hanabi.training import Transition, GameResult



class ReplayBuffer:
    """Experience replay buffer for storing and sampling transitions."""

    def __init__(
        self,
        max_size: int = 100_000,
        save_dir: Optional[Path] = None,
    ):
        self.max_size = max_size
        self.games: deque[List[Transition]] = deque()
        self.save_dir = save_dir
        self._lock = threading.Lock()

        self.total_added = 0
        self.current_size = 0  # Track current number of transitions in buffer
        self.game_results: List[Dict] = []

        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)

    def _evict_old_games(self) -> int:
        """Remove old games until buffer is under max_size. Returns number of games removed."""
        games_removed = 0
        while self.current_size > self.max_size and self.games:
            old_game = self.games.popleft()
            self.current_size -= len(old_game)
            games_removed += 1
        return games_removed

    def add_game_result(self, result: GameResult) -> None:
        with self._lock:
            num_transitions = len(result.transitions)
            self.total_added += num_transitions
            self.current_size += num_transitions

            self.games.append(result.transitions)

            self.game_results.append({
                "final_score": result.final_score,
                "max_score": result.max_possible_score,
                "num_turns": result.num_turns,
                "num_transitions": num_transitions,
                "game_config": result.game_config,
            })

            # Evict old games if buffer exceeds max_size
            self._evict_old_games()



    def __len__(self) -> int:
        return self.current_size

    def num_games(self) -> int:
        with self._lock:
            return len(self.games)

    def get_games_snapshot(self) -> List[List[Transition]]:
        with self._lock:
            return list(self.games)

    def get_statistics(self) -> Dict:
        with self._lock:
            if not self.game_results:
                return {
                    "total_added": self.total_added,
                    "current_size": self.current_size,
                    "num_games": 0,
                }

            scores = [r["final_score"] for r in self.game_results]
            max_scores = [r["max_score"] for r in self.game_results]
            normalized_scores = [s / m for s, m in zip(scores, max_scores) if m > 0]

            return {
                "total_added": self.total_added,
                "current_size": self.current_size,
                "num_games": len(self.games),
                "avg_score": np.mean(scores).item() if scores else 0,
                "max_score": max(scores) if scores else 0,
                "min_score": min(scores) if scores else 0,
                "max_turns": max(r["num_turns"] for r in self.game_results) if self.game_results else 0,
                "min_turns": min(r["num_turns"] for r in self.game_results) if self.game_results else 0,
                "avg_normalized_score": np.mean(normalized_scores).item() if normalized_scores else 0,
                "avg_turns": np.mean([r["num_turns"] for r in self.game_results]).item(),
            }

    def save(self, filename: str) -> None:
        if self.save_dir is None:
            return

        filepath = self.save_dir / filename
        with self._lock:
            with open(filepath, "wb") as f:
                pickle.dump({
                    "games": list(self.games),
                    "game_results": self.game_results,
                    "total_added": self.total_added,
                    "current_size": self.current_size,
                }, f)

    def load(self, filename: str) -> None:
        if self.save_dir is None:
            return

        filepath = self.save_dir / filename
        if filepath.exists():
            with open(filepath, "rb") as f:
                data = pickle.load(f)
            with self._lock:
                self.games = deque(data.get("games", []))
                self.game_results = data["game_results"]
                self.total_added = data["total_added"]
                # Recompute current_size for backward compatibility
                self.current_size = data.get(
                    "current_size",
                    sum(len(g) for g in self.games)
                )

    def clear_game_results(self) -> None:
        with self._lock:
            self.game_results = []


class GameSequenceDataset(IterableDataset):
    """Iterable Dataset yielding per-step batches across games.

    Each batch contains at most one transition per game, aligned by step index.
    """

    def __init__(
        self,
        buffer: ReplayBuffer,
        token_config: TokenizationConfig,
        batch_size: int,
        *,
        shuffle_games: bool = False,
        device: Optional[torch.device] = None,
    ):
        self.buffer = buffer
        self.token_config = token_config
        self.batch_size = batch_size
        self.shuffle_games = shuffle_games
        self.device = device

    def _pad_tokens(self, tokens: List[int]) -> List[int]:
        context_size = self.token_config.context_size
        if len(tokens) > context_size:
            raise ValueError(
                f"Token sequence too long: {len(tokens)} > context_size={context_size}"
            )
        if len(tokens) < context_size:
            tokens = tokens + [self.token_config.pad_token] * (context_size - len(tokens))
        return tokens

    def _collate(
        self,
        transitions: List[Transition],
        reset_mask: List[bool],
    ) -> Dict[str, torch.Tensor]:
        tokens_list = []
        legal_moves_mask_list = []
        chosen_action_idx_list = []
        reward_list = []
        advantage_list = []
        return_list = []
        done_list = []
        current_player_list = []
        teacher_action_idx_list = []
        teacher_mask_list = []
        num_players_list = []
        num_colors_list = []
        num_ranks_list = []
        hand_size_list = []

        for t in transitions:
            tokens_list.append(self._pad_tokens(t.tokens))
            legal_moves_mask_list.append(t.legal_moves_mask)
            chosen_action_idx_list.append(t.chosen_action_idx)
            reward_list.append(t.reward)
            advantage_list.append(t.advantage)
            return_list.append(t.return_value)
            done_list.append(t.done)
            current_player_list.append(t.current_player)
            teacher_action_idx_list.append(getattr(t, "teacher_action_idx", -1))
            teacher_mask_list.append(bool(getattr(t, "teacher_mask", False)))
            num_players_list.append(t.game_config.get("num_players", 2))
            num_colors_list.append(t.game_config.get("num_colors", 5))
            num_ranks_list.append(t.game_config.get("num_ranks", 5))
            hand_size_list.append(t.game_config.get("hand_size", 5))

        device = self.device
        tokens = torch.tensor(tokens_list, dtype=torch.long, device=device)
        legal_moves_mask = torch.tensor(legal_moves_mask_list, dtype=torch.bool, device=device)
        chosen_action_idx = torch.tensor(chosen_action_idx_list, dtype=torch.long, device=device)
        reward = torch.tensor(reward_list, dtype=torch.float32, device=device)
        advantage = torch.tensor(advantage_list, dtype=torch.float32, device=device)
        returns = torch.tensor(return_list, dtype=torch.float32, device=device)
        done = torch.tensor(done_list, dtype=torch.bool, device=device)
        current_player = torch.tensor(current_player_list, dtype=torch.long, device=device)
        teacher_action_idx = torch.tensor(teacher_action_idx_list, dtype=torch.long, device=device)
        teacher_mask = torch.tensor(teacher_mask_list, dtype=torch.bool, device=device)
        num_players = torch.tensor(num_players_list, dtype=torch.long, device=device)
        num_colors = torch.tensor(num_colors_list, dtype=torch.long, device=device)
        num_ranks = torch.tensor(num_ranks_list, dtype=torch.long, device=device)
        hand_size = torch.tensor(hand_size_list, dtype=torch.long, device=device)

        return {
            "tokens": tokens,
            "legal_moves_mask": legal_moves_mask,
            "chosen_action_idx": chosen_action_idx,
            "reward": reward,
            "advantage": advantage,
            "returns": returns,
            "done": done,
            "current_player": current_player,
            "teacher_action_idx": teacher_action_idx,
            "teacher_mask": teacher_mask,
            "num_players": num_players,
            "num_colors": num_colors,
            "num_ranks": num_ranks,
            "hand_size": hand_size,
            "reset_mask": torch.tensor(reset_mask, dtype=torch.bool, device=device),
        }

    def __iter__(self) -> Iterator:
        games = self.buffer.get_games_snapshot()
        if self.shuffle_games:
            random.shuffle(games)

        for start in range(0, len(games), self.batch_size):
            group = games[start:start + self.batch_size]
            if not group:
                continue

            max_len = max(len(game) for game in group)
            for step_idx in range(max_len):
                transitions: List[Transition] = []
                reset_mask: List[bool] = []
                for game in group:
                    if step_idx < len(game):
                        transitions.append(game[step_idx])
                        reset_mask.append(step_idx == 0)

                if not transitions:
                    continue

                yield self._collate(transitions, reset_mask)


