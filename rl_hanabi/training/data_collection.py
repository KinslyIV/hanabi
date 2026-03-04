"""
Data collection and dataset classes for token-based Hanabi self-play training.
"""

from __future__ import annotations

import random
import pickle
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional
import threading

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from rl_hanabi.model.tokenizer import TokenizationConfig
from rl_hanabi.training.game_simulator import Transition, GameResult
from rl_hanabi.training.token_utils import pad_tokens


class ReplayBuffer:
    """Experience replay buffer for storing and sampling transitions."""

    def __init__(
        self,
        max_size: int = 100_000,
        save_dir: Optional[Path] = None,
    ):
        self.max_size = max_size
        self.buffer: deque[Transition] = deque(maxlen=max_size)
        self.save_dir = save_dir
        self._lock = threading.Lock()

        self.total_added = 0
        self.game_results: List[Dict] = []

        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)

    def add(self, transition: Transition) -> None:
        with self._lock:
            self.buffer.append(transition)
            self.total_added += 1

    def add_game_result(self, result: GameResult) -> None:
        with self._lock:
            for transition in result.transitions:
                self.buffer.append(transition)
                self.total_added += 1

            self.game_results.append({
                "final_score": result.final_score,
                "max_score": result.max_possible_score,
                "num_turns": result.num_turns,
                "num_transitions": len(result.transitions),
                "game_config": result.game_config,
            })

    def sample(self, batch_size: int) -> List[Transition]:
        with self._lock:
            if len(self.buffer) < batch_size:
                return list(self.buffer)
            return random.sample(list(self.buffer), batch_size)

    def __len__(self) -> int:
        return len(self.buffer)

    def get_statistics(self) -> Dict:
        with self._lock:
            if not self.game_results:
                return {
                    "buffer_size": len(self.buffer),
                    "total_added": self.total_added,
                    "num_games": 0,
                }

            scores = [r["final_score"] for r in self.game_results]
            max_scores = [r["max_score"] for r in self.game_results]
            normalized_scores = [s / m for s, m in zip(scores, max_scores) if m > 0]

            return {
                "buffer_size": len(self.buffer),
                "total_added": self.total_added,
                "num_games": len(self.game_results),
                "avg_score": np.mean(scores) if scores else 0,
                "max_score": max(scores) if scores else 0,
                "min_score": min(scores) if scores else 0,
                "avg_normalized_score": np.mean(normalized_scores) if normalized_scores else 0,
                "avg_turns": np.mean([r["num_turns"] for r in self.game_results]),
            }

    def save(self, filename: str) -> None:
        if self.save_dir is None:
            return

        filepath = self.save_dir / filename
        with self._lock:
            with open(filepath, "wb") as f:
                pickle.dump({
                    "buffer": list(self.buffer),
                    "game_results": self.game_results,
                    "total_added": self.total_added,
                }, f)

    def load(self, filename: str) -> None:
        if self.save_dir is None:
            return

        filepath = self.save_dir / filename
        if filepath.exists():
            with open(filepath, "rb") as f:
                data = pickle.load(f)
            with self._lock:
                self.buffer = deque(data["buffer"], maxlen=self.max_size)
                self.game_results = data["game_results"]
                self.total_added = data["total_added"]

    def clear_game_results(self) -> None:
        with self._lock:
            self.game_results = []


class HanabiDataset(Dataset):
    """PyTorch Dataset for tokenized Hanabi transitions."""

    def __init__(
        self,
        buffer: ReplayBuffer,
        token_config: TokenizationConfig,
    ):
        self.buffer = buffer
        self.token_config = token_config

    def __len__(self) -> int:
        return len(self.buffer)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        with self.buffer._lock:
            transition = self.buffer.buffer[idx]

        tokens = pad_tokens(transition.tokens, self.token_config)
        return {
            "tokens": torch.tensor(tokens, dtype=torch.long),
            "legal_moves_mask": torch.tensor(transition.legal_moves_mask, dtype=torch.bool),
            "chosen_action_idx": torch.tensor(transition.chosen_action_idx, dtype=torch.long),
            "reward": torch.tensor(transition.reward, dtype=torch.float32),
            "done": torch.tensor(transition.done, dtype=torch.bool),
            "current_player": torch.tensor(transition.current_player, dtype=torch.long),
            "num_players": torch.tensor(transition.game_config.get("num_players", 2), dtype=torch.long),
            "num_colors": torch.tensor(transition.game_config.get("num_colors", 5), dtype=torch.long),
            "num_ranks": torch.tensor(transition.game_config.get("num_ranks", 5), dtype=torch.long),
            "hand_size": torch.tensor(transition.game_config.get("hand_size", 5), dtype=torch.long),
        }


def create_dataloader(
    buffer: ReplayBuffer,
    batch_size: int,
    token_config: TokenizationConfig,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    dataset = HanabiDataset(buffer, token_config)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
