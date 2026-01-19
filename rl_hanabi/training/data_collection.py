"""
Data collection and dataset classes for Hanabi self-play training.
"""

from __future__ import annotations

import random
import pickle
from pathlib import Path
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Iterator
import threading
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset

from rl_hanabi.training.game_simulator import Transition, GameResult, GameConfig


@dataclass
class BatchedTransition:
    """Batched transitions for training."""
    slot_beliefs: torch.Tensor          # [B, P, H, C+R]
    affected_mask: torch.Tensor         # [B, P, H]
    move_target_player: torch.Tensor    # [B]
    acting_player: torch.Tensor         # [B]
    action: torch.Tensor                # [B, action_dim]
    fireworks: torch.Tensor             # [B, C]
    discard_pile: torch.Tensor          # [B, C*R]
    
    # Targets
    true_colors: torch.Tensor           # [B, H]
    true_ranks: torch.Tensor            # [B, H]
    chosen_action_idx: torch.Tensor     # [B]
    legal_moves_mask: torch.Tensor      # [B, action_space_size]
    
    # Metadata
    rewards: torch.Tensor               # [B] - normalized final score
    failed_plays: torch.Tensor          # [B] - whether this was a failed play (bomb)
    dones: torch.Tensor                 # [B]


class ReplayBuffer:
    """
    Experience replay buffer for storing and sampling transitions.
    Thread-safe for use with multiprocessing.
    """
    
    def __init__(
        self,
        max_size: int = 100_000,
        save_dir: Optional[Path] = None,
    ):
        self.max_size = max_size
        self.buffer: deque[Transition] = deque(maxlen=max_size)
        self.save_dir = save_dir
        self._lock = threading.Lock()
        
        # Statistics
        self.total_added = 0
        self.game_results: List[Dict] = []
        
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
    
    def add(self, transition: Transition) -> None:
        """Add a single transition to the buffer."""
        with self._lock:
            self.buffer.append(transition)
            self.total_added += 1
    
    def add_game_result(self, result: GameResult) -> None:
        """Add all transitions from a game result."""
        with self._lock:
            for transition in result.transitions:
                self.buffer.append(transition)
                self.total_added += 1
            
            # Track game statistics
            self.game_results.append({
                "final_score": result.final_score,
                "max_score": result.max_possible_score,
                "num_turns": result.num_turns,
                "num_transitions": len(result.transitions),
                "game_config": result.game_config,
            })
    
    def sample(self, batch_size: int) -> List[Transition]:
        """Sample a batch of transitions."""
        with self._lock:
            if len(self.buffer) < batch_size:
                return list(self.buffer)
            return random.sample(list(self.buffer), batch_size)
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def get_statistics(self) -> Dict:
        """Get buffer statistics."""
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
        """Save buffer to disk."""
        if self.save_dir is None:
            return
        
        filepath = self.save_dir / filename
        with self._lock:
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'buffer': list(self.buffer),
                    'game_results': self.game_results,
                    'total_added': self.total_added,
                }, f)
    
    def load(self, filename: str) -> None:
        """Load buffer from disk."""
        if self.save_dir is None:
            return
        
        filepath = self.save_dir / filename
        if filepath.exists():
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                with self._lock:
                    self.buffer = deque(data['buffer'], maxlen=self.max_size)
                    self.game_results = data['game_results']
                    self.total_added = data['total_added']
    
    def clear_game_results(self) -> None:
        """Clear game results (keep transitions)."""
        with self._lock:
            self.game_results = []


class HanabiDataset(Dataset):
    """
    PyTorch Dataset for Hanabi transitions.
    Pads/normalizes transitions to handle varying game configurations.
    """
    
    def __init__(
        self,
        buffer: ReplayBuffer,
        max_num_players: int = 5,
        max_num_colors: int = 5,
        max_num_ranks: int = 5,
        max_hand_size: int = 5,
    ):
        self.buffer = buffer
        self.max_num_players = max_num_players
        self.max_num_colors = max_num_colors
        self.max_num_ranks = max_num_ranks
        self.max_hand_size = max_hand_size
        
        # Max action space size
        self.max_action_space_size = (
            2 * max_hand_size +  # Play + Discard
            (max_num_players - 1) * max_num_colors +  # Color clues
            (max_num_players - 1) * max_num_ranks     # Rank clues
        )
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        with self.buffer._lock:
            transition = self.buffer.buffer[idx]
        
        return self._process_transition(transition)
    
    def _remap_action_index(
        self,
        orig_idx: int,
        orig_players: int,
        orig_colors: int,
        orig_ranks: int,
        orig_hand_size: int,
    ) -> int:
        """Remap an action index from original game config to max config.
        
        Action space layout:
        - [0, H): Discard card i
        - [H, 2H): Play card i
        - [2H, 2H + (N-1)*C): Color clues: (player_offset-1)*C + color
        - [2H + (N-1)*C, ...): Rank clues: (player_offset-1)*R + rank
        """
        H = orig_hand_size
        N = orig_players
        C = orig_colors
        R = orig_ranks
        
        max_H = self.max_hand_size
        max_N = self.max_num_players
        max_C = self.max_num_colors
        max_R = self.max_num_ranks
        
        # Discard
        if orig_idx < H:
            return orig_idx  # Same index (card slot)
        
        # Play
        if orig_idx < 2 * H:
            card_idx = orig_idx - H
            return max_H + card_idx
        
        # Color clues
        color_clue_start = 2 * H
        num_color_clues = (N - 1) * C
        if orig_idx < color_clue_start + num_color_clues:
            rel_idx = orig_idx - color_clue_start
            player_offset_minus_1 = rel_idx // C  # 0 to N-2
            color = rel_idx % C
            # Map to max action space
            max_color_clue_start = 2 * max_H
            return max_color_clue_start + player_offset_minus_1 * max_C + color
        
        # Rank clues
        rank_clue_start = color_clue_start + num_color_clues
        rel_idx = orig_idx - rank_clue_start
        player_offset_minus_1 = rel_idx // R
        rank = rel_idx % R
        # Map to max action space
        max_rank_clue_start = 2 * max_H + (max_N - 1) * max_C
        return max_rank_clue_start + player_offset_minus_1 * max_R + rank
    
    def _process_transition(self, t: Transition) -> Dict[str, torch.Tensor]:
        """Process and pad a transition to fixed dimensions."""
        config = t.game_config
        num_players = config.get("num_players", 2)
        num_colors = config.get("num_colors", 5)
        num_ranks = config.get("num_ranks", 5)
        hand_size = config.get("hand_size", 5)
        
        # Pad slot beliefs to max dimensions
        # Original: [P, H, C+R]
        # Target: [max_P, max_H, max_C + max_R]
        slot_beliefs = np.zeros(
            (self.max_num_players, self.max_hand_size, self.max_num_colors + self.max_num_ranks),
            dtype=np.float32
        )
        slot_beliefs[:num_players, :hand_size, :num_colors] = t.slot_beliefs[:, :, :num_colors]
        slot_beliefs[:num_players, :hand_size, self.max_num_colors:self.max_num_colors + num_ranks] = \
            t.slot_beliefs[:, :, num_colors:num_colors + num_ranks]
        
        # Pad affected mask
        affected_mask = np.zeros((self.max_num_players, self.max_hand_size), dtype=np.float32)
        affected_mask[:num_players, :hand_size] = t.affected_mask[:num_players, :hand_size]
        
        # Pad fireworks
        fireworks = np.zeros(self.max_num_colors, dtype=np.float32)
        fireworks[:num_colors] = t.fireworks[:num_colors]
        
        # Pad discard pile
        discard_pile = np.zeros(self.max_num_colors * self.max_num_ranks, dtype=np.float32)
        # Map original indices to padded indices
        for orig_idx in range(len(t.discard_pile)):
            if t.discard_pile[orig_idx] > 0:
                orig_color = orig_idx // num_ranks
                orig_rank = orig_idx % num_ranks
                if orig_color < self.max_num_colors and orig_rank < self.max_num_ranks:
                    new_idx = orig_color * self.max_num_ranks + orig_rank
                    discard_pile[new_idx] = t.discard_pile[orig_idx]
        
        # Remap legal moves mask with proper action index mapping
        legal_moves_mask = np.zeros(self.max_action_space_size, dtype=np.float32)
        for orig_idx, is_legal in enumerate(t.legal_moves_mask):
            if is_legal:
                new_idx = self._remap_action_index(
                    orig_idx, num_players, num_colors, num_ranks, hand_size
                )
                if new_idx < self.max_action_space_size:
                    legal_moves_mask[new_idx] = 1.0
        
        # Remap chosen action index
        chosen_action_idx_remapped = self._remap_action_index(
            t.chosen_action_idx, num_players, num_colors, num_ranks, hand_size
        )
        
        # Pad true colors and ranks
        true_colors = np.full(self.max_hand_size, -1, dtype=np.int64)
        true_ranks = np.full(self.max_hand_size, -1, dtype=np.int64)
        valid_len = min(len(t.true_colors), self.max_hand_size)
        true_colors[:valid_len] = t.true_colors[:valid_len]
        true_ranks[:valid_len] = t.true_ranks[:valid_len]
        
        return {
            "slot_beliefs": torch.from_numpy(slot_beliefs),
            "affected_mask": torch.from_numpy(affected_mask),
            "move_target_player": torch.tensor(t.move_target_player, dtype=torch.long),
            "acting_player": torch.tensor(t.acting_player, dtype=torch.long),
            "action": torch.from_numpy(t.action.astype(np.float32)),
            "fireworks": torch.from_numpy(fireworks),
            "discard_pile": torch.from_numpy(discard_pile),
            "true_colors": torch.from_numpy(true_colors),
            "true_ranks": torch.from_numpy(true_ranks),
            "chosen_action_idx": torch.tensor(chosen_action_idx_remapped, dtype=torch.long),
            "legal_moves_mask": torch.from_numpy(legal_moves_mask),
            "reward": torch.tensor(t.reward, dtype=torch.float32),
            "failed_play": torch.tensor(t.failed_play, dtype=torch.bool),
            "done": torch.tensor(t.done, dtype=torch.bool),
            # Game config info for potential conditional processing
            "num_players": torch.tensor(num_players, dtype=torch.long),
            "num_colors": torch.tensor(num_colors, dtype=torch.long),
            "num_ranks": torch.tensor(num_ranks, dtype=torch.long),
            "hand_size": torch.tensor(hand_size, dtype=torch.long),
        }


class StreamingHanabiDataset(IterableDataset):
    """
    Iterable dataset that continuously samples from the replay buffer.
    Useful for training while games are being played.
    """
    
    def __init__(
        self,
        buffer: ReplayBuffer,
        batch_size: int,
        max_num_players: int = 5,
        max_num_colors: int = 5,
        max_num_ranks: int = 5,
        max_hand_size: int = 5,
    ):
        self.buffer = buffer
        self.batch_size = batch_size
        self.max_num_players = max_num_players
        self.max_num_colors = max_num_colors
        self.max_num_ranks = max_num_ranks
        self.max_hand_size = max_hand_size
        
        # Helper dataset for processing
        self._processor = HanabiDataset(
            buffer, max_num_players, max_num_colors, max_num_ranks, max_hand_size
        )
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        while True:
            if len(self.buffer) < self.batch_size:
                continue
            
            transitions = self.buffer.sample(self.batch_size)
            
            for t in transitions:
                yield self._processor._process_transition(t)


def create_dataloader(
    buffer: ReplayBuffer,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    max_num_players: int = 5,
    max_num_colors: int = 5,
    max_num_ranks: int = 5,
    max_hand_size: int = 5,
) -> DataLoader:
    """Create a DataLoader from a replay buffer."""
    dataset = HanabiDataset(
        buffer,
        max_num_players=max_num_players,
        max_num_colors=max_num_colors,
        max_num_ranks=max_num_ranks,
        max_hand_size=max_hand_size,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
