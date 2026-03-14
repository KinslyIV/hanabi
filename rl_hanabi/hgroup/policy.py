"""
H-Group convention policy (ported from hanabi-bot beginner conventions).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from hanabi_learning_environment.pyhanabi import (
    HanabiCard,
    HanabiMove,
    HanabiMoveType,
)

from rl_hanabi.game.hle_state import HLEGameState


@dataclass
class HGroupPolicy:
    """Beginner H-Group convention policy for HLEGameState."""

    play_weight: float = 10.0
    clue_weight: float = 5.0
    save_weight: float = 8.0
    discard_weight: float = 1.0

    def get_fireworks(self, state: HLEGameState) -> List[int]:
        return state.fireworks()

    def is_playable(self, card: HanabiCard, fireworks: List[int]) -> bool:
        return card.rank() == fireworks[card.color()]

    def is_critical(self, card: HanabiCard, state: HLEGameState) -> bool:
        color, rank = card.color(), card.rank()
        if rank == 4:
            return True
        discard_pile = state.discard_pile()
        discarded_count = sum(1 for c in discard_pile if c.color() == color and c.rank() == rank)
        total_copies = state.game.num_cards(color, rank)
        return (total_copies - discarded_count) == 1

    def is_already_played(self, card: HanabiCard, fireworks: List[int]) -> bool:
        return card.rank() < fireworks[card.color()]

    def is_card_clued(self, card_knowledge) -> bool:
        return card_knowledge.color() is not None or card_knowledge.rank() is not None

    def get_clued_status(self, state: HLEGameState, player_idx: int) -> List[bool]:
        observation = state.observation_for_player(player_idx)
        card_knowledge = observation.card_knowledge()
        player_knowledge = card_knowledge[0]
        return [self.is_card_clued(ck) for ck in player_knowledge]

    def get_other_player_clued_status(
        self,
        state: HLEGameState,
        current_player: int,
        target_player: int,
    ) -> List[bool]:
        observation = state.observation_for_player(current_player)
        card_knowledge = observation.card_knowledge()
        num_players = state.num_players
        target_offset = (target_player - current_player) % num_players
        player_knowledge = card_knowledge[target_offset]
        return [self.is_card_clued(ck) for ck in player_knowledge]

    def get_chop_index(self, hand: List[HanabiCard], state: HLEGameState, player_idx: int) -> int:
        if not hand:
            return 0
        try:
            current_player = state.current_player_index
            if player_idx == current_player:
                clued_status = self.get_clued_status(state, player_idx)
            else:
                clued_status = self.get_other_player_clued_status(state, current_player, player_idx)
            for i in range(len(hand)):
                if i < len(clued_status) and not clued_status[i]:
                    return i
            return 0
        except Exception:
            return 0

    def get_playable_cards_in_hand(self, hand: List[HanabiCard], fireworks: List[int]) -> List[int]:
        playable = []
        for i, card in enumerate(hand):
            if self.is_playable(card, fireworks):
                playable.append(i)
        return playable

    def get_save_candidates(
        self,
        hand: List[HanabiCard],
        state: HLEGameState,
        fireworks: List[int],
    ) -> Tuple[bool, bool, bool]:
        if not hand:
            return False, False, False
        chop_idx = self.get_chop_index(hand, state, -1)
        chop_card = hand[chop_idx]
        if self.is_already_played(chop_card, fireworks):
            return False, False, False
        needs_5_save = chop_card.rank() == 4
        needs_2_save = chop_card.rank() == 1
        needs_critical_save = (not needs_5_save and self.is_critical(chop_card, state))
        return needs_5_save, needs_2_save, needs_critical_save

    def evaluate_play_clue(
        self,
        target_hand: List[HanabiCard],
        fireworks: List[int],
        clue_type: str,
        clue_value: int,
        target_clued_status: Optional[List[bool]] = None,
    ) -> float:
        if clue_type not in ("color", "rank"):
            return 0.0
        playable_cards = []
        new_clued_cards = []
        touched_cards = []
        for i, card in enumerate(target_hand):
            touched = False
            if clue_type == "color" and card.color() == clue_value:
                touched = True
            elif clue_type == "rank" and card.rank() == clue_value:
                touched = True
            if not touched:
                continue
            touched_cards.append(i)
            if self.is_playable(card, fireworks):
                playable_cards.append(i)
            if target_clued_status and i < len(target_clued_status) and not target_clued_status[i]:
                new_clued_cards.append(i)
        if not playable_cards:
            return 0.0
        score = 2.0 * len(playable_cards)
        score += 0.25 * len(new_clued_cards)
        score -= 0.1 * max(0, len(touched_cards) - len(playable_cards))
        return score

    def _visible_count(
        self,
        state: HLEGameState,
        current_player: int,
        color: int,
        rank: int,
    ) -> int:
        count = 0
        for p_idx, hand in enumerate(state.state.player_hands()):
            if p_idx == current_player:
                continue
            count += sum(1 for card in hand if card.color() == color and card.rank() == rank)
        count += sum(1 for card in state.discard_pile() if card.color() == color and card.rank() == rank)
        fireworks = state.fireworks()
        if fireworks[color] > rank:
            count += 1
        return count

    def _needs_save2(
        self,
        state: HLEGameState,
        current_player: int,
        card: HanabiCard,
    ) -> bool:
        if card.rank() != 1:
            return False
        fireworks = state.fireworks()
        if fireworks[card.color()] >= 2:
            return False
        visible = self._visible_count(state, current_player, card.color(), card.rank())
        return visible == 1

    def _known_playable_indices(self, state: HLEGameState) -> List[int]:
        current_player = state.current_player_index
        observation = state.observation_for_player(current_player)
        knowledge = observation.card_knowledge()[0]
        fireworks = state.fireworks()
        playable = []
        for idx, know in enumerate(knowledge):
            color = know.color()
            rank = know.rank()
            if color is None or rank is None:
                continue
            if 0 <= color < len(fireworks) and fireworks[color] == rank:
                playable.append(idx)
        return playable

    def _clue_targets(self, state: HLEGameState) -> List[int]:
        current_player = state.current_player_index
        return [p for p in range(state.num_players) if p != current_player]

    def _clue_touches(
        self,
        hand: List[HanabiCard],
        move: HanabiMove,
    ) -> List[int]:
        if move.type() == HanabiMoveType.REVEAL_COLOR:
            return [i for i, card in enumerate(hand) if card.color() == move.color()]
        if move.type() == HanabiMoveType.REVEAL_RANK:
            return [i for i, card in enumerate(hand) if card.rank() == move.rank()]
        return []

    def _find_best_play_clue(self, state: HLEGameState) -> Tuple[Optional[HanabiMove], float]:
        current_player = state.current_player_index
        fireworks = state.fireworks()
        best_move = None
        best_score = 0.0
        for move in state.legal_moves():
            if move.type() not in (HanabiMoveType.REVEAL_COLOR, HanabiMoveType.REVEAL_RANK):
                continue
            target = (current_player + move.target_offset()) % state.num_players
            target_hand = state.state.player_hands()[target]
            if not target_hand:
                continue
            target_clued = self.get_other_player_clued_status(state, current_player, target)
            clue_type = "color" if move.type() == HanabiMoveType.REVEAL_COLOR else "rank"
            clue_value = move.color() if clue_type == "color" else move.rank()
            score = self.evaluate_play_clue(
                target_hand, fireworks, clue_type, clue_value, target_clued
            )
            if score > best_score:
                best_score = score
                best_move = move
        return best_move, best_score

    def _find_best_save_clue(self, state: HLEGameState) -> Tuple[Optional[HanabiMove], float]:
        current_player = state.current_player_index
        fireworks = state.fireworks()
        best_move = None
        best_score = 0.0
        for move in state.legal_moves():
            if move.type() not in (HanabiMoveType.REVEAL_COLOR, HanabiMoveType.REVEAL_RANK):
                continue
            target = (current_player + move.target_offset()) % state.num_players
            target_hand = state.state.player_hands()[target]
            if not target_hand:
                continue
            target_clued = self.get_other_player_clued_status(state, current_player, target)
            chop_idx = self.get_chop_index(target_hand, state, target)
            if chop_idx >= len(target_hand):
                continue
            if chop_idx < len(target_clued) and target_clued[chop_idx]:
                continue
            chop_card = target_hand[chop_idx]
            needs_5_save = chop_card.rank() == 4
            needs_2_save = self._needs_save2(state, current_player, chop_card)
            needs_critical_save = self.is_critical(chop_card, state)
            if not (needs_5_save or needs_2_save or needs_critical_save):
                continue
            touched = self._clue_touches(target_hand, move)
            if chop_idx not in touched:
                continue
            play_bonus = 0.0
            for idx in touched:
                if self.is_playable(target_hand[idx], fireworks):
                    play_bonus += 0.5
            score = 3.0 + play_bonus - 0.1 * len(touched)
            if needs_5_save and move.type() == HanabiMoveType.REVEAL_RANK and move.rank() == 4:
                score += 1.0
            if needs_2_save and move.type() == HanabiMoveType.REVEAL_RANK and move.rank() == 1:
                score += 0.5
            if score > best_score:
                best_score = score
                best_move = move
        return best_move, best_score

    def get_move_weights(self, state: HLEGameState) -> np.ndarray:
        legal_moves = state.legal_moves()
        weights = np.zeros(state.action_space_size, dtype=np.float64)
        if not legal_moves:
            return weights
        current_player = state.current_player_index
        fireworks = self.get_fireworks(state)
        current_hand = state.state.player_hands()[current_player]
        known_playables = set(self._known_playable_indices(state))
        best_play_clue, play_score = self._find_best_play_clue(state)
        best_save_clue, save_score = self._find_best_save_clue(state)
        for move in legal_moves:
            move_idx = state.move_to_index(move)
            move_type = move.type()
            if move_type == HanabiMoveType.PLAY:
                weights[move_idx] = self.play_weight if move.card_index() in known_playables else 0.1
                continue
            if move_type == HanabiMoveType.DISCARD:
                chop_idx = self.get_chop_index(current_hand, state, current_player)
                weights[move_idx] = self.discard_weight if move.card_index() == chop_idx else 0.2
                continue
            if move == best_save_clue and save_score > 0:
                weights[move_idx] = self.save_weight + save_score
                continue
            if move == best_play_clue and play_score > 0:
                weights[move_idx] = self.clue_weight + play_score
                continue
            weights[move_idx] = 0.05
        total = weights.sum()
        if total > 0:
            weights = weights / total
        else:
            for move in legal_moves:
                weights[state.move_to_index(move)] = 1.0 / len(legal_moves)
        return weights

    def select_action_index(self, state: HLEGameState, *, stochastic: bool = False) -> int:
        legal_moves = state.legal_moves()
        if not legal_moves:
            raise ValueError("No legal moves available")
        if stochastic:
            weights = self.get_move_weights(state)
            legal_indices = [state.move_to_index(m) for m in legal_moves]
            legal_weights = weights[legal_indices]
            legal_weights = legal_weights / legal_weights.sum()
            chosen_idx = int(np.random.choice(len(legal_indices), p=legal_weights))
            return legal_indices[chosen_idx]

        current_player = state.current_player_index
        fireworks = state.fireworks()
        known_playables = self._known_playable_indices(state)
        if known_playables:
            return state.move_to_index(HanabiMove.get_play_move(known_playables[0]))

        best_play_clue, _ = self._find_best_play_clue(state)
        best_save_clue, _ = self._find_best_save_clue(state)
        clue_tokens = state.information_tokens()
        if clue_tokens > 0:
            if best_play_clue and (best_save_clue is None or clue_tokens > 1):
                return state.move_to_index(best_play_clue)
            if best_save_clue:
                return state.move_to_index(best_save_clue)
            if best_play_clue:
                return state.move_to_index(best_play_clue)

        current_hand = state.state.player_hands()[current_player]
        chop_idx = self.get_chop_index(current_hand, state, current_player)
        discard_move = HanabiMove.get_discard_move(chop_idx)
        if state.move_to_index(discard_move) in [state.move_to_index(m) for m in legal_moves]:
            return state.move_to_index(discard_move)

        return state.move_to_index(legal_moves[0])

    def select_move(self, state: HLEGameState, *, stochastic: bool = False) -> HanabiMove:
        return state.index_to_move(self.select_action_index(state, stochastic=stochastic))
