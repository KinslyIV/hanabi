"""
H-Group convention policy (level 1) using HLE card knowledge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
    """Beginner H-Group convention policy for HLEGameState (standard variants only)."""

    play_weight: float = 10.0
    clue_weight: float = 5.0
    save_weight: float = 8.0
    discard_weight: float = 1.0
    _called_to_play: List[List[bool]] = field(default_factory=list, init=False)

    def reset(self, state: HLEGameState) -> None:
        self._called_to_play = [
            [False for _ in hand] for hand in state.state.player_hands()
        ]

    def _ensure_state(self, state: HLEGameState) -> None:
        if not self._called_to_play or len(self._called_to_play) != state.num_players:
            self.reset(state)
            return
        for idx, hand in enumerate(state.state.player_hands()):
            if idx >= len(self._called_to_play):
                self.reset(state)
                return
            if len(self._called_to_play[idx]) != len(hand):
                self.reset(state)
                return

    def observe_move(self, state_before: HLEGameState, move: HanabiMove) -> None:
        self._ensure_state(state_before)
        current_player = state_before.current_player_index

        if move.type() in (HanabiMoveType.PLAY, HanabiMoveType.DISCARD):
            idx = move.card_index()
            if 0 <= idx < len(self._called_to_play[current_player]):
                self._called_to_play[current_player].pop(idx)
            self._called_to_play[current_player].append(False)
            return

        if move.type() not in (HanabiMoveType.REVEAL_COLOR, HanabiMoveType.REVEAL_RANK):
            return

        target = (current_player + move.target_offset()) % state_before.num_players
        target_hand = state_before.state.player_hands()[target]
        if not target_hand:
            return
        touched = self._clue_touches(target_hand, move)
        if not touched:
            return
        clued_status = self._clued_status(state_before, current_player, target)
        chop_idx = self._get_chop_index(state_before, target, target)
        newly_touched = [i for i in touched if i < len(clued_status) and not clued_status[i]]
        if chop_idx in touched:
            focus_idx = chop_idx
        elif newly_touched:
            focus_idx = newly_touched[0]
        else:
            focus_idx = touched[0]
        if focus_idx >= len(target_hand):
            return
        focus_card = target_hand[focus_idx]
        playable = state_before.fireworks()[focus_card.color()] == focus_card.rank()
        self._called_to_play[target][focus_idx] = playable

    def _possible_identities(
        self,
        state: HLEGameState,
        observer: int,
        target: int,
    ) -> List[List[Tuple[int, int]]]:
        observation = state.observation_for_player(observer)
        knowledge = observation.card_knowledge()
        offset = (target - observer) % state.num_players
        target_knowledge = knowledge[offset]
        num_colors = state.game.num_colors()
        num_ranks = state.game.num_ranks()
        possibilities: List[List[Tuple[int, int]]] = []
        for card_knowledge in target_knowledge:
            poss: List[Tuple[int, int]] = []
            for color in range(num_colors):
                if not card_knowledge.color_plausible(color):
                    continue
                for rank in range(num_ranks):
                    if card_knowledge.rank_plausible(rank):
                        poss.append((color, rank))
            possibilities.append(poss)
        return possibilities

    def _known_playable_indices(self, state: HLEGameState) -> List[int]:
        current_player = state.current_player_index
        observation = state.observation_for_player(current_player)
        knowledge = observation.card_knowledge()[0]
        fireworks = state.fireworks()
        playable = []
        for idx, know in enumerate(knowledge):
            color = know.color()
            rank = know.rank()
            if color is not None and rank is not None:
                if fireworks[color] == rank:
                    playable.append(idx)
                continue
            possibilities = [
                (c, r)
                for c in range(state.game.num_colors())
                for r in range(state.game.num_ranks())
                if know.color_plausible(c) and know.rank_plausible(r)
            ]
            if possibilities and all(fireworks[c] == r for c, r in possibilities):
                playable.append(idx)
        return playable

    def _clued_status(self, state: HLEGameState, observer: int, target: int) -> List[bool]:
        observation = state.observation_for_player(observer)
        knowledge = observation.card_knowledge()
        offset = (target - observer) % state.num_players
        target_knowledge = knowledge[offset]
        return [ck.color() is not None or ck.rank() is not None for ck in target_knowledge]

    def _clue_touches(self, hand: List[HanabiCard], move: HanabiMove) -> List[int]:
        if move.type() == HanabiMoveType.REVEAL_COLOR:
            return [i for i, card in enumerate(hand) if card.color() == move.color()]
        if move.type() == HanabiMoveType.REVEAL_RANK:
            return [i for i, card in enumerate(hand) if card.rank() == move.rank()]
        return []

    def _get_chop_index(self, state: HLEGameState, observer: int, target: int) -> int:
        hand = state.state.player_hands()[target]
        if not hand:
            return 0
        clued_status = self._clued_status(state, observer, target)
        for i in range(len(hand)):
            if i < len(clued_status) and not clued_status[i]:
                return i
        return 0

    def _is_critical(self, card: HanabiCard, state: HLEGameState) -> bool:
        color, rank = card.color(), card.rank()
        if rank == 4:
            return True
        discarded = sum(1 for c in state.discard_pile() if c.color() == color and c.rank() == rank)
        total = state.game.num_cards(color, rank)
        return total - discarded == 1

    def _visible_count(self, state: HLEGameState, observer: int, card: HanabiCard) -> int:
        color, rank = card.color(), card.rank()
        count = 0
        for p_idx, hand in enumerate(state.state.player_hands()):
            if p_idx == observer:
                continue
            count += sum(1 for c in hand if c.color() == color and c.rank() == rank)
        count += sum(1 for c in state.discard_pile() if c.color() == color and c.rank() == rank)
        if state.fireworks()[color] > rank:
            count += 1
        return count

    def _needs_save2(self, state: HLEGameState, observer: int, card: HanabiCard) -> bool:
        if card.rank() != 1:
            return False
        if state.fireworks()[card.color()] >= 2:
            return False
        return self._visible_count(state, observer, card) == 1

    def _simulate_clue(
        self,
        state: HLEGameState,
        move: HanabiMove,
        observer: int,
        target: int,
    ) -> Tuple[List[List[Tuple[int, int]]], List[List[Tuple[int, int]]]]:
        before = self._possible_identities(state, observer, target)
        next_state = state.copy()
        next_state.apply_move(move)
        after = self._possible_identities(next_state, observer, target)
        return before, after

    def _evaluate_play_clue(self, state: HLEGameState, move: HanabiMove) -> float:
        current_player = state.current_player_index
        target = (current_player + move.target_offset()) % state.num_players
        target_hand = state.state.player_hands()[target]
        if not target_hand:
            return 0.0
        fireworks = state.fireworks()
        before, after = self._simulate_clue(state, move, current_player, target)
        touched = self._clue_touches(target_hand, move)
        if not touched:
            return 0.0
        clued_status = self._clued_status(state, current_player, target)
        chop_idx = self._get_chop_index(state, target, target)
        newly_touched = [i for i in touched if i < len(clued_status) and not clued_status[i]]
        if chop_idx in touched:
            focus_idx = chop_idx
        elif newly_touched:
            focus_idx = newly_touched[0]
        else:
            focus_idx = touched[0]
        focus_card = target_hand[focus_idx]
        if fireworks[focus_card.color()] != focus_card.rank():
            return 0.0
        newly_playable = 0
        newly_informed = 0
        for idx in touched:
            before_poss = before[idx]
            after_poss = after[idx]
            if not after_poss:
                continue
            before_playable = before_poss and all(fireworks[c] == r for c, r in before_poss)
            after_playable = all(fireworks[c] == r for c, r in after_poss)
            if after_playable and not before_playable:
                newly_playable += 1
            if len(after_poss) < len(before_poss):
                newly_informed += 1
        score = 2.5 + 0.5 * newly_playable + 0.25 * newly_informed
        score -= 0.2 * max(0, len(touched) - max(1, newly_playable))
        return score

    def _evaluate_save_clue(self, state: HLEGameState, move: HanabiMove) -> float:
        current_player = state.current_player_index
        target = (current_player + move.target_offset()) % state.num_players
        target_hand = state.state.player_hands()[target]
        if not target_hand:
            return 0.0
        chop_idx = self._get_chop_index(state, target, target)
        if chop_idx >= len(target_hand):
            return 0.0
        chop_card = target_hand[chop_idx]
        needs_5_save = chop_card.rank() == 4
        needs_2_save = self._needs_save2(state, current_player, chop_card)
        needs_critical = self._is_critical(chop_card, state)
        if not (needs_5_save or needs_2_save or needs_critical):
            return 0.0
        touched = self._clue_touches(target_hand, move)
        if chop_idx not in touched:
            return 0.0
        score = 3.0 - 0.1 * len(touched)
        if needs_5_save and move.type() == HanabiMoveType.REVEAL_RANK and move.rank() == 4:
            score += 1.0
        if needs_2_save and move.type() == HanabiMoveType.REVEAL_RANK and move.rank() == 1:
            score += 0.5
        return score

    def _best_play_clue(self, state: HLEGameState) -> Tuple[Optional[HanabiMove], float]:
        best_move = None
        best_score = 0.0
        for move in state.legal_moves():
            if move.type() not in (HanabiMoveType.REVEAL_COLOR, HanabiMoveType.REVEAL_RANK):
                continue
            score = self._evaluate_play_clue(state, move)
            if score > best_score:
                best_score = score
                best_move = move
        return best_move, best_score

    def _best_save_clue(self, state: HLEGameState) -> Tuple[Optional[HanabiMove], float]:
        best_move = None
        best_score = 0.0
        for move in state.legal_moves():
            if move.type() not in (HanabiMoveType.REVEAL_COLOR, HanabiMoveType.REVEAL_RANK):
                continue
            score = self._evaluate_save_clue(state, move)
            if score > best_score:
                best_score = score
                best_move = move
        return best_move, best_score

    def _discard_index(self, state: HLEGameState) -> int:
        current_player = state.current_player_index
        hand = state.state.player_hands()[current_player]
        return self._get_chop_index(state, current_player, current_player)

    def select_action_index(self, state: HLEGameState, *, stochastic: bool = False) -> int:
        self._ensure_state(state)
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
        info_full = state.information_tokens() >= state.game.max_information_tokens()
        called = [
            idx for idx, flag in enumerate(self._called_to_play[current_player]) if flag
        ]
        if called:
            return state.move_to_index(HanabiMove.get_play_move(called[0]))

        known_playables = self._known_playable_indices(state)
        if known_playables:
            return state.move_to_index(HanabiMove.get_play_move(known_playables[0]))

        clue_tokens = state.information_tokens()
        play_clue, play_score = self._best_play_clue(state)
        save_clue, save_score = self._best_save_clue(state)
        if clue_tokens > 0:
            if play_clue and (save_clue is None or clue_tokens > 1):
                return state.move_to_index(play_clue)
            if save_clue:
                return state.move_to_index(save_clue)
            if play_clue:
                return state.move_to_index(play_clue)

        if info_full:
            for move in legal_moves:
                if move.type() in (HanabiMoveType.REVEAL_COLOR, HanabiMoveType.REVEAL_RANK):
                    return state.move_to_index(move)

        discard_idx = self._discard_index(state)
        discard_move = HanabiMove.get_discard_move(discard_idx)
        if state.state.move_is_legal(discard_move):
            return state.move_to_index(discard_move)

        return state.move_to_index(legal_moves[0])

    def get_move_weights(self, state: HLEGameState) -> np.ndarray:
        self._ensure_state(state)
        legal_moves = state.legal_moves()
        weights = np.zeros(state.action_space_size, dtype=np.float64)
        if not legal_moves:
            return weights
        play_clue, play_score = self._best_play_clue(state)
        save_clue, save_score = self._best_save_clue(state)
        known_playables = set(self._known_playable_indices(state))
        current_player = state.current_player_index
        discard_idx = self._get_chop_index(state, current_player, current_player)
        called = {
            idx for idx, flag in enumerate(self._called_to_play[current_player]) if flag
        }
        info_full = state.information_tokens() >= state.game.max_information_tokens()
        for move in legal_moves:
            idx = state.move_to_index(move)
            if move.type() == HanabiMoveType.PLAY:
                if move.card_index() in called:
                    weights[idx] = self.play_weight + 2.0
                elif move.card_index() in known_playables:
                    weights[idx] = self.play_weight
                else:
                    weights[idx] = 0.05
            elif move.type() == HanabiMoveType.DISCARD:
                if info_full:
                    weights[idx] = 0.0
                else:
                    weights[idx] = self.discard_weight if move.card_index() == discard_idx else 0.05
            elif move == play_clue and play_score > 0:
                weights[idx] = self.clue_weight + play_score
            elif move == save_clue and save_score > 0:
                weights[idx] = self.save_weight + save_score
            else:
                weights[idx] = 0.05
        total = weights.sum()
        if total > 0:
            weights = weights / total
        else:
            for move in legal_moves:
                weights[state.move_to_index(move)] = 1.0 / len(legal_moves)
        return weights
