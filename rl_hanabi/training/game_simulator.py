"""
Game simulator for self-play training using tokenized state/action inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import torch

from hanabi_learning_environment import pyhanabi

from rl_hanabi.game import HLEGameState, GameConfig
from rl_hanabi.model import ActionDecoder
from rl_hanabi.model import HLETokenizer


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    T = len(rewards)

    advantages = torch.zeros_like(rewards)
    last_adv = torch.zeros((), dtype=rewards.dtype, device=rewards.device)

    for t in reversed(range(T)):
        if t == T - 1:
            next_value = torch.zeros_like(values[t])
        else:
            next_value = values[t + 1]

        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        last_adv = delta + gamma * lam * (1 - dones[t]) * last_adv
        advantages[t] = last_adv

    returns = advantages + values

    return advantages, returns


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
    advantage: float = 0.0
    return_value: float = 0.0


@dataclass
class GameResult:
    transitions: List[Transition]
    final_score: int
    max_possible_score: int
    num_turns: int
    game_config: Dict[str, int]
    debug_log: List[str] = field(default_factory=list)


class GameSimulator:
    def __init__(
        self,
        model: ActionDecoder,
        tokenizer: HLETokenizer,
        device: torch.device,
        temperature: float = 1.0,
        gamma: float = 0.99,
        lam: float = 0.95,
        early_play_bonus: float = 1.0,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.temperature = temperature
        self.gamma = gamma
        self.lam = lam
        self.early_play_bonus = early_play_bonus
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

    def _blocked_colors(self, state: HLEGameState) -> set[int]:
        fireworks = state.fireworks()
        discard_pile = state.discard_pile()
        num_colors = state.game.num_colors()
        num_ranks = state.game.num_ranks()

        discarded: Dict[tuple[int, int], int] = {}
        for card in discard_pile:
            key = (card.color(), card.rank())
            discarded[key] = discarded.get(key, 0) + 1

        blocked: set[int] = set()
        for color in range(num_colors):
            for rank in range(num_ranks):
                total_copies = state.game.num_cards(color, rank)
                if discarded.get((color, rank), 0) >= total_copies and fireworks[color] <= rank:
                    blocked.add(color)
                    break

        return blocked

    def _compute_step_reward(
        self,
        move: pyhanabi.HanabiMove,
        fireworks_before: List[int],
        fireworks_after: List[int],
        max_score: int,
        blocked_before: set[int] | None,
        state_after: HLEGameState,
    ) -> float:
        reward = 0.0
        move_type = move.type()

        if move_type == pyhanabi.HanabiMoveType.PLAY:
            if sum(fireworks_after) > sum(fireworks_before):
                # Reward early successful plays slightly more.
                # fireworks entries are counts (0..num_ranks); sum is current score.
                progress = 0.0
                if max_score > 0:
                    progress = min(1.0, max(0.0, float(sum(fireworks_before)) / float(max_score)))
                reward += 1.0 + self.early_play_bonus * (1.0 - progress)
            else:
                reward -= 1.0
        elif move_type == pyhanabi.HanabiMoveType.DISCARD:
            blocked_after = self._blocked_colors(state_after)
            blocked_before = blocked_before or set()
            if len(blocked_after) > len(blocked_before):
                reward -= 1.0

        return reward

    @staticmethod
    def _format_move(move: pyhanabi.HanabiMove, current_player: int, num_players: int) -> str:
        move_type = move.type()
        if move_type == pyhanabi.HanabiMoveType.PLAY:
            return f"PLAY slot {move.card_index()}"
        if move_type == pyhanabi.HanabiMoveType.DISCARD:
            return f"DISCARD slot {move.card_index()}"
        if move_type == pyhanabi.HanabiMoveType.REVEAL_COLOR:
            target = (current_player + move.target_offset()) % num_players
            return f"CLUE color {move.color()} -> player {target}"
        if move_type == pyhanabi.HanabiMoveType.REVEAL_RANK:
            target = (current_player + move.target_offset()) % num_players
            return f"CLUE rank {move.rank() + 1} -> player {target}"
        return str(move)

    @torch.no_grad()
    def simulate_game(
        self,
        config: GameConfig,
        *,
        capture_states: bool = False,
    ) -> GameResult:
        state = HLEGameState.from_table_options(config)

        player_models = self._get_player_models(state.num_players)

        # Model device/eval state should be set once in the worker.
        transitions: List[Transition] = []
        debug_log: List[str] = []
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
            current_player_action_prob: float | None = None

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
                        sampled_idx = int(torch.multinomial(probs, 1).item())
                        current_player_action_idx = sampled_idx
                        current_player_action_prob = float(probs[0, sampled_idx].item())
                    else:
                        probs = torch.softmax(masked_logits, dim=-1)
                        current_player_action_idx = int(masked_logits.argmax(dim=-1).item())
                        current_player_action_prob = float(
                            probs[0, current_player_action_idx].item()
                        )

                    current_player_value = value
                    current_player_tokens = tokens

            if current_player_action_idx is None or current_player_tokens is None:
                break

            action_idx = current_player_action_idx

            move = state.index_to_move(action_idx)
            fireworks_before = state.fireworks()
            score_before = sum(fireworks_before)
            blocked_before = None
            if move.type() == pyhanabi.HanabiMoveType.DISCARD:
                blocked_before = self._blocked_colors(state)

            state.apply_move_by_index(action_idx)
            previous_action_idx = action_idx

            reward = self._compute_step_reward(
                move=move,
                fireworks_before=fireworks_before,
                fireworks_after=state.fireworks(),
                max_score=state.max_score(),
                blocked_before=blocked_before,
                state_after=state,
            )

            score_after = state.score()
            if capture_states:
                debug_log.append(
                    "\n".join(
                        [
                            f"Turn {num_turns:02d} P{current_player} idx={action_idx:3d} "
                            f"{self._format_move(move, current_player, state.num_players)} "
                            f"score {score_before}->{score_after} r={reward:+.3f} "
                            f"p={current_player_action_prob:.4f}",
                            f"Score {score_after}/{state.max_score()}",
                            str(state.state),
                            "-" * 72,
                        ]
                    )
                )

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
                    reward=reward,
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

        if transitions:
            for transition in transitions:
                transition.reward += final_score

        if transitions:
            rewards = torch.tensor([t.reward for t in transitions], dtype=torch.float32)
            values = torch.tensor([t.value for t in transitions], dtype=torch.float32)
            dones = torch.tensor([t.done for t in transitions], dtype=torch.float32)
            advantages, returns = compute_gae(
                rewards,
                values,
                dones,
                gamma=self.gamma,
                lam=self.lam,
            )
            for transition, advantage, return_value in zip(
                transitions,
                advantages.tolist(),
                returns.tolist(),
            ):
                transition.advantage = advantage
                transition.return_value = return_value

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
            debug_log=debug_log,
        )

