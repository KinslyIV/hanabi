"""
Game simulator for bot-vs-bot training using H-Group conventions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from hanabi_learning_environment import pyhanabi

from rl_hanabi.bot.hgroup_bot import HGroupBotPolicy
from rl_hanabi.game import HLEGameState, GameConfig
from rl_hanabi.model import HLETokenizer
from rl_hanabi.training.game_simulator import Transition, GameResult


@dataclass
class BotGameSimulator:
    tokenizer: HLETokenizer
    bot_policy: HGroupBotPolicy = field(default_factory=HGroupBotPolicy)

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

    def simulate_game(
        self,
        config: GameConfig,
        *,
        capture_states: bool = False,
    ) -> GameResult:
        state = HLEGameState.from_table_options(config)
        self.bot_policy.reset(state)
        transitions: List[Transition] = []
        debug_log: List[str] = []
        num_turns = 0
        previous_action_idx = -1

        while not state.is_terminal():
            current_player = state.current_player_index
            legal_moves_mask = state.legal_moves_mask()
            if not legal_moves_mask.any():
                break

            tokens = self.tokenizer.tokenize_state_and_action(
                state,
                previous_action_idx,
                current_player,
            )

            action_idx = self.bot_policy.select_action_index(state)
            move = state.index_to_move(action_idx)
            self.bot_policy.observe_move(state, move)
            state.apply_move_by_index(action_idx)
            previous_action_idx = action_idx

            if capture_states:
                debug_log.append(
                    "\n".join(
                        [
                            f"Turn {num_turns:02d} P{current_player} idx={action_idx:3d} "
                            f"{self._format_move(move, current_player, state.num_players)}",
                            f"Score {state.score()}/{state.max_score()}",
                            str(state.state),
                            "-" * 72,
                        ]
                    )
                )

            transitions.append(
                Transition(
                    tokens=tokens,
                    legal_moves_mask=legal_moves_mask.tolist(),
                    chosen_action_idx=action_idx,
                    value=0.0,
                    reward=0.0,
                    done=False,
                    current_player=current_player,
                    game_config={
                        "num_players": config.num_players,
                        "num_colors": config.num_colors,
                        "num_ranks": config.num_ranks,
                        "hand_size": config.hand_size,
                    },
                    advantage=0.0,
                    return_value=0.0,
                    teacher_action_idx=-1,
                    teacher_mask=False,
                )
            )
            num_turns += 1

        if transitions:
            transitions[-1].done = True

        return GameResult(
            transitions=transitions,
            final_score=state.score(),
            max_possible_score=state.max_score(),
            num_turns=num_turns,
            game_config={
                "num_players": config.num_players,
                "num_colors": config.num_colors,
                "num_ranks": config.num_ranks,
                "hand_size": config.hand_size,
            },
            debug_log=debug_log,
        )
