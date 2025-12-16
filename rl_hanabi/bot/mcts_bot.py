import logging
import numpy as np
from typing import Any, Dict, Optional

from rl_hanabi.bot.base_bot import BaseBot
from rl_hanabi.game.game_types import ACTION
from rl_hanabi.mcts.mcts import MCTS
from hanabi_learning_environment import pyhanabi

class MCTSBot(BaseBot):
    def __init__(self, time_limit_ms: int = 1000) -> None:
        super().__init__(enable_hle=True)
        self.logger = logging.getLogger("rl_hanabi.mcts_bot")
        self.time_limit_ms = time_limit_ms

    def make_move(self) -> None:
        if not self.state or not self.state.hle_state:
            self.logger.error("Cannot make move: state or hle_state is missing")
            return

        self.logger.info("It's my turn! Thinking...")

        # Initialize MCTS
        mcts = MCTS(model=None, device=None, time_ms=self.time_limit_ms)
        
        # Use a copy of the current HLE state as root
        root_state = self.state.hle_state.copy()
        print(f"Root state before MCTS:\n{root_state}\n")
        mcts.init_root(root_state, c=1)
        self.logger.info("Starting MCTS for %d ms...", self.time_limit_ms)
        
        # Run MCTS
        try:
            prob_dist, q_value = mcts.run(use_rollouts=True)
        except Exception as e:
            self.logger.error(f"MCTS run failed: {e}", exc_info=True)
            return

        # Select best move
        best_move_index = np.argmax(prob_dist)
        self.logger.info(f"Best move index: {best_move_index}")
        best_move = self.state.hle_state.index_to_move(int(best_move_index))
        
        self.logger.info(f"MCTS selected move: {best_move} with Q-value: {q_value:.4f}")
        
        self._send_hanabi_move(best_move)

    def _send_hanabi_move(self, move: pyhanabi.HanabiMove) -> None:
        if not self.state:
            return

        move_type = move.type()
        payload = {"tableID": self.table_id}
        
        if move_type == pyhanabi.HanabiMoveType.PLAY:
            card_index = move.card_index()
            # HLE index 0 is oldest, hanab.live index 0 is newest.
            # We need to convert HLE index to hanab.live index.
            hand_size = len(self.state.our_hand)
            hanab_index = hand_size - 1 - card_index

            if 0 <= hanab_index < hand_size:
                card_order = self.state.our_hand[hanab_index]
                payload["type"] = ACTION.PLAY
                payload["target"] = card_order
                self.send_cmd("action", payload)
            else:
                self.logger.error(f"Invalid card index for play: {card_index} -> {hanab_index}")

        elif move_type == pyhanabi.HanabiMoveType.DISCARD:
            card_index = move.card_index()
            hand_size = len(self.state.our_hand)
            hanab_index = hand_size - 1 - card_index

            if 0 <= hanab_index < hand_size:
                card_order = self.state.our_hand[hanab_index]
                payload["type"] = ACTION.DISCARD
                payload["target"] = card_order
                self.send_cmd("action", payload)
            else:
                self.logger.error(f"Invalid card index for discard: {card_index} -> {hanab_index}")

        elif move_type == pyhanabi.HanabiMoveType.REVEAL_COLOR:
            target_offset = move.target_offset()
            target_player = (self.state.our_index + target_offset) % self.state.num_players
            color = move.color()
            
            payload["type"] = ACTION.COLOUR
            payload["target"] = target_player
            payload["value"] = color
            self.send_cmd("action", payload)

        elif move_type == pyhanabi.HanabiMoveType.REVEAL_RANK:
            target_offset = move.target_offset()
            target_player = (self.state.our_index + target_offset) % self.state.num_players
            rank = move.rank() # 0-based
            
            payload["type"] = ACTION.RANK
            payload["target"] = target_player
            payload["value"] = rank + 1 # 1-based for hanab.live
            self.send_cmd("action", payload)

    # Optional: Override chat to give specific version info
    def _handle_chat(self, data: Dict[str, Any]) -> None:
        msg: str = data.get("msg", "")
        who: str = data.get("who", "")
        
        if msg.startswith("/version"):
            self.send_pm(who, "mcts-bot v0.1")
            return
        elif msg.startswith("/settings"):
            self.send_pm(who, "This bot uses MCTS.")
            return
            
        super()._handle_chat(data)
