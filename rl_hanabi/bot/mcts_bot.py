import logging
import threading
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
        self._thinking = False
        self._thinking_lock = threading.Lock()

    def make_move(self) -> None:
        """Start MCTS computation in a background thread to avoid blocking WebSocket."""
        if not self.state or not self.state.hle_state:
            self.logger.error("Cannot make move: state or hle_state is missing")
            return

        # Prevent multiple concurrent MCTS runs
        with self._thinking_lock:
            if self._thinking:
                self.logger.warning("Already thinking, ignoring duplicate make_move call")
                return
            self._thinking = True

        self.logger.info("It's my turn! Starting MCTS in background thread...")
        
        # Copy state data needed for MCTS before starting thread
        root_state = self.state.hle_state.copy()
        
        # Run MCTS in background thread to keep WebSocket alive
        thread = threading.Thread(
            target=self._run_mcts_and_send_move,
            args=(root_state,),
            daemon=True
        )
        thread.start()

    def _run_mcts_and_send_move(self, root_state) -> None:
        """Run MCTS computation and send the result. Called in background thread."""
        try:
            # Initialize MCTS
            mcts = MCTS(model=None, device=None, time_ms=self.time_limit_ms)
            
            print(f"Root state before MCTS:\n{root_state}\n")
            mcts.init_root(root_state, c=1.4)
            self.logger.info("Starting MCTS for %d ms...", self.time_limit_ms)
            
            # Run MCTS with reduced parameters to prevent timeout
            prob_dist, q_value = mcts.run(
                use_rollouts=True,
                num_rollouts=1,
                rollout_depth=30
            )

            # Select best move
            best_move_index = np.argmax(prob_dist)
            self.logger.info(f"Best move index: {best_move_index}")
            best_move = root_state.index_to_move(int(best_move_index))
            
            self.logger.info(f"MCTS selected move: {best_move} with Q-value: {q_value:.4f}")
            
            # Send the move (thread-safe via send queue)
            self._send_hanabi_move(best_move)
            
        except Exception as e:
            self.logger.error(f"MCTS run failed: {e}", exc_info=True)
        finally:
            with self._thinking_lock:
                self._thinking = False

    def _send_hanabi_move(self, move: pyhanabi.HanabiMove) -> None:
        if not self.state:
            return

        logging.info(f"Sending Hanabi move: \n{move}\n")
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
