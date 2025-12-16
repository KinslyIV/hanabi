from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Optional

from rl_hanabi.bot.base_bot import BaseBot
from rl_hanabi.game.game_types import ACTION


class RandomBot(BaseBot):
    def __init__(self) -> None:
        super().__init__(enable_hle=False)
        self.logger = logging.getLogger("rl_hanabi.random_bot")

    def make_move(self) -> None:
        if not self.state:
            return
        
        pa = self.state.random_action()
        if pa:
            # small delay to look human-ish
            def _send():
                if self.table_id is None:
                    return
                    
                payload = {"tableID": self.table_id, "type": pa._type, "target": pa.target}
                if pa._type in (ACTION.COLOUR, ACTION.RANK) and pa.value is not None:
                    payload["value"] = pa.value
                
                if self.state:
                    self.logger.info(
                        "Sending action: type=%s target=%s our_hand=%s clues=%s turn=%s",
                        pa._type,
                        pa.target,
                        list(self.state.our_hand),
                        self.state.clue_tokens,
                        self.state.turn_count,
                    )
                self.send_cmd("action", payload)

            threading.Timer(0.5, _send).start()

    # Optional: Override chat to give specific version info
    def _handle_chat(self, data: Dict[str, Any]) -> None:
        msg: str = data.get("msg", "")
        who: str = data.get("who", "")
        
        if msg.startswith("/version"):
            self.send_pm(who, "rl-random-bot v0.1")
            return
        elif msg.startswith("/settings"):
            self.send_pm(who, "This bot ignores convention settings; plays randomly.")
            return
            
        super()._handle_chat(data)
