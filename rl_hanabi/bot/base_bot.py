from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional

from rl_hanabi.bot.bot import Bot
from rl_hanabi.game.game_types import ACTION
from rl_hanabi.game.state import GameState


class BaseBot(Bot):
    def __init__(self, enable_hle: bool = False) -> None:
        super().__init__()
        self.logger = logging.getLogger("rl_hanabi.base_bot")
        self.state: Optional[GameState] = None
        self.tables: Dict[str, Dict[str, Any]] = {}
        self.last_sender: Optional[str] = None
        self.game_started: bool = False
        self.enable_hle = enable_hle
        self.catchup = False

    # --- Core message handler ---
    def handle_msg(self, command: str, data: Dict[str, Any]) -> None:
        self.logger.debug("Received command=%s data=%s", command, data)
        
        if command == "welcome":
            self.self_info = data
        elif command == "tableList":
            for t in data:
                self.tables[t["id"]] = t # type: ignore
        elif command == "table":
            self.tables[data["id"]] = data
            self.logger.info(f"Table data received: {data}")
        elif command == "tableGone":
            tid = data.get("tableID")
            if tid in self.tables:
                del self.tables[tid]
        elif command == "joined":
            self.table_id = data.get("tableID")
            self.game_started = False
        elif command == "left":
            self.table_id = None
            self.game_started = False
            self.state = None
        elif command == "tableStart":
            self.logger.info(f"Table start command received: {data}")
            self.table_id = data.get("tableID")
            self.game_started = True
            # Request initial info (mirrors JS bot)
            self.send_cmd("getGameInfo1", {"tableID": self.table_id})
        elif command == "init":
            # Initialize minimal state and ask for more info
            self.logger.info(f"Init command received: {data}")
            self.table_id = data.get("tableID")
            player_names: List[str] = data.get("playerNames", [])
            our_idx: int = data.get("ourPlayerIndex") # type: ignore
            options: Dict[str, Any] = data.get("options", {})
            
            self._init_game_state(player_names, our_idx, options)
            
            self.send_cmd("getGameInfo2", {"tableID": self.table_id})
        elif command == "gameActionList":
            # Catch-up: apply all historical actions into our state/HLE shadow.
            self.catchup = True
            lst: List[Dict[str, Any]] = data.get("list", [])
            for a in lst:
                self._apply_action(a)
                self.logger.info(f"Applied action from history: {a}")
            self.catchup = False
            
            # Notify loaded like JS bot
            self.logger.info(f"Game loaded for tableID: {self.table_id}")
            if self.table_id is not None:
                self.send_cmd("loaded", {"tableID": self.table_id})
            
            # Check if it's our turn immediately after loading history
            if self.state:
                self.logger.info(f"History loaded. Turn: {self.state.current_player_index} Our: {self.state.our_index} Table: {self.table_id}")
                if self.state.current_player_index == self.state.our_index and self.table_id is not None:
                    self.logger.info("It's our turn after history load. Making move.")
                    self.make_move()
                else:
                    self.logger.info("Not our turn after history load.")

        elif command == "gameAction":
            self.logger.info(f"Game action received: {data}")
            self._apply_action(data.get("action", {}))
        elif command == "warning":
            warn = data.get("warning")
            self.logger.warning(f"Server warning: {warn}")
            if self.last_sender:
                self.send_pm(self.last_sender, str(warn))
                self.last_sender = None
        elif command == "chat":
            self._handle_chat(data)

    def _init_game_state(self, player_names: List[str], our_idx: int, options: Dict[str, Any]) -> None:
        self.state = GameState(player_names, our_idx, len(player_names), options, enable_hle=self.enable_hle)
        self.logger.info(f"Game initialized. Options: {options}")

    # --- Apply game events ---
    def _apply_action(self, action: Dict[str, Any]) -> None:
        if not self.state:
            return
        t = action.get("type")
        self.logger.debug(f"Apply action type={t} payload={action}")
        if t == "status":
            self.state.apply_status(
                action.get("clues", self.state.clue_tokens),
                action.get("lives"),
                action.get("score")
            )
        elif t == "turn":
            self.state.apply_turn(action.get("num", self.state.turn_count), action.get("currentPlayerIndex", self.state.current_player_index))
            # If it's our turn, make a move
            if not self.catchup and self.state.current_player_index == self.state.our_index and self.table_id is not None:
                self.make_move()
        elif t == "draw":
            # Draw action has card info at top level (order, suitIndex, rank)
            self.state.apply_draw(action.get("playerIndex"), action) # type: ignore
        elif t == "play":
            self.state.apply_play(action.get("order"), action.get("playerIndex"), action) # type: ignore
        elif t == "discard":
            self.state.apply_discard(action.get("order"), action.get("playerIndex"), action) # type: ignore
        elif t == "clue":
            # Mirror clues into the HLE shadow state when present.
            clue = action.get("clue", {})
            clue_type = clue.get("type")
            value = clue.get("value")
            giver = action.get("giver")
            target = action.get("target")
            if clue_type is not None and value is not None and giver is not None and target is not None:
                self.state.apply_clue(giver, target, clue_type, value, action)
        elif t == "gameOver":
            self.game_started = False
            self.logger.info("Game Over")

    def make_move(self) -> None:
        """Override this method to implement bot logic."""
        pass

    # --- Chat commands ---
    def _handle_chat(self, data: Dict[str, Any]) -> None:
        msg: str = data.get("msg", "")
        who: str = data.get("who", "")
        room: str = data.get("room", "")
        recipient: str = data.get("recipient", "")

        self.logger.info(f"Chat message from={who} room={room} recipient={recipient} msg={msg}")

        within_room = recipient == "" and room.startswith("table")
        if within_room:
            if msg.startswith("/leaveall"):
                self.leave_room()
            # ignore '/setall' for random bot
            return

        # Only respond to PMs directed to us
        if not self.self_info or recipient != self.self_info.get("username"):
            return

        self.last_sender = who

        if msg.startswith("/join"):
            table = self._find_table_for_user(who)
            if table is None:
                self.send_pm(who, "Could not join; you are not in a room.")
                return
            if not table.get("passwordProtected", False):
                self.send_cmd("tableJoin", {"tableID": table["id"]})
            else:
                parts = msg.split(" ", 1)
                if len(parts) == 1:
                    self.send_pm(who, "Room is password protected; please provide a password.")
                else:
                    self.send_cmd("tableJoin", {"tableID": table["id"], "password": parts[1]})
        elif msg.startswith("/rejoin"):
            if self.table_id is not None:
                self.send_pm(who, "Already in a game.")
                return
            table = self._find_table_with_player(self.self_info.get("username", ""))
            if table is None:
                self.send_pm(who, "Not a player in any open room.")
            else:
                self.send_cmd("tableReattend", {"tableID": table["id"]})
        elif msg.startswith("/leave"):
            if self.table_id is None:
                self.send_pm(who, "Not currently in a room.")
                return
            self.leave_room()
        elif msg.startswith("/create"):
            parts = msg.split(" ")
            name = parts[1] if len(parts) > 1 else "rl-bot"
            max_players = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 2
            password = parts[3] if len(parts) > 3 else None
            arg: Dict[str, Any] = {"name": name, "maxPlayers": max_players}
            if password:
                arg["password"] = password
            self.send_cmd("tableCreate", arg)
        elif msg.startswith("/start"):
            if self.table_id is not None:
                self.send_cmd("tableStart", {"tableID": self.table_id})
        elif msg.startswith("/restart"):
            if self.table_id is not None:
                self.send_cmd("tableRestart", {"tableID": self.table_id, "hidePregame": True})
        elif msg.startswith("/remake"):
            if self.table_id is not None:
                self.send_cmd("tableRestart", {"tableID": self.table_id, "hidePregame": False})
        elif msg.startswith("/version"):
            self.send_pm(who, "rl-base-bot v0.1")
        elif msg.startswith("/settings"):
            self.send_pm(who, "This bot ignores convention settings.")
        else:
            self.send_pm(who, "Unrecognized command.")

    def _find_table_for_user(self, username: str) -> Optional[Dict[str, Any]]:
        # Prefer a table where the user is a player or spectator
        candidates = []
        for t in self.tables.values():
            if username in t.get("players", []) or any(s.get("name") == username for s in t.get("spectators", [])):
                candidates.append(t)
        if not candidates:
            return None
        # Return the one with max id (latest)
        return max(candidates, key=lambda x: x.get("id", -1))

    def _find_table_with_player(self, username: str) -> Optional[Dict[str, Any]]:
        for t in self.tables.values():
            if username in t.get("players", []):
                return t
        return None

    def leave_room(self) -> None:
        if self.table_id is not None:
            self.send_cmd("tableLeave", {"tableID": self.table_id})
            self.table_id = None
            self.state = None
            self.game_started = False

    def send_pm(self, recipient: str, msg: str) -> None:
        self.send_cmd("chatPM", {"recipient": recipient, "msg": msg})
