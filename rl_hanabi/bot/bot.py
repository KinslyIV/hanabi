import os
import json
import threading
import time
from typing import Any, Callable, Dict, Optional

import requests
from websocket import WebSocketApp


class HanabiConnectionError(Exception):
    pass


class Bot:
    """
    Minimal Hanabi bot base that handles:
    - Login via HTTP POST to `/login` and stores session cookie
    - WebSocket connect to `/ws` with Cookie header
    - Receiving messages in the format: `COMMAND {json}`
    - Sending commands in the format: `COMMAND {json}` with a small send queue
    - Automatic reconnection on connection loss

    Extend this class and override `handle_msg(command, data)` to implement logic.
    """

    def __init__(
        self,
        hostname: Optional[str] = None,
        port: Optional[int] = None,
        ssl_enabled: Optional[bool] = None,
        username_env: str = "HANABI_USERNAME",
        password_env: str = "HANABI_PASSWORD",
        send_interval_ms: int = 500,
        reconnect_delay_s: float = 2.0,
        max_reconnect_attempts: int = 5,
    ) -> None:
        self.hostname = hostname or os.environ.get("HANABI_HOSTNAME", "hanab.live")
        self.ssl_enabled = (
            True
            if os.environ.get("SSL_ENABLED") in (None, "true", "True", "1")
            else False if os.environ.get("SSL_ENABLED") in ("false", "False", "0") else (ssl_enabled if ssl_enabled is not None else True)
        )
        default_port = 443 if self.ssl_enabled else 80
        self.port = int(os.environ.get("HANABI_PORT", str(port or default_port)))
        self.username_env = username_env
        self.password_env = password_env

        self.cookie: Optional[str] = None
        self.ws_app: Optional[WebSocketApp] = None
        self.ws_thread: Optional[threading.Thread] = None

        self.table_id: Optional[int] = None
        self.self_info: Optional[Dict[str, Any]] = None

        # Send queue
        self._queue: list[str] = []
        self._queue_lock = threading.Lock()
        self._queue_timer_ms = send_interval_ms
        self._queue_running = False

        # Reconnection settings
        self._reconnect_delay_s = reconnect_delay_s
        self._max_reconnect_attempts = max_reconnect_attempts
        self._reconnect_attempts = 0
        self._should_reconnect = True
        self._reconnect_lock = threading.Lock()
        self._index_suffix = ""  # Store for reconnection
        self._is_connected = False

        # Callbacks (optional)
        self.on_open: Optional[Callable[[], None]] = None
        self.on_close: Optional[Callable[[int, str], None]] = None
        self.on_error: Optional[Callable[[Exception], None]] = None
        self.on_reconnect: Optional[Callable[[int], None]] = None  # Called with attempt number
        self.on_ping: Optional[Callable[[bytes], None]] = None  # Called when ping received

    # ---- Public API ----
    def connect(self, index_suffix: str = "") -> None:
        """Login and establish websocket connection."""
        self._index_suffix = index_suffix  # Store for reconnection
        self._should_reconnect = True
        self._reconnect_attempts = 0
        self._connect_internal(index_suffix)

    def _connect_internal(self, index_suffix: str = "") -> None:
        """Internal connection logic, used for initial connect and reconnect."""
        self.cookie = self._login(index_suffix)
        ws_url = self._ws_url()

        headers = {"Cookie": self.cookie} if self.cookie else {}
        self.ws_app = WebSocketApp(
            ws_url,
            header=[f"Cookie: {headers['Cookie']}"] if headers else None,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
            on_ping=self._on_ping,
        )

        # Run websocket in background thread
        self.ws_thread = threading.Thread(target=self.ws_app.run_forever, daemon=True)
        self.ws_thread.start()

        # Start send queue pump (only if not already running)
        if not self._queue_running:
            self._start_queue_pump()

    def reconnect(self) -> bool:
        """Attempt to reconnect to the server. Returns True if successful."""
        with self._reconnect_lock:
            if not self._should_reconnect:
                return False
            
            if self._reconnect_attempts >= self._max_reconnect_attempts:
                if self.on_error:
                    self.on_error(HanabiConnectionError(
                        f"Max reconnect attempts ({self._max_reconnect_attempts}) reached"
                    ))
                return False

            self._reconnect_attempts += 1
            attempt = self._reconnect_attempts

        if self.on_reconnect:
            self.on_reconnect(attempt)

        try:
            # Close existing connection if any
            if self.ws_app:
                try:
                    self.ws_app.close()
                except Exception:
                    pass

            # Wait before reconnecting
            time.sleep(self._reconnect_delay_s)

            # Attempt to reconnect
            self._connect_internal(self._index_suffix)
            return True

        except Exception as e:
            if self.on_error:
                self.on_error(e)
            # Schedule another reconnect attempt
            threading.Thread(target=self.reconnect, daemon=True).start()
            return False

    def close(self) -> None:
        self._should_reconnect = False  # Prevent reconnection on intentional close
        self._is_connected = False
        if self.ws_app:
            try:
                self.ws_app.close()
            except Exception:
                pass
        self._stop_queue_pump()

    def send_cmd(self, command: str, arg: Dict[str, Any]) -> None:
        """Queue a command to send: 'command {json}'."""
        payload = f"{command} {json.dumps(arg, separators=(',', ':'))}"
        with self._queue_lock:
            self._queue.append(payload)

    # To be overridden by subclasses
    def handle_msg(self, command: str, data: Dict[str, Any]) -> None:  # pragma: no cover
        pass

    # ---- Internal: HTTP and WS ----
    def _login(self, index_suffix: str) -> str:
        u_field = f"{self.username_env}{index_suffix}"
        p_field = f"{self.password_env}{index_suffix}"
        username = os.environ.get(u_field)
        password = os.environ.get(p_field)
        if not username or not password:
            raise HanabiConnectionError(f"Missing {u_field} and/or {p_field} environment variables.")

        data = {
            "username": username,
            "password": password,
            "version": "bot",
        }
        scheme = "https" if self.ssl_enabled else "http"
        url = f"{scheme}://{self.hostname}:{self.port}/login" if self._custom_port() else f"{scheme}://{self.hostname}/login"
        try:
            resp = requests.post(url, data=data, timeout=10)
        except requests.RequestException as e:
            raise HanabiConnectionError(f"Login request error: {e}")

        if resp.status_code != 200:
            raise HanabiConnectionError(f"Login failed: HTTP {resp.status_code}")

        cookie = resp.headers.get("Set-Cookie")
        if not cookie:
            raise HanabiConnectionError("Failed to parse cookie from auth headers.")

        return cookie.split(",")[0]  # take first cookie only

    def _ws_url(self) -> str:
        protocol = "wss" if self.ssl_enabled else "ws"
        if self._custom_port():
            return f"{protocol}://{self.hostname}:{self.port}/ws"
        return f"{protocol}://{self.hostname}/ws"

    def _custom_port(self) -> bool:
        return self.port not in (80, 443)

    # ---- WS callbacks ----
    def _on_open(self, ws: WebSocketApp) -> None:
        self._is_connected = True
        self._reconnect_attempts = 0  # Reset on successful connection
        if self.on_open:
            self.on_open()

    def _on_error(self, ws: WebSocketApp, error: Exception) -> None:
        self._is_connected = False
        if self.on_error:
            self.on_error(error)
        # Trigger reconnection on error
        if self._should_reconnect:
            threading.Thread(target=self.reconnect, daemon=True).start()

    def _on_close(self, ws: WebSocketApp, status_code: int, msg: str) -> None:
        self._is_connected = False
        if self.on_close:
            self.on_close(status_code, msg)
        # Trigger reconnection on unexpected close
        if self._should_reconnect:
            threading.Thread(target=self.reconnect, daemon=True).start()

    def _on_ping(self, ws: WebSocketApp, data: bytes) -> None:
        if self.on_ping:
            self.on_ping(data)

    def _on_message(self, ws: WebSocketApp, message: str) -> None:
        # Messages are: "command {json}"
        try:
            ind = message.find(" ")
            command = message[:ind]
            arg_str = message[ind + 1:]
            data = json.loads(arg_str)
        except Exception as e:
            # malformed message
            if self.on_error:
                self.on_error(e)
            return
        self.handle_msg(command, data)

    # ---- Send queue pump ----
    def _start_queue_pump(self) -> None:
        self._queue_running = True
        threading.Thread(target=self._queue_loop, daemon=True).start()

    def _stop_queue_pump(self) -> None:
        self._queue_running = False

    def _queue_loop(self) -> None:
        while self._queue_running:
            payload: Optional[str] = None
            with self._queue_lock:
                if self._queue:
                    payload = self._queue.pop(0)
            if payload and self.ws_app:
                try:
                    self.ws_app.send(payload)
                except Exception as e:
                    if self.on_error:
                        self.on_error(e)
                    # Put the message back in the queue for retry after reconnect
                    with self._queue_lock:
                        self._queue.insert(0, payload)
                    # Trigger reconnection
                    if self._should_reconnect and not self._is_connected:
                        threading.Thread(target=self.reconnect, daemon=True).start()
                    # Wait longer before retrying to avoid spam
                    time.sleep(self._reconnect_delay_s)
                    continue
            time.sleep(self._queue_timer_ms / 1000.0)

    # ---- Convenience helpers ----
    def send_pm(self, recipient: str, msg: str) -> None:
        self.send_cmd("chatPM", {"msg": msg, "recipient": recipient, "room": "lobby"})

    def send_chat(self, msg: str) -> None:
        if self.table_id is None:
            return
        self.send_cmd("chat", {"msg": msg, "room": f"table{self.table_id}"})

    def join_table(self, table_id: int, password: Optional[str] = None) -> None:
        arg: Dict[str, Any] = {"tableID": table_id}
        if password:
            arg["password"] = password
        self.send_cmd("tableJoin", arg)

    def leave_room(self) -> None:
        if self.table_id is None:
            return
        # Choose command based on whether game started; callers should set table_id correctly
        self.send_cmd("tableLeave", {"tableID": self.table_id})
        self.table_id = None

