"""
GPU Client - Connects to the GPU server with automatic reconnection and resilience.
Used by the coordinator to offload GPU training.
"""

from __future__ import annotations

import asyncio
import logging
import pickle
import struct
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional

import numpy as np

from rl_hanabi.training.distributed.protocol import TrainingRequest, TrainingResponse

logger = logging.getLogger('gpu_client')


def _deserialize_response(data: bytes) -> TrainingResponse:
    """Deserialize a response, handling both Pydantic models and dicts."""
    obj = pickle.loads(data)
    if isinstance(obj, dict):
        return TrainingResponse.model_validate(obj)
    return obj


class ConnectionState(Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    AUTHENTICATING = "authenticating"
    AUTHENTICATED = "authenticated"
    ERROR = "error"


@dataclass
class ConnectionStats:
    """Statistics about the connection."""
    connect_attempts: int = 0
    successful_connects: int = 0
    failed_connects: int = 0
    requests_sent: int = 0
    requests_failed: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    total_latency: float = 0.0
    last_successful_request: Optional[float] = None
    last_error: Optional[str] = None
    last_error_time: Optional[float] = None


@dataclass
class GPUClientConfig:
    """Configuration for GPU client."""
    host: str = "localhost"
    port: int = 5555
    auth_token: Optional[str] = None
    
    # Reconnection settings
    initial_retry_delay: float = 1.0
    max_retry_delay: float = 60.0
    retry_backoff_factor: float = 2.0
    max_connect_attempts: int = 0  # 0 = infinite
    connection_timeout: float = 30.0
    request_timeout: float = 300.0
    
    # Health check
    ping_interval: float = 30.0
    ping_timeout: float = 10.0
    
    # Callbacks
    on_connect: Optional[Callable[[], None]] = None
    on_disconnect: Optional[Callable[[str], None]] = None
    on_reconnecting: Optional[Callable[[int, float], None]] = None


class GPUClient:
    """
    Resilient async client for connecting to the GPU server.
    Automatically reconnects on failure and pauses operations when disconnected.
    """
    
    def __init__(self, config: GPUClientConfig):
        self.config = config
        self.state = ConnectionState.DISCONNECTED
        self.stats = ConnectionStats()
        
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._lock = asyncio.Lock()
        self._shutdown_event = asyncio.Event()
        self._connected_event = asyncio.Event()
        self._reconnect_task: Optional[asyncio.Task] = None
        self._ping_task: Optional[asyncio.Task] = None
        
        self._current_retry_delay = config.initial_retry_delay
    
    @property
    def is_connected(self) -> bool:
        return self.state == ConnectionState.AUTHENTICATED
    
    async def _send(self, data: bytes) -> None:
        """Send data with length prefix."""
        if self._writer is None:
            raise ConnectionError("Not connected")
        
        length = struct.pack('>I', len(data))
        self._writer.write(length + data)
        await self._writer.drain()
        self.stats.bytes_sent += len(data) + 4
    
    async def _recv(self, timeout: float) -> bytes:
        """Receive data with length prefix."""
        if self._reader is None:
            raise ConnectionError("Not connected")
        
        length_bytes = await asyncio.wait_for(
            self._reader.readexactly(4),
            timeout=timeout
        )
        length = struct.unpack('>I', length_bytes)[0]
        
        data = await asyncio.wait_for(
            self._reader.readexactly(length),
            timeout=timeout
        )
        self.stats.bytes_received += len(data) + 4
        return data
    
    async def _do_connect(self) -> bool:
        """Attempt a single connection."""
        self.state = ConnectionState.CONNECTING
        self.stats.connect_attempts += 1
        
        try:
            logger.info(f"Connecting to GPU server at {self.config.host}:{self.config.port}")
            
            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_connection(self.config.host, self.config.port),
                timeout=self.config.connection_timeout
            )
            
            self.state = ConnectionState.CONNECTED
            
            # Authenticate if needed
            if self.config.auth_token:
                self.state = ConnectionState.AUTHENTICATING
                auth_request = TrainingRequest(request_type='auth', payload={'token': self.config.auth_token})
                await self._send(pickle.dumps(auth_request))
                
                response_data = await self._recv(self.config.connection_timeout)
                response = _deserialize_response(response_data)
                
                if not response.success:
                    raise ConnectionError(f"Authentication failed: {response.error}")
                
                logger.info("Authenticated with GPU server")
            
            self.state = ConnectionState.AUTHENTICATED
            self.stats.successful_connects += 1
            self._current_retry_delay = self.config.initial_retry_delay
            self._connected_event.set()
            
            if self.config.on_connect:
                try:
                    self.config.on_connect()
                except Exception as e:
                    logger.warning(f"on_connect callback error: {e}")
            
            logger.info("Connected to GPU server")
            return True
        
        except Exception as e:
            self.stats.failed_connects += 1
            self.stats.last_error = str(e)
            self.stats.last_error_time = time.time()
            self.state = ConnectionState.ERROR
            logger.warning(f"Connection failed: {e}")
            
            await self._cleanup_connection()
            return False
    
    async def _cleanup_connection(self) -> None:
        """Clean up connection resources."""
        self._connected_event.clear()
        
        if self._writer:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception:
                pass
        
        self._reader = None
        self._writer = None
    
    async def _reconnect_loop(self) -> None:
        """Background task that maintains connection."""
        attempt = 0
        
        while not self._shutdown_event.is_set():
            if self.state == ConnectionState.AUTHENTICATED:
                await asyncio.sleep(1)
                continue
            
            attempt += 1
            
            if self.config.max_connect_attempts > 0 and attempt > self.config.max_connect_attempts:
                logger.error("Max connection attempts reached")
                break
            
            if self.config.on_reconnecting:
                try:
                    self.config.on_reconnecting(attempt, self._current_retry_delay)
                except Exception as e:
                    logger.warning(f"on_reconnecting callback error: {e}")
            
            if await self._do_connect():
                attempt = 0
            else:
                # Exponential backoff
                await asyncio.sleep(self._current_retry_delay)
                self._current_retry_delay = min(
                    self._current_retry_delay * self.config.retry_backoff_factor,
                    self.config.max_retry_delay
                )
    
    async def _ping_loop(self) -> None:
        """Background task that periodically pings the server."""
        while not self._shutdown_event.is_set():
            await asyncio.sleep(self.config.ping_interval)
            
            if self.state != ConnectionState.AUTHENTICATED:
                continue
            
            try:
                await self.ping(timeout=self.config.ping_timeout)
            except Exception as e:
                logger.warning(f"Ping failed: {e}")
                await self._handle_disconnect(f"Ping failed: {e}")
    
    async def _handle_disconnect(self, reason: str) -> None:
        """Handle disconnection."""
        if self.state == ConnectionState.DISCONNECTED:
            return
        
        logger.warning(f"Disconnected: {reason}")
        self.state = ConnectionState.DISCONNECTED
        
        await self._cleanup_connection()
        
        if self.config.on_disconnect:
            try:
                self.config.on_disconnect(reason)
            except Exception as e:
                logger.warning(f"on_disconnect callback error: {e}")
    
    async def connect(self) -> None:
        """Start connection and background tasks."""
        self._shutdown_event.clear()
        
        # Start reconnect loop
        self._reconnect_task = asyncio.create_task(self._reconnect_loop())
        
        # Start ping loop
        self._ping_task = asyncio.create_task(self._ping_loop())
        
        # Wait for initial connection
        try:
            await asyncio.wait_for(
                self._connected_event.wait(),
                timeout=self.config.connection_timeout * 2
            )
        except asyncio.TimeoutError:
            logger.warning("Initial connection timeout, will keep trying in background")
    
    async def disconnect(self) -> None:
        """Disconnect and stop background tasks."""
        logger.info("Disconnecting from GPU server...")
        self._shutdown_event.set()
        
        if self._reconnect_task:
            self._reconnect_task.cancel()
            try:
                await self._reconnect_task
            except asyncio.CancelledError:
                pass
        
        if self._ping_task:
            self._ping_task.cancel()
            try:
                await self._ping_task
            except asyncio.CancelledError:
                pass
        
        await self._cleanup_connection()
        self.state = ConnectionState.DISCONNECTED
        logger.info("Disconnected")
    
    async def wait_for_connection(self, timeout: Optional[float] = None) -> bool:
        """Wait until connected."""
        try:
            await asyncio.wait_for(
                self._connected_event.wait(),
                timeout=timeout
            )
            return True
        except asyncio.TimeoutError:
            return False
    
    async def _request(
        self,
        request_type: str,
        payload: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Send a request and wait for response."""
        if timeout is None:
            timeout = self.config.request_timeout
        
        async with self._lock:
            if not self.is_connected:
                # Wait for reconnection
                logger.info("Waiting for connection...")
                connected = await self.wait_for_connection(timeout=timeout)
                if not connected:
                    raise ConnectionError("Not connected and connection timeout")
            
            request = TrainingRequest(request_type=request_type, payload=payload)
            start_time = time.time()
            
            try:
                await self._send(pickle.dumps(request))
                response_data = await self._recv(timeout)
                response = _deserialize_response(response_data)
                
                self.stats.requests_sent += 1
                self.stats.total_latency += time.time() - start_time
                self.stats.last_successful_request = time.time()
                
                if not response.success:
                    raise RuntimeError(f"Request failed: {response.error}")
                
                return response.payload if response.payload else {}
            
            except Exception as e:
                self.stats.requests_failed += 1
                await self._handle_disconnect(str(e))
                raise
    
    async def ping(self, timeout: float = 10.0) -> Dict[str, Any]:
        """Ping the server."""
        return await self._request('ping', timeout=timeout)
    
    async def train_step(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Send a batch for training and get metrics back."""
        result = await self._request('train_step', {'batch': batch})
        return result.get('metrics', {})
    
    async def get_weights(self) -> Dict[str, Any]:
        """Get current model weights."""
        result = await self._request('get_weights')
        return result.get('weights', {})
    
    async def get_state(self) -> Dict[str, Any]:
        """Get complete trainer state."""
        result = await self._request('get_state')
        return result.get('state', {})
    
    async def set_state(self, state: Dict[str, Any]) -> None:
        """Set trainer state."""
        await self._request('set_state', {'state': state})
    
    async def save_checkpoint(self, filename: str) -> str:
        """Save checkpoint on GPU server."""
        result = await self._request('save_checkpoint', {'filename': filename})
        return result.get('filepath', '')
    
    async def load_checkpoint(self, filepath: str) -> None:
        """Load checkpoint on GPU server."""
        await self._request('load_checkpoint', {'filepath': filepath})
    
    async def load_pretrained_model(self, filepath: str) -> Dict[str, Any]:
        """
        Load pretrained model weights on GPU server.
        This loads only the model weights, resetting optimizer/scheduler state.
        Use this to start fresh training with a pretrained model.
        
        If the file is not found, a fresh model will be initialized instead.
        
        Returns:
            Dict with 'loaded' (bool) indicating if weights were loaded,
            and 'message' describing what happened.
        """
        return await self._request('load_pretrained_model', {'filepath': filepath})
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get server statistics."""
        return await self._request('get_stats')
    
    async def stop_training(self, shutdown_server: bool = False) -> None:
        """
        Request to stop all training operations.
        
        Args:
            shutdown_server: If True, also shuts down the GPU server process.
                             If False, just stops training but keeps the server running.
        """
        try:
            await self._request('stop_training', {'shutdown_server': shutdown_server})
        except Exception as e:
            logger.warning(f"Error sending stop_training request: {e}")
        finally:
            # Always disconnect the client after stopping training
            await self.disconnect()
