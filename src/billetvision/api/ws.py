"""WebSocket connection manager for real-time events and telemetry."""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages active WebSocket connections and fan-out broadcasts."""

    def __init__(self) -> None:
        self._connections: List[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._connections.append(websocket)
        logger.debug("WS connected; total=%d", len(self._connections))

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            try:
                self._connections.remove(websocket)
            except ValueError:
                pass
        logger.debug("WS disconnected; total=%d", len(self._connections))

    async def broadcast(self, message: Dict[str, Any]) -> None:
        """Send ``message`` as JSON to every connected client.

        Dead connections are silently dropped.
        """
        payload = json.dumps(message, default=str)
        dead: List[WebSocket] = []
        async with self._lock:
            targets = list(self._connections)
        for ws in targets:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        if dead:
            async with self._lock:
                for ws in dead:
                    try:
                        self._connections.remove(ws)
                    except ValueError:
                        pass


# Module-level singleton shared between pipeline and the WS endpoint
ws_manager = ConnectionManager()
