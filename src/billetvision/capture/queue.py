"""Bounded drop-oldest frame queue to maintain constant low latency."""
import queue
from typing import Any, Optional

class DropOldestQueue:
    """A bounded FIFO queue that drops the oldest element when full."""

    def __init__(self, maxsize: int = 5):
        self.queue: queue.Queue = queue.Queue(maxsize=maxsize)
        self.maxsize = maxsize
        self.dropped_frames = 0

    def put(self, item: Any) -> None:
        """Put item, dropping oldest item if queue is full."""
        if self.queue.full():
            try:
                self.queue.get_nowait()
                self.dropped_frames += 1
            except queue.Empty:
                pass
        try:
            self.queue.put_nowait(item)
        except queue.Full:
            pass

    def get(self, timeout: Optional[float] = None) -> Any:
        """Get next item from queue."""
        return self.queue.get(timeout=timeout)

    def empty(self) -> bool:
        """Check if queue is empty."""
        return self.queue.empty()
