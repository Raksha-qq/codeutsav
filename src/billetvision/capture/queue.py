"""Bounded drop-oldest frame queue and capture producer to maintain low latency."""
import collections
import logging
import queue
import threading
import time
from typing import Any, Optional

from billetvision.capture.sources import FrameReadError, FrameSource

logger = logging.getLogger(__name__)


class DropOldestQueue:
    """A thread-safe bounded FIFO queue that drops the oldest element when full.
    
    This ensures that downstream slow consumers never cause capture latency to accumulate.
    """

    def __init__(self, maxsize: int = 5) -> None:
        if maxsize <= 0:
            raise ValueError("maxsize must be greater than 0")
        self.maxsize = int(maxsize)
        self._deque: collections.deque = collections.deque()
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._dropped_count = 0

    def put(self, item: Any) -> bool:
        """Put item into queue, dropping oldest item if queue is at capacity.
        
        Returns:
            bool: True if an older item was dropped, False otherwise.
        """
        dropped = False
        with self._lock:
            if len(self._deque) >= self.maxsize:
                self._deque.popleft()
                self._dropped_count += 1
                dropped = True
            self._deque.append(item)
            self._not_empty.notify()
        return dropped

    def get(self, timeout: Optional[float] = None) -> Any:
        """Get the next item from the queue.
        
        Blocks until an item is available or timeout expires.
        Raises queue.Empty if timeout occurs.
        """
        with self._not_empty:
            if not self._deque:
                if not self._not_empty.wait(timeout=timeout):
                    raise queue.Empty
                if not self._deque:
                    raise queue.Empty
            return self._deque.popleft()

    def get_nowait(self) -> Any:
        """Get the next item immediately without waiting."""
        return self.get(timeout=0)

    def empty(self) -> bool:
        """Check if queue is empty."""
        with self._lock:
            return len(self._deque) == 0

    def full(self) -> bool:
        """Check if queue is at capacity."""
        with self._lock:
            return len(self._deque) >= self.maxsize

    def qsize(self) -> int:
        """Return the current number of items in the queue."""
        with self._lock:
            return len(self._deque)

    @property
    def dropped_count(self) -> int:
        """Total number of frames dropped due to buffer overflow."""
        with self._lock:
            return self._dropped_count

    @property
    def dropped_frames(self) -> int:
        """Alias for dropped_count."""
        return self.dropped_count

    def clear(self) -> None:
        """Remove all items from the queue."""
        with self._lock:
            self._deque.clear()


class FPSCounter:
    """Thread-safe rolling-window FPS counter."""

    def __init__(self, window_size: int = 30) -> None:
        if window_size < 2:
            raise ValueError("window_size must be at least 2")
        self.window_size = int(window_size)
        self._timestamps: collections.deque = collections.deque()
        self._lock = threading.Lock()

    def tick(self) -> float:
        """Record frame arrival time and return current rolling FPS."""
        now = time.perf_counter()
        with self._lock:
            self._timestamps.append(now)
            if len(self._timestamps) > self.window_size:
                self._timestamps.popleft()
            return self._compute_fps()

    @property
    def fps(self) -> float:
        """Current rolling FPS estimate."""
        with self._lock:
            return self._compute_fps()

    def _compute_fps(self) -> float:
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / elapsed

    def reset(self) -> None:
        """Clear recorded timestamps."""
        with self._lock:
            self._timestamps.clear()


class CaptureProducer(threading.Thread):
    """Background thread reading from a FrameSource into a bounded DropOldestQueue.
    
    Guarantees latency does not grow by discarding older frames when processing lags behind.
    """

    def __init__(
        self,
        source: FrameSource,
        queue: Optional[DropOldestQueue] = None,
        fps_counter: Optional[FPSCounter] = None,
        target_fps: Optional[float] = None,
        name: str = "CaptureProducerThread",
        daemon: bool = True,
    ) -> None:
        super().__init__(name=name, daemon=daemon)
        self.source = source
        self.queue = queue if queue is not None else DropOldestQueue(maxsize=5)
        self.fps_counter = fps_counter if fps_counter is not None else FPSCounter()
        self.target_fps = target_fps
        self._stop_event = threading.Event()
        self.last_error: Optional[Exception] = None
        self.frames_read = 0

    def run(self) -> None:
        try:
            self.source.open()
        except Exception as e:
            self.last_error = e
            logger.error("Capture producer failed to open source: %s", e)
            return

        # Live sources (like webcams) deliver at their hardware rate.
        # Canned sources (video/image folders) can be paced according to target_fps or source fps.
        effective_fps = self.target_fps if (self.target_fps and self.target_fps > 0) else (
            self.source.fps if (not self.source.is_live and self.source.fps > 0) else None
        )
        interval = (1.0 / effective_fps) if (effective_fps and effective_fps > 0) else 0.0

        while not self._stop_event.is_set():
            t0 = time.perf_counter()
            try:
                frame = self.source.read()
                self.queue.put(frame)
                self.frames_read += 1
                self.fps_counter.tick()
            except FrameReadError as e:
                self.last_error = e
                logger.info("Frame source reading stopped: %s", e)
                break
            except Exception as e:
                self.last_error = e
                logger.error("Unexpected error in capture producer: %s", e)
                break

            if interval > 0 and not self._stop_event.is_set():
                elapsed = time.perf_counter() - t0
                sleep_time = interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

    def stop(self, timeout: Optional[float] = None) -> None:
        """Signal thread to stop, wait for completion, and release source resources."""
        self._stop_event.set()
        if self.is_alive():
            self.join(timeout=timeout)
        try:
            self.source.release()
        except Exception as e:
            logger.warning("Error releasing source: %s", e)


# Backward-compatible and semantic alias
ProducerThread = CaptureProducer
