"""MJPEG video streaming — latest-frame holder + sync generator.

The pipeline writes annotated BGR frames via ``latest_frame.update()``.
The MJPEG generator in ``frame_generator()`` reads and yields them as
JPEG boundaries.  Clients that lag behind always see the current frame,
never a backlog.
"""
from __future__ import annotations

import threading
import time
from typing import Generator, Optional

import cv2
import numpy as np


class LatestFrame:
    """Thread-safe holder for the most-recently rendered frame (as JPEG bytes)."""

    def __init__(self, jpeg_quality: int = 80) -> None:
        self._jpeg: Optional[bytes] = None
        self._cond = threading.Condition(threading.Lock())
        self._quality = jpeg_quality

    def update(self, frame_bgr: np.ndarray) -> None:
        """Encode ``frame_bgr`` to JPEG and notify waiting generators."""
        ok, buf = cv2.imencode(
            ".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, self._quality]
        )
        if not ok:
            return
        with self._cond:
            self._jpeg = bytes(buf)
            self._cond.notify_all()

    def get(self, timeout: float = 1.0) -> Optional[bytes]:
        """Block until a new frame is available (or ``timeout`` elapses).

        Returns the latest JPEG bytes, or None on timeout.
        """
        with self._cond:
            self._cond.wait(timeout=timeout)
            return self._jpeg

    @property
    def has_frame(self) -> bool:
        return self._jpeg is not None


# Module-level singleton shared between pipeline and MJPEG endpoint
latest_frame = LatestFrame()

# Placeholder single-frame PNG used when the pipeline has not started yet
_PLACEHOLDER: Optional[bytes] = None


def _placeholder_jpeg() -> bytes:
    global _PLACEHOLDER
    if _PLACEHOLDER is None:
        img = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.putText(
            img,
            "BilletVision — waiting for pipeline",
            (60, 185),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (180, 180, 180),
            2,
            cv2.LINE_AA,
        )
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        _PLACEHOLDER = bytes(buf)
    return _PLACEHOLDER


def frame_generator() -> Generator[bytes, None, None]:
    """Yield MJPEG boundaries for a StreamingResponse.

    Falls back to the placeholder at ~5 fps when no pipeline frames arrive.
    """
    _BOUNDARY = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
    _END = b"\r\n"

    while True:
        jpeg = latest_frame.get(timeout=0.2)
        if jpeg is None:
            jpeg = _placeholder_jpeg()
            time.sleep(0.2)  # throttle placeholder to ~5 fps
        yield _BOUNDARY + jpeg + _END
