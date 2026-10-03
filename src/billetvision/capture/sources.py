"""FrameSource abstraction for webcam, video file, and image folder sources."""
from abc import ABC, abstractmethod
from typing import Generator, Optional, Tuple, Any

class FrameSource(ABC):
    """Abstract base class for frame sources."""

    @abstractmethod
    def open(self) -> None:
        """Open the frame source."""
        pass

    @abstractmethod
    def read(self) -> Tuple[bool, Optional[Any]]:
        """Read the next frame. Returns (success, frame)."""
        pass

    @abstractmethod
    def release(self) -> None:
        """Release underlying resources."""
        pass

    @abstractmethod
    def get_fps(self) -> float:
        """Return FPS of the source."""
        pass
