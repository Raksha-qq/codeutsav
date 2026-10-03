"""FrameSource abstraction for webcam, video file, and image folder sources."""
from abc import ABC, abstractmethod
import logging
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


class FrameSourceError(Exception):
    """Base exception for frame source errors."""
    pass


class FrameReadError(FrameSourceError):
    """Exception raised when reading a frame fails or source reaches EOF."""
    pass


def natural_sort_key(path: Path) -> List[Union[int, str]]:
    """Sort key for natural alphanumeric ordering (e.g. frame_2 before frame_10)."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", path.name)]


class FrameSource(ABC):
    """Abstract base class for frame sources."""

    @abstractmethod
    def open(self) -> None:
        """Open the frame source."""
        pass

    @abstractmethod
    def read(self) -> np.ndarray:
        """Read the next frame.
        
        Returns:
            np.ndarray: Decoded video frame.
            
        Raises:
            FrameReadError: When reading fails or EOF is reached without looping.
            FrameSourceError: When source is not open or misconfigured.
        """
        pass

    @abstractmethod
    def release(self) -> None:
        """Release underlying resources."""
        pass

    @property
    @abstractmethod
    def fps(self) -> float:
        """FPS of the source."""
        pass

    @property
    @abstractmethod
    def is_live(self) -> bool:
        """Whether the source is live (e.g. webcam) or canned (video/images)."""
        pass

    def get_fps(self) -> float:
        """Compatibility helper returning source FPS."""
        return self.fps

    def __enter__(self) -> "FrameSource":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()


class WebcamSource(FrameSource):
    """Frame source streaming from a connected webcam."""

    def __init__(self, index: int = 0, fps: Optional[float] = None) -> None:
        self.index = int(index)
        self._custom_fps = float(fps) if fps is not None else None
        self.cap: Optional[cv2.VideoCapture] = None

    def open(self) -> None:
        if self.cap is not None and self.cap.isOpened():
            return
        self.cap = cv2.VideoCapture(self.index)
        if not self.cap.isOpened():
            self.cap = None
            raise FrameSourceError(f"Cannot open webcam at index {self.index}")
        logger.info("Opened webcam index %d", self.index)

    def read(self) -> np.ndarray:
        if self.cap is None or not self.cap.isOpened():
            raise FrameSourceError(f"Webcam at index {self.index} is not open. Call open() first.")
        ret, frame = self.cap.read()
        if not ret or frame is None:
            raise FrameReadError(f"Failed to read frame from webcam {self.index}")
        return frame

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            logger.info("Released webcam index %d", self.index)

    @property
    def fps(self) -> float:
        if self._custom_fps is not None and self._custom_fps > 0:
            return self._custom_fps
        if self.cap is not None and self.cap.isOpened():
            val = self.cap.get(cv2.CAP_PROP_FPS)
            if val and val > 0:
                return float(val)
        return 30.0

    @property
    def is_live(self) -> bool:
        return True


class VideoFileSource(FrameSource):
    """Frame source reading from a video file with optional looping."""

    def __init__(self, path: Union[str, Path], loop: bool = False) -> None:
        self.path = Path(path)
        self.loop = bool(loop)
        self.cap: Optional[cv2.VideoCapture] = None
        self._fps: Optional[float] = None

    def open(self) -> None:
        if not self.path.exists():
            raise FrameSourceError(f"Video file does not exist: {self.path}")
        if self.cap is not None and self.cap.isOpened():
            return
        self.cap = cv2.VideoCapture(str(self.path))
        if not self.cap.isOpened():
            self.cap = None
            raise FrameSourceError(f"Failed to open video file: {self.path}")
        val = self.cap.get(cv2.CAP_PROP_FPS)
        self._fps = float(val) if val and val > 0 else 25.0
        logger.info("Opened video file %s (FPS: %.2f, loop: %s)", self.path, self._fps, self.loop)

    def read(self) -> np.ndarray:
        if self.cap is None or not self.cap.isOpened():
            raise FrameSourceError(f"Video file {self.path} is not open. Call open() first.")
        ret, frame = self.cap.read()
        if not ret or frame is None:
            if self.loop:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    raise FrameReadError(f"Failed to read looped frame from video: {self.path}")
            else:
                raise FrameReadError(f"Reached end of video file or read failed: {self.path}")
        return frame

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            logger.info("Released video file %s", self.path)

    @property
    def fps(self) -> float:
        if self._fps is not None and self._fps > 0:
            return self._fps
        if self.cap is not None and self.cap.isOpened():
            val = self.cap.get(cv2.CAP_PROP_FPS)
            if val and val > 0:
                return float(val)
        return 25.0

    @property
    def is_live(self) -> bool:
        return False


class ImageFolderSource(FrameSource):
    """Frame source reading images sequentially from a folder with optional delay."""

    def __init__(
        self,
        dir: Union[str, Path],
        sorted: bool = True,
        delay: float = 0.0,
        loop: bool = False,
    ) -> None:
        self.dir = Path(dir)
        self.sort_files = bool(sorted)
        self.delay = float(delay)
        self.loop = bool(loop)
        self._files: List[Path] = []
        self._index: int = 0
        self._is_open: bool = False

    def open(self) -> None:
        if not self.dir.is_dir():
            raise FrameSourceError(f"Directory not found: {self.dir}")
        found_files = [
            p for p in self.dir.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ]
        if not found_files:
            raise FrameSourceError(f"No supported images found in directory: {self.dir}")
        if self.sort_files:
            self._files = sorted(found_files, key=natural_sort_key)
        else:
            self._files = list(found_files)
        self._index = 0
        self._is_open = True
        logger.info("Opened image folder %s (%d images, delay: %.3fs)", self.dir, len(self._files), self.delay)

    def read(self) -> np.ndarray:
        if not self._is_open:
            raise FrameSourceError(f"Image folder {self.dir} is not open. Call open() first.")
        if self._index >= len(self._files):
            if self.loop:
                self._index = 0
            else:
                raise FrameReadError(f"Reached end of image sequence in folder: {self.dir}")

        if self.delay > 0:
            time.sleep(self.delay)

        img_path = self._files[self._index]
        self._index += 1
        frame = cv2.imread(str(img_path))
        if frame is None:
            raise FrameReadError(f"Failed to read/decode image: {img_path}")
        return frame

    def release(self) -> None:
        self._is_open = False
        self._files = []
        self._index = 0
        logger.info("Released image folder %s", self.dir)

    @property
    def fps(self) -> float:
        if self.delay > 0:
            return 1.0 / self.delay
        return 15.0

    @property
    def is_live(self) -> bool:
        return False


def build_source(config: Union[Dict[str, Any], str, Path, int]) -> FrameSource:
    """Factory creating an appropriate FrameSource based on configuration.
    
    Args:
        config: Can be an int (webcam index), a string/Path (file or folder),
                or a dictionary configuring the capture source.
                
    Returns:
        FrameSource: Configured instance of WebcamSource, VideoFileSource, or ImageFolderSource.
    """
    if isinstance(config, int):
        return WebcamSource(index=config)

    if isinstance(config, (str, Path)):
        s = str(config).strip()
        if s.isdigit():
            return WebcamSource(index=int(s))
        p = Path(s)
        if p.is_dir():
            return ImageFolderSource(dir=p)
        return VideoFileSource(path=p)

    if isinstance(config, dict):
        cfg = config.get("capture", config) if "capture" in config and isinstance(config.get("capture"), dict) else config
        source_type = str(cfg.get("type", "")).lower()
        loop = bool(cfg.get("loop", False))
        delay = float(cfg.get("delay", 0.0))
        sorted_flag = bool(cfg.get("sorted", cfg.get("sort", True)))

        if source_type in ("webcam", "camera"):
            idx = int(cfg.get("index", cfg.get("source", 0)))
            return WebcamSource(index=idx, fps=cfg.get("fps"))
        elif source_type in ("video", "videofile", "file"):
            path = cfg.get("path", cfg.get("source"))
            if not path:
                raise ValueError("Video source configuration requires 'path' or 'source'")
            return VideoFileSource(path=path, loop=loop)
        elif source_type in ("image_folder", "folder", "images", "image"):
            folder = cfg.get("dir", cfg.get("path", cfg.get("source")))
            if not folder:
                raise ValueError("Image folder source configuration requires 'dir', 'path', or 'source'")
            return ImageFolderSource(dir=folder, sorted=sorted_flag, delay=delay, loop=loop)

        # Infer from 'source' or 'path' if type was not explicitly passed
        raw_source = cfg.get("source", cfg.get("path", cfg.get("index", cfg.get("dir"))))
        if raw_source is None:
            raise ValueError(f"Missing source specification in config: {config}")

        if isinstance(raw_source, int) or (isinstance(raw_source, str) and raw_source.isdigit()):
            return WebcamSource(index=int(raw_source), fps=cfg.get("fps"))

        p = Path(str(raw_source))
        if p.is_dir():
            return ImageFolderSource(dir=p, sorted=sorted_flag, delay=delay, loop=loop)
        return VideoFileSource(path=p, loop=loop)

    raise ValueError(f"Unsupported config type for build_source: {type(config)}")
