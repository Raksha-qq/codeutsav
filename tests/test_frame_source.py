"""Tests for capture/frame_source.py using synthetic video and images."""
from pathlib import Path
import time
import cv2
import numpy as np
import pytest

from billetvision.capture.frame_source import FrameSource


@pytest.fixture
def synthetic_10_frame_video(tmp_path: Path) -> Path:
    """Generate a synthetic 10-frame MP4 video using OpenCV."""
    video_path = tmp_path / "synthetic_10.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 30.0
    width, height = 64, 64
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
    assert out.isOpened(), "Could not create synthetic video writer"

    for i in range(10):
        # Frame intensity proportional to frame index: 20, 40, ..., 200
        val = int((i + 1) * 20)
        frame = np.full((height, width, 3), val, dtype=np.uint8)
        out.write(frame)

    out.release()
    return video_path


@pytest.fixture
def synthetic_10_frame_folder(tmp_path: Path) -> Path:
    """Generate a folder containing 10 synthetic image files."""
    folder = tmp_path / "synthetic_images"
    folder.mkdir()
    width, height = 64, 64

    for i in range(10):
        val = int((i + 1) * 20)
        img = np.full((height, width, 3), val, dtype=np.uint8)
        cv2.imwrite(str(folder / f"frame_{i:02d}.png"), img)

    return folder


def test_frame_source_video_reading(synthetic_10_frame_video: Path):
    """Test reading all 10 frames from a video file."""
    with FrameSource(source=synthetic_10_frame_video, mode="video", maxsize=4, pace=False) as fs:
        assert fs.maxsize == 4
        assert fs.mode == "video"

        frames_read = 0
        intensities = []
        while True:
            ret, frame = fs.read(timeout=1.0)
            if not ret or frame is None:
                break
            frames_read += 1
            intensities.append(float(np.mean(frame)))

        assert frames_read == 10
        # Intensities should be monotonically increasing (frame order preserved)
        for i in range(len(intensities) - 1):
            assert intensities[i] < intensities[i + 1]


def test_frame_source_drop_oldest_under_slow_consumer(synthetic_10_frame_video: Path):
    """Test that a bounded queue (maxsize=4) drops the oldest frames when consumer is slow."""
    fs = FrameSource(source=synthetic_10_frame_video, mode="video", maxsize=4, pace=False)
    fs.start()

    # Let producer push all 10 frames into the queue without consumer reading
    time.sleep(0.1)

    # 10 frames put into maxsize=4 -> 6 oldest frames should have been dropped
    assert fs.dropped_frames == 6

    # Only 4 newest frames remain
    remaining = []
    while True:
        ret, frame = fs.read(timeout=0.1)
        if not ret or frame is None:
            break
        remaining.append(frame)

    assert len(remaining) == 4
    fs.release()


def test_frame_source_image_folder(synthetic_10_frame_folder: Path):
    """Test image folder mode with 10 synthetic frames."""
    with FrameSource(source=synthetic_10_frame_folder, mode="folder", maxsize=4, pace=False) as fs:
        assert fs.mode == "folder"
        frames = list(fs)
        assert len(frames) == 10
        assert frames[0].shape == (64, 64, 3)


def test_frame_source_webcam_mode_config():
    """Test webcam mode initialization with cv2.VideoCapture(0)."""
    fs = FrameSource(mode="webcam", source=0, maxsize=4)
    assert fs.mode == "webcam"
    assert fs.source == 0
    assert fs.maxsize == 4
    # Clean release without crashing even if physical device is absent
    fs.release()


def test_frame_source_get_frame_convenience(synthetic_10_frame_video: Path):
    """Test get_frame convenience method and EOF."""
    fs = FrameSource(source=synthetic_10_frame_video, pace=False)
    frame = fs.get_frame()
    assert frame is not None
    assert frame.shape == (64, 64, 3)
    fs.release()
