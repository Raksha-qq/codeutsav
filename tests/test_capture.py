"""Tests for FrameSource, DropOldestQueue, CaptureProducer, and FPSCounter."""
from pathlib import Path
import time
import cv2
import numpy as np
import pytest

from billetvision.capture import (
    CaptureProducer,
    DropOldestQueue,
    FPSCounter,
    FrameReadError,
    FrameSourceError,
    ImageFolderSource,
    VideoFileSource,
    WebcamSource,
    build_source,
)


@pytest.fixture
def synthetic_image_folder(tmp_path: Path) -> Path:
    """Generate a folder containing 5 synthetic indexed images."""
    folder = tmp_path / "images"
    folder.mkdir()
    for i in range(5):
        # Create 64x64 image where every pixel has value (i + 1) * 40
        val = (i + 1) * 40
        img = np.full((64, 64, 3), val, dtype=np.uint8)
        img_path = folder / f"frame_{i:02d}.png"
        cv2.imwrite(str(img_path), img)
    return folder


@pytest.fixture
def synthetic_video_file(tmp_path: Path) -> Path:
    """Generate a tiny 5-frame synthetic video using OpenCV."""
    video_path = tmp_path / "synthetic.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, 20.0, (64, 64))
    assert out.isOpened(), "Failed to create synthetic video writer"
    for i in range(5):
        val = (i + 1) * 40
        frame = np.full((64, 64, 3), val, dtype=np.uint8)
        out.write(frame)
    out.release()
    return video_path


def test_image_folder_source_order_and_eof(synthetic_image_folder: Path):
    source = ImageFolderSource(dir=synthetic_image_folder, sorted=True, delay=0.0, loop=False)
    assert not source.is_live
    assert source.fps > 0

    source.open()
    read_values = []
    for _ in range(5):
        frame = source.read()
        assert frame is not None
        assert frame.shape == (64, 64, 3)
        read_values.append(int(np.mean(frame)))

    # Verify frame order preserved
    expected_values = [(i + 1) * 40 for i in range(5)]
    assert read_values == expected_values

    # Next read must raise FrameReadError (EOF)
    with pytest.raises(FrameReadError):
        source.read()

    source.release()


def test_image_folder_source_loop(synthetic_image_folder: Path):
    source = ImageFolderSource(dir=synthetic_image_folder, sorted=True, loop=True)
    source.open()
    # Read 7 frames (5 + 2 wrapped)
    values = [int(np.mean(source.read())) for _ in range(7)]
    source.release()

    expected = [(0 + 1) * 40, (1 + 1) * 40, (2 + 1) * 40, (3 + 1) * 40, (4 + 1) * 40, (0 + 1) * 40, (1 + 1) * 40]
    assert values == expected


def test_video_file_source_order_and_eof(synthetic_video_file: Path):
    source = VideoFileSource(path=synthetic_video_file, loop=False)
    assert not source.is_live

    source.open()
    assert source.fps > 0

    intensities = []
    for _ in range(5):
        frame = source.read()
        assert frame is not None
        assert frame.shape == (64, 64, 3)
        intensities.append(float(np.mean(frame)))

    # Intensities must be monotonically increasing (frame order preserved)
    for i in range(len(intensities) - 1):
        assert intensities[i] < intensities[i + 1]

    # Reaching EOF must raise FrameReadError
    with pytest.raises(FrameReadError):
        source.read()

    source.release()


def test_video_file_source_loop(synthetic_video_file: Path):
    source = VideoFileSource(path=synthetic_video_file, loop=True)
    source.open()
    first_frame = source.read()
    for _ in range(4):
        source.read()

    # 6th read should loop back
    looped_frame = source.read()
    source.release()

    assert np.allclose(first_frame, looped_frame, atol=10.0)


def test_factory_build_source(synthetic_image_folder: Path, synthetic_video_file: Path):
    src_video = build_source(synthetic_video_file)
    assert isinstance(src_video, VideoFileSource)

    src_folder = build_source(synthetic_image_folder)
    assert isinstance(src_folder, ImageFolderSource)

    src_cam = build_source({"type": "webcam", "index": 0})
    assert isinstance(src_cam, WebcamSource)
    assert src_cam.is_live

    src_cfg = build_source({"capture": {"source": str(synthetic_video_file), "loop": True}})
    assert isinstance(src_cfg, VideoFileSource)
    assert src_cfg.loop is True

    with pytest.raises(ValueError):
        build_source(None)  # type: ignore


def test_drop_oldest_queue_behavior():
    q = DropOldestQueue(maxsize=3)
    assert q.empty()

    for i in range(5):
        q.put(i)

    # 5 items put into maxsize=3 -> 2 items dropped
    assert q.dropped_count == 2
    assert q.dropped_frames == 2
    assert q.full()
    assert q.qsize() == 3

    # Items in queue should be 2, 3, 4
    assert q.get() == 2
    assert q.get() == 3
    assert q.get() == 4
    assert q.empty()


def test_producer_slow_consumer_drop_oldest(tmp_path: Path):
    # Create 12 synthetic images
    folder = tmp_path / "producer_images"
    folder.mkdir()
    for i in range(12):
        img = np.full((32, 32, 3), i, dtype=np.uint8)
        cv2.imwrite(str(folder / f"f_{i:02d}.png"), img)

    source = ImageFolderSource(dir=folder, sorted=True, delay=0.005)
    queue = DropOldestQueue(maxsize=3)
    producer = CaptureProducer(source=source, queue=queue)

    producer.start()

    # Simulate slow consumer that consumes fewer frames than produced
    consumed_frames = []
    time.sleep(0.08)  # Let producer generate multiple frames

    while True:
        try:
            frame = queue.get(timeout=0.05)
            consumed_frames.append(int(np.mean(frame)))
            time.sleep(0.02)  # Slow consumer delay
        except Exception:
            break

    producer.stop(timeout=1.0)

    # Ensure queue never blocked the producer and frames were produced
    assert producer.frames_read > 0
    # Over capacity should have led to dropped frames under the slow consumer
    assert queue.dropped_count > 0
    assert len(consumed_frames) > 0


def test_fps_counter_sanity():
    counter = FPSCounter(window_size=10)
    assert counter.fps == 0.0

    # 1 tick still cannot compute interval
    counter.tick()
    assert counter.fps == 0.0

    # Record 6 ticks spaced by ~20ms -> expected ~50 FPS
    for _ in range(5):
        time.sleep(0.02)
        fps = counter.tick()

    measured_fps = counter.fps
    # Allow reasonable operating system timing tolerances (25 to 75 FPS for ~20ms sleeps)
    assert 25.0 <= measured_fps <= 75.0

    counter.reset()
    assert counter.fps == 0.0


def test_read_failure_exception(tmp_path: Path):
    # Non-existent file
    src = VideoFileSource(path=tmp_path / "non_existent.mp4")
    with pytest.raises(FrameSourceError):
        src.open()

    # Empty image folder
    empty_folder = tmp_path / "empty_dir"
    empty_folder.mkdir()
    src_folder = ImageFolderSource(dir=empty_folder)
    with pytest.raises(FrameSourceError):
        src_folder.open()
