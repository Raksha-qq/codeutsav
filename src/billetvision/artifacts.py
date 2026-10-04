"""Per-billet artifacts: annotated snapshot, ID crop, frame thumbnails, detail JSON.

Everything for billet ``seq`` lives in the snapshot directory under
``{seq:06d}*`` names so the dashboard drill-down (FR-21) can show frames,
measurements and the decision reasoning without touching the pipeline.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def artifact_name(seq: int, suffix: str) -> str:
    """File name for a billet artifact, e.g. ``000012.jpg`` / ``000012_id.png``."""
    return f"{seq:06d}{suffix}"


def _write_image(path: Path, image: np.ndarray, params: Optional[list] = None) -> bool:
    try:
        return bool(cv2.imwrite(str(path), image, params or []))
    except Exception as exc:  # disk full / bad path must not crash the pipeline
        logger.warning("Could not write %s: %s", path, exc)
        return False


def save_snapshot(snapshot_dir: Path, seq: int, image: np.ndarray) -> Path:
    """Save the annotated billet card; returns its path (even if the write failed)."""
    path = snapshot_dir / artifact_name(seq, ".jpg")
    _write_image(path, image, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return path


def save_id_crop(snapshot_dir: Path, seq: int, crop: Optional[np.ndarray]) -> Optional[Path]:
    """Save the ID crop used for the review queue (PNG, lossless)."""
    if crop is None or crop.size == 0:
        return None
    path = snapshot_dir / artifact_name(seq, "_id.png")
    return path if _write_image(path, crop) else None


def save_frames(snapshot_dir: Path, seq: int, frames: List[np.ndarray]) -> List[str]:
    """Save the sharpest frames as JPEG thumbnails; returns their file names."""
    names: List[str] = []
    for i, frame in enumerate(frames):
        name = artifact_name(seq, f"_f{i}.jpg")
        if _write_image(snapshot_dir / name, frame, [cv2.IMWRITE_JPEG_QUALITY, 80]):
            names.append(name)
    return names


def save_detail(snapshot_dir: Path, seq: int, detail: Dict[str, Any]) -> Path:
    """Write the drill-down JSON (measurements, tolerances, OCR, per-frame data)."""
    path = snapshot_dir / artifact_name(seq, "_detail.json")
    try:
        path.write_text(json.dumps(detail, indent=2, default=str), encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not write %s: %s", path, exc)
    return path


def load_detail(snapshot_dir: Path, seq: int) -> Optional[Dict[str, Any]]:
    """Load the drill-down JSON for ``seq`` (None if missing or unreadable)."""
    path = snapshot_dir / artifact_name(seq, "_detail.json")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
