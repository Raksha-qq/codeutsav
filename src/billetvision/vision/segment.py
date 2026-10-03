"""Segmentation: Otsu, adaptive thresholding, morphology, contour extraction.

Coordinate frame: image origin (0,0) top-left, +X right, +Y down.
All returned contours are in pixel coordinates.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from billetvision.vision.preprocess import preprocess

logger = logging.getLogger(__name__)

# Minimum contour area as a fraction of the image area to ignore noise.
_MIN_AREA_FRAC = 0.005


@dataclass
class SegmentResult:
    """Output of the segmentation stage."""

    contour: np.ndarray  # (N,1,2) int32 pixel contour
    mask: np.ndarray  # binary uint8 mask, same size as input
    bounding_rect: tuple[int, int, int, int]  # x,y,w,h axis-aligned bbox
    method: str  # "otsu" | "adaptive" | "hot"


def _largest_valid_contour(
    contours: list[np.ndarray], min_area_px: float
) -> Optional[np.ndarray]:
    """Return the largest contour whose area exceeds *min_area_px*."""
    best: Optional[np.ndarray] = None
    best_area = 0.0
    for c in contours:
        area = cv2.contourArea(c)
        if area > min_area_px and area > best_area:
            best = c
            best_area = area
    return best


def _threshold_otsu(gray: np.ndarray) -> np.ndarray:
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def _threshold_adaptive(gray: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5
    )


def _threshold_hot(gray: np.ndarray, brightness_thresh: int = 200) -> np.ndarray:
    """High-brightness thresholding for glowing hot billets."""
    _, binary = cv2.threshold(gray, brightness_thresh, 255, cv2.THRESH_BINARY)
    return binary


def _clean_binary(binary: np.ndarray) -> np.ndarray:
    """Morphological open+close to remove noise and fill gaps."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    result = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel, iterations=2)
    return result


def segment(
    frame: np.ndarray,
    *,
    hot_billet_mode: bool = False,
    brightness_thresh: int = 200,
    min_area_frac: float = _MIN_AREA_FRAC,
    preprocess_kwargs: Optional[dict] = None,
) -> Optional[SegmentResult]:
    """Segment the billet from the background and return the dominant contour.

    Args:
        frame: Input BGR or gray image.
        hot_billet_mode: Use brightness thresholding instead of Otsu/adaptive.
        brightness_thresh: Minimum pixel intensity for hot-billet mode (0–255).
        min_area_frac: Contour must cover this fraction of image area to be valid.
        preprocess_kwargs: Extra keyword args forwarded to ``preprocess()``.

    Returns:
        SegmentResult with the best contour, or None if nothing valid was found.
    """
    kw = preprocess_kwargs or {}
    gray = preprocess(frame, hot_billet_mode=hot_billet_mode, **kw)

    h, w = gray.shape[:2]
    min_area_px = h * w * min_area_frac

    if hot_billet_mode:
        binary = _threshold_hot(gray, brightness_thresh)
        method = "hot"
    else:
        binary_otsu = _threshold_otsu(gray)
        contours_otsu, _ = cv2.findContours(
            _clean_binary(binary_otsu), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        best_otsu = _largest_valid_contour(contours_otsu, min_area_px)

        if best_otsu is not None:
            binary = binary_otsu
            method = "otsu"
        else:
            binary = _threshold_adaptive(gray)
            method = "adaptive"

    cleaned = _clean_binary(binary)

    # Invert if background is white (billet appears dark)
    if cv2.countNonZero(cleaned) > (h * w * 0.5):
        cleaned = cv2.bitwise_not(cleaned)

    contours, _ = cv2.findContours(
        cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    best = _largest_valid_contour(contours, min_area_px)
    if best is None:
        logger.warning("segment: no valid contour found (method=%s)", method)
        return None

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [best], -1, 255, cv2.FILLED)

    x, y, bw, bh = cv2.boundingRect(best)
    return SegmentResult(
        contour=best,
        mask=mask,
        bounding_rect=(x, y, bw, bh),
        method=method,
    )
