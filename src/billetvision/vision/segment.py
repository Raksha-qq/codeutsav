"""Segmentation: Otsu, adaptive thresholding, background subtraction, morphology.

Coordinate frame: image origin (0,0) top-left, +X right, +Y down.
All returned contours and masks are in **full-frame** pixel coordinates, even
when segmentation is restricted to a region of interest.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np

from billetvision.vision.preprocess import preprocess

logger = logging.getLogger(__name__)

# Minimum contour area as a fraction of the image area to ignore noise.
_MIN_AREA_FRAC = 0.005
# A "billet" covering more than this fraction of the search area is the whole
# background (bad threshold), not a billet.
_MAX_AREA_FRAC = 0.85
# Otsu always finds *some* split, even in an empty scene.  Require the
# foreground to differ from the background by at least this many grey levels,
# and to be solid (not an outline / scattered text) to count as a billet.
_MIN_CONTRAST = 45.0
_MIN_FILL = 0.6


@dataclass
class SegmentResult:
    """Output of the segmentation stage."""

    contour: np.ndarray  # (N,1,2) int32 pixel contour, full-frame coords
    mask: np.ndarray  # binary uint8 mask, same size as input frame
    bounding_rect: tuple[int, int, int, int]  # x,y,w,h axis-aligned bbox
    method: str  # "otsu" | "adaptive" | "hot" | "background"


def _largest_valid_contour(
    contours: Sequence[np.ndarray], min_area_px: float, max_area_px: float = float("inf")
) -> Optional[np.ndarray]:
    """Return the largest contour whose area is within (min_area_px, max_area_px]."""
    best: Optional[np.ndarray] = None
    best_area = 0.0
    for c in contours:
        area = cv2.contourArea(c)
        if min_area_px < area <= max_area_px and area > best_area:
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


def _border_fraction(mask: np.ndarray) -> float:
    """Fraction of the outermost pixel ring that is set in ``mask``."""
    ring = np.concatenate([mask[0, :], mask[-1, :], mask[1:-1, 0], mask[1:-1, -1]])
    return float(np.count_nonzero(ring)) / max(ring.size, 1)


def column_envelope(contour: np.ndarray) -> np.ndarray:
    """Re-trace ``contour`` as the per-column vertical envelope of its mask.

    Dark stamped text that touches the billet end, the edge or the ROI border
    opens notches in the mask.  For a conveyor running along X the true body is
    everything between the top-most and bottom-most mask pixel of each column,
    so filling that range closes such notches without hiding real shape
    information: a bent bar keeps its curved envelope and a width change keeps
    its step (neither is a "hole inside a column").

    Args:
        contour: Billet contour, full-frame pixel coordinates, (N,1,2).

    Returns:
        New contour (same coordinate frame); the input if nothing changed.
    """
    if len(contour) < 8:
        return contour
    x, y, w, h = cv2.boundingRect(contour)
    mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.drawContours(mask, [contour - np.array([[[x - 1, y - 1]]], dtype=contour.dtype)], -1, 255, cv2.FILLED)
    filled = mask > 0
    cols = filled.any(axis=0)
    top = np.argmax(filled, axis=0)
    bottom = filled.shape[0] - 1 - np.argmax(filled[::-1], axis=0)
    rows = np.arange(filled.shape[0])[:, None]
    envelope = (rows >= top[None, :]) & (rows <= bottom[None, :]) & cols[None, :]
    if int(envelope.sum()) == int(filled.sum()):
        return contour
    found, _ = cv2.findContours(envelope.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not found:
        return contour
    best = max(found, key=cv2.contourArea)
    return best + np.array([[[x - 1, y - 1]]], dtype=best.dtype)


def _is_billet_like(
    contour: np.ndarray, cleaned: np.ndarray, gray: Optional[np.ndarray]
) -> bool:
    """Reject hollow outlines, text blobs and low-contrast noise."""
    x, y, w, h = cv2.boundingRect(contour)
    local = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(local, [contour - np.array([[[x, y]]], dtype=contour.dtype)], -1, 255, cv2.FILLED)
    area = cv2.countNonZero(local)
    if area == 0:
        return False
    solid = cv2.countNonZero(cv2.bitwise_and(local, cleaned[y:y + h, x:x + w]))
    if solid / area < _MIN_FILL:
        return False
    if gray is not None:
        crop = gray[y:y + h, x:x + w]
        inside = crop[local > 0]
        outside = gray[cleaned == 0]
        if inside.size and outside.size and abs(float(inside.mean()) - float(outside.mean())) < _MIN_CONTRAST:
            return False
    return True


def _find_billet(
    binary: np.ndarray,
    min_area_px: float,
    max_area_px: float,
    gray: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Clean ``binary`` and return the largest valid, billet-like external contour.

    Polarity: the background is whatever dominates the search-area border, so if
    the border is mostly foreground-coloured the mask is flipped (dark billet on
    bright background).  A billet covering more than half the area is therefore
    not mistaken for the background.  ``gray`` (when given) enables the
    foreground/background contrast check.
    """
    cleaned = _clean_binary(binary)
    if _border_fraction(cleaned) > 0.5:
        cleaned = cv2.bitwise_not(cleaned)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(c)
        if area <= min_area_px:
            break
        if area <= max_area_px and _is_billet_like(c, cleaned, gray):
            return column_envelope(c)
    return None


class BackgroundModel:
    """MOG2 background subtractor used as a fallback when thresholding fails.

    Feed every frame through ``apply`` so the model keeps learning the static
    scene; ``foreground`` returns the latest binary foreground mask.
    """

    def __init__(self, history: int = 200, var_threshold: float = 32.0) -> None:
        self._mog = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_threshold, detectShadows=False
        )
        self._last: Optional[np.ndarray] = None

    def apply(self, gray: np.ndarray) -> np.ndarray:
        self._last = self._mog.apply(gray)
        return self._last

    @property
    def foreground(self) -> Optional[np.ndarray]:
        return self._last


def segment(
    frame: np.ndarray,
    *,
    hot_billet_mode: bool = False,
    brightness_thresh: int = 200,
    min_area_frac: float = _MIN_AREA_FRAC,
    preprocess_kwargs: Optional[dict] = None,
    roi: Optional[Sequence[int]] = None,
    background: Optional[BackgroundModel] = None,
) -> Optional[SegmentResult]:
    """Segment the billet from the background and return the dominant contour.

    Args:
        frame: Input BGR or gray image.
        hot_billet_mode: Use brightness thresholding instead of Otsu/adaptive.
        brightness_thresh: Minimum pixel intensity for hot-billet mode (0-255).
        min_area_frac: Contour must cover this fraction of the search area.
        preprocess_kwargs: Extra keyword args forwarded to ``preprocess()``.
        roi: Optional ``(x1, y1, x2, y2)`` pixel box; only this region is
            searched (HUD/overlay areas outside it are ignored).
        background: Optional ``BackgroundModel`` whose foreground mask is used
            as a last-resort fallback when thresholding finds nothing valid.

    Returns:
        SegmentResult in full-frame coordinates, or None if nothing valid.
    """
    kw = preprocess_kwargs or {}
    gray_full = preprocess(frame, hot_billet_mode=hot_billet_mode, **kw)
    fh, fw = gray_full.shape[:2]

    ox = oy = 0
    gray = gray_full
    if roi is not None:
        x1, y1 = max(0, int(roi[0])), max(0, int(roi[1]))
        x2, y2 = min(fw, int(roi[2])), min(fh, int(roi[3]))
        if x2 - x1 > 8 and y2 - y1 > 8:
            ox, oy = x1, y1
            gray = gray_full[y1:y2, x1:x2]

    h, w = gray.shape[:2]
    min_area_px = h * w * min_area_frac
    max_area_px = h * w * _MAX_AREA_FRAC

    best: Optional[np.ndarray] = None
    method = "otsu"
    if hot_billet_mode:
        method = "hot"
        best = _find_billet(_threshold_hot(gray, brightness_thresh), min_area_px, max_area_px, gray)
    else:
        best = _find_billet(_threshold_otsu(gray), min_area_px, max_area_px, gray)
        if best is None:
            method = "adaptive"
            best = _find_billet(_threshold_adaptive(gray), min_area_px, max_area_px, gray)

    if best is None and background is not None and background.foreground is not None:
        fg = background.foreground[oy:oy + h, ox:ox + w]
        if fg.shape[:2] == (h, w):
            method = "background"
            best = _find_billet(fg, min_area_px, max_area_px)

    if best is None:
        logger.debug("segment: no valid contour found")
        return None

    if ox or oy:
        best = best + np.array([[[ox, oy]]], dtype=best.dtype)

    mask = np.zeros((fh, fw), dtype=np.uint8)
    cv2.drawContours(mask, [best], -1, 255, cv2.FILLED)
    return SegmentResult(
        contour=best,
        mask=mask,
        bounding_rect=cv2.boundingRect(best),
        method=method,
    )
