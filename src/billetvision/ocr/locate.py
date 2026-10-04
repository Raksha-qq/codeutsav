"""ID-region localisation: find the stamped/painted/printed text on a billet.

Two modes (FR-8):

* **Fixed ROI per config** — ``fixed_roi = (fx, fy, fw, fh)`` as fractions of the
  billet bounding box (0-1).  Deterministic; use when the marking position is
  known.
* **Auto** — classical text-region detection: adaptive threshold in both
  polarities, a wide horizontal close to merge characters into lines, then
  connected components filtered by size/aspect.

All boxes are ``(x, y, w, h)`` in the pixel coordinates of the image passed in.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

Box = Tuple[int, int, int, int]

_MIN_TEXT_HEIGHT_PX = 10
_MAX_TEXT_HEIGHT_FRAC = 0.5   # text taller than this fraction of the search area is not a line of text
_MIN_ASPECT = 1.5             # w / h of a merged text line
_PAD_FRAC = 0.25              # padding added around the detected line (fraction of height)


def fixed_region(image_shape: Sequence[int], fixed_roi: Sequence[float]) -> Box:
    """Convert a fractional ROI ``(fx, fy, fw, fh)`` to pixel ``(x, y, w, h)``."""
    h, w = image_shape[:2]
    fx, fy, fw, fh = (float(v) for v in fixed_roi)
    x, y = int(round(fx * w)), int(round(fy * h))
    bw, bh = int(round(fw * w)), int(round(fh * h))
    x, y = max(0, min(x, w - 1)), max(0, min(y, h - 1))
    return x, y, max(1, min(bw, w - x)), max(1, min(bh, h - y))


def _text_lines(binary: np.ndarray, max_h: int) -> List[Box]:
    """Merge characters into line boxes and keep text-shaped ones."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
    merged = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    count, _, stats, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)
    boxes: List[Box] = []
    for i in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[i])
        if h < _MIN_TEXT_HEIGHT_PX or h > max_h or w / max(h, 1) < _MIN_ASPECT:
            continue
        fill = cv2.countNonZero(binary[y:y + h, x:x + w]) / float(w * h)
        if 0.12 <= fill <= 0.85:  # solid blobs and hairlines are not text
            boxes.append((x, y, w, h))
    return boxes


def find_text_regions(gray: np.ndarray, max_regions: int = 2) -> List[Box]:
    """Detect candidate ID text lines in ``gray`` (billet crop).

    Args:
        gray: Single-channel uint8 image of the billet face.
        max_regions: Maximum number of boxes returned (largest first).

    Returns:
        Padded ``(x, y, w, h)`` boxes, possibly empty.
    """
    if gray is None or gray.size == 0:
        return []
    h, w = gray.shape[:2]
    # Ignore the outer margin where the billet edge / outline would dominate.
    m = max(4, int(0.03 * min(h, w)))
    inner = gray[m:h - m, m:w - m]
    if inner.size == 0:
        return []
    max_h = int(_MAX_TEXT_HEIGHT_FRAC * inner.shape[0])
    blurred = cv2.GaussianBlur(inner, (3, 3), 0)

    found: List[Box] = []
    for invert in (False, True):  # dark-on-light and light-on-dark text
        mode = cv2.THRESH_BINARY_INV if not invert else cv2.THRESH_BINARY
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, mode, 31, 12)
        found.extend(_text_lines(binary, max_h))

    found.sort(key=lambda b: b[2] * b[3], reverse=True)
    padded: List[Box] = []
    for x, y, bw, bh in found:
        pad = int(_PAD_FRAC * bh)
        x0, y0 = max(0, x + m - pad), max(0, y + m - pad)
        x1, y1 = min(w, x + m + bw + pad), min(h, y + m + bh + pad)
        box = (x0, y0, x1 - x0, y1 - y0)
        if all(_iou(box, p) < 0.5 for p in padded):
            padded.append(box)
        if len(padded) >= max_regions:
            break
    return padded


def _iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    iw, ih = max(0, min(ax2, bx2) - max(ax1, bx1)), max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union else 0.0


def locate_id_regions(
    gray: np.ndarray,
    fixed_roi: Optional[Sequence[float]] = None,
    max_regions: int = 2,
) -> List[Box]:
    """Return candidate ID boxes for ``gray``: the fixed ROI if configured,
    otherwise auto-detected text lines (falling back to the whole image)."""
    if fixed_roi:
        return [fixed_region(gray.shape, fixed_roi)]
    regions = find_text_regions(gray, max_regions=max_regions)
    return regions or [(0, 0, gray.shape[1], gray.shape[0])]


def crop(gray: np.ndarray, box: Box) -> np.ndarray:
    """Crop ``gray`` to ``box`` ``(x, y, w, h)``."""
    x, y, w, h = box
    return gray[y:y + h, x:x + w]
