"""Defect detection: surface anomalies, edge irregularity, and defect labelling.

Profile variation along the length and camber are computed in ``measure.py``;
this module adds the surface/edge metrics and turns every defect metric into
human-readable labels for the ``defects`` log column.

Units: millimetres unless stated; ``surface_anomaly_score`` is a 0-1 fraction of
the billet area flagged as anomalous.  Coordinate frame: image origin top-left,
+X right, +Y down.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from billetvision.vision.calibrate import px_to_mm

logger = logging.getLogger(__name__)

# A pixel is anomalous when it deviates from the local surface by more than
# max(_MIN_DEVIATION, _SIGMA_K * robust sigma) grey levels.
_MIN_DEVIATION = 25.0
_SIGMA_K = 4.0
# Anomalous blobs smaller than this many pixels are treated as sensor noise.
_MIN_BLOB_PX = 12
# Fraction of the billet bounding box trimmed off each side so edges (and any
# burned-in outline) do not count as surface defects.
_EDGE_ERODE_PX = 6


def surface_anomaly_score(gray: np.ndarray, mask: np.ndarray) -> float:
    """Fraction (0-1) of the billet surface that deviates from its local mean.

    Args:
        gray: Single-channel uint8 image containing the billet.
        mask: uint8 mask (255 = billet), same size as ``gray``.

    Returns:
        Anomalous-pixel area divided by the eroded billet area; 0.0 when the
        mask is empty or too small to analyse.
    """
    if gray is None or mask is None or gray.shape[:2] != mask.shape[:2]:
        return 0.0
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (2 * _EDGE_ERODE_PX + 1, 2 * _EDGE_ERODE_PX + 1)
    )
    inner = cv2.erode(mask, kernel)
    area = int(cv2.countNonZero(inner))
    if area < 100:
        return 0.0

    background = cv2.medianBlur(gray, 31)
    deviation = cv2.absdiff(gray, background).astype(np.float32)
    values = deviation[inner > 0]
    sigma = 1.4826 * float(np.median(np.abs(values - np.median(values))))
    threshold = max(_MIN_DEVIATION, _SIGMA_K * sigma)

    anomalous = ((deviation > threshold) & (inner > 0)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(anomalous, connectivity=8)
    flagged = 0
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] >= _MIN_BLOB_PX:
            flagged += int(stats[i, cv2.CC_STAT_AREA])
    return float(min(1.0, flagged / area))


def edge_irregularity_px(contour: np.ndarray, shape: str = "square") -> float:
    """Robust (95th-percentile) deviation of the outline from its ideal shape.

    The ideal shape is the minAreaRect for square billets and the fitted
    ellipse for round ones.

    Args:
        contour: Billet contour, pixel coordinates, (N,1,2).
        shape: ``"square"`` or ``"round"``.

    Returns:
        Deviation in pixels (0.0 for degenerate contours).
    """
    if contour is None or len(contour) < 8:
        return 0.0
    pts = contour.reshape(-1, 2).astype(np.float32)
    if shape == "round" and len(pts) >= 5:
        (cx, cy), (d1, d2), angle = cv2.fitEllipse(contour)
        theta = np.deg2rad(angle)
        dx, dy = pts[:, 0] - cx, pts[:, 1] - cy
        u = dx * np.cos(theta) + dy * np.sin(theta)
        v = -dx * np.sin(theta) + dy * np.cos(theta)
        a, b = max(d1, 1e-6) / 2.0, max(d2, 1e-6) / 2.0
        radial = np.sqrt((u / a) ** 2 + (v / b) ** 2)
        residual = np.abs(radial - 1.0) * min(a, b)
        return float(np.percentile(residual, 95))

    box = cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.float32).reshape(-1, 1, 2)
    residual = np.array(
        [abs(cv2.pointPolygonTest(box, (float(x), float(y)), True)) for x, y in pts]
    )
    return float(np.percentile(residual, 95))


def edge_irregularity_mm(contour: np.ndarray, mm_per_px: float, shape: str = "square") -> float:
    """``edge_irregularity_px`` converted to millimetres."""
    return round(float(px_to_mm(edge_irregularity_px(contour, shape), mm_per_px)), 3)


def classify_defects(
    *,
    camber_mm: Optional[float],
    cross_section_var_mm: Optional[float],
    surface_anomaly: float,
    edge_irregularity: Optional[float],
    tol: Dict[str, Any],
) -> List[str]:
    """Return defect labels for every metric exceeding its tolerance limit.

    Pure function: compares metrics against the ``max_*`` keys of one profile's
    tolerance dict and never raises on missing keys.
    """
    labels: List[str] = []
    checks = (
        ("bent", camber_mm, "max_camber_mm"),
        ("profile_variation", cross_section_var_mm, "max_cross_section_var_mm"),
        ("surface_anomaly", surface_anomaly, "max_surface_anomaly_score"),
        ("edge_irregular", edge_irregularity, "max_edge_irregularity_mm"),
    )
    for label, value, key in checks:
        limit = tol.get(key)
        if value is not None and limit is not None and value > limit:
            labels.append(label)
    return labels
