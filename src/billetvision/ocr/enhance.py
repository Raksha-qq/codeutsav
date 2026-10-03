"""Image enhancement variants for OCR pre-processing.

All functions accept and return single-channel uint8 (grayscale) numpy arrays
unless noted otherwise.  The caller is responsible for converting BGR → gray
before calling.
"""
from __future__ import annotations

import logging
from typing import List, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Individual enhancement steps
# ---------------------------------------------------------------------------

def clahe_enhance(
    gray: np.ndarray,
    clip_limit: float = 3.0,
    tile_grid: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Contrast-limited adaptive histogram equalisation."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return clahe.apply(gray)


def bilateral_denoise(gray: np.ndarray, d: int = 9, sigma: float = 75.0) -> np.ndarray:
    """Edge-preserving bilateral filter — keeps text edges sharp."""
    return cv2.bilateralFilter(gray, d, sigma, sigma)


def unsharp_mask(
    gray: np.ndarray,
    blur_sigma: float = 1.0,
    strength: float = 1.5,
) -> np.ndarray:
    """Unsharp-mask sharpening: original + strength × (original − blurred)."""
    blurred = cv2.GaussianBlur(gray, (0, 0), blur_sigma)
    sharpened = cv2.addWeighted(gray, 1.0 + strength, blurred, -strength, 0)
    return sharpened


def binarize(gray: np.ndarray, block_size: int = 21, C: int = 10) -> np.ndarray:
    """Adaptive thresholding — good for uneven lighting on stamped text."""
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, C
    )


def upscale(gray: np.ndarray, factor: float = 2.0) -> np.ndarray:
    """Upscale image by ``factor`` using Lanczos interpolation."""
    if abs(factor - 1.0) < 1e-6:
        return gray
    h, w = gray.shape[:2]
    return cv2.resize(
        gray, (int(w * factor), int(h * factor)), interpolation=cv2.INTER_LANCZOS4
    )


def ensure_min_width(gray: np.ndarray, min_width: int = 200) -> np.ndarray:
    """Upscale if the image is narrower than ``min_width`` pixels."""
    h, w = gray.shape[:2]
    if w < min_width:
        scale = min_width / w
        return upscale(gray, factor=scale)
    return gray


def deskew(gray: np.ndarray) -> np.ndarray:
    """Estimate text skew and rotate to straighten.

    Uses the projection-profile method on a binarised copy.
    Skew is clamped to ±15 ° to avoid wild rotations on noise.
    """
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(binary > 0))
    if len(coords) < 10:
        return gray
    angle = cv2.minAreaRect(coords)[-1]  # degrees in [-90, 0)
    # Convert to a [-45, 45] range
    if angle < -45:
        angle += 90
    if abs(angle) < 0.5:
        return gray  # not worth rotating
    angle = float(np.clip(angle, -15, 15))
    h, w = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=255)
    return rotated


# ---------------------------------------------------------------------------
# Composite pipelines
# ---------------------------------------------------------------------------

def enhance_for_ocr(gray: np.ndarray) -> np.ndarray:
    """Default full enhancement chain: ensure min size → CLAHE → bilateral → sharpen."""
    gray = ensure_min_width(gray, min_width=200)
    gray = clahe_enhance(gray)
    gray = bilateral_denoise(gray)
    gray = unsharp_mask(gray)
    return gray


def enhancement_variants(gray: np.ndarray) -> List[np.ndarray]:
    """Return a ranked list of enhancement variants to try with OCR.

    The list is ordered from most to least likely to help; the caller tries
    each variant and picks the highest-confidence result.
    """
    base = ensure_min_width(gray, min_width=200)

    variants: List[np.ndarray] = []

    # 1. Default pipeline
    variants.append(enhance_for_ocr(base))

    # 2. Aggressive CLAHE + sharpen (low-contrast stamps)
    v2 = clahe_enhance(base, clip_limit=5.0, tile_grid=(4, 4))
    v2 = unsharp_mask(v2, strength=2.0)
    variants.append(v2)

    # 3. Raw (sometimes the original is already good)
    variants.append(base)

    # 4. Binarised — useful for printed / QR-adjacent text
    v4 = clahe_enhance(base)
    v4 = binarize(v4)
    variants.append(v4)

    # 5. Deskewed default pipeline
    try:
        v5 = deskew(base)
        v5 = enhance_for_ocr(v5)
        variants.append(v5)
    except Exception:
        pass  # deskew can fail on degenerate images

    return variants
