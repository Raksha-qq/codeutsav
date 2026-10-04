"""Preprocessing pipeline: CLAHE, undistort, noise reduction, exposure normalization."""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def to_gray(frame: np.ndarray) -> np.ndarray:
    """Convert BGR or already-gray frame to single-channel uint8."""
    if frame.ndim == 2:
        return frame
    if frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def apply_clahe(
    gray: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Apply CLAHE (contrast-limited adaptive histogram equalisation)."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray)


def gamma_correct(gray: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Apply power-law (gamma) correction.

    gamma < 1 brightens; gamma > 1 darkens.
    """
    if abs(gamma - 1.0) < 1e-6:
        return gray
    inv_gamma = 1.0 / gamma
    lut = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8
    )
    return cv2.LUT(gray, lut)


def denoise(gray: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Apply Gaussian blur for noise reduction."""
    return cv2.GaussianBlur(gray, (ksize, ksize), 0)


def auto_gamma(
    gray: np.ndarray, target_mean: float = 110.0, lo: float = 0.4, hi: float = 2.5
) -> float:
    """Pick a gamma that moves the mean grey level towards ``target_mean``.

    Returns 1.0 (identity) for already well-exposed frames (mean within 35% of
    the target) so normal footage is untouched.
    """
    mean = float(np.mean(gray))
    if mean < 1.0 or mean > 254.0 or abs(mean - target_mean) / target_mean < 0.35:
        return 1.0
    gamma = float(np.log(mean / 255.0) / np.log(target_mean / 255.0))
    return float(np.clip(gamma, lo, hi))


def preprocess(
    frame: np.ndarray,
    *,
    auto_exposure: bool = False,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
    gamma: float = 1.0,
    denoise_ksize: int = 3,
    hot_billet_mode: bool = False,
) -> np.ndarray:
    """Full preprocessing chain: BGR → gray → CLAHE → gamma → denoise.

    Args:
        frame: Input BGR (or gray) image.
        clip_limit: CLAHE clip limit.
        tile_grid_size: CLAHE tile grid size.
        gamma: Gamma correction exponent (1.0 = identity).
        denoise_ksize: Gaussian blur kernel size (odd integer ≥ 1).
        auto_exposure: If True, derive ``gamma`` from the frame's mean
            brightness (exposure normalisation for varied lighting).
        hot_billet_mode: If True, skip CLAHE (image is already high-contrast from
            brightness thresholding) and apply a larger denoise kernel.

    Returns:
        Preprocessed single-channel uint8 image.
    """
    gray = to_gray(frame)
    if auto_exposure and not hot_billet_mode:
        gamma = auto_gamma(gray)
    if hot_billet_mode:
        gray = denoise(gray, ksize=max(denoise_ksize, 5))
    else:
        gray = apply_clahe(gray, clip_limit=clip_limit, tile_grid_size=tile_grid_size)
        gray = gamma_correct(gray, gamma=gamma)
        if denoise_ksize > 1:
            gray = denoise(gray, ksize=denoise_ksize)
    return gray
