"""QR and Barcode decoder module.

Uses cv2.QRCodeDetector (always available) and pyzbar (optional).
Falls back gracefully if pyzbar / libzbar is not installed.
"""
from __future__ import annotations

import logging
from typing import List, Optional

import cv2
import numpy as np

from billetvision.ocr.reader import OcrResult

logger = logging.getLogger(__name__)

try:
    from pyzbar import pyzbar as _pyzbar  # type: ignore
    _PYZBAR_AVAILABLE = True
except ImportError:
    _pyzbar = None
    _PYZBAR_AVAILABLE = False
    logger.debug("pyzbar not available — barcode decoding disabled (install pyzbar + libzbar)")


def read_qr(image: np.ndarray) -> Optional[OcrResult]:
    """Decode the first QR code found in ``image`` using OpenCV.

    Args:
        image: Grayscale or BGR image.

    Returns:
        OcrResult with source="qr" and confidence=1.0, or None if no QR found.
    """
    detector = cv2.QRCodeDetector()
    data, _, _ = detector.detectAndDecode(image)
    if data:
        return OcrResult(text=data.strip(), confidence=1.0, source="qr")
    return None


def read_barcodes(image: np.ndarray) -> List[OcrResult]:
    """Decode all barcodes in ``image`` using pyzbar.

    Returns an empty list if pyzbar is not installed or no barcode is found.

    Args:
        image: Grayscale or BGR image.

    Returns:
        List of OcrResult with source="barcode".
    """
    if not _PYZBAR_AVAILABLE:
        return []

    results: List[OcrResult] = []
    try:
        decoded = _pyzbar.decode(image)
        for d in decoded:
            text = d.data.decode("utf-8", errors="replace").strip()
            if text:
                results.append(OcrResult(text=text, confidence=1.0, source="barcode"))
    except Exception as exc:
        logger.warning("pyzbar decode error: %s", exc)
    return results


def read_codes(image: np.ndarray) -> List[OcrResult]:
    """Try QR then barcodes; return all successfully decoded results.

    QR results are prepended (higher priority than barcode in the voter).

    Args:
        image: Grayscale or BGR image.

    Returns:
        List of OcrResult, possibly empty.
    """
    results: List[OcrResult] = []

    qr = read_qr(image)
    if qr is not None:
        results.append(qr)

    results.extend(read_barcodes(image))
    return results
