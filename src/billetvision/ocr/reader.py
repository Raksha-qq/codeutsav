"""OCR reader: PaddleOCR primary, EasyOCR fallback, Tesseract last resort.

Tries each enhancement variant and returns the highest-confidence result that
passes the ID regex (or the best raw result if none pass).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class OcrResult:
    """Single OCR read from one image crop."""

    text: str
    confidence: float
    source: str = "ocr"  # "ocr" | "qr" | "barcode" | "manual"
    engine: str = ""     # "paddle" | "easyocr" | "tesseract" | ""


# ---------------------------------------------------------------------------
# Engine availability probes (lazy import to avoid crash on missing deps)
# ---------------------------------------------------------------------------

def _try_import_paddle():
    try:
        from paddleocr import PaddleOCR  # type: ignore
        return PaddleOCR
    except ImportError:
        logger.debug("PaddleOCR not installed")
        return None


def _try_import_easyocr():
    try:
        import easyocr  # type: ignore
        return easyocr
    except ImportError:
        logger.debug("EasyOCR not installed")
        return None


def _try_import_tesseract():
    try:
        import pytesseract  # type: ignore
        return pytesseract
    except ImportError:
        logger.debug("pytesseract not installed")
        return None


# ---------------------------------------------------------------------------
# Per-engine reader callables
# ---------------------------------------------------------------------------

def _read_with_paddle(
    image: np.ndarray,
    paddle_instance,
    lang: str = "en",
) -> List[OcrResult]:
    """Run PaddleOCR on ``image``.  Returns list of OcrResult (may be empty)."""
    try:
        results = paddle_instance.ocr(image, cls=True)
        out: List[OcrResult] = []
        if not results or results[0] is None:
            return out
        for line in results[0]:
            if line is None:
                continue
            text_info = line[1]  # (text, confidence)
            text, conf = str(text_info[0]), float(text_info[1])
            if text.strip():
                out.append(OcrResult(text=text.strip(), confidence=conf, engine="paddle"))
        return out
    except Exception as exc:
        logger.warning("PaddleOCR error: %s", exc)
        return []


def _read_with_easyocr(
    image: np.ndarray,
    reader_instance,
) -> List[OcrResult]:
    """Run EasyOCR on ``image``.  Returns list of OcrResult."""
    try:
        results = reader_instance.readtext(image, detail=1)
        out: List[OcrResult] = []
        for (_bbox, text, conf) in results:
            if str(text).strip():
                out.append(OcrResult(text=str(text).strip(), confidence=float(conf), engine="easyocr"))
        return out
    except Exception as exc:
        logger.warning("EasyOCR error: %s", exc)
        return []


def _read_with_tesseract(image: np.ndarray, pytesseract) -> List[OcrResult]:
    """Run Tesseract on ``image``.  Returns list of OcrResult."""
    try:
        config = "--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        data = pytesseract.image_to_data(
            image, config=config, output_type=pytesseract.Output.DICT
        )
        out: List[OcrResult] = []
        for i, text in enumerate(data["text"]):
            text = str(text).strip()
            conf = float(data["conf"][i])
            if text and conf > 0:
                out.append(OcrResult(text=text, confidence=conf / 100.0, engine="tesseract"))
        return out
    except Exception as exc:
        logger.warning("Tesseract error: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Main reader class
# ---------------------------------------------------------------------------

class OcrReader:
    """Multi-engine OCR reader with enhancement-variant iteration.

    Usage::

        reader = OcrReader.from_config(config["ocr"])
        result = reader.read_best(crop_gray, pattern=r"^[A-Z]\\d{5,7}$")
    """

    def __init__(
        self,
        engine: str = "easyocr",
        languages: List[str] | None = None,
        min_confidence: float = 0.60,
        pattern: str = r"^[A-Z]\d{5,7}$",
    ) -> None:
        self.engine = engine
        self.languages = languages or ["en"]
        self.min_confidence = min_confidence
        self.pattern = pattern

        self._paddle = None
        self._easyocr = None
        self._tesseract = None
        self._init_engines()

    def _init_engines(self) -> None:
        if self.engine in ("paddleocr", "paddle"):
            PaddleOCR = _try_import_paddle()
            if PaddleOCR:
                try:
                    self._paddle = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
                    logger.info("PaddleOCR initialised")
                except Exception as exc:
                    logger.warning("PaddleOCR init failed: %s", exc)

        if self._paddle is None:
            easyocr_mod = _try_import_easyocr()
            if easyocr_mod:
                try:
                    self._easyocr = easyocr_mod.Reader(self.languages, gpu=False, verbose=False)
                    logger.info("EasyOCR initialised (fallback)")
                except Exception as exc:
                    logger.warning("EasyOCR init failed: %s", exc)

        if self._paddle is None and self._easyocr is None:
            self._tesseract = _try_import_tesseract()
            if self._tesseract:
                logger.info("Tesseract initialised (last-resort fallback)")
            else:
                logger.warning("No OCR engine available — all reads will return empty")

    # ------------------------------------------------------------------

    def _run_engines(self, image: np.ndarray) -> List[OcrResult]:
        """Run whichever engines are available; return combined candidates."""
        candidates: List[OcrResult] = []
        if self._paddle is not None:
            candidates.extend(_read_with_paddle(image, self._paddle))
        if self._easyocr is not None:
            candidates.extend(_read_with_easyocr(image, self._easyocr))
        if self._tesseract is not None:
            candidates.extend(_read_with_tesseract(image, self._tesseract))
        return candidates

    def read_candidates(self, gray: np.ndarray) -> List[OcrResult]:
        """Try all enhancement variants and return all candidates, deduped.

        Candidates are sorted by confidence descending.
        """
        from billetvision.ocr.enhance import enhancement_variants

        all_candidates: List[OcrResult] = []
        variants = enhancement_variants(gray)
        for variant in variants:
            all_candidates.extend(self._run_engines(variant))

        # Deduplicate by text, keeping max confidence
        best: dict[str, OcrResult] = {}
        for r in all_candidates:
            key = r.text.strip().upper()
            if key not in best or r.confidence > best[key].confidence:
                best[key] = r

        return sorted(best.values(), key=lambda r: r.confidence, reverse=True)

    def read_best(
        self,
        gray: np.ndarray,
        pattern: str | None = None,
    ) -> OcrResult:
        """Return the single best OcrResult for ``gray``.

        If ``pattern`` is given, prefers results that pass after charset correction.

        Args:
            gray: Cropped grayscale image of the ID region.
            pattern: Regex to validate the corrected text.

        Returns:
            OcrResult with text, confidence, and engine set.
            Falls back to OcrResult("", 0.0) if nothing is detected.
        """
        from billetvision.ocr.validate import correct_and_validate

        pat = pattern or self.pattern
        candidates = self.read_candidates(gray)

        if not candidates:
            return OcrResult(text="", confidence=0.0, source="ocr")

        # First, look for a candidate that passes the regex after correction
        for r in candidates:
            corrected, matched, penalty = correct_and_validate(r.text, pat)
            if matched:
                final_conf = max(0.0, r.confidence - penalty)
                return OcrResult(
                    text=corrected,
                    confidence=final_conf,
                    source=r.source,
                    engine=r.engine,
                )

        # No match — return the highest-confidence raw result
        best = candidates[0]
        from billetvision.ocr.validate import normalize
        return OcrResult(
            text=normalize(best.text),
            confidence=best.confidence,
            source=best.source,
            engine=best.engine,
        )

    @classmethod
    def from_config(cls, ocr_cfg: dict) -> "OcrReader":
        """Build OcrReader from a config dict (the 'ocr' section of config.yaml)."""
        return cls(
            engine=ocr_cfg.get("engine", "easyocr"),
            languages=ocr_cfg.get("languages", ["en"]),
            min_confidence=float(ocr_cfg.get("min_confidence", 0.60)),
            pattern=ocr_cfg.get("heat_id_regex", r"^[A-Z]\d{5,7}$"),
        )
