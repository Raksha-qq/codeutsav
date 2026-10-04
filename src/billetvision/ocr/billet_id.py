"""Billet-ID readout: localise → read codes/OCR → validate → multi-frame vote.

Ties the OCR building blocks together for one tracked billet (FR-8..FR-12):

1. QR / barcode decode on the whole billet crop (a decoded code wins, FR-10).
2. Otherwise locate text regions (fixed ROI or automatic) and OCR each one.
3. Confidence-weighted vote across the billet's sharpest frames.
4. Regex + charset-correction validation; anything uncertain → ``REVIEW`` and
   the best crop is kept so the operator can type the right ID (rule 7).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from billetvision.ocr.codes import read_codes
from billetvision.ocr.locate import crop, locate_id_regions
from billetvision.ocr.reader import OcrReader, OcrResult
from billetvision.ocr.validate import correct_and_validate
from billetvision.ocr.vote import MultiFrameVoter

logger = logging.getLogger(__name__)

# Stop reading further frames once this many frames agree on a valid ID.
_EARLY_AGREEMENT = 2


@dataclass
class IdReadout:
    """Final ID decision for one billet."""

    text: str = ""
    confidence: float = 0.0
    status: str = "REVIEW"          # "PASS" | "REVIEW"
    matched: bool = False           # passed the format regex
    source: str = "ocr"             # "ocr" | "qr" | "barcode"
    engine: str = ""
    crop: Optional[np.ndarray] = None   # best ID crop (kept for the review queue)
    candidates: List[str] = field(default_factory=list)
    frames_read: int = 0

    def reason(self, min_confidence: float, regex: str) -> str:
        """Human-readable explanation used when status is REVIEW."""
        if not self.text:
            return "ID not readable — operator verification required"
        fmt = "format ok" if self.matched else f"does not match {regex}"
        return (
            f"ID '{self.text}' unconfirmed (confidence {self.confidence:.2f} "
            f"< {min_confidence:.2f} or {fmt}) — operator verification required"
        )


def _touches_border(box: tuple, shape: Sequence[int], margin: int = 3) -> bool:
    """True if ``box`` (x, y, w, h) touches the left/right edge of an image of ``shape``."""
    x, _, w, _ = box
    return x <= margin or x + w >= shape[1] - margin


def _observe_frame(
    gray: np.ndarray,
    reader: OcrReader,
    regex: str,
    fixed_roi: Optional[Sequence[float]],
    state: dict,
) -> None:
    """Collect this frame's code/OCR observations into ``state`` and track the best crop."""
    for code in read_codes(gray):
        state["obs"].append(code)
        state["code"] = True
    if state["code"]:
        return
    matched_obs: List[OcrResult] = []
    boxes = locate_id_regions(gray, fixed_roi)
    # A text line touching the crop border is probably cut off (a dropped last
    # digit still matches the format regex) — ignore it unless it is all we have.
    uncut = [b for b in boxes if not _touches_border(b, gray.shape)]
    for box in uncut or boxes:
        region = crop(gray, box)
        res = reader.read_best(region, pattern=regex)
        if not res.text:
            continue
        state["candidates"].append(res.text)
        _, matched, _ = correct_and_validate(res.text, regex)
        if matched:
            matched_obs.append(res)
        if res.confidence > state["best_conf"]:
            state["best_conf"], state["best_crop"] = res.confidence, region
    # Format-valid reads outrank noise from other regions of the same frame.
    state["obs"].extend(matched_obs)
    if not matched_obs and state["candidates"]:
        state["obs"].append(OcrResult(text=state["candidates"][-1], confidence=state["best_conf"], engine=""))


def _merge_truncated(observations: List[OcrResult], regex: str) -> List[OcrResult]:
    r"""Credit a truncated read to the longer read it is a prefix of.

    OCR drops trailing characters (cut-off text, a faint last glyph) far more
    often than it invents them, and a dropped digit can still satisfy a format
    regex such as ``\d{5,7}``.  A format-valid read that is a strict prefix of
    another format-valid read is therefore re-labelled as the longer text.
    """
    texts = [o.text for o in observations if correct_and_validate(o.text, regex)[1]]
    merged: List[OcrResult] = []
    for obs in observations:
        longer = [t for t in texts if len(t) > len(obs.text) and t.startswith(obs.text)]
        if longer:
            obs = OcrResult(text=max(longer, key=len), confidence=obs.confidence * 0.9,
                            source=obs.source, engine=obs.engine)
        merged.append(obs)
    return merged


def read_billet_id(
    frames: Sequence[np.ndarray],
    reader: OcrReader,
    regex: str,
    min_confidence: float = 0.60,
    fixed_roi: Optional[Sequence[float]] = None,
    max_frames: int = 3,
) -> IdReadout:
    """Read one billet's ID from its sharpest grayscale frames.

    Args:
        frames: Grayscale billet crops, sharpest first.
        reader: Configured ``OcrReader``.
        regex: Heat-ID format regex (``^[A-Z]\\d{5,7}$`` by default).
        min_confidence: Voted confidence below this means REVIEW.
        fixed_roi: Optional ``(fx, fy, fw, fh)`` fractions of the billet crop
            where the marking is; None = automatic localisation.
        max_frames: Upper bound on frames read (latency guard).

    Returns:
        IdReadout; ``status == "PASS"`` only for a format-valid, confident read.
    """
    state: dict = {"code": False, "candidates": [], "best_conf": -1.0, "best_crop": None, "obs": []}
    valid_texts: List[str] = []
    frames_read = 0

    for gray in list(frames)[: max(1, max_frames)]:
        frames_read += 1
        before = len(state["candidates"])
        try:
            _observe_frame(gray, reader, regex, fixed_roi, state)
        except Exception as exc:  # OCR must never crash the pipeline
            logger.warning("ID read failed on a frame: %s", exc)
            continue
        if state["code"]:
            break
        new = state["candidates"][before:]
        valid_texts.extend(t for t in new if correct_and_validate(t, regex)[1])
        if any(valid_texts.count(t) >= _EARLY_AGREEMENT for t in set(valid_texts)):
            break

    voter = MultiFrameVoter(min_confidence=min_confidence)
    for obs in _merge_truncated(state["obs"], regex):
        voter.add(obs)
    text, conf, _ = voter.decide()
    if text == "UNKNOWN":
        text = ""
    corrected, matched, penalty = correct_and_validate(text, regex) if text else ("", False, 0.0)
    final_text = corrected if matched else text
    final_conf = round(max(0.0, conf - (penalty if matched else 0.0)), 4)

    source, engine = "ocr", ""
    if state["code"]:
        source = "qr"
    status = "PASS" if (matched and final_conf >= min_confidence) else "REVIEW"
    return IdReadout(
        text=final_text,
        confidence=final_conf,
        status=status,
        matched=matched,
        source=source,
        engine=engine,
        crop=state["best_crop"],
        candidates=list(dict.fromkeys(state["candidates"])),
        frames_read=frames_read,
    )
