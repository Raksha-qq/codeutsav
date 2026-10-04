"""Dependency-free fallback OCR for uppercase/digit IDs (template matching).

Used only when PaddleOCR, EasyOCR and Tesseract are all unavailable, so the
pipeline can still read clean printed/painted IDs offline.  It is a *fallback*:
it has no language model, no robustness to heavy distortion and no support for
dot-matrix or heavily stylised stamps.  Characters are segmented by connected
components and classified by normalised cross-correlation against glyphs
rendered from system fonts (via Pillow when available) plus OpenCV Hershey fonts.

Units: pixels.  Coordinate frame: image origin top-left.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
_GLYPH_W, _GLYPH_H = 20, 28
_RENDER_H = 64
_FONT_CANDIDATES = (
    "C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/segoeuib.ttf",
    "C:/Windows/Fonts/verdana.ttf", "C:/Windows/Fonts/calibri.ttf",
    "C:/Windows/Fonts/consola.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/Library/Fonts/Arial.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
)
_HERSHEY = (cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_TRIPLEX)

# (normalised bitmap flattened & zero-mean/unit-norm, aspect ratio w/h, char)
_Template = Tuple[np.ndarray, float, str]


def _normalise(glyph_bin: np.ndarray) -> Tuple[np.ndarray, float]:
    """Resize a tight binary glyph to the template grid; return (vector, aspect)."""
    h, w = glyph_bin.shape[:2]
    resized = cv2.resize(glyph_bin, (_GLYPH_W, _GLYPH_H), interpolation=cv2.INTER_AREA)
    vec = resized.astype(np.float32).ravel() / 255.0
    vec -= vec.mean()
    norm = float(np.linalg.norm(vec))
    return (vec / norm if norm > 1e-6 else vec), w / float(max(h, 1))


def _tight(binary: np.ndarray) -> Optional[np.ndarray]:
    ys, xs = np.where(binary > 0)
    if len(xs) == 0:
        return None
    return binary[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def _render_pillow(ch: str, path: str) -> Optional[np.ndarray]:
    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore

        font = ImageFont.truetype(path, _RENDER_H)
        img = Image.new("L", (_RENDER_H * 2, _RENDER_H * 2), 0)
        ImageDraw.Draw(img).text((_RENDER_H // 2, _RENDER_H // 2), ch, fill=255, font=font)
        arr = np.array(img)
        return _tight((arr > 127).astype(np.uint8) * 255)
    except Exception:  # missing/corrupt font or Pillow absent
        return None


def _render_hershey(ch: str, font: int) -> Optional[np.ndarray]:
    img = np.zeros((_RENDER_H * 2, _RENDER_H * 2), dtype=np.uint8)
    cv2.putText(img, ch, (_RENDER_H // 2, int(_RENDER_H * 1.4)), font, 2.0, 255, 3, cv2.LINE_AA)
    return _tight((img > 127).astype(np.uint8) * 255)


_BANK: Optional[List[_Template]] = None


def _template_bank() -> List[_Template]:
    """Build (once) the glyph template bank from all available fonts."""
    global _BANK
    if _BANK is not None:
        return _BANK
    bank: List[_Template] = []
    fonts = [p for p in _FONT_CANDIDATES if os.path.exists(p)]
    for ch in CHARSET:
        renders = [_render_pillow(ch, p) for p in fonts] + [_render_hershey(ch, f) for f in _HERSHEY]
        for glyph in renders:
            if glyph is not None and glyph.size:
                vec, aspect = _normalise(glyph)
                bank.append((vec, aspect, ch))
    _BANK = bank
    logger.info("Builtin OCR template bank: %d glyphs from %d system fonts", len(bank), len(fonts))
    return bank


def _classify(glyph_bin: np.ndarray, bank: List[_Template]) -> Tuple[str, float]:
    """Return (best_char, score in 0-1) for one tight binary glyph."""
    vec, aspect = _normalise(glyph_bin)
    best: Dict[str, float] = {}
    for tvec, taspect, ch in bank:
        score = float(np.dot(vec, tvec)) - 0.35 * abs(math.log(max(aspect, 0.05) / max(taspect, 0.05)))
        if score > best.get(ch, -9.0):
            best[ch] = score
    ch = max(best, key=lambda c: best[c])
    return ch, float(np.clip(best[ch], 0.0, 1.0))


def _binarise(gray: np.ndarray) -> np.ndarray:
    """Otsu-binarise with text forced to white (text is the minority class)."""
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if cv2.countNonZero(binary) > binary.size * 0.5:
        binary = cv2.bitwise_not(binary)
    return binary


def _glyph_boxes(binary: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Character-sized connected components, left to right."""
    h, w = binary.shape[:2]
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    comps = [tuple(int(v) for v in stats[i][:4]) for i in range(1, count) if stats[i][4] >= 12]
    comps = [c for c in comps if not (c[0] <= 0 or c[1] <= 0 or c[0] + c[2] >= w or c[1] + c[3] >= h)]
    if not comps:
        return []
    heights = sorted(c[3] for c in comps)
    ref = heights[int(0.75 * (len(heights) - 1))]  # robust "character height"
    keep = [c for c in comps if 0.6 * ref <= c[3] <= 1.5 * ref and c[2] <= 4.5 * ref]
    boxes: List[Tuple[int, int, int, int]] = []
    for box in sorted(keep, key=lambda c: c[0]):
        boxes.extend(_split_touching(binary, box, ref))
    return boxes


def _split_touching(
    binary: np.ndarray, box: Tuple[int, int, int, int], ref_h: int
) -> List[Tuple[int, int, int, int]]:
    """Split a component that is several touching characters into glyph boxes.

    A single character is rarely wider than ~1.2x its height; wider components
    are cut at the emptiest columns near the expected character pitch.
    """
    x, y, w, h = box
    if w <= 1.25 * ref_h:
        return [box]
    n = max(2, int(round(w / (0.8 * ref_h))))
    cols = (binary[y:y + h, x:x + w] > 0).sum(axis=0)
    cuts = [0]
    for i in range(1, n):
        centre = int(round(i * w / n))
        lo, hi = max(cuts[-1] + 2, centre - int(0.2 * w / n)), min(w - 2, centre + int(0.2 * w / n))
        cuts.append(lo + int(np.argmin(cols[lo:hi + 1])) if hi > lo else centre)
    cuts.append(w)
    return [(x + a, y, b - a, h) for a, b in zip(cuts[:-1], cuts[1:]) if b - a >= 3]


def read(gray: np.ndarray) -> List[Tuple[str, float]]:
    """Read word tokens from a grayscale image of one text region.

    Returns:
        ``(text, confidence)`` per whitespace-separated token, left to right.
    """
    if gray is None or gray.size == 0:
        return []
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    if gray.shape[0] < 96:  # small text: upscale so glyph strokes are resolvable
        scale = min(4.0, 96.0 / gray.shape[0])
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    padded = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_REPLICATE)
    binary = _binarise(padded)

    boxes = _glyph_boxes(binary)
    if not boxes:
        return []
    bank = _template_bank()
    ref_h = float(np.median([b[3] for b in boxes]))

    tokens: List[List[Tuple[str, float]]] = [[]]
    prev_right: Optional[int] = None
    for x, y, w, h in boxes:
        if prev_right is not None and x - prev_right > 0.55 * ref_h:
            tokens.append([])
        glyph = binary[y:y + h, x:x + w]
        tokens[-1].append(_classify(glyph, bank))
        prev_right = x + w
    out: List[Tuple[str, float]] = []
    for tok in tokens:
        if tok:
            out.append(("".join(c for c, _ in tok), float(np.mean([s for _, s in tok]))))
    return out
