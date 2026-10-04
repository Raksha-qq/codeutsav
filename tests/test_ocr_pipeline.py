"""Tests for ID localisation, the builtin OCR fallback and the billet-ID readout."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from billetvision.ocr import builtin
from billetvision.ocr.billet_id import _merge_truncated, read_billet_id
from billetvision.ocr.locate import find_text_regions, fixed_region, locate_id_regions
from billetvision.ocr.reader import OcrReader, OcrResult

REGEX = r"^[A-Z]\d{5,7}$"


def _text_img(text, size=(420, 120), fg=30, bg=170, scale=1.6, thick=3):
    img = np.full((size[1], size[0]), bg, dtype=np.uint8)
    cv2.putText(img, text, (20, 75), cv2.FONT_HERSHEY_DUPLEX, scale, fg, thick, cv2.LINE_AA)
    return img


class FakeReader:
    """Stands in for OcrReader: returns scripted results per call."""

    def __init__(self, texts):
        self._texts = list(texts)
        self.calls = 0

    def read_best(self, gray, pattern=None):
        text = self._texts[min(self.calls, len(self._texts) - 1)]
        self.calls += 1
        return OcrResult(text=text, confidence=0.9, engine="fake")


@pytest.mark.parametrize("text", ["H123456", "A000001", "Z999999", "B789012"])
@pytest.mark.parametrize("polarity", ["dark_on_light", "light_on_dark"])
def test_builtin_reads_clean_ids(text, polarity):
    img = _text_img(text) if polarity == "dark_on_light" else _text_img(text, fg=230, bg=30)
    tokens = [t for t, _ in builtin.read(img)]
    assert text in tokens


def test_builtin_splits_label_and_id_and_empty_is_safe():
    tokens = [t for t, _ in builtin.read(_text_img("HEAT: H123456", size=(640, 120), scale=1.4))]
    assert "H123456" in tokens
    assert builtin.read(np.full((50, 50), 128, np.uint8)) == []
    assert builtin.read(None) == []


def test_find_text_regions_on_billet_like_image():
    img = np.full((260, 700), 165, np.uint8)
    cv2.putText(img, "H123456", (200, 140), cv2.FONT_HERSHEY_DUPLEX, 1.6, 30, 3, cv2.LINE_AA)
    boxes = find_text_regions(img)
    assert boxes
    x, y, w, h = boxes[0]
    assert x < 260 and x + w > 420 and y < 120 and y + h > 140


def test_fixed_region_and_fallback_to_whole_image():
    assert fixed_region((200, 400), (0.5, 0.25, 0.25, 0.5)) == (200, 50, 100, 100)
    blank = np.full((100, 200), 120, np.uint8)
    assert locate_id_regions(blank) == [(0, 0, 200, 100)]
    assert locate_id_regions(blank, fixed_roi=(0, 0, 0.5, 0.5)) == [(0, 0, 100, 50)]


def test_reader_falls_back_to_builtin_and_reads():
    reader = OcrReader(engine="builtin")
    res = reader.read_best(_text_img("H123456"), pattern=REGEX)
    assert res.text == "H123456" and res.engine == "builtin" and res.confidence > 0.6


def test_read_billet_id_valid_pass():
    frames = [_text_img("H123456")] * 3
    out = read_billet_id(frames, OcrReader(engine="builtin"), REGEX, min_confidence=0.6)
    assert out.status == "PASS" and out.text == "H123456" and out.matched


def test_read_billet_id_unreadable_goes_to_review_with_crop():
    blank = np.full((120, 300), 120, np.uint8)
    out = read_billet_id([blank], FakeReader([""]), REGEX)
    assert out.status == "REVIEW" and out.text == ""
    assert "not readable" in out.reason(0.6, REGEX)


def test_read_billet_id_bad_format_is_review_not_silent_pass():
    out = read_billet_id([_text_img("HEAT")], FakeReader(["XYZ"]), REGEX)
    assert out.status == "REVIEW" and not out.matched
    assert "unconfirmed" in out.reason(0.6, REGEX)


def test_merge_truncated_prefers_longer_valid_read():
    obs = [OcrResult("H12345", 0.9), OcrResult("H123456", 0.8), OcrResult("H12345", 0.9)]
    merged = _merge_truncated(obs, REGEX)
    assert [o.text for o in merged] == ["H123456"] * 3


def test_read_billet_id_prefix_truncation_resolved():
    out = read_billet_id([_text_img("A")] * 3, FakeReader(["H12345", "H123456", "H12345"]), REGEX, max_frames=3)
    assert out.text == "H123456" and out.status == "PASS"


def test_read_billet_id_never_raises_on_engine_error():
    class Boom:
        def read_best(self, *_a, **_k):
            raise RuntimeError("engine died")

    out = read_billet_id([_text_img("H123456")], Boom(), REGEX)
    assert out.status == "REVIEW"


def test_qr_code_wins_when_present():
    enc = getattr(cv2, "QRCodeEncoder", None)
    if enc is None:
        pytest.skip("cv2.QRCodeEncoder unavailable")
    qr = enc.create().encode("H654321")
    qr = cv2.resize(qr, None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST)
    img = cv2.copyMakeBorder(qr, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=255)
    out = read_billet_id([img], FakeReader(["WRONG1"]), REGEX)
    assert out.text == "H654321" and out.source == "qr" and out.status == "PASS"
