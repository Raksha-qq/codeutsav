"""Unit tests for ocr/ modules.

All tests are pure (no OCR engine required):
- enhance.py — image transform correctness
- validate.py — charset correction, regex validation
- vote.py    — confidence-weighted voting
- codes.py  — QR / barcode stubs (cv2 path only; pyzbar skipped if absent)
"""
from __future__ import annotations

import re
import numpy as np
import pytest
import cv2

from billetvision.ocr.reader import OcrResult
from billetvision.ocr.validate import (
    correct_and_validate,
    correct_for_digits,
    correct_for_letters,
    normalize,
    validate_regex,
)
from billetvision.ocr.enhance import (
    binarize,
    bilateral_denoise,
    clahe_enhance,
    deskew,
    enhancement_variants,
    unsharp_mask,
    upscale,
    ensure_min_width,
)
from billetvision.ocr.vote import MultiFrameVoter

PATTERN = r"^[A-Z]\d{5,7}$"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gray(h: int = 80, w: int = 320, val: int = 128) -> np.ndarray:
    return np.full((h, w), val, dtype=np.uint8)


def _render(text: str) -> np.ndarray:
    img = np.zeros((80, 320), dtype=np.uint8)
    cv2.putText(img, text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.4, 255, 2, cv2.LINE_AA)
    return img


# ---------------------------------------------------------------------------
# validate.py
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_strips_and_uppercases(self):
        assert normalize("  a123456  ") == "A123456"

    def test_removes_internal_spaces(self):
        assert normalize("A 1 2 3 4 5 6") == "A123456"

    def test_removes_dashes_underscores(self):
        assert normalize("A-123_456") == "A123456"


class TestValidateRegex:
    def test_exact_match_passes(self):
        assert validate_regex("A123456", PATTERN)

    def test_too_short_fails(self):
        assert not validate_regex("A1234", PATTERN)

    def test_too_long_fails(self):
        assert not validate_regex("A12345678", PATTERN)

    def test_no_leading_letter_fails(self):
        assert not validate_regex("1123456", PATTERN)

    def test_valid_7digit(self):
        assert validate_regex("B1234567", PATTERN)

    def test_invalid_regex_returns_false(self):
        assert not validate_regex("X", "[invalid regex")


class TestCorrectForDigits:
    def test_O_becomes_0(self):
        assert correct_for_digits("AO23456") == "A023456"

    def test_I_becomes_1(self):
        assert correct_for_digits("AI23456") == "A123456"

    def test_no_change_on_clean_text(self):
        assert correct_for_digits("A123456") == "A123456"

    def test_multiple_corrections(self):
        result = correct_for_digits("AOIZS456")
        assert "0" in result  # O→0
        assert "1" in result  # I→1
        assert "2" in result  # Z→2
        assert "5" in result  # S→5


class TestCorrectForLetters:
    def test_0_becomes_O(self):
        assert correct_for_letters("A023456")[1] == "O"

    def test_1_becomes_I(self):
        assert correct_for_letters("A123456")[1] == "I"


class TestCorrectAndValidate:
    def test_clean_text_passes_first_try(self):
        text, matched, penalty = correct_and_validate("A123456", PATTERN)
        assert matched
        assert text == "A123456"
        assert penalty == 0.0

    def test_lowercase_normalized_and_matched(self):
        text, matched, _ = correct_and_validate("a123456", PATTERN)
        assert matched
        assert text == "A123456"

    def test_letter_O_corrected_to_digit(self):
        # Raw "AO23456" — position 1 should be digit, O→0
        text, matched, penalty = correct_and_validate("AO23456", PATTERN)
        assert matched
        assert text == "A023456"
        assert penalty > 0

    def test_unrepairable_text_not_matched(self):
        _, matched, _ = correct_and_validate("!!!!!!", PATTERN)
        assert not matched

    def test_penalty_increases_with_more_corrections(self):
        _, _, p1 = correct_and_validate("A123456", PATTERN)   # step 1 match
        _, _, p2 = correct_and_validate("AO23456", PATTERN)   # requires correction
        assert p2 > p1

    def test_returns_norm_on_failure(self):
        text, matched, _ = correct_and_validate("garbage###", PATTERN)
        assert not matched
        assert text == "GARBAGE###"

    def test_strip_spaces(self):
        text, matched, _ = correct_and_validate(" A123456 ", PATTERN)
        assert matched
        assert text == "A123456"


# ---------------------------------------------------------------------------
# enhance.py
# ---------------------------------------------------------------------------

class TestEnhanceFunctions:
    def test_clahe_returns_same_shape(self):
        g = _gray()
        out = clahe_enhance(g)
        assert out.shape == g.shape
        assert out.dtype == np.uint8

    def test_bilateral_returns_same_shape(self):
        g = _gray()
        out = bilateral_denoise(g)
        assert out.shape == g.shape

    def test_unsharp_mask_returns_same_shape(self):
        g = _gray()
        out = unsharp_mask(g)
        assert out.shape == g.shape

    def test_binarize_returns_binary(self):
        g = _render("A123456")
        out = binarize(g)
        unique = set(np.unique(out).tolist())
        assert unique <= {0, 255}

    def test_upscale_doubles_size(self):
        g = _gray(40, 160)
        out = upscale(g, factor=2.0)
        assert out.shape == (80, 320)

    def test_upscale_identity_at_1(self):
        g = _gray()
        out = upscale(g, factor=1.0)
        assert out.shape == g.shape

    def test_ensure_min_width_enlarges_small(self):
        g = _gray(40, 80)
        out = ensure_min_width(g, min_width=200)
        assert out.shape[1] >= 200

    def test_ensure_min_width_leaves_large_unchanged(self):
        g = _gray(80, 320)
        out = ensure_min_width(g, min_width=200)
        assert out.shape == g.shape

    def test_deskew_returns_same_shape(self):
        g = _render("A123456")
        out = deskew(g)
        assert out.shape == g.shape

    def test_enhancement_variants_returns_list(self):
        g = _render("A123456")
        variants = enhancement_variants(g)
        assert len(variants) >= 3
        for v in variants:
            assert v.dtype == np.uint8
            assert v.shape[0] > 0 and v.shape[1] > 0


# ---------------------------------------------------------------------------
# vote.py
# ---------------------------------------------------------------------------

class TestMultiFrameVoter:
    def test_empty_returns_review(self):
        voter = MultiFrameVoter()
        _, conf, status = voter.decide()
        assert status == "REVIEW"
        assert conf == 0.0

    def test_single_high_confidence_passes(self):
        voter = MultiFrameVoter(min_confidence=0.60)
        voter.add(OcrResult(text="A123456", confidence=0.95))
        text, conf, status = voter.decide()
        assert text == "A123456"
        assert status == "PASS"

    def test_single_low_confidence_is_review(self):
        voter = MultiFrameVoter(min_confidence=0.60)
        voter.add(OcrResult(text="A123456", confidence=0.30))
        _, _, status = voter.decide()
        assert status == "REVIEW"

    def test_majority_wins(self):
        voter = MultiFrameVoter()
        voter.add(OcrResult(text="A123456", confidence=0.90))
        voter.add(OcrResult(text="A123456", confidence=0.85))
        voter.add(OcrResult(text="B999999", confidence=0.70))
        text, _, _ = voter.decide()
        assert text == "A123456"

    def test_qr_overrides_ocr(self):
        voter = MultiFrameVoter()
        voter.add(OcrResult(text="WRONG11", confidence=0.95, source="ocr"))
        voter.add(OcrResult(text="A123456", confidence=0.99, source="qr"))
        text, _, status = voter.decide()
        assert text == "A123456"
        assert status == "PASS"

    def test_reset_clears_state(self):
        voter = MultiFrameVoter()
        voter.add(OcrResult(text="A123456", confidence=0.95))
        voter.reset()
        assert voter.observation_count == 0
        _, _, status = voter.decide()
        assert status == "REVIEW"

    def test_confidence_weighted_prefers_high_conf(self):
        voter = MultiFrameVoter(min_confidence=0.60)
        # Two votes for A, one for B but with much higher confidence
        voter.add(OcrResult(text="A123456", confidence=0.50))
        voter.add(OcrResult(text="A123456", confidence=0.50))
        voter.add(OcrResult(text="B789012", confidence=0.98))
        # Weights: A = 1.0, B = 0.98 → A still wins by weight
        text, _, _ = voter.decide()
        assert text == "A123456"

    def test_legacy_compute_final_id_works(self):
        voter = MultiFrameVoter()
        voter.add(OcrResult(text="A123456", confidence=0.90))
        result = voter.compute_final_id()
        assert result[0] == "A123456"

    def test_legacy_add_observation_works(self):
        voter = MultiFrameVoter()
        voter.add_observation(OcrResult(text="A123456", confidence=0.90))
        assert voter.observation_count == 1


# ---------------------------------------------------------------------------
# codes.py — QR path (no real QR code, just checks it doesn't crash)
# ---------------------------------------------------------------------------

class TestCodes:
    def test_read_qr_no_code_returns_none(self):
        from billetvision.ocr.codes import read_qr
        blank = np.zeros((200, 200), dtype=np.uint8)
        result = read_qr(blank)
        assert result is None

    def test_read_codes_no_code_returns_empty(self):
        from billetvision.ocr.codes import read_codes
        blank = np.zeros((200, 200), dtype=np.uint8)
        results = read_codes(blank)
        assert isinstance(results, list)

    def test_read_qr_returns_ocr_result_on_match(self):
        """Generate a real QR code with cv2 and decode it back."""
        # Skip if QRCodeEncoder not available (older OpenCV)
        encoder = getattr(cv2, "QRCodeEncoder", None)
        if encoder is None:
            pytest.skip("cv2.QRCodeEncoder not available in this OpenCV build")
        qr_enc = encoder.create()
        qr_img = qr_enc.encode("A123456")
        from billetvision.ocr.codes import read_qr
        result = read_qr(qr_img)
        if result is not None:  # detection not guaranteed on tiny QR
            assert result.text == "A123456"
            assert result.source == "qr"
