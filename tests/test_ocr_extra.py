"""Tests for OCR validation and MultiFrameVoter.

Verifies PRD §7.2 (FR-10, FR-11, FR-12) & §8 #4:
- Regex validation for heat/batch numbers
- Optical character set substitution (O<->0, I<->1, S<->5, B<->8, Z<->2)
- Multi-frame voting across frames
- Barcode/QR code priority over standard OCR
- Low-confidence routing to REVIEW status
"""

import pytest
from billetvision.ocr.reader import OcrResult
from billetvision.ocr.validate import correct_id_format, validate_heat_id
from billetvision.ocr.vote import MultiFrameVoter


class TestOcrValidation:
    def test_regex_valid_heat_ids(self):
        assert validate_heat_id("H123456") is True
        assert validate_heat_id("A998877") is True
        assert validate_heat_id("K12345") is True

    def test_regex_invalid_heat_ids(self):
        assert validate_heat_id("123456") is False  # Missing prefix letter
        assert validate_heat_id("HH12345") is False # Double prefix letter
        assert validate_heat_id("H12") is False     # Too short

    def test_charset_correction(self):
        # 'O' confused with '0', 'I' confused with '1', 'S' with '5'
        assert correct_id_format("H12345O") == "H123450"
        assert correct_id_format("H12345I") == "H123451"
        assert correct_id_format("H12345S") == "H123455"
        assert correct_id_format("H12345B") == "H123458"
        assert correct_id_format("H12345Z") == "H123452"
        # Whitespace removal
        assert correct_id_format(" H 123 456 ") == "H123456"


class TestMultiFrameVoter:

    def test_qr_code_priority(self):
        voter = MultiFrameVoter()
        # Even if OCR saw something else, QR/barcode takes priority if confident
        voter.add_observation(OcrResult(text="H123450", confidence=0.70, source="ocr"))
        voter.add_observation(OcrResult(text="H123456", confidence=0.99, source="qr"))

        best_id, conf, status = voter.compute_final_id()
        assert best_id == "H123456"
        assert conf == 0.99
        assert status == "PASS"

    def test_low_confidence_routes_to_review(self):
        voter = MultiFrameVoter(min_confidence=0.75)
        # Average confidence is 0.55 < 0.75
        voter.add_observation(OcrResult(text="H123456", confidence=0.55, source="ocr"))
        best_id, conf, status = voter.compute_final_id()
        assert status == "REVIEW"
        assert best_id == "H123456"

    def test_empty_voter_routes_to_review(self):
        voter = MultiFrameVoter()
        best_id, conf, status = voter.compute_final_id()
        assert status == "REVIEW"
        assert best_id == "UNKNOWN"
