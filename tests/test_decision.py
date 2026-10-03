"""Tests for decision engine (billetvision.decision.engine).

Verifies PRD §7.3 and §8 #2:
- Pure functions evaluating tolerances
- Explainable fail reasons (e.g. 'width 131.8 mm out of range (130.0 ± 1.0 mm)')
- PASS / FAIL / REVIEW / REWORK logic
- Support for square and round billet profiles
- Boundary edge cases
"""

import pytest
from billetvision.vision.measure import Measurement
from billetvision.decision.engine import evaluate_tolerances, Verdict


@pytest.fixture
def square_tolerances():
    return {
        "shape": "square",
        "width_nominal_mm": 130.0,
        "width_tol_mm": 1.0,
        "height_nominal_mm": 130.0,
        "height_tol_mm": 1.0,
        "length_nominal_mm": 1000.0,
        "length_tol_mm": 10.0,
        "max_diag_diff_mm": 2.0,
        "max_camber_mm": 3.0,
        "max_cross_section_var_mm": 2.5,
    }


@pytest.fixture
def round_tolerances():
    return {
        "shape": "round",
        "diameter_nominal_mm": 150.0,
        "diameter_tol_mm": 1.5,
        "length_nominal_mm": 1000.0,
        "length_tol_mm": 10.0,
        "max_ovality_pct": 1.5,
        "max_camber_mm": 3.0,
        "max_cross_section_var_mm": 2.5,
    }


class TestDecisionEngineSquare:
    def test_ideal_billet_passes(self, square_tolerances):
        m = Measurement(
            shape="square",
            length_mm=1000.0,
            width_mm=130.0,
            height_mm=130.0,
            diag_diff_mm=0.2,
            camber_mm=0.5,
            cross_section_var_mm=0.4,
        )
        v = evaluate_tolerances(m, square_tolerances)
        assert v.status == "PASS"
        assert v.is_acceptable is True
        assert len(v.fail_reasons) == 0

    def test_oversized_width_fails_with_explainable_reason(self, square_tolerances):
        m = Measurement(
            shape="square",
            length_mm=1000.0,
            width_mm=131.8,
            height_mm=130.0,
            diag_diff_mm=0.2,
        )
        v = evaluate_tolerances(m, square_tolerances)
        assert v.status == "FAIL"
        assert v.is_acceptable is False
        assert any("width" in r and "131.8" in r for r in v.fail_reasons)

    def test_undersized_height_fails(self, square_tolerances):
        m = Measurement(
            shape="square",
            length_mm=1000.0,
            width_mm=130.0,
            height_mm=128.5,
        )
        v = evaluate_tolerances(m, square_tolerances)
        assert v.status == "FAIL"
        assert any("height" in r for r in v.fail_reasons)

    def test_excessive_rhomboidity_fails(self, square_tolerances):
        m = Measurement(
            shape="square",
            length_mm=1000.0,
            width_mm=130.0,
            height_mm=130.0,
            diag_diff_mm=3.2,  # Max allowed is 2.0
        )
        v = evaluate_tolerances(m, square_tolerances)
        assert v.status == "FAIL"
        assert any("diagonal difference" in r for r in v.fail_reasons)

    def test_excessive_camber_fails(self, square_tolerances):
        m = Measurement(
            shape="square",
            length_mm=1000.0,
            width_mm=130.0,
            height_mm=130.0,
            camber_mm=4.5,  # Max is 3.0
        )
        v = evaluate_tolerances(m, square_tolerances)
        assert v.status == "FAIL"
        assert any("camber" in r for r in v.fail_reasons)

    def test_excessive_cross_section_var_fails(self, square_tolerances):
        m = Measurement(
            shape="square",
            length_mm=1000.0,
            width_mm=130.0,
            height_mm=130.0,
            cross_section_var_mm=3.8,  # Max is 2.5
        )
        v = evaluate_tolerances(m, square_tolerances)
        assert v.status == "FAIL"
        assert any("cross-section variation" in r for r in v.fail_reasons)

    def test_multiple_failures_listed(self, square_tolerances):
        m = Measurement(
            shape="square",
            length_mm=1000.0,
            width_mm=132.5,
            height_mm=127.0,
            diag_diff_mm=4.0,
        )
        v = evaluate_tolerances(m, square_tolerances)
        assert v.status == "FAIL"
        assert len(v.fail_reasons) == 3


class TestDecisionEngineRound:
    def test_round_pass(self, round_tolerances):
        m = Measurement(
            shape="round",
            length_mm=1000.0,
            width_mm=150.0,
            height_mm=150.0,
            diameter_mm=150.2,
            ovality=0.8,
        )
        v = evaluate_tolerances(m, round_tolerances)
        assert v.status == "PASS"

    def test_round_diameter_out_of_spec(self, round_tolerances):
        m = Measurement(
            shape="round",
            length_mm=1000.0,
            width_mm=152.5,
            height_mm=152.5,
            diameter_mm=152.5,  # Max allowed is 150.0 + 1.5 = 151.5
            ovality=0.5,
        )
        v = evaluate_tolerances(m, round_tolerances)
        assert v.status == "FAIL"
        assert any("diameter" in r for r in v.fail_reasons)

    def test_round_excessive_ovality(self, round_tolerances):
        m = Measurement(
            shape="round",
            length_mm=1000.0,
            width_mm=150.0,
            height_mm=150.0,
            diameter_mm=150.0,
            ovality=2.2,  # Max allowed is 1.5%
        )
        v = evaluate_tolerances(m, round_tolerances)
        assert v.status == "FAIL"
        assert any("ovality" in r for r in v.fail_reasons)


class TestDecisionEngineOcrReview:
    def test_ocr_review_status_propagated(self, square_tolerances):
        m = Measurement(
            shape="square",
            length_mm=1000.0,
            width_mm=130.0,
            height_mm=130.0,
        )
        v = evaluate_tolerances(m, square_tolerances, ocr_status="REVIEW")
        assert v.status == "REVIEW"
        assert v.is_acceptable is False
        assert any("OCR" in r for r in v.fail_reasons)
