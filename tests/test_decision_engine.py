"""Unit tests for decision/engine.py.

All tests are pure — no filesystem, no OpenCV, no camera.
Measurement objects are built directly from known values.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from billetvision.decision.engine import (
    REWORK_FACTOR,
    Verdict,
    evaluate,
    evaluate_tolerances,
    load_tolerances,
)
from billetvision.vision.measure import Measurement

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TOLERANCES_PATH = Path("config/tolerances.yaml")


@pytest.fixture(scope="module")
def tols() -> dict:
    return load_tolerances(TOLERANCES_PATH)


def _rect(
    length_mm: float = 1000.0,
    width_mm: float = 130.0,
    height_mm: float = 130.0,
    diag_diff_mm: float = 0.0,
    camber_mm: float = 0.0,
    cross_section_var_mm: float = 0.0,
    surface_anomaly_score: float = 0.0,
) -> Measurement:
    return Measurement(
        shape="square",
        length_mm=length_mm,
        width_mm=width_mm,
        height_mm=height_mm,
        diag_diff_mm=diag_diff_mm,
        camber_mm=camber_mm,
        cross_section_var_mm=cross_section_var_mm,
        surface_anomaly_score=surface_anomaly_score,
    )


def _round_billet(
    length_mm: float = 1000.0,
    diameter_mm: float = 150.0,
    ovality: float = 0.0,
    camber_mm: float = 0.0,
    cross_section_var_mm: float = 0.0,
    surface_anomaly_score: float = 0.0,
) -> Measurement:
    return Measurement(
        shape="round",
        length_mm=length_mm,
        width_mm=diameter_mm,
        height_mm=diameter_mm,
        diameter_mm=diameter_mm,
        ovality=ovality,
        camber_mm=camber_mm,
        cross_section_var_mm=cross_section_var_mm,
        surface_anomaly_score=surface_anomaly_score,
    )


# ---------------------------------------------------------------------------
# load_tolerances
# ---------------------------------------------------------------------------

class TestLoadTolerances:
    def test_returns_dict(self, tols):
        assert isinstance(tols, dict)

    def test_known_profiles_present(self, tols):
        assert "square_130" in tols
        assert "square_100" in tols
        assert "round_150" in tols

    def test_square_130_fields(self, tols):
        p = tols["square_130"]
        assert p["width_nominal_mm"] == pytest.approx(130.0)
        assert p["width_tol_mm"] == pytest.approx(1.0)
        assert p["max_diag_diff_mm"] == pytest.approx(2.0)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_tolerances(tmp_path / "nonexistent.yaml")

    def test_bad_yaml_raises(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("- not\n- a\n- mapping\n")
        with pytest.raises(ValueError):
            load_tolerances(bad)


# ---------------------------------------------------------------------------
# PASS cases
# ---------------------------------------------------------------------------

class TestPass:
    def test_perfect_nominal(self, tols):
        v = evaluate_tolerances(_rect(), tols["square_130"])
        assert v.status == "PASS"
        assert v.reasons == []

    def test_exact_tolerance_boundary_is_pass(self, tols):
        # width exactly at nominal + tol → should PASS (≤ not >)
        p = tols["square_130"]
        w = p["width_nominal_mm"] + p["width_tol_mm"]
        v = evaluate_tolerances(_rect(width_mm=w), tols["square_130"])
        assert v.status == "PASS"

    def test_negative_boundary_is_pass(self, tols):
        p = tols["square_130"]
        w = p["width_nominal_mm"] - p["width_tol_mm"]
        v = evaluate_tolerances(_rect(width_mm=w), tols["square_130"])
        assert v.status == "PASS"

    def test_is_acceptable_true_on_pass(self, tols):
        v = evaluate_tolerances(_rect(), tols["square_130"])
        assert v.is_acceptable is True

    def test_fail_reasons_alias_empty_on_pass(self, tols):
        v = evaluate_tolerances(_rect(), tols["square_130"])
        assert v.fail_reasons == []


# ---------------------------------------------------------------------------
# REWORK cases (violation ≤ REWORK_FACTOR × tolerance)
# ---------------------------------------------------------------------------

class TestRework:
    def _just_over_tol(self, tols, field: str = "width_mm") -> Measurement:
        p = tols["square_130"]
        # 1.01 × tol over nominal: beyond tol but within REWORK_FACTOR × tol
        deviation = p["width_tol_mm"] * 1.01
        return _rect(width_mm=p["width_nominal_mm"] + deviation)

    def test_minor_width_violation_is_rework(self, tols):
        v = evaluate_tolerances(self._just_over_tol(tols), tols["square_130"])
        assert v.status == "REWORK"
        assert len(v.reasons) == 1
        assert "width" in v.reasons[0]

    def test_rework_is_not_acceptable(self, tols):
        v = evaluate_tolerances(self._just_over_tol(tols), tols["square_130"])
        assert v.is_acceptable is False

    def test_minor_height_violation_is_rework(self, tols):
        p = tols["square_130"]
        deviation = p["height_tol_mm"] * 1.5
        m = _rect(height_mm=p["height_nominal_mm"] + deviation)
        v = evaluate_tolerances(m, tols["square_130"])
        assert v.status == "REWORK"
        assert "height" in v.reasons[0]

    def test_minor_camber_violation_is_rework(self, tols):
        p = tols["square_130"]
        # Just over the max_camber_mm limit but within REWORK_FACTOR × limit
        camber = p["max_camber_mm"] * 1.5
        m = _rect(camber_mm=camber)
        v = evaluate_tolerances(m, tols["square_130"])
        assert v.status == "REWORK"

    def test_multiple_minor_violations_stay_rework(self, tols):
        p = tols["square_130"]
        w = p["width_nominal_mm"] + p["width_tol_mm"] * 1.5
        h = p["height_nominal_mm"] + p["height_tol_mm"] * 1.5
        m = _rect(width_mm=w, height_mm=h)
        v = evaluate_tolerances(m, tols["square_130"])
        assert v.status == "REWORK"
        assert len(v.reasons) == 2


# ---------------------------------------------------------------------------
# FAIL cases (violation > REWORK_FACTOR × tolerance)
# ---------------------------------------------------------------------------

class TestFail:
    def test_severe_width_violation_is_fail(self, tols):
        p = tols["square_130"]
        # Well beyond REWORK_FACTOR × tol
        w = p["width_nominal_mm"] + p["width_tol_mm"] * (REWORK_FACTOR + 1)
        v = evaluate_tolerances(_rect(width_mm=w), tols["square_130"])
        assert v.status == "FAIL"
        assert "width" in v.reasons[0]

    def test_fail_is_not_acceptable(self, tols):
        p = tols["square_130"]
        w = p["width_nominal_mm"] + p["width_tol_mm"] * (REWORK_FACTOR + 1)
        v = evaluate_tolerances(_rect(width_mm=w), tols["square_130"])
        assert v.is_acceptable is False

    def test_fail_overrides_rework(self, tols):
        """One severe + one minor violation → FAIL, both reasons present."""
        p = tols["square_130"]
        severe_w = p["width_nominal_mm"] + p["width_tol_mm"] * (REWORK_FACTOR + 1)
        minor_h = p["height_nominal_mm"] + p["height_tol_mm"] * 1.5
        m = _rect(width_mm=severe_w, height_mm=minor_h)
        v = evaluate_tolerances(m, tols["square_130"])
        assert v.status == "FAIL"
        assert len(v.reasons) == 2  # both reasons included

    def test_multiple_fail_reasons(self, tols):
        p = tols["square_130"]
        factor = REWORK_FACTOR + 1
        w = p["width_nominal_mm"] + p["width_tol_mm"] * factor
        h = p["height_nominal_mm"] + p["height_tol_mm"] * factor
        m = _rect(width_mm=w, height_mm=h)
        v = evaluate_tolerances(m, tols["square_130"])
        assert v.status == "FAIL"
        assert len(v.reasons) == 2

    def test_severe_camber_is_fail(self, tols):
        p = tols["square_130"]
        camber = p["max_camber_mm"] * (REWORK_FACTOR + 1)
        m = _rect(camber_mm=camber)
        v = evaluate_tolerances(m, tols["square_130"])
        assert v.status == "FAIL"

    def test_severe_diag_diff_is_fail(self, tols):
        p = tols["square_130"]
        diag = p["max_diag_diff_mm"] * (REWORK_FACTOR + 1)
        m = _rect(diag_diff_mm=diag)
        v = evaluate_tolerances(m, tols["square_130"])
        assert v.status == "FAIL"

    def test_severe_cross_section_var_is_fail(self, tols):
        p = tols["square_130"]
        var = p["max_cross_section_var_mm"] * (REWORK_FACTOR + 1)
        m = _rect(cross_section_var_mm=var)
        v = evaluate_tolerances(m, tols["square_130"])
        assert v.status == "FAIL"

    def test_surface_anomaly_fail(self, tols):
        p = tols["square_130"]
        score = p["max_surface_anomaly_score"] * (REWORK_FACTOR + 1)
        m = _rect(surface_anomaly_score=score)
        v = evaluate_tolerances(m, tols["square_130"])
        assert v.status == "FAIL"

    def test_surface_anomaly_rework(self, tols):
        p = tols["square_130"]
        score = p["max_surface_anomaly_score"] * 1.5  # within REWORK_FACTOR
        m = _rect(surface_anomaly_score=score)
        v = evaluate_tolerances(m, tols["square_130"])
        assert v.status == "REWORK"


# ---------------------------------------------------------------------------
# REVIEW cases (OCR uncertainty)
# ---------------------------------------------------------------------------

class TestReview:
    def test_uncertain_ocr_with_good_dimensions_is_review(self, tols):
        v = evaluate_tolerances(_rect(), tols["square_130"], ocr_status="REVIEW")
        assert v.status == "REVIEW"
        assert len(v.reasons) == 1
        assert "OCR" in v.reasons[0]

    def test_dimensional_fail_overrides_review(self, tols):
        p = tols["square_130"]
        w = p["width_nominal_mm"] + p["width_tol_mm"] * (REWORK_FACTOR + 1)
        m = _rect(width_mm=w)
        v = evaluate_tolerances(m, tols["square_130"], ocr_status="REVIEW")
        assert v.status == "FAIL"

    def test_dimensional_rework_overrides_review(self, tols):
        p = tols["square_130"]
        w = p["width_nominal_mm"] + p["width_tol_mm"] * 1.5
        m = _rect(width_mm=w)
        v = evaluate_tolerances(m, tols["square_130"], ocr_status="REVIEW")
        assert v.status == "REWORK"

    def test_ocr_pass_does_not_cause_review(self, tols):
        v = evaluate_tolerances(_rect(), tols["square_130"], ocr_status="PASS")
        assert v.status == "PASS"


# ---------------------------------------------------------------------------
# Length checks
# ---------------------------------------------------------------------------

class TestLengthCheck:
    def test_length_pass(self, tols):
        v = evaluate_tolerances(_rect(length_mm=1000.0), tols["square_130"])
        assert v.status == "PASS"

    def test_length_at_boundary(self, tols):
        p = tols["square_130"]
        length = p["length_nominal_mm"] + p["length_tol_mm"]
        v = evaluate_tolerances(_rect(length_mm=length), tols["square_130"])
        assert v.status == "PASS"

    def test_length_minor_fail_is_rework(self, tols):
        p = tols["square_130"]
        length = p["length_nominal_mm"] + p["length_tol_mm"] * 1.5
        v = evaluate_tolerances(_rect(length_mm=length), tols["square_130"])
        assert v.status == "REWORK"
        assert "length" in v.reasons[0]

    def test_length_severe_fail_is_fail(self, tols):
        p = tols["square_130"]
        length = p["length_nominal_mm"] + p["length_tol_mm"] * (REWORK_FACTOR + 1)
        v = evaluate_tolerances(_rect(length_mm=length), tols["square_130"])
        assert v.status == "FAIL"


# ---------------------------------------------------------------------------
# Round billet checks
# ---------------------------------------------------------------------------

class TestRoundBillet:
    def test_perfect_round_pass(self, tols):
        m = _round_billet()
        v = evaluate_tolerances(m, tols["round_150"])
        assert v.status == "PASS"

    def test_diameter_out_of_tol_rework(self, tols):
        p = tols["round_150"]
        d = p["diameter_nominal_mm"] + p["diameter_tol_mm"] * 1.5
        m = _round_billet(diameter_mm=d)
        v = evaluate_tolerances(m, tols["round_150"])
        assert v.status == "REWORK"
        assert "diameter" in v.reasons[0]

    def test_ovality_rework(self, tols):
        p = tols["round_150"]
        ovality = p["max_ovality_pct"] * 1.5
        m = _round_billet(ovality=ovality)
        v = evaluate_tolerances(m, tols["round_150"])
        assert v.status == "REWORK"

    def test_ovality_fail(self, tols):
        p = tols["round_150"]
        ovality = p["max_ovality_pct"] * (REWORK_FACTOR + 1)
        m = _round_billet(ovality=ovality)
        v = evaluate_tolerances(m, tols["round_150"])
        assert v.status == "FAIL"


# ---------------------------------------------------------------------------
# evaluate() top-level wrapper
# ---------------------------------------------------------------------------

class TestEvaluateWrapper:
    def test_pass_via_evaluate(self, tols):
        v = evaluate(_rect(), "square_130", tols)
        assert v.status == "PASS"

    def test_fail_via_evaluate(self, tols):
        p = tols["square_130"]
        w = p["width_nominal_mm"] + p["width_tol_mm"] * (REWORK_FACTOR + 1)
        v = evaluate(_rect(width_mm=w), "square_130", tols)
        assert v.status == "FAIL"

    def test_unknown_profile_raises_key_error(self, tols):
        with pytest.raises(KeyError, match="not found"):
            evaluate(_rect(), "nonexistent_profile", tols)

    def test_ocr_review_forwarded(self, tols):
        v = evaluate(_rect(), "square_130", tols, ocr_status="REVIEW")
        assert v.status == "REVIEW"


# ---------------------------------------------------------------------------
# Reason message quality
# ---------------------------------------------------------------------------

class TestReasonMessages:
    def test_reason_includes_measured_value(self, tols):
        p = tols["square_130"]
        w = p["width_nominal_mm"] + p["width_tol_mm"] * 1.5
        m = _rect(width_mm=w)
        v = evaluate_tolerances(m, tols["square_130"])
        assert f"{w:.2f}" in v.reasons[0]

    def test_reason_includes_nominal_and_tolerance(self, tols):
        p = tols["square_130"]
        w = p["width_nominal_mm"] + p["width_tol_mm"] * 1.5
        m = _rect(width_mm=w)
        v = evaluate_tolerances(m, tols["square_130"])
        assert "130.00" in v.reasons[0]  # nominal
        assert "1.00" in v.reasons[0]    # tol

    def test_reason_includes_deviation(self, tols):
        p = tols["square_130"]
        w = p["width_nominal_mm"] + p["width_tol_mm"] * 1.5
        m = _rect(width_mm=w)
        v = evaluate_tolerances(m, tols["square_130"])
        assert "deviation" in v.reasons[0]

    def test_ocr_reason_mentions_operator(self, tols):
        v = evaluate_tolerances(_rect(), tols["square_130"], ocr_status="REVIEW")
        assert "operator" in v.reasons[0].lower()
