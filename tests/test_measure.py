"""Tests for vision/segment.py and vision/measure.py.

Synthetic images are used throughout — a rectangle or circle of known pixel
size is drawn, segmented, then measured.  All values must land within ±1% of
the true physical dimension (the ≤±1% requirement from CLAUDE.md).
"""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import pytest

from billetvision.vision.measure import Measurement, measure, measure_rect, measure_round
from billetvision.vision.segment import segment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MM_PER_PX = 0.5  # 2 px per mm — easy round numbers


def _rect_image(
    length_px: int, width_px: int, canvas: tuple[int, int] = (600, 1400)
) -> tuple[np.ndarray, np.ndarray]:
    """Draw a white rectangle on a black canvas; return (bgr_image, contour)."""
    h, w = canvas
    img = np.zeros((h, w), dtype=np.uint8)
    x0 = (w - length_px) // 2
    y0 = (h - width_px) // 2
    cv2.rectangle(img, (x0, y0), (x0 + length_px, y0 + width_px), 255, -1)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return bgr, contours[0]


def _circle_image(
    diameter_px: int, canvas: tuple[int, int] = (600, 600)
) -> tuple[np.ndarray, np.ndarray]:
    """Draw a white circle on a black canvas; return (bgr_image, contour)."""
    h, w = canvas
    img = np.zeros((h, w), dtype=np.uint8)
    cx, cy = w // 2, h // 2
    r = diameter_px // 2
    cv2.circle(img, (cx, cy), r, 255, -1)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return bgr, contours[0]


def _within_1pct(measured: float, expected: float) -> bool:
    if expected == 0:
        return measured == 0
    return abs(measured - expected) / expected <= 0.01


# ---------------------------------------------------------------------------
# segment() tests
# ---------------------------------------------------------------------------


class TestSegment:
    def test_segment_finds_rect(self):
        bgr, _ = _rect_image(800, 260)
        result = segment(bgr)
        assert result is not None, "segment() should find the rectangle"
        assert result.contour is not None
        assert result.mask.sum() > 0

    def test_segment_returns_none_on_blank(self):
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        result = segment(blank)
        assert result is None

    def test_segment_hot_billet_mode(self):
        bgr, _ = _rect_image(800, 260)
        # Make it bright (simulate hot billet)
        hot = bgr.copy()
        hot[hot > 0] = 220
        result = segment(hot, hot_billet_mode=True)
        assert result is not None
        assert result.method == "hot"

    def test_segment_method_label(self):
        bgr, _ = _rect_image(800, 260)
        result = segment(bgr)
        assert result is not None
        assert result.method in ("otsu", "adaptive")


# ---------------------------------------------------------------------------
# measure_rect() tests
# ---------------------------------------------------------------------------


class TestMeasureRect:
    def test_length_within_1pct(self):
        length_px = 2000
        width_px = 260
        _, contour = _rect_image(length_px, width_px, canvas=(500, 2400))
        m = measure_rect(contour, MM_PER_PX)
        expected_mm = length_px * MM_PER_PX
        assert _within_1pct(m.length_mm, expected_mm), (
            f"length {m.length_mm:.2f} not within 1% of {expected_mm:.2f}"
        )

    def test_width_within_1pct(self):
        length_px = 2000
        width_px = 260
        _, contour = _rect_image(length_px, width_px, canvas=(500, 2400))
        m = measure_rect(contour, MM_PER_PX)
        expected_mm = width_px * MM_PER_PX
        assert _within_1pct(m.width_mm, expected_mm), (
            f"width {m.width_mm:.2f} not within 1% of {expected_mm:.2f}"
        )

    def test_shape_field(self):
        _, contour = _rect_image(800, 260)
        m = measure_rect(contour, MM_PER_PX)
        assert m.shape == "square"

    def test_diag_diff_perfect_rect_is_near_zero(self):
        _, contour = _rect_image(800, 260)
        m = measure_rect(contour, MM_PER_PX)
        # Perfect synthetic rectangle should have near-zero diagonal difference
        assert m.diag_diff_mm is not None
        assert m.diag_diff_mm < 1.0, f"diag_diff={m.diag_diff_mm} mm unexpectedly large"

    def test_measurement_dataclass_fields(self):
        _, contour = _rect_image(800, 260)
        m = measure_rect(contour, MM_PER_PX)
        assert isinstance(m, Measurement)
        assert m.length_mm > 0
        assert m.width_mm > 0
        assert m.height_mm > 0


# ---------------------------------------------------------------------------
# measure_round() tests
# ---------------------------------------------------------------------------


class TestMeasureRound:
    def test_diameter_within_1pct(self):
        diameter_px = 300
        _, contour = _circle_image(diameter_px)
        m = measure_round(contour, MM_PER_PX)
        expected_mm = diameter_px * MM_PER_PX
        assert _within_1pct(m.diameter_mm, expected_mm), (
            f"diameter {m.diameter_mm:.2f} not within 1% of {expected_mm:.2f}"
        )

    def test_ovality_perfect_circle_is_near_zero(self):
        _, contour = _circle_image(300)
        m = measure_round(contour, MM_PER_PX)
        assert m.ovality is not None
        assert m.ovality < 2.0, f"ovality={m.ovality:.3f}% unexpectedly large"

    def test_shape_field(self):
        _, contour = _circle_image(300)
        m = measure_round(contour, MM_PER_PX)
        assert m.shape == "round"


# ---------------------------------------------------------------------------
# measure() dispatcher
# ---------------------------------------------------------------------------


class TestMeasureDispatch:
    def test_dispatches_square(self):
        _, contour = _rect_image(800, 260)
        m = measure(contour, MM_PER_PX, shape="square")
        assert m.shape == "square"

    def test_dispatches_round(self):
        _, contour = _circle_image(300)
        m = measure(contour, MM_PER_PX, shape="round")
        assert m.shape == "round"


# ---------------------------------------------------------------------------
# Ground-truth CSV validation
# ---------------------------------------------------------------------------


GROUND_TRUTH_PATH = Path(__file__).parent.parent / "data" / "ground_truth.csv"


@pytest.mark.skipif(
    not GROUND_TRUTH_PATH.exists(), reason="data/ground_truth.csv not found"
)
class TestGroundTruth:
    """Verify that synthetic images matching each ground-truth row produce
    measurements within ±1% of the caliper reference values."""

    @staticmethod
    def _build_contour_for_row(row: dict) -> tuple[np.ndarray, str, float]:
        """Build a synthetic contour that matches the GT row dimensions."""
        shape = row["shape"]
        # Use a representative mm/px that keeps images sensible
        mm_per_px = 0.5

        if shape == "round":
            d_mm = float(row["caliper_diameter_mm"])
            d_px = int(d_mm / mm_per_px)
            _, contour = _circle_image(d_px, canvas=(d_px + 200, d_px + 200))
        else:
            l_mm = float(row["caliper_length_mm"])
            w_mm = float(row["caliper_width_mm"])
            l_px = int(l_mm / mm_per_px)
            w_px = int(w_mm / mm_per_px)
            _, contour = _rect_image(l_px, w_px, canvas=(w_px + 200, l_px + 200))

        return contour, shape, mm_per_px

    def test_all_rows_within_tolerance(self):
        errors = []
        with open(GROUND_TRUTH_PATH, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                contour, shape, mm_per_px = self._build_contour_for_row(row)
                m = measure(contour, mm_per_px, shape=shape)

                pid = row["prop_id"]
                if shape == "round":
                    expected = float(row["caliper_diameter_mm"])
                    measured = m.diameter_mm
                    if not _within_1pct(measured, expected):
                        errors.append(
                            f"{pid}: diameter {measured:.2f} vs {expected:.2f} mm"
                        )
                else:
                    # Check length and width
                    for field_name, col in [
                        ("length_mm", "caliper_length_mm"),
                        ("width_mm", "caliper_width_mm"),
                    ]:
                        expected = float(row[col])
                        measured = getattr(m, field_name)
                        if not _within_1pct(measured, expected):
                            errors.append(
                                f"{pid} {field_name}: {measured:.2f} vs {expected:.2f} mm"
                            )

        assert not errors, "Ground-truth ±1% check failed:\n" + "\n".join(errors)
