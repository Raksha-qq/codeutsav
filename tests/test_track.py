"""Tests for vision/track.py.

All tests use synthetic data — no real images needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from billetvision.vision.measure import Measurement
from billetvision.vision.track import (
    BilletTrack,
    CentroidTracker,
    FrameSample,
    TrackedBillet,
    TrackState,
    median_measurement,
    sharpness,
    sharpness_of_contour,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MM_PER_PX = 0.5


def _dummy_measurement(
    length: float = 1000.0,
    width: float = 130.0,
    height: float = 130.0,
) -> Measurement:
    return Measurement(
        shape="square",
        length_mm=length,
        width_mm=width,
        height_mm=height,
    )


def _gray(brightness: int = 128, size: tuple[int, int] = (100, 200)) -> np.ndarray:
    return np.full(size, brightness, dtype=np.uint8)


def _rect_contour(
    x: int = 10, y: int = 10, w: int = 80, h: int = 40
) -> np.ndarray:
    """Return a 4-point rectangular contour."""
    return np.array(
        [[[x, y]], [[x + w, y]], [[x + w, y + h]], [[x, y + h]]], dtype=np.int32
    )


def _detection(
    cx: float,
    cy: float = 360.0,
    length: float = 1000.0,
    width: float = 130.0,
) -> tuple:
    """Build a synthetic (centroid, contour, gray, measurement) detection tuple."""
    contour = _rect_contour(int(cx) - 40, int(cy) - 30, 80, 60)
    gray = _gray()
    meas = _dummy_measurement(length=length, width=width)
    return ((cx, cy), contour, gray, meas)


def _tracker(entry_x: int = 100, exit_x: int = 1100, **kwargs) -> CentroidTracker:
    defaults = dict(max_distance_px=80.0, max_lost_frames=5, best_n=3)
    defaults.update(kwargs)
    return CentroidTracker(entry_x=entry_x, exit_x=exit_x, **defaults)


# ---------------------------------------------------------------------------
# sharpness helpers
# ---------------------------------------------------------------------------


class TestSharpness:
    def test_blank_is_zero(self):
        blank = np.zeros((100, 100), dtype=np.uint8)
        assert sharpness(blank) == pytest.approx(0.0)

    def test_noise_higher_than_blank(self):
        rng = np.random.default_rng(42)
        noisy = rng.integers(0, 255, (100, 100), dtype=np.uint8)
        assert sharpness(noisy) > 100.0

    def test_none_returns_zero(self):
        assert sharpness(None) == 0.0

    def test_contour_crop_matches_roi(self):
        gray = _gray(200)
        contour = _rect_contour(10, 10, 80, 40)
        score = sharpness_of_contour(gray, contour)
        assert score >= 0.0


# ---------------------------------------------------------------------------
# median_measurement
# ---------------------------------------------------------------------------


class TestMedianMeasurement:
    def _make_samples(self, lengths: list[float], sharpnesses: list[float]) -> list[FrameSample]:
        return [
            FrameSample(
                frame_gray=_gray(int(s * 0.5) % 255),
                contour=_rect_contour(),
                measurement=_dummy_measurement(length=l),
                sharpness_score=s,
            )
            for l, s in zip(lengths, sharpnesses)
        ]

    def test_selects_best_n_by_sharpness(self):
        # Sharpest 3 have lengths 1000, 1001, 1002 — median 1001
        samples = self._make_samples(
            lengths=[1000.0, 1001.0, 1002.0, 500.0, 500.0],
            sharpnesses=[90.0, 95.0, 92.0, 10.0, 5.0],
        )
        med, best = median_measurement(samples, best_n=3)
        assert med.length_mm == pytest.approx(1001.0)

    def test_best_sample_is_sharpest(self):
        samples = self._make_samples(
            lengths=[1000.0, 1001.0],
            sharpnesses=[50.0, 99.0],
        )
        _, best = median_measurement(samples, best_n=5)
        assert best.sharpness_score == pytest.approx(99.0)

    def test_single_sample(self):
        samples = self._make_samples([1000.0], [50.0])
        med, _ = median_measurement(samples, best_n=5)
        assert med.length_mm == pytest.approx(1000.0)

    def test_raises_on_empty(self):
        with pytest.raises(ValueError):
            median_measurement([], best_n=5)

    def test_optional_fields_none_handled(self):
        # Measurements where diameter_mm is None
        samples = [
            FrameSample(
                frame_gray=_gray(),
                contour=_rect_contour(),
                measurement=Measurement(
                    shape="square",
                    length_mm=1000.0,
                    width_mm=130.0,
                    height_mm=130.0,
                    diameter_mm=None,
                ),
                sharpness_score=80.0,
            )
        ]
        med, _ = median_measurement(samples)
        assert med.diameter_mm is None


# ---------------------------------------------------------------------------
# CentroidTracker — one ID per billet
# ---------------------------------------------------------------------------


class TestCentroidTracker:
    def test_single_billet_emits_one_record(self):
        """A billet sweeping entry→exit should produce exactly one TrackedBillet."""
        tracker = _tracker(entry_x=100, exit_x=1100)
        emitted: list[TrackedBillet] = []

        for x in range(50, 1200, 50):
            emitted.extend(tracker.update([_detection(float(x))]))

        assert len(emitted) == 1
        assert emitted[0].track_id == 1

    def test_two_billets_emit_two_records(self):
        """Two billets with different y-coordinates should get separate IDs."""
        tracker = _tracker()
        emitted: list[TrackedBillet] = []

        # Simulate two billets moving left-to-right with different y
        for x in range(50, 1200, 50):
            dets = [
                _detection(float(x), cy=200.0),
                _detection(float(x), cy=500.0),
            ]
            emitted.extend(tracker.update(dets))

        assert len(emitted) == 2
        ids = {b.track_id for b in emitted}
        assert len(ids) == 2, "Two billets must get different track IDs"

    def test_no_duplicate_records_for_one_billet(self):
        """A billet should never produce more than one record."""
        tracker = _tracker()
        emitted: list[TrackedBillet] = []
        for x in range(0, 1300, 30):
            emitted.extend(tracker.update([_detection(float(x))]))
        assert len(emitted) == 1

    def test_timeout_emits_record(self):
        """A billet that disappears before the exit line should emit via timeout."""
        tracker = _tracker(max_lost_frames=3)
        emitted: list[TrackedBillet] = []

        # Move through entry but stop before exit
        for x in range(100, 600, 50):
            emitted.extend(tracker.update([_detection(float(x))]))

        # Feed empty frames to trigger timeout
        for _ in range(5):
            emitted.extend(tracker.update([]))

        assert len(emitted) == 1

    def test_flush_emits_remaining_tracks(self):
        """flush() should finalise tracks still in-progress."""
        tracker = _tracker()
        emitted: list[TrackedBillet] = []

        for x in range(100, 600, 50):
            emitted.extend(tracker.update([_detection(float(x))]))

        # No exit yet — flush should emit the in-progress track
        flushed = tracker.flush()
        assert len(flushed) == 1
        assert len(emitted) == 0  # nothing exited naturally

    def test_record_has_frame_count(self):
        tracker = _tracker()
        emitted: list[TrackedBillet] = []
        xs = list(range(50, 1200, 50))
        for x in xs:
            emitted.extend(tracker.update([_detection(float(x))]))
        assert emitted[0].frame_count >= 1

    def test_measurement_is_median_not_raw(self):
        """Median measurement should be used, not the last raw frame's value."""
        tracker = _tracker(best_n=3)
        emitted: list[TrackedBillet] = []

        # Vary width per frame; median of widths 120,130,130,140,100 (best 3 by sharpness)
        # sharpness is uniform here so best_n picks first 3 → median([130,130,130]) = 130
        for i, x in enumerate(range(50, 1200, 50)):
            widths = [130.0, 130.0, 140.0, 120.0, 100.0]
            w = widths[i % len(widths)]
            emitted.extend(tracker.update([_detection(float(x), width=w)]))

        assert len(emitted) == 1
        # Median over top-3 sharpest (all equal sharpness, so first 3 frames)
        # widths: 130,130,140 → median=130
        assert emitted[0].measurement.width_mm == pytest.approx(130.0, abs=20.0)

    def test_pending_billet_before_entry_line_not_emitted(self):
        """A billet that never crosses the entry line should not emit a record."""
        tracker = _tracker(entry_x=500, exit_x=1100)
        emitted: list[TrackedBillet] = []

        # Never reaches entry_x=500
        for x in range(50, 400, 50):
            emitted.extend(tracker.update([_detection(float(x))]))

        # Expire by timeout
        for _ in range(10):
            emitted.extend(tracker.update([]))

        assert len(emitted) == 0, "Pre-entry billet should not produce a record"

    def test_reset_clears_state(self):
        tracker = _tracker()
        for x in range(50, 600, 50):
            tracker.update([_detection(float(x))])
        tracker.reset()
        assert tracker._tracks == {}
        assert tracker._next_id == 1

    def test_right_to_left_direction(self):
        """Billets moving right-to-left trigger on reversed thresholds."""
        tracker = CentroidTracker(
            entry_x=1100,
            exit_x=100,
            max_distance_px=80.0,
            max_lost_frames=5,
            best_n=3,
            direction="right_to_left",
        )
        emitted: list[TrackedBillet] = []
        for x in range(1150, 50, -50):
            emitted.extend(tracker.update([_detection(float(x))]))
        assert len(emitted) == 1
