"""Tests for defects, segmentation hardening, belt-speed length, tracker extras, synthetic scenes."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from billetvision import synthetic as S
from billetvision.vision.defects import (
    classify_defects,
    edge_irregularity_px,
    surface_anomaly_score,
)
from billetvision.vision.length import length_from_belt_speed
from billetvision.vision.measure import Measurement, measure
from billetvision.vision.preprocess import auto_gamma
from billetvision.vision.segment import BackgroundModel, column_envelope, segment
from billetvision.vision.track import CentroidTracker, FrameSample, top_samples


def _bar_frame(w=640, h=360, rect=(100, 120, 400, 100), bg=30, fg=170):
    img = np.full((h, w, 3), bg, dtype=np.uint8)
    x, y, rw, rh = rect
    cv2.rectangle(img, (x, y), (x + rw - 1, y + rh - 1), (fg, fg, fg), -1)
    return img


# --------------------------- defects ---------------------------------

def test_surface_anomaly_clean_vs_spotted():
    gray = np.full((200, 300), 150, dtype=np.uint8)
    mask = np.full_like(gray, 255)
    assert surface_anomaly_score(gray, mask) < 0.01
    spotted = gray.copy()
    for x in range(40, 260, 30):  # pits/stains much smaller than the median window
        cv2.rectangle(spotted, (x, 90), (x + 11, 101), 40, -1)
    assert surface_anomaly_score(spotted, mask) > 0.01


def test_edge_irregularity_rect_vs_notched():
    img = np.zeros((300, 600), dtype=np.uint8)
    cv2.rectangle(img, (50, 80), (549, 219), 255, -1)
    clean = max(cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea)
    assert edge_irregularity_px(clean) < 1.5
    # bite a large wavy chunk out of the top edge
    for x in range(100, 500, 40):
        cv2.rectangle(img, (x, 80), (x + 20, 100), 0, -1)
    rough = max(cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea)
    assert edge_irregularity_px(rough) > 10


def test_classify_defects_labels_and_missing_keys():
    tol = {"max_camber_mm": 3.0, "max_surface_anomaly_score": 0.15}
    labels = classify_defects(camber_mm=5.0, cross_section_var_mm=99.0, surface_anomaly=0.3,
                              edge_irregularity=None, tol=tol)
    assert labels == ["bent", "surface_anomaly"]  # no cs/edge limits configured -> not flagged
    assert classify_defects(camber_mm=1.0, cross_section_var_mm=None, surface_anomaly=0.0,
                            edge_irregularity=None, tol=tol) == []


# --------------------------- segmentation ----------------------------

def test_segment_roi_ignores_outside_blob():
    img = _bar_frame()
    cv2.rectangle(img, (5, 5), (90, 60), (255, 255, 255), -1)  # HUD blob outside ROI
    seg = segment(img, roi=(100, 100, 600, 300))
    assert seg is not None
    x, y, w, h = seg.bounding_rect
    assert x >= 100 and y >= 100 and abs(w - 400) <= 2


def test_segment_empty_scene_returns_none():
    rng = np.random.default_rng(0)
    img = np.clip(30 + rng.normal(0, 2, (360, 640, 3)), 0, 255).astype(np.uint8)
    cv2.line(img, (50, 50), (600, 50), (70, 70, 70), 1)
    assert segment(img, roi=(20, 20, 620, 340)) is None


def test_segment_billet_covering_over_half_roi_not_inverted():
    img = _bar_frame(rect=(0, 110, 640, 160))  # 160/360 of the frame, clipped left/right
    seg = segment(img, roi=(0, 60, 640, 320))  # bar = 61% of ROI
    assert seg is not None
    assert abs(seg.bounding_rect[3] - 160) <= 2


def test_column_envelope_closes_notch_but_keeps_curvature():
    mask = np.zeros((200, 400), dtype=np.uint8)
    cv2.rectangle(mask, (20, 50), (379, 149), 255, -1)
    cv2.rectangle(mask, (20, 90), (60, 110), 0, -1)  # text notch opening on the left end
    c = max(cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea)
    filled = column_envelope(c)
    assert cv2.contourArea(filled) > cv2.contourArea(c)
    assert abs(cv2.contourArea(filled) - 359 * 99) < 50  # full body, notch closed


def test_background_model_fallback_does_not_crash():
    bg = BackgroundModel()
    empty = np.full((200, 300), 40, dtype=np.uint8)
    for _ in range(5):
        bg.apply(empty)
    assert bg.foreground is not None
    assert segment(np.full((200, 300, 3), 40, np.uint8), background=bg) is None


def test_auto_gamma_identity_for_normal_and_brightens_dark():
    assert auto_gamma(np.full((10, 10), 110, np.uint8)) == 1.0
    assert auto_gamma(np.full((10, 10), 30, np.uint8)) > 1.0


def test_measure_travel_axis_gives_across_belt_width_for_short_piece():
    c = max(cv2.findContours(
        cv2.rectangle(np.zeros((300, 600), np.uint8), (100, 50), (199, 249), 255, -1),
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea)  # 100 wide (x), 200 tall (y)
    m = measure(c, 1.0, travel_axis="x")
    assert m.length_mm == pytest.approx(100, abs=1.5)
    assert m.width_mm == pytest.approx(200, abs=1.5)


# --------------------------- length ----------------------------------

def _timeline(length_px, speed_px_s, fps=15, line=500, start=-3000):
    out = []
    for i in range(400):
        t = i / fps
        head = start + speed_px_s * t
        out.append((t, max(160.0, head - length_px), min(1240.0, head)))
    return out


def test_length_from_belt_speed_accurate():
    tl = _timeline(2000, 500.0)  # 500 px/s * 0.5 mm/px = 250 mm/s
    assert length_from_belt_speed(tl, 700, 250.0) == pytest.approx(1000.0, rel=0.01)


def test_length_unavailable_when_tail_never_crosses():
    tl = [(i / 15, 160.0, 100.0 + 20 * i) for i in range(30)]
    assert length_from_belt_speed(tl, 700, 250.0) is None
    assert length_from_belt_speed(tl[:1], 700, 250.0) is None
    assert length_from_belt_speed(tl, 700, 0.0) is None


# --------------------------- tracker extras --------------------------

def _meas():
    return Measurement(length_mm=1000, width_mm=130, height_mm=130)


def test_tracker_timeline_and_partial_frames():
    tr = CentroidTracker(entry_x=100, exit_x=500, max_distance_px=300, max_lost_frames=3, best_n=2)
    gray = np.zeros((100, 100), np.uint8)
    cnt = np.array([[[10, 10]], [[80, 10]], [[80, 80]], [[10, 80]]], dtype=np.int32)
    out = []
    for i, x in enumerate((150, 300, 520)):
        meas = None if i == 0 else _meas()  # first frame: timing only
        out += tr.update([((x, 50), cnt, gray, meas, {"x_range": (x - 50, x + 50)})], timestamp=i * 0.1)
    assert len(out) == 1
    assert len(out[0].timeline) == 3
    assert out[0].frame_count == 3


def test_tracker_drops_track_without_samples():
    tr = CentroidTracker(entry_x=100, exit_x=500, max_lost_frames=2)
    gray = np.zeros((10, 10), np.uint8)
    cnt = np.array([[[1, 1]], [[5, 1]], [[5, 5]]], dtype=np.int32)
    assert tr.update([((200, 5), cnt, gray, None)]) == []
    assert tr.update([]) == [] and tr.update([]) == []
    assert tr.flush() == []


def test_top_samples_prefers_complete_views():
    cnt = np.zeros((4, 1, 2), np.int32)
    g = np.zeros((4, 4), np.uint8)
    big = FrameSample(g, cnt, _meas(), 100.0, area_px=1000.0)
    tiny_sharp = FrameSample(g, cnt, _meas(), 900.0, area_px=100.0)
    assert top_samples([big, tiny_sharp], 1)[0] is big


# --------------------------- synthetic -------------------------------

def test_ground_truth_loader_and_split():
    props = S.load_ground_truth()
    assert len(props) == 12
    assert {p.shape for p in S.demo_props()} == {"square"}


def test_draw_bar_has_exact_subpixel_coverage():
    canvas = np.zeros((20, 40, 3), np.uint8)
    tex = np.full((10, 40, 3), 200, np.uint8)
    S.draw_bar(canvas, tex, 5.5, 25.0, 4.0, 14.0)
    row = canvas[8, :, 0].astype(int)
    assert row[5] == pytest.approx(100, abs=2)      # half-covered edge pixel
    assert row[10] == 200 and row[30] == 0
