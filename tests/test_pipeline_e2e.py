"""End-to-end tests: synthetic belt through the real pipeline, API drill-down/review, camera loss."""
from __future__ import annotations

import json
import time
import unittest.mock
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from billetvision import synthetic as S
from billetvision.pipeline import BilletVisionPipeline, _deep_merge


def _overrides(out: Path) -> dict:
    return _deep_merge(S.PIPELINE_OVERRIDES, {"logging": {
        "db_path": str(out / "b.db"), "csv_path": str(out / "b.csv"),
        "xlsx_path": str(out / "b.xlsx"), "snapshot_dir": str(out / "snap"),
    }})


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    """Two props (one PASS, one out of tolerance) through the full pipeline."""
    out = tmp_path_factory.mktemp("e2e")
    props = [S.demo_props()[0], S.demo_props()[3]]  # PROP-01 (PASS), PROP-04 (131.8 wide)
    pipe = BilletVisionPipeline("config/config.yaml", overrides=_overrides(out))
    rows = pipe.run_offline(S.render_belt_frames(props, seed=3), fps=S.FPS)
    return {"pipe": pipe, "rows": rows, "props": props, "out": out}


def test_one_record_per_billet(run):
    assert len(run["rows"]) == 2
    assert [r["billet_seq"] for r in run["rows"]] == [1, 2]


def test_dimensions_within_one_percent(run):
    for prop, rec in zip(run["props"], run["rows"]):
        assert abs(rec["width_mm"] - prop.width_mm) / prop.width_mm < 0.01
        assert abs(rec["length_mm"] - prop.length_mm) / prop.length_mm < 0.01


def test_ids_and_verdicts(run):
    ok, bad = run["rows"]
    assert ok["billet_id"] == "H123456" and ok["status"] == "PASS" and ok["fail_reasons"] == ""
    assert bad["billet_id"] == "H123459" and bad["status"] == "FAIL"
    assert "width" in bad["fail_reasons"] and "130.00" in bad["fail_reasons"]


def test_latency_under_two_seconds_and_fps(run):
    assert all(r["processing_ms"] < 2000 for r in run["rows"])
    assert run["pipe"].offline_stats["mean_frame_ms"] < 1000 / 15


def test_artifacts_and_detail_json_written(run):
    snap = run["out"] / "snap"
    for name in ("000001.jpg", "000001_detail.json", "000002.jpg"):
        assert (snap / name).is_file()
    detail = json.loads((snap / "000002_detail.json").read_text())
    assert detail["status"] == "FAIL" and detail["frames"] and detail["tolerances"]["width_nominal_mm"] == 130.0
    assert detail["ocr"]["text"] == "H123459"


def test_calibration_recovered_from_marker(run):
    assert run["pipe"].get_calibration()["mm_per_px"] == pytest.approx(0.5, rel=0.01)


def test_rerun_continues_sequence_and_flags_duplicate(run):
    """A restarted pipeline must not reuse seq numbers, and re-seeing an ID is flagged."""
    props = [S.demo_props()[0]]
    pipe = BilletVisionPipeline("config/config.yaml", overrides=_overrides(run["out"]))
    rows = pipe.run_offline(S.render_belt_frames(props, seed=4), fps=S.FPS)
    assert [r["billet_seq"] for r in rows] == [3]
    assert "duplicate_id" in rows[0]["defects"]
    assert any(a.kind == "warning" for a in pipe.alert_manager.history)


# ------------------------------ API ----------------------------------

@pytest.fixture()
def api(run):
    from billetvision.api import main as api_main
    db = str(run["out"] / "b.db")
    with unittest.mock.patch.object(api_main, "_log_cfg", return_value={"db_path": db, "snapshot_dir": str(run["out"] / "snap")}), \
         unittest.mock.patch.object(api_main.pipeline, "start"), unittest.mock.patch.object(api_main.pipeline, "stop"), \
         unittest.mock.patch.object(type(api_main.pipeline), "snapshot_dir", new=property(lambda self: run["out"] / "snap")), \
         unittest.mock.patch.object(type(api_main.pipeline), "is_running", new=property(lambda self: True)):
        with TestClient(api_main.app) as client:
            yield client


def test_api_log_has_image_url_and_reasons_and_heat_filter(api):
    rows = api.get("/api/log?q=H123459").json()
    assert len(rows) == 1 and rows[0]["image_url"] == "/snapshots/000002.jpg"
    assert rows[0]["reasons"] and rows[0]["status"] == "FAIL"
    assert api.get("/api/log?q=NOPE").json() == []


def test_api_billet_drilldown_and_snapshot_serving(api):
    body = api.get("/api/billet/2").json()
    assert body["record"]["billet_id"] == "H123459"
    assert body["detail"]["measurement"]["width_mm"] > 131
    assert api.get("/api/billet/999").status_code == 404
    assert api.get("/snapshots/000002.jpg").headers["content-type"] == "image/jpeg"
    assert api.get("/snapshots/..%2Fb.db").status_code in (400, 404)
    assert api.get("/snapshots/nope.jpg").status_code == 404


def test_api_json_export(api):
    resp = api.get("/api/export/json")
    assert resp.status_code == 200 and len(resp.json()) >= 2


def test_api_tolerance_validation(api):
    assert api.put("/api/tolerances/square_130", json={"updates": {"width_tol_mm": "wide"}}).status_code == 400
    assert api.put("/api/tolerances/square_130", json={"updates": {"width_tol_mm": -1}}).status_code == 400
    assert api.put("/api/tolerances/square_130", json={"updates": {"shape": "round"}}).status_code == 400


def test_api_review_corrected_id_roundtrip(api, run):
    from billetvision.logging_.db import InspectionRecord, fetch_record, insert_record, open_writer_connection, commit
    db = run["out"] / "b.db"
    conn = open_writer_connection(db)
    insert_record(conn, InspectionRecord.make(billet_seq=500, billet_id="UNREAD-000500", status="REVIEW", defects="x"))
    commit(conn)
    conn.close()
    assert api.post("/api/review/500", json={"action": "approve", "corrected_id": "bad id"}).status_code == 422
    resp = api.post("/api/review/500", json={"action": "approve", "corrected_id": "h123499"})
    assert resp.status_code == 200 and resp.json()["billet_id"] == "H123499"
    row = fetch_record(db, 500)
    assert row["status"] == "PASS" and row["billet_id"] == "H123499" and "manual_id" in row["defects"]
    assert row["ocr_confidence"] == 1.0


def test_api_recalibrate_errors(api):
    from billetvision.api import main as api_main
    with unittest.mock.patch.object(api_main.pipeline, "recalibrate", side_effect=ValueError("no marker")):
        assert api.post("/api/calibration/recalibrate", json={}).status_code == 422
    with unittest.mock.patch.object(api_main.pipeline, "recalibrate", side_effect=RuntimeError("no frame")):
        assert api.post("/api/calibration/recalibrate", json={}).status_code == 409
    assert api.post("/api/calibration/recalibrate", json={"marker_size_mm": -5}).status_code == 400


def test_recalibrate_uses_latest_frame(run):
    pipe = run["pipe"]
    pipe._last_raw = next(S.render_belt_frames([S.demo_props()[0]]))
    pipe._cal_path = run["out"] / "cal.json"
    info = pipe.recalibrate()
    assert info["mm_per_px"] == pytest.approx(0.5, rel=0.01) and pipe._cal_path.exists()


# --------------------------- camera loss -----------------------------

class _StubSource:
    mode = "webcam"
    fps = 15.0
    def is_alive(self):
        return False

    def __init__(self):
        self.stopped = False
        self.released = False

    def stop(self):
        self.stopped = True

    def release(self):
        self.stopped = self.released = True

    def start(self):
        return self


def test_camera_lost_raises_alert_and_reconnects_then_recovers(tmp_path):
    pipe = BilletVisionPipeline("config/config.yaml", overrides=_overrides(tmp_path))
    pipe._load_config()
    pipe._init_components(start_source=False)
    stub = _StubSource()
    pipe._source = stub
    pipe._make_source = lambda: _StubSource()  # type: ignore[method-assign]
    try:
        last = pipe._on_no_frame(now=100.0, last_frame_t=90.0, last_reconnect=0.0)
        assert last == 100.0 and not pipe.camera_ok and stub.released   # the old device handle is freed
        lost = [a for a in pipe.alert_manager.history if a.status == "CAMERA_LOST"]
        assert len(lost) == 1 and lost[0].kind == "camera"
        pipe._set_camera(True)
        assert pipe.camera_ok
        assert pipe.alert_manager.history[-1].status == "CAMERA_RESTORED"
        assert not pipe.alert_manager.history[-1].sound
    finally:
        pipe._writer.stop()
        pipe.alert_manager.close()


def test_short_gap_does_not_trigger_camera_lost(tmp_path):
    pipe = BilletVisionPipeline("config/config.yaml", overrides=_overrides(tmp_path))
    pipe._load_config()
    pipe._init_components(start_source=False)
    pipe._source = _StubSource()
    try:
        pipe._on_no_frame(now=100.0, last_frame_t=99.0, last_reconnect=0.0)
        assert pipe.camera_ok
    finally:
        pipe._writer.stop()


def test_end_of_stream_detected_with_real_frame_source(tmp_path):
    """Non-looping file source: the pipeline must notice the end and flush, not wait forever."""
    from billetvision.capture.frame_source import FrameSource

    video = tmp_path / "short.mp4"
    S.write_video(iter([next(S.render_belt_frames([S.demo_props()[0]]))] * 6), video)
    pipe = BilletVisionPipeline("config/config.yaml", overrides=_overrides(tmp_path) | {"capture": {"loop": False}})
    pipe._load_config()
    pipe._init_components(start_source=False)
    src = FrameSource(source=str(video), loop=False, fps=200).start()
    pipe._source = src
    try:
        while src.read(timeout=0.5)[0]:
            pass
        assert not src.is_alive()
        pipe._on_no_frame(now=time.monotonic(), last_frame_t=time.monotonic(), last_reconnect=0.0)
        assert pipe.ended and pipe.camera_ok
    finally:
        src.stop()
        pipe._writer.stop()


def test_direct_length_mode_measures_piece_inside_view(tmp_path):
    prop = S.PropSpec("SHORT", "square", 400.0, 130.0, 130.0, None, "H777777", "PASS")
    ov = _deep_merge(_overrides(tmp_path), {"vision": {"length_mode": "direct"}})
    pipe = BilletVisionPipeline("config/config.yaml", overrides=ov)
    rows = pipe.run_offline(S.render_belt_frames([prop], seed=5), fps=S.FPS)
    assert len(rows) == 1
    assert abs(rows[0]["length_mm"] - 400.0) / 400.0 < 0.01
    assert abs(rows[0]["width_mm"] - 130.0) / 130.0 < 0.01
    assert rows[0]["billet_id"] == "H777777"
