"""Tests: alert dispatch/debounce, CSV lock fallback, daily rotation, review write-back."""
from __future__ import annotations

import threading
import time
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from billetvision.alerts.manager import AlertManager
from billetvision.alerts.webhook import build_notifiers, send_webhook
from billetvision.logging_.csv_writer import append_to_csv, read_csv_rows
from billetvision.logging_.db import (
    InspectionRecord,
    billet_id_exists,
    fetch_record,
    fetch_recent,
    max_billet_seq,
)
from billetvision.logging_.rotation import archive_path, rotate_file
from billetvision.logging_.writer import LogWriter


def _rec(seq: int, status="PASS", billet_id=None) -> InspectionRecord:
    return InspectionRecord.make(billet_seq=seq, billet_id=billet_id or f"H{seq:06d}",
                                 batch_id="B", status=status, defects="")


# ------------------------------ alerts -------------------------------

def test_notifiers_run_off_thread_and_failures_are_contained():
    got, done = [], threading.Event()
    main = threading.current_thread()

    def good(alert):
        got.append((alert.status, threading.current_thread() is main))
        done.set()

    def bad(alert):
        raise RuntimeError("endpoint down")

    mgr = AlertManager(notifiers=[bad, good])
    mgr.trigger("H1", "FAIL", ["width out of range"])
    assert done.wait(2)
    assert got == [("FAIL", False)]
    mgr.close()


def test_system_alerts_debounced_but_billet_alerts_are_not():
    mgr = AlertManager(debounce_s=60)
    a = mgr.trigger("-", "CAMERA_LOST", ["no frames"], kind="camera")
    b = mgr.trigger("-", "CAMERA_LOST", ["no frames"], kind="camera")
    assert a is b and len(mgr.history) == 1
    mgr.trigger("H1", "FAIL", ["x"])
    mgr.trigger("H1", "FAIL", ["x"])
    assert len(mgr.history) == 3
    assert [x.alert_id for x in mgr.history] == [1, 2, 3]


def test_webhook_inert_without_url_and_posts_when_configured(monkeypatch):
    monkeypatch.delenv("BILLETVISION_WEBHOOK_URL", raising=False)
    alert = AlertManager().trigger("H1", "FAIL", ["r"])
    with patch("billetvision.alerts.webhook.requests.post") as post:
        assert send_webhook(alert) is False
        post.assert_not_called()
        post.return_value.status_code = 200
        assert send_webhook(alert, "http://example.invalid/hook") is True
        assert post.call_args.kwargs["json"]["status"] == "FAIL"
    assert len(build_notifiers({})) == 1
    assert len(build_notifiers({"webhook_url": "http://x"})) == 2


# --------------------------- csv / rotation --------------------------

def test_csv_append_falls_back_to_part_file_when_locked(tmp_path, monkeypatch):
    path = tmp_path / "log.csv"
    append_to_csv(path, _rec(1))
    real_open = open

    def locked(file, mode="r", *a, **k):
        if Path(file) == path and "a" in mode:
            raise PermissionError("locked by Excel")
        return real_open(file, mode, *a, **k)

    monkeypatch.setattr("builtins.open", locked)
    written = append_to_csv(path, _rec(2))
    monkeypatch.undo()
    assert written != path and "part" in written.name
    assert len(read_csv_rows(path)) == 1 and len(read_csv_rows(written)) == 1


def test_rotate_file_archives_and_never_overwrites(tmp_path):
    p = tmp_path / "log.csv"
    p.write_text("a\n1\n")
    day = date(2026, 10, 2)
    first = rotate_file(p, day)
    assert first == archive_path(p, day) and not p.exists()
    p.write_text("a\n2\n")
    second = rotate_file(p, day)
    assert second != first and first.exists() and second.exists()
    assert rotate_file(p, day) is None  # nothing left to rotate


def test_writer_rotates_when_date_changes(tmp_path):
    today = {"d": date(2026, 10, 2)}
    w = LogWriter(tmp_path / "b.db", tmp_path / "b.csv", tmp_path / "b.xlsx", clock=lambda: today["d"])
    w.start()
    w.submit(_rec(1))
    time.sleep(0.5)
    today["d"] += timedelta(days=1)
    w.submit(_rec(2))
    w.stop()
    archive = archive_path(tmp_path / "b.csv", date(2026, 10, 2))
    assert [r["billet_seq"] for r in read_csv_rows(archive)] == ["1"]
    assert [r["billet_seq"] for r in read_csv_rows(tmp_path / "b.csv")] == ["2"]
    assert len(fetch_recent(tmp_path / "b.db", n=None)) == 2   # SQLite is never rotated


def test_writer_survives_locked_xlsx_and_loses_nothing(tmp_path):
    w = LogWriter(tmp_path / "b.db", tmp_path / "b.csv", tmp_path / "b.xlsx")
    w.start()
    with patch("billetvision.logging_.xlsx_writer.os.replace", side_effect=PermissionError("locked")):
        for i in range(1, 4):
            w.submit(_rec(i))
        w.stop()
    assert len(fetch_recent(tmp_path / "b.db", n=None)) == 3
    assert len(read_csv_rows(tmp_path / "b.csv")) == 3


# ----------------------------- review --------------------------------

def test_update_record_patches_db_csv_and_xlsx(tmp_path):
    w = LogWriter(tmp_path / "b.db", tmp_path / "b.csv", tmp_path / "b.xlsx")
    w.start()
    w.submit(_rec(1, status="REVIEW", billet_id="UNREAD-000001"))
    w.submit(_rec(2))
    time.sleep(0.5)
    assert w.update_record(1, {"billet_id": "H123456", "status": "PASS", "ocr_confidence": 1.0})
    assert not w.update_record(99, {"status": "PASS"})
    w.stop()
    row = fetch_record(tmp_path / "b.db", 1)
    assert row["billet_id"] == "H123456" and row["status"] == "PASS"
    csv_row = next(r for r in read_csv_rows(tmp_path / "b.csv") if r["billet_seq"] == "1")
    assert csv_row["billet_id"] == "H123456" and csv_row["status"] == "PASS"
    xl = pd.read_excel(tmp_path / "b.xlsx")
    assert xl.loc[xl.billet_seq == 1, "billet_id"].iloc[0] == "H123456"


def test_db_helpers_seq_and_duplicates(tmp_path):
    w = LogWriter(tmp_path / "b.db", tmp_path / "b.csv", tmp_path / "b.xlsx")
    w.start()
    w.submit(_rec(7, billet_id="H000007"))
    w.stop()
    assert max_billet_seq(tmp_path / "b.db") == 7
    assert billet_id_exists(tmp_path / "b.db", "H000007")
    assert not billet_id_exists(tmp_path / "b.db", "H000008")
    assert max_billet_seq(tmp_path / "missing.db") == 0
    assert [r["billet_id"] for r in fetch_recent(tmp_path / "b.db", q="0007")] == ["H000007"]
