"""Tests for logging_/ : SQLite, CSV, XLSX, and the single-writer thread.

Covers:
- DB schema init and basic insert/read
- CSV init and append, count_csv_rows
- XLSX flush, atomic replace, PermissionError → part-file fallback
- LogWriter concurrency: many threads submit, zero rows lost
- LogWriter stop() drains queue cleanly
- LogWriter restart is a no-op
"""
from __future__ import annotations

import csv
import os
import queue
import sqlite3
import threading
import time
import unittest.mock
from pathlib import Path

import pytest

from billetvision.logging_.db import (
    InspectionRecord,
    RECORD_COLUMNS,
    commit,
    count_records,
    fetch_recent,
    init_db,
    insert_record,
    open_writer_connection,
)
from billetvision.logging_.csv_writer import append_to_csv, count_csv_rows, init_csv
from billetvision.logging_.xlsx_writer import XlsxLogWriter
from billetvision.logging_.writer import LogWriter

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _rec(seq: int = 1, status: str = "PASS") -> InspectionRecord:
    return InspectionRecord.make(
        billet_seq=seq,
        billet_id=f"A{seq:06d}",
        batch_id="BATCH01",
        length_mm=1000.0,
        width_mm=130.0,
        height_mm=130.0,
        status=status,
        processing_ms=42.0,
    )


@pytest.fixture()
def tmp_dir(tmp_path: Path) -> Path:
    return tmp_path


# ---------------------------------------------------------------------------
# InspectionRecord
# ---------------------------------------------------------------------------


class TestInspectionRecord:
    def test_make_fills_timestamp(self):
        r = _rec()
        assert r.timestamp  # not empty
        assert "T" in r.timestamp  # ISO-8601

    def test_to_dict_keys(self):
        r = _rec()
        d = r.to_dict()
        for col in RECORD_COLUMNS:
            assert col in d

    def test_to_row_length(self):
        r = _rec()
        assert len(r.to_row()) == len(RECORD_COLUMNS)

    def test_to_row_order_matches_columns(self):
        r = _rec(seq=7)
        row = r.to_row()
        d = r.to_dict()
        for i, col in enumerate(RECORD_COLUMNS):
            assert row[i] == d[col]


# ---------------------------------------------------------------------------
# SQLite db.py
# ---------------------------------------------------------------------------


class TestDatabase:
    def test_init_creates_table(self, tmp_dir):
        db = tmp_dir / "test.db"
        init_db(db)
        with sqlite3.connect(str(db)) as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='records'"
            ).fetchall()
        assert len(rows) == 1

    def test_init_is_idempotent(self, tmp_dir):
        db = tmp_dir / "test.db"
        init_db(db)
        init_db(db)  # should not raise

    def test_insert_and_count(self, tmp_dir):
        db = tmp_dir / "test.db"
        init_db(db)
        conn = open_writer_connection(db)
        insert_record(conn, _rec(1))
        insert_record(conn, _rec(2))
        commit(conn)
        conn.close()
        assert count_records(db) == 2

    def test_fetch_recent_newest_first(self, tmp_dir):
        db = tmp_dir / "test.db"
        init_db(db)
        conn = open_writer_connection(db)
        for i in range(5):
            insert_record(conn, _rec(i))
        commit(conn)
        conn.close()
        rows = fetch_recent(db, n=3)
        assert len(rows) == 3
        # newest first → highest billet_seq first (inserted last)
        assert rows[0]["billet_seq"] > rows[-1]["billet_seq"]

    def test_fetch_recent_respects_limit(self, tmp_dir):
        db = tmp_dir / "test.db"
        init_db(db)
        conn = open_writer_connection(db)
        for i in range(20):
            insert_record(conn, _rec(i))
        commit(conn)
        conn.close()
        assert len(fetch_recent(db, n=5)) == 5

    def test_count_empty_db(self, tmp_dir):
        db = tmp_dir / "test.db"
        init_db(db)
        assert count_records(db) == 0


# ---------------------------------------------------------------------------
# CSV csv_writer.py
# ---------------------------------------------------------------------------


class TestCsvWriter:
    def test_init_creates_header(self, tmp_dir):
        csv_path = tmp_dir / "log.csv"
        init_csv(csv_path)
        with csv_path.open(newline="", encoding="utf-8") as fh:
            header = next(csv.reader(fh))
        assert header == RECORD_COLUMNS

    def test_init_idempotent(self, tmp_dir):
        csv_path = tmp_dir / "log.csv"
        init_csv(csv_path)
        init_csv(csv_path)
        assert count_csv_rows(csv_path) == 0

    def test_append_adds_row(self, tmp_dir):
        csv_path = tmp_dir / "log.csv"
        append_to_csv(csv_path, _rec(1))
        assert count_csv_rows(csv_path) == 1

    def test_append_multiple(self, tmp_dir):
        csv_path = tmp_dir / "log.csv"
        for i in range(10):
            append_to_csv(csv_path, _rec(i))
        assert count_csv_rows(csv_path) == 10

    def test_count_empty_file(self, tmp_dir):
        assert count_csv_rows(tmp_dir / "nonexistent.csv") == 0

    def test_values_round_trip(self, tmp_dir):
        csv_path = tmp_dir / "log.csv"
        r = _rec(42, status="FAIL")
        append_to_csv(csv_path, r)
        with csv_path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            row = next(reader)
        assert row["billet_seq"] == "42"
        assert row["status"] == "FAIL"
        assert row["billet_id"] == "A000042"


# ---------------------------------------------------------------------------
# XLSX xlsx_writer.py
# ---------------------------------------------------------------------------


class TestXlsxWriter:
    def test_flush_creates_file(self, tmp_dir):
        xlsx_path = tmp_dir / "log.xlsx"
        w = XlsxLogWriter(xlsx_path)
        w.append(_rec(1))
        w.flush()
        assert xlsx_path.exists()

    def test_flush_multiple_records(self, tmp_dir):
        import pandas as pd

        xlsx_path = tmp_dir / "log.xlsx"
        w = XlsxLogWriter(xlsx_path)
        for i in range(5):
            w.append(_rec(i))
        w.flush()
        df = pd.read_excel(xlsx_path)
        assert len(df) == 5

    def test_flush_noop_on_empty(self, tmp_dir):
        xlsx_path = tmp_dir / "log.xlsx"
        w = XlsxLogWriter(xlsx_path)
        w.flush()  # should not raise or create file
        assert not xlsx_path.exists()

    def test_pending_count(self, tmp_dir):
        xlsx_path = tmp_dir / "log.xlsx"
        w = XlsxLogWriter(xlsx_path)
        w.append(_rec(1))
        w.append(_rec(2))
        assert w.pending_count == 2
        w.flush()
        assert w.pending_count == 0

    def test_permission_error_creates_part_file(self, tmp_dir):
        xlsx_path = tmp_dir / "log.xlsx"
        w = XlsxLogWriter(xlsx_path)
        w.append(_rec(1))

        with unittest.mock.patch("os.replace") as mock_replace:
            # First call (temp → target) raises PermissionError;
            # second call (temp → part file) succeeds.
            mock_replace.side_effect = [PermissionError("locked"), None]
            w.flush()

        assert "FALLBACK" in w.last_status
        assert w._part_counter == 1

    def test_permission_error_status_healthy_on_success(self, tmp_dir):
        xlsx_path = tmp_dir / "log.xlsx"
        w = XlsxLogWriter(xlsx_path)
        w.append(_rec(1))
        w.flush()
        assert w.last_status == "HEALTHY"

    def test_loads_existing_records_on_init(self, tmp_dir):
        import pandas as pd

        xlsx_path = tmp_dir / "log.xlsx"
        # Write a file with 3 rows manually
        df = pd.DataFrame([_rec(i).to_dict() for i in range(3)], columns=RECORD_COLUMNS)
        df.to_excel(xlsx_path, index=False, engine="openpyxl")

        # New writer should load those 3 rows
        w = XlsxLogWriter(xlsx_path)
        assert len(w._records) == 3

    def test_extend_adds_multiple(self, tmp_dir):
        import pandas as pd

        xlsx_path = tmp_dir / "log.xlsx"
        w = XlsxLogWriter(xlsx_path)
        w.extend([_rec(i) for i in range(4)])
        w.flush()
        df = pd.read_excel(xlsx_path)
        assert len(df) == 4


# ---------------------------------------------------------------------------
# LogWriter — concurrency and correctness
# ---------------------------------------------------------------------------


class TestLogWriter:
    def test_start_stop_clean(self, tmp_dir):
        w = LogWriter(
            tmp_dir / "bv.db",
            tmp_dir / "bv.csv",
            tmp_dir / "bv.xlsx",
        )
        w.start()
        w.stop(timeout=5.0)
        assert not w._thread.is_alive()

    def test_single_record_written(self, tmp_dir):
        w = LogWriter(
            tmp_dir / "bv.db",
            tmp_dir / "bv.csv",
            tmp_dir / "bv.xlsx",
        )
        w.start()
        w.submit(_rec(1))
        w.stop(timeout=10.0)

        assert count_records(tmp_dir / "bv.db") == 1
        assert count_csv_rows(tmp_dir / "bv.csv") == 1

    def test_concurrency_zero_lost_rows(self, tmp_dir):
        """10 threads × 50 records = 500 total — all must reach the DB and CSV."""
        n_threads = 10
        n_per_thread = 50
        total = n_threads * n_per_thread

        w = LogWriter(
            tmp_dir / "bv.db",
            tmp_dir / "bv.csv",
            tmp_dir / "bv.xlsx",
        )
        w.start()

        seq_counter = [0]
        counter_lock = threading.Lock()

        def producer():
            for _ in range(n_per_thread):
                with counter_lock:
                    seq = seq_counter[0]
                    seq_counter[0] += 1
                w.submit(_rec(seq))

        threads = [threading.Thread(target=producer) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        w.stop(timeout=30.0)

        db_count = count_records(tmp_dir / "bv.db")
        csv_count = count_csv_rows(tmp_dir / "bv.csv")

        assert db_count == total, f"DB has {db_count}/{total} rows"
        assert csv_count == total, f"CSV has {csv_count}/{total} rows"
        assert w.written == total

    def test_stop_drains_queue(self, tmp_dir):
        """Records submitted just before stop() must not be lost."""
        w = LogWriter(
            tmp_dir / "bv.db",
            tmp_dir / "bv.csv",
            tmp_dir / "bv.xlsx",
        )
        w.start()
        # Flood the queue then stop immediately
        n = 200
        for i in range(n):
            w.submit(_rec(i))
        w.stop(timeout=30.0)

        assert count_records(tmp_dir / "bv.db") == n
        assert count_csv_rows(tmp_dir / "bv.csv") == n

    def test_double_start_is_noop(self, tmp_dir):
        w = LogWriter(
            tmp_dir / "bv.db",
            tmp_dir / "bv.csv",
            tmp_dir / "bv.xlsx",
        )
        w.start()
        thread_id = id(w._thread)
        w.start()  # second call should be a no-op
        assert id(w._thread) == thread_id
        w.stop()

    def test_submitted_counter(self, tmp_dir):
        w = LogWriter(
            tmp_dir / "bv.db",
            tmp_dir / "bv.csv",
            tmp_dir / "bv.xlsx",
        )
        w.start()
        for i in range(5):
            w.submit(_rec(i))
        w.stop(timeout=10.0)
        assert w.submitted == 5

    def test_xlsx_written_after_stop(self, tmp_dir):
        xlsx_path = tmp_dir / "bv.xlsx"
        w = LogWriter(tmp_dir / "bv.db", tmp_dir / "bv.csv", xlsx_path)
        w.start()
        for i in range(15):  # > XLSX_FLUSH_RECORDS threshold
            w.submit(_rec(i))
        w.stop(timeout=15.0)
        assert xlsx_path.exists()

    def test_permission_error_fallback_no_data_loss(self, tmp_dir):
        """PermissionError on XLSX must not lose DB or CSV records."""
        w = LogWriter(
            tmp_dir / "bv.db",
            tmp_dir / "bv.csv",
            tmp_dir / "bv.xlsx",
        )
        w.start()

        original_replace = os.replace

        def flaky_replace(src, dst):
            # Fail only when writing to the main xlsx target
            if str(dst) == str(tmp_dir / "bv.xlsx"):
                raise PermissionError("locked by Excel")
            original_replace(src, dst)

        with unittest.mock.patch("os.replace", side_effect=flaky_replace):
            for i in range(20):
                w.submit(_rec(i))
            w.stop(timeout=15.0)

        assert count_records(tmp_dir / "bv.db") == 20
        assert count_csv_rows(tmp_dir / "bv.csv") == 20
        # Part files should have been created
        part_files = list(tmp_dir.glob("bv_*_part*.xlsx"))
        assert len(part_files) > 0
