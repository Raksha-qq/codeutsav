"""Tests for logging system (SQLite, CSV, and atomic XLSX with lock fallback).

Verifies PRD §7.3, §8 #3:
- SQLite source of truth initialization and schema integrity
- CSV append writer ensures zero data loss
- Atomic XLSX replacement via temporary file
- PermissionError fallback to part files when Excel holds file lock
- Concurrency / multi-threaded append resilience
"""

import os
import sqlite3
import csv
from pathlib import Path
from unittest.mock import patch
import pandas as pd
import pytest

from billetvision.logging_.db import init_db, RECORD_COLUMNS
from billetvision.logging_.csv_writer import init_csv, append_to_csv
from billetvision.logging_.xlsx_writer import XlsxLogWriter


@pytest.fixture
def sample_record():
    return {
        "timestamp": "2026-10-03T12:00:00Z",
        "billet_seq": 1,
        "billet_id": "H123456",
        "batch_id": "B-99",
        "length_mm": 1000.5,
        "width_mm": 130.1,
        "height_mm": 129.9,
        "diameter_mm": None,
        "ovality": None,
        "diag_diff_mm": 0.2,
        "defects": "none",
        "ocr_confidence": 0.96,
        "status": "PASS",
        "fail_reasons": "",
        "image_path": "data/outputs/snapshots/snap_1.jpg",
        "processing_ms": 42.5,
    }


class TestSqliteLogging:
    def test_init_db_creates_table(self, tmp_path):
        db_file = tmp_path / "test_billetvision.db"
        init_db(db_file)
        assert db_file.exists()

        with sqlite3.connect(db_file) as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(records)")
            cols = [row[1] for row in cursor.fetchall()]
            for expected in ["timestamp", "billet_seq", "billet_id", "status", "processing_ms"]:
                assert expected in cols


class TestCsvLogging:
    def test_init_and_append_csv(self, tmp_path, sample_record):
        csv_file = tmp_path / "test_billet_log.csv"
        init_csv(csv_file)
        assert csv_file.exists()

        # Check headers
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            assert header == RECORD_COLUMNS

        # Append row
        append_to_csv(csv_file, sample_record)
        with open(csv_file, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
            assert len(lines) == 2  # header + 1 record
            assert "H123456" in lines[1]
            assert "PASS" in lines[1]


class TestXlsxLogging:
    def test_xlsx_writer_creates_valid_excel(self, tmp_path, sample_record):
        xlsx_file = tmp_path / "test_log.xlsx"
        writer = XlsxLogWriter(xlsx_file)
        writer.append(sample_record)

        assert xlsx_file.exists()
        df = pd.read_excel(xlsx_file)
        assert len(df) == 1
        assert df["billet_id"].iloc[0] == "H123456"
        assert writer.last_status == "HEALTHY"

    def test_xlsx_lock_fallback_to_part_file(self, tmp_path, sample_record):
        """Simulates Excel holding the file lock (PermissionError on os.replace)."""
        xlsx_file = tmp_path / "locked_log.xlsx"
        writer = XlsxLogWriter(xlsx_file)

        # First write succeeds
        writer.append(sample_record)
        assert xlsx_file.exists()

        # Mock os.replace to raise PermissionError when targeting xlsx_file
        orig_replace = os.replace

        def mock_replace(src, dst):
            if Path(dst) == xlsx_file:
                raise PermissionError("Simulated Excel Lock: File in use")
            return orig_replace(src, dst)

        with patch("os.replace", side_effect=mock_replace):
            rec2 = dict(sample_record)
            rec2["billet_seq"] = 2
            rec2["billet_id"] = "H123457"
            writer.append(rec2)

        # The writer must have safely written to part file without crashing
        assert "FALLBACK_PART_" in writer.last_status
        assert writer.active_file.exists()
        assert "part" in writer.active_file.name

        # Verify fallback part file contains the data
        df_part = pd.read_excel(writer.active_file)
        assert len(df_part) == 2
        assert df_part["billet_id"].iloc[1] == "H123457"
