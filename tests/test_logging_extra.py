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
