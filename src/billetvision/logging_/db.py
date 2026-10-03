"""SQLite source-of-truth database for billet inspection logs."""
import sqlite3
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

RECORD_COLUMNS = [
    "timestamp", "billet_seq", "billet_id", "batch_id", "length_mm",
    "width_mm", "height_mm", "diameter_mm", "ovality", "diag_diff_mm",
    "defects", "ocr_confidence", "status", "fail_reasons", "image_path", "processing_ms"
]

@dataclass
class InspectionRecord:
    timestamp: str
    billet_seq: int
    billet_id: str
    batch_id: str
    length_mm: float
    width_mm: float
    height_mm: float
    diameter_mm: Optional[float]
    ovality: Optional[float]
    diag_diff_mm: Optional[float]
    defects: str
    ocr_confidence: float
    status: str # PASS, FAIL, REWORK, REVIEW
    fail_reasons: str
    image_path: str
    processing_ms: float

def init_db(db_path: str | Path) -> None:
    """Initialize SQLite database with records table."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                billet_seq INTEGER NOT NULL,
                billet_id TEXT NOT NULL,
                batch_id TEXT NOT NULL,
                length_mm REAL,
                width_mm REAL,
                height_mm REAL,
                diameter_mm REAL,
                ovality REAL,
                diag_diff_mm REAL,
                defects TEXT,
                ocr_confidence REAL,
                status TEXT NOT NULL,
                fail_reasons TEXT,
                image_path TEXT,
                processing_ms REAL
            )
        """)
        conn.commit()
