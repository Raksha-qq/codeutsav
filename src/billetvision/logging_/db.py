"""SQLite source-of-truth database for billet inspection logs.

All writes must come from the single LogWriter thread.
Read helpers (count_records, fetch_recent) open their own short-lived
connections and are safe to call from any thread (WAL mode).
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

RECORD_COLUMNS = [
    "timestamp", "billet_seq", "billet_id", "batch_id",
    "length_mm", "width_mm", "height_mm", "diameter_mm",
    "ovality", "diag_diff_mm", "defects", "ocr_confidence",
    "status", "fail_reasons", "image_path", "processing_ms",
]

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS records (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT    NOT NULL,
    billet_seq      INTEGER NOT NULL,
    billet_id       TEXT    NOT NULL,
    batch_id        TEXT    NOT NULL,
    length_mm       REAL,
    width_mm        REAL,
    height_mm       REAL,
    diameter_mm     REAL,
    ovality         REAL,
    diag_diff_mm    REAL,
    defects         TEXT,
    ocr_confidence  REAL,
    status          TEXT    NOT NULL,
    fail_reasons    TEXT,
    image_path      TEXT,
    processing_ms   REAL
)
"""

_INSERT = """
INSERT INTO records
    (timestamp, billet_seq, billet_id, batch_id,
     length_mm, width_mm, height_mm, diameter_mm,
     ovality, diag_diff_mm, defects, ocr_confidence,
     status, fail_reasons, image_path, processing_ms)
VALUES
    (:timestamp, :billet_seq, :billet_id, :batch_id,
     :length_mm, :width_mm, :height_mm, :diameter_mm,
     :ovality, :diag_diff_mm, :defects, :ocr_confidence,
     :status, :fail_reasons, :image_path, :processing_ms)
"""


@dataclass
class InspectionRecord:
    """One billet inspection result.  All numeric fields are millimetres."""

    timestamp: str          # ISO-8601 UTC
    billet_seq: int
    billet_id: str
    batch_id: str
    length_mm: float
    width_mm: float
    height_mm: float
    diameter_mm: Optional[float]
    ovality: Optional[float]
    diag_diff_mm: Optional[float]
    defects: str            # comma-separated defect labels
    ocr_confidence: float
    status: str             # PASS | FAIL | REWORK | REVIEW
    fail_reasons: str       # semicolon-separated reason strings
    image_path: str
    processing_ms: float

    def to_dict(self) -> Dict[str, Any]:
        """Return a dict keyed by RECORD_COLUMNS (excludes the DB auto-id)."""
        return asdict(self)

    def to_row(self) -> list:
        """Return values in RECORD_COLUMNS order for CSV writing."""
        d = self.to_dict()
        return [d.get(col, "") for col in RECORD_COLUMNS]

    @classmethod
    def make(
        cls,
        billet_seq: int,
        billet_id: str = "UNKNOWN",
        batch_id: str = "",
        length_mm: float = 0.0,
        width_mm: float = 0.0,
        height_mm: float = 0.0,
        diameter_mm: Optional[float] = None,
        ovality: Optional[float] = None,
        diag_diff_mm: Optional[float] = None,
        defects: str = "",
        ocr_confidence: float = 0.0,
        status: str = "PASS",
        fail_reasons: str = "",
        image_path: str = "",
        processing_ms: float = 0.0,
    ) -> "InspectionRecord":
        """Convenience constructor that fills timestamp automatically."""
        ts = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        return cls(
            timestamp=ts,
            billet_seq=billet_seq,
            billet_id=billet_id,
            batch_id=batch_id,
            length_mm=length_mm,
            width_mm=width_mm,
            height_mm=height_mm,
            diameter_mm=diameter_mm,
            ovality=ovality,
            diag_diff_mm=diag_diff_mm,
            defects=defects,
            ocr_confidence=ocr_confidence,
            status=status,
            fail_reasons=fail_reasons,
            image_path=image_path,
            processing_ms=processing_ms,
        )


# ---------------------------------------------------------------------------
# Schema management
# ---------------------------------------------------------------------------

def init_db(db_path: str | Path) -> None:
    """Create the records table and enable WAL mode."""
    p = Path(db_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(p)) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(_CREATE_TABLE)
        conn.commit()


def open_writer_connection(db_path: str | Path) -> sqlite3.Connection:
    """Open a long-lived write connection for the writer thread.

    Caller is responsible for calling conn.close() when done.
    WAL mode is set so concurrent readers are not blocked.
    """
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


# ---------------------------------------------------------------------------
# Write (called from the single writer thread only)
# ---------------------------------------------------------------------------

def insert_record(conn: sqlite3.Connection, record: InspectionRecord) -> None:
    """Insert one InspectionRecord using an existing open connection.

    Does NOT commit — caller must commit after inserting (batching commits
    is more efficient for high throughput).
    """
    conn.execute(_INSERT, record.to_dict())


def commit(conn: sqlite3.Connection) -> None:
    conn.commit()


# ---------------------------------------------------------------------------
# Read helpers (any thread)
# ---------------------------------------------------------------------------

def count_records(db_path: str | Path) -> int:
    """Return total row count in the records table."""
    with sqlite3.connect(str(db_path)) as conn:
        row = conn.execute("SELECT COUNT(*) FROM records").fetchone()
        return row[0] if row else 0


def fetch_recent(
    db_path: str | Path,
    n: Optional[int] = 50,
    status: Optional[str] = None,
    q: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return the last ``n`` records as a list of dicts, newest first.

    Args:
        db_path: Path to the SQLite file.
        n: Maximum number of rows to return (None = all rows).
        status: Optional status filter (PASS | FAIL | REWORK | REVIEW).
        q: Optional case-insensitive substring filter on billet_id / batch_id
            (heat/batch lookup).
    """
    clauses: List[str] = []
    params: List[Any] = []
    if status:
        clauses.append("status=?")
        params.append(status)
    if q:
        clauses.append("(billet_id LIKE ? OR batch_id LIKE ?)")
        params.extend([f"%{q}%", f"%{q}%"])
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    limit = "LIMIT ?" if n is not None else ""
    if n is not None:
        params.append(n)
    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"SELECT * FROM records {where} ORDER BY id DESC {limit}", params
        ).fetchall()
        return [dict(r) for r in rows]


def fetch_record(db_path: str | Path, billet_seq: int) -> Optional[Dict[str, Any]]:
    """Return the newest record with this ``billet_seq``, or None."""
    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM records WHERE billet_seq=? ORDER BY id DESC LIMIT 1",
            (billet_seq,),
        ).fetchone()
        return dict(row) if row else None


def max_billet_seq(db_path: str | Path) -> int:
    """Highest billet_seq stored (0 when the table is empty or missing)."""
    try:
        with sqlite3.connect(str(db_path)) as conn:
            row = conn.execute("SELECT MAX(billet_seq) FROM records").fetchone()
            return int(row[0]) if row and row[0] is not None else 0
    except sqlite3.Error:
        return 0


def billet_id_exists(db_path: str | Path, billet_id: str) -> bool:
    """True if a record with this exact billet_id is already logged."""
    try:
        with sqlite3.connect(str(db_path)) as conn:
            row = conn.execute(
                "SELECT 1 FROM records WHERE billet_id=? LIMIT 1", (billet_id,)
            ).fetchone()
            return row is not None
    except sqlite3.Error:
        return False


_UPDATABLE = ("billet_id", "status", "fail_reasons", "ocr_confidence", "defects")


def update_record(
    db_path: str | Path, billet_seq: int, fields: Dict[str, Any]
) -> bool:
    """Update whitelisted columns of the record with ``billet_seq``.

    Returns True if a row was updated.  Unknown column names raise ValueError
    (never interpolated into SQL).
    """
    bad = set(fields) - set(_UPDATABLE)
    if bad:
        raise ValueError(f"Cannot update columns: {sorted(bad)}")
    if not fields:
        return False
    assignments = ", ".join(f"{k}=?" for k in fields)
    with sqlite3.connect(str(db_path)) as conn:
        cur = conn.execute(
            f"UPDATE records SET {assignments} WHERE billet_seq=?",
            [*fields.values(), billet_seq],
        )
        conn.commit()
        return cur.rowcount > 0


def update_record_status(
    db_path: str | Path,
    billet_seq: int,
    new_status: str,
    fail_reasons: Optional[str] = None,
) -> bool:
    """Update the status (and optionally reasons) of a record by billet_seq.

    Returns True if a row was updated, False if billet_seq was not found.
    """
    with sqlite3.connect(str(db_path)) as conn:
        if fail_reasons is not None:
            cur = conn.execute(
                "UPDATE records SET status=?, fail_reasons=? WHERE billet_seq=?",
                (new_status, fail_reasons, billet_seq),
            )
        else:
            cur = conn.execute(
                "UPDATE records SET status=? WHERE billet_seq=?",
                (new_status, billet_seq),
            )
        conn.commit()
        return cur.rowcount > 0
