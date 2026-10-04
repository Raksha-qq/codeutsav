"""CSV append writer — append-mode, never loses a row, never crashes on a lock."""
from __future__ import annotations

import csv
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from billetvision.logging_.db import RECORD_COLUMNS, InspectionRecord

logger = logging.getLogger(__name__)


def init_csv(csv_path: str | Path) -> None:
    """Create the file with a header row if it does not exist or is empty."""
    p = Path(csv_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists() or p.stat().st_size == 0:
        with p.open("w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(RECORD_COLUMNS)


def _part_path(csv_path: Path) -> Path:
    """Dated side file used while the main CSV is locked (e.g. open in Excel)."""
    return csv_path.with_name(f"{csv_path.stem}_{datetime.now().strftime('%Y%m%d')}_part{csv_path.suffix}")


def append_to_csv(csv_path: str | Path, record: InspectionRecord) -> Path:
    """Append one record row and return the file actually written.

    Thread-safe only if called from a single thread.  If the main file is
    locked (``PermissionError``) the row goes to a dated ``_part`` CSV so the
    record is never dropped and the caller never crashes.
    """
    p = Path(csv_path)
    try:
        init_csv(p)
        with open(p, "a", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(record.to_row())
        return p
    except PermissionError:
        part = _part_path(p)
        logger.warning("CSV %s is locked — appending to %s", p, part.name)
        init_csv(part)
        with open(part, "a", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(record.to_row())
        return part


def read_csv_rows(csv_path: str | Path) -> List[Dict[str, Any]]:
    """Read all data rows of a CSV as dicts (empty list if missing)."""
    p = Path(csv_path)
    if not p.exists():
        return []
    with p.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def rewrite_csv(csv_path: str | Path, rows: List[Dict[str, Any]]) -> bool:
    """Atomically replace the CSV with ``rows`` (temp file + ``os.replace``).

    Returns False (leaving the original untouched) if the file is locked.
    """
    p = Path(csv_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".csv", dir=p.parent)
    os.close(fd)
    try:
        with open(tmp, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=RECORD_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp, p)
        return True
    except PermissionError:
        logger.warning("CSV %s is locked — review update not applied to the CSV", p)
    except OSError as exc:
        logger.error("CSV rewrite failed: %s", exc)
    try:
        os.remove(tmp)
    except OSError:
        pass
    return False


def count_csv_rows(csv_path: str | Path) -> int:
    """Count data rows (excluding the header) in an existing CSV file."""
    p = Path(csv_path)
    if not p.exists():
        return 0
    with p.open(newline="", encoding="utf-8") as fh:
        # subtract 1 for the header row
        return max(0, sum(1 for _ in fh) - 1)
