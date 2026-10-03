"""CSV append writer — always open in append mode, never rewrites the file."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict

from billetvision.logging_.db import RECORD_COLUMNS, InspectionRecord


def init_csv(csv_path: str | Path) -> None:
    """Create the file with a header row if it does not exist or is empty."""
    p = Path(csv_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists() or p.stat().st_size == 0:
        with p.open("w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(RECORD_COLUMNS)


def append_to_csv(csv_path: str | Path, record: InspectionRecord) -> None:
    """Append one record row.  Thread-safe only if called from a single thread."""
    init_csv(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow(record.to_row())


def count_csv_rows(csv_path: str | Path) -> int:
    """Count data rows (excluding the header) in an existing CSV file."""
    p = Path(csv_path)
    if not p.exists():
        return 0
    with p.open(newline="", encoding="utf-8") as fh:
        # subtract 1 for the header row
        return max(0, sum(1 for _ in fh) - 1)
