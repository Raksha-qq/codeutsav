"""CSV append writer for continuous billet inspection records."""
import csv
from pathlib import Path
from typing import Dict, Any
from billetvision.logging_.db import RECORD_COLUMNS

def init_csv(csv_path: str | Path) -> None:
    """Ensure CSV exists with standard header."""
    p = Path(csv_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists() or p.stat().st_size == 0:
        with open(p, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(RECORD_COLUMNS)

def append_to_csv(csv_path: str | Path, record: Dict[str, Any]) -> None:
    """Append single record row to CSV."""
    init_csv(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([record.get(col, "") for col in RECORD_COLUMNS])
