"""Atomic XLSX writer with Windows PermissionError fallback to part files."""
import os
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import pandas as pd
from billetvision.logging_.db import RECORD_COLUMNS

class XlsxLogWriter:
    """Safely writes logs to Excel with atomic replace and fallback if locked by Excel."""

    def __init__(self, target_path: str | Path):
        self.target_path = Path(target_path)
        self.target_path.parent.mkdir(parents=True, exist_ok=True)
        self.records: List[Dict[str, Any]] = []
        self.active_file = self.target_path
        self.part_counter = 1
        self.last_status = "HEALTHY"

    def append(self, record: Dict[str, Any]) -> None:
        """Add record and flush atomically to Excel."""
        self.records.append(record)
        self._flush()

    def _flush(self) -> None:
        df = pd.DataFrame(self.records, columns=RECORD_COLUMNS)
        temp_fd, temp_file = tempfile.mkstemp(suffix=".xlsx", dir=self.target_path.parent)
        os.close(temp_fd)

        try:
            df.to_excel(temp_file, index=False, engine="openpyxl")
            # Try atomic replace
            try:
                os.replace(temp_file, self.target_path)
                self.active_file = self.target_path
                self.last_status = "HEALTHY"
            except PermissionError:
                # File is open in Excel, fallback to part file
                self.part_counter += 1
                today_str = datetime.now().strftime("%Y%m%d")
                fallback_path = self.target_path.parent / f"log_{today_str}_part{self.part_counter}.xlsx"
                os.replace(temp_file, fallback_path)
                self.active_file = fallback_path
                self.last_status = f"FALLBACK_PART_{self.part_counter}"
        except Exception as e:
            self.last_status = f"ERROR: {str(e)}"
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except OSError:
                    pass
