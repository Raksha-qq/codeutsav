"""Atomic XLSX writer with Windows PermissionError fallback to part files.

Design
------
- Keeps all records in memory; writes to a temp file; atomically renames
  over the target with ``os.replace``.
- If the target is locked (Excel has it open), catches ``PermissionError``
  and redirects to a dated part file in the same directory.
- The temp file is always created in the **same directory** as the target so
  that ``os.replace`` is guaranteed to be on the same filesystem.
- ``flush()`` is called explicitly by the writer thread on a schedule (every
  N records or T seconds) — not on every ``append()``.
"""
from __future__ import annotations

import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from billetvision.logging_.db import RECORD_COLUMNS, InspectionRecord

logger = logging.getLogger(__name__)


class XlsxLogWriter:
    """In-memory accumulator with batched atomic-replace XLSX flushing."""

    def __init__(self, target_path: str | Path) -> None:
        self.target_path = Path(target_path)
        self.target_path.parent.mkdir(parents=True, exist_ok=True)

        self._records: List[Dict[str, Any]] = []
        self._unflushed: int = 0        # records added since last flush
        self._part_counter: int = 0
        self.active_file: Path = self.target_path
        self.last_status: str = "HEALTHY"

        # Load existing records so a restart doesn't lose history
        self._load_existing()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def append(self, record: InspectionRecord) -> None:
        """Add one record to the in-memory buffer (does not flush)."""
        self._records.append(record.to_dict())
        self._unflushed += 1

    def extend(self, records: List[InspectionRecord]) -> None:
        """Add a batch of records (does not flush)."""
        for r in records:
            self._records.append(r.to_dict())
        self._unflushed += len(records)

    def flush(self) -> None:
        """Write all buffered records to XLSX atomically.

        On PermissionError (target open in Excel) writes to a part file.
        Never raises — logs errors and continues.
        """
        if not self._records:
            return
        self._unflushed = 0

        df = pd.DataFrame(self._records, columns=RECORD_COLUMNS)

        # Temp file must be on the same volume for os.replace to be atomic.
        tmp_fd, tmp_path = tempfile.mkstemp(
            suffix=".xlsx", dir=self.target_path.parent
        )
        os.close(tmp_fd)

        try:
            df.to_excel(tmp_path, index=False, engine="openpyxl")
        except Exception as exc:
            logger.error("XLSX write to temp file failed: %s", exc)
            self.last_status = f"ERROR:{exc}"
            _safe_remove(tmp_path)
            return

        # Try atomic replace onto the target
        try:
            os.replace(tmp_path, self.target_path)
            self.active_file = self.target_path
            self.last_status = "HEALTHY"
            return
        except PermissionError:
            logger.warning(
                "XLSX target locked (%s) — writing to part file", self.target_path
            )

        # Fallback: rename to a part file (part file should not be open)
        self._part_counter += 1
        date_str = datetime.now().strftime("%Y%m%d")
        part_path = self.target_path.parent / (
            f"{self.target_path.stem}_{date_str}_part{self._part_counter}.xlsx"
        )
        try:
            os.replace(tmp_path, part_path)
            self.active_file = part_path
            self.last_status = f"FALLBACK:{part_path.name}"
            logger.info("XLSX part file written: %s", part_path)
        except Exception as exc:
            logger.error("XLSX part file write failed: %s", exc)
            self.last_status = f"ERROR:{exc}"
            _safe_remove(tmp_path)

    @property
    def pending_count(self) -> int:
        """Number of records buffered since the last flush."""
        return self._unflushed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_existing(self) -> None:
        """Load records from the target XLSX if it already exists."""
        if not self.target_path.exists():
            return
        try:
            df = pd.read_excel(self.target_path, engine="openpyxl")
            self._records = df.to_dict(orient="records")
            logger.debug(
                "XlsxLogWriter loaded %d existing records from %s",
                len(self._records),
                self.target_path,
            )
        except Exception as exc:
            logger.warning(
                "Could not load existing XLSX (%s): %s — starting fresh", self.target_path, exc
            )
            self._records = []


def _safe_remove(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass
