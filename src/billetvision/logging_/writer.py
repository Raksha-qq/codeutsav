"""Single writer thread that funnels all log writes through one path.

Architecture
------------
- Callers on any thread call ``submit(record)`` which puts the record on an
  unbounded ``queue.Queue`` (we never drop log records).
- One background thread pulls records off the queue and writes:
    1. SQLite  — insert + commit immediately (source of truth)
    2. CSV     — append-mode write (part-file fallback if locked)
    3. XLSX    — buffered; flushed every XLSX_FLUSH_RECORDS records or
                 every XLSX_FLUSH_SECONDS seconds, whichever comes first.
- Operator corrections (review queue) go through the same thread via
  ``update_record`` so SQLite, CSV and XLSX stay consistent.
- CSV/XLSX rotate daily (``<stem>_YYYYMMDD.<ext>`` archives); SQLite does not.
- ``stop(timeout)`` sends a sentinel, then joins the thread so the queue
  drains cleanly before the process exits.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from billetvision.logging_.csv_writer import (
    append_to_csv,
    init_csv,
    read_csv_rows,
    rewrite_csv,
)
from billetvision.logging_.db import (
    InspectionRecord,
    commit,
    init_db,
    insert_record,
    open_writer_connection,
)
from billetvision.logging_.db import update_record as db_update_record
from billetvision.logging_.rotation import file_day, rotate_file
from billetvision.logging_.xlsx_writer import XlsxLogWriter

logger = logging.getLogger(__name__)

_SENTINEL = object()  # poison pill

# Flush XLSX after this many unflushed records …
XLSX_FLUSH_RECORDS: int = 10
# … or this many seconds since the last flush, whichever comes first.
XLSX_FLUSH_SECONDS: float = 5.0


@dataclass
class _Update:
    """A queued correction to an already-written record."""

    billet_seq: int
    fields: Dict[str, Any]
    done: threading.Event = field(default_factory=threading.Event)
    ok: bool = False


class LogWriter:
    """Thread-safe single-writer log manager.

    Usage::

        writer = LogWriter("data/outputs/billetvision.db",
                           "data/outputs/billet_log.csv",
                           "data/outputs/billet_log.xlsx")
        writer.start()
        writer.submit(record)
        ...
        writer.stop()
    """

    def __init__(
        self,
        db_path: str | Path,
        csv_path: str | Path,
        xlsx_path: str | Path,
        rotate_daily: bool = True,
        clock: Callable[[], date] = date.today,
    ) -> None:
        self.db_path = Path(db_path)
        self.csv_path = Path(csv_path)
        self.xlsx_path = Path(xlsx_path)
        self.rotate_daily = rotate_daily
        self._clock = clock

        self._queue: queue.Queue = queue.Queue()
        self._thread: Optional[threading.Thread] = None

        # Counters (written only by the writer thread after start)
        self._written: int = 0
        # Submitted counter is written by caller threads — protect with a lock
        self._submitted: int = 0
        self._submitted_lock = threading.Lock()

        self._running = False
        self._day: date = clock()
        self.xlsx_status: str = "HEALTHY"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Initialise storage files and start the background writer thread."""
        if self._running:
            return
        init_db(self.db_path)
        self._day = self._clock()
        if self.rotate_daily:
            self._rotate_stale_files()
        init_csv(self.csv_path)
        self._running = True
        self._thread = threading.Thread(
            target=self._worker, name="LogWriter", daemon=False
        )
        self._thread.start()
        logger.info("LogWriter started (db=%s)", self.db_path)

    def submit(self, record: InspectionRecord) -> None:
        """Enqueue a record for writing.  Safe to call from any thread."""
        with self._submitted_lock:
            self._submitted += 1
        self._queue.put(record)

    def update_record(
        self, billet_seq: int, fields: Dict[str, Any], timeout: float = 10.0
    ) -> bool:
        """Apply an operator correction to SQLite, CSV and XLSX.

        Blocks until the writer thread has applied it (or ``timeout``).  If the
        writer is not running only SQLite is updated.

        Returns True if a record with ``billet_seq`` was updated in SQLite.
        """
        if not self._running:
            return db_update_record(self.db_path, billet_seq, fields)
        item = _Update(billet_seq, dict(fields))
        self._queue.put(item)
        item.done.wait(timeout)
        return item.ok

    def stop(self, timeout: float = 30.0) -> None:
        """Drain the queue, flush XLSX, and shut down the writer thread."""
        if not self._running:
            return
        self._running = False
        self._queue.put(_SENTINEL)
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        logger.info(
            "LogWriter stopped — submitted=%d written=%d",
            self._submitted,
            self._written,
        )

    @property
    def submitted(self) -> int:
        with self._submitted_lock:
            return self._submitted

    @property
    def written(self) -> int:
        """Records successfully written (read from writer thread — approximate)."""
        return self._written

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    # ------------------------------------------------------------------
    # Rotation
    # ------------------------------------------------------------------

    def _rotate_stale_files(self) -> None:
        """Archive CSV/XLSX files last written on an earlier day (startup)."""
        for path in (self.csv_path, self.xlsx_path):
            day = file_day(path)
            if day is not None and day != self._day:
                rotate_file(path, day)

    def _maybe_rotate(self, xlsx: XlsxLogWriter) -> XlsxLogWriter:
        """Roll the CSV/XLSX over when the local date changed; returns the active XLSX writer."""
        today = self._clock()
        if not self.rotate_daily or today == self._day:
            return xlsx
        old_day = self._day
        self._day = today
        xlsx.flush()
        rotate_file(self.csv_path, old_day)
        if rotate_file(self.xlsx_path, old_day) is not None:
            xlsx = XlsxLogWriter(self.xlsx_path)
        init_csv(self.csv_path)
        logger.info("Daily rotation complete (%s -> %s)", old_day, today)
        return xlsx

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------

    def _write_record(self, conn, xlsx: XlsxLogWriter, record: InspectionRecord) -> None:
        """Write one record to SQLite, CSV and the XLSX buffer; never raises."""
        try:
            insert_record(conn, record)
            commit(conn)
        except Exception as exc:
            logger.error("SQLite insert failed for seq=%s: %s", record.billet_seq, exc)
        try:
            append_to_csv(self.csv_path, record)
        except Exception as exc:
            logger.error("CSV append failed for seq=%s: %s", record.billet_seq, exc)
        xlsx.append(record)
        self._written += 1

    def _apply_update(self, xlsx: XlsxLogWriter, item: _Update) -> None:
        """Apply a correction to SQLite, then patch the current CSV and XLSX."""
        try:
            item.ok = db_update_record(self.db_path, item.billet_seq, item.fields)
            if item.ok:
                rows = read_csv_rows(self.csv_path)
                touched = False
                for row in rows:
                    if str(row.get("billet_seq")) == str(item.billet_seq):
                        row.update({k: v for k, v in item.fields.items()})
                        touched = True
                if touched:
                    rewrite_csv(self.csv_path, rows)
                if xlsx.update_record(item.billet_seq, item.fields):
                    xlsx.flush()
                    self.xlsx_status = xlsx.last_status
        except Exception as exc:
            logger.error("Record update failed for seq=%s: %s", item.billet_seq, exc)
        finally:
            item.done.set()

    def _worker(self) -> None:
        conn = open_writer_connection(self.db_path)
        xlsx = XlsxLogWriter(self.xlsx_path)
        last_xlsx_flush = time.monotonic()

        try:
            while True:
                # Block up to XLSX_FLUSH_SECONDS so we flush on a schedule
                # even when the queue is idle.
                try:
                    item = self._queue.get(timeout=XLSX_FLUSH_SECONDS)
                except queue.Empty:
                    xlsx = self._maybe_rotate(xlsx)
                    self._flush_xlsx_if_needed(xlsx, last_xlsx_flush, force=True)
                    self.xlsx_status = xlsx.last_status
                    last_xlsx_flush = time.monotonic()
                    continue

                if item is _SENTINEL:
                    break

                if isinstance(item, _Update):
                    self._apply_update(xlsx, item)
                    continue

                xlsx = self._maybe_rotate(xlsx)
                self._write_record(conn, xlsx, item)
                self._queue.task_done()

                # Flush XLSX on schedule
                now = time.monotonic()
                if self._flush_xlsx_if_needed(xlsx, last_xlsx_flush, force=False):
                    last_xlsx_flush = now
                    self.xlsx_status = xlsx.last_status

        finally:
            # Drain any remaining items that arrived after the sentinel
            while True:
                try:
                    item = self._queue.get_nowait()
                except queue.Empty:
                    break
                if item is _SENTINEL:
                    continue
                if isinstance(item, _Update):
                    self._apply_update(xlsx, item)
                    continue
                self._write_record(conn, xlsx, item)

            # Final XLSX flush
            xlsx.flush()
            self.xlsx_status = xlsx.last_status
            conn.close()
            logger.debug("LogWriter thread exited cleanly")

    @staticmethod
    def _flush_xlsx_if_needed(
        xlsx: XlsxLogWriter,
        last_flush: float,
        force: bool,
    ) -> bool:
        """Flush XLSX if threshold is met.  Returns True if a flush happened."""
        elapsed = time.monotonic() - last_flush
        should = (
            force
            or xlsx.pending_count >= XLSX_FLUSH_RECORDS
            or elapsed >= XLSX_FLUSH_SECONDS
        )
        if should and xlsx.pending_count > 0:
            xlsx.flush()
            return True
        return False
