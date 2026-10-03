"""Single writer thread that funnels all log writes through one path.

Architecture
------------
- Callers on any thread call ``submit(record)`` which puts the record on an
  unbounded ``queue.Queue`` (we never drop log records).
- One background thread pulls records off the queue and writes:
    1. SQLite  — insert + commit immediately (source of truth)
    2. CSV     — append-mode write (always consistent with SQLite)
    3. XLSX    — buffered; flushed every XLSX_FLUSH_RECORDS records or
                 every XLSX_FLUSH_SECONDS seconds, whichever comes first.
- ``stop(timeout)`` sends a sentinel, then joins the thread so the queue
  drains cleanly before the process exits.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from pathlib import Path
from typing import Optional

from billetvision.logging_.db import (
    InspectionRecord,
    commit,
    init_db,
    insert_record,
    open_writer_connection,
)
from billetvision.logging_.csv_writer import append_to_csv, init_csv
from billetvision.logging_.xlsx_writer import XlsxLogWriter

logger = logging.getLogger(__name__)

_SENTINEL = object()  # poison pill

# Flush XLSX after this many unflushed records …
XLSX_FLUSH_RECORDS: int = 10
# … or this many seconds since the last flush, whichever comes first.
XLSX_FLUSH_SECONDS: float = 5.0


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
    ) -> None:
        self.db_path = Path(db_path)
        self.csv_path = Path(csv_path)
        self.xlsx_path = Path(xlsx_path)

        self._queue: queue.Queue = queue.Queue()
        self._thread: Optional[threading.Thread] = None

        # Counters (written only by the writer thread after start)
        self._written: int = 0
        # Submitted counter is written by caller threads — protect with a lock
        self._submitted: int = 0
        self._submitted_lock = threading.Lock()

        self._running = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Initialise storage files and start the background writer thread."""
        if self._running:
            return
        init_db(self.db_path)
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
    # Worker
    # ------------------------------------------------------------------

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
                    self._flush_xlsx_if_needed(xlsx, last_xlsx_flush, force=True)
                    last_xlsx_flush = time.monotonic()
                    continue

                if item is _SENTINEL:
                    break

                record: InspectionRecord = item

                # 1 — SQLite (source of truth)
                try:
                    insert_record(conn, record)
                    commit(conn)
                except Exception as exc:
                    logger.error("SQLite insert failed for seq=%s: %s", record.billet_seq, exc)

                # 2 — CSV
                try:
                    append_to_csv(self.csv_path, record)
                except Exception as exc:
                    logger.error("CSV append failed for seq=%s: %s", record.billet_seq, exc)

                # 3 — XLSX buffer
                xlsx.append(record)

                self._written += 1
                self._queue.task_done()

                # Flush XLSX on schedule
                now = time.monotonic()
                if self._flush_xlsx_if_needed(xlsx, last_xlsx_flush, force=False):
                    last_xlsx_flush = now

        finally:
            # Drain any remaining items that arrived after the sentinel
            while True:
                try:
                    item = self._queue.get_nowait()
                except queue.Empty:
                    break
                if item is _SENTINEL:
                    continue
                record = item
                try:
                    insert_record(conn, record)
                    commit(conn)
                    append_to_csv(self.csv_path, record)
                    xlsx.append(record)
                    self._written += 1
                except Exception as exc:
                    logger.error("Final drain write failed: %s", exc)

            # Final XLSX flush
            xlsx.flush()
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
