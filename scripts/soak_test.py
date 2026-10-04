#!/usr/bin/env python3
"""Logging integrity soak test.

Spawns multiple producer threads that submit InspectionRecords at a
configurable rate for the requested duration, then verifies that every
submitted record reached SQLite and CSV with no data loss or corruption.

Exit code 0 = all checks passed.
Exit code 1 = at least one check failed.

Usage
-----
    python scripts/soak_test.py                # 5-minute run, defaults
    python scripts/soak_test.py --minutes 1    # quick smoke test
    python scripts/soak_test.py --minutes 30 --threads 8 --rate 20
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
import tempfile

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from billetvision.logging_.db import count_records, fetch_recent
from billetvision.logging_.csv_writer import count_csv_rows
from billetvision.logging_.writer import LogWriter
from billetvision.logging_.db import InspectionRecord


# ---------------------------------------------------------------------------
# Producer
# ---------------------------------------------------------------------------

def _producer(
    writer: LogWriter,
    rate_per_sec: float,
    duration_sec: float,
    seq_start: int,
    results: list,
    idx: int,
) -> None:
    """Submit records at ``rate_per_sec`` for ``duration_sec`` seconds."""
    interval = 1.0 / rate_per_sec
    deadline = time.monotonic() + duration_sec
    seq = seq_start
    submitted = 0

    while time.monotonic() < deadline:
        record = InspectionRecord.make(
            billet_seq=seq,
            billet_id=f"T{seq:07d}",
            batch_id=f"SOAK{idx:02d}",
            length_mm=1000.0 + (seq % 5),
            width_mm=130.0,
            height_mm=130.0,
            status="PASS" if seq % 10 != 0 else "FAIL",
            fail_reasons="width out of range" if seq % 10 == 0 else "",
            processing_ms=float(seq % 100),
        )
        writer.submit(record)
        submitted += 1
        seq += 1
        time.sleep(interval)

    results[idx] = submitted


# ---------------------------------------------------------------------------
# Integrity checks
# ---------------------------------------------------------------------------

def _check_integrity(
    db_path: Path,
    csv_path: Path,
    expected: int,
) -> list[str]:
    """Return a list of failure descriptions; empty = all passed."""
    failures = []

    db_count = count_records(db_path)
    if db_count != expected:
        failures.append(f"SQLite: expected {expected} rows, got {db_count}")

    csv_count = count_csv_rows(csv_path)
    if csv_count != expected:
        failures.append(f"CSV: expected {expected} rows, got {csv_count}")

    # Spot-check CSV header
    if csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as fh:
            first_line = fh.readline().strip()
        if not first_line.startswith("timestamp"):
            failures.append(f"CSV header missing or wrong: {first_line!r}")

    # Check DB readable (WAL integrity)
    try:
        rows = fetch_recent(db_path, n=5)
        if expected > 0 and not rows:
            failures.append("DB fetch_recent returned no rows on non-empty DB")
    except Exception as exc:
        failures.append(f"DB read failed: {exc}")

    return failures


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252
    parser = argparse.ArgumentParser(description="Logging soak test")
    parser.add_argument("--minutes", type=float, default=5.0, help="Run duration (minutes)")
    parser.add_argument("--threads", type=int, default=4, help="Producer thread count")
    parser.add_argument("--rate", type=float, default=10.0, help="Records/sec per thread")
    parser.add_argument("--outdir", default=None, help="Output directory (default: temp dir)")
    args = parser.parse_args(argv)

    duration_sec = args.minutes * 60.0
    n_threads = args.threads
    rate = args.rate
    total_expected = int(duration_sec * n_threads * rate)

    if args.outdir:
        out = Path(args.outdir)
        out.mkdir(parents=True, exist_ok=True)
        cleanup = False
    else:
        tmp = tempfile.mkdtemp(prefix="bv_soak_")
        out = Path(tmp)
        cleanup = True

    db_path = out / "soak.db"
    csv_path = out / "soak.csv"
    xlsx_path = out / "soak.xlsx"

    print(f"[soak] Output dir : {out}")
    print(f"[soak] Duration   : {args.minutes:.1f} min ({duration_sec:.0f} s)")
    print(f"[soak] Threads    : {n_threads}")
    print(f"[soak] Rate       : {rate:.0f} rec/s per thread")
    print(f"[soak] Expected ≈ : {total_expected} records")
    print()

    writer = LogWriter(db_path, csv_path, xlsx_path)
    writer.start()

    results = [0] * n_threads
    threads = []
    SEQ_BLOCK = 10_000_000  # ensure no seq collision between threads
    for i in range(n_threads):
        t = threading.Thread(
            target=_producer,
            args=(writer, rate, duration_sec, i * SEQ_BLOCK, results, i),
            daemon=True,
        )
        threads.append(t)

    start = time.monotonic()
    for t in threads:
        t.start()

    # Progress reporting
    try:
        while any(t.is_alive() for t in threads):
            elapsed = time.monotonic() - start
            print(
                f"\r[soak] {elapsed:5.0f}s  submitted={writer.submitted:6d}  "
                f"written={writer.written:6d}  queue={writer.queue_depth:4d}",
                end="",
                flush=True,
            )
            time.sleep(2.0)
    except KeyboardInterrupt:
        print("\n[soak] Interrupted — draining …")

    for t in threads:
        t.join()

    actual_submitted = sum(results)
    print(f"\n[soak] Producers done — submitted={actual_submitted}  waiting for writer …")

    writer.stop(timeout=60.0)
    elapsed_total = time.monotonic() - start
    print(f"[soak] Writer stopped after {elapsed_total:.1f}s total")

    print()
    print("=" * 60)
    print("  Integrity check")
    print("=" * 60)

    failures = _check_integrity(db_path, csv_path, actual_submitted)

    db_count = count_records(db_path)
    csv_count = count_csv_rows(csv_path)
    xlsx_exists = xlsx_path.exists()

    print(f"  Submitted   : {actual_submitted}")
    print(f"  DB rows     : {db_count}")
    print(f"  CSV rows    : {csv_count}")
    print(f"  XLSX exists : {xlsx_exists}")
    print(f"  Writer.written : {writer.written}")
    print()

    if failures:
        print("  FAILURES:")
        for f in failures:
            print(f"    ✗ {f}")
        print()
        print("  → SOAK TEST FAILED")
        return 1
    else:
        print("  → SOAK TEST PASSED — zero data loss")

    # Optional cleanup
    if cleanup:
        import shutil
        try:
            shutil.rmtree(out, ignore_errors=True)
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
