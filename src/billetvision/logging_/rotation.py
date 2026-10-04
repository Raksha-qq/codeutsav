"""Daily log rotation for the CSV / XLSX exports (FR-15).

The live files keep today's rows.  When the date changes, each file is renamed
to ``<stem>_YYYYMMDD<suffix>`` (the day its rows belong to) and a fresh file is
started.  SQLite is never rotated — it remains the complete source of truth.
"""
from __future__ import annotations

import logging
import os
from datetime import date, datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def archive_path(path: Path, day: date) -> Path:
    """Archive name for ``path`` holding ``day``'s rows."""
    return path.with_name(f"{path.stem}_{day.strftime('%Y%m%d')}{path.suffix}")


def file_day(path: Path) -> Optional[date]:
    """Local date the file was last modified, or None if it does not exist."""
    try:
        return datetime.fromtimestamp(path.stat().st_mtime).date()
    except OSError:
        return None


def rotate_file(path: Path, day: date) -> Optional[Path]:
    """Rename ``path`` to its dated archive name.

    Returns the archive path, or None when there was nothing to rotate or the
    file is locked (Excel open) — the caller keeps writing to the live file and
    rotation is retried on the next record.  Never overwrites an existing
    archive: a numeric suffix is added instead.
    """
    if not path.exists() or path.stat().st_size == 0:
        return None
    target = archive_path(path, day)
    n = 1
    while target.exists():
        target = target.with_name(f"{archive_path(path, day).stem}_{n}{path.suffix}")
        n += 1
    try:
        os.replace(path, target)
    except PermissionError:
        logger.warning("Cannot rotate %s (locked) — will retry", path)
        return None
    except OSError as exc:
        logger.error("Rotation of %s failed: %s", path, exc)
        return None
    logger.info("Rotated %s -> %s", path.name, target.name)
    return target
