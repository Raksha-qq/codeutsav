"""Logging package: SQLite source of truth, CSV append, atomic XLSX, single writer thread."""
from billetvision.logging_.db import (
    InspectionRecord,
    RECORD_COLUMNS,
    init_db,
    insert_record,
    count_records,
    fetch_recent,
)
from billetvision.logging_.csv_writer import append_to_csv, init_csv, count_csv_rows
from billetvision.logging_.xlsx_writer import XlsxLogWriter
from billetvision.logging_.writer import LogWriter

__all__ = [
    "InspectionRecord",
    "RECORD_COLUMNS",
    "init_db",
    "insert_record",
    "count_records",
    "fetch_recent",
    "append_to_csv",
    "init_csv",
    "count_csv_rows",
    "XlsxLogWriter",
    "LogWriter",
]
