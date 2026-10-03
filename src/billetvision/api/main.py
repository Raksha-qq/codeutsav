"""FastAPI application: MJPEG video, WebSocket events, REST for log/tolerances/export/review."""
from __future__ import annotations

import asyncio
import dataclasses
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from billetvision.api.mjpeg import frame_generator
from billetvision.api.ws import ws_manager
from billetvision.logging_.db import count_records, fetch_recent, update_record_status
from billetvision.pipeline import pipeline

logger = logging.getLogger(__name__)

_CFG_PATH = Path("config/config.yaml")
_WEB_DIR = Path(__file__).resolve().parent.parent.parent.parent / "web"


def _log_cfg() -> Dict[str, Any]:
    with _CFG_PATH.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh).get("logging", {})


# ---------------------------------------------------------------------------
# Lifespan: start/stop pipeline around the server lifetime
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_running_loop()
    pipeline.set_event_loop(loop)
    try:
        pipeline.start(event_loop=loop)
    except Exception as exc:
        logger.error("Pipeline failed to start: %s", exc)
    yield
    pipeline.stop()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="BilletVision API", version="1.0.0", lifespan=lifespan)

if _WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_WEB_DIR)), name="static")


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def get_index():
    index = _WEB_DIR / "index.html"
    if index.exists():
        return HTMLResponse(content=index.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>BilletVision — pipeline running</h1>")


# ---------------------------------------------------------------------------
# MJPEG video stream
# ---------------------------------------------------------------------------

@app.get("/video", include_in_schema=False)
def video_feed():
    """MJPEG stream of the annotated live feed."""
    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ---------------------------------------------------------------------------
# WebSocket events
# ---------------------------------------------------------------------------


@app.websocket("/events")
async def websocket_events(websocket: WebSocket):
    """Real-time billet events and telemetry (JSON messages)."""
    await ws_manager.connect(websocket)
    try:
        while True:
            # Keep the connection alive; pipeline pushes unsolicited events
            await websocket.receive_text()
    except WebSocketDisconnect:
        await ws_manager.disconnect(websocket)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "app": "BilletVision",
        "pipeline_running": pipeline._running,
        "queue_depth": pipeline._writer.queue_depth if pipeline._writer else 0,
    }


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@app.get("/api/stats")
def get_stats():
    """Live pipeline performance counters."""
    return pipeline.stats.snapshot()


# ---------------------------------------------------------------------------
# Log (SQLite read)
# ---------------------------------------------------------------------------

@app.get("/api/log")
def get_log(
    n: int = Query(default=50, ge=1, le=1000),
    status: Optional[str] = Query(default=None, pattern="^(PASS|FAIL|REWORK|REVIEW)$"),
):
    """Return the most-recent ``n`` inspection records (newest first)."""
    cfg = _log_cfg()
    db_path = cfg.get("db_path", "data/outputs/billetvision.db")
    try:
        return fetch_recent(db_path, n=n, status=status)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/log/count")
def log_count():
    cfg = _log_cfg()
    db_path = cfg.get("db_path", "data/outputs/billetvision.db")
    return {"count": count_records(db_path)}


# ---------------------------------------------------------------------------
# Export (file download)
# ---------------------------------------------------------------------------

@app.get("/api/export/csv")
def export_csv():
    """Download the full inspection log as CSV."""
    cfg = _log_cfg()
    path = Path(cfg.get("csv_path", "data/outputs/billet_log.csv"))
    if not path.exists():
        raise HTTPException(status_code=404, detail="CSV log not found")
    return FileResponse(
        str(path),
        media_type="text/csv",
        filename=path.name,
    )


@app.get("/api/export/xlsx")
def export_xlsx():
    """Download the full inspection log as Excel."""
    cfg = _log_cfg()
    path = Path(cfg.get("xlsx_path", "data/outputs/billet_log.xlsx"))
    if not path.exists():
        raise HTTPException(status_code=404, detail="XLSX log not found")
    return FileResponse(
        str(path),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=path.name,
    )


# ---------------------------------------------------------------------------
# Tolerances / profiles
# ---------------------------------------------------------------------------

@app.get("/api/profiles")
def get_profiles():
    """List the configured billet profiles."""
    path = Path("config/billet_profiles.yaml")
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return (yaml.safe_load(fh) or {}).get("profiles", [])


@app.get("/api/tolerances")
def get_tolerances():
    """Return all tolerance profiles."""
    return pipeline.get_tolerances()


@app.get("/api/tolerances/{profile}")
def get_profile(profile: str):
    tols = pipeline.get_tolerances()
    if profile not in tols:
        raise HTTPException(status_code=404, detail=f"Profile '{profile}' not found")
    return tols[profile]


class ToleranceUpdate(BaseModel):
    updates: Dict[str, Any]


@app.put("/api/tolerances/{profile}")
def update_tolerances(profile: str, body: ToleranceUpdate):
    """Patch tolerance values for a profile (in-memory; restarts reverts)."""
    try:
        updated = pipeline.update_profile_tolerances(profile, body.updates)
        return updated
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Profile '{profile}' not found")


@app.put("/api/tolerances/{profile}/activate")
def activate_profile(profile: str):
    """Switch the active inspection profile."""
    try:
        pipeline.set_active_profile(profile)
        return {"active_profile": profile}
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Profile '{profile}' not found")


# ---------------------------------------------------------------------------
# Review queue
# ---------------------------------------------------------------------------

class ReviewResolution(BaseModel):
    action: str          # "approve" | "reject"
    notes: Optional[str] = None


@app.post("/api/review/{billet_seq}")
def resolve_review(billet_seq: int, body: ReviewResolution):
    """Resolve a REVIEW-status billet as PASS (approve) or FAIL (reject).

    The operator can add notes which become the fail_reasons entry.
    """
    if body.action not in ("approve", "reject"):
        raise HTTPException(status_code=400, detail="action must be 'approve' or 'reject'")

    new_status = "PASS" if body.action == "approve" else "FAIL"
    fail_reasons = body.notes if body.action == "reject" else ""

    cfg = _log_cfg()
    db_path = cfg.get("db_path", "data/outputs/billetvision.db")
    updated = update_record_status(db_path, billet_seq, new_status, fail_reasons)
    if not updated:
        raise HTTPException(status_code=404, detail=f"billet_seq {billet_seq} not found")
    return {"billet_seq": billet_seq, "new_status": new_status}


@app.get("/api/review/pending")
def pending_reviews(n: int = Query(default=50, ge=1, le=500)):
    """List records still in REVIEW status."""
    cfg = _log_cfg()
    db_path = cfg.get("db_path", "data/outputs/billetvision.db")
    return fetch_recent(db_path, n=n, status="REVIEW")


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------

@app.get("/api/alerts")
def get_alerts(n: int = Query(default=50, ge=1, le=500)):
    """Return the last ``n`` alerts (newest first)."""
    history = pipeline.alert_manager.history[-n:]
    return [dataclasses.asdict(a) for a in reversed(history)]


@app.post("/api/alerts/clear")
def clear_active_alert():
    """Dismiss the currently active alert banner."""
    pipeline.alert_manager.clear_active()
    return {"ok": True}


@app.get("/api/alerts/active")
def active_alert():
    a = pipeline.alert_manager.active_alert
    return dataclasses.asdict(a) if a else None
