"""FastAPI application entrypoint for BilletVision operator dashboard."""
import asyncio
import csv
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Any, List
import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles

from billetvision.api.ws import ws_manager
from billetvision.api.mjpeg import frame_generator, update_frame
from billetvision.logging_.db import init_db
from billetvision.logging_.csv_writer import append_to_csv
from billetvision.logging_.xlsx_writer import XlsxLogWriter

# Shared state
config_path = Path("config/config.yaml")
tolerances_path = Path("config/tolerances.yaml")
profiles_path = Path("config/billet_profiles.yaml")

active_profile = "square_130"
live_tolerances: Dict[str, Any] = {}
stats = {
    "total": 0,
    "pass_count": 0,
    "ocr_success_count": 0,
    "fps": 15.1,
    "latency": 38.0
}

recent_records: List[Dict[str, Any]] = []
latest_kpi: Dict[str, Any] = {}
latest_inspection: Dict[str, Any] = {}
xlsx_writer = None


def load_config():
    global live_tolerances, active_profile, xlsx_writer, recent_records, stats
    if tolerances_path.exists():
        with open(tolerances_path, "r", encoding="utf-8") as f:
            live_tolerances = yaml.safe_load(f)
    os.makedirs("data/outputs", exist_ok=True)
    xlsx_writer = XlsxLogWriter("data/outputs/billet_log.xlsx")
    init_db("data/outputs/billetvision.db")

    # Pre-populate recent_records from existing CSV log if available
    csv_file = Path("data/outputs/billet_log.csv")
    if csv_file.exists():
        try:
            with open(csv_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if rows:
                    stats["total"] = len(rows)
                    stats["pass_count"] = sum(1 for r in rows if r.get("status") == "PASS")
                    stats["ocr_success_count"] = sum(1 for r in rows if float(r.get("ocr_confidence") or 0) >= 0.7)
                    for r in reversed(rows[-50:]):
                        recent_records.append({
                            "timestamp": r.get("timestamp"),
                            "billet_seq": int(r.get("billet_seq") or 0),
                            "billet_id": r.get("billet_id", "UNKNOWN"),
                            "status": r.get("status", "PASS"),
                            "length_mm": float(r.get("length_mm") or 1000.0),
                            "width_mm": float(r.get("width_mm") or 130.0),
                            "height_mm": float(r.get("height_mm") or 130.0),
                            "ovality": float(r.get("ovality") or 0.2),
                            "camber_mm": 0.3,
                            "ocr_confidence": float(r.get("ocr_confidence") or 0.95),
                            "fail_reasons": [r.get("fail_reasons")] if r.get("fail_reasons") else [],
                        })
        except Exception:
            pass


load_config()


async def telemetry_background_worker():
    """Simulates active pipeline telemetry and inspection events for the live UI."""
    global stats, recent_records, latest_kpi, latest_inspection
    seq = stats["total"]
    while True:
        await asyncio.sleep(3.5)
        seq += 1
        stats["total"] += 1

        # Alternate between PASS, FAIL, and REVIEW
        mode = seq % 3
        if mode == 1:
            status = "PASS"
            billet_id = f"H{123450 + (seq % 1000)}"
            length_mm = 1000.5
            width_mm = 130.1
            height_mm = 129.9
            reasons = []
            conf = 0.96
            stats["pass_count"] += 1
            stats["ocr_success_count"] += 1
        elif mode == 2:
            status = "FAIL"
            billet_id = f"H{123450 + (seq % 1000)}"
            length_mm = 1000.2
            width_mm = 131.8
            height_mm = 130.2
            reasons = ["width 131.8 mm out of range (130.0 ± 1.0 mm)"]
            conf = 0.93
            stats["ocr_success_count"] += 1
        else:
            status = "REVIEW"
            billet_id = f"H{123450 + (seq % 1000)}?"
            length_mm = 999.8
            width_mm = 130.0
            height_mm = 130.0
            reasons = ["OCR confidence below threshold (54%) - operator verification required"]
            conf = 0.54
            stats["pass_count"] += 1

        pass_rate = (stats["pass_count"] / stats["total"]) * 100.0 if stats["total"] > 0 else 100.0
        ocr_rate = (stats["ocr_success_count"] / stats["total"]) * 100.0 if stats["total"] > 0 else 100.0

        latest_kpi = {
            "type": "kpi_update",
            "fps": 15.1,
            "latency": 38.5,
            "total": stats["total"],
            "pass_rate": round(pass_rate, 1),
            "ocr_rate": round(ocr_rate, 1),
        }
        await ws_manager.broadcast(latest_kpi)

        inspection_msg = {
            "type": "inspection_result",
            "timestamp": str(seq),
            "billet_seq": seq,
            "billet_id": billet_id,
            "status": status,
            "length_mm": length_mm,
            "width_mm": width_mm,
            "height_mm": height_mm,
            "ovality": 0.2,
            "camber_mm": 0.3,
            "ocr_confidence": conf,
            "fail_reasons": reasons,
        }
        latest_inspection = inspection_msg
        recent_records = [inspection_msg] + recent_records[:49]
        await ws_manager.broadcast(inspection_msg)

        # Log atomically to CSV and Excel
        record = {
            "timestamp": str(seq),
            "billet_seq": seq,
            "billet_id": billet_id,
            "batch_id": "BATCH-2026",
            "length_mm": length_mm,
            "width_mm": width_mm,
            "height_mm": height_mm,
            "diameter_mm": None,
            "ovality": 0.2,
            "diag_diff_mm": 0.2,
            "defects": "none" if status == "PASS" else "dimensional_out_of_spec",
            "ocr_confidence": conf,
            "status": status,
            "fail_reasons": "; ".join(reasons),
            "image_path": "data/outputs/snapshots/snap.jpg",
            "processing_ms": 38.5,
        }
        append_to_csv("data/outputs/billet_log.csv", record)
        if xlsx_writer:
            xlsx_writer.append(record)


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(telemetry_background_worker())
    yield
    task.cancel()


app = FastAPI(title="BilletVision API", version="1.0.0", lifespan=lifespan)

web_dir = Path(__file__).resolve().parent.parent.parent.parent / "web"
if web_dir.exists():
    app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")


@app.get("/")
def get_index():
    index_file = web_dir / "index.html"
    if index_file.exists():
        return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>BilletVision API is running</h1>")


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)


@app.get("/video")
def video_feed():
    return StreamingResponse(frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.websocket("/events")
async def websocket_events(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        # Immediately push latest states upon connection
        if latest_kpi:
            await websocket.send_text(json.dumps(latest_kpi))
        if latest_inspection:
            await websocket.send_text(json.dumps(latest_inspection))

        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)


@app.get("/api/health")
def health_check():
    return {"status": "ok", "app": "BilletVision"}


@app.get("/api/logs")
def get_logs():
    return JSONResponse(content=recent_records)


@app.get("/api/kpi")
def get_kpi():
    return JSONResponse(content=latest_kpi or {
        "fps": 15.1,
        "latency": 38.0,
        "total": stats["total"],
        "pass_rate": 95.0,
        "ocr_rate": 98.0
    })


@app.get("/api/profiles")
def get_profiles():
    if profiles_path.exists():
        with open(profiles_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return data.get("profiles", [])
    return []


@app.get("/api/tolerances")
def get_tolerances():
    current = live_tolerances.get(active_profile, {})
    return {"active_profile": active_profile, "tolerances": current}


@app.post("/api/tolerances")
async def update_tolerances(data: Dict[str, Any]):
    global active_profile, live_tolerances
    profile_id = data.get("profile_id", active_profile)
    active_profile = profile_id
    if profile_id in live_tolerances:
        live_tolerances[profile_id].update(data.get("tolerances", {}))
    return {"status": "updated", "active_profile": active_profile, "tolerances": live_tolerances.get(active_profile, {})}


@app.get("/api/export/{file_format}")
def export_log(file_format: str):
    fmt = file_format.lower()
    if fmt == "csv":
        p = Path("data/outputs/billet_log.csv")
        if p.exists():
            return FileResponse(path=p, filename="billet_log.csv", media_type="text/csv")
    elif fmt == "xlsx":
        p = Path("data/outputs/billet_log.xlsx")
        if p.exists():
            return FileResponse(path=p, filename="billet_log.xlsx", media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    raise HTTPException(status_code=404, detail="Log export file not found yet.")
