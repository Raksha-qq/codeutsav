"""FastAPI application entrypoint for BilletVision operator dashboard."""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from billetvision.api.ws import ws_manager
from billetvision.api.mjpeg import frame_generator

app = FastAPI(title="BilletVision API", version="1.0.0")

web_dir = Path(__file__).resolve().parent.parent.parent.parent / "web"
if web_dir.exists():
    app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")

@app.get("/")
def get_index():
    index_file = web_dir / "index.html"
    if index_file.exists():
        return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>BilletVision API is running</h1>")

@app.get("/video")
def video_feed():
    return StreamingResponse(frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.websocket("/events")
async def websocket_events(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)

@app.get("/api/health")
def health_check():
    return {"status": "ok", "app": "BilletVision"}
