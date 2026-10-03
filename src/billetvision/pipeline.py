"""BilletVision end-to-end inspection pipeline.

Wires together capture → preprocess → segment → track → OCR → decision
→ log → annotate → MJPEG + WebSocket broadcast.

All I/O-heavy work (capture, processing) runs in background threads.
The FastAPI app calls ``pipeline.start()`` on startup and
``pipeline.stop()`` on shutdown.  The WS event loop is injected via
``set_event_loop()`` so the sync pipeline thread can safely broadcast.
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import yaml

from billetvision.alerts.manager import AlertManager
from billetvision.api.mjpeg import latest_frame
from billetvision.api.ws import ws_manager
from billetvision.capture.frame_source import FrameSource
from billetvision.decision.engine import evaluate, load_tolerances
from billetvision.logging_.db import InspectionRecord, update_record_status
from billetvision.logging_.writer import LogWriter
from billetvision.ocr.reader import OcrReader
from billetvision.ocr.validate import correct_and_validate
from billetvision.vision.calibrate import CalibrationData
from billetvision.vision.measure import measure
from billetvision.vision.preprocess import preprocess
from billetvision.vision.segment import segment
from billetvision.vision.track import CentroidTracker, TrackedBillet

logger = logging.getLogger(__name__)

# Status → BGR colour for annotation
_STATUS_COLOR: Dict[str, tuple] = {
    "PASS":   (0, 200, 0),
    "FAIL":   (0, 0, 220),
    "REWORK": (0, 140, 255),
    "REVIEW": (0, 215, 255),
}
_DEFAULT_COLOR = (180, 180, 180)


# ---------------------------------------------------------------------------
# Pipeline stats (read by the API)
# ---------------------------------------------------------------------------

class PipelineStats:
    """Counters and rates updated by the processing thread."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.fps: float = 0.0
        self.latency_ms: float = 0.0
        self.total: int = 0
        self.by_status: Dict[str, int] = {"PASS": 0, "FAIL": 0, "REWORK": 0, "REVIEW": 0}
        self.ocr_ok: int = 0
        self._fps_frames: int = 0
        self._fps_t0: float = time.monotonic()

    def tick_frame(self, latency_ms: float) -> None:
        with self._lock:
            self.latency_ms = latency_ms
            self._fps_frames += 1
            elapsed = time.monotonic() - self._fps_t0
            if elapsed >= 1.0:
                self.fps = self._fps_frames / elapsed
                self._fps_frames = 0
                self._fps_t0 = time.monotonic()

    def add_result(self, status: str, ocr_matched: bool) -> None:
        with self._lock:
            self.total += 1
            self.by_status[status] = self.by_status.get(status, 0) + 1
            if ocr_matched:
                self.ocr_ok += 1

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            total = max(self.total, 1)
            pass_count = self.by_status.get("PASS", 0)
            return {
                "fps": round(self.fps, 1),
                "latency_ms": round(self.latency_ms, 1),
                "total": self.total,
                "pass_rate": round(pass_count / total, 4),
                "ocr_rate": round(self.ocr_ok / total, 4),
                "by_status": dict(self.by_status),
            }


# ---------------------------------------------------------------------------
# Main pipeline class
# ---------------------------------------------------------------------------

class BilletVisionPipeline:
    """Single-instance inspection pipeline.

    Usage (managed by FastAPI lifespan)::

        pipeline = BilletVisionPipeline("config/config.yaml")
        pipeline.start(event_loop)
        ...
        pipeline.stop()
    """

    def __init__(self, config_path: str | Path = "config/config.yaml") -> None:
        self._config_path = Path(config_path)
        self._cfg: Dict[str, Any] = {}
        self._running = False
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Components — created in start()
        self._source: Optional[FrameSource] = None
        self._tracker: Optional[CentroidTracker] = None
        self._ocr: Optional[OcrReader] = None
        self._writer: Optional[LogWriter] = None
        self.alert_manager = AlertManager()
        self.stats = PipelineStats()

        # Config-derived values
        self._mm_per_px: float = 1.0
        self._profile_name: str = "square_130"
        self._profile_shape: str = "square"
        self._tolerances: Dict[str, Any] = {}
        self._id_regex: str = r"^[A-Z]\d{5,7}$"
        self._min_confidence: float = 0.60
        self._batch_id: str = ""
        self._billet_seq: int = 0
        self._snapshot_dir: Path = Path("data/outputs/snapshots")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, event_loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
        if self._running:
            return
        self._loop = event_loop
        self._load_config()
        self._init_components()
        self._stop_event.clear()
        self._running = True
        self._thread = threading.Thread(
            target=self._process_loop, name="BVPipeline", daemon=True
        )
        self._thread.start()
        logger.info("BilletVision pipeline started")

    def stop(self, timeout: float = 10.0) -> None:
        if not self._running:
            return
        self._running = False
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=timeout)
        if self._source:
            self._source.stop()
        if self._writer:
            self._writer.stop(timeout=15.0)
        logger.info("BilletVision pipeline stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    # ------------------------------------------------------------------
    # Config / tolerance access for the API
    # ------------------------------------------------------------------

    def get_tolerances(self) -> Dict[str, Any]:
        return self._tolerances

    def update_profile_tolerances(
        self, profile: str, updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        if profile not in self._tolerances:
            raise KeyError(profile)
        self._tolerances[profile].update(updates)
        return self._tolerances[profile]

    def set_active_profile(self, profile: str) -> None:
        if profile not in self._tolerances:
            raise KeyError(profile)
        self._profile_name = profile
        shape = self._tolerances[profile].get("shape", "square")
        self._profile_shape = shape

    # ------------------------------------------------------------------
    # Internal init
    # ------------------------------------------------------------------

    def _load_config(self) -> None:
        with self._config_path.open(encoding="utf-8") as fh:
            self._cfg = yaml.safe_load(fh)

        vis = self._cfg.get("vision", {})
        self._profile_name = vis.get("active_profile", "square_130")

        tol_path = Path(self._cfg.get("decision", {}).get("tolerances_file", "config/tolerances.yaml"))
        self._tolerances = load_tolerances(tol_path)
        profile_tol = self._tolerances.get(self._profile_name, {})
        self._profile_shape = profile_tol.get("shape", "square")

        ocr_cfg = self._cfg.get("ocr", {})
        self._id_regex = ocr_cfg.get("heat_id_regex", r"^[A-Z]\d{5,7}$")
        self._min_confidence = float(ocr_cfg.get("min_confidence", 0.60))

        log_cfg = self._cfg.get("logging", {})
        self._snapshot_dir = Path(log_cfg.get("snapshot_dir", "data/outputs/snapshots"))
        self._snapshot_dir.mkdir(parents=True, exist_ok=True)

        self._batch_id = datetime.now(timezone.utc).strftime("BATCH-%Y%m%d-%H%M")

        # Calibration
        cal_path = Path(vis.get("calibration_file", "config/calibration.json"))
        self._mm_per_px = self._load_mm_per_px(cal_path)

    def _load_mm_per_px(self, cal_path: Path) -> float:
        if cal_path.exists():
            try:
                import json as _json
                data = _json.loads(cal_path.read_text())
                v = data.get("mm_per_pixel") or data.get("mm_per_px")
                if v and float(v) > 0:
                    return float(v)
            except Exception as exc:
                logger.warning("Could not load calibration: %s", exc)
        logger.warning("No valid calibration found — defaulting mm_per_px=0.5")
        return 0.5  # rough default for demo props

    def _init_components(self) -> None:
        cap_cfg = self._cfg.get("capture", {})
        vis_cfg = self._cfg.get("vision", {})
        log_cfg = self._cfg.get("logging", {})
        ocr_cfg = self._cfg.get("ocr", {})

        # Frame source
        source = cap_cfg.get("source", 0)
        self._source = FrameSource(
            source=source,
            maxsize=cap_cfg.get("queue_maxsize", 5),
            loop=True,  # loop video for demo
            fps=cap_cfg.get("fps_target", 15),
        )
        self._source.start()

        # Tracker — ROI from config
        roi = vis_cfg.get("roi_box", [100, 100, 1180, 620])
        entry_x = roi[0] + 20
        exit_x = roi[2] - 20
        self._tracker = CentroidTracker(
            entry_x=entry_x,
            exit_x=exit_x,
            max_distance_px=100.0,
            max_lost_frames=15,
            best_n=5,
        )

        # OCR (lazy — may not have engines installed)
        self._ocr = OcrReader.from_config(ocr_cfg)

        # Log writer
        self._writer = LogWriter(
            db_path=log_cfg.get("db_path", "data/outputs/billetvision.db"),
            csv_path=log_cfg.get("csv_path", "data/outputs/billet_log.csv"),
            xlsx_path=log_cfg.get("xlsx_path", "data/outputs/billet_log.xlsx"),
        )
        self._writer.start()

        logger.info(
            "Pipeline components ready — profile=%s mm_per_px=%.4f",
            self._profile_name, self._mm_per_px,
        )

    # ------------------------------------------------------------------
    # Processing loop
    # ------------------------------------------------------------------

    def _process_loop(self) -> None:
        assert self._source and self._tracker and self._writer

        vis_cfg = self._cfg.get("vision", {})
        hot_mode = vis_cfg.get("hot_billet_mode", False)
        min_area = vis_cfg.get("min_contour_area", 5000)
        roi_box = vis_cfg.get("roi_box", [0, 0, 9999, 9999])

        consecutive_empty = 0

        while not self._stop_event.is_set():
            ok, frame = self._source.read(timeout=0.5)
            if not ok or frame is None:
                consecutive_empty += 1
                if consecutive_empty > 30:
                    self._broadcast_sync({"type": "camera_status", "status": "lost"})
                    consecutive_empty = 0
                continue
            consecutive_empty = 0

            t0 = time.perf_counter()

            try:
                finalized, annotated = self._process_frame(
                    frame, hot_mode=hot_mode, min_area=min_area, roi_box=roi_box
                )
                for tb in finalized:
                    self._handle_billet(tb, annotated)
            except Exception as exc:
                logger.exception("Frame processing error: %s", exc)
                annotated = frame.copy()

            # Push to MJPEG
            latest_frame.update(annotated)

            latency = (time.perf_counter() - t0) * 1000
            self.stats.tick_frame(latency)

            # Broadcast telemetry every second
            self._maybe_broadcast_stats()

        logger.info("Pipeline processing loop exited")

    def _process_frame(
        self,
        frame: np.ndarray,
        *,
        hot_mode: bool,
        min_area: int,
        roi_box: List[int],
    ):
        gray = preprocess(frame, hot_billet_mode=hot_mode)

        seg = segment(
            gray,
            hot_billet_mode=hot_mode,
            min_area_frac=min_area / max(gray.shape[0] * gray.shape[1], 1),
        )

        detections = []
        if seg is not None:
            M = cv2.moments(seg.contour)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                meas = measure(
                    seg.contour, self._mm_per_px, shape=self._profile_shape
                )
                detections.append(((cx, cy), seg.contour, gray, meas))

        finalized = self._tracker.update(detections)

        annotated = self._annotate(frame, seg, roi_box)
        return finalized, annotated

    # ------------------------------------------------------------------
    # Per-billet handling
    # ------------------------------------------------------------------

    def _handle_billet(self, tb: TrackedBillet, annotated_frame: np.ndarray) -> None:
        self._billet_seq += 1
        t0 = time.perf_counter()

        # OCR
        ocr_result = self._ocr.read_best(tb.best_frame_gray, pattern=self._id_regex)
        corrected, matched, penalty = correct_and_validate(ocr_result.text, self._id_regex)
        final_conf = round(max(0.0, ocr_result.confidence - penalty), 4)
        ocr_status = "PASS" if matched and final_conf >= self._min_confidence else "REVIEW"

        billet_id = corrected if corrected else f"BLT{self._billet_seq:06d}"

        # Decision
        verdict = evaluate(
            tb.measurement,
            self._profile_name,
            self._tolerances,
            ocr_status=ocr_status,
        )

        # Snapshot
        image_path = self._save_snapshot(annotated_frame, self._billet_seq, verdict.status)

        processing_ms = (time.perf_counter() - t0) * 1000

        # Log record
        record = InspectionRecord.make(
            billet_seq=self._billet_seq,
            billet_id=billet_id,
            batch_id=self._batch_id,
            length_mm=tb.measurement.length_mm,
            width_mm=tb.measurement.width_mm,
            height_mm=tb.measurement.height_mm,
            diameter_mm=tb.measurement.diameter_mm,
            ovality=tb.measurement.ovality,
            diag_diff_mm=tb.measurement.diag_diff_mm,
            defects=", ".join(tb.measurement.defects),
            ocr_confidence=final_conf,
            status=verdict.status,
            fail_reasons="; ".join(verdict.reasons),
            image_path=str(image_path),
            processing_ms=round(processing_ms, 1),
        )
        self._writer.submit(record)

        # Alert
        if verdict.status in ("FAIL", "REWORK", "REVIEW"):
            self.alert_manager.trigger(
                billet_id=billet_id,
                status=verdict.status,
                reasons=verdict.reasons,
                image_path=str(image_path),
            )

        # Stats
        self.stats.add_result(verdict.status, matched)

        # Broadcast billet event
        self._broadcast_sync({
            "type": "billet",
            "billet_seq": self._billet_seq,
            "billet_id": billet_id,
            "status": verdict.status,
            "length_mm": round(tb.measurement.length_mm, 2),
            "width_mm": round(tb.measurement.width_mm, 2),
            "height_mm": round(tb.measurement.height_mm, 2),
            "diameter_mm": tb.measurement.diameter_mm,
            "ovality": tb.measurement.ovality,
            "reasons": verdict.reasons,
            "ocr_confidence": final_conf,
            "processing_ms": round(processing_ms, 1),
            "timestamp": record.timestamp,
        })

    # ------------------------------------------------------------------
    # Annotation
    # ------------------------------------------------------------------

    def _annotate(
        self,
        frame: np.ndarray,
        seg,
        roi_box: List[int],
    ) -> np.ndarray:
        out = frame.copy()

        # Draw ROI
        rx1, ry1, rx2, ry2 = roi_box[0], roi_box[1], roi_box[2], roi_box[3]
        cv2.rectangle(out, (rx1, ry1), (rx2, ry2), (60, 60, 60), 1)

        # Draw detected contour
        if seg is not None:
            cv2.drawContours(out, [seg.contour], -1, (0, 200, 255), 2)
            x, y, w, h = seg.bounding_rect
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 200, 255), 1)

        # FPS + latency overlay (top-left)
        snap = self.stats.snapshot()
        cv2.putText(
            out,
            f"FPS {snap['fps']:.1f}  {snap['latency_ms']:.0f}ms  #={snap['total']}",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA,
        )
        return out

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _save_snapshot(
        self, frame: np.ndarray, seq: int, status: str
    ) -> Path:
        fname = f"{seq:06d}_{status}_{datetime.now(timezone.utc).strftime('%H%M%S')}.jpg"
        path = self._snapshot_dir / fname
        try:
            cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        except Exception as exc:
            logger.warning("Snapshot save failed: %s", exc)
        return path

    _last_stats_broadcast: float = 0.0

    def _maybe_broadcast_stats(self) -> None:
        now = time.monotonic()
        if now - self._last_stats_broadcast >= 1.0:
            self._last_stats_broadcast = now
            self._broadcast_sync({"type": "stats", **self.stats.snapshot()})

    def _broadcast_sync(self, message: Dict[str, Any]) -> None:
        """Fire-and-forget broadcast from the sync processing thread."""
        if self._loop and not self._loop.is_closed():
            try:
                asyncio.run_coroutine_threadsafe(
                    ws_manager.broadcast(message), self._loop
                )
            except Exception as exc:
                logger.debug("WS broadcast error: %s", exc)


# ---------------------------------------------------------------------------
# Module-level singleton — shared with the API
# ---------------------------------------------------------------------------

pipeline = BilletVisionPipeline()
