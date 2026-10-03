"""MJPEG video streaming utilities with live frame buffer and fallback simulation."""
import time
import threading
import logging
from typing import Generator, Optional
import cv2
import numpy as np

logger = logging.getLogger(__name__)

_latest_frame: Optional[np.ndarray] = None
_frame_lock = threading.Lock()


def update_frame(frame: np.ndarray) -> None:
    """Update the latest broadcast frame from pipeline or capture."""
    global _latest_frame
    if frame is None:
        return
    with _frame_lock:
        _latest_frame = frame.copy()


def get_latest_frame() -> Optional[np.ndarray]:
    """Retrieve the latest broadcast frame."""
    with _frame_lock:
        return _latest_frame.copy() if _latest_frame is not None else None


def create_synthetic_frame(frame_idx: int) -> np.ndarray:
    """Generate dynamic synthetic conveyor belt frame with moving billet and ArUco marker."""
    w, h = 1280, 720
    # Industrial dark conveyor background
    img = np.full((h, w, 3), 32, dtype=np.uint8)

    # Conveyor belt lane
    belt_top, belt_bottom = 120, 600
    img[belt_top:belt_bottom, :] = 45

    # Moving belt texture / rollers
    offset = int((frame_idx * 8) % 80)
    for x in range(-80 + offset, w + 80, 80):
        cv2.line(img, (x, belt_top), (x, belt_bottom), (55, 55, 55), 2)

    # ArUco marker in top-left
    try:
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        marker = cv2.aruco.generateImageMarker(aruco_dict, 0, 90)
        marker_bgr = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
        img[30:120, 30:120] = marker_bgr
        cv2.rectangle(img, (28, 28), (122, 122), (0, 255, 200), 1)
        cv2.putText(img, "ArUco: 50mm", (30, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 200), 1)
    except Exception:
        cv2.rectangle(img, (30, 30), (120, 120), (255, 255, 255), -1)
        cv2.putText(img, "CALIB REF", (35, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

    # Inspection ROI Box
    roi_x1, roi_y1, roi_x2, roi_y2 = 180, 140, 1100, 580
    cv2.rectangle(img, (roi_x1, roi_y1), (roi_x2, roi_y2), (70, 70, 100), 2)
    cv2.putText(img, "INSPECTION ROI ZONE", (roi_x1 + 10, roi_y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 120, 180), 1)

    # Moving Billet cycle (160 frames per cycle, ~10 seconds at 15 FPS)
    cycle = frame_idx % 160
    billet_len_px = 620
    billet_width_px = 260
    bx = int(-billet_len_px + cycle * (w + billet_len_px) / 160)
    by = int(h // 2 - billet_width_px // 2)

    # Alternate between normal (PASS) and oversized (FAIL)
    is_oversized = ((frame_idx // 160) % 2) == 1
    if is_oversized:
        billet_width_px = 275  # ~131.8 mm (fail)
        heat_id = "H123459"
    else:
        billet_width_px = 260  # ~130.0 mm (pass)
        heat_id = "H123456"

    # Draw billet if in visible range
    if bx + billet_len_px > 0 and bx < w:
        x1 = max(0, bx)
        x2 = min(w, bx + billet_len_px)
        cv2.rectangle(img, (x1, by), (x2, by + billet_width_px), (160, 165, 175), -1)
        cv2.rectangle(img, (x1, by), (x2, by + billet_width_px), (210, 215, 225), 2)

        # Stamped Heat ID text
        if bx + 120 > 0 and bx + 120 < w:
            cv2.putText(img, f"HEAT: {heat_id}", (bx + 80, by + billet_width_px // 2 + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (30, 30, 40), 3)

        # Inspection bounding box & overlays when inside ROI
        if bx > roi_x1 - 100 and bx + billet_len_px < roi_x2 + 100:
            box_color = (0, 0, 240) if is_oversized else (0, 230, 120)
            status_label = "FAIL: OVERSIZED (+1.8mm)" if is_oversized else "PASS: 1000x130mm"
            cv2.rectangle(img, (bx - 4, by - 4), (bx + billet_len_px + 4, by + billet_width_px + 4), box_color, 2)
            cv2.putText(img, f"Length: 1000.5mm  Width: {131.8 if is_oversized else 130.1}mm",
                        (bx, by - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
            cv2.putText(img, f"Status: {status_label}",
                        (bx, by + billet_width_px + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

    # Top Telemetry HUD
    t_str = time.strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(img, f"BILLET-VISION LIVE FEED | {t_str} | 15.0 FPS", (140, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 230, 230), 2)
    cv2.putText(img, "Profile: square_130 | 1 px = 0.21 mm | Conveyor: 250 mm/s", (140, 88),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 185, 210), 1)

    return img


def frame_generator() -> Generator[bytes, None, None]:
    """Generates continuous MJPEG boundary frames with proper JPEG bytes."""
    frame_idx = 0
    while True:
        frame = get_latest_frame()
        if frame is None:
            frame = create_synthetic_frame(frame_idx)
            frame_idx += 1

        ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ret:
            jpg_bytes = buffer.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(jpg_bytes)).encode("utf-8") + b"\r\n\r\n" +
                jpg_bytes + b"\r\n"
            )
        time.sleep(0.066)  # ~15 FPS
