#!/usr/bin/env python3
"""Calibration wizard CLI.

Detects ArUco marker (or optional checkerboard), computes mm/px and plane homography,
draws an overlay, prints '1 px = x mm', and saves calibration to config/calibration.json.
"""

from __future__ import annotations

import argparse
import glob
import logging
from pathlib import Path
import sys
from typing import Optional, Tuple

# Ensure src is in sys.path when running as a standalone script
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import cv2
import numpy as np

from billetvision.capture import build_source
from billetvision.vision.calibrate import (
    CalibrationData,
    calibrate_camera_checkerboard,
    calibrate_from_marker,
    detect_aruco_marker,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BilletVision Calibration Wizard: ArUco / Checkerboard camera calibration."
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Capture source: webcam index ('0', '1'), image file path, or video file path.",
    )
    parser.add_argument(
        "--marker-size",
        type=float,
        default=50.0,
        help="Physical side length of the ArUco marker in millimetres (default: 50.0).",
    )
    parser.add_argument(
        "--marker-dict",
        type=str,
        default="DICT_4X4_50",
        help="ArUco dictionary name (default: DICT_4X4_50).",
    )
    parser.add_argument(
        "--marker-id",
        type=int,
        default=None,
        help="Specific ArUco marker ID to calibrate against (default: auto-select largest).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="config/calibration.json",
        help="Destination path for calibration JSON (default: config/calibration.json).",
    )
    parser.add_argument(
        "--checkerboard",
        type=str,
        default=None,
        help="Optional glob pattern or folder path of checkerboard images for camera distortion calibration.",
    )
    parser.add_argument(
        "--checkerboard-size",
        type=str,
        default="9x6",
        help="Checkerboard inner corners as COLUMNSxROWS (default: 9x6).",
    )
    parser.add_argument(
        "--checkerboard-square-mm",
        type=float,
        default=25.0,
        help="Checkerboard square size in millimetres (default: 25.0).",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Run headless without interactive window display.",
    )
    return parser.parse_args()


def draw_calibration_overlay(
    frame: np.ndarray,
    corners: Optional[np.ndarray],
    calib: Optional[CalibrationData],
    marker_dict: str,
    marker_size_mm: float,
) -> np.ndarray:
    """Render status overlay with marker boundary and scale metrics."""
    overlay = frame.copy()
    h, w = overlay.shape[:2]

    # Draw header background
    cv2.rectangle(overlay, (0, 0), (w, 65), (25, 25, 25), -1)
    cv2.line(overlay, (0, 65), (w, 65), (70, 70, 70), 1)

    if calib is not None and corners is not None:
        # Draw marker boundary
        pts = corners.astype(np.int32)
        cv2.polylines(overlay, [pts], True, (0, 255, 0), 2, cv2.LINE_AA)
        for i, pt in enumerate(pts):
            cv2.circle(overlay, tuple(pt), 4, (0, 165, 255) if i == 0 else (0, 255, 0), -1)

        # Text banner
        line1 = f"1 px = {calib.mm_per_pixel:.4f} mm ({calib.pixels_per_mm:.2f} px/mm)"
        err_str = f"{calib.reprojection_error:.4f} px" if calib.reprojection_error is not None else "N/A"
        line2 = f"Marker: {marker_size_mm:.1f} mm ({marker_dict}) | Reproj: {err_str} | Press 's' to save, 'q' to quit"

        cv2.putText(overlay, line1, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(overlay, line2, (15, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 1, cv2.LINE_AA)
    else:
        line1 = f"Searching for ArUco marker ({marker_dict}, {marker_size_mm:.1f} mm)..."
        line2 = "Place marker flat and unobstructed in camera view. Press 'q' to quit."
        cv2.putText(overlay, line1, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 215, 255), 2, cv2.LINE_AA)
        cv2.putText(overlay, line2, (15, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 180, 180), 1, cv2.LINE_AA)

    return overlay


def run_checkerboard_calibration(
    pattern_glob: str,
    pattern_str: str,
    square_size_mm: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Run checkerboard camera calibration from file glob."""
    paths = glob.glob(pattern_glob)
    if not paths and Path(pattern_glob).is_dir():
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
            paths.extend(glob.glob(str(Path(pattern_glob) / ext)))

    if not paths:
        raise FileNotFoundError(f"No checkerboard images found matching '{pattern_glob}'")

    parts = pattern_str.lower().split("x")
    pattern_size = (int(parts[0]), int(parts[1]))

    images = []
    for p in paths:
        img = cv2.imread(p)
        if img is not None:
            images.append(img)

    logger.info("Calibrating camera distortion with %d images...", len(images))
    return calibrate_camera_checkerboard(images, pattern_size, square_size_mm)


def main() -> int:
    args = parse_args()

    camera_matrix = None
    dist_coeffs = None
    if args.checkerboard:
        try:
            camera_matrix, dist_coeffs, reproj_err = run_checkerboard_calibration(
                args.checkerboard,
                args.checkerboard_size,
                args.checkerboard_square_mm,
            )
            logger.info("Camera calibration complete: reprojection error = %.4f px", reproj_err)
        except Exception as e:
            logger.error("Checkerboard camera calibration failed: %s", e)
            return 1

    source_path = Path(args.source)
    is_image_file = source_path.is_file() and source_path.suffix.lower() in {
        ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"
    }

    if is_image_file:
        frame = cv2.imread(str(source_path))
        if frame is None:
            logger.error("Could not read image from %s", source_path)
            return 1

        try:
            calib = calibrate_from_marker(
                frame,
                marker_size_mm=args.marker_size,
                marker_dict=args.marker_dict,
                marker_id=args.marker_id,
                camera_matrix=camera_matrix,
                dist_coeffs=dist_coeffs,
            )
        except Exception as e:
            logger.error("Calibration failed: %s", e)
            return 1

        # Required print format: prints '1 px = x mm'
        print(f"1 px = {calib.mm_per_pixel:.4f} mm")
        calib.save(args.output)
        print(f"Calibration saved to {args.output}")

        if not args.no_gui:
            corners = detect_aruco_marker(frame, marker_dict=args.marker_dict, marker_id=args.marker_id)
            overlay = draw_calibration_overlay(
                frame, corners, calib, args.marker_dict, args.marker_size
            )
            try:
                cv2.imshow("BilletVision Calibration", overlay)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            except Exception as e:
                logger.debug("GUI display skipped: %s", e)
        return 0

    # Webcam or Video source
    try:
        src = build_source(args.source)
        src.open()
    except Exception as e:
        logger.error("Failed to open capture source '%s': %s", args.source, e)
        return 1

    logger.info("Starting calibration wizard on source '%s'...", args.source)
    latest_calib: Optional[CalibrationData] = None

    try:
        while True:
            try:
                frame = src.read()
            except Exception:
                break

            corners = detect_aruco_marker(frame, marker_dict=args.marker_dict, marker_id=args.marker_id)
            if corners is not None:
                try:
                    latest_calib = calibrate_from_marker(
                        frame,
                        marker_size_mm=args.marker_size,
                        marker_dict=args.marker_dict,
                        marker_id=args.marker_id,
                        camera_matrix=camera_matrix,
                        dist_coeffs=dist_coeffs,
                    )
                except Exception:
                    pass

            if args.no_gui:
                if latest_calib is not None:
                    print(f"1 px = {latest_calib.mm_per_pixel:.4f} mm")
                    latest_calib.save(args.output)
                    print(f"Calibration saved to {args.output}")
                    break
                continue

            overlay = draw_calibration_overlay(
                frame, corners, latest_calib, args.marker_dict, args.marker_size
            )

            try:
                cv2.imshow("BilletVision Calibration Wizard", overlay)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("s") and latest_calib is not None:
                    print(f"1 px = {latest_calib.mm_per_pixel:.4f} mm")
                    latest_calib.save(args.output)
                    print(f"Calibration saved to {args.output}")
                    break
            except Exception as e:
                logger.debug("GUI error: %s", e)
                if latest_calib is not None:
                    print(f"1 px = {latest_calib.mm_per_pixel:.4f} mm")
                    latest_calib.save(args.output)
                    print(f"Calibration saved to {args.output}")
                    break

    finally:
        src.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
