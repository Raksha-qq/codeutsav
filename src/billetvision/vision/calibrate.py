"""Camera and perspective calibration: ArUco / checkerboard detection and mm/px calculation.

All units internally are millimetres (mm).
Coordinate frame:
  - Image coordinates: origin (0, 0) top-left, +X right, +Y down.
  - Fronto-parallel plane: rectified plane aligned with the conveyor surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Standard ArUco dictionary lookup
ARUCO_DICTIONARIES: Dict[str, int] = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
}


def px_to_mm(
    pixel_length: Union[float, int, np.ndarray], mm_per_px: float
) -> Union[float, np.ndarray]:
    """Convert length in pixels to millimetres.

    Args:
        pixel_length: Length in pixels (scalar or numpy array).
        mm_per_px: Millimetres per pixel scale factor.

    Returns:
        Converted length in millimetres.
    """
    return pixel_length * mm_per_px


def mm_to_px(
    mm_length: Union[float, int, np.ndarray], mm_per_px: float
) -> Union[float, np.ndarray]:
    """Convert length in millimetres to pixels.

    Args:
        mm_length: Length in millimetres (scalar or numpy array).
        mm_per_px: Millimetres per pixel scale factor.

    Returns:
        Converted length in pixels.
    """
    if mm_per_px <= 0:
        raise ValueError(f"mm_per_px must be positive, got {mm_per_px}")
    return mm_length / mm_per_px


# Aliases
pixels_to_mm = px_to_mm
mm_to_pixels = mm_to_px


@dataclass
class CalibrationData:
    """Represents camera calibration parameters, scale factor, and plane homography."""

    mm_per_pixel: float
    pixels_per_mm: float
    homography_matrix: Optional[Union[List[List[float]], np.ndarray]] = None
    camera_matrix: Optional[Union[List[List[float]], np.ndarray]] = None
    dist_coeffs: Optional[Union[List[float], np.ndarray]] = None
    marker_type: Optional[str] = "DICT_4X4_50"
    marker_size_mm: Optional[float] = 50.0
    calibrated_at: Optional[str] = None
    reprojection_error: Optional[float] = None

    @property
    def mm_per_px(self) -> float:
        """Alias for mm_per_pixel."""
        return self.mm_per_pixel

    @property
    def pixels_per_px(self) -> float:
        """Alias for pixels_per_mm."""
        return self.pixels_per_mm

    @property
    def px_per_mm(self) -> float:
        """Alias for pixels_per_mm."""
        return self.pixels_per_mm

    @property
    def timestamp(self) -> Optional[str]:
        """Alias for calibrated_at."""
        return self.calibrated_at

    def get_homography_matrix(self) -> Optional[np.ndarray]:
        """Return homography as a 3x3 float64 numpy array."""
        if self.homography_matrix is None:
            return None
        return np.asarray(self.homography_matrix, dtype=np.float64)

    def get_camera_matrix(self) -> Optional[np.ndarray]:
        """Return camera matrix as a 3x3 float64 numpy array."""
        if self.camera_matrix is None:
            return None
        return np.asarray(self.camera_matrix, dtype=np.float64)

    def get_dist_coeffs(self) -> Optional[np.ndarray]:
        """Return distortion coefficients as a 1D float64 numpy array."""
        if self.dist_coeffs is None:
            return None
        return np.asarray(self.dist_coeffs, dtype=np.float64).reshape(-1)

    def px_to_mm(
        self, pixel_length: Union[float, int, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Convert pixels to mm using this calibration."""
        return px_to_mm(pixel_length, self.mm_per_pixel)

    def mm_to_px(
        self, mm_length: Union[float, int, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Convert mm to pixels using this calibration."""
        return mm_to_px(mm_length, self.mm_per_pixel)

    def warp_and_undistort(
        self,
        frame: np.ndarray,
        output_size: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """Warp and undistort frame using this calibration."""
        return warp_undistort_frame(frame, self, output_size=output_size)

    def to_dict(self) -> Dict[str, Any]:
        """Convert calibration data to a JSON-serializable dictionary."""

        def _to_list(val: Any) -> Any:
            if val is None:
                return None
            if isinstance(val, np.ndarray):
                return val.tolist()
            if isinstance(val, list):
                return [
                    v.tolist() if isinstance(v, np.ndarray) else v for v in val
                ]
            return val

        return {
            "mm_per_pixel": float(self.mm_per_pixel),
            "pixels_per_mm": float(self.pixels_per_mm),
            "marker_type": self.marker_type,
            "marker_size_mm": (
                float(self.marker_size_mm)
                if self.marker_size_mm is not None
                else None
            ),
            "calibrated_at": self.calibrated_at,
            "homography_matrix": _to_list(self.homography_matrix),
            "camera_matrix": _to_list(self.camera_matrix),
            "dist_coeffs": _to_list(self.dist_coeffs),
            "reprojection_error": (
                float(self.reprojection_error)
                if self.reprojection_error is not None
                else None
            ),
        }

    def save(self, path: Union[str, Path]) -> None:
        """Save calibration data to JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "CalibrationData":
        """Load calibration data from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        mm_per_pixel = data.get("mm_per_pixel", data.get("mm_per_px"))
        if mm_per_pixel is None:
            raise KeyError("Missing 'mm_per_pixel' in calibration file")

        pixels_per_mm = data.get("pixels_per_mm", data.get("px_per_mm"))
        if pixels_per_mm is None:
            pixels_per_mm = 1.0 / float(mm_per_pixel)

        calibrated_at = data.get("calibrated_at", data.get("timestamp"))
        marker_size_mm = data.get("marker_size_mm", data.get("marker_size"))

        return cls(
            mm_per_pixel=float(mm_per_pixel),
            pixels_per_mm=float(pixels_per_mm),
            homography_matrix=data.get("homography_matrix"),
            camera_matrix=data.get("camera_matrix"),
            dist_coeffs=data.get("dist_coeffs"),
            marker_type=data.get("marker_type", "DICT_4X4_50"),
            marker_size_mm=(
                float(marker_size_mm) if marker_size_mm is not None else None
            ),
            calibrated_at=calibrated_at,
            reprojection_error=(
                float(data["reprojection_error"])
                if data.get("reprojection_error") is not None
                else None
            ),
        )


def save_calibration(
    calibration: CalibrationData,
    path: Union[str, Path] = "config/calibration.json",
) -> None:
    """Save CalibrationData to a JSON file."""
    calibration.save(path)


def load_calibration(
    path: Union[str, Path] = "config/calibration.json",
) -> CalibrationData:
    """Load CalibrationData from a JSON file."""
    return CalibrationData.load(path)


def get_aruco_dictionary(dict_name: str = "DICT_4X4_50") -> cv2.aruco.Dictionary:
    """Get OpenCV ArUco dictionary by name."""
    if dict_name in ARUCO_DICTIONARIES:
        dict_id = ARUCO_DICTIONARIES[dict_name]
    elif hasattr(cv2.aruco, dict_name):
        dict_id = getattr(cv2.aruco, dict_name)
    else:
        raise ValueError(
            f"Unknown ArUco dictionary: {dict_name}. Available: {list(ARUCO_DICTIONARIES.keys())}"
        )
    return cv2.aruco.getPredefinedDictionary(dict_id)


def detect_aruco_marker(
    image: np.ndarray,
    marker_dict: str = "DICT_4X4_50",
    marker_id: Optional[int] = None,
) -> Optional[np.ndarray]:
    """Detect ArUco marker in image and return its 4 corners.

    Args:
        image: Input image (BGR or grayscale).
        marker_dict: ArUco dictionary name.
        marker_id: Specific marker ID to look for. If None, selects the marker with
            the largest perimeter.

    Returns:
        np.ndarray of shape (4, 2) with corner coordinates [TL, TR, BR, BL],
        or None if no matching marker was found.
    """
    if image is None or image.size == 0:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    dictionary = get_aruco_dictionary(marker_dict)
    parameters = cv2.aruco.DetectorParameters()

    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        corners_list, ids, _ = detector.detectMarkers(gray)
    else:
        corners_list, ids, _ = cv2.aruco.detectMarkers(
            gray, dictionary, parameters=parameters
        )

    if ids is None or len(corners_list) == 0:
        return None

    ids_flat = ids.flatten().tolist()

    if marker_id is not None:
        if marker_id not in ids_flat:
            return None
        idx = ids_flat.index(marker_id)
        return corners_list[idx].reshape(4, 2).astype(np.float32)

    # If multiple markers, pick the one with largest perimeter
    best_idx = 0
    best_perimeter = -1.0
    for idx, c in enumerate(corners_list):
        pts = c.reshape(4, 2)
        perimeter = cv2.arcLength(pts, True)
        if perimeter > best_perimeter:
            best_perimeter = perimeter
            best_idx = idx

    return corners_list[best_idx].reshape(4, 2).astype(np.float32)


def compute_marker_homography(
    corners: np.ndarray,
    marker_size_mm: float = 50.0,
    target_px_per_mm: Optional[float] = None,
    target_center: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, float, float, float]:
    """Compute fronto-parallel homography and scale factors from ArUco corners.

    Args:
        corners: (4, 2) array of marker corner coordinates [TL, TR, BR, BL].
        marker_size_mm: Physical side length of marker in mm.
        target_px_per_mm: Optional target scale factor in rectified view.
        target_center: Optional (cx, cy) center coordinate in rectified view.

    Returns:
        Tuple of (homography_3x3, mm_per_px, px_per_mm, reprojection_error).
    """
    c = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    sides = [float(np.linalg.norm(c[(i + 1) % 4] - c[i])) for i in range(4)]
    avg_side_px = float(np.mean(sides))

    if target_px_per_mm is not None and target_px_per_mm > 0:
        px_per_mm = float(target_px_per_mm)
        mm_per_px = 1.0 / px_per_mm
        rect_side_px = marker_size_mm * px_per_mm
    else:
        rect_side_px = avg_side_px
        mm_per_px = marker_size_mm / rect_side_px
        px_per_mm = 1.0 / mm_per_px

    if target_center is not None:
        cx, cy = float(target_center[0]), float(target_center[1])
    else:
        cx, cy = float(np.mean(c[:, 0])), float(np.mean(c[:, 1]))

    half = rect_side_px / 2.0
    dst_pts = np.float32(
        [
            [cx - half, cy - half],  # Top-Left
            [cx + half, cy - half],  # Top-Right
            [cx + half, cy + half],  # Bottom-Right
            [cx - half, cy + half],  # Bottom-Left
        ]
    )

    H, _ = cv2.findHomography(c, dst_pts)
    if H is None:
        raise ValueError("Failed to compute homography matrix from corners")

    # Reprojection error of the 4 marker corners
    proj = cv2.perspectiveTransform(c.reshape(-1, 1, 2), H).reshape(-1, 2)
    reproj_err = float(np.sqrt(np.mean(np.sum((proj - dst_pts) ** 2, axis=1))))

    return H, mm_per_px, px_per_mm, reproj_err


def calibrate_from_marker(
    image: np.ndarray,
    marker_size_mm: float = 50.0,
    marker_dict: str = "DICT_4X4_50",
    marker_id: Optional[int] = None,
    camera_matrix: Optional[np.ndarray] = None,
    dist_coeffs: Optional[np.ndarray] = None,
    target_px_per_mm: Optional[float] = None,
    target_center: Optional[Tuple[float, float]] = None,
) -> CalibrationData:
    """Calibrate camera perspective and scale from an image with an ArUco marker.

    Args:
        image: Frame containing ArUco marker.
        marker_size_mm: Known side length of marker in mm.
        marker_dict: ArUco dictionary name.
        marker_id: Optional marker ID.
        camera_matrix: Optional 3x3 camera intrinsic matrix.
        dist_coeffs: Optional distortion coefficients.
        target_px_per_mm: Optional target pixels per mm in rectified view.
        target_center: Optional center coordinate in rectified view.

    Returns:
        CalibrationData object.

    Raises:
        ValueError: If marker is not detected.
    """
    img_to_detect = image
    if camera_matrix is not None and dist_coeffs is not None:
        dist_arr = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1)
        if np.any(dist_arr != 0):
            img_to_detect = cv2.undistort(
                image,
                np.asarray(camera_matrix, dtype=np.float64),
                dist_arr,
            )

    corners = detect_aruco_marker(
        img_to_detect, marker_dict=marker_dict, marker_id=marker_id
    )
    if corners is None:
        raise ValueError(
            f"No ArUco marker ({marker_dict}) detected in frame. "
            "Ensure marker is visible, well-lit, and surrounded by a white margin."
        )

    H, mm_per_px, px_per_mm, reproj_err = compute_marker_homography(
        corners=corners,
        marker_size_mm=marker_size_mm,
        target_px_per_mm=target_px_per_mm,
        target_center=target_center,
    )

    now_iso = datetime.now(timezone.utc).isoformat()

    return CalibrationData(
        mm_per_pixel=mm_per_px,
        pixels_per_mm=px_per_mm,
        homography_matrix=H.tolist(),
        camera_matrix=(
            camera_matrix.tolist()
            if isinstance(camera_matrix, np.ndarray)
            else camera_matrix
        ),
        dist_coeffs=(
            dist_coeffs.tolist()
            if isinstance(dist_coeffs, np.ndarray)
            else dist_coeffs
        ),
        marker_type=marker_dict,
        marker_size_mm=marker_size_mm,
        calibrated_at=now_iso,
        reprojection_error=reproj_err,
    )


def calibrate_camera_checkerboard(
    images: List[np.ndarray],
    pattern_size: Tuple[int, int] = (9, 6),
    square_size_mm: float = 25.0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Calibrate camera intrinsics and distortion using checkerboard images.

    Args:
        images: List of images containing the checkerboard pattern.
        pattern_size: Inner corner count as (columns, rows).
        square_size_mm: Physical side length of each checkerboard square in mm.

    Returns:
        Tuple of (camera_matrix, dist_coeffs, reprojection_error).

    Raises:
        ValueError: If no valid checkerboard corners could be detected.
    """
    if not images:
        raise ValueError("No images provided for checkerboard calibration")

    # 3D points in real world space (Z = 0)
    num_corners = pattern_size[0] * pattern_size[1]
    objp = np.zeros((num_corners, 3), np.float32)
    objp[:, :2] = (
        np.mgrid[0 : pattern_size[0], 0 : pattern_size[1]].T.reshape(-1, 2)
        * square_size_mm
    )

    objpoints: List[np.ndarray] = []
    imgpoints: List[np.ndarray] = []
    image_size: Optional[Tuple[int, int]] = None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for img in images:
        if img is None or img.size == 0:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])

        ret, corners = cv2.findChessboardCorners(
            gray,
            pattern_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        if not ret:
            ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        if ret and corners is not None:
            corners_sub = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria
            )
            objpoints.append(objp)
            imgpoints.append(corners_sub)

    if not objpoints or image_size is None:
        raise ValueError(
            f"Checkerboard corners {pattern_size} not detected in any of the {len(images)} images."
        )

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )

    # Compute RMS reprojection error
    total_error_sq = 0.0
    total_points = 0
    for i in range(len(objpoints)):
        proj, _ = cv2.projectPoints(
            objpoints[i], rvecs[i], tvecs[i], mtx, dist
        )
        pts_obs = imgpoints[i].reshape(-1, 2).astype(np.float64)
        pts_proj = proj.reshape(-1, 2).astype(np.float64)
        err_sq = np.sum((pts_obs - pts_proj) ** 2, axis=1)
        total_error_sq += float(np.sum(err_sq))
        total_points += len(pts_obs)

    rms_error = (
        float(np.sqrt(total_error_sq / total_points))
        if total_points > 0
        else float(ret)
    )
    return mtx, dist.ravel(), rms_error


def warp_undistort_frame(
    frame: np.ndarray,
    calibration: CalibrationData,
    output_size: Optional[Tuple[int, int]] = None,
    interpolation: int = cv2.INTER_LINEAR,
    border_mode: int = cv2.BORDER_CONSTANT,
    border_value: Union[int, Tuple[int, ...]] = 0,
) -> np.ndarray:
    """Undistort and perspective-warp a frame to fronto-parallel view.

    Args:
        frame: Input frame image.
        calibration: CalibrationData instance.
        output_size: Optional (width, height) of destination frame. Defaults to
            frame's original size.
        interpolation: OpenCV interpolation flag.
        border_mode: OpenCV border mode.
        border_value: Border fill value.

    Returns:
        Transformed (undistorted and rectified) frame.
    """
    if frame is None or frame.size == 0:
        return frame

    result = frame

    # 1. Undistort camera lens distortion if parameters exist and non-zero
    cam_mat = calibration.get_camera_matrix()
    dist_coeffs = calibration.get_dist_coeffs()
    if cam_mat is not None and dist_coeffs is not None:
        if np.any(dist_coeffs != 0):
            result = cv2.undistort(result, cam_mat, dist_coeffs)

    # 2. Warp perspective to fronto-parallel view
    H = calibration.get_homography_matrix()
    if H is not None:
        if output_size is None:
            output_size = (result.shape[1], result.shape[0])
        result = cv2.warpPerspective(
            result,
            H,
            output_size,
            flags=interpolation,
            borderMode=border_mode,
            borderValue=border_value,
        )

    return result
