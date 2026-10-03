"""Unit tests for camera calibration, ArUco detection, homography rectification, and CLI."""

from pathlib import Path
import subprocess
import sys
from typing import Tuple
import cv2
import numpy as np
import pytest

from billetvision.vision.calibrate import (
    CalibrationData,
    calibrate_camera_checkerboard,
    calibrate_from_marker,
    compute_marker_homography,
    detect_aruco_marker,
    get_aruco_dictionary,
    load_calibration,
    mm_to_pixels,
    mm_to_px,
    pixels_to_mm,
    px_to_mm,
    save_calibration,
    warp_undistort_frame,
)


@pytest.fixture
def synthetic_marker_image() -> Tuple[np.ndarray, float, float]:
    """Generate a clean synthetic image containing an ArUco marker at known scale.
    
    Returns:
        Tuple of (image, marker_size_mm, mm_per_px_ground_truth).
    """
    canvas_size = (1000, 1000)
    img = np.full(canvas_size, 255, dtype=np.uint8)

    # 50.0 mm marker rendered at 200x200 px -> 0.25 mm/px
    marker_size_mm = 50.0
    marker_px = 200
    d = get_aruco_dictionary("DICT_4X4_50")
    marker_img = cv2.aruco.generateImageMarker(d, 0, marker_px)

    pos_x, pos_y = 400, 400
    img[pos_y : pos_y + marker_px, pos_x : pos_x + marker_px] = marker_img
    ground_truth_mm_per_px = marker_size_mm / marker_px

    return img, marker_size_mm, ground_truth_mm_per_px


def test_aruco_detection_and_perspective_rectification_accuracy(synthetic_marker_image):
    """Test ArUco marker detection under perspective warp and assert mm error < 0.5%."""
    img, marker_size_mm, gt_mm_per_px = synthetic_marker_image
    h, w = img.shape[:2]

    # Apply perspective distortion simulating camera tilt
    src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst_pts = np.float32([[80, 50], [w - 80, 100], [w - 20, h - 80], [20, h - 120]])
    H_cam = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, H_cam, (w, h), borderValue=255)

    # Calibrate from the warped camera frame
    calib = calibrate_from_marker(warped, marker_size_mm=marker_size_mm, marker_dict="DICT_4X4_50")
    assert calib.mm_per_pixel > 0
    assert calib.pixels_per_mm > 0
    assert calib.homography_matrix is not None

    # Warp back to fronto-parallel view
    rectified = warp_undistort_frame(warped, calib)

    # Detect marker in rectified view and measure dimensions
    rect_corners = detect_aruco_marker(rectified, "DICT_4X4_50")
    assert rect_corners is not None, "Marker must be detected in rectified frame"

    # Compute measured side lengths of rectified marker
    sides = [
        float(np.linalg.norm(rect_corners[(i + 1) % 4] - rect_corners[i]))
        for i in range(4)
    ]
    measured_px = float(np.mean(sides))
    measured_mm = px_to_mm(measured_px, calib.mm_per_pixel)

    # Assert mm error < 0.5% (requirement: error < 0.5%)
    mm_error_pct = abs(measured_mm - marker_size_mm) / marker_size_mm * 100.0
    print(f"Measured: {measured_mm:.4f} mm, Ground Truth: {marker_size_mm:.1f} mm, Error: {mm_error_pct:.4f}%")
    assert mm_error_pct < 0.5, f"Expected mm error < 0.5%, got {mm_error_pct:.4f}%"


def test_px_to_mm_and_mm_to_px():
    """Test scalar and vector conversion between pixels and millimetres."""
    mm_per_px = 0.25

    # Scalar checks
    assert px_to_mm(100.0, mm_per_px) == 25.0
    assert mm_to_px(25.0, mm_per_px) == 100.0
    assert pixels_to_mm(200.0, mm_per_px) == 50.0
    assert mm_to_pixels(50.0, mm_per_px) == 200.0

    # Array checks
    px_arr = np.array([100.0, 200.0, 400.0])
    mm_arr = px_to_mm(px_arr, mm_per_px)
    assert np.allclose(mm_arr, [25.0, 50.0, 100.0])
    assert np.allclose(mm_to_px(mm_arr, mm_per_px), px_arr)

    # Error handling
    with pytest.raises(ValueError):
        mm_to_px(10.0, 0.0)
    with pytest.raises(ValueError):
        mm_to_px(10.0, -0.5)


def test_calibration_data_methods():
    """Test methods on CalibrationData dataclass."""
    calib = CalibrationData(
        mm_per_pixel=0.20,
        pixels_per_mm=5.0,
        homography_matrix=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        camera_matrix=[[1000.0, 0.0, 500.0], [0.0, 1000.0, 500.0], [0.0, 0.0, 1.0]],
        dist_coeffs=[0.0, 0.0, 0.0, 0.0, 0.0],
    )
    assert calib.mm_per_px == 0.20
    assert calib.px_per_mm == 5.0
    assert calib.px_to_mm(50.0) == 10.0
    assert calib.mm_to_px(10.0) == 50.0

    H = calib.get_homography_matrix()
    assert isinstance(H, np.ndarray)
    assert H.shape == (3, 3)

    K = calib.get_camera_matrix()
    assert isinstance(K, np.ndarray)
    assert K.shape == (3, 3)

    D = calib.get_dist_coeffs()
    assert isinstance(D, np.ndarray)
    assert len(D) == 5


def test_save_and_load_calibration_json(tmp_path: Path):
    """Test saving and loading config/calibration.json with full schema."""
    calib_orig = CalibrationData(
        mm_per_pixel=0.21,
        pixels_per_mm=4.76190476,
        homography_matrix=[[1.0, 0.1, 0.0], [-0.1, 1.0, 0.0], [0.0, 0.0, 1.0]],
        camera_matrix=[[1000.0, 0.0, 640.0], [0.0, 1000.0, 360.0], [0.0, 0.0, 1.0]],
        dist_coeffs=[0.01, -0.02, 0.0, 0.0, 0.0],
        marker_type="DICT_4X4_50",
        marker_size_mm=50.0,
        calibrated_at="2026-10-03T12:00:00Z",
        reprojection_error=0.08,
    )

    save_path = tmp_path / "calibration.json"
    calib_orig.save(save_path)
    assert save_path.exists()

    calib_loaded = CalibrationData.load(save_path)
    assert np.isclose(calib_loaded.mm_per_pixel, 0.21)
    assert np.isclose(calib_loaded.pixels_per_mm, 4.76190476)
    assert calib_loaded.marker_type == "DICT_4X4_50"
    assert calib_loaded.marker_size_mm == 50.0
    assert calib_loaded.calibrated_at == "2026-10-03T12:00:00Z"
    assert np.isclose(calib_loaded.reprojection_error, 0.08)
    assert np.allclose(calib_loaded.homography_matrix, calib_orig.homography_matrix)
    assert np.allclose(calib_loaded.camera_matrix, calib_orig.camera_matrix)
    assert np.allclose(calib_loaded.dist_coeffs, calib_orig.dist_coeffs)


def test_load_existing_workspace_calibration():
    """Verify that existing config/calibration.json loads seamlessly."""
    config_file = Path("config/calibration.json")
    if config_file.exists():
        calib = load_calibration(config_file)
        assert calib.mm_per_pixel > 0
        assert calib.pixels_per_mm > 0
        assert calib.get_homography_matrix() is not None
        assert calib.get_camera_matrix() is not None


def test_warp_undistort_frame():
    """Test warp_undistort_frame with identity, perspective homography, and distortion."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[25:75, 25:75] = 255

    # 1. Identity calibration
    calib_ident = CalibrationData(
        mm_per_pixel=1.0,
        pixels_per_mm=1.0,
        homography_matrix=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    )
    warped_ident = warp_undistort_frame(img, calib_ident)
    assert np.array_equal(img, warped_ident)

    # 2. Translation homography (+10 px in X)
    calib_shift = CalibrationData(
        mm_per_pixel=1.0,
        pixels_per_mm=1.0,
        homography_matrix=[[1.0, 0.0, 10.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    )
    warped_shift = warp_undistort_frame(img, calib_shift)
    assert warped_shift[50, 35, 0] == 255
    assert warped_shift[50, 25, 0] == 0  # original white region shifted


def test_checkerboard_camera_calibration():
    """Test optional camera calibration with cv2.calibrateCamera on synthetic checkerboard."""
    pattern_size = (7, 6)
    square_size_mm = 25.0
    squares_x, squares_y = 8, 7
    sq_px = 50

    # Draw planar checkerboard
    board = np.full((squares_y * sq_px, squares_x * sq_px), 255, dtype=np.uint8)
    for r in range(squares_y):
        for c in range(squares_x):
            if (r + c) % 2 == 0:
                board[r * sq_px : (r + 1) * sq_px, c * sq_px : (c + 1) * sq_px] = 0

    scale = square_size_mm / sq_px
    board_corners_3d = np.float32([
        [0, 0, 0],
        [squares_x * sq_px * scale, 0, 0],
        [squares_x * sq_px * scale, squares_y * sq_px * scale, 0],
        [0, squares_y * sq_px * scale, 0],
    ])

    K_synth = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    dist_synth = np.zeros(5, dtype=np.float64)

    poses = [
        (np.array([0.10, -0.10, 0.05]), np.array([-50.0, -40.0, 700.0])),
        (np.array([-0.12, 0.10, -0.05]), np.array([40.0, -30.0, 750.0])),
        (np.array([0.05, 0.12, 0.08]), np.array([-20.0, 30.0, 720.0])),
        (np.array([-0.08, -0.12, -0.04]), np.array([30.0, 40.0, 680.0])),
    ]

    images = []
    src = np.float32([
        [0, 0],
        [board.shape[1], 0],
        [board.shape[1], board.shape[0]],
        [0, board.shape[0]],
    ])

    for rvec, tvec in poses:
        proj_board, _ = cv2.projectPoints(board_corners_3d, rvec, tvec, K_synth, dist_synth)
        proj_board = proj_board.reshape(-1, 2)
        H = cv2.getPerspectiveTransform(src, proj_board.astype(np.float32))
        warped = cv2.warpPerspective(board, H, (640, 480), borderValue=255)
        images.append(warped)

    mtx, dist, reproj_err = calibrate_camera_checkerboard(
        images, pattern_size=pattern_size, square_size_mm=square_size_mm
    )

    assert mtx.shape == (3, 3)
    assert mtx[0, 0] > 0 and mtx[1, 1] > 0  # focal lengths positive
    assert len(dist) >= 4
    assert reproj_err < 1.0  # sub-pixel reprojection error


def test_calibrate_missing_marker_raises():
    """Test that calibrate_from_marker raises ValueError on images with no ArUco marker."""
    blank = np.full((400, 400, 3), 255, dtype=np.uint8)
    assert detect_aruco_marker(blank) is None
    with pytest.raises(ValueError, match="No ArUco marker"):
        calibrate_from_marker(blank, marker_size_mm=50.0)


def test_cli_calibrate_image(tmp_path: Path, synthetic_marker_image):
    """Test scripts/calibrate.py CLI with an image source in headless mode."""
    img, marker_size_mm, _ = synthetic_marker_image
    img_path = tmp_path / "test_marker.png"
    out_json = tmp_path / "output_calib.json"
    cv2.imwrite(str(img_path), img)

    cmd = [
        sys.executable,
        "scripts/calibrate.py",
        "--source",
        str(img_path),
        "--output",
        str(out_json),
        "--marker-size",
        str(marker_size_mm),
        "--no-gui",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)

    assert result.returncode == 0
    # Must print '1 px = x mm'
    assert "1 px = " in result.stdout
    assert " mm" in result.stdout
    assert out_json.exists()

    loaded = CalibrationData.load(out_json)
    assert loaded.mm_per_pixel > 0
    assert loaded.pixels_per_mm > 0
