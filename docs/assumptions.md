# BilletVision: Scope Decisions & Technical Assumptions

Based on PRD.pdf §5 and §12:

## 1. Input Sources
- The system supports 3 interchangeable sources through the unified `FrameSource` interface:
  1. Live USB/industrial webcam (OpenCV video capture)
  2. Pre-recorded MP4 video file (deterministic replay for demonstrations)
  3. Directory of static images (batch processing)

## 2. Test Material & Calibration
- Ground truth is calibrated against physical props (wooden/aluminium/steel bars or PVC blocks) measured with digital calipers.
- Camera calibration uses an ArUco marker (DICT_4X4_50) or checkerboard placed in the conveyor plane to compute pixels-per-millimetre and a homography matrix correcting for perspective distortion.

## 3. Dimensional Measurement
- Billets longer than the camera's FOV (e.g. 6–12 m in production) are measured via:
  $$\text{Length} = \text{Belt Speed} \times \text{Time in View}$$
  Direct FOV measurement is used for scaled test props.
- Square/rectangular billets output length, width, height, diagonal difference (rhomboidity), and camber.
- Round billets output diameter and ovality.

## 4. Hot Billets
- Production continuous casting lines operate at 900–1100 °C. Hot billets emit visible glow (bright-on-dark). BilletVision provides an intensity thresholding mode for glowing billets and accounts for thermal expansion factor (~1.2%).

## 5. Resilient Logging
- SQLite is the durable source of truth.
- CSV is continuously appended.
- XLSX is written to a temporary file and atomically swapped (`os.replace`). If an operator or engineer has the Excel file open, the system gracefully falls back to timestamped part files (`log_YYYYMMDD_partN.xlsx`) without blocking or losing data.

## 6. Implementation decisions (added during build)
- **Calibration scale only.** The marker gives mm/px; its homography is *not* applied unless `vision.use_homography: true`
  (one 100 px marker extrapolates poorly across the belt). A lens distortion model from a checkerboard is applied.
- **Measurement frames.** `direct` length mode only measures frames where the whole billet is inside the ROI;
  `belt_speed` mode also accepts clipped frames (width only) and times head/tail across the ROI centre line.
  Width is the side across the belt (`travel_axis: x`); height is not observable from a top view and equals width.
- **Pixel-centre bias.** Contour extents are one pixel short; `measure.py` adds 1 px per dimension.
- **Text notches.** Dark stamped text touching the billet edge is closed by a per-column envelope (conveyor along X).
- **Review semantics.** The engine grades 1–2x tolerance as REWORK, >2x as FAIL (both reject).
- **Duplicate IDs** only warn (alert + `duplicate_id` defect); they do not change the verdict.
- **Rotation.** CSV/XLSX roll over daily to `<name>_YYYYMMDD.<ext>`; SQLite is never rotated. If Excel holds the file at
  rollover it keeps writing to the live file and rolls on the next start.
- **Tolerance edits** apply live but are not written back to `tolerances.yaml`.
