# PRD: BilletVision — Automated Billet Inspection & Traceability
**Event:** Code Utsava X.0 (Turing Club of Programmers) | **Problem Statement:** 1 | **Duration:** 28 hours | **Version:** 1.0 | **Date:** 3 Oct 2026

---

## 1. One-line pitch
A non-contact computer-vision system that watches billets pass on a conveyor, measures them to within ±1%, reads their heat/batch markings, flags out-of-tolerance pieces instantly, and writes an audit-ready Excel/CSV log, with no human near the hot line.

---

## 2. Problem recap (from PS 1)
Manual billet inspection is slow, error-prone, dangerous near hot steel, and creates bad records. We need: live capture, dimension + defect measurement, OCR/barcode reading, automated Excel logging, instant tolerance alerts, and an operator dashboard.

---

## 3. Goals, non-goals, success metrics

### Goals (must hit by hour 28)
| # | Goal | Target (from PS) |
|---|------|------------------|
| G1 | Dimensional accuracy after calibration | ≤ ±1% error vs. ground truth |
| G2 | Speed | ≥ 15 FPS live, or < 2 s inspection latency per billet |
| G3 | OCR / ID reading | ≥ 90% on test set of stamped/painted/printed IDs |
| G4 | Logging | Append-mode Excel/CSV, no corruption, no file-lock crash during continuous run |
| G5 | Alerts | Out-of-tolerance → visual + audio alert within 1 s of decision |
| G6 | Robustness | Works under varied lighting, dust/noise, mild scale |
| G7 | Dashboard | Live feed, bounding boxes, measurements, alert panel, log table |

### Non-goals
- No claim of production deployment on a real hot mill. We prototype on video / scaled props.
- No custom-trained detector on thousands of real billet images (no public dataset to rely on).
- No PLC/MES integration (we show an export + REST hook as the integration path).

---

## 4. Users & scenarios
- **Line operator:** watches dashboard, hears alert, diverts the billet.
- **QA engineer:** reviews pass/fail history, filters by heat number, exports audit logs.
- **Plant manager:** looks at yield / rejection trends.

**Primary scenario:** billet enters the field of view → trigger → N frames captured → measured, ID read, verdict computed → row appended to log → dashboard updates → alert if FAIL.

---

## 5. Scope decisions and honest assumptions
| Question | Decision | Why |
|---|---|---|
| Input source | Support 3 sources behind one interface: (a) webcam, (b) video file, (c) image folder | Demo never depends on one thing working |
| Test material | Phone-recorded video of scaled billet props (wooden/aluminium/steel bars, PVC or metal blocks) moving on a belt or hand-slid on a table, with stamped/printed ID labels, plus any hot-billet / steel-bar footage you can find for the robustness slide | Nobody has a mill nearby; props give you real ground truth via calipers |
| Dimension model | Square/rectangular and round billet modes (config switch) | PS says "width, height/diameter" |
| Length of real billets (6–12 m) | One camera can't see that. Real answer = line-scan camera or multi-camera stitching or length from belt-speed × time-in-view. We implement time-in-view × calibrated belt speed as a length mode + direct measurement for props within the FOV | Shows you understand the real constraint |
| Calibration | Reference object of known size in the scene (ArUco marker or checkerboard) → pixels-per-mm; homography for perspective correction | Core to the ±1% claim |
| Hot billets | Glow → bright-on-dark; use intensity thresholding mode + ND/exposure lock; note IR/thermal camera as production path | Keeps it realistic |

---

## 6. System architecture
```
Camera / Video / Folder
│
[Capture thread] ──► frame queue (drop-oldest, keeps FPS up)
│
[Vision worker]
1. Preprocess (undistort, exposure normalise, CLAHE)
2. Segment billet (classical: threshold/Otsu/adaptive + morphology + contours; optional YOLOv8-seg)
3. Trigger / tracking (billet fully in ROI; centroid tracker → one ID per billet, no double count)
4. Measure (minAreaRect / ellipse fit → mm via calibration; ovality, diagonal diff, edge straightness)
5. Defect check (surface anomaly score, cross-section deviation, bend)
6. ID read (ROI crop → enhance → OCR + QR/barcode → regex validate → multi-frame vote)
7. Decision engine (tolerance config → PASS / FAIL / REWORK / REVIEW)
│
[Event bus]
├─► Logger (SQLite truth → CSV append → Excel writer thread, atomic)
├─► Alert manager (UI banner, sound, optional Telegram/webhook)
└─► WebSocket → Dashboard
│
[FastAPI backend] ◄──► [Operator dashboard (browser)]
```

### Tech stack
- Python 3.11/3.12, OpenCV, NumPy, scikit-image
- OCR: PaddleOCR (best on stamped/odd fonts) with EasyOCR fallback, Tesseract as last resort; pyzbar / cv2.QRCodeDetector for codes
- Optional DL: Ultralytics YOLOv8n-seg (if needed; classical CV is default)
- Backend: FastAPI + Uvicorn, WebSocket for live events, MJPEG endpoint for the video feed
- Storage: SQLite (source of truth) → pandas + openpyxl / CSV export
- Frontend: single-page HTML + vanilla JS
- Tests: pytest

---

## 7. Functional requirements
- **FR-1:** Pluggable source (webcam index / video path / folder) with FPS counter.
- **FR-2:** Bounded queue; if processing lags, drop frames, never grow latency.
- **FR-3:** Calibration wizard: detect ArUco/checkerboard → compute mm/px → save calibration.json; re-calibrate from UI.
- **FR-4:** Segmentation robust to lighting: adaptive threshold + background subtraction fallback + auto exposure/gamma normalisation.
- **FR-5:** Dimension outputs: length, width, height/diameter (mm), plus derived: ovality, diagonal difference (rhomboidity), straightness/camber.
- **FR-6:** Defect flags: cross-sectional variation (profile width sampled along length, flag if std/max-min exceeds limit), edge irregularity, surface anomaly.
- **FR-7:** Tracking so each physical billet yields exactly one record, using median of N good frames.
- **FR-8:** ID-region localisation (text detector or fixed ROI per config).
- **FR-9:** Enhancement pipeline: grayscale → CLAHE → denoise → sharpen → deskew → upscale.
- **FR-10:** Barcode/QR decode in parallel; QR wins if present.
- **FR-11:** Format validation: configurable regex (`^[A-Z]\d{5,7}$`) and charset correction (`O↔0, I↔1, S↔5, B↔8`).
- **FR-12:** Multi-frame voting across billet's track → single confident ID; if confidence < threshold, status = REVIEW.
- **FR-13:** Write path: event → SQLite (instant, durable) → in-memory queue → single writer thread appends to CSV and XLSX.
- **FR-14:** No corruption / no lock crash: XLSX written to temp file then atomically replaced; fallback to part files if locked.
- **FR-15:** Daily rotation, header auto-create, duplicate-ID warning, and export (CSV / XLSX / JSON).
- **FR-16:** Live video with bounding boxes + overlaid mm values + ID text.
- **FR-17:** Big status tile (PASS green / FAIL red flashing) + audio beep + alert history; optional Telegram/webhook push.
- **FR-18:** Tolerance editor in UI, applied live.
- **FR-19:** Log table (last 100, filter by status/heat), KPIs: total, pass rate, FPS, avg latency, OCR success rate.
- **FR-20:** Review queue: REVIEW items show crop; operator types right ID, saved back to log.
- **FR-21:** Drill-down on any billet: frames, measurements, decision reasoning.

---

## 8. Log Schema
Columns:
`timestamp, billet_seq, billet_id, batch_id, length_mm, width_mm, height_mm, diameter_mm, ovality, diag_diff_mm, defects, ocr_confidence, status, fail_reasons, image_path, processing_ms`
Status: `PASS | FAIL | REWORK | REVIEW`
