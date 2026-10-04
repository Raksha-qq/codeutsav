# CLAUDE.md — BilletVision

Project context for Claude Code. Read this and `PRD.md` before doing anything.

## What this project is
BilletVision is a 28-hour hackathon prototype (Code Utsava X.0, Problem Statement 1). It is a non-contact computer-vision system that watches steel billets on a conveyor, measures their dimensions, reads their ID markings (OCR / QR / barcode), decides PASS / FAIL / REWORK / REVIEW against tolerances, alerts the operator, and logs every record to SQLite, CSV and Excel.

Time is extremely limited. **Prefer working, simple, demoable code over elegant abstractions.** The demo must never crash.

## Hard requirements (from the problem statement)
- Dimensional error ≤ ±1% after calibration
- ≥ 15 FPS live, or < 2 s latency per billet
- OCR ≥ 90% exact match on the test set
- Append-mode Excel/CSV logging with no corruption or file-lock crash during continuous scanning
- Instant alert when out of tolerance
- Operator dashboard: live feed, bounding boxes, measurements, alerts

## Tech stack
- Python 3.11, OpenCV, NumPy, scikit-image
- OCR: PaddleOCR (primary), EasyOCR (fallback), `pyzbar` / `cv2.QRCodeDetector`
- Backend: FastAPI + Uvicorn (WebSocket for events, MJPEG for video)
- Storage: SQLite (source of truth) → CSV + XLSX (`pandas`, `openpyxl`)
- Frontend: single-page HTML + vanilla JS in `web/` (no build step)
- Tests: pytest
- Optional only if classical CV fails: Ultralytics YOLOv8n-seg

Do not add new heavy dependencies without asking. Pin versions in `requirements.txt`.

## Commands
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/run_demo.py --source data/raw/demo.mp4  # run the whole pipeline + dashboard
uvicorn billetvision.api.main:app --reload --port 8000  # API/dashboard only
pytest -q                                               # run all tests
python scripts/calibrate.py --source webcam             # calibration wizard
python scripts/accuracy_report.py                       # writes docs/accuracy_report.md
python scripts/ocr_eval.py                              # OCR accuracy on data/ocr_testset
python scripts/soak_test.py --minutes 30                # logging integrity soak test
```
If a command above does not exist yet, create it as part of the relevant task.

## Repo layout
```
config/            config.yaml, tolerances.yaml, billet_profiles.yaml, calibration.json
src/billetvision/
  capture/         FrameSource (webcam | video | folder), bounded drop-oldest queue
  vision/          preprocess, segment, measure, defects, track, calibrate
  ocr/             enhance, reader, codes, validate, vote
  decision/        engine.py  (tolerances -> verdict + reasons)
  logging_/        db.py, csv_writer.py, xlsx_writer.py
  alerts/          manager.py, telegram.py
  api/             main.py, ws.py, mjpeg.py
  pipeline.py      wires everything
web/               dashboard (index.html, app.js, styles.css)
data/              raw/, ground_truth.csv, ocr_testset/, outputs/
tests/  scripts/  docs/
```

## Architecture rules (do not break these)
1. **One pipeline, pluggable sources.** All input goes through the `FrameSource` interface. The video-file source must always work, since it is the demo's safety net.
2. **Never block the capture thread.** Use a bounded queue that drops the oldest frame. Latency must not grow.
3. **One record per physical billet.** Use the tracker plus a median of the best-N sharpest frames. Never log from a single noisy frame.
4. **Single writer for logs.** All log writes go through one writer thread. SQLite is the source of truth; CSV is always appended; XLSX is written to a temp file and swapped in with `os.replace`. On `PermissionError` (Excel open), fall back to a part file (`log_YYYYMMDD_partN.xlsx`) and never crash or drop the record.
5. **Decision engine is pure.** `decision/engine.py` takes a `Measurement` and tolerances and returns `Verdict(status, reasons[])`. No I/O inside it. It must be fully unit-tested.
6. **Every verdict is explainable.** FAIL and REVIEW always carry human-readable reasons, e.g. `width 131.8 mm > 130.0 ± 1.0 mm`, plus the saved annotated image path.
7. **Uncertain OCR goes to REVIEW.** If OCR confidence is below the threshold or the regex check fails, never write a guessed ID silently. Mark REVIEW and save the crop.
8. **All units are millimetres** internally. Convert pixel to mm in exactly one place (`vision/calibrate.py`). Never scatter magic scale constants.
9. **Config over constants.** Tolerances, regexes, thresholds, and billet profiles live in `config/*.yaml`.
10. **No real side effects in tests/demo:** Telegram or webhook sending is disabled unless env vars are set.

## Measurement conventions
- Segmentation is classical first: grayscale → CLAHE/gamma normalise → Otsu/adaptive threshold → morphology → largest valid contour. Hot-billet mode uses brightness thresholding.
- Rectangular billets: `cv2.minAreaRect` → length, width, height. Round billets: `cv2.fitEllipse` → diameter, ovality.
- Derived metrics: ovality, diagonal difference (rhomboidity), straightness/camber, cross-section profile variation along the length.
- Length for pieces longer than the FOV = calibrated belt speed × time-in-view (config flag). For pieces within the FOV, measure directly.
- Calibration: ArUco marker (or checkerboard) of known size → mm/px and homography, stored in `config/calibration.json`.

## Log schema (CSV / XLSX / SQLite table `records`)
`timestamp, billet_seq, billet_id, batch_id, length_mm, width_mm, height_mm, diameter_mm, ovality, diag_diff_mm, defects, ocr_confidence, status, fail_reasons, image_path, processing_ms`
`status` ∈ `PASS | FAIL | REWORK | REVIEW`. Do not change columns without updating the writers, API, dashboard, and tests together.

## Coding conventions
- Type hints everywhere; dataclasses (or pydantic) for `Measurement`, `OcrResult`, `Verdict`, `Record`.
- Small pure functions; no global mutable state outside `pipeline.py`.
- Use the `logging` module (no bare `print` in `src/`).
- Handle errors at boundaries (camera lost, OCR exception, file locked), log them, and continue running.
- Format with `ruff`/`black` defaults. Keep functions under about 50 lines where reasonable.
- Add a docstring that states units and coordinate frames for any function dealing with geometry.

## Testing rules
- Every module gets at least one pytest. Vision tests use **synthetic images** (draw a rectangle of known pixel size and verify mm output within 1%).
- Required tests: decision engine (boundaries, ties, multiple failures), writer concurrency (many threads submit, zero lost rows), lock fallback (simulate `PermissionError`), OCR charset correction + regex, tracker (one ID per billet), calibration round-trip.
- After any change, run `pytest -q` and report the result. Do not claim something works without running it.

## How I want you to work
- **Read first, then plan briefly, then implement.** For anything touching more than 3 files, give me a 5-line plan first.
- Implement **only what I ask for**. No bonus features, no refactors of unrelated code.
- Work in small, committable steps. After each step: run tests, show the result, suggest a commit message.
- If something in the PRD is ambiguous or two options are similar, choose the simpler one and tell me in one line.
- Never fabricate metrics. Accuracy, FPS, and OCR numbers must come from running the scripts. If a number misses its target, say so plainly.
- If a dependency fails to install or a model won't download offline, tell me and propose a fallback instead of silently working around it.
- Keep responses short: what changed, how to run it, what's next.

## Current priorities (update this block as the hackathon progresses)
- [x] Checkpoint 1 (~hour 8): thin slice working end to end (video → measure → log → dashboard)
- [x] Checkpoint 2 (~hour 15): all PS "must" bullets demoable (see docs/accuracy_report.md for measured numbers)
- [ ] Hour 22: feature freeze for new features. Hour 27: freeze everything.
- Current focus: install a real OCR engine and record real props; re-run `scripts/accuracy_report.py` and `scripts/ocr_eval.py` on them

## Demo safety checklist
- Video-file demo path works offline (`--source data/raw/demo.mp4`).
- Dependencies cached for offline install (`pip download -r requirements.txt -d wheels/`).
- Excel-open-during-run test passes.
- Camera-unplug test shows a "camera lost" alert and recovers.
- `data/raw/demo_backup.mp4` is a recorded screen capture of the full demo.

## Out of scope (do not build unless explicitly asked)
PLC/OPC-UA/MES integration, user authentication, cloud deployment, custom model training pipelines, mobile app, multi-camera stitching.