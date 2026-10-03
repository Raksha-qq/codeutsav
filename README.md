# BilletVision — Automated Billet Inspection & Traceability System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

BilletVision is a non-contact industrial computer-vision system for continuous steel billet inspection and traceability. It measures cross-sectional dimensions to within $\pm 1\%$ accuracy, reads heat/batch IDs via optical character and code recognition, flags out-of-tolerance billets in real-time, and logs audit-ready records to SQLite, CSV, and Excel without data corruption.

---

## Key Features

1. **Precision Geometry:** Sub-millimetre non-contact measurement of width, height, diameter, ovality, diagonal difference (rhomboidity), and camber.
2. **ArUco / Homography Calibration:** Fast calibration wizard establishing pixel-to-millimetre scale with perspective correction.
3. **Pluggable Capture Pipeline:** Unified `FrameSource` interface supporting live webcams, pre-recorded MP4 video streams, and static image folders with a bounded drop-oldest frame queue ($\ge 15\text{ FPS}$).
4. **Robust OCR & Code Detection:** Multi-frame voting OCR with character-set correction and automatic fallback for QR / DataMatrix / Barcodes.
5. **Zero-Loss Durable Logging:** Multi-tiered logging (SQLite source-of-truth $\rightarrow$ CSV append $\rightarrow$ atomic XLSX writer) with automatic fallback to part files if Excel holds the file lock.
6. **Live Operator Dashboard:** Modern web UI with live video stream, real-time PASS/FAIL alerts with audio, tolerance editor, and historical audit log.

---

## Quick Start

### 1. Environment Setup

```bash
# Windows
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run Demonstration

```bash
python scripts/run_demo.py --source data/raw/demo.mp4
```

### 3. Start Operator Dashboard & API

```bash
uvicorn billetvision.api.main:app --reload --port 8000
```
Open [http://localhost:8000](http://localhost:8000) in your browser.

### 4. Run Test Suite

```bash
pytest -q
```

---

## Repository Structure

```
billetvision/
├── CLAUDE.md                   # Agent guidelines and commands
├── PRD.md / PRD.pdf            # Product Requirements Document
├── README.md                   # Project overview & instructions
├── pyproject.toml              # Build & package configuration
├── requirements.txt            # Python dependencies
├── config/                     # YAML configuration files
│   ├── config.yaml             # Main system configuration
│   ├── tolerances.yaml         # Dimensional tolerances
│   ├── billet_profiles.yaml    # Standard billet profiles
│   └── calibration.json        # Calibration parameters
├── src/billetvision/           # Application source code
│   ├── capture/                # Frame sources & bounded queues
│   ├── vision/                 # Preprocessing, segmentation, measurement, defects
│   ├── ocr/                    # OCR enhancement, reader, barcode/QR, voting
│   ├── decision/               # Pure tolerance decision engine
│   ├── logging_/               # SQLite, CSV, and atomic XLSX writers
│   ├── alerts/                 # Alert manager & Telegram notifications
│   ├── api/                    # FastAPI routes, WebSocket, MJPEG video
│   └── pipeline.py             # Main processing pipeline
├── web/                        # Operator dashboard (HTML5 / Vanilla JS / CSS3)
├── data/                       # Ground truth data, test images, output logs
├── scripts/                    # CLI scripts (run_demo, calibrate, accuracy_report, etc.)
├── tests/                      # Automated unit and integration tests
└── docs/                       # Technical documentation & reports
```
