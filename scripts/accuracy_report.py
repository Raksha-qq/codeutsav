#!/usr/bin/env python3
"""Measure dimensional accuracy, OCR, speed and write docs/accuracy_report.md.

Every number in the report is produced by running the real pipeline code on
synthetic scenes whose true dimensions are known exactly (``billetvision.
synthetic``), using the caliper values in ``data/ground_truth.csv`` as truth:

* **Belt run** — the square props cross a simulated belt as one video, through
  the full pipeline (marker auto-calibration → segmentation → tracking → belt-speed
  length → ID read → verdict → log).
* **Round props** — end-on views, calibrated from the marker in each frame,
  repeated with 20 noise seeds for repeatability.

This validates the algorithms against known geometry.  It does **not** replace
calibration against calipers on real camera footage — the report says so.

    python scripts/accuracy_report.py [--out docs/accuracy_report.md] [--seeds 1]
"""
from __future__ import annotations

import argparse
import logging
import platform
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import cv2  # noqa: E402

from billetvision import synthetic  # noqa: E402
from billetvision.pipeline import BilletVisionPipeline, _deep_merge  # noqa: E402
from billetvision.vision.calibrate import calibrate_from_marker  # noqa: E402
from billetvision.vision.measure import measure  # noqa: E402
from billetvision.vision.segment import segment  # noqa: E402

logger = logging.getLogger("accuracy_report")

TARGET_PCT = 1.0
TARGET_SIGMA_MM = 0.3
TARGET_FPS = 15.0
TARGET_LATENCY_S = 2.0
TARGET_OCR = 0.90


def pct(measured: float, truth: float) -> float:
    return (measured - truth) / truth * 100.0


# ---------------------------------------------------------------------------
# Belt run (square props, full pipeline)
# ---------------------------------------------------------------------------

def run_belt(props: List[synthetic.PropSpec], seed: int, workdir: Path) -> Dict[str, Any]:
    """Run the full pipeline offline on the belt scene; pair each record with its prop."""
    out = workdir / f"belt_{seed}"
    overrides = _deep_merge(
        synthetic.PIPELINE_OVERRIDES,
        {"logging": {
            "db_path": str(out / "b.db"), "csv_path": str(out / "b.csv"),
            "xlsx_path": str(out / "b.xlsx"), "snapshot_dir": str(out / "snap"),
        }},
    )
    pipe = BilletVisionPipeline(ROOT / "config/config.yaml", overrides=overrides)
    rows = pipe.run_offline(synthetic.render_belt_frames(props, seed=seed), fps=synthetic.FPS)
    pairs = [{"prop": p, "rec": r} for p, r in zip(props, rows)]
    return {"pairs": pairs, "n_props": len(props), "n_records": len(rows), "stats": pipe.offline_stats,
            "mm_per_px": pipe.get_calibration()["mm_per_px"]}


def belt_table(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate belt-run pairs into table rows and summary metrics."""
    rows: List[Dict[str, Any]] = []
    for run in runs:
        for pair in run["pairs"]:
            p, r = pair["prop"], pair["rec"]
            rows.append({
                "prop": p.prop_id, "id_truth": p.heat_id, "id_read": r["billet_id"],
                "w_truth": p.width_mm, "w_meas": r["width_mm"],
                "l_truth": p.length_mm, "l_meas": r["length_mm"],
                "expected": p.expected_status, "verdict": r["status"],
                "ms": r["processing_ms"], "reasons": r["fail_reasons"],
            })
    w_err = [abs(pct(r["w_meas"], r["w_truth"])) for r in rows]
    l_err = [abs(pct(r["l_meas"], r["l_truth"])) for r in rows]
    ocr_ok = sum(r["id_read"] == r["id_truth"] for r in rows)
    agree = sum((r["verdict"] == "PASS") == (r["expected"] == "PASS") for r in rows)
    return {"rows": rows, "w_err": w_err, "l_err": l_err, "ocr_ok": ocr_ok, "agree": agree, "n": len(rows)}


# ---------------------------------------------------------------------------
# Round props (end view, calibrated per frame)
# ---------------------------------------------------------------------------

def measure_round_frame(diameter_mm: float, seed: int, ovality_pct: float = 0.0) -> Optional[Dict[str, float]]:
    frame = synthetic.render_end_view(diameter_mm, seed=seed, ovality_pct=ovality_pct)
    cal = calibrate_from_marker(frame)
    seg = segment(frame, roi=list(synthetic.ROI_BOX), preprocess_kwargs={"auto_exposure": True})
    if seg is None:
        return None
    m = measure(seg.contour, cal.mm_per_pixel, shape="round", travel_axis="x")
    return {"diameter": m.diameter_mm, "ovality": m.ovality or 0.0, "mm_per_px": cal.mm_per_pixel}


def run_round(props: List[synthetic.PropSpec], repeats: int) -> List[Dict[str, Any]]:
    out = []
    for p in props:
        meas = [m for m in (measure_round_frame(p.diameter_mm, seed=s) for s in range(repeats)) if m]
        diam = np.array([m["diameter"] for m in meas])
        out.append({
            "prop": p.prop_id, "truth": p.diameter_mm, "mean": float(diam.mean()), "std": float(diam.std(ddof=1)),
            "err_pct": pct(float(diam.mean()), p.diameter_mm), "max_err_pct": float(np.max(np.abs((diam - p.diameter_mm) / p.diameter_mm * 100))),
            "ovality": float(np.mean([m["ovality"] for m in meas])), "n": len(meas),
        })
    return out


# ---------------------------------------------------------------------------
# OCR test set
# ---------------------------------------------------------------------------

def run_ocr_testset() -> Optional[Dict[str, Any]]:
    """Evaluate ``data/ocr_testset`` with the configured reader (same logic as ocr_eval.py)."""
    import yaml

    import ocr_eval
    from billetvision.ocr.reader import OcrReader

    testset = ROOT / "data/ocr_testset"
    if not (testset / "ground_truth.csv").exists():
        return None
    cfg = yaml.safe_load((ROOT / "config/config.yaml").read_text(encoding="utf-8"))["ocr"]
    reader = OcrReader.from_config(cfg)
    metrics = ocr_eval.run_eval(testset, reader, pattern=cfg.get("heat_id_regex", r"^[A-Z]\d{5,7}$"))
    metrics["engine"] = "builtin" if reader._builtin else cfg.get("engine", "?")
    return metrics


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def _verdict(ok: bool) -> str:
    return "PASS" if ok else "**FAIL**"


def render_markdown(belt: Dict[str, Any], rounds: List[Dict[str, Any]], runs: List[Dict[str, Any]],
                    ocr: Optional[Dict[str, Any]], seeds: int) -> str:
    stats = runs[0]["stats"]
    fps = 1000.0 / stats["mean_frame_ms"] if stats["mean_frame_ms"] else 0.0
    billet_ms = [r["ms"] for r in belt["rows"]]
    w_max, l_max = max(belt["w_err"]), max(belt["l_err"])
    r_max = max(r["max_err_pct"] for r in rounds) if rounds else 0.0
    sigma_max = max(r["std"] for r in rounds) if rounds else 0.0
    n = belt["n"]
    lines = [
        "# BilletVision: Dimensional Accuracy & Repeatability Report",
        "",
        f"_Generated by `scripts/accuracy_report.py` on {datetime.now():%Y-%m-%d %H:%M}; "
        f"Python {platform.python_version()}, OpenCV {cv2.__version__}, NumPy {np.__version__}._",
        "",
        "> **What this is.** Every figure below was produced by running the real pipeline on **synthetic scenes "
        "with exactly known geometry** (`src/billetvision/synthetic.py`), using the caliper values in "
        "`data/ground_truth.csv` as truth. It validates the algorithms (calibration, segmentation, belt-speed "
        "length, ID reading, decision). It does **not** replace measuring real props with calipers on real camera "
        "footage; lens distortion, glare, motion blur and perspective of a real installation are not modelled "
        "beyond sensor noise and exposure drift.",
        "",
        "## Requirement check",
        "",
        "| # | Requirement | Measured | Result |",
        "|---|---|---|---|",
        f"| G1 | Dimensional error ≤ ±{TARGET_PCT:.0f}% (belt run, width) | max {w_max:.2f}% over {n} billets | {_verdict(w_max <= TARGET_PCT)} |",
        f"| G1 | Dimensional error ≤ ±{TARGET_PCT:.0f}% (belt run, length via belt speed) | max {l_max:.2f}% | {_verdict(l_max <= TARGET_PCT)} |",
        f"| G1 | Dimensional error ≤ ±{TARGET_PCT:.0f}% (round, diameter) | max {r_max:.2f}% | {_verdict(r_max <= TARGET_PCT)} |",
        f"| G1 | Repeatability σ < {TARGET_SIGMA_MM} mm (round, 20 runs) | max σ {sigma_max:.3f} mm | {_verdict(sigma_max < TARGET_SIGMA_MM)} |",
        f"| G2 | ≥ {TARGET_FPS:.0f} FPS processing capacity | {fps:.1f} FPS (mean {stats['mean_frame_ms']:.1f} ms, p95 {stats['p95_frame_ms']:.1f} ms / frame) | {_verdict(fps >= TARGET_FPS)} |",
        f"| G2 | < {TARGET_LATENCY_S:.0f} s inspection latency per billet | max {max(billet_ms)/1000:.2f} s (mean {np.mean(billet_ms)/1000:.2f} s) | {_verdict(max(billet_ms) < TARGET_LATENCY_S * 1000)} |",
        f"| G3 | ID exact match ≥ {TARGET_OCR:.0%} (belt run) | {belt['ocr_ok']}/{n} = {belt['ocr_ok']/n:.0%} | {_verdict(belt['ocr_ok']/n >= TARGET_OCR)} |",
    ]
    if ocr:
        lines.append(
            f"| G3 | ID exact match ≥ {TARGET_OCR:.0%} (`data/ocr_testset`, engine `{ocr['engine']}`) | "
            f"{ocr['exact_hits']}/{ocr['n']} = {ocr['exact_accuracy']:.0%} | {_verdict(ocr['exact_accuracy'] >= TARGET_OCR)} |"
        )
    lines += [
        f"| — | Accept/reject agrees with ground truth (PASS vs not-PASS) | {belt['agree']}/{n} | {_verdict(belt['agree'] == n)} |",
        "",
        "## Calibration",
        "",
        "- Reference: ArUco `DICT_4X4_50`, 50.0 mm, auto-detected in the first frames (sub-pixel corners).",
        f"- Recovered scale: **{runs[0]['mm_per_px']:.4f} mm/px** (scene truth 0.5000 mm/px, "
        f"{pct(runs[0]['mm_per_px'], 0.5):+.2f}%).",
        f"- Belt speed: {synthetic.BELT_SPEED_MM_S:.0f} mm/s (config `vision.conveyor_speed_mm_s`), {synthetic.FPS:.0f} FPS.",
        "",
        f"## Belt run — square props, full pipeline ({seeds} noise seed{'s' if seeds != 1 else ''})",
        "",
        "Width is measured across the belt; **height is not observable from a top view and equals the width "
        "proxy**, so only width is compared. Length comes from belt speed × time-in-view because the bars are "
        "longer than the field of view. The per-billet value is the median of the 5 sharpest frames.",
        "",
        "| Prop | ID truth | ID read | Width truth (mm) | Width measured | Err % | Length truth (mm) | Length measured | Err % | Expected | Verdict |",
        "|:--|:--|:--|--:|--:|--:|--:|--:|--:|:--|:--|",
    ]
    for r in belt["rows"]:
        lines.append(
            f"| {r['prop']} | {r['id_truth']} | {r['id_read']}{'' if r['id_read'] == r['id_truth'] else ' ✗'} | "
            f"{r['w_truth']:.1f} | {r['w_meas']:.2f} | {pct(r['w_meas'], r['w_truth']):+.2f} | "
            f"{r['l_truth']:.1f} | {r['l_meas']:.2f} | {pct(r['l_meas'], r['l_truth']):+.2f} | "
            f"{r['expected']} | {r['verdict']} |"
        )
    lines += [
        "",
        f"Mean absolute error: width **{np.mean(belt['w_err']):.2f}%**, length **{np.mean(belt['l_err']):.3f}%**; "
        f"max width {w_max:.2f}%, max length {l_max:.3f}%.",
        "",
        "`expected_status` in `data/ground_truth.csv` is binary-ish (PASS/FAIL). The decision engine grades a "
        "deviation of 1–2× the tolerance as **REWORK** and > 2× as **FAIL**; both are rejects, so agreement is "
        "counted as PASS vs not-PASS.",
        "",
        "## Round props — end view, 20 repeats each",
        "",
        "| Prop | Truth (mm) | Mean measured | Mean err % | Max err % | σ (mm) | Mean ovality % |",
        "|:--|--:|--:|--:|--:|--:|--:|",
    ]
    for r in rounds:
        lines.append(
            f"| {r['prop']} | {r['truth']:.1f} | {r['mean']:.2f} | {r['err_pct']:+.2f} | {r['max_err_pct']:.2f} | "
            f"{r['std']:.3f} | {r['ovality']:.2f} |"
        )
    lines += [
        "",
        "## Known limits",
        "",
        "- Width resolution is one pixel (0.5 mm here); a median of identical integer widths quantises to 0.5 mm. "
        "Zoom in (or use sub-pixel edge fitting) if a ±1.0 mm tolerance must be verified to better than ~0.4 mm.",
        "- The repeatability σ reflects sensor-noise repeatability of the algorithm on identical geometry; it does not "
        "include mechanical variation (vibration, belt wander) or re-seating the prop.",
        "- `data/raw/demo.mp4` is not used here: its burned-in labels (1000 × 130 mm) are inconsistent with its own "
        "ArUco marker (0.52 mm/px → the bar is ≈ 323 × 133 mm), so it has no usable ground truth.",
    ]
    if ocr and ocr["engine"] == "builtin":
        lines.append(
            "- No OCR engine (PaddleOCR / EasyOCR / Tesseract) is installed in this environment, so the **builtin "
            "template-matching fallback** produced the OCR figures. It is tuned to clean printed IDs; the test set and "
            "belt scene use fonts it was built from, so these OCR numbers are optimistic. Install PaddleOCR or EasyOCR "
            "and re-run this script and `scripts/ocr_eval.py` to measure real stamped/painted IDs."
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(ROOT / "docs/accuracy_report.md"))
    parser.add_argument("--seeds", type=int, default=1, help="belt-run noise seeds (more = slower, tighter stats)")
    parser.add_argument("--repeats", type=int, default=20, help="round-prop repeats for repeatability")
    parser.add_argument("--no-write", action="store_true", help="print to stdout instead of writing the file")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    import os

    os.chdir(ROOT)
    gt = ROOT / "data/ground_truth.csv"
    squares = synthetic.demo_props(gt)
    round_props = [p for p in synthetic.load_ground_truth(gt) if p.shape == "round"]

    workdir = Path(tempfile.mkdtemp(prefix="bv_accuracy_"))
    try:
        print(f"[accuracy] belt run: {len(squares)} props x {args.seeds} seed(s) ...", flush=True)
        runs = [run_belt(squares, seed=7 + i, workdir=workdir) for i in range(args.seeds)]
        belt = belt_table(runs)
        print(f"[accuracy] round props: {len(round_props)} x {args.repeats} repeats ...", flush=True)
        rounds = run_round(round_props, args.repeats)
        print("[accuracy] OCR test set ...", flush=True)
        ocr = run_ocr_testset()
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    report = render_markdown(belt, rounds, runs, ocr, args.seeds)
    if args.no_write:
        print(report)
    else:
        Path(args.out).write_text(report, encoding="utf-8")
        print(f"[accuracy] wrote {args.out}")
    failed = (
        max(belt["w_err"]) > TARGET_PCT or max(belt["l_err"]) > TARGET_PCT
        or (rounds and max(r["max_err_pct"] for r in rounds) > TARGET_PCT)
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
