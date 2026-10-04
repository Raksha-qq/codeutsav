#!/usr/bin/env python3
"""OCR accuracy evaluation on data/ocr_testset/.

Expected testset layout
-----------------------
data/ocr_testset/
    ground_truth.csv        columns: filename, ground_truth
    <image files>           any name; referenced by filename column

If ground_truth.csv does not exist, the script generates a small synthetic
testset (drawn text on solid backgrounds) so the pipeline can be exercised
immediately without real images.

Metrics
-------
- Exact-match accuracy  : fraction of IDs read exactly right
- Character accuracy     : 1 − CER  (character error rate via edit distance)
- Per-engine breakdown   : counts by OCR engine used

Usage
-----
    python scripts/ocr_eval.py [--testset data/ocr_testset] [--config config/config.yaml]
"""
from __future__ import annotations

import argparse
import csv
import sys
import logging
from pathlib import Path

import cv2
import numpy as np
import yaml

# Ensure src/ is on the path when run as a script
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from billetvision.ocr.reader import OcrReader
from billetvision.ocr.validate import correct_and_validate, normalize

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("ocr_eval")

# ---------------------------------------------------------------------------
# Synthetic testset generator
# ---------------------------------------------------------------------------

SYNTHETIC_IDS = [
    "A123456",
    "B789012",
    "C345678",
    "D901234",
    "A000001",
    "Z999999",
    "B111111",
    "C222222",
]

# Common OCR confusions to simulate degraded images
_CONFUSIONS = {"O": "0", "I": "1", "S": "5", "B": "8"}


def _render_id_image(text: str, size: tuple[int, int] = (320, 80)) -> np.ndarray:
    """Render ``text`` as a white-on-black image with slight degradation."""
    img = np.zeros((size[1], size[0]), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.4
    thickness = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x = max(0, (size[0] - tw) // 2)
    y = max(th, (size[1] + th) // 2)
    cv2.putText(img, text, (x, y), font, scale, 255, thickness, cv2.LINE_AA)
    # Add mild Gaussian noise
    noise = np.random.normal(0, 15, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


def generate_synthetic_testset(testset_dir: Path) -> Path:
    """Create synthetic images + ground_truth.csv in ``testset_dir``."""
    testset_dir.mkdir(parents=True, exist_ok=True)
    gt_path = testset_dir / "ground_truth.csv"
    rows = []
    for billet_id in SYNTHETIC_IDS:
        fname = f"synthetic_{billet_id}.png"
        img = _render_id_image(billet_id)
        cv2.imwrite(str(testset_dir / fname), img)
        rows.append({"filename": fname, "ground_truth": billet_id})
    with gt_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["filename", "ground_truth"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[ocr_eval] Generated {len(rows)} synthetic test images → {gt_path}")
    return gt_path


# ---------------------------------------------------------------------------
# Edit-distance (for character error rate)
# ---------------------------------------------------------------------------

def _edit_distance(a: str, b: str) -> int:
    """Standard Levenshtein distance."""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def char_accuracy(pred: str, gt: str) -> float:
    """1 − CER, clamped to [0, 1].  CER = edit_distance / max(len(gt), 1)."""
    if not gt:
        return 1.0 if not pred else 0.0
    cer = _edit_distance(pred, gt) / len(gt)
    return max(0.0, 1.0 - cer)


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def load_groundtruth(gt_path: Path) -> list[dict]:
    rows = []
    with gt_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)
    return rows


def run_eval(
    testset_dir: Path,
    ocr_reader: OcrReader,
    pattern: str,
    verbose: bool = False,
) -> dict:
    gt_path = testset_dir / "ground_truth.csv"
    if not gt_path.exists():
        print(f"[ocr_eval] No ground_truth.csv found — generating synthetic testset")
        generate_synthetic_testset(testset_dir)

    rows = load_groundtruth(gt_path)
    if not rows:
        print("[ocr_eval] ground_truth.csv is empty — nothing to evaluate")
        return {}

    exact_hits = 0
    total_char_acc = 0.0
    engine_counts: dict[str, int] = {}
    results = []

    for row in rows:
        fname = row["filename"]
        gt = normalize(row["ground_truth"])
        img_path = testset_dir / fname
        if not img_path.exists():
            logger.warning("Image not found: %s", img_path)
            results.append({"file": fname, "gt": gt, "pred": "", "exact": False, "char_acc": 0.0})
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            logger.warning("Failed to load image: %s", img_path)
            results.append({"file": fname, "gt": gt, "pred": "", "exact": False, "char_acc": 0.0})
            continue

        ocr_res = ocr_reader.read_best(img, pattern=pattern)
        pred, matched, _ = correct_and_validate(ocr_res.text, pattern)
        pred_norm = normalize(pred)

        exact = pred_norm == gt
        ca = char_accuracy(pred_norm, gt)

        if exact:
            exact_hits += 1
        total_char_acc += ca

        engine_counts[ocr_res.engine or "none"] = engine_counts.get(ocr_res.engine or "none", 0) + 1
        results.append({
            "file": fname,
            "gt": gt,
            "pred": pred_norm,
            "conf": round(ocr_res.confidence, 3),
            "engine": ocr_res.engine,
            "exact": exact,
            "char_acc": round(ca, 3),
        })

        if verbose:
            mark = "✓" if exact else "✗"
            print(f"  {mark} {fname:30s}  gt={gt:12s}  pred={pred_norm:12s}  conf={ocr_res.confidence:.2f}  ca={ca:.2f}")

    n = len(rows)
    exact_acc = exact_hits / n if n else 0.0
    mean_char_acc = total_char_acc / n if n else 0.0

    return {
        "n": n,
        "exact_hits": exact_hits,
        "exact_accuracy": exact_acc,
        "mean_char_accuracy": mean_char_acc,
        "engine_counts": engine_counts,
        "results": results,
    }


def print_report(metrics: dict) -> None:
    n = metrics.get("n", 0)
    if n == 0:
        print("No samples evaluated.")
        return

    print()
    print("=" * 60)
    print("  BilletVision OCR Evaluation Report")
    print("=" * 60)
    print(f"  Samples evaluated  : {n}")
    print(f"  Exact-match hits   : {metrics['exact_hits']} / {n}")
    print(f"  Exact-match acc    : {metrics['exact_accuracy']*100:.1f}%  (target ≥ 90%)")
    print(f"  Mean char accuracy : {metrics['mean_char_accuracy']*100:.1f}%")
    print()
    print("  Engine usage:")
    for eng, cnt in sorted(metrics.get("engine_counts", {}).items()):
        print(f"    {eng:15s}: {cnt}")
    print("=" * 60)

    target_met = metrics["exact_accuracy"] >= 0.90
    status = "PASS" if target_met else "FAIL (below 90% target)"
    print(f"  → OCR accuracy target: {status}")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252
    parser = argparse.ArgumentParser(description="Evaluate OCR accuracy on test set")
    parser.add_argument(
        "--testset",
        default="data/ocr_testset",
        help="Directory containing images and ground_truth.csv",
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print per-sample results",
    )
    args = parser.parse_args(argv)

    testset_dir = Path(args.testset)
    testset_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    ocr_cfg = cfg.get("ocr", {})
    pattern = ocr_cfg.get("heat_id_regex", r"^[A-Z]\d{5,7}$")

    print(f"[ocr_eval] Loading OCR engine (engine={ocr_cfg.get('engine','easyocr')}) …")
    reader = OcrReader.from_config(ocr_cfg)

    print(f"[ocr_eval] Evaluating on {testset_dir} …")
    metrics = run_eval(testset_dir, reader, pattern=pattern, verbose=args.verbose)

    print_report(metrics)
    return 0 if metrics.get("exact_accuracy", 0) >= 0.90 else 1


if __name__ == "__main__":
    sys.exit(main())
