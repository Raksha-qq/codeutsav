"""Decision engine: pure functions that evaluate measurements against tolerances.

No I/O inside ``evaluate_tolerances`` or ``evaluate`` — those are pure.
Only ``load_tolerances`` touches the filesystem.

All measurement values are in **millimetres** (or % for ovality / score 0-1 for anomaly).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from billetvision.vision.measure import Measurement

logger = logging.getLogger(__name__)

# Violations within REWORK_FACTOR × tolerance are REWORK; beyond that are FAIL.
REWORK_FACTOR = 2.0

_STATUS_PRIORITY = {"FAIL": 3, "REWORK": 2, "REVIEW": 1, "PASS": 0}


@dataclass
class Verdict:
    """Final inspection verdict for one billet."""

    status: str  # PASS | FAIL | REWORK | REVIEW
    reasons: List[str] = field(default_factory=list)

    # Legacy alias kept for callers that used the old field name.
    @property
    def fail_reasons(self) -> List[str]:
        return self.reasons

    @property
    def is_acceptable(self) -> bool:
        return self.status == "PASS"


def load_tolerances(path: str | Path) -> Dict[str, Any]:
    """Load tolerances.yaml and return the raw dict.

    Args:
        path: Path to tolerances.yaml (absolute or relative to cwd).

    Returns:
        Dict mapping profile name → tolerance dict.
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"tolerances.yaml at {p} must be a mapping, got {type(data)}")
    return data


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_range(
    value: float,
    nominal: float,
    tol: float,
    label: str,
    unit: str = "mm",
) -> Optional[str]:
    """Return a reason string if ``value`` is outside nominal ± tol, else None.

    Severity is baked into the returned string via the deviation amount so
    callers can decide REWORK vs FAIL by inspecting the raw deviation.
    """
    deviation = abs(value - nominal)
    if deviation > tol:
        return (
            f"{label} {value:.2f} {unit} out of range "
            f"({nominal:.2f} ± {tol:.2f} {unit}, "
            f"deviation {deviation:.2f} {unit})"
        )
    return None


def _check_max(
    value: float,
    limit: float,
    label: str,
    unit: str = "mm",
) -> Optional[str]:
    """Return a reason string if ``value`` exceeds ``limit``, else None."""
    if value > limit:
        return f"{label} {value:.2f} {unit} > limit {limit:.2f} {unit}"
    return None


# ---------------------------------------------------------------------------
# Core pure evaluation
# ---------------------------------------------------------------------------

def evaluate_tolerances(
    measurement: Measurement,
    tol_config: Dict[str, Any],
    ocr_status: str = "PASS",
) -> Verdict:
    """Evaluate a Measurement against a tolerance config dict.

    Args:
        measurement: Measured dimensions in mm.
        tol_config: One profile's worth of tolerances (a value from load_tolerances()).
        ocr_status: "PASS" or "REVIEW" from the OCR module.

    Returns:
        Verdict with status ∈ {PASS, FAIL, REWORK, REVIEW} and human-readable reasons.

    Note:
        FAIL takes priority over REWORK, which takes priority over REVIEW.
        A dimensional FAIL is always returned even when OCR is also REVIEW.
    """
    fail_reasons: List[str] = []
    rework_reasons: List[str] = []

    def _classify(reason: str, value: float, nominal: float, tol: float) -> None:
        """Sort a dimensional violation into fail or rework buckets."""
        if reason is None:
            return
        deviation = abs(value - nominal)
        if deviation > REWORK_FACTOR * tol:
            fail_reasons.append(reason)
        else:
            rework_reasons.append(reason)

    def _classify_max(reason: str, value: float, limit: float) -> None:
        """Sort a max-limit violation into fail or rework buckets."""
        if reason is None:
            return
        # For max-type limits, REWORK if ≤ REWORK_FACTOR × limit, FAIL otherwise.
        if value > REWORK_FACTOR * limit:
            fail_reasons.append(reason)
        else:
            rework_reasons.append(reason)

    # --- Length ---
    if "length_nominal_mm" in tol_config and "length_tol_mm" in tol_config:
        nom = tol_config["length_nominal_mm"]
        tol = tol_config["length_tol_mm"]
        r = _check_range(measurement.length_mm, nom, tol, "length")
        _classify(r, measurement.length_mm, nom, tol)

    # --- Width (rectangular billets) ---
    if "width_nominal_mm" in tol_config and "width_tol_mm" in tol_config:
        nom = tol_config["width_nominal_mm"]
        tol = tol_config["width_tol_mm"]
        r = _check_range(measurement.width_mm, nom, tol, "width")
        _classify(r, measurement.width_mm, nom, tol)

    # --- Height (rectangular billets) ---
    if "height_nominal_mm" in tol_config and "height_tol_mm" in tol_config:
        nom = tol_config["height_nominal_mm"]
        tol = tol_config["height_tol_mm"]
        r = _check_range(measurement.height_mm, nom, tol, "height")
        _classify(r, measurement.height_mm, nom, tol)

    # --- Diameter (round billets) ---
    if (
        measurement.diameter_mm is not None
        and "diameter_nominal_mm" in tol_config
        and "diameter_tol_mm" in tol_config
    ):
        nom = tol_config["diameter_nominal_mm"]
        tol = tol_config["diameter_tol_mm"]
        r = _check_range(measurement.diameter_mm, nom, tol, "diameter")
        _classify(r, measurement.diameter_mm, nom, tol)

    # --- Diagonal difference / rhomboidity ---
    if measurement.diag_diff_mm is not None and "max_diag_diff_mm" in tol_config:
        limit = tol_config["max_diag_diff_mm"]
        r = _check_max(measurement.diag_diff_mm, limit, "diagonal difference")
        _classify_max(r, measurement.diag_diff_mm, limit)

    # --- Ovality (round billets) ---
    if measurement.ovality is not None and "max_ovality_pct" in tol_config:
        limit = tol_config["max_ovality_pct"]
        r = _check_max(measurement.ovality, limit, "ovality", unit="%")
        _classify_max(r, measurement.ovality, limit)

    # --- Camber / straightness ---
    if measurement.camber_mm is not None and "max_camber_mm" in tol_config:
        limit = tol_config["max_camber_mm"]
        r = _check_max(measurement.camber_mm, limit, "camber")
        _classify_max(r, measurement.camber_mm, limit)

    # --- Cross-section profile variation ---
    if (
        measurement.cross_section_var_mm is not None
        and "max_cross_section_var_mm" in tol_config
    ):
        limit = tol_config["max_cross_section_var_mm"]
        r = _check_max(
            measurement.cross_section_var_mm, limit, "cross-section variation"
        )
        _classify_max(r, measurement.cross_section_var_mm, limit)

    # --- Surface anomaly score ---
    if "max_surface_anomaly_score" in tol_config:
        limit = tol_config["max_surface_anomaly_score"]
        r = _check_max(
            measurement.surface_anomaly_score,
            limit,
            "surface anomaly score",
            unit="",
        )
        _classify_max(r, measurement.surface_anomaly_score, limit)

    # --- Determine final status ---
    if fail_reasons:
        all_reasons = fail_reasons + rework_reasons
        return Verdict(status="FAIL", reasons=all_reasons)

    if rework_reasons:
        return Verdict(status="REWORK", reasons=rework_reasons)

    if ocr_status == "REVIEW":
        return Verdict(
            status="REVIEW",
            reasons=["OCR confidence below threshold — operator verification required"],
        )

    return Verdict(status="PASS", reasons=[])


def evaluate(
    measurement: Measurement,
    profile_name: str,
    tolerances: Dict[str, Any],
    ocr_status: str = "PASS",
) -> Verdict:
    """Top-level entry point: look up profile and evaluate.

    Args:
        measurement: Measured billet dimensions.
        profile_name: Key in the tolerances dict, e.g. "square_130".
        tolerances: Full tolerances dict from load_tolerances().
        ocr_status: "PASS" or "REVIEW" from the OCR module.

    Returns:
        Verdict.

    Raises:
        KeyError: If profile_name is not in tolerances.
    """
    if profile_name not in tolerances:
        raise KeyError(
            f"Profile '{profile_name}' not found in tolerances. "
            f"Available: {list(tolerances.keys())}"
        )
    return evaluate_tolerances(measurement, tolerances[profile_name], ocr_status)
