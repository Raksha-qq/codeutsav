"""Decision engine: evaluates measurements against tolerance configurations."""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from billetvision.vision.measure import Measurement

@dataclass
class Verdict:
    """Represents final inspection verdict for a billet."""
    status: str # "PASS", "FAIL", "REWORK", "REVIEW"
    fail_reasons: List[str] = field(default_factory=list)
    is_acceptable: bool = True

def evaluate_tolerances(measurement: Measurement, tol_config: Dict[str, Any], ocr_status: str = "PASS") -> Verdict:
    """Pure function evaluating Measurement against tolerance configuration."""
    reasons: List[str] = []

    # Width check
    if "width_nominal_mm" in tol_config and "width_tol_mm" in tol_config:
        nom = tol_config["width_nominal_mm"]
        tol = tol_config["width_tol_mm"]
        if abs(measurement.width_mm - nom) > tol:
            reasons.append(f"width {measurement.width_mm:.1f} mm out of range ({nom:.1f} ± {tol:.1f} mm)")

    # Height check
    if "height_nominal_mm" in tol_config and "height_tol_mm" in tol_config:
        nom = tol_config["height_nominal_mm"]
        tol = tol_config["height_tol_mm"]
        if abs(measurement.height_mm - nom) > tol:
            reasons.append(f"height {measurement.height_mm:.1f} mm out of range ({nom:.1f} ± {tol:.1f} mm)")

    # Diameter check
    if measurement.diameter_mm is not None and "diameter_nominal_mm" in tol_config and "diameter_tol_mm" in tol_config:
        nom = tol_config["diameter_nominal_mm"]
        tol = tol_config["diameter_tol_mm"]
        if abs(measurement.diameter_mm - nom) > tol:
            reasons.append(f"diameter {measurement.diameter_mm:.1f} mm out of range ({nom:.1f} ± {tol:.1f} mm)")

    # Diagonal difference (rhomboidity)
    if measurement.diag_diff_mm is not None and "max_diag_diff_mm" in tol_config:
        max_diff = tol_config["max_diag_diff_mm"]
        if measurement.diag_diff_mm > max_diff:
            reasons.append(f"diagonal difference {measurement.diag_diff_mm:.1f} mm > {max_diff:.1f} mm")

    # Ovality check
    if measurement.ovality is not None and "max_ovality_pct" in tol_config:
        max_ov = tol_config["max_ovality_pct"]
        if measurement.ovality > max_ov:
            reasons.append(f"ovality {measurement.ovality:.2f}% > {max_ov:.2f}%")

    # Camber / straightness check
    if measurement.camber_mm is not None and "max_camber_mm" in tol_config:
        max_cam = tol_config["max_camber_mm"]
        if measurement.camber_mm > max_cam:
            reasons.append(f"camber {measurement.camber_mm:.1f} mm > {max_cam:.1f} mm")

    # Profile variation check
    if measurement.cross_section_var_mm is not None and "max_cross_section_var_mm" in tol_config:
        max_var = tol_config["max_cross_section_var_mm"]
        if measurement.cross_section_var_mm > max_var:
            reasons.append(f"cross-section variation {measurement.cross_section_var_mm:.1f} mm > {max_var:.1f} mm")

    if reasons:
        return Verdict(status="FAIL", fail_reasons=reasons, is_acceptable=False)
    
    if ocr_status == "REVIEW":
        return Verdict(status="REVIEW", fail_reasons=["OCR confidence below threshold - operator verification needed"], is_acceptable=False)

    return Verdict(status="PASS", fail_reasons=[], is_acceptable=True)
