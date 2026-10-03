"""Dimensional measurements: minAreaRect, ellipse fit, ovality, diagonal difference, camber."""
from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class Measurement:
    """Represents geometric measurements of a billet in millimetres."""
    length_mm: float
    width_mm: float
    height_mm: float
    diameter_mm: Optional[float] = None
    ovality: Optional[float] = None
    diag_diff_mm: Optional[float] = None
    camber_mm: Optional[float] = None
    cross_section_var_mm: Optional[float] = None
    surface_anomaly_score: float = 0.0
    defects: List[str] = field(default_factory=list)
