"""Camera and perspective calibration: ArUco / checkerboard detection and mm/px calculation."""
from dataclasses import dataclass
from typing import Optional, List
import json
from pathlib import Path

@dataclass
class CalibrationData:
    mm_per_pixel: float
    pixels_per_mm: float
    homography_matrix: Optional[List[List[float]]] = None

    @classmethod
    def load(cls, path: str | Path) -> "CalibrationData":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            mm_per_pixel=data["mm_per_pixel"],
            pixels_per_mm=data["pixels_per_mm"],
            homography_matrix=data.get("homography_matrix"),
        )
