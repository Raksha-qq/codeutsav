"""Multi-frame voting accumulator to combine multiple frame observations into a single ID."""
from typing import List, Tuple
from collections import Counter
from billetvision.ocr.reader import OcrResult

class MultiFrameVoter:
    """Accumulates OCR and code reads across frames for a tracked billet."""

    def __init__(self, min_confidence: float = 0.60):
        self.observations: List[OcrResult] = []
        self.min_confidence = min_confidence

    def add_observation(self, res: OcrResult) -> None:
        self.observations.append(res)

    def compute_final_id(self) -> Tuple[str, float, str]:
        """Returns (best_id, confidence, status). Status is PASS or REVIEW."""
        if not self.observations:
            return ("UNKNOWN", 0.0, "REVIEW")
        
        # QR/Barcode always takes priority if present
        for obs in self.observations:
            if obs.source in ("qr", "barcode") and obs.confidence > 0.8:
                return (obs.text, obs.confidence, "PASS")
        
        counts = Counter(obs.text for obs in self.observations)
        best_text, _ = counts.most_common(1)[0]
        matching_confidences = [obs.confidence for obs in self.observations if obs.text == best_text]
        avg_conf = sum(matching_confidences) / len(matching_confidences)
        
        status = "PASS" if avg_conf >= self.min_confidence else "REVIEW"
        return (best_text, avg_conf, status)
