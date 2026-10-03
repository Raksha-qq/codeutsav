"""Multi-frame voting: accumulate OCR reads across frames for one tracked billet.

Uses confidence-weighted majority voting so high-confidence reads dominate.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from billetvision.ocr.reader import OcrResult

logger = logging.getLogger(__name__)


class MultiFrameVoter:
    """Accumulate per-frame OcrResult observations and compute a consensus ID.

    Typical usage::

        voter = MultiFrameVoter(min_confidence=0.60)
        for frame in frames:
            result = reader.read_best(crop)
            voter.add(result)
        billet_id, confidence, status = voter.decide()
        voter.reset()
    """

    def __init__(self, min_confidence: float = 0.60) -> None:
        self.min_confidence = min_confidence
        self._observations: List[OcrResult] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, result: OcrResult) -> None:
        """Record one frame's OCR result."""
        self._observations.append(result)

    # Legacy alias
    def add_observation(self, result: OcrResult) -> None:
        self.add(result)

    def reset(self) -> None:
        """Clear all accumulated observations (call after a billet is logged)."""
        self._observations.clear()

    @property
    def observation_count(self) -> int:
        return len(self._observations)

    def decide(self) -> Tuple[str, float, str]:
        """Compute consensus ID, confidence, and status.

        Returns:
            (billet_id, confidence, status) where status ∈ {"PASS", "REVIEW"}.
        """
        if not self._observations:
            return ("UNKNOWN", 0.0, "REVIEW")

        # QR / barcode always wins if high-confidence
        for obs in self._observations:
            if obs.source in ("qr", "barcode") and obs.confidence >= 0.9:
                return (obs.text, obs.confidence, "PASS")

        # Confidence-weighted voting
        weights: Dict[str, float] = defaultdict(float)
        counts: Dict[str, int] = defaultdict(int)
        for obs in self._observations:
            if obs.text:
                weights[obs.text] += obs.confidence
                counts[obs.text] += 1

        if not weights:
            return ("UNKNOWN", 0.0, "REVIEW")

        winner = max(weights, key=lambda t: weights[t])
        winner_weight = weights[winner]
        total_weight = sum(weights.values()) or 1.0
        vote_fraction = winner_weight / total_weight

        # Average confidence for the winning text
        avg_conf = winner_weight / counts[winner]

        # Blended confidence: 60% avg_conf + 40% vote_fraction
        final_conf = 0.6 * avg_conf + 0.4 * vote_fraction
        status = "PASS" if final_conf >= self.min_confidence else "REVIEW"

        return (winner, round(final_conf, 4), status)

    # Legacy alias
    def compute_final_id(self) -> Tuple[str, float, str]:
        return self.decide()
