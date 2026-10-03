"""Centroid tracker: entry/exit line, one record per billet, median over best-N frames.

Design
------
* Each detection in a frame is a (centroid_px, contour, frame_gray) tuple.
* ``CentroidTracker.update()`` matches detections to existing tracks by nearest
  centroid (Hungarian-free greedy, sufficient for a single conveyor lane).
* A track becomes ACTIVE when its centroid first crosses the entry line.
* A track emits a ``TrackedBillet`` (finalised record) when its centroid crosses
  the exit line, or when it has not been seen for ``max_lost_frames`` frames.
* ``median_measurement()`` selects the best-N frames by Laplacian-variance
  sharpness and returns a per-field median ``Measurement``.

Coordinate frame: image origin (0,0) top-left, +X right, +Y down.
All pixel coordinates are in the rectified/undistorted image plane.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import cv2
import numpy as np

from billetvision.vision.measure import Measurement

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sharpness
# ---------------------------------------------------------------------------

def sharpness(gray: np.ndarray) -> float:
    """Laplacian-variance sharpness score for a grayscale image or crop.

    Higher is sharper.  Returns 0.0 for empty/None inputs.
    """
    if gray is None or gray.size == 0:
        return 0.0
    lap = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F)
    return float(np.var(lap))


def sharpness_of_contour(gray: np.ndarray, contour: np.ndarray) -> float:
    """Compute sharpness restricted to the bounding-box crop of *contour*."""
    if gray is None or contour is None:
        return 0.0
    x, y, w, h = cv2.boundingRect(contour)
    crop = gray[y : y + h, x : x + w]
    return sharpness(crop)


# ---------------------------------------------------------------------------
# Frame sample stored per track
# ---------------------------------------------------------------------------

@dataclass
class FrameSample:
    """One frame contribution to a track's measurement pool."""

    frame_gray: np.ndarray      # grayscale crop or full frame
    contour: np.ndarray         # billet contour in full-frame pixel coords
    measurement: Measurement    # measurement derived from this frame
    sharpness_score: float      # Laplacian variance — higher = sharper


# ---------------------------------------------------------------------------
# Track state machine
# ---------------------------------------------------------------------------

class TrackState(Enum):
    PENDING = auto()   # seen but not yet past entry line
    ACTIVE = auto()    # past entry line, collecting frames
    DONE = auto()      # past exit line or timed out — record emitted


@dataclass
class BilletTrack:
    """State for a single billet being tracked across frames."""

    track_id: int
    state: TrackState = TrackState.PENDING
    centroid: tuple[float, float] = (0.0, 0.0)
    centroid_history: list[tuple[float, float]] = field(default_factory=list)
    frames: list[FrameSample] = field(default_factory=list)
    lost_count: int = 0          # consecutive frames without a matching detection
    age: int = 0                 # total frames this track has been alive
    frame_index: int = 0         # pipeline frame counter at creation


# ---------------------------------------------------------------------------
# Finalised record emitted when a track exits
# ---------------------------------------------------------------------------

@dataclass
class TrackedBillet:
    """Finalised billet record produced when a track crosses the exit line."""

    track_id: int
    measurement: Measurement     # median over best-N frames
    best_frame_gray: np.ndarray  # sharpest frame for OCR / snapshot
    best_contour: np.ndarray     # contour from the sharpest frame
    centroid_path: list[tuple[float, float]]
    frame_count: int             # total frames the billet was observed


# ---------------------------------------------------------------------------
# Median measurement
# ---------------------------------------------------------------------------

def median_measurement(
    samples: list[FrameSample],
    best_n: int = 5,
) -> tuple[Measurement, FrameSample]:
    """Compute a per-field median ``Measurement`` from the best-N sharpest frames.

    Args:
        samples: All frame samples collected for a track.
        best_n: Maximum number of sharpest frames to include.

    Returns:
        (median_measurement, sharpest_sample) where sharpest_sample is the
        single best frame (for OCR / snapshot use).
    """
    if not samples:
        raise ValueError("median_measurement: no samples provided")

    ranked = sorted(samples, key=lambda s: s.sharpness_score, reverse=True)
    top = ranked[: max(1, min(best_n, len(ranked)))]
    best = top[0]

    def _med(values: list[Optional[float]]) -> Optional[float]:
        valid = [v for v in values if v is not None]
        if not valid:
            return None
        return float(np.median(valid))

    ms = [s.measurement for s in top]

    shape = ms[0].shape
    med = Measurement(
        shape=shape,
        length_mm=float(np.median([m.length_mm for m in ms])),
        width_mm=float(np.median([m.width_mm for m in ms])),
        height_mm=float(np.median([m.height_mm for m in ms])),
        diameter_mm=_med([m.diameter_mm for m in ms]),
        ovality=_med([m.ovality for m in ms]),
        diag_diff_mm=_med([m.diag_diff_mm for m in ms]),
        camber_mm=_med([m.camber_mm for m in ms]),
        cross_section_var_mm=_med([m.cross_section_var_mm for m in ms]),
        surface_anomaly_score=float(np.median([m.surface_anomaly_score for m in ms])),
        defects=ms[0].defects,  # defects from best frame; union if needed
    )
    return med, best


# ---------------------------------------------------------------------------
# Centroid tracker
# ---------------------------------------------------------------------------

class CentroidTracker:
    """Greedy nearest-centroid tracker for a single-lane conveyor.

    Parameters
    ----------
    entry_x : int
        X-coordinate of the entry line.  A billet becomes ACTIVE when its
        centroid's x first exceeds this value (left-to-right conveyor).
    exit_x : int
        X-coordinate of the exit line.  A ACTIVE billet emits a record when
        its centroid's x first exceeds this value.
    max_distance_px : float
        Maximum centroid distance (px) to associate a detection with a track.
    max_lost_frames : int
        Expire an unseen track after this many consecutive frames without a
        matching detection.
    best_n : int
        Number of sharpest frames used by ``median_measurement``.
    direction : str
        "left_to_right" (default) or "right_to_left".  Controls which
        direction triggers entry/exit.
    """

    def __init__(
        self,
        entry_x: int = 50,
        exit_x: int = 1230,
        max_distance_px: float = 80.0,
        max_lost_frames: int = 10,
        best_n: int = 5,
        direction: str = "left_to_right",
    ) -> None:
        self.entry_x = entry_x
        self.exit_x = exit_x
        self.max_distance_px = max_distance_px
        self.max_lost_frames = max_lost_frames
        self.best_n = best_n
        self.direction = direction

        self._next_id: int = 1
        self._tracks: dict[int, BilletTrack] = {}
        self._frame_index: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        detections: list[tuple[tuple[float, float], np.ndarray, np.ndarray, Measurement]],
    ) -> list[TrackedBillet]:
        """Process one frame's detections and return any newly finalised billets.

        Args:
            detections: List of (centroid_px, contour, frame_gray, measurement).
                ``centroid_px`` is (x, y) in pixel coords.
                ``frame_gray`` may be the full frame or a crop.
                ``measurement`` is the per-frame Measurement from measure().

        Returns:
            List of ``TrackedBillet`` records that exited or timed out this frame.
        """
        self._frame_index += 1
        emitted: list[TrackedBillet] = []

        # ---- match detections to tracks --------------------------------
        active_track_ids = list(self._tracks.keys())
        unmatched_detections = list(range(len(detections)))
        matched_pairs: list[tuple[int, int]] = []  # (track_id, det_idx)

        if active_track_ids and detections:
            track_centroids = np.array(
                [self._tracks[tid].centroid for tid in active_track_ids],
                dtype=np.float64,
            )
            det_centroids = np.array(
                [d[0] for d in detections], dtype=np.float64
            )

            # Greedy matching: for each detection find the nearest track
            used_tracks: set[int] = set()
            for det_idx, dc in enumerate(det_centroids):
                dists = np.linalg.norm(track_centroids - dc, axis=1)
                order = np.argsort(dists)
                for ti in order:
                    tid = active_track_ids[ti]
                    if tid in used_tracks:
                        continue
                    if dists[ti] <= self.max_distance_px:
                        matched_pairs.append((tid, det_idx))
                        used_tracks.add(tid)
                        if det_idx in unmatched_detections:
                            unmatched_detections.remove(det_idx)
                    break

        # ---- update matched tracks ------------------------------------
        for tid, det_idx in matched_pairs:
            cx, cy = detections[det_idx][0]
            contour = detections[det_idx][1]
            frame_gray = detections[det_idx][2]
            meas = detections[det_idx][3]

            track = self._tracks[tid]
            track.centroid = (cx, cy)
            track.centroid_history.append((cx, cy))
            track.lost_count = 0
            track.age += 1

            sharp = sharpness_of_contour(frame_gray, contour)
            track.frames.append(
                FrameSample(
                    frame_gray=frame_gray,
                    contour=contour,
                    measurement=meas,
                    sharpness_score=sharp,
                )
            )

            # State transitions
            if track.state == TrackState.PENDING and self._past_entry(cx):
                track.state = TrackState.ACTIVE
                logger.debug("Track %d became ACTIVE at x=%.1f", tid, cx)

            if track.state == TrackState.ACTIVE and self._past_exit(cx):
                track.state = TrackState.DONE
                emitted.append(self._finalise(track))
                del self._tracks[tid]
                continue

        # ---- increment lost counter for unmatched tracks ----------------
        for tid in list(self._tracks.keys()):
            matched_tids = {p[0] for p in matched_pairs}
            if tid not in matched_tids:
                self._tracks[tid].lost_count += 1
                if self._tracks[tid].lost_count >= self.max_lost_frames:
                    track = self._tracks[tid]
                    if track.state == TrackState.ACTIVE and track.frames:
                        logger.debug(
                            "Track %d timed out after %d lost frames", tid, track.lost_count
                        )
                        emitted.append(self._finalise(track))
                    del self._tracks[tid]

        # ---- create new tracks for unmatched detections -----------------
        for det_idx in unmatched_detections:
            cx, cy = detections[det_idx][0]

            # Never start a new track for a detection that is already past the
            # exit line — it is a billet tail or noise from a departing billet.
            if self._past_exit(cx):
                continue

            contour = detections[det_idx][1]
            frame_gray = detections[det_idx][2]
            meas = detections[det_idx][3]

            tid = self._next_id
            self._next_id += 1
            track = BilletTrack(
                track_id=tid,
                centroid=(cx, cy),
                frame_index=self._frame_index,
            )
            track.centroid_history.append((cx, cy))

            sharp = sharpness_of_contour(frame_gray, contour)
            track.frames.append(
                FrameSample(
                    frame_gray=frame_gray,
                    contour=contour,
                    measurement=meas,
                    sharpness_score=sharp,
                )
            )

            if self._past_entry(cx):
                track.state = TrackState.ACTIVE

            self._tracks[tid] = track
            logger.debug("New track %d at (%.1f, %.1f)", tid, cx, cy)

        return emitted

    def flush(self) -> list[TrackedBillet]:
        """Finalise and return all active tracks (call at end-of-stream)."""
        emitted: list[TrackedBillet] = []
        for tid in list(self._tracks.keys()):
            track = self._tracks[tid]
            if track.state == TrackState.ACTIVE and track.frames:
                emitted.append(self._finalise(track))
            del self._tracks[tid]
        return emitted

    def reset(self) -> None:
        """Reset tracker state (e.g. between test clips)."""
        self._tracks.clear()
        self._next_id = 1
        self._frame_index = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _past_entry(self, cx: float) -> bool:
        if self.direction == "left_to_right":
            return cx >= self.entry_x
        return cx <= self.entry_x

    def _past_exit(self, cx: float) -> bool:
        if self.direction == "left_to_right":
            return cx >= self.exit_x
        return cx <= self.exit_x

    def _finalise(self, track: BilletTrack) -> TrackedBillet:
        """Compute median measurement and return a TrackedBillet."""
        med_meas, best_sample = median_measurement(track.frames, self.best_n)
        return TrackedBillet(
            track_id=track.track_id,
            measurement=med_meas,
            best_frame_gray=best_sample.frame_gray,
            best_contour=best_sample.contour,
            centroid_path=list(track.centroid_history),
            frame_count=len(track.frames),
        )
