"""Billet length from belt speed x time-in-view.

For pieces longer than the camera field of view (real billets are 6-12 m) the
length cannot be measured directly.  Instead the head and tail of the billet
are timed crossing a fixed line in the image and

    length_mm = belt_speed_mm_s * (t_tail - t_head)

Units: seconds, pixels (full-frame, +X right) and millimetres.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

Timeline = Sequence[Tuple[float, float, float]]  # (t_s, min_x_px, max_x_px)


def _crossing_time(
    points: Sequence[Tuple[float, float]], line_x: float, increasing: bool
) -> Optional[float]:
    """Time at which ``x`` first reaches ``line_x`` (linear interpolation).

    Args:
        points: ``(t, x)`` samples in time order.
        line_x: Line position in pixels.
        increasing: True if the edge moves towards larger x.

    Returns:
        Interpolated crossing time, or None if the edge never crossed the line
        or it was already beyond the line at the first sample.
    """
    prev: Optional[Tuple[float, float]] = None
    for t, x in points:
        past = x >= line_x if increasing else x <= line_x
        if past:
            if prev is None:
                return None  # already past the line when first seen
            t0, x0 = prev
            if x == x0:
                return t
            frac = (line_x - x0) / (x - x0)
            return t0 + frac * (t - t0)
        prev = (t, x)
    return None


def length_from_belt_speed(
    timeline: Timeline,
    line_x: float,
    speed_mm_s: float,
    direction: str = "left_to_right",
) -> Optional[float]:
    """Estimate billet length in mm from head/tail crossing times.

    Args:
        timeline: Per-frame ``(t_s, min_x_px, max_x_px)`` of the billet's
            bounding box, in time order.
        line_x: X coordinate (px) of the timing line.
        speed_mm_s: Calibrated belt speed in mm/s.
        direction: ``"left_to_right"`` or ``"right_to_left"``.

    Returns:
        Length in mm, or None when head and tail were not both seen crossing
        the line (caller should fall back to direct measurement).
    """
    if speed_mm_s <= 0 or len(timeline) < 2:
        return None
    ltr = direction == "left_to_right"
    head = [(t, hi if ltr else lo) for t, lo, hi in timeline]
    tail = [(t, lo if ltr else hi) for t, lo, hi in timeline]
    t_head = _crossing_time(head, line_x, increasing=ltr)
    t_tail = _crossing_time(tail, line_x, increasing=ltr)
    if t_head is None or t_tail is None or t_tail <= t_head:
        return None
    return float(speed_mm_s * (t_tail - t_head))
