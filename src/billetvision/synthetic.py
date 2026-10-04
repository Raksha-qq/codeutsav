"""Synthetic conveyor scenes with exact ground truth.

Used for the demo video, the accuracy report and the end-to-end tests.  Every
billet is rendered with analytic sub-pixel anti-aliasing at a known scale, so
the true dimensions in millimetres are known exactly and the measured error
reflects the vision pipeline, not the scene.

Geometry (all in the image plane, +X right, +Y down)::

    scale            = SCALE_MM_PER_PX mm/px     (marker is 50 mm = 100 px)
    belt speed       = BELT_SPEED_MM_S           (billets travel left -> right)
    billet centre y  = frame_height / 2

This is a *simulation*: it validates the algorithms against known geometry; it
does not replace calibration against calipers on real footage.
"""
from __future__ import annotations

import csv
import dataclasses
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence

import cv2
import numpy as np

FRAME_W, FRAME_H = 1280, 720
FPS = 15.0
SCALE_MM_PER_PX = 0.5
BELT_SPEED_MM_S = 250.0
MARKER_MM = 50.0
GAP_MM = 150.0               # empty belt between consecutive billets
ROI_BOX = (160, 170, 1240, 650)   # inspection ROI the synthetic scene is designed for

# Config overrides under which the pipeline reads these scenes correctly:
# billets are far longer than the field of view, so length = belt speed x time.
PIPELINE_OVERRIDES = {
    "vision": {
        "roi_box": list(ROI_BOX),
        "length_mode": "belt_speed",
        "conveyor_speed_mm_s": BELT_SPEED_MM_S,
        "direction": "left_to_right",
    },
}


@dataclass(frozen=True)
class Scene:
    """Camera/belt geometry of a synthetic scene (all lengths in mm, ROI in px)."""

    scale: float                  # mm per pixel
    marker_mm: float              # side of the ArUco calibration marker
    gap_mm: float                 # empty belt between consecutive billets
    belt_speed: float             # mm/s
    roi: tuple                    # inspection ROI (x1, y1, x2, y2) in px
    length_mode: str              # "belt_speed" (longer than the FOV) | "direct" (whole billet in view)
    look: str = "plain"           # "plain" (exact test scene) | "realistic" (varied tone, marks, position, skew)

    def overrides(self) -> dict:
        """Pipeline config overrides under which this scene is read correctly."""
        return {"vision": {
            "roi_box": list(self.roi), "length_mode": self.length_mode,
            "conveyor_speed_mm_s": self.belt_speed, "direction": "left_to_right",
            "marker_size_mm": self.marker_mm,
        }}


# Accuracy report / tests: 1 m billets are far longer than the field of view.
DEFAULT_SCENE = Scene(SCALE_MM_PER_PX, MARKER_MM, GAP_MM, BELT_SPEED_MM_S, ROI_BOX, "belt_speed")
# Operator demo: camera far enough back that a whole 1 m billet (1111 px of 1280) is in view
# with clear belt either side, so billets read as separate pieces and are measured directly.
DEMO_SCENE = Scene(0.9, 100.0, 400.0, 250.0, (0, 170, FRAME_W, 650), "direct", look="realistic")
HUD_H = 160                  # dark strip at the top that holds the marker
_BG = 34
_BILLET_BGR = (168, 164, 160)
_TEXT_BGR = (28, 30, 34)


@dataclass
class PropSpec:
    """One test billet and its caliper ground truth (millimetres)."""

    prop_id: str
    shape: str                        # "square" | "round"
    length_mm: float
    width_mm: Optional[float]
    height_mm: Optional[float]
    diameter_mm: Optional[float]
    heat_id: str
    expected_status: str


def _opt(value: str) -> Optional[float]:
    return float(value) if value not in ("", None) else None


def load_ground_truth(path: str | Path = "data/ground_truth.csv") -> List[PropSpec]:
    """Read ``data/ground_truth.csv`` into PropSpec rows."""
    with Path(path).open(newline="", encoding="utf-8") as fh:
        return [
            PropSpec(
                prop_id=r["prop_id"], shape=r["shape"],
                length_mm=float(r["caliper_length_mm"]),
                width_mm=_opt(r["caliper_width_mm"]), height_mm=_opt(r["caliper_height_mm"]),
                diameter_mm=_opt(r["caliper_diameter_mm"]),
                heat_id=r["heat_id"], expected_status=r["expected_status"],
            )
            for r in csv.DictReader(fh)
        ]


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def background(rng: np.random.Generator, with_marker: bool = True, scene: Scene = None) -> np.ndarray:
    """Dark belt with faint cross-rails, a HUD strip and the ArUco calibration marker."""
    bg = np.full((FRAME_H, FRAME_W, 3), _BG, dtype=np.uint8)
    bg[:HUD_H] = 28
    for x in range(0, FRAME_W, 80):
        bg[HUD_H:, x:x + 2] = _BG + 8
    scene = scene or DEFAULT_SCENE
    if with_marker:
        side = int(round(scene.marker_mm / scene.scale))
        marker = cv2.aruco.generateImageMarker(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), 0, side
        )
        quiet = 12  # white margin around the marker improves detection
        tile = np.full((side + 2 * quiet, side + 2 * quiet), 255, dtype=np.uint8)
        tile[quiet:quiet + side, quiet:quiet + side] = marker
        y0, x0 = 20, 20  # the detector needs some margin to the frame border
        bg[y0:y0 + tile.shape[0], x0:x0 + tile.shape[1]] = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
    return bg


def _stamp(tex: np.ndarray, spec: PropSpec, x: int, y_jitter: int = 0) -> None:
    """Stamp ``HEAT: <id>`` once, starting at column ``x``."""
    height_px, width_px = tex.shape[:2]
    label = f"HEAT: {spec.heat_id}"
    scale = max(0.8, height_px / 190.0)
    thickness = max(2, int(round(scale * 2.2)))
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, scale, thickness)
    if x + tw + 10 < width_px:
        cv2.putText(tex, label, (x, height_px // 2 + th // 2 + y_jitter), cv2.FONT_HERSHEY_DUPLEX,
                    scale, _TEXT_BGR, thickness, cv2.LINE_AA)


def billet_texture(
    spec: PropSpec, width_px: int, height_px: int, rng: np.random.Generator,
    look: str = "plain", tone: int = 0, stamp_frac: float = 0.2,
) -> np.ndarray:
    """Billet surface strip with a stamped ``HEAT: <id>`` label.

    ``plain`` (the test scene): uniform steel with mild texture, stamped twice along the bar.
    ``realistic`` (the demo): per-billet steel tone, soft oxide-scale blotches, a few scratches,
    slightly darkened long edges and a single stamp ``stamp_frac`` of the way along the bar.
    """
    tex = np.empty((height_px, width_px, 3), dtype=np.uint8)
    tex[:] = _BILLET_BGR
    # gentle lengthwise shading + grain, so thresholds are not trivially perfect
    shade = np.linspace(-6, 6, height_px, dtype=np.float32)[:, None, None]
    grain = rng.normal(0, 3.0, tex.shape).astype(np.float32)
    if look != "realistic":
        tex = np.clip(tex.astype(np.float32) + shade + grain, 0, 255).astype(np.uint8)
        for x in (int(0.18 * width_px), int(0.62 * width_px)):  # stamped twice along the bar
            _stamp(tex, spec, x)
        return tex

    warm = rng.uniform(-5, 5, 3).astype(np.float32)               # slight per-billet colour cast
    img = tex.astype(np.float32) + tone + warm + shade + grain
    scale_mask = np.zeros((height_px, width_px), np.float32)
    for _ in range(int(rng.integers(6, 11))):                      # oxide-scale blotches
        centre = (int(rng.uniform(0, width_px)), int(rng.uniform(0, height_px)))
        axes = (int(rng.uniform(40, 170)), int(rng.uniform(8, 40)))
        cv2.ellipse(scale_mask, centre, axes, float(rng.uniform(-8, 8)), 0, 360, float(rng.uniform(0.4, 1.0)), -1)
    scale_mask = cv2.GaussianBlur(scale_mask, (0, 0), 14)
    img -= (scale_mask * rng.uniform(10, 20))[..., None]
    for _ in range(int(rng.integers(2, 5))):                       # longitudinal scratches
        y0 = int(rng.uniform(0.15, 0.85) * height_px)
        x0 = int(rng.uniform(0, 0.7) * width_px)
        x1 = x0 + int(rng.uniform(0.1, 0.3) * width_px)
        layer = np.zeros((height_px, width_px), np.float32)
        cv2.line(layer, (x0, y0), (x1, y0 + int(rng.integers(-3, 4))), 1.0, 1, cv2.LINE_AA)
        img += (layer * rng.choice([-14.0, 12.0]))[..., None]
    edge = np.minimum(np.arange(height_px), np.arange(height_px)[::-1]).astype(np.float32)
    img *= (0.92 + 0.08 * np.clip(edge / 7.0, 0, 1))[:, None, None]  # rounded, slightly darker long edges
    tex = np.clip(img, 0, 255).astype(np.uint8)
    _stamp(tex, spec, int(stamp_frac * width_px), int(rng.integers(-10, 11)))
    return tex


def _coverage(lo: float, hi: float, size: int) -> np.ndarray:
    """Per-pixel coverage (0-1) of the interval [lo, hi) over ``size`` unit pixels."""
    idx = np.arange(size, dtype=np.float32)
    return np.clip(np.minimum(idx + 1.0, hi) - np.maximum(idx, lo), 0.0, 1.0)


def draw_bar(
    canvas: np.ndarray, texture: np.ndarray, x_left: float, x_right: float,
    y_top: float, y_bottom: float,
) -> None:
    """Composite a texture-filled rectangle with exact analytic edge anti-aliasing."""
    h, w = canvas.shape[:2]
    cov_x = _coverage(x_left, x_right, w)
    cov_y = _coverage(y_top, y_bottom, h)
    xs = np.nonzero(cov_x)[0]
    ys = np.nonzero(cov_y)[0]
    if xs.size == 0 or ys.size == 0:
        return
    xa, xb, ya, yb = xs[0], xs[-1] + 1, ys[0], ys[-1] + 1
    tx = np.clip(np.arange(xa, xb) - int(np.floor(x_left)), 0, texture.shape[1] - 1)
    ty = np.clip(np.arange(ya, yb) - int(np.floor(y_top)), 0, texture.shape[0] - 1)
    layer = texture[np.ix_(ty, tx)].astype(np.float32)
    alpha = (cov_y[ya:yb, None] * cov_x[None, xa:xb])[..., None]
    region = canvas[ya:yb, xa:xb].astype(np.float32)
    canvas[ya:yb, xa:xb] = np.clip(region * (1 - alpha) + layer * alpha, 0, 255).astype(np.uint8)


def _variant(rng: np.random.Generator) -> dict:
    """Per-billet variation for the realistic look.

    No skew on purpose: any tilt adds a whole pixel (~0.8% of a 130 mm width at this scale) to the
    contour-based width, which would push in-spec pieces into REWORK - a measurement limit, not realism.
    """
    return {
        "tone": int(rng.integers(-14, 15)),          # steel brightness
        "dy": float(rng.uniform(-45, 45)),           # lateral drift on the belt (px)
        "gap": float(rng.uniform(0.75, 1.7)),        # gap to the next billet, x scene gap
        "stamp": float(rng.uniform(0.08, 0.32)),     # where along the bar the heat ID is stamped
    }


def render_belt_frames(
    props: Sequence[PropSpec],
    seed: int = 7,
    noise_sigma: float = 2.0,
    fps: float = FPS,
    lead_in_frames: int = 8,
    scene: Scene = None,
) -> Iterator[np.ndarray]:
    """Yield BGR frames of ``props`` (square billets, top view) crossing the belt.

    Each billet enters from the left at the scene's belt speed; its width is the caliper
    width, so the image is a sub-pixel-accurate rendering of the ground truth.  Frames carry
    sensor noise and slow brightness drift.  With ``scene.look == "realistic"`` every billet
    also gets its own tone, surface marks, lateral position and gap.
    """
    scene = scene or DEFAULT_SCENE
    realistic = scene.look == "realistic"
    rng = _rng(seed)
    base = background(rng, scene=scene)
    px_per_frame = scene.belt_speed / scene.scale / fps
    cy0 = (scene.roi[1] + scene.roi[3]) / 2.0
    frame_idx = 0
    for _ in range(lead_in_frames):
        yield _finish(base, rng, noise_sigma, frame_idx)
        frame_idx += 1
    for spec in props:
        length_px = spec.length_mm / scene.scale
        width_px = (spec.width_mm or spec.diameter_mm or 130.0) / scene.scale
        var = _variant(rng) if realistic else {"tone": 0, "dy": 0.0, "gap": 1.0, "stamp": 0.2}
        tex = billet_texture(spec, int(np.ceil(length_px)) + 2, int(np.ceil(width_px)) + 2, rng,
                             look=scene.look, tone=var["tone"], stamp_frac=var["stamp"])
        cy = cy0 + var["dy"]
        x_left = -length_px - 2.0
        travel = FRAME_W + length_px + scene.gap_mm / scene.scale * var["gap"]
        steps = int(np.ceil(travel / px_per_frame))
        for i in range(steps):
            canvas = base.copy()
            xl = x_left + i * px_per_frame
            draw_bar(canvas, tex, xl, xl + length_px, cy - width_px / 2, cy + width_px / 2)
            yield _finish(canvas, rng, noise_sigma, frame_idx, already_copy=True)
            frame_idx += 1


_NOISE_BANK: dict = {}


def _noise(rng: np.random.Generator, sigma: float, shape: tuple, idx: int) -> np.ndarray:
    """Gaussian noise drawn from a small pre-generated bank (fast; rolled per frame)."""
    key = (sigma, shape)
    if key not in _NOISE_BANK:
        _NOISE_BANK[key] = [rng.normal(0, sigma, shape).astype(np.float32) for _ in range(6)]
    bank = _NOISE_BANK[key]
    return np.roll(bank[idx % len(bank)], shift=(idx * 37) % shape[1], axis=1)


def _finish(
    canvas: np.ndarray, rng: np.random.Generator, sigma: float, idx: int, already_copy: bool = False
) -> np.ndarray:
    """Add slow exposure drift and Gaussian sensor noise."""
    img = canvas if already_copy else canvas.copy()
    gain = 1.0 + 0.04 * np.sin(idx / 40.0)
    noisy = img.astype(np.float32) * gain + _noise(rng, sigma, img.shape, idx)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def demo_props(path: str | Path = "data/ground_truth.csv") -> List[PropSpec]:
    """Square props from the ground-truth file (the belt demo's billets)."""
    return [p for p in load_ground_truth(path) if p.shape == "square"]


# Operator demo line-up: the ground-truth squares, with three pieces pushed clearly out of the
# 130 +/- 1 mm x 1000 +/- 10 mm spec (each beyond 2x tolerance, so FAIL rather than REWORK).
_DEMO_OVERRIDES = {
    "PROP-04": {"width_mm": 138.0, "height_mm": 138.0},     # oversize section (worn finishing roll)
    "PROP-06": {"width_mm": 121.0, "height_mm": 121.0},     # undersize section
    "PROP-11": {"length_mm": 940.0},                         # 60 mm short (cut-to-length error)
}


def show_props(path: str | Path = "data/ground_truth.csv") -> List[PropSpec]:
    """Billets for the operator demo: mostly in-spec, three clearly out of tolerance."""
    out = []
    for p in demo_props(path):
        change = _DEMO_OVERRIDES.get(p.prop_id)
        out.append(dataclasses.replace(p, expected_status="FAIL", **change) if change else p)
    return out


_RENDER_VERSION = 3        # bump when the demo rendering code changes, so cached videos are rebuilt


def demo_video_name() -> str:
    """Cache file name that changes whenever the demo scene or line-up changes (no stale videos)."""
    sig = hashlib.md5(repr((_RENDER_VERSION, DEMO_SCENE, show_props())).encode()).hexdigest()[:8]
    return f"demo_belt_{sig}.mp4"


def write_video(frames: Iterator[np.ndarray], out_path: str | Path, fps: float = FPS) -> int:
    """Encode ``frames`` to an MP4 (mp4v); returns the number of frames written."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (FRAME_W, FRAME_H))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {out}")
    n = 0
    try:
        for frame in frames:
            writer.write(frame)
            n += 1
    finally:
        writer.release()
    return n


def render_end_view(
    diameter_mm: float, seed: int = 0, noise_sigma: float = 2.0, ovality_pct: float = 0.0,
    scale_mm_per_px: float = SCALE_MM_PER_PX,
) -> np.ndarray:
    """End-on image of a round billet (ellipse) for the diameter/ovality check."""
    rng = _rng(seed)
    img = background(rng)  # includes the calibration marker
    d_px = diameter_mm / scale_mm_per_px
    major = d_px * (1 + ovality_pct / 200.0)
    minor = d_px * (1 - ovality_pct / 200.0)
    big = 4  # supersample for sub-pixel accurate edges
    cx, cy = (ROI_BOX[0] + ROI_BOX[2]) / 2.0, (ROI_BOX[1] + ROI_BOX[3]) / 2.0
    layer = np.zeros((FRAME_H * big, FRAME_W * big), dtype=np.uint8)
    cv2.ellipse(layer, (int(cx * big), int(cy * big)),
                (int(round(major * big / 2)), int(round(minor * big / 2))), 0, 0, 360, 255, -1)
    alpha = cv2.resize(layer, (FRAME_W, FRAME_H), interpolation=cv2.INTER_AREA).astype(np.float32)[..., None] / 255.0
    fill = np.array(_BILLET_BGR, dtype=np.float32)
    out = img.astype(np.float32) * (1 - alpha) + fill * alpha
    return np.clip(out + rng.normal(0, noise_sigma, out.shape), 0, 255).astype(np.uint8)

