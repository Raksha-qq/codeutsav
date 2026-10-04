#!/usr/bin/env python3
"""Generate the synthetic belt demo video (physically consistent, exact ground truth).

Renders the square props of ``data/ground_truth.csv`` crossing the belt at a
known scale and belt speed (see ``billetvision.synthetic``).  The output is a
simulation: it exercises the whole pipeline against known geometry, it is not
a substitute for footage of real props measured with calipers.

    python scripts/make_demo_video.py [--out data/raw/demo_belt_<hash>.mp4]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from billetvision import synthetic  # noqa: E402

logger = logging.getLogger("make_demo_video")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(ROOT / "data/raw" / synthetic.demo_video_name()), help="output MP4 path")
    parser.add_argument("--ground-truth", default=str(ROOT / "data/ground_truth.csv"))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    props = synthetic.show_props(args.ground_truth)
    logger.info("Rendering %d props: %s", len(props), ", ".join(p.prop_id for p in props))
    frames = synthetic.write_video(synthetic.render_belt_frames(props, seed=args.seed, scene=synthetic.DEMO_SCENE), args.out)
    size_mb = Path(args.out).stat().st_size / 1e6
    logger.info("Wrote %s (%d frames, %.1f s, %.1f MB)", args.out, frames, frames / synthetic.FPS, size_mb)
    return 0


if __name__ == "__main__":
    sys.exit(main())
