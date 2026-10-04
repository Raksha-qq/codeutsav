#!/usr/bin/env python3
"""Run the whole BilletVision pipeline + operator dashboard.

    python scripts/run_demo.py --source data/raw/demo.mp4
    python scripts/run_demo.py --synthetic            # physically consistent belt demo
    python scripts/run_demo.py --source 0             # webcam index 0

The pipeline starts with the API; open http://localhost:8000.  Everything runs
offline — Telegram/webhook delivery stays disabled unless its env vars are set.
"""
from __future__ import annotations

import argparse
import logging
import sys
import threading
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def build_overrides(args: argparse.Namespace) -> dict:
    """Translate CLI flags into config overrides (merged over config/config.yaml)."""
    from billetvision import synthetic

    synthetic_video = ROOT / "data/raw" / synthetic.demo_video_name()
    overrides: dict = {}
    source = args.source
    if args.synthetic:
        if not synthetic_video.exists():
            logging.getLogger("run_demo").info("Generating %s (one-off, ~1 min)...", synthetic_video.name)
            frames = synthetic.render_belt_frames(synthetic.show_props(ROOT / "data/ground_truth.csv"), scene=synthetic.DEMO_SCENE)
            synthetic.write_video(frames, synthetic_video)
        source = str(synthetic_video)
        overrides = synthetic.DEMO_SCENE.overrides()
    if source is not None:
        overrides.setdefault("capture", {})["source"] = int(source) if str(source).isdigit() else source
    if args.once:
        overrides.setdefault("capture", {})["loop"] = False
    if args.profile:
        overrides.setdefault("vision", {})["active_profile"] = args.profile
    return overrides


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", help="video file, image folder, or webcam index (default: config/config.yaml)")
    parser.add_argument("--synthetic", action="store_true", help="use the generated belt demo (creates it if missing)")
    parser.add_argument("--profile", help="active billet profile, e.g. square_130 or round_150")
    parser.add_argument("--once", action="store_true", help="play a video/folder once instead of looping")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true", help="do not open the dashboard automatically")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    import os

    os.chdir(ROOT)  # config paths in config.yaml are relative to the repo root

    import uvicorn

    from billetvision.api.main import app
    from billetvision.pipeline import pipeline

    pipeline.overrides = build_overrides(args)
    if not args.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(f"http://{args.host}:{args.port}")).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
