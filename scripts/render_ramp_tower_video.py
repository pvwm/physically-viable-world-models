from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from npm_sim.materials import MATERIALS
from npm_sim.ramp_tower import VIDEO_NUM_FRAMES, render_video


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--ball-material",
        type=str,
        default="steel",
        choices=sorted(MATERIALS),
        help="Material preset for the rolling sphere.",
    )
    parser.add_argument(
        "--cube-material",
        type=str,
        default="wood",
        choices=sorted(MATERIALS),
        help="Material preset for the cubes and static ground surfaces.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=VIDEO_NUM_FRAMES,
        help="Number of simulation frames to render into the video.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="outputs/ramp_tower.mp4",
        help="MP4 output path.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional Warp device override, for example cpu or cuda:0.",
    )
    return parser


def main(argv: list[str] | None = None) -> Path:
    parser = create_parser()
    args = parser.parse_args(argv)
    if args.num_frames <= 0:
        parser.error("--num-frames must be positive")
    return render_video(
        output_path=args.output_path,
        ball_material=args.ball_material,
        cube_material=args.cube_material,
        num_frames=args.num_frames,
        device=args.device,
    )


if __name__ == "__main__":
    output = main()
    print(output)
