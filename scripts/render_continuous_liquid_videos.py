"""Render continuous-surface versions of the robot-arm pour videos.

The existing pour videos use Genesis ``vis_mode="particle"``. This script
keeps those outputs untouched, renders matching ``*_continuous.mp4`` files with
``vis_mode="recon"``, and rebuilds the two calibration comparison videos from
the continuous sources.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_DIR = ROOT / "outputs" / "calibration_demo"
DEFAULT_ESTIMATED_VISCOSITY = 0.005760280121128763


def _python_cmd(args: argparse.Namespace) -> list[str]:
    if args.use_current_python:
        return [sys.executable]
    return ["conda", "run", "-n", args.conda_env, "python"]


def _run(command: list[str], *, dry_run: bool = False) -> None:
    print("+", " ".join(str(part) for part in command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True, cwd=ROOT)


def _best_estimated_viscosity() -> float:
    history_path = CALIBRATION_DIR / "bo_history.json"
    if not history_path.exists():
        return DEFAULT_ESTIMATED_VISCOSITY
    payload = json.loads(history_path.read_text())
    best = payload.get("best") or {}
    return float(best.get("viscosity", DEFAULT_ESTIMATED_VISCOSITY))


def _estimated_video_name(viscosity: float) -> str:
    return f"robotic_arm_pour_estimated_mu_{viscosity:.6f}".replace(".", "p") + "_continuous.mp4"


def _ffmpeg_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


def _labeled_scale(input_index: int, label: str, out_label: str, *, height: int = 360) -> str:
    return (
        f"[{input_index}:v]"
        f"scale=-2:{height}:flags=lanczos,"
        "crop=640:360,"
        "setsar=1,"
        "drawtext="
        f"text='{_ffmpeg_text(label)}':"
        "x=10:y=10:fontsize=26:fontcolor=white:"
        "box=1:boxcolor=black@0.65:boxborderw=8"
        f"[{out_label}]"
    )


def _build_reference_grid(
    *,
    target: Path,
    estimated: Path,
    water: Path,
    honey: Path,
    output: Path,
    estimated_viscosity: float,
    dry_run: bool,
) -> None:
    filters = [
        _labeled_scale(0, "Target unknown (mu = 0.006)", "v0"),
        _labeled_scale(1, f"Predicted (mu = {estimated_viscosity:.5f})", "v1"),
        _labeled_scale(2, "Water (mu = 0.001)", "v2"),
        _labeled_scale(3, "Honey (mu = 0.030)", "v3"),
        "[v0][v1]hstack=inputs=2:shortest=1[top]",
        "[v2][v3]hstack=inputs=2:shortest=1[bottom]",
        "[top][bottom]vstack=inputs=2:shortest=1,format=yuv420p[v]",
    ]
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(target),
        "-i",
        str(estimated),
        "-i",
        str(water),
        "-i",
        str(honey),
        "-filter_complex",
        ";".join(filters),
        "-map",
        "[v]",
        "-an",
        "-r",
        "60",
        "-vcodec",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    _run(command, dry_run=dry_run)


def _build_target_vs_estimated(
    *,
    target: Path,
    estimated: Path,
    output: Path,
    estimated_viscosity: float,
    dry_run: bool,
) -> None:
    filters = [
        _labeled_scale(0, "Target unknown (mu = 0.006)", "v0"),
        _labeled_scale(1, f"Predicted (mu = {estimated_viscosity:.5f})", "v1"),
        "[v0][v1]hstack=inputs=2:shortest=1,format=yuv420p[v]",
    ]
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(target),
        "-i",
        str(estimated),
        "-filter_complex",
        ";".join(filters),
        "-map",
        "[v]",
        "-an",
        "-r",
        "60",
        "-vcodec",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    _run(command, dry_run=dry_run)


def _render_video(
    args: argparse.Namespace,
    script: str,
    output: Path,
    *,
    extra_args: list[str] | None = None,
) -> None:
    if output.exists() and args.skip_existing:
        print(f"skip existing {output}", flush=True)
        return
    command = [
        *_python_cmd(args),
        str(ROOT / "scripts" / script),
        "--liquid-vis-mode",
        "recon",
        "--output-path",
        str(output),
    ]
    if extra_args:
        command.extend(extra_args)
    _run(command, dry_run=args.dry_run)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--conda-env", default="genesis-sim")
    parser.add_argument(
        "--use-current-python",
        action="store_true",
        help="Run the Genesis scripts with the current interpreter instead of conda run.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--only-composites",
        action="store_true",
        help="Only rebuild composite videos from already-rendered continuous sources.",
    )
    args = parser.parse_args(argv)

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
    (ROOT / "outputs").mkdir(exist_ok=True)
    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)

    estimated_viscosity = _best_estimated_viscosity()
    estimated_output = CALIBRATION_DIR / _estimated_video_name(estimated_viscosity)
    videos = {
        "water": ROOT / "outputs" / "robotic_arm_pour_genesis_continuous.mp4",
        "honey": ROOT / "outputs" / "robotic_arm_pour_honey_genesis_continuous.mp4",
        "medium": ROOT / "outputs" / "robotic_arm_pour_medium_genesis_continuous.mp4",
        "target": ROOT / "outputs" / "robotic_arm_pour_target_genesis_continuous.mp4",
        "estimated": estimated_output,
    }

    if not args.only_composites:
        _render_video(args, "run_robotic_arm_pour_genesis.py", videos["water"])
        _render_video(args, "run_robotic_arm_pour_honey_genesis.py", videos["honey"])
        _render_video(args, "run_robotic_arm_pour_medium_genesis.py", videos["medium"])
        _render_video(args, "run_robotic_arm_pour_target_genesis.py", videos["target"])
        _render_video(
            args,
            "run_robotic_arm_pour_viscosity_genesis.py",
            videos["estimated"],
            extra_args=["--viscosity", f"{estimated_viscosity:.15g}"],
        )

    missing = [path for path in videos.values() if not path.exists()]
    if missing and not args.dry_run:
        raise FileNotFoundError("missing continuous source videos: " + ", ".join(str(p) for p in missing))

    _build_reference_grid(
        target=videos["target"],
        estimated=videos["estimated"],
        water=videos["water"],
        honey=videos["honey"],
        output=CALIBRATION_DIR / "reference_liquids_grid_continuous.mp4",
        estimated_viscosity=estimated_viscosity,
        dry_run=args.dry_run,
    )
    _build_target_vs_estimated(
        target=videos["target"],
        estimated=videos["estimated"],
        output=CALIBRATION_DIR / "target_vs_estimated_continuous.mp4",
        estimated_viscosity=estimated_viscosity,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
