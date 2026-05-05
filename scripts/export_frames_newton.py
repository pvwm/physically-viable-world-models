"""Export per-frame meshes (OBJ) and particle clouds (PLY) for Blender.

Output layout per variant (under ``outputs/blender/<variant>/``):
  - ramp.obj, floor.obj                  (static, world coords)
  - ball_NNNN.obj, cube_*_NNNN.obj, ...  (rigid bodies, world coords per frame)
  - ball_NNNN.ply, wall_*_NNNN.ply, ... (soft-body particle clouds, jelly only)

Usage:
    python scripts/export_frames_newton.py --variant rigid
    python scripts/export_frames_newton.py --variant rigid-steel-cubes
    python scripts/export_frames_newton.py --variant jelly-single
    python scripts/export_frames_newton.py --variant jelly-domino
    python scripts/export_frames_newton.py --variant jelly-domino-flip
    python scripts/export_frames_newton.py --variant superball
    python scripts/export_frames_newton.py --variant superball-steel-cubes
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import warp as wp

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import newton.viewer

from npm_sim import export as ex
from npm_sim import ramp_tower, rigid_ramp_tower


def _write_static_geometry(
    out_dir: Path,
    *,
    ramp_angle_deg: float,
    ramp_length: float,
    ramp_width: float,
    ramp_thickness: float,
    floor_length: float,
    floor_width: float,
    floor_thickness: float,
    floor_ramp_overlap: float,
) -> None:
    angle = math.radians(ramp_angle_deg)
    ramp_half_length = ramp_length * 0.5
    ramp_half_thickness = ramp_thickness * 0.5
    ramp_pos = np.array(
        [
            0.0,
            -ramp_half_length * math.cos(angle) - ramp_half_thickness * math.sin(angle),
            ramp_half_length * math.sin(angle) - ramp_half_thickness * math.cos(angle),
        ]
    )
    ramp_quat = ex.axis_angle_quat_xyzw((1.0, 0.0, 0.0), -ramp_angle_deg)
    ramp_v, ramp_f = ex.make_box_mesh(ramp_width * 0.5, ramp_half_length, ramp_half_thickness)
    ex.write_obj(out_dir / "ramp.obj", ex.transform_verts_xyzw(ramp_v, ramp_pos, ramp_quat), ramp_f)

    floor_half_length = floor_length * 0.5
    floor_half_thickness = floor_thickness * 0.5
    floor_pos = np.array([0.0, floor_half_length - floor_ramp_overlap, -floor_half_thickness])
    floor_v, floor_f = ex.make_box_mesh(floor_width * 0.5, floor_half_length, floor_half_thickness)
    ex.write_obj(
        out_dir / "floor.obj",
        ex.transform_verts_xyzw(floor_v, floor_pos, np.array([0.0, 0.0, 0.0, 1.0])),
        floor_f,
    )


def _static_args(module) -> dict:
    return dict(
        ramp_angle_deg=module.RAMP_ANGLE_DEG,
        ramp_length=module.RAMP_LENGTH,
        ramp_width=module.RAMP_WIDTH,
        ramp_thickness=module.RAMP_THICKNESS,
        floor_length=module.FLOOR_LENGTH,
        floor_width=module.FLOOR_WIDTH,
        floor_thickness=module.FLOOR_THICKNESS,
        floor_ramp_overlap=module.FLOOR_RAMP_OVERLAP,
    )


def export_rigid(
    num_frames: int,
    out_dir: Path,
    *,
    ball_material: str = "steel",
    cube_material: str = "wood",
    target_material: str | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_static_geometry(out_dir, **_static_args(rigid_ramp_tower))

    viewer = newton.viewer.ViewerNull(num_frames=num_frames)
    demo = rigid_ramp_tower.RampTowerDemo(
        viewer=viewer,
        ball_material=ball_material,
        cube_material=cube_material,
        target_material=target_material,
    )

    ball_v, ball_f = ex.make_sphere_mesh(rigid_ramp_tower.BALL_RADIUS, subdivisions=2)
    half = rigid_ramp_tower.CUBE_SIZE * 0.5
    cube_v, cube_f = ex.make_box_mesh(half, half, half)
    sphere_target_v, sphere_target_f = ex.make_sphere_mesh(half, subdivisions=2)
    superball_layout = ball_material == "superball" and cube_material == "superball"
    target_name = "target" if superball_layout else "cube"

    try:
        for frame_idx in range(num_frames):
            if frame_idx > 0:
                demo.step()
            body_q = demo.state_0.body_q.numpy()
            ball_pose = body_q[demo.ball_body_index]
            ex.write_obj(
                out_dir / f"ball_{frame_idx:04d}.obj",
                ex.transform_verts_xyzw(ball_v, ball_pose[:3], ball_pose[3:]),
                ball_f,
            )
            for i, body_idx in enumerate(demo.cube_body_indices):
                pose = body_q[body_idx]
                if superball_layout and i == 2:
                    ex.write_obj(
                        out_dir / f"{target_name}_{i}_{frame_idx:04d}.obj",
                        ex.transform_verts_xyzw(sphere_target_v, pose[:3], pose[3:]),
                        sphere_target_f,
                    )
                else:
                    ex.write_obj(
                        out_dir / f"{target_name}_{i}_{frame_idx:04d}.obj",
                        ex.transform_verts_xyzw(cube_v, pose[:3], pose[3:]),
                        cube_f,
                    )
    finally:
        viewer.close()


def export_jelly(
    wall_count: int,
    num_frames: int,
    out_dir: Path,
    *,
    ramp_length: float = ramp_tower.RAMP_LENGTH,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    static_args = _static_args(ramp_tower)
    static_args["ramp_length"] = ramp_length
    _write_static_geometry(out_dir, **static_args)

    viewer = newton.viewer.ViewerNull(num_frames=num_frames)
    demo = ramp_tower.RampTowerDemo(viewer=viewer, wall_count=wall_count, ramp_length=ramp_length)

    try:
        for frame_idx in range(num_frames):
            if frame_idx > 0:
                demo.step()
            positions = demo.particle_positions()
            b0, b1 = demo.ball_particle_range
            ex.write_ply_points(out_dir / f"ball_{frame_idx:04d}.ply", positions[b0:b1])
            for i, (start, end) in enumerate(demo.target_particle_ranges):
                ex.write_ply_points(out_dir / f"wall_{i}_{frame_idx:04d}.ply", positions[start:end])
    finally:
        viewer.close()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--variant",
        required=True,
        choices=[
            "rigid",
            "rigid-steel-cubes",
            "superball",
            "superball-steel-cubes",
            "jelly-single",
            "jelly-domino",
            "jelly-domino-flip",
        ],
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=None,
        help="Defaults to the rendered video's frame count for this variant.",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional Warp device override, for example cpu or cuda:0.",
    )
    args = parser.parse_args(argv)

    if args.device:
        wp.set_device(args.device)

    default_frames = {
        "rigid": rigid_ramp_tower.VIDEO_NUM_FRAMES,
        "rigid-steel-cubes": rigid_ramp_tower.VIDEO_NUM_FRAMES,
        "superball": rigid_ramp_tower.VIDEO_NUM_FRAMES,
        "superball-steel-cubes": rigid_ramp_tower.FRAME_RATE * 4,
        "jelly-single": ramp_tower.VIDEO_NUM_FRAMES,
        "jelly-domino": ramp_tower.VIDEO_NUM_FRAMES,
        "jelly-domino-flip": ramp_tower.DOMINO_FLIP_VIDEO_NUM_FRAMES,
    }
    num_frames = args.num_frames if args.num_frames is not None else default_frames[args.variant]

    default_dirs = {
        "rigid": ROOT / "outputs" / "blender" / "rigid",
        "rigid-steel-cubes": ROOT / "outputs" / "blender" / "rigid_steel_cubes",
        "superball": ROOT / "outputs" / "blender" / "superball",
        "superball-steel-cubes": ROOT / "outputs" / "blender" / "superball_steel_cubes",
        "jelly-single": ROOT / "outputs" / "blender" / "jelly_single",
        "jelly-domino": ROOT / "outputs" / "blender" / "jelly_domino",
        "jelly-domino-flip": ROOT / "outputs" / "blender" / "jelly_domino_flip",
    }
    out_dir = Path(args.output_dir) if args.output_dir else default_dirs[args.variant]

    if args.variant == "rigid":
        export_rigid(num_frames, out_dir)
    elif args.variant == "rigid-steel-cubes":
        export_rigid(num_frames, out_dir, cube_material="steel")
    elif args.variant == "superball":
        export_rigid(num_frames, out_dir, ball_material="superball", cube_material="superball")
    elif args.variant == "superball-steel-cubes":
        export_rigid(
            num_frames,
            out_dir,
            ball_material="superball",
            cube_material="superball",
            target_material="steel",
        )
    elif args.variant == "jelly-single":
        export_jelly(1, num_frames, out_dir)
    elif args.variant == "jelly-domino":
        export_jelly(2, num_frames, out_dir)
    elif args.variant == "jelly-domino-flip":
        export_jelly(2, num_frames, out_dir, ramp_length=ramp_tower.DOMINO_FLIP_RAMP_LENGTH)

    print(f"wrote {num_frames} frames to {out_dir}")


if __name__ == "__main__":
    main()
