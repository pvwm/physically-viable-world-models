"""Export per-frame meshes (OBJ) and water particles (PLY) for the Genesis cup-water sim.

Output layout (default ``outputs/blender/cup_water_genesis/``):
    ramp.obj, floor.obj          (static, world coords)
    ball_NNNN.obj                (rigid sphere, world coords per frame)
    cup_NNNN.obj                 (rigid hollow cup, world coords per frame)
    water_NNNN.ply               (SPH particle cloud, in-cavity particles only)

Loaded by path so it works from the ``genesis-sim`` conda env without going
through ``npm_sim/__init__`` (which imports Newton).
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def _load_by_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _genesis_quat_to_xyzw(q_wxyz: np.ndarray) -> np.ndarray:
    """Genesis stores quaternions as (w, x, y, z); ex.transform_verts_xyzw wants (x, y, z, w)."""
    q = np.asarray(q_wxyz).reshape(-1)
    return np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--num-frames", type=int, default=240)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT / "outputs" / "blender" / "cup_water_genesis"),
    )
    parser.add_argument(
        "--floor-length",
        type=float,
        default=4.00,
        help="Length (Y) of the exported floor box, matching the Newton variants.",
    )
    parser.add_argument(
        "--floor-width",
        type=float,
        default=2.00,
        help="Width (X) of the exported floor box, matching the Newton variants.",
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ex = _load_by_path("npm_export", SRC / "npm_sim" / "export.py")
    mod = _load_by_path("rigid_ramp_cup_water_genesis", SRC / "npm_sim" / "rigid_ramp_cup_water_genesis.py")

    angle = math.radians(mod.RAMP_ANGLE_DEG)
    ramp_half_length = mod.RAMP_LENGTH * 0.5
    ramp_half_thickness = mod.RAMP_THICKNESS * 0.5
    ramp_pos = np.array(
        [
            0.0,
            -ramp_half_length * math.cos(angle) - ramp_half_thickness * math.sin(angle),
            ramp_half_length * math.sin(angle) - ramp_half_thickness * math.cos(angle),
        ]
    )
    ramp_quat = ex.axis_angle_quat_xyzw((1.0, 0.0, 0.0), -mod.RAMP_ANGLE_DEG)
    ramp_v, ramp_f = ex.make_box_mesh(mod.RAMP_WIDTH * 0.5, ramp_half_length, ramp_half_thickness)
    ex.write_obj(out_dir / "ramp.obj", ex.transform_verts_xyzw(ramp_v, ramp_pos, ramp_quat), ramp_f)

    floor_thickness = 0.08
    floor_ramp_overlap = 0.03
    fl_half = args.floor_length * 0.5
    fw_half = args.floor_width * 0.5
    floor_pos = np.array([0.0, fl_half - floor_ramp_overlap, -floor_thickness * 0.5])
    floor_v, floor_f = ex.make_box_mesh(fw_half, fl_half, floor_thickness * 0.5)
    ex.write_obj(
        out_dir / "floor.obj",
        ex.transform_verts_xyzw(floor_v, floor_pos, np.array([0.0, 0.0, 0.0, 1.0])),
        floor_f,
    )

    if not mod.SETTLED_PARTICLES_CACHE.exists():
        mod.bake_settled_particles()

    demo = mod.RampCupWaterGenesisDemo(
        num_frames=args.num_frames, show_viewer=False, enable_camera=False
    )
    if not demo.load_settled_particles(mod.SETTLED_PARTICLES_CACHE):
        mod.bake_settled_particles()
        demo.load_settled_particles(mod.SETTLED_PARTICLES_CACHE)

    ball_v, ball_f = ex.make_sphere_mesh(mod.BALL_RADIUS, subdivisions=2)
    cup_v, cup_f = ex.make_hollow_cup_mesh(
        height=mod.CUP_HEIGHT,
        bottom_radius=mod.CUP_BOTTOM_RADIUS,
        top_radius=mod.CUP_TOP_RADIUS,
        wall_thickness=mod.CUP_WALL_THICKNESS,
        base_thickness=mod.CUP_BASE_THICKNESS,
    )

    for frame_idx in range(args.num_frames):
        if frame_idx > 0:
            demo.step()

        ball_pos = demo.ball.get_pos().cpu().numpy().reshape(-1)
        ball_q = _genesis_quat_to_xyzw(demo.ball.get_quat().cpu().numpy())
        ex.write_obj(
            out_dir / f"ball_{frame_idx:04d}.obj",
            ex.transform_verts_xyzw(ball_v, ball_pos, ball_q),
            ball_f,
        )

        cup_pos = demo.cup.get_pos().cpu().numpy().reshape(-1)
        cup_q = _genesis_quat_to_xyzw(demo.cup.get_quat().cpu().numpy())
        ex.write_obj(
            out_dir / f"cup_{frame_idx:04d}.obj",
            ex.transform_verts_xyzw(cup_v, cup_pos, cup_q),
            cup_f,
        )

        particles = demo._particle_positions()
        # Filter out the (10, 10, -10) stash where overflow particles live.
        live = particles[particles[:, 2] > -1.0]
        ex.write_ply_points(out_dir / f"water_{frame_idx:04d}.ply", live)

    print(f"wrote {args.num_frames} frames to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
