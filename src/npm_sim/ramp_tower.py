from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import newton
import newton.viewer
import numpy as np
import warp as wp

from .materials import MATERIALS, MaterialPreset

# Fixed v1 scene constants, tuned for one readable passive rigid-body demo.
# Axis convention matches current Newton contact examples:
#   z = up
#   y = forward / downhill motion
#   x = sideways
FRAME_RATE = 120
SIM_SUBSTEPS = 10
SOLVER_ITERATIONS = 8
SOLVER_CONTACT_RELAXATION = 0.8
RIGID_GAP = 0.005

RAMP_ANGLE_DEG = 20.0
RAMP_LENGTH = 1.25
RAMP_WIDTH = 0.45
RAMP_THICKNESS = 0.08

FLOOR_LENGTH = 2.00
FLOOR_WIDTH = 0.80
FLOOR_THICKNESS = 0.08
FLOOR_RAMP_OVERLAP = 0.03

BALL_RADIUS = 0.09
BALL_START_MARGIN = 0.18
BALL_LATERAL_OFFSET = 0.02
BALL_CLEARANCE = 0.003

CUBE_SIZE = 0.14
TOP_CUBE_YAW_DEG = 5.0
TOWER_CENTER_Y = 0.95

CAMERA_POS = wp.vec3(2.10, -1.40, 0.92)
CAMERA_PITCH = -13.0
CAMERA_YAW = 142.0
CAMERA_FOV = 55.0

VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
VIDEO_NUM_FRAMES = FRAME_RATE * 3
VIDEO_CROP_X = 150
VIDEO_CROP_Y = 230
VIDEO_CROP_WIDTH = 940
VIDEO_CROP_HEIGHT = 529

RAMP_COLOR = wp.vec3(0.32, 0.35, 0.39)
FLOOR_COLOR = wp.vec3(0.55, 0.58, 0.62)
BALL_COLOR = wp.vec3(0.68, 0.74, 0.80)
CUBE_COLOR = wp.vec3(0.61, 0.42, 0.23)


@dataclass(frozen=True)
class SimulationResult:
    initial_ball_position: np.ndarray
    final_ball_position: np.ndarray
    cube_body_indices: tuple[int, ...]
    initial_cube_transforms: np.ndarray
    final_cube_transforms: np.ndarray
    initial_body_poses: np.ndarray
    final_body_poses: np.ndarray


def _shape_cfg(material: MaterialPreset) -> newton.ModelBuilder.ShapeConfig:
    return newton.ModelBuilder.ShapeConfig(
        density=material.density,
        mu=material.friction,
        restitution=material.restitution,
    )


def _static_shape_cfg(material: MaterialPreset) -> newton.ModelBuilder.ShapeConfig:
    return newton.ModelBuilder.ShapeConfig(
        density=0.0,
        mu=material.friction,
        restitution=material.restitution,
    )


def _quat_x(degrees: float) -> wp.quat:
    return wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.radians(degrees))


def _quat_z(degrees: float) -> wp.quat:
    return wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), math.radians(degrees))


def _build_viewer(viewer: str, num_frames: int, output_path: str | None) -> Any:
    if viewer == "gl":
        headless = not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        if headless:
            os.environ.setdefault("PYGLET_HEADLESS", "1")
        return newton.viewer.ViewerGL(headless=headless)
    if viewer == "usd":
        if output_path is None:
            raise ValueError("--output-path is required when using usd viewer")
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        return newton.viewer.ViewerUSD(
            output_path=str(output),
            fps=FRAME_RATE,
            up_axis="Z",
            num_frames=num_frames,
        )
    if viewer == "null":
        return newton.viewer.ViewerNull(num_frames=num_frames)
    raise ValueError(f"Unsupported viewer: {viewer}")


class RampTowerDemo:
    def __init__(self, viewer: Any, ball_material: str = "steel", cube_material: str = "wood"):
        self.viewer = viewer
        self.fps = FRAME_RATE
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = SIM_SUBSTEPS
        self.sim_dt = self.frame_dt / self.sim_substeps

        ball_preset = MATERIALS[ball_material]
        cube_preset = MATERIALS[cube_material]
        ball_cfg = _shape_cfg(ball_preset)
        cube_cfg = _shape_cfg(cube_preset)
        ground_cfg = _static_shape_cfg(cube_preset)

        builder = newton.ModelBuilder()
        builder.rigid_gap = RIGID_GAP

        ramp_angle = math.radians(RAMP_ANGLE_DEG)
        ramp_half_length = RAMP_LENGTH * 0.5
        ramp_half_thickness = RAMP_THICKNESS * 0.5
        floor_half_length = FLOOR_LENGTH * 0.5
        floor_half_thickness = FLOOR_THICKNESS * 0.5
        cube_half = CUBE_SIZE * 0.5

        floor_top_z = 0.0
        ramp_center = wp.vec3(
            0.0,
            -ramp_half_length * math.cos(ramp_angle) - ramp_half_thickness * math.sin(ramp_angle),
            ramp_half_length * math.sin(ramp_angle) - ramp_half_thickness * math.cos(ramp_angle),
        )
        ramp_xform = wp.transform(p=ramp_center, q=_quat_x(-RAMP_ANGLE_DEG))
        floor_xform = wp.transform(
            p=wp.vec3(0.0, floor_half_length - FLOOR_RAMP_OVERLAP, floor_top_z - floor_half_thickness),
            q=wp.quat_identity(),
        )

        builder.add_shape_box(
            body=-1,
            xform=ramp_xform,
            hx=RAMP_WIDTH * 0.5,
            hy=ramp_half_length,
            hz=ramp_half_thickness,
            cfg=ground_cfg,
            color=RAMP_COLOR,
            label="ramp",
        )
        builder.add_shape_box(
            body=-1,
            xform=floor_xform,
            hx=FLOOR_WIDTH * 0.5,
            hy=floor_half_length,
            hz=floor_half_thickness,
            cfg=ground_cfg,
            color=FLOOR_COLOR,
            label="floor",
        )

        ball_local = wp.vec3(
            BALL_LATERAL_OFFSET,
            -ramp_half_length + BALL_START_MARGIN,
            ramp_half_thickness + BALL_RADIUS + BALL_CLEARANCE,
        )
        ball_start = wp.transform_point(ramp_xform, ball_local)
        self.ball_body_index = builder.add_body(
            xform=wp.transform(p=ball_start, q=wp.quat_identity()),
            label="ball",
        )
        builder.add_shape_sphere(
            self.ball_body_index,
            radius=BALL_RADIUS,
            cfg=ball_cfg,
            color=BALL_COLOR,
            label="ball_shape",
        )

        tower_base_z = floor_top_z + cube_half
        self.cube_body_indices: list[int] = []
        cube_positions = (
            wp.vec3(-cube_half, TOWER_CENTER_Y, tower_base_z),
            wp.vec3(cube_half, TOWER_CENTER_Y, tower_base_z),
            wp.vec3(0.0, TOWER_CENTER_Y, tower_base_z + CUBE_SIZE),
        )
        cube_rotations = (
            wp.quat_identity(),
            wp.quat_identity(),
            _quat_z(TOP_CUBE_YAW_DEG),
        )
        cube_labels = ("cube_bottom_left", "cube_bottom_right", "cube_top")

        for position, rotation, label in zip(cube_positions, cube_rotations, cube_labels, strict=True):
            body = builder.add_body(xform=wp.transform(p=position, q=rotation), label=label)
            builder.add_shape_box(
                body,
                hx=cube_half,
                hy=cube_half,
                hz=cube_half,
                cfg=cube_cfg,
                color=CUBE_COLOR,
                label=f"{label}_shape",
            )
            self.cube_body_indices.append(body)

        self.model = builder.finalize()
        self.collision_pipeline = newton.CollisionPipeline(self.model)
        self.solver = newton.solvers.SolverXPBD(
            self.model,
            iterations=SOLVER_ITERATIONS,
            rigid_contact_relaxation=SOLVER_CONTACT_RELAXATION,
            enable_restitution=True,
        )
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.contacts = self.collision_pipeline.contacts()

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=CAMERA_POS, pitch=CAMERA_PITCH, yaw=CAMERA_YAW)
        if hasattr(self.viewer, "camera") and hasattr(self.viewer.camera, "fov"):
            self.viewer.camera.fov = CAMERA_FOV

        self.initial_body_poses = self.body_poses()
        self.capture()

    def body_poses(self) -> np.ndarray:
        return self.state_0.body_q.numpy().copy()

    def capture(self) -> None:
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self) -> None:
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self) -> None:
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def result(self) -> SimulationResult:
        final_body_poses = self.body_poses()
        cube_indices = tuple(self.cube_body_indices)
        return SimulationResult(
            initial_ball_position=self.initial_body_poses[self.ball_body_index, :3].copy(),
            final_ball_position=final_body_poses[self.ball_body_index, :3].copy(),
            cube_body_indices=cube_indices,
            initial_cube_transforms=self.initial_body_poses[list(cube_indices)].copy(),
            final_cube_transforms=final_body_poses[list(cube_indices)].copy(),
            initial_body_poses=self.initial_body_poses.copy(),
            final_body_poses=final_body_poses,
        )


def build_scene(
    *,
    ball_material: str = "steel",
    cube_material: str = "wood",
    viewer: str | Any = "null",
    num_frames: int = 240,
    output_path: str | None = None,
    device: str | None = None,
) -> RampTowerDemo:
    if device:
        wp.set_device(device)

    if isinstance(viewer, str):
        viewer_obj = _build_viewer(viewer, num_frames=num_frames, output_path=output_path)
    else:
        viewer_obj = viewer

    return RampTowerDemo(
        viewer=viewer_obj,
        ball_material=ball_material,
        cube_material=cube_material,
    )


def run_simulation(
    *,
    ball_material: str = "steel",
    cube_material: str = "wood",
    viewer: str = "null",
    num_frames: int = 240,
    output_path: str | None = None,
    device: str | None = None,
) -> SimulationResult:
    demo = build_scene(
        ball_material=ball_material,
        cube_material=cube_material,
        viewer=viewer,
        num_frames=num_frames,
        output_path=output_path,
        device=device,
    )

    try:
        while demo.viewer.is_running():
            if not demo.viewer.is_paused():
                demo.step()
            demo.render()
        return demo.result()
    finally:
        demo.viewer.close()


def render_video(
    *,
    output_path: str = "outputs/ramp_tower.mp4",
    ball_material: str = "steel",
    cube_material: str = "wood",
    num_frames: int = VIDEO_NUM_FRAMES,
    device: str | None = None,
) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to render video output")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("PYGLET_HEADLESS", "1")
    viewer = newton.viewer.ViewerGL(width=VIDEO_WIDTH, height=VIDEO_HEIGHT, headless=True)
    demo = build_scene(
        ball_material=ball_material,
        cube_material=cube_material,
        viewer=viewer,
        num_frames=num_frames,
        output_path=None,
        device=device,
    )

    command = [
        ffmpeg,
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{VIDEO_WIDTH}x{VIDEO_HEIGHT}",
        "-r",
        str(FRAME_RATE),
        "-i",
        "-",
        "-an",
        "-vf",
        (
            f"crop={VIDEO_CROP_WIDTH}:{VIDEO_CROP_HEIGHT}:{VIDEO_CROP_X}:{VIDEO_CROP_Y},"
            f"scale={VIDEO_WIDTH}:{VIDEO_HEIGHT}:flags=lanczos"
        ),
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
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    try:
        for frame_index in range(num_frames):
            if frame_index > 0:
                demo.step()
            demo.render()
            frame = viewer.get_frame()
            frame_np = np.ascontiguousarray(frame.numpy())
            assert process.stdin is not None
            process.stdin.write(frame_np.tobytes())

        assert process.stdin is not None
        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr is not None else ""
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg failed with exit code {return_code}: {stderr.strip()}")
        return output
    finally:
        if process.stdin is not None and not process.stdin.closed:
            process.stdin.close()
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)
        if process.stderr is not None:
            process.stderr.close()
        viewer.close()


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
        "--viewer",
        type=str,
        default="gl",
        choices=["gl", "usd", "null"],
        help="Viewer mode.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=240,
        help="Frame count for null and usd viewers.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="USD output path. Required when --viewer usd is selected.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional Warp device override, for example cpu or cuda:0.",
    )
    return parser


def main(argv: list[str] | None = None) -> SimulationResult:
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.num_frames <= 0:
        parser.error("--num-frames must be positive")
    if args.viewer == "usd" and args.output_path is None:
        parser.error("--output-path is required when using --viewer usd")

    return run_simulation(
        ball_material=args.ball_material,
        cube_material=args.cube_material,
        viewer=args.viewer,
        num_frames=args.num_frames,
        output_path=args.output_path,
        device=args.device,
    )


if __name__ == "__main__":
    main()
