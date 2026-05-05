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

# Fixed scene constants for one readable deformable demo.
# Axis convention matches current Newton contact examples:
#   z = up
#   y = forward / downhill motion
#   x = sideways
FRAME_RATE = 120
SIM_SUBSTEPS = 10
SOLVER_ITERATIONS = 14
RIGID_GAP = 0.005

RAMP_ANGLE_DEG = 20.0
RAMP_LENGTH = 1.25
RAMP_WIDTH = 0.45
RAMP_THICKNESS = 0.08
DOMINO_FLIP_RAMP_LENGTH = 4.25

FLOOR_LENGTH = 4.00
FLOOR_WIDTH = 2.00
FLOOR_THICKNESS = 0.08
FLOOR_RAMP_OVERLAP = 0.03

BALL_RADIUS = 0.09
BALL_START_MARGIN = 0.18
BALL_LATERAL_OFFSET = 0.02
BALL_CLEARANCE = 0.006

# One or two connected jelly walls read clearly in motion and stay quieter at rest than stacked soft cubes.
WALL_SIZE_X = 0.28
WALL_SIZE_Y = 0.14
WALL_SIZE_Z = 0.32
WALL_CENTER_Y = 1.02
WALL_DOMINO_SPACING_Y = 0.22
SINGLE_WALL_COUNT = 1
DOMINO_WALL_COUNT = 2
DEFAULT_WALL_COUNT = DOMINO_WALL_COUNT
WALL_CELLS_X = 4
WALL_CELLS_Y = 2
WALL_CELLS_Z = 5
# Start each wall just above its particle-floor contact envelope so it settles instead of popping.
WALL_BASE_CLEARANCE = 0.028

# Soft-body tuning.
BALL_JELLY_K_MU = 6.0e4
BALL_JELLY_K_LAMBDA = 1.5e5
BALL_JELLY_K_DAMP = 2.0e-2
CUBE_JELLY_K_MU = 3.0e4
CUBE_JELLY_K_LAMBDA = 7.5e4
CUBE_JELLY_K_DAMP = 3.0e-2
PARTICLE_CONTACT_RADIUS = 0.026
PARTICLE_SELF_CONTACT_RADIUS = 0.028
PARTICLE_SELF_CONTACT_MARGIN = 0.032
SOFT_CONTACT_KE = 1.5e5
SOFT_CONTACT_KD = 1.0e-4
SOFT_CONTACT_FRICTION_FLOOR = 0.8
SOFT_CONTACT_RESTITUTION_CAP = 0.02

CAMERA_POS = wp.vec3(2.10, -1.40, 0.92)
CAMERA_PITCH = -13.0
CAMERA_YAW = 142.0
CAMERA_FOV = 55.0
DOMINO_FLIP_CAMERA_POS = wp.vec3(3.20, -4.40, 2.10)
DOMINO_FLIP_CAMERA_PITCH = -20.0
DOMINO_FLIP_CAMERA_YAW = 126.0
DOMINO_FLIP_CAMERA_FOV = 65.0

VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
VIDEO_NUM_FRAMES = FRAME_RATE * 3
DOMINO_FLIP_VIDEO_NUM_FRAMES = FRAME_RATE * 4
VIDEO_CROP_X = 150
VIDEO_CROP_Y = 230
VIDEO_CROP_WIDTH = 940
VIDEO_CROP_HEIGHT = 529
DOMINO_FLIP_VIDEO_CROP_X = 0
DOMINO_FLIP_VIDEO_CROP_Y = 0
DOMINO_FLIP_VIDEO_CROP_WIDTH = VIDEO_WIDTH
DOMINO_FLIP_VIDEO_CROP_HEIGHT = VIDEO_HEIGHT

RAMP_COLOR = wp.vec3(0.32, 0.35, 0.39)
FLOOR_COLOR = wp.vec3(0.55, 0.58, 0.62)

GOLDEN_RATIO = (1.0 + math.sqrt(5.0)) * 0.5
BALL_SPHERE_SUBDIVISIONS = 1
BALL_SURFACE_VERTICES = np.array(
    [
        (-1.0, GOLDEN_RATIO, 0.0),
        (1.0, GOLDEN_RATIO, 0.0),
        (-1.0, -GOLDEN_RATIO, 0.0),
        (1.0, -GOLDEN_RATIO, 0.0),
        (0.0, -1.0, GOLDEN_RATIO),
        (0.0, 1.0, GOLDEN_RATIO),
        (0.0, -1.0, -GOLDEN_RATIO),
        (0.0, 1.0, -GOLDEN_RATIO),
        (GOLDEN_RATIO, 0.0, -1.0),
        (GOLDEN_RATIO, 0.0, 1.0),
        (-GOLDEN_RATIO, 0.0, -1.0),
        (-GOLDEN_RATIO, 0.0, 1.0),
    ],
    dtype=np.float32,
)
BALL_SURFACE_VERTICES /= np.linalg.norm(BALL_SURFACE_VERTICES[0])
BALL_SURFACE_FACES = np.array(
    [
        (0, 11, 5),
        (0, 5, 1),
        (0, 1, 7),
        (0, 7, 10),
        (0, 10, 11),
        (1, 5, 9),
        (5, 11, 4),
        (11, 10, 2),
        (10, 7, 6),
        (7, 1, 8),
        (3, 9, 4),
        (3, 4, 2),
        (3, 2, 6),
        (3, 6, 8),
        (3, 8, 9),
        (4, 9, 5),
        (2, 4, 11),
        (6, 2, 10),
        (8, 6, 7),
        (9, 8, 1),
    ],
    dtype=np.int32,
)


@dataclass(frozen=True)
class SimulationResult:
    ball_particle_range: tuple[int, int]
    target_particle_ranges: tuple[tuple[int, int], ...]
    target_particle_range: tuple[int, int]
    initial_ball_position: np.ndarray
    final_ball_position: np.ndarray
    initial_target_centers: np.ndarray
    final_target_centers: np.ndarray
    initial_target_center: np.ndarray
    final_target_center: np.ndarray
    initial_particle_positions: np.ndarray
    final_particle_positions: np.ndarray


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


def _make_ball_tet_mesh(radius: float) -> tuple[np.ndarray, np.ndarray]:
    surface_vertices, surface_faces = _subdivide_unit_icosphere(BALL_SPHERE_SUBDIVISIONS)
    vertices = np.vstack([np.zeros((1, 3), dtype=np.float32), surface_vertices * radius])
    tets: list[int] = []

    for face in surface_faces:
        tet = [0, int(face[0]) + 1, int(face[1]) + 1, int(face[2]) + 1]
        a, b, c, d = vertices[tet]
        signed_volume6 = float(np.dot(b - a, np.cross(c - a, d - a)))
        if signed_volume6 < 0.0:
            tet[2], tet[3] = tet[3], tet[2]
        tets.extend(tet)

    return vertices, np.array(tets, dtype=np.int32)


def _subdivide_unit_icosphere(subdivisions: int) -> tuple[np.ndarray, np.ndarray]:
    vertices = BALL_SURFACE_VERTICES.copy()
    faces = BALL_SURFACE_FACES.copy()

    for _ in range(subdivisions):
        vertex_list = vertices.tolist()
        midpoint_cache: dict[tuple[int, int], int] = {}
        refined_faces: list[tuple[int, int, int]] = []

        def midpoint_index(i0: int, i1: int) -> int:
            key = (i0, i1) if i0 < i1 else (i1, i0)
            cached = midpoint_cache.get(key)
            if cached is not None:
                return cached

            midpoint = 0.5 * (vertices[i0] + vertices[i1])
            midpoint /= np.linalg.norm(midpoint)
            vertex_list.append(midpoint.astype(np.float32).tolist())
            index = len(vertex_list) - 1
            midpoint_cache[key] = index
            return index

        for i0, i1, i2 in faces:
            a = midpoint_index(int(i0), int(i1))
            b = midpoint_index(int(i1), int(i2))
            c = midpoint_index(int(i2), int(i0))
            refined_faces.extend(
                [
                    (int(i0), a, c),
                    (int(i1), b, a),
                    (int(i2), c, b),
                    (a, b, c),
                ]
            )

        vertices = np.array(vertex_list, dtype=np.float32)
        faces = np.array(refined_faces, dtype=np.int32)

    return vertices, faces


def _particle_centroid(positions: np.ndarray, particle_range: tuple[int, int]) -> np.ndarray:
    start, end = particle_range
    return positions[start:end].mean(axis=0)


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
    def __init__(
        self,
        viewer: Any,
        ball_material: str = "steel",
        cube_material: str = "wood",
        wall_count: int = DEFAULT_WALL_COUNT,
        ramp_length: float = RAMP_LENGTH,
        ramp_angle_deg: float = RAMP_ANGLE_DEG,
        ramp_height_offset: float = 0.0,
        camera_pos: wp.vec3 | None = None,
        camera_pitch: float = CAMERA_PITCH,
        camera_yaw: float = CAMERA_YAW,
        camera_fov: float = CAMERA_FOV,
    ):
        if wall_count not in (SINGLE_WALL_COUNT, DOMINO_WALL_COUNT):
            raise ValueError("wall_count must be 1 or 2")

        self.viewer = viewer
        self.fps = FRAME_RATE
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = SIM_SUBSTEPS
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.ramp_angle_deg = ramp_angle_deg
        self.ramp_height_offset = ramp_height_offset

        ball_preset = MATERIALS[ball_material]
        cube_preset = MATERIALS[cube_material]

        builder = newton.ModelBuilder()
        builder.rigid_gap = RIGID_GAP

        ramp_angle = math.radians(ramp_angle_deg)
        ramp_half_length = ramp_length * 0.5
        ramp_half_thickness = RAMP_THICKNESS * 0.5
        floor_half_length = FLOOR_LENGTH * 0.5
        floor_half_thickness = FLOOR_THICKNESS * 0.5
        floor_top_z = 0.0
        ramp_center = wp.vec3(
            0.0,
            -ramp_half_length * math.cos(ramp_angle) - ramp_half_thickness * math.sin(ramp_angle),
            ramp_half_length * math.sin(ramp_angle) - ramp_half_thickness * math.cos(ramp_angle) + ramp_height_offset,
        )
        ramp_xform = wp.transform(p=ramp_center, q=_quat_x(-ramp_angle_deg))
        floor_xform = wp.transform(
            p=wp.vec3(0.0, floor_half_length - FLOOR_RAMP_OVERLAP, floor_top_z - floor_half_thickness),
            q=wp.quat_identity(),
        )

        ground_cfg = _static_shape_cfg(cube_preset)
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
        ball_center = wp.transform_point(ramp_xform, ball_local)
        ball_vertices, ball_tets = _make_ball_tet_mesh(BALL_RADIUS)
        ball_start = builder.particle_count
        builder.add_soft_mesh(
            pos=ball_center,
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=ball_vertices,
            indices=ball_tets,
            density=ball_preset.density,
            k_mu=BALL_JELLY_K_MU,
            k_lambda=BALL_JELLY_K_LAMBDA,
            k_damp=BALL_JELLY_K_DAMP,
            tri_ke=0.0,
            tri_ka=0.0,
            tri_kd=0.0,
            edge_ke=0.0,
            edge_kd=0.0,
            particle_radius=PARTICLE_CONTACT_RADIUS,
        )
        self.ball_particle_range = (ball_start, builder.particle_count)

        self.target_particle_ranges: list[tuple[int, int]] = []
        for wall_index in range(wall_count):
            wall_start = builder.particle_count
            wall_center_y = WALL_CENTER_Y + wall_index * WALL_DOMINO_SPACING_Y
            builder.add_soft_grid(
                pos=wp.vec3(-WALL_SIZE_X * 0.5, wall_center_y - WALL_SIZE_Y * 0.5, floor_top_z + WALL_BASE_CLEARANCE),
                rot=wp.quat_identity(),
                vel=wp.vec3(0.0, 0.0, 0.0),
                dim_x=WALL_CELLS_X,
                dim_y=WALL_CELLS_Y,
                dim_z=WALL_CELLS_Z,
                cell_x=WALL_SIZE_X / WALL_CELLS_X,
                cell_y=WALL_SIZE_Y / WALL_CELLS_Y,
                cell_z=WALL_SIZE_Z / WALL_CELLS_Z,
                density=cube_preset.density,
                k_mu=CUBE_JELLY_K_MU,
                k_lambda=CUBE_JELLY_K_LAMBDA,
                k_damp=CUBE_JELLY_K_DAMP,
                tri_ke=0.0,
                tri_ka=0.0,
                tri_kd=0.0,
                edge_ke=0.0,
                edge_kd=0.0,
                particle_radius=PARTICLE_CONTACT_RADIUS,
            )
            self.target_particle_ranges.append((wall_start, builder.particle_count))
        self.target_particle_range = self.target_particle_ranges[0]

        builder.color()

        self.model = builder.finalize()
        self.model.soft_contact_ke = SOFT_CONTACT_KE
        self.model.soft_contact_kd = SOFT_CONTACT_KD
        self.model.soft_contact_mu = max(
            SOFT_CONTACT_FRICTION_FLOOR,
            0.5 * (ball_preset.friction + cube_preset.friction),
        )
        self.model.soft_contact_restitution = min(
            SOFT_CONTACT_RESTITUTION_CAP,
            ball_preset.restitution,
            cube_preset.restitution,
        )

        self.solver = newton.solvers.SolverVBD(
            model=self.model,
            iterations=SOLVER_ITERATIONS,
            particle_enable_self_contact=True,
            particle_self_contact_radius=PARTICLE_SELF_CONTACT_RADIUS,
            particle_self_contact_margin=PARTICLE_SELF_CONTACT_MARGIN,
            particle_topological_contact_filter_threshold=1,
            particle_enable_tile_solve=True,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        self.viewer.set_model(self.model)
        if camera_pos is None:
            camera_pos = CAMERA_POS
        self.viewer.set_camera(pos=camera_pos, pitch=camera_pitch, yaw=camera_yaw)
        if hasattr(self.viewer, "camera") and hasattr(self.viewer.camera, "fov"):
            self.viewer.camera.fov = camera_fov

        self.initial_particle_positions = self.particle_positions()
        self.capture()

    def particle_positions(self) -> np.ndarray:
        return self.state_0.particle_q.numpy().copy()

    def target_centers(self, positions: np.ndarray) -> np.ndarray:
        return np.stack([_particle_centroid(positions, particle_range) for particle_range in self.target_particle_ranges])

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
            self.viewer.apply_forces(self.state_0)
            self.model.collide(self.state_0, self.contacts)
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
        final_positions = self.particle_positions()
        initial_target_centers = self.target_centers(self.initial_particle_positions)
        final_target_centers = self.target_centers(final_positions)
        return SimulationResult(
            ball_particle_range=self.ball_particle_range,
            target_particle_ranges=tuple(self.target_particle_ranges),
            target_particle_range=self.target_particle_range,
            initial_ball_position=_particle_centroid(self.initial_particle_positions, self.ball_particle_range),
            final_ball_position=_particle_centroid(final_positions, self.ball_particle_range),
            initial_target_centers=initial_target_centers,
            final_target_centers=final_target_centers,
            initial_target_center=initial_target_centers[0].copy(),
            final_target_center=final_target_centers[0].copy(),
            initial_particle_positions=self.initial_particle_positions.copy(),
            final_particle_positions=final_positions,
        )


def build_scene(
    *,
    ball_material: str = "steel",
    cube_material: str = "wood",
    wall_count: int = DEFAULT_WALL_COUNT,
    viewer: str | Any = "null",
    num_frames: int = 240,
    output_path: str | None = None,
    device: str | None = None,
    ramp_length: float = RAMP_LENGTH,
    ramp_angle_deg: float = RAMP_ANGLE_DEG,
    ramp_height_offset: float = 0.0,
    camera_pos: wp.vec3 | None = None,
    camera_pitch: float = CAMERA_PITCH,
    camera_yaw: float = CAMERA_YAW,
    camera_fov: float = CAMERA_FOV,
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
        wall_count=wall_count,
        ramp_length=ramp_length,
        ramp_angle_deg=ramp_angle_deg,
        ramp_height_offset=ramp_height_offset,
        camera_pos=camera_pos,
        camera_pitch=camera_pitch,
        camera_yaw=camera_yaw,
        camera_fov=camera_fov,
    )


def run_simulation(
    *,
    ball_material: str = "steel",
    cube_material: str = "wood",
    wall_count: int = DEFAULT_WALL_COUNT,
    viewer: str = "null",
    num_frames: int = 240,
    output_path: str | None = None,
    device: str | None = None,
    ramp_length: float = RAMP_LENGTH,
    ramp_angle_deg: float = RAMP_ANGLE_DEG,
    ramp_height_offset: float = 0.0,
    camera_pos: wp.vec3 | None = None,
    camera_pitch: float = CAMERA_PITCH,
    camera_yaw: float = CAMERA_YAW,
    camera_fov: float = CAMERA_FOV,
) -> SimulationResult:
    demo = build_scene(
        ball_material=ball_material,
        cube_material=cube_material,
        wall_count=wall_count,
        viewer=viewer,
        num_frames=num_frames,
        output_path=output_path,
        device=device,
        ramp_length=ramp_length,
        ramp_angle_deg=ramp_angle_deg,
        ramp_height_offset=ramp_height_offset,
        camera_pos=camera_pos,
        camera_pitch=camera_pitch,
        camera_yaw=camera_yaw,
        camera_fov=camera_fov,
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
    wall_count: int = DEFAULT_WALL_COUNT,
    num_frames: int = VIDEO_NUM_FRAMES,
    device: str | None = None,
    ramp_length: float = RAMP_LENGTH,
    ramp_angle_deg: float = RAMP_ANGLE_DEG,
    ramp_height_offset: float = 0.0,
    camera_pos: wp.vec3 | None = None,
    camera_pitch: float = CAMERA_PITCH,
    camera_yaw: float = CAMERA_YAW,
    camera_fov: float = CAMERA_FOV,
    video_crop_x: int = VIDEO_CROP_X,
    video_crop_y: int = VIDEO_CROP_Y,
    video_crop_width: int = VIDEO_CROP_WIDTH,
    video_crop_height: int = VIDEO_CROP_HEIGHT,
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
        wall_count=wall_count,
        viewer=viewer,
        num_frames=num_frames,
        output_path=None,
        device=device,
        ramp_length=ramp_length,
        ramp_angle_deg=ramp_angle_deg,
        ramp_height_offset=ramp_height_offset,
        camera_pos=camera_pos,
        camera_pitch=camera_pitch,
        camera_yaw=camera_yaw,
        camera_fov=camera_fov,
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
            f"crop={video_crop_width}:{video_crop_height}:{video_crop_x}:{video_crop_y},"
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
        help="Material preset for the deformable ball.",
    )
    parser.add_argument(
        "--cube-material",
        type=str,
        default="wood",
        choices=sorted(MATERIALS),
        help="Material preset for the jelly walls and static ground surfaces.",
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
