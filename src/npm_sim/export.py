"""Per-frame mesh / point-cloud writers for Blender re-rendering.

Pure numpy — no Newton/Genesis imports — so this module loads from either
the Newton conda env or the genesis-sim env (the genesis script imports it
by path to skip ``npm_sim.__init__``).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple

import numpy as np


def write_ply_points(path: Path, points: np.ndarray) -> None:
    """Binary little-endian PLY point cloud (Nx3, float32)."""
    points = np.ascontiguousarray(points, dtype=np.float32)
    n = points.shape[0]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(points.tobytes())


def write_obj(path: Path, verts: np.ndarray, faces: np.ndarray) -> None:
    """Triangle OBJ. ``faces`` are 0-indexed; written 1-indexed per OBJ spec."""
    verts = np.asarray(verts, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    lines = [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in verts]
    lines.extend(f"f {a + 1} {b + 1} {c + 1}" for a, b, c in faces)
    path.write_text("\n".join(lines) + "\n")


def make_box_mesh(hx: float, hy: float, hz: float) -> Tuple[np.ndarray, np.ndarray]:
    verts = np.array(
        [
            (-hx, -hy, -hz), (+hx, -hy, -hz), (+hx, +hy, -hz), (-hx, +hy, -hz),
            (-hx, -hy, +hz), (+hx, -hy, +hz), (+hx, +hy, +hz), (-hx, +hy, +hz),
        ],
        dtype=np.float32,
    )
    faces = np.array(
        [
            (0, 2, 1), (0, 3, 2),
            (4, 5, 6), (4, 6, 7),
            (0, 1, 5), (0, 5, 4),
            (1, 2, 6), (1, 6, 5),
            (2, 3, 7), (2, 7, 6),
            (3, 0, 4), (3, 4, 7),
        ],
        dtype=np.int64,
    )
    return verts, faces


def make_sphere_mesh(radius: float, subdivisions: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """Icosphere centred at origin."""
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    base = np.array(
        [
            (-1, phi, 0), (1, phi, 0), (-1, -phi, 0), (1, -phi, 0),
            (0, -1, phi), (0, 1, phi), (0, -1, -phi), (0, 1, -phi),
            (phi, 0, -1), (phi, 0, 1), (-phi, 0, -1), (-phi, 0, 1),
        ],
        dtype=np.float64,
    )
    base /= np.linalg.norm(base[0])
    faces = np.array(
        [
            (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
            (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
            (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
            (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
        ],
        dtype=np.int64,
    )

    verts = base.tolist()
    cache: dict[Tuple[int, int], int] = {}

    def midpoint(i: int, j: int) -> int:
        key = (i, j) if i < j else (j, i)
        cached = cache.get(key)
        if cached is not None:
            return cached
        m = 0.5 * (np.asarray(verts[i]) + np.asarray(verts[j]))
        m /= np.linalg.norm(m)
        verts.append(m.tolist())
        cache[key] = len(verts) - 1
        return cache[key]

    for _ in range(subdivisions):
        new_faces = []
        for a, b, c in faces:
            ab = midpoint(int(a), int(b))
            bc = midpoint(int(b), int(c))
            ca = midpoint(int(c), int(a))
            new_faces.extend([(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)])
        faces = np.array(new_faces, dtype=np.int64)

    return (np.asarray(verts, dtype=np.float32) * radius), faces


def make_hollow_cup_mesh(
    *,
    height: float,
    bottom_radius: float,
    top_radius: float,
    wall_thickness: float,
    base_thickness: float,
    n_segments: int = 48,
) -> Tuple[np.ndarray, np.ndarray]:
    """Smooth hollow-cylinder cup mesh in local frame (centre at origin, base at z=-h/2)."""
    half_h = height * 0.5
    base_z = -half_h
    rim_z = +half_h
    inner_base_z = base_z + base_thickness
    inner_bottom_r = bottom_radius - wall_thickness
    inner_top_r = top_radius - wall_thickness

    angles = np.linspace(0.0, 2.0 * np.pi, n_segments, endpoint=False)

    def ring(radius: float, z: float) -> np.ndarray:
        return np.stack(
            [radius * np.cos(angles), radius * np.sin(angles), np.full(n_segments, z)],
            axis=1,
        )

    outer_bot = ring(bottom_radius, base_z)
    outer_top = ring(top_radius, rim_z)
    inner_bot = ring(inner_bottom_r, inner_base_z)
    inner_top = ring(inner_top_r, rim_z)
    base_outer = ring(bottom_radius, base_z)
    floor_inner = ring(inner_bottom_r, inner_base_z)

    verts = np.concatenate(
        [outer_bot, outer_top, inner_bot, inner_top, base_outer, floor_inner],
        axis=0,
    )
    o_ob, o_ot, o_ib, o_it, o_base, o_floor = (
        0, n_segments, 2 * n_segments, 3 * n_segments, 4 * n_segments, 5 * n_segments,
    )
    base_centre_idx = verts.shape[0]
    verts = np.concatenate([verts, np.array([[0.0, 0.0, base_z]])], axis=0)
    floor_centre_idx = verts.shape[0]
    verts = np.concatenate([verts, np.array([[0.0, 0.0, inner_base_z]])], axis=0)

    faces = []
    for i in range(n_segments):
        j = (i + 1) % n_segments
        faces.append([o_ob + i, o_ob + j, o_ot + j])
        faces.append([o_ob + i, o_ot + j, o_ot + i])
        faces.append([o_ib + i, o_it + j, o_ib + j])
        faces.append([o_ib + i, o_it + i, o_it + j])
        faces.append([o_ot + i, o_it + j, o_it + i])
        faces.append([o_ot + i, o_ot + j, o_it + j])
        faces.append([base_centre_idx, o_base + j, o_base + i])
        faces.append([floor_centre_idx, o_floor + i, o_floor + j])

    return verts.astype(np.float32), np.asarray(faces, dtype=np.int64)


def axis_angle_quat_xyzw(axis: Tuple[float, float, float], degrees: float) -> np.ndarray:
    """Build a (x, y, z, w) quaternion from axis-angle in degrees."""
    half = math.radians(degrees) * 0.5
    s = math.sin(half)
    c = math.cos(half)
    ax = np.asarray(axis, dtype=np.float64)
    ax = ax / np.linalg.norm(ax)
    return np.array([ax[0] * s, ax[1] * s, ax[2] * s, c], dtype=np.float64)


def quat_rotate_points_xyzw(quat: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Rotate points by quaternion in (x, y, z, w) order."""
    qx, qy, qz, qw = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
    R = np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ]
    )
    return np.asarray(points) @ R.T


def transform_verts_xyzw(verts: np.ndarray, pos: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    """Apply rigid transform: world = R(quat_xyzw) @ verts + pos."""
    return (quat_rotate_points_xyzw(quat_xyzw, verts) + np.asarray(pos)).astype(np.float32)
