from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from npm_sim.ramp_tower import run_simulation


def _quat_angle_delta_rad(q0: np.ndarray, q1: np.ndarray) -> float:
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.clip(abs(np.dot(q0, q1)), -1.0, 1.0))
    return 2.0 * np.arccos(dot)


class RampTowerSmokeTest(unittest.TestCase):
    def test_ramp_tower_smoke(self) -> None:
        result = run_simulation(viewer="null", num_frames=180, device="cpu")

        self.assertTrue(np.isfinite(result.final_body_poses).all())
        self.assertGreater(result.final_ball_position[1], result.initial_ball_position[1] + 0.20)

        cube_translation = np.linalg.norm(
            result.final_cube_transforms[:, :3] - result.initial_cube_transforms[:, :3],
            axis=1,
        )
        cube_rotation = np.array(
            [
                _quat_angle_delta_rad(result.initial_cube_transforms[i, 3:], result.final_cube_transforms[i, 3:])
                for i in range(len(result.cube_body_indices))
            ]
        )

        moved_cube = np.any(cube_translation > 0.02) or np.any(cube_rotation > np.deg2rad(5.0))
        self.assertTrue(moved_cube)


if __name__ == "__main__":
    unittest.main()
