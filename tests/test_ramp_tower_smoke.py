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


class RampTowerSmokeTest(unittest.TestCase):
    def _assert_common_outcome(self, result) -> None:
        self.assertTrue(np.isfinite(result.final_particle_positions).all())
        self.assertGreater(result.final_ball_position[1], result.initial_ball_position[1] + 0.50)

    def _target_metrics(self, result) -> list[tuple[float, float]]:
        target_metrics = []
        for wall_index, (start, end) in enumerate(result.target_particle_ranges):
            initial_center = result.initial_target_centers[wall_index]
            final_center = result.final_target_centers[wall_index]
            initial_target = result.initial_particle_positions[start:end]
            final_target = result.final_particle_positions[start:end]
            translation = float(np.linalg.norm(final_center - initial_center))
            deformation = float(
                np.linalg.norm(
                    (final_target - final_center) - (initial_target - initial_center),
                    axis=1,
                ).mean()
            )
            target_metrics.append((translation, deformation))
        return target_metrics

    def test_jelly_single_wall_smoke(self) -> None:
        result = run_simulation(viewer="null", num_frames=180, device="cpu", wall_count=1)

        self._assert_common_outcome(result)
        self.assertEqual(len(result.target_particle_ranges), 1)

        wall_translation, wall_deformation = self._target_metrics(result)[0]
        self.assertTrue(wall_translation > 0.04 or wall_deformation > 0.03)

    def test_jelly_domino_smoke(self) -> None:
        result = run_simulation(viewer="null", num_frames=180, device="cpu", wall_count=2)

        self._assert_common_outcome(result)
        self.assertEqual(len(result.target_particle_ranges), 2)

        target_metrics = self._target_metrics(result)
        first_wall_translation, first_wall_deformation = target_metrics[0]
        second_wall_translation, second_wall_deformation = target_metrics[1]

        self.assertTrue(first_wall_translation > 0.04 or first_wall_deformation > 0.03)
        self.assertTrue(second_wall_translation > 0.03 or second_wall_deformation > 0.01)


if __name__ == "__main__":
    unittest.main()
