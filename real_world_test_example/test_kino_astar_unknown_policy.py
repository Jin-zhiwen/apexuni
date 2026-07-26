import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class KinoAstarUnknownPolicyTest(unittest.TestCase):
    def test_real_world_kinoastar_unknown_collision_policy_is_configurable(self) -> None:
        header = (
            REPO_ROOT
            / "src"
            / "planner"
            / "path_searching"
            / "include"
            / "path_searching"
            / "kino_astar.h"
        ).read_text()
        source = (
            REPO_ROOT
            / "src"
            / "planner"
            / "path_searching"
            / "src"
            / "kino_astar.cpp"
        ).read_text()
        launch_source = (
            REPO_ROOT
            / "src"
            / "planner"
            / "exploration_manager"
            / "launch"
            / "algorithm_traj.xml"
        ).read_text()

        self.assertIn("bool unknown_as_collision_ = true;", header)
        self.assertIn('"/kinoastar/unknown_as_collision"', source)
        self.assertIn(
            "occupancy == SDFMap2D::OCCUPIED ||\n"
            "        (unknown_as_collision_ && occupancy == SDFMap2D::UNKNOWN)",
            source,
        )
        self.assertIn(
            '<param name="kinoastar/unknown_as_collision" value="false" type="bool"/>',
            launch_source,
        )


if __name__ == "__main__":
    unittest.main()
