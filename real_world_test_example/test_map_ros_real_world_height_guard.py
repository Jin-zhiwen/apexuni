import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class MapRosRealWorldHeightGuardTest(unittest.TestCase):
    def test_real_world_launch_uses_more_conservative_min_height(self):
        launch_source = (
            REPO_ROOT
            / "src"
            / "planner"
            / "exploration_manager"
            / "launch"
            / "algorithm_traj.xml"
        ).read_text()

        self.assertIn('<arg name="obstacle_min_above_floor_" default="0.28"/>', launch_source)


if __name__ == "__main__":
    unittest.main()
