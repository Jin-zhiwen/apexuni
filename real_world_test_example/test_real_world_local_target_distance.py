import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class RealWorldLocalTargetDistanceTest(unittest.TestCase):
    def test_real_world_traj_fsm_loads_configured_local_target_distance(self):
        launch_source = (
            REPO_ROOT
            / "src"
            / "planner"
            / "exploration_manager"
            / "launch"
            / "algorithm_traj.xml"
        ).read_text()
        fsm_source = (
            REPO_ROOT
            / "src"
            / "planner"
            / "exploration_manager"
            / "src"
            / "exploration_fsm_traj.cpp"
        ).read_text()

        self.assertIn('name="fsm/local_target_distance" value="1.5"', launch_source)
        self.assertIn(
            'nh.param("fsm/local_target_distance", fp_->local_target_distance_',
            fsm_source,
        )

    def test_real_world_traj_fsm_loads_configured_local_target_yaw_limit(self):
        launch_source = (
            REPO_ROOT
            / "src"
            / "planner"
            / "exploration_manager"
            / "launch"
            / "algorithm_traj.xml"
        ).read_text()
        fsm_source = (
            REPO_ROOT
            / "src"
            / "planner"
            / "exploration_manager"
            / "src"
            / "exploration_fsm_traj.cpp"
        ).read_text()

        self.assertIn('name="fsm/local_target_max_yaw_error"', launch_source)
        self.assertIn(
            'nh.param("fsm/local_target_max_yaw_error", local_target_max_yaw_error_',
            fsm_source,
        )


if __name__ == "__main__":
    unittest.main()
