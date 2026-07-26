import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class RealWorldTrackingLimitsTest(unittest.TestCase):
    def test_real_world_launch_gives_controller_catchup_velocity_margin(self) -> None:
        launch_source = (
            REPO_ROOT
            / "src"
            / "planner"
            / "exploration_manager"
            / "launch"
            / "exploration_traj.launch"
        ).read_text()

        self.assertIn('<param name="max_correction_vel" value="0.35" type="double"/>', launch_source)
        self.assertIn('<param name="max_correction_omega" value="0.50" type="double"/>', launch_source)

    def test_traj_server_derates_linear_speed_when_tracking_error_grows(self) -> None:
        source = (
            REPO_ROOT
            / "src"
            / "planner"
            / "trajectory_manager"
            / "src"
            / "traj_server.cpp"
        ).read_text()

        self.assertIn("tracking_slowdown_error_", source)
        self.assertIn("tracking_stop_error_", source)
        self.assertIn("tracking_min_speed_scale_", source)
        self.assertIn("tracking_min_effective_vx_", source)
        self.assertIn("applyTrackingErrorSpeedLimit", source)
        self.assertIn("applyTrackingErrorYawLimit", source)
        self.assertIn("track_err >=", source)
        self.assertIn("twist_msg.linear.x = applyTrackingErrorSpeedLimit", source)
        self.assertIn("twist_msg.angular.z = applyTrackingErrorYawLimit", source)
        self.assertIn("std::copysign(tracking_min_effective_vx_, limited_vx)", source)

    def test_traj_server_uses_wall_clock_elapsed_time_for_reference_sampling(self) -> None:
        source = (
            REPO_ROOT
            / "src"
            / "planner"
            / "trajectory_manager"
            / "src"
            / "traj_server.cpp"
        ).read_text()

        self.assertIn(
            "double elapsed_time = (current_time - start_time_).toSec();",
            source,
        )
        self.assertIn("traj_->getPos(elapsed_time)", source)
        self.assertIn("double temp_t = elapsed_time + i * mpc_dt_;", source)
        self.assertNotIn("updateAlignedTrackingTime", source)
        self.assertNotIn("findOdomAlignedTrajectoryTime", source)
        self.assertNotIn("tracking_time", source)
        self.assertNotIn("ref_t=", source)

    def test_real_world_launch_does_not_enable_progress_based_tracking(self) -> None:
        launch_source = (
            REPO_ROOT
            / "src"
            / "planner"
            / "exploration_manager"
            / "launch"
            / "exploration_traj.launch"
        ).read_text()

        self.assertNotIn("progress_alignment", launch_source)


if __name__ == "__main__":
    unittest.main()
