from pathlib import Path
import subprocess
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_COMMIT = "c3e91d4"

EXACT_MATCH_FILES = [
    "src/planner/exploration_manager/include/exploration_manager/exploration_fsm_traj.h",
    "src/planner/exploration_manager/include/exploration_manager/exploration_manager.h",
    "src/planner/exploration_manager/src/exploration_manager.cpp",
]


def git_show(path: str) -> str:
    return subprocess.check_output(
        ["git", "show", f"{BASELINE_COMMIT}:{path}"],
        cwd=REPO_ROOT,
        text=True,
    )


def read_repo_file(path: str) -> str:
    return (REPO_ROOT / path).read_text()


class VioTrajPlanningBaselineTest(unittest.TestCase):
    def test_vio_traj_planning_files_match_snapshot_baseline(self) -> None:
        for rel_path in EXACT_MATCH_FILES:
            with self.subTest(path=rel_path):
                self.assertEqual(read_repo_file(rel_path), git_show(rel_path))

    def test_real_world_launch_only_enables_init_rotation_by_default(self) -> None:
        launch_source = read_repo_file("src/planner/exploration_manager/launch/exploration_traj.launch")

        self.assertIn('<arg name="need_init_rotation" default="true"/>', launch_source)
        self.assertIn('<param name="max_correction_vel" value="0.35" type="double"/>', launch_source)
        self.assertIn('<param name="max_correction_omega" value="0.50" type="double"/>', launch_source)
        self.assertIn('<param name="tracking_slowdown_error" value="0.15" type="double"/>', launch_source)
        self.assertIn('<param name="tracking_stop_error" value="0.55" type="double"/>', launch_source)
        self.assertIn('<param name="tracking_min_speed_scale" value="0.25" type="double"/>', launch_source)
        self.assertIn('<param name="tracking_min_effective_vx" value="0.14" type="double"/>', launch_source)
        self.assertNotIn("progress_alignment", launch_source)

    def test_current_collision_body_is_preserved(self) -> None:
        planning_param = read_repo_file("src/planner/trajectory_manager/config/planning_param.yaml")

        self.assertIn("length: 0.50", planning_param)
        self.assertIn("width: 0.30", planning_param)
        self.assertIn("height: 0.40", planning_param)
        self.assertIn("odom_to_center_x: -0.10", planning_param)

    def test_real_world_visualization_hides_global_point_marker(self) -> None:
        source = read_repo_file("src/planner/exploration_manager/src/exploration_fsm_traj.cpp")

        self.assertIn('visualization_->drawSpheres({}, fp_->vis_scale_ * 3.5,', source)
        self.assertIn('"global_point"', source)


if __name__ == "__main__":
    unittest.main()
