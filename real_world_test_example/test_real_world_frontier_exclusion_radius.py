import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class RealWorldFrontierExclusionRadiusTest(unittest.TestCase):
    def test_real_world_launch_covers_head_camera_near_field_hole(self) -> None:
        launch_source = (
            REPO_ROOT
            / "src"
            / "planner"
            / "exploration_manager"
            / "launch"
            / "algorithm_traj.xml"
        ).read_text()

        self.assertIn(
            '<param name="frontier/real_world_exclusion_radius" value="0.65" type="double"/>',
            launch_source,
        )

    def test_frontier_map_tracks_real_world_exclusion_radius_and_latest_sensor_pose(self) -> None:
        header = (
            REPO_ROOT
            / "src"
            / "planner"
            / "plan_env"
            / "include"
            / "plan_env"
            / "frontier_map2d.h"
        ).read_text()
        source = (
            REPO_ROOT
            / "src"
            / "planner"
            / "plan_env"
            / "src"
            / "frontier_map2d.cpp"
        ).read_text()

        self.assertIn("bool shouldExcludeFrontierNearRobot(const Frontier2D& frontier) const;", header)
        self.assertIn("bool is_real_world_;", header)
        self.assertIn("double real_world_exclusion_radius_;", header)
        self.assertIn("bool have_latest_sensor_pos_;", header)
        self.assertIn("Vector2d latest_sensor_pos_;", header)

        self.assertIn('nh.param("is_real_world", is_real_world_, false);', source)
        self.assertIn(
            'nh.param("frontier/real_world_exclusion_radius", real_world_exclusion_radius_, 0.0);',
            source,
        )
        self.assertIn("latest_sensor_pos_ = sensor_pos;", source)
        self.assertIn("have_latest_sensor_pos_ = true;", source)
        self.assertIn("bool FrontierMap2D::shouldExcludeFrontierNearRobot(", source)
        self.assertIn("latest_sensor_pos_, real_world_exclusion_radius_", source)

    def test_frontier_outputs_skip_frontiers_inside_robot_near_field_hole(self) -> None:
        source = (
            REPO_ROOT
            / "src"
            / "planner"
            / "plan_env"
            / "src"
            / "frontier_map2d.cpp"
        ).read_text()

        self.assertIn("if (shouldExcludeFrontierNearRobot(frontier))", source)
        self.assertIn("continue;", source)

if __name__ == "__main__":
    unittest.main()
