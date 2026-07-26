import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class RealWorldClearanceAndVisualizationTest(unittest.TestCase):
    def test_real_world_clearance_margins_are_aligned_across_planners(self) -> None:
        source = (
            REPO_ROOT
            / "src"
            / "planner"
            / "path_searching"
            / "src"
            / "kino_astar.cpp"
        ).read_text()
        planning_param = (
            REPO_ROOT
            / "src"
            / "planner"
            / "trajectory_manager"
            / "config"
            / "planning_param.yaml"
        ).read_text()
        launch_source = (
            REPO_ROOT
            / "src"
            / "planner"
            / "exploration_manager"
            / "launch"
            / "algorithm_traj.xml"
        ).read_text()

        self.assertIn("hard_collision_padding_", source)
        self.assertIn('"/hard_collision_padding"', source)
        self.assertIn("query_pos += hard_collision_padding_ * offset;", source)
        self.assertIn("const int occupancy = map_->getOccupancy(query_pos);", source)
        self.assertNotIn("map_->getInflateOccupancy(pos) == 1", source)
        self.assertIn('<arg name="obstacles_inflation_" default="0.08"/>', launch_source)
        self.assertIn('<param name="astar/preferred_clearance" value="0.35" type="double"/>',
                      launch_source)
        self.assertIn('<param name="astar/wall_penalty_weight" value="8.0" type="double"/>',
                      launch_source)
        self.assertIn("hard_collision_padding: 0.08", planning_param)
        self.assertIn("safe_dist: 0.14", planning_param)

    def test_real_world_visualization_clears_global_point_and_keeps_local_point(self) -> None:
        source = (
            REPO_ROOT
            / "src"
            / "planner"
            / "exploration_manager"
            / "src"
            / "exploration_fsm_traj.cpp"
        ).read_text()

        self.assertNotIn("global_points.push_back(ed_ptr->next_pos_);", source)
        self.assertIn('"global_point"', source)
        self.assertIn("drawSpheres({}, fp_->vis_scale_ * 3.5,", source)
        self.assertIn('"local_point"', source)


if __name__ == "__main__":
    unittest.main()
