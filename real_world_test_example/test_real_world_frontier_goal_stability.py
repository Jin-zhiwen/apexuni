import re
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class RealWorldFrontierGoalStabilityTest(unittest.TestCase):
    def test_frontier_map_tracks_safe_navigation_anchor_separately_from_boundary_centroid(self):
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

        self.assertIn("Vector2d average_;", header)
        self.assertIn("Vector2d navigation_point_;", header)
        self.assertIn("void refineFrontierNavigationPoint(Frontier2D& frontier);", header)
        self.assertIn("refineFrontierNavigationPoint(", source)
        self.assertIn("averages.push_back(frontier.navigation_point_);", source)

    def test_exploration_frontier_paths_are_refined_to_safe_free_space_targets(self):
        source = (
            REPO_ROOT
            / "src"
            / "planner"
            / "exploration_manager"
            / "src"
            / "exploration_manager.cpp"
        ).read_text()

        self.assertIn("refineFrontierGoalFromPath(", source)
        self.assertRegex(
            source,
            re.compile(
                r"if\s*\(\s*path_finder_->astarSearch\(start,\s*end,\s*0\.25,\s*0\.01\)"
                r"\s*==\s*Astar2D::REACH_END\s*\)\s*\{",
                re.MULTILINE,
            ),
        )
        self.assertIn("shortenPath(refined_path);", source)
        self.assertIn("refined_pos = refineFrontierGoalFromPath(refined_path);", source)

    def test_real_world_traj_fsm_reuses_previous_exploration_goal_when_frontier_is_stable(self):
        source = (
            REPO_ROOT
            / "src"
            / "planner"
            / "exploration_manager"
            / "src"
            / "exploration_fsm_traj.cpp"
        ).read_text()

        self.assertIn("fd_->replan_flag_", source)
        self.assertIn("fd_->last_next_pos_", source)
        self.assertIn("fd_->stucking_next_pos_count_", source)
        self.assertIn("shouldKeepPreviousExplorationGoal(", source)
        self.assertIn("expl_manager_->ed_->next_best_path_ = previous_path;", source)
        self.assertIn("expl_manager_->ed_->next_pos_ = previous_goal;", source)

    def test_real_world_traj_server_has_catchup_margin_above_planner_limit(self):
        launch_source = (
            REPO_ROOT
            / "src"
            / "planner"
            / "exploration_manager"
            / "launch"
            / "exploration_traj.launch"
        ).read_text()
        planning_source = (
            REPO_ROOT
            / "src"
            / "planner"
            / "trajectory_manager"
            / "config"
            / "planning_param.yaml"
        ).read_text()

        self.assertIn("max_vel: 0.25", planning_source)
        self.assertIn('<param name="max_correction_vel" value="0.35" type="double"/>', launch_source)

    def test_exploration_manager_rejects_post_optimization_wall_grazing_trajectory(self):
        source = (
            REPO_ROOT
            / "src"
            / "planner"
            / "exploration_manager"
            / "src"
            / "exploration_manager.cpp"
        ).read_text()

        self.assertIn("isTrajectoryCollisionFree(", source)
        self.assertIn("if (!isTrajectoryCollisionFree(gcopter_->local_trajectory_))", source)
        self.assertIn("Trajectory rejected after optimization due to footprint collision", source)


if __name__ == "__main__":
    unittest.main()
