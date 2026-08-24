import re
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class KinoAstarSafetyGuardsTest(unittest.TestCase):
    def test_sampled_collision_invalidates_frontend_trajectory(self):
        source = (
            REPO_ROOT
            / "src"
            / "planner"
            / "path_searching"
            / "src"
            / "kino_astar.cpp"
        ).read_text()
        manager_source = (
            REPO_ROOT
            / "src"
            / "planner"
            / "exploration_manager"
            / "src"
            / "exploration_manager.cpp"
        ).read_text()

        self.assertIn("rejectSampledCollisionTrajectory", source)
        self.assertRegex(
            source,
            re.compile(
                r"if\s*\(\s*occ\s*\)\s*\{\s*"
                r"rejectSampledCollisionTrajectory\(Eigen::Vector2d\(px,\s*py\)\);\s*"
                r"return;",
                re.MULTILINE,
            ),
        )
        self.assertIn("has_path_ = false;", source)
        self.assertIn("flat_trajs_.clear();", source)
        self.assertIn("totalTrajTime = 0.0;", source)
        self.assertIn("if (sample_traj_.size() < 2)", source)
        self.assertIn("if (!kinoastar_->has_path_ || kinoastar_->flat_trajs_.empty())", manager_source)

    def test_evaluate_pos_guards_empty_and_out_of_range_time(self):
        source = (
            REPO_ROOT
            / "src"
            / "planner"
            / "path_searching"
            / "src"
            / "kino_astar.cpp"
        ).read_text()
        evaluate_pos = source.split("Eigen::Vector4d KinoAstar::evaluatePos", 1)[1]

        self.assertIn("if (sample_traj_.empty() || shot_timeList.empty() || shotindex.size() < 2)", evaluate_pos)
        self.assertIn("return Eigen::Vector4d::Zero();", evaluate_pos)
        self.assertIn("if (index < 0)", evaluate_pos)
        self.assertIn("return sample_traj_.back();", evaluate_pos)
        self.assertIn("double CutTime = 0.0;", evaluate_pos)
        self.assertIn("double normalize_yaw = state[2];", evaluate_pos)
        self.assertIn("constexpr double kTimeBoundaryEps", evaluate_pos)
        self.assertIn("time_diff", evaluate_pos)
        self.assertIn("input_t=%.9f", evaluate_pos)
        self.assertIn("totalTrajTime=%.9f", evaluate_pos)
        self.assertIn("diff=%.9f", evaluate_pos)
        self.assertIn("ROS_WARN_THROTTLE", evaluate_pos)

    def test_one_shot_collision_check_uses_dense_sampling_and_terminal_sample(self):
        source = (
            REPO_ROOT
            / "src"
            / "planner"
            / "path_searching"
            / "src"
            / "kino_astar.cpp"
        ).read_text()
        compute_shot = source.split("double KinoAstar::computeShotTraj", 1)[1]

        self.assertIn("dense_check_step", compute_shot)
        self.assertIn("0.5 * collision_interval_", compute_shot)
        self.assertIn("0.5 * sampletime_ * max_vel_", compute_shot)
        self.assertIn("std::min(l, len)", compute_shot)
        self.assertRegex(
            compute_shot,
            re.compile(
                r"for\s*\(\s*double\s+l\s*=\s*0.0;\s*"
                r"l\s*<=\s*len\s*\+\s*1.0e-6;\s*"
                r"l\s*\+=\s*dense_check_step",
                re.MULTILINE,
            ),
        )

    def test_repeated_frontend_collision_failures_force_target_reconsideration(self):
        source = (
            REPO_ROOT
            / "src"
            / "planner"
            / "exploration_manager"
            / "src"
            / "exploration_fsm_traj.cpp"
        ).read_text()

        self.assertIn("planning_failure_count_", source)
        self.assertIn("MAX_CONSECUTIVE_PLANNING_FAILURES", source)
        self.assertIn("setForceDormantFrontier(object_goal_pos)", source)
        self.assertIn("dormant_frontier_flag_ = true", source)

    def test_blocked_in_place_rotation_falls_back_without_dormanting_frontier(self):
        source = (
            REPO_ROOT
            / "src"
            / "planner"
            / "exploration_manager"
            / "src"
            / "exploration_fsm_traj.cpp"
        ).read_text()
        rotation_branch = source.split(
            "if (shouldRotateBeforeTranslation(", 1
        )[1].split("Eigen::Vector2d goal_pos", 1)[0]

        self.assertIn("fall back to KinoAstar curved trajectory", rotation_branch)
        self.assertNotIn("setForceDormantFrontier", rotation_branch)
        self.assertNotIn("return TrajPlannerResult::FAILED", rotation_branch)

    def test_unsafe_local_target_requires_repeated_failures_before_dormancy(self):
        source = (
            REPO_ROOT
            / "src"
            / "planner"
            / "exploration_manager"
            / "src"
            / "exploration_fsm_traj.cpp"
        ).read_text()
        unsafe_target_branch = source.split(
            "if (!local_target_valid)", 1
        )[1].split("const Eigen::Vector2d rotation_pos", 1)[0]

        self.assertIn("planning_failure_count_++", unsafe_target_branch)
        self.assertIn("MAX_CONSECUTIVE_PLANNING_FAILURES", unsafe_target_branch)
        self.assertRegex(
            unsafe_target_branch,
            re.compile(
                r"if\s*\(planning_failure_count_\s*>=\s*"
                r"FSMConstantsReal::MAX_CONSECUTIVE_PLANNING_FAILURES\)\s*\{\s*"
                r"ROS_WARN\([^;]+;\s*"
                r"expl_manager_->frontier_map2d_->setForceDormantFrontier",
                re.MULTILINE,
            ),
        )


if __name__ == "__main__":
    unittest.main()
