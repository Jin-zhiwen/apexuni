import unittest
from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[1]


class ValueMapVisualSmoothingTest(unittest.TestCase):
    def test_insinav_uses_full_frame_similarity_for_dense_value_map(self) -> None:
        config_source = (
            REPO_ROOT
            / "real_world_test_example"
            / "config"
            / "real_world_test_insinav.yaml"
        ).read_text()

        self.assertIn("dense_full_frame_value_map: true", config_source)
        self.assertIn('no_candidate_mode: "full_frame"', config_source)
        self.assertRegex(config_source, r"(?m)^\s+no_candidate_score:\s+0\.0(?:\s|#|$)")

    def test_value_map_visualization_smooths_only_published_rviz_intensity(self) -> None:
        map_ros_header = (
            REPO_ROOT / "src" / "planner" / "plan_env" / "include" / "plan_env" / "map_ros.h"
        ).read_text()
        map_ros_source = (
            REPO_ROOT / "src" / "planner" / "plan_env" / "src" / "map_ros.cpp"
        ).read_text()

        self.assertIn("double smoothValueMapIntensity(const Eigen::Vector2i& idx) const;", map_ros_header)
        self.assertIn("bool isValueMapVisualCellFree(const Eigen::Vector2i& idx) const;", map_ros_header)
        self.assertIn("if (!isValueMapVisualCellFree(idx))", map_ros_header)
        self.assertIn("value = smoothValueMapIntensity(idx);", map_ros_header)
        self.assertIn("pt.intensity = value;", map_ros_header)
        self.assertIn("constexpr double kVisualMinIntensity = 0.0;", map_ros_header)
        self.assertIn("constexpr double kVisualMaxIntensity = 1.0;", map_ros_header)
        self.assertIn("constexpr double kVisualRawMin = 0.08;", map_ros_header)
        self.assertIn("constexpr double kVisualRawMax = 0.80;", map_ros_header)
        self.assertIn("semantic_visual_free_space_floor_", map_ros_header)
        self.assertIn("std::max(0.0, std::min(1.0, semantic_visual_free_space_floor_))", map_ros_header)
        self.assertIn("return visual_floor;", map_ros_header)
        self.assertIn("std::max(visual_floor, visual_intensity)", map_ros_header)
        self.assertIn(
            'node_.param("semantic/visual_free_space_floor", semantic_visual_free_space_floor_, 0.06);',
            map_ros_source,
        )
        self.assertIn("const double visual_ratio =", map_ros_header)
        self.assertIn(
            "const double visual_intensity =",
            map_ros_header,
        )
        self.assertNotIn("getSemanticObjectVisualOverlayIntensity", map_ros_header)
        self.assertNotIn("updateSemanticObjectVisualOverlay", map_ros_header)
        self.assertNotIn("semantic_object_visual_buffer_", map_ros_header)
        self.assertNotIn("semantic/object_visual_overlay_duration", map_ros_source)
        self.assertNotIn("std::sqrt(clamped)", map_ros_header)
        self.assertRegex(
            map_ros_source,
            re.compile(
                r"updateValueMap\(\s*camera_pos,\s*camera_yaw,\s*semantic_value_grids,\s*"
                r"itm_score_\s*\)",
                re.MULTILINE,
            ),
        )

    def test_visual_free_space_floor_is_configured_without_polluting_planning_values(self) -> None:
        map_ros_header = (
            REPO_ROOT / "src" / "planner" / "plan_env" / "include" / "plan_env" / "map_ros.h"
        ).read_text()
        value_map_source = (
            REPO_ROOT / "src" / "planner" / "plan_env" / "src" / "value_map2d.cpp"
        ).read_text()
        exploration_source = (
            REPO_ROOT
            / "src"
            / "planner"
            / "exploration_manager"
            / "src"
            / "exploration_manager.cpp"
        ).read_text()
        real_world_launch = (
            REPO_ROOT
            / "src"
            / "planner"
            / "exploration_manager"
            / "launch"
            / "algorithm_traj.xml"
        ).read_text()

        self.assertIn('name="semantic/visual_free_space_floor" value="0.06"', real_world_launch)
        self.assertIn("semantic_visual_free_space_floor_", map_ros_header)
        self.assertNotIn("visual_free_space_floor", value_map_source)
        self.assertNotIn("semantic_visual_free_space_floor", value_map_source)
        self.assertIn("sdf_map_->value_map_->getValue", exploration_source)
        self.assertNotIn("visual_free_space_floor", exploration_source)

    def test_value_map_real_scores_follow_apexnavmain_full_frame_fusion(self) -> None:
        value_map_header = (
            REPO_ROOT / "src" / "planner" / "plan_env" / "include" / "plan_env" / "value_map2d.h"
        ).read_text()
        value_map_source = (
            REPO_ROOT / "src" / "planner" / "plan_env" / "src" / "value_map2d.cpp"
        ).read_text()
        real_world_launch = (
            REPO_ROOT
            / "src"
            / "planner"
            / "exploration_manager"
            / "launch"
            / "algorithm_traj.xml"
        ).read_text()

        self.assertNotIn("getDirectionalValueWeight", value_map_header)
        self.assertNotIn("directional_value_weight_enabled_", value_map_header)
        self.assertNotIn("semantic/main_path_weight_enabled", value_map_source)
        self.assertIn("double now_value = itm_score;", value_map_source)
        self.assertIn('name="semantic/use_raycast_free_grids" value="true"', real_world_launch)

    def test_semantic_object_evidence_fuses_as_directional_fov_bias_without_extra_rviz_layer(self) -> None:
        map_ros_header = (
            REPO_ROOT / "src" / "planner" / "plan_env" / "include" / "plan_env" / "map_ros.h"
        ).read_text()
        map_ros_source = (
            REPO_ROOT / "src" / "planner" / "plan_env" / "src" / "map_ros.cpp"
        ).read_text()
        value_map_header = (
            REPO_ROOT / "src" / "planner" / "plan_env" / "include" / "plan_env" / "value_map2d.h"
        ).read_text()
        value_map_source = (
            REPO_ROOT / "src" / "planner" / "plan_env" / "src" / "value_map2d.cpp"
        ).read_text()
        real_world_bridge = (
            REPO_ROOT / "real_world_test_example" / "real_world_test_insinav.py"
        ).read_text()
        real_world_config = (
            REPO_ROOT
            / "real_world_test_example"
            / "config"
            / "real_world_test_insinav.yaml"
        ).read_text()
        real_world_launch = (
            REPO_ROOT
            / "src"
            / "planner"
            / "exploration_manager"
            / "launch"
            / "algorithm_traj.xml"
        ).read_text()

        self.assertIn("/detector/semantic_clouds_with_scores", map_ros_source)
        self.assertIn("detectedSemanticCloudCallback", map_ros_header)
        self.assertIn("object_value_evidence_enabled_", map_ros_header)
        self.assertIn("object_fov_angle_sigma_", map_ros_header)
        self.assertIn("object_fov_range_sigma_", map_ros_header)
        self.assertIn("object_fov_max_angle_", map_ros_header)
        self.assertIn("object_fov_max_range_error_", map_ros_header)
        self.assertIn("object_fov_weight_", map_ros_header)
        self.assertIn("object_fov_decay_tau_", map_ros_header)
        self.assertIn("latest_semantic_value_grids_", map_ros_header)
        self.assertNotIn("semantic_object_visual_overlay_enabled_", map_ros_header)
        self.assertIn(
            'node_.param("semantic/object_evidence_enabled", object_value_evidence_enabled_, false);',
            map_ros_source,
        )
        self.assertIn("semantic/object_fov_angle_sigma", map_ros_source)
        self.assertIn("semantic/object_fov_range_sigma", map_ros_source)
        self.assertIn("semantic/object_fov_max_angle", map_ros_source)
        self.assertIn("semantic/object_fov_max_range_error", map_ros_source)
        self.assertIn("semantic/object_fov_weight", map_ros_source)
        self.assertIn("semantic/object_fov_decay_tau", map_ros_source)
        self.assertNotIn("object_visual_overlay_enabled", map_ros_source)
        self.assertIn('name="semantic/object_evidence_enabled" value="true"', real_world_launch)
        self.assertIn('name="semantic/object_fov_angle_sigma" value="0.35"', real_world_launch)
        self.assertIn('name="semantic/object_fov_range_sigma" value="0.60"', real_world_launch)
        self.assertIn('name="semantic/object_fov_max_angle" value="0.75"', real_world_launch)
        self.assertIn('name="semantic/object_fov_max_range_error" value="1.00"', real_world_launch)
        self.assertIn('name="semantic/object_fov_weight" value="0.70"', real_world_launch)
        self.assertIn('name="semantic/object_fov_decay_tau" value="3.00"', real_world_launch)
        self.assertNotIn("object_visual_overlay_enabled", real_world_launch)
        guard_idx = map_ros_source.index("if (!object_value_evidence_enabled_)")
        update_idx = map_ros_source.index("updateObjectFovEvidence")
        self.assertLess(guard_idx, update_idx)
        self.assertIn("computeCloudCentroid2D", map_ros_source)
        self.assertIn("latest_semantic_value_grids_", map_ros_source)
        self.assertIn("decayObjectEvidence", value_map_header)
        self.assertIn("decayObjectEvidence", value_map_source)
        self.assertIn("object_value_buffer_", value_map_header)
        self.assertIn("return fuseBaseAndObjectValue", value_map_header)
        self.assertIn("updateObjectFovEvidence", value_map_header)
        self.assertIn("updateObjectFovEvidence", value_map_source)
        self.assertIn("updateObjectEvidence", value_map_header)
        self.assertIn("updateObjectEvidence", value_map_source)
        self.assertIn("object_value_update_radius_", value_map_header)
        self.assertIn("semantic/object_update_radius", value_map_source)
        self.assertIn("score_weight <= 1e-6", value_map_source)
        self.assertIn("const double angular_weight = std::exp", value_map_source)
        self.assertIn("const double range_weight = std::exp", value_map_source)
        self.assertIn("std::abs(range - object_range) > max_range_error", value_map_source)
        self.assertIn("std::abs(angle_error) > max_angle", value_map_source)
        self.assertIn("object_value_buffer_[adr] = std::max", value_map_source)
        self.assertNotIn("value_buffer_[adr] = std::max(last_value, std::min(1.0, fused_value));", value_map_source)
        self.assertIn("publish_semantic_object_clouds: true", real_world_config)
        self.assertIn("self.publish_semantic_object_clouds", real_world_bridge)
        self.assertIn("self.publish_semantic_object_clouds", real_world_bridge)
        self.assertIn("and dino_score_source == \"crop\"", real_world_bridge)
        self.assertNotIn("itmScoreIsValid()", map_ros_source)
        self.assertNotIn("itm_score_weight_ <= 1e-6", map_ros_header)
        self.assertNotIn("itm_score_weight_ = msg->data;", map_ros_source)

    def test_dense_value_map_publishes_full_frame_similarity_every_goal_frame(self) -> None:
        bridge_source = (
            REPO_ROOT / "real_world_test_example" / "real_world_test_insinav.py"
        ).read_text()
        map_ros_source = (
            REPO_ROOT / "src" / "planner" / "plan_env" / "src" / "map_ros.cpp"
        ).read_text()

        self.assertIn("self.dino_dense_full_frame_value_map", bridge_source)
        self.assertIn("dense_value_score = float(self.clip_client.cosine(rgb_cv))", bridge_source)
        self.assertIn("self.itm_score_pub_.publish(Float64(data=float(dense_value_score)))", bridge_source)
        self.assertIn("if (itm_score_ != -1.0)", map_ros_source)

    def test_real_world_rviz_uses_full_value_map_intensity_range(self) -> None:
        rviz_traj = (
            REPO_ROOT
            / "src"
            / "planner"
            / "exploration_manager"
            / "config"
            / "ApexNav_Traj.rviz"
        ).read_text()
        rviz_real = (
            REPO_ROOT
            / "src"
            / "planner"
            / "exploration_manager"
            / "config"
            / "ApexNav.rviz"
        ).read_text()

        for rviz_source in (rviz_traj, rviz_real):
            block = rviz_source.split("Name: semantic_score_map", 1)[0].split("- Alpha: 1")[-1]
            self.assertIn("Min Intensity: 0", block)
            self.assertIn("Max Intensity: 1", block)

    def test_semantic_frontier_scoring_guards_value_map_bounds(self) -> None:
        manager_header = (
            REPO_ROOT
            / "src"
            / "planner"
            / "exploration_manager"
            / "include"
            / "exploration_manager"
            / "exploration_manager.h"
        ).read_text()
        manager_source = (
            REPO_ROOT
            / "src"
            / "planner"
            / "exploration_manager"
            / "src"
            / "exploration_manager.cpp"
        ).read_text()

        self.assertIn("double getSemanticValueSafe(const Eigen::Vector2i& idx) const;", manager_header)
        self.assertIn("double ExplorationManager::getSemanticValueSafe(", manager_source)
        self.assertIn("if (!sdf_map_->isInMap(idx))", manager_source)
        self.assertIn("return 0.0;", manager_source)
        self.assertIn("getSemanticValueSafe(idx)", manager_source)
        self.assertIn("getSemanticValueSafe(nbr)", manager_source)
        self.assertNotIn("value_map_->getValue(nbr)", manager_source)
        self.assertIn("if (mean <= 1.0e-9)", manager_source)


if __name__ == "__main__":
    unittest.main()
