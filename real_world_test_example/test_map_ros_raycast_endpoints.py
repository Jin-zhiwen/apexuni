import re
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class MapRosRaycastEndpointsTest(unittest.TestCase):
    def test_mapping_uses_all_depth_points_for_free_space_raycasting(self):
        map_ros_header = (
            REPO_ROOT / "src" / "planner" / "plan_env" / "include" / "plan_env" / "map_ros.h"
        ).read_text()
        map_ros_source = (
            REPO_ROOT / "src" / "planner" / "plan_env" / "src" / "map_ros.cpp"
        ).read_text()
        sdf_map_header = (
            REPO_ROOT / "src" / "planner" / "plan_env" / "include" / "plan_env" / "sdf_map2d.h"
        ).read_text()

        self.assertIn("raycast_depth_cloud2d_", map_ros_header)
        self.assertIn("raycast_depth_cloud2d_->clear()", map_ros_source)
        self.assertRegex(
            map_ros_source,
            re.compile(
                r"map_->inputDepthCloud2D\(\s*filtered_depth_cloud2d_,\s*"
                r"raycast_depth_cloud2d_,\s*camera_pos_,\s*free_grids,\s*"
                r"&semantic_free_grids\s*\)",
                re.MULTILINE,
            ),
        )
        self.assertIn(
            "void inputDepthCloud2D(const pcl::PointCloud<pcl::PointXY>::Ptr& occupied_points,",
            sdf_map_header,
        )
        self.assertIn(
            "const pcl::PointCloud<pcl::PointXY>::Ptr& raycast_points",
            sdf_map_header,
        )

    def test_value_map_can_use_full_view_free_grids_for_smooth_semantic_sector(self):
        map_ros_source = (
            REPO_ROOT / "src" / "planner" / "plan_env" / "src" / "map_ros.cpp"
        ).read_text()
        map_ros_header = (
            REPO_ROOT / "src" / "planner" / "plan_env" / "include" / "plan_env" / "map_ros.h"
        ).read_text()
        sdf_map_header = (
            REPO_ROOT / "src" / "planner" / "plan_env" / "include" / "plan_env" / "sdf_map2d.h"
        ).read_text()
        sdf_map_source = (
            REPO_ROOT / "src" / "planner" / "plan_env" / "src" / "sdf_map2d.cpp"
        ).read_text()
        launch_source = (
            REPO_ROOT
            / "src"
            / "planner"
            / "exploration_manager"
            / "launch"
            / "algorithm_traj.xml"
        ).read_text()

        self.assertIn("vector<Eigen::Vector2i> semantic_free_grids;", map_ros_source)
        self.assertIn("semantic_use_raycast_free_grids_", map_ros_header)
        self.assertIn("semantic/use_raycast_free_grids", map_ros_source)
        self.assertIn('name="semantic/use_raycast_free_grids" value="true"', launch_source)
        input_idx = map_ros_source.index("map_->inputDepthCloud2D(")
        dilate_idx = map_ros_source.index("dilateGrids(free_grids, 1);", input_idx)
        self.assertLess(input_idx, dilate_idx)
        self.assertRegex(
            map_ros_source,
            re.compile(
                r"inputDepthCloud2D\(\s*filtered_depth_cloud2d_,\s*"
                r"raycast_depth_cloud2d_,\s*camera_pos_,\s*free_grids,\s*"
                r"&semantic_free_grids\s*\)",
                re.MULTILINE,
            ),
        )
        self.assertRegex(
            map_ros_source,
            re.compile(
                r"semantic_value_grids\s*=\s*semantic_use_raycast_free_grids_\s*\?\s*"
                r"free_grids\s*:\s*semantic_free_grids",
                re.MULTILINE,
            ),
        )
        self.assertRegex(
            map_ros_source,
            re.compile(
                r"updateValueMap\(\s*camera_pos,\s*camera_yaw,\s*semantic_value_grids,\s*"
                r"itm_score_\s*\)",
                re.MULTILINE,
            ),
        )
        self.assertIn(
            "vector<Eigen::Vector2i>* semantic_free_grids",
            sdf_map_header,
        )
        self.assertIn("semantic_free_grids->clear();", sdf_map_source)
        self.assertIn("Semantic obstacle rays use obstacle-height endpoints", sdf_map_source)

    def test_raycast_only_endpoints_do_not_create_occupied_hits(self):
        sdf_map_source = (
            REPO_ROOT / "src" / "planner" / "plan_env" / "src" / "sdf_map2d.cpp"
        ).read_text()

        input_depth = sdf_map_source.split("void SDFMap2D::inputDepthCloud2D", 1)[1]
        input_depth = input_depth.split("void SDFMap2D::setForceOccGrid", 1)[0]
        raycast_pass = input_depth.split("for (int i = 0; i < raycast_point_num; ++i)", 1)[1]

        self.assertNotIn("setCacheOccupancy(vox_adr, tmp_flag);", raycast_pass)
        self.assertIn("tmp_flag == 1 && flag_occ.count(vox_adr)", raycast_pass)
        self.assertIn("setCacheOccupancy(vox_adr, 1)", raycast_pass)

    def test_real_world_raycast_stops_at_current_frame_obstacles(self):
        launch_source = (
            REPO_ROOT
            / "src"
            / "planner"
            / "exploration_manager"
            / "launch"
            / "algorithm_traj.xml"
        ).read_text()
        sdf_map_source = (
            REPO_ROOT / "src" / "planner" / "plan_env" / "src" / "sdf_map2d.cpp"
        ).read_text()

        self.assertIn('name="sdf_map/ray_mode" value="1"', launch_source)
        self.assertIn('name="sdf_map/ray_stop_dilation" value="1"', launch_source)

        input_depth = sdf_map_source.split("void SDFMap2D::inputDepthCloud2D", 1)[1]
        input_depth = input_depth.split("void SDFMap2D::setForceOccGrid", 1)[0]
        self.assertNotIn("Skip if marked as occupied", input_depth)
        self.assertIn("Stop if hit occupied grid", input_depth)
        self.assertIn("flag_occ_stop", input_depth)
        self.assertIn("if (flag_occ_stop.count(adr) && flag_occ_stop[adr] == 1)", input_depth)

    def test_dilated_stop_cells_are_cleared_before_breaking(self):
        sdf_map_source = (
            REPO_ROOT / "src" / "planner" / "plan_env" / "src" / "sdf_map2d.cpp"
        ).read_text()

        input_depth = sdf_map_source.split("void SDFMap2D::inputDepthCloud2D", 1)[1]
        input_depth = input_depth.split("void SDFMap2D::setForceOccGrid", 1)[0]
        ray_mode_one = input_depth.split("// Ray mode 1: Cast from sensor to point", 1)[1]
        stop_block = ray_mode_one.split("if (md_->virtual_ground_buffer_[adr])", 1)[0]

        self.assertIn("if (flag_occ.count(adr) && flag_occ[adr] == 1)", stop_block)
        self.assertIn("setCacheOccupancy(adr, 0);", stop_block)
        self.assertLess(
            stop_block.index("if (flag_occ.count(adr) && flag_occ[adr] == 1)"),
            stop_block.index("setCacheOccupancy(adr, 0);"),
        )

    def test_raycast_frame_stamps_do_not_wrap_after_short_runs(self):
        sdf_map_header = (
            REPO_ROOT / "src" / "planner" / "plan_env" / "include" / "plan_env" / "sdf_map2d.h"
        ).read_text()
        sdf_map_source = (
            REPO_ROOT / "src" / "planner" / "plan_env" / "src" / "sdf_map2d.cpp"
        ).read_text()

        self.assertIn("vector<int> flag_rayend_", sdf_map_header)
        self.assertIn("int raycast_num_", sdf_map_header)
        self.assertIn("md_->flag_rayend_ = vector<int>(mp_->buffer_size_, -1);", sdf_map_source)
        self.assertNotIn("vector<char> flag_rayend_", sdf_map_header)
        self.assertNotIn("char raycast_num_", sdf_map_header)


if __name__ == "__main__":
    unittest.main()
