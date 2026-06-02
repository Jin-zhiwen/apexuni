import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class VirtualGroundBufferResetTest(unittest.TestCase):
    def test_virtual_ground_buffer_is_cleared_before_each_frame(self):
        sdf_map_source = (
            REPO_ROOT / "src" / "planner" / "plan_env" / "src" / "sdf_map2d.cpp"
        ).read_text()

        input_virtual_ground = sdf_map_source.split("void SDFMap2D::inputVirtualGround", 1)[1]
        input_virtual_ground = input_virtual_ground.split(
            "void SDFMap2D::inputObjectCloud2D", 1
        )[0]

        clear_stmt = "std::fill(md_->virtual_ground_buffer_.begin(), md_->virtual_ground_buffer_.end(), 0);"
        empty_guard = "if (point_num == 0)"

        self.assertIn(clear_stmt, input_virtual_ground)
        self.assertLess(input_virtual_ground.index(clear_stmt), input_virtual_ground.index(empty_guard))

    def test_map_ros_pushes_current_frame_virtual_ground_once(self):
        map_ros_source = (
            REPO_ROOT / "src" / "planner" / "plan_env" / "src" / "map_ros.cpp"
        ).read_text()

        filter_fn = map_ros_source.split("void MapROS::filterPointCloudToXY()", 1)[1]
        filter_fn = filter_fn.split("bool MapROS::interpolateLineAtZ", 1)[0]

        self.assertEqual(filter_fn.count("map_->inputVirtualGround(under_ground_cloud_2d);"), 1)

    def test_virtual_ground_points_do_not_become_occupied_endpoints(self):
        map_ros_source = (
            REPO_ROOT / "src" / "planner" / "plan_env" / "src" / "map_ros.cpp"
        ).read_text()

        filter_fn = map_ros_source.split("void MapROS::filterPointCloudToXY()", 1)[1]
        filter_fn = filter_fn.split("bool MapROS::interpolateLineAtZ", 1)[0]
        virtual_ground_block = filter_fn.split(
            "if (camera_pitch > object_process_min_pitch_ && !under_ground_cloud_3d->points.empty()) {",
            1,
        )[1]
        virtual_ground_block = virtual_ground_block.split(
            "map_->inputVirtualGround(under_ground_cloud_2d);", 1
        )[0]

        self.assertIn("under_ground_cloud_2d->points.push_back(pt_xy);", virtual_ground_block)
        self.assertNotIn("filtered_depth_cloud2d_->points.push_back(pt_xy);", virtual_ground_block)


if __name__ == "__main__":
    unittest.main()
