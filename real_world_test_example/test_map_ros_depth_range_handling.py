import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class MapRosDepthRangeHandlingTest(unittest.TestCase):
    def test_over_max_depth_is_not_clamped_into_an_occupied_endpoint(self):
        map_ros_source = (
            REPO_ROOT / "src" / "planner" / "plan_env" / "src" / "map_ros.cpp"
        ).read_text()
        sdf_map_source = (
            REPO_ROOT / "src" / "planner" / "plan_env" / "src" / "sdf_map2d.cpp"
        ).read_text()

        process_depth_image = map_ros_source.split("void MapROS::processDepthImage()", 1)[1]
        process_depth_image = process_depth_image.split("void MapROS::getObservationObjectsCloud", 1)[
            0
        ]

        self.assertNotIn("depth = depth_filter_maxdist_", process_depth_image)
        self.assertIn("std::isfinite(depth)", process_depth_image)
        self.assertIn("depth < depth_filter_mindist_", process_depth_image)

        self.assertIn("if (length > mp_->max_ray_length_)", sdf_map_source)
        self.assertIn("tmp_flag = 0;", sdf_map_source)


if __name__ == "__main__":
    unittest.main()
