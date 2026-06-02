import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class MapRosLatestPoseFallbackTest(unittest.TestCase):
    def test_real_world_mapping_keeps_pose_depth_sync_path(self):
        map_ros_source = (
            REPO_ROOT / "src" / "planner" / "plan_env" / "src" / "map_ros.cpp"
        ).read_text()
        map_ros_header = (
            REPO_ROOT / "src" / "planner" / "plan_env" / "include" / "plan_env" / "map_ros.h"
        ).read_text()
        launch_source = (
            REPO_ROOT
            / "src"
            / "planner"
            / "exploration_manager"
            / "launch"
            / "algorithm_traj.xml"
        ).read_text()

        self.assertNotIn("map_ros/use_latest_pose_on_depth", launch_source)
        self.assertNotIn("map_ros/latest_pose_max_age", launch_source)

        self.assertNotIn(
            "void poseCallback(const nav_msgs::OdometryConstPtr& pose);",
            map_ros_header,
        )
        self.assertNotIn(
            "void depthCallback(const sensor_msgs::ImageConstPtr& img);",
            map_ros_header,
        )
        self.assertNotIn("use_latest_pose_on_depth_", map_ros_header)
        self.assertNotIn("latest_pose_max_age_", map_ros_header)
        self.assertNotIn("latest_depth_msg_", map_ros_header)

        self.assertIn('depth_sub_.reset(', map_ros_source)
        self.assertIn('pose_sub_.reset(', map_ros_source)
        self.assertIn('sync_image_pose_->registerCallback', map_ros_source)
        self.assertNotIn("[MAP_ASYNC]", map_ros_source)
        self.assertIn("[MAP_SYNC]", map_ros_source)


if __name__ == "__main__":
    unittest.main()
