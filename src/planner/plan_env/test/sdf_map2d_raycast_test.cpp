#include <gtest/gtest.h>

#include <memory>

#include <ros/ros.h>

#include <plan_env/map_ros.h>
#include <plan_env/raycast2d.h>
#include <plan_env/sdf_map2d.h>

namespace {

using apexnav_planner::SDFMap2D;

class SDFMap2DRaycastTest : public ::testing::Test {
protected:
  static void SetUpTestSuite()
  {
    if (ros::isInitialized())
      return;

    int argc = 1;
    char node_name[] = "sdf_map2d_raycast_test";
    char* argv[] = {node_name, nullptr};
    ros::init(argc, argv, node_name, ros::init_options::AnonymousName |
        ros::init_options::NoSigintHandler);
  }

  void SetUp() override
  {
    nh_.reset(new ros::NodeHandle("~raycast_test"));
    nh_->setParam("is_real_world", true);
    nh_->setParam("sdf_map/ray_mode", 1);
    nh_->setParam("sdf_map/resolution", 0.05);
    nh_->setParam("sdf_map/map_size_x", 10.0);
    nh_->setParam("sdf_map/map_size_y", 10.0);
    nh_->setParam("sdf_map/obstacles_inflation", 0.05);
    nh_->setParam("sdf_map/local_bound", 3.0);
    nh_->setParam("sdf_map/optimistic", false);
    nh_->setParam("sdf_map/signed_dist", false);
    nh_->setParam("sdf_map/p_hit", 0.90);
    nh_->setParam("sdf_map/p_miss", 0.48);
    nh_->setParam("sdf_map/p_min", 0.10);
    nh_->setParam("sdf_map/p_max", 0.98);
    nh_->setParam("sdf_map/p_occ", 0.80);
    nh_->setParam("sdf_map/max_ray_length", 3.0);
    nh_->setParam("sdf_map/ray_stop_dilation", 0);
    nh_->setParam("sdf_map/stop_at_persistent_occupancy", true);
    nh_->setParam("map_ros/skip_pixel", 1);

    map_.reset(new SDFMap2D());
    map_->initMap(*nh_);
  }

  pcl::PointCloud<pcl::PointXY>::Ptr cloudAt(double x, double y) const
  {
    pcl::PointCloud<pcl::PointXY>::Ptr cloud(new pcl::PointCloud<pcl::PointXY>());
    pcl::PointXY point;
    point.x = x;
    point.y = y;
    cloud->push_back(point);
    return cloud;
  }

  pcl::PointCloud<pcl::PointXY>::Ptr emptyCloud() const
  {
    return pcl::PointCloud<pcl::PointXY>::Ptr(new pcl::PointCloud<pcl::PointXY>());
  }

  void input(const pcl::PointCloud<pcl::PointXY>::Ptr& occupied,
      const pcl::PointCloud<pcl::PointXY>::Ptr& rays,
      const pcl::PointCloud<pcl::PointXY>::Ptr& stops)
  {
    std::vector<Eigen::Vector2i> free_grids;
    map_->inputDepthCloud2D(
        occupied, rays, stops, Eigen::Vector3d(0.025, 0.025, 0.41), free_grids);
  }

  std::unique_ptr<ros::NodeHandle> nh_;
  std::unique_ptr<SDFMap2D> map_;
};

TEST_F(SDFMap2DRaycastTest, CurrentObstacleCandidateStopsNeighboringFreeRay)
{
  input(emptyCloud(), cloudAt(2.025, 0.025), cloudAt(1.025, 0.025));

  EXPECT_EQ(map_->getOccupancy(Eigen::Vector2d(0.525, 0.025)), SDFMap2D::FREE);
  EXPECT_EQ(map_->getOccupancy(Eigen::Vector2d(1.025, 0.025)), SDFMap2D::UNKNOWN);
  EXPECT_EQ(map_->getOccupancy(Eigen::Vector2d(1.525, 0.025)), SDFMap2D::UNKNOWN);
}

TEST_F(SDFMap2DRaycastTest, PersistentObstacleGetsMissButBlocksSpaceBehindIt)
{
  map_->setForceOccGrid(Eigen::Vector2d(1.025, 0.025));
  input(emptyCloud(), cloudAt(2.025, 0.025), emptyCloud());

  EXPECT_EQ(map_->getOccupancy(Eigen::Vector2d(1.025, 0.025)), SDFMap2D::OCCUPIED);
  EXPECT_EQ(map_->getOccupancy(Eigen::Vector2d(1.525, 0.025)), SDFMap2D::UNKNOWN);
}

TEST_F(SDFMap2DRaycastTest, FilteredObstacleEndpointStillBecomesOccupied)
{
  const auto obstacle = cloudAt(1.025, 0.025);
  input(obstacle, obstacle, obstacle);

  EXPECT_EQ(map_->getOccupancy(Eigen::Vector2d(1.025, 0.025)), SDFMap2D::OCCUPIED);
}

}  // namespace
