#ifndef _EXPLORATION_MANAGER_H_
#define _EXPLORATION_MANAGER_H_

// Third-party libraries
#include <Eigen/Eigen>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

// Standard C++ libraries
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

// ROS core
#include <ros/ros.h>

// Plan environment
#include <plan_env/frontier_map2d.h>
#include <plan_env/object_map2d.h>
#include <plan_env/sdf_map2d.h>
#include <plan_env/value_map2d.h>

// Path searching
#include <path_searching/astar2d.h>

using Eigen::Vector2d;
using Eigen::Vector3d;
using std::shared_ptr;
using std::unique_ptr;
using std::vector;

namespace apexnav_planner {
class SDFMap2D;
class FrontierMap2D;
class Gcopter;
class KinoAstar;
struct LocalTrajectory;
struct ExplorationParam;
struct ExplorationData;

struct SemanticFrontier {
  Vector2d position;      ///< 2D position of the frontier
  double semantic_value;  ///< Semantic value at the frontier location
  double path_length;     ///< Path length to reach this frontier
  vector<Vector2d> path;  ///< Complete path to the frontier

  bool operator<(const SemanticFrontier& other) const
  {
    if (fabs(semantic_value - other.semantic_value) < 1e-4) {
      // If semantic values are equal, sort by path length (ascending)
      return path_length < other.path_length;
    }
    // Otherwise, sort by semantic value (descending)
    return semantic_value > other.semantic_value;
  }
};

enum EXPL_RESULT {
  EXPLORATION,               ///< Normal exploration mode
  SEARCH_BEST_OBJECT,        ///< Found high-confidence object
  SEARCH_OVER_DEPTH_OBJECT,  ///< Searching over-depth object
  SEARCH_SUSPICIOUS_OBJECT,  ///< Investigating suspicious object
  NO_PASSABLE_FRONTIER,      ///< No reachable frontiers available
  NO_COVERABLE_FRONTIER,     ///< No coverable frontiers found
  SEARCH_EXTREME             ///< Extreme search mode activated
};

class ExplorationManager {
public:
  ExplorationManager() = default;
  ~ExplorationManager();  // Explicit destructor declaration for shared_ptr with forward declaration

  void initialize(ros::NodeHandle& nh);

  int planNextBestPoint(const Vector3d& pos, const double& yaw);
  bool replanPathToGoal(const Vector2d& start, const Vector2d& goal,
      Vector2d& refined_goal, vector<Vector2d>& replanned_path);
  void setSkipObjectNavigationOnce(bool skip = true);
  bool planTrajectory(const Eigen::VectorXd& start, const Eigen::VectorXd& end, const Vector3d& ctrl);
  void getSortedSemanticFrontiers(const Vector2d& cur_pos, const vector<Vector2d>& frontiers,
      vector<SemanticFrontier>& sem_frontiers);
  void calcSemanticFrontierInfo(const vector<SemanticFrontier>& sem_frontiers, double& std_dev,
      double& max_to_mean, double& mean, bool if_print = false);

  shared_ptr<ExplorationData> ed_;            ///< Exploration data container
  shared_ptr<ExplorationParam> ep_;           ///< Exploration parameters
  unique_ptr<Astar2D> path_finder_;           ///< A* path finding algorithm
  shared_ptr<FrontierMap2D> frontier_map2d_;  ///< 2D frontier map
  shared_ptr<ObjectMap2D> object_map2d_;      ///< 2D object map
  shared_ptr<SDFMap2D> sdf_map_;              ///< Signed distance field map
  shared_ptr<Gcopter> gcopter_;               ///< Trajectory optimizer (for real-world)
  shared_ptr<KinoAstar> kinoastar_;           ///< Kinodynamic A* planner (for real-world)

  typedef shared_ptr<ExplorationManager> Ptr;

private:
  // Exploration Policy
  void chooseExplorationPolicy(Vector2d cur_pos, vector<Vector2d> frontiers,
      Vector2d& next_best_pos, vector<Vector2d>& next_best_path);
  void findClosestFrontierPolicy(Vector2d cur_pos, vector<Vector2d> frontiers,
      Vector2d& next_best_pos, vector<Vector2d>& next_best_path);
  void findHighestSemanticsFrontierPolicy(Vector2d cur_pos, vector<Vector2d> frontiers,
      Vector2d& next_best_pos, vector<Vector2d>& next_best_path);
  void hybridExplorePolicy(Vector2d cur_pos, vector<Vector2d> frontiers, Vector2d& next_best_pos,
      vector<Vector2d>& next_best_path);
  void findTSPTourPolicy(Vector2d cur_pos, vector<Vector2d> frontiers, Vector2d& next_best_pos,
      vector<Vector2d>& next_best_path);

  // Path Search Utils
  bool searchObjectPath(const Vector3d& start,
      const pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& object_cloud,
      Eigen::Vector2d& refined_pos, std::vector<Eigen::Vector2d>& refined_path);
  bool searchObjectPathExtreme(const Vector3d& start,
      const pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& object_cloud,
      Eigen::Vector2d& refined_pos, std::vector<Eigen::Vector2d>& refined_path);
  bool searchFrontierPath(const Vector2d& start, const Vector2d& end, Eigen::Vector2d& refined_pos,
      std::vector<Eigen::Vector2d>& refined_path);
  void shortenPath(vector<Vector2d>& path);
  bool isTrajectoryCollisionFree(const LocalTrajectory& local_traj) const;
  void publishFootprintCollisionMarker(
      const Eigen::Vector2d& collision_pos,
      double collision_time) const;

  // Helper functions for object path searching
  Vector2d findNearestObjectPoint(
      const Vector3d& start, const pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& object_cloud);
  bool trySearchObjectPathWithDistance(const Vector2d& start2d, const Vector2d& object_pose,
      double distance, double max_search_time, Eigen::Vector2d& refined_pos,
      std::vector<Eigen::Vector2d>& refined_path, const std::string& debug_msg);

  // TSP Optimization Methods
  void computeATSPTour(
      const Vector2d& cur_pos, const vector<Vector2d>& frontiers, vector<int>& indices);
  void computeATSPCostMatrix(
      const Vector2d& cur_pos, const vector<Vector2d>& frontiers, Eigen::MatrixXd& cost_matrix);
  double computePathCost(const Vector2d& pos1, const Vector2d& pos2);
  double getSemanticValueSafe(const Eigen::Vector2i& idx) const;
  vector<Vector2i> allNeighbors(const Eigen::Vector2i& idx, int grid_radius);

  ros::ServiceClient tsp_client_;         ///< ROS service client for TSP solver
  ros::Publisher footprint_collision_marker_pub_;
  unique_ptr<RayCaster2D> ray_caster2d_;  ///< Ray casting for collision checking
  std::string collision_marker_frame_id_ = "odom";
  double path_shortcut_max_segment_ = 0.75;
  // Must match the FSM lookahead check so policy selection does not accept a
  // point-path that the rectangular Go2 footprint will reject immediately.
  double local_target_distance_ = 1.50;
  double local_target_min_distance_ = 0.30;
  bool skip_object_navigation_once_ = false;
};

inline bool ExplorationManager::searchObjectPathExtreme(const Vector3d& start,
    const pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& object_cloud,
    Eigen::Vector2d& refined_pos, std::vector<Eigen::Vector2d>& refined_path)
{
  Vector2d object_pose = findNearestObjectPoint(start, object_cloud);
  if (object_pose.x() < -999.0)
    return false;  // Error finding nearest point

  Vector2d start2d = Vector2d(start(0), start(1));
  path_finder_->reset();
  if (path_finder_->astarSearch(start2d, object_pose, 0.25, 0.2, Astar2D::SAFETY_MODE::EXTREME) ==
      Astar2D::REACH_END) {
    refined_pos = object_pose;
    refined_path = path_finder_->getPath();
    return true;
  }
  return false;
}

inline void ExplorationManager::shortenPath(vector<Vector2d>& path)
{
  if (path.size() < 3) {
    return;
  }

  // Preserve the same collision model used by A*. The previous point-map ray
  // cast could replace a valid detour with a long line that a Go2 body cannot
  // traverse. Keeping short segments also guarantees a local target exists.
  vector<Vector2d> shortened;
  shortened.reserve(path.size());
  shortened.push_back(path.front());
  size_t anchor = 0;
  while (anchor + 1 < path.size()) {
    size_t furthest_safe = anchor + 1;
    for (size_t candidate = anchor + 1; candidate < path.size(); ++candidate) {
      if ((path[candidate] - path[anchor]).norm() > path_shortcut_max_segment_ + 1.0e-6)
        continue;
      if (path_finder_->isSegmentSafe(path[anchor], path[candidate]))
        furthest_safe = candidate;
    }
    shortened.push_back(path[furthest_safe]);
    anchor = furthest_safe;
  }
  path.swap(shortened);
}

inline vector<Eigen::Vector2i> ExplorationManager::allNeighbors(
    const Eigen::Vector2i& idx, int grid_radius)
{
  vector<Eigen::Vector2i> neighbors;

  for (int x = -grid_radius; x <= grid_radius; ++x) {
    for (int y = -grid_radius; y <= grid_radius; ++y) {
      if (x == 0 && y == 0)
        continue;  // Skip center point
      Eigen::Vector2i offset(x, y);
      neighbors.push_back(idx + offset);
    }
  }
  return neighbors;
}

}  // namespace apexnav_planner

#endif
