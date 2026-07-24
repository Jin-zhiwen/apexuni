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
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

// ROS core
#include <ros/ros.h>
#include <std_msgs/String.h>

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
  bool planTrajectory(const Eigen::VectorXd& start, const Eigen::VectorXd& end, const Vector3d& ctrl);
  bool hasLockedObjectViewpoint() const;
  Vector2d getLockedObjectViewpoint() const;
  Vector2d getLockedObjectCenter() const;
  double getObjectViewpointReachDistance() const;
  bool isVerifiedApproachActive() const;
  void setVerifiedApproachState(int state);
  void setVerifiedApproachCloud(
      const pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& object_cloud);
  // Finish the current DINO inspection view and activate the next precomputed view, if any.
  // The task remains a single active endpoint at all times.
  bool advanceLockedObjectViewpoint(const Vector3d& pos, const std::string& reason);
  void releaseObjectViewpointLock(bool reject_candidate, const std::string& reason);
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
  struct ObjectViewpointPlan {
    Vector2d viewpoint = Vector2d::Zero();
    Vector2d aim_point = Vector2d::Zero();
    vector<Vector2d> path;
    double score = -std::numeric_limits<double>::infinity();
    double visible_ratio = 0.0;
    double fov_deg = 0.0;
    double standoff = 0.0;
    double azimuth = 0.0;
    int generated_candidates = 0;
  };

  struct ObjectViewpointSearchStats {
    int cloud_points = 0;
    int generated = 0;
    int out_of_map = 0;
    int non_free = 0;
    int inflated = 0;
    int safe = 0;
    int ray_samples = 0;
    int ray_unknown = 0;
    int ray_known_blocked = 0;
    int visibility_rejected = 0;
    int visible = 0;
    int astar_attempted = 0;
    int astar_reached = 0;
    int short_path = 0;
    int selected = 0;
  };

  enum class ObjectInspectionGate { ELIGIBLE, COOLDOWN, EXHAUSTED };

  struct ObjectInspectionFailure {
    vector<int> object_ids;
    int label = -1;
    Vector2d center = Vector2d::Zero();
    int failure_count = 0;
    double best_failed_score = -std::numeric_limits<double>::infinity();
    unsigned long cooldown_until_sequence = 0;
  };

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
  int planSemanticObjectOrFrontier(const Vector3d& pos, const double& yaw);
  bool planVerifiedApproach(const Vector3d& pos);
  double getFrontierBaseValue(const Vector2d& frontier_pos);
  bool continueLockedObjectViewpoint(const Vector3d& pos);
  bool activateLockedObjectViewpoint(const Vector3d& pos, size_t index,
      const std::string& reason);
  bool isObjectInspectionCompleted(const SemanticObjectCandidate& candidate) const;
  void markObjectInspectionCompleted(int object_id, int label, unsigned long evidence_epoch);
  ObjectInspectionFailure* findObjectInspectionFailure(
      int object_id, int label, const Vector2d& center);
  const ObjectInspectionFailure* findObjectInspectionFailure(
      int object_id, int label, const Vector2d& center) const;
  ObjectInspectionGate getObjectInspectionGate(const SemanticObjectCandidate& candidate,
      unsigned long semantic_sequence, bool* stronger_retry = nullptr) const;
  void recordObjectInspectionFailure(int object_id, int label, unsigned long evidence_epoch,
      const Vector2d& center, double score, const std::string& reason);
  bool prepareObjectViewpointPlans(const Vector3d& pos,
      const SemanticObjectCandidate& candidate, vector<ObjectViewpointPlan>& plans);
  void lockObjectViewpoint(
      const SemanticObjectCandidate& candidate, const vector<ObjectViewpointPlan>& plans);
  void publishObjectViewpointDebug(const std::string& text);

  // Path Search Utils
  bool searchObjectPath(const Vector3d& start,
      const pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& object_cloud,
      Eigen::Vector2d& refined_pos, std::vector<Eigen::Vector2d>& refined_path);
  bool searchObjectViewpointPaths(const Vector3d& start, const SemanticObjectCandidate& candidate,
      vector<ObjectViewpointPlan>& plans, ObjectViewpointSearchStats* stats);
  bool searchObjectPathExtreme(const Vector3d& start,
      const pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& object_cloud,
      Eigen::Vector2d& refined_pos, std::vector<Eigen::Vector2d>& refined_path);
  bool searchFrontierPath(const Vector2d& start, const Vector2d& end, Eigen::Vector2d& refined_pos,
      std::vector<Eigen::Vector2d>& refined_path);
  void shortenPath(vector<Vector2d>& path);

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
  vector<Vector2i> allNeighbors(const Eigen::Vector2i& idx, int grid_radius);

  ros::ServiceClient tsp_client_;         ///< ROS service client for TSP solver
  ros::Publisher object_viewpoint_debug_pub_;
  unique_ptr<RayCaster2D> ray_caster2d_;  ///< Ray casting for collision checking

  bool verified_approach_active_ = false;
  bool verified_approach_target_locked_ = false;
  bool verified_approach_target_failed_ = false;
  Vector2d verified_approach_target_ = Vector2d::Zero();
  pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> verified_approach_cloud_;
  bool object_viewpoint_locked_ = false;
  int locked_object_id_ = -1;
  int locked_object_label_ = -1;
  int object_viewpoint_lock_steps_ = 0;
  size_t locked_object_viewpoint_index_ = 0;
  unsigned long locked_object_evidence_epoch_ = 0;
  double locked_object_score_ = -1.0;
  Vector2d locked_object_center_ = Vector2d::Zero();
  Vector2d locked_object_viewpoint_ = Vector2d::Zero();
  vector<ObjectViewpointPlan> locked_object_viewpoint_plans_;
  std::map<std::pair<int, int>, unsigned long> completed_object_evidence_epochs_;
  vector<ObjectInspectionFailure> object_inspection_failures_;
};

inline bool ExplorationManager::searchFrontierPath(const Vector2d& start, const Vector2d& end,
    Eigen::Vector2d& refined_pos, std::vector<Eigen::Vector2d>& refined_path)
{
  path_finder_->reset();
  if (path_finder_->astarSearch(start, end, 0.25, 0.01) == Astar2D::REACH_END) {
    refined_pos = end;
    refined_path = path_finder_->getPath();
    return true;
  }
  return false;
}

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
  if (path.empty()) {
    ROS_ERROR("Empty path to shorten");
    return;
  }

  // Shorten the path by keeping only critical intermediate points
  const double dist_thresh = 3.0;  // Minimum distance threshold for waypoint retention
  vector<Vector2d> short_tour = { path.front() };

  for (int i = 1; i < (int)path.size() - 1; ++i) {
    if ((path[i] - short_tour.back()).norm() > dist_thresh)
      short_tour.push_back(path[i]);
    else {
      // Add waypoints only when necessary to avoid collision
      ray_caster2d_->input(short_tour.back(), path[i + 1]);
      Eigen::Vector2i idx;
      while (ray_caster2d_->nextId(idx) && ros::ok()) {
        if (sdf_map_->getInflateOccupancy(idx) == 1 ||
            sdf_map_->getOccupancy(idx) == SDFMap2D::UNKNOWN) {
          short_tour.push_back(path[i]);
          break;
        }
      }
    }
  }

  // Always include the final destination
  if ((path.back() - short_tour.back()).norm() > 1e-3)
    short_tour.push_back(path.back());

  // Ensure minimum path complexity (at least three points)
  if (short_tour.size() == 2)
    short_tour.insert(short_tour.begin() + 1, 0.5 * (short_tour[0] + short_tour[1]));

  path = short_tour;
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
