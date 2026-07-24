#ifndef _FAST_EXPLORATION_FSM_H_
#define _FAST_EXPLORATION_FSM_H_

// Third-party libraries
#include <Eigen/Eigen>

// Standard C++ libraries
#include <memory>
#include <string>
#include <vector>

// ROS core
#include <ros/ros.h>

// ROS message types
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>
#include <plan_env/MultipleMasksWithConfidence.h>
#include <std_msgs/Float64.h>
#include <std_msgs/Int32.h>
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/String.h>
#include <visualization_msgs/Marker.h>

using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::Vector4d;
using std::shared_ptr;
using std::string;
using std::unique_ptr;
using std::vector;

namespace apexnav_planner {
// Centralized constants for ExplorationFSM (mirrors the style of FSMConstants in fsm2.h)
namespace FSMConstants {
// Timers (s)
constexpr double EXEC_TIMER_DURATION = 0.01;
constexpr double FRONTIER_TIMER_DURATION = 0.25;

// Robot Action
constexpr double ACTION_DISTANCE = 0.25;
constexpr double ACTION_ANGLE = M_PI / 6.0;
constexpr double MAST3R_FINE_YAW_ACTION_ANGLE = M_PI / 12.0;
constexpr double MAST3R_FINE_YAW_HALF_ANGLE = MAST3R_FINE_YAW_ACTION_ANGLE / 2.0;
constexpr double OBJECT_VIEWPOINT_YAW_TOLERANCE = ACTION_ANGLE / 2.0 + M_PI / 90.0;

// Distances (m)
constexpr double STUCKING_DISTANCE = 0.05;       // consider stuck if movement < this
constexpr double REACH_DISTANCE = 0.20;          // reach object distance
constexpr double SOFT_REACH_DISTANCE = 0.45;     // soft reach distance for object
constexpr double LOCAL_DISTANCE = 0.80;          // local target lookahead
constexpr double FORWARD_DISTANCE = 0.15;        // min clearance for marking obstacles
constexpr double FORCE_DORMANT_DISTANCE = 0.35;  // force dormant frontier if very close
constexpr double MIN_SAFE_DISTANCE = 0.15;       // min safe distance to obstacles

// Counters / thresholds
constexpr int MAX_STUCKING_COUNT = 25;           // max consecutive stuck actions -> stop
constexpr int MAX_STUCKING_NEXT_POS_COUNT = 14;  // times next_pos unchanged while stuck
constexpr int MAST3R_LOCAL_GOAL_MAX_AGE = 4;     // fallback guard before an A* path is locked
constexpr int MAST3R_LOCAL_GOAL_MAX_ACTIVE_AGE = 48;
constexpr int MAST3R_LOCAL_GOAL_FORWARD_FAILURES_BEFORE_REPLAN = 2;
constexpr int MAST3R_LOCAL_GOAL_MAX_REPLANS = 4;
constexpr double MAST3R_EXECUTION_HORIZON_DISTANCE = 0.75;
constexpr int OBJECT_VIEWPOINT_FORWARD_FAILURES_BEFORE_ADVANCE = 2;
constexpr int OBJECT_VIEWPOINT_MAX_ALIGNMENT_STEPS = 8;
constexpr double MAST3R_LOCAL_GOAL_REACH_DISTANCE = ACTION_DISTANCE;
constexpr double INSTANCE_REJECT_RADIUS = 0.75;
constexpr double INSINAV_SEMANTIC_GATE_THRESHOLD = 0.50;

// Cost weights
constexpr double TARGET_WEIGHT = 150.0;
constexpr double TARGET_CLOSE_WEIGHT_1 = 2000.0;  // penalize moving away
constexpr double TARGET_CLOSE_WEIGHT_2 = 200.0;   // encourage moving closer
constexpr double SAFETY_WEIGHT = 1.0;
constexpr double SAMPLE_NUM = 10.0;  // samples along a step for safety cost

// Visualization / robot marker
constexpr double VIS_SCALE_FACTOR = 1.8;  // multiply by map resolution
constexpr double ROBOT_HEIGHT = 0.15;
constexpr double ROBOT_RADIUS = 0.18;
}  // namespace FSMConstants

class FastPlannerManager;
class ExplorationManager;
class PlanningVisualization;
struct FSMParam;
struct FSMData;

enum ROS_STATE { INIT, WAIT_TRIGGER, PLAN_ACTION, WAIT_ACTION_FINISH, PUB_ACTION, FINISH };
enum ACTION { STOP, MOVE_FORWARD, TURN_LEFT, TURN_RIGHT, TURN_DOWN, TURN_UP };
enum HABITAT_STATE { READY, ACTION_EXEC, ACTION_FINISH, EPISODE_FINISH };
class ExplorationFSM {
private:
  /* Planning Utils */
  ros::NodeHandle nh_;
  shared_ptr<FastPlannerManager> planner_manager_;
  shared_ptr<ExplorationManager> expl_manager_;
  shared_ptr<PlanningVisualization> visualization_;

  shared_ptr<FSMParam> fp_;
  shared_ptr<FSMData> fd_;
  ROS_STATE state_;

  /* ROS Utils */
  ros::NodeHandle node_;
  ros::Timer exec_timer_, vis_timer_, frontier_timer_;
  ros::Subscriber trigger_sub_, odom_sub_, habitat_state_sub_, confidence_threshold_sub_;
  ros::Subscriber mast3r_hint_sub_;
  ros::Subscriber instance_stop_gate_sub_;
  ros::Subscriber verified_approach_target_sub_;
  ros::Subscriber resume_exploration_sub_;
  ros::Publisher action_pub_, ros_state_pub_, expl_state_pub_, expl_result_pub_;
  ros::Publisher mast3r_refine_status_pub_;
  ros::Publisher mast3r_debug_pub_;
  ros::Publisher robot_marker_pub_;

  /* Action Planner */
  int callActionPlanner();
  int planNextBestAction(Vector2d current_pos, double current_yaw, const vector<Vector2d>& path,
      bool need_safety = true);
  Vector2d selectLocalTarget(
      const Vector2d& current_pos, const vector<Vector2d>& path, const double& local_distance);
  int decideNextAction(double current_yaw, double target_yaw);
  bool alignMast3rTargetYaw(double current_yaw);
  bool planMast3rPath(const Vector2d& current_pos, const Vector2d& target_pos, bool is_replan);
  void markMast3rForwardCollision(const Vector2d& current_pos, double current_yaw);
  int planMast3rPathAction(
      const Vector2d& current_pos, double current_yaw, const vector<Vector2d>& path);
  Vector2d computeBestStep(
      const Vector2d& current_pos, double current_yaw, const Vector2d& target_pos);
  bool buildMast3rLocalTarget(
      const Vector2d& current_pos, double current_yaw, Vector2d& target_pos, double& target_yaw);
  void clearMast3rLocalGoal();
  void publishMast3rRefineStatus(int status);
  double computeActionSafetyCost(const Vector2d& current_pos, const Vector2d& step);
  double computeActionTotalCost(const Vector2d& current_pos, double current_yaw,
      const Vector2d& target_pos, const Vector2d& step);

  /* Helper functions */
  bool updateFrontierAndObject(bool enable_dormant = true);
  void transitState(ROS_STATE new_state, string pos_call);
  void wrapAngle(double& angle);
  void publishRobotMarker();
  void visualize();
  void clearVisMarker();

  /* ROS callbacks */
  void FSMCallback(const ros::TimerEvent& e);
  void frontierCallback(const ros::TimerEvent& e);
  void triggerCallback(const geometry_msgs::PoseStampedConstPtr& msg);
  void odometryCallback(const nav_msgs::OdometryConstPtr& msg);
  void habitatStateCallback(const std_msgs::Int32ConstPtr& msg);
  void confidenceThresholdCallback(const std_msgs::Float64ConstPtr& msg);
  void mast3rHintCallback(const std_msgs::Float32MultiArrayConstPtr& msg);
  void instanceStopGateCallback(const std_msgs::Int32ConstPtr& msg);
  void verifiedApproachTargetCallback(
      const plan_env::MultipleMasksWithConfidenceConstPtr& msg);
  void resumeExplorationCallback(const std_msgs::Int32ConstPtr& msg);

public:
  ExplorationFSM() = default;
  ~ExplorationFSM() = default;

  void init(ros::NodeHandle& nh);

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

inline void ExplorationFSM::wrapAngle(double& angle)
{
  while (angle < -M_PI) angle += 2 * M_PI;
  while (angle > M_PI) angle -= 2 * M_PI;
}
}  // namespace apexnav_planner

#endif
