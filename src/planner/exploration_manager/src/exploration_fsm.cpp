
#include <exploration_manager/exploration_manager.h>
#include <exploration_manager/exploration_fsm.h>
#include <exploration_manager/exploration_data.h>
#include <vis_utils/planning_visualization.h>
#include <pcl_conversions/pcl_conversions.h>

#include <iomanip>
#include <sstream>

namespace apexnav_planner {
namespace {
enum StrictStepState {
  STRICT_STEP_SAFE = 0,
  STRICT_STEP_OUT_OF_MAP,
  STRICT_STEP_UNKNOWN,
  STRICT_STEP_OCCUPIED,
  STRICT_STEP_INFLATED
};

struct MapPointDebug {
  bool in_map = false;
  Eigen::Vector2i index = Eigen::Vector2i::Zero();
  int occupancy = -1;
  int inflated = -1;
  double esdf = -1.0;
};

struct StrictStepDebug {
  int safe = 0;
  int out_of_map = 0;
  int unknown = 0;
  int occupied = 0;
  int inflated = 0;
};

const char* occupancyStateName(int occupancy)
{
  if (occupancy == SDFMap2D::UNKNOWN)
    return "unknown";
  if (occupancy == SDFMap2D::FREE)
    return "free";
  if (occupancy == SDFMap2D::OCCUPIED)
    return "occupied";
  return "invalid";
}

MapPointDebug inspectMapPoint(const SDFMap2D::Ptr& map, const Vector2d& position)
{
  MapPointDebug debug;
  map->posToIndex(position, debug.index);
  debug.in_map = map->isInMap(position);
  if (debug.in_map) {
    debug.occupancy = map->getOccupancy(position);
    debug.inflated = map->getInflateOccupancy(position);
    debug.esdf = map->getDistance(position);
  }
  return debug;
}

StrictStepState classifyStrictPoint(const SDFMap2D::Ptr& map, const Vector2d& position)
{
  if (!map->isInMap(position))
    return STRICT_STEP_OUT_OF_MAP;
  if (map->getInflateOccupancy(position) == 1)
    return STRICT_STEP_INFLATED;

  const int occupancy = map->getOccupancy(position);
  if (occupancy == SDFMap2D::OCCUPIED)
    return STRICT_STEP_OCCUPIED;
  if (occupancy == SDFMap2D::UNKNOWN)
    return STRICT_STEP_UNKNOWN;
  return STRICT_STEP_SAFE;
}

StrictStepState classifyStrictStep(
    const SDFMap2D::Ptr& map, const Vector2d& start, const Vector2d& step)
{
  const Vector2d end = start + step;
  StrictStepState state = classifyStrictPoint(map, end);
  if (state != STRICT_STEP_SAFE)
    return state;

  const double length = step.norm();
  if (length <= 1e-9)
    return STRICT_STEP_SAFE;
  const Vector2d direction = step / length;
  for (double distance = 0.025; distance < length; distance += 0.025) {
    state = classifyStrictPoint(map, start + distance * direction);
    if (state != STRICT_STEP_SAFE)
      return state;
  }
  return STRICT_STEP_SAFE;
}

StrictStepDebug inspectStrictFirstSteps(const SDFMap2D::Ptr& map,
    const Vector2d& start, const std::vector<Vector2d>& steps)
{
  StrictStepDebug debug;
  for (const auto& step : steps) {
    switch (classifyStrictStep(map, start, step)) {
      case STRICT_STEP_SAFE:
        ++debug.safe;
        break;
      case STRICT_STEP_OUT_OF_MAP:
        ++debug.out_of_map;
        break;
      case STRICT_STEP_UNKNOWN:
        ++debug.unknown;
        break;
      case STRICT_STEP_OCCUPIED:
        ++debug.occupied;
        break;
      case STRICT_STEP_INFLATED:
        ++debug.inflated;
        break;
    }
  }
  return debug;
}
}  // namespace

void ExplorationFSM::init(ros::NodeHandle& nh)
{
  nh_ = nh;
  fp_.reset(new FSMParam);
  fd_.reset(new FSMData);

  /* Initialize main modules */
  expl_manager_.reset(new ExplorationManager);
  expl_manager_->initialize(nh);
  visualization_.reset(new PlanningVisualization(nh));
  fp_->vis_scale_ = expl_manager_->sdf_map_->getResolution() * FSMConstants::VIS_SCALE_FACTOR;

  state_ = ROS_STATE::INIT;

  /* ROS Timer */
  exec_timer_ = nh.createTimer(
      ros::Duration(FSMConstants::EXEC_TIMER_DURATION), &ExplorationFSM::FSMCallback, this);
  frontier_timer_ = nh.createTimer(ros::Duration(FSMConstants::FRONTIER_TIMER_DURATION),
      &ExplorationFSM::frontierCallback, this);

  /* ROS Subscriber */
  trigger_sub_ = nh.subscribe("/move_base_simple/goal", 10, &ExplorationFSM::triggerCallback, this);
  odom_sub_ = nh.subscribe("/odom_world", 10, &ExplorationFSM::odometryCallback, this);
  habitat_state_sub_ =
      nh.subscribe("/habitat/state", 10, &ExplorationFSM::habitatStateCallback, this);
  confidence_threshold_sub_ = node_.subscribe(
      "/detector/confidence_threshold", 10, &ExplorationFSM::confidenceThresholdCallback, this);
  mast3r_hint_sub_ = nh.subscribe(
      "/habitat/mast3r_hint", 10, &ExplorationFSM::mast3rHintCallback, this);
  instance_stop_gate_sub_ = nh.subscribe(
      "/habitat/instance_stop_gate", 10, &ExplorationFSM::instanceStopGateCallback, this);
  verified_approach_target_sub_ = nh.subscribe(
      "/habitat/verified_approach_target", 10,
      &ExplorationFSM::verifiedApproachTargetCallback, this);
  resume_exploration_sub_ = nh.subscribe(
      "/habitat/resume_exploration", 10, &ExplorationFSM::resumeExplorationCallback, this);

  /* ROS Publisher */
  ros_state_pub_ = nh.advertise<std_msgs::Int32>("/ros/state", 10);
  expl_state_pub_ = nh.advertise<std_msgs::Int32>("/ros/expl_state", 10);
  action_pub_ = nh.advertise<std_msgs::Int32>("/habitat/plan_action", 10);
  expl_result_pub_ = nh.advertise<std_msgs::Int32>("/ros/expl_result", 10);
  mast3r_refine_status_pub_ = nh.advertise<std_msgs::Int32>("/habitat/mast3r_refine_status", 10);
  mast3r_debug_pub_ = nh.advertise<std_msgs::String>("/ros/object_viewpoint_debug", 20);
  robot_marker_pub_ = nh.advertise<visualization_msgs::Marker>("/robot", 10);
}

// FSM between ROS and Habitat for action planning and execution
void ExplorationFSM::FSMCallback(const ros::TimerEvent& e)
{
  exec_timer_.stop();
  std_msgs::Int32 ros_state_msg;
  ros_state_msg.data = state_;
  ros_state_pub_.publish(ros_state_msg);
  switch (state_) {
    case ROS_STATE::INIT: {
      // Wait for odometry and target confidence threshold
      if (!fd_->have_odom_ || !fd_->have_confidence_) {
        ROS_WARN_THROTTLE(1.0, "No odom || No target confidence threshold.");
        exec_timer_.start();
        return;
      }
      // Go to WAIT_TRIGGER when prerequisites are ready
      clearVisMarker();
      transitState(ROS_STATE::WAIT_TRIGGER, "FSM");
      break;
    }

    case ROS_STATE::WAIT_TRIGGER: {
      // Do nothing but wait for trigger
      ROS_WARN_THROTTLE(1.0, "Wait for trigger.");
      break;
    }

    case ROS_STATE::FINISH: {
      if (!fd_->have_finished_) {
        fd_->have_finished_ = true;
        clearVisMarker();
        std_msgs::Int32 action_msg;
        action_msg.data = ACTION::STOP;
        action_pub_.publish(action_msg);
      }
      ROS_WARN_THROTTLE(1.0, "Finish One Episode!!!");
      break;
    }

    case ROS_STATE::PLAN_ACTION: {
      // Initial action sequence: perform orientation calibration turns
      if (fd_->init_action_count_ < 1 + 12 + 1 + 12) {
        if (fd_->init_action_count_ < 1)
          fd_->newest_action_ = ACTION::TURN_DOWN;
        else if (fd_->init_action_count_ < 1 + 12)
          fd_->newest_action_ = ACTION::TURN_LEFT;
        else if (fd_->init_action_count_ < 1 + 12 + 1)
          fd_->newest_action_ = ACTION::TURN_UP;
        else
          fd_->newest_action_ = ACTION::TURN_LEFT;
        ROS_WARN("Init Mode Process -----> (%d/26)", fd_->init_action_count_);
        fd_->init_action_count_++;
        transitState(ROS_STATE::PUB_ACTION, "FSM");
        updateFrontierAndObject();
      }
      else {
        // Main planning phase: determine robot pose and call action planner
        fd_->start_pt_ = fd_->odom_pos_;
        fd_->start_yaw_(0) = fd_->odom_yaw_;

        auto t1 = ros::Time::now();
        fd_->final_result_ = callActionPlanner();
        double call_action_planner_time = (ros::Time::now() - t1).toSec();
        ROS_INFO_THROTTLE(
            10.0, "[Calculating Time] Planning process time = %.3f s", call_action_planner_time);

        std_msgs::Int32 expl_state_msg;
        expl_state_msg.data = fd_->final_result_;
        expl_state_pub_.publish(expl_state_msg);
        if (fd_->final_result_ == FINAL_RESULT::EXPLORE ||
            fd_->final_result_ == FINAL_RESULT::SEARCH_OBJECT)
          transitState(ROS_STATE::PUB_ACTION, "FSM");
        else
          transitState(ROS_STATE::FINISH, "FSM");
      }
      visualize();
      break;
    }

    case ROS_STATE::PUB_ACTION: {
      std_msgs::Int32 action_msg;
      action_msg.data = fd_->newest_action_;
      action_pub_.publish(action_msg);
      transitState(ROS_STATE::WAIT_ACTION_FINISH, "FSM");
      break;
    }

    case ROS_STATE::WAIT_ACTION_FINISH: {
      exec_timer_.start();
      break;
    }
  }
  exec_timer_.start();
}

/**
 * @brief Plan the next action based on current state and environment
 * @return Final result indicating the planned action type and exploration state
 *
 * This is the core planning function that decides what action the robot should take next.
 * It handles obstacle avoidance, frontier exploration, object search, and stuck recovery.
 */
int ExplorationFSM::callActionPlanner()
{
  const double stucking_distance = FSMConstants::STUCKING_DISTANCE;
  const double reach_distance = FSMConstants::REACH_DISTANCE;
  const double soft_reach_distance = FSMConstants::SOFT_REACH_DISTANCE;
  const double object_viewpoint_arrival_distance =
      expl_manager_->getObjectViewpointReachDistance() + stucking_distance;

  bool frontier_change_flag = updateFrontierAndObject();

  int expl_res, final_res;
  Eigen::Vector2d current_pos = Eigen::Vector2d(fd_->start_pt_(0), fd_->start_pt_(1));
  Eigen::Vector2d last_pos = Eigen::Vector2d(fd_->last_start_pos_(0), fd_->last_start_pos_(1));
  double current_yaw = fd_->start_yaw_(0);
  fd_->last_start_pos_ = fd_->start_pt_;
  bool mast3r_refinement_active = fd_->mast3r_hint_active_ || fd_->mast3r_goal_active_;
  bool instance_stop_gate_blocks_object_finish =
      fd_->instance_stop_gate_enabled_ && !fd_->mast3r_allow_stop_;
  auto resetObjectViewpointAlignment = [&]() {
    fd_->object_viewpoint_alignment_steps_ = 0;
    fd_->object_viewpoint_last_alignment_action_ = -1;
  };
  const bool verified_approach_active = expl_manager_->isVerifiedApproachActive();
  if (mast3r_refinement_active || verified_approach_active)
    resetObjectViewpointAlignment();

  if (mast3r_refinement_active && fd_->escape_stucking_flag_) {
    fd_->escape_stucking_flag_ = false;
    fd_->escape_stucking_count_ = 0;
    ROS_INFO("Cancel generic escape sequence because MASt3R now owns a locked target.");
  }

  // A committed MASt3R path owns motion until it reaches the locked pose or fails.
  // Its final SE(2) target stays fixed, while A* executes one short path segment at a time.
  // Do not let object rejection or generic escape actions change its viewpoint.
  if (fd_->mast3r_goal_active_ && fd_->mast3r_path_active_) {
    const Vector2d mast3r_target_pos = fd_->mast3r_target_pos_;
    const double mast3r_target_distance = (mast3r_target_pos - current_pos).norm();
    if (mast3r_target_distance < FSMConstants::MAST3R_LOCAL_GOAL_REACH_DISTANCE) {
      double target_yaw = fd_->mast3r_target_yaw_;
      double yaw_error = target_yaw - current_yaw;
      wrapAngle(yaw_error);
      if (fd_->mast3r_goal_age_ >= FSMConstants::MAST3R_LOCAL_GOAL_MAX_ACTIVE_AGE &&
          std::fabs(yaw_error) > FSMConstants::MAST3R_FINE_YAW_HALF_ANGLE) {
        ROS_WARN(
            "MASt3R final yaw alignment timed out: yaw_error=%.2f deg age=%d; "
            "resume locked object inspection.",
            yaw_error * 180.0 / M_PI, fd_->mast3r_goal_age_);
        clearMast3rLocalGoal();
        publishMast3rRefineStatus(-3);
        fd_->newest_action_ = ACTION::TURN_LEFT;
        return FINAL_RESULT::SEARCH_OBJECT;
      }
      if (!alignMast3rTargetYaw(current_yaw)) {
        fd_->mast3r_goal_age_++;
        return FINAL_RESULT::SEARCH_OBJECT;
      }
      clearMast3rLocalGoal();
      publishMast3rRefineStatus(1);
      fd_->newest_action_ = ACTION::STOP;
      ROS_INFO_THROTTLE(0.5,
          "MASt3R local target position and yaw reached; report locked-pose arrival.");
      return FINAL_RESULT::SEARCH_OBJECT;
    }

    bool mast3r_segment_completed = false;
    if (fd_->mast3r_segment_target_valid_ &&
        (fd_->mast3r_segment_target_pos_ - current_pos).norm() <
            FSMConstants::MAST3R_LOCAL_GOAL_REACH_DISTANCE) {
      mast3r_segment_completed = true;
      fd_->mast3r_path_active_ = false;
      fd_->mast3r_segment_target_valid_ = false;
      expl_manager_->ed_->next_best_path_.clear();
      ROS_INFO(
          "MASt3R rolling horizon reached: segment=%d pos=(%.3f, %.3f); "
          "replan toward fixed final target=(%.3f, %.3f).",
          fd_->mast3r_segment_count_, current_pos(0), current_pos(1),
          mast3r_target_pos(0), mast3r_target_pos(1));
    }

    if (!fd_->mast3r_last_pos_valid_) {
      fd_->mast3r_last_pos_ = current_pos;
      fd_->mast3r_last_pos_valid_ = true;
      fd_->mast3r_forward_blocked_count_ = 0;
    }
    else {
      const double moved_distance = (current_pos - fd_->mast3r_last_pos_).norm();
      if (fd_->newest_action_ == ACTION::MOVE_FORWARD &&
          moved_distance < FSMConstants::STUCKING_DISTANCE) {
        ++fd_->mast3r_forward_blocked_count_;
      }
      else {
        fd_->mast3r_forward_blocked_count_ = 0;
      }
      fd_->mast3r_last_pos_ = current_pos;
    }

    const bool mast3r_stalled =
        fd_->mast3r_forward_blocked_count_ >=
        FSMConstants::MAST3R_LOCAL_GOAL_FORWARD_FAILURES_BEFORE_REPLAN;
    const bool mast3r_expired =
        fd_->mast3r_goal_age_ >= FSMConstants::MAST3R_LOCAL_GOAL_MAX_ACTIVE_AGE;
    if (mast3r_expired) {
      ROS_WARN(
          "MASt3R local goal failed: reason=active_age_limit target_distance=%.3f age=%d "
          "replans=%d; resume locked object inspection.",
          mast3r_target_distance, fd_->mast3r_goal_age_, fd_->mast3r_path_replan_count_);
      clearMast3rLocalGoal();
      publishMast3rRefineStatus(-3);
      fd_->newest_action_ = ACTION::TURN_LEFT;
      return FINAL_RESULT::SEARCH_OBJECT;
    }

    if (mast3r_segment_completed) {
      if (!planMast3rPath(current_pos, mast3r_target_pos, false)) {
        ROS_WARN(
            "MASt3R fixed target cannot continue after segment=%d: target=(%.3f, %.3f) "
            "distance=%.3f; release the target.",
            fd_->mast3r_segment_count_, mast3r_target_pos(0), mast3r_target_pos(1),
            mast3r_target_distance);
        clearMast3rLocalGoal();
        publishMast3rRefineStatus(-2);
        fd_->newest_action_ = ACTION::TURN_LEFT;
        return FINAL_RESULT::SEARCH_OBJECT;
      }
    }
    else if (mast3r_stalled || expl_manager_->ed_->next_best_path_.empty()) {
      const char* reason = mast3r_stalled ? "forward_blocked" : "path_lost";
      ROS_WARN(
          "MASt3R local path needs replanning: reason=%s target_distance=%.3f age=%d "
          "blocked_forwards=%d replan=%d/%d.",
          reason, mast3r_target_distance, fd_->mast3r_goal_age_,
          fd_->mast3r_forward_blocked_count_, fd_->mast3r_path_replan_count_ + 1,
          FSMConstants::MAST3R_LOCAL_GOAL_MAX_REPLANS);
      if (mast3r_stalled)
        markMast3rForwardCollision(current_pos, current_yaw);
      if (!planMast3rPath(current_pos, mast3r_target_pos, true)) {
        ROS_WARN(
            "MASt3R fixed target cannot be replanned: target=(%.3f, %.3f) "
            "distance=%.3f replans=%d; release the target.",
            mast3r_target_pos(0), mast3r_target_pos(1), mast3r_target_distance,
            fd_->mast3r_path_replan_count_);
        clearMast3rLocalGoal();
        publishMast3rRefineStatus(-3);
        fd_->newest_action_ = ACTION::TURN_LEFT;
        return FINAL_RESULT::SEARCH_OBJECT;
      }
    }

    fd_->local_pos_ = fd_->mast3r_segment_target_valid_
                         ? fd_->mast3r_segment_target_pos_
                         : mast3r_target_pos;
    fd_->newest_action_ = planMast3rPathAction(
        current_pos, current_yaw, expl_manager_->ed_->next_best_path_);
    ROS_INFO_THROTTLE(0.5,
        "MASt3R rolling A* target active: age=%d segment=%d replans=%d "
        "final_distance=%.3f segment_target=(%.3f, %.3f) forward_err=%.3f "
        "lateral_err=%.3f yaw_err=%.2f deg transl_err=%.3f",
        fd_->mast3r_goal_age_, fd_->mast3r_segment_count_,
        fd_->mast3r_path_replan_count_, mast3r_target_distance, fd_->local_pos_(0),
        fd_->local_pos_(1),
        fd_->mast3r_forward_error_, fd_->mast3r_lateral_error_, fd_->mast3r_yaw_error_deg_,
        fd_->mast3r_transl_error_);
    fd_->mast3r_goal_age_++;
    return FINAL_RESULT::SEARCH_OBJECT;
  }

  bool object_viewpoint_completed = false;
  const bool object_viewpoint_locked = expl_manager_->hasLockedObjectViewpoint();
  if (!object_viewpoint_locked) {
    fd_->object_viewpoint_scan_target_valid_ = false;
    fd_->object_viewpoint_scan_phase_ = 0;
    fd_->object_viewpoint_forward_blocked_count_ = 0;
    resetObjectViewpointAlignment();
  }
  if (!mast3r_refinement_active && !verified_approach_active && object_viewpoint_locked) {
    const Vector2d viewpoint = expl_manager_->getLockedObjectViewpoint();
    const Vector2d object_center = expl_manager_->getLockedObjectCenter();
    if (!fd_->object_viewpoint_scan_target_valid_ ||
        (viewpoint - fd_->object_viewpoint_scan_target_).norm() > 1e-3) {
      fd_->object_viewpoint_scan_target_valid_ = true;
      fd_->object_viewpoint_scan_target_ = viewpoint;
      fd_->object_viewpoint_scan_phase_ = 0;
      resetObjectViewpointAlignment();
    }
    // Discrete Habitat forward actions can stop just outside the A* reach radius. Treat the
    // final stuck-distance band as arrival so a locked view cannot fall between "arrived" and
    // "forward blocked" handling.
    if ((viewpoint - current_pos).norm() <= object_viewpoint_arrival_distance) {
      double target_yaw = std::atan2(
          object_center.y() - current_pos.y(), object_center.x() - current_pos.x());
      double yaw_diff = target_yaw - current_yaw;
      wrapAngle(yaw_diff);
      if (std::fabs(yaw_diff) > FSMConstants::OBJECT_VIEWPOINT_YAW_TOLERANCE) {
        const int alignment_action =
            yaw_diff > 0.0 ? ACTION::TURN_LEFT : ACTION::TURN_RIGHT;
        const bool reversed = fd_->object_viewpoint_alignment_steps_ > 0 &&
            fd_->object_viewpoint_last_alignment_action_ != alignment_action;
        ++fd_->object_viewpoint_alignment_steps_;

        if (reversed) {
          ROS_WARN(
              "[OBJECT_VIEWPOINT_LOCK] final yaw crossed the target between discrete "
              "headings; accept the nearest heading: yaw_diff=%.2f deg attempts=%d",
              yaw_diff * 180.0 / M_PI, fd_->object_viewpoint_alignment_steps_);
          resetObjectViewpointAlignment();
        }
        else if (fd_->object_viewpoint_alignment_steps_ >=
                 FSMConstants::OBJECT_VIEWPOINT_MAX_ALIGNMENT_STEPS) {
          ROS_WARN(
              "[OBJECT_VIEWPOINT_LOCK] final yaw did not converge after %d turns; "
              "advance the locked viewpoint: yaw_diff=%.2f deg",
              fd_->object_viewpoint_alignment_steps_, yaw_diff * 180.0 / M_PI);
          const bool advanced = expl_manager_->advanceLockedObjectViewpoint(
              fd_->start_pt_, "yaw_alignment_timeout");
          fd_->object_viewpoint_scan_target_valid_ = false;
          fd_->object_viewpoint_scan_phase_ = 0;
          resetObjectViewpointAlignment();
          ROS_WARN("[OBJECT_VIEWPOINT_LOCK] yaw alignment timeout; %s",
              advanced ? "switch to next object view" : "all object views exhausted");
          object_viewpoint_completed = true;
        }
        else {
          fd_->object_viewpoint_last_alignment_action_ = alignment_action;
          fd_->newest_action_ = alignment_action;
          ROS_WARN(
              "[OBJECT_VIEWPOINT_LOCK] position reached; face object before releasing target: "
              "yaw_diff=%.2f deg action=%s attempt=%d/%d",
              yaw_diff * 180.0 / M_PI,
              fd_->newest_action_ == ACTION::TURN_LEFT ? "turn_left" : "turn_right",
              fd_->object_viewpoint_alignment_steps_,
              FSMConstants::OBJECT_VIEWPOINT_MAX_ALIGNMENT_STEPS);
          return FINAL_RESULT::SEARCH_OBJECT;
        }
      }
      else {
        resetObjectViewpointAlignment();
      }

      if (!object_viewpoint_completed && fd_->object_viewpoint_scan_phase_ == 0) {
        fd_->object_viewpoint_scan_phase_ = 1;
        fd_->newest_action_ = ACTION::TURN_LEFT;
        ROS_WARN(
            "[OBJECT_VIEWPOINT_LOCK] position reached and aligned; collect a side-view "
            "confirmation frame before considering another viewpoint");
        return FINAL_RESULT::SEARCH_OBJECT;
      }

      if (!object_viewpoint_completed) {
        // A new image is only available after an action. Sweep once to collect an off-axis frame,
        // then the following turn re-aligns the camera and supplies a fresh centered frame before
        // any fallback viewpoint can take control.
        const bool advanced = expl_manager_->advanceLockedObjectViewpoint(
            fd_->start_pt_, "inspection_sweep_completed_without_confirmation");
        fd_->object_viewpoint_scan_target_valid_ = false;
        fd_->object_viewpoint_scan_phase_ = 0;
        resetObjectViewpointAlignment();
        ROS_WARN("[OBJECT_VIEWPOINT_LOCK] fresh aligned observation completed; %s",
            advanced ? "switch to next object view" : "all object views exhausted");
        object_viewpoint_completed = true;
      }
    }
    else {
      resetObjectViewpointAlignment();
    }
  }

  // Reach the object - check if close enough to target object
  if (!object_viewpoint_completed && !object_viewpoint_locked &&
      fd_->final_result_ == FINAL_RESULT::SEARCH_OBJECT &&
      (current_pos - expl_manager_->ed_->next_pos_).norm() < reach_distance) {
    if (mast3r_refinement_active && !fd_->mast3r_allow_stop_) {
      ROS_WARN("Near target but MASt3R requests further locked-pose refinement.");
    }
    else if (instance_stop_gate_blocks_object_finish) {
      ROS_WARN(
          "Near object-map candidate, but INSiNav requires visual instance verification before STOP.");
      expl_manager_->object_map2d_->rejectInstanceCandidate(
          expl_manager_->ed_->next_pos_, FSMConstants::INSTANCE_REJECT_RADIUS);
    }
    else {
      ROS_ERROR("Reach the object successfully!!!");
      clearMast3rLocalGoal();
      final_res = FINAL_RESULT::REACH_OBJECT;
      return final_res;
    }
  }

  /*******  Escape-from-stuck logic START *******/
  // Detect if robot is stuck and initiate escape sequence
  int last_action = fd_->newest_action_;
  bool object_viewpoint_forward_failed = false;
  if (!object_viewpoint_completed && !mast3r_refinement_active && object_viewpoint_locked &&
      !fd_->escape_stucking_flag_ && (current_pos - last_pos).norm() < stucking_distance &&
      last_action == ACTION::MOVE_FORWARD) {
    const double distance_to_locked_viewpoint =
        (current_pos - expl_manager_->getLockedObjectViewpoint()).norm();
    if (distance_to_locked_viewpoint > expl_manager_->getObjectViewpointReachDistance()) {
      ++fd_->object_viewpoint_forward_blocked_count_;
      if (fd_->object_viewpoint_forward_blocked_count_ >=
          FSMConstants::OBJECT_VIEWPOINT_FORWARD_FAILURES_BEFORE_ADVANCE) {
        expl_manager_->advanceLockedObjectViewpoint(fd_->start_pt_, "forward_blocked");
        fd_->object_viewpoint_scan_target_valid_ = false;
        fd_->object_viewpoint_scan_phase_ = 0;
        fd_->object_viewpoint_forward_blocked_count_ = 0;
        resetObjectViewpointAlignment();
        object_viewpoint_forward_failed = true;
      }
      else {
        ROS_WARN(
            "[OBJECT_VIEWPOINT_LOCK] forward made no progress; retry current view before "
            "advancing (%d/%d)",
            fd_->object_viewpoint_forward_blocked_count_,
            FSMConstants::OBJECT_VIEWPOINT_FORWARD_FAILURES_BEFORE_ADVANCE);
      }
    }
    else {
      fd_->object_viewpoint_forward_blocked_count_ = 0;
    }
  }
  else {
    fd_->object_viewpoint_forward_blocked_count_ = 0;
  }

  if (!mast3r_refinement_active && !object_viewpoint_locked && !fd_->escape_stucking_flag_ &&
      (current_pos - last_pos).norm() < stucking_distance &&
      last_action == ACTION::MOVE_FORWARD) {
    if (!object_viewpoint_locked && !object_viewpoint_forward_failed &&
        fd_->final_result_ == FINAL_RESULT::SEARCH_OBJECT &&
        (current_pos - expl_manager_->ed_->next_pos_).norm() < soft_reach_distance) {
      if (mast3r_refinement_active && !fd_->mast3r_allow_stop_) {
        ROS_WARN("Soft-reach satisfied but MASt3R still owns the locked target.");
      }
      else if (instance_stop_gate_blocks_object_finish) {
        ROS_WARN(
            "Soft-reach object-map candidate, but INSiNav keeps exploring until visual verification.");
        expl_manager_->object_map2d_->rejectInstanceCandidate(
            expl_manager_->ed_->next_pos_, FSMConstants::INSTANCE_REJECT_RADIUS);
      }
      else {
        ROS_ERROR("Reach the object successfully!!!");
        clearMast3rLocalGoal();
        final_res = FINAL_RESULT::REACH_OBJECT;
        return final_res;
      }
    }

    bool past_stucking_flag = false;
    for (auto stucking_point : fd_->stucking_points_) {
      Vector2d stucking_pos = Vector2d(stucking_point(0), stucking_point(1));
      double stucking_yaw = stucking_point(2);
      if ((stucking_pos - current_pos).norm() < stucking_distance &&
          fabs(stucking_yaw - current_yaw) < FSMConstants::ACTION_ANGLE) {
        past_stucking_flag = true;
        ROS_ERROR("Still stuck at the same place");
        break;
      }
    }
    if (!past_stucking_flag) {
      fd_->escape_stucking_flag_ = true;
      fd_->escape_stucking_count_ = 0;
      fd_->escape_stucking_pos_ = current_pos;
      fd_->escape_stucking_yaw_ = current_yaw;
    }
  }

  if (fd_->escape_stucking_flag_ && (current_pos - last_pos).norm() >= stucking_distance) {
    ROS_ERROR("Escaped from stuck state.");
    fd_->escape_stucking_flag_ = false;
  }

  if (fd_->escape_stucking_flag_) {
    ROS_ERROR("Escaping stuck...");
    if (fd_->escape_stucking_count_ == 0)
      fd_->newest_action_ = ACTION::TURN_RIGHT;
    else if (fd_->escape_stucking_count_ == 1)
      fd_->newest_action_ = ACTION::MOVE_FORWARD;
    else if (fd_->escape_stucking_count_ == 2)
      fd_->newest_action_ = ACTION::TURN_RIGHT;
    else if (fd_->escape_stucking_count_ == 3)
      fd_->newest_action_ = ACTION::MOVE_FORWARD;
    else if (fd_->escape_stucking_count_ == 4)
      fd_->newest_action_ = ACTION::TURN_LEFT;
    else if (fd_->escape_stucking_count_ == 5)
      fd_->newest_action_ = ACTION::TURN_LEFT;
    else if (fd_->escape_stucking_count_ == 6)
      fd_->newest_action_ = ACTION::TURN_LEFT;
    else if (fd_->escape_stucking_count_ == 7)
      fd_->newest_action_ = ACTION::MOVE_FORWARD;
    else if (fd_->escape_stucking_count_ == 8)
      fd_->newest_action_ = ACTION::TURN_LEFT;
    else if (fd_->escape_stucking_count_ == 9)
      fd_->newest_action_ = ACTION::MOVE_FORWARD;
    else {
      // Failed to escape - mark area as occupied and add to stuck points
      ROS_ERROR("Cannot escape stuck state.");
      fd_->escape_stucking_flag_ = false;
      expl_manager_->sdf_map_->setForceOccGrid(current_pos);
      double forward_distance = FSMConstants::FORWARD_DISTANCE;
      Eigen::Vector2d forward_pos = fd_->escape_stucking_pos_;
      forward_pos(0) += forward_distance * cos(fd_->escape_stucking_yaw_);
      forward_pos(1) += forward_distance * sin(fd_->escape_stucking_yaw_);
      expl_manager_->sdf_map_->setForceOccGrid(forward_pos);
      forward_distance = FSMConstants::FORWARD_DISTANCE * 2.0;
      forward_pos = fd_->escape_stucking_pos_;
      forward_pos(0) += forward_distance * cos(fd_->escape_stucking_yaw_);
      forward_pos(1) += forward_distance * sin(fd_->escape_stucking_yaw_);
      expl_manager_->sdf_map_->setForceOccGrid(forward_pos);
      fd_->dormant_frontier_flag_ = true;
      Vector3d stucking_point(
          fd_->escape_stucking_pos_(0), fd_->escape_stucking_pos_(1), fd_->escape_stucking_yaw_);
      fd_->stucking_points_.push_back(stucking_point);
    }

    if (fd_->escape_stucking_flag_) {
      fd_->escape_stucking_count_++;
      return fd_->final_result_;
    }
  }

  /*******  Decide whether to replan path (stability heuristic) START *******/
  // Use path stability to reduce oscillation between different frontier targets
  vector<Vector2d> last_next_best_path = expl_manager_->ed_->next_best_path_;
  Vector2d last_next_pos = expl_manager_->ed_->next_pos_;
  if (fd_->dormant_frontier_flag_) {
    fd_->replan_flag_ = true;
    fd_->dormant_frontier_flag_ = false;
  }
  else if (fd_->final_result_ == FINAL_RESULT::EXPLORE && !frontier_change_flag)
    fd_->replan_flag_ = false;

  expl_res = expl_manager_->planNextBestPoint(fd_->start_pt_, fd_->start_yaw_(0));

  if (expl_res != EXPL_RESULT::EXPLORATION) {
    fd_->replan_flag_ = true;
  }
  if (expl_res == EXPL_RESULT::EXPLORATION && !fd_->replan_flag_) {
    expl_manager_->ed_->next_best_path_ = last_next_best_path;
    expl_manager_->ed_->next_pos_ = last_next_pos;
    fd_->replan_flag_ = true;
  }
  /*******  Decide whether to replan path (stability heuristic) END *******/

  // Publish exploration result to monitor
  std_msgs::Int32 expl_result_msg;
  expl_result_msg.data = expl_res;
  expl_result_pub_.publish(expl_result_msg);

  // Determine current high-level state based on exploration results
  if (expl_res == EXPL_RESULT::EXPLORATION)
    final_res = FINAL_RESULT::EXPLORE;
  else if (expl_res == EXPL_RESULT::NO_COVERABLE_FRONTIER ||
           expl_res == EXPL_RESULT::NO_PASSABLE_FRONTIER)
    final_res = FINAL_RESULT::NO_FRONTIER;
  else
    final_res = FINAL_RESULT::SEARCH_OBJECT;

  if ((final_res == FINAL_RESULT::NO_FRONTIER || expl_manager_->ed_->next_best_path_.empty()) &&
      !mast3r_refinement_active) {
    ROS_WARN("No (passable) frontier");
    clearMast3rLocalGoal();
    return final_res;
  }

  Eigen::Vector2d end_pos = expl_manager_->ed_->next_pos_;
  Eigen::Vector2d last_end_pos = fd_->last_next_pos_;
  fd_->last_next_pos_ = end_pos;
  double min_dist = (current_pos - end_pos).norm();
  ROS_WARN("To the next point (%.2fm %.2fm), distance = %.2f m", end_pos(0), end_pos(1), min_dist);

  // Handling being stuck while exploring toward a specific frontier
  if (final_res == FINAL_RESULT::EXPLORE) {
    // Force dormant if very close to target but still exploring
    if (min_dist < FSMConstants::FORCE_DORMANT_DISTANCE) {
      ROS_ERROR("Force set dormant frontier.");
      expl_manager_->frontier_map2d_->setForceDormantFrontier(end_pos);
      fd_->dormant_frontier_flag_ = true;
    }

    // Count consecutive times with same target position while stuck
    if ((end_pos - last_end_pos).norm() < 1e-3 &&
        (current_pos - last_pos).norm() < stucking_distance) {
      fd_->stucking_next_pos_count_++;
      ROS_ERROR_COND(fd_->stucking_next_pos_count_ > 8, "stucking_next_pos_count_ = %d",
          fd_->stucking_next_pos_count_);
    }
    else
      fd_->stucking_next_pos_count_ = 0;

    // Mark frontier as dormant if stuck too long with same target
    if (fd_->stucking_next_pos_count_ >= FSMConstants::MAX_STUCKING_NEXT_POS_COUNT) {
      ROS_ERROR("Set dormant frontier.");
      fd_->stucking_action_count_ = 0;
      fd_->stucking_next_pos_count_ = 0;
      expl_manager_->frontier_map2d_->setForceDormantFrontier(end_pos);
      fd_->dormant_frontier_flag_ = true;
    }
  }

  // Object-viewpoint and MASt3R locks have bounded, target-specific recovery paths. The global
  // frontier stuck counter must not terminate one while it is still being resolved.
  if (object_viewpoint_locked || mast3r_refinement_active) {
    fd_->stucking_action_count_ = 0;
  }
  else {
    // Track consecutive stuck actions globally.
    if ((current_pos - last_pos).norm() < stucking_distance) {
      fd_->stucking_action_count_++;
      ROS_ERROR_COND(fd_->stucking_action_count_ > 15, "Stucking action count = %d",
          fd_->stucking_action_count_);
    }
    else {
      fd_->stucking_action_count_ = 0;
    }

    if (fd_->stucking_action_count_ >= FSMConstants::MAX_STUCKING_COUNT) {
      ROS_ERROR("Stuck for too long, stopping episode.");
      clearMast3rLocalGoal();
      final_res = FINAL_RESULT::STUCKING;
      return final_res;
    }
  }

  // Plan specific action based on exploration result
  if (!expl_manager_->ed_->next_best_path_.empty()) {
    if (expl_res == EXPL_RESULT::SEARCH_EXTREME)
      fd_->newest_action_ =
          planNextBestAction(current_pos, current_yaw, expl_manager_->ed_->next_best_path_, false);
    else
      fd_->newest_action_ =
          planNextBestAction(current_pos, current_yaw, expl_manager_->ed_->next_best_path_);
  }

  mast3r_refinement_active = fd_->mast3r_hint_active_ || fd_->mast3r_goal_active_;
  if (mast3r_refinement_active) {
    if (fd_->mast3r_allow_stop_) {
      ROS_INFO_THROTTLE(0.5, "MASt3R hint allows stop when planner reaches the target.");
      clearMast3rLocalGoal();
    }
    else {
      Vector2d mast3r_target_pos;
      double mast3r_target_yaw = current_yaw;
      if (fd_->mast3r_hint_active_) {
        if (buildMast3rLocalTarget(current_pos, current_yaw, mast3r_target_pos, mast3r_target_yaw)) {
          fd_->mast3r_target_pos_ = mast3r_target_pos;
          fd_->mast3r_target_yaw_ = mast3r_target_yaw;
          fd_->mast3r_goal_active_ = true;
          fd_->mast3r_path_active_ = false;
          fd_->mast3r_segment_target_valid_ = false;
          fd_->mast3r_segment_count_ = 0;
          fd_->mast3r_goal_age_ = 0;
          fd_->mast3r_path_replan_count_ = 0;
        }
        fd_->mast3r_hint_active_ = false;
      }

      if (fd_->mast3r_goal_active_) {
        mast3r_target_pos = fd_->mast3r_target_pos_;
        mast3r_target_yaw = fd_->mast3r_target_yaw_;

        if ((mast3r_target_pos - current_pos).norm() < FSMConstants::MAST3R_LOCAL_GOAL_REACH_DISTANCE) {
          if (!alignMast3rTargetYaw(current_yaw)) {
            fd_->mast3r_goal_age_++;
            return final_res;
          }
          clearMast3rLocalGoal();
          publishMast3rRefineStatus(1);
          fd_->newest_action_ = ACTION::STOP;
          ROS_INFO_THROTTLE(0.5,
              "MASt3R local target position and yaw reached; report locked-pose arrival.");
          return final_res;
        }
        else if (!fd_->mast3r_path_active_ &&
                 fd_->mast3r_goal_age_ >= FSMConstants::MAST3R_LOCAL_GOAL_MAX_AGE) {
          clearMast3rLocalGoal();
          publishMast3rRefineStatus(-2);
          fd_->newest_action_ = ACTION::TURN_LEFT;
          ROS_WARN_THROTTLE(0.5, "MASt3R local target expired; fall back to object-map planning.");
        }
      }

      if (fd_->mast3r_goal_active_ && !fd_->mast3r_path_active_) {
        if (!planMast3rPath(current_pos, mast3r_target_pos, false)) {
          clearMast3rLocalGoal();
          publishMast3rRefineStatus(-2);
          fd_->newest_action_ = ACTION::TURN_LEFT;
          ROS_WARN_THROTTLE(0.5,
              "MASt3R fixed target has no strict A* path; fall back to object-map planning.");
        }
      }

      if (fd_->mast3r_goal_active_ && fd_->mast3r_path_active_) {
        fd_->local_pos_ = fd_->mast3r_segment_target_valid_
                             ? fd_->mast3r_segment_target_pos_
                             : mast3r_target_pos;
        fd_->newest_action_ = planMast3rPathAction(
            current_pos, current_yaw, expl_manager_->ed_->next_best_path_);
        ROS_INFO_THROTTLE(0.5,
            "MASt3R rolling A* target accepted: final_target=(%.3f, %.3f) "
            "final_distance=%.3f yaw_target=%.2f deg segment=%d "
            "segment_target=(%.3f, %.3f) segment_len=%.3f",
            mast3r_target_pos(0), mast3r_target_pos(1),
            (mast3r_target_pos - current_pos).norm(), mast3r_target_yaw * 180.0 / M_PI,
            fd_->mast3r_segment_count_, fd_->local_pos_(0), fd_->local_pos_(1),
            Astar2D::pathLength(expl_manager_->ed_->next_best_path_));
        fd_->mast3r_goal_age_++;
      }
    }
  }
  else if (final_res != FINAL_RESULT::SEARCH_OBJECT) {
    clearMast3rLocalGoal();
  }

  return final_res;
}

int ExplorationFSM::planNextBestAction(
    Vector2d current_pos, double current_yaw, const vector<Vector2d>& path, bool need_safety)
{
  const double local_distance = FSMConstants::LOCAL_DISTANCE;

  // Update target position based on path and local distance
  Vector2d local_pos = selectLocalTarget(current_pos, path, local_distance);
  fd_->local_pos_ = local_pos;

  // Compute the best step considering obstacles and safety
  Vector2d best_step;
  if ((current_pos - path.back()).norm() > FSMConstants::ACTION_DISTANCE && need_safety)
    best_step = computeBestStep(current_pos, current_yaw, local_pos);
  else
    best_step = local_pos;

  // Calculate target orientation from best step direction
  double target_yaw = std::atan2(best_step(1) - current_pos(1), best_step(0) - current_pos(0));
  return decideNextAction(current_yaw, target_yaw);
}

Vector2d ExplorationFSM::selectLocalTarget(
    const Vector2d& current_pos, const vector<Vector2d>& path, const double& local_distance)
{
  Vector2d target_pos = path.back();

  // Find the closest path point to current position as starting search index
  int start_path_id = 0;
  double min_dist = std::numeric_limits<double>::max();
  for (int i = 0; i < (int)path.size() - 1; i++) {
    Eigen::Vector2d pos = path[i];
    if ((pos - current_pos).norm() < min_dist) {
      min_dist = (pos - current_pos).norm();
      start_path_id = i + 1;
    }
  }

  // Select a local target position within the specified distance
  double len = (path[start_path_id] - current_pos).norm();
  for (int i = start_path_id + 1; i < (int)path.size(); i++) {
    len += (path[i] - path[i - 1]).norm();
    if (len > local_distance && (current_pos - path[i - 1]).norm() > 0.30) {
      target_pos = path[i - 1];
      break;
    }
  }

  return target_pos;
}

Vector2d ExplorationFSM::computeBestStep(
    const Vector2d& current_pos, double current_yaw, const Vector2d& target_pos)
{
  Vector2d best_step = target_pos;

  double min_cost = std::numeric_limits<double>::max();
  for (auto step : fp_->action_steps_) {
    double cost = computeActionTotalCost(current_pos, current_yaw, target_pos, step);
    if (cost < min_cost) {
      best_step = current_pos + step;
      min_cost = cost;
    }
  }

  return best_step;
}

bool ExplorationFSM::planMast3rPath(
    const Vector2d& current_pos, const Vector2d& target_pos, bool is_replan)
{
  if (is_replan) {
    if (fd_->mast3r_path_replan_count_ >= FSMConstants::MAST3R_LOCAL_GOAL_MAX_REPLANS)
      return false;
    ++fd_->mast3r_path_replan_count_;
  }

  expl_manager_->path_finder_->reset();
  const int search_result = expl_manager_->path_finder_->astarSearch(current_pos, target_pos,
      0.20, 0.20, Astar2D::SAFETY_MODE::NORMAL, false);
  vector<Vector2d> path;
  if (search_result != Astar2D::REACH_END ||
      (path = expl_manager_->path_finder_->getPath()).empty()) {
    const auto map = expl_manager_->sdf_map_;
    const MapPointDebug start_debug = inspectMapPoint(map, current_pos);
    const MapPointDebug target_debug = inspectMapPoint(map, target_pos);
    const StrictStepDebug first_step_debug = inspectStrictFirstSteps(
        map, current_pos, expl_manager_->path_finder_->generateSteps(current_pos));
    std::ostringstream debug;
    debug << std::fixed << std::setprecision(3)
          << "event=mast3r_astar_failure"
          << " phase=" << (is_replan ? "replan" : "initial")
          << " result=" << search_result
          << " termination="
          << expl_manager_->path_finder_->getLastSearchTerminationName()
          << " duration_ms="
          << 1000.0 * expl_manager_->path_finder_->getLastSearchDuration()
          << " nodes=" << expl_manager_->path_finder_->getLastUsedNodeNum()
          << " iterations=" << expl_manager_->path_finder_->getLastIterationNum()
          << " start_x=" << current_pos(0)
          << " start_y=" << current_pos(1)
          << " start_ix=" << start_debug.index(0)
          << " start_iy=" << start_debug.index(1)
          << " start_in_map=" << int(start_debug.in_map)
          << " start_occ=" << occupancyStateName(start_debug.occupancy)
          << " start_occ_id=" << start_debug.occupancy
          << " start_inflated=" << start_debug.inflated
          << " start_esdf=" << start_debug.esdf
          << " target_x=" << target_pos(0)
          << " target_y=" << target_pos(1)
          << " target_ix=" << target_debug.index(0)
          << " target_iy=" << target_debug.index(1)
          << " target_in_map=" << int(target_debug.in_map)
          << " target_occ=" << occupancyStateName(target_debug.occupancy)
          << " target_occ_id=" << target_debug.occupancy
          << " target_inflated=" << target_debug.inflated
          << " target_esdf=" << target_debug.esdf
          << " first_step_safe=" << first_step_debug.safe
          << " first_step_out=" << first_step_debug.out_of_map
          << " first_step_unknown=" << first_step_debug.unknown
          << " first_step_occupied=" << first_step_debug.occupied
          << " first_step_inflated=" << first_step_debug.inflated;
    ROS_WARN("[MASt3R_ASTAR_DEBUG] %s", debug.str().c_str());
    std_msgs::String debug_msg;
    debug_msg.data = debug.str();
    mast3r_debug_pub_.publish(debug_msg);
    fd_->mast3r_path_active_ = false;
    return false;
  }

  vector<Vector2d> segment_path;
  segment_path.reserve(path.size());
  segment_path.push_back(path.front());
  double segment_length = 0.0;
  for (size_t idx = 1; idx < path.size(); ++idx) {
    segment_length += (path[idx] - path[idx - 1]).norm();
    segment_path.push_back(path[idx]);
    if (segment_length + 1e-6 >= FSMConstants::MAST3R_EXECUTION_HORIZON_DISTANCE)
      break;
  }

  const Vector2d segment_target = segment_path.back();
  const bool segment_ends_before_final_target =
      (target_pos - segment_target).norm() >= FSMConstants::MAST3R_LOCAL_GOAL_REACH_DISTANCE;
  expl_manager_->ed_->next_best_path_ = segment_path;
  expl_manager_->ed_->next_pos_ = segment_target;
  fd_->mast3r_path_active_ = true;
  fd_->mast3r_segment_target_pos_ = segment_target;
  fd_->mast3r_segment_target_valid_ = segment_ends_before_final_target;
  ++fd_->mast3r_segment_count_;
  fd_->mast3r_last_pos_ = current_pos;
  fd_->mast3r_last_pos_valid_ = true;
  fd_->mast3r_forward_blocked_count_ = 0;
  ROS_INFO(
      "MASt3R strict A* %s: final_target=(%.3f, %.3f) full_nodes=%zu full_len=%.3f "
      "segment=%d segment_target=(%.3f, %.3f) segment_nodes=%zu segment_len=%.3f "
      "replan=%d/%d",
      is_replan ? "replanned" : "planned", target_pos(0), target_pos(1), path.size(),
      Astar2D::pathLength(path), fd_->mast3r_segment_count_, segment_target(0),
      segment_target(1), segment_path.size(), Astar2D::pathLength(segment_path),
      fd_->mast3r_path_replan_count_,
      FSMConstants::MAST3R_LOCAL_GOAL_MAX_REPLANS);
  return true;
}

void ExplorationFSM::markMast3rForwardCollision(
    const Vector2d& current_pos, double current_yaw)
{
  int marked_cells = 0;
  for (int multiplier = 1; multiplier <= 2; ++multiplier) {
    const double distance = FSMConstants::FORWARD_DISTANCE * multiplier;
    const Vector2d blocked_pos(
        current_pos(0) + distance * std::cos(current_yaw),
        current_pos(1) + distance * std::sin(current_yaw));
    if (!expl_manager_->sdf_map_->isInMap(blocked_pos))
      continue;
    expl_manager_->sdf_map_->setForceOccGrid(blocked_pos);
    ++marked_cells;
  }
  ROS_WARN(
      "MASt3R forward collision feedback: marked %d occupied cells ahead of "
      "(%.3f, %.3f) at yaw=%.2f deg before strict A* replanning.",
      marked_cells, current_pos(0), current_pos(1), current_yaw * 180.0 / M_PI);
}

int ExplorationFSM::planMast3rPathAction(
    const Vector2d& current_pos, double current_yaw, const vector<Vector2d>& path)
{
  if (path.empty())
    return ACTION::TURN_LEFT;

  const int action = planNextBestAction(current_pos, current_yaw, path, true);
  ROS_INFO_THROTTLE(0.5,
      "MASt3R clearance-aware path follow: local_target=(%.3f, %.3f) "
      "distance=%.3f action=%d",
      fd_->local_pos_(0), fd_->local_pos_(1), (fd_->local_pos_ - current_pos).norm(), action);
  return action;
}

bool ExplorationFSM::buildMast3rLocalTarget(
    const Vector2d& current_pos, double current_yaw, Vector2d& target_pos, double& target_yaw)
{
  if (!fd_->mast3r_hint_active_ || fd_->mast3r_allow_stop_) return false;

  Vector2d forward_dir(std::cos(current_yaw), std::sin(current_yaw));
  Vector2d left_dir(-std::sin(current_yaw), std::cos(current_yaw));

  Vector2d local_offset =
      fd_->mast3r_forward_error_ * forward_dir + fd_->mast3r_lateral_error_ * left_dir;
  if (!std::isfinite(local_offset(0)) || !std::isfinite(local_offset(1)))
    return false;
  target_pos = current_pos + local_offset;

  // MASt3R yaw_error sign is aligned with Python's suggested_action:
  // positive -> turn right, negative -> turn left.
  target_yaw = current_yaw - fd_->mast3r_yaw_error_deg_ * M_PI / 180.0;
  wrapAngle(target_yaw);
  return true;
}

void ExplorationFSM::clearMast3rLocalGoal()
{
  fd_->mast3r_goal_active_ = false;
  fd_->mast3r_path_active_ = false;
  fd_->mast3r_segment_target_pos_ = Eigen::Vector2d::Zero();
  fd_->mast3r_segment_target_valid_ = false;
  fd_->mast3r_segment_count_ = 0;
  fd_->mast3r_goal_age_ = 0;
  fd_->mast3r_target_pos_ = Eigen::Vector2d(0, 0);
  fd_->mast3r_target_yaw_ = 0.0;
  fd_->mast3r_last_pos_ = Eigen::Vector2d::Zero();
  fd_->mast3r_last_pos_valid_ = false;
  fd_->mast3r_forward_blocked_count_ = 0;
  fd_->mast3r_path_replan_count_ = 0;
}

void ExplorationFSM::publishMast3rRefineStatus(int status)
{
  std_msgs::Int32 msg;
  msg.data = status;
  mast3r_refine_status_pub_.publish(msg);
}

// Compute total cost of taking a step towards target
// Considers distance-to-target, movement efficiency, and collision safety
double ExplorationFSM::computeActionTotalCost(const Vector2d& current_pos, double current_yaw,
    const Vector2d& target_pos, const Vector2d& step)
{
  const double traget_weight = FSMConstants::TARGET_WEIGHT;
  const double traget_close_weight1 = FSMConstants::TARGET_CLOSE_WEIGHT_1;
  const double traget_close_weight2 = FSMConstants::TARGET_CLOSE_WEIGHT_2;
  const double safety_weight = FSMConstants::SAFETY_WEIGHT;
  double cost = 0.0;

  // Distance-to-target cost
  Vector2d step_pos = current_pos + step;
  double target_cost = traget_weight * (step_pos - target_pos).norm();

  // Change-in-distance cost (negative if moving closer)
  double target_close_cost = (step_pos - target_pos).norm() - (current_pos - target_pos).norm();
  if (target_close_cost > 0)
    target_close_cost *= traget_close_weight1;
  else
    target_close_cost *= traget_close_weight2;

  // Safety distance cost
  double safety_cost = safety_weight * computeActionSafetyCost(current_pos, step);

  cost += target_cost + target_close_cost + safety_cost;
  return cost;
}

// Compute safety cost along the step using SDF distance to obstacles
// Returns higher cost for paths that go too close to obstacles
double ExplorationFSM::computeActionSafetyCost(const Vector2d& current_pos, const Vector2d& step)
{
  const double min_safe_distance = FSMConstants::MIN_SAFE_DISTANCE;
  const double sample_num = FSMConstants::SAMPLE_NUM;

  Vector2d dir = step;
  double len = dir.norm();
  dir.normalize();

  double safety_cost = 0.0;
  for (double l = len / sample_num; l < len; l += len / sample_num) {
    Vector2d ckpt = current_pos + l * dir;
    Vector2d grad;
    double dist_to_occ = expl_manager_->sdf_map_->getDistWithGrad(ckpt, grad);
    if (dist_to_occ < min_safe_distance)
      safety_cost += 1 / (dist_to_occ + 1e-2);
  }

  return safety_cost;
}

// Decide whether to turn or move forward based on yaw difference
// Uses action angle threshold to determine if orientation adjustment is needed
int ExplorationFSM::decideNextAction(double current_yaw, double target_yaw)
{
  wrapAngle(target_yaw);
  wrapAngle(current_yaw);
  double yaw_diff = target_yaw - current_yaw;
  wrapAngle(yaw_diff);

  int next_action;
  if (std::fabs(yaw_diff) > FSMConstants::ACTION_ANGLE / 1.9) {
    if (yaw_diff > 0)
      next_action = ACTION::TURN_LEFT;
    else
      next_action = ACTION::TURN_RIGHT;
  }
  else
    next_action = ACTION::MOVE_FORWARD;

  return next_action;
}

bool ExplorationFSM::alignMast3rTargetYaw(double current_yaw)
{
  double target_yaw = fd_->mast3r_target_yaw_;
  wrapAngle(target_yaw);
  wrapAngle(current_yaw);
  double yaw_diff = target_yaw - current_yaw;
  wrapAngle(yaw_diff);

  const double yaw_threshold = FSMConstants::MAST3R_FINE_YAW_HALF_ANGLE;
  if (std::fabs(yaw_diff) <= yaw_threshold) {
    return true;
  }

  fd_->newest_action_ = yaw_diff > 0.0 ? ACTION::TURN_LEFT : ACTION::TURN_RIGHT;
  publishMast3rRefineStatus(2);
  ROS_INFO_THROTTLE(0.5,
      "MASt3R local target position reached; align yaw before reporting arrival: "
      "yaw_diff=%.2f deg, target_yaw=%.2f deg, current_yaw=%.2f deg, action=%s, fine_turn=15.00 deg",
      yaw_diff * 180.0 / M_PI, target_yaw * 180.0 / M_PI, current_yaw * 180.0 / M_PI,
      fd_->newest_action_ == ACTION::TURN_LEFT ? "turn_left" : "turn_right");
  return false;
}

void ExplorationFSM::visualize()
{
  auto ed_ptr = expl_manager_->ed_;

  // Lambda function to convert 2D vectors to 3D for visualization
  auto vec2dTo3d = [](const vector<Eigen::Vector2d>& vec2d, double z = 0.15) {
    vector<Eigen::Vector3d> vec3d;
    for (auto v : vec2d) vec3d.push_back(Vector3d(v(0), v(1), z));
    return vec3d;
  };

  // Draw frontier
  static int last_ftr2d_num = 0;
  for (int i = 0; i < (int)ed_ptr->frontiers_.size(); ++i) {
    visualization_->drawCubes(vec2dTo3d(ed_ptr->frontiers_[i]), fp_->vis_scale_,
        visualization_->getColor(double(i) / ed_ptr->frontiers_.size(), 1.0), "frontier", i, 4);
  }
  for (int i = ed_ptr->frontiers_.size(); i < last_ftr2d_num; ++i) {
    visualization_->drawCubes({}, fp_->vis_scale_, Vector4d(0, 0, 0, 1), "frontier", i, 4);
  }
  last_ftr2d_num = ed_ptr->frontiers_.size();

  // Draw dormant frontier
  static int last_dftr2d_num = 0;
  for (int i = 0; i < (int)ed_ptr->dormant_frontiers_.size(); ++i) {
    visualization_->drawCubes(vec2dTo3d(ed_ptr->dormant_frontiers_[i]), fp_->vis_scale_,
        Vector4d(0, 0, 0, 1), "dormant_frontier", i, 4);
  }
  for (int i = ed_ptr->dormant_frontiers_.size(); i < last_dftr2d_num; ++i) {
    visualization_->drawCubes({}, fp_->vis_scale_, Vector4d(0, 0, 0, 1), "dormant_frontier", i, 4);
  }
  last_dftr2d_num = ed_ptr->dormant_frontiers_.size();

  // Draw object
  // static int last_obj_num = 0;
  // for (int i = 0; i < (int)ed_ptr->objects_.size(); ++i) {
  //   visualization_->drawCubes(vec2dTo3d(ed_ptr->objects_[i]), fp_->vis_scale_,
  //       visualization_->getColor(double(i) / ed_ptr->objects_.size(), 1.0), "object", i, 4);
  // }
  // for (int i = ed_ptr->objects_.size(); i < last_obj_num; ++i) {
  //   visualization_->drawCubes({}, fp_->vis_scale_, Vector4d(0, 0, 0, 1), "object", i, 4);
  // }
  // last_obj_num = ed_ptr->objects_.size();

  static int last_obj_num = 0;
  for (int i = 0; i < (int)ed_ptr->objects_.size(); ++i) {
    int label = ed_ptr->object_labels_[i];
    visualization_->drawCubes(vec2dTo3d(ed_ptr->objects_[i]), fp_->vis_scale_,
        visualization_->getColor(double(label) / 5.0, 1.0), "object", i, 4);
  }
  for (int i = ed_ptr->objects_.size(); i < last_obj_num; ++i) {
    visualization_->drawCubes({}, fp_->vis_scale_, Vector4d(0, 0, 0, 1), "object", i, 4);
  }
  last_obj_num = ed_ptr->objects_.size();

  // Draw next best path
  visualization_->drawLines(vec2dTo3d(ed_ptr->next_best_path_), fp_->vis_scale_,
      Vector4d(1, 0.2, 0.2, 1), "next_path", 1, 6);

  // Draw the actual global endpoint separately from the 0.8 m local lookahead marker.
  vector<Vector2d> global_targets;
  if (!ed_ptr->next_best_path_.empty())
    global_targets.push_back(ed_ptr->next_pos_);
  const Vector4d global_target_color = expl_manager_->hasLockedObjectViewpoint()
                                           ? Vector4d(0.1, 1.0, 0.2, 1.0)
                                           : Vector4d(1.0, 0.8, 0.0, 1.0);
  visualization_->drawSpheres(vec2dTo3d(global_targets, 0.24), fp_->vis_scale_ * 5,
      global_target_color, "global_next_target", 0, 6);

  vector<Vector2d> locked_object_centers;
  vector<Vector2d> locked_object_viewpoints;
  vector<Vector2d> locked_object_link;
  if (expl_manager_->hasLockedObjectViewpoint()) {
    locked_object_centers.push_back(expl_manager_->getLockedObjectCenter());
    locked_object_viewpoints.push_back(expl_manager_->getLockedObjectViewpoint());
    locked_object_link.push_back(expl_manager_->getLockedObjectCenter());
    locked_object_link.push_back(expl_manager_->getLockedObjectViewpoint());
  }
  visualization_->drawSpheres(vec2dTo3d(locked_object_centers, 0.28), fp_->vis_scale_ * 5,
      Vector4d(1.0, 0.1, 0.1, 1.0), "locked_object_center", 0, 6);
  visualization_->drawSpheres(vec2dTo3d(locked_object_viewpoints, 0.28), fp_->vis_scale_ * 5,
      Vector4d(0.0, 0.9, 1.0, 1.0), "locked_object_viewpoint", 0, 6);
  visualization_->drawLines(vec2dTo3d(locked_object_link, 0.25), fp_->vis_scale_ * 0.8,
      Vector4d(0.0, 0.9, 1.0, 0.9), "locked_object_link", 0, 6);

  // Draw next local point
  vector<Vector2d> local_points;
  local_points.push_back(fd_->local_pos_);
  visualization_->drawSpheres(vec2dTo3d(local_points), fp_->vis_scale_ * 3,
      Vector4d(0.2, 0.2, 1.0, 1), "local_point", 1, 6);

  visualization_->drawLines(vec2dTo3d(ed_ptr->tsp_tour_), fp_->vis_scale_ / 1.25,
      Vector4d(0.2, 1, 0.2, 1), "tsp_tour", 0, 6);

  visualization_->drawSpheres(vec2dTo3d(fd_->traveled_path_), fp_->vis_scale_ * 1.5,
      Vector4d(2.0 / 255.0, 111.0 / 255.0, 197.0 / 255.0, 1), "traveled_path", 1, 6);
}

void ExplorationFSM::clearVisMarker()
{
  auto ed_ptr = expl_manager_->ed_;
  for (int i = 0; i < 500; ++i) {
    visualization_->drawCubes({}, fp_->vis_scale_, Vector4d(0, 0, 0, 1), "frontier", i, 4);
    visualization_->drawCubes({}, fp_->vis_scale_, Vector4d(0, 0, 0, 1), "dormant_frontier", i, 4);
    visualization_->drawCubes({}, fp_->vis_scale_, Vector4d(0, 0, 0, 1), "object", i, 4);
  }

  visualization_->drawLines({}, fp_->vis_scale_, Vector4d(0, 0, 1, 1), "next_path", 1, 6);
  visualization_->drawSpheres(
      {}, fp_->vis_scale_, Vector4d(0, 0, 0, 1), "global_next_target", 0, 6);
  visualization_->drawSpheres(
      {}, fp_->vis_scale_, Vector4d(0, 0, 0, 1), "locked_object_center", 0, 6);
  visualization_->drawSpheres(
      {}, fp_->vis_scale_, Vector4d(0, 0, 0, 1), "locked_object_viewpoint", 0, 6);
  visualization_->drawLines(
      {}, fp_->vis_scale_, Vector4d(0, 0, 0, 1), "locked_object_link", 0, 6);
}

bool ExplorationFSM::updateFrontierAndObject(bool enable_dormant)
{
  bool change_flag = false;
  auto frt_map = expl_manager_->frontier_map2d_;
  auto obj_map = expl_manager_->object_map2d_;
  auto ed = expl_manager_->ed_;
  Eigen::Vector2d start_pos2d = Eigen::Vector2d(fd_->start_pt_(0), fd_->start_pt_(1));

  change_flag = frt_map->isAnyFrontierChanged();
  frt_map->searchFrontiers();
  change_flag |= frt_map->dormantSeenFrontiers(start_pos2d, fd_->start_yaw_(0));
  frt_map->getFrontiers(ed->frontiers_, ed->frontier_averages_);
  frt_map->getDormantFrontiers(ed->dormant_frontiers_, ed->dormant_frontier_averages_);
  obj_map->getObjects(ed->objects_, ed->object_averages_, ed->object_labels_);

  return change_flag;
}

// Receive Habitat state messages
void ExplorationFSM::habitatStateCallback(const std_msgs::Int32ConstPtr& msg)
{
  if (msg->data == HABITAT_STATE::ACTION_FINISH && state_ == ROS_STATE::WAIT_ACTION_FINISH)
    transitState(PLAN_ACTION, "Habitat Finish Action");
  if (msg->data == HABITAT_STATE::EPISODE_FINISH) {
    clearMast3rLocalGoal();
    init(nh_);
  }
  return;
}

// Periodically update frontiers and visualize in idle states
void ExplorationFSM::frontierCallback(const ros::TimerEvent& e)
{
  if (state_ != ROS_STATE::WAIT_TRIGGER && state_ != ROS_STATE::FINISH)
    return;

  updateFrontierAndObject();
  visualize();
}

// Receive user trigger to start exploration
void ExplorationFSM::triggerCallback(const geometry_msgs::PoseStampedConstPtr& msg)
{
  if (state_ != ROS_STATE::WAIT_TRIGGER)
    return;
  fd_->trigger_ = true;
  cout << "Triggered!" << endl;
  transitState(PLAN_ACTION, "triggerCallback");
}

// Receive robot odometry and update traveled path + marker
void ExplorationFSM::odometryCallback(const nav_msgs::OdometryConstPtr& msg)
{
  fd_->odom_pos_(0) = msg->pose.pose.position.x;
  fd_->odom_pos_(1) = msg->pose.pose.position.y;
  fd_->odom_pos_(2) = msg->pose.pose.position.z;

  fd_->odom_orient_.w() = msg->pose.pose.orientation.w;
  fd_->odom_orient_.x() = msg->pose.pose.orientation.x;
  fd_->odom_orient_.y() = msg->pose.pose.orientation.y;
  fd_->odom_orient_.z() = msg->pose.pose.orientation.z;

  Eigen::Vector3d rot_x = fd_->odom_orient_.toRotationMatrix().block<3, 1>(0, 0);
  fd_->odom_yaw_ = atan2(rot_x(1), rot_x(0));

  fd_->have_odom_ = true;

  Vector2d odom_pos2d = Vector2d(fd_->odom_pos_(0), fd_->odom_pos_(1));
  if (fd_->traveled_path_.empty())
    fd_->traveled_path_.push_back(odom_pos2d);
  else if ((fd_->traveled_path_.back() - odom_pos2d).norm() > 1e-2)
    fd_->traveled_path_.push_back(odom_pos2d);

  publishRobotMarker();
}

void ExplorationFSM::publishRobotMarker()
{
  const double robot_height = FSMConstants::ROBOT_HEIGHT;
  const double robot_radius = FSMConstants::ROBOT_RADIUS;

  // Create robot body cylinder marker
  visualization_msgs::Marker robot_marker;
  robot_marker.header.frame_id = "world";
  robot_marker.header.stamp = ros::Time::now();
  robot_marker.ns = "robot_position";
  robot_marker.id = 0;
  robot_marker.type = visualization_msgs::Marker::CYLINDER;
  robot_marker.action = visualization_msgs::Marker::ADD;

  // Set cylinder position
  robot_marker.pose.position.x = fd_->odom_pos_(0);
  robot_marker.pose.position.y = fd_->odom_pos_(1);
  robot_marker.pose.position.z = fd_->odom_pos_(2) + robot_height / 2.0;

  // Set cylinder orientation
  robot_marker.pose.orientation.x = fd_->odom_orient_.x();
  robot_marker.pose.orientation.y = fd_->odom_orient_.y();
  robot_marker.pose.orientation.z = fd_->odom_orient_.z();
  robot_marker.pose.orientation.w = fd_->odom_orient_.w();

  // Set cylinder dimensions
  robot_marker.scale.x = robot_radius * 2;  // Diameter
  robot_marker.scale.y = robot_radius * 2;  // Diameter
  robot_marker.scale.z = robot_height;      // Height

  // Set cylinder color (blue)
  robot_marker.color.r = 50.0 / 255.0;
  robot_marker.color.g = 50.0 / 255.0;
  robot_marker.color.b = 255.0 / 255.0;
  robot_marker.color.a = 1.0;

  // Create direction arrow marker
  visualization_msgs::Marker arrow_marker;
  arrow_marker.header.frame_id = "world";
  arrow_marker.header.stamp = ros::Time::now();
  arrow_marker.ns = "robot_direction";
  arrow_marker.id = 1;
  arrow_marker.type = visualization_msgs::Marker::ARROW;
  arrow_marker.action = visualization_msgs::Marker::ADD;

  // Set arrow position
  arrow_marker.pose.position.x = fd_->odom_pos_(0);
  arrow_marker.pose.position.y = fd_->odom_pos_(1);
  arrow_marker.pose.position.z = fd_->odom_pos_(2) + robot_height;

  // Set arrow orientation
  arrow_marker.pose.orientation.x = fd_->odom_orient_.x();
  arrow_marker.pose.orientation.y = fd_->odom_orient_.y();
  arrow_marker.pose.orientation.z = fd_->odom_orient_.z();
  arrow_marker.pose.orientation.w = fd_->odom_orient_.w();

  // Set arrow dimensions
  arrow_marker.scale.x = robot_radius + 0.13;  // Arrow length
  arrow_marker.scale.y = 0.08;                 // Arrow width
  arrow_marker.scale.z = 0.08;                 // Arrow thickness

  // Set arrow color (green)
  arrow_marker.color.r = 10.0 / 255.0;
  arrow_marker.color.g = 255.0 / 255.0;
  arrow_marker.color.b = 10.0 / 255.0;
  arrow_marker.color.a = 1.0;

  // Publish both markers
  robot_marker_pub_.publish(robot_marker);
  robot_marker_pub_.publish(arrow_marker);
}

void ExplorationFSM::confidenceThresholdCallback(const std_msgs::Float64ConstPtr& msg)
{
  if (fd_->have_confidence_)
    return;
  fd_->have_confidence_ = true;
  expl_manager_->sdf_map_->object_map2d_->setConfidenceThreshold(msg->data);
}

void ExplorationFSM::instanceStopGateCallback(const std_msgs::Int32ConstPtr& msg)
{
  const bool enabled = msg->data != 0;
  if (fd_->instance_stop_gate_enabled_ == enabled)
    return;

  fd_->instance_stop_gate_enabled_ = enabled;
}

void ExplorationFSM::verifiedApproachTargetCallback(
    const plan_env::MultipleMasksWithConfidenceConstPtr& msg)
{
  if (msg->point_clouds.empty()) {
    expl_manager_->setVerifiedApproachState(0);
    return;
  }

  pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> object_cloud(
      new pcl::PointCloud<pcl::PointXYZ>());
  pcl::fromROSMsg(msg->point_clouds.front(), *object_cloud);
  if (object_cloud->points.empty()) {
    expl_manager_->setVerifiedApproachState(0);
    return;
  }

  expl_manager_->setVerifiedApproachCloud(object_cloud);
  expl_manager_->setVerifiedApproachState(1);
}

void ExplorationFSM::resumeExplorationCallback(const std_msgs::Int32ConstPtr& msg)
{
  if (msg->data == 0 || state_ != ROS_STATE::FINISH)
    return;

  // NO_FRONTIER and STUCKING are genuine planner terminal outcomes and must remain terminal.
  // Python consumes those outcomes directly instead of sending a resume request.
  if (fd_->final_result_ == FINAL_RESULT::NO_FRONTIER ||
      fd_->final_result_ == FINAL_RESULT::STUCKING) {
    ROS_WARN("[RESUME_EXPLORATION] ignore request for terminal planner result=%d",
        fd_->final_result_);
    return;
  }

  clearMast3rLocalGoal();
  expl_manager_->setVerifiedApproachState(0);
  fd_->have_finished_ = false;
  fd_->escape_stucking_flag_ = false;
  fd_->escape_stucking_count_ = 0;
  fd_->stucking_action_count_ = 0;
  fd_->stucking_next_pos_count_ = 0;
  fd_->final_result_ = FINAL_RESULT::EXPLORE;
  transitState(ROS_STATE::PLAN_ACTION, "resume exploration after verification failure");
  ROS_WARN("[RESUME_EXPLORATION] revive planner after an unverified FINISH");
  exec_timer_.start();
}

void ExplorationFSM::mast3rHintCallback(const std_msgs::Float32MultiArrayConstPtr& msg)
{
  if (msg->data.size() < 7) {
    fd_->mast3r_hint_active_ = false;
    fd_->mast3r_allow_stop_ = false;
    fd_->mast3r_yaw_error_deg_ = 0.0;
    fd_->mast3r_forward_error_ = 0.0;
    fd_->mast3r_lateral_error_ = 0.0;
    fd_->mast3r_transl_error_ = 0.0;
    fd_->mast3r_depth_error_ = 0.0;
    return;
  }

  const bool hint_active = msg->data[0] > 0.5f;
  const bool allow_stop = msg->data[1] > 0.5f;
  if (hint_active || allow_stop) {
    expl_manager_->setVerifiedApproachState(0);
    // MASt3R owns motion while active, but a failed refinement must resume the same fixed
    // DINO observation target rather than return to frontier selection.
    if (expl_manager_->hasLockedObjectViewpoint())
      ROS_INFO("[OBJECT_VIEWPOINT_LOCK] suspend DINO inspection for MASt3R refinement");
  }
  const bool refine_from_finish = state_ == ROS_STATE::FINISH && hint_active && !allow_stop;
  if (!refine_from_finish && fd_->mast3r_goal_active_ && hint_active && !allow_stop)
    return;

  fd_->mast3r_hint_active_ = hint_active;
  fd_->mast3r_allow_stop_ = allow_stop;
  if (!hint_active && !allow_stop) {
    clearMast3rLocalGoal();
    return;
  }

  fd_->mast3r_yaw_error_deg_ = msg->data[2];
  fd_->mast3r_forward_error_ = msg->data[3];
  fd_->mast3r_lateral_error_ = msg->data[4];
  fd_->mast3r_transl_error_ = msg->data[5];
  fd_->mast3r_depth_error_ = msg->data[6];

  if (fd_->mast3r_allow_stop_) {
    clearMast3rLocalGoal();
    return;
  }

  if (refine_from_finish) {
    clearMast3rLocalGoal();
    fd_->have_finished_ = false;
    fd_->escape_stucking_flag_ = false;
    fd_->escape_stucking_count_ = 0;
    fd_->stucking_action_count_ = 0;
    fd_->stucking_next_pos_count_ = 0;
    fd_->final_result_ = FINAL_RESULT::SEARCH_OBJECT;
    transitState(ROS_STATE::PLAN_ACTION, "MASt3R refine hint");
    ROS_WARN(
        "[MASt3R_HINT] revive planner from FINISH with a locked local target; reset stale "
        "stuck counters before A* planning.");
    exec_timer_.start();
  }
}

// Transition FSM state and log the change
void ExplorationFSM::transitState(ROS_STATE new_state, string pos_call)
{
  int pre_s = int(state_);
  state_ = new_state;
  cout << "[ " + pos_call + "]: from " + fd_->state_str_[pre_s] + " to " +
              fd_->state_str_[int(new_state)]
       << endl;
}
}  // namespace apexnav_planner
