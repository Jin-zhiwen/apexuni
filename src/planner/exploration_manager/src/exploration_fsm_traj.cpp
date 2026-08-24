#include <exploration_manager/exploration_manager.h>
#include <exploration_manager/exploration_fsm_traj.h>
#include <exploration_manager/exploration_fsm_traj_logic.h>
#include <exploration_manager/exploration_data.h>
#include <vis_utils/planning_visualization.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <tf/transform_datatypes.h>

namespace apexnav_planner {

void ExplorationFSMReal::init(ros::NodeHandle& nh)
{
  nh_ = nh;
  fp_.reset(new FSMParam);
  fd_.reset(new FSMData);

  /* Initialize main modules */
  expl_manager_.reset(new ExplorationManager);
  expl_manager_->initialize(nh);
  visualization_.reset(new PlanningVisualization(nh));
  fp_->vis_scale_ = expl_manager_->sdf_map_->getResolution() * FSMConstantsReal::VIS_SCALE_FACTOR;

  state_ = RealFSM::State::INIT;

  // Load real-world specific parameters
  nh.param("fsm/replan_time", fp_->replan_time_, 0.2);
  nh.param("fsm/replan_traj_end_threshold", fp_->replan_traj_end_threshold_, 1.0);
  nh.param("fsm/replan_frontier_change_delay", fp_->replan_frontier_change_delay_, 0.5);
  nh.param("fsm/replan_timeout", fp_->replan_timeout_, 2.0);
  nh.param("fsm/local_target_distance", fp_->local_target_distance_, fp_->local_target_distance_);
  nh.param("fsm/local_target_corner_angle",
      local_target_corner_angle_, local_target_corner_angle_);
  nh.param("fsm/corner_replan_end_threshold",
      corner_replan_end_threshold_, corner_replan_end_threshold_);
  nh.param("fsm/rotation_before_translate_yaw_error",
      rotation_before_translate_yaw_error_, rotation_before_translate_yaw_error_);
  nh.param("fsm/rotation_tolerance", rotation_tolerance_, rotation_tolerance_);
  nh.param("fsm/rotation_settle_time", rotation_settle_time_, rotation_settle_time_);
  nh.param("fsm/rotation_timeout", rotation_timeout_, rotation_timeout_);
  nh.param("fsm/rotation_footprint_step", rotation_footprint_step_, rotation_footprint_step_);
  nh.param("fsm/plan_failure_retry_delay",
      plan_failure_retry_delay_, plan_failure_retry_delay_);
  nh.param("fsm/visualize_objects", visualize_object_markers_, false);
  nh.param("length", robot_marker_length_, robot_marker_length_);
  nh.param("width", robot_marker_width_, robot_marker_width_);
  nh.param("height", robot_marker_height_, robot_marker_height_);
  robot_marker_length_ = std::max(0.01, robot_marker_length_);
  robot_marker_width_ = std::max(0.01, robot_marker_width_);
  robot_marker_height_ = std::max(0.01, robot_marker_height_);
  nh.param("fsm/tracking_abort_distance", tracking_abort_distance_,
      FSMConstantsReal::DEFAULT_TRACKING_ABORT_DISTANCE);
  nh.param("fsm/tracking_replan_odom_distance", tracking_replan_odom_distance_, 0.20);
  nh.param("fsm/replan_on_frontier_change",
      replan_on_frontier_change_, replan_on_frontier_change_);
  local_target_corner_angle_ = std::max(
      0.0, std::min(local_target_corner_angle_, M_PI));
  corner_replan_end_threshold_ = std::max(0.0, corner_replan_end_threshold_);

  // Initialize status before ROS callbacks can observe the object.
  insinav_stop_verified_status_ = "PENDING";
  finish_wait_start_time_ = ros::Time(0);
  lightglue_verified_start_time_ = ros::Time(0);
  finish_completed_time_ = ros::Time(0);
  lightglue_close_stop_candidate_active_ = false;

  /* ROS Publisher */
  ros_state_pub_ = nh.advertise<std_msgs::Int32>("/ros/state", 10);
  expl_state_pub_ = nh.advertise<std_msgs::Int32>("/ros/expl_state", 10);
  expl_result_pub_ = nh.advertise<std_msgs::Int32>("/ros/expl_result", 10);
  robot_marker_pub_ = nh.advertise<visualization_msgs::Marker>("/robot", 10);

  // Real-world trajectory publishers
  poly_traj_pub_ = nh.advertise<trajectory_manager::PolyTraj>("/planning/trajectory", 10);
  stop_pub_ = nh.advertise<std_msgs::Empty>("/traj_server/stop", 10);
  target_angle_pub_ = nh.advertise<std_msgs::Float32>("/traj_server/target_angle", 10);

  /* ROS Subscriber */
  trigger_sub_ =
      nh.subscribe("/move_base_simple/goal", 10, &ExplorationFSMReal::triggerCallback, this);
  goal_sub_ = nh.subscribe("/initialpose", 10, &ExplorationFSMReal::goalCallback, this);
  odom_sub_ = nh.subscribe("/odom_world", 10, &ExplorationFSMReal::odometryCallback, this);
  confidence_threshold_sub_ = nh.subscribe(
      "/detector/confidence_threshold", 10, &ExplorationFSMReal::confidenceThresholdCallback, this);
  insinav_stop_verified_sub_ = nh.subscribe(
      "/insinav/stop_verified", 10, &ExplorationFSMReal::insinnavStopVerifiedCallback, this);

  /* ROS Timer */
  exec_timer_ = nh.createTimer(ros::Duration(FSMConstantsReal::EXEC_TIMER_DURATION),
      &ExplorationFSMReal::FSMCallback, this, false, false);
  frontier_timer_ = nh.createTimer(ros::Duration(FSMConstantsReal::FRONTIER_TIMER_DURATION),
      &ExplorationFSMReal::frontierCallback, this, false, false);
  safety_timer_ =
      nh.createTimer(ros::Duration(0.05), &ExplorationFSMReal::safetyCallback, this, false, false);

  exec_timer_.start();
  frontier_timer_.start();
  safety_timer_.start();

  ROS_INFO("[ExplorationFSMReal] Initialization complete.");
}

// Main FSM callback for real-world exploration
void ExplorationFSMReal::FSMCallback(const ros::TimerEvent& e)
{
  exec_timer_.stop();

  // Publish current state
  std_msgs::Int32 ros_state_msg;
  ros_state_msg.data = static_cast<int>(state_);
  ros_state_pub_.publish(ros_state_msg);

  switch (state_) {
    case RealFSM::State::INIT: {
      // For real-world manual navigation, odometry is mandatory, but the confidence
      // threshold from the INSiNav bridge should not block entering WAIT_TRIGGER.
      if (!fd_->have_odom_) {
        ROS_WARN_THROTTLE(1.0, "[Real] No odom.");
        exec_timer_.start();
        return;
      }
      if (!fd_->have_confidence_) {
        ROS_WARN_THROTTLE(2.0,
            "[Real] No target confidence threshold yet. Exploration trigger will wait, but manual goals are allowed.");
      }
      // Go to WAIT_TRIGGER when prerequisites are ready
      clearVisMarker();
      transitState(RealFSM::State::WAIT_TRIGGER, "FSM");
      break;
    }

    case RealFSM::State::WAIT_TRIGGER: {
      // Do nothing but wait for trigger
      ROS_WARN_THROTTLE(1.0, "[Real] Waiting for trigger...");
      break;
    }

    case RealFSM::State::FINISH: {
      fd_->static_state_ = true;
      if (!fd_->have_finished_) {
        if (finish_wait_start_time_.isZero()) {
          finish_wait_start_time_ = ros::Time::now();
        }

        // Check LightGlue verification status before completing
        if (insinav_stop_verified_status_ == "VERIFIED") {
          if (lightglue_verified_start_time_.isZero()) {
            lightglue_verified_start_time_ = ros::Time::now();
          }
          const double verified_hold_time =
              (ros::Time::now() - lightglue_verified_start_time_).toSec();
          if (shouldFinalizeVerifiedStop(insinav_stop_verified_status_, verified_hold_time,
                  FSMConstantsReal::LIGHTGLUE_VERIFIED_HOLD_TIME)) {
            fd_->have_finished_ = true;
            finish_completed_time_ = ros::Time::now();
            clearVisMarker();
            ROS_INFO("[ExplorationFSMReal] LightGlue stop gate VERIFIED held for %.1f s - Mission complete!",
                verified_hold_time);
          } else {
            ROS_WARN_THROTTLE(1.0,
                "[ExplorationFSMReal] Holding verified stop for stability: %.1f/%.1f s.",
                verified_hold_time, FSMConstantsReal::LIGHTGLUE_VERIFIED_HOLD_TIME);
          }
        } else if (insinav_stop_verified_status_ == "PENDING_FAR") {
          lightglue_verified_start_time_ = ros::Time(0);
          finish_wait_start_time_ = ros::Time(0);
          finish_goal_pos_valid_ = false;
          lightglue_close_stop_candidate_active_ = false;
          ROS_WARN("[ExplorationFSMReal] LightGlue target is verified but still far; resume normal planning.");
          transitState(RealFSM::State::PLAN_TRAJ, "LightGluePendingFar");
          exec_timer_.start();
          return;
        } else if (shouldHoldForLightGlueConfirmation(
                       insinav_stop_verified_status_, lightglue_close_stop_candidate_active_)) {
          lightglue_verified_start_time_ = ros::Time(0);
          if (insinav_stop_verified_status_ == "PENDING_CLOSE") {
            lightglue_close_stop_candidate_active_ = true;
          }
          const double wait_time = (ros::Time::now() - finish_wait_start_time_).toSec();
          if (wait_time > FSMConstantsReal::LIGHTGLUE_UNVERIFIED_REPLAN_WAIT) {
            finish_wait_start_time_ = ros::Time(0);
            finish_goal_pos_valid_ = false;
            lightglue_close_stop_candidate_active_ = false;
            ROS_WARN("[ExplorationFSMReal] LightGlue not VERIFIED after %.1f s (status=%s); resume planning.",
                wait_time, insinav_stop_verified_status_.c_str());
            transitState(RealFSM::State::PLAN_TRAJ, "LightGlueNotVerified");
            exec_timer_.start();
            return;
          } else {
            ROS_WARN_THROTTLE(1.0,
                "[ExplorationFSMReal] Holding position for LightGlue verification (status=%s)...",
                insinav_stop_verified_status_.c_str());
          }
        } else if (insinav_stop_verified_status_ == "PENDING") {
          lightglue_verified_start_time_ = ros::Time(0);
          finish_wait_start_time_ = ros::Time(0);
          finish_goal_pos_valid_ = false;
          lightglue_close_stop_candidate_active_ = false;
          fd_->have_finished_ = true;
          finish_completed_time_ = ros::Time::now();
          clearVisMarker();
          ROS_WARN("[ExplorationFSMReal] No frontier remains while LightGlue is still PENDING; stop replanning to avoid a FINISH/PLAN loop.");
        } else if (insinav_stop_verified_status_ == "NO_GOAL_IMAGE") {
          // No goal image available, allow finish anyway
          fd_->have_finished_ = true;
          clearVisMarker();
          ROS_WARN("[ExplorationFSMReal] No goal image - allow finish anyway");
        } else {
          ROS_WARN_THROTTLE(1.0, "[ExplorationFSMReal] Unknown verification status: %s",
                            insinav_stop_verified_status_.c_str());
        }
      }
      if (shouldReportFinishExploration(fd_->have_finished_)) {
        ROS_WARN_THROTTLE(1.0, "[Real] Finish exploration!");
      }
      break;
    }

    case RealFSM::State::PLAN_TRAJ: {
      if (!planning_retry_after_.isZero() && ros::Time::now() < planning_retry_after_) {
        break;
      }
      planning_retry_after_ = ros::Time(0);

      // Plan trajectory based on current state
      LocalTrajectory* active_traj =
          expl_manager_->gcopter_ ? &expl_manager_->gcopter_->local_trajectory_ : nullptr;
      const bool have_active_traj =
          active_traj != nullptr && active_traj->duration > 1.0e-3 && !fd_->static_state_;
      double current_tracking_error = 0.0;
      double active_time_to_end = std::numeric_limits<double>::infinity();
      if (have_active_traj) {
        const double elapsed = (ros::Time::now() - active_traj->start_time).toSec();
        active_time_to_end = active_traj->duration - elapsed;
        const double t_track = std::min(std::max(elapsed, 0.0), active_traj->duration);
        current_tracking_error =
            (active_traj->traj.getPos(t_track).head(2) - fd_->odom_pos_.head(2)).norm();
      }

      // Keep using the predicted moving state while the active trajectory can
      // cover the configured future start delay. Fall back to odometry only
      // when there is genuinely not enough trajectory left for that handoff.
      const bool insufficient_handoff_time = have_active_traj &&
          active_time_to_end <= fp_->replan_time_ + 0.05;
      plan_from_odom_ = shouldUseOdometryStartForReplan(
          fd_->static_state_ || !have_active_traj,
          current_tracking_error,
          tracking_replan_odom_distance_) || insufficient_handoff_time;

      if (plan_from_odom_) {
        // Robot is static, use current odometry
        fd_->start_pt_ = fd_->odom_pos_;
        fd_->start_vel_ = fd_->odom_vel_;
        fd_->start_yaw_(0) = fd_->odom_yaw_;
        fd_->start_yaw_(1) = fd_->start_yaw_(2) = 0.0;
        if (have_active_traj) {
          if (insufficient_handoff_time &&
              current_tracking_error <= tracking_replan_odom_distance_) {
            ROS_INFO("[Real] Replan handoff from odom: active trajectory has %.2f s remaining; "
                     "new trajectory will start immediately.", active_time_to_end);
          }
          else {
            ROS_WARN("[Real] Replan from odom because tracking error %.2f exceeds %.2f.",
                current_tracking_error, tracking_replan_odom_distance_);
          }
        }
      }
      else {
        // Robot is moving, predict future state for smooth replanning
        LocalTrajectory* info = active_traj;
        double t_plan = (ros::Time::now() - info->start_time).toSec() + fp_->replan_time_;
        t_plan = min(t_plan, info->duration);

        Eigen::Vector3d cur_pos = info->traj.getPos(t_plan);
        Eigen::Vector3d cur_vel = info->traj.getVel(t_plan);
        Eigen::Vector3d cur_acc = info->traj.getAcc(t_plan);
        double cur_yaw = atan2(cur_vel(1), cur_vel(0));

        // Calculate yaw rate from acceleration
        Eigen::Matrix2d B_h;
        B_h << 0, -1.0, 1.0, 0;
        Eigen::Vector2d cur_vel_2d = cur_vel.head(2);
        Eigen::Vector2d cur_acc_2d = cur_acc.head(2);
        double norm_vel = cur_vel_2d.norm();
        double help1 = 1.0 / (norm_vel * norm_vel + 1e-2);
        double omega = help1 * cur_acc_2d.transpose() * B_h * cur_vel_2d;

        fd_->start_pt_ = cur_pos;
        fd_->start_vel_ = cur_vel;
        fd_->start_yaw_(0) = cur_yaw;
        fd_->start_yaw_(1) = omega;
      }

      TrajPlannerResult res = callTrajectoryPlanner();

      if (res == TrajPlannerResult::FAILED) {
        // A failed replacement must revoke the old command. Otherwise traj_server
        // can keep following a stale path while the FSM has already rejected it.
        emergencyStop();
        local_target_marker_valid_ = false;
        planning_retry_after_ = ros::Time::now() +
            ros::Duration(std::max(0.0, plan_failure_retry_delay_));
        ROS_WARN("[Real] Plan trajectory failed; motion stopped and retry delayed by %.2f s.",
            std::max(0.0, plan_failure_retry_delay_));
      }
      else if (res == TrajPlannerResult::NEED_ROTATION) {
        fd_->static_state_ = true;
        active_corner_approach_ = false;
        rotation_command_sent_ = false;
        rotation_command_time_ = ros::Time(0);
        rotation_in_tolerance_time_ = ros::Time(0);
        transitState(RealFSM::State::ROTATE_TO_PATH, "FSM");
      }
      else if (res == TrajPlannerResult::SUCCESS) {
        transitState(RealFSM::State::EXEC_TRAJ, "FSM");
      }
      else {  // TrajPlannerResult::MISSION_COMPLETE
        transitState(RealFSM::State::FINISH, "FSM");
      }

      visualize();
      break;
    }

    case RealFSM::State::ROTATE_TO_PATH: {
      if (!fd_->have_odom_) {
        ROS_WARN_THROTTLE(1.0, "[Real] Waiting for odometry before rotating to path.");
        break;
      }

      if (!rotation_command_sent_) {
        // target_angle atomically takes ownership of the controller in
        // traj_server, so no independent stop/topic ordering is required.
        std_msgs::Float32 target_msg;
        target_msg.data = pending_rotation_yaw_;
        target_angle_pub_.publish(target_msg);
        rotation_command_sent_ = true;
        rotation_command_time_ = ros::Time::now();
        rotation_in_tolerance_time_ = ros::Time(0);
        ROS_WARN("[Real] Rotate in place before planning: current=%.2f, target=%.2f rad.",
            fd_->odom_yaw_, pending_rotation_yaw_);
      }

      double yaw_error = pending_rotation_yaw_ - fd_->odom_yaw_;
      wrapAngle(yaw_error);
      if (std::fabs(yaw_error) <= rotation_tolerance_) {
        if (rotation_in_tolerance_time_.isZero()) {
          rotation_in_tolerance_time_ = ros::Time::now();
        }
        else if ((ros::Time::now() - rotation_in_tolerance_time_).toSec() >=
                 rotation_settle_time_) {
          rotation_command_sent_ = false;
          rotation_in_tolerance_time_ = ros::Time(0);
          reuse_rotation_path_once_ = true;
          transitState(RealFSM::State::PLAN_TRAJ, "RotateToPathComplete");
        }
      }
      else {
        rotation_in_tolerance_time_ = ros::Time(0);
      }

      if (rotation_command_sent_ &&
          (ros::Time::now() - rotation_command_time_).toSec() > rotation_timeout_) {
        ROS_ERROR("[Real] Rotate-to-path timed out; dormant the current frontier and replan.");
        emergencyStop();
        expl_manager_->frontier_map2d_->setForceDormantFrontier(pending_rotation_goal_);
        fd_->dormant_frontier_flag_ = true;
        rotation_command_sent_ = false;
        reuse_rotation_path_once_ = false;
        pending_rotation_path_.clear();
        local_target_marker_valid_ = false;
        planning_retry_after_ = ros::Time::now() +
            ros::Duration(std::max(0.0, plan_failure_retry_delay_));
        transitState(RealFSM::State::PLAN_TRAJ, "RotateToPathTimeout");
      }
      break;
    }

    case RealFSM::State::EXEC_TRAJ: {
      // Publish trajectory and transition to execution monitoring
      double dt = (ros::Time::now() - fd_->newest_traj_.start_time).toSec();
      if (dt > 0) {
        trajectory_manager::PolyTraj poly_msg;
        polyTraj2ROSMsg(fd_->newest_traj_, poly_msg);
        poly_traj_pub_.publish(poly_msg);
        fd_->static_state_ = false;
        transitState(RealFSM::State::REPLAN, "FSM");
      }
      break;
    }

    case RealFSM::State::REPLAN: {
      // Monitor trajectory execution and decide when to replan
      LocalTrajectory* info = &expl_manager_->gcopter_->local_trajectory_;
      double t_cur = (ros::Time::now() - info->start_time).toSec();
      double time_to_end = info->duration - t_cur;

      // A corner-staging trajectory must reach its zero-speed endpoint before
      // the outgoing segment is planned. Replanning it one second early would
      // reintroduce the same inside-corner shortcut this state is preventing.
      const double replan_end_threshold = active_corner_approach_ ?
          corner_replan_end_threshold_ : fp_->replan_traj_end_threshold_;
      if (time_to_end < replan_end_threshold) {
        transitState(RealFSM::State::PLAN_TRAJ, "FSM");
        ROS_WARN("[Real] Replan: trajectory endpoint approaching (%.2f s remaining%s).",
            time_to_end, active_corner_approach_ ? ", corner staging" : "");
        exec_timer_.start();
        return;
      }

      const bool frontier_changed =
          pending_frontier_change_ || expl_manager_->frontier_map2d_->isAnyFrontierChanged();
      if (shouldReplanForFrontierChange(
              t_cur,
              fp_->replan_frontier_change_delay_,
              fd_->final_result_,
              frontier_changed,
              replan_on_frontier_change_)) {
        pending_frontier_change_ = false;
        transitState(RealFSM::State::PLAN_TRAJ, "FSM");
        ROS_WARN("[Real] Replan: frontier changed");
        exec_timer_.start();
        return;
      }

      // Replan if trajectory timeout
      if (t_cur > fp_->replan_timeout_) {
        transitState(RealFSM::State::PLAN_TRAJ, "FSM");
        ROS_WARN("[Real] Replan: time out");
        exec_timer_.start();
        return;
      }
      break;
    }
  }

  exec_timer_.start();
}

TrajPlannerResult ExplorationFSMReal::callTrajectoryPlanner()
{
  ros::Time time_r = ros::Time::now() + ros::Duration(fp_->replan_time_);
  const bool reuse_rotation_path = reuse_rotation_path_once_;
  bool temporarily_no_passable_frontier = false;
  reuse_rotation_path_once_ = false;

  if (reuse_rotation_path) {
    const bool saved_path_missing = pending_rotation_path_.empty();
    const Eigen::Vector2d saved_rotation_goal = pending_rotation_goal_;
    const int saved_rotation_result = pending_rotation_final_result_;
    pending_rotation_path_.clear();

    if (saved_path_missing) {
      local_target_marker_valid_ = false;
      ROS_ERROR("[Real] Rotate-to-path completed without a saved path; retry normal frontier selection.");
      return TrajPlannerResult::FAILED;
    }

    // Keep the target that requested the turn, but rebuild its 2D A* path from
    // the latest body-center odometry. Reusing the pre-rotation path is unsafe:
    // camera lever-arm motion or VIO drift can move the estimated XY while Go2
    // rotates, leaving the old path disconnected from the new planning start.
    Eigen::Vector2d replanned_goal;
    std::vector<Eigen::Vector2d> replanned_path;
    if (!expl_manager_->replanPathToGoal(
            fd_->start_pt_.head(2), saved_rotation_goal,
            replanned_goal, replanned_path)) {
      local_target_marker_valid_ = false;
      ROS_WARN("[Real] Could not rebuild the path from the post-rotation body pose; retry target selection.");
      return TrajPlannerResult::FAILED;
    }

    expl_manager_->ed_->next_best_path_ = replanned_path;
    expl_manager_->ed_->next_pos_ = replanned_goal;
    fd_->final_result_ = saved_rotation_result;
    pending_frontier_change_ = false;
    ROS_INFO("[Real] Rebuilt path to the same target from the post-rotation body pose.");
  }
  else {
    const bool forced_dormant_frontier = fd_->dormant_frontier_flag_;
    const auto previous_path = expl_manager_->ed_->next_best_path_;
    const Eigen::Vector2d previous_goal = expl_manager_->ed_->next_pos_;
    const int previous_final_result = fd_->final_result_;

    fd_->dormant_frontier_flag_ = false;
    const bool frontier_changed = updateFrontierAndObject();
    pending_frontier_change_ = false;

    // A local trajectory has ended or was explicitly replanned. Select from the
    // current map and current odometry; retaining the old frontier here can make
    // Go2 stop at a local waypoint and keep selecting that already-reached path.
    int expl_res = expl_manager_->planNextBestPoint(fd_->start_pt_, fd_->start_yaw_(0));

    temporarily_no_passable_frontier =
        expl_res == EXPL_RESULT::NO_PASSABLE_FRONTIER;
    if (expl_res == EXPL_RESULT::EXPLORATION)
      fd_->final_result_ = FINAL_RESULT::EXPLORE;
    else if (expl_res == EXPL_RESULT::NO_COVERABLE_FRONTIER)
      fd_->final_result_ = FINAL_RESULT::NO_FRONTIER;
    else if (temporarily_no_passable_frontier)
      fd_->final_result_ = FINAL_RESULT::EXPLORE;
    else
      fd_->final_result_ = FINAL_RESULT::SEARCH_OBJECT;

    if (shouldKeepPreviousExplorationGoal(
            previous_final_result,
            fd_->final_result_,
            frontier_changed,
            !previous_path.empty(),
            forced_dormant_frontier)) {
      expl_manager_->ed_->next_best_path_ = previous_path;
      expl_manager_->ed_->next_pos_ = previous_goal;
      ROS_INFO("[Real] Keep current exploration path for stable local trajectory handoff.");
    }
  }

  // Publish exploration result
  std_msgs::Int32 expl_result_msg;
  expl_result_msg.data = fd_->final_result_;
  expl_result_pub_.publish(expl_result_msg);

  if (temporarily_no_passable_frontier) {
    // Frontiers still exist, but the partial depth map cannot connect the body
    // center to one yet. This is transient during startup and map updates; do
    // not report mission completion. PLAN_TRAJ will retry after its configured
    // failure delay while the robot remains stopped.
    finish_goal_pos_valid_ = false;
    local_target_marker_valid_ = false;
    pending_rotation_path_.clear();
    ROS_WARN_THROTTLE(1.0,
        "[Real] Frontiers exist but none is currently passable; keep exploration active and retry after map update.");
    return TrajPlannerResult::FAILED;
  }

  if (fd_->final_result_ == FINAL_RESULT::NO_FRONTIER) {
    finish_goal_pos_valid_ = false;
    local_target_marker_valid_ = false;
    pending_rotation_path_.clear();
    ROS_WARN("[Real] No (passable) frontier");
    return TrajPlannerResult::MISSION_COMPLETE;
  }

  // Use the global goal for trajectory planning so exploration keeps moving outward.
  // The path still provides a yaw reference near the current lookahead point.
  Eigen::Vector2d object_goal_pos = expl_manager_->ed_->next_pos_;

  finish_goal_pos_ = object_goal_pos;
  finish_goal_pos_valid_ = true;

  Eigen::Vector2d local_goal_pos = object_goal_pos;
  double local_goal_yaw = 0.0;
  auto path = expl_manager_->ed_->next_best_path_;
  const bool local_target_valid =
      selectLocalTarget(fd_->start_pt_.head(2), path, fp_->local_target_distance_,
          local_goal_pos, local_goal_yaw);
  if (!local_target_valid) {
    local_target_marker_valid_ = false;
    pending_rotation_path_.clear();
    if (fd_->final_result_ == FINAL_RESULT::EXPLORE) {
      planning_failure_count_++;
      ROS_WARN("[Real] Frontier path has no footprint-safe local target; retry after map update (%d/%d).",
          planning_failure_count_, FSMConstantsReal::MAX_CONSECUTIVE_PLANNING_FAILURES);
      if (planning_failure_count_ >= FSMConstantsReal::MAX_CONSECUTIVE_PLANNING_FAILURES) {
        ROS_WARN("[Real] Force dormant current frontier after repeated unsafe local-target selections.");
        expl_manager_->frontier_map2d_->setForceDormantFrontier(object_goal_pos);
        fd_->dormant_frontier_flag_ = true;
        planning_failure_count_ = 0;
      }
      else {
        fd_->dormant_frontier_flag_ = false;
      }
    }
    else {
      // This goal came from object navigation, not from the frontier map. Do
      // not pass it to setForceDormantFrontier: that cannot remove an object
      // goal and caused the same object to be selected indefinitely.
      ROS_WARN("[Real] Object path has no footprint-safe local target; skip object once and reselect frontier.");
      expl_manager_->setSkipObjectNavigationOnce();
      fd_->dormant_frontier_flag_ = false;
    }
    return TrajPlannerResult::FAILED;
  }

  // The safety test uses the actual stopped/present body pose, rather than a
  // short-horizon predicted trajectory pose. This makes the rotation check match
  // where Go2 will physically turn.
  const Eigen::Vector2d rotation_pos = fd_->odom_pos_.head(2);
  const double rotation_start_yaw = fd_->odom_yaw_;
  const double rotation_yaw_error =
      computeTargetYawError(rotation_pos, rotation_start_yaw, local_goal_pos);
  if (shouldRotateBeforeTranslation(
          rotation_yaw_error, rotation_before_translate_yaw_error_)) {
    pending_rotation_yaw_ = std::atan2(
        local_goal_pos.y() - rotation_pos.y(), local_goal_pos.x() - rotation_pos.x());
    if (isInPlaceRotationSafe(rotation_pos, rotation_start_yaw, pending_rotation_yaw_)) {
      pending_rotation_goal_ = object_goal_pos;
      pending_rotation_path_ = path;
      pending_rotation_final_result_ = fd_->final_result_;
      ROS_WARN("[Real] Safe local target needs %.1f deg heading alignment; rotate before translation.",
          rotation_yaw_error * 180.0 / M_PI);
      return TrajPlannerResult::NEED_ROTATION;
    }

    pending_rotation_path_.clear();
    ROS_WARN("[Real] Safe local target needs %.1f deg rotation, but the rectangular body sweep is blocked; fall back to KinoAstar curved trajectory.",
        rotation_yaw_error * 180.0 / M_PI);
  }

  Eigen::Vector2d goal_pos =
      selectTrajectoryGoalForMode(fd_->final_result_, object_goal_pos, local_goal_pos);
  double goal_yaw = local_goal_yaw;
  expl_manager_->ed_->next_local_pos_ = goal_pos;

  // Check if reached object
  if (shouldCompleteSearchObjectMission(
          fd_->final_result_,
          fd_->start_pt_.head(2),
          object_goal_pos,
          local_goal_pos,
          FSMConstantsReal::REACH_DISTANCE)) {
    local_target_marker_valid_ = false;
    pending_rotation_path_.clear();
    ROS_WARN("[Real] Object-map approach point reached.");
    return TrajPlannerResult::MISSION_COMPLETE;
  }

  // Prepare state for trajectory planning
  Eigen::VectorXd goal_state(5), current_state(5);
  Eigen::Vector3d current_control(0.0, 0.0, 0.0);
  double start_vel = Eigen::Vector2d(fd_->start_vel_(0), fd_->start_vel_(1)).norm();
  current_state << fd_->start_pt_(0), fd_->start_pt_(1), fd_->start_yaw_(0), 0.0, start_vel;

  goal_state << goal_pos(0), goal_pos(1), goal_yaw, 0.0, 0.0;

  bool traj_res = expl_manager_->planTrajectory(current_state, goal_state, current_control);
  if (traj_res) {
    planning_failure_count_ = 0;
    active_corner_approach_ = local_target_is_corner_staging_;
    pending_rotation_path_.clear();
    expl_manager_->ed_->next_local_pos_ = goal_pos;
    auto info = &expl_manager_->gcopter_->local_trajectory_;
    const ros::Time plan_now = ros::Time::now();
    if (plan_from_odom_) {
      // No future-start gap when replacing a trajectory close to its end.
      info->start_time = plan_now;
    }
    else {
      info->start_time = (plan_now - time_r).toSec() > 0 ? plan_now : time_r;
    }
    fd_->newest_traj_ = expl_manager_->gcopter_->local_trajectory_;
    ROS_INFO("[Real] Trajectory handoff mode: %s, start in %.3f s.",
        plan_from_odom_ ? "odom-immediate" : "predicted-delayed",
        (info->start_time - plan_now).toSec());
    return TrajPlannerResult::SUCCESS;
  }

  planning_failure_count_++;
  pending_rotation_path_.clear();
  ROS_WARN("[Real] Trajectory planning failed for goal=(%.2f, %.2f), consecutive_failures=%d/%d",
      goal_pos(0), goal_pos(1), planning_failure_count_,
      FSMConstantsReal::MAX_CONSECUTIVE_PLANNING_FAILURES);
  if (planning_failure_count_ >= FSMConstantsReal::MAX_CONSECUTIVE_PLANNING_FAILURES) {
    if (fd_->final_result_ == FINAL_RESULT::EXPLORE) {
      ROS_WARN("[Real] Force dormant current frontier after repeated trajectory planning failures.");
      expl_manager_->frontier_map2d_->setForceDormantFrontier(object_goal_pos);
      fd_->dormant_frontier_flag_ = true;
    }
    else {
      ROS_WARN("[Real] Repeated object trajectory failures; skip object once and reselect frontier.");
      expl_manager_->setSkipObjectNavigationOnce();
      fd_->dormant_frontier_flag_ = false;
    }
    planning_failure_count_ = 0;
  }
  else if (fd_->final_result_ != FINAL_RESULT::EXPLORE) {
    // Avoid spending all retry cycles on an object path that is not currently
    // executable while the frontier map has other exploration targets.
    expl_manager_->setSkipObjectNavigationOnce();
  }

  return TrajPlannerResult::FAILED;
}

bool ExplorationFSMReal::isInPlaceRotationSafe(
    const Eigen::Vector2d& pos, double start_yaw, double target_yaw)
{
  return isInPlaceRotationFootprintSafe(
      pos, start_yaw, target_yaw, rotation_footprint_step_,
      [this](const Eigen::Vector2d& collision_pos, double yaw) {
        return expl_manager_->kinoastar_->isCollisionPosYaw(collision_pos, yaw);
      });
}

void ExplorationFSMReal::polyTraj2ROSMsg(
    const LocalTrajectory& local_traj, trajectory_manager::PolyTraj& poly_msg)
{
  auto data = &local_traj;
  Eigen::VectorXd durs = data->traj.getDurations();
  int piece_num = data->traj.getPieceNum();

  poly_msg.drone_id = 0;
  poly_msg.traj_id = data->traj_id;
  poly_msg.start_time = data->start_time;
  poly_msg.order = 7;
  poly_msg.duration.resize(piece_num);
  poly_msg.coef_x.resize(8 * piece_num);
  poly_msg.coef_y.resize(8 * piece_num);
  poly_msg.coef_z.resize(8 * piece_num);

  for (int i = 0; i < piece_num; ++i) {
    poly_msg.duration[i] = durs(i);

    auto cMat = data->traj.operator[](i).getCoeffMat();
    int i8 = i * 8;
    for (int j = 0; j < 8; j++) {
      poly_msg.coef_x[i8 + j] = cMat(0, j);
      poly_msg.coef_y[i8 + j] = cMat(1, j);
      poly_msg.coef_z[i8 + j] = cMat(2, j);
    }
  }
}

bool ExplorationFSMReal::selectLocalTarget(const Eigen::Vector2d& current_pos,
    const std::vector<Eigen::Vector2d>& path, const double& local_distance,
    Eigen::Vector2d& target_pos, double& target_yaw)
{
  local_target_is_corner_staging_ = false;
  if (path.empty()) {
    return false;
  }

  PathPoseCandidate safe_pose;
  bool corner_limited = false;
  if (!selectCornerAwareFootprintSafeLocalTargetFromPath(
          current_pos,
          path,
          local_distance,
          0.30,
          local_target_corner_angle_,
          [this](const Eigen::Vector2d& pos, double yaw) {
            return expl_manager_->kinoastar_->isCollisionPosYaw(pos, yaw);
          },
          safe_pose,
          corner_limited)) {
    ROS_WARN("[Real] Selected path has no footprint-safe local target within %.2f m.",
        local_distance);
    return false;
  }

  target_pos = safe_pose.pos;
  target_yaw = safe_pose.yaw;
  const Eigen::Vector2d footprint_safe_target = target_pos;
  const double footprint_safe_yaw = target_yaw;
  local_target_is_corner_staging_ = corner_limited;

  if (corner_limited) {
    ROS_INFO("[Real] Limit local target at first sharp path corner: target=(%.2f, %.2f), yaw=%.1f deg.",
        target_pos.x(), target_pos.y(), target_yaw * 180.0 / M_PI);
  }

  // Gradient-based safety adjustment
  double step_size = 0.05;
  double tolerance = 1e-3;
  int max_iterations = 30;

  for (int i = 0; !corner_limited && i < max_iterations; ++i) {
    Eigen::Vector2d prev_pos = target_pos;

    // Get gradient from SDF map
    Eigen::Vector2d grad;
    double dist = expl_manager_->sdf_map_->getDistWithGrad(target_pos, grad);

    if (dist > 0.26)
      break;

    // Move along gradient to safer position
    if (grad.norm() > 1e-6) {
      target_pos += step_size * grad.normalized();
    }

    // Check convergence
    if ((target_pos - prev_pos).norm() < tolerance) {
      break;
    }
  }

  target_pos = clampLocalTargetToLookahead(current_pos, target_pos, local_distance);
  if (expl_manager_->kinoastar_->isCollisionPosYaw(target_pos, target_yaw)) {
    ROS_WARN("[Real] Gradient-adjusted local target is not footprint-safe; use path-safe target instead.");
    target_pos = footprint_safe_target;
    target_yaw = footprint_safe_yaw;
  }

  // Store selected local target
  expl_manager_->ed_->next_local_pos_ = target_pos;
  local_target_marker_valid_ = true;
  return true;
}

void ExplorationFSMReal::visualize()
{
  auto ed_ptr = expl_manager_->ed_;

  auto vec2dTo3d = [](const std::vector<Eigen::Vector2d>& vec2d, double z = 0.15) {
    std::vector<Eigen::Vector3d> vec3d;
    for (auto v : vec2d) vec3d.push_back(Eigen::Vector3d(v(0), v(1), z));
    return vec3d;
  };

  // Draw frontiers
  static int last_ftr2d_num = 0;
  for (int i = 0; i < (int)ed_ptr->frontiers_.size(); ++i) {
    visualization_->drawCubes(vec2dTo3d(ed_ptr->frontiers_[i]), fp_->vis_scale_,
        visualization_->getColor(double(i) / ed_ptr->frontiers_.size(), 1.0), "frontier", i, 4);
  }
  for (int i = ed_ptr->frontiers_.size(); i < last_ftr2d_num; ++i) {
    visualization_->drawCubes({}, fp_->vis_scale_, Eigen::Vector4d(0, 0, 0, 1), "frontier", i, 4);
  }
  last_ftr2d_num = ed_ptr->frontiers_.size();

  // Draw dormant frontiers
  static int last_dftr2d_num = 0;
  for (int i = 0; i < (int)ed_ptr->dormant_frontiers_.size(); ++i) {
    visualization_->drawCubes(vec2dTo3d(ed_ptr->dormant_frontiers_[i]), fp_->vis_scale_,
        Eigen::Vector4d(0, 0, 0, 1), "dormant_frontier", i, 4);
  }
  for (int i = ed_ptr->dormant_frontiers_.size(); i < last_dftr2d_num; ++i) {
    visualization_->drawCubes(
        {}, fp_->vis_scale_, Eigen::Vector4d(0, 0, 0, 1), "dormant_frontier", i, 4);
  }
  last_dftr2d_num = ed_ptr->dormant_frontiers_.size();

  // Draw objects
  static int last_obj_num = 0;
  if (!visualize_object_markers_) {
    for (int i = 0; i < last_obj_num; ++i) {
      visualization_->drawCubes({}, fp_->vis_scale_, Eigen::Vector4d(0, 0, 0, 1), "object", i, 4);
    }
    last_obj_num = 0;
  } else {
    for (int i = 0; i < (int)ed_ptr->objects_.size(); ++i) {
      int label = ed_ptr->object_labels_[i];
      if (!shouldVisualizeObjectMarker(visualize_object_markers_, label)) {
        visualization_->drawCubes({}, fp_->vis_scale_, Eigen::Vector4d(0, 0, 0, 1), "object", i, 4);
        continue;
      }
      visualization_->drawCubes(vec2dTo3d(ed_ptr->objects_[i]), fp_->vis_scale_,
          visualization_->getColor(double(label) / 5.0, 1.0), "object", i, 4);
    }
    for (int i = ed_ptr->objects_.size(); i < last_obj_num; ++i) {
      visualization_->drawCubes({}, fp_->vis_scale_, Eigen::Vector4d(0, 0, 0, 1), "object", i, 4);
    }
    last_obj_num = ed_ptr->objects_.size();
  }

  // Draw next best path
  visualization_->drawLines(vec2dTo3d(ed_ptr->next_best_path_), fp_->vis_scale_,
      Eigen::Vector4d(1, 0.2, 0.2, 1), "next_path", 1, 6);

  // Apexnavmain does not visualize the global target point; clear any stale marker.
  visualization_->drawSpheres({}, fp_->vis_scale_ * 3.5,
      Eigen::Vector4d(1.0, 0.85, 0.1, 1), "global_point", 1, 6);

  std::vector<Eigen::Vector2d> local_points;
  if (local_target_marker_valid_) {
    local_points.push_back(ed_ptr->next_local_pos_);
  }
  visualization_->drawSpheres(vec2dTo3d(local_points), fp_->vis_scale_ * 3,
      Eigen::Vector4d(0.2, 0.2, 1.0, 1), "local_point", 1, 6);

  visualization_->drawLines(vec2dTo3d(ed_ptr->tsp_tour_), fp_->vis_scale_ / 1.25,
      Eigen::Vector4d(0.2, 1, 0.2, 1), "tsp_tour", 0, 6);
}

void ExplorationFSMReal::clearVisMarker()
{
  local_target_marker_valid_ = false;
  for (int i = 0; i < 500; ++i) {
    visualization_->drawCubes({}, fp_->vis_scale_, Eigen::Vector4d(0, 0, 0, 1), "frontier", i, 4);
    visualization_->drawCubes(
        {}, fp_->vis_scale_, Eigen::Vector4d(0, 0, 0, 1), "dormant_frontier", i, 4);
    visualization_->drawCubes({}, fp_->vis_scale_, Eigen::Vector4d(0, 0, 0, 1), "object", i, 4);
  }
  visualization_->drawLines({}, fp_->vis_scale_, Eigen::Vector4d(0, 0, 1, 1), "next_path", 1, 6);
  visualization_->drawSpheres({}, fp_->vis_scale_ * 3.5,
      Eigen::Vector4d(1.0, 0.85, 0.1, 1), "global_point", 1, 6);
  visualization_->drawSpheres({}, fp_->vis_scale_ * 3,
      Eigen::Vector4d(0.2, 0.2, 1.0, 1), "local_point", 1, 6);
}

bool ExplorationFSMReal::updateFrontierAndObject()
{
  bool change_flag = false;
  auto frt_map = expl_manager_->frontier_map2d_;
  auto obj_map = expl_manager_->object_map2d_;
  auto ed = expl_manager_->ed_;
  Eigen::Vector2d sensor_pos = Eigen::Vector2d(fd_->odom_pos_(0), fd_->odom_pos_(1));

  change_flag = frt_map->isAnyFrontierChanged();
  frt_map->searchFrontiers();
  change_flag |= frt_map->dormantSeenFrontiers(sensor_pos, fd_->odom_yaw_);
  pending_frontier_change_ = pending_frontier_change_ || change_flag;
  frt_map->getFrontiers(ed->frontiers_, ed->frontier_averages_);
  frt_map->getDormantFrontiers(ed->dormant_frontiers_, ed->dormant_frontier_averages_);
  obj_map->getObjects(ed->objects_, ed->object_averages_, ed->object_labels_);

  return change_flag;
}

void ExplorationFSMReal::frontierCallback(const ros::TimerEvent& e)
{
  if (state_ == RealFSM::State::INIT)
    return;

  updateFrontierAndObject();
  visualize();
}

void ExplorationFSMReal::triggerCallback(const geometry_msgs::PoseStampedConstPtr& msg)
{
  if (state_ != RealFSM::State::WAIT_TRIGGER)
    return;
  if (!fd_->have_confidence_) {
    ROS_WARN("[Real] Exploration trigger ignored: waiting for confidence threshold from INSiNav.");
    return;
  }

  fd_->trigger_ = true;
  expl_manager_->frontier_map2d_->setPersistentExclusionZone(fd_->odom_pos_.head(2));
  ROS_INFO("[Real] Exploration triggered!");
  transitState(RealFSM::State::PLAN_TRAJ, "triggerCallback");
}

void ExplorationFSMReal::odometryCallback(const nav_msgs::OdometryConstPtr& msg)
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

  // Extract linear velocity
  fd_->odom_vel_(0) = msg->twist.twist.linear.x;
  fd_->odom_vel_(1) = msg->twist.twist.linear.y;
  fd_->odom_vel_(2) = msg->twist.twist.linear.z;

  // Extract angular velocity
  fd_->odom_omega_(0) = msg->twist.twist.angular.x;
  fd_->odom_omega_(1) = msg->twist.twist.angular.y;
  fd_->odom_omega_(2) = msg->twist.twist.angular.z;

  fd_->have_odom_ = true;

  // Publish robot marker for visualization
  publishRobotMarker();
}

void ExplorationFSMReal::confidenceThresholdCallback(const std_msgs::Float64ConstPtr& msg)
{
  if (fd_->have_confidence_)
    return;
  fd_->have_confidence_ = true;
  expl_manager_->sdf_map_->object_map2d_->setConfidenceThreshold(msg->data);
  ROS_INFO("[Real] Confidence threshold set to: %.2f", msg->data);
}

void ExplorationFSMReal::goalCallback(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr& msg)
{
  double x = msg->pose.pose.position.x;
  double y = msg->pose.pose.position.y;

  tf::Quaternion q(msg->pose.pose.orientation.x, msg->pose.pose.orientation.y,
      msg->pose.pose.orientation.z, msg->pose.pose.orientation.w);

  double roll, pitch, yaw;
  tf::Matrix3x3(q).getRPY(roll, pitch, yaw);

  Eigen::VectorXd goal_state(5), current_state(5);
  Eigen::Vector3d current_control;
  double start_vel = Eigen::Vector2d(fd_->odom_vel_(0), fd_->odom_vel_(1)).norm();
  current_state << fd_->odom_pos_(0), fd_->odom_pos_(1), fd_->odom_yaw_, 0.0, start_vel;
  goal_state << x, y, yaw, 0.0, 0.0;

  if ((current_state.head(2) - goal_state.head(2)).norm() > 0.2) {
    current_control << 0.0, 0.0, 0.0;

    if (!expl_manager_->planTrajectory(current_state, goal_state, current_control)) {
      ROS_WARN("[Real] Manual goal trajectory planning failed: clicked=(%.2f, %.2f, %.2f).",
          x, y, yaw);
      return;
    }

    trajectory_manager::PolyTraj poly_msg;
    const Eigen::Vector2d goal_pos = goal_state.head(2);
    expl_manager_->ed_->next_local_pos_ = goal_pos;
    local_target_marker_valid_ = true;
    polyTraj2ROSMsg(expl_manager_->gcopter_->local_trajectory_, poly_msg);
    poly_traj_pub_.publish(poly_msg);
    ROS_INFO("[Real] Published manual goal trajectory: x=%.2f, y=%.2f, yaw=%.2f",
        goal_pos(0), goal_pos(1), yaw);
  }
  ROS_INFO("[Real] Received goal pose: x=%.2f, y=%.2f, yaw=%.2f", x, y, yaw);
}

void ExplorationFSMReal::emergencyStop()
{
  fd_->static_state_ = true;
  active_corner_approach_ = false;
  stop_pub_.publish(std_msgs::Empty());
}

void ExplorationFSMReal::safetyCallback(const ros::TimerEvent& e)
{
  if (state_ != RealFSM::State::REPLAN)
    return;

  // Check if robot deviates from planned trajectory
  double t_cur = (ros::Time::now() - expl_manager_->gcopter_->local_trajectory_.start_time).toSec();
  t_cur = min(t_cur, expl_manager_->gcopter_->local_trajectory_.duration);
  Eigen::Vector3d cur_pos = expl_manager_->gcopter_->local_trajectory_.traj.getPos(t_cur);
  const Eigen::Vector2d planned_pos = cur_pos.head(2);
  const Eigen::Vector2d odom_pos = fd_->odom_pos_.head(2);
  const double tracking_error = (planned_pos - odom_pos).norm();

  if (shouldAbortTrajectoryForTrackingError(
          planned_pos, odom_pos, tracking_abort_distance_)) {
    ROS_ERROR("[Real] Odom far from traj: planned=(%.2f, %.2f), odom=(%.2f, %.2f), err=%.2f, t=%.2f, status=%s, result=%d. Stop!!!",
        planned_pos(0), planned_pos(1), odom_pos(0), odom_pos(1), tracking_error,
        t_cur, insinav_stop_verified_status_.c_str(), fd_->final_result_);
    emergencyStop();
    fd_->static_state_ = true;

    transitState(RealFSM::State::PLAN_TRAJ, "Odom Far From Trajectory");
    return;
  }

  // Time-sampled safety check using the same oriented rectangular footprint as
  // KinoAstar and the post-optimization validator.
  double time_horizon = 2.5;  // Check trajectory for next 2.5 seconds
  double sample_dt = 0.05;

  for (double t_check = t_cur;
      t_check <= min(t_cur + time_horizon, expl_manager_->gcopter_->local_trajectory_.duration);
      t_check += sample_dt) {
    Eigen::Vector3d check_pos = expl_manager_->gcopter_->local_trajectory_.traj.getPos(t_check);
    Eigen::Vector3d check_vel = expl_manager_->gcopter_->local_trajectory_.traj.getVel(t_check);
    Eigen::Vector2d check_pos_2d = check_pos.head(2);
    double check_yaw = fd_->odom_yaw_;
    if (check_vel.head(2).norm() > 1.0e-3) {
      check_yaw = std::atan2(check_vel.y(), check_vel.x());
    }
    else {
      const double t0 = std::max(0.0, t_check - sample_dt);
      const double t1 = std::min(
          expl_manager_->gcopter_->local_trajectory_.duration, t_check + sample_dt);
      const Eigen::Vector2d delta =
          expl_manager_->gcopter_->local_trajectory_.traj.getPos(t1).head(2) -
          expl_manager_->gcopter_->local_trajectory_.traj.getPos(t0).head(2);
      if (delta.norm() > 1.0e-4) {
        check_yaw = std::atan2(delta.y(), delta.x());
      }
    }

    if (expl_manager_->kinoastar_->isCollisionPosYaw(check_pos_2d, check_yaw)) {
      ROS_ERROR("[Real] Safety Stop: rectangular footprint collision predicted at "
                "(%.2f, %.2f), yaw=%.1f deg, trajectory time %.2f.",
          check_pos_2d.x(), check_pos_2d.y(), check_yaw * 180.0 / M_PI, t_check);
      emergencyStop();
      transitState(RealFSM::State::PLAN_TRAJ, "Trajectory Safety Stop");
      return;
    }
  }
}

void ExplorationFSMReal::publishRobotMarker()
{
  // Use the same centered rectangular envelope as KinoAstar so RViz shows the
  // actual footprint being collision-checked.
  visualization_msgs::Marker robot_marker;
  robot_marker.header.frame_id = "odom";
  robot_marker.header.stamp = ros::Time::now();
  robot_marker.ns = "robot_position";
  robot_marker.id = 0;
  robot_marker.type = visualization_msgs::Marker::CUBE;
  robot_marker.action = visualization_msgs::Marker::ADD;

  robot_marker.pose.position.x = fd_->odom_pos_(0);
  robot_marker.pose.position.y = fd_->odom_pos_(1);
  robot_marker.pose.position.z = fd_->odom_pos_(2) + robot_marker_height_ / 2.0;

  robot_marker.pose.orientation.x = fd_->odom_orient_.x();
  robot_marker.pose.orientation.y = fd_->odom_orient_.y();
  robot_marker.pose.orientation.z = fd_->odom_orient_.z();
  robot_marker.pose.orientation.w = fd_->odom_orient_.w();

  robot_marker.scale.x = robot_marker_length_;
  robot_marker.scale.y = robot_marker_width_;
  robot_marker.scale.z = robot_marker_height_;

  robot_marker.color.r = 50.0 / 255.0;
  robot_marker.color.g = 50.0 / 255.0;
  robot_marker.color.b = 255.0 / 255.0;
  robot_marker.color.a = 1.0;

  // Create direction arrow marker
  visualization_msgs::Marker arrow_marker;
  arrow_marker.header.frame_id = "odom";
  arrow_marker.header.stamp = ros::Time::now();
  arrow_marker.ns = "robot_direction";
  arrow_marker.id = 1;
  arrow_marker.type = visualization_msgs::Marker::ARROW;
  arrow_marker.action = visualization_msgs::Marker::ADD;

  arrow_marker.pose.position.x = fd_->odom_pos_(0);
  arrow_marker.pose.position.y = fd_->odom_pos_(1);
  arrow_marker.pose.position.z = fd_->odom_pos_(2) + robot_marker_height_;

  arrow_marker.pose.orientation.x = fd_->odom_orient_.x();
  arrow_marker.pose.orientation.y = fd_->odom_orient_.y();
  arrow_marker.pose.orientation.z = fd_->odom_orient_.z();
  arrow_marker.pose.orientation.w = fd_->odom_orient_.w();

  arrow_marker.scale.x = robot_marker_length_ / 2.0;
  arrow_marker.scale.y = 0.08;
  arrow_marker.scale.z = 0.08;

  arrow_marker.color.r = 10.0 / 255.0;
  arrow_marker.color.g = 255.0 / 255.0;
  arrow_marker.color.b = 10.0 / 255.0;
  arrow_marker.color.a = 1.0;

  robot_marker_pub_.publish(robot_marker);
  robot_marker_pub_.publish(arrow_marker);
}

void ExplorationFSMReal::transitState(RealFSM::State new_state, std::string pos_call)
{
  std::string state_str[] = { "INIT", "WAIT_TRIGGER", "PLAN_TRAJ", "ROTATE_TO_PATH",
    "EXEC_TRAJ", "REPLAN", "FINISH" };
  ROS_INFO("[Real FSM]: %s -> from %s to %s", pos_call.c_str(),
      state_str[static_cast<int>(state_)].c_str(), state_str[static_cast<int>(new_state)].c_str());
  state_ = new_state;
}

void ExplorationFSMReal::insinnavStopVerifiedCallback(const std_msgs::StringConstPtr& msg)
{
  insinav_stop_verified_status_ = msg->data;

  ROS_INFO_THROTTLE(1.0,
      "[ExplorationFSMReal] LightGlue stop verification status: %s",
      insinav_stop_verified_status_.c_str());

  if (fd_->have_finished_ && !finish_completed_time_.isZero()) {
    const double time_since_finish = (ros::Time::now() - finish_completed_time_).toSec();
    if (shouldRevokeFinishedStop(insinav_stop_verified_status_, time_since_finish,
            FSMConstantsReal::LIGHTGLUE_FINISH_REVOKE_TIME)) {
      fd_->have_finished_ = false;
      finish_completed_time_ = ros::Time(0);
      finish_wait_start_time_ = ros::Time(0);
      lightglue_verified_start_time_ = ros::Time(0);
      finish_goal_pos_valid_ = false;
      lightglue_close_stop_candidate_active_ = false;
      ROS_WARN("[ExplorationFSMReal] LightGlue became PENDING_FAR %.1f s after finish; revoke finish and resume normal planning.",
          time_since_finish);
      if (state_ == RealFSM::State::FINISH) {
        transitState(RealFSM::State::PLAN_TRAJ, "LightGlueFinishRevoked");
      }
      return;
    }
  }

  if (!shouldStopMotionForLightGlueStatus(insinav_stop_verified_status_)) {
    if (insinav_stop_verified_status_ == "PENDING_FAR") {
      lightglue_close_stop_candidate_active_ = false;
    }
    return;
  }

  if (state_ != RealFSM::State::EXEC_TRAJ &&
      state_ != RealFSM::State::REPLAN &&
      state_ != RealFSM::State::PLAN_TRAJ) {
    return;
  }

  emergencyStop();
  fd_->static_state_ = true;

  if (insinav_stop_verified_status_ == "VERIFIED") {
    if (lightglue_verified_start_time_.isZero()) {
      lightglue_verified_start_time_ = ros::Time::now();
    }
    ROS_WARN("[ExplorationFSMReal] INSiNav target verified. Stop motion and hold briefly before final finish.");
  } else {
    lightglue_verified_start_time_ = ros::Time(0);
    finish_wait_start_time_ = ros::Time(0);
    lightglue_close_stop_candidate_active_ = true;
    ROS_WARN("[ExplorationFSMReal] INSiNav stop candidate is close; stop motion and wait for confirmation.");
  }

  transitState(RealFSM::State::FINISH, "insinnavStopVerifiedCallback");
}


}  // namespace apexnav_planner
