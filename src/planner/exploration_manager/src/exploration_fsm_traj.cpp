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
  nh.param("fsm/visualize_objects", visualize_object_markers_, false);
  nh.param("fsm/tracking_abort_distance", tracking_abort_distance_,
      FSMConstantsReal::DEFAULT_TRACKING_ABORT_DISTANCE);

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
  manual_cmd_pub_ = nh.advertise<geometry_msgs::Twist>("/cmd_vel", 10);

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
      if (manual_goal_active_) {
        stepManualGoalFallback();
        exec_timer_.start();
        return;
      }
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
          expl_manager_->setCloseObjectApproachOnce(true);
          ROS_WARN("[ExplorationFSMReal] LightGlue target is verified but still far; continue approaching instead of waiting in FINISH.");
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
            expl_manager_->setCloseObjectApproachOnce(true);
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
          if (shouldResumePlanningFromPendingLightGlue(
                  fd_->final_result_, insinav_stop_verified_status_)) {
            expl_manager_->setSkipObjectNavigationOnce(true);
            ROS_WARN("[ExplorationFSMReal] LightGlue status is PENDING without a close stop candidate; resume exploration instead of holding.");
            transitState(RealFSM::State::PLAN_TRAJ, "LightGluePendingResume");
            exec_timer_.start();
            return;
          }

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
      if (manual_goal_active_) {
        stepManualGoalFallback();
        exec_timer_.start();
        return;
      }
      // Plan trajectory based on current state
      if (fd_->static_state_) {
        // Robot is static, use current odometry
        fd_->start_pt_ = fd_->odom_pos_;
        fd_->start_vel_ = fd_->odom_vel_;
        fd_->start_yaw_(0) = fd_->odom_yaw_;
        fd_->start_yaw_(1) = fd_->start_yaw_(2) = 0.0;
      }
      else {
        // Robot is moving, predict future state for smooth replanning
        LocalTrajectory* info = &expl_manager_->gcopter_->local_trajectory_;
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
        ROS_WARN("[Real] Plan trajectory failed");
        fd_->static_state_ = true;
        if (shouldSkipObjectNavigationAfterPlanFailure(
                fd_->final_result_, insinav_stop_verified_status_)) {
          expl_manager_->setSkipObjectNavigationOnce(true);
          finish_wait_start_time_ = ros::Time(0);
          ROS_WARN("[Real] Search-object trajectory failed while LightGlue is not verified (status=%s); skip object navigation once and resume exploration.",
              insinav_stop_verified_status_.c_str());
        }
        else if (shouldContinueApproachingObject(
                     fd_->final_result_, insinav_stop_verified_status_)) {
          expl_manager_->setCloseObjectApproachOnce(true);
          finish_wait_start_time_ = ros::Time(0);
          ROS_WARN("[Real] Search-object trajectory failed while target is PENDING_FAR; retry closer object approach.");
        }
        else if (shouldWaitForLightGlueAfterPlanFailure(fd_->final_result_)) {
          finish_wait_start_time_ = ros::Time(0);
          ROS_WARN("[Real] Search-object trajectory failed; waiting for LightGlue verification before retry.");
          transitState(RealFSM::State::FINISH, "SearchObjectPlanFailed");
        }
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

      // Replan if trajectory is almost finished
      if (time_to_end < fp_->replan_traj_end_threshold_) {
        transitState(RealFSM::State::PLAN_TRAJ, "FSM");
        ROS_WARN("[Real] Replan: traj fully executed");
        exec_timer_.start();
        return;
      }

      if (shouldReplanForFrontierChange(
              t_cur,
              fp_->replan_frontier_change_delay_,
              fd_->final_result_,
              expl_manager_->frontier_map2d_->isAnyFrontierChanged())) {
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
  const bool frontier_changed = updateFrontierAndObject();

  const auto previous_path = expl_manager_->ed_->next_best_path_;
  const auto previous_goal = expl_manager_->ed_->next_pos_;
  const int previous_final_result = fd_->final_result_;

  // Call exploration manager to find next best point
  int expl_res = expl_manager_->planNextBestPoint(fd_->start_pt_, fd_->start_yaw_(0));

  // Determine final result based on exploration result
  if (expl_res == EXPL_RESULT::EXPLORATION)
    fd_->final_result_ = FINAL_RESULT::EXPLORE;
  else if (expl_res == EXPL_RESULT::NO_COVERABLE_FRONTIER ||
           expl_res == EXPL_RESULT::NO_PASSABLE_FRONTIER)
    fd_->final_result_ = FINAL_RESULT::NO_FRONTIER;
  else
    fd_->final_result_ = FINAL_RESULT::SEARCH_OBJECT;

  if (shouldKeepPreviousExplorationGoal(
          previous_final_result,
          fd_->final_result_,
          frontier_changed,
          !previous_path.empty())) {
    expl_manager_->ed_->next_best_path_ = previous_path;
    expl_manager_->ed_->next_pos_ = previous_goal;
  }

  // Publish exploration result
  std_msgs::Int32 expl_result_msg;
  expl_result_msg.data = fd_->final_result_;
  expl_result_pub_.publish(expl_result_msg);

  if (fd_->final_result_ == FINAL_RESULT::NO_FRONTIER) {
    finish_goal_pos_valid_ = false;
    ROS_WARN("[Real] No (passable) frontier");
    return TrajPlannerResult::MISSION_COMPLETE;
  }

  // Select a safe local target from the global path. The global next_pos_ remains
  // the exploration target, but the vehicle tracks this local point like apexnavmain.
  Eigen::Vector2d object_goal_pos = expl_manager_->ed_->next_pos_;
  finish_goal_pos_ = object_goal_pos;
  finish_goal_pos_valid_ = true;

  if (shouldDormantExplorationGoalBeforeTrajectory(
          fd_->final_result_,
          fd_->start_pt_.head(2),
          object_goal_pos,
          FSMConstantsReal::FORCE_DORMANT_DISTANCE)) {
    expl_manager_->frontier_map2d_->setForceDormantFrontier(object_goal_pos);
    fd_->dormant_frontier_flag_ = true;
    ROS_WARN("[Real] Exploration frontier is too close for a stable trajectory; robot=(%.2f,%.2f) global_goal=(%.2f,%.2f), dist=%.2f. force dormant and replan.",
        fd_->start_pt_(0), fd_->start_pt_(1), object_goal_pos(0), object_goal_pos(1),
        (object_goal_pos - fd_->start_pt_.head(2)).norm());
    return TrajPlannerResult::FAILED;
  }

  Eigen::Vector2d local_goal_pos = object_goal_pos;
  double local_goal_yaw = 0.0;
  auto path = expl_manager_->ed_->next_best_path_;
  bool local_target_valid =
      selectLocalTarget(fd_->start_pt_.head(2), path, 4.0, local_goal_pos, local_goal_yaw);
  Eigen::Vector2d goal_pos =
      selectTrajectoryGoalForMode(fd_->final_result_, object_goal_pos, local_goal_pos);
  double goal_yaw = local_goal_yaw;
  LocalTargetPlanningDebugInfo planning_debug_info;
  planning_debug_info.robot_pos = fd_->start_pt_.head(2);
  planning_debug_info.global_goal_pos = object_goal_pos;
  planning_debug_info.local_target_pos = goal_pos;
  planning_debug_info.local_target_valid = local_target_valid;
  planning_debug_info.path_size = static_cast<int>(path.size());
  planning_debug_info.path_back_pos = path.empty() ? object_goal_pos : path.back();

  // Check if reached object
  if (shouldCompleteSearchObjectMission(
          fd_->final_result_,
          fd_->start_pt_.head(2),
          object_goal_pos,
          local_goal_pos,
          FSMConstantsReal::REACH_DISTANCE)) {
    if (shouldCompleteObjectApproach(fd_->final_result_, insinav_stop_verified_status_)) {
      ROS_WARN("[Real] Object-map approach point reached and LightGlue status=%s; finishing.",
          insinav_stop_verified_status_.c_str());
      return TrajPlannerResult::MISSION_COMPLETE;
    }

    if (shouldRetryObjectApproachAfterMapPointReached(
            fd_->final_result_, insinav_stop_verified_status_)) {
      if (shouldContinueApproachingObject(fd_->final_result_, insinav_stop_verified_status_)) {
        expl_manager_->setCloseObjectApproachOnce(true);
        ROS_WARN("[Real] Object-map approach point reached but LightGlue status=%s; retry perception-guided approach.",
            insinav_stop_verified_status_.c_str());
      } else {
        ROS_WARN("[Real] Object-map approach point reached but LightGlue status=%s; resume exploration instead of waiting.",
            insinav_stop_verified_status_.c_str());
      }
      return TrajPlannerResult::FAILED;
    }

    ROS_WARN("[Real] Object-map approach point reached; waiting for LightGlue verification.");
    return TrajPlannerResult::MISSION_COMPLETE;
  }

  // Prepare state for trajectory planning
  Eigen::VectorXd goal_state(5), current_state(5);
  Eigen::Vector3d current_control(0.0, 0.0, 0.0);
  double start_vel = Eigen::Vector2d(fd_->start_vel_(0), fd_->start_vel_(1)).norm();
  current_state << fd_->start_pt_(0), fd_->start_pt_(1), fd_->start_yaw_(0), 0.0, start_vel;

  std::vector<std::pair<Eigen::Vector2d, double>> candidate_targets;
  LocalPlannerCandidateDebugStats candidate_debug_stats;
  auto appendCandidate = [&](const Eigen::Vector2d& pos, const std::vector<double>& yaw_candidates) {
    appendLocalPlannerCandidate(pos, yaw_candidates, fd_->start_pt_.head(2), 0.20, 0.10, 0.20,
        [&](const Eigen::Vector2d& candidate_pos, double candidate_yaw) {
          return expl_manager_->kinoastar_->isCollisionPosYaw(candidate_pos, candidate_yaw);
        },
        candidate_targets, &candidate_debug_stats);
  };

  if (local_target_valid || fd_->final_result_ == FINAL_RESULT::EXPLORE) {
    appendCandidate(goal_pos, { goal_yaw, fd_->start_yaw_(0) });
  }

  if (!path.empty()) {
    const Eigen::Vector2d fallback_anchor = selectFallbackCandidateAnchor(
        fd_->start_pt_.head(2), goal_pos, object_goal_pos, 0.30, 0.60);
    int nearest_idx = 0;
    double nearest_dist = std::numeric_limits<double>::max();
    for (int i = 0; i < (int)path.size(); ++i) {
      double dist = (path[i] - fallback_anchor).norm();
      if (dist < nearest_dist) {
        nearest_dist = dist;
        nearest_idx = i;
      }
    }

    const auto sampled_candidates = sampleBackwardPathCandidates(path, nearest_idx, 0.10);
    for (const auto& sampled_candidate : sampled_candidates) {
      appendCandidate(
          sampled_candidate.pos, { sampled_candidate.yaw, fd_->start_yaw_(0) });
    }
  }

  for (const auto& candidate : candidate_targets) {
    planning_debug_info.nearest_candidate_distance = std::min(
        planning_debug_info.nearest_candidate_distance,
        (candidate.first - planning_debug_info.robot_pos).norm());
  }

  if (candidate_targets.empty()) {
    if (fd_->final_result_ == FINAL_RESULT::EXPLORE) {
      expl_manager_->frontier_map2d_->setForceDormantFrontier(expl_manager_->ed_->next_pos_);
      fd_->dormant_frontier_flag_ = true;
      ROS_WARN("[Real] No vehicle-feasible local target for current frontier; %s; candidates=0, rejected: collision=%d too_close=%d duplicate=%d. force dormant and replan.",
          formatLocalTargetPlanningDebugSummary(planning_debug_info).c_str(),
          candidate_debug_stats.collision_rejections, candidate_debug_stats.too_close_rejections,
          candidate_debug_stats.duplicate_rejections);
    }
    return TrajPlannerResult::FAILED;
  }

  // Plan trajectory using GCopter, falling back to earlier safe points if needed.
  for (const auto& candidate : candidate_targets) {
    goal_state << candidate.first(0), candidate.first(1), candidate.second, 0.0, 0.0;
    bool traj_res = expl_manager_->planTrajectory(current_state, goal_state, current_control);
    if (!traj_res)
      continue;

    expl_manager_->ed_->next_local_pos_ = candidate.first;
    auto info = &expl_manager_->gcopter_->local_trajectory_;
    info->start_time = (ros::Time::now() - time_r).toSec() > 0 ? ros::Time::now() : time_r;
    fd_->newest_traj_ = expl_manager_->gcopter_->local_trajectory_;
    return TrajPlannerResult::SUCCESS;
  }

  if (fd_->final_result_ == FINAL_RESULT::EXPLORE) {
    const auto& fallback_target = candidate_targets.front();
    const double fallback_dist = (fallback_target.first - current_state.head(2)).norm();
    if (shouldUseExplorationManualFallback(
            fallback_dist, FSMConstantsReal::EXPLORATION_FALLBACK_MAX_DIRECT_DISTANCE)) {
      activateManualFallback(
          fallback_target.first, fallback_target.second, true, expl_manager_->ed_->next_pos_);
      expl_manager_->ed_->next_local_pos_ = fallback_target.first;
      ROS_WARN("[Real] Switching to direct cmd_vel fallback for exploration local target: dist=%.2f.",
          fallback_dist);
    }
    else {
      expl_manager_->frontier_map2d_->setForceDormantFrontier(expl_manager_->ed_->next_pos_);
      fd_->dormant_frontier_flag_ = true;
      ROS_WARN("[Real] Exploration local target %.2f m away is not suitable for direct cmd_vel fallback; force dormant and replan.",
          fallback_dist);
    }
  }
  else if (shouldUseSearchObjectManualFallback(
               fd_->final_result_, insinav_stop_verified_status_)) {
    const auto& fallback_target = candidate_targets.front();
    activateManualFallback(fallback_target.first, fallback_target.second, false, object_goal_pos);
    expl_manager_->ed_->next_local_pos_ = fallback_target.first;
    ROS_WARN("[Real] Switching to direct cmd_vel fallback for verified search-object local target.");
  }

  return TrajPlannerResult::FAILED;
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
  const double target_clearance =
      computeLocalTargetClearance(
          expl_manager_->kinoastar_->width_, 0.005,
          expl_manager_->sdf_map_->getObstaclesInflation(), 0.06);

  if (path.empty()) {
    expl_manager_->ed_->next_local_pos_ = target_pos;
    return false;
  }

  auto isFootprintCollision = [&](const Eigen::Vector2d& pos, double yaw) {
    return expl_manager_->kinoastar_->isCollisionPosYaw(pos, yaw);
  };

  // First, try to find a collision-free target from the end of path
  Eigen::Vector2d safe_target_pos = target_pos;
  double safe_target_yaw = target_yaw;
  PathPoseCandidate safe_pose;
  if (findLastFootprintSafePathPose(path, isFootprintCollision, safe_pose)) {
    safe_target_pos = safe_pose.pos;
    safe_target_yaw = safe_pose.yaw;
  }
  target_pos = safe_target_pos;
  target_yaw = safe_target_yaw;

  // Find closest path point to current position
  int start_path_id = 0;
  double min_dist = std::numeric_limits<double>::max();
  for (int i = 0; i < (int)path.size() - 1; i++) {
    Eigen::Vector2d pos = path[i];
    if ((pos - current_pos).norm() < min_dist) {
      min_dist = (pos - current_pos).norm();
      start_path_id = i + 1;
    }
  }

  // Select local target within local_distance
  double len = (path[start_path_id] - current_pos).norm();
  for (int i = start_path_id + 1; i < (int)path.size(); i++) {
    len += (path[i] - path[i - 1]).norm();
    if (len > local_distance && (current_pos - path[i - 1]).norm() > target_clearance) {
      Eigen::Vector2d candidate_pos = path[i - 1];
      double candidate_yaw = atan2(path[i](1) - path[i - 1](1), path[i](0) - path[i - 1](0));
      if (!isFootprintCollision(candidate_pos, candidate_yaw)) {
        target_pos = candidate_pos;
        target_yaw = candidate_yaw;
      }
      break;
    }
  }

  // Gradient-based safety adjustment
  double step_size = 0.05;
  double tolerance = 1e-3;
  int max_iterations = 30;

  for (int i = 0; i < max_iterations; ++i) {
    Eigen::Vector2d prev_pos = target_pos;

    // Get gradient from SDF map
    Eigen::Vector2d grad;
    double dist = expl_manager_->sdf_map_->getDistWithGrad(target_pos, grad);

    if (dist > target_clearance)
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

  // Final safeguard: if local target is still not collision-free, walk backward along path
  // and keep the first feasible point.
  if (isFootprintCollision(target_pos, target_yaw)) {
    int nearest_idx = 0;
    double nearest_dist = std::numeric_limits<double>::max();
    for (int i = 0; i < (int)path.size(); ++i) {
      double dist = (path[i] - target_pos).norm();
      if (dist < nearest_dist) {
        nearest_dist = dist;
        nearest_idx = i;
      }
    }
    const auto sampled_candidates = sampleBackwardPathCandidates(path, nearest_idx, 0.05);
    for (const auto& candidate : sampled_candidates) {
      if (!isFootprintCollision(candidate.pos, candidate.yaw)) {
        target_pos = candidate.pos;
        target_yaw = candidate.yaw;
        break;
      }
    }
  }

  if (isFootprintCollision(target_pos, target_yaw)) {
    expl_manager_->ed_->next_local_pos_ = target_pos;
    return false;
  }

  // Store selected local target
  expl_manager_->ed_->next_local_pos_ = target_pos;
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

  // Draw global frontier/object target. This is the actual next_pos_ selected by
  // the exploration manager, separate from the local tracking point below.
  std::vector<Eigen::Vector2d> global_points;
  if (!ed_ptr->next_best_path_.empty()) {
    global_points.push_back(ed_ptr->next_pos_);
  }
  visualization_->drawSpheres(vec2dTo3d(global_points, 0.22), fp_->vis_scale_ * 3.5,
      Eigen::Vector4d(1.0, 0.85, 0.1, 1), "global_point", 1, 6);

  // Draw next local point
  std::vector<Eigen::Vector2d> local_points;
  local_points.push_back(ed_ptr->next_local_pos_);
  visualization_->drawSpheres(vec2dTo3d(local_points), fp_->vis_scale_ * 3,
      Eigen::Vector4d(0.2, 0.2, 1.0, 1), "local_point", 1, 6);

  visualization_->drawLines(vec2dTo3d(ed_ptr->tsp_tour_), fp_->vis_scale_ / 1.25,
      Eigen::Vector4d(0.2, 1, 0.2, 1), "tsp_tour", 0, 6);
}

void ExplorationFSMReal::clearVisMarker()
{
  for (int i = 0; i < 500; ++i) {
    visualization_->drawCubes({}, fp_->vis_scale_, Eigen::Vector4d(0, 0, 0, 1), "frontier", i, 4);
    visualization_->drawCubes(
        {}, fp_->vis_scale_, Eigen::Vector4d(0, 0, 0, 1), "dormant_frontier", i, 4);
    visualization_->drawCubes({}, fp_->vis_scale_, Eigen::Vector4d(0, 0, 0, 1), "object", i, 4);
  }
  visualization_->drawLines({}, fp_->vis_scale_, Eigen::Vector4d(0, 0, 1, 1), "next_path", 1, 6);
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
  frt_map->getFrontiers(ed->frontiers_, ed->frontier_averages_);
  frt_map->getDormantFrontiers(ed->dormant_frontiers_, ed->dormant_frontier_averages_);
  obj_map->getObjects(ed->objects_, ed->object_averages_, ed->object_labels_);

  return change_flag;
}

void ExplorationFSMReal::frontierCallback(const ros::TimerEvent& e)
{
  // Update frontiers and visualize in idle states
  if (state_ != RealFSM::State::WAIT_TRIGGER && state_ != RealFSM::State::FINISH)
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

    // Manual goals from RViz are often clicked on the free/unknown boundary.
    // First find a point-robot path, then refine the clicked pose to the last safe point
    // on that path before invoking the vehicle-sized kinodynamic planner.
    expl_manager_->path_finder_->reset();
    if (expl_manager_->path_finder_->astarSearch(
            current_state.head(2), goal_state.head(2), 0.25, 0.05) != Astar2D::REACH_END) {
      ROS_WARN("[Real] Manual goal path search failed: no point-robot path to clicked pose.");
      return;
    }

    auto manual_path = expl_manager_->path_finder_->getPath();
    if (manual_path.empty()) {
      ROS_WARN("[Real] Manual goal path search returned empty path.");
      return;
    }

    auto computePathYaw = [&](int idx) {
      if (idx < (int)manual_path.size() - 1) {
        return atan2(manual_path[idx + 1](1) - manual_path[idx](1),
            manual_path[idx + 1](0) - manual_path[idx](0));
      }
      if (idx > 0) {
        return atan2(manual_path[idx](1) - manual_path[idx - 1](1),
            manual_path[idx](0) - manual_path[idx - 1](0));
      }
      return yaw;
    };

    LocalPlannerCandidateDebugStats manual_candidate_debug_stats;
    auto appendCandidate = [&](const Eigen::Vector2d& pos,
                               const std::vector<double>& yaw_candidates,
                               std::vector<std::pair<Eigen::Vector2d, double>>& candidates) {
      appendLocalPlannerCandidate(pos, yaw_candidates, current_state.head(2), 0.25, 0.08, 0.20,
          [&](const Eigen::Vector2d& candidate_pos, double candidate_yaw) {
            return expl_manager_->kinoastar_->isCollisionPosYaw(candidate_pos, candidate_yaw);
          },
          candidates, &manual_candidate_debug_stats);
    };

    Eigen::Vector2d refined_goal_pos = goal_state.head(2);
    double refined_goal_yaw = yaw;
    bool refined_goal_valid =
        selectLocalTarget(current_state.head(2), manual_path, 4.0, refined_goal_pos, refined_goal_yaw);

    std::vector<std::pair<Eigen::Vector2d, double>> candidate_goals;
    if (refined_goal_valid) {
      appendCandidate(refined_goal_pos, { refined_goal_yaw, current_state[2], yaw }, candidate_goals);
    }

    // Try multiple fallback points along the point-robot path. In tight spaces the clicked
    // endpoint is often too close to occupied/unknown boundaries for the vehicle-sized planner.
    for (int i = (int)manual_path.size() - 1; i >= 0; --i) {
      Eigen::Vector2d pos = manual_path[i];
      std::vector<double> yaw_candidates = { computePathYaw(i), current_state[2], yaw };
      appendCandidate(pos, yaw_candidates, candidate_goals);
    }

    bool plan_ok = false;
    for (const auto& candidate : candidate_goals) {
      goal_state << candidate.first(0), candidate.first(1), candidate.second, 0.0, 0.0;
      plan_ok = expl_manager_->planTrajectory(current_state, goal_state, current_control);
      if (plan_ok) {
        refined_goal_pos = candidate.first;
        refined_goal_yaw = candidate.second;
        break;
      }
    }

    if (!plan_ok) {
      ROS_WARN("[Real] Manual goal trajectory planning failed after goal refinement. clicked=(%.2f, %.2f, %.2f), refined=(%.2f, %.2f, %.2f), candidates=%zu, rejected: collision=%d too_close=%d duplicate=%d",
          x, y, yaw, refined_goal_pos(0), refined_goal_pos(1), refined_goal_yaw, candidate_goals.size(),
          manual_candidate_debug_stats.collision_rejections,
          manual_candidate_debug_stats.too_close_rejections,
          manual_candidate_debug_stats.duplicate_rejections);
      if (!candidate_goals.empty()) {
        activateManualFallback(candidate_goals.front().first, candidate_goals.front().second, false,
            candidate_goals.front().first);
        ROS_WARN("[Real] Switching to direct cmd_vel fallback for manual goal.");
      }
      return;
    }

    trajectory_manager::PolyTraj poly_msg;
    expl_manager_->ed_->next_local_pos_ = refined_goal_pos;
    polyTraj2ROSMsg(expl_manager_->gcopter_->local_trajectory_, poly_msg);
    poly_traj_pub_.publish(poly_msg);
    manual_goal_active_ = false;
    manual_goal_replan_after_finish_ = false;
    ROS_INFO("[Real] Published manual goal trajectory toward refined target: x=%.2f, y=%.2f, yaw=%.2f",
        refined_goal_pos(0), refined_goal_pos(1), refined_goal_yaw);
  }
  ROS_INFO("[Real] Received goal pose: x=%.2f, y=%.2f, yaw=%.2f", x, y, yaw);
}

void ExplorationFSMReal::activateManualFallback(
    const Eigen::Vector2d& target_pos, double target_yaw, bool replan_after_finish,
    const Eigen::Vector2d& dormant_pos)
{
  if (shouldStopTrajectoryServerForManualFallback(
          manual_goal_active_, manual_goal_pos_, target_pos, 0.10)) {
    stop_pub_.publish(std_msgs::Empty());
  }

  manual_goal_pos_ = target_pos;
  manual_goal_dormant_pos_ = dormant_pos;
  manual_goal_yaw_ = target_yaw;
  manual_goal_active_ = true;
  manual_goal_replan_after_finish_ = replan_after_finish;
  manual_goal_best_dist_ = std::numeric_limits<double>::infinity();
  manual_goal_last_progress_time_ = ros::Time::now();
  manual_goal_turn_dir_ = 1.0;
}

double ExplorationFSMReal::stabilizeFallbackYawError(double yaw_error)
{
  constexpr double kPiFlipBand = 2.8;
  constexpr double kPiReleaseBand = 2.4;

  if (std::fabs(yaw_error) > kPiFlipBand) {
    if (std::fabs(yaw_error) < M_PI - 1e-3) {
      manual_goal_turn_dir_ = yaw_error >= 0.0 ? 1.0 : -1.0;
    }
    return manual_goal_turn_dir_ * std::fabs(yaw_error);
  }

  if (std::fabs(yaw_error) < kPiReleaseBand) {
    manual_goal_turn_dir_ = yaw_error >= 0.0 ? 1.0 : -1.0;
  }

  return yaw_error;
}

void ExplorationFSMReal::stepManualGoalFallback()
{
  geometry_msgs::Twist cmd;
  Eigen::Vector2d current_pos(fd_->odom_pos_(0), fd_->odom_pos_(1));
  Eigen::Vector2d diff = manual_goal_pos_ - current_pos;
  double dist = diff.norm();

  if (dist + FSMConstantsReal::MANUAL_FALLBACK_PROGRESS_EPS < manual_goal_best_dist_) {
    manual_goal_best_dist_ = dist;
    manual_goal_last_progress_time_ = ros::Time::now();
  }

  const double reach_distance = manual_goal_replan_after_finish_ ?
      FSMConstantsReal::EXPLORATION_FALLBACK_REACH_DISTANCE :
      FSMConstantsReal::MANUAL_FALLBACK_REACH_DISTANCE;

  if (dist < reach_distance) {
    const bool force_dormant = shouldForceDormantAfterExplorationFallback(
        manual_goal_replan_after_finish_, dist, reach_distance);

    manual_goal_active_ = false;
    manual_cmd_pub_.publish(cmd);
    ROS_INFO("[Real] Manual cmd_vel fallback reached target: dist=%.2f threshold=%.2f.",
        dist, reach_distance);
    if (manual_goal_replan_after_finish_) {
      manual_goal_replan_after_finish_ = false;
      if (force_dormant) {
        expl_manager_->frontier_map2d_->setForceDormantFrontier(manual_goal_dormant_pos_);
        fd_->dormant_frontier_flag_ = true;
        ROS_WARN("[Real] Exploration fallback target reached; force dormant current frontier.");
      }
      transitState(RealFSM::State::PLAN_TRAJ, "ManualCmdFallbackReached");
    }
    return;
  }

  if ((ros::Time::now() - manual_goal_last_progress_time_).toSec() >
      FSMConstantsReal::MANUAL_FALLBACK_STUCK_TIMEOUT) {
    manual_goal_active_ = false;
    manual_cmd_pub_.publish(cmd);
    ROS_WARN("[Real] Manual cmd_vel fallback stuck: dist=%.2f best=%.2f. Replanning.",
        dist, manual_goal_best_dist_);
    if (manual_goal_replan_after_finish_) {
      manual_goal_replan_after_finish_ = false;
      expl_manager_->frontier_map2d_->setForceDormantFrontier(manual_goal_dormant_pos_);
      transitState(RealFSM::State::PLAN_TRAJ, "ManualCmdFallbackStuck");
    }
    return;
  }

  double desired_yaw = std::atan2(diff(1), diff(0));
  double yaw_error = std::atan2(std::sin(desired_yaw - fd_->odom_yaw_),
                                std::cos(desired_yaw - fd_->odom_yaw_));
  yaw_error = stabilizeFallbackYawError(yaw_error);

  const auto fallback_cmd = computeManualFallbackCommand(dist, yaw_error);
  cmd.linear.x = fallback_cmd.first;
  cmd.angular.z = fallback_cmd.second;

  manual_cmd_pub_.publish(cmd);
  ROS_INFO_THROTTLE(0.5, "[Real] Manual fallback cmd: dist=%.2f yaw_err=%.2f vx=%.2f wz=%.2f",
      dist, yaw_error, cmd.linear.x, cmd.angular.z);
}

void ExplorationFSMReal::emergencyStop()
{
  fd_->static_state_ = true;
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
    ROS_ERROR("[Real] Odom far from traj: planned=(%.2f, %.2f), odom=(%.2f, %.2f), err=%.2f, status=%s, result=%d. Stop!!!",
        planned_pos(0), planned_pos(1), odom_pos(0), odom_pos(1), tracking_error,
        insinav_stop_verified_status_.c_str(), fd_->final_result_);
    emergencyStop();
    fd_->static_state_ = true;

    if (shouldUseSearchObjectManualFallback(fd_->final_result_, insinav_stop_verified_status_)) {
      const Eigen::Vector2d fallback_target = expl_manager_->ed_->next_local_pos_;
      const Eigen::Vector2d diff = fallback_target - odom_pos;
      const double fallback_yaw = diff.norm() > 1e-3 ? std::atan2(diff(1), diff(0)) : fd_->odom_yaw_;
      activateManualFallback(fallback_target, fallback_yaw, false, finish_goal_pos_);
      ROS_WARN("[Real] Switching to direct cmd_vel fallback after search-object trajectory tracking error.");
    }

    transitState(RealFSM::State::PLAN_TRAJ, "Odom Far From Trajectory");
    return;
  }

  // Time-sampled safety check - use inflated map to detect obstacles
  double time_horizon = 2.5;  // Check trajectory for next 2.5 seconds
  double sample_dt = 0.1;     // Sample every 0.1 seconds

  for (double t_check = t_cur;
      t_check <= min(t_cur + time_horizon, expl_manager_->gcopter_->local_trajectory_.duration);
      t_check += sample_dt) {
    Eigen::Vector3d check_pos = expl_manager_->gcopter_->local_trajectory_.traj.getPos(t_check);
    Eigen::Vector2d check_pos_2d = check_pos.head(2);

    // Skip positions too close to origin
    if ((check_pos_2d - Eigen::Vector2d(0.0, 0.0)).norm() < 1.5)
      continue;

    if (expl_manager_->sdf_map_->getInflateOccupancy(check_pos_2d)) {
      ROS_ERROR("[Real] Safety Stop!!! Obstacle detected (%.2f, %.2f) at time %.2f",
          check_pos_2d(0), check_pos_2d(1), t_check);
      emergencyStop();
      transitState(RealFSM::State::PLAN_TRAJ, "Trajectory Safety Stop");
      break;
    }
  }
}

void ExplorationFSMReal::publishRobotMarker()
{
  const double robot_height = FSMConstantsReal::ROBOT_HEIGHT;
  const double robot_radius = FSMConstantsReal::ROBOT_RADIUS;

  // Create robot body cylinder marker
  visualization_msgs::Marker robot_marker;
  robot_marker.header.frame_id = "odom";
  robot_marker.header.stamp = ros::Time::now();
  robot_marker.ns = "robot_position";
  robot_marker.id = 0;
  robot_marker.type = visualization_msgs::Marker::CYLINDER;
  robot_marker.action = visualization_msgs::Marker::ADD;

  robot_marker.pose.position.x = fd_->odom_pos_(0);
  robot_marker.pose.position.y = fd_->odom_pos_(1);
  robot_marker.pose.position.z = fd_->odom_pos_(2) + robot_height / 2.0;

  robot_marker.pose.orientation.x = fd_->odom_orient_.x();
  robot_marker.pose.orientation.y = fd_->odom_orient_.y();
  robot_marker.pose.orientation.z = fd_->odom_orient_.z();
  robot_marker.pose.orientation.w = fd_->odom_orient_.w();

  robot_marker.scale.x = robot_radius * 2;
  robot_marker.scale.y = robot_radius * 2;
  robot_marker.scale.z = robot_height;

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
  arrow_marker.pose.position.z = fd_->odom_pos_(2) + robot_height;

  arrow_marker.pose.orientation.x = fd_->odom_orient_.x();
  arrow_marker.pose.orientation.y = fd_->odom_orient_.y();
  arrow_marker.pose.orientation.z = fd_->odom_orient_.z();
  arrow_marker.pose.orientation.w = fd_->odom_orient_.w();

  arrow_marker.scale.x = robot_radius + 0.13;
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
  std::string state_str[] = { "INIT", "WAIT_TRIGGER", "PLAN_TRAJ", "EXEC_TRAJ", "REPLAN",
    "FINISH" };
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
      expl_manager_->setCloseObjectApproachOnce(true);
      ROS_WARN("[ExplorationFSMReal] LightGlue became PENDING_FAR %.1f s after finish; revoke finish and continue approaching.",
          time_since_finish);
      if (state_ == RealFSM::State::FINISH) {
        transitState(RealFSM::State::PLAN_TRAJ, "LightGlueFinishRevoked");
      }
      return;
    }
  }

  if (shouldPreemptExplorationFallbackForLightGlueStatus(
          insinav_stop_verified_status_, manual_goal_active_, manual_goal_replan_after_finish_)) {
    manual_goal_active_ = false;
    manual_goal_replan_after_finish_ = false;
    manual_cmd_pub_.publish(geometry_msgs::Twist());
    stop_pub_.publish(std_msgs::Empty());
    fd_->static_state_ = true;
    expl_manager_->setCloseObjectApproachOnce(true);
    ROS_WARN("[ExplorationFSMReal] LightGlue became PENDING_FAR during exploration fallback; preempt fallback and resume object approach.");
    if (state_ == RealFSM::State::PLAN_TRAJ ||
        state_ == RealFSM::State::WAIT_TRIGGER ||
        state_ == RealFSM::State::FINISH) {
      transitState(RealFSM::State::PLAN_TRAJ, "LightGluePendingFarPreempt");
    }
    return;
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
  manual_goal_active_ = false;
  manual_cmd_pub_.publish(geometry_msgs::Twist());
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
