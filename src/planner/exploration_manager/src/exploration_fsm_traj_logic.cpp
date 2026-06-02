#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>

#include <exploration_manager/exploration_data.h>
#include <exploration_manager/exploration_fsm_traj_logic.h>

namespace apexnav_planner {

bool shouldCompleteSearchObjectMission(int final_result,
    const Eigen::Vector2d& current_pos,
    const Eigen::Vector2d& object_goal_pos,
    const Eigen::Vector2d& local_goal_pos,
    double reach_distance)
{
  if (final_result != FINAL_RESULT::SEARCH_OBJECT) {
    return false;
  }

  const double object_goal_dist = (current_pos - object_goal_pos).norm();
  const double local_goal_dist = (current_pos - local_goal_pos).norm();

  return object_goal_dist < reach_distance && object_goal_dist <= local_goal_dist + 1e-6;
}

bool shouldUseKinoAstarPath(int path_node_count, bool has_path)
{
  return has_path && path_node_count > 0 && path_node_count <= 2;
}

std::vector<double> sampleKinoAstarArcInputs(double step_arc, bool include_reverse)
{
  std::vector<double> arcs;
  if (step_arc <= 1e-6) {
    return arcs;
  }

  if (include_reverse) {
    for (double arc = -step_arc; arc < -1e-6; arc += 0.5 * step_arc) {
      arcs.push_back(arc);
    }
  }

  for (double arc = 0.5 * step_arc; arc <= step_arc + 1e-6; arc += 0.5 * step_arc) {
    arcs.push_back(arc);
  }

  return arcs;
}

bool shouldTreatOccupancyAsKinoCollision(int occupancy, bool unknown_as_collision)
{
  return occupancy == SDFMap2D::OCCUPIED ||
      (unknown_as_collision && occupancy == SDFMap2D::UNKNOWN);
}

double computeFootprintSupportRadius(double length,
    double width,
    double odom_to_center_x,
    double odom_to_center_y)
{
  const double half_length = 0.5 * length;
  const double half_width = 0.5 * width;
  const double xs[] = { odom_to_center_x + half_length, odom_to_center_x - half_length };
  const double ys[] = { odom_to_center_y + half_width, odom_to_center_y - half_width };

  double max_radius = 0.0;
  for (double x : xs) {
    for (double y : ys) {
      max_radius = std::max(max_radius, std::hypot(x, y));
    }
  }
  return max_radius;
}

double computeLocalTargetClearance(
    double width,
    double safety_margin,
    double obstacles_inflation,
    double minimum_clearance)
{
  return std::max(minimum_clearance, 0.5 * width + safety_margin - obstacles_inflation);
}

double computeWallProximityPenalty(
    double obstacle_distance,
    double preferred_clearance,
    double penalty_weight)
{
  if (obstacle_distance >= preferred_clearance) {
    return 0.0;
  }

  const double clearance_deficit = preferred_clearance - obstacle_distance;
  return penalty_weight * clearance_deficit * clearance_deficit;
}

bool shouldQueueLocalPlannerCandidate(
    bool collision_free,
    double distance_to_robot,
    double min_distance)
{
  return collision_free && distance_to_robot >= min_distance;
}

bool isDuplicateLocalPlannerCandidate(const Eigen::Vector2d& pos,
    double yaw,
    const std::vector<std::pair<Eigen::Vector2d, double>>& candidates,
    double position_tolerance,
    double yaw_tolerance)
{
  for (const auto& candidate : candidates) {
    if ((candidate.first - pos).norm() >= position_tolerance) {
      continue;
    }

    const double yaw_error = std::atan2(
        std::sin(candidate.second - yaw), std::cos(candidate.second - yaw));
    if (std::fabs(yaw_error) < yaw_tolerance) {
      return true;
    }
  }
  return false;
}

bool appendLocalPlannerCandidate(const Eigen::Vector2d& pos,
    const std::vector<double>& yaw_candidates,
    const Eigen::Vector2d& current_pos,
    double min_distance,
    double position_tolerance,
    double yaw_tolerance,
    const std::function<bool(const Eigen::Vector2d&, double)>& is_collision,
    std::vector<std::pair<Eigen::Vector2d, double>>& candidates,
    LocalPlannerCandidateDebugStats* stats)
{
  const double distance_to_robot = (pos - current_pos).norm();
  bool saw_collision = false;
  bool saw_duplicate = false;

  for (double yaw : yaw_candidates) {
    const bool collision_free = !is_collision(pos, yaw);
    if (!collision_free) {
      saw_collision = true;
      continue;
    }

    if (!shouldQueueLocalPlannerCandidate(collision_free, distance_to_robot, min_distance)) {
      if (stats != nullptr) {
        stats->too_close_rejections += 1;
      }
      return false;
    }

    if (isDuplicateLocalPlannerCandidate(
            pos, yaw, candidates, position_tolerance, yaw_tolerance)) {
      saw_duplicate = true;
      continue;
    }

    candidates.emplace_back(pos, yaw);
    if (stats != nullptr) {
      stats->accepted_candidates += 1;
      if (saw_collision) {
        stats->collision_rejections += 1;
      }
      if (saw_duplicate) {
        stats->duplicate_rejections += 1;
      }
    }
    return true;
  }

  if (stats != nullptr) {
    if (distance_to_robot < min_distance) {
      stats->too_close_rejections += 1;
    } else if (saw_collision) {
      stats->collision_rejections += 1;
    } else if (saw_duplicate) {
      stats->duplicate_rejections += 1;
    }
  }

  return false;
}

Eigen::Vector2d selectFallbackCandidateAnchor(const Eigen::Vector2d& robot_pos,
    const Eigen::Vector2d& local_target_pos,
    const Eigen::Vector2d& global_goal_pos,
    double min_candidate_distance,
    double collapse_tolerance)
{
  const double local_target_dist = (local_target_pos - robot_pos).norm();
  const double global_goal_dist = (global_goal_pos - robot_pos).norm();
  const double goal_gap = (global_goal_pos - local_target_pos).norm();

  if (local_target_dist < min_candidate_distance &&
      global_goal_dist >= min_candidate_distance &&
      goal_gap > 1e-6) {
    return global_goal_pos;
  }

  if (local_target_dist < min_candidate_distance &&
      global_goal_dist > local_target_dist + collapse_tolerance &&
      goal_gap > collapse_tolerance) {
    return global_goal_pos;
  }

  return local_target_pos;
}

Eigen::Vector2d selectTrajectoryGoalForMode(
    int final_result,
    const Eigen::Vector2d& global_goal_pos,
    const Eigen::Vector2d& local_target_pos)
{
  return local_target_pos;
}

bool shouldDormantExplorationGoalBeforeTrajectory(
    int final_result,
    const Eigen::Vector2d& robot_pos,
    const Eigen::Vector2d& global_goal_pos,
    double min_exploration_goal_distance)
{
  return final_result == FINAL_RESULT::EXPLORE &&
      (global_goal_pos - robot_pos).norm() < min_exploration_goal_distance;
}

std::vector<PathPoseCandidate> sampleBackwardPathCandidates(
    const std::vector<Eigen::Vector2d>& path,
    int nearest_idx,
    double sample_step)
{
  std::vector<PathPoseCandidate> candidates;
  if (path.empty()) {
    return candidates;
  }

  const int start_idx = std::max(0, std::min(nearest_idx, static_cast<int>(path.size()) - 1));
  for (int i = start_idx; i >= 0; --i) {
    int next_idx = std::min(i + 1, static_cast<int>(path.size()) - 1);
    Eigen::Vector2d segment = path[next_idx] - path[i];
    if (segment.norm() <= 1e-6 && i > 0) {
      next_idx = i;
      segment = path[i] - path[i - 1];
    }
    const double yaw = segment.norm() > 1e-6 ? std::atan2(segment.y(), segment.x()) : 0.0;

    PathPoseCandidate discrete_candidate;
    discrete_candidate.pos = path[i];
    discrete_candidate.yaw = yaw;
    candidates.push_back(discrete_candidate);

    if (i == next_idx)
      continue;

    const double segment_len = segment.norm();
    if (segment_len <= sample_step)
      continue;

    const Eigen::Vector2d segment_dir = segment / segment_len;
    for (double traveled = sample_step; traveled < segment_len; traveled += sample_step) {
      PathPoseCandidate interpolated_candidate;
      interpolated_candidate.pos = path[next_idx] - segment_dir * traveled;
      interpolated_candidate.yaw = yaw;
      candidates.push_back(interpolated_candidate);
    }
  }

  return candidates;
}

bool findLastFootprintSafePathPose(const std::vector<Eigen::Vector2d>& path,
    const std::function<bool(const Eigen::Vector2d&, double)>& is_collision,
    PathPoseCandidate& safe_pose)
{
  if (path.empty()) {
    return false;
  }

  for (int i = static_cast<int>(path.size()) - 1; i >= 0; --i) {
    int next_idx = std::min(i + 1, static_cast<int>(path.size()) - 1);
    Eigen::Vector2d segment = path[next_idx] - path[i];
    if (segment.norm() <= 1e-6 && i > 0) {
      next_idx = i;
      segment = path[i] - path[i - 1];
    }

    const double yaw = segment.norm() > 1e-6 ? std::atan2(segment.y(), segment.x()) : 0.0;
    if (!is_collision(path[i], yaw)) {
      safe_pose.pos = path[i];
      safe_pose.yaw = yaw;
      return true;
    }
  }

  return false;
}

std::string formatLocalTargetPlanningDebugSummary(
    const LocalTargetPlanningDebugInfo& debug_info)
{
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(2);
  oss << "robot=(" << debug_info.robot_pos.x() << "," << debug_info.robot_pos.y() << ")";
  oss << " global_goal=(" << debug_info.global_goal_pos.x() << ","
      << debug_info.global_goal_pos.y() << ")";
  oss << " path_back=(" << debug_info.path_back_pos.x() << ","
      << debug_info.path_back_pos.y() << ")";
  oss << " local_target=(" << debug_info.local_target_pos.x() << ","
      << debug_info.local_target_pos.y() << ")";
  oss << " local_target_valid=" << (debug_info.local_target_valid ? "true" : "false");
  oss << " path_size=" << debug_info.path_size;
  if (std::isfinite(debug_info.nearest_candidate_distance)) {
    oss << " nearest_candidate_dist=" << debug_info.nearest_candidate_distance;
  } else {
    oss << " nearest_candidate_dist=inf";
  }
  return oss.str();
}

bool shouldAcceptRefinedFrontierGoal(const Eigen::Vector2d& start_pos,
    const Eigen::Vector2d& refined_pos,
    double min_progress_distance)
{
  return (refined_pos - start_pos).norm() >= min_progress_distance;
}

std::pair<double, double> computeManualFallbackCommand(double dist, double yaw_error)
{
  const double abs_yaw_error = std::fabs(yaw_error);

  if (abs_yaw_error > 0.25) {
    const double wz = std::max(-0.5, std::min(0.5, 1.5 * yaw_error));
    return { 0.0, wz };
  }

  // Go2 often needs a stronger minimum forward command than the previous
  // 0.12 m/s to overcome stance dithering at startup.
  const double desired_vx = 0.45 * dist;
  double vx = std::max(0.0, std::min(0.24, desired_vx));
  if (dist > 0.30) {
    vx = std::max(vx, 0.22);
  }
  const double wz = std::max(-0.3, std::min(0.3, 1.0 * yaw_error));
  return { vx, wz };
}

bool shouldAbortTrajectoryForTrackingError(
    const Eigen::Vector2d& planned_pos,
    const Eigen::Vector2d& odom_pos,
    double abort_distance)
{
  return (planned_pos - odom_pos).norm() > abort_distance;
}

bool shouldReplanForFrontierChange(
    double trajectory_elapsed_time,
    double replan_frontier_change_delay,
    int final_result,
    bool frontier_changed)
{
  return trajectory_elapsed_time > replan_frontier_change_delay &&
      trajectory_elapsed_time < replan_frontier_change_delay + 1.0 &&
      final_result == FINAL_RESULT::EXPLORE &&
      frontier_changed;
}

bool shouldKeepPreviousExplorationGoal(
    int previous_final_result,
    int new_final_result,
    bool frontier_changed,
    bool previous_path_available)
{
  return previous_final_result == FINAL_RESULT::EXPLORE &&
      new_final_result == FINAL_RESULT::EXPLORE &&
      !frontier_changed &&
      previous_path_available;
}

bool shouldReuseGoalAfterTrackingAbort(
    int final_result,
    bool finish_goal_valid,
    const Eigen::Vector2d& current_goal,
    const Eigen::Vector2d& finish_goal,
    double goal_reuse_distance_threshold)
{
  return final_result == FINAL_RESULT::SEARCH_OBJECT &&
      finish_goal_valid &&
      (current_goal - finish_goal).norm() <= goal_reuse_distance_threshold;
}

bool shouldStopTrajectoryServerForManualFallback(
    bool manual_goal_active,
    const Eigen::Vector2d& previous_goal,
    const Eigen::Vector2d& new_goal,
    double goal_change_threshold)
{
  if (!manual_goal_active) {
    return true;
  }
  return (previous_goal - new_goal).norm() > goal_change_threshold;
}

bool shouldForceDormantAfterExplorationFallback(
    bool replan_after_finish,
    double dist,
    double reach_distance)
{
  return replan_after_finish && dist < reach_distance;
}

bool shouldUseExplorationManualFallback(double target_distance, double max_direct_distance)
{
  return target_distance <= max_direct_distance;
}

bool shouldWaitForLightGlueAfterPlanFailure(int final_result)
{
  return final_result == FINAL_RESULT::SEARCH_OBJECT;
}

bool shouldSkipObjectNavigationAfterPlanFailure(
    int final_result, const std::string& stop_verification_status)
{
  return final_result == FINAL_RESULT::SEARCH_OBJECT &&
      stop_verification_status == "PENDING";
}

bool shouldUseSearchObjectManualFallback(
    int final_result, const std::string& stop_verification_status)
{
  return final_result == FINAL_RESULT::SEARCH_OBJECT &&
      (stop_verification_status == "VERIFIED" ||
          stop_verification_status == "PENDING_FAR" ||
          stop_verification_status == "NO_GOAL_IMAGE");
}

bool shouldCompleteObjectApproach(
    int final_result, const std::string& stop_verification_status)
{
  return final_result == FINAL_RESULT::SEARCH_OBJECT &&
      (stop_verification_status == "VERIFIED" ||
          stop_verification_status == "NO_GOAL_IMAGE");
}

bool shouldContinueApproachingObject(
    int final_result, const std::string& stop_verification_status)
{
  return final_result == FINAL_RESULT::SEARCH_OBJECT &&
      stop_verification_status == "PENDING_FAR";
}

bool shouldRetryObjectApproachAfterMapPointReached(
    int final_result, const std::string& stop_verification_status)
{
  return final_result == FINAL_RESULT::SEARCH_OBJECT &&
      (stop_verification_status == "PENDING" ||
          stop_verification_status == "PENDING_FAR");
}

bool shouldStopMotionForLightGlueStatus(const std::string& stop_verification_status)
{
  return stop_verification_status == "VERIFIED" ||
      stop_verification_status == "PENDING_CLOSE";
}

bool shouldHoldForLightGlueConfirmation(
    const std::string& stop_verification_status,
    bool close_stop_candidate_active)
{
  return stop_verification_status == "PENDING_CLOSE" ||
      (stop_verification_status == "PENDING" && close_stop_candidate_active);
}

bool shouldResumePlanningFromPendingLightGlue(
    int final_result, const std::string& stop_verification_status)
{
  return stop_verification_status == "PENDING" &&
      final_result != FINAL_RESULT::NO_FRONTIER;
}

bool shouldPreemptExplorationFallbackForLightGlueStatus(
    const std::string& stop_verification_status,
    bool manual_goal_active,
    bool manual_goal_replan_after_finish)
{
  return manual_goal_active &&
      manual_goal_replan_after_finish &&
      stop_verification_status == "PENDING_FAR";
}

bool shouldFinalizeVerifiedStop(
    const std::string& stop_verification_status,
    double verified_hold_time,
    double required_hold_time)
{
  return stop_verification_status == "VERIFIED" &&
      verified_hold_time >= required_hold_time;
}

bool shouldRevokeFinishedStop(
    const std::string& stop_verification_status,
    double time_since_finish,
    double revoke_window)
{
  return stop_verification_status == "PENDING_FAR" &&
      time_since_finish >= 0.0 &&
      time_since_finish <= revoke_window;
}

bool shouldReportFinishExploration(bool have_finished)
{
  return have_finished;
}

bool shouldVisualizeObjectMarker(bool visualize_object_markers, int label)
{
  return visualize_object_markers && label >= 0;
}

}  // namespace apexnav_planner
