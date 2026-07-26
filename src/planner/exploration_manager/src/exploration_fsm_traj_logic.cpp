#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>

#include <exploration_manager/exploration_data.h>
#include <exploration_manager/exploration_fsm_traj_logic.h>

namespace apexnav_planner {
namespace {

double pathYawAtIndex(const std::vector<Eigen::Vector2d>& path, int idx)
{
  if (path.size() < 2 || idx < 0 || idx >= static_cast<int>(path.size())) {
    return 0.0;
  }

  int next_idx = std::min(idx + 1, static_cast<int>(path.size()) - 1);
  Eigen::Vector2d segment = path[next_idx] - path[idx];
  if (segment.norm() <= 1e-6 && idx > 0) {
    segment = path[idx] - path[idx - 1];
  }

  return segment.norm() > 1e-6 ? std::atan2(segment.y(), segment.x()) : 0.0;
}

int selectLookaheadPathIndex(const Eigen::Vector2d& current_pos,
    const std::vector<Eigen::Vector2d>& path,
    double local_distance,
    double min_target_distance)
{
  if (path.empty()) {
    return -1;
  }
  int start_path_id = 0;
  double min_dist = std::numeric_limits<double>::max();
  for (int i = 0; i < static_cast<int>(path.size()) - 1; ++i) {
    const double dist = (path[i] - current_pos).norm();
    if (dist < min_dist) {
      min_dist = dist;
      start_path_id = i + 1;
    }
  }
  start_path_id = std::min(start_path_id, static_cast<int>(path.size()) - 1);

  int selected_idx = -1;
  double len = (path[start_path_id] - current_pos).norm();
  if (len <= local_distance && len > min_target_distance) {
    selected_idx = start_path_id;
  }

  for (int i = start_path_id + 1; i < static_cast<int>(path.size()); ++i) {
    len += (path[i] - path[i - 1]).norm();
    if (len > local_distance) {
      break;
    }
    if ((current_pos - path[i]).norm() > min_target_distance) {
      selected_idx = i;
    }
  }

  return selected_idx;
}

}  // namespace

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

std::vector<double> sampleInitialKinoAstarArcInputs(
    double grid_interval, double step_arc, bool start_nearly_static)
{
  const double base_arc = start_nearly_static ?
      std::max(grid_interval, 0.5 * step_arc) :
      std::max(grid_interval, 1e-6);

  std::vector<double> arcs;
  for (double arc = base_arc; arc <= 2.0 * base_arc + 1e-6; arc += base_arc) {
    arcs.push_back(arc);
  }

  return arcs;
}

bool shouldTreatOccupancyAsKinoCollision(int occupancy)
{
  return occupancy == SDFMap2D::OCCUPIED ||
      occupancy == SDFMap2D::UNKNOWN;
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

Eigen::Vector2d selectTrajectoryGoalForMode(
    int final_result,
    const Eigen::Vector2d& global_goal_pos,
    const Eigen::Vector2d& local_target_pos)
{
  (void)final_result;
  (void)global_goal_pos;
  return local_target_pos;
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

bool selectFootprintSafeLocalTargetFromPath(const Eigen::Vector2d& current_pos,
    const std::vector<Eigen::Vector2d>& path,
    double local_distance,
    double min_target_distance,
    const std::function<bool(const Eigen::Vector2d&, double)>& is_collision,
    PathPoseCandidate& safe_pose)
{
  const int selected_idx =
      selectLookaheadPathIndex(current_pos, path, local_distance, min_target_distance);
  if (selected_idx < 0) {
    return false;
  }

  for (int i = selected_idx; i >= 0; --i) {
    if ((path[i] - current_pos).norm() < min_target_distance) {
      continue;
    }

    const double yaw = pathYawAtIndex(path, i);
    if (!is_collision(path[i], yaw)) {
      safe_pose.pos = path[i];
      safe_pose.yaw = yaw;
      return true;
    }
  }

  return false;
}

bool isLocalTargetFacingForward(
    const Eigen::Vector2d& current_pos,
    double current_yaw,
    const Eigen::Vector2d& target_pos,
    double max_yaw_error)
{
  const Eigen::Vector2d delta = target_pos - current_pos;
  if (delta.norm() <= 1.0e-6) {
    return true;
  }

  const double target_yaw = std::atan2(delta.y(), delta.x());
  const double yaw_error =
      std::atan2(std::sin(target_yaw - current_yaw), std::cos(target_yaw - current_yaw));
  const double clamped_limit =
      std::max(0.0, std::min(max_yaw_error, 3.14159265358979323846));
  return std::fabs(yaw_error) <= clamped_limit + 1.0e-6;
}

bool selectFootprintSafeForwardLocalTargetFromPath(const Eigen::Vector2d& current_pos,
    double current_yaw,
    const std::vector<Eigen::Vector2d>& path,
    double local_distance,
    double min_target_distance,
    double max_yaw_error,
    const std::function<bool(const Eigen::Vector2d&, double)>& is_collision,
    PathPoseCandidate& safe_pose)
{
  const int selected_idx =
      selectLookaheadPathIndex(current_pos, path, local_distance, min_target_distance);
  if (selected_idx < 0) {
    return false;
  }

  for (int i = selected_idx; i >= 0; --i) {
    if ((path[i] - current_pos).norm() < min_target_distance) {
      continue;
    }
    if (!isLocalTargetFacingForward(current_pos, current_yaw, path[i], max_yaw_error)) {
      continue;
    }

    const double yaw = pathYawAtIndex(path, i);
    if (!is_collision(path[i], yaw)) {
      safe_pose.pos = path[i];
      safe_pose.yaw = yaw;
      return true;
    }
  }

  return false;
}

Eigen::Vector2d clampLocalTargetToLookahead(
    const Eigen::Vector2d& current_pos,
    const Eigen::Vector2d& target_pos,
    double local_distance)
{
  if (local_distance <= 1.0e-6) {
    return current_pos;
  }

  const Eigen::Vector2d delta = target_pos - current_pos;
  const double distance = delta.norm();
  if (distance <= local_distance || distance <= 1.0e-6) {
    return target_pos;
  }

  return current_pos + delta.normalized() * local_distance;
}

bool shouldAcceptRefinedFrontierGoal(const Eigen::Vector2d& start_pos,
    const Eigen::Vector2d& refined_pos,
    double min_progress_distance)
{
  return (refined_pos - start_pos).norm() >= min_progress_distance;
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
    bool frontier_changed,
    bool replan_on_frontier_change)
{
  return replan_on_frontier_change &&
      trajectory_elapsed_time > replan_frontier_change_delay &&
      final_result == FINAL_RESULT::EXPLORE &&
      frontier_changed;
}

bool shouldUseOdometryStartForReplan(
    bool static_state,
    double tracking_error,
    double max_tracking_error)
{
  return static_state || tracking_error > max_tracking_error;
}

bool shouldReplanForTrajectoryTimeout(
    double trajectory_elapsed_time,
    double replan_timeout,
    double time_since_last_progress,
    bool progress_tracking_active)
{
  if (trajectory_elapsed_time <= replan_timeout) {
    return false;
  }

  if (!progress_tracking_active) {
    return true;
  }

  return time_since_last_progress > replan_timeout;
}

bool shouldStopActiveTrajectoryBeforeReplan(
    bool replan_due_to_timeout_without_progress)
{
  return replan_due_to_timeout_without_progress;
}

bool shouldKeepPreviousExplorationGoal(
    int previous_final_result,
    int new_final_result,
    bool frontier_changed,
    bool previous_path_available,
    bool forced_dormant_frontier)
{
  return previous_final_result == FINAL_RESULT::EXPLORE &&
      new_final_result == FINAL_RESULT::EXPLORE &&
      !frontier_changed &&
      !forced_dormant_frontier &&
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

visualization_msgs::Marker makeFootprintCollisionPointMarker(
    const Eigen::Vector2d& collision_pos,
    double collision_time,
    const std::string& frame_id)
{
  visualization_msgs::Marker marker;
  marker.header.frame_id = frame_id;
  marker.header.stamp = ros::Time(0);
  marker.ns = "footprint_collision";
  marker.id = 0;
  marker.type = visualization_msgs::Marker::SPHERE;
  marker.action = visualization_msgs::Marker::ADD;
  marker.pose.position.x = collision_pos.x();
  marker.pose.position.y = collision_pos.y();
  marker.pose.position.z = 0.08;
  marker.pose.orientation.w = 1.0;
  marker.scale.x = 0.18;
  marker.scale.y = 0.18;
  marker.scale.z = 0.18;
  marker.color.r = 1.0;
  marker.color.g = 0.05;
  marker.color.b = 0.05;
  marker.color.a = 1.0;
  marker.lifetime = ros::Duration(0.0);

  std::ostringstream text;
  text << "footprint collision t=" << std::fixed << std::setprecision(2)
       << collision_time << "s";
  marker.text = text.str();

  return marker;
}

bool shouldWaitForLightGlueAfterPlanFailure(int final_result)
{
  return final_result == FINAL_RESULT::SEARCH_OBJECT;
}

bool shouldSkipObjectNavigationAfterPlanFailure(
    int final_result,
    const std::string& stop_verification_status)
{
  return final_result == FINAL_RESULT::SEARCH_OBJECT &&
      stop_verification_status == "PENDING";
}

bool shouldHoldObjectNavigationSuppressed(const std::string& stop_verification_status)
{
  return stop_verification_status == "PENDING";
}

bool shouldCompleteObjectApproach(
    int final_result,
    const std::string& stop_verification_status)
{
  return final_result == FINAL_RESULT::SEARCH_OBJECT &&
      (stop_verification_status == "VERIFIED" ||
          stop_verification_status == "NO_GOAL_IMAGE");
}

bool shouldRetryObjectApproachAfterMapPointReached(
    int final_result,
    const std::string& stop_verification_status)
{
  return final_result == FINAL_RESULT::SEARCH_OBJECT &&
      (stop_verification_status == "PENDING" ||
          stop_verification_status == "PENDING_FAR");
}

bool shouldResumePlanningFromPendingLightGlue(
    int final_result,
    const std::string& stop_verification_status)
{
  return stop_verification_status == "PENDING" &&
      final_result != FINAL_RESULT::NO_FRONTIER;
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
