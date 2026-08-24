#ifndef _EXPLORATION_FSM_TRAJ_LOGIC_H_
#define _EXPLORATION_FSM_TRAJ_LOGIC_H_

#include <Eigen/Core>
#include <functional>
#include <string>
#include <vector>
#include <visualization_msgs/Marker.h>

namespace apexnav_planner {

struct PathPoseCandidate {
  Eigen::Vector2d pos = Eigen::Vector2d::Zero();
  double yaw = 0.0;
};

bool shouldCompleteSearchObjectMission(int final_result,
    const Eigen::Vector2d& current_pos,
    const Eigen::Vector2d& object_goal_pos,
    const Eigen::Vector2d& local_goal_pos,
    double reach_distance);

bool shouldUseKinoAstarPath(int path_node_count, bool has_path);

std::vector<double> sampleKinoAstarArcInputs(double step_arc, bool include_reverse);

std::vector<double> sampleInitialKinoAstarArcInputs(
    double grid_interval, double step_arc, bool start_nearly_static);

bool shouldTreatOccupancyAsKinoCollision(int occupancy);

double computeWallProximityPenalty(
    double obstacle_distance,
    double preferred_clearance,
    double penalty_weight);

Eigen::Vector2d selectTrajectoryGoalForMode(
    int final_result,
    const Eigen::Vector2d& global_goal_pos,
    const Eigen::Vector2d& local_target_pos);

bool findLastFootprintSafePathPose(const std::vector<Eigen::Vector2d>& path,
    const std::function<bool(const Eigen::Vector2d&, double)>& is_collision,
    PathPoseCandidate& safe_pose);

bool selectFootprintSafeLocalTargetFromPath(const Eigen::Vector2d& current_pos,
    const std::vector<Eigen::Vector2d>& path,
    double local_distance,
    double min_target_distance,
    const std::function<bool(const Eigen::Vector2d&, double)>& is_collision,
    PathPoseCandidate& safe_pose);

bool selectCornerAwareFootprintSafeLocalTargetFromPath(
    const Eigen::Vector2d& current_pos,
    const std::vector<Eigen::Vector2d>& path,
    double local_distance,
    double min_target_distance,
    double max_path_heading_change,
    const std::function<bool(const Eigen::Vector2d&, double)>& is_collision,
    PathPoseCandidate& safe_pose,
    bool& corner_limited);

bool isLocalTargetFacingForward(
    const Eigen::Vector2d& current_pos,
    double current_yaw,
    const Eigen::Vector2d& target_pos,
    double max_yaw_error);

double computeTargetYawError(
    const Eigen::Vector2d& current_pos,
    double current_yaw,
    const Eigen::Vector2d& target_pos);

bool shouldRotateBeforeTranslation(
    double yaw_error,
    double rotation_yaw_error_threshold);

bool selectFootprintSafeForwardLocalTargetFromPath(const Eigen::Vector2d& current_pos,
    double current_yaw,
    const std::vector<Eigen::Vector2d>& path,
    double local_distance,
    double min_target_distance,
    double max_yaw_error,
    const std::function<bool(const Eigen::Vector2d&, double)>& is_collision,
    PathPoseCandidate& safe_pose);

bool isInPlaceRotationFootprintSafe(const Eigen::Vector2d& current_pos,
    double start_yaw,
    double target_yaw,
    double max_yaw_step,
    const std::function<bool(const Eigen::Vector2d&, double)>& is_collision);

Eigen::Vector2d clampLocalTargetToLookahead(
    const Eigen::Vector2d& current_pos,
    const Eigen::Vector2d& target_pos,
    double local_distance);

bool shouldAcceptRefinedFrontierGoal(const Eigen::Vector2d& start_pos,
    const Eigen::Vector2d& refined_pos,
    double min_progress_distance);

bool shouldAbortTrajectoryForTrackingError(
    const Eigen::Vector2d& planned_pos,
    const Eigen::Vector2d& odom_pos,
    double abort_distance);

bool shouldReplanForFrontierChange(
    double trajectory_elapsed_time,
    double replan_frontier_change_delay,
    int final_result,
    bool frontier_changed,
    bool replan_on_frontier_change);

bool shouldUseOdometryStartForReplan(
    bool static_state,
    double tracking_error,
    double max_tracking_error);

bool shouldReplanForTrajectoryTimeout(
    double trajectory_elapsed_time,
    double replan_timeout,
    double time_since_last_progress,
    bool progress_tracking_active);

bool shouldStopActiveTrajectoryBeforeReplan(
    bool replan_due_to_timeout_without_progress);

bool shouldKeepPreviousExplorationGoal(
    int previous_final_result,
    int new_final_result,
    bool frontier_changed,
    bool previous_path_available,
    bool forced_dormant_frontier = false);

bool shouldReuseGoalAfterTrackingAbort(
    int final_result,
    bool finish_goal_valid,
    const Eigen::Vector2d& current_goal,
    const Eigen::Vector2d& finish_goal,
    double goal_reuse_distance_threshold);

visualization_msgs::Marker makeFootprintCollisionPointMarker(
    const Eigen::Vector2d& collision_pos,
    double collision_time,
    const std::string& frame_id);

bool shouldWaitForLightGlueAfterPlanFailure(int final_result);

bool shouldSkipObjectNavigationAfterPlanFailure(
    int final_result,
    const std::string& stop_verification_status);

bool shouldHoldObjectNavigationSuppressed(const std::string& stop_verification_status);

bool shouldCompleteObjectApproach(
    int final_result,
    const std::string& stop_verification_status);

bool shouldRetryObjectApproachAfterMapPointReached(
    int final_result,
    const std::string& stop_verification_status);

bool shouldResumePlanningFromPendingLightGlue(
    int final_result,
    const std::string& stop_verification_status);

bool shouldStopMotionForLightGlueStatus(const std::string& stop_verification_status);

bool shouldHoldForLightGlueConfirmation(
    const std::string& stop_verification_status,
    bool close_stop_candidate_active);

bool shouldFinalizeVerifiedStop(
    const std::string& stop_verification_status,
    double verified_hold_time,
    double required_hold_time);

bool shouldRevokeFinishedStop(
    const std::string& stop_verification_status,
    double time_since_finish,
    double revoke_window);

bool shouldReportFinishExploration(bool have_finished);

bool shouldVisualizeObjectMarker(bool visualize_object_markers, int label);

}  // namespace apexnav_planner

#endif
