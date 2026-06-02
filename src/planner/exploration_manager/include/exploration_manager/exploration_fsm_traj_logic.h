#ifndef _EXPLORATION_FSM_TRAJ_LOGIC_H_
#define _EXPLORATION_FSM_TRAJ_LOGIC_H_

#include <Eigen/Core>
#include <functional>
#include <limits>
#include <vector>
#include <string>
#include <utility>

namespace apexnav_planner {

struct LocalPlannerCandidateDebugStats {
  int collision_rejections = 0;
  int too_close_rejections = 0;
  int duplicate_rejections = 0;
  int accepted_candidates = 0;
};

struct LocalTargetPlanningDebugInfo {
  Eigen::Vector2d robot_pos = Eigen::Vector2d::Zero();
  Eigen::Vector2d global_goal_pos = Eigen::Vector2d::Zero();
  Eigen::Vector2d path_back_pos = Eigen::Vector2d::Zero();
  Eigen::Vector2d local_target_pos = Eigen::Vector2d::Zero();
  bool local_target_valid = false;
  int path_size = 0;
  double nearest_candidate_distance = std::numeric_limits<double>::infinity();
};

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

bool shouldTreatOccupancyAsKinoCollision(int occupancy, bool unknown_as_collision);

double computeFootprintSupportRadius(double length,
    double width,
    double odom_to_center_x,
    double odom_to_center_y);

double computeLocalTargetClearance(
    double width,
    double safety_margin,
    double obstacles_inflation,
    double minimum_clearance);

double computeWallProximityPenalty(
    double obstacle_distance,
    double preferred_clearance,
    double penalty_weight);

bool shouldQueueLocalPlannerCandidate(
    bool collision_free,
    double distance_to_robot,
    double min_distance);

bool isDuplicateLocalPlannerCandidate(const Eigen::Vector2d& pos,
    double yaw,
    const std::vector<std::pair<Eigen::Vector2d, double>>& candidates,
    double position_tolerance,
    double yaw_tolerance);

bool appendLocalPlannerCandidate(const Eigen::Vector2d& pos,
    const std::vector<double>& yaw_candidates,
    const Eigen::Vector2d& current_pos,
    double min_distance,
    double position_tolerance,
    double yaw_tolerance,
    const std::function<bool(const Eigen::Vector2d&, double)>& is_collision,
    std::vector<std::pair<Eigen::Vector2d, double>>& candidates,
    LocalPlannerCandidateDebugStats* stats = nullptr);

Eigen::Vector2d selectFallbackCandidateAnchor(const Eigen::Vector2d& robot_pos,
    const Eigen::Vector2d& local_target_pos,
    const Eigen::Vector2d& global_goal_pos,
    double min_candidate_distance,
    double collapse_tolerance);

Eigen::Vector2d selectTrajectoryGoalForMode(
    int final_result,
    const Eigen::Vector2d& global_goal_pos,
    const Eigen::Vector2d& local_target_pos);

bool shouldDormantExplorationGoalBeforeTrajectory(
    int final_result,
    const Eigen::Vector2d& robot_pos,
    const Eigen::Vector2d& global_goal_pos,
    double min_exploration_goal_distance);

std::vector<PathPoseCandidate> sampleBackwardPathCandidates(
    const std::vector<Eigen::Vector2d>& path,
    int nearest_idx,
    double sample_step);

bool findLastFootprintSafePathPose(const std::vector<Eigen::Vector2d>& path,
    const std::function<bool(const Eigen::Vector2d&, double)>& is_collision,
    PathPoseCandidate& safe_pose);

std::string formatLocalTargetPlanningDebugSummary(
    const LocalTargetPlanningDebugInfo& debug_info);

bool shouldAcceptRefinedFrontierGoal(const Eigen::Vector2d& start_pos,
    const Eigen::Vector2d& refined_pos,
    double min_progress_distance);

std::pair<double, double> computeManualFallbackCommand(double dist, double yaw_error);

bool shouldAbortTrajectoryForTrackingError(
    const Eigen::Vector2d& planned_pos,
    const Eigen::Vector2d& odom_pos,
    double abort_distance);

bool shouldReplanForFrontierChange(
    double trajectory_elapsed_time,
    double replan_frontier_change_delay,
    int final_result,
    bool frontier_changed);

bool shouldKeepPreviousExplorationGoal(
    int previous_final_result,
    int new_final_result,
    bool frontier_changed,
    bool previous_path_available);

bool shouldReuseGoalAfterTrackingAbort(
    int final_result,
    bool finish_goal_valid,
    const Eigen::Vector2d& current_goal,
    const Eigen::Vector2d& finish_goal,
    double goal_reuse_distance_threshold);

bool shouldStopTrajectoryServerForManualFallback(
    bool manual_goal_active,
    const Eigen::Vector2d& previous_goal,
    const Eigen::Vector2d& new_goal,
    double goal_change_threshold);

bool shouldForceDormantAfterExplorationFallback(
    bool replan_after_finish,
    double dist,
    double reach_distance);

bool shouldUseExplorationManualFallback(
    double target_distance,
    double max_direct_distance);

bool shouldWaitForLightGlueAfterPlanFailure(int final_result);

bool shouldSkipObjectNavigationAfterPlanFailure(
    int final_result, const std::string& stop_verification_status);

bool shouldUseSearchObjectManualFallback(
    int final_result, const std::string& stop_verification_status);

bool shouldCompleteObjectApproach(
    int final_result, const std::string& stop_verification_status);

bool shouldContinueApproachingObject(
    int final_result, const std::string& stop_verification_status);

bool shouldRetryObjectApproachAfterMapPointReached(
    int final_result, const std::string& stop_verification_status);

bool shouldStopMotionForLightGlueStatus(const std::string& stop_verification_status);

bool shouldHoldForLightGlueConfirmation(
    const std::string& stop_verification_status,
    bool close_stop_candidate_active);

bool shouldResumePlanningFromPendingLightGlue(
    int final_result, const std::string& stop_verification_status);

bool shouldPreemptExplorationFallbackForLightGlueStatus(
    const std::string& stop_verification_status,
    bool manual_goal_active,
    bool manual_goal_replan_after_finish);

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
