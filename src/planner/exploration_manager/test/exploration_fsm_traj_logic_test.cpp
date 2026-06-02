#include <gtest/gtest.h>

#include <Eigen/Core>

#include <exploration_manager/exploration_data.h>
#include <exploration_manager/exploration_fsm_traj_logic.h>
#include <exploration_manager/exploration_fsm_traj.h>
#include <path_searching/kino_astar.h>

namespace {

TEST(ExplorationFSMRealLogicTest, DoesNotFinishWhenOnlyLocalGoalIsNear)
{
  const Eigen::Vector2d current_pos(0.0, 0.0);
  const Eigen::Vector2d object_goal_pos(2.7, 0.0);
  const Eigen::Vector2d local_goal_pos(0.1, 0.0);

  EXPECT_FALSE(apexnav_planner::shouldCompleteSearchObjectMission(
      apexnav_planner::FINAL_RESULT::SEARCH_OBJECT,
      current_pos,
      object_goal_pos,
      local_goal_pos,
      0.25));
}

TEST(ExplorationFSMRealLogicTest, FinishesWhenGlobalObjectGoalIsNear)
{
  const Eigen::Vector2d current_pos(0.0, 0.0);
  const Eigen::Vector2d object_goal_pos(0.1, 0.0);
  const Eigen::Vector2d local_goal_pos(1.5, 0.0);

  EXPECT_TRUE(apexnav_planner::shouldCompleteSearchObjectMission(
      apexnav_planner::FINAL_RESULT::SEARCH_OBJECT,
      current_pos,
      object_goal_pos,
      local_goal_pos,
      0.25));
}

TEST(ExplorationFSMRealLogicTest, IgnoresNonSearchObjectResults)
{
  const Eigen::Vector2d current_pos(0.0, 0.0);
  const Eigen::Vector2d object_goal_pos(0.0, 0.0);
  const Eigen::Vector2d local_goal_pos(0.0, 0.0);

  EXPECT_FALSE(apexnav_planner::shouldCompleteSearchObjectMission(
      apexnav_planner::FINAL_RESULT::EXPLORE,
      current_pos,
      object_goal_pos,
      local_goal_pos,
      0.25));
}

TEST(ExplorationFSMRealLogicTest, UsesKinoAstarPrefixPathForRollingReplan)
{
  EXPECT_TRUE(apexnav_planner::shouldUseKinoAstarPath(2, true));
  EXPECT_TRUE(apexnav_planner::shouldUseKinoAstarPath(1, true));
  EXPECT_FALSE(apexnav_planner::shouldUseKinoAstarPath(3, false));
  EXPECT_FALSE(apexnav_planner::shouldUseKinoAstarPath(3, true));
  EXPECT_FALSE(apexnav_planner::shouldUseKinoAstarPath(2, false));
}

TEST(ExplorationFSMRealLogicTest, KinoAstarRollingArcSamplesIncludeReverseAndSkipZero)
{
  const auto arcs = apexnav_planner::sampleKinoAstarArcInputs(0.2, true);

  ASSERT_EQ(arcs.size(), 4u);
  EXPECT_NEAR(arcs[0], -0.2, 1e-6);
  EXPECT_NEAR(arcs[1], -0.1, 1e-6);
  EXPECT_NEAR(arcs[2], 0.1, 1e-6);
  EXPECT_NEAR(arcs[3], 0.2, 1e-6);
  for (const double arc : arcs) {
    EXPECT_GT(std::fabs(arc), 1e-6);
  }
}

TEST(ExplorationFSMRealLogicTest, KinoAstarCollisionPolicyCanAllowUnknownCells)
{
  EXPECT_TRUE(apexnav_planner::shouldTreatOccupancyAsKinoCollision(
      apexnav_planner::SDFMap2D::OCCUPIED, false));
  EXPECT_FALSE(apexnav_planner::shouldTreatOccupancyAsKinoCollision(
      apexnav_planner::SDFMap2D::UNKNOWN, false));
  EXPECT_TRUE(apexnav_planner::shouldTreatOccupancyAsKinoCollision(
      apexnav_planner::SDFMap2D::UNKNOWN, true));
  EXPECT_FALSE(apexnav_planner::shouldTreatOccupancyAsKinoCollision(
      apexnav_planner::SDFMap2D::FREE, true));
}

TEST(ExplorationFSMRealLogicTest, ComputesSupportRadiusFromFootprintAndOffset)
{
  const double radius =
      apexnav_planner::computeFootprintSupportRadius(0.50, 0.15, -0.10, 0.0);

  EXPECT_NEAR(radius, 0.3579455, 1e-6);
  EXPECT_GT(radius, 0.26);
}

TEST(ExplorationFSMRealLogicTest, LocalTargetClearanceAvoidsDoubleCountingInflation)
{
  const double footprint_radius =
      apexnav_planner::computeFootprintSupportRadius(0.75, 0.35, -0.10, 0.0);
  const double local_clearance =
      apexnav_planner::computeLocalTargetClearance(0.35, 0.005, 0.08, 0.06);

  EXPECT_GT(footprint_radius, 0.50);
  EXPECT_NEAR(local_clearance, 0.10, 1e-6);
  EXPECT_LT(local_clearance, 0.13);
  EXPECT_LT(local_clearance, footprint_radius);
}

TEST(ExplorationFSMRealLogicTest, LocalTargetClearanceKeepsMinimumBuffer)
{
  const double local_clearance =
      apexnav_planner::computeLocalTargetClearance(0.35, 0.005, 0.30, 0.06);

  EXPECT_NEAR(local_clearance, 0.06, 1e-6);
}

TEST(ExplorationFSMRealLogicTest, NarrowDoorTuningLeavesMoreFreeGapWhenInflationDrops)
{
  const double nominal_gap =
      0.77 - 2.0 * (0.08 + 0.08);
  const double tuned_gap =
      0.77 - 2.0 * (0.06 + 0.05);

  EXPECT_LT(nominal_gap, tuned_gap);
  EXPECT_GT(tuned_gap, 0.50);
}

TEST(ExplorationFSMRealLogicTest, WallProximityPenaltyPrefersDoorCenterline)
{
  const double center_penalty =
      apexnav_planner::computeWallProximityPenalty(0.38, 0.25, 3.0);
  const double near_wall_penalty =
      apexnav_planner::computeWallProximityPenalty(0.12, 0.25, 3.0);

  EXPECT_NEAR(center_penalty, 0.0, 1e-6);
  EXPECT_GT(near_wall_penalty, center_penalty);
}

TEST(ExplorationFSMRealLogicTest, RejectsCollidingLocalPlannerCandidate)
{
  EXPECT_FALSE(apexnav_planner::shouldQueueLocalPlannerCandidate(false, 1.20, 0.30));
}

TEST(ExplorationFSMRealLogicTest, RejectsTooCloseLocalPlannerCandidate)
{
  EXPECT_FALSE(apexnav_planner::shouldQueueLocalPlannerCandidate(true, 0.15, 0.30));
}

TEST(ExplorationFSMRealLogicTest, AcceptsCollisionFreeLocalPlannerCandidateBeyondMinimumDistance)
{
  EXPECT_TRUE(apexnav_planner::shouldQueueLocalPlannerCandidate(true, 0.45, 0.30));
}

TEST(ExplorationFSMRealLogicTest, AcceptsNearLocalPlannerCandidateWhenAlreadyAtDoorway)
{
  EXPECT_TRUE(apexnav_planner::shouldQueueLocalPlannerCandidate(true, 0.21, 0.20));
}

TEST(ExplorationFSMRealLogicTest, LocalPlannerCandidateTriesAlternateYawWhenPathYawCollides)
{
  const Eigen::Vector2d current_pos(0.0, 0.0);
  const Eigen::Vector2d target_pos(1.0, 0.0);
  std::vector<std::pair<Eigen::Vector2d, double>> candidates;
  apexnav_planner::LocalPlannerCandidateDebugStats stats;
  const std::vector<double> yaw_candidates{ 0.0, M_PI_2 };

  const bool appended = apexnav_planner::appendLocalPlannerCandidate(target_pos, yaw_candidates,
      current_pos, 0.30, 0.10, 0.20,
      [](const Eigen::Vector2d&, double yaw) { return std::fabs(yaw) < 1e-6; }, candidates,
      &stats);

  EXPECT_TRUE(appended);
  ASSERT_EQ(candidates.size(), 1u);
  EXPECT_NEAR(candidates.front().first.x(), 1.0, 1e-6);
  EXPECT_NEAR(candidates.front().first.y(), 0.0, 1e-6);
  EXPECT_NEAR(candidates.front().second, M_PI_2, 1e-6);
  EXPECT_EQ(stats.accepted_candidates, 1);
  EXPECT_EQ(stats.collision_rejections, 1);
  EXPECT_EQ(stats.too_close_rejections, 0);
  EXPECT_EQ(stats.duplicate_rejections, 0);
}

TEST(ExplorationFSMRealLogicTest, PlanningDiagnosticsReportKeyGoalPositionsAndDistances)
{
  apexnav_planner::LocalTargetPlanningDebugInfo debug_info;
  debug_info.robot_pos = Eigen::Vector2d(0.0, 0.0);
  debug_info.global_goal_pos = Eigen::Vector2d(1.2, 0.4);
  debug_info.path_back_pos = Eigen::Vector2d(0.28, 0.02);
  debug_info.local_target_pos = Eigen::Vector2d(0.24, 0.01);
  debug_info.local_target_valid = false;
  debug_info.path_size = 5;
  debug_info.nearest_candidate_distance = 0.24020824;

  const std::string summary =
      apexnav_planner::formatLocalTargetPlanningDebugSummary(debug_info);

  EXPECT_NE(summary.find("robot=(0.00,0.00)"), std::string::npos);
  EXPECT_NE(summary.find("global_goal=(1.20,0.40)"), std::string::npos);
  EXPECT_NE(summary.find("path_back=(0.28,0.02)"), std::string::npos);
  EXPECT_NE(summary.find("local_target=(0.24,0.01)"), std::string::npos);
  EXPECT_NE(summary.find("local_target_valid=false"), std::string::npos);
  EXPECT_NE(summary.find("path_size=5"), std::string::npos);
  EXPECT_NE(summary.find("nearest_candidate_dist=0.24"), std::string::npos);
}

TEST(ExplorationFSMRealLogicTest, FallbackCandidateAnchorPrefersGlobalGoalWhenLocalTargetCollapses)
{
  const Eigen::Vector2d robot_pos(0.0, 0.44);
  const Eigen::Vector2d local_target_pos(0.0, 0.44);
  const Eigen::Vector2d global_goal_pos(-0.50, -2.61);

  const Eigen::Vector2d anchor = apexnav_planner::selectFallbackCandidateAnchor(
      robot_pos, local_target_pos, global_goal_pos, 0.30, 0.60);

  EXPECT_NEAR(anchor.x(), global_goal_pos.x(), 1e-6);
  EXPECT_NEAR(anchor.y(), global_goal_pos.y(), 1e-6);
}

TEST(ExplorationFSMRealLogicTest, FallbackCandidateAnchorKeepsLocalTargetWhenItIsProgressed)
{
  const Eigen::Vector2d robot_pos(0.0, 0.44);
  const Eigen::Vector2d local_target_pos(-0.22, -1.10);
  const Eigen::Vector2d global_goal_pos(-0.50, -2.61);

  const Eigen::Vector2d anchor = apexnav_planner::selectFallbackCandidateAnchor(
      robot_pos, local_target_pos, global_goal_pos, 0.30, 0.60);

  EXPECT_NEAR(anchor.x(), local_target_pos.x(), 1e-6);
  EXPECT_NEAR(anchor.y(), local_target_pos.y(), 1e-6);
}

TEST(ExplorationFSMRealLogicTest, FallbackCandidateAnchorUsesShortButValidGlobalGoal)
{
  const Eigen::Vector2d robot_pos(0.07, 0.36);
  const Eigen::Vector2d local_target_pos(0.07, 0.36);
  const Eigen::Vector2d global_goal_pos(-0.03, 0.02);

  const Eigen::Vector2d anchor = apexnav_planner::selectFallbackCandidateAnchor(
      robot_pos, local_target_pos, global_goal_pos, 0.30, 0.60);

  EXPECT_NEAR(anchor.x(), global_goal_pos.x(), 1e-6);
  EXPECT_NEAR(anchor.y(), global_goal_pos.y(), 1e-6);
}

TEST(ExplorationFSMRealLogicTest, ExplorationTrajectoryTargetUsesLocalPathGoal)
{
  const Eigen::Vector2d global_goal_pos(2.4, -0.8);
  const Eigen::Vector2d local_target_pos(0.5, -0.1);

  const Eigen::Vector2d target = apexnav_planner::selectTrajectoryGoalForMode(
      apexnav_planner::FINAL_RESULT::EXPLORE, global_goal_pos, local_target_pos);

  EXPECT_NEAR(target.x(), local_target_pos.x(), 1e-6);
  EXPECT_NEAR(target.y(), local_target_pos.y(), 1e-6);
}

TEST(ExplorationFSMRealLogicTest, RejectsExplorationFrontierTooCloseForTrajectory)
{
  const Eigen::Vector2d robot_pos(0.05, -0.47);
  const Eigen::Vector2d global_goal_pos(0.22, -0.40);

  EXPECT_TRUE(apexnav_planner::shouldDormantExplorationGoalBeforeTrajectory(
      apexnav_planner::FINAL_RESULT::EXPLORE, robot_pos, global_goal_pos, 0.35));
  EXPECT_FALSE(apexnav_planner::shouldDormantExplorationGoalBeforeTrajectory(
      apexnav_planner::FINAL_RESULT::SEARCH_OBJECT, robot_pos, global_goal_pos, 0.35));
  EXPECT_FALSE(apexnav_planner::shouldDormantExplorationGoalBeforeTrajectory(
      apexnav_planner::FINAL_RESULT::EXPLORE, robot_pos, Eigen::Vector2d(0.80, -0.40), 0.35));
}

TEST(ExplorationFSMRealLogicTest, SearchObjectTrajectoryTargetKeepsLocalSafeGoal)
{
  const Eigen::Vector2d global_goal_pos(2.4, -0.8);
  const Eigen::Vector2d local_target_pos(0.5, -0.1);

  const Eigen::Vector2d target = apexnav_planner::selectTrajectoryGoalForMode(
      apexnav_planner::FINAL_RESULT::SEARCH_OBJECT, global_goal_pos, local_target_pos);

  EXPECT_NEAR(target.x(), local_target_pos.x(), 1e-6);
  EXPECT_NEAR(target.y(), local_target_pos.y(), 1e-6);
}

TEST(ExplorationFSMRealLogicTest, RejectsRefinedFrontierGoalWithoutMeaningfulProgress)
{
  const Eigen::Vector2d start_pos(0.01, 0.17);
  const Eigen::Vector2d refined_pos(0.01, 0.17);

  EXPECT_FALSE(apexnav_planner::shouldAcceptRefinedFrontierGoal(
      start_pos, refined_pos, 0.25));
}

TEST(ExplorationFSMRealLogicTest, AcceptsRefinedFrontierGoalWhenItMovesPastThreshold)
{
  const Eigen::Vector2d start_pos(0.01, 0.17);
  const Eigen::Vector2d refined_pos(0.17, 0.00);

  EXPECT_TRUE(apexnav_planner::shouldAcceptRefinedFrontierGoal(
      start_pos, refined_pos, 0.20));
}

TEST(ExplorationFSMRealLogicTest, FrontierChangeReplanWaitsForDelayDuringExploration)
{
  EXPECT_FALSE(apexnav_planner::shouldReplanForFrontierChange(
      0.8, 2.0, apexnav_planner::FINAL_RESULT::EXPLORE, true));
  EXPECT_TRUE(apexnav_planner::shouldReplanForFrontierChange(
      2.1, 2.0, apexnav_planner::FINAL_RESULT::EXPLORE, true));
  EXPECT_FALSE(apexnav_planner::shouldReplanForFrontierChange(
      4.8, 2.0, apexnav_planner::FINAL_RESULT::EXPLORE, true));
}

TEST(ExplorationFSMRealLogicTest, FrontierChangeDoesNotTriggerOutsideExploreMode)
{
  EXPECT_FALSE(apexnav_planner::shouldReplanForFrontierChange(
      3.0, 2.0, apexnav_planner::FINAL_RESULT::SEARCH_OBJECT, true));
  EXPECT_FALSE(apexnav_planner::shouldReplanForFrontierChange(
      3.0, 2.0, apexnav_planner::FINAL_RESULT::EXPLORE, false));
}

TEST(ExplorationFSMRealLogicTest, KeepsPreviousExplorationGoalWhenFrontierIsStable)
{
  EXPECT_TRUE(apexnav_planner::shouldKeepPreviousExplorationGoal(
      apexnav_planner::FINAL_RESULT::EXPLORE,
      apexnav_planner::FINAL_RESULT::EXPLORE,
      false,
      true));
}

TEST(ExplorationFSMRealLogicTest, DoesNotKeepPreviousGoalAcrossModeOrMapChanges)
{
  EXPECT_FALSE(apexnav_planner::shouldKeepPreviousExplorationGoal(
      apexnav_planner::FINAL_RESULT::SEARCH_OBJECT,
      apexnav_planner::FINAL_RESULT::EXPLORE,
      false,
      true));
  EXPECT_FALSE(apexnav_planner::shouldKeepPreviousExplorationGoal(
      apexnav_planner::FINAL_RESULT::EXPLORE,
      apexnav_planner::FINAL_RESULT::EXPLORE,
      true,
      true));
  EXPECT_FALSE(apexnav_planner::shouldKeepPreviousExplorationGoal(
      apexnav_planner::FINAL_RESULT::EXPLORE,
      apexnav_planner::FINAL_RESULT::EXPLORE,
      false,
      false));
}

TEST(ExplorationFSMRealLogicTest, ReusesSearchObjectGoalAfterTrackingAbortWhenCloseEnough)
{
  const Eigen::Vector2d current_goal(1.0, 1.0);
  const Eigen::Vector2d finish_goal(1.08, 1.02);

  EXPECT_TRUE(apexnav_planner::shouldReuseGoalAfterTrackingAbort(
      apexnav_planner::FINAL_RESULT::SEARCH_OBJECT, true, current_goal, finish_goal, 0.25));
}

TEST(ExplorationFSMRealLogicTest, DoesNotReuseTrackingAbortGoalWhenFarOrInvalid)
{
  const Eigen::Vector2d current_goal(1.0, 1.0);
  const Eigen::Vector2d finish_goal(1.40, 1.00);

  EXPECT_FALSE(apexnav_planner::shouldReuseGoalAfterTrackingAbort(
      apexnav_planner::FINAL_RESULT::SEARCH_OBJECT, true, current_goal, finish_goal, 0.25));
  EXPECT_FALSE(apexnav_planner::shouldReuseGoalAfterTrackingAbort(
      apexnav_planner::FINAL_RESULT::EXPLORE, true, current_goal, current_goal, 0.25));
  EXPECT_FALSE(apexnav_planner::shouldReuseGoalAfterTrackingAbort(
      apexnav_planner::FINAL_RESULT::SEARCH_OBJECT, false, current_goal, current_goal, 0.25));
}

TEST(ExplorationFSMRealLogicTest, BackwardPathSamplingAddsIntermediateCandidates)
{
  const std::vector<Eigen::Vector2d> path{
    Eigen::Vector2d(0.0, 0.0),
    Eigen::Vector2d(0.0, 1.0),
  };

  const auto candidates =
      apexnav_planner::sampleBackwardPathCandidates(path, 1, 0.25);

  ASSERT_GE(candidates.size(), 4u);
  EXPECT_NEAR(candidates.front().pos.x(), 0.0, 1e-6);
  EXPECT_NEAR(candidates.front().pos.y(), 1.0, 1e-6);
  EXPECT_NEAR(candidates.front().yaw, M_PI_2, 1e-6);

  bool found_midpoint = false;
  for (const auto& candidate : candidates) {
    if (std::fabs(candidate.pos.y() - 0.50) < 1e-6) {
      found_midpoint = true;
      EXPECT_NEAR(candidate.yaw, M_PI_2, 1e-6);
    }
  }
  EXPECT_TRUE(found_midpoint);
}

TEST(ExplorationFSMRealLogicTest, FindsLastFootprintSafePoseBeforeFrontierBoundary)
{
  const std::vector<Eigen::Vector2d> path{
    Eigen::Vector2d(0.0, 0.0),
    Eigen::Vector2d(1.0, 0.0),
    Eigen::Vector2d(2.0, 0.0),
    Eigen::Vector2d(3.0, 0.0),
  };
  apexnav_planner::PathPoseCandidate safe_pose;

  const bool found = apexnav_planner::findLastFootprintSafePathPose(
      path,
      [](const Eigen::Vector2d& pos, double yaw) {
        return pos.x() > 1.5 || std::fabs(yaw) > 1e-6;
      },
      safe_pose);

  EXPECT_TRUE(found);
  EXPECT_NEAR(safe_pose.pos.x(), 1.0, 1e-6);
  EXPECT_NEAR(safe_pose.pos.y(), 0.0, 1e-6);
  EXPECT_NEAR(safe_pose.yaw, 0.0, 1e-6);
}

TEST(ExplorationFSMRealLogicTest, ManualFallbackTurnsInPlaceForLargeYawError)
{
  const auto cmd = apexnav_planner::computeManualFallbackCommand(0.40, 0.30);

  EXPECT_DOUBLE_EQ(cmd.first, 0.0);
  EXPECT_NEAR(cmd.second, 0.45, 1e-6);
}

TEST(ExplorationFSMRealLogicTest, ManualFallbackUsesMinimumWalkingSpeedWhenFarEnough)
{
  const auto cmd = apexnav_planner::computeManualFallbackCommand(0.45, 0.08);

  EXPECT_NEAR(cmd.first, 0.22, 1e-6);
  EXPECT_NEAR(cmd.second, 0.08, 1e-6);
}

TEST(ExplorationFSMRealLogicTest, ManualFallbackSlowsDownNearGoal)
{
  const auto cmd = apexnav_planner::computeManualFallbackCommand(0.24, -0.10);

  EXPECT_NEAR(cmd.first, 0.108, 1e-6);
  EXPECT_NEAR(cmd.second, -0.10, 1e-6);
}

TEST(ExplorationFSMRealLogicTest, ExplorationFallbackRequiresCloseContact)
{
  EXPECT_GT(apexnav_planner::FSMConstantsReal::EXPLORATION_FALLBACK_REACH_DISTANCE,
      apexnav_planner::FSMConstantsReal::MANUAL_FALLBACK_REACH_DISTANCE);
  EXPECT_LT(apexnav_planner::FSMConstantsReal::EXPLORATION_FALLBACK_REACH_DISTANCE, 0.20);
}

TEST(ExplorationFSMRealLogicTest, ExplorationFallbackReachForcesDormantFrontier)
{
  EXPECT_TRUE(apexnav_planner::shouldForceDormantAfterExplorationFallback(
      true, 0.12, apexnav_planner::FSMConstantsReal::EXPLORATION_FALLBACK_REACH_DISTANCE));
  EXPECT_FALSE(apexnav_planner::shouldForceDormantAfterExplorationFallback(
      true, 0.30, apexnav_planner::FSMConstantsReal::EXPLORATION_FALLBACK_REACH_DISTANCE));
  EXPECT_FALSE(apexnav_planner::shouldForceDormantAfterExplorationFallback(
      false, 0.26, apexnav_planner::FSMConstantsReal::EXPLORATION_FALLBACK_REACH_DISTANCE));
  EXPECT_FALSE(apexnav_planner::shouldForceDormantAfterExplorationFallback(
      true, 0.45, apexnav_planner::FSMConstantsReal::EXPLORATION_FALLBACK_REACH_DISTANCE));
}

TEST(ExplorationFSMRealLogicTest, ExplorationFallbackRejectsTargetsBeyondDirectRange)
{
  EXPECT_TRUE(apexnav_planner::shouldUseExplorationManualFallback(0.35, 0.50));
  EXPECT_TRUE(apexnav_planner::shouldUseExplorationManualFallback(0.50, 0.50));
  EXPECT_FALSE(apexnav_planner::shouldUseExplorationManualFallback(0.51, 0.50));
  EXPECT_FALSE(apexnav_planner::shouldUseExplorationManualFallback(1.45, 0.50));
}

TEST(ExplorationFSMRealLogicTest, SearchObjectPlanFailureWaitsForLightGlue)
{
  EXPECT_TRUE(apexnav_planner::shouldWaitForLightGlueAfterPlanFailure(
      apexnav_planner::FINAL_RESULT::SEARCH_OBJECT));
  EXPECT_FALSE(apexnav_planner::shouldWaitForLightGlueAfterPlanFailure(
      apexnav_planner::FINAL_RESULT::EXPLORE));
  EXPECT_FALSE(apexnav_planner::shouldWaitForLightGlueAfterPlanFailure(
      apexnav_planner::FINAL_RESULT::NO_FRONTIER));
}

TEST(ExplorationFSMRealLogicTest, SearchObjectPlanFailureSkipsUnverifiedObject)
{
  EXPECT_FALSE(apexnav_planner::shouldSkipObjectNavigationAfterPlanFailure(
      apexnav_planner::FINAL_RESULT::SEARCH_OBJECT, "PENDING_FAR"));
  EXPECT_TRUE(apexnav_planner::shouldSkipObjectNavigationAfterPlanFailure(
      apexnav_planner::FINAL_RESULT::SEARCH_OBJECT, "PENDING"));
  EXPECT_FALSE(apexnav_planner::shouldSkipObjectNavigationAfterPlanFailure(
      apexnav_planner::FINAL_RESULT::SEARCH_OBJECT, "VERIFIED"));
  EXPECT_FALSE(apexnav_planner::shouldSkipObjectNavigationAfterPlanFailure(
      apexnav_planner::FINAL_RESULT::SEARCH_OBJECT, "NO_GOAL_IMAGE"));
  EXPECT_FALSE(apexnav_planner::shouldSkipObjectNavigationAfterPlanFailure(
      apexnav_planner::FINAL_RESULT::EXPLORE, "PENDING_FAR"));
}

TEST(ExplorationFSMRealLogicTest, SearchObjectManualFallbackAllowsVerifiedFarTarget)
{
  EXPECT_FALSE(apexnav_planner::shouldUseSearchObjectManualFallback(
      apexnav_planner::FINAL_RESULT::SEARCH_OBJECT, "PENDING"));
  EXPECT_TRUE(apexnav_planner::shouldUseSearchObjectManualFallback(
      apexnav_planner::FINAL_RESULT::SEARCH_OBJECT, "PENDING_FAR"));
  EXPECT_TRUE(apexnav_planner::shouldUseSearchObjectManualFallback(
      apexnav_planner::FINAL_RESULT::SEARCH_OBJECT, "VERIFIED"));
  EXPECT_TRUE(apexnav_planner::shouldUseSearchObjectManualFallback(
      apexnav_planner::FINAL_RESULT::SEARCH_OBJECT, "NO_GOAL_IMAGE"));
  EXPECT_FALSE(apexnav_planner::shouldUseSearchObjectManualFallback(
      apexnav_planner::FINAL_RESULT::EXPLORE, "VERIFIED"));
}

TEST(ExplorationFSMRealLogicTest, ObjectApproachCompletesOnlyAfterVerified)
{
  EXPECT_TRUE(apexnav_planner::shouldCompleteObjectApproach(
      apexnav_planner::FINAL_RESULT::SEARCH_OBJECT, "VERIFIED"));
  EXPECT_TRUE(apexnav_planner::shouldCompleteObjectApproach(
      apexnav_planner::FINAL_RESULT::SEARCH_OBJECT, "NO_GOAL_IMAGE"));
  EXPECT_FALSE(apexnav_planner::shouldCompleteObjectApproach(
      apexnav_planner::FINAL_RESULT::SEARCH_OBJECT, "PENDING_FAR"));
  EXPECT_FALSE(apexnav_planner::shouldCompleteObjectApproach(
      apexnav_planner::FINAL_RESULT::SEARCH_OBJECT, "PENDING"));
  EXPECT_FALSE(apexnav_planner::shouldCompleteObjectApproach(
      apexnav_planner::FINAL_RESULT::EXPLORE, "VERIFIED"));
}

TEST(ExplorationFSMRealLogicTest, PendingFarContinuesObjectApproach)
{
  EXPECT_TRUE(apexnav_planner::shouldContinueApproachingObject(
      apexnav_planner::FINAL_RESULT::SEARCH_OBJECT, "PENDING_FAR"));
  EXPECT_FALSE(apexnav_planner::shouldContinueApproachingObject(
      apexnav_planner::FINAL_RESULT::SEARCH_OBJECT, "PENDING"));
  EXPECT_FALSE(apexnav_planner::shouldContinueApproachingObject(
      apexnav_planner::FINAL_RESULT::SEARCH_OBJECT, "VERIFIED"));
  EXPECT_FALSE(apexnav_planner::shouldContinueApproachingObject(
      apexnav_planner::FINAL_RESULT::EXPLORE, "PENDING_FAR"));
}

TEST(ExplorationFSMRealLogicTest, ObjectMapPointReachedRetriesUntilLightGlueConfirms)
{
  EXPECT_TRUE(apexnav_planner::shouldRetryObjectApproachAfterMapPointReached(
      apexnav_planner::FINAL_RESULT::SEARCH_OBJECT, "PENDING"));
  EXPECT_TRUE(apexnav_planner::shouldRetryObjectApproachAfterMapPointReached(
      apexnav_planner::FINAL_RESULT::SEARCH_OBJECT, "PENDING_FAR"));
  EXPECT_FALSE(apexnav_planner::shouldRetryObjectApproachAfterMapPointReached(
      apexnav_planner::FINAL_RESULT::SEARCH_OBJECT, "PENDING_CLOSE"));
  EXPECT_FALSE(apexnav_planner::shouldRetryObjectApproachAfterMapPointReached(
      apexnav_planner::FINAL_RESULT::SEARCH_OBJECT, "VERIFIED"));
  EXPECT_FALSE(apexnav_planner::shouldRetryObjectApproachAfterMapPointReached(
      apexnav_planner::FINAL_RESULT::SEARCH_OBJECT, "NO_GOAL_IMAGE"));
  EXPECT_FALSE(apexnav_planner::shouldRetryObjectApproachAfterMapPointReached(
      apexnav_planner::FINAL_RESULT::EXPLORE, "PENDING"));
}

TEST(ExplorationFSMRealLogicTest, FinishLogOnlyReportsConfirmedCompletion)
{
  EXPECT_FALSE(apexnav_planner::shouldReportFinishExploration(false));
  EXPECT_TRUE(apexnav_planner::shouldReportFinishExploration(true));
}

TEST(ExplorationFSMRealLogicTest, StopCandidateStatusStopsMotionWhileConfirming)
{
  EXPECT_TRUE(apexnav_planner::shouldStopMotionForLightGlueStatus("PENDING_CLOSE"));
  EXPECT_TRUE(apexnav_planner::shouldStopMotionForLightGlueStatus("VERIFIED"));
  EXPECT_FALSE(apexnav_planner::shouldStopMotionForLightGlueStatus("PENDING_FAR"));
  EXPECT_FALSE(apexnav_planner::shouldStopMotionForLightGlueStatus("PENDING"));
}

TEST(ExplorationFSMRealLogicTest, HoldsTransientPendingAfterCloseStopCandidate)
{
  EXPECT_TRUE(apexnav_planner::shouldHoldForLightGlueConfirmation("PENDING_CLOSE", false));
  EXPECT_TRUE(apexnav_planner::shouldHoldForLightGlueConfirmation("PENDING", true));
  EXPECT_FALSE(apexnav_planner::shouldHoldForLightGlueConfirmation("PENDING", false));
  EXPECT_FALSE(apexnav_planner::shouldHoldForLightGlueConfirmation("PENDING_FAR", true));
}

TEST(ExplorationFSMRealLogicTest, PendingLightGlueDoesNotResumePlanningWhenNoFrontierRemains)
{
  EXPECT_TRUE(apexnav_planner::shouldResumePlanningFromPendingLightGlue(
      apexnav_planner::FINAL_RESULT::SEARCH_OBJECT, "PENDING"));
  EXPECT_TRUE(apexnav_planner::shouldResumePlanningFromPendingLightGlue(
      apexnav_planner::FINAL_RESULT::EXPLORE, "PENDING"));
  EXPECT_FALSE(apexnav_planner::shouldResumePlanningFromPendingLightGlue(
      apexnav_planner::FINAL_RESULT::NO_FRONTIER, "PENDING"));
  EXPECT_FALSE(apexnav_planner::shouldResumePlanningFromPendingLightGlue(
      apexnav_planner::FINAL_RESULT::SEARCH_OBJECT, "PENDING_FAR"));
}

TEST(ExplorationFSMRealLogicTest, PendingFarPreemptsExplorationFallback)
{
  EXPECT_TRUE(apexnav_planner::shouldPreemptExplorationFallbackForLightGlueStatus(
      "PENDING_FAR", true, true));
  EXPECT_FALSE(apexnav_planner::shouldPreemptExplorationFallbackForLightGlueStatus(
      "PENDING", true, true));
  EXPECT_FALSE(apexnav_planner::shouldPreemptExplorationFallbackForLightGlueStatus(
      "PENDING_FAR", false, true));
  EXPECT_FALSE(apexnav_planner::shouldPreemptExplorationFallbackForLightGlueStatus(
      "PENDING_FAR", true, false));
}

TEST(ExplorationFSMRealLogicTest, VerifiedStopRequiresHoldBeforeFinalFinish)
{
  EXPECT_TRUE(apexnav_planner::shouldFinalizeVerifiedStop("VERIFIED", 0.0, 0.0));
  EXPECT_FALSE(apexnav_planner::shouldFinalizeVerifiedStop("PENDING_FAR", 0.0, 0.0));
  EXPECT_FALSE(apexnav_planner::shouldFinalizeVerifiedStop("VERIFIED", 0.2, 1.0));
  EXPECT_TRUE(apexnav_planner::shouldFinalizeVerifiedStop("VERIFIED", 1.2, 1.0));
  EXPECT_DOUBLE_EQ(apexnav_planner::FSMConstantsReal::LIGHTGLUE_VERIFIED_HOLD_TIME, 0.0);
}

TEST(ExplorationFSMRealLogicTest, RecentFinishedStopCanBeRevokedWhenTargetIsFarAgain)
{
  EXPECT_TRUE(apexnav_planner::shouldRevokeFinishedStop("PENDING_FAR", 0.3, 3.0));
  EXPECT_FALSE(apexnav_planner::shouldRevokeFinishedStop("PENDING_FAR", 3.5, 3.0));
  EXPECT_FALSE(apexnav_planner::shouldRevokeFinishedStop("VERIFIED", 0.3, 3.0));
}

TEST(ExplorationFSMRealLogicTest, ObjectMarkersStayHiddenWhenVisualizationDisabled)
{
  EXPECT_FALSE(apexnav_planner::shouldVisualizeObjectMarker(false, 0));
  EXPECT_FALSE(apexnav_planner::shouldVisualizeObjectMarker(false, 2));
}

TEST(ExplorationFSMRealLogicTest, ObjectMarkersSkipInvalidLabels)
{
  EXPECT_FALSE(apexnav_planner::shouldVisualizeObjectMarker(true, -1));
  EXPECT_TRUE(apexnav_planner::shouldVisualizeObjectMarker(true, 0));
  EXPECT_TRUE(apexnav_planner::shouldVisualizeObjectMarker(true, 3));
}

TEST(ExplorationFSMRealLogicTest, DoesNotAbortTrajectoryForModerateTrackingError)
{
  const Eigen::Vector2d planned_pos(0.0, 0.0);
  const Eigen::Vector2d odom_pos(0.55, 0.0);

  EXPECT_FALSE(
      apexnav_planner::shouldAbortTrajectoryForTrackingError(planned_pos, odom_pos, 0.65));
}

TEST(ExplorationFSMRealLogicTest, AbortsTrajectoryForLargeTrackingError)
{
  const Eigen::Vector2d planned_pos(0.0, 0.0);
  const Eigen::Vector2d odom_pos(0.75, 0.0);

  EXPECT_TRUE(
      apexnav_planner::shouldAbortTrajectoryForTrackingError(planned_pos, odom_pos, 0.65));
}

TEST(ExplorationFSMRealLogicTest, StopsTrajectoryServerWhenManualFallbackStarts)
{
  EXPECT_TRUE(apexnav_planner::shouldStopTrajectoryServerForManualFallback(
      false, Eigen::Vector2d::Zero(), Eigen::Vector2d(1.0, 0.0), 0.10));
}

TEST(ExplorationFSMRealLogicTest, StopsTrajectoryServerWhenFallbackGoalChanges)
{
  EXPECT_TRUE(apexnav_planner::shouldStopTrajectoryServerForManualFallback(
      true, Eigen::Vector2d(0.0, 0.0), Eigen::Vector2d(0.2, 0.0), 0.10));
  EXPECT_FALSE(apexnav_planner::shouldStopTrajectoryServerForManualFallback(
      true, Eigen::Vector2d(0.0, 0.0), Eigen::Vector2d(0.05, 0.0), 0.10));
}

}  // namespace
