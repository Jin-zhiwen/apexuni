#include <gtest/gtest.h>

#include <Eigen/Core>
#include <visualization_msgs/Marker.h>

#include <exploration_manager/exploration_data.h>
#include <exploration_manager/exploration_fsm_traj_logic.h>
#include <exploration_manager/exploration_fsm_traj.h>
#include <path_searching/kino_astar.h>
#include <plan_env/frontier_map2d.h>

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

TEST(ExplorationFSMRealLogicTest, InitialKinoAstarArcSamplesUseForwardEscapeScaleWhenRobotStartsStatic)
{
  const auto arcs =
      apexnav_planner::sampleInitialKinoAstarArcInputs(0.05, 0.2, true);

  ASSERT_EQ(arcs.size(), 2u);
  EXPECT_NEAR(arcs[0], 0.10, 1e-6);
  EXPECT_NEAR(arcs[1], 0.20, 1e-6);
  for (const double arc : arcs) {
    EXPECT_GT(arc, 0.0);
  }
}

TEST(ExplorationFSMRealLogicTest, InitialKinoAstarArcSamplesKeepForwardGridScaleWhenAlreadyMoving)
{
  const auto arcs =
      apexnav_planner::sampleInitialKinoAstarArcInputs(0.05, 0.2, false);

  ASSERT_EQ(arcs.size(), 2u);
  EXPECT_NEAR(arcs[0], 0.05, 1e-6);
  EXPECT_NEAR(arcs[1], 0.10, 1e-6);
  for (const double arc : arcs) {
    EXPECT_GT(arc, 0.0);
  }
}

TEST(ExplorationFSMRealLogicTest, BuildsFootprintCollisionMarkerAtRejectedPoint)
{
  const Eigen::Vector2d collision_pos(0.06, 0.68);

  const auto marker = apexnav_planner::makeFootprintCollisionPointMarker(
      collision_pos, 3.35, "odom");

  EXPECT_EQ(marker.header.frame_id, "odom");
  EXPECT_EQ(marker.ns, "footprint_collision");
  EXPECT_EQ(marker.id, 0);
  EXPECT_EQ(marker.type, visualization_msgs::Marker::SPHERE);
  EXPECT_EQ(marker.action, visualization_msgs::Marker::ADD);
  EXPECT_NEAR(marker.pose.position.x, collision_pos.x(), 1e-6);
  EXPECT_NEAR(marker.pose.position.y, collision_pos.y(), 1e-6);
  EXPECT_NEAR(marker.pose.position.z, 0.08, 1e-6);
  EXPECT_NEAR(marker.scale.x, 0.18, 1e-6);
  EXPECT_NEAR(marker.scale.y, 0.18, 1e-6);
  EXPECT_NEAR(marker.scale.z, 0.18, 1e-6);
  EXPECT_NEAR(marker.color.r, 1.0, 1e-6);
  EXPECT_NEAR(marker.color.g, 0.05, 1e-6);
  EXPECT_NEAR(marker.color.b, 0.05, 1e-6);
  EXPECT_NEAR(marker.color.a, 1.0, 1e-6);
  EXPECT_EQ(marker.text, "footprint collision t=3.35s");
}

TEST(ExplorationFSMRealLogicTest, KinoAstarCollisionPolicyAlwaysBlocksUnknownCells)
{
  EXPECT_TRUE(apexnav_planner::shouldTreatOccupancyAsKinoCollision(
      apexnav_planner::SDFMap2D::OCCUPIED));
  EXPECT_TRUE(apexnav_planner::shouldTreatOccupancyAsKinoCollision(
      apexnav_planner::SDFMap2D::UNKNOWN));
  EXPECT_FALSE(apexnav_planner::shouldTreatOccupancyAsKinoCollision(
      apexnav_planner::SDFMap2D::FREE));
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

TEST(ExplorationFSMRealLogicTest, ExplorationTrajectoryTargetUsesLocalGoal)
{
  const Eigen::Vector2d global_goal_pos(2.4, -0.8);
  const Eigen::Vector2d local_target_pos(0.5, -0.1);

  const Eigen::Vector2d target = apexnav_planner::selectTrajectoryGoalForMode(
      apexnav_planner::FINAL_RESULT::EXPLORE, global_goal_pos, local_target_pos);

  EXPECT_NEAR(target.x(), local_target_pos.x(), 1e-6);
  EXPECT_NEAR(target.y(), local_target_pos.y(), 1e-6);
}

TEST(ExplorationFSMRealLogicTest, SearchObjectTrajectoryTargetUsesLocalGoal)
{
  const Eigen::Vector2d global_goal_pos(2.4, -0.8);
  const Eigen::Vector2d local_target_pos(0.5, -0.1);

  const Eigen::Vector2d target = apexnav_planner::selectTrajectoryGoalForMode(
      apexnav_planner::FINAL_RESULT::SEARCH_OBJECT, global_goal_pos, local_target_pos);

  EXPECT_NEAR(target.x(), local_target_pos.x(), 1e-6);
  EXPECT_NEAR(target.y(), local_target_pos.y(), 1e-6);
}

TEST(ExplorationFSMRealLogicTest, FindsLastFootprintSafePoseBeforeCollidingPathEnd)
{
  const std::vector<Eigen::Vector2d> path{
    Eigen::Vector2d(0.0, 0.0),
    Eigen::Vector2d(1.0, 0.0),
    Eigen::Vector2d(2.0, 0.0),
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

TEST(ExplorationFSMRealLogicTest, LocalTargetBacksOffToFootprintSafePose)
{
  const std::vector<Eigen::Vector2d> path{
    Eigen::Vector2d(0.0, 0.0),
    Eigen::Vector2d(1.0, 0.0),
    Eigen::Vector2d(2.0, 0.0),
    Eigen::Vector2d(3.0, 0.0),
  };
  apexnav_planner::PathPoseCandidate safe_pose;

  const bool found = apexnav_planner::selectFootprintSafeLocalTargetFromPath(
      Eigen::Vector2d(0.0, 0.0),
      path,
      2.5,
      0.30,
      [](const Eigen::Vector2d& pos, double yaw) {
        return pos.x() >= 2.0 || std::fabs(yaw) > 1e-6;
      },
      safe_pose);

  EXPECT_TRUE(found);
  EXPECT_NEAR(safe_pose.pos.x(), 1.0, 1e-6);
  EXPECT_NEAR(safe_pose.pos.y(), 0.0, 1e-6);
  EXPECT_NEAR(safe_pose.yaw, 0.0, 1e-6);
}

TEST(ExplorationFSMRealLogicTest, LocalTargetDoesNotFallThroughToDistantFrontier)
{
  const std::vector<Eigen::Vector2d> path{
    Eigen::Vector2d(0.0, 0.0),
    Eigen::Vector2d(3.0, 0.0),
  };
  apexnav_planner::PathPoseCandidate safe_pose;

  const bool found = apexnav_planner::selectFootprintSafeLocalTargetFromPath(
      Eigen::Vector2d(0.0, 0.0),
      path,
      1.5,
      0.30,
      [](const Eigen::Vector2d&, double) {
        return false;
      },
      safe_pose);

  EXPECT_FALSE(found);
}

TEST(ExplorationFSMRealLogicTest, ForwardLocalTargetRejectsPointBehindRobot)
{
  const std::vector<Eigen::Vector2d> path{
    Eigen::Vector2d(0.0, 0.0),
    Eigen::Vector2d(-0.8, 0.0),
    Eigen::Vector2d(-1.2, 0.0),
  };
  apexnav_planner::PathPoseCandidate safe_pose;

  const bool found = apexnav_planner::selectFootprintSafeForwardLocalTargetFromPath(
      Eigen::Vector2d(0.0, 0.0),
      0.0,
      path,
      1.5,
      0.30,
      1.57079632679,
      [](const Eigen::Vector2d&, double) {
        return false;
      },
      safe_pose);

  EXPECT_FALSE(found);
}

TEST(ExplorationFSMRealLogicTest, ForwardLocalTargetAcceptsPointInsideForwardSector)
{
  const std::vector<Eigen::Vector2d> path{
    Eigen::Vector2d(0.0, 0.0),
    Eigen::Vector2d(0.5, 0.5),
    Eigen::Vector2d(1.0, 0.8),
  };
  apexnav_planner::PathPoseCandidate safe_pose;

  const bool found = apexnav_planner::selectFootprintSafeForwardLocalTargetFromPath(
      Eigen::Vector2d(0.0, 0.0),
      0.0,
      path,
      1.5,
      0.30,
      1.57079632679,
      [](const Eigen::Vector2d&, double) {
        return false;
      },
      safe_pose);

  EXPECT_TRUE(found);
  EXPECT_GT(safe_pose.pos.x(), 0.0);
  EXPECT_GT(safe_pose.pos.y(), 0.0);
}

TEST(ExplorationFSMRealLogicTest, LocalTargetClampKeepsGradientAdjustmentInsideLookahead)
{
  const Eigen::Vector2d current_pos(0.0, 0.0);
  const Eigen::Vector2d adjusted_pos(2.4, 1.8);

  const Eigen::Vector2d clamped_pos =
      apexnav_planner::clampLocalTargetToLookahead(current_pos, adjusted_pos, 1.5);

  EXPECT_NEAR((clamped_pos - current_pos).norm(), 1.5, 1e-6);
  EXPECT_NEAR(clamped_pos.x(), 1.2, 1e-6);
  EXPECT_NEAR(clamped_pos.y(), 0.9, 1e-6);
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
      0.8, 2.0, apexnav_planner::FINAL_RESULT::EXPLORE, true, true));
  EXPECT_TRUE(apexnav_planner::shouldReplanForFrontierChange(
      2.1, 2.0, apexnav_planner::FINAL_RESULT::EXPLORE, true, true));
  EXPECT_TRUE(apexnav_planner::shouldReplanForFrontierChange(
      4.8, 2.0, apexnav_planner::FINAL_RESULT::EXPLORE, true, true));
}

TEST(ExplorationFSMRealLogicTest, FrontierChangeReplanCanBeDisabledDuringExecution)
{
  EXPECT_FALSE(apexnav_planner::shouldReplanForFrontierChange(
      3.6, 3.5, apexnav_planner::FINAL_RESULT::EXPLORE, true, false));
}

TEST(ExplorationFSMRealLogicTest, FrontierChangeDoesNotTriggerOutsideExploreMode)
{
  EXPECT_FALSE(apexnav_planner::shouldReplanForFrontierChange(
      3.0, 2.0, apexnav_planner::FINAL_RESULT::SEARCH_OBJECT, true, true));
  EXPECT_FALSE(apexnav_planner::shouldReplanForFrontierChange(
      3.0, 2.0, apexnav_planner::FINAL_RESULT::EXPLORE, false, true));
}

TEST(ExplorationFSMRealLogicTest, MovingReplanUsesOdomWhenTrackingErrorIsLarge)
{
  EXPECT_TRUE(apexnav_planner::shouldUseOdometryStartForReplan(false, 0.35, 0.20));
  EXPECT_FALSE(apexnav_planner::shouldUseOdometryStartForReplan(false, 0.08, 0.20));
  EXPECT_TRUE(apexnav_planner::shouldUseOdometryStartForReplan(true, 0.0, 0.20));
}

TEST(ExplorationFSMRealLogicTest, RealFsmUsesApexnavMainScaleLocalTargetDistanceByDefault)
{
  const apexnav_planner::FSMParam params;

  EXPECT_NEAR(params.local_target_distance_, 4.0, 1e-6);
}

TEST(ExplorationFSMRealLogicTest, TrajectoryTimeoutDoesNotReplanBeforeTimeout)
{
  EXPECT_FALSE(apexnav_planner::shouldReplanForTrajectoryTimeout(
      8.0, 10.0, 20.0, true));
}

TEST(ExplorationFSMRealLogicTest, TrajectoryTimeoutDoesNotReplanAfterRecentProgress)
{
  EXPECT_FALSE(apexnav_planner::shouldReplanForTrajectoryTimeout(
      12.0, 10.0, 1.5, true));
}

TEST(ExplorationFSMRealLogicTest, TrajectoryTimeoutReplansWhenProgressStops)
{
  EXPECT_TRUE(apexnav_planner::shouldReplanForTrajectoryTimeout(
      12.0, 10.0, 10.5, true));
}

TEST(ExplorationFSMRealLogicTest, StopsActiveTrajectoryBeforeTimeoutReplan)
{
  EXPECT_TRUE(apexnav_planner::shouldStopActiveTrajectoryBeforeReplan(true));
  EXPECT_FALSE(apexnav_planner::shouldStopActiveTrajectoryBeforeReplan(false));
}

TEST(ExplorationFSMRealLogicTest, ExcludesFrontierCellInsidePersistentStartZone)
{
  const std::vector<Eigen::Vector2d> frontier_cells{
    Eigen::Vector2d(0.45, 0.0),
    Eigen::Vector2d(0.39, 0.0),
  };

  EXPECT_TRUE(apexnav_planner::FrontierMap2D::shouldExcludeCellsNearZone(
      frontier_cells, true, Eigen::Vector2d::Zero(), 0.40));
}

TEST(ExplorationFSMRealLogicTest, KeepsFrontierOutsidePersistentStartZone)
{
  const std::vector<Eigen::Vector2d> frontier_cells{
    Eigen::Vector2d(0.45, 0.0),
    Eigen::Vector2d(0.50, 0.0),
  };

  EXPECT_FALSE(apexnav_planner::FrontierMap2D::shouldExcludeCellsNearZone(
      frontier_cells, true, Eigen::Vector2d::Zero(), 0.40));
}

TEST(ExplorationFSMRealLogicTest, DisabledPersistentStartZoneDoesNotExclude)
{
  const std::vector<Eigen::Vector2d> frontier_cells{
    Eigen::Vector2d(0.10, 0.0),
  };

  EXPECT_FALSE(apexnav_planner::FrontierMap2D::shouldExcludeCellsNearZone(
      frontier_cells, false, Eigen::Vector2d::Zero(), 0.40));
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

TEST(ExplorationFSMRealLogicTest, DoesNotKeepPreviousGoalAfterForcedDormantFrontier)
{
  EXPECT_FALSE(apexnav_planner::shouldKeepPreviousExplorationGoal(
      apexnav_planner::FINAL_RESULT::EXPLORE,
      apexnav_planner::FINAL_RESULT::EXPLORE,
      false,
      true,
      true));
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

TEST(ExplorationFSMRealLogicTest, HoldsObjectNavigationSuppressedWhileLightGlueStaysPending)
{
  EXPECT_TRUE(apexnav_planner::shouldHoldObjectNavigationSuppressed("PENDING"));
  EXPECT_FALSE(apexnav_planner::shouldHoldObjectNavigationSuppressed("PENDING_FAR"));
  EXPECT_FALSE(apexnav_planner::shouldHoldObjectNavigationSuppressed("PENDING_CLOSE"));
  EXPECT_FALSE(apexnav_planner::shouldHoldObjectNavigationSuppressed("VERIFIED"));
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
  const Eigen::Vector2d odom_pos(0.25, 0.0);

  EXPECT_FALSE(
      apexnav_planner::shouldAbortTrajectoryForTrackingError(
          planned_pos, odom_pos, apexnav_planner::FSMConstantsReal::DEFAULT_TRACKING_ABORT_DISTANCE));
}

TEST(ExplorationFSMRealLogicTest, AbortsTrajectoryForLargeTrackingError)
{
  const Eigen::Vector2d planned_pos(0.0, 0.0);
  const Eigen::Vector2d odom_pos(0.35, 0.0);

  EXPECT_TRUE(
      apexnav_planner::shouldAbortTrajectoryForTrackingError(
          planned_pos, odom_pos, apexnav_planner::FSMConstantsReal::DEFAULT_TRACKING_ABORT_DISTANCE));
}

}  // namespace
