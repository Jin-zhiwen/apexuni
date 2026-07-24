/**
 * @file exploration_manager.cpp
 * @brief Implementation of exploration manager for autonomous semantic navigation
 * @author Zager-Zhang
 *
 * This file implements the ExplorationManager class that handles various
 * exploration strategies including distance-based, semantic-based, hybrid,
 * and TSP-optimized frontier selection for autonomous robot exploration.
 */

#include <exploration_manager/exploration_manager.h>
#include <exploration_manager/exploration_data.h>
#include <lkh_mtsp_solver/SolveMTSP.h>
#include <plan_env/map_ros.h>
#include <path_searching/kino_astar.h>
#include <trajectory_manager/optimizer.h>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>

using namespace Eigen;

namespace apexnav_planner {

ExplorationManager::~ExplorationManager() = default;

void ExplorationManager::initialize(ros::NodeHandle& nh)
{
  // Initialize SDF map and get object map reference
  sdf_map_.reset(new SDFMap2D);
  sdf_map_->initMap(nh);
  object_map2d_ = sdf_map_->object_map2d_;

  // Initialize frontier map and path finder
  frontier_map2d_.reset(new FrontierMap2D(sdf_map_, nh));
  path_finder_.reset(new Astar2D);
  path_finder_->init(nh, sdf_map_);

  // Initialize exploration data and parameter containers
  ed_.reset(new ExplorationData);
  ep_.reset(new ExplorationParam);

  // Load exploration parameters from ROS parameter server
  nh.param("exploration/policy", ep_->policy_mode_, 0);
  nh.param("exploration/sigma_threshold", ep_->sigma_threshold_, 0.030);
  nh.param("exploration/max_to_mean_threshold", ep_->max_to_mean_threshold_, 1.2);
  nh.param("exploration/max_to_mean_percentage", ep_->max_to_mean_percentage_, 0.95);
  nh.param("exploration/tsp_dir", ep_->tsp_dir_, string("null"));
  nh.param("exploration/object_viewpoint/enabled", ep_->object_viewpoint_enabled_, false);
  nh.param("exploration/object_viewpoint/min_score", ep_->object_viewpoint_min_score_, 0.50);
  nh.param("exploration/object_viewpoint/standoff_near",
      ep_->object_viewpoint_standoff_near_, 1.40);
  nh.param("exploration/object_viewpoint/standoff_mid",
      ep_->object_viewpoint_standoff_mid_, 1.70);
  nh.param("exploration/object_viewpoint/standoff_far",
      ep_->object_viewpoint_standoff_far_, 2.00);
  nh.param("exploration/object_viewpoint/standoff_extra",
      ep_->object_viewpoint_standoff_extra_, 2.30);
  nh.param("exploration/object_viewpoint/reach_distance",
      ep_->object_viewpoint_reach_distance_, 0.25);
  nh.param("exploration/object_viewpoint/min_visible_ratio",
      ep_->object_viewpoint_min_visible_ratio_, 0.55);
  nh.param("exploration/object_viewpoint/min_fov_deg", ep_->object_viewpoint_min_fov_deg_, 25.0);
  nh.param("exploration/object_viewpoint/max_fov_deg", ep_->object_viewpoint_max_fov_deg_, 65.0);
  nh.param("exploration/object_viewpoint/ideal_fov_deg",
      ep_->object_viewpoint_ideal_fov_deg_, 48.0);
  nh.param("exploration/object_viewpoint/fov_sigma_deg",
      ep_->object_viewpoint_fov_sigma_deg_, 15.0);
  nh.param("exploration/object_viewpoint/astar_time", ep_->object_viewpoint_astar_time_, 0.03);
  nh.param("exploration/object_viewpoint/astar_budget",
      ep_->object_viewpoint_astar_budget_, 0.15);
  nh.param("exploration/object_viewpoint/min_path_length",
      ep_->object_viewpoint_min_path_length_, 0.30);
  nh.param("exploration/object_viewpoint/min_azimuth_sep_deg",
      ep_->object_viewpoint_min_azimuth_sep_deg_, 60.0);
  nh.param("exploration/object_viewpoint/min_position_sep",
      ep_->object_viewpoint_min_position_sep_, 0.70);
  nh.param("exploration/object_viewpoint/pre_path_distance_weight",
      ep_->object_viewpoint_pre_path_distance_weight_, 0.10);
  nh.param("exploration/object_viewpoint/unknown_ray_penalty",
      ep_->object_viewpoint_unknown_ray_penalty_, 0.35);
  nh.param("exploration/object_viewpoint/target_surface_clearance",
      ep_->object_viewpoint_target_surface_clearance_, 0.40);
  nh.param("exploration/object_viewpoint/min_observations",
      ep_->object_viewpoint_min_observations_, 2);
  nh.param("exploration/object_viewpoint/max_lock_steps",
      ep_->object_viewpoint_max_lock_steps_, 60);
  nh.param("exploration/object_viewpoint/direction_count",
      ep_->object_viewpoint_direction_count_, 16);
  nh.param("exploration/object_viewpoint/visibility_samples",
      ep_->object_viewpoint_visibility_samples_, 32);
  nh.param("exploration/object_viewpoint/max_astar_candidates",
      ep_->object_viewpoint_max_astar_candidates_, 16);
  nh.param("exploration/object_viewpoint/max_views", ep_->object_viewpoint_max_views_, 3);
  nh.param("exploration/object_viewpoint/max_failed_inspections",
      ep_->object_viewpoint_max_failed_inspections_, 2);
  nh.param("exploration/object_viewpoint/failure_cooldown_sequences",
      ep_->object_viewpoint_failure_cooldown_sequences_, 30);
  nh.param("exploration/object_viewpoint/retry_score_margin",
      ep_->object_viewpoint_retry_score_margin_, 0.05);
  nh.param("exploration/object_viewpoint/failure_spatial_radius",
      ep_->object_viewpoint_failure_spatial_radius_, 0.75);
  ep_->object_viewpoint_max_failed_inspections_ =
      std::max(1, ep_->object_viewpoint_max_failed_inspections_);
  ep_->object_viewpoint_failure_cooldown_sequences_ =
      std::max(0, ep_->object_viewpoint_failure_cooldown_sequences_);
  ep_->object_viewpoint_retry_score_margin_ =
      std::max(0.0, ep_->object_viewpoint_retry_score_margin_);
  ep_->object_viewpoint_failure_spatial_radius_ =
      std::max(0.0, ep_->object_viewpoint_failure_spatial_radius_);

  verified_approach_active_ = false;
  verified_approach_target_locked_ = false;
  verified_approach_target_failed_ = false;
  verified_approach_target_.setZero();
  verified_approach_cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>());
  object_viewpoint_locked_ = false;
  locked_object_id_ = -1;
  locked_object_label_ = -1;
  object_viewpoint_lock_steps_ = 0;
  locked_object_viewpoint_index_ = 0;
  locked_object_evidence_epoch_ = 0;
  locked_object_score_ = -1.0;
  locked_object_viewpoint_plans_.clear();
  completed_object_evidence_epochs_.clear();
  object_inspection_failures_.clear();

  ROS_INFO(
      "[OBJECT_VIEWPOINT_RETRY] max_failures=%d cooldown_sequences=%d "
      "retry_score_margin=%.3f spatial_radius=%.2f",
      ep_->object_viewpoint_max_failed_inspections_,
      ep_->object_viewpoint_failure_cooldown_sequences_,
      ep_->object_viewpoint_retry_score_margin_,
      ep_->object_viewpoint_failure_spatial_radius_);

  // Get map parameters for ray casting initialization
  double resolution = sdf_map_->getResolution();
  Eigen::Vector2d origin, size;
  sdf_map_->getRegion(origin, size);

  // Initialize ray caster for collision checking and TSP service client
  ray_caster2d_.reset(new RayCaster2D);
  ray_caster2d_->setParams(resolution, origin);
  tsp_client_ = nh.serviceClient<lkh_mtsp_solver::SolveMTSP>("/solve_tsp", true);
  object_viewpoint_debug_pub_ =
      nh.advertise<std_msgs::String>("/ros/object_viewpoint_debug", 20);

  // Initialize KinoAstar and GCopter for real-world trajectory planning
  kinoastar_.reset(new KinoAstar(nh, sdf_map_));
  kinoastar_->init();
  
  Config gcopter_config(nh);
  gcopter_.reset(new Gcopter(gcopter_config, nh, sdf_map_, kinoastar_));
  
  ROS_INFO("[ExplorationManager] KinoAstar and GCopter initialized for real-world mode");
}

int ExplorationManager::planNextBestPoint(const Vector3d& pos, const double& yaw)
{
  Vector2d pos2d = Vector2d(pos(0), pos(1));
  ros::Time t1 = ros::Time::now();
  auto t2 = t1;

  // Clear previous planning results
  ed_->tsp_tour_.clear();
  ed_->next_best_path_.clear();

  if (verified_approach_active_ ||
      (ep_->object_viewpoint_enabled_ && object_map2d_->isSemanticGateEnabled()))
    return planSemanticObjectOrFrontier(pos, yaw);

  vector<pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>> object_clouds;
  sdf_map_->object_map2d_->getTopConfidenceObjectCloud(object_clouds);

  // ==================== Navigation Mode: High-Confidence Objects ====================
  if (!object_clouds.empty()) {
    ROS_WARN("[Navigation Mode] Get object_cloud num = %ld", object_clouds.size());

    // Try to find path to each detected object in order of confidence
    for (auto object_cloud : object_clouds) {
      if (searchObjectPath(pos, object_cloud, ed_->next_pos_, ed_->next_best_path_))
        return SEARCH_BEST_OBJECT;
    }
  }

  // ==================== Navigation Mode: Over-Depth Objects ====================
  if (!object_map2d_->over_depth_object_cloud_->points.empty()) {
    ROS_WARN("[Navigation Mode (Over Depth)] Get over depth object cloud");
    if (searchObjectPath(
            pos, object_map2d_->over_depth_object_cloud_, ed_->next_pos_, ed_->next_best_path_))
      return SEARCH_OVER_DEPTH_OBJECT;
  }

  // In INSiNav, semantic-gated objects are instance candidates worth actively verifying.
  // They may not yet satisfy object-map confidence/observation counts, so try them before
  // falling back to frontier exploration.
  if (sdf_map_->object_map2d_->isSemanticGateEnabled()) {
    sdf_map_->object_map2d_->getTopConfidenceObjectCloud(object_clouds, false);
    if (!object_clouds.empty()) {
      ROS_WARN("[Navigation Mode (Semantic Candidate)] Get object_cloud num = %ld",
          object_clouds.size());
      for (auto object_cloud : object_clouds) {
        if (!object_cloud->points.empty() &&
            searchObjectPath(pos, object_cloud, ed_->next_pos_, ed_->next_best_path_))
          return SEARCH_SUSPICIOUS_OBJECT;
      }
    }
  }

  // ==================== Exploration Mode: Frontier-Based Planning ====================
  sdf_map_->object_map2d_->getTopConfidenceObjectCloud(object_clouds, false);
  pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> top_object_cloud(
      new pcl::PointCloud<pcl::PointXYZ>);
  if (object_clouds.size() >= 1)
    top_object_cloud = object_clouds[0];

  // Apply selected exploration policy to choose next frontier
  Eigen::Vector2d next_best_pos;
  std::vector<Eigen::Vector2d> next_best_path;
  chooseExplorationPolicy(pos2d, ed_->frontier_averages_, next_best_pos, next_best_path);

  // Handle case when no passable frontiers are found
  if (next_best_path.empty()) {
    ROS_WARN("Maybe no passable frontier.");

    // Try suspicious objects as backup
    if (!top_object_cloud->points.empty() &&
        searchObjectPath(pos, top_object_cloud, ed_->next_pos_, ed_->next_best_path_))
      return SEARCH_SUSPICIOUS_OBJECT;
    else
      // Try dormant frontiers as last resort
      chooseExplorationPolicy(
          pos2d, ed_->dormant_frontier_averages_, next_best_pos, next_best_path);

    // Extreme search mode when all normal options fail
    if (next_best_path.empty()) {
      ROS_ERROR("search exterme case!!!");

      // Try extreme object search with relaxed constraints
      for (auto object_cloud : object_clouds) {
        if (!object_cloud->points.empty() &&
            searchObjectPathExtreme(pos, object_cloud, ed_->next_pos_, ed_->next_best_path_))
          return SEARCH_EXTREME;
      }

      // Include lower confidence objects in extreme search
      sdf_map_->object_map2d_->getTopConfidenceObjectCloud(object_clouds, false, true);
      for (auto object_cloud : object_clouds) {
        if (!object_cloud->points.empty() &&
            searchObjectPathExtreme(pos, object_cloud, ed_->next_pos_, ed_->next_best_path_))
          return SEARCH_EXTREME;
      }

      // Try cached over-depth objects as final option
      static auto last_over_depth_object_cloud = object_map2d_->over_depth_object_cloud_;
      if (!object_map2d_->over_depth_object_cloud_->points.empty())
        last_over_depth_object_cloud = object_map2d_->over_depth_object_cloud_;

      if (!last_over_depth_object_cloud->points.empty() &&
          searchObjectPathExtreme(
              pos, last_over_depth_object_cloud, ed_->next_pos_, ed_->next_best_path_)) {
        return SEARCH_EXTREME;
      }
    }

    // Final error handling when no valid targets exist
    if (next_best_path.empty()) {
      if (ed_->frontiers_.empty()) {
        ROS_ERROR("No coverable frontier!!");
        return NO_COVERABLE_FRONTIER;
      }
      else {
        ROS_ERROR("No passable frontier!!");
        return NO_PASSABLE_FRONTIER;
      }
    }
  }

  // Store successful planning results
  ed_->next_pos_ = next_best_pos;
  ed_->next_best_path_ = next_best_path;

  // Performance monitoring
  double total_time = (ros::Time::now() - t2).toSec();
  ROS_ERROR_COND(total_time > 0.25, "[Plan NBV] Total time %.2lf s too long!!!", total_time);

  return EXPLORATION;
}

bool ExplorationManager::hasLockedObjectViewpoint() const
{
  return object_viewpoint_locked_;
}

Vector2d ExplorationManager::getLockedObjectViewpoint() const
{
  return locked_object_viewpoint_;
}

Vector2d ExplorationManager::getLockedObjectCenter() const
{
  return locked_object_center_;
}

double ExplorationManager::getObjectViewpointReachDistance() const
{
  return ep_ != nullptr ? ep_->object_viewpoint_reach_distance_ : 0.25;
}

bool ExplorationManager::isVerifiedApproachActive() const
{
  return verified_approach_active_;
}

void ExplorationManager::publishObjectViewpointDebug(const std::string& text)
{
  std_msgs::String msg;
  msg.data = text;
  object_viewpoint_debug_pub_.publish(msg);
}

void ExplorationManager::setVerifiedApproachState(int state)
{
  const bool active = state > 0;
  if (verified_approach_active_ == active)
    return;

  const bool was_active = verified_approach_active_;
  verified_approach_active_ = active;
  if (active && !was_active) {
    verified_approach_target_failed_ = false;
    if (object_viewpoint_locked_) {
      ROS_INFO("[OBJECT_VIEWPOINT_LOCK] suspend DINO inspection id=%d for LightGlue target",
          locked_object_id_);
      std::ostringstream debug;
      debug << "event=lock_suspend reason=lightglue_verified_preempted id=" << locked_object_id_;
      publishObjectViewpointDebug(debug.str());
    }
  }
  else {
    if (!active) {
      verified_approach_target_locked_ = false;
      verified_approach_target_failed_ = false;
      verified_approach_target_.setZero();
      verified_approach_cloud_->clear();
      if (object_viewpoint_locked_) {
        ROS_INFO("[OBJECT_VIEWPOINT_LOCK] resume DINO inspection id=%d after LightGlue release",
            locked_object_id_);
        std::ostringstream debug;
        debug << "event=lock_resume reason=lightglue_released id=" << locked_object_id_;
        publishObjectViewpointDebug(debug.str());
      }
    }
  }

  ROS_WARN("[VERIFIED_APPROACH] state=%d active=%d locked=%d", state, active ? 1 : 0,
      verified_approach_target_locked_ ? 1 : 0);
}

void ExplorationManager::setVerifiedApproachCloud(
    const pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& object_cloud)
{
  if (object_cloud == nullptr || object_cloud->points.empty())
    return;

  verified_approach_cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>(*object_cloud));
}

void ExplorationManager::releaseObjectViewpointLock(
    bool reject_candidate, const std::string& reason)
{
  if (!object_viewpoint_locked_)
    return;

  const int object_id = locked_object_id_;
  const int label = locked_object_label_;
  const double score = locked_object_score_;
  const Vector2d center = locked_object_center_;
  const Vector2d viewpoint = locked_object_viewpoint_;
  const int lock_steps = object_viewpoint_lock_steps_;
  const size_t viewpoint_index = locked_object_viewpoint_index_;
  const size_t viewpoint_count = locked_object_viewpoint_plans_.size();
  const unsigned long evidence_epoch = locked_object_evidence_epoch_;

  if (reject_candidate)
    object_map2d_->rejectSemanticObjectCandidate(object_id);

  object_viewpoint_locked_ = false;
  locked_object_id_ = -1;
  locked_object_label_ = -1;
  object_viewpoint_lock_steps_ = 0;
  locked_object_viewpoint_index_ = 0;
  locked_object_evidence_epoch_ = 0;
  locked_object_score_ = -1.0;
  locked_object_center_.setZero();
  locked_object_viewpoint_.setZero();
  locked_object_viewpoint_plans_.clear();

  ROS_WARN(
      "[OBJECT_VIEWPOINT_LOCK] release reason=%s reject=%d id=%d label=%d score=%.3f "
      "center=(%.2f, %.2f) viewpoint=(%.2f, %.2f) view=%zu/%zu epoch=%lu lock_steps=%d",
      reason.c_str(), reject_candidate ? 1 : 0, object_id, label, score, center.x(), center.y(),
      viewpoint.x(), viewpoint.y(), viewpoint_index + 1, viewpoint_count, evidence_epoch,
      lock_steps);

  std::ostringstream debug;
  debug << std::fixed << std::setprecision(3)
        << "event=lock_release reason=" << reason << " reject=" << (reject_candidate ? 1 : 0)
        << " id=" << object_id << " label=" << label << " crop=" << score
        << " center_x=" << center.x() << " center_y=" << center.y()
        << " target_x=" << viewpoint.x() << " target_y=" << viewpoint.y()
        << " view_index=" << (viewpoint_index + 1) << " view_count=" << viewpoint_count
        << " evidence_epoch=" << evidence_epoch
        << " lock_steps=" << lock_steps;
  publishObjectViewpointDebug(debug.str());
}

double ExplorationManager::getFrontierBaseValue(const Vector2d& frontier_pos)
{
  Vector2i idx;
  sdf_map_->posToIndex(frontier_pos, idx);
  if (!sdf_map_->isInMap(idx))
    return 0.0;

  double value = sdf_map_->value_map_->getBaseValue(idx);
  for (const auto& nbr : allNeighbors(idx, 2)) {
    if (!sdf_map_->isInMap(nbr) || sdf_map_->getInflateOccupancy(nbr) == 1 ||
        sdf_map_->getOccupancy(nbr) == SDFMap2D::OCCUPIED)
      continue;
    value = std::max(value, sdf_map_->value_map_->getBaseValue(nbr));
  }
  return std::max(0.0, std::min(1.0, value));
}

bool ExplorationManager::planVerifiedApproach(const Vector3d& pos)
{
  const Vector2d current_pos(pos.x(), pos.y());
  if (verified_approach_target_locked_) {
    Vector2d replanned_pos;
    vector<Vector2d> replanned_path;
    if (searchFrontierPath(
        current_pos, verified_approach_target_, replanned_pos, replanned_path) &&
        !replanned_path.empty()) {
      ed_->next_pos_ = verified_approach_target_;
      ed_->next_best_path_ = replanned_path;
      ROS_INFO_THROTTLE(0.5,
          "[VERIFIED_APPROACH_LOCK] active target=(%.2f, %.2f) distance=%.2f",
          verified_approach_target_.x(), verified_approach_target_.y(),
          (current_pos - verified_approach_target_).norm());
      std::ostringstream debug;
      debug << std::fixed << std::setprecision(3)
            << "event=route mode=lightglue_lock target_x=" << verified_approach_target_.x()
            << " target_y=" << verified_approach_target_.y()
            << " distance=" << (current_pos - verified_approach_target_).norm();
      publishObjectViewpointDebug(debug.str());
      return true;
    }

    ROS_WARN_THROTTLE(0.5,
        "[VERIFIED_APPROACH_LOCK] fixed target path lost; keep this LightGlue epoch from "
        "switching to another object");
    verified_approach_target_locked_ = false;
    verified_approach_target_failed_ = true;
    verified_approach_target_.setZero();
    return false;
  }

  if (verified_approach_target_failed_)
    return false;

  if (verified_approach_cloud_ != nullptr && !verified_approach_cloud_->points.empty() &&
      searchObjectPath(
          pos, verified_approach_cloud_, ed_->next_pos_, ed_->next_best_path_)) {
    verified_approach_target_locked_ = true;
    verified_approach_target_ = ed_->next_pos_;
    ROS_WARN(
        "[VERIFIED_APPROACH_LOCK] acquire exact LightGlue target=(%.2f, %.2f)",
        ed_->next_pos_.x(), ed_->next_pos_.y());
    std::ostringstream debug;
    debug << std::fixed << std::setprecision(3)
          << "event=lightglue_lock_acquire target_x=" << ed_->next_pos_.x()
          << " target_y=" << ed_->next_pos_.y();
    publishObjectViewpointDebug(debug.str());
    return true;
  }

  ROS_WARN_THROTTLE(0.5,
      "[VERIFIED_APPROACH] active but exact verified cloud is not yet path-ready");
  return false;
}

bool ExplorationManager::isObjectInspectionCompleted(
    const SemanticObjectCandidate& candidate) const
{
  if (candidate.evidence_epoch == 0)
    return false;

  const std::pair<int, int> key(candidate.object_id, candidate.label);
  const auto completed = completed_object_evidence_epochs_.find(key);
  return completed != completed_object_evidence_epochs_.end() &&
         completed->second == candidate.evidence_epoch;
}

void ExplorationManager::markObjectInspectionCompleted(
    int object_id, int label, unsigned long evidence_epoch)
{
  if (object_id < 0 || label < 0 || evidence_epoch == 0)
    return;
  completed_object_evidence_epochs_[std::make_pair(object_id, label)] = evidence_epoch;
}

ExplorationManager::ObjectInspectionFailure* ExplorationManager::findObjectInspectionFailure(
    int object_id, int label, const Vector2d& center)
{
  ObjectInspectionFailure* nearest = nullptr;
  double nearest_distance = std::numeric_limits<double>::infinity();
  for (auto& failure : object_inspection_failures_) {
    if (failure.label != label)
      continue;
    if (std::find(failure.object_ids.begin(), failure.object_ids.end(), object_id) !=
        failure.object_ids.end())
      return &failure;

    const double distance = (failure.center - center).norm();
    if (distance <= ep_->object_viewpoint_failure_spatial_radius_ &&
        distance < nearest_distance) {
      nearest = &failure;
      nearest_distance = distance;
    }
  }
  return nearest;
}

const ExplorationManager::ObjectInspectionFailure* ExplorationManager::findObjectInspectionFailure(
    int object_id, int label, const Vector2d& center) const
{
  const ObjectInspectionFailure* nearest = nullptr;
  double nearest_distance = std::numeric_limits<double>::infinity();
  for (const auto& failure : object_inspection_failures_) {
    if (failure.label != label)
      continue;
    if (std::find(failure.object_ids.begin(), failure.object_ids.end(), object_id) !=
        failure.object_ids.end())
      return &failure;

    const double distance = (failure.center - center).norm();
    if (distance <= ep_->object_viewpoint_failure_spatial_radius_ &&
        distance < nearest_distance) {
      nearest = &failure;
      nearest_distance = distance;
    }
  }
  return nearest;
}

ExplorationManager::ObjectInspectionGate ExplorationManager::getObjectInspectionGate(
    const SemanticObjectCandidate& candidate, unsigned long semantic_sequence,
    bool* stronger_retry) const
{
  if (stronger_retry != nullptr)
    *stronger_retry = false;

  const ObjectInspectionFailure* failure = findObjectInspectionFailure(
      candidate.object_id, candidate.label, candidate.center);
  if (failure == nullptr)
    return ObjectInspectionGate::ELIGIBLE;

  const bool materially_stronger = ep_->object_viewpoint_retry_score_margin_ > 0.0 &&
      candidate.semantic_score >=
          failure->best_failed_score + ep_->object_viewpoint_retry_score_margin_;
  if (materially_stronger) {
    if (stronger_retry != nullptr)
      *stronger_retry = true;
    return ObjectInspectionGate::ELIGIBLE;
  }

  if (failure->failure_count >= ep_->object_viewpoint_max_failed_inspections_)
    return ObjectInspectionGate::EXHAUSTED;
  if (semantic_sequence < failure->cooldown_until_sequence)
    return ObjectInspectionGate::COOLDOWN;
  return ObjectInspectionGate::ELIGIBLE;
}

void ExplorationManager::recordObjectInspectionFailure(int object_id, int label,
    unsigned long evidence_epoch, const Vector2d& center, double score,
    const std::string& reason)
{
  ObjectInspectionFailure* failure = findObjectInspectionFailure(object_id, label, center);
  if (failure == nullptr) {
    ObjectInspectionFailure new_failure;
    new_failure.object_ids.push_back(object_id);
    new_failure.label = label;
    new_failure.center = center;
    object_inspection_failures_.push_back(new_failure);
    failure = &object_inspection_failures_.back();
  }
  else if (std::find(failure->object_ids.begin(), failure->object_ids.end(), object_id) ==
           failure->object_ids.end()) {
    failure->object_ids.push_back(object_id);
  }

  const int previous_failures = failure->failure_count;
  failure->center =
      (failure->center * previous_failures + center) / std::max(1, previous_failures + 1);
  ++failure->failure_count;
  failure->best_failed_score = std::max(failure->best_failed_score, score);
  const unsigned long semantic_sequence = object_map2d_->getSemanticInputDebugInfo().sequence;
  failure->cooldown_until_sequence = semantic_sequence +
      static_cast<unsigned long>(ep_->object_viewpoint_failure_cooldown_sequences_);
  const bool exhausted =
      failure->failure_count >= ep_->object_viewpoint_max_failed_inspections_;

  ROS_WARN(
      "[OBJECT_VIEWPOINT_RETRY] failure reason=%s id=%d label=%d epoch=%lu "
      "attempt=%d/%d score=%.3f best_failed=%.3f center=(%.2f, %.2f) "
      "cooldown_until=%lu exhausted=%d spatial_ids=%zu",
      reason.c_str(), object_id, label, evidence_epoch, failure->failure_count,
      ep_->object_viewpoint_max_failed_inspections_, score, failure->best_failed_score,
      failure->center.x(), failure->center.y(), failure->cooldown_until_sequence,
      exhausted ? 1 : 0, failure->object_ids.size());

  std::ostringstream debug;
  debug << std::fixed << std::setprecision(3)
        << "event=inspection_failure reason=" << reason << " id=" << object_id
        << " label=" << label << " evidence_epoch=" << evidence_epoch
        << " attempt=" << failure->failure_count
        << " max_attempts=" << ep_->object_viewpoint_max_failed_inspections_
        << " score=" << score << " best_failed=" << failure->best_failed_score
        << " center_x=" << failure->center.x() << " center_y=" << failure->center.y()
        << " cooldown_until=" << failure->cooldown_until_sequence
        << " exhausted=" << (exhausted ? 1 : 0)
        << " spatial_ids=" << failure->object_ids.size();
  publishObjectViewpointDebug(debug.str());
}

bool ExplorationManager::activateLockedObjectViewpoint(
    const Vector3d& pos, size_t index, const std::string& reason)
{
  if (!object_viewpoint_locked_ || index >= locked_object_viewpoint_plans_.size())
    return false;

  const ObjectViewpointPlan& plan = locked_object_viewpoint_plans_[index];
  const Vector2d current_pos(pos.x(), pos.y());
  vector<Vector2d> replanned_path;
  path_finder_->reset();
  if (path_finder_->astarSearch(current_pos, plan.viewpoint,
          ep_->object_viewpoint_reach_distance_, ep_->object_viewpoint_astar_time_) !=
          Astar2D::REACH_END ||
      (replanned_path = path_finder_->getPath()).empty()) {
    ROS_WARN(
        "[OBJECT_VIEWPOINT_LOCK] skip unreachable view=%zu/%zu id=%d reason=%s target=(%.2f, %.2f)",
        index + 1, locked_object_viewpoint_plans_.size(), locked_object_id_, reason.c_str(),
        plan.viewpoint.x(), plan.viewpoint.y());
    return false;
  }

  locked_object_viewpoint_index_ = index;
  object_viewpoint_lock_steps_ = 1;
  locked_object_center_ = plan.aim_point;
  locked_object_viewpoint_ = plan.viewpoint;
  ed_->next_pos_ = plan.viewpoint;
  ed_->next_best_path_ = replanned_path;

  ROS_WARN(
      "[OBJECT_VIEWPOINT_LOCK] advance id=%d label=%d epoch=%lu view=%zu/%zu reason=%s "
      "target=(%.2f, %.2f) path_len=%.2f visible=%.2f fov_deg=%.1f",
      locked_object_id_, locked_object_label_, locked_object_evidence_epoch_, index + 1,
      locked_object_viewpoint_plans_.size(), reason.c_str(), plan.viewpoint.x(),
      plan.viewpoint.y(), Astar2D::pathLength(replanned_path), plan.visible_ratio, plan.fov_deg);

  std::ostringstream debug;
  debug << std::fixed << std::setprecision(3)
        << "event=lock_advance mode=dino id=" << locked_object_id_
        << " label=" << locked_object_label_ << " evidence_epoch=" << locked_object_evidence_epoch_
        << " view_index=" << (index + 1)
        << " view_count=" << locked_object_viewpoint_plans_.size()
        << " reason=" << reason << " target_x=" << plan.viewpoint.x()
        << " target_y=" << plan.viewpoint.y()
        << " path_len=" << Astar2D::pathLength(replanned_path)
        << " visible=" << plan.visible_ratio << " fov_deg=" << plan.fov_deg;
  publishObjectViewpointDebug(debug.str());
  return true;
}

bool ExplorationManager::advanceLockedObjectViewpoint(const Vector3d& pos, const std::string& reason)
{
  if (!object_viewpoint_locked_)
    return false;

  for (size_t index = locked_object_viewpoint_index_ + 1;
       index < locked_object_viewpoint_plans_.size(); ++index) {
    if (activateLockedObjectViewpoint(pos, index, reason))
      return true;
  }

  const int object_id = locked_object_id_;
  const int label = locked_object_label_;
  const unsigned long evidence_epoch = locked_object_evidence_epoch_;
  markObjectInspectionCompleted(object_id, label, evidence_epoch);
  recordObjectInspectionFailure(object_id, label, evidence_epoch, locked_object_center_,
      locked_object_score_, reason);

  std::ostringstream debug;
  debug << "event=inspection_complete mode=dino id=" << object_id << " label=" << label
        << " evidence_epoch=" << evidence_epoch << " reason=" << reason
        << " attempted_views=" << locked_object_viewpoint_plans_.size();
  publishObjectViewpointDebug(debug.str());
  releaseObjectViewpointLock(false, "viewpoints_exhausted_" + reason);
  return false;
}

bool ExplorationManager::continueLockedObjectViewpoint(const Vector3d& pos)
{
  if (!object_viewpoint_locked_)
    return false;

  const Vector2d current_pos(pos.x(), pos.y());
  if ((current_pos - locked_object_viewpoint_).norm() <=
      ep_->object_viewpoint_reach_distance_) {
    // Arrival is not evidence that this view was inspected. The action FSM owns the
    // turn-and-observe sequence and is the only place allowed to advance a completed view.
    ROS_INFO_THROTTLE(0.5,
        "[OBJECT_VIEWPOINT_LOCK] arrived at id=%d view=%zu/%zu; hold for visual confirmation",
        locked_object_id_, locked_object_viewpoint_index_ + 1,
        locked_object_viewpoint_plans_.size());
    return true;
  }

  if (object_viewpoint_lock_steps_ >= ep_->object_viewpoint_max_lock_steps_)
    return advanceLockedObjectViewpoint(pos, "timeout");

  vector<Vector2d> replanned_path;
  path_finder_->reset();
  if (path_finder_->astarSearch(current_pos, locked_object_viewpoint_,
          ep_->object_viewpoint_reach_distance_, ep_->object_viewpoint_astar_time_) !=
          Astar2D::REACH_END ||
      (replanned_path = path_finder_->getPath()).empty()) {
    return advanceLockedObjectViewpoint(pos, "path_lost");
  }

  ed_->next_pos_ = locked_object_viewpoint_;
  ed_->next_best_path_ = replanned_path;
  ++object_viewpoint_lock_steps_;
  ROS_INFO_THROTTLE(0.5,
      "[OBJECT_VIEWPOINT_LOCK] active id=%d label=%d epoch=%lu view=%zu/%zu score=%.3f "
      "target=(%.2f, %.2f) distance=%.2f lock_steps=%d",
      locked_object_id_, locked_object_label_, locked_object_evidence_epoch_,
      locked_object_viewpoint_index_ + 1, locked_object_viewpoint_plans_.size(),
      locked_object_score_, locked_object_viewpoint_.x(), locked_object_viewpoint_.y(),
      (current_pos - locked_object_viewpoint_).norm(), object_viewpoint_lock_steps_);
  std::ostringstream debug;
  debug << std::fixed << std::setprecision(3)
        << "event=route mode=dino_lock id=" << locked_object_id_
        << " label=" << locked_object_label_ << " evidence_epoch=" << locked_object_evidence_epoch_
        << " view_index=" << (locked_object_viewpoint_index_ + 1)
        << " view_count=" << locked_object_viewpoint_plans_.size()
        << " crop=" << locked_object_score_ << " target_x=" << locked_object_viewpoint_.x()
        << " target_y=" << locked_object_viewpoint_.y()
        << " distance=" << (current_pos - locked_object_viewpoint_).norm()
        << " lock_steps=" << object_viewpoint_lock_steps_;
  publishObjectViewpointDebug(debug.str());
  return true;
}

bool ExplorationManager::prepareObjectViewpointPlans(const Vector3d& pos,
    const SemanticObjectCandidate& candidate, vector<ObjectViewpointPlan>& plans)
{
  ObjectViewpointSearchStats viewpoint_stats;
  if (searchObjectViewpointPaths(pos, candidate, plans, &viewpoint_stats))
    return true;

  markObjectInspectionCompleted(candidate.object_id, candidate.label, candidate.evidence_epoch);
  recordObjectInspectionFailure(candidate.object_id, candidate.label, candidate.evidence_epoch,
      candidate.center, candidate.semantic_score, "no_valid_viewpoint");
  ROS_WARN(
      "[OBJECT_VIEWPOINT_DECISION] no valid observation viewpoint for id=%d label=%d "
      "epoch=%lu center=(%.2f, %.2f) generated=%d safe=%d visible=%d astar=%d/%d",
      candidate.object_id, candidate.label, candidate.evidence_epoch, candidate.center.x(),
      candidate.center.y(), viewpoint_stats.generated, viewpoint_stats.safe,
      viewpoint_stats.visible, viewpoint_stats.astar_reached, viewpoint_stats.astar_attempted);

  std::ostringstream debug;
  debug << std::fixed << std::setprecision(3)
        << "event=candidate_no_viewpoint id=" << candidate.object_id
        << " label=" << candidate.label << " evidence_epoch=" << candidate.evidence_epoch
        << " streak=" << candidate.observation_count << " crop=" << candidate.semantic_score
        << " center_x=" << candidate.center.x() << " center_y=" << candidate.center.y()
        << " cloud_points=" << viewpoint_stats.cloud_points
        << " generated=" << viewpoint_stats.generated
        << " out_of_map=" << viewpoint_stats.out_of_map
        << " non_free=" << viewpoint_stats.non_free
        << " inflated=" << viewpoint_stats.inflated << " safe=" << viewpoint_stats.safe
        << " ray_samples=" << viewpoint_stats.ray_samples
        << " ray_unknown=" << viewpoint_stats.ray_unknown
        << " ray_known_blocked=" << viewpoint_stats.ray_known_blocked
        << " visibility_rejected=" << viewpoint_stats.visibility_rejected
        << " visible=" << viewpoint_stats.visible
        << " astar_attempted=" << viewpoint_stats.astar_attempted
        << " astar_reached=" << viewpoint_stats.astar_reached
        << " short_path=" << viewpoint_stats.short_path
        << " selected=" << viewpoint_stats.selected;
  publishObjectViewpointDebug(debug.str());
  return false;
}

void ExplorationManager::lockObjectViewpoint(
    const SemanticObjectCandidate& candidate, const vector<ObjectViewpointPlan>& plans)
{
  if (plans.empty())
    return;

  const ObjectViewpointPlan& plan = plans.front();
  object_viewpoint_locked_ = true;
  locked_object_id_ = candidate.object_id;
  locked_object_label_ = candidate.label;
  object_viewpoint_lock_steps_ = 1;
  locked_object_viewpoint_index_ = 0;
  locked_object_evidence_epoch_ = candidate.evidence_epoch;
  locked_object_score_ = candidate.semantic_score;
  locked_object_center_ = plan.aim_point;
  locked_object_viewpoint_ = plan.viewpoint;
  locked_object_viewpoint_plans_ = plans;
  ed_->next_pos_ = plan.viewpoint;
  ed_->next_best_path_ = plan.path;

  ROS_WARN(
      "[OBJECT_VIEWPOINT_LOCK] acquire id=%d label=%d epoch=%lu observations=%d crop=%.3f "
      "view=1/%zu center=(%.2f, %.2f) viewpoint=(%.2f, %.2f) path_len=%.2f "
      "visible=%.2f fov_deg=%.1f standoff=%.2f sampled=%d",
      candidate.object_id, candidate.label, candidate.evidence_epoch, candidate.observation_count,
      candidate.semantic_score, plans.size(), plan.aim_point.x(), plan.aim_point.y(),
      plan.viewpoint.x(), plan.viewpoint.y(), Astar2D::pathLength(plan.path), plan.visible_ratio,
      plan.fov_deg, plan.standoff, plan.generated_candidates);

  std::ostringstream debug;
  debug << std::fixed << std::setprecision(3)
        << "event=lock_acquire mode=dino id=" << candidate.object_id
        << " label=" << candidate.label << " evidence_epoch=" << candidate.evidence_epoch
        << " streak=" << candidate.observation_count << " crop=" << candidate.semantic_score
        << " view_index=1 view_count=" << plans.size()
        << " center_x=" << plan.aim_point.x() << " center_y=" << plan.aim_point.y()
        << " target_x=" << plan.viewpoint.x() << " target_y=" << plan.viewpoint.y()
        << " path_len=" << Astar2D::pathLength(plan.path)
        << " visible=" << plan.visible_ratio << " fov_deg=" << plan.fov_deg
        << " standoff=" << plan.standoff << " sampled=" << plan.generated_candidates;
  publishObjectViewpointDebug(debug.str());
}

int ExplorationManager::planSemanticObjectOrFrontier(const Vector3d& pos, const double& yaw)
{
  (void)yaw;
  if (verified_approach_active_ && planVerifiedApproach(pos))
    return SEARCH_BEST_OBJECT;

  if (!verified_approach_active_ && continueLockedObjectViewpoint(pos))
    return SEARCH_SUSPICIOUS_OBJECT;

  const Vector2d current_pos(pos.x(), pos.y());
  vector<SemanticObjectCandidate> candidates;
  object_map2d_->getSemanticObjectCandidates(candidates);
  const SemanticInputDebugInfo& semantic_debug = object_map2d_->getSemanticInputDebugInfo();
  SemanticObjectCandidate stable_candidate;
  bool has_stable_candidate = false;
  bool has_eligible_candidate = false;
  bool has_completed_eligible_candidate = false;
  int completed_epoch_skips = 0;
  int cooldown_skips = 0;
  int exhausted_skips = 0;
  int stronger_retry_candidates = 0;
  const SemanticObjectCandidate* cooldown_fallback_candidate = nullptr;
  for (const auto& current_candidate : candidates) {
    if (current_candidate.observation_count < ep_->object_viewpoint_min_observations_)
      continue;
    if (!has_stable_candidate) {
      stable_candidate = current_candidate;
      has_stable_candidate = true;
    }
    if (current_candidate.semantic_score < ep_->object_viewpoint_min_score_)
      continue;
    if (isObjectInspectionCompleted(current_candidate)) {
      ++completed_epoch_skips;
      has_completed_eligible_candidate = true;
      continue;
    }
    bool stronger_retry = false;
    const ObjectInspectionGate inspection_gate = getObjectInspectionGate(
        current_candidate, semantic_debug.sequence, &stronger_retry);
    if (inspection_gate == ObjectInspectionGate::COOLDOWN) {
      ++cooldown_skips;
      if (cooldown_fallback_candidate == nullptr)
        cooldown_fallback_candidate = &current_candidate;
      continue;
    }
    if (inspection_gate == ObjectInspectionGate::EXHAUSTED) {
      ++exhausted_skips;
      continue;
    }
    if (stronger_retry)
      ++stronger_retry_candidates;
    has_eligible_candidate = true;
  }

  const SemanticObjectCandidate* top_candidate = candidates.empty() ? nullptr : &candidates.front();
  bool frontier_evaluated = false;
  bool frontier_available = false;
  double frontier_base = 0.0;
  auto publish_decision = [&](const char* reason, const SemanticObjectCandidate* selected) {
    std::ostringstream decision_debug;
    decision_debug << std::fixed << std::setprecision(3)
                   << "event=decision reason=" << reason
                   << " semantic_seq=" << semantic_debug.sequence
                   << " received=" << semantic_debug.received_clouds
                   << " valid=" << semantic_debug.valid_clouds
                   << " matched=" << semantic_debug.matched_clouds
                   << " unmatched=" << semantic_debug.unmatched_clouds
                   << " invalid_label=" << semantic_debug.invalid_label_clouds
                   << " above_gate=" << semantic_debug.above_gate_pairs
                   << " stable_pairs=" << semantic_debug.stable_pairs
                   << " gate=" << semantic_debug.gate_threshold
                   << " map_best_id=" << semantic_debug.best_object_id
                   << " map_best_label=" << semantic_debug.best_label
                   << " map_best_score=" << semantic_debug.best_score
                   << " map_best_recent=" << semantic_debug.best_recent_score
                   << " map_best_streak=" << semantic_debug.best_streak
                   << " candidates=" << candidates.size()
                   << " top_id=" << (top_candidate != nullptr ? top_candidate->object_id : -1)
                   << " top_label=" << (top_candidate != nullptr ? top_candidate->label : -1)
                   << " top_streak="
                   << (top_candidate != nullptr ? top_candidate->observation_count : 0)
                   << " top_crop="
                   << (top_candidate != nullptr ? top_candidate->semantic_score : -1.0)
                   << " eligible_id=" << (selected != nullptr ? selected->object_id : -1)
                   << " eligible_streak="
                   << (selected != nullptr ? selected->observation_count : 0)
                   << " eligible_crop="
                   << (selected != nullptr ? selected->semantic_score : -1.0)
                   << " eligible_epoch="
                   << (selected != nullptr ? selected->evidence_epoch : 0)
                   << " min_crop=" << ep_->object_viewpoint_min_score_
                   << " completed_epoch_skips=" << completed_epoch_skips
                   << " cooldown_skips=" << cooldown_skips
                   << " exhausted_skips=" << exhausted_skips
                   << " stronger_retries=" << stronger_retry_candidates
                   << " frontier_evaluated=" << (frontier_evaluated ? 1 : 0)
                   << " frontier=" << (frontier_available ? 1 : 0)
                   << " frontier_base=" << frontier_base
                   << " inspection_trigger=" << (has_eligible_candidate ? 1 : 0);
    publishObjectViewpointDebug(decision_debug.str());
  };

  bool attempted_viewpoint = false;
  if (has_eligible_candidate) {
    vector<std::pair<int, int>> attempted_object_labels;
    for (const auto& current_candidate : candidates) {
      if (current_candidate.observation_count < ep_->object_viewpoint_min_observations_ ||
          current_candidate.semantic_score < ep_->object_viewpoint_min_score_)
        continue;
      if (isObjectInspectionCompleted(current_candidate))
        continue;
      bool stronger_retry = false;
      if (getObjectInspectionGate(current_candidate, semantic_debug.sequence,
              &stronger_retry) != ObjectInspectionGate::ELIGIBLE)
        continue;

      const std::pair<int, int> object_label(
          current_candidate.object_id, current_candidate.label);
      if (std::find(attempted_object_labels.begin(), attempted_object_labels.end(), object_label) !=
          attempted_object_labels.end())
        continue;

      attempted_object_labels.push_back(object_label);
      attempted_viewpoint = true;
      vector<ObjectViewpointPlan> plans;
      if (prepareObjectViewpointPlans(pos, current_candidate, plans)) {
        publish_decision(stronger_retry ? "try_object_stronger_retry" : "try_object",
            &current_candidate);
        lockObjectViewpoint(current_candidate, plans);
        return SEARCH_SUSPICIOUS_OBJECT;
      }
    }
  }

  // DINO inspection has higher priority than normal frontier exploration. Only pay for TSP and
  // frontier scoring after all currently eligible inspection tasks have been considered.
  Vector2d frontier_pos;
  vector<Vector2d> frontier_path;
  chooseExplorationPolicy(current_pos, ed_->frontier_averages_, frontier_pos, frontier_path);
  frontier_evaluated = true;
  frontier_available = !frontier_path.empty();
  frontier_base = frontier_available ? getFrontierBaseValue(frontier_pos) : 0.0;

  const char* decision_reason = "no_stable_candidate";
  if (exhausted_skips > 0 && !has_eligible_candidate)
    decision_reason = "inspection_retry_exhausted";
  else if (cooldown_skips > 0 && !has_eligible_candidate)
    decision_reason = "inspection_cooldown";
  else if (has_completed_eligible_candidate && !has_eligible_candidate)
    decision_reason = "evidence_epoch_completed";
  else if (has_stable_candidate && !has_eligible_candidate)
    decision_reason = "crop_below_min";
  else if (attempted_viewpoint)
    decision_reason = "no_valid_viewpoint";
  else if (!frontier_available)
    decision_reason = "no_frontier_no_eligible_object";
  publish_decision(decision_reason, has_stable_candidate ? &stable_candidate : nullptr);

  if (frontier_available) {
    ed_->next_pos_ = frontier_pos;
    ed_->next_best_path_ = frontier_path;
    std::ostringstream debug;
    debug << std::fixed << std::setprecision(3)
          << "event=route mode=frontier target_x=" << frontier_pos.x()
          << " target_y=" << frontier_pos.y() << " frontier_base=" << frontier_base
          << " reason=" << decision_reason;
    publishObjectViewpointDebug(debug.str());
    return EXPLORATION;
  }

  Vector2d dormant_pos;
  vector<Vector2d> dormant_path;
  chooseExplorationPolicy(current_pos, ed_->dormant_frontier_averages_, dormant_pos, dormant_path);
  if (!dormant_path.empty()) {
    ed_->next_pos_ = dormant_pos;
    ed_->next_best_path_ = dormant_path;
    ROS_WARN("[OBJECT_VIEWPOINT_DECISION] no active frontier/object target; use dormant frontier");
    return EXPLORATION;
  }

  // A cooldown should make frontier exploration win, not manufacture a premature NO_FRONTIER.
  // If neither active nor dormant frontiers are usable, spend the still-available second
  // inspection attempt instead of ending the episode. Exhausted candidates never use this path.
  if (cooldown_fallback_candidate != nullptr) {
    vector<ObjectViewpointPlan> plans;
    attempted_viewpoint = true;
    if (prepareObjectViewpointPlans(pos, *cooldown_fallback_candidate, plans)) {
      publish_decision("try_object_cooldown_no_frontier", cooldown_fallback_candidate);
      lockObjectViewpoint(*cooldown_fallback_candidate, plans);
      return SEARCH_SUSPICIOUS_OBJECT;
    }
  }

  if (ed_->frontiers_.empty()) {
    ROS_ERROR("No coverable frontier or eligible DINO object viewpoint!!");
    return NO_COVERABLE_FRONTIER;
  }

  ROS_ERROR("No passable frontier or eligible DINO object viewpoint!!");
  return NO_PASSABLE_FRONTIER;
}

void ExplorationManager::chooseExplorationPolicy(Vector2d cur_pos, vector<Vector2d> frontiers,
    Vector2d& next_best_pos, vector<Vector2d>& next_best_path)
{
  switch (ep_->policy_mode_) {
    case ExplorationParam::DISTANCE:
      ROS_WARN("[Exploration Mode] Find Closest Frontier");
      findClosestFrontierPolicy(cur_pos, frontiers, next_best_pos, next_best_path);
      break;

    case ExplorationParam::SEMANTIC:
      ROS_WARN("[Exploration Mode] Find Highest Semantic Value Frontier");
      findHighestSemanticsFrontierPolicy(cur_pos, frontiers, next_best_pos, next_best_path);
      break;

    case ExplorationParam::HYBRID:
      ROS_WARN("[Exploration Mode] Working on Hybrid Mode");
      hybridExplorePolicy(cur_pos, frontiers, next_best_pos, next_best_path);
      break;

    case ExplorationParam::TSP_DIST:
      ROS_WARN("[Exploration Mode] Working on TSP Distance Mode");
      findTSPTourPolicy(cur_pos, frontiers, next_best_pos, next_best_path);
      break;

    default:
      ROS_WARN("[Exploration Mode] Unknown Mode");
      break;
  }
}

void ExplorationManager::hybridExplorePolicy(Vector2d cur_pos, vector<Vector2d> frontiers,
    Vector2d& next_best_pos, vector<Vector2d>& next_best_path)
{
  double std_dev_threshold = ep_->sigma_threshold_;
  double max_to_mean_threshold = ep_->max_to_mean_threshold_;
  vector<SemanticFrontier> sem_frontiers;
  getSortedSemanticFrontiers(cur_pos, frontiers, sem_frontiers);
  if (sem_frontiers.empty())
    return;

  double std_dev, max_to_mean, mean;
  calcSemanticFrontierInfo(sem_frontiers, std_dev, max_to_mean, mean);

  // Decide between exploitation and exploration based on semantic statistics
  if (std_dev > std_dev_threshold && max_to_mean > max_to_mean_threshold) {
    ROS_WARN("Exploit the semantic value (TSP)!!");
    vector<Vector2d> high_sem_frontiers;

    // Select high-value frontiers for TSP optimization
    for (auto sem_frontier : sem_frontiers) {
      double auto_max_to_mean_threshold =
          max(max_to_mean_threshold, ep_->max_to_mean_percentage_ * max_to_mean);
      if (sem_frontier.semantic_value / mean < auto_max_to_mean_threshold)
        break;
      high_sem_frontiers.push_back(sem_frontier.position);
    }
    findTSPTourPolicy(cur_pos, high_sem_frontiers, next_best_pos, next_best_path);
  }
  else {
    ROS_WARN("Explore the environment (Closest)!!");
    findClosestFrontierPolicy(cur_pos, frontiers, next_best_pos, next_best_path);
  }
}

void ExplorationManager::findHighestSemanticsFrontierPolicy(Vector2d cur_pos,
    vector<Vector2d> frontiers, Vector2d& next_best_pos, vector<Vector2d>& next_best_path)
{
  next_best_path.clear();

  // Container for frontier-value pairs for sorting
  vector<pair<Vector2d, double>> frontier_values;

  // Compute semantic value for each frontier
  for (auto frontier : frontiers) {
    Vector2i idx;
    sdf_map_->posToIndex(frontier, idx);
    auto nbrs = allNeighbors(idx, 2);  // 5x5 neighborhood

    // Find maximum semantic value in local neighborhood
    double value = sdf_map_->value_map_->getBaseValue(idx);
    for (auto nbr : nbrs) {
      if (!sdf_map_->isInMap(nbr))
        continue;
      value = max(value, sdf_map_->value_map_->getBaseValue(nbr));
    }

    frontier_values.emplace_back(frontier, value);
  }

  // Sort by semantic value (descending), then by distance (ascending)
  auto compareFrontiers = [&cur_pos](
                              const pair<Vector2d, double>& a, const pair<Vector2d, double>& b) {
    if (fabs(a.second - b.second) > 1e-5) {
      return a.second > b.second;  // Higher semantic value first
    }
    else {
      double dist_a = (a.first - cur_pos).norm();
      double dist_b = (b.first - cur_pos).norm();
      return dist_a < dist_b;  // Closer distance first for tie-breaking
    }
  };

  std::sort(frontier_values.begin(), frontier_values.end(), compareFrontiers);

  // Update frontier list with sorted order
  frontiers.clear();
  for (const auto& fv : frontier_values) {
    frontiers.push_back(fv.first);
  }

  // Select first reachable frontier from sorted list
  for (int i = 0; i < (int)frontiers.size(); i++) {
    std::vector<Eigen::Vector2d> tmp_path;
    Eigen::Vector2d tmp_pos;
    if (!searchFrontierPath(cur_pos, frontiers[i], tmp_pos, tmp_path))
      continue;
    next_best_pos = tmp_pos;
    next_best_path = tmp_path;
    break;
  }
}

void ExplorationManager::findClosestFrontierPolicy(Vector2d cur_pos, vector<Vector2d> frontiers,
    Vector2d& next_best_pos, vector<Vector2d>& next_best_path)
{
  next_best_path.clear();

  // Sort frontiers by Euclidean distance for efficient processing
  std::sort(frontiers.begin(), frontiers.end(), [&cur_pos](const Vector2d& a, const Vector2d& b) {
    return (a - cur_pos).norm() < (b - cur_pos).norm();
  });

  double min_len = std::numeric_limits<double>::max();

  // Find the frontier with shortest actual path length
  for (int i = 0; i < (int)frontiers.size(); i++) {
    // Skip if Euclidean distance already exceeds best path length
    if ((frontiers[i] - cur_pos).norm() >= min_len)
      continue;

    std::vector<Eigen::Vector2d> tmp_path;
    Eigen::Vector2d tmp_pos;

    // Attempt path planning to this frontier
    if (!searchFrontierPath(cur_pos, frontiers[i], tmp_pos, tmp_path))
      continue;

    // Update best solution if this path is shorter
    double len = Astar2D::pathLength(tmp_path);
    if (len < min_len) {
      min_len = len;
      next_best_pos = tmp_pos;
      next_best_path = tmp_path;
    }
  }
}

void ExplorationManager::findTSPTourPolicy(Vector2d cur_pos, vector<Vector2d> frontiers,
    Vector2d& next_best_pos, vector<Vector2d>& next_best_path)
{
  next_best_path.clear();
  vector<Vector2d> filter_frontiers;
  for (auto frontier : frontiers) {
    Vector2d tmp_pos;
    vector<Vector2d> tmp_path;
    if (searchFrontierPath(cur_pos, frontier, tmp_pos, tmp_path))
      filter_frontiers.push_back(frontier);
  }

  vector<int> indices;
  computeATSPTour(cur_pos, filter_frontiers, indices);
  ed_->tsp_tour_.push_back(cur_pos);
  for (auto idx : indices) ed_->tsp_tour_.push_back(filter_frontiers[idx]);

  if (!indices.empty()) {
    for (auto idx : indices) {
      Vector2d next_bext_frontier = filter_frontiers[idx];
      if (searchFrontierPath(cur_pos, next_bext_frontier, next_best_pos, next_best_path))
        break;
    }
  }
}

double ExplorationManager::computePathCost(const Vector2d& pos1, const Vector2d& pos2)
{
  path_finder_->reset();
  if (path_finder_->astarSearch(pos1, pos2, 0.25, 0.002) == Astar2D::REACH_END)
    return Astar2D::pathLength(path_finder_->getPath());
  return 10000.0;
}

void ExplorationManager::computeATSPCostMatrix(
    const Vector2d& cur_pos, const vector<Vector2d>& frontiers, Eigen::MatrixXd& mat)
{
  int dimen = frontiers.size() + 1;
  mat.resize(dimen, dimen);

  // Agent to frontiers
  for (int i = 1; i < dimen; i++) {
    mat(0, i) = computePathCost(cur_pos, frontiers[i - 1]);
    mat(i, 0) = 0;
  }

  // Costs between frontiers
  for (int i = 1; i < dimen; ++i) {
    for (int j = i + 1; j < dimen; ++j) {
      double cost = computePathCost(frontiers[i - 1], frontiers[j - 1]);
      mat(i, j) = cost;
      mat(j, i) = cost;
    }
  }

  // Diag
  for (int i = 0; i < dimen; ++i) {
    mat(i, i) = 100000.0;
  }
}

void ExplorationManager::computeATSPTour(
    const Vector2d& cur_pos, const vector<Vector2d>& frontiers, vector<int>& indices)
{
  indices.clear();
  if (frontiers.empty()) {
    ROS_ERROR("No frontier to compute tsp!");
    return;
  }
  else if (frontiers.size() == 1) {
    indices.push_back(0);
    return;
  }
  /* change ATSP to lhk3 */
  auto t1 = ros::Time::now();

  // Get cost matrix for current state and clusters
  Eigen::MatrixXd cost_mat;
  computeATSPCostMatrix(cur_pos, frontiers, cost_mat);
  const int dimension = cost_mat.rows();

  double mat_time = (ros::Time::now() - t1).toSec();
  t1 = ros::Time::now();

  // Initialize ATSP par file
  // Create problem file
  ofstream file(ep_->tsp_dir_ + "/atsp_tour.atsp");
  file << "NAME : amtsp\n";
  file << "TYPE : ATSP\n";
  file << "DIMENSION : " + to_string(dimension) + "\n";
  file << "EDGE_WEIGHT_TYPE : EXPLICIT\n";
  file << "EDGE_WEIGHT_FORMAT : FULL_MATRIX\n";
  file << "EDGE_WEIGHT_SECTION\n";
  for (int i = 0; i < dimension; ++i) {
    for (int j = 0; j < dimension; ++j) {
      int int_cost = 100 * cost_mat(i, j);
      file << int_cost << " ";
    }
    file << "\n";
  }
  file.close();

  // Create par file
  const int drone_num = 1;
  file.open(ep_->tsp_dir_ + "/atsp_tour.par");
  file << "SPECIAL\n";
  file << "PROBLEM_FILE = " + ep_->tsp_dir_ + "/atsp_tour.atsp\n";
  file << "SALESMEN = " << to_string(drone_num) << "\n";
  file << "MTSP_OBJECTIVE = MINSUM\n";
  file << "RUNS = 1\n";
  file << "TRACE_LEVEL = 0\n";
  file << "TOUR_FILE = " + ep_->tsp_dir_ + "/atsp_tour.tour\n";
  file.close();

  auto par_dir = ep_->tsp_dir_ + "/atsp_tour.atsp";

  lkh_mtsp_solver::SolveMTSP srv;
  srv.request.prob = 1;
  if (!tsp_client_.call(srv)) {
    ROS_ERROR("Fail to solve ATSP.");
    return;
  }

  // Read optimal tour from the tour section of result file
  ifstream res_file(ep_->tsp_dir_ + "/atsp_tour.tour");
  string res;
  while (getline(res_file, res)) {
    // Go to tour section
    if (res.compare("TOUR_SECTION") == 0)
      break;
  }

  // Read path for ATSP formulation
  while (getline(res_file, res)) {
    // Read indices of frontiers in optimal tour
    int id = stoi(res);
    if (id == 1)  // Ignore the current state
      continue;
    if (id == -1)
      break;
    indices.push_back(id - 2);  // Idx of solver-2 == Idx of frontier
  }

  res_file.close();

  // for (auto idx : indices) ROS_WARN("ATSP idx = %d", idx);

  double tsp_time = (ros::Time::now() - t1).toSec();
  ROS_WARN("[ATSP Tour] Cost mat: %lf, TSP: %lf", mat_time, tsp_time);
}

Vector2d ExplorationManager::findNearestObjectPoint(
    const Vector3d& start, const pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& object_cloud)
{
  pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
  kdtree.setInputCloud(object_cloud);
  std::vector<int> pointIdxNKNSearch(1);
  std::vector<float> pointNKNSquaredDistance(1);

  pcl::PointXYZ cur_pt;
  cur_pt.x = start(0);
  cur_pt.y = start(1);
  cur_pt.z = start(2);

  if (kdtree.nearestKSearch(cur_pt, 1, pointIdxNKNSearch, pointNKNSquaredDistance) <= 0) {
    ROS_ERROR("[Bug] No nearest object point found.");
    return Vector2d(-1000.0, -1000.0);  // Error indicator
  }

  int nearest_idx = pointIdxNKNSearch[0];
  auto nearest_point = object_cloud->points[nearest_idx];
  return Vector2d(nearest_point.x, nearest_point.y);
}

bool ExplorationManager::trySearchObjectPathWithDistance(const Vector2d& start2d,
    const Vector2d& object_pose, double distance, double max_search_time,
    Eigen::Vector2d& refined_pos, std::vector<Eigen::Vector2d>& refined_path,
    const std::string& debug_msg)
{
  path_finder_->reset();
  if (path_finder_->astarSearch(start2d, object_pose, distance, max_search_time) ==
      Astar2D::REACH_END) {
    std::vector<Eigen::Vector2d> path = path_finder_->getPath();
    Vector2d tmp_pos(-1000.0, -1000.0);

    // Find valid position along the path (from end to start)
    for (int i = path.size() - 1; i >= 0; i--) {
      if (sdf_map_->getOccupancy(path[i]) != SDFMap2D::OCCUPIED &&
          sdf_map_->getOccupancy(path[i]) != SDFMap2D::UNKNOWN &&
          sdf_map_->getInflateOccupancy(path[i]) != 1) {
        tmp_pos = path[i];
        break;
      }
    }

    // Search path to the valid position
    path_finder_->reset();
    if (path_finder_->astarSearch(start2d, tmp_pos, 0.2, max_search_time) == Astar2D::REACH_END) {
      refined_path = path_finder_->getPath();
      refined_pos = tmp_pos;
      if (!debug_msg.empty()) {
        ROS_WARN("%s", debug_msg.c_str());
      }
      return true;
    }
  }
  return false;
}

bool ExplorationManager::searchObjectPath(const Vector3d& start,
    const pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& object_cloud,
    Eigen::Vector2d& refined_pos, std::vector<Eigen::Vector2d>& refined_path)
{
  const double max_search_time = 0.2;  // Maximum planning time per attempt
  Vector2d start2d = Vector2d(start(0), start(1));

  // Find nearest accessible point in object cloud
  Vector2d object_pose = findNearestObjectPoint(start, object_cloud);
  if (object_pose.x() < -999.0)
    return false;  // Error indicator from findNearestObjectPoint

  // Try different safety distances in order of preference
  const std::vector<double> distances = { 0.5, 0.70, 0.85 };
  const std::vector<std::string> debug_messages = { "I'm going to the object! dist = 0.5m!",
    "I'm going to the object! dist = 0.70m!", "I'm going to the object! dist = 0.85m!" };

  // Attempt path planning with each safety distance
  for (size_t i = 0; i < distances.size(); ++i) {
    if (trySearchObjectPathWithDistance(start2d, object_pose, distances[i], max_search_time,
            refined_pos, refined_path, debug_messages[i])) {
      return true;
    }
  }

  ROS_ERROR("Failed to find object path.");
  return false;
}

bool ExplorationManager::searchObjectViewpointPaths(const Vector3d& start,
    const SemanticObjectCandidate& candidate, vector<ObjectViewpointPlan>& plans,
    ObjectViewpointSearchStats* stats)
{
  ObjectViewpointSearchStats local_stats;
  if (stats == nullptr)
    stats = &local_stats;
  *stats = ObjectViewpointSearchStats();

  plans.clear();
  if (candidate.cloud == nullptr || candidate.cloud->points.empty())
    return false;

  vector<Vector2d> object_points;
  object_points.reserve(candidate.cloud->points.size());
  vector<double> xs, ys;
  xs.reserve(candidate.cloud->points.size());
  ys.reserve(candidate.cloud->points.size());
  for (const auto& point : candidate.cloud->points) {
    if (!std::isfinite(point.x) || !std::isfinite(point.y))
      continue;
    const Vector2d point2d(point.x, point.y);
    if (!sdf_map_->isInMap(point2d))
      continue;
    object_points.push_back(point2d);
    xs.push_back(point2d.x());
    ys.push_back(point2d.y());
  }
  stats->cloud_points = object_points.size();
  if (object_points.size() < 3)
    return false;

  // A median anchor is robust to partial clouds and does not fall into the center of a C-shaped
  // fused cluster as easily as a raw mean.
  std::sort(xs.begin(), xs.end());
  std::sort(ys.begin(), ys.end());
  const size_t mid = xs.size() / 2;
  const Vector2d aim_point(xs[mid], ys[mid]);

  Matrix2d covariance = Matrix2d::Zero();
  for (const auto& point : object_points) {
    const Vector2d delta = point - aim_point;
    covariance += delta * delta.transpose();
  }
  covariance /= static_cast<double>(object_points.size());

  Vector2d short_axis(1.0, 0.0);
  Eigen::SelfAdjointEigenSolver<Matrix2d> eigen_solver(covariance);
  if (eigen_solver.info() == Eigen::Success &&
      eigen_solver.eigenvectors().col(0).norm() > 1e-6) {
    short_axis = eigen_solver.eigenvectors().col(0).normalized();
  }
  const Vector2d long_axis(-short_axis.y(), short_axis.x());

  vector<Vector2d> directions;
  auto add_direction = [&directions](const Vector2d& direction) {
    if (direction.norm() <= 1e-6)
      return;
    const Vector2d normalized = direction.normalized();
    for (const auto& existing : directions) {
      if (existing.dot(normalized) > 0.999)
        return;
    }
    directions.push_back(normalized);
  };

  // Broad-side views are especially useful for sofas. Ring samples keep the method general when
  // the cloud is partial, L-shaped, or one broad side is blocked.
  add_direction(short_axis);
  add_direction(-short_axis);
  add_direction(long_axis);
  add_direction(-long_axis);
  const int direction_count = std::max(4, ep_->object_viewpoint_direction_count_);
  for (int i = 0; i < direction_count; ++i) {
    const double angle = 2.0 * M_PI * static_cast<double>(i) / direction_count;
    add_direction(Vector2d(std::cos(angle), std::sin(angle)));
  }

  struct CandidateEvaluation {
    Vector2d viewpoint;
    double standoff = 0.0;
    double azimuth = 0.0;
    double visible_ratio = 0.0;
    double unknown_ray_ratio = 0.0;
    double fov_deg = 0.0;
    double pre_score = -std::numeric_limits<double>::infinity();
    bool fov_valid = false;
  };
  vector<CandidateEvaluation> viewpoint_candidates;

  const vector<double> standoffs = { ep_->object_viewpoint_standoff_near_,
    ep_->object_viewpoint_standoff_mid_, ep_->object_viewpoint_standoff_far_,
    ep_->object_viewpoint_standoff_extra_ };
  const int visibility_samples = std::max(1, ep_->object_viewpoint_visibility_samples_);
  const double resolution = sdf_map_->getResolution();
  const double target_surface_clearance = std::max(
      2.0 * resolution, ep_->object_viewpoint_target_surface_clearance_);
  const Vector2d start2d(start.x(), start.y());

  for (const auto& outward : directions) {
    Vector2d surface_point = object_points.front();
    double max_projection = -std::numeric_limits<double>::infinity();
    for (const auto& point : object_points) {
      const double projection = (point - aim_point).dot(outward);
      if (projection > max_projection) {
        max_projection = projection;
        surface_point = point;
      }
    }

    // A viewpoint outside the object should only be checked against the face it can actually
    // see. Sampling through the full fused cloud makes the object's own occupied shell look
    // like a wall in front of every back-side target point.
    const double front_surface_band = std::max(2.0 * resolution,
        std::min(target_surface_clearance, 0.40));
    vector<Vector2d> front_surface_points;
    for (const auto& point : object_points) {
      const double projection = (point - aim_point).dot(outward);
      if (max_projection - projection <= front_surface_band)
        front_surface_points.push_back(point);
    }
    if (front_surface_points.empty())
      front_surface_points.push_back(surface_point);
    const int sample_count =
        std::min<int>(visibility_samples, front_surface_points.size());

    for (const double standoff : standoffs) {
      if (standoff <= resolution)
        continue;
      const Vector2d viewpoint = surface_point + outward * standoff;
      ++stats->generated;
      if (!sdf_map_->isInMap(viewpoint)) {
        ++stats->out_of_map;
        continue;
      }
      if (sdf_map_->getOccupancy(viewpoint) != SDFMap2D::FREE) {
        ++stats->non_free;
        continue;
      }
      if (sdf_map_->getInflateOccupancy(viewpoint) == 1) {
        ++stats->inflated;
        continue;
      }
      ++stats->safe;

      const double viewpoint_yaw =
          std::atan2(aim_point.y() - viewpoint.y(), aim_point.x() - viewpoint.x());
      int sampled = 0;
      int visible = 0;
      int unknown_rays = 0;
      vector<double> visible_angles;
      visible_angles.reserve(sample_count);
      for (int sample = 0; sample < sample_count; ++sample) {
        const size_t point_index =
            std::min(front_surface_points.size() - 1,
                static_cast<size_t>(sample) * front_surface_points.size() / sample_count);
        const Vector2d target = front_surface_points[point_index];
        if ((target - viewpoint).norm() <= resolution)
          continue;
        ++sampled;
        ++stats->ray_samples;

        bool blocked = false;
        bool has_unknown = false;
        if (!ray_caster2d_->input(viewpoint, target))
          continue;
        Vector2i ray_idx;
        while (ray_caster2d_->nextId(ray_idx)) {
          if (!sdf_map_->isInMap(ray_idx)) {
            blocked = true;
            ++stats->ray_known_blocked;
            break;
          }
          Vector2d ray_pos;
          sdf_map_->indexToPos(ray_idx, ray_pos);
          // The target object occupies and inflates a shell around its front surface. That shell
          // is the intended ray endpoint, not an occluder; only obstacles farther before it
          // should reject this view.
          if ((ray_pos - target).norm() <= target_surface_clearance)
            continue;
          if (sdf_map_->getOccupancy(ray_idx) == SDFMap2D::UNKNOWN) {
            has_unknown = true;
            continue;
          }
          if (sdf_map_->getOccupancy(ray_idx) == SDFMap2D::OCCUPIED ||
              sdf_map_->getInflateOccupancy(ray_idx) == 1) {
            blocked = true;
            ++stats->ray_known_blocked;
            break;
          }
        }
        if (has_unknown) {
          ++unknown_rays;
          ++stats->ray_unknown;
        }
        if (!blocked) {
          ++visible;
          double relative_angle =
              std::atan2(target.y() - viewpoint.y(), target.x() - viewpoint.x()) - viewpoint_yaw;
          while (relative_angle > M_PI)
            relative_angle -= 2.0 * M_PI;
          while (relative_angle < -M_PI)
            relative_angle += 2.0 * M_PI;
          visible_angles.push_back(relative_angle);
        }
      }

      if (sampled == 0)
        continue;
      const double visible_ratio = static_cast<double>(visible) / sampled;
      const int min_visible_angles = std::max(1, std::min(3, sampled - 1));
      if (visible_ratio < ep_->object_viewpoint_min_visible_ratio_ ||
          static_cast<int>(visible_angles.size()) < min_visible_angles) {
        ++stats->visibility_rejected;
        continue;
      }
      ++stats->visible;

      std::sort(visible_angles.begin(), visible_angles.end());
      const size_t low_index = static_cast<size_t>(
          std::floor(0.10 * static_cast<double>(visible_angles.size() - 1)));
      const size_t high_index = static_cast<size_t>(
          std::ceil(0.90 * static_cast<double>(visible_angles.size() - 1)));
      const double fov_deg =
          (visible_angles[high_index] - visible_angles[low_index]) * 180.0 / M_PI;
      const double fov_sigma = std::max(1.0, ep_->object_viewpoint_fov_sigma_deg_);
      const double fov_error = (fov_deg - ep_->object_viewpoint_ideal_fov_deg_) / fov_sigma;
      const double image_fit = std::exp(-0.5 * fov_error * fov_error);
      const double broadside = std::abs(outward.dot(short_axis));
      const bool fov_valid = fov_deg >= ep_->object_viewpoint_min_fov_deg_ &&
                             fov_deg <= ep_->object_viewpoint_max_fov_deg_;

      CandidateEvaluation evaluation;
      evaluation.viewpoint = viewpoint;
      evaluation.standoff = standoff;
      evaluation.azimuth = std::atan2(viewpoint.y() - aim_point.y(), viewpoint.x() - aim_point.x());
      evaluation.visible_ratio = visible_ratio;
      evaluation.unknown_ray_ratio = static_cast<double>(unknown_rays) / sampled;
      evaluation.fov_deg = fov_deg;
      evaluation.fov_valid = fov_valid;
      evaluation.pre_score = 3.0 * visible_ratio + 2.0 * image_fit + 0.5 * broadside +
                             (fov_valid ? 0.25 : 0.0) -
                             ep_->object_viewpoint_pre_path_distance_weight_ *
                                 (viewpoint - start2d).norm() -
                             ep_->object_viewpoint_unknown_ray_penalty_ *
                                 evaluation.unknown_ray_ratio;
      viewpoint_candidates.push_back(evaluation);
    }
  }

  if (viewpoint_candidates.empty()) {
    ROS_WARN("[OBJECT_VIEWPOINT_PATH] no safe visible viewpoint candidates for object id=%d",
        candidate.object_id);
    return false;
  }

  std::sort(viewpoint_candidates.begin(), viewpoint_candidates.end(),
      [](const CandidateEvaluation& lhs, const CandidateEvaluation& rhs) {
        return lhs.pre_score > rhs.pre_score;
      });

  vector<ObjectViewpointPlan> strict_plans;
  vector<ObjectViewpointPlan> relaxed_plans;
  const int max_astar_candidates = std::min<int>(
      std::max(1, ep_->object_viewpoint_max_astar_candidates_),
      static_cast<int>(viewpoint_candidates.size()));
  const double astar_budget = std::max(0.0, ep_->object_viewpoint_astar_budget_);
  const ros::WallTime astar_start = ros::WallTime::now();
  for (int i = 0; i < max_astar_candidates; ++i) {
    const double elapsed = (ros::WallTime::now() - astar_start).toSec();
    if (astar_budget > 1e-6 && elapsed >= astar_budget)
      break;

    double search_time = std::max(1e-3, ep_->object_viewpoint_astar_time_);
    if (astar_budget > 1e-6)
      search_time = std::min(search_time, astar_budget - elapsed);
    if (search_time <= 1e-6)
      break;

    const auto& evaluation = viewpoint_candidates[i];
    path_finder_->reset();
    ++stats->astar_attempted;
    if (path_finder_->astarSearch(start2d, evaluation.viewpoint,
            ep_->object_viewpoint_reach_distance_, search_time) !=
        Astar2D::REACH_END)
      continue;
    ++stats->astar_reached;

    vector<Vector2d> path = path_finder_->getPath();
    const double path_length = Astar2D::pathLength(path);
    if (path_length < ep_->object_viewpoint_min_path_length_) {
      ++stats->short_path;
      continue;
    }

    ObjectViewpointPlan evaluation_plan;
    evaluation_plan.viewpoint = evaluation.viewpoint;
    evaluation_plan.aim_point = aim_point;
    evaluation_plan.path = path;
    evaluation_plan.score = evaluation.pre_score - 0.15 * path_length;
    evaluation_plan.visible_ratio = evaluation.visible_ratio;
    evaluation_plan.fov_deg = evaluation.fov_deg;
    evaluation_plan.standoff = evaluation.standoff;
    evaluation_plan.azimuth = evaluation.azimuth;
    evaluation_plan.generated_candidates = viewpoint_candidates.size();
    if (evaluation.fov_valid)
      strict_plans.push_back(evaluation_plan);
    else
      relaxed_plans.push_back(evaluation_plan);
  }

  vector<ObjectViewpointPlan>& ranked_plans =
      !strict_plans.empty() ? strict_plans : relaxed_plans;
  if (ranked_plans.empty()) {
    ROS_WARN("[OBJECT_VIEWPOINT_PATH] no reachable fixed viewpoint for object id=%d candidates=%zu",
        candidate.object_id, viewpoint_candidates.size());
    return false;
  }

  std::sort(ranked_plans.begin(), ranked_plans.end(),
      [](const ObjectViewpointPlan& lhs, const ObjectViewpointPlan& rhs) {
        return lhs.score > rhs.score;
      });

  const int max_views = std::max(1, ep_->object_viewpoint_max_views_);
  const double min_azimuth_sep = std::max(0.0, ep_->object_viewpoint_min_azimuth_sep_deg_) *
                                 M_PI / 180.0;
  const double min_position_sep = std::max(0.0, ep_->object_viewpoint_min_position_sep_);
  auto angular_distance = [](double lhs, double rhs) {
    double difference = std::fabs(lhs - rhs);
    while (difference > M_PI)
      difference = std::fabs(difference - 2.0 * M_PI);
    return difference;
  };

  plans.reserve(std::min<int>(max_views, ranked_plans.size()));
  for (const auto& ranked_plan : ranked_plans) {
    bool sufficiently_distinct = true;
    for (const auto& selected_plan : plans) {
      if (angular_distance(ranked_plan.azimuth, selected_plan.azimuth) < min_azimuth_sep ||
          (ranked_plan.viewpoint - selected_plan.viewpoint).norm() < min_position_sep) {
        sufficiently_distinct = false;
        break;
      }
    }
    if (!sufficiently_distinct)
      continue;

    plans.push_back(ranked_plan);
    if (static_cast<int>(plans.size()) >= max_views)
      break;
  }
  stats->selected = plans.size();

  const ObjectViewpointPlan& plan = plans.front();
  ROS_WARN(
      "[OBJECT_VIEWPOINT_PATH] id=%d views=%zu target=(%.2f, %.2f) aim=(%.2f, %.2f) "
      "visible=%.2f fov_deg=%.1f standoff=%.2f candidates=%d",
      candidate.object_id, plans.size(), plan.viewpoint.x(), plan.viewpoint.y(), plan.aim_point.x(),
      plan.aim_point.y(), plan.visible_ratio, plan.fov_deg, plan.standoff,
      plan.generated_candidates);
  return true;
}

void ExplorationManager::getSortedSemanticFrontiers(const Vector2d& cur_pos,
    const vector<Vector2d>& frontiers, vector<SemanticFrontier>& sem_frontiers)
{
  // Filter and sort frontiers based on semantic values and reachability
  sem_frontiers.clear();

  for (auto& frontier : frontiers) {
    SemanticFrontier sem_frontier;
    sem_frontier.position = frontier;

    // Compute semantic value from local neighborhood
    Vector2i idx;
    sdf_map_->posToIndex(frontier, idx);
    auto nbrs = allNeighbors(idx, 2);  // 5x5 grid neighborhood
    double value = sdf_map_->value_map_->getBaseValue(idx);

    // Find maximum semantic value in neighborhood (ignoring occupied cells)
    for (auto& nbr : nbrs) {
      if (!sdf_map_->isInMap(nbr) || sdf_map_->getInflateOccupancy(nbr) == 1 ||
          sdf_map_->getOccupancy(nbr) == SDFMap2D::OCCUPIED)
        continue;
      value = std::max(value, sdf_map_->value_map_->getBaseValue(nbr));
    }
    sem_frontier.semantic_value = value;

    // Validate reachability and compute path cost
    Vector2d tmp_pos;
    vector<Vector2d> tmp_path;
    if (!searchFrontierPath(cur_pos, frontier, tmp_pos, tmp_path)) {
      // Assign high cost penalty for unreachable frontiers
      sem_frontier.path_length = 1000000;
      sem_frontier.path.clear();
    }
    else {
      sem_frontier.path_length = Astar2D::pathLength(tmp_path);
      sem_frontier.path = tmp_path;
    }

    // Only include frontiers with valid paths
    if (!sem_frontier.path.empty())
      sem_frontiers.push_back(sem_frontier);
  }

  // Sort by semantic value (desc) then by path length (asc)
  std::sort(sem_frontiers.begin(), sem_frontiers.end());
}

void ExplorationManager::calcSemanticFrontierInfo(const vector<SemanticFrontier>& sem_frontiers,
    double& std_dev, double& max_to_mean, double& mean, bool if_print)
{
  // Handle empty frontier list
  if (sem_frontiers.empty()) {
    std::cout << "No semantic frontiers available." << std::endl;
    max_to_mean = 1.0;  // Neutral ratio
    std_dev = 0.0;      // No variation
    return;
  }

  // Compute mean and maximum semantic values
  double sum = 0.0;
  double max_value = 0.0;
  for (const auto& frontier : sem_frontiers) {
    sum += frontier.semantic_value;
    max_value = max(max_value, frontier.semantic_value);
  }
  mean = sum / sem_frontiers.size();

  // Compute standard deviation
  double variance_sum = 0.0;
  for (const auto& frontier : sem_frontiers)
    variance_sum += (frontier.semantic_value - mean) * (frontier.semantic_value - mean);

  max_to_mean = max_value / mean;
  std_dev = std::sqrt(variance_sum / sem_frontiers.size());

  // Print summary statistics
  std::cout << "Mean Value: " << std::fixed << std::setprecision(3) << mean;
  std::cout << " , Standard Deviation: " << std::fixed << std::setprecision(3) << std_dev;
  std::cout << " , Max-to-Mean: " << std::fixed << std::setprecision(3) << max_to_mean << std::endl;

  // Print detailed frontier values if requested
  if (if_print) {
    for (const auto& sem_frontier : sem_frontiers)
      std::cout << "Value: " << std::fixed << std::setprecision(3) << sem_frontier.semantic_value
                << std::endl;
  }
}

bool ExplorationManager::planTrajectory(
    const Eigen::VectorXd& start, const Eigen::VectorXd& end, const Vector3d& ctrl)
{
  if (!gcopter_ || !kinoastar_) {
    ROS_WARN_THROTTLE(1.0, "[ExplorationManager] GCopter or KinoAstar not initialized for real-world mode");
    return false;
  }
  
  Eigen::VectorXd goal_state, current_state;
  Vector3d control = ctrl;
  goal_state = end;
  current_state = start;

  // Kinodynamic A* search
  kinoastar_->reset();
  kinoastar_->search(goal_state, current_state, control);
  kinoastar_->getKinoNode();
  
  if (kinoastar_->has_path_) {
    kinoastar_->kinoastarFlatPathPub(kinoastar_->flat_trajs_);
    gcopter_->minco_plan();
    std::vector<Trajectory<7, 3>> final_trajes = gcopter_->final_trajes;
    gcopter_->mincoPathPub(gcopter_->final_trajes, gcopter_->final_singuls);
    return true;
  }
  
  return false;
}

}  // namespace apexnav_planner
