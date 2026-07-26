/**
 * @file value_map2d.cpp
 * @brief Implementation of semantic value mapping system with confidence-weighted ITM score fusion
 *
 * This file implements the ValueMap class which provides semantic value mapping capabilities
 * for autonomous navigation systems. The implementation focuses on confidence-weighted fusion
 * of ITM (Image-Text Matching) scores using field-of-view based confidence modeling.
 *
 * Reference paper "VLFM: Vision-Language Frontier Maps for Zero-Shot Semantic Navigation"
 *
 * @author Zager-Zhang
 */

#include <plan_env/value_map2d.h>
#include <algorithm>
#include <cmath>

namespace apexnav_planner {
ValueMap::ValueMap(SDFMap2D* sdf_map, ros::NodeHandle& nh)
{
  this->sdf_map_ = sdf_map;
  int voxel_num = sdf_map_->getVoxelNum();
  value_buffer_ = vector<double>(voxel_num, 0.0);
  object_value_buffer_ = vector<double>(voxel_num, 0.0);
  confidence_buffer_ = vector<double>(voxel_num, 0.0);

  nh.param("semantic/object_update_radius", object_value_update_radius_, 0.18);
}

void ValueMap::updateValueMap(const Vector2d& sensor_pos, const double& sensor_yaw,
    const vector<Vector2i>& free_grids, const double& itm_score)
{
  for (const auto& grid : free_grids) {
    Vector2d pos;
    sdf_map_->indexToPos(grid, pos);
    int adr = sdf_map_->toAddress(grid);

    // Calculate FOV-based confidence for current observation
    double now_confidence = getFovConfidence(sensor_pos, sensor_yaw, pos);
    if (now_confidence <= 1e-6)
      continue;
    double now_value = itm_score;

    // Retrieve existing confidence and value
    double last_confidence = confidence_buffer_[adr];
    double last_value = value_buffer_[adr];

    // Apply confidence-weighted fusion with quadratic confidence combination
    confidence_buffer_[adr] =
        (now_confidence * now_confidence + last_confidence * last_confidence) /
        (now_confidence + last_confidence);
    value_buffer_[adr] = (now_confidence * now_value + last_confidence * last_value) /
                         (now_confidence + last_confidence);
  }
}

void ValueMap::decayObjectEvidence(const double& dt, const double& decay_tau)
{
  if (dt <= 1e-6 || decay_tau <= 1e-6)
    return;

  const double decay = std::exp(-dt / decay_tau);
  for (auto& value : object_value_buffer_)
    value *= decay;
}

void ValueMap::updateObjectFovEvidence(const Vector2d& sensor_pos,
    const vector<Vector2i>& free_grids, const Vector2d& object_pos, const double& score,
    const double& score_weight, const double& angle_sigma, const double& range_sigma,
    const double& max_angle, const double& max_range_error)
{
  if (score < 0.0 || score_weight <= 1e-6 || free_grids.empty() || !sdf_map_->isInMap(object_pos))
    return;

  const Vector2d object_delta = object_pos - sensor_pos;
  const double object_range = object_delta.norm();
  if (object_range <= sdf_map_->getResolution())
    return;

  const double object_bearing = std::atan2(object_delta.y(), object_delta.x());
  const double clamped_score = std::max(0.0, std::min(1.0, score));
  const double clamped_weight = std::max(0.0, std::min(1.0, score_weight));
  const double safe_angle_sigma = std::max(1e-3, angle_sigma);
  const double safe_range_sigma = std::max(1e-3, range_sigma);

  for (const auto& grid : free_grids) {
    if (!sdf_map_->isInMap(grid))
      continue;

    Vector2d cell_pos;
    sdf_map_->indexToPos(grid, cell_pos);
    const Vector2d cell_delta = cell_pos - sensor_pos;
    const double range = cell_delta.norm();
    if (range <= sdf_map_->getResolution())
      continue;

    const double angle_error = normalizeAngle(std::atan2(cell_delta.y(), cell_delta.x()) - object_bearing);
    if (std::abs(angle_error) > max_angle)
      continue;

    if (std::abs(range - object_range) > max_range_error)
      continue;
    const double range_error = range - object_range;

    const double angular_weight = std::exp(
        -0.5 * angle_error * angle_error / (safe_angle_sigma * safe_angle_sigma));
    const double range_weight = std::exp(
        -0.5 * range_error * range_error / (safe_range_sigma * safe_range_sigma));
    const double boost = clamped_weight * clamped_score * angular_weight * range_weight;
    if (boost <= 1e-6)
      continue;

    int adr = sdf_map_->toAddress(grid);
    object_value_buffer_[adr] = std::max(object_value_buffer_[adr], std::min(1.0, boost));
  }
}

void ValueMap::updateObjectEvidence(
    const vector<Vector2d>& object_points, const double& score, const double& score_weight)
{
  if (score < 0.0 || score_weight <= 1e-6 || object_points.empty())
    return;

  const double clamped_score = std::max(0.0, std::min(1.0, score));
  const double clamped_score_weight = std::max(0.0, std::min(1.0, score_weight));
  const double radius = std::max(object_value_update_radius_, sdf_map_->getResolution());
  const int radius_step = std::max(0, static_cast<int>(std::ceil(radius / sdf_map_->getResolution())));

  for (const auto& pt_pos : object_points) {
    if (!sdf_map_->isInMap(pt_pos))
      continue;

    Vector2i center_idx;
    sdf_map_->posToIndex(pt_pos, center_idx);

    for (int dx = -radius_step; dx <= radius_step; ++dx) {
      for (int dy = -radius_step; dy <= radius_step; ++dy) {
        Vector2i idx = center_idx + Vector2i(dx, dy);
        if (!sdf_map_->isInMap(idx))
          continue;

        Vector2d cell_pos;
        sdf_map_->indexToPos(idx, cell_pos);
        double dist = (cell_pos - pt_pos).norm();
        if (dist > radius)
          continue;

        double spatial_weight = 1.0 - dist / radius;
        double now_confidence = std::max(0.05, spatial_weight) * clamped_score_weight;
        double now_value = std::min(1.0, clamped_score * clamped_score_weight);
        int adr = sdf_map_->toAddress(idx);

        object_value_buffer_[adr] = std::max(object_value_buffer_[adr],
            std::min(1.0, now_value * now_confidence));
      }
    }
  }
}

double ValueMap::getFovConfidence(
    const Vector2d& sensor_pos, const double& sensor_yaw, const Vector2d& pt_pos)
{
  // Calculate relative position vector from sensor to target point
  Vector2d rel_pos = pt_pos - sensor_pos;
  double angle_to_point = atan2(rel_pos(1), rel_pos(0));

  // Normalize angles to [-π, π] range for consistent angular arithmetic
  double normalized_sensor_yaw = normalizeAngle(sensor_yaw);
  double normalized_angle_to_point = normalizeAngle(angle_to_point);
  double relative_angle = normalizeAngle(normalized_angle_to_point - normalized_sensor_yaw);

  // Apply cosine-squared FOV confidence model
  // FOV angle: 79° total field of view (typical RGB camera)
  double fov_angle = 79.0 * M_PI / 180.0;
  double value = std::cos(relative_angle / (fov_angle / 2) * (M_PI / 2));
  return value * value;  // Square for stronger center weighting
}

}  // namespace apexnav_planner
