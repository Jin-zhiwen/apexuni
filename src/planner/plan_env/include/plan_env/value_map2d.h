#ifndef _VALUE_MAP_H_
#define _VALUE_MAP_H_

#include <ros/ros.h>
#include <Eigen/Eigen>
#include <algorithm>
#include <vector>

#include <plan_env/sdf_map2d.h>

using Eigen::Vector2d;
using Eigen::Vector2i;
using Eigen::Vector3d;
using std::shared_ptr;
using std::unique_ptr;
using std::vector;

namespace apexnav_planner {
class SDFMap2D;

class ValueMap {
public:
  ValueMap(SDFMap2D* sdf_map, ros::NodeHandle& nh);
  ~ValueMap(){};

  void updateValueMap(const Vector2d& sensor_pos, const double& sensor_yaw,
      const vector<Vector2i>& free_grids, const double& itm_score);
  void decayObjectEvidence(const double& dt, const double& decay_tau);
  void updateObjectFovEvidence(const Vector2d& sensor_pos, const vector<Vector2i>& free_grids,
      const Vector2d& object_pos, const double& score, const double& score_weight,
      const double& angle_sigma, const double& range_sigma, const double& max_angle,
      const double& max_range_error);
  void updateObjectEvidence(const vector<Vector2d>& object_points, const double& score,
      const double& score_weight = 1.0);
  double getValue(const Vector2d& pos);
  double getValue(const Vector2i& idx);
  double getConfidence(const Vector2d& pos);
  double getConfidence(const Vector2i& idx);

private:
  double getFovConfidence(
      const Vector2d& sensor_pos, const double& sensor_yaw, const Vector2d& pt_pos);
  double fuseBaseAndObjectValue(const double& base_value, const double& object_value);
  double normalizeAngle(double angle);

  vector<double> value_buffer_;  // Grid-based semantic value storage
  vector<double> object_value_buffer_;  // Decaying object/crop FOV evidence storage
  vector<double> confidence_buffer_;  // Grid-based confidence storage for weighted fusion

  double object_value_update_radius_;

  // Utils
  SDFMap2D* sdf_map_;
};

inline double ValueMap::normalizeAngle(double angle)
{
  while (angle > M_PI) angle -= 2.0 * M_PI;
  while (angle < -M_PI) angle += 2.0 * M_PI;
  return angle;
}

inline double ValueMap::getConfidence(const Vector2d& pos)
{
  Vector2i idx;
  sdf_map_->posToIndex(pos, idx);
  return getConfidence(idx);
}

inline double ValueMap::getConfidence(const Vector2i& idx)
{
  int adr = sdf_map_->toAddress(idx);
  return confidence_buffer_[adr];
}

inline double ValueMap::getValue(const Vector2d& pos)
{
  Vector2i idx;
  sdf_map_->posToIndex(pos, idx);
  return getValue(idx);
}

inline double ValueMap::getValue(const Vector2i& idx)
{
  int adr = sdf_map_->toAddress(idx);
  return fuseBaseAndObjectValue(value_buffer_[adr], object_value_buffer_[adr]);
}

inline double ValueMap::fuseBaseAndObjectValue(const double& base_value, const double& object_value)
{
  const double clamped_base = std::max(0.0, std::min(1.0, base_value));
  const double clamped_object = std::max(0.0, std::min(1.0, object_value));
  return clamped_base + clamped_object * (1.0 - clamped_base);
}

}  // namespace apexnav_planner
#endif
