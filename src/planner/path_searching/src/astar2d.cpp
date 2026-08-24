#include <path_searching/astar2d.h>
#include <algorithm>
#include <cmath>
#include <sstream>

using namespace std;
using namespace Eigen;

namespace apexnav_planner {
namespace {

double computeWallPenalty(double obstacle_distance, double preferred_clearance, double penalty_weight)
{
  if (obstacle_distance >= preferred_clearance) {
    return 0.0;
  }

  const double clearance_deficit = preferred_clearance - obstacle_distance;
  return penalty_weight * clearance_deficit * clearance_deficit;
}

}  // namespace

Astar2D::~Astar2D()
{
  for (int i = 0; i < allocate_num_; i++) delete path_node_pool_[i];
}

void Astar2D::init(ros::NodeHandle& nh, const SDFMap2D::Ptr& sdf_map)
{
  nh.param("astar/resolution_astar", resolution_, -1.0);
  nh.param("astar/lambda_heu", lambda_heu_, -1.0);
  nh.param("astar/preferred_clearance", preferred_clearance_, 0.25);
  nh.param("astar/wall_penalty_weight", wall_penalty_weight_, 3.0);
  nh.param("astar/footprint_clearance", footprint_clearance_, -1.0);
  nh.param("astar/start_clearance_grace", start_clearance_grace_, 0.25);
  start_clearance_grace_ = std::max(0.0, start_clearance_grace_);
  if (footprint_clearance_ < 0.0) {
    double length = 0.70;
    double width = 0.30;
    double hard_collision_padding = 0.03;
    nh.param("length", length, length);
    nh.param("width", width, width);
    nh.param("hard_collision_padding", hard_collision_padding, hard_collision_padding);
    const double body_radius = 0.5 * std::hypot(length, width) + hard_collision_padding;
    // The ESDF starts at the already-inflated obstacle boundary.
    footprint_clearance_ = std::max(0.0, body_radius - sdf_map->getObstaclesInflation());
  }
  allocate_num_ = 1000000;

  this->sdf_map_ = sdf_map;

  /* ---------- map params ---------- */
  this->inv_resolution_ = 1.0 / resolution_;
  sdf_map_->getRegion(origin_, map_size_2d_);
  cout << "origin_: " << origin_.transpose() << endl;
  cout << "map size: " << map_size_2d_.transpose() << endl;
  ROS_INFO("[Astar2D] footprint clearance after map inflation: %.3f m", footprint_clearance_);

  path_node_pool_.resize(allocate_num_);
  for (int i = 0; i < allocate_num_; i++) path_node_pool_[i] = new Node2D;
  use_node_num_ = 0;
  iter_num_ = 0;
  early_terminate_cost_ = 0.0;
}

void Astar2D::reset()
{
  open_set_map_.clear();
  close_set_map_.clear();
  path_nodes_.clear();

  std::priority_queue<Node2DPtr, std::vector<Node2DPtr>, NodeComparator2D> empty_queue;
  open_set_.swap(empty_queue);
  for (int i = 0; i < use_node_num_; i++) {
    path_node_pool_[i]->parent = nullptr;
  }
  use_node_num_ = 0;
  iter_num_ = 0;
}

void Astar2D::setResolution(const double& res)
{
  resolution_ = res;
  this->inv_resolution_ = 1.0 / resolution_;
}

int Astar2D::astarSearch(const Eigen::Vector2d& start_pt, const Eigen::Vector2d& end_pt,
    double success_dist, double max_time, int safety_mode)
{
  if (!sdf_map_->isInMap(start_pt)) {
    ROS_WARN("[Astar2D] Reject search: start is outside the map at (%.2f, %.2f).",
        start_pt.x(), start_pt.y());
    return NO_PATH;
  }

  // With a head-mounted depth camera, cells around the body can remain
  // unknown or be inflated by a near-field return even while the stationary
  // robot is physically safe. The original planner allowed a short escape
  // from such a pose. Restrict that exception to this case and this radius;
  // every sample after it still uses the full map and footprint tests.
  const bool start_requires_escape = !checkPointSafety(start_pt, safety_mode, false);
  if (start_requires_escape) {
    ROS_WARN_THROTTLE(1.0,
        "[Astar2D] Start (%.2f, %.2f) is not map-safe (occupancy=%d, inflated=%d); "
        "allowing a bounded %.2f m startup escape.",
        start_pt.x(), start_pt.y(), sdf_map_->getOccupancy(start_pt),
        sdf_map_->getInflateOccupancy(start_pt), start_clearance_grace_);
  }

  int rejected_out_of_map = 0;
  int rejected_occupied = 0;
  int rejected_unknown = 0;
  int rejected_clearance = 0;
  auto logNoPath = [&]() {
    ROS_WARN_THROTTLE(1.0,
        "[Astar2D] No path from (%.2f, %.2f) to (%.2f, %.2f): expanded=%d, "
        "rejected[out=%d, occupied_or_inflated=%d, unknown=%d, clearance=%d].",
        start_pt.x(), start_pt.y(), end_pt.x(), end_pt.y(), iter_num_, rejected_out_of_map,
        rejected_occupied, rejected_unknown, rejected_clearance);
  };

  Node2DPtr cur_node = path_node_pool_[0];
  cur_node->parent = nullptr;
  cur_node->position = start_pt;
  posToIndex(start_pt, cur_node->index);
  cur_node->g_score = 0.0;
  cur_node->f_score = lambda_heu_ * getDiagHeu(cur_node->position, end_pt);

  Eigen::Vector2i end_index;
  posToIndex(end_pt, end_index);

  open_set_.push(cur_node);
  open_set_map_.insert(make_pair(cur_node->index, cur_node));
  use_node_num_ += 1;

  const auto t1 = ros::Time::now();

  /* ---------- search loop ---------- */
  while (!open_set_.empty()) {
    cur_node = open_set_.top();
    bool reach_end =
        abs(cur_node->index(0) - end_index(0)) <= 1 && abs(cur_node->index(1) - end_index(1)) <= 1;
    if ((cur_node->position - end_pt).norm() < success_dist)
      reach_end = true;
    if (reach_end) {
      // A frontier endpoint can be unknown or too close to an obstacle. End
      // at the safe node that reached the success radius instead of appending
      // the unchecked requested endpoint to the path.
      backtrack(cur_node, cur_node->position);
      return REACH_END;
    }

    // Early termination if time up
    if ((ros::Time::now() - t1).toSec() > max_time) {
      early_terminate_cost_ = cur_node->g_score + getDiagHeu(cur_node->position, end_pt);
      logNoPath();
      return NO_PATH;
    }

    open_set_.pop();
    open_set_map_.erase(cur_node->index);
    close_set_map_.insert(make_pair(cur_node->index, 1));
    iter_num_ += 1;

    Eigen::Vector2d cur_pos = cur_node->position;
    Eigen::Vector2d nbr_pos;

    std::vector<Eigen::Vector2d> steps = generateSteps(cur_pos);
    for (auto step : steps) {
      nbr_pos = cur_pos + step;

      // A self-marked starting cell needs a short, bounded escape. Do not use
      // the exception for an already map-safe start, and never permit leaving
      // the map. Once outside the escape radius, all occupancy, unknown, and
      // footprint-clearance checks apply again.
      const Vector2d segment = nbr_pos - cur_pos;
      const double segment_length = segment.norm();
      const int sample_count = std::max(1,
          static_cast<int>(std::ceil(segment_length / std::max(0.01, std::min(0.025, 0.5 * resolution_)))));
      bool safe = true;
      for (int i = 1; i <= sample_count; ++i) {
        const Vector2d sample = cur_pos + segment * (static_cast<double>(i) / sample_count);
        const bool in_startup_escape = start_requires_escape &&
            (sample - start_pt).norm() <= start_clearance_grace_ + 1.0e-6;
        if (!sdf_map_->isInMap(sample)) {
          ++rejected_out_of_map;
          safe = false;
          break;
        }
        if (!in_startup_escape && !checkPointSafety(sample, safety_mode)) {
          const int occupancy = sdf_map_->getOccupancy(sample);
          if (sdf_map_->getInflateOccupancy(sample) == 1 || occupancy == SDFMap2D::OCCUPIED)
            ++rejected_occupied;
          else if (occupancy == SDFMap2D::UNKNOWN && safety_mode == SAFETY_MODE::NORMAL)
            ++rejected_unknown;
          else
            ++rejected_clearance;
          safe = false;
          break;
        }
      }
      if (!safe)
        continue;

      // Check not in close set
      Eigen::Vector2i nbr_idx;
      posToIndex(nbr_pos, nbr_idx);
      if (close_set_map_.find(nbr_idx) != close_set_map_.end())
        continue;

      Node2DPtr neighbor;
      double tmp_g_score = computeTraversalCost(cur_pos, nbr_pos) + cur_node->g_score;
      auto node_iter = open_set_map_.find(nbr_idx);
      if (node_iter == open_set_map_.end()) {
        neighbor = path_node_pool_[use_node_num_];
        use_node_num_ += 1;
        if (use_node_num_ == allocate_num_) {
          cout << "run out of node pool." << endl;
          return NO_PATH;
        }
        neighbor->index = nbr_idx;
        neighbor->position = nbr_pos;
      }
      else if (tmp_g_score < node_iter->second->g_score) {
        neighbor = node_iter->second;
      }
      else
        continue;

      neighbor->parent = cur_node;
      neighbor->g_score = tmp_g_score;
      neighbor->f_score = tmp_g_score + lambda_heu_ * getDiagHeu(nbr_pos, end_pt);
      open_set_.push(neighbor);
      open_set_map_[nbr_idx] = neighbor;
    }
  }
  logNoPath();
  return NO_PATH;
}

std::vector<Eigen::Vector2d> Astar2D::generateSteps(Eigen::Vector2d pos)
{
  vector<Eigen::Vector2d> steps;

  // Normal Astar Step
  // for (double dx = -resolution_; dx <= resolution_ + 1e-3; dx += resolution_)
  //   for (double dy = -resolution_; dy <= resolution_ + 1e-3; dy += resolution_) {
  //     Eigen::Vector2d step;
  //     step << dx, dy;
  //     if (step.norm() < 1e-3)
  //       continue;
  //     steps.push_back(step);
  //   }

  // Habitat-like stepping (12 directions)
  const double step_length = 0.25;
  const double angle_increment = M_PI / 6;

  for (int i = 0; i < 12; ++i) {
    double angle = i * angle_increment;
    Eigen::Vector2d step(step_length * cos(angle), step_length * sin(angle));
    steps.push_back(step);
  }
  return steps;
}

double Astar2D::getDiagHeu(const Eigen::Vector2d& x1, const Eigen::Vector2d& x2)
{
  double dx = fabs(x1(0) - x2(0));
  double dy = fabs(x1(1) - x2(1));
  double tie_breaker = 1.0 + 1e-6 * (dx + dy);
  // Diagonal distance heuristic for 2D
  return tie_breaker * (sqrt(2.0) * min(dx, dy) + abs(dx - dy));
}

double Astar2D::getManhHeu(const Eigen::Vector2d& x1, const Eigen::Vector2d& x2)
{
  double dx = fabs(x1(0) - x2(0));
  double dy = fabs(x1(1) - x2(1));
  double tie_breaker = 1.0 + 1e-6 * (dx + dy);
  // Manhattan distance heuristic for 2D
  return tie_breaker * (dx + dy);
}

double Astar2D::getEuclHeu(const Eigen::Vector2d& x1, const Eigen::Vector2d& x2)
{
  double dx = fabs(x1(0) - x2(0));
  double dy = fabs(x1(1) - x2(1));
  double tie_breaker = 1.0 + 1e-6 * (dx + dy);
  // Euclidean distance heuristic for 2D
  return tie_breaker * (x2 - x1).norm();
}

double Astar2D::computeTraversalCost(const Eigen::Vector2d& from, const Eigen::Vector2d& to)
{
  const double base_cost = (to - from).norm();
  Eigen::Vector2d grad;
  const double obstacle_distance = sdf_map_->getDistWithGrad(to, grad);
  const double wall_penalty = computeWallPenalty(
      obstacle_distance, preferred_clearance_, wall_penalty_weight_);
  return base_cost + wall_penalty;
}

void Astar2D::backtrack(const Node2DPtr& end_node, const Eigen::Vector2d& end)
{
  if ((end - end_node->position).norm() > 1.0e-6)
    path_nodes_.push_back(end);
  path_nodes_.push_back(end_node->position);
  Node2DPtr cur_node = end_node;
  while (cur_node->parent != nullptr) {
    cur_node = cur_node->parent;
    path_nodes_.push_back(cur_node->position);
  }
  reverse(path_nodes_.begin(), path_nodes_.end());
}

std::vector<Eigen::Vector2d> Astar2D::getVisited()
{
  vector<Eigen::Vector2d> visited;
  for (int i = 0; i < use_node_num_; ++i) visited.push_back(path_node_pool_[i]->position);
  return visited;
}

void Astar2D::posToIndex(const Eigen::Vector2d& pt, Eigen::Vector2i& idx)
{
  idx = ((pt - origin_) * inv_resolution_).array().floor().cast<int>();
}

std::vector<Eigen::Vector2d> Astar2D::getPath()
{
  return path_nodes_;
}

double Astar2D::pathLength(const vector<Eigen::Vector2d>& path)
{
  double length = 0.0;
  if (path.size() < 2)
    return length;
  for (int i = 0; i < (int)path.size() - 1; ++i) length += (path[i + 1] - path[i]).norm();
  return length;
}

bool Astar2D::checkPointSafety(
    const Eigen::Vector2d& pos, int safety_mode, bool enforce_footprint_clearance)
{
  // Outside map bounds is always unsafe
  if (!sdf_map_->isInMap(pos))
    return false;

  // EXTREME: allow any position inside the map (bypass occupancy checks)
  if (safety_mode == SAFETY_MODE::EXTREME)
    return true;

  // Occupancy checks
  const auto occ = sdf_map_->getOccupancy(pos);
  // If inflated occupancy marks collision, or cell is definitely occupied -> unsafe
  if (sdf_map_->getInflateOccupancy(pos) == 1 || occ == SDFMap2D::OCCUPIED)
    return false;

  // In NORMAL mode, treat unknown as unsafe. In OPTIMISTIC, unknown is allowed.
  if (occ == SDFMap2D::UNKNOWN && safety_mode == SAFETY_MODE::NORMAL)
    return false;

  if (enforce_footprint_clearance && footprint_clearance_ > 1.0e-6) {
    Eigen::Vector2d grad;
    if (sdf_map_->getDistWithGrad(pos, grad) < footprint_clearance_)
      return false;
  }

  return true;
}

bool Astar2D::isPointSafe(const Eigen::Vector2d& pos, int safety_mode)
{
  return checkPointSafety(pos, safety_mode);
}

bool Astar2D::isSegmentSafe(
    const Eigen::Vector2d& from, const Eigen::Vector2d& to, int safety_mode)
{
  if (!checkPointSafety(from, safety_mode) || !checkPointSafety(to, safety_mode))
    return false;

  const Eigen::Vector2d delta = to - from;
  const double length = delta.norm();
  if (length <= 1.0e-6)
    return true;

  const double sample_spacing = std::max(0.01, std::min(0.025, 0.5 * resolution_));
  const int sample_count = std::max(1, static_cast<int>(std::ceil(length / sample_spacing)));
  for (int i = 1; i < sample_count; ++i) {
    const Eigen::Vector2d sample = from + delta * (static_cast<double>(i) / sample_count);
    if (!checkPointSafety(sample, safety_mode))
      return false;
  }
  return true;
}
}  // namespace apexnav_planner
