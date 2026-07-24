#ifndef _EXPL_DATA_H_
#define _EXPL_DATA_H_

#include <Eigen/Eigen>
#include <iostream>
#include <vector>
#include <trajectory_manager/optimizer.h>

// Undefine uint macro from optimizer.h to avoid conflict with OpenCV
#ifdef uint
#undef uint
#endif

namespace apexnav_planner {

enum FINAL_RESULT { EXPLORE, SEARCH_OBJECT, STUCKING, NO_FRONTIER, REACH_OBJECT };

struct FSMData {
  FSMData()
  {
    trigger_ = false;
    have_odom_ = false;
    have_confidence_ = false;
    have_finished_ = false;
    static_state_ = true;
    state_str_ = { "INIT", "WAIT_TRIGGER", "PLAN_ACTION", "WAIT_ACTION_FINISH", "PUB_ACTION",
      "FINISH" };

    odom_pos_ = Eigen::Vector3d::Zero();
    odom_vel_ = Eigen::Vector3d::Zero();
    odom_omega_ = Eigen::Vector3d::Zero();
    odom_orient_ = Eigen::Quaterniond::Identity();
    odom_yaw_ = 0.0;
    start_pt_ = Eigen::Vector3d::Zero();
    start_vel_ = Eigen::Vector3d::Zero();
    start_yaw_ = Eigen::Vector3d::Zero();
    last_start_pos_ = Eigen::Vector3d(-100, -100, -100);
    last_next_pos_ = Eigen::Vector2d(-100, -100);
    newest_action_ = -1;
    init_action_count_ = 0;
    no_frontier_count_ = 0;
    stucking_action_count_ = 0;
    stucking_next_pos_count_ = 0;
    traveled_path_.clear();

    final_result_ = -1;
    replan_flag_ = true;
    dormant_frontier_flag_ = false;
    escape_stucking_flag_ = false;
    escape_stucking_count_ = 0;
    stucking_points_.clear();

    local_pos_ = Eigen::Vector2d(0, 0);
    mast3r_hint_active_ = false;
    mast3r_allow_stop_ = false;
    mast3r_yaw_error_deg_ = 0.0;
    mast3r_forward_error_ = 0.0;
    mast3r_lateral_error_ = 0.0;
    mast3r_transl_error_ = 0.0;
    mast3r_depth_error_ = 0.0;
    mast3r_target_pos_ = Eigen::Vector2d(0, 0);
    mast3r_target_yaw_ = 0.0;
    mast3r_goal_active_ = false;
    mast3r_path_active_ = false;
    mast3r_segment_target_pos_ = Eigen::Vector2d::Zero();
    mast3r_segment_target_valid_ = false;
    mast3r_segment_count_ = 0;
    mast3r_goal_age_ = 0;
    mast3r_last_pos_ = Eigen::Vector2d::Zero();
    mast3r_last_pos_valid_ = false;
    mast3r_forward_blocked_count_ = 0;
    mast3r_path_replan_count_ = 0;
    instance_stop_gate_enabled_ = false;
    object_viewpoint_scan_target_valid_ = false;
    object_viewpoint_scan_phase_ = 0;
    object_viewpoint_scan_target_ = Eigen::Vector2d::Zero();
    object_viewpoint_forward_blocked_count_ = 0;
    object_viewpoint_alignment_steps_ = 0;
    object_viewpoint_last_alignment_action_ = -1;
  }
  // FSM data
  bool trigger_, have_odom_, have_confidence_;
  bool have_finished_;
  std::vector<string> state_str_;
  std::vector<Eigen::Vector2d> traveled_path_;

  // odometry state
  Eigen::Vector3d odom_pos_, odom_vel_, odom_omega_;
  Eigen::Quaterniond odom_orient_;
  double odom_yaw_;
  bool static_state_;  // Track if robot is static or moving

  Eigen::Vector3d start_pt_, start_vel_, start_yaw_;
  Eigen::Vector3d last_start_pos_;
  Eigen::Vector2d last_next_pos_;
  int newest_action_;
  int init_action_count_;
  int no_frontier_count_;
  int stucking_action_count_;
  int stucking_next_pos_count_;

  int final_result_;
  bool replan_flag_, dormant_frontier_flag_;
  bool escape_stucking_flag_;
  int escape_stucking_count_;
  Eigen::Vector2d escape_stucking_pos_;
  double escape_stucking_yaw_;
  std::vector<Eigen::Vector3d> stucking_points_;

  Eigen::Vector2d local_pos_;
  bool mast3r_hint_active_;
  bool mast3r_allow_stop_;
  double mast3r_yaw_error_deg_;
  double mast3r_forward_error_;
  double mast3r_lateral_error_;
  double mast3r_transl_error_;
  double mast3r_depth_error_;
  Eigen::Vector2d mast3r_target_pos_;
  double mast3r_target_yaw_;
  bool mast3r_goal_active_;
  bool mast3r_path_active_;
  Eigen::Vector2d mast3r_segment_target_pos_;
  bool mast3r_segment_target_valid_;
  int mast3r_segment_count_;
  int mast3r_goal_age_;
  Eigen::Vector2d mast3r_last_pos_;
  bool mast3r_last_pos_valid_;
  int mast3r_forward_blocked_count_;
  int mast3r_path_replan_count_;
  bool instance_stop_gate_enabled_;
  // A locked DINO observation site gets a side-view frame and a re-aligned frame before
  // the FSM is allowed to consume a precomputed fallback viewpoint.
  bool object_viewpoint_scan_target_valid_;
  int object_viewpoint_scan_phase_;
  Eigen::Vector2d object_viewpoint_scan_target_;
  int object_viewpoint_forward_blocked_count_;
  int object_viewpoint_alignment_steps_;
  int object_viewpoint_last_alignment_action_;
  LocalTrajectory newest_traj_;  // Store latest planned trajectory
};

struct FSMParam {
  FSMParam()
  {
    vis_scale_ = 0.1;
    replan_time_ = 0.2;
    replan_traj_end_threshold_ = 1.0;
    replan_frontier_change_delay_ = 0.5;
    replan_timeout_ = 2.0;

    const double step_length = 0.25;
    const double angle_increment = M_PI / 6;
    action_steps_.clear();
    for (int i = 0; i < 12; ++i) {
      double angle = i * angle_increment;
      Eigen::Vector2d step(step_length * cos(angle), step_length * sin(angle));
      action_steps_.push_back(step);
    }
  }
  double vis_scale_;
  std::vector<Eigen::Vector2d> action_steps_;
  // replan timing parameters (loaded from ros params in ExplorationFSM::init)
  double replan_time_;
  double replan_traj_end_threshold_;
  double replan_frontier_change_delay_;
  double replan_timeout_;
};

struct ExplorationData {
  ExplorationData()
  {
    frontiers_.clear();
    frontier_averages_.clear();
    dormant_frontiers_.clear();
    dormant_frontier_averages_.clear();
    objects_.clear();
    object_averages_.clear();
    object_labels_.clear();
    next_pos_ = Eigen::Vector2d(0, 0);
    next_best_path_.clear();
    tsp_tour_.clear();
  }
  std::vector<std::vector<Eigen::Vector2d>> frontiers_, dormant_frontiers_;
  std::vector<Eigen::Vector2d> frontier_averages_, dormant_frontier_averages_;
  std::vector<std::vector<Eigen::Vector2d>> objects_;
  std::vector<Eigen::Vector2d> object_averages_;
  std::vector<int> object_labels_;
  Eigen::Vector2d next_pos_;
  Eigen::Vector2d next_local_pos_;  // Local target position along path
  std::vector<Eigen::Vector2d> next_best_path_;
  std::vector<Eigen::Vector2d> tsp_tour_;
};

struct ExplorationParam {
  enum POLICY_MODE { DISTANCE, SEMANTIC, HYBRID, TSP_DIST };
  // params
  int policy_mode_;
  double sigma_threshold_, max_to_mean_threshold_, max_to_mean_percentage_;
  bool object_viewpoint_enabled_;
  double object_viewpoint_min_score_;
  double object_viewpoint_standoff_near_;
  double object_viewpoint_standoff_mid_;
  double object_viewpoint_standoff_far_;
  double object_viewpoint_standoff_extra_;
  double object_viewpoint_reach_distance_;
  double object_viewpoint_min_visible_ratio_;
  double object_viewpoint_min_fov_deg_;
  double object_viewpoint_max_fov_deg_;
  double object_viewpoint_ideal_fov_deg_;
  double object_viewpoint_fov_sigma_deg_;
  double object_viewpoint_astar_time_;
  double object_viewpoint_astar_budget_;
  double object_viewpoint_min_path_length_;
  double object_viewpoint_min_azimuth_sep_deg_;
  double object_viewpoint_min_position_sep_;
  double object_viewpoint_pre_path_distance_weight_;
  double object_viewpoint_unknown_ray_penalty_;
  double object_viewpoint_target_surface_clearance_;
  int object_viewpoint_min_observations_;
  int object_viewpoint_max_lock_steps_;
  int object_viewpoint_direction_count_;
  int object_viewpoint_visibility_samples_;
  int object_viewpoint_max_astar_candidates_;
  int object_viewpoint_max_views_;
  int object_viewpoint_max_failed_inspections_;
  int object_viewpoint_failure_cooldown_sequences_;
  double object_viewpoint_retry_score_margin_;
  double object_viewpoint_failure_spatial_radius_;
  std::string tsp_dir_;
};

}  // namespace apexnav_planner

#endif
