#include <ros/ros.h>
#include <gcopter/trajectory.hpp>
#include <trajectory_manager/PolyTraj.h>
#include <Eigen/Dense>
#include <geometry_msgs/Twist.h>
#include <nav_msgs/Odometry.h>
#include <visualization_msgs/Marker.h>
#include <std_msgs/Empty.h>
#include <std_msgs/Float32.h>
#include "controller/mpc.h"

using namespace std;
using namespace Eigen;
// PIDController removed — using MPC (and simple P for small-angle rotation) instead

class TrajectoryServer {
public:
  TrajectoryServer(ros::NodeHandle& nh)
  {
    nh_ = nh;
    receive_traj_ = false;
    have_odom_ = false;
    has_target_angle_ = false;
    target_yaw_ = 0.0;
    bool need_init;
    nh.param("need_init", need_init, false);
    nh.param("init_rotation_omega", init_rotation_omega_, M_PI / 3.0);
    nh.param("init_odom_warmup", init_odom_warmup_, 3.0);
    nh.param("init_rotation_timeout", init_rotation_timeout_, 10.0);
    nh.param("init_odom_timeout", init_odom_timeout_, 1.5);
    nh.param("init_no_progress_timeout", init_no_progress_timeout_, 1.5);
    nh.param("max_correction_vel", max_correction_vel_, 0.6);
    nh.param("max_correction_omega", max_correction_omega_, 1.2);
    nh.param("rotation_tolerance", rotation_tolerance_, 0.10);
    nh.param("min_rotation_omega", min_rotation_omega_, 0.25);
    nh.param("tracking_slowdown_error", tracking_slowdown_error_, 0.20);
    nh.param("tracking_stop_error", tracking_stop_error_, 0.55);
    nh.param("tracking_min_speed_scale", tracking_min_speed_scale_, 0.25);
    nh.param("tracking_min_effective_vx", tracking_min_effective_vx_, 0.22);
    traj_sub_ = nh_.subscribe("trajectory", 10, &TrajectoryServer::polyTrajCallback, this);
    odom_sub_ = nh_.subscribe("odometry", 10, &TrajectoryServer::odometryCallback, this);
    stop_sub_ = nh_.subscribe("/traj_server/stop", 10, &TrajectoryServer::stopCallback, this);
    target_angle_sub_ = nh_.subscribe(
        "/traj_server/target_angle", 10, &TrajectoryServer::targetAngleCallback, this);
    robot_marker_pub_ = nh.advertise<visualization_msgs::Marker>("/robot", 10);
    vel_cmd_pub_ = nh_.advertise<geometry_msgs::Twist>("cmd_vel", 10);
    traj_vis_pub_ = nh_.advertise<visualization_msgs::Marker>("/travel_traj", 10);
    current_desire_pub_ = nh_.advertise<geometry_msgs::Pose>("/current_desire", 10);
    vis_timer_ = nh_.createTimer(ros::Duration(0.20), &TrajectoryServer::visCallback, this);
    cmd_timer_ = nh_.createTimer(ros::Duration(0.02), &TrajectoryServer::cmdCallBack, this);
    init_cmd_timer_ =
        nh_.createTimer(ros::Duration(0.1), &TrajectoryServer::initCmdCallback, this, false, false);
    std::cout << "[traj_server] TrajectoryServer initialized, waiting for messages..." << std::endl;

    nh.param("mpc/predict_steps", mpc_N_, -1);
    nh.param("mpc/dt", mpc_dt_, -1.0);
    if (mpc_N_ <= 0 || mpc_dt_ <= 0.0) {
      ROS_ERROR("[traj_server] Wrong MPC parameters!");
      return;
    }
    mpc_controller_.reset(new MPC);
    mpc_controller_->init(nh_);
    xref_.resize(mpc_N_);

    if (need_init) {
      init_state_ = 0;
      init_warmup_started_ = false;
      init_rotation_started_ = false;
      rotation_accum_ = 0.0;
      last_odom_yaw_ = 0.0;
      init_cmd_timer_.start();
    }
  }

  void initCmdCallback(const ros::TimerEvent& event)
  {
    geometry_msgs::Twist twist_msg;
    switch (init_state_) {
      case 0: {
        // Prefer odom-based rotation stop: accumulate yaw change from odom
        if (have_odom_) {
          const ros::WallTime now_wall = ros::WallTime::now();
          if ((now_wall - last_odom_wall_time_).toSec() > init_odom_timeout_) {
            abortInitialRotation("VIO odometry became stale");
            return;
          }

          if (!init_rotation_started_) {
            if (!init_warmup_started_) {
              init_warmup_started_ = true;
              init_warmup_start_wall_time_ = now_wall;
            }

            const double warmup_elapsed = (now_wall - init_warmup_start_wall_time_).toSec();
            if (warmup_elapsed < init_odom_warmup_) {
              publishZeroVelocity();
              ROS_INFO_THROTTLE(1.0,
                  "[traj_server] Holding still for VIO warmup: %.1f / %.1f s",
                  warmup_elapsed, init_odom_warmup_);
              return;
            }

            last_odom_yaw_ = odom_yaw_;
            rotation_accum_ = 0.0;
            init_rotation_started_ = true;
            init_rotation_start_wall_time_ = now_wall;
            init_last_progress_wall_time_ = now_wall;
          }

          if ((now_wall - init_rotation_start_wall_time_).toSec() > init_rotation_timeout_) {
            abortInitialRotation("initial scan exceeded its time limit");
            return;
          }

          // accumulate yaw change using shortest-angle difference
          double delta = atan2(sin(odom_yaw_ - last_odom_yaw_), cos(odom_yaw_ - last_odom_yaw_));
          rotation_accum_ += fabs(delta);
          last_odom_yaw_ = odom_yaw_;
          if (fabs(delta) >= 0.005) {
            init_last_progress_wall_time_ = now_wall;
          }
          else if ((now_wall - init_last_progress_wall_time_).toSec() >
                   init_no_progress_timeout_) {
            abortInitialRotation("VIO yaw stopped making progress");
            return;
          }

          // Stop after approximately one full rotation
          if (rotation_accum_ >= 2.0 * M_PI - 0.05) {
            twist_msg.angular.z = 0.0;
            vel_cmd_pub_.publish(twist_msg);
            init_state_++;
          }
          else {
            // Mapping density is controlled by Kimera's bounded keyframe
            // interval, so the original one-revolution speed can be used.
            twist_msg.angular.z = init_rotation_omega_;
            vel_cmd_pub_.publish(twist_msg);
          }
        }
        else {
          // Waiting for odom: do not start rotation until odom is available.
          ROS_WARN_THROTTLE(5, "[traj_server] Waiting for odom to start init rotation.");
          return;
        }
      } break;

      case 1: {
        twist_msg.linear.x = 0.0;
        vel_cmd_pub_.publish(twist_msg);
        init_cmd_timer_.stop();
      } break;

      default:
        break;
    }
  }

  void polyTrajCallback(const trajectory_manager::PolyTrajConstPtr& msg)
  {
    if (msg->order != 7) {
      ROS_ERROR("[traj_server] Only support trajectory order equals 7 now!");
      return;
    }

    if (msg->duration.size() * (msg->order + 1) != msg->coef_x.size()) {
      ROS_ERROR("[traj_server] WRONG trajectory parameters, ");
      return;
    }

    int piece_nums = msg->duration.size();
    std::vector<double> dura(piece_nums);
    std::vector<Eigen::Matrix<double, 3, 8>> cMats(piece_nums);

    for (int i = 0; i < piece_nums; ++i) {
      int i8 = i * 8;
      cMats[i].row(0) << msg->coef_x[i8 + 0], msg->coef_x[i8 + 1], msg->coef_x[i8 + 2],
          msg->coef_x[i8 + 3], msg->coef_x[i8 + 4], msg->coef_x[i8 + 5], msg->coef_x[i8 + 6],
          msg->coef_x[i8 + 7];
      cMats[i].row(1) << msg->coef_y[i8 + 0], msg->coef_y[i8 + 1], msg->coef_y[i8 + 2],
          msg->coef_y[i8 + 3], msg->coef_y[i8 + 4], msg->coef_y[i8 + 5], msg->coef_y[i8 + 6],
          msg->coef_y[i8 + 7];
      cMats[i].row(2) << msg->coef_z[i8 + 0], msg->coef_z[i8 + 1], msg->coef_z[i8 + 2],
          msg->coef_z[i8 + 3], msg->coef_z[i8 + 4], msg->coef_z[i8 + 5], msg->coef_z[i8 + 6],
          msg->coef_z[i8 + 7];
      dura[i] = msg->duration[i];
    }

    std::unique_ptr<Trajectory<7, 3>> new_traj(new Trajectory<7, 3>(dura, cMats));
    const ros::Time now = ros::Time::now();
    const bool active_traj = receive_traj_ && traj_ &&
        (now - start_time_).toSec() < traj_duration_;
    const bool starts_in_future = (msg->start_time - now).toSec() > 0.01;

    // Do not replace a running trajectory with one that starts in the future.
    // Replacing it would make cmdCallBack() return during the waiting period,
    // which appears on Go2 as a stop and creates a visible odometry offset at
    // the old/new trajectory handoff. Keep the old trajectory active and swap
    // the replacement atomically when its start time arrives.
    if (active_traj && starts_in_future) {
      pending_traj_ = std::move(new_traj);
      pending_start_time_ = msg->start_time;
      pending_traj_id_ = msg->traj_id;
      ROS_INFO("[traj_server] Queued trajectory ID %d, starts in %.3f s; keeping active trajectory.",
          pending_traj_id_, (pending_start_time_ - now).toSec());
      return;
    }

    pending_traj_.reset();
    activateTrajectory(std::move(new_traj), msg->start_time, msg->traj_id);
  }

  void odometryCallback(const nav_msgs::OdometryConstPtr& msg)
  {
    odom_pos_(0) = msg->pose.pose.position.x;
    odom_pos_(1) = msg->pose.pose.position.y;
    odom_pos_(2) = msg->pose.pose.position.z;

    odom_orient_.w() = msg->pose.pose.orientation.w;
    odom_orient_.x() = msg->pose.pose.orientation.x;
    odom_orient_.y() = msg->pose.pose.orientation.y;
    odom_orient_.z() = msg->pose.pose.orientation.z;

    odom_linear_vel_(0) = msg->twist.twist.linear.x;
    odom_linear_vel_(1) = msg->twist.twist.linear.y;
    odom_linear_vel_(2) = msg->twist.twist.linear.z;

    Eigen::Vector3d rot_x = odom_orient_.toRotationMatrix().block<3, 1>(0, 0);
    odom_yaw_ = atan2(rot_x(1), rot_x(0));
    have_odom_ = true;
    last_odom_wall_time_ = ros::WallTime::now();

    const ros::Time odom_stamp = msg->header.stamp.isZero() ? ros::Time::now() : msg->header.stamp;
    if (have_last_odom_sample_) {
      const double dt = (odom_stamp - last_odom_stamp_).toSec();
      const double step = (odom_pos_ - last_odom_pos_).head<2>().norm();
      // At the configured Go2 speed a normal odometry sample cannot move this
      // far. Report discontinuities instead of silently drawing them into the
      // blue /travel_traj history.
      if (dt > 0.0 && dt < 0.5 && step > 0.20) {
        ROS_WARN_THROTTLE(1.0,
            "[traj_server] Odom position jump %.3f m in %.3f s; blue /travel_traj may show an offset.",
            step, dt);
      }
    }
    last_odom_stamp_ = odom_stamp;
    last_odom_pos_ = odom_pos_;
    have_last_odom_sample_ = true;
    traj_real_.push_back(Eigen::Vector3d(odom_pos_(0), odom_pos_(1), 0.15));
    if (traj_real_.size() > 50000)
      traj_real_.erase(traj_real_.begin(), traj_real_.begin() + 10000);
  }

  void stopCallback(const std_msgs::EmptyConstPtr& msg)
  {
    // Immediate emergency stop
    if (receive_traj_) {
      const double t_stop = (ros::Time::now() - start_time_).toSec();
      traj_duration_ = min(t_stop, traj_duration_);
    }
    receive_traj_ = false;
    pending_traj_.reset();
    has_target_angle_ = false;
    init_cmd_timer_.stop();
    init_warmup_started_ = false;
    init_rotation_started_ = false;
    init_state_ = 2;

    publishZeroVelocity();
  }

  void targetAngleCallback(const std_msgs::Float32ConstPtr& msg)
  {
    // A pure turn replaces any active polynomial trajectory.
    receive_traj_ = false;
    pending_traj_.reset();
    publishZeroVelocity();
    target_yaw_ = msg->data;
    has_target_angle_ = true;
    rotation_start_time_ = ros::Time::now();
    rotation_start_pos_ = odom_pos_;
    rotation_start_pos_valid_ = have_odom_;

    ROS_INFO("Received target angle: %.3f radians (%.1f degrees)", target_yaw_,
        target_yaw_ * 180.0 / M_PI);
  }

  void visCallback(const ros::TimerEvent& e)
  {
    displayTrajWithColor(
        traj_real_, 0.10, Vector4d(2.0 / 255.0, 111.0 / 255.0, 197.0 / 255.0, 1), 0);
  }

  void cmdCallBack(const ros::TimerEvent& event)
  {
    const ros::Time current_time = ros::Time::now();

    // Activate a queued replacement exactly at its requested start time. Until
    // then the currently active trajectory continues to command the robot.
    if (pending_traj_ && current_time + ros::Duration(0.005) >= pending_start_time_) {
      std::unique_ptr<Trajectory<7, 3>> replacement = std::move(pending_traj_);
      activateTrajectory(std::move(replacement), pending_start_time_, pending_traj_id_);
    }

    // Check for rotate-to-target-angle task
    if (has_target_angle_) {
      executeRotationToTarget();
      return;
    }

    if (!receive_traj_) {
      return;
    }

    double elapsed_time = (current_time - start_time_).toSec();

    if (elapsed_time < 0)
      return;  // Wait for start time to pass

    if (elapsed_time > traj_duration_) {
      // Trajectory finished, stop publishing
      publishZeroVelocity();            // Publish zero velocity
      receive_traj_ = false;            // Reset flag so that no more commands are published
      return;
    }

    if (use_mpc_) {
      const Eigen::Vector3d current_desire_pos = traj_->getPos(elapsed_time);
      const Eigen::Vector3d current_desire_vel = traj_->getVel(elapsed_time);
      Eigen::Vector3d pos = current_desire_pos;
      Eigen::Vector3d vel = current_desire_vel;

      Eigen::Vector3d ref;
      ref(0) = pos(0);
      ref(1) = pos(1);
      ref(2) = atan2(vel(1), vel(0));
      for (int i = 0; i < mpc_N_; ++i) {
        double temp_t = elapsed_time + i * mpc_dt_;
        if (temp_t <= traj_duration_) {
          pos = traj_->getPos(temp_t);
          vel = traj_->getVel(temp_t);
          ref(0) = pos(0);
          ref(1) = pos(1);
          ref(2) = atan2(vel(1), vel(0));
        }
        xref_[i] = ref;
      }
      Eigen::Vector2d cmd;
      mpc_controller_->setOdom(
          Eigen::Vector4d(odom_pos_(0), odom_pos_(1), odom_yaw_, odom_linear_vel_.head(2).norm()));
      cmd = mpc_controller_->calCmd(xref_);
      const double track_err =
          Eigen::Vector2d(odom_pos_(0) - current_desire_pos(0),
              odom_pos_(1) - current_desire_pos(1)).norm();
      geometry_msgs::Twist twist_msg;
      twist_msg.linear.x = applyTrackingErrorSpeedLimit(cmd(0), track_err);
      twist_msg.linear.y = 0.0;
      twist_msg.linear.z = 0.0;
      twist_msg.angular.x = 0.0;
      twist_msg.angular.y = 0.0;
      twist_msg.angular.z = applyTrackingErrorYawLimit(cmd(1), track_err);
      vel_cmd_pub_.publish(twist_msg);
      ROS_INFO_THROTTLE(1.0, "[traj_server] cmd_vel vx=%.2f wz=%.2f track_err=%.2f",
          twist_msg.linear.x, twist_msg.angular.z, track_err);

      // Publish current desired pose
      geometry_msgs::Pose desire_pose;
      double current_desire_yaw = atan2(current_desire_vel(1), current_desire_vel(0));

      desire_pose.position.x = current_desire_pos(0);
      desire_pose.position.y = current_desire_pos(1);
      desire_pose.position.z = current_desire_pos(2);

      // Convert yaw to quaternion
      Eigen::Quaterniond q(Eigen::AngleAxisd(current_desire_yaw, Eigen::Vector3d::UnitZ()));
      desire_pose.orientation.x = q.x();
      desire_pose.orientation.y = q.y();
      desire_pose.orientation.z = q.z();
      desire_pose.orientation.w = q.w();

      current_desire_pub_.publish(desire_pose);
      // ROS_ERROR("mpc cmd: [%f, %f]", cmd(0), cmd(1));
    }
    else {
      // Non-MPC branch removed: use MPC controller to compute commands here as well.
      const Eigen::Vector3d current_desire_pos = traj_->getPos(elapsed_time);
      const Eigen::Vector3d current_desire_vel = traj_->getVel(elapsed_time);
      Eigen::Vector3d pos = current_desire_pos;
      Eigen::Vector3d vel = current_desire_vel;

      Eigen::Vector3d ref;
      ref(0) = pos(0);
      ref(1) = pos(1);
      ref(2) = atan2(vel(1), vel(0));
      for (int i = 0; i < mpc_N_; ++i) {
        double temp_t = elapsed_time + i * mpc_dt_;
        if (temp_t <= traj_duration_) {
          pos = traj_->getPos(temp_t);
          vel = traj_->getVel(temp_t);
          ref(0) = pos(0);
          ref(1) = pos(1);
          ref(2) = atan2(vel(1), vel(0));
        }
        xref_[i] = ref;
      }
      Eigen::Vector2d cmd;
      mpc_controller_->setOdom(
          Eigen::Vector4d(odom_pos_(0), odom_pos_(1), odom_yaw_, odom_linear_vel_.head(2).norm()));
      cmd = mpc_controller_->calCmd(xref_);
      const double track_err =
          Eigen::Vector2d(odom_pos_(0) - current_desire_pos(0),
              odom_pos_(1) - current_desire_pos(1)).norm();
      geometry_msgs::Twist twist_msg;
      twist_msg.linear.x = applyTrackingErrorSpeedLimit(cmd(0), track_err);
      twist_msg.linear.y = 0.0;
      twist_msg.linear.z = 0.0;
      twist_msg.angular.x = 0.0;
      twist_msg.angular.y = 0.0;
      twist_msg.angular.z = applyTrackingErrorYawLimit(cmd(1), track_err);
      vel_cmd_pub_.publish(twist_msg);
      ROS_INFO_THROTTLE(1.0, "[traj_server] cmd_vel vx=%.2f wz=%.2f track_err=%.2f",
          twist_msg.linear.x, twist_msg.angular.z, track_err);

      // Publish current desired pose
      geometry_msgs::Pose desire_pose;
      double current_desire_yaw = atan2(current_desire_vel(1), current_desire_vel(0));

      desire_pose.position.x = current_desire_pos(0);
      desire_pose.position.y = current_desire_pos(1);
      desire_pose.position.z = current_desire_pos(2);

      // Convert yaw to quaternion
      Eigen::Quaterniond q(Eigen::AngleAxisd(current_desire_yaw, Eigen::Vector3d::UnitZ()));
      desire_pose.orientation.x = q.x();
      desire_pose.orientation.y = q.y();
      desire_pose.orientation.z = q.z();
      desire_pose.orientation.w = q.w();

      current_desire_pub_.publish(desire_pose);
    }
  }

  void publishZeroVelocity()
  {
    geometry_msgs::Twist twist_msg;
    twist_msg.linear.x = 0.0;
    twist_msg.angular.z = 0.0;
    vel_cmd_pub_.publish(twist_msg);
  }

  void abortInitialRotation(const std::string& reason)
  {
    publishZeroVelocity();
    init_cmd_timer_.stop();
    init_warmup_started_ = false;
    init_rotation_started_ = false;
    init_state_ = 2;
    ROS_ERROR("[traj_server] Initial rotation aborted: %s (progress %.1f deg). Robot stopped.",
        reason.c_str(), rotation_accum_ * 180.0 / M_PI);
  }

  void activateTrajectory(
      std::unique_ptr<Trajectory<7, 3>> new_traj,
      const ros::Time& new_start_time,
      int new_traj_id)
  {
    has_target_angle_ = false;
    traj_ = std::move(new_traj);
    start_time_ = new_start_time;
    traj_duration_ = traj_->getTotalDuration();
    traj_id_ = new_traj_id;
    receive_traj_ = true;

    ROS_INFO("[traj_server] Activated trajectory ID %d, total duration %.3f s, starts in %.3f s.",
        traj_id_, traj_duration_, (start_time_ - ros::Time::now()).toSec());
  }

  void executeRotationToTarget()
  {
    if (!have_odom_) {
      return;
    }

    // Compute yaw error
    double yaw_error =
        std::atan2(std::sin(target_yaw_ - odom_yaw_), std::cos(target_yaw_ - odom_yaw_));

    if (std::abs(yaw_error) <= rotation_tolerance_) {
      // Target angle reached: stop rotating
      geometry_msgs::Twist twist_msg;
      twist_msg.linear.x = 0.0;
      twist_msg.angular.z = 0.0;
      vel_cmd_pub_.publish(twist_msg);

      has_target_angle_ = false;
      const double xy_displacement = rotation_start_pos_valid_ ?
          (odom_pos_ - rotation_start_pos_).head<2>().norm() : -1.0;
      ROS_INFO("Reached target angle: %.3f radians (%.1f degrees), body XY displacement=%.3f m",
          target_yaw_, target_yaw_ * 180.0 / M_PI, xy_displacement);
      rotation_start_pos_valid_ = false;
      return;
    }

    // Compute angular velocity: use a simple P controller instead of PID
    const double Kp_rotation = 2.0;  // proportional gain; adjust if needed
    double angular_velocity = Kp_rotation * yaw_error;

    // Limit maximum angular velocity
    const double max_angular_velocity = max_correction_omega_;  // rad/s
    angular_velocity = std::max(-max_angular_velocity, std::min(max_angular_velocity, angular_velocity));
    if (std::fabs(angular_velocity) < min_rotation_omega_) {
      angular_velocity = std::copysign(min_rotation_omega_, yaw_error);
    }

    // Send rotation command
    geometry_msgs::Twist twist_msg;
    twist_msg.linear.x = 0.0;
    twist_msg.linear.y = 0.0;
    twist_msg.linear.z = 0.0;
    twist_msg.angular.x = 0.0;
    twist_msg.angular.y = 0.0;
    twist_msg.angular.z = angular_velocity;
    vel_cmd_pub_.publish(twist_msg);

    // Log debug info
    double elapsed_rotation_time = (ros::Time::now() - rotation_start_time_).toSec();
    if (static_cast<int>(elapsed_rotation_time * 10) % 10 == 0) {  // print every 0.1s
      ROS_INFO("Rotating to target: current=%.2f°, target=%.2f°, error=%.2f°, vel=%.2f rad/s",
          odom_yaw_ * 180.0 / M_PI, target_yaw_ * 180.0 / M_PI, yaw_error * 180.0 / M_PI,
          angular_velocity);
    }
  }

  double applyTrackingErrorSpeedLimit(double commanded_vx, double track_err) const
  {
    double limited_vx = std::max(-max_correction_vel_, std::min(max_correction_vel_, commanded_vx));

    if (track_err <= tracking_slowdown_error_) {
      return limited_vx;
    }

    if (track_err >= tracking_stop_error_) {
      return 0.0;
    }

    const double raw_scale =
        (tracking_stop_error_ - track_err) /
        std::max(1.0e-6, tracking_stop_error_ - tracking_slowdown_error_);
    const double speed_scale = std::max(tracking_min_speed_scale_, raw_scale);
    limited_vx *= speed_scale;
    if (std::fabs(limited_vx) > 1.0e-3 && std::fabs(limited_vx) < tracking_min_effective_vx_) {
      limited_vx = std::copysign(tracking_min_effective_vx_, limited_vx);
    }
    return limited_vx;
  }

  double applyTrackingErrorYawLimit(double commanded_wz, double track_err) const
  {
    double limited_wz = std::max(-max_correction_omega_, std::min(max_correction_omega_, commanded_wz));

    if (track_err <= tracking_slowdown_error_) {
      return limited_wz;
    }

    if (track_err >= tracking_stop_error_) {
      return 0.0;
    }

    const double raw_scale =
        (tracking_stop_error_ - track_err) /
        std::max(1.0e-6, tracking_stop_error_ - tracking_slowdown_error_);
    const double yaw_scale = std::max(0.10, raw_scale);
    return limited_wz * yaw_scale;
  }

    void publishRobotMarker()
  {
    const double robot_height = 0.15;
    const double robot_radius = 0.18;

    visualization_msgs::Marker marker;
    marker.header.frame_id = "odom";  // Set reference frame
    marker.header.stamp = ros::Time::now();
    marker.ns = "robot_position";
    marker.id = 0;
    marker.type = visualization_msgs::Marker::CYLINDER;  // Set to CYLINDER
    marker.action = visualization_msgs::Marker::ADD;

    // Set cylinder position
    marker.pose.position.x = odom_pos_(0);
    marker.pose.position.y = odom_pos_(1);
    marker.pose.position.z = odom_pos_(2) + robot_height / 2.0;

    // Set cylinder orientation (quaternion)
    marker.pose.orientation.x = odom_orient_.x();
    marker.pose.orientation.y = odom_orient_.y();
    marker.pose.orientation.z = odom_orient_.z();
    marker.pose.orientation.w = odom_orient_.w();

    // Set cylinder size
    marker.scale.x = robot_radius * 2;  // diameter
    marker.scale.y = robot_radius * 2;  // diameter
    marker.scale.z = robot_height;      // height

    marker.color.r = 50.0 / 255.0;
    marker.color.g = 50.0 / 255.0;
    marker.color.b = 255.0 / 255.0;
    marker.color.a = 1.0;  // opaque

    // Create and publish arrow (direction)
    visualization_msgs::Marker arrow_marker;
    arrow_marker.header.frame_id = "odom";
    arrow_marker.header.stamp = ros::Time::now();
    arrow_marker.ns = "robot_direction";
    arrow_marker.id = 1;
    arrow_marker.type = visualization_msgs::Marker::ARROW;  // Set to ARROW
    arrow_marker.action = visualization_msgs::Marker::ADD;

    // Set arrow position (start)
    arrow_marker.pose.position.x = odom_pos_(0);
    arrow_marker.pose.position.y = odom_pos_(1);
    arrow_marker.pose.position.z = odom_pos_(2) + robot_height;

    // Set arrow orientation (from quaternion)
    arrow_marker.pose.orientation.x = odom_orient_.x();
    arrow_marker.pose.orientation.y = odom_orient_.y();
    arrow_marker.pose.orientation.z = odom_orient_.z();
    arrow_marker.pose.orientation.w = odom_orient_.w();

    // Set arrow size
    arrow_marker.scale.x = robot_radius + 0.13;  // arrow length
    arrow_marker.scale.y = 0.08;                 // arrow width
    arrow_marker.scale.z = 0.08;                 // arrow thickness

    arrow_marker.color.r = 10.0 / 255.0;
    arrow_marker.color.g = 255.0 / 255.0;
    arrow_marker.color.b = 10.0 / 255.0;
    arrow_marker.color.a = 1.0;  // opaque

    robot_marker_pub_.publish(marker);
    robot_marker_pub_.publish(arrow_marker);
  }

  void displayTrajWithColor(
      vector<Eigen::Vector3d> path, double resolution, Eigen::Vector4d color, int id)
  {
    visualization_msgs::Marker mk;
    mk.header.frame_id = "odom";
    mk.header.stamp = ros::Time::now();
    mk.type = visualization_msgs::Marker::SPHERE_LIST;
    mk.action = visualization_msgs::Marker::DELETE;
    mk.id = id;
    traj_vis_pub_.publish(mk);

    mk.action = visualization_msgs::Marker::ADD;
    mk.pose.orientation.x = 0.0;
    mk.pose.orientation.y = 0.0;
    mk.pose.orientation.z = 0.0;
    mk.pose.orientation.w = 1.0;
    mk.color.r = color(0);
    mk.color.g = color(1);
    mk.color.b = color(2);
    mk.color.a = color(3);
    mk.scale.x = resolution;
    mk.scale.y = resolution;
    mk.scale.z = resolution;
    geometry_msgs::Point pt;
    for (int i = 0; i < int(path.size()); i++) {
      pt.x = path[i](0);
      pt.y = path[i](1);
      pt.z = path[i](2);
      mk.points.push_back(pt);
    }
    traj_vis_pub_.publish(mk);
    ros::Duration(0.0001).sleep();
  }

private:
  ros::NodeHandle nh_;
  ros::Subscriber traj_sub_, odom_sub_, stop_sub_, target_angle_sub_;
  ros::Publisher vel_cmd_pub_, robot_marker_pub_, traj_vis_pub_, current_desire_pub_;
  ros::Timer cmd_timer_, vis_timer_, init_cmd_timer_;

  // Trajectory Data
  std::unique_ptr<Trajectory<7, 3>> traj_;
  std::unique_ptr<Trajectory<7, 3>> pending_traj_;
  ros::Time start_time_;
  ros::Time pending_start_time_;
  double traj_duration_;
  int traj_id_;
  int pending_traj_id_;
  bool receive_traj_;

  bool use_mpc_ = true;
  MPC::Ptr mpc_controller_;
  std::vector<Eigen::Vector3d> xref_;
  int mpc_N_;
  double mpc_dt_;

  // Target Angle Data
  double target_yaw_;
  bool has_target_angle_;
  ros::Time rotation_start_time_;
  Vector3d rotation_start_pos_ = Vector3d::Zero();
  bool rotation_start_pos_valid_ = false;

  // Data
  Vector3d odom_pos_, odom_linear_vel_;
  Quaterniond odom_orient_;
  double odom_yaw_;
  bool have_odom_;
  bool have_last_odom_sample_ = false;
  ros::Time last_odom_stamp_;
  Vector3d last_odom_pos_;
  double replan_time_ = 0.5;
  vector<Eigen::Vector3d> traj_real_;
  int init_state_;
  // init rotation: prefer odom-based stopping (accumulate yaw change);
  bool init_warmup_started_;
  bool init_rotation_started_;
  double rotation_accum_;  // accumulated absolute yaw change (rad)
  double last_odom_yaw_;   // last odom yaw used for accumulation
  ros::WallTime last_odom_wall_time_;
  ros::WallTime init_warmup_start_wall_time_;
  ros::WallTime init_rotation_start_wall_time_;
  ros::WallTime init_last_progress_wall_time_;
  double init_rotation_omega_;
  double init_odom_warmup_;
  double init_rotation_timeout_;
  double init_odom_timeout_;
  double init_no_progress_timeout_;
  double max_correction_vel_, max_correction_omega_;
  double rotation_tolerance_, min_rotation_omega_;
  double tracking_slowdown_error_, tracking_stop_error_, tracking_min_speed_scale_;
  double tracking_min_effective_vx_;
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "trajectory_server_node");
  ros::NodeHandle nh("~");
  TrajectoryServer traj_server(nh);
  ros::spin();
  return 0;
}
