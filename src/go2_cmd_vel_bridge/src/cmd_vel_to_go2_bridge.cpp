#include <algorithm>
#include <string>
#include <thread>
#include <chrono>
#include <mutex>
#include <cstdlib>
#include <sys/stat.h>

#include <geometry_msgs/Twist.h>
#include <ros/ros.h>

#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/go2/sport/sport_client.hpp>

class CmdVelToGo2Bridge
{
public:
  CmdVelToGo2Bridge(ros::NodeHandle& nh, ros::NodeHandle& pnh)
    : nh_(nh), pnh_(pnh), initialized_(false), init_attempted_(false)
  {
    pnh_.param<std::string>("network_interface", network_interface_, std::string(""));
    pnh_.param<int>("domain_id", domain_id_, 0);
    pnh_.param<double>("max_vx", max_vx_, 0.6);
    pnh_.param<double>("max_vy", max_vy_, 0.4);
    pnh_.param<double>("max_yaw", max_yaw_, 1.2);
    pnh_.param<double>("cmd_timeout", cmd_timeout_, 0.5);
    pnh_.param<double>("publish_rate", publish_rate_, 50.0);
    pnh_.param<bool>("auto_stand", auto_stand_, true);
    pnh_.param<double>("init_retry_interval", init_retry_interval_, 3.0);
    pnh_.param<std::string>("cyclonedds_uri", cyclonedds_uri_, std::string(""));

    if (network_interface_.empty())
    {
      ROS_FATAL("[go2_cmd_vel_bridge] network_interface param is empty.");
      throw std::runtime_error("network_interface param is empty");
    }

    ROS_INFO("[go2_cmd_vel_bridge] Using interface=%s, domain_id=%d",
             network_interface_.c_str(), domain_id_);

    // Try immediate initialization, but don't fail if it doesn't work
    last_init_attempt_time_ = ros::Time(0);
    initializeSdkSafe();

    cmd_sub_ = nh_.subscribe("/cmd_vel", 1, &CmdVelToGo2Bridge::cmdVelCallback, this);

    timer_ = nh_.createTimer(
      ros::Duration(1.0 / std::max(1.0, publish_rate_)),
      &CmdVelToGo2Bridge::timerCallback,
      this);

    last_cmd_time_ = ros::Time(0);
    has_cmd_ = false;

    ROS_INFO("[go2_cmd_vel_bridge] Started. network_interface=%s domain_id=%d sdk_initialized=%s",
             network_interface_.c_str(), domain_id_, initialized_ ? "true" : "false");
  }

private:
  void initializeSdkSafe()
  {
    const ros::Time now = ros::Time::now();
    if (init_attempted_ && (now - last_init_attempt_time_).toSec() < init_retry_interval_)
      return;
    init_attempted_ = true;
    last_init_attempt_time_ = now;

    // Set environment for DDS only if we actually have a valid config file.
    if (!cyclonedds_uri_.empty())
    {
      struct stat st;
      if (stat(cyclonedds_uri_.c_str(), &st) == 0)
      {
        const std::string cyclone_uri = "dds://" + cyclonedds_uri_;
        setenv("CYCLONEDDS_URI", cyclone_uri.c_str(), 1);
        ROS_INFO("[go2_cmd_vel_bridge] Using CYCLONEDDS_URI=%s", cyclone_uri.c_str());
      }
      else
      {
        ROS_WARN("[go2_cmd_vel_bridge] cyclonedds_uri does not exist: %s. Proceeding without overriding CYCLONEDDS_URI.",
                 cyclonedds_uri_.c_str());
      }
    }

    try
    {
      ROS_INFO("[go2_cmd_vel_bridge] Initializing DDS channel...");
      unitree::robot::ChannelFactory::Instance()->Init(domain_id_, network_interface_);
      ROS_INFO("[go2_cmd_vel_bridge] DDS channel initialized.");

      ROS_INFO("[go2_cmd_vel_bridge] Initializing SportClient...");
      sport_client_.SetTimeout(10.0f);
      sport_client_.Init();
      ROS_INFO("[go2_cmd_vel_bridge] SportClient initialized.");

      initialized_ = true;

      if (auto_stand_)
      {
        ROS_INFO("[go2_cmd_vel_bridge] Sending StandUp...");
        try
        {
          sport_client_.StandUp();
          std::this_thread::sleep_for(std::chrono::seconds(1));
          ROS_INFO("[go2_cmd_vel_bridge] StandUp sent.");
        }
        catch (const std::exception& e)
        {
          ROS_WARN("[go2_cmd_vel_bridge] StandUp failed: %s", e.what());
        }
      }
    }
    catch (const std::exception& e)
    {
      ROS_WARN("[go2_cmd_vel_bridge] SDK initialization failed: %s", e.what());
      ROS_WARN("[go2_cmd_vel_bridge] Will operate in listen-only mode (no commands sent)");
      initialized_ = false;
    }
    catch (...)
    {
      ROS_WARN("[go2_cmd_vel_bridge] SDK initialization failed (unknown exception)");
      ROS_WARN("[go2_cmd_vel_bridge] Will operate in listen-only mode (no commands sent)");
      initialized_ = false;
    }
  }

private:
  static double clamp(double value, double min_value, double max_value)
  {
    return std::max(min_value, std::min(max_value, value));
  }

  void cmdVelCallback(const geometry_msgs::Twist::ConstPtr& msg)
  {
    last_cmd_ = *msg;
    last_cmd_time_ = ros::Time::now();
    has_cmd_ = true;
  }

  void timerCallback(const ros::TimerEvent&)
  {
    if (!initialized_)
    {
      initializeSdkSafe();
      ROS_WARN_THROTTLE(5.0,
                        "[go2_cmd_vel_bridge] SDK not initialized - bridge is in listen-only mode and commands are not being sent");
      return;
    }

    try
    {
      const ros::Time now = ros::Time::now();
      if (!has_cmd_ || (now - last_cmd_time_).toSec() > cmd_timeout_)
      {
        sport_client_.StopMove();
        return;
      }

      const double vx = clamp(last_cmd_.linear.x, -max_vx_, max_vx_);
      const double vy = clamp(last_cmd_.linear.y, -max_vy_, max_vy_);
      const double yaw = clamp(last_cmd_.angular.z, -max_yaw_, max_yaw_);

      sport_client_.Move(vx, vy, yaw);
      ROS_DEBUG_THROTTLE(1.0, "[go2_cmd_vel_bridge] vx=%.2f vy=%.2f yaw=%.2f", vx, vy, yaw);
    }
    catch (const std::exception& e)
    {
      ROS_WARN_THROTTLE(2.0, "[go2_cmd_vel_bridge] Command failed: %s", e.what());
    }
    catch (...)
    {
      ROS_WARN_THROTTLE(2.0, "[go2_cmd_vel_bridge] Command failed (unknown error)");
    }
  }

  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;
  ros::Subscriber cmd_sub_;
  ros::Timer timer_;

  unitree::robot::go2::SportClient sport_client_;

  geometry_msgs::Twist last_cmd_;
  ros::Time last_cmd_time_;
  bool has_cmd_;
  bool initialized_;
  bool init_attempted_;
  ros::Time last_init_attempt_time_;

  std::string network_interface_;
  int domain_id_;
  double max_vx_;
  double max_vy_;
  double max_yaw_;
  double cmd_timeout_;
  double publish_rate_;
  bool auto_stand_;
  double init_retry_interval_;
  std::string cyclonedds_uri_;
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "go2_cmd_vel_bridge");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");

  try
  {
    CmdVelToGo2Bridge bridge(nh, pnh);
    ros::spin();
  }
  catch (const std::exception& ex)
  {
    ROS_FATAL("[go2_cmd_vel_bridge] init failed: %s", ex.what());
    return 1;
  }

  return 0;
}
