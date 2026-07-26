#include <algorithm>
#include <cmath>
#include <string>
#include <thread>
#include <chrono>
#include <mutex>
#include <atomic>
#include <memory>

#include <geometry_msgs/Twist.h>
#include <ros/ros.h>

// Only try to include Unitree SDK if we can safely handle failures
#ifdef UNITREE_AVAILABLE
  #include <unitree/robot/channel/channel_factory.hpp>
  #include <unitree/robot/go2/sport/sport_client.hpp>
#endif

class CmdVelToGo2Bridge
{
public:
  CmdVelToGo2Bridge(ros::NodeHandle& nh, ros::NodeHandle& pnh)
    : nh_(nh), pnh_(pnh), initialized_(false), init_attempted_(false), sim_mode_(false)
  {
    pnh_.param<std::string>("network_interface", network_interface_, std::string(""));
    pnh_.param<int>("domain_id", domain_id_, 0);
    pnh_.param<double>("max_vx", max_vx_, 0.6);
    pnh_.param<double>("max_vy", max_vy_, 0.4);
    pnh_.param<double>("max_yaw", max_yaw_, 1.2);
    pnh_.param<double>("cmd_timeout", cmd_timeout_, 0.5);
    pnh_.param<double>("publish_rate", publish_rate_, 50.0);
    pnh_.param<bool>("auto_stand", auto_stand_, true);
    pnh_.param<bool>("sim_mode", sim_mode_, false);

    if (network_interface_.empty())
    {
      ROS_FATAL("[go2_cmd_vel_bridge] network_interface param is empty.");
      throw std::runtime_error("network_interface param is empty");
    }

    ROS_INFO("[go2_cmd_vel_bridge] network_interface=%s domain_id=%d sim_mode=%s",
             network_interface_.c_str(), domain_id_, sim_mode_ ? "true" : "false");

    if (sim_mode_)
    {
      ROS_WARN("[go2_cmd_vel_bridge] Running in SIMULATION MODE - no GO2 commands sent");
      initialized_ = true;  // In sim mode, consider it initialized
      last_sim_cmd_time_ = ros::Time::now();
    }
    else
    {
      // Try SDK initialization - may fail in non-hardware environment
      initializeSdkSafe();
    }

    cmd_sub_ = nh_.subscribe("/cmd_vel", 1, &CmdVelToGo2Bridge::cmdVelCallback, this);

    timer_ = nh_.createTimer(
      ros::Duration(1.0 / std::max(1.0, publish_rate_)),
      &CmdVelToGo2Bridge::timerCallback,
      this);

    last_cmd_time_ = ros::Time(0);
    has_cmd_ = false;

    ROS_INFO("[go2_cmd_vel_bridge] Ready. interface=%s domain_id=%d sdk=%s",
             network_interface_.c_str(), domain_id_,
             sim_mode_ ? "simulated" : (initialized_ ? "initialized" : "not-initialized"));
  }

private:
  void initializeSdkSafe()
  {
    const ros::Time now = ros::Time::now();
    if (init_attempted_ && (now - last_init_attempt_time_).toSec() < init_retry_interval_)
      return;
    init_attempted_ = true;
    last_init_attempt_time_ = now;

    try
    {
#ifdef UNITREE_AVAILABLE
      // Respect externally provided CYCLONEDDS_URI; if not set, provide a safe file:// fallback
      const char* env_uri = std::getenv("CYCLONEDDS_URI");
      if (env_uri && std::strlen(env_uri) > 0)
      {
        ROS_INFO("[go2_cmd_vel_bridge] Using existing CYCLONEDDS_URI=%s", env_uri);
      }
      else
      {
        std::string cyclone_uri = "file:///home/jzw/apexuni/unitree_sdk2/thirdparty/share/config/dds_default.xml";
        setenv("CYCLONEDDS_URI", cyclone_uri.c_str(), 1);
        ROS_INFO("[go2_cmd_vel_bridge] CYCLONEDDS_URI not set; using fallback %s", cyclone_uri.c_str());
      }

      ROS_INFO("[go2_cmd_vel_bridge] Initializing DDS (domain_id=%d, interface=%s)...", domain_id_, network_interface_.c_str());
      unitree::robot::ChannelFactory::Instance()->Init(domain_id_, network_interface_);
      ROS_INFO("[go2_cmd_vel_bridge] DDS initialized.");

      ROS_INFO("[go2_cmd_vel_bridge] Initializing SportClient...");
      sport_client_.reset(new unitree::robot::go2::SportClient());
      sport_client_->SetTimeout(10.0f);
      sport_client_->Init();
      ROS_INFO("[go2_cmd_vel_bridge] SportClient initialized.");

      initialized_ = true;

      if (auto_stand_ && sport_client_)
      {
        ROS_INFO("[go2_cmd_vel_bridge] Sending StandUp...");
        const int32_t stand_ret = sport_client_->StandUp();
        ROS_INFO("[go2_cmd_vel_bridge] StandUp returned %d.", stand_ret);
        std::this_thread::sleep_for(std::chrono::seconds(1));
        ROS_INFO("[go2_cmd_vel_bridge] Sending BalanceStand...");
        const int32_t balance_ret = sport_client_->BalanceStand();
        ROS_INFO("[go2_cmd_vel_bridge] BalanceStand returned %d.", balance_ret);
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
      }
#else
      ROS_WARN("[go2_cmd_vel_bridge] Unitree SDK not available - will operate in monitor-only mode");
      initialized_ = false;
#endif
    }
    catch (const std::exception& e)
    {
      ROS_WARN("[go2_cmd_vel_bridge] SDK init exception: %s", e.what());
      initialized_ = false;
    }
    catch (...)
    {
      ROS_WARN("[go2_cmd_vel_bridge] SDK init failed (unknown exception)");
      initialized_ = false;
    }
  }

  static double clamp(double value, double min_value, double max_value)
  {
    return std::max(min_value, std::min(max_value, value));
  }

  void cmdVelCallback(const geometry_msgs::Twist::ConstPtr& msg)
  {
    last_cmd_ = *msg;
    last_cmd_time_ = ros::Time::now();
    has_cmd_ = true;

    if (std::fabs(msg->linear.x) > 1e-3 || std::fabs(msg->linear.y) > 1e-3 ||
        std::fabs(msg->angular.z) > 1e-3)
    {
      ROS_INFO_THROTTLE(1.0, "[go2_cmd_vel_bridge] received cmd_vel vx=%.2f vy=%.2f yaw=%.2f",
                        msg->linear.x, msg->linear.y, msg->angular.z);
    }
  }

  void timerCallback(const ros::TimerEvent&)
  {
    if (!initialized_)
    {
      ROS_WARN_THROTTLE(
          2.0,
          "[go2_cmd_vel_bridge] SDK not ready - skipping cmd_vel. Check network_interface=%s, "
          "domain_id=%d, GO2 SDK network reachability, and sim_mode.",
          network_interface_.c_str(), domain_id_);
      return;
    }

    try
    {
      const ros::Time now = ros::Time::now();
      
      // Check timeout
      if (!has_cmd_ || (now - last_cmd_time_).toSec() > cmd_timeout_)
      {
        if (sim_mode_)
        {
          ROS_DEBUG_THROTTLE(1.0, "[go2_cmd_vel_bridge] [SIM] Stop");
        }
        else
        {
#ifdef UNITREE_AVAILABLE
          if (sport_client_)
          {
            const int32_t stop_ret = sport_client_->StopMove();
            ROS_INFO_THROTTLE(1.0, "[go2_cmd_vel_bridge] StopMove returned %d.", stop_ret);
          }
#endif
        }
        return;
      }

      // Parse velocities
      const double vx = clamp(last_cmd_.linear.x, -max_vx_, max_vx_);
      const double vy = clamp(last_cmd_.linear.y, -max_vy_, max_vy_);
      const double yaw = clamp(last_cmd_.angular.z, -max_yaw_, max_yaw_);

      // Send command
      if (sim_mode_)
      {
        ROS_DEBUG_THROTTLE(1.0, "[go2_cmd_vel_bridge] [SIM] vx=%.2f vy=%.2f yaw=%.2f", vx, vy, yaw);
        last_sim_cmd_time_ = now;
      }
      else
      {
#ifdef UNITREE_AVAILABLE
  if (sport_client_)
  {
    const int32_t move_ret = sport_client_->Move(vx, vy, yaw);
    ROS_INFO_THROTTLE(1.0, "[go2_cmd_vel_bridge] Move returned %d; sent Move vx=%.2f vy=%.2f yaw=%.2f",
                      move_ret, vx, vy, yaw);
  }
#endif
      }
    }
    catch (const std::exception& e)
    {
      ROS_WARN_THROTTLE(2.0, "[go2_cmd_vel_bridge] Command exception: %s", e.what());
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

#ifdef UNITREE_AVAILABLE
  std::unique_ptr<unitree::robot::go2::SportClient> sport_client_;
#endif

  geometry_msgs::Twist last_cmd_;
  ros::Time last_cmd_time_;
  ros::Time last_sim_cmd_time_;
  bool has_cmd_;
  bool initialized_;
  bool init_attempted_;
  bool sim_mode_;
  ros::Time last_init_attempt_time_;
  double init_retry_interval_ = 3.0;

  std::string network_interface_;
  int domain_id_;
  double max_vx_;
  double max_vy_;
  double max_yaw_;
  double cmd_timeout_;
  double publish_rate_;
  bool auto_stand_;
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "go2_cmd_vel_bridge");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");

  try
  {
    CmdVelToGo2Bridge bridge(nh, pnh);
    ROS_INFO("[go2_cmd_vel_bridge] Entering main loop...");
    ros::spin();
  }
  catch (const std::exception& ex)
  {
    ROS_FATAL("[go2_cmd_vel_bridge] Fatal error: %s", ex.what());
    return 1;
  }
  catch (...)
  {
    ROS_FATAL("[go2_cmd_vel_bridge] Fatal error (unknown)");
    return 1;
  }

  return 0;
}
