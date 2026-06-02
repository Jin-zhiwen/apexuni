#include <iostream>
#include <string>
#include <thread>
#include <chrono>
#include <cstring>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>

#include <geometry_msgs/Twist.h>
#include <ros/ros.h>

#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/go2/sport/sport_client.hpp>

class DiagnosticBridge
{
public:
  DiagnosticBridge(ros::NodeHandle& nh, ros::NodeHandle& pnh)
    : nh_(nh), pnh_(pnh), initialized_(false)
  {
    pnh_.param<std::string>("network_interface", network_interface_, "");
    pnh_.param<int>("domain_id", domain_id_, 0);
    pnh_.param<double>("publish_rate", publish_rate_, 50.0);

    if (network_interface_.empty())
    {
      ROS_FATAL("network_interface param is empty");
      throw std::runtime_error("network_interface param is empty");
    }

    ROS_INFO("[DIAGNOSE] network_interface=%s domain_id=%d", network_interface_.c_str(), domain_id_);

    // Test network connectivity first
    testNetworkConnectivity();

    // Try SDK initialization with detailed logging
    initializeSdk();

    // Subscribe to cmd_vel
    cmd_sub_ = nh_.subscribe("/cmd_vel", 1, &DiagnosticBridge::cmdVelCallback, this);

    // Create timer for command execution
    timer_ = nh_.createTimer(
      ros::Duration(1.0 / std::max(1.0, publish_rate_)),
      &DiagnosticBridge::timerCallback,
      this);

    ROS_INFO("[DIAGNOSE] Initialization complete. SDK initialized=%s", initialized_ ? "YES" : "NO");
  }

private:
  void testNetworkConnectivity()
  {
    ROS_INFO("[DIAGNOSE] Testing network connectivity...");

    // Test 1: Check if network interface exists
    std::string cmd = "ip addr show " + network_interface_ + " 2>/dev/null | grep 'inet ' | head -1";
    int ret = system(cmd.c_str());
    if (ret != 0)
    {
      ROS_WARN("[DIAGNOSE] Network interface %s may not exist or have no IP", network_interface_.c_str());
    }
    else
    {
      ROS_INFO("[DIAGNOSE] Network interface %s is configured", network_interface_.c_str());
    }

    // Test 2: Ping common GO2 addresses
    std::vector<std::string> go2_ips = {
      "192.168.123.161",  // Default GO2 IP
      "192.168.123.1",    // Gateway
      "255.255.255.255"   // Broadcast
    };

    for (const auto& ip : go2_ips)
    {
      std::string ping_cmd = "timeout 1 ping -c 1 " + ip + " 2>&1 > /dev/null";
      ret = system(ping_cmd.c_str());
      if (ret == 0)
      {
        ROS_INFO("[DIAGNOSE] Successfully pinged %s", ip.c_str());
      }
      else
      {
        ROS_WARN("[DIAGNOSE] Cannot ping %s (timeout or unreachable)", ip.c_str());
      }
    }

    // Test 3: Check ifconfig output
    ROS_INFO("[DIAGNOSE] Interface details:");
    cmd = "ifconfig " + network_interface_ + " 2>/dev/null | head -10";
    system(cmd.c_str());
  }

  void initializeSdk()
  {
    try
    {
      ROS_INFO("[DIAGNOSE] Setting CYCLONEDDS_URI...");
      std::string cyclone_uri = "dds:///home/jzw/apexuni/unitree_sdk2/thirdparty/share/config/dds_default.xml";
      setenv("CYCLONEDDS_URI", cyclone_uri.c_str(), 1);
      ROS_INFO("[DIAGNOSE] CYCLONEDDS_URI=%s", getenv("CYCLONEDDS_URI"));

      ROS_INFO("[DIAGNOSE] Creating ChannelFactory...");
      auto factory = unitree::robot::ChannelFactory::Instance();
      ROS_INFO("[DIAGNOSE] ChannelFactory created, calling Init(domain_id=%d, interface=%s)...", 
               domain_id_, network_interface_.c_str());

      // This might timeout/hang if network is not properly configured
      std::cout << std::flush;
      fflush(stdout);
      factory->Init(domain_id_, network_interface_);
      ROS_INFO("[DIAGNOSE] ChannelFactory Init completed successfully!");

      ROS_INFO("[DIAGNOSE] Creating SportClient...");
      sport_client_ = std::make_unique<unitree::robot::go2::SportClient>();
      ROS_INFO("[DIAGNOSE] SportClient created, setting timeout...");
      sport_client_->SetTimeout(10.0f);
      
      ROS_INFO("[DIAGNOSE] Calling SportClient::Init()...");
      std::cout << std::flush;
      fflush(stdout);
      sport_client_->Init();
      ROS_INFO("[DIAGNOSE] SportClient Init completed successfully!");

      initialized_ = true;
      ROS_INFO("[DIAGNOSE] SDK fully initialized!");
    }
    catch (const std::exception& e)
    {
      ROS_ERROR("[DIAGNOSE] SDK initialization failed: %s", e.what());
      initialized_ = false;
    }
    catch (...)
    {
      ROS_ERROR("[DIAGNOSE] SDK initialization failed (unknown exception)");
      initialized_ = false;
    }
  }

  void cmdVelCallback(const geometry_msgs::Twist::ConstPtr& msg)
  {
    ROS_INFO("[DIAGNOSE] Received cmd_vel: vx=%.2f vy=%.2f yaw=%.2f",
             msg->linear.x, msg->linear.y, msg->angular.z);

    if (!initialized_)
    {
      ROS_WARN("[DIAGNOSE] SDK not initialized - cannot send command");
      return;
    }

    try
    {
      sport_client_->Move(msg->linear.x, msg->linear.y, msg->angular.z);
      ROS_INFO("[DIAGNOSE] Command sent successfully");
    }
    catch (const std::exception& e)
    {
      ROS_ERROR("[DIAGNOSE] Command failed: %s", e.what());
    }
  }

  void timerCallback(const ros::TimerEvent&)
  {
    // Just keep-alive timer, actual commands come from cmd_vel subscriber
    if (initialized_)
    {
      static int count = 0;
      if (++count % 50 == 0)  // Log every 1 second at 50Hz
      {
        ROS_DEBUG("[DIAGNOSE] Timer tick (running normally)");
      }
    }
  }

  ros::NodeHandle& nh_;
  ros::NodeHandle& pnh_;
  ros::Subscriber cmd_sub_;
  ros::Timer timer_;

  std::string network_interface_;
  int domain_id_;
  double publish_rate_;

  bool initialized_;
  std::unique_ptr<unitree::robot::go2::SportClient> sport_client_;
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "cmd_vel_to_go2_bridge_diagnose");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");

  try
  {
    DiagnosticBridge bridge(nh, pnh);
    ROS_INFO("[DIAGNOSE] Bridge created successfully, entering spin...");
    ros::spin();
  }
  catch (const std::exception& e)
  {
    ROS_FATAL("[DIAGNOSE] Fatal error: %s", e.what());
    return 1;
  }
  catch (...)
  {
    ROS_FATAL("[DIAGNOSE] Fatal error (unknown exception)");
    return 1;
  }

  return 0;
}
