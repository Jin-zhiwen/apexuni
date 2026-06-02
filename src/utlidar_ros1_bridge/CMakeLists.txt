cmake_minimum_required(VERSION 3.0.2)
project(utlidar_ros1_bridge)

find_package(catkin REQUIRED COMPONENTS
  rospy
  nav_msgs
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/ros1_odom_publisher.py
  scripts/ros2_utlidar_to_udp.py
  scripts/start_utlidar_bridge.sh
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  OPTIONAL
)
