cmake_minimum_required(VERSION 3.0.2)
project(unilidar_fastlio_bridge)

find_package(catkin REQUIRED COMPONENTS
  rospy
  nav_msgs
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/republish_fastlio_odom.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
