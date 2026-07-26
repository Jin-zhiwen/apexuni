cmake_minimum_required(VERSION 2.8.3)
project(exploration_manager)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall")

find_package(Eigen3 REQUIRED)
find_package(PCL 1.7 REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  geometry_msgs
  visualization_msgs
  plan_env
  path_searching
  vis_utils
  cv_bridge
  lkh_mtsp_solver
  trajectory_manager
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES exploration_manager
  CATKIN_DEPENDS plan_env path_searching vis_utils lkh_mtsp_solver visualization_msgs
  #  DEPENDS system_lib
)

include_directories(
  include
  SYSTEM
  ${catkin_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/include
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
)

add_executable(exploration_node
  src/exploration_node.cpp
  src/exploration_fsm.cpp
  src/exploration_fsm_traj.cpp
  src/exploration_fsm_traj_logic.cpp
  src/exploration_manager.cpp
)
add_dependencies(exploration_node
  ${catkin_EXPORTED_TARGETS}
)
target_link_libraries(exploration_node
  ${catkin_LIBRARIES}
)

if(CATKIN_ENABLE_TESTING)
  catkin_add_gtest(exploration_fsm_traj_logic_test
    test/exploration_fsm_traj_logic_test.cpp
    src/exploration_fsm_traj_logic.cpp
  )
  if(TARGET exploration_fsm_traj_logic_test)
    add_dependencies(exploration_fsm_traj_logic_test
      ${catkin_EXPORTED_TARGETS}
    )
    target_link_libraries(exploration_fsm_traj_logic_test
      ${catkin_LIBRARIES}
      gtest_main
    )
  endif()
endif()

# Install Python scripts
catkin_install_python(PROGRAMS
  scripts/camera_pose_tf_converter.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
