cmake_minimum_required(VERSION 3.0.2)
project(go2_cmd_vel_bridge)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  geometry_msgs
)

find_package(Threads REQUIRED)

catkin_package()

get_filename_component(APEXUNI_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
set(UNITREE_SDK_ROOT "${APEXUNI_ROOT}/unitree_sdk2")

set(UNITREE_SDK_LIB_DIR "${UNITREE_SDK_ROOT}/lib/${CMAKE_SYSTEM_PROCESSOR}")
find_library(UNITREE_SDK_LIB unitree_sdk2 PATHS ${UNITREE_SDK_LIB_DIR} NO_DEFAULT_PATH)

set(UNITREE_DDS_LIB_DIR "${UNITREE_SDK_ROOT}/thirdparty/lib/${CMAKE_SYSTEM_PROCESSOR}")
find_library(DDSC_LIB ddsc PATHS ${UNITREE_DDS_LIB_DIR} NO_DEFAULT_PATH)
find_library(DDSCXX_LIB ddscxx PATHS ${UNITREE_DDS_LIB_DIR} NO_DEFAULT_PATH)

# Detect whether SDK + DDS libs are available
set(SDK_AVAILABLE TRUE)
if(NOT UNITREE_SDK_LIB)
  message(WARNING "Unitree SDK library not found in ${UNITREE_SDK_LIB_DIR} - building without SDK support")
  set(SDK_AVAILABLE FALSE)
endif()
if(NOT DDSC_LIB OR NOT DDSCXX_LIB)
  message(WARNING "Cyclone DDS libraries not found in ${UNITREE_DDS_LIB_DIR} - building without SDK support")
  set(SDK_AVAILABLE FALSE)
endif()

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${UNITREE_SDK_ROOT}/include
  ${UNITREE_SDK_ROOT}/thirdparty/include
  ${UNITREE_SDK_ROOT}/thirdparty/include/ddscxx
  ${UNITREE_SDK_ROOT}/thirdparty/include/ddsc
)

# Main bridge
add_executable(cmd_vel_to_go2_bridge src/cmd_vel_to_go2_bridge_safe.cpp)
# Diagnostic tool
add_executable(cmd_vel_to_go2_bridge_diagnose src/cmd_vel_to_go2_bridge_diagnose.cpp)

if(SDK_AVAILABLE)
  target_compile_definitions(cmd_vel_to_go2_bridge PRIVATE UNITREE_AVAILABLE)
  target_compile_definitions(cmd_vel_to_go2_bridge_diagnose PRIVATE UNITREE_AVAILABLE)

  target_link_libraries(cmd_vel_to_go2_bridge
    ${catkin_LIBRARIES}
    ${UNITREE_SDK_LIB}
    ${DDSC_LIB}
    ${DDSCXX_LIB}
    Threads::Threads
  )

  target_link_libraries(cmd_vel_to_go2_bridge_diagnose
    ${catkin_LIBRARIES}
    ${UNITREE_SDK_LIB}
    ${DDSC_LIB}
    ${DDSCXX_LIB}
    Threads::Threads
  )

  message(STATUS "Building cmd_vel_to_go2_bridge WITH Unitree SDK support")
else()
  target_link_libraries(cmd_vel_to_go2_bridge
    ${catkin_LIBRARIES}
    Threads::Threads
  )
  target_link_libraries(cmd_vel_to_go2_bridge_diagnose
    ${catkin_LIBRARIES}
    Threads::Threads
  )
  message(STATUS "Building cmd_vel_to_go2_bridge WITHOUT Unitree SDK support")
endif()
