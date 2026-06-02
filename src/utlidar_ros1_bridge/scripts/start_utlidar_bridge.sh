#!/usr/bin/env bash
set -euo pipefail

ROS1_SETUP="${ROS1_SETUP:-/home/jzw/apexuni/ApexNav/devel/setup.bash}"
GO2_HOST="${GO2_HOST:-unitree@192.168.123.18}"
ROS2_SETUP="${ROS2_SETUP:-/opt/ros/foxy/setup.bash}"
UDP_HOST="${UDP_HOST:-0.0.0.0}"
UDP_PORT="${UDP_PORT:-25100}"
ROS1_TOPIC="${ROS1_TOPIC:-/kimera_vio_ros/body_odometry}"
ROS2_TOPIC="${ROS2_TOPIC:-/utlidar/robot_odom}"
LOCAL_UDP_HOST="${LOCAL_UDP_HOST:-192.168.123.222}"

cleanup() {
  jobs -p | xargs -r kill 2>/dev/null || true
}
trap cleanup EXIT INT TERM

source /opt/ros/noetic/setup.bash
source "${ROS1_SETUP}"

if ! rostopic list >/dev/null 2>&1; then
  roscore >/tmp/utlidar_ros1_bridge_roscore.log 2>&1 &
  sleep 2
fi

rosrun utlidar_ros1_bridge ros1_odom_publisher.py \
  _listen_host:="${UDP_HOST}" \
  _listen_port:="${UDP_PORT}" \
  _output_topic:="${ROS1_TOPIC}" &

ssh -o StrictHostKeyChecking=no "${GO2_HOST}" \
  "source ${ROS2_SETUP} >/dev/null 2>&1 && python3 ~/ros2_utlidar_to_udp.py --host '${LOCAL_UDP_HOST}' --port '${UDP_PORT}' --topic '${ROS2_TOPIC}'" &

wait
