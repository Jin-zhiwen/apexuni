#!/bin/bash
# Build script for go2_cmd_vel_bridge

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

echo "[go2_cmd_vel_bridge] Build script"
echo "  Workspace: ${WORKSPACE_ROOT}"
echo "  Script dir: ${SCRIPT_DIR}"

# Check if catkin is sourced
if [ -z "$CMAKE_PREFIX_PATH" ]; then
    echo "[ERROR] ROS setup not sourced. Please run 'source /opt/ros/<distro>/setup.bash'"
    exit 1
fi

# Create build directory if it doesn't exist
BUILD_DIR="${WORKSPACE_ROOT}/build"
if [ ! -d "$BUILD_DIR" ]; then
    mkdir -p "$BUILD_DIR"
fi

# Build using catkin
cd "$WORKSPACE_ROOT"
echo "[go2_cmd_vel_bridge] Running catkin_make for go2_cmd_vel_bridge..."
catkin_make --pkg go2_cmd_vel_bridge -DCMAKE_BUILD_TYPE=Release

echo "[go2_cmd_vel_bridge] Build complete!"
echo ""
echo "To run the bridge:"
echo "  source ${WORKSPACE_ROOT}/devel/setup.bash"
echo "  roslaunch go2_cmd_vel_bridge cmd_vel_bridge.launch network_interface:=enp2s0 domain_id:=0"
