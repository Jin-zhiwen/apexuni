#!/bin/bash
# 快速测试脚本 - go2_cmd_vel_bridge

set -e

echo "========================================"
echo "GO2 cmd_vel Bridge - 快速测试"
echo "========================================"
echo ""

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Step 1: 检查编译结果
echo -e "${YELLOW}[1] 检查编译结果...${NC}"
BINARY_PATH="/home/jzw/apexuni/ApexNav/devel/lib/go2_cmd_vel_bridge/cmd_vel_to_go2_bridge"
if [ -f "$BINARY_PATH" ]; then
    echo -e "${GREEN}✓ 可执行文件存在${NC}"
    echo "  $BINARY_PATH"
    ls -lh "$BINARY_PATH"
else
    echo -e "${RED}✗ 可执行文件不存在，请先编译${NC}"
    exit 1
fi
echo ""

# Step 2: 检查网络连接
echo -e "${YELLOW}[2] 检查网络连接...${NC}"
echo "检查网口 enp2s0..."
if ip link show | grep -q "enp2s0"; then
    echo -e "${GREEN}✓ 网口 enp2s0 存在${NC}"
    ifconfig enp2s0 | grep "inet " || echo "  (未分配IP或需要配置)"
else
    echo -e "${RED}✗ 网口 enp2s0 不存在${NC}"
    echo "  可用网口:"
    ip link show | grep "^[0-9]" | awk '{print "  "$2}' | sort -u
    exit 1
fi
echo ""

# Step 3: 检查 GO2 连接
echo -e "${YELLOW}[3] 检查 GO2 机器人连接...${NC}"
echo "Ping 192.168.123.161 (3次)..."
if ping -c 3 -W 1 192.168.123.161 > /dev/null 2>&1; then
    echo -e "${GREEN}✓ GO2 机器人可达${NC}"
else
    echo -e "${RED}✗ GO2 机器人不可达${NC}"
    echo "  请检查网络连接"
    exit 1
fi
echo ""

# Step 4: 检查 ROS
echo -e "${YELLOW}[4] 检查 ROS 环境...${NC}"
source /home/jzw/apexuni/ApexNav/devel/setup.bash > /dev/null 2>&1
if command -v roscore &> /dev/null; then
    echo -e "${GREEN}✓ ROS 环境已配置${NC}"
else
    echo -e "${RED}✗ ROS 环境未配置${NC}"
    exit 1
fi
echo ""

# Step 5: 显示启动命令
echo -e "${YELLOW}[5] 显示启动命令${NC}"
echo ""
echo "选项 A - 使用 launch 文件启动 (推荐):"
echo "  source /home/jzw/apexuni/ApexNav/devel/setup.bash"
echo "  roslaunch go2_cmd_vel_bridge cmd_vel_bridge.launch \\"
echo "    network_interface:=enp2s0 domain_id:=0"
echo ""
echo "选项 B - 直接运行可执行文件:"
echo "  source /home/jzw/apexuni/ApexNav/devel/setup.bash"
echo "  rosrun go2_cmd_vel_bridge cmd_vel_to_go2_bridge \\"
echo "    _network_interface:=enp2s0 _domain_id:=0"
echo ""

# Step 6: 验证依赖库
echo -e "${YELLOW}[6] 验证依赖库...${NC}"
echo "检查 DDS 库..."
if ldd "$BINARY_PATH" | grep -q "libddsc"; then
    echo -e "${GREEN}✓ DDS 库链接正确${NC}"
else
    echo -e "${YELLOW}⚠ 未找到 DDS 库引用${NC}"
fi
echo ""

# 总结
echo "========================================"
echo -e "${GREEN}✓ 所有检查通过！${NC}"
echo "========================================"
echo ""
echo "接下来:"
echo "1. 打开新终端，运行: roscore"
echo "2. 打开另一个终端，运行上面的启动命令"
echo "3. 在第三个终端发送测试命令:"
echo "   rostopic pub /cmd_vel geometry_msgs/Twist"
echo "   \"linear: {x: 0.2, y: 0.0, z: 0.0} angular: {x: 0.0, y: 0.0, z: 0.3}\""
echo ""
