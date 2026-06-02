# GO2 cmd_vel Bridge

## 概述

将ROS `cmd_vel` (`geometry_msgs/Twist`) 命令转换为Unitree GO2 SDK的`SportClient::Move()`调用。

**完整工作流：**
1. **轨迹规划器** (`traj_server.cpp`) → 发布 `cmd_vel`
2. **本桥接器** → 订阅 `cmd_vel` → 调用GO2 SDK
3. **GO2机器人** → 执行运动命令

## 构建

```bash
cd /home/jzw/apexuni/ApexNav
source devel/setup.bash
cd src/go2_cmd_vel_bridge
chmod +x build.sh
./build.sh
```

或直接：
```bash
cd /home/jzw/apexuni/ApexNav
catkin_make --pkg go2_cmd_vel_bridge
```

## 运行

```bash
source devel/setup.bash
roslaunch go2_cmd_vel_bridge cmd_vel_bridge.launch network_interface:=enp2s0 domain_id:=0
```

## 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `network_interface` | `enp2s0` | 网口名称（用于DDS通信） |
| `domain_id` | `0` | DDS域ID（0或1） |
| `max_vx` | `0.6` | 最大前进速度 (m/s) |
| `max_vy` | `0.4` | 最大横向速度 (m/s) |
| `max_yaw` | `1.2` | 最大角速度 (rad/s) |
| `cmd_timeout` | `0.5` | 命令超时时间 (s)，超时后停止 |
| `publish_rate` | `50.0` | 命令发布频率 (Hz) |
| `auto_stand` | `true` | 启动时自动让机器人站起 |

## 常见用法

### 1. 使用默认参数
```bash
roslaunch go2_cmd_vel_bridge cmd_vel_bridge.launch
```

### 2. 指定网口和域ID
```bash
roslaunch go2_cmd_vel_bridge cmd_vel_bridge.launch \
  network_interface:=enp2s0 \
  domain_id:=0
```

### 3. 增加速度限制
```bash
roslaunch go2_cmd_vel_bridge cmd_vel_bridge.launch \
  max_vx:=1.0 \
  max_vy:=0.8 \
  max_yaw:=1.5
```

### 4. 不自动站起
```bash
roslaunch go2_cmd_vel_bridge cmd_vel_bridge.launch auto_stand:=false
```

## 诊断

### 检查网口连接
```bash
ping 192.168.123.161  # GO2默认IP
```

### 查看运行日志
```bash
roslaunch go2_cmd_vel_bridge cmd_vel_bridge.launch
# 或
rosbag record /cmd_vel  # 记录命令话题
```

### 手动测试
```bash
# 在另一终端发布测试命令
rostopic pub /cmd_vel geometry_msgs/Twist "linear: {x: 0.2, y: 0.0, z: 0.0} angular: {x: 0.0, y: 0.0, z: 0.3}"
```

## 集成到INSiNav

完整的INSiNav实机流程：

```
INSiNav检测器
    ↓
轨迹规划器 (traj_server) → cmd_vel
    ↓
go2_cmd_vel_bridge → SportClient::Move()
    ↓
GO2机器人
```

## 文件结构

```
go2_cmd_vel_bridge/
├── CMakeLists.txt
├── package.xml
├── build.sh
├── launch/
│   └── cmd_vel_bridge.launch
├── src/
│   └── cmd_vel_to_go2_bridge.cpp
└── README.md
```
