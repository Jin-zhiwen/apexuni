#!/usr/bin/env python3

import argparse
import json
import socket

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node


class Ros2UtlidarToUdp(Node):
    def __init__(self, input_topic: str, target_host: str, target_port: int):
        super().__init__("ros2_utlidar_to_udp")
        self.input_topic = input_topic
        self.target_host = target_host
        self.target_port = target_port

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sub = self.create_subscription(Odometry, self.input_topic, self.callback, 10)

    def callback(self, msg: Odometry):
        payload = {
            "frame_id": msg.header.frame_id,
            "child_frame_id": msg.child_frame_id,
            "pose": {
                "position": {
                    "x": msg.pose.pose.position.x,
                    "y": msg.pose.pose.position.y,
                    "z": msg.pose.pose.position.z,
                },
                "orientation": {
                    "x": msg.pose.pose.orientation.x,
                    "y": msg.pose.pose.orientation.y,
                    "z": msg.pose.pose.orientation.z,
                    "w": msg.pose.pose.orientation.w,
                },
            },
            "twist": {
                "linear": {
                    "x": msg.twist.twist.linear.x,
                    "y": msg.twist.twist.linear.y,
                    "z": msg.twist.twist.linear.z,
                },
                "angular": {
                    "x": msg.twist.twist.angular.x,
                    "y": msg.twist.twist.angular.y,
                    "z": msg.twist.twist.angular.z,
                },
            },
        }
        self.sock.sendto(json.dumps(payload).encode("utf-8"), (self.target_host, self.target_port))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--topic", default=None)
    args, _ = parser.parse_known_args()

    rclpy.init()
    temp_node = Node("ros2_utlidar_to_udp_params")
    input_topic = args.topic or temp_node.declare_parameter("input_topic", "/utlidar/robot_odom").value
    target_host = args.host or temp_node.declare_parameter("target_host", "127.0.0.1").value
    target_port = args.port or int(temp_node.declare_parameter("target_port", 25100).value)
    temp_node.destroy_node()

    node = Ros2UtlidarToUdp(input_topic, target_host, target_port)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
