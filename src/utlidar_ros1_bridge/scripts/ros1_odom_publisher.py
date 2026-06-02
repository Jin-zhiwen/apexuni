#!/usr/bin/env python3

import json
import socket

import rospy
from geometry_msgs.msg import Quaternion, Point, Pose, Twist, Vector3
from nav_msgs.msg import Odometry


def to_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class Ros1OdomPublisher:
    def __init__(self):
        self.listen_host = rospy.get_param("~listen_host", "0.0.0.0")
        self.listen_port = int(rospy.get_param("~listen_port", 25100))
        self.output_topic = rospy.get_param("~output_topic", "/kimera_vio_ros/body_odometry")
        self.frame_id = rospy.get_param("~frame_id", "odom")
        self.child_frame_id = rospy.get_param("~child_frame_id", "base_link")

        self.pub = rospy.Publisher(self.output_topic, Odometry, queue_size=20)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.listen_host, self.listen_port))
        self.sock.settimeout(0.5)

    def run(self):
        while not rospy.is_shutdown():
            try:
                data, _ = self.sock.recvfrom(65535)
            except socket.timeout:
                continue

            try:
                payload = json.loads(data.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue

            msg = Odometry()
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = payload.get("frame_id", self.frame_id)
            msg.child_frame_id = payload.get("child_frame_id", self.child_frame_id)

            pose = payload.get("pose", {})
            pos = pose.get("position", {})
            ori = pose.get("orientation", {})

            msg.pose.pose = Pose(
                position=Point(
                    x=to_float(pos.get("x")),
                    y=to_float(pos.get("y")),
                    z=to_float(pos.get("z")),
                ),
                orientation=Quaternion(
                    x=to_float(ori.get("x")),
                    y=to_float(ori.get("y")),
                    z=to_float(ori.get("z")),
                    w=to_float(ori.get("w"), 1.0),
                ),
            )

            twist = payload.get("twist", {})
            lin = twist.get("linear", {})
            ang = twist.get("angular", {})
            msg.twist.twist = Twist(
                linear=Vector3(
                    x=to_float(lin.get("x")),
                    y=to_float(lin.get("y")),
                    z=to_float(lin.get("z")),
                ),
                angular=Vector3(
                    x=to_float(ang.get("x")),
                    y=to_float(ang.get("y")),
                    z=to_float(ang.get("z")),
                ),
            )

            self.pub.publish(msg)


if __name__ == "__main__":
    rospy.init_node("utlidar_ros1_odom_publisher")
    Ros1OdomPublisher().run()
