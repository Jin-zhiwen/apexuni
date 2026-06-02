#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Broadcast TF from camera depth optical frame pose.

Input:
    /camera/sensor_pose
        nav_msgs/Odometry
        header.frame_id = odom
        child_frame_id  = camera_depth_optical_frame

Output TF:
    odom -> camera_depth_optical_frame

This node is used to make RViz / mapping / TF tree know the dynamic pose of
camera_depth_optical_frame in odom frame.
"""

import rospy
import tf2_ros
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped


class CameraDepthPoseToTf:
    def __init__(self):
        rospy.init_node("camera_depth_pose_to_tf", anonymous=False)

        self.input_topic = rospy.get_param("~input_topic", "/camera/sensor_pose")
        self.parent_frame = rospy.get_param("~parent_frame", "odom")
        self.child_frame = rospy.get_param("~child_frame", "camera_depth_optical_frame")
        self.use_msg_frame = rospy.get_param("~use_msg_frame", False)

        self.br = tf2_ros.TransformBroadcaster()

        self.sub = rospy.Subscriber(
            self.input_topic,
            Odometry,
            self.odom_callback,
            queue_size=50,
        )

        rospy.loginfo("========================================")
        rospy.loginfo("[camera_depth_pose_to_tf] initialized")
        rospy.loginfo("[camera_depth_pose_to_tf] input_topic  : %s", self.input_topic)
        rospy.loginfo("[camera_depth_pose_to_tf] parent_frame : %s", self.parent_frame)
        rospy.loginfo("[camera_depth_pose_to_tf] child_frame  : %s", self.child_frame)
        rospy.loginfo("[camera_depth_pose_to_tf] output TF    : %s -> %s",
                      self.parent_frame, self.child_frame)
        rospy.loginfo("========================================")

    def odom_callback(self, msg):
        tf_msg = TransformStamped()

        tf_msg.header.stamp = msg.header.stamp

        if self.use_msg_frame:
            tf_msg.header.frame_id = msg.header.frame_id
            tf_msg.child_frame_id = msg.child_frame_id
        else:
            tf_msg.header.frame_id = self.parent_frame
            tf_msg.child_frame_id = self.child_frame

        tf_msg.transform.translation.x = msg.pose.pose.position.x
        tf_msg.transform.translation.y = msg.pose.pose.position.y
        tf_msg.transform.translation.z = msg.pose.pose.position.z

        tf_msg.transform.rotation.x = msg.pose.pose.orientation.x
        tf_msg.transform.rotation.y = msg.pose.pose.orientation.y
        tf_msg.transform.rotation.z = msg.pose.pose.orientation.z
        tf_msg.transform.rotation.w = msg.pose.pose.orientation.w

        self.br.sendTransform(tf_msg)

        rospy.loginfo_throttle(
            2.0,
            "[camera_depth_pose_to_tf] broadcasting TF: %s -> %s",
            tf_msg.header.frame_id,
            tf_msg.child_frame_id,
        )


def main():
    CameraDepthPoseToTf()
    rospy.spin()


if __name__ == "__main__":
    main()

