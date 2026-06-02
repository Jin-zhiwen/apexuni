#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Camera Pose TF Converter

Purpose:
    Convert base_link odometry to camera optical-frame odometry.

Typical real-world mapping chain:
    /kimera_vio_ros/odometry
        base_link pose w.r.t odom
            ↓
    camera_pose_tf_converter
        camera_depth_optical_frame pose w.r.t odom
            ↓
    /camera/sensor_pose

Important:
    If mapping uses /camera/depth/image_rect_raw, camera_frame should be:
        camera_depth_optical_frame

    If mapping uses /camera/aligned_depth_to_color/image_raw, camera_frame should be:
        camera_color_optical_frame

This script avoids tf2_geometry_msgs / PyKDL dependency.
"""

import math

import numpy as np
import rospy
import tf2_ros
import tf.transformations as tft

from nav_msgs.msg import Odometry


def normalize_quaternion(q):
    """
    Normalize quaternion [x, y, z, w].
    Does NOT force w positive, because q and -q represent the same rotation.
    """
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q)

    if norm < 1e-12 or not np.isfinite(norm):
        raise ValueError("Invalid quaternion norm")

    return q / norm


def rotate_vector_by_quaternion(vec, quat):
    """
    Rotate a 3D vector using quaternion [x, y, z, w].
    """
    quat = normalize_quaternion(quat)
    rot_mat = tft.quaternion_matrix(quat)[:3, :3]
    return rot_mat.dot(vec)


class CameraPoseTfConverter:
    def __init__(self):
        rospy.init_node("camera_pose_tf_converter", anonymous=False)

        # Parameters
        self.base_frame = rospy.get_param("~base_frame", "base_link")

        # Important default:
        # Since current mapping uses /camera/depth/image_rect_raw,
        # this should be camera_depth_optical_frame, not camera_color_optical_frame.
        self.camera_frame = rospy.get_param(
            "~camera_frame",
            "camera_depth_optical_frame",
        )

        self.odom_frame = rospy.get_param("~odom_frame", "odom")
        self.input_topic = rospy.get_param("~input_topic", "/kimera_vio_ros/odometry")
        self.output_topic = rospy.get_param("~output_topic", "/camera/sensor_pose")
        self.tf_timeout = float(rospy.get_param("~tf_timeout", 0.5))

        # TF listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # ROS I/O
        self.odom_sub = rospy.Subscriber(
            self.input_topic,
            Odometry,
            self.odom_callback,
            queue_size=20,
        )

        self.camera_pose_pub = rospy.Publisher(
            self.output_topic,
            Odometry,
            queue_size=20,
        )

        rospy.loginfo("========================================")
        rospy.loginfo("[camera_pose_converter] initialized")
        rospy.loginfo("[camera_pose_converter] input_topic  : %s", self.input_topic)
        rospy.loginfo("[camera_pose_converter] output_topic : %s", self.output_topic)
        rospy.loginfo("[camera_pose_converter] base_frame   : %s", self.base_frame)
        rospy.loginfo("[camera_pose_converter] camera_frame : %s", self.camera_frame)
        rospy.loginfo("[camera_pose_converter] odom_frame   : %s", self.odom_frame)
        rospy.loginfo("[camera_pose_converter] TF used      : %s -> %s",
                      self.base_frame, self.camera_frame)
        rospy.loginfo("========================================")

    def odom_callback(self, msg):
        """
        Input:
            msg: base_link pose w.r.t odom frame

        Output:
            camera optical frame pose w.r.t odom frame

        Math:
            T_odom_camera = T_odom_base * T_base_camera
        """
        try:
            # Warn if odometry child_frame_id does not match expected base_frame.
            if msg.child_frame_id and msg.child_frame_id != self.base_frame:
                rospy.logwarn_throttle(
                    5.0,
                    "[camera_pose_converter] odom child_frame_id is '%s', "
                    "but expected base_frame is '%s'. "
                    "Please confirm Kimera odometry represents base_link.",
                    msg.child_frame_id,
                    self.base_frame,
                )

            if msg.header.frame_id and msg.header.frame_id != self.odom_frame:
                rospy.logwarn_throttle(
                    5.0,
                    "[camera_pose_converter] odom header.frame_id is '%s', "
                    "but configured odom_frame is '%s'. Output will use '%s'.",
                    msg.header.frame_id,
                    self.odom_frame,
                    self.odom_frame,
                )

            # Lookup T_base_camera.
            #
            # lookup_transform(target, source, time) returns transform from source to target.
            # Therefore:
            #   target = base_link
            #   source = camera_depth_optical_frame
            # gives:
            #   T_base_camera
            tf_transform = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.camera_frame,
                rospy.Time(0),
                timeout=rospy.Duration(self.tf_timeout),
            )

            trans = tf_transform.transform.translation
            rot = tf_transform.transform.rotation

            # T_odom_base from odometry.
            base_pos_odom = np.array(
                [
                    msg.pose.pose.position.x,
                    msg.pose.pose.position.y,
                    msg.pose.pose.position.z,
                ],
                dtype=np.float64,
            )

            base_quat_odom = normalize_quaternion(
                [
                    msg.pose.pose.orientation.x,
                    msg.pose.pose.orientation.y,
                    msg.pose.pose.orientation.z,
                    msg.pose.pose.orientation.w,
                ]
            )

            # T_base_camera from TF.
            camera_pos_base = np.array(
                [
                    trans.x,
                    trans.y,
                    trans.z,
                ],
                dtype=np.float64,
            )

            camera_quat_base = normalize_quaternion(
                [
                    rot.x,
                    rot.y,
                    rot.z,
                    rot.w,
                ]
            )

            # Position:
            #   p_odom_camera = p_odom_base + R_odom_base * p_base_camera
            camera_pos_odom = base_pos_odom + rotate_vector_by_quaternion(
                camera_pos_base,
                base_quat_odom,
            )

            # Orientation:
            #   q_odom_camera = q_odom_base * q_base_camera
            camera_quat_odom = tft.quaternion_multiply(
                base_quat_odom,
                camera_quat_base,
            )
            camera_quat_odom = normalize_quaternion(camera_quat_odom)

            # Build output Odometry.
            camera_odom = Odometry()
            camera_odom.header.stamp = msg.header.stamp
            camera_odom.header.frame_id = self.odom_frame
            camera_odom.child_frame_id = self.camera_frame

            camera_odom.pose.pose.position.x = float(camera_pos_odom[0])
            camera_odom.pose.pose.position.y = float(camera_pos_odom[1])
            camera_odom.pose.pose.position.z = float(camera_pos_odom[2])

            camera_odom.pose.pose.orientation.x = float(camera_quat_odom[0])
            camera_odom.pose.pose.orientation.y = float(camera_quat_odom[1])
            camera_odom.pose.pose.orientation.z = float(camera_quat_odom[2])
            camera_odom.pose.pose.orientation.w = float(camera_quat_odom[3])

            # Copy covariance directly for now.
            camera_odom.pose.covariance = msg.pose.covariance

            self.camera_pose_pub.publish(camera_odom)

            rospy.loginfo_throttle(
                2.0,
                "[camera_pose_converter] published %s pose in %s frame",
                self.camera_frame,
                self.odom_frame,
            )

        except tf2_ros.LookupException as exc:
            rospy.logwarn_throttle(
                2.0,
                "[camera_pose_converter] TF lookup failed: %s",
                str(exc),
            )

        except tf2_ros.ConnectivityException as exc:
            rospy.logwarn_throttle(
                2.0,
                "[camera_pose_converter] TF connectivity failed: %s",
                str(exc),
            )

        except tf2_ros.ExtrapolationException as exc:
            rospy.logwarn_throttle(
                2.0,
                "[camera_pose_converter] TF extrapolation failed: %s",
                str(exc),
            )

        except Exception as exc:
            rospy.logerr_throttle(
                1.0,
                "[camera_pose_converter] odom_callback error: %s",
                str(exc),
            )


def main():
    CameraPoseTfConverter()
    rospy.spin()


if __name__ == "__main__":
    main()
