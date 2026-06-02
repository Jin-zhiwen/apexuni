#!/usr/bin/env python3
"""
Camera Pose TF Converter (Fixed for grayscale map issue)
Converts base_link odometry to camera_color_optical_frame odometry using TF transforms.

问题修复：处理四元数w为负数的情况，避免180度翻转导致的点云倒置

This node:
1. Subscribes to odom (base_link w.r.t odom frame)
2. Queries TF for base_link -> camera_color_optical_frame transform
3. Publishes camera_color_optical_frame pose w.r.t odom frame
4. Normalizes quaternion to w >= 0 to avoid inversion issues
"""

import rospy
import tf2_ros
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, TransformStamped
import numpy as np
from scipy.spatial.transform import Rotation


class CameraPoseTfConverter:
    def __init__(self):
        rospy.init_node('camera_pose_tf_converter', anonymous=False)
        
        # TF listener and buffer
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Parameters
        self.base_frame = rospy.get_param('~base_frame', 'base_link')
        self.camera_frame = rospy.get_param('~camera_frame', 'camera_color_optical_frame')
        self.odom_frame = rospy.get_param('~odom_frame', 'odom')
        self.output_topic = rospy.get_param('~output_topic', '/camera/sensor_pose')
        self.input_topic = rospy.get_param('~input_topic', '/odom')
        self.tf_timeout = rospy.get_param('~tf_timeout', 0.5)
        
        # Subscribers and Publishers
        self.odom_sub = rospy.Subscriber(self.input_topic, Odometry, self.odom_callback, queue_size=10)
        self.camera_pose_pub = rospy.Publisher(self.output_topic, Odometry, queue_size=10)
        
        rospy.loginfo(f"Camera Pose TF Converter (Fixed) initialized")
        rospy.loginfo(f"Input: {self.input_topic} ({self.base_frame} w.r.t {self.odom_frame})")
        rospy.loginfo(f"Output: {self.output_topic} ({self.camera_frame} w.r.t {self.odom_frame})")
        rospy.loginfo(f"TF: {self.base_frame} -> {self.camera_frame}")
        
    def odom_callback(self, msg):
        """
        Convert base_link odometry to camera_color_optical_frame odometry
        """
        try:
            # Get the transform from base_link to camera_color_optical_frame
            tf_transform = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.camera_frame,
                rospy.Time(0),  # Get latest transform
                timeout=rospy.Duration(self.tf_timeout)
            )
            
            # Extract transform data
            trans = tf_transform.transform.translation
            rot = tf_transform.transform.rotation
            
            # Get base_link position and orientation in odom frame
            base_pos = np.array([
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z
            ])
            
            base_quat = np.array([
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
                msg.pose.pose.orientation.w
            ])
            
            # Get base_link->camera transform
            camera_offset = np.array([trans.x, trans.y, trans.z])
            camera_rot_quat = np.array([rot.x, rot.y, rot.z, rot.w])
            
            # Convert quaternions to rotation matrices
            base_rot = Rotation.from_quat(base_quat)
            camera_offset_rot = Rotation.from_quat(camera_rot_quat)
            
            # Compute camera position in odom frame
            # P_camera_odom = P_base_odom + R_base_odom * P_camera_base
            camera_pos_odom = base_pos + base_rot.apply(camera_offset)
            
            # Compute camera orientation in odom frame
            # R_camera_odom = R_base_odom * R_camera_base
            camera_rot_odom = base_rot * camera_offset_rot
            camera_quat_odom = camera_rot_odom.as_quat()  # [x, y, z, w]

            # === FIX: Normalize quaternion to ensure w >= 0 ===
            # 如果w < 0，翻转整个四元数以得到等价的短路径旋转
            # 这避免了180度翻转导致的点云倒置问题
            if camera_quat_odom[3] < 0:
                rospy.logwarn_throttle(5.0, "Quaternion w < 0 detected, flipping to positive!")
                camera_quat_odom = -camera_quat_odom
            
            # Create output Odometry message
            camera_odom = Odometry()
            camera_odom.header.stamp = msg.header.stamp
            camera_odom.header.frame_id = self.odom_frame
            camera_odom.child_frame_id = self.camera_frame
            
            # Set pose
            camera_odom.pose.pose.position.x = camera_pos_odom[0]
            camera_odom.pose.pose.position.y = camera_pos_odom[1]
            camera_odom.pose.pose.position.z = camera_pos_odom[2]
            
            camera_odom.pose.pose.orientation.x = camera_quat_odom[0]
            camera_odom.pose.pose.orientation.y = camera_quat_odom[1]
            camera_odom.pose.pose.orientation.z = camera_quat_odom[2]
            camera_odom.pose.pose.orientation.w = camera_quat_odom[3]

            # Optional: Copy covariance from base_link (can be improved)
            camera_odom.pose.covariance = msg.pose.covariance

            # Odometry twist is typically expressed in child_frame_id (base_frame here).
            # Rotate it into the output child frame so downstream controllers do not
            # see zero / frame-inconsistent body velocities.
            base_linear = np.array([
                msg.twist.twist.linear.x,
                msg.twist.twist.linear.y,
                msg.twist.twist.linear.z,
            ])
            base_angular = np.array([
                msg.twist.twist.angular.x,
                msg.twist.twist.angular.y,
                msg.twist.twist.angular.z,
            ])
            camera_rot_from_base = camera_offset_rot.inv()
            camera_linear = camera_rot_from_base.apply(base_linear)
            camera_angular = camera_rot_from_base.apply(base_angular)

            camera_odom.twist.twist.linear.x = camera_linear[0]
            camera_odom.twist.twist.linear.y = camera_linear[1]
            camera_odom.twist.twist.linear.z = camera_linear[2]
            camera_odom.twist.twist.angular.x = camera_angular[0]
            camera_odom.twist.twist.angular.y = camera_angular[1]
            camera_odom.twist.twist.angular.z = camera_angular[2]
            camera_odom.twist.covariance = msg.twist.covariance
            
            self.camera_pose_pub.publish(camera_odom)
            
        except tf2_ros.TransformException as e:
            rospy.logwarn_throttle(2.0, f"Transform lookup failed: {e}")
        except Exception as e:
            rospy.logerr(f"Error in odom_callback: {e}")


def main():
    converter = CameraPoseTfConverter()
    rospy.spin()


if __name__ == '__main__':
    main()
