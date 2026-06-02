#!/usr/bin/env python3

import rospy
from nav_msgs.msg import Odometry


class FastLioOdomRepublisher:
    def __init__(self):
        self.input_topic = rospy.get_param("~input_topic", "/Odometry")
        self.output_topic = rospy.get_param("~output_topic", "/odom")
        self.frame_id = rospy.get_param("~frame_id", "odom")
        self.child_frame_id = rospy.get_param("~child_frame_id", "base_link")

        self.pub = rospy.Publisher(self.output_topic, Odometry, queue_size=20)
        self.sub = rospy.Subscriber(self.input_topic, Odometry, self.callback, queue_size=100)

    def callback(self, msg: Odometry):
        out = Odometry()
        out.header = msg.header
        out.child_frame_id = msg.child_frame_id
        out.pose = msg.pose
        out.twist = msg.twist

        out.header.frame_id = self.frame_id
        out.child_frame_id = self.child_frame_id

        self.pub.publish(out)


if __name__ == "__main__":
    rospy.init_node("fastlio_odom_republisher")
    FastLioOdomRepublisher()
    rospy.spin()
