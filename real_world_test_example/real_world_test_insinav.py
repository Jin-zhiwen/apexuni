#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""INSiNav real-world deployment bridge.

This node adapts the existing ApexNav real-world perception pipeline to the
instance-image-navigation workflow:
- it accepts a goal image from a file or ROS topic,
- infers a target label candidate list for detector compatibility,
- publishes the same perception outputs used by the current real-world stack,
- and optionally enables LightGlue verification on goal-conditioned crops.

The motion side is still handled by the existing ROS launch files and the
trajectory manager; this script focuses on the perception / goal-conditioning
part that needs to differ from object-goal deployment.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional

import hydra
import message_filters
import numpy as np
import rospy
import tf.transformations as tft
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
from omegaconf import DictConfig
from PIL import Image
from sensor_msgs.msg import Image as RosImage
from std_msgs.msg import Float64, String

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from basic_utils.object_point_cloud_utils.object_point_cloud import (
    get_object_point_cloud,
)
from llm.answer_reader.answer_reader import read_answer
from plan_env.msg import MultipleMasksWithConfidence
from vlm.detector.yolov7 import YOLOv7Client
from vlm.itm.dino_similarity import DINOSimilarity
from vlm.itm.lightglue_verifier import LightGlueVerifier
from vlm.utils.get_object_utils import get_object


@dataclass
class GoalState:
    image: Optional[np.ndarray] = None
    label_candidates: List[str] = field(default_factory=list)
    primary_label: Optional[str] = None
    similar_answers: List[str] = field(default_factory=list)
    room_hint: Optional[str] = None
    fusion_score: float = 0.0
    clean_crop: Optional[np.ndarray] = None


def crop_from_mask(rgb_image: np.ndarray, object_mask: np.ndarray, padding_ratio: float = 0.1):
    ys, xs = np.where(object_mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    height, width = rgb_image.shape[:2]

    box_width = x2 - x1 + 1
    box_height = y2 - y1 + 1
    pad_x = int(box_width * padding_ratio)
    pad_y = int(box_height * padding_ratio)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(width - 1, x2 + pad_x)
    y2 = min(height - 1, y2 + pad_y)

    if x2 <= x1 or y2 <= y1:
        return None
    return rgb_image[y1 : y2 + 1, x1 : x2 + 1]


def infer_goal_categories_from_image(
    goal_image: np.ndarray,
    detector_cfg,
    yolo_client: YOLOv7Client,
    fallback_label: Optional[str] = None,
    topk: int = 3,
) -> List[str]:
    """Infer target label candidates from a goal image using YOLO detections."""
    try:
        detections = yolo_client.predict(
            goal_image,
            agnostic_nms=detector_cfg.yolo.agnostic_nms,
            conf_thres=detector_cfg.yolo.confidence_threshold_yolo,
            iou_thres=detector_cfg.yolo.iou_threshold_yolo,
        )
    except Exception as exc:
        rospy.logwarn(
            "[INSiNav] goal-label inference failed (%s); fallback_label=%s",
            exc,
            fallback_label,
        )
        return [fallback_label] if fallback_label else []

    num_det = len(detections.logits)
    if num_det == 0:
        if fallback_label:
            rospy.logwarn(
                "[INSiNav] no detection on goal image; fallback_label=%s", fallback_label
            )
            return [fallback_label]
        return []

    ranked = []
    for idx in range(num_det):
        x1, y1, x2, y2 = [float(v) for v in detections.boxes[idx]]
        score = float(detections.logits[idx].item())
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        center_dist = float(np.sqrt((cx - 0.5) ** 2 + (cy - 0.5) ** 2))
        rank = score + 0.2 * area - 0.3 * center_dist
        ranked.append((rank, score, detections.phrases[idx]))

    ranked.sort(key=lambda item: item[0], reverse=True)
    candidates = []
    seen = set()
    for _, score, label in ranked:
        if label in seen:
            continue
        seen.add(label)
        candidates.append((label, score))
        if len(candidates) >= max(1, int(topk)):
            break

    if not candidates:
        return [fallback_label] if fallback_label else []

    candidate_msg = ", ".join([f"{label}:{score:.3f}" for label, score in candidates])
    rospy.loginfo(
        "[INSiNav] goal label candidates inferred by YOLO: [%s], total_det=%d",
        candidate_msg,
        num_det,
    )
    return [label for label, _ in candidates]


def inverse_habitat_publisher_transform(sensor_pose_msg: Odometry, camera_height: float):
    """Recover Habitat-style GPS and compass from ROS sensor pose."""
    pos = sensor_pose_msg.pose.pose.position
    orn = sensor_pose_msg.pose.pose.orientation

    gps = np.array([-pos.y, pos.z - camera_height, -pos.x], dtype=np.float32)
    euler = tft.euler_from_quaternion([orn.x, orn.y, orn.z, orn.w])
    compass = np.array([euler[2] + np.pi / 2.0], dtype=np.float32)
    return gps, compass


class RealWorldINSiNavNode:
    def __init__(self, cfg):
        self.config = cfg
        rospy.init_node("real_world_insinav_node", anonymous=False)

        self.bridge = CvBridge()
        self.goal = GoalState()
        self.goal_ready = False
        self.goal_image_path = cfg.goal.get("image_path", "")
        if self.goal_image_path:
            self.goal_image_topic = ""
        else:
            self.goal_image_topic = cfg.goal.get("goal_image_topic", "") or "/habitat/goal_image"
        self.goal_label = cfg.goal.get("goal_label", "")
        self.goal_topk = max(1, int(cfg.insinav_filter.get("goal_topk", 3)))
        self.camera_height = float(cfg.habitat_sensor.get("camera_height", 0.88))
        self.goal_infer_label_enabled = bool(cfg.goal.get("infer_label", True))

        self.rgb_sub_ = message_filters.Subscriber("/habitat/camera_rgb", RosImage)
        self.depth_sub_ = message_filters.Subscriber("/habitat/camera_depth", RosImage)
        self.sensor_pose_sub_ = message_filters.Subscriber("/habitat/sensor_pose", Odometry)

        self.goal_image_sub = None
        if self.goal_image_topic:
            self.goal_image_sub = rospy.Subscriber(
                self.goal_image_topic, RosImage, self.goal_image_callback, queue_size=1
            )

        rospy.Subscriber("/habitat/odom", Odometry, self.odom_callback, queue_size=10)

        self.confidence_threshold_pub_ = rospy.Publisher(
            "/detector/confidence_threshold", Float64, queue_size=10
        )
        self.itm_score_pub_ = rospy.Publisher("/blip2/cosine_score", Float64, queue_size=10)
        self.cld_with_score_pub_ = rospy.Publisher(
            "/detector/clouds_with_scores", MultipleMasksWithConfidence, queue_size=10
        )
        self.detect_img_pub_ = rospy.Publisher("/detector/detect_img", RosImage, queue_size=10)
        self.goal_label_pub_ = rospy.Publisher("/detector/label", String, queue_size=1)

        self.sync_detect = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub_, self.depth_sub_, self.sensor_pose_sub_], queue_size=5, slop=0.01
        )
        self.sync_detect.registerCallback(self.sync_detect_callback)

        self.sync_value = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub_, self.depth_sub_, self.sensor_pose_sub_], queue_size=5, slop=0.01
        )
        self.sync_value.registerCallback(self.sync_value_callback)

        self.processing_detect = False
        self.processing_value = False
        self.robot_odom = None
        self.odom_stamp = None

        llm_cfg = self.config.get("llm", None)
        self.llm_answer_path = None
        self.llm_response_path = None
        self.llm_client = None
        if llm_cfg is not None:
            self.llm_answer_path = llm_cfg.get("llm_answer_path", None)
            self.llm_response_path = llm_cfg.get("llm_response_path", None)
            llm_client_cfg = llm_cfg.get("llm_client", None)
            if llm_client_cfg is not None:
                self.llm_client = llm_client_cfg.get("llm_client", None)

        self.clip_client = DINOSimilarity(
            model_name=cfg.clip.get("model_name", "dino_vits16"),
            device=cfg.clip.get("device", "cuda"),
        )

        self.lightglue_cfg = cfg.get("lightglue", None)
        self.lightglue_verifier = None
        if self.lightglue_cfg is not None and self.lightglue_cfg.get("enabled", False):
            self.lightglue_verifier = LightGlueVerifier(
                device=self.lightglue_cfg.get("device", "cuda"),
                max_num_keypoints=self.lightglue_cfg.get("max_num_keypoints", 1024),
            )

        self.yolo_client = YOLOv7Client(port=int(cfg.goal.get("yolo_port", 12184)))
        self.insinav_filter_enabled = bool(cfg.insinav_filter.get("enabled", False))
        self.no_detection_patience = max(
            1, int(cfg.insinav_filter.get("no_detection_patience", 8))
        )
        self.no_detection_steps = 0

        if self.goal_image_path:
            self._load_goal_image_from_path(self.goal_image_path)

        rospy.Timer(rospy.Duration(1.0), self.publish_confidence_threshold)

    def _load_goal_image_from_path(self, image_path: str):
        if not image_path:
            return
        if not os.path.isabs(image_path):
            image_path = os.path.join(parent_dir, image_path)
        if not os.path.exists(image_path):
            rospy.logwarn("[INSiNav] goal image path does not exist: %s", image_path)
            return

        pil_image = Image.open(image_path).convert("RGB")
        self.goal.image = np.array(pil_image)
        self._update_goal_state(reason=f"loaded from file {image_path}")

    def goal_image_callback(self, msg: RosImage):
        try:
            self.goal.image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            self._update_goal_state(reason=f"received from topic {self.goal_image_topic}")
        except Exception as exc:
            rospy.logerr("[INSiNav] failed to parse goal image: %s", exc)

    def _update_goal_state(self, reason: str):
        if self.goal.image is None:
            return

        rospy.loginfo("[INSiNav] goal image ready (%s)", reason)
        self.clip_client.set_goal_image(self.goal.image)

        goal_label_candidates = []
        if self.goal_label:
            goal_label_candidates = [self.goal_label]
        elif self.goal_infer_label_enabled:
            goal_label_candidates = infer_goal_categories_from_image(
                self.goal.image,
                self.config,
                self.yolo_client,
                fallback_label=None,
                topk=self.goal_topk,
            )

        self.goal.label_candidates = goal_label_candidates
        self.goal.primary_label = goal_label_candidates[0] if goal_label_candidates else None
        self.goal.clean_crop = None

        if self.goal.primary_label:
            self.goal_label_pub_.publish(String(data=self.goal.primary_label))
            self._refresh_llm_answer(self.goal.primary_label)
            rospy.loginfo(
                "[INSiNav] primary goal label set to '%s'", self.goal.primary_label
            )
            try:
                _, goal_scores, goal_masks, _ = get_object(
                    self.goal.primary_label,
                    self.goal.image,
                    self.config.detector,
                    self.goal.similar_answers if self.insinav_filter_enabled else [],
                )
                if goal_masks:
                    best_idx = int(np.argmax(goal_scores)) if goal_scores else 0
                    crop_padding_ratio = float(
                        self.lightglue_cfg.get("crop_padding_ratio", 0.1)
                    ) if self.lightglue_cfg is not None else 0.1
                    self.goal.clean_crop = crop_from_mask(
                        self.goal.image, goal_masks[best_idx], crop_padding_ratio
                    )
            except Exception as exc:
                rospy.logwarn("[INSiNav] failed to extract goal crop: %s", exc)
        else:
            rospy.logwarn(
                "[INSiNav] goal label could not be inferred; perception will still use image similarity"
            )

        if self.lightglue_verifier is not None and self.lightglue_verifier.available:
            if self.goal.clean_crop is not None:
                self.lightglue_verifier.set_goal_image(self.goal.clean_crop, goal_key="crop")
            else:
                self.lightglue_verifier.clear_goal(goal_key="crop")

        self.goal_ready = True

    def _refresh_llm_answer(self, label: str):
        self.goal.similar_answers = []
        self.goal.room_hint = None
        self.goal.fusion_score = 0.0
        if not self.llm_answer_path or not self.llm_response_path or not self.llm_client:
            return

        try:
            llm_answer, room, fusion_score = read_answer(
                self.llm_answer_path,
                self.llm_response_path,
                label,
                self.llm_client,
            )
            self.goal.similar_answers = llm_answer or []
            self.goal.room_hint = room
            self.goal.fusion_score = float(fusion_score)
        except Exception as exc:
            rospy.logwarn("[INSiNav] failed to read LLM answers for '%s': %s", label, exc)

    def _current_goal_label(self) -> Optional[str]:
        if self.goal.primary_label:
            return self.goal.primary_label
        if self.goal.label_candidates:
            return self.goal.label_candidates[0]
        if self.goal_label:
            return self.goal_label
        return None

    def sync_detect_callback(self, rgb_msg, depth_msg, sensor_pose_msg):
        if self.processing_detect:
            return
        self.processing_detect = True
        try:
            if not self.goal_ready and self.goal_label:
                self.goal.label_candidates = [self.goal_label]
                self.goal.primary_label = self.goal_label
                self.goal_label_pub_.publish(String(data=self.goal_label))
                self._refresh_llm_answer(self.goal_label)
                self.goal_ready = True

            if not self.goal_ready:
                rospy.logwarn_throttle(5.0, "[INSiNav] waiting for goal image")
                return

            stamp = rgb_msg.header.stamp
            time_diff = abs((stamp - sensor_pose_msg.header.stamp).to_sec())
            if time_diff > 0.1:
                return

            goal_label = self._current_goal_label()
            if goal_label is None:
                rospy.logwarn_throttle(
                    5.0,
                    "[INSiNav] no goal label available; skipping detector-cloud publication",
                )
                return

            rgb_cv = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="rgb8")
            depth_img = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
            depth_cv = np.expand_dims(depth_img.astype(np.float32), axis=-1)

            similar_answer = self.goal.similar_answers if self.insinav_filter_enabled else []
            detect_img, score_list, object_masks_list, label_list = get_object(
                goal_label, rgb_cv, self.config.detector, similar_answer
            )

            gps, compass = inverse_habitat_publisher_transform(
                sensor_pose_msg, self.camera_height
            )
            observations = {"depth": depth_cv, "gps": gps, "compass": compass}
            obj_point_cloud_list = get_object_point_cloud(
                self.config, observations, object_masks_list
            )

            if (
                self.lightglue_verifier is not None
                and self.lightglue_verifier.available
                and len(score_list) == len(object_masks_list)
            ):
                min_match_points = max(1, int(self.lightglue_cfg.get("min_match_points", 30)))
                blend_weight = float(self.lightglue_cfg.get("blend_weight", 0.5))
                hard_gate = bool(self.lightglue_cfg.get("hard_gate", False))
                crop_padding_ratio = float(self.lightglue_cfg.get("crop_padding_ratio", 0.1))
                lightglue_points = []
                for object_mask in object_masks_list:
                    crop = crop_from_mask(rgb_cv, object_mask, crop_padding_ratio)
                    lightglue_points.append(
                        self.lightglue_verifier.match_points(crop, goal_key="crop")
                    )

                if lightglue_points:
                    rospy.loginfo(
                        "[INSiNav] LightGlue points: max=%d mean=%.1f min=%d",
                        max(lightglue_points),
                        float(np.mean(lightglue_points)),
                        min(lightglue_points),
                    )
                for idx, base_score in enumerate(score_list):
                    match_points = lightglue_points[idx]
                    if hard_gate:
                        score_list[idx] = float(base_score) if match_points >= min_match_points else 0.0
                    else:
                        lg_score = min(1.0, match_points / float(min_match_points))
                        score_list[idx] = float((1.0 - blend_weight) * float(base_score) + blend_weight * lg_score)

            self.detect_img_pub_.publish(self.bridge.cv2_to_imgmsg(detect_img, encoding="rgb8"))

            cld_with_score_msg = MultipleMasksWithConfidence()
            cld_with_score_msg.point_clouds = obj_point_cloud_list
            cld_with_score_msg.confidence_scores = score_list
            cld_with_score_msg.label_indices = label_list
            self.cld_with_score_pub_.publish(cld_with_score_msg)

            self.no_detection_steps = 0 if len(score_list) > 0 else self.no_detection_steps + 1
            if self.no_detection_steps >= self.no_detection_patience:
                rospy.logwarn(
                    "[INSiNav] no valid detections for %d frames; keeping goal label '%s'",
                    self.no_detection_steps,
                    goal_label,
                )

        except Exception as exc:
            rospy.logerr("[INSiNav] detection callback failed: %s", exc)
        finally:
            self.processing_detect = False

    def sync_value_callback(self, rgb_msg, depth_msg, sensor_pose_msg):
        if self.processing_value:
            return
        self.processing_value = True
        try:
            if not self.goal_ready:
                return
            if self.goal.image is None:
                return

            stamp = rgb_msg.header.stamp
            time_diff = abs((stamp - sensor_pose_msg.header.stamp).to_sec())
            if time_diff > 0.1:
                return

            rgb_cv = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="rgb8")
            cosine = self.clip_client.cosine(rgb_cv)
            rospy.loginfo("[INSiNav] goal similarity score: %.3f", cosine)
            itm_score_msg = Float64()
            itm_score_msg.data = float(cosine)
            self.itm_score_pub_.publish(itm_score_msg)
        except Exception as exc:
            rospy.logerr("[INSiNav] similarity callback failed: %s", exc)
        finally:
            self.processing_value = False

    def odom_callback(self, msg: Odometry):
        self.robot_odom = msg
        self.odom_stamp = msg.header.stamp

    def publish_confidence_threshold(self, event):
        confidence_threshold_msg = Float64()
        confidence_threshold_msg.data = 0.5
        self.confidence_threshold_pub_.publish(confidence_threshold_msg)

    def run(self):
        rospy.loginfo("[INSiNav] real-world bridge running")
        rospy.spin()


@hydra.main(version_base=None, config_path="config", config_name="real_world_test_insinav")
def main(cfg: DictConfig):
    node = RealWorldINSiNavNode(cfg)
    node.run()


if __name__ == "__main__":
    main()
