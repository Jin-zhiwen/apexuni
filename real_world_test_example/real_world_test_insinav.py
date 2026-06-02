#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

from basic_utils.object_point_cloud_utils.object_point_cloud import get_object_point_cloud
from insinav_stop_gate import (
    compensate_frame_age_for_clock_offset,
    crop_image_by_roi,
    estimate_mask_distance,
    filter_positive_detection_results,
    frame_timing_status,
    locked_stop_gate_status,
    lightglue_candidate_passes,
    resolve_similar_answers,
    select_goal_object_crop,
    stop_gate_status_from_config,
    summarize_mask_depth,
    update_lightglue_approach_lock,
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
    matching_image: Optional[np.ndarray] = None
    label_candidates: List[str] = field(default_factory=list)
    primary_label: Optional[str] = None
    similar_answers: List[str] = field(default_factory=list)
    room_hint: Optional[str] = None
    fusion_score: float = 0.0


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

def compute_dino_scores(
    rgb_image: np.ndarray,
    object_masks_list: List[np.ndarray],
    clip_client: DINOSimilarity,
    use_candidate_crops: bool = True,
    crop_padding_ratio: float = 0.1,
    topk: int = 1,
    no_candidate_mode: str = "low_score",
    no_candidate_score: float = 0.0,
) -> float:
    clip_scores = []

    if use_candidate_crops and len(object_masks_list) > 0:
        for object_mask in object_masks_list:
            crop = crop_from_mask(rgb_image, object_mask, crop_padding_ratio)
            if crop is not None:
                clip_scores.append(float(clip_client.cosine(crop)))

    if len(clip_scores) > 0:
        clip_scores_sorted = sorted(clip_scores, reverse=True)
        k = min(max(1, topk), len(clip_scores_sorted))
        cosine = float(np.mean(clip_scores_sorted[:k]))

        rospy.loginfo_throttle(
            1.0,
            "[INSiNav] DINO crop similarity: %.3f, top%d/%d, max=%.3f",
            cosine,
            k,
            len(clip_scores_sorted),
            max(clip_scores_sorted),
        )
        return cosine

    if no_candidate_mode == "full_frame":
        cosine = float(clip_client.cosine(rgb_image))
        rospy.loginfo_throttle(
            1.0,
            "[INSiNav] DINO full-frame similarity fallback: %.3f",
            cosine,
        )
        return cosine

    rospy.loginfo_throttle(
        1.0,
        "[INSiNav] no object candidates; using no_candidate_score=%.3f",
        no_candidate_score,
    )
    return float(no_candidate_score)


def infer_goal_categories_from_image(
    goal_image: np.ndarray,
    detector_cfg,
    yolo_client: YOLOv7Client,
    fallback_label: Optional[str] = None,
    topk: int = 3,
) -> List[str]:
    try:
        detections = yolo_client.predict(
            goal_image,
            agnostic_nms=detector_cfg.yolo.agnostic_nms,
            conf_thres=detector_cfg.yolo.confidence_threshold_yolo,
            iou_thres=detector_cfg.yolo.iou_threshold_yolo,
        )
    except Exception as exc:
        rospy.logwarn(
            "[INSiNav] goal-label inference failed: %s; fallback_label=%s",
            exc,
            fallback_label,
        )
        return [fallback_label] if fallback_label else []

    num_det = len(detections.logits)
    if num_det == 0:
        rospy.logwarn("[INSiNav] no YOLO detection on goal image")
        return [fallback_label] if fallback_label else []

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

    rospy.loginfo(
        "[INSiNav] goal label candidates: %s",
        ", ".join([f"{label}:{score:.3f}" for label, score in candidates]),
    )

    return [label for label, _ in candidates]


def inverse_habitat_publisher_transform(sensor_pose_msg: Odometry, camera_height: float):
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

        self.has_goal_image = False
        self.has_goal_label = False
        self.processing = False  # Initialize before subscribing to topics

        self.goal_image_path = cfg.goal.get("image_path", "")
        self.goal_label = cfg.goal.get("goal_label", "")
        self.goal_topk = max(1, int(cfg.insinav_filter.get("goal_topk", 3)))
        self.goal_infer_label_enabled = bool(cfg.goal.get("infer_label", True))

        self.camera_height = float(cfg.habitat_sensor.get("camera_height", 0.88))
        self.depth_min = float(cfg.habitat_sensor.get("depth_min", 0.05))
        self.depth_max = float(cfg.habitat_sensor.get("depth_max", 10.0))
        self.depth_scale = float(cfg.habitat_sensor.get("depth_scale", 1.0))
        self.depth_normalized = bool(cfg.habitat_sensor.get("depth_normalized", False))
        self.depth_max_range = float(
            cfg.habitat_sensor.get("depth_max_range", self.depth_max)
        )
        self.depth_max_effective = (
            self.depth_max_range if self.depth_normalized else self.depth_max
        )

        if self.goal_image_path:
            self.goal_image_topic = ""
        else:
            self.goal_image_topic = cfg.goal.get("goal_image_topic", "") or "/insinav/goal_image"

        self.rgb_topic = cfg.get("ros_topics", {}).get("rgb_topic", "/camera/color/image_raw")
        self.depth_topic = cfg.get("ros_topics", {}).get("depth_topic", "/camera/depth/image_rect_raw")
        self.sensor_pose_topic = cfg.get("ros_topics", {}).get("sensor_pose_topic", "/kimera_vio_ros/odometry")
        self.odom_topic = cfg.get("ros_topics", {}).get("odom_topic", "/kimera_vio_ros/odometry")
        self.subscriber_queue_size = max(
            1, int(cfg.get("subscriber_queue_size", 1))
        )
        self.sync_queue_size = max(1, int(cfg.get("sync_queue_size", 2)))
        self.max_frame_age = float(cfg.get("max_frame_age", 0.5))
        self.max_sensor_pose_dt = float(cfg.get("max_sensor_pose_dt", 0.1))
        self.max_rgb_depth_dt = float(cfg.get("max_rgb_depth_dt", 0.08))
        self.clock_offset_tolerance = float(cfg.get("clock_offset_tolerance", 900.0))
        self.frame_clock_offset = None

        self.rgb_sub_ = message_filters.Subscriber(
            self.rgb_topic, RosImage, queue_size=self.subscriber_queue_size
        )
        self.depth_sub_ = message_filters.Subscriber(
            self.depth_topic, RosImage, queue_size=self.subscriber_queue_size
        )
        self.sensor_pose_sub_ = message_filters.Subscriber(
            self.sensor_pose_topic, Odometry, queue_size=self.subscriber_queue_size
        )

        sync_slop = float(cfg.get("sync_slop", 0.05))
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub_, self.depth_sub_, self.sensor_pose_sub_],
            queue_size=self.sync_queue_size,
            slop=sync_slop,
        )
        self.sync.registerCallback(self.sync_callback)

        rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback, queue_size=10)

        self.goal_image_sub = None
        if self.goal_image_topic:
            self.goal_image_sub = rospy.Subscriber(
                self.goal_image_topic,
                RosImage,
                self.goal_image_callback,
                queue_size=1,
            )

        self.confidence_threshold_pub_ = rospy.Publisher(
            "/detector/confidence_threshold", Float64, queue_size=10
        )
        self.itm_score_pub_ = rospy.Publisher(
            "/blip2/cosine_score", Float64, queue_size=10
        )
        self.cld_with_score_pub_ = rospy.Publisher(
            "/detector/clouds_with_scores", MultipleMasksWithConfidence, queue_size=10
        )
        self.detect_img_pub_ = rospy.Publisher(
            "/detector/detect_img", RosImage, queue_size=10
        )
        self.goal_img_pub_ = rospy.Publisher(
            "/insinav/goal_image", RosImage, queue_size=1, latch=True
        )
        self.goal_label_pub_ = rospy.Publisher(
            "/detector/label", String, queue_size=1, latch=True
        )

        self.lightglue_match_points_pub_ = rospy.Publisher(
            "/insinav/lightglue_match_points", Float64, queue_size=10
        )
        self.insinav_stop_verified_pub_ = rospy.Publisher(
            "/insinav/stop_verified", String, queue_size=1
        )
        self.target_distance_pub_ = rospy.Publisher(
            "/insinav/target_distance", Float64, queue_size=10
        )

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
        self.lightglue_stop_confirm_count = 0
        self.lightglue_stop_confirm_frames = 1
        self.lightglue_target_locked = False
        self.lightglue_target_lost_frames = 0
        self.last_locked_target_distance = float("inf")

        if self.lightglue_cfg is not None and self.lightglue_cfg.get("enabled", False):
            self.lightglue_verifier = LightGlueVerifier(
                device=self.lightglue_cfg.get("device", "cuda"),
                max_num_keypoints=self.lightglue_cfg.get("max_num_keypoints", 1024),
            )
            self.lightglue_stop_confirm_frames = max(
                1, int(self.lightglue_cfg.get("stop_confirm_frames", 2))
            )

        self.yolo_client = YOLOv7Client(port=int(cfg.goal.get("yolo_port", 12184)))

        self.insinav_filter_enabled = bool(cfg.insinav_filter.get("enabled", False))
        self.insinav_use_similar_set = bool(
            cfg.insinav_filter.get("use_similar_set", True)
        )
        self.no_detection_patience = max(
            1, int(cfg.insinav_filter.get("no_detection_patience", 8))
        )
        self.no_detection_steps = 0

        self.dino_use_candidate_crops = bool(cfg.clip.get("use_candidate_crops", True))
        self.dino_topk = max(1, int(cfg.clip.get("topk", 1)))
        self.dino_crop_padding = float(cfg.clip.get("crop_padding_ratio", 0.1))
        self.dino_no_candidate_mode = str(cfg.clip.get("no_candidate_mode", "low_score"))
        self.dino_no_candidate_score = float(cfg.clip.get("no_candidate_score", 0.0))
        self.use_goal_object_crop_for_matching = bool(
            self.lightglue_cfg.get("use_object_crop_for_matching", True)
            if self.lightglue_cfg is not None
            else True
        )
        self.goal_crop_padding = float(
            self.lightglue_cfg.get("goal_crop_padding_ratio", self.dino_crop_padding)
            if self.lightglue_cfg is not None
            else self.dino_crop_padding
        )
        self.goal_crop_roi = []
        self.max_goal_auto_crop_area_ratio = 0.90
        if self.lightglue_cfg is not None:
            roi_cfg = self.lightglue_cfg.get("goal_crop_roi", [])
            self.goal_crop_roi = list(roi_cfg) if roi_cfg is not None else []
            self.max_goal_auto_crop_area_ratio = float(
                self.lightglue_cfg.get("max_goal_auto_crop_area_ratio", 0.90)
            )

        self.robot_odom = None
        self.odom_stamp = None

        if self.goal_label:
            self._set_goal_label(self.goal_label, reason="configured goal_label")

        if self.goal_image_path:
            self._load_goal_image_from_path(self.goal_image_path)

        rospy.Timer(rospy.Duration(1.0), self.publish_confidence_threshold)

        rospy.loginfo("[INSiNav] subscribed rgb=%s", self.rgb_topic)
        rospy.loginfo("[INSiNav] subscribed depth=%s", self.depth_topic)
        rospy.loginfo("[INSiNav] subscribed sensor_pose=%s", self.sensor_pose_topic)
        rospy.loginfo("[INSiNav] subscribed odom=%s", self.odom_topic)
        rospy.loginfo("[INSiNav] goal_image_path=%s", self.goal_image_path)
        rospy.loginfo("[INSiNav] goal_image_topic=%s", self.goal_image_topic)

    def _set_goal_label(self, label: str, reason: str):
        if not label:
            return

        self.goal.primary_label = label
        self.goal.label_candidates = [label]
        self.has_goal_label = True

        self.goal_label_pub_.publish(String(data=label))
        self._refresh_llm_answer(label)

        rospy.loginfo("[INSiNav] goal label set to '%s' (%s)", label, reason)

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
        self._update_goal_image_state(reason=f"loaded from file {image_path}")

    def goal_image_callback(self, msg: RosImage):
        try:
            self.goal.image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            self._update_goal_image_state(reason=f"received from topic {self.goal_image_topic}")
        except Exception as exc:
            rospy.logerr("[INSiNav] failed to parse goal image: %s", exc)

    def _update_goal_image_state(self, reason: str):
        if self.goal.image is None:
            return

        self.has_goal_image = True

        try:
            goal_msg = self.bridge.cv2_to_imgmsg(self.goal.image, encoding="rgb8")
            self.goal_img_pub_.publish(goal_msg)
        except Exception as exc:
            rospy.logwarn("[INSiNav] failed to publish goal image: %s", exc)

        rospy.loginfo(
            "[INSiNav] goal image ready (%s), shape=%s",
            reason,
            self.goal.image.shape,
        )

        if not self.has_goal_label:
            goal_label_candidates = []

            if self.goal_infer_label_enabled:
                goal_label_candidates = infer_goal_categories_from_image(
                    self.goal.image,
                    self.config.detector,
                    self.yolo_client,
                    fallback_label=None,
                    topk=self.goal_topk,
                )

            if goal_label_candidates:
                self._set_goal_label(
                    goal_label_candidates[0],
                    reason="inferred from goal image",
                )
                self.goal.label_candidates = goal_label_candidates
            else:
                rospy.logwarn(
                    "[INSiNav] goal label could not be inferred from goal image"
                )

        self._set_goal_matching_image(reason)

    def _set_goal_matching_image(self, reason: str):
        if self.goal.image is None:
            return

        matching_image = self.goal.image
        primary_label = self.goal.primary_label

        if self.use_goal_object_crop_for_matching and primary_label:
            if self.goal_crop_roi:
                crop, meta = crop_image_by_roi(self.goal.image, self.goal_crop_roi)
                if crop is not None:
                    matching_image = crop
                    rospy.loginfo(
                        "[INSiNav] goal matching image uses manual ROI box=%s area=%.3f shape=%s (%s)",
                        str(meta.get("box", [])),
                        float(meta.get("area_ratio", 0.0)),
                        matching_image.shape,
                        reason,
                    )
                else:
                    rospy.logwarn(
                        "[INSiNav] invalid goal_crop_roi=%s; trying automatic goal crop",
                        str(self.goal_crop_roi),
                    )

            if matching_image is self.goal.image:
                try:
                    detections = self.yolo_client.predict(
                        self.goal.image,
                        agnostic_nms=self.config.detector.yolo.agnostic_nms,
                        conf_thres=self.config.detector.yolo.confidence_threshold_yolo,
                        iou_thres=self.config.detector.yolo.iou_threshold_yolo,
                    )
                    crop, meta = select_goal_object_crop(
                        self.goal.image,
                        detections.boxes,
                        detections.phrases,
                        detections.logits,
                        target_label=primary_label,
                        padding_ratio=self.goal_crop_padding,
                        max_area_ratio=self.max_goal_auto_crop_area_ratio,
                    )
                    if crop is not None:
                        matching_image = crop
                        rospy.loginfo(
                            "[INSiNav] goal matching image uses %s auto crop idx=%s score=%.3f box=%s area=%.3f shape=%s (%s)",
                            meta.get("label", primary_label),
                            str(meta.get("idx", "?")),
                            float(meta.get("score", 0.0)),
                            str(meta.get("box", [])),
                            float(meta.get("area_ratio", 0.0)),
                            matching_image.shape,
                            reason,
                        )
                    else:
                        rospy.logwarn(
                            "[INSiNav] no usable '%s' crop found in goal image; using full goal image for matching (auto crop area limit=%.2f)",
                            primary_label,
                            self.max_goal_auto_crop_area_ratio,
                        )
                except Exception as exc:
                    rospy.logwarn(
                        "[INSiNav] failed to crop goal image for matching: %s; using full goal image",
                        exc,
                    )
        else:
            rospy.loginfo(
                "[INSiNav] goal matching image uses full goal image shape=%s (%s)",
                matching_image.shape,
                reason,
            )

        if matching_image is self.goal.image and self.use_goal_object_crop_for_matching and primary_label:
            rospy.logwarn(
                "[INSiNav] goal matching image remains full frame; consider setting lightglue.goal_crop_roi or using a tighter goal image"
            )

        self.goal.matching_image = matching_image
        self.clip_client.set_goal_image(matching_image)

        if self.lightglue_verifier is not None and self.lightglue_verifier.available:
            self.lightglue_verifier.set_goal_image(matching_image)

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

            rospy.loginfo(
                "[INSiNav] LLM answer for '%s': similar=%s, room=%s, fusion=%.3f",
                label,
                self.goal.similar_answers,
                self.goal.room_hint,
                self.goal.fusion_score,
            )

        except Exception as exc:
            rospy.logwarn("[INSiNav] failed to read LLM answers for '%s': %s", label, exc)

    def _current_goal_label(self) -> Optional[str]:
        if self.goal.primary_label:
            return self.goal.primary_label
        if self.goal.label_candidates:
            return self.goal.label_candidates[0]
        return None

    def _publish_detect_img(self, detect_img: np.ndarray, rgb_header):
        if detect_img is None:
            return

        try:
            # Keep detect_img as RGB.
            # If the displayed image becomes red/blue swapped, then change encoding to bgr8 instead.
            msg = self.bridge.cv2_to_imgmsg(detect_img, encoding="rgb8")
            msg.header = rgb_header
            self.detect_img_pub_.publish(msg)

            rospy.loginfo_throttle(
                1.0,
                "[INSiNav] published /detector/detect_img, shape=%s",
                detect_img.shape,
            )

        except Exception as exc:
            rospy.logerr("[INSiNav] failed to publish detect_img: %s", exc)

    def _apply_lightglue_gate(self, rgb_cv, depth_img, object_masks_list, score_list, label_list):
        max_match_points = 0.0
        best_distance = float("inf")
        selected_match_points = 0.0
        selected_idx = -1

        if (
            self.lightglue_verifier is None
            or not self.lightglue_verifier.available
            or len(object_masks_list) == 0
            or len(score_list) != len(object_masks_list)
        ):
            self.lightglue_match_points_pub_.publish(Float64(data=0.0))
            self.target_distance_pub_.publish(Float64(data=-1.0))
            if self.lightglue_target_locked:
                approach_lock_max_lost_frames = max(
                    0, int(self.lightglue_cfg.get("approach_lock_max_lost_frames", 3))
                )
                approach_lock_max_distance = float(
                    self.lightglue_cfg.get(
                        "approach_lock_max_distance",
                        max(float(self.lightglue_cfg.get("stop_distance", 0.7)), 1.5),
                    )
                )
                self.lightglue_target_locked, self.lightglue_target_lost_frames = (
                    update_lightglue_approach_lock(
                        locked=self.lightglue_target_locked,
                        lost_frames=self.lightglue_target_lost_frames,
                        acquire_ok=False,
                        tracking_ok=False,
                        best_distance=self.last_locked_target_distance,
                        max_lost_frames=approach_lock_max_lost_frames,
                        max_distance=approach_lock_max_distance,
                    )
                )
            if self.lightglue_target_locked:
                status = locked_stop_gate_status(
                    best_distance=self.last_locked_target_distance,
                    stop_distance=float(self.lightglue_cfg.get("stop_distance", 0.7)),
                    stop_enter_distance=self.lightglue_cfg.get("stop_enter_distance", None),
                    stop_exit_distance=self.lightglue_cfg.get("stop_exit_distance", None),
                )
                self.insinav_stop_verified_pub_.publish(String(data=status))
            else:
                self.lightglue_stop_confirm_count = 0
                self.last_locked_target_distance = float("inf")
                self.insinav_stop_verified_pub_.publish(String(data="PENDING"))
            return score_list, max_match_points, best_distance

        min_match_points = max(1, int(self.lightglue_cfg.get("min_match_points", 60)))
        candidate_min_match_points = max(
            1, int(self.lightglue_cfg.get("candidate_min_match_points", min_match_points))
        )
        score_threshold = float(self.lightglue_cfg.get("score_threshold", 60.0))
        stop_distance = float(self.lightglue_cfg.get("stop_distance", 0.7))
        hard_gate = bool(self.lightglue_cfg.get("hard_gate", True))
        crop_padding_ratio = float(self.lightglue_cfg.get("crop_padding_ratio", 0.1))
        distance_percentile = float(self.lightglue_cfg.get("distance_percentile", 50.0))
        min_inlier_points = float(self.lightglue_cfg.get("min_inlier_points", 0.0))
        min_inlier_ratio = float(self.lightglue_cfg.get("min_inlier_ratio", 0.0))
        candidate_min_inlier_points = float(
            self.lightglue_cfg.get("candidate_min_inlier_points", min_inlier_points)
        )
        candidate_min_inlier_ratio = float(
            self.lightglue_cfg.get("candidate_min_inlier_ratio", min_inlier_ratio)
        )
        approach_min_match_points = float(
            self.lightglue_cfg.get(
                "approach_min_match_points", min(candidate_min_match_points, score_threshold)
            )
        )
        approach_min_inlier_points = float(
            self.lightglue_cfg.get(
                "approach_min_inlier_points", min(candidate_min_inlier_points, min_inlier_points)
            )
        )
        approach_min_inlier_ratio = float(
            self.lightglue_cfg.get(
                "approach_min_inlier_ratio", min(candidate_min_inlier_ratio, min_inlier_ratio)
            )
        )
        approach_lock_max_lost_frames = max(
            0, int(self.lightglue_cfg.get("approach_lock_max_lost_frames", 3))
        )
        approach_lock_max_distance = float(
            self.lightglue_cfg.get("approach_lock_max_distance", max(stop_distance, 1.5))
        )
        ransac_reproj_threshold = float(
            self.lightglue_cfg.get("ransac_reproj_threshold", 5.0)
        )

        lightglue_points = []
        lightglue_inliers = []
        lightglue_inlier_ratios = []
        mask_distances = []
        mask_depth_summaries = []

        for object_mask in object_masks_list:
            crop = crop_from_mask(rgb_cv, object_mask, crop_padding_ratio)

            if crop is None:
                match_points = 0.0
                inlier_points = 0.0
                inlier_ratio = 0.0
            else:
                try:
                    match_stats = self.lightglue_verifier.match_stats(
                        crop,
                        ransac_reproj_threshold=ransac_reproj_threshold,
                    )
                    match_points = float(match_stats.get("matches", 0.0))
                    inlier_points = float(match_stats.get("inliers", 0.0))
                    inlier_ratio = float(match_stats.get("inlier_ratio", 0.0))
                except Exception as exc:
                    rospy.logwarn("[INSiNav] LightGlue failed on crop: %s", exc)
                    match_points = 0.0
                    inlier_points = 0.0
                    inlier_ratio = 0.0

            dist = estimate_mask_distance(
                depth_img,
                object_mask,
                min_depth=self.depth_min,
                max_depth=self.depth_max_effective,
                percentile=distance_percentile,
            )

            lightglue_points.append(match_points)
            lightglue_inliers.append(inlier_points)
            lightglue_inlier_ratios.append(inlier_ratio)
            mask_distances.append(dist)
            mask_depth_summaries.append(
                summarize_mask_depth(
                    depth_img,
                    object_mask,
                    min_depth=self.depth_min,
                    max_depth=self.depth_max_effective,
                )
            )

        if len(lightglue_points) > 0:
            max_match_points = max(lightglue_points)

            # Prefer target-label candidates if label_list uses 0 for the primary target.
            candidate_indices = [
                i for i in range(min(len(label_list), len(lightglue_points)))
                if label_list[i] == 0
            ]

            if not candidate_indices:
                candidate_indices = list(range(len(lightglue_points)))

            # Use the same candidate for both match and distance so the stop gate
            # does not mix a far textured object with a nearby low-texture one.
            selected_idx = max(
                candidate_indices,
                key=lambda i: (
                    lightglue_inliers[i],
                    lightglue_inlier_ratios[i],
                    lightglue_points[i],
                ),
            )
            selected_match_points = lightglue_points[selected_idx]
            selected_inlier_points = lightglue_inliers[selected_idx]
            selected_inlier_ratio = lightglue_inlier_ratios[selected_idx]
            best_distance = mask_distances[selected_idx]

            rospy.loginfo_throttle(
                1.0,
                "[INSiNav] LightGlue: raw_max=%.0f, selected_idx=%d, selected_match=%.0f, inliers=%.0f, ratio=%.2f, best_dist=%.3f m, stop_dist=%.3f m",
                max_match_points,
                selected_idx,
                selected_match_points,
                selected_inlier_points,
                selected_inlier_ratio,
                best_distance if np.isfinite(best_distance) else -1.0,
                stop_distance,
            )
            depth_summary = mask_depth_summaries[selected_idx]
            rospy.loginfo_throttle(
                1.0,
                "[INSiNav][DEPTH_DEBUG] selected_idx=%d rgb_shape=%s depth_shape=%s mask_shape=%s shape_match=%s mask_px=%d valid_px=%d valid_ratio=%.2f depth_m[min/p10/p30/p50/p90/max]=[%.3f %.3f %.3f %.3f %.3f %.3f]",
                selected_idx,
                tuple(rgb_cv.shape[:2]),
                tuple(np.squeeze(depth_img).shape[:2]),
                tuple(object_masks_list[selected_idx].shape[:2]),
                str(depth_summary["shape_match"]),
                depth_summary["mask_pixels"],
                depth_summary["valid_pixels"],
                depth_summary["valid_ratio"],
                depth_summary["min"],
                depth_summary["p10"],
                depth_summary["p30"],
                depth_summary["p50"],
                depth_summary["p90"],
                depth_summary["max"],
            )
        else:
            selected_inlier_points = 0.0
            selected_inlier_ratio = 0.0

        acquire_ok = lightglue_candidate_passes(
            selected_match_points,
            selected_inlier_points,
            selected_inlier_ratio,
            score_threshold,
            min_inlier_points,
            min_inlier_ratio,
        )
        tracking_ok = lightglue_candidate_passes(
            selected_match_points,
            selected_inlier_points,
            selected_inlier_ratio,
            approach_min_match_points,
            approach_min_inlier_points,
            approach_min_inlier_ratio,
        )
        self.lightglue_target_locked, self.lightglue_target_lost_frames = (
            update_lightglue_approach_lock(
                locked=self.lightglue_target_locked,
                lost_frames=self.lightglue_target_lost_frames,
                acquire_ok=acquire_ok,
                tracking_ok=tracking_ok,
                best_distance=best_distance,
                max_lost_frames=approach_lock_max_lost_frames,
                max_distance=approach_lock_max_distance,
            )
        )
        if self.lightglue_target_locked and np.isfinite(best_distance):
            self.last_locked_target_distance = best_distance
        elif not self.lightglue_target_locked:
            self.last_locked_target_distance = float("inf")

        if hard_gate:
            accepted_candidates = 0
            for idx, base_score in enumerate(score_list):
                if idx >= len(lightglue_points):
                    continue
                candidate_ok = lightglue_candidate_passes(
                    lightglue_points[idx],
                    lightglue_inliers[idx],
                    lightglue_inlier_ratios[idx],
                    candidate_min_match_points,
                    candidate_min_inlier_points,
                    candidate_min_inlier_ratio,
                )
                keep_locked_selected = (
                    self.lightglue_target_locked
                    and idx == selected_idx
                    and lightglue_candidate_passes(
                        lightglue_points[idx],
                        lightglue_inliers[idx],
                        lightglue_inlier_ratios[idx],
                        approach_min_match_points,
                        approach_min_inlier_points,
                        approach_min_inlier_ratio,
                    )
                )
                if candidate_ok or keep_locked_selected:
                    accepted_candidates += 1
                else:
                    score_list[idx] = 0.0
            rospy.loginfo_throttle(
                1.0,
                "[INSiNav] LightGlue candidate gate: accepted=%d/%d, require match>=%.0f inliers>=%.0f ratio>=%.2f",
                accepted_candidates,
                len(lightglue_points),
                candidate_min_match_points,
                candidate_min_inlier_points,
                candidate_min_inlier_ratio,
            )
            rospy.loginfo_throttle(
                1.0,
                "[INSiNav] LightGlue approach lock: active=%s lost=%d/%d tracking>=%.0f/%.0f/%.2f dist<=%.2f",
                str(self.lightglue_target_locked),
                self.lightglue_target_lost_frames,
                approach_lock_max_lost_frames,
                approach_min_match_points,
                approach_min_inlier_points,
                approach_min_inlier_ratio,
                approach_lock_max_distance,
            )

        self.lightglue_match_points_pub_.publish(Float64(data=selected_match_points))
        self.target_distance_pub_.publish(
            Float64(data=best_distance if np.isfinite(best_distance) else -1.0)
        )

        status, self.lightglue_stop_confirm_count = stop_gate_status_from_config(
            selected_match_points,
            best_distance,
            self.lightglue_cfg,
            self.lightglue_stop_confirm_count,
            selected_inlier_points=selected_inlier_points,
            selected_inlier_ratio=selected_inlier_ratio,
            approach_lock_active=self.lightglue_target_locked,
        )

        if status == "VERIFIED":
            self.insinav_stop_verified_pub_.publish(String(data="VERIFIED"))
            rospy.logwarn_throttle(
                1.0,
                "[INSiNav] STOP VERIFIED: match=%.0f >= %.0f, inliers=%.0f >= %.0f, ratio=%.2f >= %.2f, dist=%.3f <= %.3f, confirm=%d/%d",
                selected_match_points,
                score_threshold,
                selected_inlier_points,
                min_inlier_points,
                selected_inlier_ratio,
                min_inlier_ratio,
                best_distance,
                stop_distance,
                self.lightglue_stop_confirm_count,
                self.lightglue_stop_confirm_frames,
            )
        elif status == "PENDING_FAR":
            self.insinav_stop_verified_pub_.publish(String(data="PENDING_FAR"))
            rospy.loginfo_throttle(
                1.0,
                "[INSiNav] LightGlue verified but target is still far: match=%.0f, inliers=%.0f, ratio=%.2f, dist=%.3f, required<=%.3f",
                selected_match_points,
                selected_inlier_points,
                selected_inlier_ratio,
                best_distance if np.isfinite(best_distance) else -1.0,
                stop_distance,
            )
        elif status == "PENDING_CLOSE":
            self.insinav_stop_verified_pub_.publish(String(data="PENDING_CLOSE"))
            rospy.loginfo_throttle(
                1.0,
                "[INSiNav] STOP candidate pending: match=%.0f >= %.0f, inliers=%.0f >= %.0f, ratio=%.2f >= %.2f, dist=%.3f <= %.3f, confirm=%d/%d",
                selected_match_points,
                score_threshold,
                selected_inlier_points,
                min_inlier_points,
                selected_inlier_ratio,
                min_inlier_ratio,
                best_distance,
                stop_distance,
                self.lightglue_stop_confirm_count,
                self.lightglue_stop_confirm_frames,
            )
        else:
            self.insinav_stop_verified_pub_.publish(String(data="PENDING"))
            if selected_match_points >= score_threshold and np.isfinite(best_distance):
                match_ok = selected_match_points >= score_threshold
                inliers_ok = selected_inlier_points >= min_inlier_points
                ratio_ok = selected_inlier_ratio >= min_inlier_ratio
                distance_ok = best_distance <= stop_distance
                rospy.loginfo_throttle(
                    1.0,
                    "[INSiNav] STOP candidate pending: match=%.0f/%.0f(%s), inliers=%.0f/%.0f(%s), ratio=%.2f/%.2f(%s), dist=%.3f/%.3f(%s), confirm=%d/%d",
                    selected_match_points,
                    score_threshold,
                    "ok" if match_ok else "fail",
                    selected_inlier_points,
                    min_inlier_points,
                    "ok" if inliers_ok else "fail",
                    selected_inlier_ratio,
                    min_inlier_ratio,
                    "ok" if ratio_ok else "fail",
                    best_distance,
                    stop_distance,
                    "ok" if distance_ok else "fail",
                    self.lightglue_stop_confirm_count,
                    self.lightglue_stop_confirm_frames,
                )

        return score_list, max_match_points, best_distance

    def sync_callback(self, rgb_msg, depth_msg, sensor_pose_msg):
        rospy.loginfo_throttle(1.0, "[INSiNav] sync_callback triggered")

        if self.processing:
            rospy.loginfo_throttle(1.0, "[INSiNav] still processing previous frame")
            return

        self.processing = True

        try:
            stamp = rgb_msg.header.stamp
            raw_frame_age = max(0.0, (rospy.Time.now() - stamp).to_sec())
            frame_age, self.frame_clock_offset, used_clock_offset = (
                compensate_frame_age_for_clock_offset(
                    raw_frame_age,
                    self.frame_clock_offset,
                    self.max_frame_age,
                    self.clock_offset_tolerance,
                )
            )
            if used_clock_offset:
                rospy.logwarn_throttle(
                    5.0,
                    "[INSiNav] compensating camera/ROS clock offset: raw_age=%.3f effective_age=%.3f offset=%.3f",
                    raw_frame_age,
                    frame_age,
                    self.frame_clock_offset,
                )
            sensor_pose_dt = abs((stamp - sensor_pose_msg.header.stamp).to_sec())
            rgb_depth_dt = abs((rgb_msg.header.stamp - depth_msg.header.stamp).to_sec())
            timing_ok, timing_reason = frame_timing_status(
                frame_age,
                sensor_pose_dt,
                rgb_depth_dt,
                self.max_frame_age,
                self.max_sensor_pose_dt,
                self.max_rgb_depth_dt,
            )
            if not timing_ok:
                rospy.logwarn_throttle(
                    1.0,
                    "[INSiNav] drop stale/desynced frame: reason=%s age=%.3f rgb_sensor_dt=%.3f rgb_depth_dt=%.3f",
                    timing_reason,
                    frame_age,
                    sensor_pose_dt,
                    rgb_depth_dt,
                )
                return

            goal_label = self._current_goal_label()
            if goal_label is None:
                rospy.logwarn_throttle(
                    5.0,
                    "[INSiNav] waiting for goal label; no detection will be published",
                )
                return

            rgb_cv = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="rgb8")
            depth_raw = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
            depth_raw = depth_raw.astype(np.float32)
            if self.depth_normalized:
                depth_img = depth_raw * self.depth_max_range
            else:
                depth_img = depth_raw * self.depth_scale
            depth_cv = np.expand_dims(depth_img, axis=-1)

            finite_depth = depth_img[np.isfinite(depth_img)]
            finite_depth = finite_depth[
                (finite_depth > self.depth_min) & (finite_depth < self.depth_max_effective)
            ]
            if finite_depth.size > 0:
                depth_min_dbg = float(np.min(finite_depth))
                depth_med_dbg = float(np.percentile(finite_depth, 50.0))
                depth_max_dbg = float(np.max(finite_depth))
            else:
                depth_min_dbg = depth_med_dbg = depth_max_dbg = float("nan")
            rospy.loginfo_throttle(
                2.0,
                "[INSiNav][DEPTH_DEBUG] rgb=%s/%s depth=%s/%s encoding=%s scale=%.6f normalized=%s rgb_depth_dt=%.3f global_depth_m[min/median/max]=[%.3f %.3f %.3f]",
                tuple(rgb_cv.shape[:2]),
                rgb_msg.header.frame_id,
                tuple(np.squeeze(depth_img).shape[:2]),
                depth_msg.header.frame_id,
                depth_msg.encoding,
                self.depth_scale,
                str(self.depth_normalized),
                rgb_depth_dt,
                depth_min_dbg,
                depth_med_dbg,
                depth_max_dbg,
            )

            similar_answer = resolve_similar_answers(
                self.insinav_filter_enabled,
                self.insinav_use_similar_set,
                self.goal.similar_answers,
            )

            rospy.loginfo_throttle(
                1.0,
                "[INSiNav] get_object label=%s, similar=%s",
                goal_label,
                similar_answer,
            )

            detect_img, score_list, object_masks_list, label_list = get_object(
                goal_label,
                rgb_cv,
                self.config.detector,
                similar_answer,
            )

            rospy.loginfo_throttle(
                1.0,
                "[INSiNav] get_object returned %d objects",
                len(object_masks_list),
            )

            # 关键：检测图像必须立刻发布，不要等点云 / LightGlue / DINO
            self._publish_detect_img(detect_img, rgb_msg.header)

            # LightGlue 只用于过滤 score 和发布 stop verification，不应该阻塞 detect_img
            if self.has_goal_image:
                score_list, max_match_points, best_distance = self._apply_lightglue_gate(
                rgb_cv,
                depth_img,
                object_masks_list,
                score_list,
                label_list,
            )
            else:
                self.lightglue_match_points_pub_.publish(Float64(data=0.0))
                self.insinav_stop_verified_pub_.publish(String(data="NO_GOAL_IMAGE"))

            # 点云生成放在 detect_img 和 LightGlue 之后
            obj_point_cloud_list = []
            try:
                gps, compass = inverse_habitat_publisher_transform(
                    sensor_pose_msg,
                    self.camera_height,
                )
                observations = {
                    "depth": depth_cv,
                    "gps": gps,
                    "compass": compass,
                }

                obj_point_cloud_list = get_object_point_cloud(
                    self.config,
                    observations,
                    object_masks_list,
                )

            except Exception as exc:
                rospy.logwarn_throttle(
                    1.0,
                    "[INSiNav] failed to build object point clouds: %s",
                    exc,
                )

            obj_point_cloud_list, score_list, label_list = filter_positive_detection_results(
                obj_point_cloud_list,
                score_list,
                label_list,
            )

            if obj_point_cloud_list:
                cld_with_score_msg = MultipleMasksWithConfidence()
                cld_with_score_msg.point_clouds = obj_point_cloud_list
                cld_with_score_msg.confidence_scores = score_list
                cld_with_score_msg.label_indices = label_list
                self.cld_with_score_pub_.publish(cld_with_score_msg)

                rospy.loginfo_throttle(
                    1.0,
                    "[INSiNav] published /detector/clouds_with_scores, num=%d",
                    len(obj_point_cloud_list),
                )
            else:
                rospy.loginfo_throttle(
                    1.0,
                    "[INSiNav] no positive detection clouds to publish after gating",
                )

            # DINO 只有在 goal image 存在时才有意义
            if self.has_goal_image:
                try:
                    cosine = compute_dino_scores(
                        rgb_cv,
                        object_masks_list,
                        self.clip_client,
                        use_candidate_crops=self.dino_use_candidate_crops,
                        crop_padding_ratio=self.dino_crop_padding,
                        topk=self.dino_topk,
                        no_candidate_mode=self.dino_no_candidate_mode,
                        no_candidate_score=self.dino_no_candidate_score,
                    )

                    self.itm_score_pub_.publish(Float64(data=float(cosine)))

                    rospy.loginfo_throttle(
                        1.0,
                        "[INSiNav] published /blip2/cosine_score: %.3f",
                        cosine,
                    )

                except Exception as exc:
                    rospy.logerr("[INSiNav] DINO similarity failed: %s", exc)

            self.no_detection_steps = 0 if len(object_masks_list) > 0 else self.no_detection_steps + 1

            if self.no_detection_steps >= self.no_detection_patience:
                rospy.logwarn_throttle(
                    2.0,
                    "[INSiNav] no valid detections for %d frames; goal_label=%s",
                    self.no_detection_steps,
                    goal_label,
                )

        except Exception as exc:
            rospy.logerr("[INSiNav] sync callback failed: %s", exc)

        finally:
            self.processing = False

    def odom_callback(self, msg: Odometry):
        self.robot_odom = msg
        self.odom_stamp = msg.header.stamp

    def publish_confidence_threshold(self, event):
        self.confidence_threshold_pub_.publish(Float64(data=0.5))

    def run(self):
        rospy.loginfo("[INSiNav] real-world bridge running")
        rospy.spin()


@hydra.main(version_base=None, config_path="config", config_name="real_world_test_insinav")
def main(cfg: DictConfig):
    node = RealWorldINSiNavNode(cfg)
    node.run()


if __name__ == "__main__":
    main()
