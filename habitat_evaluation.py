"""
Habitat ObjectNav Evaluation Script for HM3D/MP3D Datasets

This script evaluates object navigation performance using the Habitat simulator
with support for HM3D-v1, HM3D-v2, and MP3D datasets. It communicates with ROS for
real-time planning and decision making, incorporates vision-language models
for object detection and image-text matching, and generates comprehensive
evaluation metrics.

Usage:
    # Run with HM3D-v1 dataset
    python habitat_evaluation.py --dataset hm3dv1

    # Run with HM3D-v2 dataset (default)
    python habitat_evaluation.py --dataset hm3dv2

    # Run with MP3D dataset
    python habitat_evaluation.py --dataset mp3d

    # Test specific episode
    python habitat_evaluation.py --dataset hm3dv2 test_epi_num=10

Author: Zager-Zhang
"""

# Standard library imports
import argparse
import gzip
import json
import os
import signal
import time
from copy import deepcopy
from PIL import Image

# Third-party library imports
from hydra import initialize, compose
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from omegaconf import DictConfig
from prettytable import PrettyTable
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Int32, Int32MultiArray, Float32MultiArray, Float64
import tqdm

# Habitat-related imports
import habitat
from habitat.config.default import patch_config
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.utils.visualizations.utils import (
    images_to_video,
    observations_to_image,
    overlay_frame,
)

# ROS message imports
from plan_env.msg import MultipleMasksWithConfidence

# Local project imports
from basic_utils.failure_check.count_files import count_files_in_directory
from basic_utils.failure_check.failure_check import check_failure, is_on_same_floor
from basic_utils.object_point_cloud_utils.object_point_cloud import (
    get_object_point_cloud,
)
from basic_utils.record_episode.read_record import read_record
from basic_utils.record_episode.write_record import write_record
from habitat2ros import habitat_publisher
from llm.answer_reader.answer_reader import read_answer
from params import HABITAT_STATE, ROS_STATE, ACTION, RESULT_TYPES
from vlm.Labels import MP3D_ID_TO_NAME
from vlm.utils.get_itm_message import get_itm_message_cosine
from vlm.itm.dino_similarity import DINOSimilarity
from vlm.itm.lightglue_verifier import LightGlueVerifier
from vlm.detector.yolov7 import YOLOv7Client
from vlm.utils.get_object_utils import get_object


def publish_int32(publisher, data):
    msg = Int32()
    msg.data = data
    publisher.publish(msg)


def publish_float64(publisher, data):
    msg = Float64()
    msg.data = data
    publisher.publish(msg)


def publish_int32_array(publisher, data_list):
    msg = Int32MultiArray()
    msg.data = data_list
    publisher.publish(msg)


def publish_float32_array(publisher, data_list):
    msg = Float32MultiArray()
    msg.data = data_list
    publisher.publish(msg)


def signal_handler(sig, frame):
    """Handle Ctrl+C signal for graceful shutdown"""
    print("Ctrl+C detected! Shutting down...")
    rospy.signal_shutdown("Manual shutdown")
    os._exit(0)


def transform_rgb_bgr(image):
    """Convert RGB image to BGR format"""
    return image[:, :, [2, 1, 0]]


def overlay_goal_thumbnail(frame, goal_image, thumb_w=200, thumb_h=150, margin=12):
    """Overlay goal image thumbnail on the top-right corner of visualization frame."""
    if frame is None or goal_image is None:
        return frame

    h, w = frame.shape[:2]
    tw = min(thumb_w, max(32, w // 3))
    th = min(thumb_h, max(24, h // 3))

    goal_thumb = np.array(Image.fromarray(goal_image).convert("RGB").resize((tw, th)))

    x1 = max(0, w - tw - margin)
    y1 = margin
    x2 = x1 + tw
    y2 = y1 + th

    frame[y1:y2, x1:x2] = goal_thumb
    return frame


def crop_from_mask(rgb_image, object_mask, padding_ratio=0.1):
    """Crop RGB patch from binary mask bounding box with optional padding."""
    ys, xs = np.where(object_mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    h, w = rgb_image.shape[:2]

    bw = x2 - x1 + 1
    bh = y2 - y1 + 1
    pad_x = int(bw * padding_ratio)
    pad_y = int(bh * padding_ratio)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w - 1, x2 + pad_x)
    y2 = min(h - 1, y2 + pad_y)

    if x2 <= x1 or y2 <= y1:
        return None
    return rgb_image[y1 : y2 + 1, x1 : x2 + 1]


def infer_goal_categories_from_image(
    goal_image,
    detector_cfg,
    yolo_client,
    fallback_label,
    topk=3,
):
    """Infer top-k goal categories from goal image using YOLO detections.

    Ranking uses confidence + area + center prior.
    """
    try:
        detections = yolo_client.predict(
            goal_image,
            agnostic_nms=detector_cfg.yolo.agnostic_nms,
            conf_thres=detector_cfg.yolo.confidence_threshold_yolo,
            iou_thres=detector_cfg.yolo.iou_threshold_yolo,
        )
    except Exception as e:
        print(
            f"[INSiNav] Goal category inference failed ({e}); fallback to episode label='{fallback_label}'"
        )
        return [fallback_label]

    num_det = len(detections.logits)
    if num_det == 0:
        print(
            f"[INSiNav] No YOLO detection on goal image; fallback to episode label='{fallback_label}'"
        )
        return [fallback_label]

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

    ranked.sort(key=lambda x: x[0], reverse=True)
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
        return [fallback_label]

    inferred_labels = [label for label, _ in candidates]
    candidate_msg = ", ".join([f"{lb}:{sc:.3f}" for lb, sc in candidates])
    print(
        f"[INSiNav] Goal label candidates inferred by YOLO: [{candidate_msg}], total_det={num_det}"
    )
    return inferred_labels


def publish_observations(event):
    """Timer callback to publish habitat observations and trigger messages"""
    global msg_observations, fusion_threshold
    global ros_pub, trigger_pub, confidence_threshold_pub
    tmp = deepcopy(msg_observations)
    ros_pub.habitat_publish_ros_topic(tmp)
    publish_float64(confidence_threshold_pub, fusion_threshold)
    trigger = PoseStamped()
    trigger_pub.publish(trigger)


def ros_action_callback(msg):
    global global_action
    global_action = msg.data


def ros_state_callback(msg):
    global ros_state
    ros_state = msg.data


def ros_final_state_callback(msg):
    global final_state
    final_state = msg.data


def ros_expl_result_callback(msg):
    global expl_result
    expl_result = msg.data


def _parse_dataset_arg():
    """Parse CLI to choose dataset and capture remaining Hydra overrides."""
    parser = argparse.ArgumentParser(
        description="Habitat ObjectNav Evaluation", add_help=True
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["hm3dv1", "hm3dv2", "mp3d", "insinav"],
        default="hm3dv2",
        help="Choose dataset: hm3dv1, hm3dv2, mp3d or insinav (default: hm3dv2)",
    )
    # Keep unknown so users can still pass Hydra-style overrides (e.g., key=value)
    args, unknown = parser.parse_known_args()
    return args.dataset, unknown


def main(cfg: DictConfig) -> None:
    global msg_observations, global_action, ros_state, fusion_threshold
    global ros_pub, trigger_pub, obj_point_cloud_pub, confidence_threshold_pub
    global final_state, expl_result

    # Navigation mode: object-goal (default) or instance-image-goal
    goal_cfg = cfg.get("goal", None)
    goal_type = goal_cfg.goal_type if goal_cfg is not None else "object"
    is_instance_imagenav = goal_type == "ins-image"
    insinav_filter_cfg = cfg.get("insinav_filter", None)
    insinav_filter_enabled = (
        bool(insinav_filter_cfg.get("enabled", False))
        if insinav_filter_cfg is not None
        else False
    )
    insinav_use_similar_set = (
        bool(insinav_filter_cfg.get("use_similar_set", True))
        if insinav_filter_cfg is not None
        else True
    )
    insinav_goal_topk = (
        max(1, int(insinav_filter_cfg.get("goal_topk", 3)))
        if insinav_filter_cfg is not None
        else 3
    )
    insinav_no_det_patience = (
        max(1, int(insinav_filter_cfg.get("no_detection_patience", 8)))
        if insinav_filter_cfg is not None
        else 8
    )
    insinav_stop_gate_enabled = (
        bool(insinav_filter_cfg.get("stop_gate_enabled", True))
        if insinav_filter_cfg is not None
        else True
    )
    insinav_stop_reject_turn_patience = (
        max(1, int(insinav_filter_cfg.get("stop_reject_turn_patience", 3)))
        if insinav_filter_cfg is not None
        else 3
    )

    category_to_coco = {}
    id_to_name = {}
    if not is_instance_imagenav:
        # Load MP3D validation data for object category mapping
        with gzip.open(
            "data/datasets/objectnav/mp3d/v1/val/val.json.gz", "rt", encoding="utf-8"
        ) as f:
            val_data = json.load(f)
        category_to_coco = val_data.get("category_to_mp3d_category_id", {})
        id_to_name = {
            category_to_coco[cat]: MP3D_ID_TO_NAME[idx]
            for idx, cat in enumerate(category_to_coco)
        }

    start_time = time.time()

    final_state = 0
    expl_result = 0
    result_list = [0] * len(RESULT_TYPES)

    cfg = patch_config(cfg)

    # Extract configuration parameters
    video_output_path = cfg.video_output_path.format(split=cfg.habitat.dataset.split)
    need_video = cfg.need_video
    record_file_path = os.path.join(video_output_path, cfg.record_file_name)
    continue_path = os.path.join(video_output_path, cfg.continue_file_name)
    max_episode_steps = cfg.habitat.environment.max_episode_steps
    success_distance = cfg.habitat.task.measurements.success.success_distance

    detector_cfg = cfg.detector

    llm_client = None
    llm_answer_path = None
    llm_response_path = None
    if not is_instance_imagenav or (is_instance_imagenav and insinav_filter_enabled):
        llm_cfg = cfg.get("llm", None)
        if llm_cfg is None:
            llm_cfg = None
        else:
            llm_client = llm_cfg.llm_client
            llm_answer_path = llm_cfg.llm_answer_path
            llm_response_path = llm_cfg.llm_response_path

    if not is_instance_imagenav and (llm_client is None or llm_answer_path is None):
        raise ValueError("ObjectNav mode requires 'llm' config section.")

    clip_client = None
    goal_yolo_client = None
    lightglue_cfg = cfg.get("lightglue", None)
    lightglue_verifier = None
    if is_instance_imagenav:
        clip_cfg = cfg.clip
        clip_client = DINOSimilarity(
            model_name=clip_cfg.model_name, device=clip_cfg.device
        )
        goal_yolo_client = YOLOv7Client(port=12184)
        print(
            f"InstanceImageNav mode enabled, DINO model={clip_cfg.model_name}, device={clip_client.device}"
        )

        if lightglue_cfg is not None and lightglue_cfg.get("enabled", False):
            lightglue_verifier = LightGlueVerifier(
                device=lightglue_cfg.get("device", "cuda"),
                max_num_keypoints=lightglue_cfg.get("max_num_keypoints", 1024),
            )
            if lightglue_verifier.available:
                print(
                    "LightGlue instance verification enabled "
                    f"(device={lightglue_verifier.device}, max_num_keypoints={lightglue_cfg.get('max_num_keypoints', 1024)})"
                )
            else:
                print(
                    "LightGlue is not available in current environment. "
                    "Falling back to CLIP-only insinav."
                )

    # Single test parameters
    env_num_once = cfg.test_epi_num  # Which episode to test for single run
    flag_once = env_num_once != -1  # Whether to run single test

    # Create directories if they don't exist
    if llm_answer_path:
        os.makedirs(os.path.dirname(llm_answer_path), exist_ok=True)
    os.makedirs(video_output_path, exist_ok=True)

    # Fallback threshold used by ROS side if LLM guidance is disabled
    fusion_threshold = cfg.get("default_fusion_threshold", 0.5)

    # Add top_down_map and collisions visualization
    with habitat.config.read_write(cfg):
        cfg.habitat.task.measurements.update(
            {
                "top_down_map": TopDownMapMeasurementConfig(
                    map_padding=3,
                    map_resolution=256,
                    draw_source=True,
                    draw_border=True,
                    draw_shortest_path=True,
                    draw_view_points=True,
                    draw_goal_positions=True,
                    draw_goal_aabbs=False,
                    fog_of_war=FogOfWarConfig(
                        draw=True,
                        visibility_dist=5.0,
                        fov=79,
                    ),
                ),
                "collisions": CollisionsMeasurementConfig(),
            }
        )

    env = habitat.Env(cfg)
    print("Environment creation successful")
    number_of_episodes = env.number_of_episodes

    # Read previous records and set initial values
    (
        num_total,
        num_success,
        spl_all,
        soft_spl_all,
        distance_to_goal_all,
        distance_to_goal_reward_all,
        last_time,
    ) = read_record(continue_path, flag_once)

    if num_total >= number_of_episodes:
        raise ValueError("Already finished all episodes.")

    pbar = tqdm.tqdm(total=env.number_of_episodes)

    env_count = num_total if not flag_once else env_num_once
    while env_count:
        pbar.update()
        env.current_episode = next(env.episode_iterator)
        env_count -= 1

    # Initialize ROS publishers, subscribers, and timers
    obj_point_cloud_pub = rospy.Publisher(
        "habitat/object_point_cloud", PointCloud2, queue_size=10
    )
    ros_pub = habitat_publisher.ROSPublisher()
    rospy.Subscriber("/habitat/plan_action", Int32, ros_action_callback, queue_size=10)
    rospy.Subscriber("/ros/state", Int32, ros_state_callback, queue_size=10)
    rospy.Subscriber("/ros/expl_state", Int32, ros_final_state_callback, queue_size=10)
    rospy.Subscriber("/ros/expl_result", Int32, ros_expl_result_callback, queue_size=10)
    state_pub = rospy.Publisher("/habitat/state", Int32, queue_size=10)
    trigger_pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=10)
    itm_score_pub = rospy.Publisher("/blip2/cosine_score", Float64, queue_size=10)
    confidence_threshold_pub = rospy.Publisher(
        "/detector/confidence_threshold", Float64, queue_size=10
    )
    cld_with_score_pub = rospy.Publisher(
        "/detector/clouds_with_scores", MultipleMasksWithConfidence, queue_size=10
    )
    progress_pub = rospy.Publisher("/habitat/progress", Int32MultiArray, queue_size=10)
    record_pub = rospy.Publisher("/habitat/record", Float32MultiArray, queue_size=10)

    for epi in range(number_of_episodes - num_total):
        # Publish progress information
        publish_int32_array(progress_pub, [num_total, number_of_episodes])

        if flag_once:
            while env_count:
                env.current_episode = next(env.episode_iterator)
                env_count -= 1

        # Initialize episode variables
        pass_object = 0.0
        near_object = 0.0
        global_action = None
        cld_with_score_msg = MultipleMasksWithConfidence()
        count_steps = 0

        camera_pitch = 0.0
        observations = env.reset()
        observations["camera_pitch"] = camera_pitch
        msg_observations = deepcopy(observations)
        del observations["camera_pitch"]
        label = env.current_episode.object_category
        target_label_for_detection = label
        goal_label_candidates = [label]
        similar_label_set = []
        no_detection_steps = 0
        insinav_stop_gate_passed = False
        insinav_stop_reject_count = 0
        pending_stop_verification = False
        stop_gate_silent_steps = 0
        episode_goal_image = None

        # Convert object category to coco name format
        if not is_instance_imagenav and label in category_to_coco:
            coco_id = category_to_coco[label]
            label = id_to_name.get(coco_id, label)

        # Get guidance for target object
        if is_instance_imagenav:
            llm_answer = []
            room = "everywhere"

            goal_image_override_path = goal_cfg.get("image_path", "")
            if goal_image_override_path:
                goal_image = np.array(Image.open(goal_image_override_path).convert("RGB"))
            elif "instance_imagegoal" in observations:
                goal_image = observations["instance_imagegoal"]
            else:
                raise KeyError(
                    "Instance-image-goal task requires 'instance_imagegoal' in observations or goal.image_path override"
                )

            if insinav_filter_enabled:
                goal_label_candidates = infer_goal_categories_from_image(
                    goal_image=goal_image,
                    detector_cfg=detector_cfg,
                    yolo_client=goal_yolo_client,
                    fallback_label=label,
                    topk=insinav_goal_topk,
                )
                target_label_for_detection = goal_label_candidates[0]

                if (
                    insinav_use_similar_set
                    and llm_client
                    and llm_answer_path
                    and llm_response_path
                ):
                    similar_label_set = []
                    for candidate_label in goal_label_candidates:
                        try:
                            llm_answer, _, _ = read_answer(
                                llm_answer_path,
                                llm_response_path,
                                candidate_label,
                                llm_client,
                            )
                            for item in llm_answer:
                                if item not in similar_label_set:
                                    similar_label_set.append(item)
                        except Exception as e:
                            print(
                                f"[INSiNav] Failed to read similar answers for '{candidate_label}' ({e}); continue."
                            )
                    print(
                        "[INSiNav] Similar label set enabled: "
                        f"candidates={goal_label_candidates}, similar={similar_label_set}"
                    )
                    if len(similar_label_set) == 0:
                        print(
                            "[INSiNav] Warning: similar_label_set is empty. "
                            "Only top-k candidate fallback labels will be used."
                        )

            clip_client.set_goal_image(goal_image)
            episode_goal_image = goal_image
            ros_pub.publish_goal_image(goal_image)
            
            # Extract clean goal object crop for better matching (Step 0 & 1)
            # Detect target object in goal image to get clean crop without background
            clip_crop_padding = float(clip_cfg.get("crop_padding_ratio", 0.1))
            print(
                "[INSiNav] Preprocessing goal image: detecting and extracting clean "
                f"target object (target_label={target_label_for_detection})..."
            )
            goal_rgb_vis, goal_score_list, goal_mask_list, goal_label_list = get_object(
                target_label_for_detection,
                goal_image,
                detector_cfg,
                similar_label_set if insinav_filter_enabled and insinav_use_similar_set else [],
                use_label_filter=True  # Use label filter to ensure we extract the TARGET object, not just the highest-scoring one
            )
            if len(goal_mask_list) > 0:
                # Find best detection by score among label-matched candidates
                best_idx = np.argmax(goal_score_list) if len(goal_score_list) > 0 else 0
                goal_clean_mask = goal_mask_list[best_idx]
                goal_detected_label = goal_label_list[best_idx]
                goal_clean_crop = crop_from_mask(goal_image, goal_clean_mask, clip_crop_padding)
                if goal_clean_crop is not None:
                    clip_client.set_goal_clean_crop(goal_clean_crop)
                    label_type = "target" if goal_detected_label == 0 else "similar"
                    print(
                        f"[INSiNav] Goal clean crop extracted ({label_type}): shape={goal_clean_crop.shape}, "
                        f"score={goal_score_list[best_idx]:.3f}, label_idx={goal_detected_label}"
                    )
                else:
                    print("[INSiNav] Failed to crop goal object; using full goal image")
            else:
                print(
                    f"[INSiNav] No '{target_label_for_detection}' object detected in goal image; "
                    "using full goal image as fallback"
                )
            
            if lightglue_verifier is not None and lightglue_verifier.available:
                lightglue_verifier.set_goal_image(goal_image)
        else:
            llm_answer, room, fusion_threshold = read_answer(
                llm_answer_path, llm_response_path, label, llm_client
            )

        # Initialize video frame collection
        vis_frames = []
        info = env.get_metrics()
        if need_video:
            frame = observations_to_image(observations, info)
            info.pop("top_down_map")
            frame = overlay_frame(frame, info)
            if is_instance_imagenav:
                frame = overlay_goal_thumbnail(frame, episode_goal_image)
            vis_frames = [frame]

        # Start publishing basic information and trigger messages
        pub_timer = rospy.Timer(rospy.Duration(0.25), publish_observations)

        print("Agent is waiting in the environment!!!")

        # Wait for ROS system to be ready
        rate = rospy.Rate(10)
        ros_state = ROS_STATE.INIT
        while ros_state == ROS_STATE.INIT or ros_state == ROS_STATE.WAIT_TRIGGER:
            if ros_state == ROS_STATE.INIT:
                print("Waiting for ROS to get odometry...")
            elif ros_state == ROS_STATE.WAIT_TRIGGER:
                print("Waiting for ROS trigger...")
            rate.sleep()

        # Stop timer publishing when starting action execution
        pub_timer.shutdown()

        print("Agent is ready to go!!!!")

        rate = rospy.Rate(10)
        while not rospy.is_shutdown() and not env.episode_over:
            if count_steps >= max_episode_steps:
                print(
                    f"[INSiNav] Reached max_episode_steps={max_episode_steps}. "
                    "Terminate current episode loop."
                )
                break

            # Skip episode if target is not on the same floor
            is_feasible = 0
            for goal in env.current_episode.goals:
                height = goal.position[1]
                is_feasible += is_on_same_floor(
                    height=height, episode=env.current_episode
                )
            if not is_feasible:
                break

            # Parse action from decision system
            action = None
            if global_action is not None:
                if (
                    not (is_instance_imagenav and insinav_stop_gate_enabled)
                    and count_steps == max_episode_steps - 1
                ):
                    global_action = ACTION.STOP

                if (
                    is_instance_imagenav
                    and insinav_stop_gate_enabled
                    and global_action == ACTION.STOP
                    and not insinav_stop_gate_passed
                ):
                    insinav_stop_reject_count += 1
                    print(
                        "[INSiNav_STOP_GATE] reject ROS STOP request: "
                        f"verification not passed. reject_count={insinav_stop_reject_count}/"
                        f"{insinav_stop_reject_turn_patience}"
                    )

                    # Avoid deadlock when planner keeps publishing STOP in FINISH state.
                    # After several rejected STOP requests, force a turn to keep perception updated.
                    if insinav_stop_reject_count >= insinav_stop_reject_turn_patience:
                        action = HabitatSimActions.turn_left
                        insinav_stop_reject_count = 0
                        print(
                            "[INSiNav_STOP_GATE] fallback action TURN_LEFT to continue verification"
                        )
                    pending_stop_verification = True
                    global_action = None
                elif (
                    is_instance_imagenav
                    and insinav_stop_gate_enabled
                    and global_action != ACTION.STOP
                ):
                    pending_stop_verification = False
                    stop_gate_silent_steps = 0

                if action is None:
                    if global_action == ACTION.MOVE_FORWARD:
                        action = HabitatSimActions.move_forward
                    elif global_action == ACTION.TURN_LEFT:
                        action = HabitatSimActions.turn_left
                    elif global_action == ACTION.TURN_RIGHT:
                        action = HabitatSimActions.turn_right
                    elif global_action == ACTION.TURN_DOWN:
                        action = HabitatSimActions.look_down
                        camera_pitch = camera_pitch - np.pi / 6.0
                    elif global_action == ACTION.TURN_UP:
                        action = HabitatSimActions.look_up
                        camera_pitch = camera_pitch + np.pi / 6.0
                    elif global_action == ACTION.STOP:
                        action = HabitatSimActions.stop

                global_action = None

            if (
                action is None
                and is_instance_imagenav
                and insinav_stop_gate_enabled
                and pending_stop_verification
                and not insinav_stop_gate_passed
            ):
                # Planner may stop publishing actions after entering FINISH once.
                # Keep moving camera and occasionally move forward to avoid endless spin.
                stop_gate_silent_steps += 1
                if stop_gate_silent_steps % 4 == 0:
                    action = HabitatSimActions.move_forward
                else:
                    action = HabitatSimActions.turn_left
                print(
                    "[INSiNav_STOP_GATE] planner silent after STOP reject; "
                    f"execute recovery action={action}, silent_steps={stop_gate_silent_steps}"
                )

            # If ROS remains in FINISH while gate is not passed, continue autonomous recovery.
            if (
                action is None
                and is_instance_imagenav
                and insinav_stop_gate_enabled
                and ros_state == ROS_STATE.FINISH
                and not insinav_stop_gate_passed
            ):
                stop_gate_silent_steps += 1
                if stop_gate_silent_steps % 4 == 0:
                    action = HabitatSimActions.move_forward
                else:
                    action = HabitatSimActions.turn_left
                pending_stop_verification = True
                print(
                    "[INSiNav_STOP_GATE] ROS in FINISH without verified target; "
                    f"autonomous recovery action={action}, silent_steps={stop_gate_silent_steps}"
                )

            if action is None:
                continue

            count_steps += 1
            print(f"\n--------------Step: {count_steps}--------------")
            if insinav_filter_enabled:
                if no_detection_steps >= insinav_no_det_patience and len(goal_label_candidates) > 1:
                    active_goal_labels = goal_label_candidates
                else:
                    active_goal_labels = [goal_label_candidates[0]]
                # Only primary label is treated as target(label=0).
                # Secondary goal candidates are used as similar/fallback(label=1).
                step_label = goal_label_candidates[0]
            else:
                step_label = label
            print(
                f"Finding [{step_label}] (active_candidates={active_goal_labels}); Action: {action};"
            )

            # Notify ROS system that action execution is starting
            publish_int32(state_pub, HABITAT_STATE.ACTION_EXEC)

            observations = env.step(action)
            info = env.get_metrics()

            # Detect objects in the current observation first (candidate regions for insinav CLIP)
            raw_rgb = observations["rgb"].copy()
            step_similar_set = []
            if insinav_filter_enabled:
                # Add secondary top-k candidates as non-target fallback labels.
                if len(active_goal_labels) > 1:
                    step_similar_set.extend(active_goal_labels[1:])
                if insinav_use_similar_set:
                    step_similar_set.extend(similar_label_set)
                # Deduplicate while preserving order and avoid duplicating primary target label.
                deduped = []
                seen = set([step_label])
                for s in step_similar_set:
                    if s not in seen:
                        seen.add(s)
                        deduped.append(s)
                step_similar_set = deduped
            observations["rgb"], score_list, object_masks_list, label_list = get_object(
                step_label,
                observations["rgb"],
                detector_cfg,
                step_similar_set,
                use_label_filter=(not is_instance_imagenav) or insinav_filter_enabled,
            )
            if is_instance_imagenav and insinav_filter_enabled:
                print(
                    f"[INSiNav_DEBUG] step={count_steps}, target_label={step_label}, "
                    f"step_similar_set={step_similar_set}"
                )

            if is_instance_imagenav and insinav_filter_enabled:
                if len(object_masks_list) > 0:
                    no_detection_steps = 0
                else:
                    no_detection_steps += 1
                print(
                    f"[INSiNav_DEBUG] no_detection_steps={no_detection_steps}, "
                    f"patience={insinav_no_det_patience}, goal_candidates={goal_label_candidates}"
                )
            print(
                f"[INSiNav_DEBUG] step={count_steps}, detector_scores={len(score_list)}, "
                f"masks={len(object_masks_list)}, labels={label_list}"
            )

            # Calculate semantic score for value map update
            if is_instance_imagenav:
                use_candidate_crops = bool(clip_cfg.get("use_candidate_crops", True))
                clip_topk = max(1, int(clip_cfg.get("topk", 1)))
                clip_crop_padding = float(clip_cfg.get("crop_padding_ratio", 0.1))
                no_candidate_mode = str(clip_cfg.get("no_candidate_mode", "low_score"))
                no_candidate_score = float(clip_cfg.get("no_candidate_score", 0.0))
                clip_scores = []
                max_match_points = 0
                max_target_match_points = 0

                # Step 2: Compare clean observation crops with clean goal crop
                if use_candidate_crops and len(object_masks_list) > 0:
                    for object_mask in object_masks_list:
                        crop = crop_from_mask(raw_rgb, object_mask, clip_crop_padding)
                        if crop is not None:
                            clip_scores.append(clip_client.cosine(crop))

                if len(clip_scores) > 0:
                    clip_scores_sorted = sorted(clip_scores, reverse=True)
                    k = min(clip_topk, len(clip_scores_sorted))
                    cosine = float(np.mean(clip_scores_sorted[:k]))
                    print(
                        f"[INSiNav] DINO clean-crop similarity: {cosine:.3f} "
                        f"(top{k}/{len(clip_scores_sorted)}, "
                        f"max={max(clip_scores_sorted):.3f}, "
                        f"mean={np.mean(clip_scores_sorted):.3f})"
                    )
                else:
                    if no_candidate_mode == "full_frame":
                        cosine = clip_client.cosine(raw_rgb)
                        print(
                            f"[INSiNav] DINO full-frame cosine similarity (fallback): {cosine:.3f}"
                        )
                    else:
                        cosine = no_candidate_score
                        pass
                print(
                    f"[INSiNav_DEBUG] step={count_steps}, clip_candidate_crops={len(clip_scores)}, "
                    f"clip_mode={no_candidate_mode}, dino_score={cosine:.3f}"
                )
            else:
                cosine = get_itm_message_cosine(observations["rgb"], label, room)
                print(f"Target related room: {room}")
                print(f"ITM cosine similarity: {cosine:.3f}")

            # Optional instance-level verification (LightGlue) to re-weight candidate scores
            if (
                is_instance_imagenav
                and lightglue_verifier is not None
                and lightglue_verifier.available
                and len(score_list) == len(object_masks_list)
            ):
                min_match_points = max(1, int(lightglue_cfg.get("min_match_points", 80)))
                blend_weight = float(lightglue_cfg.get("blend_weight", 0.6))
                hard_gate = bool(lightglue_cfg.get("hard_gate", True))
                crop_padding_ratio = float(lightglue_cfg.get("crop_padding_ratio", 0.1))

                lightglue_points = []
                for object_mask in object_masks_list:
                    crop = crop_from_mask(raw_rgb, object_mask, crop_padding_ratio)
                    match_points = lightglue_verifier.match_points(crop)
                    lightglue_points.append(match_points)

                if lightglue_points:
                    max_match_points = max(lightglue_points)
                    target_points = [
                        lightglue_points[i]
                        for i in range(min(len(lightglue_points), len(label_list)))
                        if label_list[i] == 0
                    ]
                    if target_points:
                        max_target_match_points = max(target_points)
                if lightglue_points:
                    print(
                        f"LightGlue match points: max={max(lightglue_points)}, "
                        f"mean={np.mean(lightglue_points):.1f}, min={min(lightglue_points)}"
                    )
                for idx, base_score in enumerate(score_list):
                    match_points = lightglue_points[idx]
                    if hard_gate:
                        # UniGoal-like hard confirmation: only keep candidates with enough matched points.
                        score_list[idx] = float(base_score) if match_points >= min_match_points else 0.0
                    else:
                        # Optional soft fusion fallback.
                        lg_score = min(1.0, match_points / float(min_match_points))
                        fused_score = (1.0 - blend_weight) * float(base_score) + blend_weight * lg_score
                        score_list[idx] = float(fused_score)

            # Use LightGlue only for confirmation (no fusion with DINO score)
            if is_instance_imagenav:
                lg_threshold = 60.0
                stop_distance = success_distance
                if lightglue_cfg is not None:
                    lg_threshold = float(lightglue_cfg.get("score_threshold", lg_threshold))
                    stop_distance = float(lightglue_cfg.get("stop_distance", stop_distance))

                distance_to_goal = info.get("distance_to_goal", float("inf"))

                insinav_stop_gate_passed = (
                    max_match_points >= lg_threshold and distance_to_goal <= stop_distance
                )
                if insinav_stop_gate_passed:
                    pending_stop_verification = False
                    stop_gate_silent_steps = 0

                print(
                    f"[INSiNav_DEBUG] step={count_steps}, lg_max_points={max_match_points}, "
                    f"lg_target_max_points={max_target_match_points}, "
                    f"dino_score={cosine:.3f}"
                )

                if max_match_points >= lg_threshold and distance_to_goal <= stop_distance:
                    print(
                        f"[INSiNav_DEBUG] stop triggered: match_points={max_match_points}, "
                        f"dino_score={cosine:.3f}, distance={distance_to_goal:.3f}, "
                        f"stop_distance={stop_distance:.3f}"
                    )
                    global_action = ACTION.STOP
                    insinav_stop_reject_count = 0

                # Important: before gate passes, do not publish any label-0 target hypothesis
                # to ROS object map. Otherwise planner may enter SEARCH_OBJECT/REACH_OBJECT
                # and publish STOP independently of verification.
                if insinav_stop_gate_enabled and not insinav_stop_gate_passed:
                    demoted = 0
                    for i in range(len(label_list)):
                        if label_list[i] == 0:
                            label_list[i] = 1
                            demoted += 1
                    if demoted > 0:
                        print(
                            "[INSiNav_STOP_GATE] demote unverified target detections "
                            f"label 0->1, count={demoted}"
                        )

            publish_float64(itm_score_pub, cosine)

            # Publish habitat observations to ROS
            observations["camera_pitch"] = camera_pitch
            msg_observations = deepcopy(observations)
            del observations["camera_pitch"]
            ros_pub.habitat_publish_ros_topic(msg_observations)

            # Generate and publish object point clouds
            obj_point_cloud_list = get_object_point_cloud(
                cfg, observations, object_masks_list
            )

            # Publish detection-related information
            cld_with_score_msg.point_clouds = obj_point_cloud_list
            cld_with_score_msg.confidence_scores = score_list
            cld_with_score_msg.label_indices = label_list
            cld_with_score_pub.publish(cld_with_score_msg)

            # Generate video frame
            if need_video:
                frame = observations_to_image(observations, info)
                info.pop("top_down_map")
                frame = overlay_frame(frame, info)
                if is_instance_imagenav:
                    frame = overlay_goal_thumbnail(frame, episode_goal_image)
                vis_frames.append(frame)

            # Track if agent has passed close to the target
            distance_to_goal = info["distance_to_goal"]
            if distance_to_goal <= success_distance and pass_object == 0:
                pass_object = 1

            # Notify ROS system that action execution is complete
            publish_int32(state_pub, HABITAT_STATE.ACTION_FINISH)
            rate.sleep()

        # Notify ROS system that current episode evaluation is complete
        publish_int32(state_pub, HABITAT_STATE.EPISODE_FINISH)

        # Collect evaluation metrics
        info = env.get_metrics()
        spl = info["spl"]
        soft_spl = info["soft_spl"]
        distance_to_goal = info["distance_to_goal"]
        distance_to_goal_reward = info["distance_to_goal_reward"]
        success = info["success"]

        # Check if agent got close to the target object
        if distance_to_goal <= success_distance:
            near_object = 1

        # Determine episode result
        if success == 1:
            num_success += 1
            result_text = "success"
        else:
            result_text = check_failure(
                env.current_episode,
                final_state,
                expl_result,
                count_steps,
                max_episode_steps,
                pass_object,
                near_object,
            )

        # Update cumulative statistics
        num_total += 1
        spl_all += spl
        soft_spl_all += soft_spl
        distance_to_goal_all += distance_to_goal
        distance_to_goal_reward_all += distance_to_goal_reward

        # Generate video file
        scene_id = env.current_episode.scene_id
        episode_id = env.current_episode.episode_id
        video_name = f"{os.path.basename(scene_id)}_{episode_id}"
        time_spend = time.time() - start_time + last_time

        img2video_output_path = os.path.join(video_output_path, result_text)

        if flag_once:
            img2video_output_path = "videos"
            video_name = "video_once"

        if need_video:
            images_to_video(
                vis_frames, img2video_output_path, video_name, fps=6, quality=9
            )
        vis_frames.clear()

        # Display average performance metrics
        table1 = PrettyTable(["Metric", "Average"])
        table1.add_row(["Average Success", f"{num_success/num_total * 100:.2f}%"])
        table1.add_row(["Average SPL", f"{spl_all/num_total * 100:.2f}%"])
        table1.add_row(["Average Soft SPL", f"{soft_spl_all/num_total * 100:.2f}%"])
        table1.add_row(
            ["Average Distance to Goal", f"{distance_to_goal_all/num_total:.4f}"]
        )
        print(table1)
        print(f"Episode {num_total} data written to {record_file_path}")
        print(f"Result: {result_text}")

        # Display total performance metrics
        table2 = PrettyTable(["Metric", "Total"])
        table2.add_row(["Total Success", f"{num_success}"])
        table2.add_row(["Total SPL", f"{spl_all:.2f}"])
        table2.add_row(["Total Soft SPL", f"{soft_spl_all:.2f}"])
        table2.add_row(["Total Distance to Goal", f"{distance_to_goal_all:.4f}"])

        if flag_once:
            break

        # Write results to record file
        write_record(
            scene_id,
            episode_id,
            table1,
            result_text,
            label,
            num_total,
            time_spend,
            record_file_path,
        )

        # Write results to continue file
        write_record(
            scene_id,
            episode_id,
            table2,
            result_text,
            label,
            num_total,
            time_spend,
            continue_path,
        )

        # Count files in each result category folder
        for i in range(len(RESULT_TYPES)):
            folder = RESULT_TYPES[i]  # Get current category (folder name)
            folder_path = os.path.join(video_output_path, folder)  # Build folder path
            file_count = count_files_in_directory(folder_path)  # Count files in folder
            result_list[i] = file_count

        # Publish comprehensive record data
        record_data = [
            num_success / num_total * 100,
            spl_all / num_total * 100,
            soft_spl_all / num_total * 100,
            distance_to_goal_all / num_total,
        ]
        record_data.extend(result_list)
        publish_float32_array(record_pub, record_data)

        pbar.update()
        env.current_episode = next(env.episode_iterator)
        rospy.sleep(0.1)  # wait a moment

    env.close()
    pbar.close()


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    rospy.init_node("habitat_eval_node", anonymous=True)

    try:
        dataset, overrides = _parse_dataset_arg()
        cfg_name = f"habitat_eval_{dataset}"
        # Compose the chosen config and pass through extra Hydra overrides
        with initialize(version_base=None, config_path="config"):
            cfg = compose(config_name=cfg_name, overrides=overrides)
        main(cfg)
    except Exception as e:
        print(f"Unexpected error occurred: {e}")
        rospy.signal_shutdown("Shutdown due to error")
        os._exit(1)
