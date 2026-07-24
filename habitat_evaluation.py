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
from datetime import datetime
import gzip
import json
import os
import signal
import sys
import time
import math
from copy import deepcopy
from PIL import Image

# Third-party library imports
import cv2
from hydra import initialize, compose
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from omegaconf import DictConfig
from prettytable import PrettyTable
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Int32, Int32MultiArray, Float32MultiArray, Float64, String
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
from basic_utils.record_episode.read_record import (
    read_diagnostic_counts,
    read_goal_view_totals,
    read_record,
)
from basic_utils.record_episode.write_record import write_record
from habitat2ros import habitat_publisher
from llm.answer_reader.answer_reader import read_answer
from params import (
    HABITAT_STATE,
    ROS_STATE,
    ACTION,
    RESULT_TYPES,
    TERMINATION_REASONS,
    RECOVERABLE_FAILURE_EVENTS,
    FINAL_RESULT,
    EXPL_RESULT,
)
from vlm.Labels import MP3D_ID_TO_NAME
from vlm.utils.get_itm_message import get_itm_message_cosine
from vlm.itm.dino_similarity import DINOSimilarity
from vlm.itm.lightglue_verifier import LightGlueVerifier
from vlm.itm.mast3r_refiner import (
    MASt3RPoseRefiner,
    _camera_intrinsics,
    _normalize_depth,
)
from vlm.detector.yolov7 import YOLOv7Client
from vlm.utils.get_object_utils import configure_detection_clients, get_object


def _load_json(path):
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _scene_name_candidates(scene_id: str):
    candidates = []
    basename = os.path.basename(scene_id)
    stem, _ = os.path.splitext(basename)
    if stem:
        candidates.append(stem)
    if stem.endswith(".basis"):
        candidates.append(stem[: -len(".basis")])

    parent = os.path.basename(os.path.dirname(scene_id))
    if parent:
        candidates.append(parent)
        if "-" in parent:
            candidates.append(parent.split("-", 1)[1])

    clean = scene_id.replace("\\", "/")
    parts = [p for p in clean.split("/") if p]
    for p in parts:
        candidates.append(p)
        if "-" in p:
            tail = p.split("-", 1)[1]
            if tail:
                candidates.append(tail)

    dedup = []
    seen = set()
    for item in candidates:
        if item and item not in seen:
            dedup.append(item)
            seen.add(item)
    return dedup


def _infer_content_json_path(dataset_path: str, scene_id: str):
    dataset_path = os.path.abspath(dataset_path)
    parent = os.path.dirname(dataset_path)
    content_dir = os.path.join(parent, "content")
    if not os.path.isdir(content_dir):
        return None
    for scene_name in _scene_name_candidates(scene_id):
        candidate = os.path.join(content_dir, f"{scene_name}.json.gz")
        if os.path.exists(candidate):
            return candidate
    return None


def _resolve_episode_goal_image_pose(dataset_path: str, episode):
    goal_object_id = str(getattr(episode, "goal_object_id"))
    goal_image_id = int(getattr(episode, "goal_image_id", 0))
    content_json_path = _infer_content_json_path(dataset_path, episode.scene_id)
    if content_json_path is None:
        return None

    content_data = _load_json(content_json_path)
    goals = content_data.get("goals", {})
    candidate_keys = [
        f"{scene_name}_{goal_object_id}" for scene_name in _scene_name_candidates(episode.scene_id)
    ]

    goal_entry = None
    if isinstance(goals, dict):
        for key in candidate_keys:
            value = goals.get(key)
            if isinstance(value, dict):
                goal_entry = value
                break
        if goal_entry is None:
            suffix = f"_{goal_object_id}"
            for key, value in goals.items():
                if str(key).endswith(suffix) and isinstance(value, dict):
                    goal_entry = value
                    break
        if goal_entry is None:
            for value in goals.values():
                if isinstance(value, dict) and str(value.get("object_id")) == goal_object_id:
                    goal_entry = value
                    break

    if goal_entry is None:
        return None

    image_goals = goal_entry.get("image_goals", [])
    if goal_image_id < 0 or goal_image_id >= len(image_goals):
        return None

    goal_pose = image_goals[goal_image_id]
    position = goal_pose.get("position")
    rotation = goal_pose.get("rotation")
    if position is None or rotation is None:
        return None

    return {
        "position": np.asarray(position, dtype=np.float32),
        "rotation": rotation,
        "content_json_path": content_json_path,
    }


def _extract_quaternion_xyzw(rotation):
    if rotation is None:
        raise ValueError("rotation is None")

    if hasattr(rotation, "x") and hasattr(rotation, "y") and hasattr(rotation, "z") and hasattr(rotation, "w"):
        return float(rotation.x), float(rotation.y), float(rotation.z), float(rotation.w)

    if hasattr(rotation, "imag") and hasattr(rotation, "real"):
        imag = np.asarray(rotation.imag, dtype=np.float64).reshape(-1)
        if imag.size >= 3:
            return float(imag[0]), float(imag[1]), float(imag[2]), float(rotation.real)

    if hasattr(rotation, "components"):
        comps = list(rotation.components)
        if len(comps) == 4:
            # numpy-quaternion stores as w, x, y, z
            return float(comps[1]), float(comps[2]), float(comps[3]), float(comps[0])

    arr = np.asarray(rotation, dtype=np.float64).reshape(-1)
    if arr.size == 4:
        return float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])

    raise ValueError(f"unsupported rotation type: {type(rotation)}")


def _yaw_from_quaternion_xyzw(rotation) -> float:
    x, y, z, w = _extract_quaternion_xyzw(rotation)
    siny_cosp = 2.0 * (w * y + x * z)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _wrap_angle_rad(angle: float) -> float:
    while angle <= -math.pi:
        angle += 2.0 * math.pi
    while angle > math.pi:
        angle -= 2.0 * math.pi
    return angle


def compute_goal_view_metrics(
    final_position: np.ndarray,
    final_rotation,
    goal_position: np.ndarray,
    goal_rotation,
    pos_threshold: float = 0.25,
    yaw_threshold_deg: float = 10.0,
):
    final_position = np.asarray(final_position, dtype=np.float32)
    goal_position = np.asarray(goal_position, dtype=np.float32)
    position_error = float(np.linalg.norm(final_position[[0, 2]] - goal_position[[0, 2]]))

    final_yaw = _yaw_from_quaternion_xyzw(final_rotation)
    goal_yaw = _yaw_from_quaternion_xyzw(goal_rotation)
    yaw_error_rad = abs(_wrap_angle_rad(final_yaw - goal_yaw))
    yaw_error_deg = float(np.rad2deg(yaw_error_rad))

    view_success = bool(position_error < pos_threshold and yaw_error_deg < yaw_threshold_deg)
    return position_error, yaw_error_deg, view_success


def publish_int32(publisher, data):
    msg = Int32()
    msg.data = data
    publisher.publish(msg)


def publish_verified_approach_target(publisher, object_cloud=None):
    msg = MultipleMasksWithConfidence()
    if object_cloud is not None:
        msg.point_clouds = [object_cloud]
        msg.confidence_scores = [1.0]
        msg.label_indices = [0]
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


def publish_mast3r_hint(
    publisher,
    active: bool,
    allow_stop: bool,
    yaw_error_deg: float,
    forward_error: float,
    lateral_error: float,
    transl_error: float,
    depth_error: float,
):
    msg = Float32MultiArray()
    msg.data = [
        1.0 if active else 0.0,
        1.0 if allow_stop else 0.0,
        float(yaw_error_deg),
        float(forward_error),
        float(lateral_error),
        float(transl_error),
        float(depth_error),
    ]
    publisher.publish(msg)


def run_mast3r_refine_on_observation(
    mast3r_refiner,
    rgb_image,
    depth_image,
    rgb_sensor_cfg,
    depth_sensor_cfg,
    goal_mask=None,
    current_mask=None,
):
    current_intrinsics = _camera_intrinsics(
        int(rgb_sensor_cfg.width),
        int(rgb_sensor_cfg.height),
        float(rgb_sensor_cfg.hfov),
    )
    current_depth_metric = _normalize_depth(
        depth_image,
        float(depth_sensor_cfg.min_depth),
        float(depth_sensor_cfg.max_depth),
    )
    return mast3r_refiner.refine(
        current_rgb=rgb_image,
        current_depth_metric=current_depth_metric,
        current_intrinsics=current_intrinsics,
        goal_mask=goal_mask,
        current_mask=current_mask,
    )


def log_mast3r_result(
    mast3r_result, geometry_mode="masked", tag="[INSiNav_MASt3R]"
):
    mask_inlier_ratio = (
        "disabled(full_frame)"
        if geometry_mode == "full_frame"
        else (
            f"{mast3r_result.goal_mask_inlier_ratio:.3f}/"
            f"{mast3r_result.current_mask_inlier_ratio:.3f}/"
            f"{mast3r_result.joint_mask_inlier_ratio:.3f}"
        )
    )
    print(
        f"{tag} "
        f"success={int(mast3r_result.success)}, matches={mast3r_result.num_matches}, "
        f"valid_matches={mast3r_result.valid_matches}, "
        f"inliers={mast3r_result.inliers}, transl_error={mast3r_result.transl_error:.3f}, "
        f"yaw_error_deg={mast3r_result.yaw_error_deg:.2f}, "
        f"forward_error={mast3r_result.forward_error:.3f}, "
        f"lateral_error={mast3r_result.lateral_error:.3f}, "
        f"depth_error={mast3r_result.depth_error:.3f}, "
        f"mask_inlier_ratio={mask_inlier_ratio}, "
        f"should_stop={int(mast3r_result.should_stop)}, "
        f"suggested_action={mast3r_result.suggested_action}, "
        f"reason={mast3r_result.debug_reason}"
    )


def mast3r_geometry_mode(mast3r_cfg) -> str:
    """Return the correspondence-selection mode used for MASt3R pose estimation."""
    mode = (
        str(mast3r_cfg.get("geometry_mode", "full_frame")).strip().lower()
        if mast3r_cfg is not None
        else "full_frame"
    )
    if mode not in ("full_frame", "masked"):
        raise ValueError(
            "mast3r_refine.geometry_mode must be 'full_frame' or 'masked', "
            f"got '{mode}'."
        )
    return mode


def validate_mast3r_result(mast3r_result, mast3r_cfg):
    """Validate a MASt3R pose before either visual route can hand it to ROS."""
    if mast3r_result is None or not mast3r_result.success:
        reason = (
            getattr(mast3r_result, "debug_reason", "missing_result")
            if mast3r_result is not None
            else "missing_result"
        )
        return False, reason

    quality_cfg = (
        mast3r_cfg.get("quality", {}) if mast3r_cfg is not None else {}
    )
    min_matches = max(1, int(quality_cfg.get("min_matches", 60)))
    min_valid_matches = max(1, int(quality_cfg.get("min_valid_matches", 40)))
    min_inliers = max(1, int(quality_cfg.get("min_inliers", 25)))
    min_inlier_ratio = float(quality_cfg.get("min_inlier_ratio", 0.30))
    min_goal_mask_inlier_ratio = float(
        quality_cfg.get("min_goal_mask_inlier_ratio", 0.90)
    )
    min_current_mask_inlier_ratio = float(
        quality_cfg.get("min_current_mask_inlier_ratio", 0.90)
    )
    min_joint_mask_inlier_ratio = float(
        quality_cfg.get("min_joint_mask_inlier_ratio", 0.90)
    )
    max_depth_error = float(quality_cfg.get("max_depth_error", 0.35))
    max_translation = float(
        quality_cfg.get(
            "max_translation",
            mast3r_cfg.get("max_local_goal_translation", 4.0)
            if mast3r_cfg is not None
            else 4.0,
        )
    )
    max_abs_yaw_deg = float(quality_cfg.get("max_abs_yaw_deg", 120.0))
    require_mask_inlier_ratios = mast3r_geometry_mode(mast3r_cfg) == "masked"

    matches = int(mast3r_result.num_matches)
    valid_matches = int(mast3r_result.valid_matches)
    inliers = int(mast3r_result.inliers)
    inlier_ratio = inliers / max(1, valid_matches)
    checks = [
        (matches >= min_matches, f"matches={matches}<{min_matches}"),
        (
            valid_matches >= min_valid_matches,
            f"valid_matches={valid_matches}<{min_valid_matches}",
        ),
        (inliers >= min_inliers, f"inliers={inliers}<{min_inliers}"),
        (
            inlier_ratio >= min_inlier_ratio,
            f"inlier_ratio={inlier_ratio:.3f}<{min_inlier_ratio:.3f}",
        ),
        (
            np.isfinite(mast3r_result.depth_error)
            and mast3r_result.depth_error <= max_depth_error,
            f"depth_error={mast3r_result.depth_error:.3f}>{max_depth_error:.3f}",
        ),
        (
            np.isfinite(mast3r_result.transl_error)
            and mast3r_result.transl_error <= max_translation,
            f"translation={mast3r_result.transl_error:.3f}>{max_translation:.3f}",
        ),
        (
            np.isfinite(mast3r_result.yaw_error_deg)
            and abs(mast3r_result.yaw_error_deg) <= max_abs_yaw_deg,
            f"abs_yaw={abs(mast3r_result.yaw_error_deg):.2f}>{max_abs_yaw_deg:.2f}",
        ),
        (
            np.isfinite(mast3r_result.forward_error)
            and np.isfinite(mast3r_result.lateral_error),
            "non_finite_local_offset",
        ),
    ]
    if require_mask_inlier_ratios:
        checks.extend(
            [
                (
                    np.isfinite(mast3r_result.goal_mask_inlier_ratio)
                    and mast3r_result.goal_mask_inlier_ratio
                    >= min_goal_mask_inlier_ratio,
                    "goal_mask_inlier_ratio="
                    f"{mast3r_result.goal_mask_inlier_ratio:.3f}<"
                    f"{min_goal_mask_inlier_ratio:.3f}",
                ),
                (
                    np.isfinite(mast3r_result.current_mask_inlier_ratio)
                    and mast3r_result.current_mask_inlier_ratio
                    >= min_current_mask_inlier_ratio,
                    "current_mask_inlier_ratio="
                    f"{mast3r_result.current_mask_inlier_ratio:.3f}<"
                    f"{min_current_mask_inlier_ratio:.3f}",
                ),
                (
                    np.isfinite(mast3r_result.joint_mask_inlier_ratio)
                    and mast3r_result.joint_mask_inlier_ratio
                    >= min_joint_mask_inlier_ratio,
                    "joint_mask_inlier_ratio="
                    f"{mast3r_result.joint_mask_inlier_ratio:.3f}<"
                    f"{min_joint_mask_inlier_ratio:.3f}",
                ),
            ]
        )
    failed_reasons = [reason for passed, reason in checks if not passed]
    return not failed_reasons, ",".join(failed_reasons) or "ok"


class TeeOutput:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)

    @property
    def encoding(self):
        return getattr(self.streams[0], "encoding", "utf-8")


def setup_run_logging(video_output_path):
    log_dir = os.path.join(video_output_path, "logs")
    os.makedirs(log_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"run_{stamp}_pid{os.getpid()}.log")
    log_file = open(log_path, "a", encoding="utf-8", buffering=1)
    sys.stdout = TeeOutput(sys.__stdout__, log_file)
    sys.stderr = TeeOutput(sys.__stderr__, log_file)
    print(f"[RUN_LOG] stdout/stderr tee enabled: {log_path}")
    return log_file, log_path


def close_run_logging(log_file):
    if log_file is None:
        return
    sys.stdout.flush()
    sys.stderr.flush()
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    log_file.close()


def action_code_to_habitat(action_code):
    """Convert planner action enum to Habitat action and camera pitch delta."""
    if action_code == ACTION.MOVE_FORWARD:
        return HabitatSimActions.move_forward, 0.0
    if action_code == ACTION.TURN_LEFT:
        return HabitatSimActions.turn_left, 0.0
    if action_code == ACTION.TURN_RIGHT:
        return HabitatSimActions.turn_right, 0.0
    if action_code == ACTION.TURN_DOWN:
        return HabitatSimActions.look_down, -np.pi / 6.0
    if action_code == ACTION.TURN_UP:
        return HabitatSimActions.look_up, np.pi / 6.0
    if action_code == ACTION.STOP:
        return HabitatSimActions.stop, 0.0
    return None, 0.0


def action_code_name(action_code) -> str:
    names = {
        ACTION.STOP: "stop",
        ACTION.MOVE_FORWARD: "move_forward",
        ACTION.TURN_LEFT: "turn_left",
        ACTION.TURN_RIGHT: "turn_right",
        ACTION.TURN_DOWN: "turn_down",
        ACTION.TURN_UP: "turn_up",
    }
    return names.get(action_code, f"unknown({action_code})")


def mast3r_yaw_only_params(cfg, mast3r_cfg, mast3r_refiner):
    """Read the fine turn used to align one locked MASt3R SE(2) goal."""
    fine_yaw_turn_deg = min(15.0, float(cfg.habitat.simulator.turn_angle))

    if mast3r_cfg is not None:
        fine_yaw_turn_deg = float(
            mast3r_cfg.get("fine_yaw_turn_deg", fine_yaw_turn_deg)
        )

    return max(1e-3, fine_yaw_turn_deg)


def _turn_action_name(action):
    if action == HabitatSimActions.turn_left:
        return "turn_left"
    if action == HabitatSimActions.turn_right:
        return "turn_right"
    return str(action)


def _set_habitat_turn_amount(env, action, amount_deg):
    """Temporarily override a Habitat turn action amount, returning restore handles."""
    if action not in (HabitatSimActions.turn_left, HabitatSimActions.turn_right):
        return []

    try:
        agent = env.sim.get_agent(0)
        action_space = agent.agent_config.action_space
    except Exception:
        return []

    turn_name = _turn_action_name(action)
    restore_handles = []
    seen_actuations = set()
    for key, spec in action_space.items():
        key_name = str(key).lower()
        if key != action and turn_name not in key_name:
            continue
        actuation = getattr(spec, "actuation", None)
        if actuation is None or not hasattr(actuation, "amount"):
            continue
        actuation_id = id(actuation)
        if actuation_id in seen_actuations:
            continue
        seen_actuations.add(actuation_id)
        try:
            restore_handles.append((actuation, float(actuation.amount)))
            actuation.amount = float(amount_deg)
        except Exception:
            continue

    return restore_handles


def _restore_habitat_turn_amount(restore_handles):
    for actuation, amount in restore_handles:
        actuation.amount = amount


def step_habitat_action(env, action, fine_yaw_turn_deg=None):
    """Step Habitat, optionally using a smaller turn amount for one yaw action."""
    restore_handles = []
    if fine_yaw_turn_deg is not None:
        restore_handles = _set_habitat_turn_amount(env, action, fine_yaw_turn_deg)
        if restore_handles:
            print(
                "[INSiNav_REFINE] apply fine Habitat yaw turn: "
                f"action={_turn_action_name(action)}, amount={fine_yaw_turn_deg:.2f}deg"
            )
        else:
            print(
                "[INSiNav_REFINE] warning: unable to override Habitat turn amount; "
                "using configured turn_angle."
            )

    try:
        return env.step(action)
    finally:
        _restore_habitat_turn_amount(restore_handles)


def filter_positive_detection_results(point_clouds, scores, labels):
    """Keep only detections with positive scores and aligned metadata."""
    filtered_point_clouds = []
    filtered_scores = []
    filtered_labels = []

    count = min(len(point_clouds), len(scores), len(labels))
    for idx in range(count):
        score = float(scores[idx])
        if score <= 1e-6:
            continue
        filtered_point_clouds.append(point_clouds[idx])
        filtered_scores.append(score)
        filtered_labels.append(int(labels[idx]))

    return filtered_point_clouds, filtered_scores, filtered_labels


def lightglue_set_goal(verifier, image, goal_key="default"):
    """Set LightGlue goal features while tolerating older verifier interfaces."""
    try:
        verifier.set_goal_image(image, goal_key=goal_key)
    except TypeError:
        verifier.set_goal_image(image)
        if (
            goal_key != "default"
            and hasattr(verifier, "goal_feats_by_key")
            and hasattr(verifier, "goal_feats")
        ):
            verifier.goal_feats_by_key[goal_key] = verifier.goal_feats


def lightglue_clear_goal(verifier, goal_key="default"):
    """Clear LightGlue goal features while tolerating older verifier interfaces."""
    if hasattr(verifier, "clear_goal"):
        try:
            verifier.clear_goal(goal_key=goal_key)
        except TypeError:
            verifier.clear_goal()
    elif hasattr(verifier, "goal_feats"):
        verifier.goal_feats = None


def lightglue_has_goal(verifier, goal_key="default") -> bool:
    """Check LightGlue goal cache while tolerating older verifier interfaces."""
    if hasattr(verifier, "has_goal"):
        try:
            return bool(verifier.has_goal(goal_key=goal_key))
        except TypeError:
            return bool(verifier.has_goal())
    return getattr(verifier, "goal_feats", None) is not None


def lightglue_match_points(verifier, image, goal_key="default") -> int:
    """Run LightGlue matching while tolerating older verifier interfaces."""
    try:
        return int(verifier.match_points(image, goal_key=goal_key))
    except TypeError:
        return int(verifier.match_points(image))


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


def mask_bbox_iou(mask_a, mask_b):
    """Return bounding-box IoU for lightweight candidate association."""
    if mask_a is None or mask_b is None:
        return 0.0

    def mask_box(mask):
        mask = np.asarray(mask)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        ys, xs = np.where(mask > 0)
        if xs.size == 0 or ys.size == 0:
            return None
        return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)

    box_a = mask_box(mask_a)
    box_b = mask_box(mask_b)
    if box_a is None or box_b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = inter_w * inter_h
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return intersection / union if union > 0.0 else 0.0


def estimate_mask_depth_distance(
    depth_image,
    object_mask,
    min_depth,
    max_depth,
    percentile=35.0,
    erosion_iters=1,
):
    """Estimate object distance from depth pixels inside a mask.

    Uses a lower percentile instead of the mean to reduce background leakage,
    which is closer to the robust heuristic used in UniGoal's instance-image flow.
    """
    if depth_image is None or object_mask is None:
        return float("inf")

    mask_uint8 = (object_mask > 0).astype(np.uint8) * 255
    if erosion_iters > 0:
        mask_uint8 = cv2.erode(mask_uint8, None, iterations=erosion_iters)

    valid_mask = mask_uint8 > 0
    if not np.any(valid_mask):
        valid_mask = object_mask > 0
    if not np.any(valid_mask):
        return float("inf")

    if depth_image.ndim == 3:
        depth_norm = depth_image[:, :, 0].astype(np.float32)
    else:
        depth_norm = depth_image.astype(np.float32)

    depth_metric = depth_norm * (max_depth - min_depth) + min_depth
    depth_values = depth_metric[valid_mask]
    depth_values = depth_values[np.isfinite(depth_values)]
    depth_values = depth_values[(depth_values > min_depth + 1e-4) & (depth_values < max_depth - 1e-4)]

    if depth_values.size == 0:
        return float("inf")

    return float(np.percentile(depth_values, percentile))


def estimate_center_depth_distance(depth_image, min_depth, max_depth, window_ratio=0.18):
    """Estimate free distance in the central image window for visual approach fallback."""
    if depth_image is None:
        return float("inf")
    if depth_image.ndim == 3:
        depth_norm = depth_image[:, :, 0].astype(np.float32)
    else:
        depth_norm = depth_image.astype(np.float32)

    h, w = depth_norm.shape[:2]
    win_w = max(4, int(w * window_ratio))
    win_h = max(4, int(h * window_ratio))
    x1 = max(0, (w - win_w) // 2)
    y1 = max(0, (h - win_h) // 2)
    center = depth_norm[y1 : y1 + win_h, x1 : x1 + win_w]
    depth_metric = center * (max_depth - min_depth) + min_depth
    depth_values = depth_metric[np.isfinite(depth_metric)]
    depth_values = depth_values[
        (depth_values > min_depth + 1e-4) & (depth_values < max_depth - 1e-4)
    ]
    if depth_values.size == 0:
        return float("inf")
    return float(np.percentile(depth_values, 30.0))


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
    global msg_observations, fusion_threshold, instance_stop_gate_enabled_flag
    global ros_pub, trigger_pub, confidence_threshold_pub, instance_stop_gate_pub
    tmp = deepcopy(msg_observations)
    ros_pub.habitat_publish_ros_topic(tmp)
    publish_float64(confidence_threshold_pub, fusion_threshold)
    publish_int32(instance_stop_gate_pub, int(instance_stop_gate_enabled_flag))
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


def planner_failure_termination_reason(current_final_state, current_expl_result):
    """Return a terminal planner failure reason, or None for a resumable FINISH."""
    if current_final_state == FINAL_RESULT.STUCKING:
        return "stucking"
    if current_final_state != FINAL_RESULT.NO_FRONTIER:
        return None
    if current_expl_result == EXPL_RESULT.NO_PASSABLE_FRONTIER:
        return "no_passable_frontier"
    return "no_coverable_frontier"


def mast3r_refine_status_callback(msg):
    global mast3r_refine_status
    mast3r_refine_status = int(msg.data)


def object_viewpoint_debug_callback(msg):
    print(f"[ROS_OBJECT_VIEWPOINT_DEBUG] {msg.data}", flush=True)


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
    global instance_stop_gate_pub, instance_stop_gate_enabled_flag
    global final_state, expl_result, mast3r_refine_status

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
    ablation_cfg = cfg.get("ablation", {}) or {}
    ablation_name = str(ablation_cfg.get("name", "full"))
    route_cfg = ablation_cfg.get("routes", {}) or {}
    boxed_cfg = ablation_cfg.get("boxed", {}) or {}
    terminal_cfg = ablation_cfg.get("terminal", {}) or {}
    boxed_lightglue_enabled = bool(route_cfg.get("boxed_lightglue", True))
    no_box_lightglue_enabled = bool(route_cfg.get("no_box_lightglue", True))
    dino_direct_route_enabled = bool(route_cfg.get("dino_direct", True))
    boxed_full_frame_gate_enabled = bool(boxed_cfg.get("full_frame_gate", True))
    boxed_crop_ownership_enabled = bool(boxed_cfg.get("crop_ownership", True))
    terminal_mode = str(terminal_cfg.get("mode", "geometry")).strip().lower()
    if terminal_mode not in {"geometry", "rgbd_after_geometry", "rgbd"}:
        raise ValueError(
            "ablation.terminal.mode must be geometry, rgbd_after_geometry, or rgbd; "
            f"got {terminal_mode!r}"
        )
    print(
        "[INSiNav_ABLATION] "
        f"name={ablation_name}, routes="
        f"R1:{int(boxed_lightglue_enabled)}/"
        f"R2:{int(no_box_lightglue_enabled)}/"
        f"R3:{int(dino_direct_route_enabled)}, "
        f"boxed_full_frame_gate={int(boxed_full_frame_gate_enabled)}, "
        f"crop_ownership={int(boxed_crop_ownership_enabled)}, "
        f"terminal={terminal_mode}"
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
    route_metrics_path = os.path.join(video_output_path, "route_metrics.jsonl")
    max_episode_steps = cfg.habitat.environment.max_episode_steps
    success_distance = cfg.habitat.task.measurements.success.success_distance
    rgb_sensor_cfg = cfg.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor
    depth_sensor_cfg = cfg.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor

    detector_cfg = cfg.detector
    configure_detection_clients(detector_cfg)

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
    lightglue_verification_mode = (
        str(lightglue_cfg.get("verification_mode", "full_frame")).lower()
        if lightglue_cfg is not None
        else "full_frame"
    )
    lightglue_no_box_cfg = (
        lightglue_cfg.get("no_box_direct", {})
        if lightglue_cfg is not None
        else {}
    ) or {}
    lightglue_no_box_enabled = bool(
        no_box_lightglue_enabled and lightglue_no_box_cfg.get("enabled", False)
    )
    lightglue_no_box_match_threshold = max(
        1.0, float(lightglue_no_box_cfg.get("match_points_threshold", 300))
    )
    lightglue_no_box_retry_cooldown_steps = max(
        1, int(lightglue_no_box_cfg.get("retry_cooldown_steps", 15))
    )
    mast3r_cfg = cfg.get("mast3r_refine", None)
    mast3r_geometry_matching_mode = mast3r_geometry_mode(mast3r_cfg)
    mast3r_use_detection_masks = mast3r_geometry_matching_mode == "masked"
    mast3r_refiner = None
    mast3r_execution_retry_cooldown_steps = max(
        1,
        int(
            mast3r_cfg.get("execution_retry_cooldown_steps", 15)
            if mast3r_cfg is not None
            else 15
        ),
    )
    if is_instance_imagenav:
        clip_cfg = cfg.clip
        clip_client = DINOSimilarity(
            model_name=clip_cfg.model_name, device=clip_cfg.device
        )
        goal_yolo_client = YOLOv7Client(
            port=int(detector_cfg.get("yolo_server_port", 12184))
        )
        print(
            f"InstanceImageNav mode enabled, DINO model={clip_cfg.model_name}, device={clip_client.device}"
        )

        if (
            lightglue_cfg is not None
            and lightglue_cfg.get("enabled", False)
            and (boxed_lightglue_enabled or lightglue_no_box_enabled)
        ):
            lightglue_verifier = LightGlueVerifier(
                device=lightglue_cfg.get("device", "cuda"),
                max_num_keypoints=lightglue_cfg.get("max_num_keypoints", 1024),
            )
            if lightglue_verifier.available:
                lightglue_mode = lightglue_cfg.get("verification_mode", "full_frame")
                print(
                    "LightGlue instance verification enabled "
                    f"(device={lightglue_verifier.device}, "
                    f"max_num_keypoints={lightglue_cfg.get('max_num_keypoints', 1024)}, "
                    f"mode={lightglue_mode})"
                )
                if lightglue_no_box_enabled:
                    if lightglue_verification_mode == "full_frame":
                        print(
                            "LightGlue detector-independent MASt3R route enabled "
                            f"(match_points_threshold="
                            f"{lightglue_no_box_match_threshold:g}, "
                            f"retry_cooldown_steps="
                            f"{lightglue_no_box_retry_cooldown_steps})"
                        )
                    else:
                        print(
                            "LightGlue detector-independent MASt3R route requires "
                            "verification_mode=full_frame; route disabled for this run."
                        )
            else:
                print(
                    "LightGlue is not available in current environment. "
                    "Falling back to CLIP-only insinav."
                )
        if mast3r_cfg is not None and mast3r_cfg.get("enabled", False):
            mast3r_refiner = MASt3RPoseRefiner(
                weights=mast3r_cfg.get("weights", None),
                mast3r_root=mast3r_cfg.get("mast3r_root", "mast3r"),
                device=mast3r_cfg.get("device", "cuda"),
                image_size=int(mast3r_cfg.get("image_size", 512)),
                min_conf_thr=float(mast3r_cfg.get("min_conf_thr", 1.001)),
                match_subsample=int(mast3r_cfg.get("match_subsample", 8)),
                min_matches=int(mast3r_cfg.get("min_matches", 40)),
                stop_translation_tolerance=float(
                    mast3r_cfg.get("stop_translation_tolerance", 0.20)
                ),
                stop_yaw_tolerance_deg=float(
                    mast3r_cfg.get("stop_yaw_tolerance_deg", 12.0)
                ),
                action_yaw_tolerance_deg=float(
                    mast3r_cfg.get("action_yaw_tolerance_deg", 8.0)
                ),
                action_forward_tolerance=float(
                    mast3r_cfg.get("action_forward_tolerance", 0.20)
                ),
                action_lateral_tolerance=float(
                    mast3r_cfg.get("action_lateral_tolerance", 0.10)
                ),
                forward_bias=float(mast3r_cfg.get("forward_bias", 0.0)),
                stop_depth_tolerance=float(
                    mast3r_cfg.get("stop_depth_tolerance", 0.20)
                ),
                center_tolerance_px=float(
                    mast3r_cfg.get("center_tolerance_px", 30.0)
                ),
                center_gain=float(mast3r_cfg.get("center_gain", 0.003)),
                pnp_reprojection_error_px=float(
                    mast3r_cfg.get("pnp_reprojection_error_px", 5.0)
                ),
                pnp_iterations=int(mast3r_cfg.get("pnp_iterations", 100)),
                pose_consistency_translation=float(
                    mast3r_cfg.get("pose_consistency_translation", 0.75)
                ),
                pose_consistency_yaw_deg=float(
                    mast3r_cfg.get("pose_consistency_yaw_deg", 30.0)
                ),
            )
            if mast3r_refiner.available:
                print(
                    "MASt3R close-range refinement enabled "
                    f"(device={mast3r_cfg.get('device', 'cuda')}, "
                    f"image_size={mast3r_cfg.get('image_size', 512)}, "
                    f"geometry_mode={mast3r_geometry_matching_mode})"
                )
            else:
                print(
                    "MASt3R is not available in current environment. "
                    f"reason={mast3r_refiner.unavailable_reason}. "
                    "LightGlue approach remains available, but final pose confirmation is disabled."
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
    configured_eval_episodes = int(cfg.get("max_eval_episodes", -1))
    evaluation_total = (
        number_of_episodes
        if configured_eval_episodes <= 0
        else min(number_of_episodes, configured_eval_episodes)
    )

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
    termination_counts, recoverable_failure_counts = read_diagnostic_counts(
        continue_path,
        TERMINATION_REASONS,
        RECOVERABLE_FAILURE_EVENTS,
        flag_once,
    )
    goal_view_totals = read_goal_view_totals(continue_path, flag_once)

    if not flag_once and num_total >= evaluation_total:
        raise ValueError(
            f"Already finished the configured {evaluation_total} evaluation episodes."
        )

    pbar = tqdm.tqdm(total=evaluation_total)
    goal_view_error_all = goal_view_totals["position_error_sum"]
    goal_yaw_error_all = goal_view_totals["yaw_error_sum"]
    goal_view_eval_count = goal_view_totals["eval_count"]
    goal_view_success_count = goal_view_totals["success_count"]

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
    rospy.Subscriber(
        "/ros/object_viewpoint_debug",
        String,
        object_viewpoint_debug_callback,
        queue_size=50,
    )
    rospy.Subscriber(
        "/habitat/mast3r_refine_status",
        Int32,
        mast3r_refine_status_callback,
        queue_size=10,
    )
    state_pub = rospy.Publisher("/habitat/state", Int32, queue_size=10)
    trigger_pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=10)
    itm_score_pub = rospy.Publisher("/blip2/cosine_score", Float64, queue_size=10)
    confidence_threshold_pub = rospy.Publisher(
        "/detector/confidence_threshold", Float64, queue_size=10
    )
    cld_with_score_pub = rospy.Publisher(
        "/detector/clouds_with_scores", MultipleMasksWithConfidence, queue_size=10
    )
    semantic_cld_with_score_pub = rospy.Publisher(
        "/detector/semantic_clouds_with_scores", MultipleMasksWithConfidence, queue_size=10
    )
    mast3r_hint_pub = rospy.Publisher("/habitat/mast3r_hint", Float32MultiArray, queue_size=10)
    instance_stop_gate_pub = rospy.Publisher("/habitat/instance_stop_gate", Int32, queue_size=10)
    resume_exploration_pub = rospy.Publisher(
        "/habitat/resume_exploration", Int32, queue_size=10
    )
    verified_approach_target_pub = rospy.Publisher(
        "/habitat/verified_approach_target",
        MultipleMasksWithConfidence,
        queue_size=10,
    )
    progress_pub = rospy.Publisher("/habitat/progress", Int32MultiArray, queue_size=10)
    record_pub = rospy.Publisher("/habitat/record", Float32MultiArray, queue_size=10)

    episodes_remaining = 1 if flag_once else evaluation_total - num_total
    for epi in range(episodes_remaining):
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
        final_state = FINAL_RESULT.EXPLORE
        expl_result = EXPL_RESULT.EXPLORATION
        mast3r_refine_status = 0
        episode_termination_reason = None
        episode_failure_events = {event: 0 for event in RECOVERABLE_FAILURE_EVENTS}
        goal_stop_executed = False
        finish_recovery_steps = 0
        finish_resume_requests = 0
        finish_verification_max_steps = max(
            1,
            int(
                lightglue_cfg.get("finish_verification_max_steps", 8)
                if lightglue_cfg is not None
                else 8
            ),
        )
        finish_resume_max_requests = max(
            1,
            int(
                insinav_filter_cfg.get("finish_resume_max_requests", 10)
                if insinav_filter_cfg is not None
                else 10
            ),
        )
        instance_stop_gate_enabled_flag = bool(
            is_instance_imagenav and insinav_stop_gate_enabled
        )
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
        confirmed_approach_active = False
        confirmed_approach_steps = 0
        visual_approach_pending = False
        visual_approach_steps = 0
        visual_approach_last_depth = float("inf")
        visual_approach_max_steps = max(
            0,
            int(
                lightglue_cfg.get("visual_approach_max_steps", 2)
                if lightglue_cfg is not None
                else 2
            ),
        )
        visual_approach_min_depth_progress = float(
            lightglue_cfg.get("visual_approach_min_depth_progress", 0.10)
            if lightglue_cfg is not None
            else 0.10
        )
        verified_approach_candidate_idx = None
        verified_approach_cloud = None
        refine_latched = False
        refine_trigger_source = None
        refine_goal_sent = False
        refine_goal_sent_step = -1
        refine_goal_hint = (0.0, 0.0, 0.0, 0.0, 0.0)
        refine_finish_wait_cycles = 0
        mast3r_direct_score_streak = 0
        mast3r_direct_candidate_mask = None
        mast3r_direct_candidate_label = None
        mast3r_direct_latched_mask = None
        mast3r_latched_current_mask = None
        mast3r_one_shot_goal_committed = False
        mast3r_execution_retry_after_step = 0
        mast3r_direct_retry_after_step = 0
        lightglue_no_box_retry_after_step = 0
        yaw_only_turn_steps = 0
        mast3r_adjustment_used = False
        episode_route_stats = {
            route: {
                "triggers": 0,
                "geometry_passes": 0,
                "geometry_rejects": 0,
                "approaches": 0,
                "stops": 0,
            }
            for route in ("lightglue", "lightglue_no_box", "dino_direct")
        }
        episode_last_accepted_route = None
        episode_stop_route = None
        fine_yaw_action_pending = False
        pending_fine_yaw_turn_deg = None
        episode_goal_image = None
        episode_goal_pose = None
        goal_clean_crop = None
        goal_clean_mask = None
        goal_clean_label_idx = None
        last_raw_rgb = observations["rgb"].copy()
        last_depth = observations["depth"].copy()

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
            episode_goal_pose = _resolve_episode_goal_image_pose(
                cfg.habitat.dataset.data_path.format(split=cfg.habitat.dataset.split),
                env.current_episode,
            )
            ros_pub.publish_goal_image(goal_image)

            if mast3r_refiner is not None and mast3r_refiner.available:
                mast3r_refiner.set_goal_image(goal_image)
                print(
                    "[INSiNav_MASt3R] goal image initialized for non-oracle close-range refinement"
                )
            
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
                use_label_filter=True,
            )
            if len(goal_mask_list) > 0:
                # `label=0` is the primary target category; `label=1` is only a
                # similar-category fallback. Current observations may score both kinds of
                # masks, but the single DINO reference crop must not let a high-confidence
                # chair/bench detection replace an available target couch detection.
                valid_goal_indices = list(
                    range(min(len(goal_score_list), len(goal_mask_list), len(goal_label_list)))
                )
                primary_goal_indices = [
                    idx for idx in valid_goal_indices if goal_label_list[idx] == 0
                ]
                fallback_goal_indices = [
                    idx for idx in valid_goal_indices if goal_label_list[idx] != 0
                ]

                goal_reference_source = None
                best_idx = None
                for source, candidate_indices in (
                    ("target", primary_goal_indices),
                    ("fallback_similar", fallback_goal_indices),
                ):
                    for idx in sorted(
                        candidate_indices,
                        key=lambda candidate_idx: goal_score_list[candidate_idx],
                        reverse=True,
                    ):
                        candidate_crop = crop_from_mask(
                            goal_image, goal_mask_list[idx], clip_crop_padding
                        )
                        if candidate_crop is None:
                            continue
                        best_idx = idx
                        goal_clean_mask = goal_mask_list[idx]
                        goal_clean_crop = candidate_crop
                        goal_reference_source = source
                        break
                    if best_idx is not None:
                        break

                if best_idx is not None:
                    goal_detected_label = int(goal_label_list[best_idx])
                    goal_clean_label_idx = goal_detected_label
                    clip_client.set_goal_clean_crop(goal_clean_crop)
                    print(
                        f"[INSiNav] Goal clean crop extracted ({goal_reference_source}): "
                        f"shape={goal_clean_crop.shape}, score={goal_score_list[best_idx]:.3f}, "
                        f"label_idx={goal_detected_label}, primary_candidates={len(primary_goal_indices)}, "
                        f"fallback_candidates={len(fallback_goal_indices)}"
                    )
                else:
                    print("[INSiNav] No valid goal-object crop; using full goal image")
            else:
                print(
                    f"[INSiNav] No '{target_label_for_detection}' object detected in goal image; "
                    "using full goal image as fallback"
                )
            
            if lightglue_verifier is not None and lightglue_verifier.available:
                lightglue_mode = (
                    str(lightglue_cfg.get("verification_mode", "full_frame")).lower()
                    if lightglue_cfg is not None
                    else "full_frame"
                )
                if lightglue_mode == "crop":
                    if goal_clean_crop is not None:
                        lightglue_set_goal(lightglue_verifier, goal_clean_crop, goal_key="crop")
                        print("[INSiNav_LightGlue] crop-crop verification goal initialized")
                    else:
                        lightglue_clear_goal(lightglue_verifier, goal_key="crop")
                        print(
                            "[INSiNav_LightGlue] goal crop unavailable; crop-crop verification disabled"
                        )
                else:
                    lightglue_set_goal(lightglue_verifier, goal_image, goal_key="full_frame")
                    if goal_clean_crop is not None:
                        # Keep a crop reference as an association check when multiple same-class
                        # instances share a frame. Full-frame matches alone cannot identify which
                        # detected couch produced the matching keypoints.
                        lightglue_set_goal(lightglue_verifier, goal_clean_crop, goal_key="crop")
                        print(
                            "[INSiNav_LightGlue] full-frame verification goal initialized "
                            "with crop association reference"
                        )
                    else:
                        lightglue_clear_goal(lightglue_verifier, goal_key="crop")
                        print("[INSiNav_LightGlue] full-frame verification goal initialized")
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
        publish_int32(instance_stop_gate_pub, int(instance_stop_gate_enabled_flag))
        publish_verified_approach_target(verified_approach_target_pub)

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
                episode_termination_reason = planner_failure_termination_reason(
                    final_state, expl_result
                ) or "step_limit"
                print(
                    f"[INSiNav] Reached max_episode_steps={max_episode_steps}. "
                    f"Terminate current episode loop with reason={episode_termination_reason}."
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
                episode_termination_reason = "infeasible"
                break

            # A planner failure STOP is an episode outcome, not a claim that the goal was found.
            # Let it bypass the instance-verification gate so FINISH cannot devolve into an
            # unbounded sequence of Python-side TURN_LEFT refresh actions.
            planner_failure_reason = planner_failure_termination_reason(
                final_state, expl_result
            )
            if ros_state == ROS_STATE.FINISH and planner_failure_reason is not None:
                episode_termination_reason = planner_failure_reason
                print(
                    "[INSiNav_TERMINATION] accept planner failure exit without target "
                    f"verification: reason={episode_termination_reason}, "
                    f"final_state={final_state}, expl_result={expl_result}, step={count_steps}"
                )
                break

            # Parse action from decision system
            action = None
            action_source = None
            fine_yaw_turn_deg_for_step = None
            if ros_state != ROS_STATE.FINISH:
                finish_recovery_steps = 0
                finish_resume_requests = 0
            if is_instance_imagenav and refine_latched and refine_goal_sent:
                max_pending_steps = 48
                max_pending_steps = max(
                    1,
                    int(mast3r_cfg.get("max_pending_steps", max_pending_steps))
                    if mast3r_cfg is not None
                    else max_pending_steps,
                )
                if (
                    refine_goal_sent_step >= 0
                    and count_steps - refine_goal_sent_step >= max_pending_steps
                ):
                    episode_failure_events["mast3r_tracking_failure"] += 1
                    timed_out_source = refine_trigger_source or "unknown"
                    mast3r_one_shot_goal_committed = False
                    mast3r_execution_retry_after_step = (
                        count_steps + mast3r_execution_retry_cooldown_steps
                    )
                    refine_latched = False
                    refine_trigger_source = None
                    refine_goal_sent = False
                    refine_goal_sent_step = -1
                    refine_finish_wait_cycles = 0
                    mast3r_adjustment_used = False
                    confirmed_approach_active = False
                    confirmed_approach_steps = 0
                    visual_approach_pending = False
                    yaw_only_turn_steps = 0
                    fine_yaw_action_pending = False
                    pending_fine_yaw_turn_deg = None
                    mast3r_refine_status = 0
                    global_action = None
                    publish_mast3r_hint(
                        mast3r_hint_pub,
                        active=False,
                        allow_stop=False,
                        yaw_error_deg=0.0,
                        forward_error=0.0,
                        lateral_error=0.0,
                        transl_error=0.0,
                        depth_error=0.0,
                    )
                    verified_approach_cloud = None
                    publish_verified_approach_target(verified_approach_target_pub)
                    publish_int32(state_pub, HABITAT_STATE.ACTION_FINISH)
                    print(
                        "[INSiNav_REFINE] MASt3R local goal pending too long; "
                        f"source={timed_out_source}, release it without STOP; "
                        "the same visual routes may retry after step "
                        f"{mast3r_execution_retry_after_step}."
                    )
                    rate.sleep()
                    continue
            if (
                is_instance_imagenav
                and refine_latched
                and refine_goal_sent
                and mast3r_refine_status == 2
            ):
                fine_yaw_turn_deg = mast3r_yaw_only_params(
                    cfg, mast3r_cfg, mast3r_refiner
                )
                fine_yaw_action_pending = True
                pending_fine_yaw_turn_deg = fine_yaw_turn_deg
                mast3r_refine_status = 0
                print(
                    "[INSiNav_REFINE] ROS requested MASt3R final yaw alignment; "
                    f"next TURN will use fine Habitat yaw={fine_yaw_turn_deg:.2f}deg."
                )

            if (
                is_instance_imagenav
                and refine_latched
                and refine_goal_sent
                and mast3r_refine_status != 0
            ):
                status = mast3r_refine_status
                status_name = {
                    1: "local_goal_reached",
                    -1: "local_goal_failed",
                    -2: "local_goal_planning_failed",
                    -3: "local_goal_tracking_failed",
                    2: "fine_yaw_turn",
                }.get(status, f"unknown({status})")
                mast3r_refine_status = 0
                refine_goal_sent = False
                refine_goal_sent_step = -1
                refine_finish_wait_cycles = 0
                print(
                    "[INSiNav_REFINE] "
                    f"ROS reported MASt3R {status_name}; verify before taking another action."
                )
                if status == 1:
                    if mast3r_adjustment_used:
                        publish_mast3r_hint(
                            mast3r_hint_pub,
                            active=True,
                            allow_stop=True,
                            yaw_error_deg=0.0,
                            forward_error=0.0,
                            lateral_error=0.0,
                            transl_error=0.0,
                            depth_error=0.0,
                        )
                        insinav_stop_gate_passed = True
                        global_action = ACTION.STOP
                        refine_latched = False
                        refine_trigger_source = None
                        refine_goal_sent = False
                        refine_goal_sent_step = -1
                        mast3r_adjustment_used = False
                        insinav_stop_reject_count = 0
                        confirmed_approach_active = False
                        confirmed_approach_steps = 0
                        yaw_only_turn_steps = 0
                        fine_yaw_action_pending = False
                        pending_fine_yaw_turn_deg = None
                        print(
                            "[INSiNav_REFINE] ROS reached the locked MASt3R position "
                            "and yaw; accept STOP without visual re-estimation."
                        )
                        rate.sleep()
                        continue
                    refine_latched = False
                    refine_trigger_source = None
                    refine_goal_sent = False
                    refine_goal_sent_step = -1
                    confirmed_approach_active = False
                    confirmed_approach_steps = 0
                    global_action = None
                    publish_mast3r_hint(
                        mast3r_hint_pub,
                        active=False,
                        allow_stop=False,
                        yaw_error_deg=0.0,
                        forward_error=0.0,
                        lateral_error=0.0,
                        transl_error=0.0,
                        depth_error=0.0,
                    )
                    verified_approach_cloud = None
                    publish_verified_approach_target(verified_approach_target_pub)
                    publish_int32(state_pub, HABITAT_STATE.ACTION_FINISH)
                    print(
                        "[INSiNav_REFINE] ROS reported arrival without an active committed "
                        "MASt3R goal; release safely without STOP."
                    )
                    rate.sleep()
                    continue
                elif status in (-1, -2, -3):
                    failure_event = (
                        "mast3r_planning_failure"
                        if status == -2
                        else "mast3r_tracking_failure"
                    )
                    episode_failure_events[failure_event] += 1
                    failed_source = refine_trigger_source or "unknown"
                    mast3r_one_shot_goal_committed = False
                    mast3r_execution_retry_after_step = (
                        count_steps + mast3r_execution_retry_cooldown_steps
                    )
                    refine_latched = False
                    refine_trigger_source = None
                    refine_goal_sent = False
                    refine_goal_sent_step = -1
                    mast3r_adjustment_used = False
                    confirmed_approach_active = False
                    confirmed_approach_steps = 0
                    visual_approach_pending = False
                    yaw_only_turn_steps = 0
                    fine_yaw_action_pending = False
                    pending_fine_yaw_turn_deg = None
                    global_action = None
                    publish_mast3r_hint(
                        mast3r_hint_pub,
                        active=False,
                        allow_stop=False,
                        yaw_error_deg=0.0,
                        forward_error=0.0,
                        lateral_error=0.0,
                        transl_error=0.0,
                        depth_error=0.0,
                    )
                    verified_approach_cloud = None
                    publish_verified_approach_target(verified_approach_target_pub)
                    publish_int32(state_pub, HABITAT_STATE.ACTION_FINISH)
                    print(
                        "[INSiNav_REFINE] one-shot MASt3R A* planning/tracking failed; "
                        f"source={failed_source}, release all approach state without "
                        "STOP; allow a fresh fixed goal after step "
                        f"{mast3r_execution_retry_after_step}."
                    )
                    rate.sleep()
                    continue
            if (
                action is None
                and is_instance_imagenav
                and visual_approach_pending
                and confirmed_approach_active
                and not refine_latched
                and global_action is None
                and visual_approach_steps < visual_approach_max_steps
            ):
                action = HabitatSimActions.move_forward
                action_source = "lightglue_visual_approach"
                visual_approach_pending = False
                visual_approach_steps += 1
                print(
                    "[INSiNav_ACTION_ARBITER] "
                    "source=lightglue_visual_approach, action=move_forward, "
                    f"step={visual_approach_steps}/{visual_approach_max_steps}"
                )
                print(
                    "[INSiNav_APPROACH] LightGlue confirmed target without a "
                    "navigable object cloud; take one visual forward step and "
                    "re-evaluate."
                )
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
                    episode_failure_events["verification_failure"] += 1
                    insinav_stop_reject_count += 1
                    print(
                        "[INSiNav_STOP_GATE] reject ROS STOP request: "
                        f"verification not passed. reject_count={insinav_stop_reject_count}/"
                        f"{insinav_stop_reject_turn_patience}"
                    )
                    global_action = None
                    fine_yaw_action_pending = False
                    pending_fine_yaw_turn_deg = None
                elif (
                    is_instance_imagenav
                    and insinav_stop_gate_enabled
                    and global_action != ACTION.STOP
                ):
                    insinav_stop_reject_count = 0

                if action is None and global_action is not None:
                    global_action_code = global_action
                    action, pitch_delta = action_code_to_habitat(global_action)
                    if global_action_code == ACTION.STOP:
                        goal_stop_executed = True
                    camera_pitch += pitch_delta
                    if (
                        fine_yaw_action_pending
                        and global_action_code in (ACTION.TURN_LEFT, ACTION.TURN_RIGHT)
                    ):
                        fine_yaw_turn_deg_for_step = pending_fine_yaw_turn_deg
                        action_source = "mast3r_yaw"
                        fine_yaw_action_pending = False
                        pending_fine_yaw_turn_deg = None
                    else:
                        action_source = "ros"
                        fine_yaw_action_pending = False
                        pending_fine_yaw_turn_deg = None
                    print(
                        "[INSiNav_ACTION_ARBITER] "
                        f"source={action_source}, action={action_code_name(global_action_code)}"
                    )

                global_action = None

            if (
                action is None
                and global_action is None
                and fine_yaw_action_pending
                and not (is_instance_imagenav and refine_latched and refine_goal_sent)
            ):
                fine_yaw_action_pending = False
                pending_fine_yaw_turn_deg = None

            if action is None:
                if (
                    is_instance_imagenav
                    and insinav_stop_gate_enabled
                    and ros_state == ROS_STATE.FINISH
                    and not insinav_stop_gate_passed
                ):
                    finish_recovery_steps += 1
                    if (
                        refine_latched
                        and not refine_goal_sent
                        and finish_recovery_steps > finish_verification_max_steps
                    ):
                        episode_failure_events["mast3r_planning_failure"] += 1
                        refine_latched = False
                        refine_trigger_source = None
                        confirmed_approach_active = False
                        confirmed_approach_steps = 0
                        verified_approach_candidate_idx = None
                        verified_approach_cloud = None
                        visual_approach_pending = False
                        publish_mast3r_hint(
                            mast3r_hint_pub,
                            active=False,
                            allow_stop=False,
                            yaw_error_deg=0.0,
                            forward_error=0.0,
                            lateral_error=0.0,
                            transl_error=0.0,
                            depth_error=0.0,
                        )
                        publish_verified_approach_target(verified_approach_target_pub)
                        print(
                            "[INSiNav_REFINE] release FINISH refinement without a local "
                            f"goal after {finish_verification_max_steps} visual refresh steps."
                        )
                    if refine_latched:
                        if refine_goal_sent:
                            max_finish_wakeup_retries = 8
                            if mast3r_cfg is not None:
                                max_finish_wakeup_retries = max(
                                    1,
                                    int(
                                        mast3r_cfg.get(
                                            "finish_wakeup_retries",
                                            max_finish_wakeup_retries,
                                        )
                                    ),
                                )
                            refine_finish_wait_cycles += 1
                            (
                                hint_yaw,
                                hint_forward,
                                hint_lateral,
                                hint_transl,
                                hint_depth,
                            ) = refine_goal_hint
                            publish_mast3r_hint(
                                mast3r_hint_pub,
                                active=True,
                                allow_stop=False,
                                yaw_error_deg=hint_yaw,
                                forward_error=hint_forward,
                                lateral_error=hint_lateral,
                                transl_error=hint_transl,
                                depth_error=hint_depth,
                            )
                            if refine_finish_wait_cycles <= max_finish_wakeup_retries:
                                print(
                                    "[INSiNav_STOP_GATE] ROS is still FINISH while a MASt3R "
                                    "local goal is pending; resend the locked goal to wake ROS "
                                    f"A* ({refine_finish_wait_cycles}/"
                                    f"{max_finish_wakeup_retries})."
                                )
                                rate.sleep()
                                continue

                            failed_source = refine_trigger_source or "unknown"
                            episode_failure_events["mast3r_tracking_failure"] += 1
                            mast3r_one_shot_goal_committed = False
                            mast3r_execution_retry_after_step = (
                                count_steps + mast3r_execution_retry_cooldown_steps
                            )
                            refine_latched = False
                            refine_trigger_source = None
                            refine_goal_sent = False
                            refine_goal_sent_step = -1
                            refine_finish_wait_cycles = 0
                            mast3r_adjustment_used = False
                            confirmed_approach_active = False
                            confirmed_approach_steps = 0
                            visual_approach_pending = False
                            yaw_only_turn_steps = 0
                            fine_yaw_action_pending = False
                            pending_fine_yaw_turn_deg = None
                            publish_mast3r_hint(
                                mast3r_hint_pub,
                                active=False,
                                allow_stop=False,
                                yaw_error_deg=0.0,
                                forward_error=0.0,
                                lateral_error=0.0,
                                transl_error=0.0,
                                depth_error=0.0,
                            )
                            action = HabitatSimActions.turn_left
                            action_source = "mast3r_finish_wakeup_failed"
                            print(
                                "[INSiNav_STOP_GATE] ROS did not leave FINISH after "
                                f"{max_finish_wakeup_retries} locked-goal retries; release "
                                f"source={failed_source} and refresh visual evidence; retry "
                                f"after step {mast3r_execution_retry_after_step}."
                            )
                        else:
                            action = HabitatSimActions.turn_left
                            action_source = "refine_refresh"
                            refine_goal_sent = False
                            refine_goal_sent_step = -1
                            print(
                                "[INSiNav_ACTION_ARBITER] "
                                "source=refine_refresh, action=turn_left"
                            )
                            print(
                                "[INSiNav_STOP_GATE] ROS entered FINISH during latched refine; "
                                "refresh visual alignment instead of forcing MOVE_FORWARD."
                            )
                    else:
                        if finish_recovery_steps > finish_verification_max_steps:
                            episode_failure_events["verification_failure"] += 1
                            confirmed_approach_active = False
                            confirmed_approach_steps = 0
                            verified_approach_candidate_idx = None
                            verified_approach_cloud = None
                            visual_approach_pending = False
                            publish_verified_approach_target(verified_approach_target_pub)
                        finish_forward_margin = 0.15
                        if lightglue_cfg is not None:
                            finish_forward_margin = float(
                                lightglue_cfg.get("finish_forward_margin", finish_forward_margin)
                            )
                        if confirmed_approach_active:
                            if (
                                np.isfinite(stop_depth_distance)
                                and stop_depth_distance > stop_distance + finish_forward_margin
                            ):
                                action = HabitatSimActions.move_forward
                                action_source = "finish_approach"
                                print(
                                    "[INSiNav_ACTION_ARBITER] "
                                    "source=finish_approach, action=move_forward"
                                )
                                print(
                                    "[INSiNav_STOP_GATE] ROS entered FINISH but stop gate not passed; "
                                    f"continue confirmed approach with MOVE_FORWARD, "
                                    f"depth_distance={stop_depth_distance:.3f}, "
                                    f"stop_distance={stop_distance:.3f}"
                                )
                            else:
                                action = HabitatSimActions.turn_left
                                action_source = "finish_refresh"
                                print(
                                    "[INSiNav_ACTION_ARBITER] "
                                    "source=finish_refresh, action=turn_left"
                                )
                                print(
                                    "[INSiNav_STOP_GATE] ROS entered FINISH near confirmed target; "
                                    "rotate to refresh close-range verification."
                                )
                        else:
                            finish_resume_requests += 1
                            publish_int32(resume_exploration_pub, 1)
                            print(
                                "[INSiNav_STOP_GATE] ROS entered FINISH without valid target "
                                "verification; request normal exploration replanning "
                                f"({finish_resume_requests}/{finish_resume_max_requests})."
                            )
                            if finish_resume_requests >= finish_resume_max_requests:
                                episode_termination_reason = "planner_resume_failure"
                                print(
                                    "[INSiNav_TERMINATION] planner did not accept bounded "
                                    "resume requests; terminate without spinning to step limit."
                                )
                                break
                            rate.sleep()
                            continue
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
                active_goal_labels = [label]
            print(
                f"Finding [{step_label}] (active_candidates={active_goal_labels}); Action: {action};"
            )

            # Notify ROS system that action execution is starting
            publish_int32(state_pub, HABITAT_STATE.ACTION_EXEC)

            observations = step_habitat_action(
                env,
                action,
                fine_yaw_turn_deg=fine_yaw_turn_deg_for_step,
            )
            info = env.get_metrics()

            if is_instance_imagenav and (refine_latched or refine_goal_sent):
                gps = np.asarray(observations.get("gps", []), dtype=np.float64).reshape(-1)
                world_x = float(-gps[2]) if gps.size >= 3 else float("nan")
                world_y = float(-gps[0]) if gps.size >= 3 else float("nan")
                compass_value = np.asarray(
                    observations.get("compass", float("nan")), dtype=np.float64
                ).reshape(-1)
                compass = (
                    float(compass_value[0])
                    if compass_value.size > 0
                    else float("nan")
                )
                collision_metrics = info.get("collisions", {})
                is_collision = (
                    bool(collision_metrics.get("is_collision", False))
                    if isinstance(collision_metrics, dict)
                    else False
                )
                collision_count = (
                    int(collision_metrics.get("count", 0))
                    if isinstance(collision_metrics, dict)
                    else -1
                )
                print(
                    "[INSiNav_MOTION] "
                    f"step={count_steps}, action={action}, world=({world_x:.3f},"
                    f"{world_y:.3f}), compass={compass:.4f}, "
                    f"collision={int(is_collision)}, collision_count={collision_count}, "
                    f"refine_sent={int(refine_goal_sent)}"
                )

            # Detect objects in the current observation first (candidate regions for insinav CLIP)
            raw_rgb = observations["rgb"].copy()
            last_raw_rgb = raw_rgb.copy()
            last_depth = observations["depth"].copy()
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
                publish_semantic_object_clouds = bool(
                    clip_cfg.get("publish_semantic_object_clouds", True)
                )
                no_candidate_mode = str(clip_cfg.get("no_candidate_mode", "low_score"))
                no_candidate_score = float(clip_cfg.get("no_candidate_score", 0.0))
                dense_full_frame_value_map = bool(
                    clip_cfg.get("dense_full_frame_value_map", True)
                )
                full_frame_weight = float(clip_cfg.get("full_frame_weight", 0.45))
                crop_score_weight = float(clip_cfg.get("crop_score_weight", 1.0))
                clip_scores = []
                ranked_crop_scores = []
                full_frame_score = clip_client.cosine(raw_rgb)
                full_frame_base_score = float(
                    np.clip(full_frame_weight * full_frame_score, 0.0, 1.0)
                )
                max_match_points = 0
                full_frame_match_points = 0
                stop_match_points = 0
                stop_depth_distance = float("inf")
                stop_candidate_count = 0
                approach_match_points = 0
                approach_depth_distance = float("inf")
                approach_candidate_idx = None
                verified_approach_candidate_idx = None
                instance_crop_match_points = 0
                instance_crop_association_required = False
                instance_crop_association_passed = True

                # Step 2: Compare clean observation crops only with the clean goal crop.
                if (
                    use_candidate_crops
                    and goal_clean_crop is not None
                    and len(object_masks_list) > 0
                ):
                    for idx, object_mask in enumerate(object_masks_list):
                        crop = crop_from_mask(raw_rgb, object_mask, clip_crop_padding)
                        if crop is not None:
                            crop_score = clip_client.cosine_clean_crop(crop)
                            clip_scores.append(crop_score)
                            ranked_crop_scores.append((idx, float(crop_score)))

                if len(clip_scores) > 0:
                    ranked_crop_scores.sort(key=lambda item: item[1], reverse=True)
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
                        cosine = full_frame_score
                        print(
                            f"[INSiNav] DINO full-frame cosine similarity (fallback): {cosine:.3f}"
                        )
                    else:
                        cosine = no_candidate_score
                        pass
                if dense_full_frame_value_map:
                    value_map_score = full_frame_base_score
                    value_map_score_source = "weighted_full_frame_base"
                else:
                    value_map_score = float(np.clip(crop_score_weight * cosine, 0.0, 1.0))
                    value_map_score_source = "crop_topk_dense"
                best_crop_score = ranked_crop_scores[0][1] if ranked_crop_scores else None
                best_crop_msg = (
                    f"{best_crop_score:.3f}" if best_crop_score is not None else "NA"
                )
                delta_raw_full_to_best_crop = (
                    full_frame_score - best_crop_score
                    if best_crop_score is not None
                    else float("nan")
                )
                delta_raw_full_to_best_crop_msg = (
                    f"{delta_raw_full_to_best_crop:.3f}"
                    if np.isfinite(delta_raw_full_to_best_crop)
                    else "NA"
                )
                top_crop_preview = ", ".join(
                    f"{score:.3f}" for _, score in ranked_crop_scores[:3]
                )
                if not top_crop_preview:
                    top_crop_preview = "NA"
                print(
                    f"[INSiNav_DEBUG] step={count_steps}, clip_candidate_crops={len(clip_scores)}, "
                    f"clip_mode={no_candidate_mode}, dino_score={cosine:.3f}, "
                    f"value_map_score={value_map_score:.3f}"
                )
                print(
                    f"[INSiNav_MAP_DEBUG] step={count_steps}, published_source={value_map_score_source}, "
                    f"published_value_map_score={value_map_score:.3f}, "
                    f"full_frame_raw={full_frame_score:.3f}, "
                    f"full_frame_base={full_frame_base_score:.3f}, "
                    f"full_frame_weight={full_frame_weight:.3f}, crop_topk_mean={cosine:.3f}, "
                    f"crop_score_weight={crop_score_weight:.3f}, "
                    f"best_crop_score={best_crop_msg}, "
                    f"delta_raw_full_minus_best_crop={delta_raw_full_to_best_crop_msg}, "
                    f"top3_crop_scores=[{top_crop_preview}]"
                )
            else:
                cosine = get_itm_message_cosine(observations["rgb"], label, room)
                value_map_score = cosine
                print(f"Target related room: {room}")
                print(f"ITM cosine similarity: {cosine:.3f}")

            # LightGlue verifies the instance for approach/STOP. The default
            # full-frame mode follows UniGoal-style current-frame vs goal-image
            # matching; crop mode is kept for ablation.
            if (
                is_instance_imagenav
                and lightglue_verifier is not None
                and lightglue_verifier.available
            ):
                lightglue_mode = (
                    str(lightglue_cfg.get("verification_mode", "full_frame")).lower()
                    if lightglue_cfg is not None
                    else "full_frame"
                )
                crop_padding_ratio = float(lightglue_cfg.get("crop_padding_ratio", 0.1))
                association_trigger_points = float(
                    lightglue_cfg.get(
                        "approach_score_threshold",
                        lightglue_cfg.get("score_threshold", 100.0),
                    )
                )
                require_instance_crop_for_multi_candidate = bool(
                    lightglue_cfg.get(
                        "require_instance_crop_for_multi_candidate", True
                    )
                )
                require_instance_crop_for_any_candidate = bool(
                    lightglue_cfg.get(
                        "require_instance_crop_for_any_candidate", True
                    )
                )
                instance_crop_min_match_points = max(
                    1, int(lightglue_cfg.get("instance_crop_min_match_points", 20))
                )

                lightglue_points = []
                crop_score_by_idx = {idx: score for idx, score in ranked_crop_scores}
                best_stop_idx = None

                full_frame_goal_key = "full_frame"
                if lightglue_mode == "full_frame" and not lightglue_has_goal(
                    lightglue_verifier, full_frame_goal_key
                ):
                    if lightglue_has_goal(lightglue_verifier, "default"):
                        full_frame_goal_key = "default"
                        print(
                            "[INSiNav_LightGlue] warning: full_frame goal key missing; "
                            "fall back to default goal features."
                        )

                if lightglue_mode == "full_frame" and lightglue_has_goal(
                    lightglue_verifier, full_frame_goal_key
                ):
                    full_frame_match_points = lightglue_match_points(
                        lightglue_verifier, raw_rgb, goal_key=full_frame_goal_key
                    )
                    lightglue_points = [full_frame_match_points]
                    max_match_points = int(full_frame_match_points)
                    stop_match_points = int(full_frame_match_points)
                    approach_match_points = int(full_frame_match_points)
                    stop_candidate_count = len(object_masks_list)

                    if ranked_crop_scores:
                        best_stop_idx = int(ranked_crop_scores[0][0])
                    elif len(object_masks_list) > 0:
                        best_stop_idx = 0

                    # A full-frame match only establishes scene-level similarity. Any current
                    # detection must independently match the goal crop before it can own
                    # approach or MASt3R; this also prevents a lone similar-class box from
                    # bypassing the old multi-candidate-only check.
                    instance_crop_association_required = (
                        boxed_lightglue_enabled
                        and boxed_crop_ownership_enabled
                        and (
                            full_frame_match_points >= association_trigger_points
                            or not boxed_full_frame_gate_enabled
                        )
                        and len(object_masks_list) > 0
                        and (
                            require_instance_crop_for_any_candidate
                            or (
                                require_instance_crop_for_multi_candidate
                                and len(object_masks_list) > 1
                            )
                        )
                    )
                    instance_matches = []
                    if instance_crop_association_required:
                        instance_crop_association_passed = False
                        if lightglue_has_goal(lightglue_verifier, "crop"):
                            for idx, object_mask in enumerate(object_masks_list):
                                crop = crop_from_mask(raw_rgb, object_mask, crop_padding_ratio)
                                match_points = (
                                    lightglue_match_points(
                                        lightglue_verifier, crop, goal_key="crop"
                                    )
                                    if crop is not None
                                    else 0
                                )
                                instance_matches.append((idx, int(match_points)))

                        if instance_matches:
                            best_stop_idx, instance_crop_match_points = max(
                                instance_matches, key=lambda item: item[1]
                            )
                            instance_crop_association_passed = (
                                instance_crop_match_points
                                >= instance_crop_min_match_points
                            )
                        else:
                            instance_crop_association_passed = False

                    if best_stop_idx is not None and best_stop_idx < len(object_masks_list):
                        depth_cfg = cfg.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor
                        stop_depth_distance = estimate_mask_depth_distance(
                            observations["depth"],
                            object_masks_list[best_stop_idx],
                            float(depth_cfg.min_depth),
                            float(depth_cfg.max_depth),
                        )
                        approach_depth_distance = stop_depth_distance
                        approach_candidate_idx = best_stop_idx

                    best_crop_score = (
                        crop_score_by_idx.get(best_stop_idx, float("nan"))
                        if best_stop_idx is not None
                        else float("nan")
                    )
                    best_crop_msg = (
                        f"{best_crop_score:.3f}"
                        if np.isfinite(best_crop_score)
                        else "NA"
                    )
                    print(
                        "LightGlue full-frame match points: "
                        f"{full_frame_match_points}, "
                        f"best_depth_idx={best_stop_idx if best_stop_idx is not None else 'NA'}, "
                        f"best_dino={best_crop_msg}, "
                        f"depth_distance={stop_depth_distance:.3f}, "
                        f"instance_crop_required={int(instance_crop_association_required)}, "
                        f"instance_crop_points={instance_crop_match_points}, "
                        f"instance_crop_passed={int(instance_crop_association_passed)}"
                    )
                    if instance_crop_association_required:
                        candidate_parts = []
                        for idx, match_points in instance_matches:
                            label_idx = label_list[idx] if idx < len(label_list) else -1
                            crop_score = crop_score_by_idx.get(idx, float("nan"))
                            crop_score_msg = (
                                f"{crop_score:.3f}" if np.isfinite(crop_score) else "NA"
                            )
                            candidate_parts.append(
                                f"{idx}:label={label_idx},lg_crop={match_points},"
                                f"dino={crop_score_msg}"
                            )
                        print(
                            "[INSiNav_LG_ASSOC] "
                            f"best_idx={best_stop_idx}, min_points="
                            f"{instance_crop_min_match_points}, candidates="
                            f"[{'; '.join(candidate_parts)}]"
                        )
                elif len(object_masks_list) > 0 and lightglue_has_goal(lightglue_verifier, "crop"):
                    for object_mask in object_masks_list:
                        crop = crop_from_mask(raw_rgb, object_mask, crop_padding_ratio)
                        match_points = (
                            lightglue_match_points(lightglue_verifier, crop, goal_key="crop")
                            if crop is not None
                            else 0
                        )
                        lightglue_points.append(match_points)

                if lightglue_points:
                    if lightglue_mode == "crop":
                        max_match_points = max(lightglue_points)
                        stop_candidate_count = len(lightglue_points)
                        best_stop_idx = max(
                            range(len(lightglue_points)), key=lambda idx: lightglue_points[idx]
                        )
                        stop_match_points = int(lightglue_points[best_stop_idx])
                        instance_crop_match_points = stop_match_points
                        depth_cfg = cfg.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor
                        stop_depth_distance = estimate_mask_depth_distance(
                            observations["depth"],
                            object_masks_list[best_stop_idx],
                            float(depth_cfg.min_depth),
                            float(depth_cfg.max_depth),
                        )
                        approach_match_points = stop_match_points
                        approach_depth_distance = stop_depth_distance
                        approach_candidate_idx = best_stop_idx
                        instance_crop_association_required = True
                        instance_crop_association_passed = (
                            instance_crop_match_points >= instance_crop_min_match_points
                        )

                        print(
                            f"LightGlue crop-crop match points: max={max(lightglue_points)}, "
                            f"mean={np.mean(lightglue_points):.1f}, min={min(lightglue_points)}"
                        )
                        candidate_parts = []
                        for idx, match_points in enumerate(lightglue_points):
                            label_idx = label_list[idx] if idx < len(label_list) else -1
                            crop_score = crop_score_by_idx.get(idx, float("nan"))
                            crop_score_msg = (
                                f"{crop_score:.3f}" if np.isfinite(crop_score) else "NA"
                            )
                            candidate_parts.append(
                                f"{idx}:label={label_idx},lg={int(match_points)},dino={crop_score_msg}"
                            )
                        print(
                            "[INSiNav_LG_DEBUG] "
                            f"best_idx={best_stop_idx}, candidates=[{'; '.join(candidate_parts)}]"
                        )
            # Use LightGlue/MASt3R only for stop confirmation and close-range refinement.
            if is_instance_imagenav:
                stop_distance = success_distance
                approach_threshold = 100.0
                approach_confirmation_patience = 8
                verified_approach_enabled = True
                verified_target_score = 0.95
                visual_approach_enabled = True
                visual_forward_clearance = 0.75
                direct_cfg = (
                    mast3r_cfg.get("direct", {}) if mast3r_cfg is not None else {}
                )
                mast3r_direct_enabled = bool(
                    dino_direct_route_enabled and direct_cfg.get("enabled", False)
                )
                mast3r_direct_dino_threshold = float(
                    direct_cfg.get("dino_threshold", 0.60)
                )
                mast3r_direct_min_stable_frames = max(
                    1, int(direct_cfg.get("min_stable_frames", 2))
                )
                mast3r_direct_min_candidate_iou = float(
                    direct_cfg.get("min_candidate_iou", 0.20)
                )
                mast3r_direct_retry_cooldown_steps = max(
                    1, int(direct_cfg.get("retry_cooldown_steps", 15))
                )
                if lightglue_cfg is not None:
                    stop_distance = float(lightglue_cfg.get("stop_distance", stop_distance))
                    approach_threshold = float(
                        lightglue_cfg.get(
                            "approach_score_threshold",
                            lightglue_cfg.get("score_threshold", approach_threshold),
                        )
                    )
                    approach_confirmation_patience = max(
                        1,
                        int(
                            lightglue_cfg.get(
                                "approach_confirmation_patience",
                                approach_confirmation_patience,
                            )
                        ),
                    )
                    verified_approach_enabled = bool(
                        lightglue_cfg.get(
                            "verified_approach_enabled",
                            verified_approach_enabled,
                        )
                    )
                    verified_target_score = float(
                        lightglue_cfg.get("verified_target_score", verified_target_score)
                    )
                    visual_approach_enabled = bool(
                        lightglue_cfg.get(
                            "visual_approach_enabled",
                            visual_approach_enabled,
                        )
                    )
                    visual_forward_clearance = float(
                        lightglue_cfg.get(
                            "visual_forward_clearance",
                            visual_forward_clearance,
                        )
                    )
                if not np.isfinite(stop_depth_distance):
                    stop_depth_distance = float("inf")
                if not np.isfinite(approach_depth_distance):
                    approach_depth_distance = float("inf")

                if not refine_latched or not refine_goal_sent:
                    publish_mast3r_hint(
                        mast3r_hint_pub,
                        active=False,
                        allow_stop=False,
                        yaw_error_deg=0.0,
                        forward_error=0.0,
                        lateral_error=0.0,
                        transl_error=0.0,
                        depth_error=0.0,
                    )

                approach_candidate_label = (
                    int(label_list[approach_candidate_idx])
                    if approach_candidate_idx is not None
                    and 0 <= approach_candidate_idx < len(label_list)
                    else None
                )
                approach_candidate_matches_goal_label = (
                    approach_candidate_idx is None
                    or (
                        goal_clean_label_idx is not None
                        and approach_candidate_label == goal_clean_label_idx
                    )
                )
                boxed_full_frame_passed = (
                    not boxed_full_frame_gate_enabled
                    or full_frame_match_points >= approach_threshold
                )
                boxed_crop_evidence_passed = (
                    not boxed_crop_ownership_enabled
                    or (
                        instance_crop_association_required
                        and instance_crop_association_passed
                    )
                )
                boxed_approach_confirmed = (
                    boxed_lightglue_enabled
                    and approach_candidate_idx is not None
                    and boxed_full_frame_passed
                    and boxed_crop_evidence_passed
                    and approach_candidate_matches_goal_label
                )
                no_box_visual_confirmed = (
                    lightglue_no_box_enabled
                    and approach_candidate_idx is None
                    and full_frame_match_points >= approach_threshold
                )
                current_approach_confirmed = (
                    boxed_approach_confirmed or no_box_visual_confirmed
                )
                if current_approach_confirmed:
                    confirmed_approach_active = True
                    confirmed_approach_steps = 0
                    if approach_candidate_idx is not None:
                        # This index is valid only for the current detector output. The
                        # corresponding world-frame cloud is snapshotted after projection.
                        verified_approach_candidate_idx = int(approach_candidate_idx)
                elif confirmed_approach_active:
                    confirmed_approach_steps += 1
                    if confirmed_approach_steps >= approach_confirmation_patience:
                        episode_failure_events["verification_failure"] += 1
                        confirmed_approach_active = False
                        confirmed_approach_steps = 0
                        verified_approach_candidate_idx = None
                        verified_approach_cloud = None
                        visual_approach_pending = False
                        visual_approach_steps = 0
                        visual_approach_last_depth = float("inf")

                approach_reference_distance = approach_depth_distance
                if (
                    current_approach_confirmed
                    and approach_candidate_idx is None
                    and not np.isfinite(approach_reference_distance)
                ):
                    depth_cfg = cfg.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor
                    approach_reference_distance = estimate_center_depth_distance(
                        observations["depth"],
                        float(depth_cfg.min_depth),
                        float(depth_cfg.max_depth),
                    )

                # This detector-independent route deliberately ignores current-frame
                # boxes. The normal box + crop-association route still wins below when
                # both routes are ready on the same frame.
                lightglue_no_box_high_confidence = (
                    lightglue_no_box_enabled
                    and lightglue_verification_mode == "full_frame"
                    and full_frame_match_points >= lightglue_no_box_match_threshold
                    and count_steps >= lightglue_no_box_retry_after_step
                    and count_steps >= mast3r_execution_retry_after_step
                    and (not mast3r_use_detection_masks or goal_clean_mask is not None)
                    and not refine_latched
                    and not mast3r_one_shot_goal_committed
                    and mast3r_refiner is not None
                    and mast3r_refiner.available
                )

                no_candidate_patience_exceeded = (
                    approach_candidate_idx is None
                    and insinav_filter_enabled
                    and no_detection_steps >= insinav_no_det_patience
                )
                if (
                    current_approach_confirmed
                    and approach_candidate_idx is None
                    and not lightglue_no_box_high_confidence
                ):
                    visual_probe_failure_reason = None
                    if no_candidate_patience_exceeded:
                        visual_probe_failure_reason = "no_detection_patience"
                    elif visual_approach_steps > 0:
                        depth_progress = (
                            visual_approach_last_depth - approach_reference_distance
                        )
                        if (
                            not np.isfinite(approach_reference_distance)
                            or not np.isfinite(visual_approach_last_depth)
                            or depth_progress < visual_approach_min_depth_progress
                        ):
                            visual_probe_failure_reason = "no_depth_progress"

                    if visual_probe_failure_reason is not None:
                        episode_failure_events["verification_failure"] += 1
                        previous_visual_depth = visual_approach_last_depth
                        current_approach_confirmed = False
                        confirmed_approach_active = False
                        confirmed_approach_steps = 0
                        verified_approach_candidate_idx = None
                        verified_approach_cloud = None
                        visual_approach_pending = False
                        # Keep this no-box episode closed until a detector candidate or a fresh
                        # low-score frame resets the confirmation state.
                        visual_approach_steps = visual_approach_max_steps
                        visual_approach_last_depth = float("inf")
                        print(
                            "[INSiNav_APPROACH] release no-cloud visual approach: "
                            f"reason={visual_probe_failure_reason}, "
                            f"no_detection_steps={no_detection_steps}, "
                            f"depth={approach_reference_distance:.3f}, "
                            f"last_depth={previous_visual_depth:.3f}"
                        )

                insinav_stop_gate_passed = False

                # B0 uses the boxed instance confirmation and sensor depth only. This never
                # reads Habitat's oracle distance-to-goal metric.
                if (
                    terminal_mode == "rgbd"
                    and boxed_approach_confirmed
                    and np.isfinite(approach_reference_distance)
                    and approach_reference_distance <= stop_distance
                ):
                    insinav_stop_gate_passed = True
                    global_action = ACTION.STOP
                    episode_stop_route = "lightglue"
                    episode_route_stats["lightglue"]["stops"] += 1
                    print(
                        "[INSiNav_RGBD_TERMINAL] boxed instance confirmed; "
                        f"sensor_depth={approach_reference_distance:.3f} <= "
                        f"stop_distance={stop_distance:.3f}"
                    )

                lightglue_geometry_ready = (
                    boxed_lightglue_enabled
                    and boxed_approach_confirmed
                    and approach_candidate_idx is not None
                    and approach_candidate_matches_goal_label
                    and (not mast3r_use_detection_masks or goal_clean_mask is not None)
                )
                lightglue_should_latch_mast3r = (
                    lightglue_geometry_ready
                    and not mast3r_one_shot_goal_committed
                    and count_steps >= mast3r_execution_retry_after_step
                    and mast3r_refiner is not None
                    and mast3r_refiner.available
                )
                lightglue_no_box_should_latch_mast3r = (
                    lightglue_no_box_high_confidence
                    and not lightglue_should_latch_mast3r
                )

                goal_label_ranked_crop_scores = [
                    (idx, score)
                    for idx, score in ranked_crop_scores
                    if goal_clean_label_idx is not None
                    and 0 <= idx < len(label_list)
                    and int(label_list[idx]) == goal_clean_label_idx
                ]
                current_best_crop_score = (
                    float(goal_label_ranked_crop_scores[0][1])
                    if goal_label_ranked_crop_scores
                    else float("nan")
                )
                current_best_crop_idx = (
                    int(goal_label_ranked_crop_scores[0][0])
                    if goal_label_ranked_crop_scores
                    else None
                )
                current_best_crop_mask = (
                    object_masks_list[current_best_crop_idx]
                    if current_best_crop_idx is not None
                    and current_best_crop_idx < len(object_masks_list)
                    else None
                )
                current_best_crop_label = (
                    label_list[current_best_crop_idx]
                    if current_best_crop_idx is not None
                    and current_best_crop_idx < len(label_list)
                    else None
                )
                if (
                    mast3r_direct_enabled
                    and np.isfinite(current_best_crop_score)
                    and current_best_crop_score >= mast3r_direct_dino_threshold
                    and not confirmed_approach_active
                    and not mast3r_one_shot_goal_committed
                    and count_steps >= mast3r_direct_retry_after_step
                    and count_steps >= mast3r_execution_retry_after_step
                    and not refine_latched
                    and current_best_crop_mask is not None
                    and (not mast3r_use_detection_masks or goal_clean_mask is not None)
                ):
                    candidate_iou = mask_bbox_iou(
                        mast3r_direct_candidate_mask, current_best_crop_mask
                    )
                    same_candidate = (
                        mast3r_direct_candidate_mask is not None
                        and mast3r_direct_candidate_label == current_best_crop_label
                        and candidate_iou >= mast3r_direct_min_candidate_iou
                    )
                    mast3r_direct_score_streak = (
                        mast3r_direct_score_streak + 1 if same_candidate else 1
                    )
                    mast3r_direct_candidate_mask = np.asarray(
                        current_best_crop_mask
                    ).copy()
                    mast3r_direct_candidate_label = current_best_crop_label
                elif not refine_latched:
                    mast3r_direct_score_streak = 0
                    mast3r_direct_candidate_mask = None
                    mast3r_direct_candidate_label = None

                mast3r_direct_should_latch = (
                    mast3r_direct_enabled
                    and not lightglue_should_latch_mast3r
                    and not lightglue_no_box_should_latch_mast3r
                    and not confirmed_approach_active
                    and not mast3r_one_shot_goal_committed
                    and count_steps >= mast3r_direct_retry_after_step
                    and count_steps >= mast3r_execution_retry_after_step
                    and (not mast3r_use_detection_masks or goal_clean_mask is not None)
                    and mast3r_direct_score_streak
                    >= mast3r_direct_min_stable_frames
                    and mast3r_refiner is not None
                    and mast3r_refiner.available
                )
                should_latch_mast3r = (
                    lightglue_should_latch_mast3r
                    or lightglue_no_box_should_latch_mast3r
                    or mast3r_direct_should_latch
                )
                if (
                    current_approach_confirmed
                    and not should_latch_mast3r
                    and verified_approach_enabled
                ):
                    if approach_candidate_idx is not None:
                        visual_approach_pending = False
                        visual_approach_steps = 0
                        visual_approach_last_depth = float("inf")
                        print(
                            "[INSiNav_APPROACH] LightGlue confirmed target; "
                            "promote current candidate for object-map approach: "
                            f"idx={verified_approach_candidate_idx}, "
                            f"match_points={approach_match_points}, "
                            f"depth={approach_reference_distance:.3f}"
                        )
                    elif visual_approach_enabled and not refine_latched:
                        center_depth_distance = approach_reference_distance
                        if visual_approach_steps >= visual_approach_max_steps:
                            episode_failure_events["verification_failure"] += 1
                            visual_approach_pending = False
                            confirmed_approach_active = False
                            confirmed_approach_steps = 0
                            print(
                                "[INSiNav_APPROACH] release no-cloud visual approach: "
                                f"reason=max_steps, steps={visual_approach_steps}/"
                                f"{visual_approach_max_steps}"
                            )
                        elif (
                            np.isfinite(center_depth_distance)
                            and center_depth_distance >= visual_forward_clearance
                        ):
                            visual_approach_pending = True
                            visual_approach_last_depth = center_depth_distance
                            print(
                                "[INSiNav_APPROACH] LightGlue confirmed target without "
                                "candidate cloud; queue visual forward approach: "
                                f"match_points={approach_match_points}, "
                                f"center_depth={center_depth_distance:.3f}, "
                                f"clearance={visual_forward_clearance:.3f}"
                            )
                        else:
                            episode_failure_events["verification_failure"] += 1
                            visual_approach_pending = False
                            confirmed_approach_active = False
                            confirmed_approach_steps = 0
                            print(
                                "[INSiNav_APPROACH] LightGlue confirmed target without "
                                "candidate cloud, but center depth is invalid or too close "
                                "for blind forward approach: "
                                f"center_depth={center_depth_distance:.3f}, "
                                f"clearance={visual_forward_clearance:.3f}"
                            )

                if (
                    should_latch_mast3r
                    and not refine_latched
                    and mast3r_refiner is not None
                    and mast3r_refiner.available
                ):
                    refine_latched = True
                    if lightglue_should_latch_mast3r:
                        refine_trigger_source = "lightglue"
                    elif lightglue_no_box_should_latch_mast3r:
                        refine_trigger_source = "lightglue_no_box"
                    else:
                        refine_trigger_source = "dino_direct"
                    episode_route_stats[refine_trigger_source]["triggers"] += 1
                    if not mast3r_use_detection_masks:
                        mast3r_latched_current_mask = None
                    elif refine_trigger_source == "dino_direct":
                        mast3r_direct_latched_mask = np.asarray(
                            current_best_crop_mask
                        ).copy()
                        mast3r_latched_current_mask = mast3r_direct_latched_mask
                    elif refine_trigger_source == "lightglue_no_box":
                        mast3r_latched_current_mask = np.ones(
                            raw_rgb.shape[:2], dtype=bool
                        )
                    else:
                        mast3r_latched_current_mask = np.asarray(
                            object_masks_list[approach_candidate_idx]
                        ).copy()
                    refine_goal_sent = False
                    refine_goal_sent_step = -1
                    visual_approach_pending = False
                    if refine_trigger_source == "lightglue":
                        print(
                            "[INSiNav_REFINE] LightGlue confirmed target; immediately "
                            "latched MASt3R refinement without a depth gate: "
                            f"depth={approach_reference_distance:.3f}"
                        )
                    elif refine_trigger_source == "lightglue_no_box":
                        print(
                            "[INSiNav_LG_NOBOX] detector-independent full-frame "
                            "LightGlue triggered one-shot MASt3R: "
                            f"match_points={full_frame_match_points}, "
                            f"threshold={lightglue_no_box_match_threshold:g}, "
                            "geometry=full_frame"
                        )
                    else:
                        print(
                            "[INSiNav_MASt3R_DIRECT] stable DINO crop candidate triggered "
                            "one-shot MASt3R without a candidate-depth gate: "
                            f"score={current_best_crop_score:.3f}, "
                            f"threshold={mast3r_direct_dino_threshold:.3f}, "
                            f"stable_frames={mast3r_direct_score_streak}, "
                            f"candidate_idx={current_best_crop_idx}"
                        )

                print(
                    f"[INSiNav_DEBUG] step={count_steps}, lg_max_points={max_match_points}, "
                    f"stop_match_points={stop_match_points}, "
                    f"stop_depth_distance={stop_depth_distance:.3f}, "
                    f"approach_active={int(confirmed_approach_active)}, "
                    f"approach_match_points={approach_match_points}, "
                    f"approach_depth_distance={approach_depth_distance:.3f}, "
                    f"approach_ref_distance={approach_reference_distance:.3f}, "
                    f"approach_idx={approach_candidate_idx}, "
                    f"approach_label={approach_candidate_label}, "
                    f"goal_label={goal_clean_label_idx}, "
                    f"label_match={int(approach_candidate_matches_goal_label)}, "
                    f"crop_assoc={int(instance_crop_association_required)}/"
                    f"{int(instance_crop_association_passed)}, "
                    f"lg_no_box_ready={int(lightglue_no_box_high_confidence)}, "
                    f"lg_no_box_retry_after={lightglue_no_box_retry_after_step}, "
                    f"mast3r_retry_after={mast3r_execution_retry_after_step}, "
                    f"stop_candidates={stop_candidate_count}, dino_score={cosine:.3f}"
                )

                mast3r_result = None
                mast3r_hint_active = False
                mast3r_result_accepted = False
                should_run_mast3r_refine = (
                    mast3r_refiner is not None
                    and mast3r_refiner.available
                    and episode_goal_image is not None
                    and refine_latched
                    and not refine_goal_sent
                )
                if (
                    should_run_mast3r_refine
                ):
                    mast3r_goal_mask = (
                        goal_clean_mask if mast3r_use_detection_masks else None
                    )
                    mast3r_current_mask = (
                        mast3r_latched_current_mask
                        if mast3r_use_detection_masks
                        else None
                    )
                    print(
                        "[INSiNav_MASt3R] correspondence_selection="
                        f"{mast3r_geometry_matching_mode}, "
                        "goal_mask="
                        f"{int(mast3r_goal_mask is not None)}, "
                        "current_mask="
                        f"{int(mast3r_current_mask is not None)}"
                    )
                    mast3r_result = run_mast3r_refine_on_observation(
                        mast3r_refiner,
                        raw_rgb,
                        observations["depth"],
                        rgb_sensor_cfg,
                        depth_sensor_cfg,
                        goal_mask=mast3r_goal_mask,
                        current_mask=mast3r_current_mask,
                    )
                    log_mast3r_result(
                        mast3r_result,
                        geometry_mode=mast3r_geometry_matching_mode,
                    )
                    (
                        mast3r_result_accepted,
                        mast3r_validation_reason,
                    ) = validate_mast3r_result(mast3r_result, mast3r_cfg)
                    print(
                        "[INSiNav_MASt3R] geometry_check="
                        f"{'pass' if mast3r_result_accepted else 'reject'}, "
                        f"route={refine_trigger_source}, "
                        f"reason={mast3r_validation_reason}"
                    )
                    route_result_key = (
                        "geometry_passes" if mast3r_result_accepted else "geometry_rejects"
                    )
                    episode_route_stats[refine_trigger_source][route_result_key] += 1
                    if mast3r_result_accepted:
                        episode_last_accepted_route = refine_trigger_source
                    mast3r_hint_active = mast3r_result_accepted

                mast3r_hint_allow_stop = False
                mast3r_hint_yaw = 0.0
                mast3r_hint_forward = 0.0
                mast3r_hint_lateral = 0.0
                mast3r_hint_transl = 0.0
                mast3r_hint_depth = 0.0
                if mast3r_result is not None and mast3r_result_accepted:
                    mast3r_hint_allow_stop = bool(
                        terminal_mode == "geometry" and mast3r_result.should_stop
                    )
                    mast3r_hint_yaw = float(mast3r_result.yaw_error_deg)
                    mast3r_hint_forward = float(mast3r_result.forward_error)
                    mast3r_hint_lateral = float(mast3r_result.lateral_error)
                    mast3r_hint_transl = float(mast3r_result.transl_error)
                    mast3r_hint_depth = float(mast3r_result.depth_error)

                rgbd_terminal_distance = float("inf")
                rgbd_terminal_candidate_idx = None
                if mast3r_result_accepted and terminal_mode == "rgbd_after_geometry":
                    depth_cfg = cfg.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor
                    if refine_trigger_source == "lightglue":
                        rgbd_terminal_distance = approach_reference_distance
                        rgbd_terminal_candidate_idx = approach_candidate_idx
                    elif refine_trigger_source == "dino_direct":
                        rgbd_terminal_candidate_idx = current_best_crop_idx
                        if current_best_crop_mask is not None:
                            rgbd_terminal_distance = estimate_mask_depth_distance(
                                observations["depth"],
                                current_best_crop_mask,
                                float(depth_cfg.min_depth),
                                float(depth_cfg.max_depth),
                            )
                    else:
                        rgbd_terminal_distance = estimate_center_depth_distance(
                            observations["depth"],
                            float(depth_cfg.min_depth),
                            float(depth_cfg.max_depth),
                        )

                if mast3r_hint_allow_stop or (
                    not mast3r_hint_active and not refine_goal_sent
                ):
                    publish_mast3r_hint(
                        mast3r_hint_pub,
                        active=mast3r_hint_active,
                        allow_stop=mast3r_hint_allow_stop,
                        yaw_error_deg=mast3r_hint_yaw,
                        forward_error=mast3r_hint_forward,
                        lateral_error=mast3r_hint_lateral,
                        transl_error=mast3r_hint_transl,
                        depth_error=mast3r_hint_depth,
                    )

                if refine_latched:
                    if (
                        mast3r_result is not None
                        and mast3r_result_accepted
                        and terminal_mode == "rgbd_after_geometry"
                    ):
                        accepted_route = refine_trigger_source
                        publish_mast3r_hint(
                            mast3r_hint_pub,
                            active=False,
                            allow_stop=False,
                            yaw_error_deg=0.0,
                            forward_error=0.0,
                            lateral_error=0.0,
                            transl_error=0.0,
                            depth_error=0.0,
                        )
                        if (
                            np.isfinite(rgbd_terminal_distance)
                            and rgbd_terminal_distance <= stop_distance
                        ):
                            insinav_stop_gate_passed = True
                            global_action = ACTION.STOP
                            episode_stop_route = accepted_route
                            episode_route_stats[accepted_route]["stops"] += 1
                            print(
                                "[INSiNav_RGBD_TERMINAL] MASt3R quality gate passed; "
                                f"route={accepted_route}, "
                                f"sensor_depth={rgbd_terminal_distance:.3f} <= "
                                f"stop_distance={stop_distance:.3f}"
                            )
                        else:
                            episode_route_stats[accepted_route]["approaches"] += 1
                            if rgbd_terminal_candidate_idx is not None:
                                confirmed_approach_active = True
                                confirmed_approach_steps = 0
                                verified_approach_candidate_idx = int(
                                    rgbd_terminal_candidate_idx
                                )
                            elif (
                                visual_approach_enabled
                                and np.isfinite(rgbd_terminal_distance)
                                and rgbd_terminal_distance >= visual_forward_clearance
                            ):
                                visual_approach_pending = True
                                visual_approach_last_depth = rgbd_terminal_distance
                            print(
                                "[INSiNav_RGBD_TERMINAL] MASt3R quality gate passed; "
                                f"route={accepted_route}, fixed_SE2=0, yaw_alignment=0, "
                                f"sensor_depth={rgbd_terminal_distance:.3f}"
                            )
                        refine_latched = False
                        refine_trigger_source = None
                        refine_goal_sent = False
                        refine_goal_sent_step = -1
                        refine_finish_wait_cycles = 0
                        mast3r_execution_retry_after_step = count_steps + 1
                        mast3r_direct_score_streak = 0
                        mast3r_direct_candidate_mask = None
                        mast3r_direct_candidate_label = None
                        mast3r_direct_latched_mask = None
                        mast3r_latched_current_mask = None
                    elif (
                        mast3r_result is not None
                        and mast3r_result_accepted
                        and mast3r_result.should_stop
                    ):
                        insinav_stop_gate_passed = True
                        episode_stop_route = refine_trigger_source
                        episode_route_stats[refine_trigger_source]["stops"] += 1
                        print(
                            "[INSiNav_DEBUG] stop triggered by latched MASt3R refinement: "
                            f"route={refine_trigger_source}, "
                            f"match_points={stop_match_points}, "
                            f"depth_distance={stop_depth_distance:.3f}, "
                            f"stop_distance={stop_distance:.3f}"
                        )
                        global_action = ACTION.STOP
                        refine_trigger_source = None
                        refine_goal_sent = False
                        refine_goal_sent_step = -1
                        refine_finish_wait_cycles = 0
                        insinav_stop_reject_count = 0
                        confirmed_approach_active = False
                        confirmed_approach_steps = 0
                    elif mast3r_result is not None and mast3r_result_accepted:
                        episode_route_stats[refine_trigger_source]["approaches"] += 1
                        publish_mast3r_hint(
                            mast3r_hint_pub,
                            active=True,
                            allow_stop=False,
                            yaw_error_deg=float(mast3r_result.yaw_error_deg),
                            forward_error=float(mast3r_result.forward_error),
                            lateral_error=float(mast3r_result.lateral_error),
                            transl_error=float(mast3r_result.transl_error),
                            depth_error=float(mast3r_result.depth_error),
                        )
                        refine_goal_sent = True
                        refine_goal_sent_step = count_steps
                        refine_goal_hint = (
                            float(mast3r_result.yaw_error_deg),
                            float(mast3r_result.forward_error),
                            float(mast3r_result.lateral_error),
                            float(mast3r_result.transl_error),
                            float(mast3r_result.depth_error),
                        )
                        refine_finish_wait_cycles = 0
                        mast3r_adjustment_used = True
                        mast3r_one_shot_goal_committed = True
                        insinav_stop_gate_passed = False
                        confirmed_approach_active = True
                        confirmed_approach_steps = 0
                        yaw_only_turn_steps = 0
                        if refine_trigger_source == "dino_direct":
                            print(
                                "[INSiNav_MASt3R_DIRECT] sent one quality-validated locked "
                                "SE(2) goal to ROS: "
                                f"forward={mast3r_result.forward_error:.3f}, "
                                f"lateral={mast3r_result.lateral_error:.3f}, "
                                f"yaw={mast3r_result.yaw_error_deg:.2f}, "
                                f"translation={mast3r_result.transl_error:.3f}"
                            )
                        elif refine_trigger_source == "lightglue_no_box":
                            print(
                                "[INSiNav_LG_NOBOX] sent one quality-validated locked "
                                "SE(2) MASt3R goal to ROS: "
                                f"forward={mast3r_result.forward_error:.3f}, "
                                f"lateral={mast3r_result.lateral_error:.3f}, "
                                f"yaw={mast3r_result.yaw_error_deg:.2f}, "
                                f"translation={mast3r_result.transl_error:.3f}"
                            )
                        else:
                            print(
                                "[INSiNav_REFINE] LightGlue route sent one quality-validated "
                                "locked SE(2) MASt3R goal to ROS without a depth gate: "
                                f"forward={mast3r_result.forward_error:.3f}, "
                                f"lateral={mast3r_result.lateral_error:.3f}, "
                                f"yaw={mast3r_result.yaw_error_deg:.2f}"
                            )
                    elif (
                        mast3r_result is not None
                        and mast3r_refiner is not None
                        and mast3r_refiner.available
                    ):
                        episode_failure_events["verification_failure"] += 1
                        failed_refine_source = refine_trigger_source or "unknown"
                        if failed_refine_source == "dino_direct":
                            mast3r_direct_retry_after_step = (
                                count_steps + mast3r_direct_retry_cooldown_steps
                            )
                            mast3r_direct_score_streak = 0
                            mast3r_direct_candidate_mask = None
                            mast3r_direct_candidate_label = None
                            mast3r_direct_latched_mask = None
                            mast3r_latched_current_mask = None
                        elif failed_refine_source == "lightglue_no_box":
                            lightglue_no_box_retry_after_step = (
                                count_steps
                                + lightglue_no_box_retry_cooldown_steps
                            )
                            mast3r_latched_current_mask = None
                            visual_approach_pending = False
                            visual_approach_steps = 0
                            visual_approach_last_depth = float("inf")
                        insinav_stop_gate_passed = False
                        refine_latched = False
                        refine_trigger_source = None
                        refine_goal_sent = False
                        refine_goal_sent_step = -1
                        publish_mast3r_hint(
                            mast3r_hint_pub,
                            active=False,
                            allow_stop=False,
                            yaw_error_deg=0.0,
                            forward_error=0.0,
                            lateral_error=0.0,
                            transl_error=0.0,
                            depth_error=0.0,
                        )
                        if failed_refine_source == "lightglue":
                            confirmed_approach_active = (
                                verified_approach_cloud is not None
                                or verified_approach_candidate_idx is not None
                            )
                            confirmed_approach_steps = 0
                            print(
                                "[INSiNav_MASt3R] LightGlue target is valid but MASt3R "
                                "geometry was rejected; keep the cached world-frame target "
                                "for approach and retry on a later confirmed frame: "
                                f"reason={mast3r_validation_reason}"
                            )
                        elif failed_refine_source == "lightglue_no_box":
                            confirmed_approach_active = False
                            confirmed_approach_steps = 0
                            verified_approach_candidate_idx = None
                            verified_approach_cloud = None
                            print(
                                "[INSiNav_LG_NOBOX] MASt3R rejected the detector-independent "
                                "target; resume normal planning after applying cooldown: "
                                f"reason={mast3r_validation_reason}, "
                                f"retry_after_step={lightglue_no_box_retry_after_step}"
                            )
                        else:
                            confirmed_approach_active = False
                            confirmed_approach_steps = 0
                            print(
                                "[INSiNav_MASt3R] direct route did not produce a reliable "
                                "local goal; release it and resume object-map planning: "
                                f"reason={mast3r_validation_reason}"
                            )
                    elif refine_goal_sent:
                        print(
                            "[INSiNav_REFINE] MASt3R local goal already sent; "
                            "keep ROS A* target locked and skip visual re-estimation."
                        )
                # Do not demote or boost label_list/score_list with LightGlue.
                # ROS may navigate to label-0 targets, but STOP remains gated above.

            publish_float64(itm_score_pub, value_map_score)

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
            verified_approach_owner_active = (
                is_instance_imagenav
                and verified_approach_enabled
                and confirmed_approach_active
                and not refine_latched
            )
            current_verified_cloud_available = (
                verified_approach_owner_active
                and verified_approach_candidate_idx is not None
                and 0 <= verified_approach_candidate_idx < len(obj_point_cloud_list)
                and obj_point_cloud_list[verified_approach_candidate_idx].width > 0
            )
            if current_verified_cloud_available:
                verified_approach_cloud = obj_point_cloud_list[
                    verified_approach_candidate_idx
                ]
            verified_approach_route_active = (
                verified_approach_owner_active and verified_approach_cloud is not None
            )
            published_point_clouds = list(obj_point_cloud_list)
            published_scores = list(score_list)
            published_labels = list(label_list)
            if verified_approach_route_active:
                promoted_label = (
                    label_list[verified_approach_candidate_idx]
                    if current_verified_cloud_available
                    and verified_approach_candidate_idx < len(label_list)
                    else -1
                )
                published_point_clouds.insert(
                    0,
                    verified_approach_cloud,
                )
                published_scores.insert(0, verified_target_score)
                published_labels.insert(0, 0)
                print(
                    "[INSiNav_APPROACH] publish LightGlue-verified approach target "
                    "as label=0 object cloud: "
                    f"idx={verified_approach_candidate_idx}, "
                    f"cached={int(not current_verified_cloud_available)}, "
                    f"orig_label={promoted_label}, "
                    f"score={verified_target_score:.3f}"
                )
            cld_with_score_msg.point_clouds = published_point_clouds
            cld_with_score_msg.confidence_scores = published_scores
            cld_with_score_msg.label_indices = published_labels
            cld_with_score_pub.publish(cld_with_score_msg)
            if not verified_approach_owner_active:
                verified_approach_cloud = None
            # The cached cloud is already in world coordinates, so transient detector-index or
            # label changes cannot redirect the verified route to another current-frame box.
            publish_verified_approach_target(
                verified_approach_target_pub,
                verified_approach_cloud if verified_approach_route_active else None,
            )

            if is_instance_imagenav and publish_semantic_object_clouds:
                # An empty message resets consecutive crop evidence when no candidate is visible.
                semantic_cloud_msg = MultipleMasksWithConfidence()
                semantic_masks = []
                semantic_scores = []
                semantic_labels = []
                for idx, crop_score in ranked_crop_scores:
                    if idx >= len(object_masks_list) or idx >= len(label_list):
                        continue
                    semantic_masks.append(object_masks_list[idx])
                    semantic_scores.append(
                        float(np.clip(crop_score_weight * crop_score, 0.0, 1.0))
                    )
                    semantic_labels.append(label_list[idx])

                if semantic_masks:
                    semantic_clouds = get_object_point_cloud(cfg, observations, semantic_masks)
                    (
                        semantic_cloud_msg.point_clouds,
                        semantic_cloud_msg.confidence_scores,
                        semantic_cloud_msg.label_indices,
                    ) = filter_positive_detection_results(
                        semantic_clouds, semantic_scores, semantic_labels
                    )
                semantic_cld_with_score_pub.publish(semantic_cloud_msg)

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
        publish_int32(instance_stop_gate_pub, 0)
        publish_verified_approach_target(verified_approach_target_pub)
        publish_int32(state_pub, HABITAT_STATE.EPISODE_FINISH)

        # Collect evaluation metrics
        info = env.get_metrics()
        spl = info["spl"]
        soft_spl = info["soft_spl"]
        distance_to_goal = info["distance_to_goal"]
        distance_to_goal_reward = info["distance_to_goal_reward"]
        success = info["success"]
        goal_view_error = None
        goal_yaw_error = None
        goal_view_success = None

        if is_instance_imagenav and episode_goal_pose is not None:
            try:
                final_agent_state = env.sim.get_agent_state()
                goal_view_error, goal_yaw_error, goal_view_success = compute_goal_view_metrics(
                    final_position=np.asarray(final_agent_state.position, dtype=np.float32),
                    final_rotation=final_agent_state.rotation,
                    goal_position=episode_goal_pose["position"],
                    goal_rotation=episode_goal_pose["rotation"],
                )
                goal_view_error_all += goal_view_error
                goal_yaw_error_all += goal_yaw_error
                goal_view_eval_count += 1
                goal_view_success_count += int(goal_view_success)
                print(
                    "[INSiNav_GOAL_VIEW] "
                    f"position_error={goal_view_error:.3f}m, "
                    f"yaw_error={goal_yaw_error:.2f}deg, "
                    f"view_success={int(goal_view_success)}"
                )
            except Exception as exc:
                print(f"[INSiNav_GOAL_VIEW] failed to compute goal-view metrics: {exc}")

        # Check if agent got close to the target object
        if distance_to_goal <= success_distance:
            near_object = 1

        # Determine episode result
        if success == 1:
            num_success += 1
            result_text = "success"
            episode_termination_reason = "success"
        else:
            if episode_termination_reason is None:
                episode_termination_reason = planner_failure_termination_reason(
                    final_state, expl_result
                )
                if episode_termination_reason is not None:
                    pass
                elif goal_stop_executed:
                    episode_termination_reason = "false_positive"
                elif count_steps >= max_episode_steps:
                    episode_termination_reason = "step_limit"
                else:
                    episode_termination_reason = "unknown"

            if episode_termination_reason == "false_positive":
                result_text = "false positive"
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

        termination_counts[episode_termination_reason] += 1
        for event, count in episode_failure_events.items():
            recoverable_failure_counts[event] += count

        # Update cumulative statistics
        num_total += 1
        spl_all += spl
        soft_spl_all += soft_spl
        distance_to_goal_all += distance_to_goal
        distance_to_goal_reward_all += distance_to_goal_reward

        # Generate video file
        scene_id = env.current_episode.scene_id
        episode_id = env.current_episode.episode_id
        if is_instance_imagenav:
            route_record = {
                "ablation": ablation_name,
                "scene_id": scene_id,
                "episode_id": str(episode_id),
                "success": int(success),
                "spl": float(spl),
                "soft_spl": float(soft_spl),
                "steps": int(count_steps),
                "termination": episode_termination_reason,
                "last_accepted_route": episode_last_accepted_route,
                "stop_route": episode_stop_route,
                "routes": episode_route_stats,
            }
            with open(route_metrics_path, "a", encoding="utf-8") as route_file:
                route_file.write(json.dumps(route_record, ensure_ascii=True) + "\n")
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
        table1.add_row(["Episode Termination Reason", episode_termination_reason])
        for event in RECOVERABLE_FAILURE_EVENTS:
            table1.add_row([f"Episode Event {event}", episode_failure_events[event]])
        if goal_view_error is not None and goal_yaw_error is not None and goal_view_success is not None:
            table1.add_row(["Episode Goal-View Pos Error", f"{goal_view_error:.4f}"])
            table1.add_row(["Episode Goal-View Yaw Error", f"{goal_yaw_error:.2f}"])
            table1.add_row(["Episode View Success@0.25m,10deg", f"{int(goal_view_success)}"])
        if goal_view_eval_count > 0:
            table1.add_row(
                ["Average Goal-View Pos Error", f"{goal_view_error_all/goal_view_eval_count:.4f}"]
            )
            table1.add_row(
                ["Average Goal-View Yaw Error", f"{goal_yaw_error_all/goal_view_eval_count:.2f}"]
            )
            table1.add_row(
                [
                    "Average View Success@0.25m,10deg",
                    f"{goal_view_success_count/goal_view_eval_count * 100:.2f}%",
                ]
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
        for reason in TERMINATION_REASONS:
            table2.add_row([f"Total Termination {reason}", termination_counts[reason]])
        for event in RECOVERABLE_FAILURE_EVENTS:
            table2.add_row([f"Total Event {event}", recoverable_failure_counts[event]])
        if goal_view_eval_count > 0:
            table2.add_row(["Total Goal-View Pos Error Sum", f"{goal_view_error_all:.6f}"])
            table2.add_row(["Total Goal-View Yaw Error Sum", f"{goal_yaw_error_all:.6f}"])
            table2.add_row(["Total Goal-View Eval Count", f"{goal_view_eval_count}"])
            table2.add_row(["Total View Success@0.25m,10deg", f"{goal_view_success_count}"])

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
        if goal_view_eval_count > 0:
            record_data.extend(
                [
                    goal_view_error_all / goal_view_eval_count,
                    goal_yaw_error_all / goal_view_eval_count,
                    goal_view_success_count / goal_view_eval_count * 100,
                ]
            )
        record_data.extend(result_list)
        publish_float32_array(record_pub, record_data)

        pbar.update()
        if flag_once:
            break
        env.current_episode = next(env.episode_iterator)
        rospy.sleep(0.1)  # wait a moment

    env.close()
    pbar.close()


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    rospy.init_node("habitat_eval_node", anonymous=True)
    run_log_file = None

    try:
        dataset, overrides = _parse_dataset_arg()
        cfg_name = f"habitat_eval_{dataset}"
        # Compose the chosen config and pass through extra Hydra overrides
        with initialize(version_base=None, config_path="config"):
            cfg = compose(config_name=cfg_name, overrides=overrides)
        video_output_path = cfg.video_output_path.format(split=cfg.habitat.dataset.split)
        run_log_file, run_log_path = setup_run_logging(video_output_path)
        print(f"[RUN_LOG] dataset={dataset}, config={cfg_name}, overrides={overrides}")
        main(cfg)
    except Exception as e:
        print(f"Unexpected error occurred: {e}")
        rospy.signal_shutdown("Shutdown due to error")
        close_run_logging(run_log_file)
        os._exit(1)
    finally:
        close_run_logging(run_log_file)
