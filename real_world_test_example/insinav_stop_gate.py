from __future__ import annotations

from typing import Iterable, List, Mapping, Optional, Tuple

import numpy as np


def resolve_similar_answers(
    filter_enabled: bool,
    use_similar_set: bool,
    similar_answers: Optional[Iterable[str]],
) -> List[str]:
    if not filter_enabled or not use_similar_set:
        return []
    return list(similar_answers or [])


def frame_timing_status(
    frame_age: float,
    sensor_pose_dt: float,
    rgb_depth_dt: float,
    max_frame_age: float,
    max_sensor_pose_dt: float,
    max_rgb_depth_dt: float,
) -> Tuple[bool, str]:
    if frame_age > max_frame_age:
        return False, "STALE_FRAME"
    if sensor_pose_dt > max_sensor_pose_dt:
        return False, "SENSOR_POSE_DESYNC"
    if rgb_depth_dt > max_rgb_depth_dt:
        return False, "RGB_DEPTH_DESYNC"
    return True, "OK"


def compensate_frame_age_for_clock_offset(
    frame_age: float,
    clock_offset: Optional[float],
    max_frame_age: float,
    clock_offset_tolerance: float,
) -> Tuple[float, Optional[float], bool]:
    if frame_age <= max_frame_age:
        return frame_age, clock_offset, False

    if clock_offset is None:
        if clock_offset_tolerance <= 0.0 or frame_age <= clock_offset_tolerance:
            return 0.0, frame_age, True
        return frame_age, clock_offset, False

    return max(0.0, frame_age - clock_offset), clock_offset, True


def filter_positive_detection_results(
    point_clouds,
    confidence_scores: Iterable[float],
    label_indices: Iterable[int],
    min_score: float = 1e-6,
):
    filtered_clouds = []
    filtered_scores = []
    filtered_labels = []

    for cloud, score, label in zip(point_clouds, confidence_scores, label_indices):
        score = float(score)
        if score <= min_score:
            continue

        filtered_clouds.append(cloud)
        filtered_scores.append(score)
        filtered_labels.append(label)

    return filtered_clouds, filtered_scores, filtered_labels


def estimate_mask_distance(
    depth_image: np.ndarray,
    object_mask: np.ndarray,
    min_depth: float = 0.05,
    max_depth: float = 10.0,
    percentile: float = 50.0,
) -> float:
    """Estimate target distance from valid depth values inside the object mask."""
    if depth_image is None or object_mask is None:
        return float("inf")

    depth = np.asarray(depth_image).astype(np.float32)
    if depth.ndim == 3:
        depth = np.squeeze(depth)

    mask = object_mask > 0
    if mask.shape != depth.shape:
        return float("inf")

    values = depth[mask]
    values = values[np.isfinite(values)]
    values = values[(values > min_depth) & (values < max_depth)]

    if values.size == 0:
        return float("inf")

    return float(np.percentile(values, percentile))


def summarize_mask_depth(
    depth_image: np.ndarray,
    object_mask: np.ndarray,
    min_depth: float,
    max_depth: float,
) -> dict:
    summary = {
        "shape_match": False,
        "mask_pixels": 0,
        "finite_pixels": 0,
        "valid_pixels": 0,
        "valid_ratio": 0.0,
        "p10": float("nan"),
        "p30": float("nan"),
        "p50": float("nan"),
        "p90": float("nan"),
        "min": float("nan"),
        "max": float("nan"),
    }

    if depth_image is None or object_mask is None:
        return summary

    depth = np.asarray(depth_image).astype(np.float32)
    if depth.ndim == 3:
        depth = np.squeeze(depth)

    mask = object_mask > 0
    summary["mask_pixels"] = int(np.count_nonzero(mask))
    if mask.shape != depth.shape:
        return summary

    summary["shape_match"] = True
    if summary["mask_pixels"] == 0:
        return summary

    values = depth[mask]
    values = values[np.isfinite(values)]
    summary["finite_pixels"] = int(values.size)
    values = values[(values > min_depth) & (values < max_depth)]
    summary["valid_pixels"] = int(values.size)
    summary["valid_ratio"] = (
        float(summary["valid_pixels"]) / float(summary["mask_pixels"])
        if summary["mask_pixels"] > 0
        else 0.0
    )

    if values.size == 0:
        return summary

    summary["min"] = float(np.min(values))
    summary["p10"] = float(np.percentile(values, 10.0))
    summary["p30"] = float(np.percentile(values, 30.0))
    summary["p50"] = float(np.percentile(values, 50.0))
    summary["p90"] = float(np.percentile(values, 90.0))
    summary["max"] = float(np.max(values))
    return summary


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def crop_image_by_roi(
    image: np.ndarray,
    roi: Iterable[float],
) -> Tuple[Optional[np.ndarray], dict]:
    if image is None or roi is None:
        return None, {}

    image_arr = np.asarray(image)
    if image_arr.ndim < 2:
        return None, {}

    roi_values = [float(v) for v in list(roi)]
    if len(roi_values) != 4 or not np.all(np.isfinite(roi_values)):
        return None, {}

    height, width = image_arr.shape[:2]
    x1, y1, x2, y2 = roi_values
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.5:
        x1 *= width
        x2 *= width
        y1 *= height
        y2 *= height

    x_min, x_max = min(x1, x2), max(x1, x2)
    y_min, y_max = min(y1, y2), max(y1, y2)

    left = max(0, int(np.floor(x_min)))
    top = max(0, int(np.floor(y_min)))
    right = min(width - 1, int(np.ceil(x_max)))
    bottom = min(height - 1, int(np.ceil(y_max)))

    if right <= left or bottom <= top:
        return None, {}

    area = float((right - left + 1) * (bottom - top + 1))
    area_ratio = area / float(width * height)
    crop = image_arr[top : bottom + 1, left : right + 1].copy()
    return crop, {
        "box": [left, top, right, bottom],
        "area_ratio": area_ratio,
    }


def select_goal_object_crop(
    image: np.ndarray,
    boxes,
    phrases: Iterable[str],
    scores,
    target_label: str,
    padding_ratio: float = 0.1,
    max_area_ratio: Optional[float] = None,
) -> Tuple[Optional[np.ndarray], dict]:
    if image is None or target_label is None:
        return None, {}

    image_arr = np.asarray(image)
    if image_arr.ndim < 2:
        return None, {}

    boxes_arr = _to_numpy(boxes).astype(np.float32)
    scores_arr = _to_numpy(scores).astype(np.float32)
    phrase_list = list(phrases or [])

    if boxes_arr.ndim != 2 or boxes_arr.shape[1] != 4:
        return None, {}

    height, width = image_arr.shape[:2]
    best = None
    max_index = min(len(phrase_list), len(scores_arr), len(boxes_arr))

    for idx in range(max_index):
        label = str(phrase_list[idx])
        if label != target_label:
            continue

        x1, y1, x2, y2 = [float(v) for v in boxes_arr[idx]]
        if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.5:
            x1 *= width
            x2 *= width
            y1 *= height
            y2 *= height

        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)
        box_w = max(1.0, x_max - x_min)
        box_h = max(1.0, y_max - y_min)
        pad_x = box_w * padding_ratio
        pad_y = box_h * padding_ratio

        left = max(0, int(np.floor(x_min - pad_x)))
        top = max(0, int(np.floor(y_min - pad_y)))
        right = min(width - 1, int(np.ceil(x_max + pad_x)))
        bottom = min(height - 1, int(np.ceil(y_max + pad_y)))

        if right <= left or bottom <= top:
            continue

        area = float((right - left + 1) * (bottom - top + 1))
        area_ratio = area / float(width * height)
        if max_area_ratio is not None and area_ratio > max_area_ratio:
            continue

        score = float(scores_arr[idx])
        rank = score + 1e-6 * area
        if best is None or rank > best[0]:
            best = (rank, idx, label, score, left, top, right, bottom, area_ratio)

    if best is None:
        return None, {}

    _, idx, label, score, left, top, right, bottom, area_ratio = best
    crop = image_arr[top : bottom + 1, left : right + 1].copy()
    return crop, {
        "idx": idx,
        "label": label,
        "score": score,
        "box": [left, top, right, bottom],
        "area_ratio": area_ratio,
    }


def stop_gate_status(
    selected_match_points: float,
    best_distance: float,
    score_threshold: float,
    stop_distance: float,
    confirm_count: int,
    confirm_frames: int,
    stop_enter_distance: Optional[float] = None,
    stop_exit_distance: Optional[float] = None,
    selected_inlier_points: Optional[float] = None,
    min_inlier_points: float = 0.0,
    selected_inlier_ratio: Optional[float] = None,
    min_inlier_ratio: float = 0.0,
    approach_lock_active: bool = False,
    tracking_min_match_points: Optional[float] = None,
    tracking_min_inlier_points: Optional[float] = None,
    tracking_min_inlier_ratio: Optional[float] = None,
) -> Tuple[str, int]:
    """Return stop verification status and updated consecutive confirmation count."""
    strict_ok = selected_match_points >= score_threshold
    if selected_inlier_points is not None:
        strict_ok = strict_ok and selected_inlier_points >= min_inlier_points
    if selected_inlier_ratio is not None:
        strict_ok = strict_ok and selected_inlier_ratio >= min_inlier_ratio

    tracking_ok = strict_ok
    if approach_lock_active:
        tracking_ok = selected_match_points >= float(
            tracking_min_match_points
            if tracking_min_match_points is not None
            else score_threshold
        )
        if selected_inlier_points is not None:
            tracking_ok = tracking_ok and selected_inlier_points >= float(
                tracking_min_inlier_points
                if tracking_min_inlier_points is not None
                else min_inlier_points
            )
        if selected_inlier_ratio is not None:
            tracking_ok = tracking_ok and selected_inlier_ratio >= float(
                tracking_min_inlier_ratio
                if tracking_min_inlier_ratio is not None
                else min_inlier_ratio
            )

    enter_distance = (
        float(stop_enter_distance)
        if stop_enter_distance is not None
        else float(stop_distance)
    )
    exit_distance = (
        float(stop_exit_distance)
        if stop_exit_distance is not None
        else float(stop_distance)
    )
    if exit_distance < enter_distance:
        exit_distance = enter_distance

    distance_finite = np.isfinite(best_distance)
    distance_enter_ok = distance_finite and best_distance <= enter_distance
    distance_exit_ok = distance_finite and best_distance <= exit_distance

    if strict_ok and (distance_enter_ok or (confirm_count > 0 and distance_exit_ok)):
        next_count = confirm_count + 1
        if next_count >= max(1, confirm_frames):
            return "VERIFIED", next_count
        return "PENDING_CLOSE", next_count

    if tracking_ok and (distance_enter_ok or (confirm_count > 0 and distance_exit_ok)):
        return "PENDING_CLOSE", max(confirm_count, 0) + 1

    if tracking_ok:
        return "PENDING_FAR", 0

    return "PENDING", 0


def update_lightglue_approach_lock(
    locked: bool,
    lost_frames: int,
    acquire_ok: bool,
    tracking_ok: bool,
    best_distance: float,
    max_lost_frames: int,
    max_distance: float,
) -> Tuple[bool, int]:
    if acquire_ok:
        return True, 0

    if not locked:
        return False, 0

    if tracking_ok:
        return True, 0

    if not np.isfinite(best_distance) or best_distance > max_distance:
        return False, 0

    next_lost_frames = lost_frames + 1
    if next_lost_frames > max(0, int(max_lost_frames)):
        return False, 0

    return True, next_lost_frames


def locked_stop_gate_status(
    best_distance: float,
    stop_distance: float,
    stop_enter_distance: Optional[float] = None,
    stop_exit_distance: Optional[float] = None,
) -> str:
    enter_distance = (
        float(stop_enter_distance)
        if stop_enter_distance is not None
        else float(stop_distance)
    )
    exit_distance = (
        float(stop_exit_distance)
        if stop_exit_distance is not None
        else float(stop_distance)
    )
    if exit_distance < enter_distance:
        exit_distance = enter_distance

    if np.isfinite(best_distance) and best_distance <= enter_distance:
        return "PENDING_CLOSE"
    if np.isfinite(best_distance):
        return "PENDING_FAR"
    return "PENDING"


def lightglue_candidate_passes(
    match_points: float,
    inlier_points: float,
    inlier_ratio: float,
    min_match_points: float,
    min_inlier_points: float,
    min_inlier_ratio: float,
) -> bool:
    return (
        match_points >= min_match_points
        and inlier_points >= min_inlier_points
        and inlier_ratio >= min_inlier_ratio
    )


def stop_gate_status_from_config(
    selected_match_points: float,
    best_distance: float,
    lightglue_cfg: Mapping,
    confirm_count: int,
    selected_inlier_points: Optional[float] = None,
    selected_inlier_ratio: Optional[float] = None,
    approach_lock_active: bool = False,
) -> Tuple[str, int]:
    return stop_gate_status(
        selected_match_points=selected_match_points,
        best_distance=best_distance,
        score_threshold=float(lightglue_cfg.get("score_threshold", 60.0)),
        stop_distance=float(lightglue_cfg.get("stop_distance", 0.7)),
        confirm_count=confirm_count,
        confirm_frames=max(1, int(lightglue_cfg.get("stop_confirm_frames", 3))),
        stop_enter_distance=lightglue_cfg.get("stop_enter_distance", None),
        stop_exit_distance=lightglue_cfg.get("stop_exit_distance", None),
        selected_inlier_points=selected_inlier_points,
        min_inlier_points=float(lightglue_cfg.get("min_inlier_points", 0.0)),
        selected_inlier_ratio=selected_inlier_ratio,
        min_inlier_ratio=float(lightglue_cfg.get("min_inlier_ratio", 0.0)),
        approach_lock_active=approach_lock_active,
        tracking_min_match_points=lightglue_cfg.get("approach_min_match_points", None),
        tracking_min_inlier_points=lightglue_cfg.get("approach_min_inlier_points", None),
        tracking_min_inlier_ratio=lightglue_cfg.get("approach_min_inlier_ratio", None),
    )
