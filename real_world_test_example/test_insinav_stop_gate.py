import numpy as np
from pathlib import Path

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
    stop_gate_status,
    update_lightglue_approach_lock,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_distance_defaults_to_median_not_low_percentile():
    depth = np.array(
        [
            [0.8, 0.9, 1.1, 1.2, 1.3],
        ],
        dtype=np.float32,
    )
    mask = np.ones_like(depth, dtype=np.uint8)

    assert estimate_mask_distance(depth, mask, min_depth=0.3, max_depth=3.0) > 1.0
    assert estimate_mask_distance(depth, mask, min_depth=0.3, max_depth=3.0, percentile=30.0) < 1.0


def test_stop_gate_requires_consecutive_confirmed_frames():
    status, confirm_count = stop_gate_status(
        selected_match_points=121.0,
        best_distance=0.93,
        score_threshold=35.0,
        stop_distance=1.0,
        confirm_count=0,
        confirm_frames=3,
    )
    assert status == "PENDING_CLOSE"
    assert confirm_count == 1

    status, confirm_count = stop_gate_status(
        selected_match_points=122.0,
        best_distance=0.95,
        score_threshold=35.0,
        stop_distance=1.0,
        confirm_count=confirm_count,
        confirm_frames=3,
    )
    assert status == "PENDING_CLOSE"
    assert confirm_count == 2

    status, confirm_count = stop_gate_status(
        selected_match_points=123.0,
        best_distance=0.94,
        score_threshold=35.0,
        stop_distance=1.0,
        confirm_count=confirm_count,
        confirm_frames=3,
    )
    assert status == "VERIFIED"
    assert confirm_count == 3


def test_stop_gate_uses_enter_exit_distance_hysteresis():
    status, confirm_count = stop_gate_status(
        selected_match_points=121.0,
        best_distance=0.88,
        score_threshold=35.0,
        stop_distance=1.0,
        confirm_count=0,
        confirm_frames=2,
        stop_enter_distance=0.90,
        stop_exit_distance=1.10,
    )
    assert status == "PENDING_CLOSE"
    assert confirm_count == 1

    status, confirm_count = stop_gate_status(
        selected_match_points=122.0,
        best_distance=1.02,
        score_threshold=35.0,
        stop_distance=1.0,
        confirm_count=confirm_count,
        confirm_frames=2,
        stop_enter_distance=0.90,
        stop_exit_distance=1.10,
    )
    assert status == "VERIFIED"
    assert confirm_count == 2


def test_stop_gate_does_not_enter_close_band_from_far_side():
    status, confirm_count = stop_gate_status(
        selected_match_points=121.0,
        best_distance=1.02,
        score_threshold=35.0,
        stop_distance=1.0,
        confirm_count=0,
        confirm_frames=2,
        stop_enter_distance=0.90,
        stop_exit_distance=1.10,
    )

    assert status == "PENDING_FAR"
    assert confirm_count == 0


def test_stop_gate_resets_confirmation_when_target_is_far():
    status, confirm_count = stop_gate_status(
        selected_match_points=121.0,
        best_distance=1.11,
        score_threshold=35.0,
        stop_distance=1.0,
        confirm_count=2,
        confirm_frames=3,
    )

    assert status == "PENDING_FAR"
    assert confirm_count == 0


def test_stop_gate_rejects_low_geometric_quality_even_when_close():
    status, confirm_count = stop_gate_status(
        selected_match_points=123.0,
        best_distance=0.81,
        score_threshold=35.0,
        stop_distance=1.0,
        confirm_count=2,
        confirm_frames=3,
        selected_inlier_points=20.0,
        min_inlier_points=60.0,
        selected_inlier_ratio=0.16,
        min_inlier_ratio=0.30,
    )

    assert status == "PENDING"
    assert confirm_count == 0


def test_stop_gate_accepts_high_geometric_quality():
    status, confirm_count = stop_gate_status(
        selected_match_points=140.0,
        best_distance=0.81,
        score_threshold=35.0,
        stop_distance=1.0,
        confirm_count=2,
        confirm_frames=3,
        selected_inlier_points=78.0,
        min_inlier_points=60.0,
        selected_inlier_ratio=0.56,
        min_inlier_ratio=0.30,
    )

    assert status == "VERIFIED"
    assert confirm_count == 3


def test_stop_gate_accepts_weak_texture_close_real_chair_frame():
    status, confirm_count = stop_gate_status(
        selected_match_points=74.0,
        best_distance=0.824,
        score_threshold=70.0,
        stop_distance=1.0,
        confirm_count=0,
        confirm_frames=2,
        stop_enter_distance=0.95,
        stop_exit_distance=1.10,
        selected_inlier_points=28.0,
        min_inlier_points=25.0,
        selected_inlier_ratio=0.38,
        min_inlier_ratio=0.30,
    )

    assert status == "PENDING_CLOSE"
    assert confirm_count == 1


def test_stop_gate_enters_close_candidate_at_nominal_stop_distance():
    status, confirm_count = stop_gate_status(
        selected_match_points=75.0,
        best_distance=0.967,
        score_threshold=70.0,
        stop_distance=1.0,
        confirm_count=0,
        confirm_frames=2,
        stop_enter_distance=1.00,
        stop_exit_distance=1.10,
        selected_inlier_points=29.0,
        min_inlier_points=25.0,
        selected_inlier_ratio=0.39,
        min_inlier_ratio=0.30,
    )

    assert status == "PENDING_CLOSE"
    assert confirm_count == 1


def test_approach_lock_latches_after_strong_identity_frame():
    locked, lost_frames = update_lightglue_approach_lock(
        locked=False,
        lost_frames=0,
        acquire_ok=True,
        tracking_ok=True,
        best_distance=1.23,
        max_lost_frames=2,
        max_distance=1.6,
    )

    assert locked is True
    assert lost_frames == 0


def test_approach_lock_tolerates_two_weak_close_frames_then_expires():
    locked, lost_frames = update_lightglue_approach_lock(
        locked=True,
        lost_frames=0,
        acquire_ok=False,
        tracking_ok=False,
        best_distance=0.94,
        max_lost_frames=2,
        max_distance=1.6,
    )
    assert locked is True
    assert lost_frames == 1

    locked, lost_frames = update_lightglue_approach_lock(
        locked=locked,
        lost_frames=lost_frames,
        acquire_ok=False,
        tracking_ok=False,
        best_distance=0.96,
        max_lost_frames=2,
        max_distance=1.6,
    )
    assert locked is True
    assert lost_frames == 2

    locked, lost_frames = update_lightglue_approach_lock(
        locked=locked,
        lost_frames=lost_frames,
        acquire_ok=False,
        tracking_ok=False,
        best_distance=0.98,
        max_lost_frames=2,
        max_distance=1.6,
    )
    assert locked is False
    assert lost_frames == 0


def test_approach_lock_stays_active_when_tracking_remains_good_even_if_far():
    locked, lost_frames = update_lightglue_approach_lock(
        locked=True,
        lost_frames=0,
        acquire_ok=False,
        tracking_ok=True,
        best_distance=2.05,
        max_lost_frames=2,
        max_distance=1.6,
    )

    assert locked is True
    assert lost_frames == 0


def test_stop_gate_keeps_locked_close_target_in_pending_close():
    status, confirm_count = stop_gate_status(
        selected_match_points=21.0,
        best_distance=0.922,
        score_threshold=70.0,
        stop_distance=1.0,
        confirm_count=0,
        confirm_frames=2,
        stop_enter_distance=1.00,
        stop_exit_distance=1.10,
        selected_inlier_points=7.0,
        min_inlier_points=25.0,
        selected_inlier_ratio=0.33,
        min_inlier_ratio=0.30,
        approach_lock_active=True,
        tracking_min_match_points=12.0,
        tracking_min_inlier_points=4.0,
        tracking_min_inlier_ratio=0.30,
    )

    assert status == "PENDING_CLOSE"
    assert confirm_count == 1


def test_stop_gate_keeps_locked_far_target_in_pending_far():
    status, confirm_count = stop_gate_status(
        selected_match_points=12.0,
        best_distance=1.206,
        score_threshold=70.0,
        stop_distance=1.0,
        confirm_count=0,
        confirm_frames=2,
        stop_enter_distance=1.00,
        stop_exit_distance=1.10,
        selected_inlier_points=5.0,
        min_inlier_points=25.0,
        selected_inlier_ratio=0.42,
        min_inlier_ratio=0.30,
        approach_lock_active=True,
        tracking_min_match_points=12.0,
        tracking_min_inlier_points=4.0,
        tracking_min_inlier_ratio=0.30,
    )

    assert status == "PENDING_FAR"
    assert confirm_count == 0


def test_locked_stop_gate_status_uses_last_distance_when_detection_drops():
    assert (
        locked_stop_gate_status(
            best_distance=0.94,
            stop_distance=1.0,
            stop_enter_distance=1.00,
            stop_exit_distance=1.10,
        )
        == "PENDING_CLOSE"
    )
    assert (
        locked_stop_gate_status(
            best_distance=1.21,
            stop_distance=1.0,
            stop_enter_distance=1.00,
            stop_exit_distance=1.10,
        )
        == "PENDING_FAR"
    )


def test_candidate_gate_rejects_same_class_weak_instance_match():
    assert not lightglue_candidate_passes(
        match_points=40.0,
        inlier_points=11.0,
        inlier_ratio=0.28,
        min_match_points=50.0,
        min_inlier_points=15.0,
        min_inlier_ratio=0.30,
    )


def test_candidate_gate_accepts_strong_instance_match():
    assert lightglue_candidate_passes(
        match_points=119.0,
        inlier_points=50.0,
        inlier_ratio=0.42,
        min_match_points=50.0,
        min_inlier_points=15.0,
        min_inlier_ratio=0.30,
    )


def test_rejected_detection_clouds_are_filtered_before_publishing():
    clouds = ["target_cloud", "rejected_cloud", "weak_cloud", "other_cloud"]
    scores = [0.8, 0.0, -0.1, 0.25]
    labels = [0, 0, 1, 1]

    kept_clouds, kept_scores, kept_labels = filter_positive_detection_results(
        clouds,
        scores,
        labels,
    )

    assert kept_clouds == ["target_cloud", "other_cloud"]
    assert kept_scores == [0.8, 0.25]
    assert kept_labels == [0, 1]


def test_goal_object_crop_selects_target_label_region():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    boxes = np.array(
        [
            [0.0, 0.0, 0.2, 0.2],
            [0.25, 0.30, 0.75, 0.90],
        ],
        dtype=np.float32,
    )
    phrases = ["book", "chair"]
    scores = np.array([0.9, 0.8], dtype=np.float32)

    crop, meta = select_goal_object_crop(
        image,
        boxes,
        phrases,
        scores,
        target_label="chair",
        padding_ratio=0.0,
    )

    assert crop is not None
    assert crop.shape[:2] == (61, 101)
    assert meta["label"] == "chair"


def test_goal_object_crop_rejects_near_full_image_detection():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    boxes = np.array([[0.0, 0.0, 1.0, 1.0]], dtype=np.float32)

    crop, meta = select_goal_object_crop(
        image,
        boxes,
        ["chair"],
        np.array([0.97], dtype=np.float32),
        target_label="chair",
        padding_ratio=0.0,
        max_area_ratio=0.90,
    )

    assert crop is None
    assert meta == {}


def test_goal_matching_roi_uses_normalized_bounds():
    image = np.zeros((100, 200, 3), dtype=np.uint8)

    crop, meta = crop_image_by_roi(image, [0.25, 0.2, 0.75, 0.8])

    assert crop is not None
    assert crop.shape[:2] == (61, 101)
    assert meta["box"] == [50, 20, 150, 80]
    assert meta["area_ratio"] < 0.40


def test_goal_object_crop_returns_none_without_target_label():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    boxes = np.array([[0.0, 0.0, 0.2, 0.2]], dtype=np.float32)

    crop, meta = select_goal_object_crop(
        image,
        boxes,
        ["book"],
        np.array([0.9], dtype=np.float32),
        target_label="chair",
        padding_ratio=0.0,
    )

    assert crop is None
    assert meta == {}


def test_similar_answers_are_disabled_for_primary_only_mode():
    assert resolve_similar_answers(True, False, ["couch", "toilet"]) == []
    assert resolve_similar_answers(True, True, ["couch"]) == ["couch"]
    assert resolve_similar_answers(False, True, ["couch"]) == []


def test_frame_timing_rejects_stale_detection_inputs():
    ok, reason = frame_timing_status(
        frame_age=0.9,
        sensor_pose_dt=0.02,
        rgb_depth_dt=0.02,
        max_frame_age=0.5,
        max_sensor_pose_dt=0.1,
        max_rgb_depth_dt=0.08,
    )

    assert not ok
    assert reason == "STALE_FRAME"


def test_frame_timing_accepts_fresh_synced_inputs():
    ok, reason = frame_timing_status(
        frame_age=0.2,
        sensor_pose_dt=0.02,
        rgb_depth_dt=0.02,
        max_frame_age=0.5,
        max_sensor_pose_dt=0.1,
        max_rgb_depth_dt=0.08,
    )

    assert ok
    assert reason == "OK"


def test_frame_timing_accepts_synchronized_inputs_with_clock_offset():
    effective_age, clock_offset, used_offset = compensate_frame_age_for_clock_offset(
        frame_age=644.6,
        clock_offset=None,
        max_frame_age=0.8,
        clock_offset_tolerance=900.0,
    )

    ok, reason = frame_timing_status(
        frame_age=effective_age,
        sensor_pose_dt=0.03,
        rgb_depth_dt=0.03,
        max_frame_age=0.8,
        max_sensor_pose_dt=0.1,
        max_rgb_depth_dt=0.08,
    )

    assert used_offset
    assert clock_offset == 644.6
    assert ok
    assert reason == "OK"


def test_frame_timing_accepts_large_synchronized_clock_offset():
    effective_age, clock_offset, used_offset = compensate_frame_age_for_clock_offset(
        frame_age=15161.6,
        clock_offset=None,
        max_frame_age=0.8,
        clock_offset_tolerance=0.0,
    )

    ok, reason = frame_timing_status(
        frame_age=effective_age,
        sensor_pose_dt=0.03,
        rgb_depth_dt=0.03,
        max_frame_age=0.8,
        max_sensor_pose_dt=0.1,
        max_rgb_depth_dt=0.08,
    )

    assert used_offset
    assert clock_offset == 15161.6
    assert ok
    assert reason == "OK"

def test_frame_timing_still_rejects_frozen_inputs_with_clock_offset():
    effective_age, clock_offset, used_offset = compensate_frame_age_for_clock_offset(
        frame_age=645.7,
        clock_offset=644.6,
        max_frame_age=0.8,
        clock_offset_tolerance=900.0,
    )

    ok, reason = frame_timing_status(
        frame_age=effective_age,
        sensor_pose_dt=0.03,
        rgb_depth_dt=0.03,
        max_frame_age=0.8,
        max_sensor_pose_dt=0.1,
        max_rgb_depth_dt=0.08,
    )

    assert used_offset
    assert clock_offset == 644.6
    assert not ok
    assert reason == "STALE_FRAME"


def test_real_world_bridge_uses_shared_stop_gate_logic():
    bridge_source = (REPO_ROOT / "real_world_test_example" / "real_world_test_insinav.py").read_text()

    assert "from insinav_stop_gate import" in bridge_source
    assert "def estimate_mask_distance(" not in bridge_source
    assert "def summarize_mask_depth(" not in bridge_source


def test_real_world_config_uses_weak_texture_stop_gate_for_real_chair_goal():
    config_source = (
        REPO_ROOT / "real_world_test_example" / "config" / "real_world_test_insinav.yaml"
    ).read_text()

    assert "max_frame_age: 0.8" in config_source
    assert "stop_confirm_frames: 2" in config_source
    assert "stop_enter_distance: 1.00" in config_source
    assert "stop_exit_distance: 1.10" in config_source
    assert "distance_percentile: 50.0" in config_source
    assert "score_threshold: 70.0" in config_source
    assert "min_inlier_points: 25" in config_source
    assert "min_inlier_ratio: 0.30" in config_source
    assert "candidate_min_match_points: 50" in config_source
    assert "candidate_min_inlier_points: 15" in config_source
    assert "candidate_min_inlier_ratio: 0.30" in config_source
    assert "use_object_crop_for_matching: true" in config_source
    assert "max_goal_auto_crop_area_ratio: 0.90" in config_source
    assert "goal_crop_roi: []" in config_source


def test_real_world_config_uses_primary_label_only_for_image_goal():
    config_source = (
        REPO_ROOT / "real_world_test_example" / "config" / "real_world_test_insinav.yaml"
    ).read_text()

    assert "use_similar_set: false" in config_source


def test_real_world_bridge_honors_use_similar_set_config():
    bridge_source = (REPO_ROOT / "real_world_test_example" / "real_world_test_insinav.py").read_text()

    assert "self.insinav_use_similar_set" in bridge_source
    assert "resolve_similar_answers(" in bridge_source


def test_real_world_bridge_drops_stale_synced_frames():
    bridge_source = (REPO_ROOT / "real_world_test_example" / "real_world_test_insinav.py").read_text()

    assert "frame_timing_status(" in bridge_source
    assert "queue_size=self.subscriber_queue_size" in bridge_source
    assert "queue_size=self.sync_queue_size" in bridge_source


def test_map_ros_ignores_rejected_detection_clouds():
    map_ros_source = (REPO_ROOT / "src" / "planner" / "plan_env" / "src" / "map_ros.cpp").read_text()

    assert "confidence_score <= 1e-6" in map_ros_source
    assert "Skip detections rejected by the perception gate" in map_ros_source


def test_real_world_mapping_uses_latest_pose_fallback_without_touching_sim():
    map_ros_source = (REPO_ROOT / "src" / "planner" / "plan_env" / "src" / "map_ros.cpp").read_text()
    map_ros_header = (REPO_ROOT / "src" / "planner" / "plan_env" / "include" / "plan_env" / "map_ros.h").read_text()
    launch_source = (
        REPO_ROOT
        / "src"
        / "planner"
        / "exploration_manager"
        / "launch"
        / "algorithm_traj.xml"
    ).read_text()

    assert 'map_ros/use_latest_pose_on_depth' in launch_source
    assert '<param name="map_ros/use_latest_pose_on_depth" value="$(arg is_real_world_)" type="bool"/>' in launch_source
    assert 'map_ros/latest_pose_max_age' in launch_source

    assert "void depthCallback(const sensor_msgs::ImageConstPtr& img);" in map_ros_header
    assert "void poseCallback(const nav_msgs::OdometryConstPtr& pose);" in map_ros_header
    assert "use_latest_pose_on_depth_" in map_ros_header
    assert "latest_pose_max_age_" in map_ros_header

    assert "if (use_latest_pose_on_depth_)" in map_ros_source
    assert 'depth_async_sub_ = node_.subscribe("/map_ros/depth"' in map_ros_source
    assert 'latest_pose_sub_ = node_.subscribe("/map_ros/pose"' in map_ros_source
    assert "[MAP_ASYNC]" in map_ros_source


def test_object_visualization_skips_unclassified_clusters():
    fsm_source = (
        REPO_ROOT
        / "src"
        / "planner"
        / "exploration_manager"
        / "src"
        / "exploration_fsm_traj.cpp"
    ).read_text()

    assert "if (label < 0)" in fsm_source
    assert "Skip unclassified object clusters" in fsm_source
