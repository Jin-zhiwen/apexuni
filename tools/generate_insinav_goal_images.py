#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Render InstanceImageNav goal images from HM3D dataset JSON.

This script reads a HM3D InstanceImageNav JSON, selects a goal viewpoint
(position + rotation + hfov + image_dimensions), renders the RGB image using
Habitat-Sim, and saves it as a PNG file.

It supports both:
  1. content/<scene>.json.gz
  2. val.json.gz

Typical usage:

  python tools/generate_insinav_goal_images.py \
    --scene-json data/datasets/instance_imagenav/hm3d/v3/val/content/4ok3usBNeis.json.gz \
    --episode-id 15

or:

  python tools/generate_insinav_goal_images.py \
    --scene-json data/datasets/instance_imagenav/hm3d/v3/val/val.json.gz \
    --episode-id 15

Output images are saved under data/insinav_goal_images by default.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

try:
    import habitat_sim
    from habitat_sim.utils.common import quat_from_coeffs
except Exception as exc:
    raise RuntimeError(
        "habitat_sim is required. Please ensure Habitat-Sim is installed and importable."
    ) from exc


def _load_json(path: str) -> Dict[str, Any]:
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_path(base_dir: str, maybe_relative: Optional[str]) -> Optional[str]:
    if maybe_relative is None:
        return None
    if maybe_relative == "":
        return None
    if os.path.isabs(maybe_relative):
        return os.path.normpath(maybe_relative)
    return os.path.normpath(os.path.join(base_dir, maybe_relative))


def _infer_default_episodes_json(scene_json_path: str) -> Optional[str]:
    """Infer val.json.gz path from content/<scene>.json.gz."""
    scene_json_path = os.path.abspath(scene_json_path)
    parent = os.path.dirname(scene_json_path)

    if os.path.basename(parent) == "content":
        val_path = os.path.join(os.path.dirname(parent), "val.json.gz")
        if os.path.exists(val_path):
            return val_path

    # If scene_json_path itself is val.json.gz, return itself.
    if os.path.basename(scene_json_path) in {"val.json.gz", "train.json.gz", "test.json.gz"}:
        return scene_json_path

    return None


def _has_non_episode_goal_entries(data: Dict[str, Any]) -> bool:
    if "goals" in data:
        return True

    for key, value in data.items():
        if key == "episodes":
            continue
        if isinstance(value, dict) and "image_goals" in value:
            return True

    return False


def _scene_name_candidates(scene_id: str) -> List[str]:
    """Generate possible scene name tokens from a Habitat scene_id."""
    candidates: List[str] = []

    basename = os.path.basename(scene_id)
    stem, ext = os.path.splitext(basename)

    if stem:
        candidates.append(stem)

    # Common HM3D basis filename: TEEsavR23oF.basis.glb
    if stem.endswith(".basis"):
        candidates.append(stem[: -len(".basis")])

    # Parent folder may be 00800-TEEsavR23oF
    parent = os.path.basename(os.path.dirname(scene_id))
    if parent:
        candidates.append(parent)
        if "-" in parent:
            candidates.append(parent.split("-", 1)[1])

    # Some scene_id strings may contain scene token without glb suffix.
    clean = scene_id.replace("\\", "/")
    parts = [p for p in clean.split("/") if p]
    for p in parts:
        if re.match(r"^[A-Za-z0-9]+$", p):
            candidates.append(p)
        if "-" in p:
            tail = p.split("-", 1)[1]
            if re.match(r"^[A-Za-z0-9]+$", tail):
                candidates.append(tail)

    # Deduplicate while keeping order.
    seen = set()
    out: List[str] = []
    for item in candidates:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _resolve_scene_path(
    workspace_root: str,
    scene_datasets_root: str,
    scene_id: str,
) -> str:
    """Resolve scene_id to an existing .glb path when possible."""

    candidates: List[str] = []

    if os.path.isabs(scene_id):
        candidates.append(scene_id)
    else:
        # Original possible paths
        candidates.append(os.path.normpath(os.path.join(workspace_root, scene_id)))
        candidates.append(os.path.normpath(os.path.join(scene_datasets_root, scene_id)))

        # If scene_id starts with data/scene_datasets
        if scene_id.startswith("data/scene_datasets/"):
            candidates.insert(0, os.path.normpath(os.path.join(workspace_root, scene_id)))

        # Your local layout:
        # JSON:   hm3d_v0.2/val/00877-xxx/xxx.basis.glb
        # Local:  data/scene_datasets/hm3d_v0.2_val/00877-xxx/xxx.basis.glb
        if scene_id.startswith("hm3d_v0.2/val/"):
            mapped = scene_id.replace("hm3d_v0.2/val/", "hm3d_v0.2_val/", 1)
            candidates.append(os.path.normpath(os.path.join(scene_datasets_root, mapped)))

        if scene_id.startswith("data/scene_datasets/hm3d_v0.2/val/"):
            mapped = scene_id.replace(
                "data/scene_datasets/hm3d_v0.2/val/",
                "data/scene_datasets/hm3d_v0.2_val/",
                1,
            )
            candidates.append(os.path.normpath(os.path.join(workspace_root, mapped)))

    for cand in candidates:
        if os.path.exists(cand):
            return cand

    raise FileNotFoundError(
        "Could not resolve scene_path.\n"
        f"scene_id: {scene_id}\n"
        f"scene_datasets_root: {scene_datasets_root}\n"
        "tried:\n  " + "\n  ".join(candidates)
    )



def _resolve_scene_dataset_config(
    workspace_root: str,
    scene_dataset_config: Optional[str],
    cli_scene_dataset_config: Optional[str],
) -> Optional[str]:
    """Resolve Habitat-Sim scene_dataset_config file.

    Use CLI override if provided.
    If CLI override is 'none', 'null', or empty string, disable scene_dataset_config.
    """

    if cli_scene_dataset_config is not None:
        raw = cli_scene_dataset_config
    else:
        raw = scene_dataset_config

    if raw is None:
        return None

    raw = str(raw).strip()

    if raw == "" or raw.lower() in {"none", "null", "no", "false"}:
        return None

    if os.path.isabs(raw):
        return os.path.normpath(raw)

    return os.path.normpath(os.path.join(workspace_root, raw))



def _find_episode(episodes_data: Dict[str, Any], episode_id: int) -> Dict[str, Any]:
    episodes = episodes_data.get("episodes", [])
    matches = [ep for ep in episodes if str(ep.get("episode_id")) == str(episode_id)]

    if not matches:
        raise ValueError(
            f"episode_id={episode_id} not found. "
            f"Loaded file has {len(episodes)} episodes."
        )

    return matches[0]


def _infer_content_json_path(
    scene_json_path: str,
    scene_id: str,
) -> Optional[str]:
    """Infer content/<scene>.json.gz from a val.json.gz or content file path."""
    scene_json_path = os.path.abspath(scene_json_path)
    parent = os.path.dirname(scene_json_path)

    # If the given file is already under content/, use it.
    if os.path.basename(parent) == "content":
        return scene_json_path

    # If the given file is val.json.gz, content/ should be a sibling directory.
    content_dir = os.path.join(parent, "content")
    if not os.path.isdir(content_dir):
        return None

    for scene_name in _scene_name_candidates(scene_id):
        cand = os.path.join(content_dir, f"{scene_name}.json.gz")
        if os.path.exists(cand):
            return cand

    return None


def _find_goal_entry(
    content_data: Dict[str, Any],
    scene_id: str,
    goal_object_id: str,
) -> Tuple[Dict[str, Any], str]:
    """Find object entry in content JSON.

    Your HM3D InstanceImageNav file layout is:

      {
        "goals": {
          "4ok3usBNeis_474": {
            "position": ...,
            "object_id": 474,
            "object_name": ...,
            "object_category": ...,
            "view_points": ...,
            "image_goals": [...]
          }
        },
        "episodes": [...]
      }
    """

    goal_object_id = str(goal_object_id)

    candidate_keys: List[str] = []
    for scene_name in _scene_name_candidates(scene_id):
        candidate_keys.append(f"{scene_name}_{goal_object_id}")

    # Case 1: old layout, object entry is directly at top level.
    for key in candidate_keys:
        if key in content_data and isinstance(content_data[key], dict):
            return content_data[key], key

    # Case 2: your actual layout, object entries are under content_data["goals"].
    if "goals" in content_data:
        goals = content_data["goals"]

        if isinstance(goals, dict):
            for key in candidate_keys:
                if key in goals and isinstance(goals[key], dict):
                    return goals[key], f"goals/{key}"

            # fallback: find key ending with _object_id
            suffix = f"_{goal_object_id}"
            matches = [
                key for key, value in goals.items()
                if str(key).endswith(suffix) and isinstance(value, dict)
            ]

            if len(matches) == 1:
                key = matches[0]
                return goals[key], f"goals/{key}"

            # fallback: search object_id field
            for key, value in goals.items():
                if not isinstance(value, dict):
                    continue
                if str(value.get("object_id")) == goal_object_id:
                    return value, f"goals/{key}"

        elif isinstance(goals, list):
            for idx, value in enumerate(goals):
                if not isinstance(value, dict):
                    continue
                if str(value.get("object_id")) == goal_object_id:
                    return value, f"goals[{idx}]"

    sample_top_keys = list(content_data.keys())[:20]
    goals_info = ""

    if "goals" in content_data:
        goals = content_data["goals"]
        if isinstance(goals, dict):
            goals_info = f"goals is dict, sample keys={list(goals.keys())[:20]}"
        elif isinstance(goals, list):
            goals_info = f"goals is list, len={len(goals)}"

    raise KeyError(
        "Could not find goal object entry.\n"
        f"scene_id: {scene_id}\n"
        f"goal_object_id: {goal_object_id}\n"
        f"tried candidate keys: {candidate_keys}\n"
        f"sample top-level keys: {sample_top_keys}\n"
        f"{goals_info}"
    )



def _select_goal_from_episode(
    scene_json_path: str,
    scene_data: Dict[str, Any],
    episodes_data: Dict[str, Any],
    episode_id: int,
) -> Tuple[Dict[str, Any], str, Dict[str, Any], Dict[str, Any]]:
    """Select goal image entry by episode id.

    Returns:
      goal_image_entry, goal_key, episode, content_data
    """
    ep = _find_episode(episodes_data, episode_id)

    scene_id = ep["scene_id"]
    goal_object_id = str(ep["goal_object_id"])
    goal_image_id = int(ep.get("goal_image_id", 0))

    # The loaded scene_data may already be a content JSON.
    if _has_non_episode_goal_entries(scene_data):
        content_data = scene_data
    else:
        content_json_path = _infer_content_json_path(scene_json_path, scene_id)
        if content_json_path is None:
            raise FileNotFoundError(
                "Could not infer content/<scene>.json.gz for this episode.\n"
                f"scene_json_path: {scene_json_path}\n"
                f"scene_id: {scene_id}"
            )
        content_data = _load_json(content_json_path)

    goal_entry, goal_key = _find_goal_entry(content_data, scene_id, goal_object_id)

    image_goals = goal_entry.get("image_goals", [])
    if goal_image_id >= len(image_goals):
        raise IndexError(
            f"goal_image_id={goal_image_id} out of range for {goal_key}; "
            f"len(image_goals)={len(image_goals)}"
        )

    return image_goals[goal_image_id], goal_key, ep, content_data


def _select_goal_from_object_key(
    scene_data: Dict[str, Any],
    object_key: str,
    goal_index: int,
) -> Tuple[Dict[str, Any], str]:
    goal_entry = scene_data.get(object_key)
    if goal_entry is None:
        sample_keys = [
            key
            for key, value in scene_data.items()
            if key != "episodes" and isinstance(value, dict)
        ][:20]
        raise KeyError(
            f"Object entry '{object_key}' not found. Sample keys: {sample_keys}"
        )

    image_goals = goal_entry.get("image_goals", [])
    if goal_index >= len(image_goals):
        raise IndexError(
            f"goal_index={goal_index} out of range for {object_key}; "
            f"len(image_goals)={len(image_goals)}"
        )

    return image_goals[goal_index], object_key


def _build_sim(
    scene_path: str,
    scene_dataset_config: Optional[str],
    width: int,
    height: int,
    hfov: float,
    camera_height: float,
) -> habitat_sim.Simulator:
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_path

    if scene_dataset_config:
        sim_cfg.scene_dataset_config_file = scene_dataset_config

    sensor_spec = habitat_sim.CameraSensorSpec()
    sensor_spec.uuid = "rgb"
    sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    sensor_spec.resolution = [height, width]
    sensor_spec.position = [0.0, camera_height, 0.0]
    sensor_spec.hfov = float(hfov)

    agent_cfg = habitat_sim.AgentConfiguration()
    agent_cfg.sensor_specifications = [sensor_spec]

    return habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))


def _render_goal_image(
    sim: habitat_sim.Simulator,
    position: List[float],
    rotation: List[float],
) -> np.ndarray:
    agent = sim.get_agent(0)
    state = agent.get_state()

    state.position = np.array(position, dtype=np.float32)
    state.rotation = quat_from_coeffs(rotation)

    agent.set_state(state, reset_sensors=True)
    obs = sim.get_sensor_observations()

    rgb = obs["rgb"]

    # Habitat-Sim COLOR sensor often returns RGBA.
    if rgb.ndim == 3 and rgb.shape[-1] == 4:
        rgb = rgb[:, :, :3]

    return rgb.astype(np.uint8)


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render InstanceImageNav goal image from HM3D JSON."
    )

    parser.add_argument(
        "--scene-json",
        required=True,
        help=(
            "Path to content/<scene>.json.gz or val.json.gz "
            "under data/datasets/instance_imagenav/hm3d/v3/val."
        ),
    )
    parser.add_argument(
        "--episodes-json",
        default=None,
        help=(
            "Optional path to val.json.gz. Needed when --scene-json is content/<scene>.json.gz "
            "and content JSON does not contain episodes. If omitted, the script tries to infer it."
        ),
    )
    parser.add_argument(
        "--scene-datasets-root",
        default="data/scene_datasets",
        help="Root directory for scene datasets. Default: data/scene_datasets",
    )
    parser.add_argument(
        "--scene-dataset-config",
        default=None,
        help=(
            "Optional override for Habitat-Sim scene_dataset_config file. "
            "If omitted, the script uses episode['scene_dataset_config'] when available."
        ),
    )

    parser.add_argument("--episode-id", type=int, default=None)
    parser.add_argument("--object-key", type=str, default=None)
    parser.add_argument("--goal-index", type=int, default=None)

    parser.add_argument(
        "--camera-height",
        type=float,
        default=0.88,
        help="Agent camera height. Default: 0.88",
    )
    parser.add_argument(
        "--output-dir",
        default="data/insinav_goal_images",
        help="Output directory for rendered goal images.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print resolved paths and selected goal metadata.",
    )

    args = parser.parse_args()

    workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    scene_json_path = _resolve_path(workspace_root, args.scene_json)
    if scene_json_path is None or not os.path.exists(scene_json_path):
        raise FileNotFoundError(f"--scene-json not found: {scene_json_path}")

    scene_data = _load_json(scene_json_path)

    episode: Optional[Dict[str, Any]] = None
    content_data = scene_data

    if args.episode_id is not None:
        # Prefer episodes from the provided scene_json itself.
        # For HM3D split datasets, val.json.gz may contain zero episodes,
        # while content/<scene>.json.gz contains the actual episodes.
        if len(scene_data.get("episodes", [])) > 0:
            episodes_data = scene_data
        else:
            episodes_json_path = args.episodes_json
            if episodes_json_path is None:
                episodes_json_path = _infer_default_episodes_json(scene_json_path)

            if episodes_json_path is not None:
                episodes_json_path = _resolve_path(workspace_root, episodes_json_path)

            if episodes_json_path and os.path.exists(episodes_json_path):
                loaded_episodes_data = _load_json(episodes_json_path)

                if len(loaded_episodes_data.get("episodes", [])) > 0:
                    episodes_data = loaded_episodes_data
                else:
                    # val.json.gz may be an index file with empty episodes.
                    # Fall back to scene_data even if it is a content file.
                    episodes_data = scene_data
            else:
                episodes_data = scene_data

        goal, goal_key, episode, content_data = _select_goal_from_episode(
            scene_json_path=scene_json_path,
            scene_data=scene_data,
            episodes_data=episodes_data,
            episode_id=args.episode_id,
        )

    else:
        if args.object_key is None or args.goal_index is None:
            raise ValueError(
                "Provide --episode-id, or provide both --object-key and --goal-index."
            )

        if not _has_non_episode_goal_entries(scene_data):
            raise ValueError(
                "--object-key mode requires --scene-json to be a content/<scene>.json.gz file "
                "that contains object entries with image_goals."
            )

        goal, goal_key = _select_goal_from_object_key(
            scene_data=scene_data,
            object_key=args.object_key,
            goal_index=args.goal_index,
        )

    # Resolve scene_id and scene_dataset_config.
    if episode is not None:
        scene_id = episode["scene_id"]
        episode_scene_dataset_config = episode.get("scene_dataset_config", None)
    else:
        # object_key mode: try to infer from content file and any available episode.
        if "episodes" in scene_data and len(scene_data["episodes"]) > 0:
            scene_id = scene_data["episodes"][0]["scene_id"]
            episode_scene_dataset_config = scene_data["episodes"][0].get(
                "scene_dataset_config", None
            )
        else:
            raise ValueError(
                "Could not infer scene_id in --object-key mode. "
                "Use --episode-id or pass a scene-json that contains episodes."
            )

    scene_datasets_root = _resolve_path(workspace_root, args.scene_datasets_root)
    if scene_datasets_root is None:
        raise ValueError("Invalid --scene-datasets-root")

    scene_path = _resolve_scene_path(
        workspace_root=workspace_root,
        scene_datasets_root=scene_datasets_root,
        scene_id=scene_id,
    )

    scene_dataset_config = _resolve_scene_dataset_config(
        workspace_root=workspace_root,
        scene_dataset_config=episode_scene_dataset_config,
        cli_scene_dataset_config=args.scene_dataset_config,
    )

    if not os.path.exists(scene_path):
        raise FileNotFoundError(
            "Resolved scene file does not exist.\n"
            f"scene_id from JSON: {scene_id}\n"
            f"resolved scene_path: {scene_path}\n"
            f"scene_datasets_root: {scene_datasets_root}"
        )

    if scene_dataset_config is not None and not os.path.exists(scene_dataset_config):
        print(
            "[WARN] Resolved scene_dataset_config does not exist; "
            "continue without scene_dataset_config.\n"
            f"       scene_dataset_config: {scene_dataset_config}"
        )
        scene_dataset_config = None

    dims = goal.get("image_dimensions", [512, 512])
    if len(dims) >= 2:
        width = int(dims[0])
        height = int(dims[1])
    else:
        width, height = 512, 512

    hfov = float(goal.get("hfov", 69.0))

    position = goal.get("position", None)
    rotation = goal.get("rotation", None)

    if position is None or rotation is None:
        raise KeyError(
            f"Selected goal entry does not contain position/rotation. goal_key={goal_key}"
        )

    if args.debug:
        print("========== Debug Info ==========")
        print(f"workspace_root:          {workspace_root}")
        print(f"scene_json_path:         {scene_json_path}")
        print(f"scene_id:                {scene_id}")
        print(f"scene_path:              {scene_path}")
        print(f"scene_dataset_config:    {scene_dataset_config}")
        print(f"goal_key:                {goal_key}")
        print(f"episode_id:              {args.episode_id}")
        print(f"image_dimensions:        {dims}")
        print(f"width x height:          {width} x {height}")
        print(f"hfov:                    {hfov}")
        print(f"position:                {position}")
        print(f"rotation:                {rotation}")
        print("================================")

    sim = _build_sim(
        scene_path=scene_path,
        scene_dataset_config=scene_dataset_config,
        width=width,
        height=height,
        hfov=hfov,
        camera_height=args.camera_height,
    )

    try:
        rgb = _render_goal_image(sim, position, rotation)
    finally:
        sim.close()

    output_dir = _resolve_path(workspace_root, args.output_dir)
    if output_dir is None:
        raise ValueError("Invalid --output-dir")

    os.makedirs(output_dir, exist_ok=True)

    if args.episode_id is not None:
        suffix = f"episode_{args.episode_id}"
    else:
        suffix = f"goal_{args.goal_index}"

    out_name = f"{_safe_name(goal_key)}_{suffix}.png"
    out_path = os.path.join(output_dir, out_name)

    Image.fromarray(rgb).convert("RGB").save(out_path)

    print(f"Saved goal image: {out_path}")


if __name__ == "__main__":
    main()
