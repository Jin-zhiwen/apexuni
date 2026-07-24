"""LightGlue-based instance verification helper for instance-image navigation."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

try:
    from lightglue import DISK, LightGlue
    from lightglue.utils import rbd

    _HAS_LIGHTGLUE = True
except Exception:
    _HAS_LIGHTGLUE = False


class LightGlueVerifier:
    def __init__(self, device: str = "cuda", max_num_keypoints: int = 1024) -> None:
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        self.available = _HAS_LIGHTGLUE
        self.goal_feats: Optional[dict] = None
        self.goal_feats_by_key: dict[str, Optional[dict]] = {}
        self.last_error = ""
        self._error_print_count = 0

        if not self.available:
            self.extractor = None
            self.matcher = None
            return

        self.extractor = DISK(max_num_keypoints=max_num_keypoints).eval().to(self.device)
        self.matcher = LightGlue(features="disk").eval().to(self.device)

    def _to_tensor(self, image: np.ndarray) -> torch.Tensor:
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        tensor = torch.from_numpy(image).float() / 255.0
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
        return tensor

    def _extract(self, image: np.ndarray) -> Optional[dict]:
        if not self.available:
            return None
        if image is None or image.size == 0:
            self.last_error = "empty_image"
            return None
        h, w = image.shape[:2]
        if h < 16 or w < 16:
            self.last_error = f"image_too_small:{w}x{h}"
            return None
        if image.ndim != 3 or image.shape[2] < 3:
            self.last_error = f"invalid_image_shape:{image.shape}"
            return None
        try:
            with torch.no_grad():
                feats = self.extractor.extract(self._to_tensor(image[:, :, :3]))
        except Exception as exc:
            self.last_error = f"extract_failed:{exc}"
            if self._error_print_count < 5:
                print(f"[LightGlueVerifier] extract failed: {exc}")
                self._error_print_count += 1
            return None
        self.last_error = ""
        return feats

    def set_goal_image(self, goal_image: np.ndarray, goal_key: str = "default") -> None:
        feats = self._extract(goal_image)
        self.goal_feats_by_key[goal_key] = feats
        if goal_key == "default":
            self.goal_feats = feats
        status = "ok" if feats is not None else f"failed:{self.last_error or 'unknown'}"
        print(f"[LightGlueVerifier] set_goal key={goal_key}, status={status}")

    def has_goal(self, goal_key: str = "default") -> bool:
        if goal_key == "default" and self.goal_feats is not None:
            return True
        return self.goal_feats_by_key.get(goal_key) is not None

    def clear_goal(self, goal_key: str = "default") -> None:
        self.goal_feats_by_key[goal_key] = None
        if goal_key == "default":
            self.goal_feats = None

    def match_points(self, query_image: np.ndarray, goal_key: str = "default") -> int:
        goal_feats = self.goal_feats_by_key.get(goal_key)
        if goal_key == "default" and goal_feats is None:
            goal_feats = self.goal_feats
        if not self.available or goal_feats is None:
            return 0
        query_feats = self._extract(query_image)
        if query_feats is None:
            return 0

        try:
            with torch.no_grad():
                matches01 = self.matcher(
                    {"image0": goal_feats, "image1": query_feats}
                )
            _, _, matches01 = [rbd(x) for x in [goal_feats, query_feats, matches01]]
            matches = matches01["matches"]
            return int(matches.shape[0])
        except Exception as exc:
            self.last_error = f"match_failed:{exc}"
            if self._error_print_count < 5:
                print(f"[LightGlueVerifier] match failed: {exc}")
                self._error_print_count += 1
            return 0
