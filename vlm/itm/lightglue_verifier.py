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
            return None
        h, w = image.shape[:2]
        if h < 16 or w < 16:
            return None
        with torch.no_grad():
            feats = self.extractor.extract(self._to_tensor(image))
        return feats

    def set_goal_image(self, goal_image: np.ndarray) -> None:
        self.goal_feats = self._extract(goal_image)

    def match_points(self, query_image: np.ndarray) -> int:
        if not self.available or self.goal_feats is None:
            return 0
        query_feats = self._extract(query_image)
        if query_feats is None:
            return 0

        try:
            with torch.no_grad():
                matches01 = self.matcher(
                    {"image0": self.goal_feats, "image1": query_feats}
                )
            _, _, matches01 = [rbd(x) for x in [self.goal_feats, query_feats, matches01]]
            matches = matches01["matches"]
            return int(matches.shape[0])
        except Exception:
            return 0
