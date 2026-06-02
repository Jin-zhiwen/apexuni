"""LightGlue-based instance verification helper for instance-image navigation."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

try:
    import cv2

    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

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

    def match_stats(self, query_image: np.ndarray, ransac_reproj_threshold: float = 5.0) -> dict:
        empty = {
            "matches": 0,
            "inliers": 0,
            "inlier_ratio": 0.0,
            "goal_keypoints": 0,
            "query_keypoints": 0,
        }
        if not self.available or self.goal_feats is None:
            return empty
        query_feats = self._extract(query_image)
        if query_feats is None:
            return empty

        try:
            with torch.no_grad():
                matches01 = self.matcher(
                    {"image0": self.goal_feats, "image1": query_feats}
                )
            goal_feats, query_feats, matches01 = [
                rbd(x) for x in [self.goal_feats, query_feats, matches01]
            ]
            matches = matches01["matches"]
            raw_matches = int(matches.shape[0])
            goal_keypoints = int(goal_feats["keypoints"].shape[0])
            query_keypoints = int(query_feats["keypoints"].shape[0])
            if raw_matches == 0:
                return {
                    "matches": 0,
                    "inliers": 0,
                    "inlier_ratio": 0.0,
                    "goal_keypoints": goal_keypoints,
                    "query_keypoints": query_keypoints,
                }

            if not _HAS_CV2 or raw_matches < 8:
                return {
                    "matches": raw_matches,
                    "inliers": 0,
                    "inlier_ratio": 0.0,
                    "goal_keypoints": goal_keypoints,
                    "query_keypoints": query_keypoints,
                }

            idx0 = matches[:, 0].detach().cpu().numpy()
            idx1 = matches[:, 1].detach().cpu().numpy()
            pts0 = goal_feats["keypoints"][idx0].detach().cpu().numpy().astype(np.float32)
            pts1 = query_feats["keypoints"][idx1].detach().cpu().numpy().astype(np.float32)
            _, inlier_mask = cv2.findHomography(
                pts0,
                pts1,
                cv2.RANSAC,
                ransac_reproj_threshold,
            )
            if inlier_mask is None:
                inliers = 0
            else:
                inliers = int(np.count_nonzero(inlier_mask))
            return {
                "matches": raw_matches,
                "inliers": inliers,
                "inlier_ratio": float(inliers) / float(raw_matches),
                "goal_keypoints": goal_keypoints,
                "query_keypoints": query_keypoints,
            }
        except Exception:
            return empty

    def match_points(self, query_image: np.ndarray) -> int:
        return int(self.match_stats(query_image).get("matches", 0))
