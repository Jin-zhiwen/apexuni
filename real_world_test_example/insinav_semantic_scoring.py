from __future__ import annotations

from collections import deque
from typing import Iterable, List, Optional, Tuple

import numpy as np


class FullFrameScoreCalibrator:
    def __init__(self, window_size: int = 50) -> None:
        self.window_size = max(1, int(window_size))
        self._scores = deque(maxlen=self.window_size)

    def update(self, score: float) -> None:
        score = float(score)
        if np.isfinite(score):
            self._scores.append(score)

    def running_p50(self, fallback: float) -> float:
        if not self._scores:
            return float(fallback)
        return float(np.percentile(np.asarray(self._scores, dtype=np.float32), 50.0))

    def calibrated_excess(self, score: float, margin: float, scale: float) -> float:
        score = float(score)
        baseline = self.running_p50(score)
        self.update(score)

        scale = max(1e-6, float(scale))
        excess = score - baseline - float(margin)
        return float(np.clip(excess / scale, 0.0, 1.0))


def semantic_similar_weight_from_fusion_score(fusion_score: float) -> float:
    fusion_score = float(fusion_score)
    if not np.isfinite(fusion_score):
        fusion_score = 0.0
    fusion_score = float(np.clip(fusion_score, 0.0, 1.0))
    return 0.5 + 0.5 * fusion_score


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


def score_candidate_crops(
    rgb_image: np.ndarray,
    object_masks_list: Iterable[np.ndarray],
    clip_client,
    crop_padding_ratio: float = 0.1,
) -> List[Tuple[int, float]]:
    crop_scores = []
    for idx, object_mask in enumerate(object_masks_list):
        crop = crop_from_mask(rgb_image, object_mask, crop_padding_ratio)
        if crop is None:
            continue
        crop_scores.append((idx, float(clip_client.cosine(crop))))

    crop_scores.sort(key=lambda item: item[1], reverse=True)
    return crop_scores


def compute_dino_scores(
    rgb_image: np.ndarray,
    object_masks_list: List[np.ndarray],
    clip_client,
    use_candidate_crops: bool = True,
    crop_padding_ratio: float = 0.1,
    topk: int = 1,
    no_candidate_mode: str = "low_score",
    no_candidate_score: float = 0.0,
    full_frame_calibrator: Optional[FullFrameScoreCalibrator] = None,
    full_frame_min_excess: float = 0.03,
    full_frame_scale: float = 0.20,
    full_frame_weight: float = 0.15,
    crop_score_weight: float = 1.0,
    return_details: bool = False,
):
    no_candidate_mode = str(no_candidate_mode).lower()
    ranked_crop_scores = []

    if use_candidate_crops and len(object_masks_list) > 0:
        ranked_crop_scores = score_candidate_crops(
            rgb_image,
            object_masks_list,
            clip_client,
            crop_padding_ratio,
        )

    if ranked_crop_scores:
        k = min(max(1, int(topk)), len(ranked_crop_scores))
        selected_crop_scores = ranked_crop_scores[:k]
        score = float(np.mean([score for _, score in selected_crop_scores]))
        result = (score, float(crop_score_weight), "crop")
        if return_details:
            return result + (selected_crop_scores,)
        return result

    if no_candidate_mode == "full_frame":
        score = float(clip_client.cosine(rgb_image))
        result = (score, 1.0, "full_frame")
        if return_details:
            return result + ([],)
        return result

    if no_candidate_mode == "calibrated_full_frame":
        score = float(clip_client.cosine(rgb_image))
        calibrator = full_frame_calibrator or FullFrameScoreCalibrator()
        calibrated = calibrator.calibrated_excess(
            score,
            margin=full_frame_min_excess,
            scale=full_frame_scale,
        )
        if calibrated <= 1e-6:
            result = (float(no_candidate_score), 0.0, "calibrated_full_frame_skip")
        else:
            result = (calibrated, float(full_frame_weight), "calibrated_full_frame")
        if return_details:
            return result + ([],)
        return result

    if no_candidate_mode == "skip":
        result = (float(no_candidate_score), 0.0, "skip")
        if return_details:
            return result + ([],)
        return result

    score_weight = 0.0 if float(no_candidate_score) < 0.0 else 1.0
    result = (float(no_candidate_score), score_weight, "low_score")
    if return_details:
        return result + ([],)
    return result
