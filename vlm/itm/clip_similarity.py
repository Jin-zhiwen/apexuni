"""CLIP image-image cosine similarity helper for instance-image goal navigation."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from PIL import Image

import clip


class CLIPSimilarity:
    def __init__(self, model_name: str = "ViT-B/32", device: str = "cuda") -> None:
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()
        self.goal_feature: Optional[torch.Tensor] = None

    @staticmethod
    def _to_pil_rgb(image: np.ndarray) -> Image.Image:
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        return Image.fromarray(image).convert("RGB")

    def _encode_image(self, image: np.ndarray) -> torch.Tensor:
        image_tensor = self.preprocess(self._to_pil_rgb(image)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feature = self.model.encode_image(image_tensor)
            feature = feature / feature.norm(dim=-1, keepdim=True)
        return feature

    def set_goal_image(self, goal_image: np.ndarray) -> None:
        self.goal_feature = self._encode_image(goal_image)

    def set_goal_clean_crop(self, goal_clean_crop: np.ndarray) -> None:
        self.goal_feature = self._encode_image(goal_clean_crop)

    def cosine(self, rgb_image: np.ndarray) -> float:
        if self.goal_feature is None:
            raise ValueError("Goal image is not set. Call set_goal_image() first.")
        cur_feature = self._encode_image(rgb_image)
        score = torch.matmul(cur_feature, self.goal_feature.T).item()
        return float(score)
