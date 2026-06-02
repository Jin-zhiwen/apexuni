"""DINO v1 image-image cosine similarity helper for instance-image goal navigation."""

from __future__ import annotations

from typing import Optional

import os
import sys
from contextlib import contextmanager
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


class DINOSimilarity:
    def __init__(self, model_name: str = "dino_vits16", device: str = "cuda") -> None:
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        self.model = self._load_model(model_name)
        self.model.eval().to(self.device)

        self.preprocess = transforms.Compose(
            [
                transforms.Resize(224, interpolation=Image.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

        self.goal_feature: Optional[torch.Tensor] = None

    @staticmethod
    def _to_pil_rgb(image: np.ndarray) -> Image.Image:
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        return Image.fromarray(image).convert("RGB")

    def _encode_image(self, image: np.ndarray) -> torch.Tensor:
        image_tensor = self.preprocess(self._to_pil_rgb(image)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feature = self.model(image_tensor)
            feature = feature / feature.norm(dim=-1, keepdim=True)
        return feature

    @staticmethod
    @contextmanager
    def _temporary_clean_import_path():
        """Temporarily remove yolov7-related import paths/modules that shadow DINO utils."""
        removed_paths = []
        removed_modules = {}

        for index in range(len(sys.path) - 1, -1, -1):
            path_entry = sys.path[index]
            normalized = os.path.normpath(path_entry or "")
            if "yolov7" in normalized:
                removed_paths.append((index, path_entry))
                sys.path.pop(index)

        for module_name in ("utils", "vision_transformer"):
            module = sys.modules.get(module_name)
            if module is None:
                continue
            module_file = getattr(module, "__file__", "") or ""
            if "yolov7" in os.path.normpath(module_file):
                removed_modules[module_name] = module
                sys.modules.pop(module_name, None)

        try:
            yield
        finally:
            for module_name, module in removed_modules.items():
                sys.modules[module_name] = module
            for index, path_entry in sorted(removed_paths, key=lambda item: item[0]):
                sys.path.insert(index, path_entry)

    def _load_model(self, model_name: str):
        repo_path = os.environ.get("DINO_REPO_PATH", "").strip()
        candidate_repos = [
            repo_path,
            os.path.expanduser("~/.cache/torch/hub/facebookresearch_dino_main"),
            os.path.expanduser("~/.cache/torch/hub/dino_main"),
        ]
        for path in candidate_repos:
            if path and os.path.isdir(path):
                with self._temporary_clean_import_path():
                    sys.path.insert(0, path)
                    try:
                        return torch.hub.load(path, model_name, source="local")
                    finally:
                        if sys.path and sys.path[0] == path:
                            sys.path.pop(0)

        try:
            with self._temporary_clean_import_path():
                return torch.hub.load("facebookresearch/dino:main", model_name)
        except Exception as exc:
            raise RuntimeError(
                "Failed to load DINO repo from GitHub (rate limit). "
                "Please clone https://github.com/facebookresearch/dino and set DINO_REPO_PATH "
                "to the local repo path."
            ) from exc

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