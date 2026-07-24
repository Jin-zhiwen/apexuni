"""MASt3R-based close-range pose refinement for instance-image navigation.

This implementation is benchmark-safe:
- it only uses the provided goal image plus current RGB/depth observations
- it never reads dataset goal viewpoint metadata
- it estimates a relative close-range alignment signal before STOP
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

try:
    import torch
except Exception:
    torch = None

_DEFAULT_MAST3R_ROOT = Path(__file__).resolve().parents[2] / "mast3r"
_DEFAULT_DUST3R_ROOT = _DEFAULT_MAST3R_ROOT / "dust3r"
for _path in (_DEFAULT_MAST3R_ROOT, _DEFAULT_DUST3R_ROOT):
    _path_str = str(_path)
    if _path.exists() and _path_str not in sys.path:
        sys.path.insert(0, _path_str)

AsymmetricMASt3R = None
fast_reciprocal_NNs = None
inference = None
_HAS_MAST3R = False
_MAST3R_IMPORT_ERROR = ""


def _try_import_mast3r() -> bool:
    global AsymmetricMASt3R, fast_reciprocal_NNs, inference
    global _HAS_MAST3R, _MAST3R_IMPORT_ERROR

    if _HAS_MAST3R:
        return True
    if torch is None:
        _MAST3R_IMPORT_ERROR = "torch_unavailable"
        return False

    try:
        from mast3r.model import AsymmetricMASt3R as _AsymmetricMASt3R
        from mast3r.fast_nn import fast_reciprocal_NNs as _fast_reciprocal_NNs
        import mast3r.utils.path_to_dust3r  # noqa: F401
        from dust3r.inference import inference as _inference

        AsymmetricMASt3R = _AsymmetricMASt3R
        fast_reciprocal_NNs = _fast_reciprocal_NNs
        inference = _inference
        _HAS_MAST3R = True
        _MAST3R_IMPORT_ERROR = ""
        return True
    except Exception as exc:
        _MAST3R_IMPORT_ERROR = str(exc)
        return False


_try_import_mast3r()


def _normalize_depth(depth_image: np.ndarray, min_depth: float, max_depth: float) -> np.ndarray:
    if depth_image.ndim == 3:
        depth_norm = depth_image[:, :, 0].astype(np.float32)
    else:
        depth_norm = depth_image.astype(np.float32)
    depth_metric = depth_norm * (max_depth - min_depth) + min_depth
    depth_metric[~np.isfinite(depth_metric)] = 0.0
    return depth_metric


def _camera_intrinsics(width: int, height: int, hfov_deg: float) -> np.ndarray:
    hfov_rad = np.deg2rad(float(hfov_deg))
    fx = 0.5 * width / np.tan(hfov_rad * 0.5)
    fy = fx
    cx = (width - 1) * 0.5
    cy = (height - 1) * 0.5
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


def _rotation_to_yaw_deg(rot_mat: np.ndarray) -> float:
    return float(np.rad2deg(np.arctan2(rot_mat[0, 2], rot_mat[2, 2])))


def _wrap_angle_deg(angle_deg: float) -> float:
    return float((angle_deg + 180.0) % 360.0 - 180.0)


@dataclass
class PoseRefinementResult:
    available: bool
    success: bool
    num_matches: int = 0
    valid_matches: int = 0
    inliers: int = 0
    goal_mask_inlier_ratio: float = float("nan")
    current_mask_inlier_ratio: float = float("nan")
    joint_mask_inlier_ratio: float = float("nan")
    transl_error: float = float("inf")
    yaw_error_deg: float = float("inf")
    forward_error: float = float("inf")
    lateral_error: float = float("inf")
    depth_error: float = float("inf")
    should_stop: bool = False
    suggested_action: Optional[str] = None
    debug_reason: str = ""


class MASt3RPoseRefiner:
    def __init__(
        self,
        weights: Optional[str] = None,
        mast3r_root: str = "mast3r",
        device: str = "cuda",
        image_size: int = 512,
        min_conf_thr: float = 1.001,
        match_subsample: int = 8,
        min_matches: int = 40,
        stop_translation_tolerance: float = 0.20,
        stop_yaw_tolerance_deg: float = 12.0,
        action_yaw_tolerance_deg: float = 8.0,
        action_forward_tolerance: float = 0.20,
        action_lateral_tolerance: float = 0.10,
        forward_bias: float = 0.0,
        stop_depth_tolerance: float = 0.20,
        center_tolerance_px: float = 30.0,
        center_gain: float = 0.003,
        pnp_reprojection_error_px: float = 5.0,
        pnp_iterations: int = 100,
        pose_consistency_translation: float = 0.75,
        pose_consistency_yaw_deg: float = 30.0,
    ) -> None:
        self.available = False
        self.unavailable_reason = ""
        self.goal_rgb: Optional[np.ndarray] = None
        self.device = device
        self.image_size = int(image_size)
        self.min_conf_thr = float(min_conf_thr)
        self.match_subsample = int(match_subsample)
        self.min_matches = int(min_matches)
        self.stop_translation_tolerance = float(stop_translation_tolerance)
        self.stop_yaw_tolerance_deg = float(stop_yaw_tolerance_deg)
        self.action_yaw_tolerance_deg = float(action_yaw_tolerance_deg)
        self.action_forward_tolerance = float(action_forward_tolerance)
        self.action_lateral_tolerance = float(action_lateral_tolerance)
        self.forward_bias = float(forward_bias)
        self.stop_depth_tolerance = float(stop_depth_tolerance)
        self.center_tolerance_px = float(center_tolerance_px)
        self.center_gain = float(center_gain)
        self.pnp_reprojection_error_px = float(pnp_reprojection_error_px)
        self.pnp_iterations = int(pnp_iterations)
        self.pose_consistency_translation = float(pose_consistency_translation)
        self.pose_consistency_yaw_deg = float(pose_consistency_yaw_deg)

        if mast3r_root:
            mast3r_root_path = Path(mast3r_root).resolve()
            dust3r_root_path = mast3r_root_path / "dust3r"
            for extra_path in (mast3r_root_path, dust3r_root_path):
                extra_path_str = str(extra_path)
                if extra_path.exists() and extra_path_str not in sys.path:
                    sys.path.insert(0, extra_path_str)

        has_mast3r = _try_import_mast3r()
        if torch is None:
            self.unavailable_reason = "torch_unavailable"
            self.model = None
            return
        if cv2 is None:
            self.unavailable_reason = "opencv_unavailable"
            self.model = None
            return
        if not has_mast3r:
            self.unavailable_reason = f"mast3r_import_failed:{_MAST3R_IMPORT_ERROR}"
            self.model = None
            return

        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = device

        model_name = weights or "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
        try:
            self.model = AsymmetricMASt3R.from_pretrained(model_name).to(self.device)
        except Exception as exc:
            self.unavailable_reason = f"mast3r_model_load_failed:{exc}"
            self.model = None
            return
        self.model.eval()
        self.available = True

    def set_goal_image(self, goal_rgb: np.ndarray) -> None:
        self.goal_rgb = goal_rgb

    def _prepare_image(self, image: np.ndarray) -> Dict:
        import PIL.Image
        import torchvision.transforms as tvf

        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        pil = tvf.functional.to_pil_image(image).convert("RGB")
        orig_w, orig_h = pil.size
        long_edge = max(orig_w, orig_h)
        if long_edge <= 0:
            raise ValueError("empty image")

        scale = float(self.image_size) / float(long_edge)
        resized_w = max(1, int(round(orig_w * scale)))
        resized_h = max(1, int(round(orig_h * scale)))
        interp = PIL.Image.LANCZOS if long_edge > self.image_size else PIL.Image.BICUBIC
        resized = pil.resize((resized_w, resized_h), interp)

        patch_size = 16
        cx, cy = resized_w // 2, resized_h // 2
        half_w = int(((2 * cx) // patch_size) * patch_size / 2)
        half_h = int(((2 * cy) // patch_size) * patch_size / 2)
        if resized_w == resized_h:
            half_h = int(3 * half_w / 4)

        left = max(0, int(round(cx - half_w)))
        top = max(0, int(round(cy - half_h)))
        right = min(resized_w, int(round(cx + half_w)))
        bottom = min(resized_h, int(round(cy + half_h)))
        if right <= left or bottom <= top:
            raise ValueError("invalid mast3r crop")

        cropped = resized.crop((left, top, right, bottom))
        img_norm = tvf.Compose(
            [tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        tensor = img_norm(cropped).unsqueeze(0).to(self.device)
        return {
            "img": tensor,
            "true_shape": np.int32([tensor.shape[-2:]]),
            "scale": scale,
            "crop_left": float(left),
            "crop_top": float(top),
            "orig_size": (orig_w, orig_h),
        }

    @staticmethod
    def _net_to_orig_xy(x: float, y: float, image_meta: Dict) -> tuple[float, float]:
        scale = float(image_meta["scale"])
        if scale <= 0:
            return x, y
        orig_w, orig_h = image_meta["orig_size"]
        orig_x = (x + float(image_meta["crop_left"])) / scale
        orig_y = (y + float(image_meta["crop_top"])) / scale
        return (
            float(np.clip(orig_x, 0.0, float(orig_w - 1))),
            float(np.clip(orig_y, 0.0, float(orig_h - 1))),
        )

    @staticmethod
    def _intrinsics_to_net_image(intrinsics: np.ndarray, image_meta: Dict) -> np.ndarray:
        scale = float(image_meta["scale"])
        intrinsics_net = intrinsics.astype(np.float32).copy()
        intrinsics_net[0, 0] *= scale
        intrinsics_net[1, 1] *= scale
        intrinsics_net[0, 2] = intrinsics_net[0, 2] * scale - float(image_meta["crop_left"])
        intrinsics_net[1, 2] = intrinsics_net[1, 2] * scale - float(image_meta["crop_top"])
        return intrinsics_net

    def _run_pair(self, current_rgb: np.ndarray):
        goal = self._prepare_image(self.goal_rgb)
        current = self._prepare_image(current_rgb)
        pair = [
            {"img": goal["img"], "idx": 0, "instance": 0, "true_shape": goal["true_shape"]},
            {"img": current["img"], "idx": 1, "instance": 1, "true_shape": current["true_shape"]},
        ]
        with torch.no_grad():
            output = inference([tuple(pair)], self.model, self.device, batch_size=1, verbose=False)
        return output, goal, current

    def refine(
        self,
        current_rgb: np.ndarray,
        current_depth_metric: np.ndarray,
        current_intrinsics: np.ndarray,
        goal_mask: Optional[np.ndarray] = None,
        current_mask: Optional[np.ndarray] = None,
    ) -> PoseRefinementResult:
        if not self.available:
            return PoseRefinementResult(
                False,
                False,
                debug_reason=self.unavailable_reason or "mast3r_unavailable",
            )
        if cv2 is None:
            return PoseRefinementResult(False, False, debug_reason="opencv_unavailable")
        if self.goal_rgb is None:
            return PoseRefinementResult(True, False, debug_reason="goal_image_not_initialized")

        try:
            output, goal_meta, current_meta = self._run_pair(current_rgb)
            view1, view2 = output["view1"], output["view2"]
            pred1, pred2 = output["pred1"], output["pred2"]
            desc1 = pred1["desc"].squeeze(0).detach()
            desc2 = pred2["desc"].squeeze(0).detach()
            conf1 = pred1["desc_conf"].squeeze(0).detach().cpu().numpy()
            conf2 = pred2["desc_conf"].squeeze(0).detach().cpu().numpy()
            pts3d_cur_in_goal = pred2["pts3d_in_other_view"].squeeze(0).detach().cpu().numpy()
            matches_goal, matches_cur = fast_reciprocal_NNs(
                desc1,
                desc2,
                subsample_or_initxy1=self.match_subsample,
                device=self.device,
                dist="dot",
                block_size=2**13,
            )
        except Exception as exc:
            return PoseRefinementResult(True, False, debug_reason=f"mast3r_inference_failed:{exc}")

        if len(matches_goal) == 0:
            return PoseRefinementResult(True, False, debug_reason="no_matches")

        h_cur, w_cur = current_rgb.shape[:2]
        goal_shape = view1["true_shape"][0]
        cur_shape = view2["true_shape"][0]
        if hasattr(goal_shape, "detach"):
            goal_shape = goal_shape.detach().cpu().numpy()
        if hasattr(cur_shape, "detach"):
            cur_shape = cur_shape.detach().cpu().numpy()
        h_goal_net, w_goal_net = int(goal_shape[0]), int(goal_shape[1])
        h_cur_net, w_cur_net = int(cur_shape[0]), int(cur_shape[1])

        current_points = []
        predicted_goal_points = []
        goal_pixels = []
        current_pixels = []
        current_pixels_net = []
        valid_count = 0

        def normalized_mask(mask, image_shape):
            if mask is None:
                return None
            binary_mask = np.asarray(mask)
            if binary_mask.ndim == 3:
                binary_mask = binary_mask[:, :, 0]
            target_h, target_w = image_shape[:2]
            if binary_mask.shape != (target_h, target_w):
                binary_mask = cv2.resize(
                    binary_mask.astype(np.uint8),
                    (target_w, target_h),
                    interpolation=cv2.INTER_NEAREST,
                )
            return binary_mask > 0

        goal_mask_binary = normalized_mask(goal_mask, self.goal_rgb.shape)
        current_mask_binary = normalized_mask(current_mask, current_rgb.shape)

        def mask_contains(mask, x, y):
            if mask is None:
                return True
            target_h, target_w = mask.shape
            xi = int(np.clip(round(float(x)), 0, target_w - 1))
            yi = int(np.clip(round(float(y)), 0, target_h - 1))
            return bool(mask[yi, xi])

        for (xg, yg), (xc, yc) in zip(matches_goal, matches_cur):
            if xg < 3 or yg < 3 or xc < 3 or yc < 3:
                continue
            if xg >= w_goal_net - 3 or yg >= h_goal_net - 3 or xc >= w_cur_net - 3 or yc >= h_cur_net - 3:
                continue
            conf_goal = float(conf1[int(yg), int(xg)])
            conf_cur = float(conf2[int(yc), int(xc)])
            if min(conf_goal, conf_cur) < self.min_conf_thr:
                continue

            xg_full, yg_full = self._net_to_orig_xy(float(xg), float(yg), goal_meta)
            xc_orig, yc_orig = self._net_to_orig_xy(float(xc), float(yc), current_meta)
            if not mask_contains(goal_mask_binary, xg_full, yg_full):
                continue
            if not mask_contains(current_mask_binary, xc_orig, yc_orig):
                continue
            xc_full = int(np.clip(round(xc_orig), 0, w_cur - 1))
            yc_full = int(np.clip(round(yc_orig), 0, h_cur - 1))

            depth_cur = float(current_depth_metric[yc_full, xc_full])
            if not np.isfinite(depth_cur) or depth_cur <= 1e-4:
                continue

            px = (xc_full - current_intrinsics[0, 2]) * depth_cur / current_intrinsics[0, 0]
            py = (yc_full - current_intrinsics[1, 2]) * depth_cur / current_intrinsics[1, 1]
            pz = depth_cur
            cur_point = np.array([px, py, pz], dtype=np.float32)

            x_net = int(np.clip(round(float(xc)), 0, w_cur_net - 1))
            y_net = int(np.clip(round(float(yc)), 0, h_cur_net - 1))
            goal_point_pred = pts3d_cur_in_goal[y_net, x_net].astype(np.float32)
            if not np.isfinite(goal_point_pred).all():
                continue
            if goal_point_pred[2] <= 1e-4:
                continue

            current_points.append(cur_point)
            predicted_goal_points.append(goal_point_pred)
            goal_pixels.append([xg_full, yg_full])
            current_pixels.append([float(xc_full), float(yc_full)])
            current_pixels_net.append([float(xc), float(yc)])
            valid_count += 1

        if valid_count < self.min_matches:
            return PoseRefinementResult(
                True,
                False,
                num_matches=len(matches_goal),
                valid_matches=valid_count,
                inliers=valid_count,
                debug_reason="insufficient_valid_matches",
            )

        cur_pts = np.asarray(current_points, dtype=np.float32)
        goal_pts_pred = np.asarray(predicted_goal_points, dtype=np.float32)
        goal_pixels = np.asarray(goal_pixels, dtype=np.float32)
        current_pixels = np.asarray(current_pixels, dtype=np.float32)
        current_pixels_net = np.asarray(current_pixels_net, dtype=np.float32)
        current_intrinsics_net = self._intrinsics_to_net_image(current_intrinsics, current_meta)

        try:
            pnp_flag = getattr(cv2, "SOLVEPNP_SQPNP", cv2.SOLVEPNP_EPNP)
            pnp_success, pnp_rvec, pnp_tvec, pnp_inliers = cv2.solvePnPRansac(
                goal_pts_pred.astype(np.float32),
                current_pixels_net.astype(np.float32),
                current_intrinsics_net.astype(np.float32),
                None,
                iterationsCount=self.pnp_iterations,
                reprojectionError=self.pnp_reprojection_error_px,
                confidence=0.99,
                flags=pnp_flag,
            )
        except Exception as exc:
            return PoseRefinementResult(
                True,
                False,
                num_matches=len(matches_goal),
                valid_matches=valid_count,
                inliers=0,
                debug_reason=f"pnp_failed:{exc}",
            )

        if not pnp_success or pnp_rvec is None or pnp_tvec is None or pnp_inliers is None:
            return PoseRefinementResult(
                True,
                False,
                num_matches=len(matches_goal),
                valid_matches=valid_count,
                inliers=0,
                debug_reason="pnp_failed",
            )

        pnp_inliers = np.asarray(pnp_inliers).reshape(-1).astype(np.int64)
        min_inliers = max(6, self.min_matches // 4)
        if pnp_inliers.size < min_inliers:
            return PoseRefinementResult(
                True,
                False,
                num_matches=len(matches_goal),
                valid_matches=valid_count,
                inliers=int(pnp_inliers.size),
                debug_reason="insufficient_pnp_inliers",
            )

        pnp_goal_pixels = goal_pixels[pnp_inliers]
        pnp_current_pixels = current_pixels[pnp_inliers]
        goal_membership = np.asarray(
            [
                mask_contains(goal_mask_binary, x, y)
                for x, y in pnp_goal_pixels
            ],
            dtype=bool,
        )
        current_membership = np.asarray(
            [
                mask_contains(current_mask_binary, x, y)
                for x, y in pnp_current_pixels
            ],
            dtype=bool,
        )
        goal_mask_inlier_ratio = (
            float(np.mean(goal_membership)) if goal_mask is not None else float("nan")
        )
        current_mask_inlier_ratio = (
            float(np.mean(current_membership))
            if current_mask is not None
            else float("nan")
        )
        joint_mask_inlier_ratio = (
            float(np.mean(goal_membership & current_membership))
            if goal_mask is not None and current_mask is not None
            else float("nan")
        )

        rot_goal_to_cur, _ = cv2.Rodrigues(pnp_rvec)
        trans_goal_to_cur = pnp_tvec.reshape(3).astype(np.float32)

        pnp_goal_inliers = goal_pts_pred[pnp_inliers]
        pnp_cur_metric_inliers = cur_pts[pnp_inliers]
        pnp_cur_pred = (rot_goal_to_cur @ pnp_goal_inliers.T).T + trans_goal_to_cur[None, :]
        valid_scale = (
            np.isfinite(pnp_cur_pred).all(axis=1)
            & np.isfinite(pnp_cur_metric_inliers).all(axis=1)
            & (pnp_cur_pred[:, 2] > 1e-4)
            & (pnp_cur_metric_inliers[:, 2] > 1e-4)
        )
        if int(valid_scale.sum()) < min_inliers:
            return PoseRefinementResult(
                True,
                False,
                num_matches=len(matches_goal),
                valid_matches=valid_count,
                inliers=int(valid_scale.sum()),
                debug_reason="insufficient_pnp_depth_inliers",
            )

        pnp_scale = float(
            np.median(
                pnp_cur_pred[valid_scale, 2] / pnp_cur_metric_inliers[valid_scale, 2]
            )
        )
        if not np.isfinite(pnp_scale) or pnp_scale <= 1e-6:
            return PoseRefinementResult(
                True,
                False,
                num_matches=len(matches_goal),
                valid_matches=valid_count,
                inliers=int(pnp_inliers.size),
                debug_reason="invalid_pnp_scale",
            )

        pnp_cur_pred_metric = pnp_cur_pred[valid_scale] / pnp_scale
        pnp_cur_metric_valid = pnp_cur_metric_inliers[valid_scale]
        depth_error = float(
            np.median(np.abs(pnp_cur_pred_metric[:, 2] - pnp_cur_metric_valid[:, 2]))
        )
        pnp_residual = float(
            np.median(np.linalg.norm(pnp_cur_pred_metric - pnp_cur_metric_valid, axis=1))
        )

        try:
            retval, affine, inlier_mask = cv2.estimateAffine3D(
                cur_pts,
                goal_pts_pred,
                ransacThreshold=0.05,
                confidence=0.99,
            )
        except Exception as exc:
            return PoseRefinementResult(
                True,
                False,
                num_matches=len(matches_goal),
                valid_matches=valid_count,
                inliers=0,
                debug_reason=f"affine3d_failed:{exc}",
            )

        if not retval or affine is None:
            return PoseRefinementResult(
                True,
                False,
                num_matches=len(matches_goal),
                valid_matches=valid_count,
                inliers=0,
                debug_reason="affine3d_failed",
            )

        rot_affine = affine[:, :3]
        trans_affine = affine[:, 3]
        try:
            u, singular_values, vt = np.linalg.svd(rot_affine)
            rot_cur_to_goal = u @ vt
            if np.linalg.det(rot_cur_to_goal) < 0:
                u[:, -1] *= -1.0
                rot_cur_to_goal = u @ vt
            scale_cur_to_goal = float(np.mean(np.abs(singular_values)))
            if not np.isfinite(scale_cur_to_goal) or scale_cur_to_goal <= 1e-6:
                return PoseRefinementResult(
                    True,
                    False,
                    num_matches=len(matches_goal),
                    valid_matches=valid_count,
                    inliers=0,
                    debug_reason="invalid_affine_scale",
                )
        except np.linalg.LinAlgError:
            return PoseRefinementResult(
                True,
                False,
                num_matches=len(matches_goal),
                valid_matches=valid_count,
                inliers=0,
                debug_reason="svd_failed",
            )

        trans_cur_to_goal = (trans_affine / scale_cur_to_goal).astype(np.float32)
        goal_pts_metric = (goal_pts_pred / scale_cur_to_goal).astype(np.float32)

        try:
            goal_from_cur = (rot_cur_to_goal @ cur_pts.T).T + trans_cur_to_goal[None, :]
            residuals = np.linalg.norm(goal_from_cur - goal_pts_metric, axis=1)
            affine_depth_error = float(np.median(np.abs(goal_from_cur[:, 2] - goal_pts_metric[:, 2])))
        except Exception:
            residuals = np.linalg.norm(trans_cur_to_goal[None, :], axis=1)
            affine_depth_error = float("inf")

        affine_inlier_count = int(inlier_mask.sum()) if inlier_mask is not None else int(np.sum(residuals < 0.10))
        if affine_inlier_count < min_inliers:
            return PoseRefinementResult(
                True,
                False,
                num_matches=len(matches_goal),
                valid_matches=valid_count,
                inliers=affine_inlier_count,
                debug_reason="insufficient_affine_inliers",
            )

        affine_goal_origin_in_current = -(rot_cur_to_goal.T @ trans_cur_to_goal)
        affine_yaw_error_deg = -_rotation_to_yaw_deg(rot_cur_to_goal)

        rot_cur_to_goal_pnp = rot_goal_to_cur.T
        goal_origin_in_current = trans_goal_to_cur / pnp_scale
        yaw_error_deg = -_rotation_to_yaw_deg(rot_cur_to_goal_pnp)
        translation_disagreement = float(
            np.linalg.norm(goal_origin_in_current[[0, 2]] - affine_goal_origin_in_current[[0, 2]])
        )
        yaw_disagreement = abs(_wrap_angle_deg(yaw_error_deg - affine_yaw_error_deg))
        if (
            translation_disagreement > self.pose_consistency_translation
            or yaw_disagreement > self.pose_consistency_yaw_deg
        ):
            return PoseRefinementResult(
                True,
                False,
                num_matches=len(matches_goal),
                valid_matches=valid_count,
                inliers=int(pnp_inliers.size),
                transl_error=float(
                    np.linalg.norm([goal_origin_in_current[2], -goal_origin_in_current[0]])
                ),
                yaw_error_deg=yaw_error_deg,
                forward_error=float(goal_origin_in_current[2] - self.forward_bias),
                lateral_error=float(-goal_origin_in_current[0]),
                depth_error=depth_error,
                should_stop=False,
                debug_reason=(
                    "pose_inconsistent:"
                    f"trans={translation_disagreement:.3f},yaw={yaw_disagreement:.2f},"
                    f"pnp_residual={pnp_residual:.3f},affine_depth={affine_depth_error:.3f}"
                ),
            )

        forward_error = float(goal_origin_in_current[2] - self.forward_bias)
        # Positive lateral error means the goal viewpoint is on the robot's left.
        lateral_error = float(-goal_origin_in_current[0])
        transl_error = float(np.linalg.norm([forward_error, lateral_error]))

        goal_center_x = float(np.median(goal_pixels[:, 0]))
        current_center_x = float(np.median(current_pixels[:, 0]))
        goal_w = float(goal_meta["orig_size"][0])
        current_center_norm = (current_center_x - 0.5 * float(w_cur - 1)) / max(1.0, float(w_cur))
        goal_center_norm = (goal_center_x - 0.5 * float(goal_w - 1)) / max(1.0, goal_w)
        center_offset_px = (current_center_norm - goal_center_norm) * float(w_cur)

        center_turn_bias_deg = float(-center_offset_px * self.center_gain)
        blended_yaw_error_deg = yaw_error_deg + center_turn_bias_deg

        should_stop = (
            transl_error <= self.stop_translation_tolerance
            and abs(blended_yaw_error_deg) <= self.stop_yaw_tolerance_deg
            and depth_error <= self.stop_depth_tolerance
            and abs(center_offset_px) <= self.center_tolerance_px
        )

        suggested_action = None
        if not should_stop:
            if abs(blended_yaw_error_deg) > self.action_yaw_tolerance_deg:
                suggested_action = "turn_right" if blended_yaw_error_deg > 0 else "turn_left"
            elif abs(center_offset_px) > self.center_tolerance_px:
                suggested_action = "turn_left" if center_offset_px < 0 else "turn_right"
            elif forward_error > self.action_forward_tolerance:
                suggested_action = "move_forward"
            elif lateral_error > self.action_lateral_tolerance:
                suggested_action = "turn_left"
            elif lateral_error < -self.action_lateral_tolerance:
                suggested_action = "turn_right"
            else:
                suggested_action = "turn_right" if blended_yaw_error_deg >= 0 else "turn_left"

        return PoseRefinementResult(
            available=True,
            success=True,
            num_matches=len(matches_goal),
            valid_matches=valid_count,
            inliers=int(pnp_inliers.size),
            goal_mask_inlier_ratio=goal_mask_inlier_ratio,
            current_mask_inlier_ratio=current_mask_inlier_ratio,
            joint_mask_inlier_ratio=joint_mask_inlier_ratio,
            transl_error=transl_error,
            yaw_error_deg=blended_yaw_error_deg,
            forward_error=forward_error,
            lateral_error=lateral_error,
            depth_error=depth_error,
            should_stop=should_stop,
            suggested_action=suggested_action,
            debug_reason=(
                "ok_pnp:"
                f"pnp_inliers={int(pnp_inliers.size)},affine_inliers={affine_inlier_count},"
                f"scale={pnp_scale:.4f},pnp_residual={pnp_residual:.3f},"
                f"affine_depth={affine_depth_error:.3f},"
                f"pose_delta={translation_disagreement:.3f}/{yaw_disagreement:.2f},"
                f"mask_ratio={goal_mask_inlier_ratio:.3f}/"
                f"{current_mask_inlier_ratio:.3f}/{joint_mask_inlier_ratio:.3f}"
            ),
        )
