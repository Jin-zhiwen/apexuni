import unittest
from pathlib import Path

import numpy as np

from insinav_semantic_scoring import (
    FullFrameScoreCalibrator,
    compute_dino_scores,
    semantic_similar_weight_from_fusion_score,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class FakeDinoClient:
    def __init__(self, scores):
        self.scores = list(scores)
        self.calls = []

    def cosine(self, image):
        self.calls.append(tuple(image.shape))
        if not self.scores:
            raise AssertionError("No fake DINO score left")
        return float(self.scores.pop(0))


class InsinavSemanticScoringTest(unittest.TestCase):
    def test_candidate_crops_return_strong_weighted_topk_score(self):
        rgb = np.zeros((10, 10, 3), dtype=np.uint8)
        mask_a = np.zeros((10, 10), dtype=np.uint8)
        mask_b = np.zeros((10, 10), dtype=np.uint8)
        mask_a[1:4, 1:4] = 1
        mask_b[5:9, 5:9] = 1

        score, weight, source = compute_dino_scores(
            rgb,
            [mask_a, mask_b],
            FakeDinoClient([0.42, 0.62]),
            use_candidate_crops=True,
            crop_padding_ratio=0.0,
            topk=2,
            no_candidate_mode="skip",
            no_candidate_score=-1.0,
            full_frame_calibrator=FullFrameScoreCalibrator(),
            crop_score_weight=1.0,
        )

        self.assertAlmostEqual(score, 0.52)
        self.assertEqual(weight, 1.0)
        self.assertEqual(source, "crop")

    def test_skip_mode_marks_value_map_update_invalid(self):
        score, weight, source = compute_dino_scores(
            np.zeros((8, 8, 3), dtype=np.uint8),
            [],
            FakeDinoClient([]),
            no_candidate_mode="skip",
            no_candidate_score=-1.0,
            full_frame_calibrator=FullFrameScoreCalibrator(),
        )

        self.assertEqual(score, -1.0)
        self.assertEqual(weight, 0.0)
        self.assertEqual(source, "skip")

    def test_calibrated_full_frame_writes_only_excess_with_weak_weight(self):
        calibrator = FullFrameScoreCalibrator(window_size=5)
        for value in [0.40, 0.41, 0.42, 0.43, 0.44]:
            calibrator.update(value)

        score, weight, source = compute_dino_scores(
            np.zeros((8, 8, 3), dtype=np.uint8),
            [],
            FakeDinoClient([0.50]),
            no_candidate_mode="calibrated_full_frame",
            no_candidate_score=-1.0,
            full_frame_calibrator=calibrator,
            full_frame_min_excess=0.03,
            full_frame_scale=0.20,
            full_frame_weight=0.15,
        )

        self.assertAlmostEqual(score, 0.25, places=6)
        self.assertAlmostEqual(weight, 0.15)
        self.assertEqual(source, "calibrated_full_frame")

    def test_calibrated_full_frame_below_margin_skips_update(self):
        calibrator = FullFrameScoreCalibrator(window_size=5)
        for value in [0.40, 0.41, 0.42, 0.43, 0.44]:
            calibrator.update(value)

        score, weight, source = compute_dino_scores(
            np.zeros((8, 8, 3), dtype=np.uint8),
            [],
            FakeDinoClient([0.44]),
            no_candidate_mode="calibrated_full_frame",
            no_candidate_score=-1.0,
            full_frame_calibrator=calibrator,
            full_frame_min_excess=0.03,
            full_frame_scale=0.20,
            full_frame_weight=0.15,
        )

        self.assertEqual(score, -1.0)
        self.assertEqual(weight, 0.0)
        self.assertEqual(source, "calibrated_full_frame_skip")

    def test_real_world_config_defaults_to_dense_view_plus_object_crop_evidence(self):
        config_source = (
            REPO_ROOT / "real_world_test_example" / "config" / "real_world_test_insinav.yaml"
        ).read_text()

        self.assertIn("dense_full_frame_value_map: true", config_source)
        self.assertIn("use_candidate_crops: true", config_source)
        self.assertIn("topk: 2", config_source)
        self.assertIn("publish_semantic_object_clouds: true", config_source)
        self.assertIn('no_candidate_mode: "full_frame"', config_source)
        self.assertIn("no_candidate_score: 0.0", config_source)
        self.assertIn("full_frame_min_excess: 0.03", config_source)
        self.assertIn("full_frame_weight: 0.15", config_source)

    def test_llm_fusion_score_softly_scales_semantic_similar_candidates(self):
        self.assertAlmostEqual(semantic_similar_weight_from_fusion_score(0.0), 0.5)
        self.assertAlmostEqual(semantic_similar_weight_from_fusion_score(0.6), 0.8)
        self.assertAlmostEqual(semantic_similar_weight_from_fusion_score(1.0), 1.0)
        self.assertAlmostEqual(semantic_similar_weight_from_fusion_score(2.0), 1.0)
        self.assertAlmostEqual(semantic_similar_weight_from_fusion_score(-1.0), 0.5)


if __name__ == "__main__":
    unittest.main()
