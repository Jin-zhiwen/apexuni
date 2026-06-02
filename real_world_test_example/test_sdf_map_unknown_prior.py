import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class SdfMapUnknownPriorTest(unittest.TestCase):
    def test_unknown_voxels_start_from_neutral_log_odds(self):
        sdf_map_source = (
            REPO_ROOT / "src" / "planner" / "plan_env" / "src" / "sdf_map2d.cpp"
        ).read_text()

        input_depth = sdf_map_source.split("void SDFMap2D::inputDepthCloud2D", 1)[1]
        input_depth = input_depth.split("void SDFMap2D::setForceOccGrid", 1)[0]

        self.assertIn("if (md_->occupancy_buffer_[adr] < mp_->clamp_min_log_ - 1e-3)", input_depth)
        self.assertIn("md_->occupancy_buffer_[adr] = 0.0;", input_depth)
        self.assertNotIn("md_->occupancy_buffer_[adr] = mp_->min_occupancy_log_;", input_depth)


if __name__ == "__main__":
    unittest.main()
