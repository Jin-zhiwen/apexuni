import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class SdfMapUnknownPriorTest(unittest.TestCase):
    def test_unknown_voxels_start_at_occupancy_threshold(self):
        sdf_map_source = (
            REPO_ROOT / "src" / "planner" / "plan_env" / "src" / "sdf_map2d.cpp"
        ).read_text()

        input_depth = sdf_map_source.split("void SDFMap2D::inputDepthCloud2D", 1)[1]
        input_depth = input_depth.split("void SDFMap2D::setForceOccGrid", 1)[0]

        self.assertIn("if (md_->occupancy_buffer_[adr] < mp_->clamp_min_log_ - 1e-3)", input_depth)
        self.assertIn("md_->occupancy_buffer_[adr] = mp_->min_occupancy_log_;", input_depth)
        self.assertNotIn("md_->occupancy_buffer_[adr] = 0.0;", input_depth)

    def test_real_world_sensor_model_prefers_stable_obstacles(self):
        sdf_map_source = (
            REPO_ROOT / "src" / "planner" / "plan_env" / "src" / "sdf_map2d.cpp"
        ).read_text()
        launch_source = (
            REPO_ROOT
            / "src"
            / "planner"
            / "exploration_manager"
            / "launch"
            / "algorithm_traj.xml"
        ).read_text()

        self.assertIn('name="sdf_map/p_hit" value="0.90"', launch_source)
        self.assertIn('name="sdf_map/p_miss" value="0.48"', launch_source)
        self.assertIn('nh.param("sdf_map/p_hit", mp_->p_hit_, 0.90);', sdf_map_source)
        self.assertIn('nh.param("sdf_map/p_miss", mp_->p_miss_, 0.48);', sdf_map_source)


if __name__ == "__main__":
    unittest.main()
