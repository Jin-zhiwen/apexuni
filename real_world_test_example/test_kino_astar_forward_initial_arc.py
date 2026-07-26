import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class KinoAstarForwardInitialArcTest(unittest.TestCase):
    def test_initial_arc_sampler_does_not_generate_reverse_arcs(self):
        source = (
            REPO_ROOT
            / "src"
            / "planner"
            / "path_searching"
            / "src"
            / "kino_astar.cpp"
        ).read_text()

        self.assertIn("sampleInitialArcInputs(", source)
        self.assertNotIn("arc = -base_arc", source)
        self.assertNotIn("arc >= -2.0 * base_arc", source)
        self.assertIn("ctrl_input << steer, arc, 0.0;", source)


if __name__ == "__main__":
    unittest.main()
