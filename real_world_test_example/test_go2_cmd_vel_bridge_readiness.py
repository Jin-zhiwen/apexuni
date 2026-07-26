import pathlib
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


class Go2CmdVelBridgeReadinessTest(unittest.TestCase):
    def test_bridge_defaults_to_auto_stand_for_real_hardware(self) -> None:
        launch_source = (
            REPO_ROOT / "src/go2_cmd_vel_bridge/launch/cmd_vel_bridge.launch"
        ).read_text()
        source = (
            REPO_ROOT / "src/go2_cmd_vel_bridge/src/cmd_vel_to_go2_bridge_safe.cpp"
        ).read_text()

        self.assertIn('<arg name="auto_stand" default="true"', launch_source)
        self.assertIn('pnh_.param<bool>("auto_stand", auto_stand_, true);', source)

    def test_bridge_enters_balance_stand_before_motion_commands(self) -> None:
        source = (
            REPO_ROOT / "src/go2_cmd_vel_bridge/src/cmd_vel_to_go2_bridge_safe.cpp"
        ).read_text()

        stand_up_index = source.find("sport_client_->StandUp()")
        balance_index = source.find("sport_client_->BalanceStand()")
        move_index = source.find("sport_client_->Move")

        self.assertGreaterEqual(stand_up_index, 0)
        self.assertGreater(balance_index, stand_up_index)
        self.assertGreater(move_index, balance_index)

    def test_bridge_logs_sdk_return_codes_for_motion_commands(self) -> None:
        source = (
            REPO_ROOT / "src/go2_cmd_vel_bridge/src/cmd_vel_to_go2_bridge_safe.cpp"
        ).read_text()

        self.assertIn("Move returned", source)
        self.assertIn("StopMove returned", source)

    def test_bridge_default_forward_limit_has_catchup_margin(self) -> None:
        launch_source = (
            REPO_ROOT / "src/go2_cmd_vel_bridge/launch/cmd_vel_bridge.launch"
        ).read_text()

        self.assertIn('<arg name="max_vx" default="0.35"', launch_source)


if __name__ == "__main__":
    unittest.main()
