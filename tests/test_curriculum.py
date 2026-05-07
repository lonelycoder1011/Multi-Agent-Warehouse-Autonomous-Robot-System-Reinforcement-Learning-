"""
test_curriculum.py — Curriculum Manager Tests
"""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from curriculum.curriculum_manager import CurriculumManager


class TestCurriculumManager:
    def setup_method(self):
        self.cm = CurriculumManager(
            promotion_threshold=0.75,
            evaluation_window=5,  # Small window for fast testing
        )

    def test_starts_at_stage_1(self):
        assert self.cm.stage_number == 1

    def test_stage_has_required_keys(self):
        stage = self.cm.current_stage
        for key in ["stage", "name", "num_robots", "orders_per_episode"]:
            assert key in stage, f"Missing key: {key}"

    def test_no_promotion_on_poor_performance(self):
        for _ in range(20):
            self.cm.record_episode(0.3)
        assert self.cm.stage_number == 1

    def test_promotion_on_good_performance(self):
        for _ in range(10):
            promoted = self.cm.record_episode(0.9)
        assert self.cm.stage_number >= 2 or self.cm.promotions > 0

    def test_rolling_success_rate(self):
        self.cm.record_episode(1.0)
        self.cm.record_episode(1.0)
        self.cm.record_episode(0.0)
        rate = self.cm.get_rolling_success_rate()
        assert 0.0 <= rate <= 1.0

    def test_get_env_config(self):
        base_config = {"grid_width": 20, "grid_height": 20}
        env_config = self.cm.get_env_config(base_config)
        assert "num_robots" in env_config
        assert "orders_per_episode" in env_config
        assert env_config["num_robots"] == self.cm.current_stage["num_robots"]

    def test_stage_5_is_final(self):
        # Force to last stage
        self.cm.current_stage_idx = 4
        assert self.cm.at_final_stage

    def test_no_promotion_beyond_final(self):
        self.cm.current_stage_idx = 4
        for _ in range(20):
            self.cm.record_episode(1.0)
        assert self.cm.current_stage_idx == 4

    def test_status_dict_keys(self):
        status = self.cm.get_status()
        for key in ["current_stage", "stage_name", "total_episodes",
                    "rolling_success_rate", "at_final_stage"]:
            assert key in status


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
