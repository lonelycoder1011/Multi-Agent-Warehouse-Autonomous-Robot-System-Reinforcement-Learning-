"""
curriculum_manager.py — Progressive Curriculum Learning
5-stage difficulty schedule with automatic promotion.
"""

from __future__ import annotations
from typing import List, Optional
from collections import deque
import yaml
import numpy as np


CURRICULUM_STAGES = [
    {
        "stage": 1,
        "name": "Basics",
        "num_robots": 5,
        "obstacle_density": 0.0,
        "orders_per_episode": 5,
        "use_deadlines": False,
        "dynamic_obstacles": False,
        "battery_management": False,
        "use_priorities": False,
        "num_obstacles": 0,
        "description": "5 robots, empty grid, simple orders",
    },
    {
        "stage": 2,
        "name": "Obstacles",
        "num_robots": 8,
        "obstacle_density": 0.05,
        "orders_per_episode": 10,
        "use_deadlines": False,
        "dynamic_obstacles": False,
        "battery_management": False,
        "use_priorities": False,
        "num_obstacles": 10,
        "description": "8 robots, sparse obstacles",
    },
    {
        "stage": 3,
        "name": "Battery & Density",
        "num_robots": 12,
        "obstacle_density": 0.1,
        "orders_per_episode": 20,
        "use_deadlines": False,
        "dynamic_obstacles": False,
        "battery_management": True,
        "use_priorities": False,
        "num_obstacles": 20,
        "description": "12 robots, battery management required",
    },
    {
        "stage": 4,
        "name": "Deadlines",
        "num_robots": 16,
        "obstacle_density": 0.12,
        "orders_per_episode": 25,
        "use_deadlines": True,
        "deadline_range": [50, 200],
        "dynamic_obstacles": True,
        "battery_management": True,
        "use_priorities": False,
        "num_obstacles": 24,
        "description": "16 robots, deadlines and dynamic obstacles",
    },
    {
        "stage": 5,
        "name": "Full Complexity",
        "num_robots": 20,
        "obstacle_density": 0.15,
        "orders_per_episode": 30,
        "use_deadlines": True,
        "deadline_range": [30, 150],
        "dynamic_obstacles": True,
        "battery_management": True,
        "use_priorities": True,
        "num_obstacles": 30,
        "description": "20 robots, full warehouse complexity",
    },
]


class CurriculumManager:
    """
    Manages curriculum progression.
    Tracks rolling success rate and promotes to next stage automatically.
    """

    def __init__(
        self,
        promotion_threshold: float = 0.75,
        evaluation_window: int = 20,
        config_path: Optional[str] = None,
        start_stage: int = 1,
    ):
        self.promotion_threshold = promotion_threshold
        self.evaluation_window = evaluation_window
        self.stages = CURRICULUM_STAGES

        if config_path:
            self._load_from_yaml(config_path)

        self.current_stage_idx = start_stage - 1
        self.success_history: deque = deque(maxlen=evaluation_window)
        self.stage_history: List[dict] = []
        self.total_episodes: int = 0
        self.promotions: int = 0

    def _load_from_yaml(self, path: str) -> None:
        with open(path) as f:
            cfg = yaml.safe_load(f)
        if "curriculum" in cfg and "stages" in cfg["curriculum"]:
            self.stages = cfg["curriculum"]["stages"]
            for stage in self.stages:
                if "order_deadlines" in stage and "use_deadlines" not in stage:
                    stage["use_deadlines"] = stage["order_deadlines"]
                if "priority_orders" in stage and "use_priorities" not in stage:
                    stage["use_priorities"] = stage["priority_orders"]
            self.promotion_threshold = cfg["curriculum"].get(
                "promotion_threshold", self.promotion_threshold
            )

    @property
    def current_stage(self) -> dict:
        return self.stages[self.current_stage_idx]

    @property
    def stage_number(self) -> int:
        return self.current_stage["stage"]

    @property
    def at_final_stage(self) -> bool:
        return self.current_stage_idx >= len(self.stages) - 1

    def get_env_config(self, base_config: dict) -> dict:
        """Return env config merged with current curriculum stage settings."""
        stage = self.current_stage
        config = {**base_config}
        config.update({
            "num_robots": stage["num_robots"],
            "num_obstacles": stage.get("num_obstacles", 15),
            "orders_per_episode": stage["orders_per_episode"],
            "use_deadlines": stage.get("use_deadlines", False),
            "dynamic_obstacles": stage.get("dynamic_obstacles", False),
            "battery_management": stage.get("battery_management", True),
            "use_priorities": stage.get("use_priorities", False),
        })
        if "deadline_range" in stage:
            config["deadline_range"] = stage["deadline_range"]
        return config

    def record_episode(self, throughput_score: float) -> bool:
        """
        Record episode outcome. Returns True if promotion occurred.
        throughput_score: fraction of orders completed (0-1).
        """
        success = throughput_score >= self.promotion_threshold
        self.success_history.append(float(success))
        self.total_episodes += 1

        self.stage_history.append({
            "episode": self.total_episodes,
            "stage": self.stage_number,
            "throughput": throughput_score,
            "success": success,
        })

        # Check promotion
        if (
            not self.at_final_stage
            and len(self.success_history) >= self.evaluation_window
        ):
            rolling_rate = np.mean(list(self.success_history))
            if rolling_rate >= self.promotion_threshold:
                self._promote()
                return True

        return False

    def _promote(self) -> None:
        """Advance to next curriculum stage."""
        old_stage = self.stage_number
        self.current_stage_idx = min(self.current_stage_idx + 1, len(self.stages) - 1)
        self.success_history.clear()
        self.promotions += 1
        print(
            f"[Curriculum] Promoted: Stage {old_stage} → Stage {self.stage_number} "
            f"({self.current_stage['name']}) after {self.total_episodes} episodes"
        )

    def get_rolling_success_rate(self) -> float:
        if not self.success_history:
            return 0.0
        return float(np.mean(list(self.success_history)))

    def get_status(self) -> dict:
        return {
            "current_stage": self.stage_number,
            "stage_name": self.current_stage["name"],
            "total_episodes": self.total_episodes,
            "promotions": self.promotions,
            "rolling_success_rate": self.get_rolling_success_rate(),
            "promotion_threshold": self.promotion_threshold,
            "at_final_stage": self.at_final_stage,
            "num_robots": self.current_stage["num_robots"],
        }
