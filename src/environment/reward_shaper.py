"""
reward_shaper.py — Modular Reward Engineering
Computes per-agent and global team rewards with full transparency.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np

from src.environment.robot import Robot, RobotState
from src.environment.order_manager import OrderManager


@dataclass
class RewardConfig:
    # Positive rewards
    delivery_success: float = 100.0    # was 10.0
    delivery_progress: float = 0.3     # was 1.0 — reduce shaping so delivery dominates
    charging_efficiency: float = 0.5

    # Negative rewards
    collision: float = -2.0             # was -5.0
    idle_penalty: float = -0.5          # was -0.1
    energy_waste: float = -0.2
    missed_deadline: float = -3.0
    staying_when_tasks: float = -0.05

    # Team vs individual blend
    team_alpha: float = 0.3            # was 0.5

    # Priority multiplier
    priority_multiplier: bool = True


class RewardShaper:
    """
    Computes rewards for all agents each timestep.
    Exposes detailed reward breakdown for debugging and curriculum tuning.
    """

    def __init__(self, config: Optional[RewardConfig] = None):
        self.config = config or RewardConfig()
        self._prev_distances: Dict[int, Optional[float]] = {}

    def reset(self, num_robots: int) -> None:
        self._prev_distances = {i: None for i in range(num_robots)}

    def compute_rewards(
        self,
        robots: list[Robot],
        events: dict,   # Events from this step: deliveries, collisions, etc.
        order_manager: OrderManager,
        current_step: int,
    ) -> Dict[int, float]:
        """
        Returns per-agent rewards for this timestep.
        events dict keys:
          - deliveries: list of (robot_id, order_id, priority)
          - collisions: list of robot_ids
          - missed_deadlines: list of order_ids
          - new_positions: dict robot_id -> Position (for progress shaping)
          - goal_positions: dict robot_id -> Position (current goal)
          - energy_waste: list of robot_ids (charged when not needed)
        """
        individual_rewards: Dict[int, float] = {r.robot_id: 0.0 for r in robots}
        reward_breakdown: Dict[int, dict] = {r.robot_id: {} for r in robots}

        cfg = self.config

        # --- Delivery rewards ---
        delivered_robots = set()
        for robot_id, order_id, priority in events.get("deliveries", []):
            mult = (priority if cfg.priority_multiplier else 1.0)
            r = cfg.delivery_success * mult
            individual_rewards[robot_id] += r
            reward_breakdown[robot_id]["delivery"] = r
            delivered_robots.add(robot_id)

        # --- Pick-up bonus (one-time, not per-step) ---
        for robot in robots:
            if (robot.state == RobotState.MOVING_TO_DELIVERY 
                and robot.carrying_item
                and robot.robot_id not in delivered_robots):
                # Only reward the step the state transitions to MOVING_TO_DELIVERY
                # Check prev_distances reset as proxy for just-picked
                if self._prev_distances.get(robot.robot_id) is None:
                    individual_rewards[robot.robot_id] += cfg.delivery_success * 0.3   

        # --- Collision penalties ---
        for robot_id in events.get("collisions", []):
            individual_rewards[robot_id] += cfg.collision
            reward_breakdown[robot_id]["collision"] = cfg.collision

        # --- Missed deadline penalties ---
        # Distributed to all robots equally (shared responsibility)
        missed = events.get("missed_deadlines", [])
        if missed:
            per_robot = cfg.missed_deadline * len(missed) / max(len(robots), 1)
            for r in robots:
                individual_rewards[r.robot_id] += per_robot
                reward_breakdown[r.robot_id]["deadline"] = per_robot

        # --- Progress shaping (potential-based) ---
        new_positions = events.get("new_positions", {})
        goal_positions = events.get("goal_positions", {})
        for robot in robots:
            rid = robot.robot_id
            if rid in goal_positions and goal_positions[rid] is not None:
                curr_pos = new_positions.get(rid, robot.position)
                goal = goal_positions[rid]
                curr_dist = abs(curr_pos.x - goal.x) + abs(curr_pos.y - goal.y)
                prev_dist = self._prev_distances.get(rid)

                if prev_dist is not None and curr_dist < prev_dist:
                    progress = cfg.delivery_progress * (prev_dist - curr_dist)
                    individual_rewards[rid] += progress
                    reward_breakdown[rid]["progress"] = progress

                self._prev_distances[rid] = curr_dist

        # --- Idle penalty ---
        for robot in robots:
            rid = robot.robot_id
            if (
                robot.state == RobotState.IDLE
                and order_manager.num_pending > 0
            ):
                individual_rewards[rid] += cfg.idle_penalty
                reward_breakdown[rid]["idle"] = cfg.idle_penalty

        # --- Energy waste penalty ---
        for robot_id in events.get("energy_waste", []):
            individual_rewards[robot_id] += cfg.energy_waste
            reward_breakdown[robot_id]["energy_waste"] = cfg.energy_waste

        # --- Global team reward ---
        team_reward = sum(individual_rewards.values()) / max(len(robots), 1)

        # --- Blend individual + team ---
        alpha = cfg.team_alpha
        final_rewards: Dict[int, float] = {}
        for r in robots:
            rid = r.robot_id
            blended = (1 - alpha) * individual_rewards[rid] + alpha * team_reward
            # Clip to reasonable range
            final_rewards[rid] = float(np.clip(blended, -50.0, 150.0))   # was (-20, 20)

        return final_rewards

    def compute_team_reward(self, individual_rewards: Dict[int, float]) -> float:
        """Average individual rewards as team metric."""
        if not individual_rewards:
            return 0.0
        return sum(individual_rewards.values()) / len(individual_rewards)
