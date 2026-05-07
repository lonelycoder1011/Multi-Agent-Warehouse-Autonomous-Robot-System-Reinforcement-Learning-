"""
test_environment.py — Warehouse Environment Test Suite
Tests reset, step, spaces, reward bounds, collision detection.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from environment.warehouse_env import WarehouseEnv
from environment.grid_world import WarehouseGrid, Position, CellType
from environment.robot import Robot, RobotState, Action
from environment.order_manager import OrderManager
from environment.reward_shaper import RewardShaper, RewardConfig


# -----------------------------------------------------------------------
# Grid Tests
# -----------------------------------------------------------------------
class TestWarehouseGrid:
    def setup_method(self):
        self.grid = WarehouseGrid(width=20, height=20, seed=42)

    def test_grid_dimensions(self):
        assert self.grid.width == 20
        assert self.grid.height == 20
        assert self.grid.static_grid.shape == (20, 20)

    def test_zones_populated(self):
        assert len(self.grid.shelf_positions) > 0
        assert len(self.grid.charging_positions) > 0
        assert len(self.grid.delivery_positions) > 0

    def test_zone_positions_in_bounds(self):
        for pos in self.grid.shelf_positions:
            assert self.grid.in_bounds(pos), f"Shelf at {pos} out of bounds"
        for pos in self.grid.charging_positions:
            assert self.grid.in_bounds(pos)

    def test_walkable_cells(self):
        # Empty cells should be walkable
        for y in range(self.grid.height):
            for x in range(self.grid.width):
                pos = Position(x, y)
                cell = self.grid.static_grid[y, x]
                if cell == CellType.OBSTACLE:
                    assert not self.grid.is_walkable(pos)

    def test_local_view_shape(self):
        pos = Position(10, 10)
        view = self.grid.get_local_view(pos, view_size=7)
        assert view.shape == (7, 7)

    def test_local_view_border_padding(self):
        # Corner position — should be padded with obstacles
        pos = Position(0, 0)
        view = self.grid.get_local_view(pos, view_size=7)
        assert view.shape == (7, 7)

    def test_robot_position_tracking(self):
        pos = Position(10, 10)
        if self.grid.is_walkable(pos):
            self.grid.set_robot_position(0, pos)
            assert self.grid.dynamic_grid[pos.y, pos.x] == 1
            self.grid.clear_robot_position(pos)
            assert self.grid.dynamic_grid[pos.y, pos.x] == 0

    def test_nearest_zone(self):
        pos = Position(5, 5)
        result = self.grid.get_nearest_zone(pos, "charging")
        assert result is not None
        nearest_pos, dist = result
        assert dist >= 0
        assert self.grid.in_bounds(nearest_pos)


# -----------------------------------------------------------------------
# Robot Tests
# -----------------------------------------------------------------------
class TestRobot:
    def setup_method(self):
        self.robot = Robot(robot_id=0, position=Position(5, 5))

    def test_initial_state(self):
        assert self.robot.state == RobotState.IDLE
        assert self.robot.battery == 100.0
        assert not self.robot.carrying_item
        assert self.robot.assigned_order is None

    def test_battery_drain(self):
        initial_battery = self.robot.battery
        self.robot.drain_battery(10.0)
        assert self.robot.battery == initial_battery - 10.0

    def test_battery_floor(self):
        self.robot.drain_battery(200.0)
        assert self.robot.battery == 0.0

    def test_battery_fraction(self):
        self.robot.battery = 50.0
        assert self.robot.battery_fraction == 0.5

    def test_needs_charging(self):
        self.robot.battery = 15.0
        assert self.robot.needs_charging
        self.robot.battery = 50.0
        assert not self.robot.needs_charging

    def test_state_vector_shape(self):
        state_vec = self.robot.get_state_vector()
        assert isinstance(state_vec, np.ndarray)
        assert state_vec.dtype == np.float32
        assert len(state_vec) == 16

    def test_state_vector_bounds(self):
        state_vec = self.robot.get_state_vector()
        # Normalized values should be reasonable
        assert np.all(state_vec >= 0.0), f"Negative state values: {state_vec}"

    def test_charge_at_station(self):
        self.robot.battery = 50.0
        charged = self.robot.step_charge(at_charging_station=True)
        assert charged > 0
        assert self.robot.battery > 50.0

    def test_no_charge_without_station(self):
        self.robot.battery = 50.0
        charged = self.robot.step_charge(at_charging_station=False)
        assert charged == 0.0
        assert self.robot.battery == 50.0


# -----------------------------------------------------------------------
# Environment Tests
# -----------------------------------------------------------------------
class TestWarehouseEnv:
    def setup_method(self):
        self.env = WarehouseEnv({
            "num_robots": 5,
            "grid_width": 15,
            "grid_height": 15,
            "max_steps": 50,
            "orders_per_episode": 10,
            "num_shelves": 20,
            "num_obstacles": 5,
            "seed": 42,
        })

    def test_reset_returns_correct_types(self):
        obs, infos = self.env.reset(seed=0)
        assert isinstance(obs, dict)
        assert isinstance(infos, dict)

    def test_reset_observation_shapes(self):
        obs, _ = self.env.reset(seed=0)
        for agent_id, o in obs.items():
            assert isinstance(o, np.ndarray), f"Obs for {agent_id} is not ndarray"
            assert o.shape == (self.env._obs_dim,), \
                f"Wrong shape for {agent_id}: {o.shape} != {(self.env._obs_dim,)}"

    def test_reset_all_agents_present(self):
        obs, _ = self.env.reset(seed=0)
        for agent_id in self.env.possible_agents:
            assert agent_id in obs

    def test_step_with_random_actions(self):
        self.env.reset(seed=0)
        actions = {agent: self.env.action_space.sample() for agent in self.env.agents}
        obs, rewards, terminateds, truncateds, infos = self.env.step(actions)

        assert isinstance(rewards, dict)
        for agent_id, r in rewards.items():
            assert isinstance(r, float), f"Reward for {agent_id} is not float"
            assert -100.0 <= r <= 100.0, f"Reward out of bounds: {r}"

    def test_step_termination(self):
        self.env.reset(seed=0)
        for _ in range(self.env.max_steps):
            if not self.env.agents:
                break
            actions = {a: self.env.action_space.sample() for a in self.env.agents}
            _, _, terminateds, _, _ = self.env.step(actions)
            if terminateds.get("__all__", False):
                break
        assert "__all__" in terminateds

    def test_observation_within_bounds(self):
        obs, _ = self.env.reset(seed=0)
        for agent_id, o in obs.items():
            assert not np.any(np.isnan(o)), f"NaN in obs for {agent_id}"
            assert not np.any(np.isinf(o)), f"Inf in obs for {agent_id}"

    def test_action_space_discrete(self):
        from gymnasium import spaces
        assert isinstance(self.env.action_space, spaces.Discrete)
        assert self.env.action_space.n == 8

    def test_observation_space_box(self):
        from gymnasium import spaces
        assert isinstance(self.env.observation_space, spaces.Box)

    def test_episode_summary(self):
        self.env.reset(seed=0)
        # Run partial episode
        for _ in range(10):
            if not self.env.agents:
                break
            actions = {a: self.env.action_space.sample() for a in self.env.agents}
            self.env.step(actions)

        summary = self.env.get_episode_summary()
        assert "total_steps" in summary
        assert "throughput_score" in summary
        assert "total_collisions" in summary


# -----------------------------------------------------------------------
# Order Manager Tests
# -----------------------------------------------------------------------
class TestOrderManager:
    def setup_method(self):
        self.grid = WarehouseGrid(width=15, height=15, seed=42)
        self.om = OrderManager(
            grid=self.grid,
            orders_per_episode=20,
            spawn_rate=0.5,
            seed=42,
        )
        self.om.reset()

    def test_initial_orders(self):
        assert self.om.total_spawned > 0

    def test_get_next_order(self):
        order = self.om.get_next_order()
        if order:
            assert order.order_id >= 0
            assert self.grid.in_bounds(order.shelf_position)
            assert self.grid.in_bounds(order.delivery_position)

    def test_complete_order(self):
        initial = self.om.num_pending
        order = self.om.get_next_order()
        if order:
            self.om.complete_order(order.order_id, current_step=10)
            assert self.om.num_completed == 1

    def test_throughput_score(self):
        score = self.om.throughput_score
        assert 0.0 <= score <= 1.0

    def test_order_queue_obs_shape(self):
        obs = self.om.get_order_queue_obs(max_orders=5)
        assert obs.shape == (30,)
        assert obs.dtype == np.float32


# -----------------------------------------------------------------------
# Reward Shaper Tests
# -----------------------------------------------------------------------
class TestRewardShaper:
    def setup_method(self):
        self.shaper = RewardShaper()
        self.grid = WarehouseGrid(15, 15, seed=42)
        self.om = OrderManager(self.grid, seed=42)
        self.om.reset()

        self.robots = [
            Robot(robot_id=i, position=Position(i * 2, i))
            for i in range(5)
        ]
        self.shaper.reset(5)

    def test_delivery_reward(self):
        events = {
            "deliveries": [(0, 1, 1)],  # robot 0, order 1, priority 1
            "collisions": [],
            "missed_deadlines": [],
            "new_positions": {},
            "goal_positions": {},
            "energy_waste": [],
        }
        rewards = self.shaper.compute_rewards(self.robots, events, self.om, 10)
        assert rewards[0] > 0, "Delivery should give positive reward"

    def test_collision_penalty(self):
        events = {
            "deliveries": [],
            "collisions": [0],
            "missed_deadlines": [],
            "new_positions": {},
            "goal_positions": {},
            "energy_waste": [],
        }
        rewards = self.shaper.compute_rewards(self.robots, events, self.om, 10)
        # Due to team blending, individual penalty may be diluted
        assert rewards[0] < 0, "Collision should give negative reward"

    def test_reward_bounds(self):
        events = {
            "deliveries": [(i, i, 3) for i in range(5)],
            "collisions": list(range(5)),
            "missed_deadlines": [1, 2, 3],
            "new_positions": {},
            "goal_positions": {},
            "energy_waste": [],
        }
        rewards = self.shaper.compute_rewards(self.robots, events, self.om, 10)
        for rid, r in rewards.items():
            assert -20.0 <= r <= 20.0, f"Reward {r} for robot {rid} out of [-20, 20]"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
