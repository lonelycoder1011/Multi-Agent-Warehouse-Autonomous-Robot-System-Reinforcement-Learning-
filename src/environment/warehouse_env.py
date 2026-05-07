"""
warehouse_env.py — Multi-Agent Warehouse Environment
RLlib-compatible MultiAgentEnv using Gymnasium spaces.
"""

from __future__ import annotations
from typing import Dict, Any, Optional, Tuple, List
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# RLlib MultiAgentEnv
try:
    if os.environ.get("WAREHOUSE_RL_DISABLE_RAY_ENV") == "1":
        raise ImportError("Ray environment base disabled")
    from ray.rllib.env.multi_agent_env import MultiAgentEnv
except Exception:
    # Fallback for dashboard/tests when Ray is unavailable or cannot initialize.
    class MultiAgentEnv:
        pass

from src.environment.grid_world import WarehouseGrid, Position, DIRECTION_DELTAS as ACTION_MOVE_DELTAS, CellType
from src.environment.robot import Robot, RobotState, Action, Order
from src.environment.order_manager import OrderManager
from src.environment.reward_shaper import RewardShaper, RewardConfig


class WarehouseEnv(MultiAgentEnv):
    """
    Multi-Agent Autonomous Warehouse Environment.

    Observation per agent:
      - Local 7x7 grid view (flattened): 49 dims
      - Robot state vector: 16 dims
      - Order queue obs (top 5 orders): 30 dims
      - Communication messages from other agents: num_robots * comm_dim
      Total (10 robots, comm_dim=16): 49 + 16 + 30 + 160 = 255 dims

    Actions (discrete): 8
      0=North, 1=South, 2=East, 3=West, 4=Pick, 5=Deliver, 6=Charge, 7=Stay

    Rewards: Shaped multi-objective (see reward_shaper.py)
    """

    metadata = {"render_modes": ["human", "rgb_array", "json"]}

    def __init__(self, config: Optional[dict] = None):
        super().__init__()
        cfg = config or {}

        # Environment parameters
        self.grid_width: int = cfg.get("grid_width", 20)
        self.grid_height: int = cfg.get("grid_height", 20)
        self.num_robots: int = cfg.get("num_robots", 10)
        self.max_steps: int = cfg.get("max_steps", 500)
        self.local_view_size: int = cfg.get("local_view_size", 7)
        self.comm_dim: int = cfg.get("comm_dim", 16)

        # Order settings
        self.orders_per_episode: int = cfg.get("orders_per_episode", 30)
        self.order_spawn_rate: float = cfg.get("order_spawn_rate", 0.1)
        self.use_deadlines: bool = cfg.get("use_deadlines", False)
        self.deadline_range: tuple = cfg.get("deadline_range", (50, 200))
        self.use_priorities: bool = cfg.get("use_priorities", False)

        # Grid settings
        self.num_shelves: int = cfg.get("num_shelves", 40)
        self.num_charging_stations: int = cfg.get("num_charging_stations", 4)
        self.num_delivery_zones: int = cfg.get("num_delivery_zones", 3)
        self.num_obstacles: int = cfg.get("num_obstacles", 15)
        self.dynamic_obstacles: bool = cfg.get("dynamic_obstacles", False)

        # Battery
        self.battery_capacity: float = cfg.get("battery_capacity", 100.0)
        self.use_battery: bool = cfg.get("battery_management", True)

        # Seed
        self._seed = cfg.get("seed", None)
        self.rng = np.random.default_rng(self._seed)

        # Agent IDs
        self.possible_agents = [f"robot_{i}" for i in range(self.num_robots)]
        self.agents = list(self.possible_agents)
        self._agent_ids = set(self.agents)

        # Compute observation dimension
        self._local_view_dim = self.local_view_size ** 2
        self._robot_state_dim = 16
        self._order_queue_dim = 30  # 5 orders * 6 features
        self._comm_obs_dim = self.num_robots * self.comm_dim
        self._obs_dim = (
            self._local_view_dim
            + self._robot_state_dim
            + self._order_queue_dim
            + self._comm_obs_dim
        )

        # Define spaces
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self._obs_dim,),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(len(Action))

        # Reward shaper
        reward_cfg = RewardConfig(
            delivery_success=cfg.get("reward_delivery", 10.0),
            collision=cfg.get("reward_collision", -5.0),
            idle_penalty=cfg.get("reward_idle", -0.1),
            team_alpha=cfg.get("team_alpha", 0.5),
        )
        self.reward_shaper = RewardShaper(reward_cfg)

        # Runtime state (initialized in reset)
        self.grid: Optional[WarehouseGrid] = None
        self.robots: Dict[str, Robot] = {}
        self.order_manager: Optional[OrderManager] = None
        self.current_step: int = 0
        self.comm_messages: np.ndarray = np.zeros(
            (self.num_robots, self.comm_dim), dtype=np.float32
        )

        # Episode metrics
        self._episode_rewards: Dict[str, float] = {}
        self._total_collisions: int = 0

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.current_step = 0
        self._total_collisions = 0
        self._episode_rewards = {agent: 0.0 for agent in self.possible_agents}
        self.agents = list(self.possible_agents)

        # Build grid
        grid_seed = int(self.rng.integers(1_000_000))
        self.grid = WarehouseGrid(
            width=self.grid_width,
            height=self.grid_height,
            num_shelves=self.num_shelves,
            num_charging_stations=self.num_charging_stations,
            num_delivery_zones=self.num_delivery_zones,
            num_obstacles=self.num_obstacles,
            seed=grid_seed,
        )

        # Spawn robots
        spawn_positions = self.grid.get_free_spawn_positions(self.num_robots)
        self.robots = {}
        for i, agent_id in enumerate(self.possible_agents):
            robot = Robot(
                robot_id=i,
                position=spawn_positions[i],
                battery_capacity=self.battery_capacity,
            )
            self.robots[agent_id] = robot
            self.grid.set_robot_position(i, spawn_positions[i])

        # Order manager
        self.order_manager = OrderManager(
            grid=self.grid,
            orders_per_episode=self.orders_per_episode,
            spawn_rate=self.order_spawn_rate,
            use_deadlines=self.use_deadlines,
            deadline_range=self.deadline_range,
            use_priorities=self.use_priorities,
            seed=int(self.rng.integers(1_000_000)),
        )
        self.order_manager.reset()

        # Reset communication
        self.comm_messages = np.zeros(
            (self.num_robots, self.comm_dim), dtype=np.float32
        )

        # Reset reward shaper
        self.reward_shaper.reset(self.num_robots)

        # Assign initial orders
        self._assign_available_orders()

        observations = {agent: self._get_obs(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(
        self, action_dict: Dict[str, int]
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Any],
    ]:
        self.current_step += 1
        events = {
            "deliveries": [],
            "collisions": [],
            "missed_deadlines": [],
            "new_positions": {},
            "goal_positions": {},
            "energy_waste": [],
        }

        # Process each agent's action
        for agent_id, action in action_dict.items():
            if agent_id not in self.robots:
                continue
            robot = self.robots[agent_id]
            self._process_action(robot, Action(action), events)

        # Update order manager (spawn new orders, check deadlines)
        new_orders = self.order_manager.step(self.current_step)
        missed = [
            o.order_id for o in self.order_manager.failed_orders
            if o.created_at == self.current_step - 1
        ]
        events["missed_deadlines"] = missed

        # Assign available orders to robots without one
        self._assign_available_orders()

        # Compute rewards
        robot_list = list(self.robots.values())
        rewards = self.reward_shaper.compute_rewards(
            robot_list, events, self.order_manager, self.current_step
        )
        # Map robot_id -> agent_id
        agent_rewards = {
            f"robot_{r.robot_id}": rewards[r.robot_id]
            for r in robot_list
        }

        for agent, r in agent_rewards.items():
            self._episode_rewards[agent] = self._episode_rewards.get(agent, 0.0) + r

        # Termination
        episode_done = (
            self.current_step >= self.max_steps
            or self.order_manager.throughput_score >= 1.0
        )

        terminateds = {agent: episode_done for agent in self.agents}
        terminateds["__all__"] = episode_done
        truncateds = {agent: False for agent in self.agents}
        truncateds["__all__"] = False

        observations = {agent: self._get_obs(agent) for agent in self.agents}
        infos = self._build_infos(events)

        if episode_done:
            self.agents = []

        return observations, agent_rewards, terminateds, truncateds, infos

    # ------------------------------------------------------------------
    # Action Processing
    # ------------------------------------------------------------------
    def _process_action(self, robot: Robot, action: Action, events: dict) -> None:
        if action in ACTION_MOVE_DELTAS:
            self._process_move(robot, action, events)
        elif action == Action.PICK:
            self._process_pick(robot, events)
        elif action == Action.DELIVER:
            self._process_deliver(robot, events)
        elif action == Action.CHARGE:
            self._process_charge(robot, events)
        elif action == Action.STAY:
            robot.step_idle()

    def _process_move(self, robot: Robot, action: Action, events: dict) -> None:
        delta = ACTION_MOVE_DELTAS[action]
        new_pos = robot.position + delta

        if not self.grid.is_walkable(new_pos):
            robot.register_collision()
            events["collisions"].append(robot.robot_id)
            self._total_collisions += 1
            robot.step_idle()
            return

        # Check for robot-robot collision
        if self.grid.dynamic_grid[new_pos.y, new_pos.x] > 0:
            robot.register_collision()
            events["collisions"].append(robot.robot_id)
            self._total_collisions += 1
            robot.step_idle()
            return

        # Move robot
        self.grid.clear_robot_position(robot.position)
        robot.step_move(new_pos)
        self.grid.set_robot_position(robot.robot_id, new_pos)
        events["new_positions"][robot.robot_id] = new_pos

        # Track goal for progress shaping
        if robot.assigned_order is not None:
            goal = (
                robot.assigned_order.delivery_position
                if robot.carrying_item
                else robot.assigned_order.shelf_position
            )
            events["goal_positions"][robot.robot_id] = goal

    def _process_pick(self, robot: Robot, events: dict) -> None:
        if (
            robot.assigned_order is not None
            and not robot.carrying_item
            and robot.position == robot.assigned_order.shelf_position
        ):
            robot.step_pick()
        else:
            robot.step_idle()

    def _process_deliver(self, robot: Robot, events: dict) -> None:
        order = robot.assigned_order
        if (
            order is not None
            and robot.carrying_item
            and robot.position == order.delivery_position
        ):
            robot.carrying_item = False
            robot.total_deliveries += 1
            robot.state = RobotState.IDLE
            self.order_manager.complete_order(order.order_id, self.current_step)
            robot.assigned_order = None
            events["deliveries"].append(
                (robot.robot_id, order.order_id, order.priority)
            )
            robot.drain_battery(robot.battery_drain_step)
        else:
            robot.step_idle()

    def _process_charge(self, robot: Robot, events: dict) -> None:
        cell = self.grid.static_grid[robot.position.y, robot.position.x]
        at_charger = cell == CellType.CHARGING_STATION

        if robot.battery > 80.0 and at_charger:
            events["energy_waste"].append(robot.robot_id)

        robot.step_charge(at_charger)

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------
    def _get_obs(self, agent_id: str) -> np.ndarray:
        robot = self.robots[agent_id]

        # 1. Local grid view
        local_view = self.grid.get_local_view(robot.position, self.local_view_size)
        local_view_norm = local_view.flatten().astype(np.float32) / 5.0

        # 2. Robot state vector
        state_vec = robot.get_state_vector()

        # 3. Order queue
        order_queue = self.order_manager.get_order_queue_obs(max_orders=5)

        # 4. Communication messages from all agents
        comm_flat = self.comm_messages.flatten()

        obs = np.concatenate([local_view_norm, state_vec, order_queue, comm_flat])
        return obs.astype(np.float32)

    def update_comm_messages(self, messages: Dict[str, np.ndarray]) -> None:
        """Called by training loop to update inter-agent messages."""
        for agent_id, msg in messages.items():
            robot = self.robots.get(agent_id)
            if robot is not None:
                self.comm_messages[robot.robot_id] = msg[:self.comm_dim]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _assign_available_orders(self) -> None:
        """Assign pending orders to any robot that doesn't have one."""
        for agent_id, robot in self.robots.items():
            if robot.assigned_order is None and not robot.needs_charging:
                order = self.order_manager.get_next_order()
                if order:
                    robot.assign_order(order)

    def _build_infos(self, events: dict) -> Dict[str, Any]:
        metrics = self.order_manager.get_metrics()
        base_info = {
            "step": self.current_step,
            "collisions_this_step": len(events["collisions"]),
            "total_collisions": self._total_collisions,
            "deliveries_this_step": len(events["deliveries"]),
            **metrics,
        }
        return {agent: base_info for agent in self.possible_agents}

    def get_episode_summary(self) -> dict:
        return {
            "total_steps": self.current_step,
            "throughput_score": self.order_manager.throughput_score if self.order_manager else 0,
            "completion_rate": self.order_manager.completion_rate if self.order_manager else 0,
            "total_collisions": self._total_collisions,
            "robot_stats": [r.get_stats() for r in self.robots.values()],
            "order_metrics": self.order_manager.get_metrics() if self.order_manager else {},
        }

    # RLlib required
    def observation_space_sample(self, agent_ids=None):
        return {a: self.observation_space.sample() for a in (agent_ids or self.agents)}

    def action_space_sample(self, agent_ids=None):
        return {a: self.action_space.sample() for a in (agent_ids or self.agents)}
