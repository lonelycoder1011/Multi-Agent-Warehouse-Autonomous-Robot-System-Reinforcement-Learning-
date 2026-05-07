"""
robot.py — Robot State Machine
Each robot has position, battery, task assignment, and a state machine.
"""

from __future__ import annotations
from enum import IntEnum
from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np

from src.environment.grid_world import Position


class RobotState(IntEnum):
    IDLE = 0
    MOVING_TO_SHELF = 1
    PICKING = 2
    MOVING_TO_DELIVERY = 3
    DELIVERING = 4
    MOVING_TO_CHARGE = 5
    CHARGING = 6


class Action(IntEnum):
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3
    PICK = 4
    DELIVER = 5
    CHARGE = 6
    STAY = 7


ACTION_MOVE_DELTAS = {
    Action.NORTH: Position(0, -1),
    Action.SOUTH: Position(0, 1),
    Action.EAST: Position(1, 0),
    Action.WEST: Position(-1, 0),
}


@dataclass
class Order:
    order_id: int
    shelf_position: Position
    delivery_position: Position
    priority: int = 1               # 1 = normal, 2 = urgent, 3 = critical
    deadline: Optional[int] = None  # timestep deadline (None = no deadline)
    created_at: int = 0
    completed_at: Optional[int] = None

    @property
    def is_expired(self) -> bool:
        return self.deadline is not None and self.completed_at is None

    def time_remaining(self, current_step: int) -> Optional[int]:
        if self.deadline is None:
            return None
        return self.deadline - current_step


@dataclass
class Robot:
    robot_id: int
    position: Position
    battery_capacity: float = 100.0
    battery_drain_step: float = 0.3
    battery_drain_move: float = 0.5
    battery_charge_rate: float = 10.0
    low_battery_threshold: float = 20.0

    # Runtime state
    battery: float = field(init=False)
    state: RobotState = field(default=RobotState.IDLE, init=False)
    assigned_order: Optional[Order] = field(default=None, init=False)
    carrying_item: bool = field(default=False, init=False)
    total_deliveries: int = field(default=0, init=False)
    total_collisions: int = field(default=0, init=False)
    steps_idle: int = field(default=0, init=False)
    steps_moving: int = field(default=0, init=False)
    total_energy_used: float = field(default=0.0, init=False)

    def __post_init__(self):
        self.battery = self.battery_capacity

    @property
    def battery_fraction(self) -> float:
        return self.battery / self.battery_capacity

    @property
    def needs_charging(self) -> bool:
        return self.battery <= self.low_battery_threshold

    @property
    def is_charging(self) -> bool:
        return self.state == RobotState.CHARGING

    @property
    def has_task(self) -> bool:
        return self.assigned_order is not None

    def assign_order(self, order: Order) -> None:
        self.assigned_order = order
        self.state = RobotState.MOVING_TO_SHELF

    def unassign_order(self) -> Optional[Order]:
        order = self.assigned_order
        self.assigned_order = None
        self.carrying_item = False
        self.state = RobotState.IDLE
        return order

    def step_idle(self) -> None:
        self.drain_battery(self.battery_drain_step)
        self.steps_idle += 1

    def step_move(self, new_position: Position) -> None:
        self.position = new_position
        self.drain_battery(self.battery_drain_step + self.battery_drain_move)
        self.steps_moving += 1

    def step_pick(self) -> bool:
        """Attempt to pick item. Returns True if successful."""
        if (
            self.assigned_order is not None
            and not self.carrying_item
            and self.position == self.assigned_order.shelf_position
        ):
            self.carrying_item = True
            self.state = RobotState.MOVING_TO_DELIVERY
            self.drain_battery(self.battery_drain_step)
            return True
        return False

    def step_deliver(self, current_step: int) -> bool:
        """Attempt delivery. Returns True if successful."""
        if (
            self.carrying_item
            and self.assigned_order is not None
            and self.position == self.assigned_order.delivery_position  # FIXED: was shelf_position
        ):
            self.assigned_order.completed_at = current_step
            self.carrying_item = False
            self.total_deliveries += 1
            self.assigned_order = None
            self.state = RobotState.IDLE
            self.drain_battery(self.battery_drain_step)
            return True
        return False

    def step_charge(self, at_charging_station: bool) -> float:
        """Charge battery. Returns amount charged."""
        if not at_charging_station:
            return 0.0
        self.state = RobotState.CHARGING
        charge_amount = min(self.battery_charge_rate, self.battery_capacity - self.battery)
        self.battery += charge_amount
        return charge_amount

    def drain_battery(self, amount: float) -> None:
        self.battery = max(0.0, self.battery - amount)
        self.total_energy_used += amount

    def register_collision(self) -> None:
        self.total_collisions += 1

    def get_state_vector(self) -> np.ndarray:
        """
        Compact state vector for RL observation.
        [x_norm, y_norm, battery_norm, state_onehot(7), carrying, has_order,
         task_x_norm, task_y_norm, deadline_norm, priority]
        Total: 2 + 1 + 7 + 1 + 1 + 2 + 1 + 1 = 16 dims
        """
        grid_norm = 20.0  # assume 20x20, override in env

        # Basic position + battery
        state_vec = [
            self.position.x / grid_norm,
            self.position.y / grid_norm,
            self.battery_fraction,
        ]

        # One-hot state
        state_onehot = [0.0] * len(RobotState)
        state_onehot[int(self.state)] = 1.0
        state_vec.extend(state_onehot)

        # Carrying + has order
        state_vec.append(float(self.carrying_item))
        state_vec.append(float(self.has_task))

        # Task info
        if self.assigned_order is not None:
            target = (
                self.assigned_order.delivery_position
                if self.carrying_item
                else self.assigned_order.shelf_position
            )

            # Relative direction to goal (critical for navigation learning)
            state_vec.append((target.x - self.position.x) / grid_norm)
            state_vec.append((target.y - self.position.y) / grid_norm)
            deadline = (
                min(self.assigned_order.deadline or 500, 500) / 500.0
                if self.assigned_order.deadline else 1.0
            )
            state_vec.append(deadline)
            state_vec.append(self.assigned_order.priority / 3.0)
        else:
            state_vec.extend([0.0, 0.0, 1.0, 0.0])

        return np.array(state_vec, dtype=np.float32)

    def get_stats(self) -> dict:
        return {
            "robot_id": self.robot_id,
            "total_deliveries": self.total_deliveries,
            "total_collisions": self.total_collisions,
            "steps_idle": self.steps_idle,
            "steps_moving": self.steps_moving,
            "total_energy_used": self.total_energy_used,
            "battery": self.battery,
        }