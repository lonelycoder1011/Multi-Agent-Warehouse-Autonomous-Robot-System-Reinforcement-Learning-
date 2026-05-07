"""
order_manager.py — Dynamic Order Queue
Generates, tracks, and manages orders throughout an episode.
"""

from __future__ import annotations
from typing import List, Optional, Dict, Set
from collections import deque
import numpy as np

from src.environment.grid_world import Position, WarehouseGrid
from src.environment.robot import Order


class OrderManager:
    """
    Manages the warehouse order queue.
    Handles generation, assignment, completion tracking, and metrics.
    """

    def __init__(
        self,
        grid: WarehouseGrid,
        orders_per_episode: int = 30,
        spawn_rate: float = 0.1,
        use_deadlines: bool = False,
        deadline_range: tuple = (50, 200),
        use_priorities: bool = False,
        seed: Optional[int] = None,
    ):
        self.grid = grid
        self.orders_per_episode = orders_per_episode
        self.spawn_rate = spawn_rate
        self.use_deadlines = use_deadlines
        self.deadline_range = deadline_range
        self.use_priorities = use_priorities
        self.rng = np.random.default_rng(seed)

        self._next_order_id: int = 0
        self.pending_orders: deque[Order] = deque()
        self.assigned_orders: Dict[int, Order] = {}   # order_id -> Order
        self.completed_orders: List[Order] = []
        self.failed_orders: List[Order] = []           # deadline missed
        self.total_spawned: int = 0

    def reset(self) -> None:
        self._next_order_id = 0
        self.pending_orders.clear()
        self.assigned_orders.clear()
        self.completed_orders.clear()
        self.failed_orders.clear()
        self.total_spawned = 0

        # Pre-spawn initial batch (25% of episode orders)
        initial = max(1, self.orders_per_episode // 4)
        for _ in range(initial):
            order = self._generate_order(current_step=0)
            if order:
                self.pending_orders.append(order)

    def step(self, current_step: int) -> List[Order]:
        """
        Called every environment step.
        Spawns new orders, checks deadlines.
        Returns list of newly spawned orders.
        """
        new_orders = []

        # Spawn new orders stochastically
        remaining = self.orders_per_episode - self.total_spawned
        if remaining > 0 and self.rng.random() < self.spawn_rate:
            order = self._generate_order(current_step)
            if order:
                self.pending_orders.append(order)
                new_orders.append(order)

        # Check for expired orders
        if self.use_deadlines:
            expired = [
                oid for oid, o in self.assigned_orders.items()
                if o.deadline is not None and current_step > o.deadline
            ]
            for oid in expired:
                order = self.assigned_orders.pop(oid)
                self.failed_orders.append(order)

        return new_orders

    def _generate_order(self, current_step: int) -> Optional[Order]:
        if not self.grid.shelf_positions or not self.grid.delivery_positions:
            return None

        shelf_idx = self.rng.integers(len(self.grid.shelf_positions))
        delivery_idx = self.rng.integers(len(self.grid.delivery_positions))

        deadline = None
        if self.use_deadlines:
            low, high = self.deadline_range
            deadline = current_step + self.rng.integers(low, high)

        priority = 1
        if self.use_priorities:
            priority = int(self.rng.choice([1, 2, 3], p=[0.6, 0.3, 0.1]))

        order = Order(
            order_id=self._next_order_id,
            shelf_position=self.grid.shelf_positions[shelf_idx],
            delivery_position=self.grid.delivery_positions[delivery_idx],
            priority=priority,
            deadline=deadline,
            created_at=current_step,
        )
        self._next_order_id += 1
        self.total_spawned += 1
        return order

    def get_next_order(self) -> Optional[Order]:
        """Pop highest-priority order from queue."""
        if not self.pending_orders:
            return None
        # Sort by priority (higher = first), then by age
        best_idx = max(
            range(len(self.pending_orders)),
            key=lambda i: (self.pending_orders[i].priority, -self.pending_orders[i].order_id)
        )
        order = self.pending_orders[best_idx]
        del self.pending_orders[best_idx]  # type: ignore[call-overload]
        self.assigned_orders[order.order_id] = order
        return order

    def complete_order(self, order_id: int, current_step: int) -> Optional[Order]:
        """Mark order as completed."""
        order = self.assigned_orders.pop(order_id, None)
        if order:
            order.completed_at = current_step
            self.completed_orders.append(order)
        return order

    def release_order(self, order_id: int) -> None:
        """Return assigned order back to queue (robot dropped/reassigned)."""
        order = self.assigned_orders.pop(order_id, None)
        if order:
            self.pending_orders.appendleft(order)

    @property
    def num_pending(self) -> int:
        return len(self.pending_orders)

    @property
    def num_assigned(self) -> int:
        return len(self.assigned_orders)

    @property
    def num_completed(self) -> int:
        return len(self.completed_orders)

    @property
    def num_failed(self) -> int:
        return len(self.failed_orders)

    @property
    def completion_rate(self) -> float:
        total = self.num_completed + self.num_failed
        if total == 0:
            return 0.0
        return self.num_completed / total

    @property
    def throughput_score(self) -> float:
        """Normalized throughput: completed / total spawned."""
        if self.total_spawned == 0:
            return 0.0
        return self.num_completed / self.total_spawned

    def get_order_queue_obs(self, max_orders: int = 5) -> np.ndarray:
        """
        Returns a flat observation of top N pending orders.
        Each order: [shelf_x, shelf_y, delivery_x, delivery_y, priority, deadline_norm]
        """
        obs = []
        orders = sorted(
            list(self.pending_orders),
            key=lambda o: (-o.priority, o.order_id)
        )[:max_orders]

        for o in orders:
            obs.extend([
                o.shelf_position.x / self.grid.width,
                o.shelf_position.y / self.grid.height,
                o.delivery_position.x / self.grid.width,
                o.delivery_position.y / self.grid.height,
                o.priority / 3.0,
                1.0 if o.deadline is None else min(o.deadline, 500) / 500.0,
            ])

        # Pad to fixed size
        target_len = max_orders * 6
        while len(obs) < target_len:
            obs.extend([0.0] * 6)

        return np.array(obs[:target_len], dtype=np.float32)

    def get_metrics(self) -> dict:
        return {
            "total_spawned": self.total_spawned,
            "completed": self.num_completed,
            "failed": self.num_failed,
            "pending": self.num_pending,
            "assigned": self.num_assigned,
            "completion_rate": self.completion_rate,
            "throughput_score": self.throughput_score,
        }
