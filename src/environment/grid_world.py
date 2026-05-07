"""
grid_world.py — Warehouse Grid Engine
Handles the physical layout: obstacles, shelves, charging stations, delivery zones.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Tuple, Set, Dict, Optional


class CellType(IntEnum):
    EMPTY = 0
    OBSTACLE = 1
    SHELF = 2
    CHARGING_STATION = 3
    DELIVERY_ZONE = 4
    ROBOT = 5


@dataclass
class Position:
    x: int
    y: int

    def __eq__(self, other) -> bool:
        return self.x == other.x and self.y == other.y

    def __hash__(self) -> int:
        return hash((self.x, self.y))

    def __add__(self, other: "Position") -> "Position":
        return Position(self.x + other.x, self.y + other.y)

    def manhattan_distance(self, other: "Position") -> int:
        return abs(self.x - other.x) + abs(self.y - other.y)

    def to_tuple(self) -> Tuple[int, int]:
        return (self.x, self.y)


# Cardinal directions: N, S, E, W
DIRECTION_DELTAS = {
    0: Position(0, -1),   # North
    1: Position(0, 1),    # South
    2: Position(1, 0),    # East
    3: Position(-1, 0),   # West
}


class WarehouseGrid:
    """
    Manages the warehouse physical layout.
    Provides spatial queries, pathfinding, and zone tracking.
    """

    def __init__(
        self,
        width: int = 20,
        height: int = 20,
        num_shelves: int = 40,
        num_charging_stations: int = 4,
        num_delivery_zones: int = 3,
        num_obstacles: int = 15,
        seed: Optional[int] = None,
    ):
        self.width = width
        self.height = height
        self.rng = np.random.default_rng(seed)

        # Grid layers
        self.static_grid = np.zeros((height, width), dtype=np.int32)
        self.dynamic_grid = np.zeros((height, width), dtype=np.int32)  # robots

        # Zone registries
        self.shelf_positions: List[Position] = []
        self.charging_positions: List[Position] = []
        self.delivery_positions: List[Position] = []
        self.obstacle_positions: Set[Position] = set()

        self._generate_layout(
            num_shelves, num_charging_stations, num_delivery_zones, num_obstacles
        )

    def _generate_layout(
        self,
        num_shelves: int,
        num_charging: int,
        num_delivery: int,
        num_obstacles: int,
    ) -> None:
        """Generate warehouse layout with zones placed strategically."""
        self.static_grid.fill(CellType.EMPTY)

        # Delivery zones — near bottom edge
        delivery_xs = np.linspace(2, self.width - 3, num_delivery, dtype=int)
        for x in delivery_xs:
            pos = Position(int(x), self.height - 2)
            self.static_grid[pos.y, pos.x] = CellType.DELIVERY_ZONE
            self.delivery_positions.append(pos)

        # Charging stations — corners and mid-edges
        charge_candidates = [
            Position(1, 1),
            Position(self.width - 2, 1),
            Position(1, self.height - 2),
            Position(self.width - 2, self.height - 2),
        ]
        for pos in charge_candidates[:num_charging]:
            self.static_grid[pos.y, pos.x] = CellType.CHARGING_STATION
            self.charging_positions.append(pos)

        # Shelves — in organized rows in upper region
        reserved = (
            {p.to_tuple() for p in self.delivery_positions}
            | {p.to_tuple() for p in self.charging_positions}
        )
        shelf_region = self._get_shelf_region(reserved)
        chosen = self.rng.choice(
            len(shelf_region), size=min(num_shelves, len(shelf_region)), replace=False
        )
        for idx in chosen:
            x, y = shelf_region[idx]
            pos = Position(x, y)
            self.static_grid[y, x] = CellType.SHELF
            self.shelf_positions.append(pos)
            reserved.add((x, y))

        # Obstacles — scattered in walkways
        empty_cells = [
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
            if (x, y) not in reserved
            and self.static_grid[y, x] == CellType.EMPTY
            and not self._is_border(x, y)
        ]
        obs_count = min(num_obstacles, len(empty_cells))
        obs_indices = self.rng.choice(len(empty_cells), size=obs_count, replace=False)
        for idx in obs_indices:
            x, y = empty_cells[idx]
            self.static_grid[y, x] = CellType.OBSTACLE
            self.obstacle_positions.add(Position(x, y))

    def _get_shelf_region(self, reserved: Set) -> List[Tuple[int, int]]:
        """Return valid shelf positions in organized rows."""
        cells = []
        for y in range(2, self.height - 4, 3):
            for x in range(2, self.width - 2, 2):
                if (x, y) not in reserved:
                    cells.append((x, y))
        return cells

    def _is_border(self, x: int, y: int) -> bool:
        return x == 0 or y == 0 or x == self.width - 1 or y == self.height - 1

    def is_walkable(self, pos: Position) -> bool:
        """Check if a position can be occupied by a robot."""
        if not self.in_bounds(pos):
            return False
        cell = self.static_grid[pos.y, pos.x]
        return cell in (CellType.EMPTY, CellType.SHELF, CellType.CHARGING_STATION, CellType.DELIVERY_ZONE)

    def in_bounds(self, pos: Position) -> bool:
        return 0 <= pos.x < self.width and 0 <= pos.y < self.height

    def get_local_view(self, pos: Position, view_size: int = 7) -> np.ndarray:
        """
        Extract NxN local observation centered on robot position.
        Pads with obstacles at boundaries.
        """
        half = view_size // 2
        view = np.full((view_size, view_size), CellType.OBSTACLE, dtype=np.int32)
        for dy in range(-half, half + 1):
            for dx in range(-half, half + 1):
                world_pos = Position(pos.x + dx, pos.y + dy)
                vy, vx = dy + half, dx + half
                if self.in_bounds(world_pos):
                    view[vy, vx] = self.static_grid[world_pos.y, world_pos.x]
                    if self.dynamic_grid[world_pos.y, world_pos.x] > 0:
                        view[vy, vx] = CellType.ROBOT
        return view

    def get_nearest_zone(
        self, pos: Position, zone_type: str
    ) -> Optional[Tuple[Position, int]]:
        """Find nearest zone of given type, returns (position, distance)."""
        zones = {
            "charging": self.charging_positions,
            "delivery": self.delivery_positions,
            "shelf": self.shelf_positions,
        }.get(zone_type, [])

        if not zones:
            return None

        best_pos = min(zones, key=lambda z: pos.manhattan_distance(z))
        return best_pos, pos.manhattan_distance(best_pos)

    def get_grid_as_array(self) -> np.ndarray:
        """Combined static + dynamic grid for rendering."""
        combined = self.static_grid.copy()
        robot_mask = self.dynamic_grid > 0
        combined[robot_mask] = CellType.ROBOT
        return combined

    def set_robot_position(self, robot_id: int, pos: Position) -> None:
        self.dynamic_grid[pos.y, pos.x] = robot_id + 1

    def clear_robot_position(self, pos: Position) -> None:
        self.dynamic_grid[pos.y, pos.x] = 0

    def get_free_spawn_positions(self, count: int) -> List[Position]:
        """Get random free positions for robot spawning."""
        candidates = [
            Position(x, y)
            for y in range(self.height)
            for x in range(self.width)
            if self.is_walkable(Position(x, y))
            and self.dynamic_grid[y, x] == 0
        ]
        if len(candidates) < count:
            raise ValueError(f"Not enough free positions to spawn {count} robots")
        indices = self.rng.choice(len(candidates), size=count, replace=False)
        return [candidates[i] for i in indices]

    def add_dynamic_obstacle(self, pos: Position) -> bool:
        """Add a dynamic obstacle (returns False if cell occupied)."""
        if not self.in_bounds(pos) or self.static_grid[pos.y, pos.x] != CellType.EMPTY:
            return False
        self.static_grid[pos.y, pos.x] = CellType.OBSTACLE
        self.obstacle_positions.add(pos)
        return True

    def remove_dynamic_obstacle(self, pos: Position) -> bool:
        if pos in self.obstacle_positions:
            self.static_grid[pos.y, pos.x] = CellType.EMPTY
            self.obstacle_positions.discard(pos)
            return True
        return False
