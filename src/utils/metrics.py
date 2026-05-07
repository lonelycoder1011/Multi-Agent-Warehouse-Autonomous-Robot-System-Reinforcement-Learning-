"""metrics.py — Performance Metrics Tracker"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from collections import deque


@dataclass
class EpisodeMetrics:
    episode: int
    total_steps: int
    total_reward: float
    deliveries: int
    collisions: int
    completion_rate: float
    avg_battery: float
    throughput: float
    algorithm: str = "MAPPO"


class MetricsTracker:
    def __init__(self, window_size: int = 100):
        self.window = window_size
        self.episodes: List[EpisodeMetrics] = []
        self._reward_window: deque = deque(maxlen=window_size)
        self._delivery_window: deque = deque(maxlen=window_size)
        self._collision_window: deque = deque(maxlen=window_size)

    def record(self, m: EpisodeMetrics) -> None:
        self.episodes.append(m)
        self._reward_window.append(m.total_reward)
        self._delivery_window.append(m.deliveries)
        self._collision_window.append(m.collisions)

    @property
    def mean_reward(self) -> float:
        return float(np.mean(self._reward_window)) if self._reward_window else 0.0

    @property
    def mean_deliveries(self) -> float:
        return float(np.mean(self._delivery_window)) if self._delivery_window else 0.0

    @property
    def mean_collisions(self) -> float:
        return float(np.mean(self._collision_window)) if self._collision_window else 0.0

    def summary(self) -> dict:
        return {
            "num_episodes": len(self.episodes),
            "mean_reward": round(self.mean_reward, 3),
            "mean_deliveries": round(self.mean_deliveries, 2),
            "mean_collisions": round(self.mean_collisions, 2),
            "best_reward": round(max((e.total_reward for e in self.episodes), default=0), 3),
        }
