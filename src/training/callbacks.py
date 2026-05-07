"""
callbacks.py — Custom RLlib Training Callbacks
Tracks warehouse-specific metrics: throughput, collision rate, battery efficiency.
"""

from __future__ import annotations
import numpy as np

try:
    from ray.rllib.algorithms.callbacks import DefaultCallbacks
    HAS_RAY = True
except ImportError:
    HAS_RAY = False
    DefaultCallbacks = object

# All agents receive identical info dicts from _build_infos().
# We pick robot_0 as the canonical agent to read env-level metrics from.
_CANONICAL_AGENT = "robot_0"


class WarehouseCallbacks(DefaultCallbacks if HAS_RAY else object):

    def on_episode_start(self, *, worker, base_env, policies, episode, **kwargs):
        episode.user_data["deliveries"] = 0
        episode.user_data["collisions"] = 0
        episode.user_data["throughput_scores"] = []

    def on_episode_step(self, *, worker, base_env, policies, episode, **kwargs):
        # FIX: must pass agent_id — last_info_for() with no arg returns None in Ray 2.x
        infos = episode.last_info_for(_CANONICAL_AGENT)
        if infos:
            episode.user_data["deliveries"] += infos.get("deliveries_this_step", 0)
            episode.user_data["collisions"] += infos.get("collisions_this_step", 0)
            episode.user_data["throughput_scores"].append(
                infos.get("throughput_score", 0.0)
            )

    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
        # FIX: must pass agent_id — same reason as above
        infos = episode.last_info_for(_CANONICAL_AGENT) or {}

        episode.custom_metrics["warehouse/throughput_score"] = infos.get("throughput_score", 0.0)
        episode.custom_metrics["warehouse/completion_rate"] = infos.get("completion_rate", 0.0)
        episode.custom_metrics["warehouse/total_deliveries"] = episode.user_data["deliveries"]
        episode.custom_metrics["warehouse/total_collisions"] = episode.user_data["collisions"]

        # These keys match get_metrics() in order_manager.py exactly
        episode.custom_metrics["warehouse/orders_completed"] = infos.get("completed", 0)
        episode.custom_metrics["warehouse/orders_failed"] = infos.get("failed", 0)
        episode.custom_metrics["warehouse/orders_spawned"] = infos.get("total_spawned", 0)

        steps = max(episode.length, 1)
        episode.custom_metrics["warehouse/collision_rate"] = (
            episode.user_data["collisions"] / steps
        )

    def on_train_result(self, *, algorithm, result, **kwargs):
        custom = result.get("custom_metrics", {})
        if "warehouse/throughput_score_mean" in custom:
            result["warehouse_throughput"] = custom["warehouse/throughput_score_mean"]
        if "warehouse/collision_rate_mean" in custom:
            result["warehouse_collision_rate"] = custom["warehouse/collision_rate_mean"]