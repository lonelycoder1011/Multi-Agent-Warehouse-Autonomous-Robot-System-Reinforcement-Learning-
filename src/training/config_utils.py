"""Shared configuration helpers for training, evaluation, and dashboard runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_train_config(path: str | Path) -> dict[str, Any]:
    """Load either mappo_config.yaml or training_config.yaml into one flat dict."""
    raw = load_yaml(path)

    if "mappo" in raw:
        cfg = dict(raw.get("mappo") or {})
        cfg.update({k: v for k, v in raw.items() if k != "mappo"})
        return cfg

    training = raw.get("training", {})
    rollout = raw.get("rollout", {})
    optimization = raw.get("optimization", {})
    logging = raw.get("logging", {})

    if training or rollout or optimization or logging:
        return {
            "algorithm": training.get("algorithm", "MAPPO"),
            "num_iterations": training.get("num_iterations", 200),
            "checkpoint_freq": training.get("checkpoint_freq", 25),
            "seed": training.get("seed", 42),
            "num_rollout_workers": rollout.get("num_workers", 2),
            "num_envs_per_worker": rollout.get("num_envs_per_worker", 1),
            "rollout_fragment_length": rollout.get("rollout_fragment_length", "auto"),
            "train_batch_size": rollout.get("batch_size", 2048),
            "sgd_minibatch_size": rollout.get("mini_batch_size", 256),
            "lr": optimization.get("lr", 3e-4),
            "gamma": optimization.get("gamma", 0.99),
            "lambda_gae": optimization.get("gae_lambda", 0.95),
            "clip_param": optimization.get("clip_param", 0.2),
            "entropy_coeff": optimization.get("entropy_coeff", 0.01),
            "vf_loss_coeff": optimization.get("vf_loss_coeff", 0.5),
            "grad_clip": optimization.get("max_grad_norm", 10.0),
            "num_sgd_iter": optimization.get("num_sgd_iter", 10),
            "save_dir": logging.get("save_dir", "./models/mappo"),
            "log_level": logging.get("log_level", "WARN"),
        }

    return raw


def build_env_config(env_cfg: dict[str, Any]) -> dict[str, Any]:
    """Convert configs/env_config.yaml into WarehouseEnv constructor keys."""
    grid = env_cfg.get("grid", {})
    robots = env_cfg.get("robots", {})
    orders = env_cfg.get("orders", {})
    rewards = env_cfg.get("rewards", {})

    deadline_steps = orders.get("order_deadline_steps")
    use_deadlines = orders.get("use_deadlines", orders.get("order_deadlines", False))
    deadline_range = orders.get("deadline_range")
    if deadline_range is None and deadline_steps:
        deadline_range = (int(deadline_steps), int(deadline_steps))

    return {
        "grid_width": grid.get("width", 20),
        "grid_height": grid.get("height", 20),
        "num_shelves": grid.get("num_shelves", 40),
        "num_charging_stations": grid.get("num_charging_stations", 4),
        "num_delivery_zones": grid.get("num_delivery_zones", 3),
        "num_obstacles": grid.get("num_obstacles", 15),
        "num_robots": robots.get("num_robots", 10),
        "max_steps": robots.get("max_steps_per_episode", 500),
        "local_view_size": robots.get("observation_radius", 7),
        "battery_capacity": robots.get("max_battery", 100.0),
        "orders_per_episode": orders.get("max_queue_size", orders.get("orders_per_episode", 30)),
        "order_spawn_rate": orders.get("order_spawn_rate", 0.1),
        "use_deadlines": bool(use_deadlines),
        "deadline_range": deadline_range or (50, 200),
        "use_priorities": bool(orders.get("use_priorities", orders.get("priority_orders", False))),
        "reward_delivery": rewards.get("delivery_reward", 100.0),
        "reward_collision": rewards.get("collision_penalty", -2.0),
        "reward_idle": rewards.get("idle_penalty", -0.5),
    }
