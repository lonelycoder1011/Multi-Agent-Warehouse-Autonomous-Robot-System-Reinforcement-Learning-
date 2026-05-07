"""
train_independent.py — Independent PPO Baseline (IPPO)
Each robot trains with its own separate policy.
Used for comparison against MAPPO.
"""

from __future__ import annotations
import sys
import argparse
import yaml
from pathlib import Path

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environment.warehouse_env import WarehouseEnv
from src.training.callbacks import WarehouseCallbacks


def train_independent(
    train_config_path: str = "configs/mappo_config.yaml",
    env_config_path: str = "configs/env_config.yaml",
    num_iterations: int = 20,
):
    with open(train_config_path) as f:
        train_cfg = yaml.safe_load(f)["mappo"]
    with open(env_config_path) as f:
        env_cfg = yaml.safe_load(f)

    ray.init(ignore_reinit_error=True)
    tune.register_env("warehouse_env", lambda cfg: WarehouseEnv(cfg))

    num_robots = env_cfg["robots"]["num_robots"]

    env_config = {
        "grid_width": env_cfg["grid"]["width"],
        "grid_height": env_cfg["grid"]["height"],
        "num_robots": env_cfg["robots"]["num_robots"],
        "max_steps": env_cfg["robots"]["max_steps_per_episode"],
        "local_view_size": env_cfg["robots"]["observation_radius"],
        "comm_dim": 16,
        "orders_per_episode": env_cfg["orders"]["max_queue_size"],
    }

    policies = {
        f"policy_{i}": PolicySpec() for i in range(num_robots)
    }

    def policy_mapping_fn(agent_id: str, episode=None, **kwargs) -> str:
        idx = int(agent_id.split("_")[1])
        return f"policy_{idx}"

    config = (
        PPOConfig()
        .environment(env="warehouse_env", env_config=env_config)
        .framework("torch")
        .rollouts(num_rollout_workers=2, num_envs_per_worker=1)
        .training(
            lr=train_cfg.get("lr", 3e-4),
            gamma=0.99,
            train_batch_size=2048,
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
        )
        .callbacks(WarehouseCallbacks)
    )

    algo = config.build()
    save_dir = Path("./models/ippo_baseline")
    save_dir.mkdir(parents=True, exist_ok=True)
    results = []

    print(f"\n{'='*50}")
    print(f"IPPO Baseline Training - Independent Learners")
    print(f"{'='*50}\n")

    for i in range(1, num_iterations + 1):
        result = algo.train()
        reward = result.get("episode_reward_mean", 0)
        throughput = result.get("warehouse_throughput", 0)
        results.append({"iter": i, "reward": reward, "throughput": throughput})

        if i % 10 == 0:
            print(f"Iter {i:4d} | Reward: {reward:7.2f} | Throughput: {throughput:.3f}")

    checkpoint = algo.save(str(save_dir / "final"))
    print(f"\nIPPO baseline saved: {checkpoint}")

    final_20 = results[-20:]
    avg_throughput = sum(r["throughput"] for r in final_20) / len(final_20)
    print(f"Final 20-episode avg throughput: {avg_throughput:.3f}")

    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-config", default="configs/mappo_config.yaml")
    parser.add_argument("--env-config", default="configs/env_config.yaml")
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()

    train_independent(args.train_config, args.env_config, args.iterations)