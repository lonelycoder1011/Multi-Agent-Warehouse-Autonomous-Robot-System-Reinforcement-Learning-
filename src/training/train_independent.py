"""
train_independent.py — Independent PPO Baseline (IPPO)
Each robot trains with its own separate policy.
Used for comparison against MAPPO.
"""

from __future__ import annotations
import os
import sys
import argparse
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_TMP_DIR = PROJECT_ROOT / ".tmp"
LOCAL_TMP_DIR.mkdir(exist_ok=True)
os.environ.setdefault("TMP", str(LOCAL_TMP_DIR))
os.environ.setdefault("TEMP", str(LOCAL_TMP_DIR))
tempfile.tempdir = str(LOCAL_TMP_DIR)

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.config_utils import build_env_config, load_train_config, load_yaml


def train_independent(
    train_config_path: str = "configs/mappo_config.yaml",
    env_config_path: str = "configs/env_config.yaml",
    num_iterations: int = 20,
    num_workers: int = 2,
):
    import ray
    from ray import tune
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.policy.policy import PolicySpec

    from src.environment.warehouse_env import WarehouseEnv
    from src.training.callbacks import WarehouseCallbacks

    train_cfg = load_train_config(train_config_path)
    env_cfg = load_yaml(env_config_path)

    ray.init(ignore_reinit_error=True)
    tune.register_env("warehouse_env", lambda cfg: WarehouseEnv(cfg))

    num_robots = env_cfg["robots"]["num_robots"]

    env_config = build_env_config(env_cfg)

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
        .rollouts(num_rollout_workers=num_workers, num_envs_per_worker=1)
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


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-config", default=None)
    parser.add_argument("--config", dest="legacy_config", default=None)
    parser.add_argument("--env-config", default="configs/env_config.yaml")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args(argv)

    train_config = args.train_config or args.legacy_config or "configs/mappo_config.yaml"

    train_independent(
        train_config,
        args.env_config,
        args.iterations,
        num_workers=args.num_workers,
    )

if __name__ == "__main__":
    main()
