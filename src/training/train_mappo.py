"""
train_mappo.py — MAPPO Training with Ray RLlib
Centralized training, decentralized execution.
Run: python train_mappo.py --config configs/mappo_config.yaml
"""

from __future__ import annotations
import os
import sys
import warnings
import argparse
import tempfile
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_TMP_DIR = PROJECT_ROOT / ".tmp"
LOCAL_TMP_DIR.mkdir(exist_ok=True)
os.environ.setdefault("TMP", str(LOCAL_TMP_DIR))
os.environ.setdefault("TEMP", str(LOCAL_TMP_DIR))
tempfile.tempdir = str(LOCAL_TMP_DIR)

# Suppress all deprecation warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.config_utils import build_env_config, load_train_config, load_yaml


def get_policy_config(env_cfg: dict) -> dict:
    """Build shared policy for all robots (parameter sharing)."""
    return {
        "model": {
            "fcnet_hiddens": [256, 256],
            "fcnet_activation": "relu",
            "use_lstm": False,
        }
    }


def train(
    train_config_path: str = "configs/mappo_config.yaml",
    env_config_path: str = "configs/env_config.yaml",
    curriculum_config_path: str = "configs/curriculum_config.yaml",
    resume: bool = False,
    checkpoint_path: str = None,
    overrides: Optional[dict] = None,
):
    """Main training entry point."""
    import ray
    from ray import tune
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.policy.policy import PolicySpec

    from src.curriculum.curriculum_manager import CurriculumManager
    from src.environment.warehouse_env import WarehouseEnv
    from src.training.callbacks import WarehouseCallbacks

    try:
        import wandb
    except ImportError:
        wandb = None

    train_cfg = load_train_config(train_config_path)
    if overrides:
        train_cfg.update({k: v for k, v in overrides.items() if v is not None})
    env_cfg = load_yaml(env_config_path)
    num_robots = env_cfg["robots"]["num_robots"]

    # Initialize Ray
    ray.init(ignore_reinit_error=True, num_cpus=os.cpu_count())
    print(f"[Ray] Initialized with {os.cpu_count()} CPUs")

    # Register environment
    tune.register_env(
        "warehouse_env",
        lambda cfg: WarehouseEnv(cfg),
    )

    # W&B initialization
    if wandb is not None:
        wandb.init(
            project="warehouse-rl",
            name="mappo-warehouse",
            config={**train_cfg, **env_cfg},
        )
        print("[W&B] Logging enabled")

    # Build env config
    env_config = build_env_config(env_cfg)

    # Curriculum manager
    curriculum = CurriculumManager(
        promotion_threshold=0.75,
        evaluation_window=20,
        config_path=curriculum_config_path,
    )

    # Multi-agent policy config with parameter sharing
    policies = {
        "shared_policy": PolicySpec(
            observation_space=None,
            action_space=None,
            config=get_policy_config(env_cfg),
        )
    }

    def policy_mapping_fn(agent_id: str, episode=None, worker=None, **kwargs) -> str:
        return "shared_policy"

    # Build PPO config (MAPPO with parameter sharing)
    config = (
        PPOConfig()
        .environment(
            env="warehouse_env",
            env_config=env_config,
        )
        .framework("torch")
        .rollouts(
            num_rollout_workers=train_cfg.get("num_rollout_workers", 2),
            num_envs_per_worker=train_cfg.get("num_envs_per_worker", 1),
            rollout_fragment_length="auto",
        )
        .training(
            lr=train_cfg.get("lr", 3e-4),
            gamma=train_cfg.get("gamma", 0.99),
            lambda_=train_cfg.get("lambda_gae", 0.95),
            clip_param=train_cfg.get("clip_param", 0.2),
            entropy_coeff=train_cfg.get("entropy_coeff", 0.01),
            vf_loss_coeff=train_cfg.get("vf_loss_coeff", 0.5),
            num_sgd_iter=train_cfg.get("num_sgd_iter", 10),
            sgd_minibatch_size=train_cfg.get("sgd_minibatch_size", 256),
            train_batch_size=train_cfg.get("train_batch_size", 2048),
            grad_clip=train_cfg.get("grad_clip", 10.0),
            model={"fcnet_hiddens": train_cfg.get("fcnet_hiddens", [256, 256])},
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
        )
        .resources(
            num_gpus=train_cfg.get("num_gpus", 0),
            num_cpus_per_worker=train_cfg.get("num_cpus_per_worker", 1),
        )
        .callbacks(WarehouseCallbacks)
        .debugging(log_level="WARN")
    )

    # Build algorithm
    algo = config.build()

    if checkpoint_path and resume:
        algo.restore(checkpoint_path)
        print(f"[Training] Resumed from checkpoint: {checkpoint_path}")

    # Training loop
    save_dir = Path(train_cfg.get("save_dir", "./models/mappo"))
    save_dir.mkdir(parents=True, exist_ok=True)
    num_iterations = train_cfg.get("num_iterations", 200)
    checkpoint_freq = train_cfg.get("checkpoint_freq", 25)
    best_throughput = 0.0

    print(f"\n{'='*60}")
    print(f"Warehouse RL Training - MAPPO (Parameter Sharing)")
    print(f"   Robots: {num_robots} | Iterations: {num_iterations}")
    print(f"   Workers: {train_cfg.get('num_rollout_workers', 2)}")
    print(f"{'='*60}\n")

    for iteration in range(1, num_iterations + 1):
        result = algo.train()

        # Extract metrics
        mean_reward = result.get("episode_reward_mean", 0)
        throughput = result.get("warehouse_throughput", 0)
        collision_rate = result.get("warehouse_collision_rate", 0)
        timesteps = result.get("timesteps_total", 0)

        # Curriculum update
        if throughput > 0:
            promoted = curriculum.record_episode(throughput)
            if promoted:
                print(f"  Curriculum Promotion! -> Stage {curriculum.stage_number}")

        # W&B logging
        if wandb is not None and wandb.run:
            wandb.log({
                "iteration": iteration,
                "reward_mean": mean_reward,
                "throughput": throughput,
                "collision_rate": collision_rate,
                "timesteps": timesteps,
                "curriculum_stage": curriculum.stage_number,
            })

        # Console output every 10 iterations
        if iteration % 10 == 0 or iteration == 1:
            print(
                f"Iter {iteration:4d}/{num_iterations} | "
                f"Reward: {mean_reward:7.2f} | "
                f"Throughput: {throughput:.3f} | "
                f"Collision: {collision_rate:.4f} | "
                f"Steps: {timesteps:,}"
            )

        # Checkpoint
        if iteration % checkpoint_freq == 0:
            ckpt = algo.save(str(save_dir / f"checkpoint_{iteration:04d}"))
            print(f"  Saved checkpoint: {ckpt}")

            if throughput > best_throughput:
                best_throughput = throughput
                best_path = algo.save(str(save_dir / "best_model"))
                print(f"  New best throughput: {best_throughput:.3f} -> {best_path}")

    # Final save
    final_path = algo.save(str(save_dir / "final_model"))
    print(f"\nTraining complete! Final model: {final_path}")
    print(f"   Best throughput: {best_throughput:.3f}")
    print(f"   Curriculum stages reached: {curriculum.stage_number}/5")

    if wandb is not None and wandb.run:
        wandb.finish()

    algo.stop()
    ray.shutdown()

    return final_path


def main(argv: Optional[list[str]] = None):
    parser = argparse.ArgumentParser(description="Train MAPPO Warehouse Agent")
    parser.add_argument("--train-config", default=None)
    parser.add_argument("--config", dest="legacy_config", default=None)
    parser.add_argument("--env-config", default="configs/env_config.yaml")
    parser.add_argument("--curriculum-config", default="configs/curriculum_config.yaml")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Accepted for script compatibility; W&B auto-enables when configured.",
    )
    args = parser.parse_args(argv)

    train_config = args.train_config or args.legacy_config or "configs/mappo_config.yaml"
    overrides = {"num_rollout_workers": args.num_workers}

    return train(
        train_config_path=train_config,
        env_config_path=args.env_config,
        curriculum_config_path=args.curriculum_config,
        resume=args.resume,
        checkpoint_path=args.checkpoint,
        overrides=overrides,
    )


if __name__ == "__main__":
    main()
