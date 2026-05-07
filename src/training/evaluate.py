"""
evaluate.py — Checkpoint Evaluation & Benchmark Comparison
Loads MAPPO checkpoints and runs structured rollout evaluation.

Usage:
  # Compare peak vs final (default benchmark)
  python -m training.evaluate --benchmark --episodes 50

  # Evaluate a single checkpoint
  python -m training.evaluate --checkpoint models/mappo/checkpoint_0050 --episodes 50

  # Full comparison: IPPO vs MAPPO peak vs MAPPO final
  python -m training.evaluate --benchmark --ippo-checkpoint models/ippo_baseline/final --episodes 50
"""

from __future__ import annotations
import os
import sys
import warnings
import argparse
import json
import tempfile
from pathlib import Path
from typing import Optional
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_TMP_DIR = PROJECT_ROOT / ".tmp"
LOCAL_TMP_DIR.mkdir(exist_ok=True)
os.environ.setdefault("TMP", str(LOCAL_TMP_DIR))
os.environ.setdefault("TEMP", str(LOCAL_TMP_DIR))
tempfile.tempdir = str(LOCAL_TMP_DIR)

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.training.config_utils import build_env_config, load_train_config, load_yaml

# ──────────────────────────────────────────────
# Config helpers (mirrors train_mappo.py exactly)
# ──────────────────────────────────────────────

def is_ippo_checkpoint(checkpoint_path: str) -> bool:
    """Detect IPPO checkpoint by checking for policy_0 folder inside policies/."""
    policies_dir = Path(checkpoint_path) / "policies"
    if not policies_dir.exists():
        # Check one level up (RLlib sometimes nests)
        policies_dir = Path(checkpoint_path).parent / "policies"
    return (policies_dir / "policy_0").exists()


def build_algo(env_config: dict, train_cfg: dict, ippo: bool = False, num_robots: int = 10):
    """Build algo config matching the checkpoint type — IPPO or MAPPO."""

    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.policy.policy import PolicySpec

    from src.training.callbacks import WarehouseCallbacks

    if ippo:
        # Mirror train_independent.py exactly
        policies = {
            f"policy_{i}": PolicySpec() for i in range(num_robots)
        }
        policy_mapping_fn = lambda agent_id, *args, **kwargs: f"policy_{int(agent_id.split('_')[1])}"
    else:
        # Mirror train_mappo.py exactly
        policies = {
            "shared_policy": PolicySpec(
                observation_space=None,
                action_space=None,
                config={
                    "model": {
                        "fcnet_hiddens": [256, 256],
                        "fcnet_activation": "relu",
                        "use_lstm": False,
                    }
                },
            )
        }
        policy_mapping_fn = lambda agent_id, *args, **kwargs: "shared_policy"

    config = (
        PPOConfig()
        .environment(env="warehouse_eval", env_config=env_config)
        .framework("torch")
        .rollouts(
            num_rollout_workers=0,
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
        .resources(num_gpus=0)
        .callbacks(WarehouseCallbacks)
        .debugging(log_level="ERROR")
    )
    return config.build()


# ──────────────────────────────────────────────
# Core rollout runner
# ──────────────────────────────────────────────

def run_episodes(algo, env_config: dict, num_episodes: int, label: str, ippo: bool = False) -> dict:
    """
    Run `num_episodes` full rollouts and collect per-episode metrics.
    Returns aggregated stats dict.
    """
    from src.environment.warehouse_env import WarehouseEnv

    env = WarehouseEnv(env_config)

    all_rewards       = []
    all_throughputs   = []
    all_completion    = []
    all_collisions    = []
    all_steps         = []
    all_deliveries    = []
    per_agent_rewards = defaultdict(list)

    print(f"\n  Running {num_episodes} episodes for [{label}]...")

    for ep in range(1, num_episodes + 1):
        obs, _ = env.reset()
        episode_rewards = defaultdict(float)
        done = False
        step = 0

        while not done:
            actions = {}
            for agent_id, agent_obs in obs.items():
                if ippo:
                    idx = int(agent_id.split("_")[1])
                    policy_id = f"policy_{idx}"
                else:
                    policy_id = "shared_policy"
                policy_output = algo.compute_single_action(
                    agent_obs,
                    policy_id=policy_id,
                    explore=False,
                )
                actions[agent_id] = policy_output
                # DEBUG: count DELIVER actions (only log ep 1)
                if ep == 1:
                    for agent_id, a in actions.items():
                        robot = env.robots[agent_id]
                        if a == 5:  # DELIVER
                            print(
                                f"    [DEBUG ep1] Step {step} {agent_id}: "
                                f"DELIVER | carrying={robot.carrying_item} | "
                                f"has_order={robot.assigned_order is not None} | "
                                f"at_delivery={robot.position == robot.assigned_order.delivery_position if robot.assigned_order else 'N/A'}"
                            )
                        if a == 4:  # PICK
                            print(
                                f"    [DEBUG ep1] Step {step} {agent_id}: "
                                f"PICK | carrying={robot.carrying_item} | "
                                f"at_shelf={robot.position == robot.assigned_order.shelf_position if robot.assigned_order else 'N/A'}"
                            )            

            obs, rewards, terminateds, truncateds, infos = env.step(actions)
            step += 1

            for agent_id, r in rewards.items():
                episode_rewards[agent_id] += r

            done = terminateds.get("__all__", False) or truncateds.get("__all__", False)

        # Collect episode-level metrics from the env directly
        summary = env.get_episode_summary()
        total_reward = sum(episode_rewards.values())

        all_rewards.append(total_reward)
        all_throughputs.append(summary["throughput_score"])
        all_completion.append(summary["completion_rate"])
        all_collisions.append(summary["total_collisions"])
        all_steps.append(summary["total_steps"])
        all_deliveries.append(summary["order_metrics"].get("completed", 0))

        for agent_id, r in episode_rewards.items():
            per_agent_rewards[agent_id].append(r)

        if ep % 10 == 0 or ep == num_episodes:
            print(
                f"    Ep {ep:3d}/{num_episodes} | "
                f"Reward: {total_reward:7.1f} | "
                f"Throughput: {summary['throughput_score']:.3f} | "
                f"Collisions: {summary['total_collisions']:3d} | "
                f"Delivered: {summary['order_metrics'].get('completed', 0):2d}"
            )

    env.close() if hasattr(env, "close") else None

    # Per-agent spread at episode level
    agent_means = {aid: np.mean(rs) for aid, rs in per_agent_rewards.items()}
    agent_mean_values = list(agent_means.values())

    return {
        "label": label,
        "n_episodes": num_episodes,
        # Reward
        "reward_mean":   float(np.mean(all_rewards)),
        "reward_std":    float(np.std(all_rewards)),
        "reward_min":    float(np.min(all_rewards)),
        "reward_max":    float(np.max(all_rewards)),
        # Throughput
        "throughput_mean": float(np.mean(all_throughputs)),
        "throughput_std":  float(np.std(all_throughputs)),
        # Completion rate
        "completion_mean": float(np.mean(all_completion)),
        # Collisions
        "collisions_mean": float(np.mean(all_collisions)),
        "collisions_std":  float(np.std(all_collisions)),
        # Deliveries
        "deliveries_mean": float(np.mean(all_deliveries)),
        # Episode length
        "steps_mean": float(np.mean(all_steps)),
        # Per-agent spread (role specialization indicator)
        "per_agent_reward_mean": float(np.mean(agent_mean_values)),
        "per_agent_reward_min":  float(np.min(agent_mean_values)),
        "per_agent_reward_max":  float(np.max(agent_mean_values)),
        "per_agent_reward_spread": float(np.max(agent_mean_values) - np.min(agent_mean_values)),
        "per_agent_details": {aid: round(float(v), 2) for aid, v in sorted(agent_means.items())},
    }


# ──────────────────────────────────────────────
# Result printing
# ──────────────────────────────────────────────

def print_results(results: list[dict]):
    sep = "=" * 70
    print(f"\n{sep}")
    print("  EVALUATION RESULTS")
    print(sep)

    for r in results:
        print(f"\n  [{r['label']}]  ({r['n_episodes']} episodes)")
        print(f"    Reward        : {r['reward_mean']:8.2f}  ±{r['reward_std']:.1f}  "
              f"(range {r['reward_min']:.0f} – {r['reward_max']:.0f})")
        print(f"    Throughput    : {r['throughput_mean']:.4f}  ±{r['throughput_std']:.4f}")
        print(f"    Completion    : {r['completion_mean']:.4f}")
        print(f"    Collisions    : {r['collisions_mean']:.1f}  ±{r['collisions_std']:.1f}")
        print(f"    Deliveries    : {r['deliveries_mean']:.1f}")
        print(f"    Avg steps     : {r['steps_mean']:.0f}")
        print(f"    Per-agent     : min={r['per_agent_reward_min']:.1f}  "
              f"max={r['per_agent_reward_max']:.1f}  "
              f"spread={r['per_agent_reward_spread']:.1f}")

    # Side-by-side comparison if multiple checkpoints
    if len(results) >= 2:
        print(f"\n{sep}")
        print("  HEAD-TO-HEAD COMPARISON")
        print(sep)
        baseline = results[0]
        for r in results[1:]:
            reward_delta = r["reward_mean"] - baseline["reward_mean"]
            reward_pct   = (reward_delta / abs(baseline["reward_mean"])) * 100 if baseline["reward_mean"] != 0 else 0
            coll_delta   = r["collisions_mean"] - baseline["collisions_mean"]
            tp_delta     = r["throughput_mean"] - baseline["throughput_mean"]

            print(f"\n  {r['label']}  vs  {baseline['label']}:")
            print(f"    Reward     : {reward_delta:+.2f}  ({reward_pct:+.1f}%)")
            print(f"    Throughput : {tp_delta:+.4f}")
            print(f"    Collisions : {coll_delta:+.1f}  ({'worse' if coll_delta > 0 else 'better'})")

    print(f"\n{sep}\n")


def save_results(results: list[dict], output_path: str = "eval_results.json"):
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to: {output_path}")


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate MAPPO Warehouse Checkpoints")
    parser.add_argument("--checkpoint",      type=str, default=None,
                        help="Path to a single checkpoint to evaluate")
    parser.add_argument("--benchmark",       action="store_true",
                        help="Compare checkpoint_0050 (peak) vs final_model")
    parser.add_argument("--ippo-checkpoint", type=str, default=None,
                        help="Optional IPPO checkpoint path for 3-way comparison")
    parser.add_argument("--episodes",        type=int, default=50)
    parser.add_argument("--train-config",    default="configs/mappo_config.yaml")
    parser.add_argument("--env-config",      default="configs/env_config.yaml")
    parser.add_argument("--output",          default="eval_results.json")
    args = parser.parse_args()

    # Load configs
    train_cfg = load_train_config(args.train_config)
    env_cfg   = load_yaml(args.env_config)
    env_config = build_env_config(env_cfg)

    # Init Ray
    import ray
    from ray import tune
    from src.environment.warehouse_env import WarehouseEnv

    ray.init(ignore_reinit_error=True, num_cpus=2)

    tune.register_env("warehouse_eval", lambda cfg: WarehouseEnv(cfg))

    all_results = []

    # ── Single checkpoint mode ──
    if args.checkpoint and not args.benchmark:
        label = Path(args.checkpoint).name
        ippo = is_ippo_checkpoint(args.checkpoint)
        print(f"\nBuilding algo for: {label}  ({'IPPO' if ippo else 'MAPPO'})")
        algo = build_algo(env_config, train_cfg, ippo=ippo, num_robots=env_cfg["robots"]["num_robots"])
        algo.restore(args.checkpoint)
        result = run_episodes(algo, env_config, args.episodes, label, ippo=ippo)
        all_results.append(result)
        algo.stop()

    # ── Benchmark mode: peak vs final ──
    elif args.benchmark:
        checkpoints = [
            ("models/mappo/checkpoint_0050", "MAPPO Peak (iter 50)"),
            ("models/mappo/final_model",     "MAPPO Final (iter 200)"),
        ]
        if args.ippo_checkpoint:
            checkpoints.insert(0, (args.ippo_checkpoint, "IPPO Baseline"))

        for ckpt_path, label in checkpoints:
            if not Path(ckpt_path).exists():
                print(f"  Skipping {label} — path not found: {ckpt_path}")
                continue
            ippo = is_ippo_checkpoint(ckpt_path)
            print(f"\nBuilding algo for: {label}  ({'IPPO' if ippo else 'MAPPO'})")
            algo = build_algo(env_config, train_cfg, ippo=ippo, num_robots=env_cfg["robots"]["num_robots"])
            algo.restore(ckpt_path)
            result = run_episodes(algo, env_config, args.episodes, label, ippo=ippo)
            all_results.append(result)
            algo.stop()

    else:
        parser.print_help()
        ray.shutdown()
        return

    # Print and save
    if all_results:
        print_results(all_results)
        save_results(all_results, args.output)

    ray.shutdown()


if __name__ == "__main__":
    main()
