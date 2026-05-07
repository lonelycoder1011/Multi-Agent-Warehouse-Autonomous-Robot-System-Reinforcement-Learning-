"""Launch the Warehouse RL dashboard and real MAPPO training together."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / ".run"
LOG_DIR = ROOT / "logs"
STACK_FILE = RUN_DIR / "warehouse-stack.json"


def port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.3)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def python_executable() -> str:
    bundled = ROOT / "rl-env" / "Scripts" / "python.exe"
    if bundled.exists():
        return str(bundled)
    return sys.executable


def creation_flags() -> int:
    flags = 0
    if os.name == "nt":
        flags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        flags |= getattr(subprocess, "DETACHED_PROCESS", 0)
    return flags


def start_process(name: str, command: list[str], log_path: Path, env: dict[str, str]) -> dict:
    log_handle = log_path.open("w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        creationflags=creation_flags(),
        close_fds=True,
    )
    log_handle.close()
    return {
        "name": name,
        "pid": proc.pid,
        "log": str(log_path.relative_to(ROOT)),
        "command": command,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Start dashboard and real MAPPO training.")
    parser.add_argument("--dashboard-port", type=int, default=8080)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--train-config", default="configs/mappo_config.yaml")
    parser.add_argument("--env-config", default="configs/env_config.yaml")
    parser.add_argument("--curriculum-config", default="configs/curriculum_config.yaml")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--online-wandb", action="store_true", help="Do not force W&B offline mode.")
    args = parser.parse_args()

    if port_open(args.dashboard_port):
        print(
            f"Port {args.dashboard_port} is already in use. "
            "Stop the existing dashboard or pass --dashboard-port 8081.",
            file=sys.stderr,
        )
        return 2

    RUN_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)

    py = python_executable()
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["WAREHOUSE_RL_DISABLE_RAY_ENV"] = "1"
    env["WAREHOUSE_RL_TRAINING_LOG"] = str(LOG_DIR / "training.log")
    if not args.online_wandb:
        env["WANDB_MODE"] = "offline"

    dashboard_cmd = [
        py,
        "-m",
        "uvicorn",
        "dashboard.app:app",
        "--host",
        args.host,
        "--port",
        str(args.dashboard_port),
        "--log-level",
        "info",
    ]

    training_cmd = [
        py,
        "-m",
        "src.training.train_mappo",
        "--train-config",
        args.train_config,
        "--env-config",
        args.env_config,
        "--curriculum-config",
        args.curriculum_config,
        "--num-workers",
        str(args.num_workers),
    ]
    if args.resume:
        training_cmd.append("--resume")
    if args.checkpoint:
        training_cmd.extend(["--checkpoint", args.checkpoint])

    dashboard = start_process("dashboard", dashboard_cmd, LOG_DIR / "dashboard.log", env)
    training = start_process("training", training_cmd, LOG_DIR / "training.log", env)

    stack = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "dashboard_url": f"http://{args.host}:{args.dashboard_port}",
        "dashboard": dashboard,
        "training": training,
    }
    STACK_FILE.write_text(json.dumps(stack, indent=2), encoding="utf-8")

    print("Warehouse RL stack started.")
    print(f"Dashboard: {stack['dashboard_url']}")
    print(f"Dashboard PID: {dashboard['pid']}")
    print(f"Training PID : {training['pid']}")
    print(f"Dashboard log: {dashboard['log']}")
    print(f"Training log : {training['log']}")
    print(f"Stack file   : {STACK_FILE.relative_to(ROOT)}")
    print()
    print("Watch logs:")
    print(r"  .\scripts\watch_logs.ps1 -Target all")
    print(r"  .\scripts\watch_logs.ps1 -Target training")
    print()
    print("Stop stack:")
    print(r"  .\scripts\stop_full_system.ps1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
