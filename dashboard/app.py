"""FastAPI dashboard backend for Warehouse RL.

The dashboard streams a live WarehouseEnv rollout by default. It falls back to a
small built-in simulator only if the environment cannot be imported.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import socket
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "env_config.yaml"
INDEX_PATH = Path(__file__).resolve().parent / "index.html"
STATIC_DIR = Path(__file__).resolve().parent / "static"
RUN_DIR = PROJECT_ROOT / ".run"
STACK_FILE = RUN_DIR / "warehouse-stack.json"
LOG_DIR = PROJECT_ROOT / "logs"
TRAINING_LOG = LOG_DIR / "training.log"
DASHBOARD_LOG = LOG_DIR / "dashboard.log"

os.environ.setdefault("WAREHOUSE_RL_DISABLE_RAY_ENV", "1")

app = FastAPI(title="Warehouse RL Dashboard", version="1.0.0")
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

try:
    from src.environment.grid_world import DIRECTION_DELTAS
    from src.environment.robot import Action, RobotState
    from src.environment.warehouse_env import WarehouseEnv
    from src.training.config_utils import build_env_config, load_yaml

    ENV_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - fallback keeps dashboard bootable.
    DIRECTION_DELTAS = {}
    Action = None
    RobotState = None
    WarehouseEnv = None
    build_env_config = None
    load_yaml = None
    ENV_IMPORT_ERROR = str(exc)


class WarehouseEnvStream:
    """Runs the real warehouse environment with a deterministic heuristic policy."""

    source = "warehouse_env"

    def __init__(self):
        if WarehouseEnv is None or build_env_config is None or load_yaml is None:
            raise RuntimeError(ENV_IMPORT_ERROR or "WarehouseEnv is unavailable")

        raw_cfg = load_yaml(CONFIG_PATH)
        self.env_config = build_env_config(raw_cfg)
        self.env_config.setdefault("seed", 42)
        self.env = WarehouseEnv(self.env_config)
        self.episode = 0
        self.obs = {}
        self.throughput_history: list[float] = []
        self.reward_history: list[float] = []
        self.battery_history: list[float] = []
        self.reset()

    def reset(self) -> None:
        self.episode += 1
        self.obs, _ = self.env.reset(seed=self.env_config.get("seed", 42) + self.episode)

    def tick(self) -> dict[str, Any]:
        actions = {
            agent_id: self._choose_action(robot)
            for agent_id, robot in self.env.robots.items()
        }
        self.obs, rewards, terminateds, truncateds, _ = self.env.step(actions)
        state = self._state()

        if terminateds.get("__all__") or truncateds.get("__all__"):
            self.reset()

        return state

    def _choose_action(self, robot) -> int:
        if robot.battery <= robot.low_battery_threshold:
            nearest = self.env.grid.get_nearest_zone(robot.position, "charging")
            if nearest and robot.position == nearest[0]:
                return int(Action.CHARGE)
            if nearest:
                return self._move_toward(robot, nearest[0])

        order = robot.assigned_order
        if order is None:
            return int(Action.STAY)

        if not robot.carrying_item:
            if robot.position == order.shelf_position:
                return int(Action.PICK)
            return self._move_toward(robot, order.shelf_position)

        if robot.position == order.delivery_position:
            return int(Action.DELIVER)
        return self._move_toward(robot, order.delivery_position)

    def _move_toward(self, robot, target) -> int:
        dx = target.x - robot.position.x
        dy = target.y - robot.position.y
        candidates: list[Action] = []

        if abs(dx) >= abs(dy):
            candidates.extend([Action.EAST if dx > 0 else Action.WEST] if dx else [])
            candidates.extend([Action.SOUTH if dy > 0 else Action.NORTH] if dy else [])
        else:
            candidates.extend([Action.SOUTH if dy > 0 else Action.NORTH] if dy else [])
            candidates.extend([Action.EAST if dx > 0 else Action.WEST] if dx else [])

        candidates.extend([Action.NORTH, Action.SOUTH, Action.EAST, Action.WEST])

        for action in candidates:
            delta = DIRECTION_DELTAS[int(action)]
            new_pos = robot.position + delta
            if (
                self.env.grid.is_walkable(new_pos)
                and self.env.grid.dynamic_grid[new_pos.y, new_pos.x] == 0
            ):
                return int(action)

        return int(Action.STAY)

    def _state(self) -> dict[str, Any]:
        metrics = self.env.order_manager.get_metrics()
        robots = [self._robot_payload(robot) for robot in self.env.robots.values()]
        avg_battery = sum(r["battery"] for r in robots) / max(len(robots), 1)
        episode_reward = sum(self.env._episode_rewards.values())
        throughput = metrics["throughput_score"] * 100.0

        self.throughput_history.append(round(throughput, 2))
        self.reward_history.append(round(float(episode_reward), 2))
        self.battery_history.append(round(avg_battery, 1))
        self.throughput_history = self.throughput_history[-100:]
        self.reward_history = self.reward_history[-100:]
        self.battery_history = self.battery_history[-100:]

        return {
            "type": "state",
            "source": self.source,
            "step": self.env.current_step,
            "episode": self.episode,
            "grid": self.env.grid.static_grid.astype(int).tolist(),
            "robots": robots,
            "metrics": {
                "orders_completed": metrics["completed"],
                "throughput": round(throughput, 2),
                "avg_battery": round(avg_battery, 1),
                "total_collisions": self.env._total_collisions,
                "episode_reward": round(float(episode_reward), 2),
                "completion_rate": round(metrics["completion_rate"], 3),
            },
            "history": {
                "throughput": self.throughput_history[-50:],
                "reward": self.reward_history[-50:],
                "battery": self.battery_history[-50:],
            },
        }

    @staticmethod
    def _robot_payload(robot) -> dict[str, Any]:
        state_name = robot.state.name
        if robot.state == RobotState.CHARGING or "CHARGE" in state_name:
            label = "CHARGING"
        elif robot.state == RobotState.PICKING:
            label = "PICKING"
        elif robot.carrying_item or "DELIVERY" in state_name:
            label = "DELIVERING"
        elif "MOVING" in state_name:
            label = "MOVING"
        else:
            label = "IDLE"

        return {
            "id": robot.robot_id,
            "x": robot.position.x,
            "y": robot.position.y,
            "battery": round(float(robot.battery), 1),
            "state": label,
            "task": robot.assigned_order.order_id if robot.assigned_order else None,
            "deliveries": robot.total_deliveries,
            "collisions": robot.total_collisions,
        }


class DemoStream:
    """Small fallback stream used only when the real environment cannot load."""

    source = "fallback_demo"
    GRID_W = 20
    GRID_H = 20
    EMPTY, OBSTACLE, SHELF, CHARGING, DELIVERY = 0, 1, 2, 3, 4

    def __init__(self):
        self.step = 0
        self.episode = 1
        self.rng = random.Random(42)
        self.grid = self._build_grid()
        self.robots = self._spawn_robots()
        self.orders_completed = 0
        self.total_collisions = 0
        self.reward_history: list[float] = []
        self.throughput_history: list[float] = []
        self.battery_history: list[float] = []
        self.episode_reward = 0.0

    def _build_grid(self) -> list[list[int]]:
        grid = [[self.EMPTY] * self.GRID_W for _ in range(self.GRID_H)]
        for row in range(2, 14, 3):
            for col in range(2, 19, 2):
                grid[row][col] = self.SHELF
        for x, y in [(1, 1), (18, 1), (1, 18), (18, 18)]:
            grid[y][x] = self.CHARGING
        for x in [5, 10, 15]:
            grid[18][x] = self.DELIVERY
        for x, y in [(3, 5), (7, 8), (12, 5), (16, 10), (4, 13), (9, 12)]:
            grid[y][x] = self.OBSTACLE
        return grid

    def _spawn_robots(self) -> list[dict[str, Any]]:
        points = [(2, 17), (4, 17), (6, 17), (8, 17), (10, 17), (12, 17), (14, 17), (2, 16), (5, 16), (8, 16)]
        return [
            {"id": i, "x": x, "y": y, "battery": self.rng.uniform(60, 100), "state": "IDLE", "task": None, "deliveries": 0, "collisions": 0}
            for i, (x, y) in enumerate(points)
        ]

    def tick(self) -> dict[str, Any]:
        self.step += 1
        if self.step > 500:
            self.step = 1
            self.episode += 1
            self.robots = self._spawn_robots()
            self.orders_completed = 0

        for robot in self.robots:
            robot["battery"] = max(5.0, robot["battery"] - self.rng.uniform(0.1, 0.6))
            robot["state"] = self.rng.choice(["MOVING", "PICKING", "DELIVERING", "IDLE"])
            dx = self.rng.choice([-1, 0, 0, 1])
            dy = self.rng.choice([-1, 0, 0, 1])
            nx = max(0, min(self.GRID_W - 1, robot["x"] + dx))
            ny = max(0, min(self.GRID_H - 1, robot["y"] + dy))
            if self.grid[ny][nx] not in (self.OBSTACLE, self.SHELF):
                robot["x"], robot["y"] = nx, ny
            if robot["state"] == "DELIVERING" and self.rng.random() < 0.08:
                robot["deliveries"] += 1
                self.orders_completed += 1

        avg_battery = sum(r["battery"] for r in self.robots) / len(self.robots)
        throughput = self.orders_completed / max(1, self.step) * 100
        self.episode_reward += throughput * 0.002 + avg_battery * 0.0005
        self.throughput_history.append(round(throughput, 2))
        self.reward_history.append(round(self.episode_reward, 2))
        self.battery_history.append(round(avg_battery, 1))

        return {
            "type": "state",
            "source": self.source,
            "step": self.step,
            "episode": self.episode,
            "grid": self.grid,
            "robots": self.robots,
            "metrics": {
                "orders_completed": self.orders_completed,
                "throughput": round(throughput, 2),
                "avg_battery": round(avg_battery, 1),
                "total_collisions": self.total_collisions,
                "episode_reward": round(self.episode_reward, 2),
                "completion_rate": round(min(1.0, self.orders_completed / max(1, self.step * 0.15)), 3),
            },
            "history": {
                "throughput": self.throughput_history[-50:],
                "reward": self.reward_history[-50:],
                "battery": self.battery_history[-50:],
            },
        }


try:
    stream = WarehouseEnvStream()
except Exception as exc:
    ENV_IMPORT_ERROR = str(exc)
    stream = DemoStream()


@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse(INDEX_PATH.read_text(encoding="utf-8", errors="replace"))


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "source": stream.source,
        "step": getattr(stream, "step", getattr(getattr(stream, "env", None), "current_step", 0)),
        "episode": getattr(stream, "episode", 1),
        "env_error": ENV_IMPORT_ERROR,
    }


@app.get("/api/state")
async def state():
    return stream.tick()


@app.get("/api/training-config")
async def training_config():
    cfg_path = PROJECT_ROOT / "configs" / "mappo_config.yaml"
    env_path = PROJECT_ROOT / "configs" / "env_config.yaml"
    return {
        "train_config": load_yaml(cfg_path) if load_yaml else {},
        "env_config": load_yaml(env_path) if load_yaml else {},
    }


@app.get("/api/eval-results")
async def eval_results():
    path = PROJECT_ROOT / "eval_results.json"
    if not path.exists():
        return {"available": False, "results": []}
    return {"available": True, "results": json.loads(path.read_text(encoding="utf-8"))}


@app.get("/api/checkpoints")
async def checkpoints():
    model_dir = PROJECT_ROOT / "models"
    if not model_dir.exists():
        return {"checkpoints": []}
    paths = [
        str(path.relative_to(PROJECT_ROOT))
        for path in model_dir.rglob("*")
        if path.is_dir() and ("checkpoint" in path.name or path.name in {"best_model", "final_model", "final"})
    ]
    return {"checkpoints": sorted(paths)}


def _is_local_port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _latest_checkpoint_file() -> dict[str, Any] | None:
    model_dir = PROJECT_ROOT / "models"
    if not model_dir.exists():
        return None
    files = [path for path in model_dir.rglob("*") if path.is_file()]
    if not files:
        return None
    latest = max(files, key=lambda path: path.stat().st_mtime)
    stat = latest.stat()
    return {
        "path": str(latest.relative_to(PROJECT_ROOT)),
        "modified_at": stat.st_mtime,
        "modified_age_seconds": round(time.time() - stat.st_mtime, 1),
        "size_bytes": stat.st_size,
    }


def _pid_running(pid: int | None) -> bool:
    if not pid:
        return False
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    try:
        import ctypes

        synchronize = 0x00100000
        wait_timeout = 0x00000102
        handle = ctypes.windll.kernel32.OpenProcess(synchronize, False, int(pid))
        if not handle:
            return False
        try:
            return ctypes.windll.kernel32.WaitForSingleObject(handle, 0) == wait_timeout
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    except Exception:
        return False


def _stack_info() -> dict[str, Any] | None:
    if not STACK_FILE.exists():
        return None
    try:
        return json.loads(STACK_FILE.read_text(encoding="utf-8"))
    except Exception:
        return None


def _file_info(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    stat = path.stat()
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "modified_at": stat.st_mtime,
        "modified_age_seconds": round(time.time() - stat.st_mtime, 1),
        "size_bytes": stat.st_size,
    }


def _tail_file(path: Path, lines: int = 80) -> list[str]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    return text.splitlines()[-max(1, min(lines, 500)):]


@app.get("/api/training-status")
async def training_status():
    latest = _latest_checkpoint_file()
    stack = _stack_info()
    training_pid = None
    dashboard_pid = None
    if stack:
        training_pid = stack.get("training", {}).get("pid")
        dashboard_pid = stack.get("dashboard", {}).get("pid")
    ray_ports_open = {
        "ray_head": _is_local_port_open(6379),
        "ray_dashboard": _is_local_port_open(8265),
    }
    training_log_info = _file_info(TRAINING_LOG)
    recent_training_log = (
        training_log_info is not None
        and training_log_info["modified_age_seconds"] <= 120
        and training_log_info["size_bytes"] > 0
    )
    recent_checkpoint_write = (
        latest is not None and latest["modified_age_seconds"] <= 120
    )
    stack_training_running = _pid_running(training_pid)
    active = (
        any(ray_ports_open.values())
        or recent_checkpoint_write
        or recent_training_log
        or stack_training_running
    )
    return {
        "training_active": active,
        "inference": "active" if active else "not_detected",
        "dashboard_source": stream.source,
        "ray_ports_open": ray_ports_open,
        "stack_file": str(STACK_FILE.relative_to(PROJECT_ROOT)) if STACK_FILE.exists() else None,
        "stack_training_pid": training_pid,
        "stack_training_running": stack_training_running,
        "stack_dashboard_pid": dashboard_pid,
        "stack_dashboard_running": _pid_running(dashboard_pid),
        "latest_checkpoint_file": latest,
        "training_log": training_log_info,
        "dashboard_log": _file_info(DASHBOARD_LOG),
        "checkpoint_save_dir": "models/",
    }


@app.get("/api/training-log")
async def training_log(lines: int = 80):
    return {
        "available": TRAINING_LOG.exists(),
        "log": str(TRAINING_LOG.relative_to(PROJECT_ROOT)),
        "lines": _tail_file(TRAINING_LOG, lines),
    }


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    current_step = getattr(stream, "step", getattr(getattr(stream, "env", None), "current_step", 0))
    current_episode = getattr(stream, "episode", 1)
    return "\n".join(
        [
            "# HELP warehouse_dashboard_up Dashboard process health.",
            "# TYPE warehouse_dashboard_up gauge",
            "warehouse_dashboard_up 1",
            "# HELP warehouse_dashboard_step Current dashboard simulation step.",
            "# TYPE warehouse_dashboard_step gauge",
            f"warehouse_dashboard_step {current_step}",
            "# HELP warehouse_dashboard_episode Current dashboard episode.",
            "# TYPE warehouse_dashboard_episode gauge",
            f"warehouse_dashboard_episode {current_episode}",
            "",
        ]
    )


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            await ws.send_text(json.dumps(stream.tick()))
            await asyncio.sleep(0.15)
    except WebSocketDisconnect:
        return


def main():
    uvicorn.run("dashboard.app:app", host="0.0.0.0", port=8080, reload=False)


if __name__ == "__main__":
    main()
