"""
app.py — FastAPI + WebSocket Dashboard Backend
Serves the live warehouse simulation dashboard with real-time metrics streaming.
"""

from __future__ import annotations
import asyncio
import json
import random
import math
import time
from typing import Dict, Any, Set
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
import uvicorn

app = FastAPI(title="Warehouse RL Dashboard", version="1.0.0")

# Mount static files
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)

# ── Connection Manager ──────────────────────────────────────────────────────
class ConnectionManager:
    def __init__(self):
        self.active: Set[WebSocket] = set()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.add(ws)

    def disconnect(self, ws: WebSocket):
        self.active.discard(ws)

    async def broadcast(self, data: dict):
        msg = json.dumps(data)
        dead = set()
        for ws in self.active:
            try:
                await ws.send_text(msg)
            except Exception:
                dead.add(ws)
        self.active -= dead

manager = ConnectionManager()

# ── Simulation State ────────────────────────────────────────────────────────
class WarehouseSimulator:
    """
    Lightweight simulation for dashboard demo.
    In production, replace with actual WarehouseEnv step() calls.
    """
    GRID_W, GRID_H = 20, 20
    NUM_ROBOTS = 10

    # Cell types
    EMPTY, OBSTACLE, SHELF, CHARGING, DELIVERY, ROBOT = 0, 1, 2, 3, 4, 5

    def __init__(self):
        self.step = 0
        self.episode = 1
        self.grid = self._build_grid()
        self.robots = self._spawn_robots()
        self.orders_completed = 0
        self.total_collisions = 0
        self.throughput_history = []
        self.reward_history = []
        self.battery_history = []
        self.rng = random.Random(42)
        self._metrics_window = []
        self.episode_reward = 0.0

    def _build_grid(self):
        g = [[self.EMPTY]*self.GRID_W for _ in range(self.GRID_H)]
        # Shelves in rows
        for row in range(2, 14, 3):
            for col in range(2, 19, 2):
                if row < self.GRID_H and col < self.GRID_W:
                    g[row][col] = self.SHELF
        # Charging corners
        for cx, cy in [(1,1),(18,1),(1,18),(18,18)]:
            g[cy][cx] = self.CHARGING
        # Delivery zones bottom
        for dx in [5, 10, 15]:
            g[18][dx] = self.DELIVERY
        # Obstacles
        obstacles = [(3,5),(7,8),(12,5),(16,10),(4,13),(9,12),(14,3),(6,17)]
        for ox, oy in obstacles:
            g[oy][ox] = self.OBSTACLE
        return g

    def _spawn_robots(self):
        robots = []
        spawn_pts = [(2,17),(4,17),(6,17),(8,17),(10,17),(12,17),(14,17),(2,16),(5,16),(8,16)]
        states = ["IDLE","MOVING","PICKING","DELIVERING","CHARGING"]
        for i, (x, y) in enumerate(spawn_pts[:self.NUM_ROBOTS]):
            robots.append({
                "id": i,
                "x": x, "y": y,
                "battery": random.uniform(60, 100),
                "state": "IDLE",
                "task": None,
                "deliveries": 0,
                "collisions": 0,
                "vx": 0, "vy": 0,
            })
        return robots

    def tick(self) -> dict:
        self.step += 1
        if self.step > 500:
            self.step = 0
            self.episode += 1
            self.robots = self._spawn_robots()
            self.orders_completed = 0

        # Simulate robot movement
        for r in self.robots:
            r["battery"] = max(5.0, r["battery"] - self.rng.uniform(0.1, 0.6))
            if r["battery"] < 20:
                r["state"] = "CHARGING"
                # Move toward charger
                target_x, target_y = 1, 1
                if r["x"] > target_x: r["x"] -= 1
                elif r["x"] < target_x: r["x"] += 1
                if r["y"] > target_y: r["y"] -= 1
                elif r["y"] < target_y: r["y"] += 1
                r["battery"] = min(100, r["battery"] + 8)
            else:
                states = ["MOVING", "MOVING", "MOVING", "PICKING", "DELIVERING", "IDLE"]
                r["state"] = self.rng.choice(states)
                dx = self.rng.choice([-1, 0, 0, 1])
                dy = self.rng.choice([-1, 0, 0, 1])
                nx = max(0, min(self.GRID_W-1, r["x"] + dx))
                ny = max(0, min(self.GRID_H-1, r["y"] + dy))
                if self.grid[ny][nx] not in (self.OBSTACLE, self.SHELF):
                    r["x"], r["y"] = nx, ny
                if r["state"] == "DELIVERING" and self.rng.random() < 0.08:
                    r["deliveries"] += 1
                    self.orders_completed += 1

        # Metrics
        avg_battery = sum(r["battery"] for r in self.robots) / len(self.robots)
        throughput = self.orders_completed / max(1, self.step) * 100
        reward = throughput * 0.1 + avg_battery * 0.05 + self.rng.uniform(-0.5, 1.5)
        self.episode_reward += reward * 0.01

        self.throughput_history.append(round(throughput, 2))
        self.reward_history.append(round(self.episode_reward, 2))
        self.battery_history.append(round(avg_battery, 1))

        if len(self.throughput_history) > 100:
            self.throughput_history.pop(0)
            self.reward_history.pop(0)
            self.battery_history.pop(0)

        return {
            "type": "state",
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
            }
        }

sim = WarehouseSimulator()

# ── Routes ──────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root():
    html_file = Path(__file__).parent / "index.html"
    return HTMLResponse(html_file.read_text())

@app.get("/health")
async def health():
    return {"status": "ok", "step": sim.step, "episode": sim.episode}

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:
            state = sim.tick()
            await manager.broadcast(state)
            await asyncio.sleep(0.15)  # ~6fps simulation
    except WebSocketDisconnect:
        manager.disconnect(ws)
    except Exception as e:
        manager.disconnect(ws)

def main():
    uvicorn.run("dashboard.app:app", host="0.0.0.0", port=8080, reload=False)

if __name__ == "__main__":
    main()
