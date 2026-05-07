# Multi-Agent Autonomous Warehouse RL

Multi-agent reinforcement learning project for coordinating autonomous warehouse robots. The system builds a custom Gymnasium/RLlib-compatible warehouse environment, trains robot policies with PPO-style multi-agent training, evaluates checkpoints, and includes a live FastAPI dashboard demo.

The main learning problem is simple to understand but rich enough for real coordination research: multiple robots navigate a grid warehouse, pick items from shelves, deliver orders, manage battery usage, avoid collisions, and improve through curriculum learning.

## Core Features

- Custom multi-agent warehouse environment with RLlib `MultiAgentEnv` semantics.
- Configurable grid world with shelves, charging stations, delivery zones, obstacles, and robot occupancy.
- Robot state machine for movement, pickup, delivery, charging, battery drain, collisions, and task assignment.
- Dynamic order manager with pending, assigned, completed, and failed order tracking.
- Reward shaping for deliveries, progress, collisions, idle time, deadline misses, and energy waste.
- MAPPO-style shared-policy training using Ray RLlib PPO.
- Independent PPO baseline for comparison against shared multi-agent learning.
- Curriculum manager with five progressive difficulty stages.
- Evaluation script for checkpoint rollouts and benchmark comparisons.
- FastAPI + WebSocket dashboard for live warehouse visualization.
- Docker and Kubernetes deployment assets.

## Project Structure

```text
warehouse-rl-final/
|-- configs/
|   |-- env_config.yaml
|   |-- mappo_config.yaml
|   |-- curriculum_config.yaml
|   `-- training_config.yaml
|-- dashboard/
|   |-- app.py
|   `-- index.html
|-- deployment/
|   |-- docker/
|   |-- kubernetes/
|   `-- ci-cd/
|-- docs/
|   `-- BUILD_PLAN.md
|-- scripts/
|   |-- run_full_system.ps1
|   |-- run_dashboard.ps1
|   |-- stop_full_system.ps1
|   |-- watch_logs.ps1
|   |-- run_training.sh
|   |-- train.sh
|   `-- evaluate.sh
|-- src/
|   |-- agents/
|   |-- communication/
|   |-- curriculum/
|   |-- environment/
|   |-- training/
|   `-- utils/
|-- tests/
|-- requirements.txt
|-- setup.py
|-- test_env.py
`-- test_env_sanity.py
```

## Main Components

### Environment

`src/environment/warehouse_env.py` defines `WarehouseEnv`, the central multi-agent environment. Each robot receives a vector observation made from a local grid view, robot state, order queue features, and communication features.

Default actions are discrete:

```text
0 = North
1 = South
2 = East
3 = West
4 = Pick
5 = Deliver
6 = Charge
7 = Stay
```

The physical world is handled by:

- `grid_world.py` - warehouse layout, cell types, local observations, spatial queries.
- `robot.py` - robot state, battery model, task state, pickup/delivery logic.
- `order_manager.py` - order generation, assignment, completion, and queue metrics.
- `reward_shaper.py` - individual and team reward calculation.

### Training

Training code lives in `src/training/`.

- `train_mappo.py` trains a shared policy for all robots with Ray RLlib PPO.
- `train_independent.py` trains an independent PPO baseline with one policy per robot.
- `callbacks.py` exports warehouse-specific throughput, completion, collision, and order metrics.
- `evaluate.py` restores checkpoints and runs benchmark rollouts.

### Curriculum

`src/curriculum/curriculum_manager.py` implements staged difficulty progression. Stages increase robot count, obstacle density, order pressure, battery requirements, deadlines, and priority orders.

### Communication And Networks

- `src/communication/comm_channel.py` stores and broadcasts fixed-size inter-agent messages.
- `src/agents/network_architectures.py` contains PyTorch actor, critic, spatial encoder, state encoder, and communication encoder modules.

### Dashboard

`dashboard/app.py` serves a FastAPI app with REST endpoints and a WebSocket stream. By default it runs the real `WarehouseEnv` with a deterministic heuristic policy so the warehouse grid, robots, batteries, orders, and activity log move live in the browser.

Important: opening the dashboard does not start RL training. It is a live environment rollout/visualization. Real training is started separately with the training commands below.

Useful dashboard endpoints:

- `/health` - server health and active stream source.
- `/api/state` - one current warehouse state tick.
- `/api/training-status` - whether Ray/checkpoint activity suggests training is active.
- `/api/training-log` - recent lines from `logs/training.log`.
- `/api/checkpoints` - saved checkpoint folders under `models/`.
- `/metrics` - Prometheus-style dashboard metrics.
- `/ws` - live WebSocket stream consumed by `index.html`.

## Requirements

- Python 3.10 or newer.
- Python 3.11.x is recommended for the pinned dependency set.
- Windows, macOS, or Linux. This copy was prepared from a Windows workspace.

Install dependencies from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On macOS or Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Quick Sanity Checks

Run these from the repository root:

```bash
python test_env.py
python test_env_sanity.py
pytest tests -v
```

`test_env.py` checks that the environment imports and resets. `test_env_sanity.py` manually verifies pickup and delivery behavior.

## Run Full System

On Windows PowerShell, start the dashboard and real MAPPO training together:

```powershell
.\scripts\run_full_system.ps1
```

This starts two background processes:

- Dashboard: `http://127.0.0.1:8080`
- Training: `python -m src.training.train_mappo ...`

Runtime files:

```text
logs/dashboard.log
logs/training.log
.run/warehouse-stack.json
```

Watch logs:

```powershell
.\scripts\watch_logs.ps1 -Target all
.\scripts\watch_logs.ps1 -Target training
.\scripts\watch_logs.ps1 -Target dashboard
```

Stop both processes:

```powershell
.\scripts\stop_full_system.ps1
```

The dashboard activity log polls `/api/training-status` and `/api/training-log`, so it can show whether real training is detected and surface recent training output.

## Training

Start MAPPO-style shared-policy training:

```bash
python -m src.training.train_mappo --train-config configs/mappo_config.yaml --env-config configs/env_config.yaml --curriculum-config configs/curriculum_config.yaml
```

On Windows PowerShell with the included virtual environment:

```powershell
.\rl-env\Scripts\python.exe -m src.training.train_mappo --train-config configs/mappo_config.yaml --env-config configs/env_config.yaml --curriculum-config configs/curriculum_config.yaml
```

Resume from a checkpoint:

```bash
python -m src.training.train_mappo --resume --checkpoint models/mappo/checkpoint_0050
```

Train the independent PPO baseline:

```bash
python -m src.training.train_independent --iterations 200
```

Training artifacts are written under `models/` by default. That folder is intentionally ignored by Git because checkpoints can become large.

MAPPO checkpoints are saved according to `checkpoint_freq` in `configs/mappo_config.yaml`, currently under:

```text
models/mappo/
```

The independent PPO baseline saves under:

```text
models/ippo_baseline/
```

To confirm whether training is running, look for Ray ports/processes and recent checkpoint writes. The dashboard also exposes this check at:

```text
http://127.0.0.1:8080/api/training-status
```

When training is started through `run_full_system.ps1`, live output is written to:

```text
logs/training.log
```

## Evaluation

Evaluate one checkpoint:

```bash
python -m src.training.evaluate --checkpoint models/mappo/final_model --episodes 50
```

Run the default benchmark comparison:

```bash
python -m src.training.evaluate --benchmark --episodes 50
```

Evaluation output is saved to `eval_results.json`, which is ignored by Git as a generated result file.

## Dashboard

Start the local dashboard:

```powershell
.\scripts\run_dashboard.ps1
```

Or run it directly:

```powershell
.\rl-env\Scripts\python.exe -m uvicorn dashboard.app:app --host 127.0.0.1 --port 8080 --reload
```

Then open:

```text
http://127.0.0.1:8080
```

Health check:

```text
http://127.0.0.1:8080/health
```

Training status:

```text
http://127.0.0.1:8080/api/training-status
```

## Deployment Assets

Docker files are in `deployment/docker/`:

```bash
docker compose -f deployment/docker/docker-compose.yml up --build
```

Kubernetes manifests are in `deployment/kubernetes/`.

GitHub Actions workflow files are stored under `deployment/ci-cd/.github/workflows/`. If you want GitHub to run those automatically, copy the workflow files into a root-level `.github/workflows/` folder before publishing.

## Repository Hygiene

The `.gitignore` is set up to keep local and generated files out of GitHub, including:

- `.env` and other secret files.
- The local virtual environment folder `rl-env/`.
- Python caches and package metadata.
- W&B runs, Ray results, logs, checkpoints, and trained models.
- Evaluation output such as `eval_results.json`.
- Accidental PowerShell brace-expansion folders.

Before uploading manually, make sure you do not include local runtime folders such as `rl-env/`, `wandb/`, `logs/`, `models/`, or any real `.env` file.

## Notes

- Commands above assume they are run from the repository root.
- W&B logging is enabled when `wandb` is installed and configured. Use your own local `.env` or W&B login, but do not publish credentials.
- No license file is currently included. Add one before making the repository public if you want others to reuse the code.
