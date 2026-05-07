# 🏭 Multi-Agent Autonomous Warehouse RL — Master Build Prompts

> A production-grade, hiring-level Reinforcement Learning project using Gymnasium, Ray RLlib, MAPPO, and an ultra-premium visualization dashboard.

---

## 📐 Project Architecture Overview

```
warehouse-rl/
├── src/
│   ├── env/                    # Custom Gymnasium Multi-Agent Environment
│   │   ├── warehouse_env.py    # Core environment
│   │   ├── grid_world.py       # Grid logic, obstacles, zones
│   │   ├── robot.py            # Robot agent logic
│   │   ├── order_manager.py    # Dynamic order queue
│   │   └── reward_shaper.py    # Advanced reward shaping
│   ├── agents/
│   │   ├── mappo_agent.py      # MAPPO implementation
│   │   ├── independent_agent.py# Independent learner baseline
│   │   └── comm_agent.py       # Agent with communication channel
│   ├── training/
│   │   ├── trainer.py          # RLlib training loop
│   │   ├── curriculum.py       # Curriculum learning scheduler
│   │   └── callbacks.py        # WandB + metrics callbacks
│   └── utils/
│       ├── config.py           # Hydra config management
│       ├── metrics.py          # Performance metrics
│       └── logger.py           # Structured logging
├── dashboard/
│   ├── app.py                  # FastAPI backend
│   ├── static/
│   │   └── index.html          # Glassmorphism UI dashboard
│   └── ws_handler.py           # WebSocket real-time updates
├── tests/
│   ├── test_env.py
│   ├── test_agents.py
│   └── test_training.py
├── deployment/
│   ├── docker/
│   │   ├── Dockerfile.training
│   │   ├── Dockerfile.dashboard
│   │   └── docker-compose.yml
│   ├── k8s/
│   │   ├── namespace.yaml
│   │   ├── training-job.yaml
│   │   ├── dashboard-deploy.yaml
│   │   ├── dashboard-service.yaml
│   │   └── hpa.yaml
│   └── ci-cd/
│       └── pipeline.yml        # GitHub Actions CI/CD
├── configs/
│   ├── env_config.yaml
│   ├── training_config.yaml
│   └── mappo_config.yaml
├── scripts/
│   ├── train.sh
│   ├── evaluate.sh
│   └── visualize.sh
├── notebooks/
│   └── analysis.ipynb
├── requirements.txt
├── setup.py
└── README.md
```

---

## 🔨 Build Prompts — Step by Step

---

### PROMPT 1 — Project Scaffolding & Dependencies

**Goal:** Create the full directory structure, `requirements.txt`, `setup.py`, and `configs/`.

**What to build:**
- Full directory tree
- `requirements.txt` with pinned versions: ray[rllib], gymnasium, torch, fastapi, uvicorn, wandb, hydra-core, numpy, pygame, websockets
- `setup.py` for installable package
- `configs/env_config.yaml` — grid size, num robots, obstacle density, charging stations
- `configs/training_config.yaml` — episodes, rollout workers, batch size, learning rate
- `configs/mappo_config.yaml` — MAPPO-specific hyperparameters

---

### PROMPT 2 — Custom Gymnasium Environment Core

**Goal:** Build the full `WarehouseEnv` as a custom `gymnasium.Env` / PettingZoo-compatible multi-agent env.

**What to build:**
- `grid_world.py`: 2D grid, cell types (EMPTY, OBSTACLE, SHELF, CHARGING, DELIVERY)
- `robot.py`: Robot class with position, battery, task state, collision detection
- `order_manager.py`: Dynamic order queue with priorities and deadlines
- `warehouse_env.py`: 
  - `reset()`, `step()`, `render()`, `observation_space`, `action_space`
  - Supports 5–20 robots
  - Partial observability: each robot sees local NxN window
  - Full state dict for centralized critic
- `reward_shaper.py`: Composite reward (delivery +10, collision -5, energy waste -0.1/step, delay -1/step overtime)

---

### PROMPT 3 — Agent Implementations

**Goal:** Build three agent variants for comparison.

**What to build:**
- `independent_agent.py`: Each robot trains its own PPO policy — baseline
- `mappo_agent.py`: MAPPO with centralized critic that sees full global state; decentralized actors with local obs
- `comm_agent.py`: Agents broadcast compressed message vectors; attention-based aggregation of neighbor messages

---

### PROMPT 4 — Ray RLlib Training Pipeline

**Goal:** Full distributed training setup with RLlib.

**What to build:**
- `trainer.py`:
  - RLlib `MultiAgentEnv` wrapper
  - Parameter sharing configuration (all robots share weights)
  - Centralized critic via custom model
  - Distributed rollout workers (configurable)
  - Checkpoint saving every N iterations
- `curriculum.py`:
  - Phase 1: 5 robots, no obstacles, simple orders
  - Phase 2: 10 robots, 20% obstacles, order deadlines
  - Phase 3: 20 robots, 35% obstacles, priority orders, battery constraints
  - Auto-advance based on mean reward threshold
- `callbacks.py`: WandB logging, per-episode metrics (throughput, collision rate, avg battery)

---

### PROMPT 5 — Evaluation & Benchmarking

**Goal:** Rigorous comparison of Independent vs MAPPO vs Comm-MAPPO.

**What to build:**
- `evaluate.py`: Load checkpoints, run N evaluation episodes, collect metrics
- Metrics: orders/hour, collision rate, mean battery efficiency, task completion rate, coordination score
- `benchmark.py`: Side-by-side comparison table + chart generation
- Statistical significance testing (t-test across seeds)

---

### PROMPT 6 — Ultra-Premium Dashboard (Glassmorphism UI)

**Goal:** A jaw-dropping real-time visualization dashboard.

**What to build:**
- `dashboard/app.py`: FastAPI + WebSocket server streaming live simulation state
- `dashboard/static/index.html`: Single-file ultra-premium HTML/CSS/JS dashboard with:
  - **Live warehouse grid** rendered on Canvas with animated robots (color-coded by state)
  - **Glassmorphism panels** with frosted-glass effect, liquid gradients
  - **Live metrics cards**: throughput, collision rate, battery avg, orders fulfilled
  - **Training curves**: animated line charts (Chart.js)
  - **Heatmap overlay**: activity density on warehouse grid
  - **Agent state panel**: per-robot status bars
  - **Dark cyberpunk aesthetic**: deep navy background, neon cyan/violet/amber accents
  - Smooth 60fps canvas animations

---

### PROMPT 7 — Docker & Docker Compose

**Goal:** Containerize the full stack.

**What to build:**
- `Dockerfile.training`: Multi-stage build, CUDA-capable, Ray cluster ready
- `Dockerfile.dashboard`: Lightweight FastAPI server image
- `docker-compose.yml`: 
  - `ray-head` service (training)
  - `ray-worker` service (scalable)
  - `dashboard` service
  - `prometheus` + `grafana` for metrics
  - Shared volumes for checkpoints
  - Health checks, restart policies

---

### PROMPT 8 — Kubernetes Deployment

**Goal:** Production-grade K8s manifests.

**What to build:**
- `namespace.yaml`: Isolated `warehouse-rl` namespace
- `training-job.yaml`: K8s Job with Ray cluster operator, GPU node selector
- `dashboard-deploy.yaml`: Deployment with 2 replicas, rolling update strategy
- `dashboard-service.yaml`: LoadBalancer service + Ingress
- `hpa.yaml`: Horizontal Pod Autoscaler for dashboard pods
- `configmap.yaml`: Environment config injection
- `secret.yaml`: WandB API key management

---

### PROMPT 9 — CI/CD Pipeline (GitHub Actions)

**Goal:** Automated test, lint, build, and deploy pipeline.

**What to build:**
- `.github/workflows/pipeline.yml`:
  - **Lint**: `ruff`, `black --check`
  - **Test**: `pytest` with coverage report
  - **Build**: Docker image build and push to GHCR
  - **Deploy**: Auto-deploy to K8s on `main` branch push
  - **Smoke test**: Hit dashboard health endpoint post-deploy
- Badge generation for README

---

### PROMPT 10 — README.md

**Goal:** A stunning, recruiter-magnet README.

**What to build:**
- Project banner ASCII art or badge
- Badges: Python, Ray, Gymnasium, Docker, K8s, CI status, License
- Architecture diagram (ASCII or Mermaid)
- Quick start (3-command setup)
- Full feature list
- Benchmark results table (Independent vs MAPPO vs Comm-MAPPO)
- Dashboard screenshots section
- Curriculum learning progression charts
- Deployment guide
- Research references (MAPPO paper, RLlib docs)
- Contribution guide

---

## 🎯 Execution Order

| Step | File(s) | Status |
|------|---------|--------|
| 1 | Scaffolding + configs | ✅ |
| 2 | Gymnasium Environment | ✅ |
| 3 | Agent Implementations | ✅ |
| 4 | RLlib Training Pipeline | ✅ |
| 5 | Evaluation & Benchmarking | ✅ |
| 6 | Dashboard UI | ✅ |
| 7 | Docker | ✅ |
| 8 | Kubernetes | ✅ |
| 9 | CI/CD | ✅ |
| 10 | README | ✅ |

---

## 💡 Key Design Decisions

1. **PettingZoo + RLlib**: Use PettingZoo's `ParallelEnv` interface, wrapped for RLlib's `MultiAgentEnv`
2. **Parameter Sharing**: All robots share one policy network — scales to 100s of robots
3. **Centralized Training, Decentralized Execution (CTDE)**: Critic sees global state during training; actors use only local obs at execution
4. **Communication**: Differentiable communication via learned message vectors + attention
5. **Curriculum**: Automatic difficulty progression prevents training collapse
6. **Reproducibility**: All seeds fixed, configs in YAML, Docker for environment
