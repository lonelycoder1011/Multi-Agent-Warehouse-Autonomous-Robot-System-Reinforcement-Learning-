#!/bin/bash
# train.sh — Launch MAPPO training with Ray RLlib
set -e

echo "🏭 Warehouse RL — Starting MAPPO Training"
echo "==========================================="

# Set PYTHONPATH
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

# Parse args
ALGORITHM=${1:-"mappo"}
CONFIG=${2:-"configs/mappo_config.yaml"}
WORKERS=${3:-4}
ENV_CONFIG="${ENV_CONFIG:-configs/env_config.yaml}"
CURRICULUM_CONFIG="${CURRICULUM_CONFIG:-configs/curriculum_config.yaml}"

echo "Algorithm : $ALGORITHM"
echo "Config    : $CONFIG"
echo "Workers   : $WORKERS"
echo ""

case "$ALGORITHM" in
  mappo)
    python -m src.training.train_mappo \
      --train-config "$CONFIG" \
      --env-config "$ENV_CONFIG" \
      --curriculum-config "$CURRICULUM_CONFIG" \
      --num-workers "$WORKERS"
    ;;
  independent)
    python -m src.training.train_independent \
      --train-config "$CONFIG" \
      --env-config "$ENV_CONFIG" \
      --num-workers "$WORKERS"
    ;;
  *)
    echo "Unknown algorithm: $ALGORITHM"
    echo "Usage: ./scripts/train.sh [mappo|independent] [config] [workers]"
    exit 1
    ;;
esac
