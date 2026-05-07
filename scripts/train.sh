#!/bin/bash
# train.sh — Launch MAPPO training with Ray RLlib
set -e

echo "🏭 Warehouse RL — Starting MAPPO Training"
echo "==========================================="

# Set PYTHONPATH
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

# Parse args
ALGORITHM=${1:-"mappo"}
CONFIG=${2:-"configs/training_config.yaml"}
WORKERS=${3:-4}

echo "Algorithm : $ALGORITHM"
echo "Config    : $CONFIG"
echo "Workers   : $WORKERS"
echo ""

case "$ALGORITHM" in
  mappo)
    python -m training.train_mappo \
      --config "$CONFIG" \
      --num-workers "$WORKERS" \
      --wandb
    ;;
  independent)
    python -m training.train_independent \
      --config "$CONFIG" \
      --num-workers "$WORKERS"
    ;;
  *)
    echo "Unknown algorithm: $ALGORITHM"
    echo "Usage: ./scripts/train.sh [mappo|independent] [config] [workers]"
    exit 1
    ;;
esac
