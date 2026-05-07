#!/usr/bin/env bash
# run_training.sh — Launch full MAPPO training
set -euo pipefail

echo "🏭 Warehouse RL — Starting MAPPO Training"
echo "============================================"

# Export src to PYTHONPATH
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

# Default args
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/mappo_config.yaml}"
ENV_CONFIG="${ENV_CONFIG:-configs/env_config.yaml}"
CURRICULUM_CONFIG="${CURRICULUM_CONFIG:-configs/curriculum_config.yaml}"
RESUME="${RESUME:-false}"
CHECKPOINT="${CHECKPOINT:-}"

echo "Train Config : $TRAIN_CONFIG"
echo "Env Config   : $ENV_CONFIG"
echo "Curriculum   : $CURRICULUM_CONFIG"
echo ""

# Launch training
python src/training/train_mappo.py \
  --train-config "$TRAIN_CONFIG" \
  --env-config "$ENV_CONFIG" \
  --curriculum-config "$CURRICULUM_CONFIG" \
  ${RESUME:+--resume} \
  ${CHECKPOINT:+--checkpoint "$CHECKPOINT"}
