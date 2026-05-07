#!/bin/bash
# evaluate.sh — Run evaluation and benchmark
set -e
export PYTHONPATH="$(pwd):$(pwd)/src:$PYTHONPATH"

CHECKPOINT=${1:-""}
EPISODES=${2:-50}

echo "🔬 Running benchmark comparison..."
python -m src.training.evaluate --benchmark --episodes "$EPISODES"

if [ -n "$CHECKPOINT" ]; then
  echo "📊 Evaluating checkpoint: $CHECKPOINT"
  python -m src.training.evaluate --checkpoint "$CHECKPOINT" --episodes "$EPISODES"
fi
