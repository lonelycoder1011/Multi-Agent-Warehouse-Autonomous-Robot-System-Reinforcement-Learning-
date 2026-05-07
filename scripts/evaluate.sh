#!/bin/bash
# evaluate.sh — Run evaluation and benchmark
set -e
export PYTHONPATH="$(pwd):$(pwd)/src:$PYTHONPATH"

CHECKPOINT=${1:-""}
EPISODES=${2:-50}

echo "🔬 Running benchmark comparison..."
python src/training/evaluate.py --benchmark --episodes "$EPISODES"

if [ -n "$CHECKPOINT" ]; then
  echo "📊 Evaluating checkpoint: $CHECKPOINT"
  python src/training/evaluate.py --checkpoint "$CHECKPOINT" --episodes "$EPISODES"
fi