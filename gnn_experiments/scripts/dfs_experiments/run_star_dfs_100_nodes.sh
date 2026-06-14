#!/usr/bin/env bash
# set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${EXPERIMENT_DIR}/.." && pwd)"

NUM_TRAINS="${NUM_TRAINS:-2000 5000 10000}"
NODE_COUNTS="${NODE_COUNTS:-100 1000 10000 100000}"
GNN_LAYERS="${GNN_LAYERS:-1 2 3}"
NUM_VAL="${NUM_VAL:-256}"
NUM_TEST="${NUM_TEST:-256}"
EPOCHS="${EPOCHS:-100}"
DEVICE="${DEVICE:-0}"

cd "${EXPERIMENT_DIR}"

for num_train in ${NUM_TRAINS}; do
  for num_nodes in ${NODE_COUNTS}; do
    for layers in ${GNN_LAYERS}; do
      run_name="train${num_train}-n${num_nodes}-layers${layers}"
      echo "Running ${run_name}"

      PYTHONPATH="${REPO_ROOT}/clrs${PYTHONPATH:+:${PYTHONPATH}}" \
        python star_graph_experiment.py \
          --cfg ./configs/GIN.yml \
          --min_nodes "${num_nodes}" \
          --max_nodes "${num_nodes}" \
          --test_min_nodes "${num_nodes}" \
          --test_max_nodes "${num_nodes}" \
          --num_train "${num_train}" \
          --num_val "${NUM_VAL}" \
          --num_test "${NUM_TEST}" \
          --gnn_layers "${layers}" \
          --epochs "${EPOCHS}" \
          --devices "${DEVICE}" \
          --run_name_suffix "${run_name}"
    done
  done
done
