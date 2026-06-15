#!/usr/bin/env bash
# set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${EXPERIMENT_DIR}/.." && pwd)"

NODE_COUNTS="${NODE_COUNTS:-4 8 12 16 20 24}" # Test for various number of nodes
EXECUTION_MODES="${EXECUTION_MODES:-recurrent direct_output}"
GNN_LAYERS="${GNN_LAYERS:-2}"
EPOCHS="${EPOCHS:-100}"
RUNS="${RUNS:-2}"
DEVICE="${DEVICE:-2}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "${EXPERIMENT_DIR}"

for num_nodes in ${NODE_COUNTS}; do
  for execution_mode in ${EXECUTION_MODES}; do
    if [[ "${execution_mode}" == "recurrent" ]]; then
      hint_loss=1.0
    else
      hint_loss=0.0
    fi

    echo "Running Dijkstra: mode=${execution_mode}, nodes=${num_nodes}"

    PYTHONPATH="${REPO_ROOT}/clrs${PYTHONPATH:+:${PYTHONPATH}}" \
      "${PYTHON_BIN}" train.py \
        --algorithm dijkstra \
        --cfg ./configs/dijkstra/GINE.yml \
        --graph_topology star \
        --num_nodes "${num_nodes}" \
        --shuffle_node_labels \
        --execution_mode "${execution_mode}" \
        --loss_weight_hint "${hint_loss}" \
        --gnn_layers "${GNN_LAYERS}" \
        --epochs "${EPOCHS}" \
        --runs "${RUNS}" \
        --devices "${DEVICE}"
  done
done

for num_nodes in ${NODE_COUNTS}; do
  for execution_mode in ${EXECUTION_MODES}; do
    if [[ "${execution_mode}" == "recurrent" ]]; then
      hint_loss=1.0
    else
      hint_loss=0.0
    fi

    echo "Running DFS: mode=${execution_mode}, nodes=${num_nodes}"

    PYTHONPATH="${REPO_ROOT}/clrs${PYTHONPATH:+:${PYTHONPATH}}" \
      "${PYTHON_BIN}" train.py \
        --algorithm dfs \
        --cfg ./configs/dfs/GIN.yml \
        --graph_topology star \
        --num_nodes "${num_nodes}" \
        --shuffle_node_labels \
        --execution_mode "${execution_mode}" \
        --loss_weight_hint "${hint_loss}" \
        --gnn_layers "${GNN_LAYERS}" \
        --epochs "${EPOCHS}" \
        --runs "${RUNS}" \
        --devices "${DEVICE}"
  done
done
