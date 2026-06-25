#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${EXPERIMENT_DIR}/.." && pwd)"

MODES="${MODES:-recurrent}"
NODE_INDEX_ENCODINGS="${NODE_INDEX_ENCODINGS:-constant}"
NODE_COUNTS="${NODE_COUNTS:-10 20 50 100}" # 1000 10000 100000
GNN_LAYERS="${GNN_LAYERS:-1 2 3}"
NUM_TRAIN="${NUM_TRAIN:-1000}"
NUM_VAL="${NUM_VAL:-256}"
NUM_TEST="${NUM_TEST:-256}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-32}"
DEVICE="${DEVICE:-0}"
CFG="${CFG:-./configs/GIN.yml}"

cd "${EXPERIMENT_DIR}"

for mode in ${MODES}; do
  for encoding in ${NODE_INDEX_ENCODINGS}; do
    for node_count in ${NODE_COUNTS}; do
      for gnn_layers in ${GNN_LAYERS}; do
        run_name="node-index-${encoding}-mode-${mode}-n${node_count}-layers${gnn_layers}"
        echo "Running ${run_name}"

    PYTHONPATH="${REPO_ROOT}/clrs${PYTHONPATH:+:${PYTHONPATH}}" \
      python star_graph_experiment.py \
        --cfg "${CFG}" \
        --execution_mode "${mode}" \
        --node_index_encoding "${encoding}" \
        --min_nodes "${node_count}" \
        --max_nodes "${node_count}" \
        --test_min_nodes "${node_count}" \
        --test_max_nodes "${node_count}" \
        --num_train "${NUM_TRAIN}" \
        --num_val "${NUM_VAL}" \
        --num_test "${NUM_TEST}" \
        --gnn_layers "${gnn_layers}" \
        --epochs "${EPOCHS}" \
        --batch_size "${BATCH_SIZE}" \
        --devices "${DEVICE}" \
        --run_name_suffix "${run_name}" \
        --runs 2
  done
done
done
done