#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${EXPERIMENT_DIR}/.." && pwd)"

NUM_TRAIN="${NUM_TRAIN:-1000}"
NUM_VAL="${NUM_VAL:-32}"
NUM_TEST="${NUM_TEST:-32}"
GNN_LAYERS="${GNN_LAYERS:-1 2 3}"
EPOCHS="${EPOCHS:-100}"
DEVICE="${DEVICE:-0}"
SEED="${SEED:-42}"
CFG="${CFG:-./configs/GIN.yml}"
MODES="${MODES:-single_pass recurrent}"

cd "${EXPERIMENT_DIR}"

for mode in ${MODES}; do
for gnn_layers in ${GNN_LAYERS}; do
  run_name="random-dfs-${mode}-train${NUM_TRAIN}-layers${gnn_layers}-seed${SEED}"
  echo "Running ${run_name}"

  PYTHONPATH="${REPO_ROOT}/clrs${PYTHONPATH:+:${PYTHONPATH}}" \
    python train.py \
      --algorithm dfs \
      --cfg "${CFG}" \
      --graph_topology dataset \
      --execution_mode "${mode}" \
      --num_train "${NUM_TRAIN}" \
      --num_val "${NUM_VAL}" \
      --num_test "${NUM_TEST}" \
      --gnn_layers "${gnn_layers}" \
      --epochs "${EPOCHS}" \
      --devices "${DEVICE}" \
      --seed "${SEED}" \
      --runs 2
done
done