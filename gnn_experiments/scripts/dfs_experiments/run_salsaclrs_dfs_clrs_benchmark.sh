#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${EXPERIMENT_DIR}/.." && pwd)"
SAMPLE_COMPLEXITY_DIR="${EXPERIMENT_DIR}/sample_complexity"
BASELINES_DIR="${SAMPLE_COMPLEXITY_DIR}/baselines"

SEED="${SEED:-42}"
CFG="${CFG:-baselines/configs/dfs/GIN.yml}"
DATA_DIR="${DATA_DIR:-${EXPERIMENT_DIR}/data/CLRS}"
NUM_TRAIN="${NUM_TRAIN:-1000}"
NUM_VAL="${NUM_VAL:-32}"
NUM_TEST="${NUM_TEST:-32}"
USE_WANDB="${USE_WANDB:-0}"

cd "${SAMPLE_COMPLEXITY_DIR}"

WANDB_ARGS=()
if [[ "${USE_WANDB}" == "1" ]]; then
  WANDB_ARGS+=(--enable-wandb)
fi

echo "Running SALSA-CLRS baseline DFS on original CLRS benchmark data"
echo "DATA_DIR=${DATA_DIR}"
echo "NUM_TRAIN=${NUM_TRAIN}, NUM_VAL=${NUM_VAL}, NUM_TEST=${NUM_TEST}, SEED=${SEED}"

PYTHONPATH="${BASELINES_DIR}:${EXPERIMENT_DIR}:${REPO_ROOT}/clrs${PYTHONPATH:+:${PYTHONPATH}}" \
  python baselines/run_experiment.py \
    --cfg "${CFG}" \
    --seed "${SEED}" \
    --data-dir "${DATA_DIR}" \
    --dataset-source benchmark \
    --hints \
    "${WANDB_ARGS[@]}" \
    --size "${NUM_TRAIN}" \
    --num-val "${NUM_VAL}" \
    --num-test "${NUM_TEST}" \
    --algorithm dfs
