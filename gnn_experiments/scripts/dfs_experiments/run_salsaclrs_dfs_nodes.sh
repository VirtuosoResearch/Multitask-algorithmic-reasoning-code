#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${EXPERIMENT_DIR}/.." && pwd)"
SAMPLE_COMPLEXITY_DIR="${EXPERIMENT_DIR}/sample_complexity"
BASELINES_DIR="${SAMPLE_COMPLEXITY_DIR}/baselines"

NODES="${NODES:-16 80 160}"
SIZE="${SIZE:-1000 2000 5000 10000}"
SEED="${SEED:-42}"
CFG="${CFG:-baselines/configs/dfs/GIN.yml}"
DATA_DIR="${DATA_DIR:-${SAMPLE_COMPLEXITY_DIR}/data}"
USE_WANDB="${USE_WANDB:-0}"

cd "${SAMPLE_COMPLEXITY_DIR}"

WANDB_ARGS=()
if [[ "${USE_WANDB}" == "1" ]]; then
  WANDB_ARGS+=(--enable-wandb)
fi

for node in ${NODES}; do
  for size in ${SIZE}; do
  echo "Running SALSA-CLRS DFS: nodes=${node}, size=${size}, seed=${SEED}"

  PYTHONPATH="${BASELINES_DIR}:${REPO_ROOT}/clrs${PYTHONPATH:+:${PYTHONPATH}}" \
    python baselines/run_experiment.py \
      --cfg "${CFG}" \
      --seed "${SEED}" \
      --data-dir "${DATA_DIR}" \
      --hints \
      "${WANDB_ARGS[@]}" \
      --size "${size}" \
      --node "${node}" \
      --algorithm dfs
done
done
