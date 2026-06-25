#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${EXPERIMENT_DIR}/.." && pwd)"
SAMPLE_COMPLEXITY_DIR="${EXPERIMENT_DIR}/sample_complexity"
BASELINES_DIR="${SAMPLE_COMPLEXITY_DIR}/baselines"

SEED="${SEED:-42}"
CFG="${CFG:-baselines/configs/dfs/MPNN.yml}"
DATA_DIR="${DATA_DIR:-${EXPERIMENT_DIR}/data/CLRS}"
NUM_TRAIN="${NUM_TRAIN:-1000}"
NUM_VAL="${NUM_VAL:-32}"
NUM_TEST="${NUM_TEST:-32}"
HINT_TEACHER_FORCING="${HINT_TEACHER_FORCING:-0.0}"
HINT_REPRED_MODE="${HINT_REPRED_MODE:-soft}"
USE_WANDB="${USE_WANDB:-0}"
ER_NODES="${ER_NODES:-16}"
ER_SIZES="${ER_SIZES:-1000 2000 5000 10000 20000}"

cd "${SAMPLE_COMPLEXITY_DIR}"

WANDB_ARGS=()
if [[ "${USE_WANDB}" == "1" ]]; then
  WANDB_ARGS+=(--enable-wandb)
fi

echo "Running CLRS-like MPNN DFS on original CLRS benchmark data"
echo "DATA_DIR=${DATA_DIR}"
echo "NUM_TRAIN=${NUM_TRAIN}, NUM_VAL=${NUM_VAL}, NUM_TEST=${NUM_TEST}, SEED=${SEED}"
echo "HINT_TEACHER_FORCING=${HINT_TEACHER_FORCING}, HINT_REPRED_MODE=${HINT_REPRED_MODE}"
for run in {1..3}; do
seed=$((SEED + run))
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="${BASELINES_DIR}:${EXPERIMENT_DIR}:${REPO_ROOT}/clrs${PYTHONPATH:+:${PYTHONPATH}}" \
  python baselines/run_experiment.py \
    --cfg "${CFG}" \
    --seed "${seed}" \
    --data-dir "${DATA_DIR}" \
    --dataset-source benchmark \
    --model-variant clrs_mpnn \
    --hint-teacher-forcing "${HINT_TEACHER_FORCING}" \
    --hint-repred-mode "${HINT_REPRED_MODE}" \
    --hints \
    "${WANDB_ARGS[@]}" \
    --size "${NUM_TRAIN}" \
    --num-val "${NUM_VAL}" \
    --num-test "${NUM_TEST}" \
    --algorithm dfs
done

echo "Running CLRS-like MPNN DFS on generated ER graphs"
echo "ER_NODES=${ER_NODES}, ER_SIZES=${ER_SIZES}, SEED=${SEED}"

for size in ${ER_SIZES}; do
  echo "Running generated ER DFS: nodes=${ER_NODES}, size=${size}, seed=${SEED}"
  for run in {1..3}; do
  seed=$((SEED + run))
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH="${BASELINES_DIR}:${EXPERIMENT_DIR}:${REPO_ROOT}/clrs${PYTHONPATH:+:${PYTHONPATH}}" \
    python baselines/run_experiment.py \
      --cfg "${CFG}" \
      --seed "${seed}" \
      --data-dir "${DATA_DIR}" \
      --dataset-source generated \
      --model-variant clrs_mpnn \
      --hint-teacher-forcing "${HINT_TEACHER_FORCING}" \
      --hint-repred-mode "${HINT_REPRED_MODE}" \
      --hints \
      "${WANDB_ARGS[@]}" \
      --node "${ER_NODES}" \
      --size "${size}" \
      --algorithm dfs
  done
done
