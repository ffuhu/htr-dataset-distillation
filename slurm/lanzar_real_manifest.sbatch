#!/bin/bash
#SBATCH --job-name=dd_real_manifest
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%A_%a.log

set -euo pipefail

# ============================================================================
# Training on REAL LINES from a manifest ("dataset K config indices_file seed tag"), SAME recipe
# as lanzar_nucleo_baseline_budget.sbatch (identical COMMON_OVERRIDES); the subset is set by
# overriding allowed_indices of the config.
# Used by X5/E12 (Random with one subset per seed), E10 (GW) and the pipeline/hardware control (E8).
#
#   sbatch --array=0-11%4 --job-name=dd_x5 --export=ALL,MANIFEST=slurm/manifests/x5_real.txt \
#     slurm/lanzar_real_manifest.sbatch
# ============================================================================

REPO_ROOT="${REPO_ROOT:?Set REPO_ROOT to the repository root}"
RUNS_DIR="${RUNS_DIR:?Set RUNS_DIR to the directory for runs, buffers and experts}"
PYTHON_BIN="${PYTHON_BIN:?Set PYTHON_BIN to the Python interpreter}"
: "${MANIFEST:?Set MANIFEST}"

export PYTHONUNBUFFERED=1
cd "$REPO_ROOT"

LINE="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")"
LINE="${LINE//\$RUNS_DIR/$RUNS_DIR}"  # manifests write paths as $RUNS_DIR/...
[[ -n "$LINE" ]] || { echo "[ERROR] Sin linea $((SLURM_ARRAY_TASK_ID + 1)) en $MANIFEST" >&2; exit 1; }
read -r DATASET K CFG INDICES SEED TAG <<< "$LINE"
RUN_DIR="$RUNS_DIR/${DATASET}_${TAG}_k${K}_s${SEED}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$RUN_DIR"

[[ -f "configs/${CFG}.yaml" ]] || { echo "[ERROR] Falta configs/${CFG}.yaml" >&2; exit 1; }
[[ -f "$INDICES" ]] || { echo "[ERROR] Falta $INDICES" >&2; exit 1; }

MINI_BS="$K"
ACCUM_OVERRIDE=()
if [[ "$K" -gt 32 ]]; then
  MINI_BS=32
  ACCUM=$(( (K + MINI_BS - 1) / MINI_BS ))
  ACCUM_OVERRIDE=("+trainer.accumulate_grad_batches=${ACCUM}")
fi

COMMON_OVERRIDES=(
  "models@model_wrapper=CRNN_CTC_Wrapper_reduced_wr8"
  "extra._target_=extras.bias_against_blank"
  "+extra.blank_bias=-5.0"
  "+optimizer.lr=3e-3"
  "trainer.max_epochs=4000"
  "trainer.check_val_every_n_epoch=20"
  "trainer_callbacks.early_stopping.patience=30"
  "batch_size=${MINI_BS}"
  "${ACCUM_OVERRIDE[@]}"
)

echo "[INFO] === ${TAG} ${DATASET} K=${K} seed=${SEED} config=${CFG} indices=${INDICES} (manifiesto ${MANIFEST}, tarea ${SLURM_ARRAY_TASK_ID}) ==="
"$PYTHON_BIN" src/run.py --config-name "$CFG" \
  "${COMMON_OVERRIDES[@]}" \
  "data_module.train_datasets.${DATASET}.filter_func.allowed_indices=\${load:${INDICES}}" \
  seed="${SEED}" \
  dataset_name="${DATASET}_${TAG}_k${K}_s${SEED}" \
  trainer_callbacks.checkpointer.dirpath="$RUN_DIR/weights_s${SEED}" \
  hydra.run.dir="$RUN_DIR/hydra_s${SEED}"

echo "[INFO] Evaluacion ${TAG} ${DATASET} K=${K} seed=${SEED} completada."
