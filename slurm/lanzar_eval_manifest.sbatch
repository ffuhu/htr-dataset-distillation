#!/bin/bash
#SBATCH --job-name=dd_eval_manifest
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%A_%a.log

set -euo pipefail

# ============================================================================
# Generic evaluation of buffers from a manifest (one line per task:
# "dataset K buffer_path seed tag"), SAME recipe as lanzar_distill_eval_noclass_budget.sbatch
# (identical COMMON_OVERRIDES).
#
#   sbatch --array=0-23%2 --job-name=dd_x1_eval_u30 --dependency=aftercorr:<x1> \
#     --export=ALL,MANIFEST=slurm/manifests/x1_eval_u30.txt slurm/lanzar_eval_manifest.sbatch
# ============================================================================

REPO_ROOT="${REPO_ROOT:?Set REPO_ROOT to the repository root}"
RUNS_DIR="${RUNS_DIR:?Set RUNS_DIR to the directory for runs, buffers and experts}"
# Python interpreter of the environment (see README).
PYTHON_BIN="${PYTHON_BIN:?Set PYTHON_BIN to the Python interpreter}"
: "${MANIFEST:?Set MANIFEST}"

export PYTHONUNBUFFERED=1
cd "$REPO_ROOT"

LINE="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")"
LINE="${LINE//\$RUNS_DIR/$RUNS_DIR}"  # manifests write paths as $RUNS_DIR/...
[[ -n "$LINE" ]] || { echo "[ERROR] Sin linea $((SLURM_ARRAY_TASK_ID + 1)) en $MANIFEST" >&2; exit 1; }
read -r DATASET K BUFFER_PATH SEED TAG <<< "$LINE"
RUN_DIR="$RUNS_DIR/${DATASET}_${TAG}_k${K}_s${SEED}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$RUN_DIR"

[[ -f "$BUFFER_PATH" ]] || { echo "[ERROR] No existe el buffer: $BUFFER_PATH" >&2; exit 1; }

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
  "+data/augmentation@data_module.train_datasets.synthetic.img_transform=default_tensor"
  "batch_size=${MINI_BS}"
  "${ACCUM_OVERRIDE[@]}"
)

echo "[INFO] === ${TAG} ${DATASET} K=${K} seed=${SEED} buffer=${BUFFER_PATH} (manifiesto ${MANIFEST}, tarea ${SLURM_ARRAY_TASK_ID}) ==="
"$PYTHON_BIN" src/run.py --config-name "${DATASET}_distilled_default" \
  "${COMMON_OVERRIDES[@]}" \
  seed="${SEED}" \
  dataset_name="${DATASET}_${TAG}_k${K}_s${SEED}" \
  data_module.train_datasets.synthetic.buffer_path="$BUFFER_PATH" \
  trainer_callbacks.checkpointer.dirpath="$RUN_DIR/weights_s${SEED}" \
  hydra.run.dir="$RUN_DIR/hydra_s${SEED}"

echo "[INFO] Evaluacion ${TAG} ${DATASET} K=${K} seed=${SEED} completada."
