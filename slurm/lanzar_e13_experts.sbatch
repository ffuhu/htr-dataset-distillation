#!/bin/bash
#SBATCH --job-name=dd_e13_experts
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=%x_%A_%a.log

set -euo pipefail

# ============================================================================
# E13 (pre-registered): one IAM expert with SGD per task
# (configs/train_expert_sgd_wr8.yaml: no augmentation, 20 epochs, snapshot every 100 steps).
# Manifest (one line per task): "lr batch_size seed".
#
#   sbatch --array=0-7%8 --export=ALL,MANIFEST=slurm/manifests/e13_g0_experts.txt slurm/lanzar_e13_experts.sbatch
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
read -r LR BATCH SEED <<< "$LINE"

EXPERT_DIR="$RUNS_DIR/e13_iam_sgd_experts/lr${LR}_b${BATCH}/expert_s${SEED}"
RUN_DIR="$RUNS_DIR/e13_iam_sgd_expert_training/lr${LR}_b${BATCH}_s${SEED}_${SLURM_ARRAY_JOB_ID}"
[[ ! -e "$EXPERT_DIR" ]] || { echo "[ERROR] $EXPERT_DIR ya existe" >&2; exit 1; }
mkdir -p "$RUN_DIR"

echo "[INFO] === E13 experto IAM SGD lr=${LR} batch=${BATCH} seed=${SEED} ==="
"$PYTHON_BIN" src/run.py --config-name train_expert_sgd_wr8 \
  seed="$SEED" \
  batch_size="$BATCH" \
  optimizer.lr="$LR" \
  dataset_name="iam_e13_sgd_lr${LR}_b${BATCH}_s${SEED}" \
  +trainer.logger.tags="[E13]" \
  trainer_callbacks.checkpointer.dirpath="$RUN_DIR/weights" \
  trainer_callbacks.expert_snapshotter.dirpath="$EXPERT_DIR" \
  hydra.run.dir="$RUN_DIR/hydra" \
  data_module.train_datasets.iam.num_workers=0
echo "[INFO] Experto guardado en $EXPERT_DIR ($(ls "$EXPERT_DIR" | wc -l) fotos)"
