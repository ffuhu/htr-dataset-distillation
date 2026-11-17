#!/bin/bash
#SBATCH --job-name=dd_distill_eval_zeroiter_budget
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-35%8
#SBATCH --output=%x_%A_%a.log

set -euo pipefail

# ============================================================================
# E3: 0-iteration control. Evaluates the DM 0-iteration buffer (the medoids as they are,
# in tensor format; written by lanzar_distill_zeroiter_budget.sbatch) through the SAME
# synthetic-buffer pipeline and the SAME recipe as DM-cond/DC-diag/no-cond (COMMON_OVERRIDES
# identical to lanzar_distill_eval_noclass_budget.sbatch). One seed per task.
#
# Job array: 12 cells (dataset,K) x 3 seeds (0,42,1337) = 36 tasks.
# Export WANDB_API_KEY before sbatch (the environment is inherited; without it wandb may fall
# back to another account's ~/.netrc and fail with 'permission denied'):
#
#   export WANDB_API_KEY=<key>
#   sbatch --dependency=afterany:<jobid_zeroiter> slurm/lanzar_distill_eval_zeroiter_budget.sbatch
# ============================================================================

REPO_ROOT="${REPO_ROOT:?Set REPO_ROOT to the repository root}"
RUNS_DIR="${RUNS_DIR:?Set RUNS_DIR to the directory for runs, buffers and experts}"
PYTHON_BIN="${PYTHON_BIN:?Set PYTHON_BIN to the Python interpreter}"
BUFFER_DIR="$RUNS_DIR/budget_buffers"

DATASETS=(iam iam iam gw gw gw read2016 read2016 read2016 lam lam lam)
KS=(62 308 616 4 18 36 84 418 835 198 991 1983)
TAG=dm_zeroiter
SEEDS=(0 42 1337)

CELL=$(( SLURM_ARRAY_TASK_ID / 3 ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % 3 ))

DATASET="${DATASETS[$CELL]}"
K="${KS[$CELL]}"
SEED="${SEEDS[$SEED_IDX]}"
BUFFER_PATH="$BUFFER_DIR/${DATASET}_${TAG}_k${K}.pt"
RUN_DIR="$RUNS_DIR/${DATASET}_distill_eval_zeroiter_k${K}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

export PYTHONUNBUFFERED=1
cd "$REPO_ROOT"
mkdir -p "$RUN_DIR"

[[ -f "$BUFFER_PATH" ]] || { echo "[ERROR] No existe el buffer: $BUFFER_PATH -- ¿corrio el array de destilacion?" >&2; exit 1; }

# See the note in lanzar_nucleo_baseline_budget.sbatch: batch_size=K does not fit in GPU memory
# for K>32 -- mini-batches of 32 + accumulate_grad_batches keep the effective batch at K.
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

echo "[INFO] === ${TAG} ${DATASET} K=${K} seed=${SEED} (tarea ${SLURM_ARRAY_TASK_ID}/35) ==="
"$PYTHON_BIN" src/run.py --config-name "${DATASET}_distilled_default" \
  "${COMMON_OVERRIDES[@]}" \
  seed="${SEED}" \
  dataset_name="${DATASET}_${TAG}_k${K}_s${SEED}" \
  data_module.train_datasets.synthetic.buffer_path="$BUFFER_PATH" \
  trainer_callbacks.checkpointer.dirpath="$RUN_DIR/weights_s${SEED}" \
  hydra.run.dir="$RUN_DIR/hydra_s${SEED}"

echo "[INFO] Evaluacion ${TAG} ${DATASET} K=${K} seed=${SEED} completada."
