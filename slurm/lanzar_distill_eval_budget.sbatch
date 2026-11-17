#!/bin/bash
#SBATCH --job-name=dd_distill_eval_budget
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.log

set -euo pipefail

# ============================================================================
# E1: evaluates a DC/DM buffer (BUFFER_PATH, TAG) of one (DATASET, K) of the sweep, 3 seeds
# (0, 42, 1337).
#   DATASET=lam K=198 BUFFER_PATH=/.../lam_dm_cond_k198.pt TAG=dm_cond \
#     sbatch slurm/lanzar_distill_eval_budget.sbatch
# ============================================================================

REPO_ROOT="${REPO_ROOT:?Set REPO_ROOT to the repository root}"
RUNS_DIR="${RUNS_DIR:?Set RUNS_DIR to the directory for runs, buffers and experts}"
PYTHON_BIN="${PYTHON_BIN:?Set PYTHON_BIN to the Python interpreter}"
DATASET="${DATASET:?Set DATASET (iam, gw, read2016, lam)}"
K="${K:?Set K (budget size)}"
BUFFER_PATH="${BUFFER_PATH:?Set BUFFER_PATH to the distilled buffer .pt file}"
TAG="${TAG:?Set TAG (e.g. dm_cond, dc_diag)}"
RUN_DIR="$RUNS_DIR/${DATASET}_distill_eval_k${K}_${SLURM_JOB_ID}"

export PYTHONUNBUFFERED=1
cd "$REPO_ROOT"
mkdir -p "$RUN_DIR"

[[ -f "$BUFFER_PATH" ]] || { echo "[ERROR] No existe el buffer: $BUFFER_PATH" >&2; exit 1; }

# See the note in lanzar_nucleo_baseline_budget.sbatch: batch_size=K does not fit in GPU memory
# for K>32 (CUDA OOM measured at K=616) -- mini-batches of 32 + accumulate_grad_batches keep the
# effective batch at exactly K (or the nearest multiple of 32 above it).
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

for seed in 0 42 1337; do
  echo "[INFO] === ${TAG} ${DATASET} K=${K} seed=${seed} ==="
  "$PYTHON_BIN" src/run.py --config-name "${DATASET}_distilled_default" \
    "${COMMON_OVERRIDES[@]}" \
    seed="${seed}" \
    dataset_name="${DATASET}_${TAG}_k${K}_s${seed}" \
    data_module.train_datasets.synthetic.buffer_path="$BUFFER_PATH" \
    trainer_callbacks.checkpointer.dirpath="$RUN_DIR/weights_s${seed}" \
    hydra.run.dir="$RUN_DIR/hydra_s${seed}"
done

echo "[INFO] Evaluacion ${TAG} ${DATASET} K=${K} completada (3 semillas)."
