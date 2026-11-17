#!/bin/bash
#SBATCH --job-name=dd_full_baseline_budget
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=72:00:00
#SBATCH --output=%x_%j.log

set -euo pipefail

# ============================================================================
# E2 (budget sweep, "100% / full pool" column): trains on the FULL training pool of a
# dataset (no subsampling), 3 seeds (0, 42, 1337) -- EXACTLY the same recipe as the rest of
# the budget sweep (default (PIL) augmentation + lr=3e-3 + wr8 + bias_against_blank +
# patience=30) so that the resulting SER is comparable row by row with random/medoids/DD.
#
# Unlike random/medoids (protocol "all K in a single batch"), there is NO subsampling here:
# it is standard mini-batch training (batch_size=32, no accumulation) on the full pool --
# only the recipe changes so that it is comparable with the rest of the sweep.
#
#   DATASET=lam sbatch slurm/lanzar_full_baseline_budget.sbatch
# ============================================================================

REPO_ROOT="${REPO_ROOT:?Set REPO_ROOT to the repository root}"
RUNS_DIR="${RUNS_DIR:?Set RUNS_DIR to the directory for runs, buffers and experts}"
PYTHON_BIN="${PYTHON_BIN:?Set PYTHON_BIN to the Python interpreter}"
DATASET="${DATASET:?Set DATASET (iam, gw, read2016, lam)}"
RUN_DIR="$RUNS_DIR/${DATASET}_full_baseline_${SLURM_JOB_ID}"

export PYTHONUNBUFFERED=1
cd "$REPO_ROOT"
mkdir -p "$RUN_DIR"

[[ -f "configs/${DATASET}_default.yaml" ]] || { echo "[ERROR] Falta configs/${DATASET}_default.yaml" >&2; exit 1; }

COMMON_OVERRIDES=(
  "models@model_wrapper=CRNN_CTC_Wrapper_reduced_wr8"
  "extra._target_=extras.bias_against_blank"
  "+extra.blank_bias=-5.0"
  "+optimizer.lr=3e-3"
  "trainer.max_epochs=4000"
  "trainer.check_val_every_n_epoch=20"
  "trainer_callbacks.early_stopping.patience=30"
  "batch_size=32"
)

i=0
for seed in 0 42 1337; do
  i=$((i + 1))
  echo "[INFO] === ${i}/3: ${DATASET} pool completo seed=${seed} ==="
  "$PYTHON_BIN" src/run.py --config-name "${DATASET}_default" \
    "${COMMON_OVERRIDES[@]}" \
    seed="${seed}" \
    dataset_name="${DATASET}_full_baseline_wr8_s${seed}" \
    trainer_callbacks.checkpointer.dirpath="$RUN_DIR/weights_s${seed}" \
    hydra.run.dir="$RUN_DIR/hydra_s${seed}"
done

echo "[INFO] Full baseline (pool completo) ${DATASET} completo -- 3 semillas (0, 42, 1337)."
