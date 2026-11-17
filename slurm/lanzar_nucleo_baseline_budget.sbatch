#!/bin/bash
#SBATCH --job-name=dd_nucleo_baseline_budget
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.log

set -euo pipefail

# ============================================================================
# E1: random + medoids for one (DATASET, K) of the 1/5/10% budget sweep, 3 seeds each
# (0, 42, 1337). Uses the configs {dataset}_{aleatorio,medoides}_k{K}_default.yaml
# (aleatorio = random subset, medoides = text medoids).
#
#   DATASET=lam K=198 sbatch slurm/lanzar_nucleo_baseline_budget.sbatch
# ============================================================================

REPO_ROOT="${REPO_ROOT:?Set REPO_ROOT to the repository root}"
RUNS_DIR="${RUNS_DIR:?Set RUNS_DIR to the directory for runs, buffers and experts}"
PYTHON_BIN="${PYTHON_BIN:?Set PYTHON_BIN to the Python interpreter}"
DATASET="${DATASET:?Set DATASET (iam, gw, read2016, lam)}"
K="${K:?Set K (budget size)}"
RUN_DIR="$RUNS_DIR/${DATASET}_nucleo_baseline_k${K}_${SLURM_JOB_ID}"

export PYTHONUNBUFFERED=1
cd "$REPO_ROOT"
mkdir -p "$RUN_DIR"

for f in "configs/${DATASET}_aleatorio_k${K}_default.yaml" "configs/${DATASET}_medoides_k${K}_default.yaml"; do
  [[ -f "$f" ]] || { echo "[ERROR] Falta $f -- sincroniza el repo." >&2; exit 1; }
done

# For K>32 the full batch (protocol "all K in a single step") does not fit on an A100 40GB
# (measured: K=616 -> CUDA OOM in backward with ~30GB in use). Gradients are accumulated over
# mini-batches of 32 (the value already validated as safe for matching_batch_size in the
# distillation scripts) so that the EFFECTIVE batch stays at exactly K (or the nearest multiple
# of 32 above it), preserving the protocol "one gradient step = all K" without changing the
# optimisation dynamics.
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

i=0
for spec in "aleatorio:0" "aleatorio:42" "aleatorio:1337" "medoides:0" "medoides:42" "medoides:1337"; do
  IFS=':' read -r tag seed <<< "$spec"
  i=$((i + 1))
  echo "[INFO] === ${i}/6: ${DATASET} ${tag} K=${K} seed=${seed} ==="
  "$PYTHON_BIN" src/run.py --config-name "${DATASET}_${tag}_k${K}_default" \
    "${COMMON_OVERRIDES[@]}" \
    seed="${seed}" \
    dataset_name="${DATASET}_${tag}_k${K}_s${seed}" \
    trainer_callbacks.checkpointer.dirpath="$RUN_DIR/weights_${tag}_s${seed}" \
    hydra.run.dir="$RUN_DIR/hydra_${tag}_s${seed}"
done

echo "[INFO] Nucleo baseline ${DATASET} K=${K} completo (aleatorio+medoides, 3 semillas cada uno)."
