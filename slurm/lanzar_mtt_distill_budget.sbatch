#!/bin/bash
#SBATCH --job-name=dd_mtt_distill_budget
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=18:00:00
#SBATCH --output=%x_%j.log

set -euo pipefail

# ============================================================================
# E1: MTT distillation for one (DATASET, K) of the sweep, reusing the expert trajectory
# buffer of that dataset (independent of K -- the experts represent standard training on
# the FULL dataset, they do not depend on the size of the synthetic buffer). Each
# EVALUATION seed needs its own distilled buffer.
#
#   DATASET=lam K=198 EXPERT_DIR=$RUNS_DIR/experts/lam SEED=0 \
#     sbatch slurm/lanzar_mtt_distill_budget.sbatch
# ============================================================================

REPO_ROOT="${REPO_ROOT:?Set REPO_ROOT to the repository root}"
RUNS_DIR="${RUNS_DIR:?Set RUNS_DIR to the directory for runs, buffers and experts}"
PYTHON_BIN="${PYTHON_BIN:?Set PYTHON_BIN to the Python interpreter}"
DATASET="${DATASET:?Set DATASET (iam, gw, read2016, lam)}"
K="${K:?Set K (budget size)}"
EXPERT_DIR="${EXPERT_DIR:?Set EXPERT_DIR to the dataset expert trajectory directory}"
SEED="${SEED:?Set SEED}"
RUN_DIR="$RUNS_DIR/${DATASET}_mtt_distill_k${K}_${SLURM_JOB_ID}"
# FIXED buffer path (independent of SLURM_JOB_ID) so that lanzar_mtt_eval_budget.sbatch can be
# chained with --dependency without parsing logs.
BUFFER_DIR="$RUNS_DIR/budget_buffers"

export PYTHONUNBUFFERED=1
cd "$REPO_ROOT"
mkdir -p "$RUN_DIR" "$BUFFER_DIR"

[[ -d "$EXPERT_DIR" ]] || { echo "[ERROR] No existe el buffer de expertos: $EXPERT_DIR" >&2; exit 1; }

# IAM uses the _wr8 variant of the MTT config (full training pool).
MTT_CFG="distill_${DATASET}_mtt_ctc"
if [[ "$DATASET" == "iam" ]]; then
  MTT_CFG="distill_iam_mtt_ctc_wr8"
fi
[[ -f "configs/${MTT_CFG}.yaml" ]] || { echo "[ERROR] Falta configs/${MTT_CFG}.yaml -- sincroniza configs/." >&2; exit 1; }

BUFFER_PATH="$BUFFER_DIR/${DATASET}_mtt_ctc_k${K}_s${SEED}.pt"

echo "[INFO] Destilacion MTT ${DATASET} K=${K} seed=${SEED}, outer_iterations=2000, config=${MTT_CFG}"
"$PYTHON_BIN" src/distill_mtt_ctc.py \
  --config-name "$MTT_CFG" \
  expert_dir="$EXPERT_DIR" \
  K="$K" \
  seed="${SEED}" \
  log_every=50 \
  buffer_path="$BUFFER_PATH"

echo "[INFO] Buffer guardado en $BUFFER_PATH"
