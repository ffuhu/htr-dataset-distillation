#!/bin/bash
#SBATCH --job-name=dd_mtt_init_buffers
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.log

set -euo pipefail

# ============================================================================
# E3: buffers with the unoptimised MTT initialisation (MTT 0-iter):
#     src/make_init_buffer.py -> {ds}_mtt_init_k{K}_s{S}.pt (36 buffers).
#
#   sbatch slurm/lanzar_mtt_init_buffers.sbatch
# ============================================================================

REPO_ROOT="${REPO_ROOT:?Set REPO_ROOT to the repository root}"
RUNS_DIR="${RUNS_DIR:?Set RUNS_DIR to the directory for runs, buffers and experts}"
PYTHON_BIN="${PYTHON_BIN:?Set PYTHON_BIN to the Python interpreter}"
B="$RUNS_DIR/budget_buffers"

export PYTHONUNBUFFERED=1
cd "$REPO_ROOT"

declare -A KS=( [iam]="62 308 616" [gw]="4 18 36" [read2016]="84 418 835" [lam]="198 991 1983" )
for DATASET in iam gw read2016 lam; do
  CFG="distill_${DATASET}_mtt_ctc"
  if [[ "$DATASET" == "iam" ]]; then CFG="distill_iam_mtt_ctc_wr8"; fi
  PAIRS=()
  for K in ${KS[$DATASET]}; do
    for S in 0 42 1337; do PAIRS+=("$B/${DATASET}_mtt_ctc_k${K}_s${S}.pt:$B/${DATASET}_mtt_init_k${K}_s${S}.pt"); done
  done
  echo "[INFO] === Buffers de init de MTT: ${DATASET} (config ${CFG}) ==="
  "$PYTHON_BIN" src/make_init_buffer.py --config-name "$CFG" --pairs "${PAIRS[@]}"
done

echo "[INFO] Buffers de init de MTT completos."
