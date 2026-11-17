#!/bin/bash
#SBATCH --job-name=dd_mtt_drift_budget
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.log

set -euo pipefail

# ============================================================================
# E3: pixel drift of the 36 MTT buffers of the sweep (random init per seed,
# cluster_file: null) against the real images of their source_indices, with the pool of the
# same distillation config (src/buffer_drift_pool.py).
# Sanity check first: the DM 0-iteration buffer with its DM config -> drift 0.
#
#   sbatch slurm/lanzar_mtt_drift_budget.sbatch
# ============================================================================

REPO_ROOT="${REPO_ROOT:?Set REPO_ROOT to the repository root}"
RUNS_DIR="${RUNS_DIR:?Set RUNS_DIR to the directory for runs, buffers and experts}"
PYTHON_BIN="${PYTHON_BIN:?Set PYTHON_BIN to the Python interpreter}"
B="$RUNS_DIR/budget_buffers"

export PYTHONUNBUFFERED=1
cd "$REPO_ROOT"

echo "[INFO] === Cordura: 0 iteraciones DM, READ2016 K=418 (esperado 0) ==="
"$PYTHON_BIN" src/buffer_drift_pool.py --config-name distill_read2016_dm_ctc \
  --buffers "$B/read2016_dm_zeroiter_k418.pt"

declare -A KS=( [iam]="62 308 616" [gw]="4 18 36" [read2016]="84 418 835" [lam]="198 991 1983" )
for DATASET in iam gw read2016 lam; do
  CFG="distill_${DATASET}_mtt_ctc"
  if [[ "$DATASET" == "iam" ]]; then CFG="distill_iam_mtt_ctc_wr8"; fi
  BUFS=()
  for K in ${KS[$DATASET]}; do
    for S in 0 42 1337; do BUFS+=("$B/${DATASET}_mtt_ctc_k${K}_s${S}.pt"); done
  done
  echo "[INFO] === MTT ${DATASET} (config ${CFG}) ==="
  "$PYTHON_BIN" src/buffer_drift_pool.py --config-name "$CFG" --buffers "${BUFS[@]}"
done
