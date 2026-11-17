#!/bin/bash
#SBATCH --job-name=dd_distill_zeroiter_budget
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-11%8
#SBATCH --output=%x_%A_%a.log

set -euo pipefail

# ============================================================================
# E3: 0-iteration control + pixel drift.
#
# (1) 0-iteration buffer: same scripts/configs/overrides as
#     lanzar_distill_budget.sbatch (DM-cond / DC-diag) but `outer_iterations=0` --
#     the matching loop does not run and the saved buffer is the medoids as they are,
#     in the same tensor format as the distilled buffers. Evaluating it through the
#     synthetic-buffer pipeline (lanzar_distill_eval_zeroiter_budget.sbatch) separates
#     "optimising pixels" from "synthetic evaluation pipeline" against real medoids.
#     Both (DM and DC) are generated to check that the initialisation is identical.
# (2) Relative L2 drift of the 4 buffers of the cell (dm_cond, dc_diag, dm_noclass,
#     dc_noclass) against their 0-iteration buffer (src/buffer_drift_budget.py).
#
#   sbatch slurm/lanzar_distill_zeroiter_budget.sbatch
# ============================================================================

REPO_ROOT="${REPO_ROOT:?Set REPO_ROOT to the repository root}"
RUNS_DIR="${RUNS_DIR:?Set RUNS_DIR to the directory for runs, buffers and experts}"
PYTHON_BIN="${PYTHON_BIN:?Set PYTHON_BIN to the Python interpreter}"
BUFFER_DIR="$RUNS_DIR/budget_buffers"

DATASETS=(iam iam iam gw gw gw read2016 read2016 read2016 lam lam lam)
KS=(62 308 616 4 18 36 84 418 835 198 991 1983)

DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
K="${KS[$SLURM_ARRAY_TASK_ID]}"
CLUSTER_FILE="data/${DATASET^^}/text_clusters_k${K}.yaml"
RUN_DIR="$RUNS_DIR/${DATASET}_distill_zeroiter_k${K}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

export PYTHONUNBUFFERED=1
cd "$REPO_ROOT"
mkdir -p "$RUN_DIR" "$BUFFER_DIR"

[[ -f "$CLUSTER_FILE" ]] || { echo "[ERROR] Falta $CLUSTER_FILE" >&2; exit 1; }

# Same IAM special case as lanzar_distill_budget.sbatch.
DM_CFG="distill_${DATASET}_dm_ctc"
DC_CFG="distill_${DATASET}_dc_ctc"
if [[ "$DATASET" == "iam" ]]; then
  DM_CFG="distill_iam_dm_ctc_wr8"
  DC_CFG="distill_iam_dc_ctc_wr8"
fi

MATCH_BS="$K"
if [[ "$K" -gt 32 ]]; then MATCH_BS=32; fi

DM_ZERO="$BUFFER_DIR/${DATASET}_dm_zeroiter_k${K}.pt"
DC_ZERO="$BUFFER_DIR/${DATASET}_dc_zeroiter_k${K}.pt"

echo "[INFO] === DM 0 iteraciones: ${DATASET} K=${K} ==="
"$PYTHON_BIN" src/distill_dm_ctc.py --config-name "$DM_CFG" \
  K="$K" \
  matching_batch_size="$MATCH_BS" \
  cluster_file="$CLUSTER_FILE" \
  outer_iterations=0 \
  buffer_path="$DM_ZERO" \
  hydra.run.dir="$RUN_DIR/hydra_dm_zeroiter"

echo "[INFO] === DC 0 iteraciones: ${DATASET} K=${K} ==="
"$PYTHON_BIN" src/distill_dc_ctc.py --config-name "$DC_CFG" \
  K="$K" \
  matching_batch_size="$MATCH_BS" \
  inner_lr=3e-3 \
  cluster_file="$CLUSTER_FILE" \
  outer_iterations=0 \
  buffer_path="$DC_ZERO" \
  hydra.run.dir="$RUN_DIR/hydra_dc_zeroiter"

echo "[INFO] === Deriva ${DATASET} K=${K} ==="
"$PYTHON_BIN" src/buffer_drift_budget.py --init "$DM_ZERO" --buffers \
  "$DC_ZERO" \
  "$BUFFER_DIR/${DATASET}_dm_cond_k${K}.pt" \
  "$BUFFER_DIR/${DATASET}_dm_noclass_k${K}.pt"
"$PYTHON_BIN" src/buffer_drift_budget.py --init "$DC_ZERO" --buffers \
  "$BUFFER_DIR/${DATASET}_dc_diag_k${K}.pt" \
  "$BUFFER_DIR/${DATASET}_dc_noclass_k${K}.pt"

echo "[INFO] Buffers 0 iteraciones: $DM_ZERO, $DC_ZERO"
