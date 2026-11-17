#!/bin/bash
#SBATCH --job-name=dd_distill_budget
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.log

set -euo pipefail

# ============================================================================
# E1: DM-cond + DC-diag for one (DATASET, K) of the budget sweep, conditioned on the
# text medoids of that K (data/{Dataset}/text_clusters_k{K}.yaml). DC directly at
# inner_lr=3e-3. Reuses the distill_{dataset}_{dc,dm}_ctc.yaml configs via command-line
# overrides (K, cluster_file, matching_batch_size, buffer_path) -- no new config per K.
#
#   DATASET=lam K=198 sbatch slurm/lanzar_distill_budget.sbatch
# ============================================================================

REPO_ROOT="${REPO_ROOT:?Set REPO_ROOT to the repository root}"
RUNS_DIR="${RUNS_DIR:?Set RUNS_DIR to the directory for runs, buffers and experts}"
PYTHON_BIN="${PYTHON_BIN:?Set PYTHON_BIN to the Python interpreter}"
DATASET="${DATASET:?Set DATASET (iam, gw, read2016, lam)}"
K="${K:?Set K (budget size)}"
RUN_DIR="$RUNS_DIR/${DATASET}_distill_k${K}"
CLUSTER_FILE="data/${DATASET^^}/text_clusters_k${K}.yaml"
# FIXED buffer paths (independent of SLURM_JOB_ID) so that lanzar_distill_eval_budget.sbatch
# can be chained with --dependency without parsing logs.
BUFFER_DIR="$RUNS_DIR/budget_buffers"
DM_BUFFER="$BUFFER_DIR/${DATASET}_dm_cond_k${K}.pt"
DC_BUFFER="$BUFFER_DIR/${DATASET}_dc_diag_k${K}.pt"

export PYTHONUNBUFFERED=1
cd "$REPO_ROOT"
mkdir -p "$RUN_DIR" "$BUFFER_DIR"

[[ -f "$CLUSTER_FILE" ]] || { echo "[ERROR] Falta $CLUSTER_FILE -- genera los indices primero." >&2; exit 1; }

# IAM uses the _wr8 variants of the distillation configs (full training pool, wr8 model);
# GW/READ2016/LAM configs have no suffix.
DM_CFG="distill_${DATASET}_dm_ctc"
DC_CFG="distill_${DATASET}_dc_ctc"
if [[ "$DATASET" == "iam" ]]; then
  DM_CFG="distill_iam_dm_ctc_wr8"
  DC_CFG="distill_iam_dc_ctc_wr8"
fi

MATCH_BS="$K"
if [[ "$K" -gt 32 ]]; then MATCH_BS=32; fi

echo "[INFO] === DM-cond ${DATASET} K=${K} (matching_batch_size=${MATCH_BS}, config=${DM_CFG}) ==="
"$PYTHON_BIN" src/distill_dm_ctc.py --config-name "$DM_CFG" \
  K="$K" \
  matching_batch_size="$MATCH_BS" \
  cluster_file="$CLUSTER_FILE" \
  buffer_path="$DM_BUFFER" \
  hydra.run.dir="$RUN_DIR/hydra_dm_cond"

echo "[INFO] === DC-diag ${DATASET} K=${K} (inner_lr=3e-3, matching_batch_size=${MATCH_BS}, config=${DC_CFG}) ==="
"$PYTHON_BIN" src/distill_dc_ctc.py --config-name "$DC_CFG" \
  K="$K" \
  matching_batch_size="$MATCH_BS" \
  inner_lr=3e-3 \
  cluster_file="$CLUSTER_FILE" \
  buffer_path="$DC_BUFFER" \
  hydra.run.dir="$RUN_DIR/hydra_dc_diag"

echo "[INFO] Buffers: $DM_BUFFER, $DC_BUFFER"
