#!/bin/bash
#SBATCH --job-name=dd_distill_noclass_budget
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --array=0-11%8
#SBATCH --output=%x_%A_%a.log

set -euo pipefail

# ============================================================================
# E4 (conditioning ablation): DM and DC with medoid init but `conditional_matching=false` --
# the matching loop uses GLOBAL batches (not grouped by text cluster as a pseudo-class),
# isolating whether conditioning adds anything beyond the initialisation.
#
# Job array: 1 task per cell (dataset, K) of the 4 datasets x 3 budgets of the sweep
# (12 cells), like lanzar_distill_budget.sbatch but WITHOUT cluster conditioning.
# `+round_robin_syn=true`: keeps the update schedule of the conditional loop (1 synthetic
# per iteration, round-robin => 2000/K updates per image) and changes ONLY where the real
# batch comes from (random global pool instead of that synthetic's cluster). Without it the
# global loop updates min(32,K) synthetics per iteration = min(32,K)x more updates per image
# than the conditional one (a dose confound). No need to repeat per seed: generating the
# buffer is a single deterministic pass (as in the conditional version).
#
#   sbatch slurm/lanzar_distill_noclass_budget.sbatch
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
RUN_DIR="$RUNS_DIR/${DATASET}_distill_noclass_k${K}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

export PYTHONUNBUFFERED=1
cd "$REPO_ROOT"
mkdir -p "$RUN_DIR" "$BUFFER_DIR"

[[ -f "$CLUSTER_FILE" ]] || { echo "[ERROR] Falta $CLUSTER_FILE" >&2; exit 1; }

# Same IAM special case as lanzar_distill_budget.sbatch: IAM uses the _wr8 configs.
DM_CFG="distill_${DATASET}_dm_ctc"
DC_CFG="distill_${DATASET}_dc_ctc"
if [[ "$DATASET" == "iam" ]]; then
  DM_CFG="distill_iam_dm_ctc_wr8"
  DC_CFG="distill_iam_dc_ctc_wr8"
fi

MATCH_BS="$K"
if [[ "$K" -gt 32 ]]; then MATCH_BS=32; fi

DM_BUFFER="$BUFFER_DIR/${DATASET}_dm_noclass_k${K}.pt"
DC_BUFFER="$BUFFER_DIR/${DATASET}_dc_noclass_k${K}.pt"

echo "[INFO] === DM sin pseudo-clase: ${DATASET} K=${K} (matching_batch_size=${MATCH_BS}) ==="
"$PYTHON_BIN" src/distill_dm_ctc.py --config-name "$DM_CFG" \
  K="$K" \
  matching_batch_size="$MATCH_BS" \
  cluster_file="$CLUSTER_FILE" \
  +conditional_matching=false \
  +round_robin_syn=true \
  buffer_path="$DM_BUFFER" \
  hydra.run.dir="$RUN_DIR/hydra_dm_noclass"

echo "[INFO] === DC sin pseudo-clase: ${DATASET} K=${K} (inner_lr=3e-3, matching_batch_size=${MATCH_BS}) ==="
"$PYTHON_BIN" src/distill_dc_ctc.py --config-name "$DC_CFG" \
  K="$K" \
  matching_batch_size="$MATCH_BS" \
  inner_lr=3e-3 \
  cluster_file="$CLUSTER_FILE" \
  +conditional_matching=false \
  +round_robin_syn=true \
  buffer_path="$DC_BUFFER" \
  hydra.run.dir="$RUN_DIR/hydra_dc_noclass"

echo "[INFO] Buffers (sin pseudo-clase): $DM_BUFFER, $DC_BUFFER"
