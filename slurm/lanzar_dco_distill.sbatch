#!/bin/bash
#SBATCH --job-name=dd_dco_distill
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=12:00:00
#SBATCH --array=0-11%2
#SBATCH --output=%x_%A_%a.log

set -euo pipefail

# ============================================================================
# E9: DC column of the main table redone with the ORIGINAL DC distance
# (match_distance=dc_original) + guard against non-finite steps. Otherwise IDENTICAL to DC-diag in
# lanzar_distill_budget.sbatch: conditional (cluster_file, round-robin), medoid init,
# inner_lr=3e-3, 2000 iterations, matching_batch_size=min(32,K). Drift against the DC 0-iter buffer.
#
#   sbatch -p <partition> slurm/lanzar_dco_distill.sbatch
# ============================================================================

REPO_ROOT="${REPO_ROOT:?Set REPO_ROOT to the repository root}"
RUNS_DIR="${RUNS_DIR:?Set RUNS_DIR to the directory for runs, buffers and experts}"
PYTHON_BIN="${PYTHON_BIN:?Set PYTHON_BIN to the Python interpreter}"
BUFFER_DIR="$RUNS_DIR/budget_buffers"
MANIFEST="$REPO_ROOT/slurm/manifests/dco_distill.txt"

export PYTHONUNBUFFERED=1
cd "$REPO_ROOT"

LINE="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")"
LINE="${LINE//\$RUNS_DIR/$RUNS_DIR}"  # manifests write paths as $RUNS_DIR/...
[[ -n "$LINE" ]] || { echo "[ERROR] Sin linea $((SLURM_ARRAY_TASK_ID + 1)) en $MANIFEST" >&2; exit 1; }
read -r DATASET K BUFFER <<< "$LINE"
CLUSTER_FILE="data/${DATASET^^}/text_clusters_k${K}.yaml"
ZEROITER="$BUFFER_DIR/${DATASET}_dc_zeroiter_k${K}.pt"
RUN_DIR="$RUNS_DIR/dco_${DATASET}_k${K}"
mkdir -p "$RUN_DIR"
[[ -f "$CLUSTER_FILE" ]] || { echo "[ERROR] Falta $CLUSTER_FILE" >&2; exit 1; }
[[ -f "$ZEROITER" ]] || { echo "[ERROR] Falta $ZEROITER" >&2; exit 1; }
[[ ! -e "$BUFFER" ]] || { echo "[ERROR] $BUFFER ya existe" >&2; exit 1; }

DC_CFG="distill_${DATASET}_dc_ctc"
if [[ "$DATASET" == "iam" ]]; then DC_CFG="distill_iam_dc_ctc_wr8"; fi
MATCH_BS="$K"; if [[ "$K" -gt 32 ]]; then MATCH_BS=32; fi

echo "[INFO] === DC-diag (distancia original) ${DATASET} K=${K} (inner_lr=3e-3, matching_batch_size=${MATCH_BS}) ==="
"$PYTHON_BIN" src/distill_dc_ctc.py --config-name "$DC_CFG" \
  K="$K" \
  matching_batch_size="$MATCH_BS" \
  inner_lr=3e-3 \
  cluster_file="$CLUSTER_FILE" \
  +match_distance=dc_original \
  +skip_nonfinite_syn_grad=true \
  buffer_path="$BUFFER" \
  hydra.run.dir="$RUN_DIR/hydra"

echo "[INFO] === Deriva frente a $ZEROITER ==="
"$PYTHON_BIN" src/buffer_drift_budget.py --init "$ZEROITER" --buffers "$BUFFER"
echo "[INFO] DCO ${DATASET} K=${K} completada."
