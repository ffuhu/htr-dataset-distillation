#!/bin/bash
#SBATCH --job-name=dd_x1_distill
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-23%6
#SBATCH --output=%x_%A_%a.log

set -euo pipefail

# ============================================================================
# E6 (X1): optimisation-strength sweep of DM/DC.
# Medoid init (cluster_file), UNCONDITIONAL batch matching (round_robin_syn=false:
# B=min(32,K) synthetics per iteration), Adam step size syn_lr in {0.1,1,10} (grey levels;
# pixels are in 0-255, not normalised) and buffer snapshots at U~{30,300,3000} mean updates
# per image. One task = (cell, method, syn_lr); single source of parameters:
# slurm/manifests/x1_distill.txt.
# At the end, L2 drift of each snapshot against the cell's 0-iteration buffer (E3).
#
#   sbatch slurm/lanzar_x1_distill.sbatch
# ============================================================================

REPO_ROOT="${REPO_ROOT:?Set REPO_ROOT to the repository root}"
RUNS_DIR="${RUNS_DIR:?Set RUNS_DIR to the directory for runs, buffers and experts}"
# MANIFEST and DC_EXTRA can be overridden (X1b: MANIFEST=slurm/manifests/x1b_distill.txt).
PYTHON_BIN="${PYTHON_BIN:?Set PYTHON_BIN to the Python interpreter}"
BUFFER_DIR="$RUNS_DIR/budget_buffers"
MANIFEST="${MANIFEST:-$REPO_ROOT/slurm/manifests/x1_distill.txt}"

export PYTHONUNBUFFERED=1
cd "$REPO_ROOT"

LINE="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")"
LINE="${LINE//\$RUNS_DIR/$RUNS_DIR}"  # manifests write paths as $RUNS_DIR/...
[[ -n "$LINE" ]] || { echo "[ERROR] Sin linea $((SLURM_ARRAY_TASK_ID + 1)) en $MANIFEST" >&2; exit 1; }
read -r DATASET K METHOD LR B N1 N2 N3 BUFFER <<< "$LINE"

CLUSTER_FILE="data/${DATASET^^}/text_clusters_k${K}.yaml"
ZEROITER="$BUFFER_DIR/${DATASET}_${METHOD}_zeroiter_k${K}.pt"
RUN_DIR="$RUNS_DIR/$(basename "${BUFFER%.pt}")"
mkdir -p "$RUN_DIR" "$BUFFER_DIR"

[[ -f "$CLUSTER_FILE" ]] || { echo "[ERROR] Falta $CLUSTER_FILE" >&2; exit 1; }
[[ -f "$ZEROITER" ]] || { echo "[ERROR] Falta $ZEROITER" >&2; exit 1; }
for N in "$N1" "$N2" "$N3"; do
  [[ ! -e "${BUFFER%.pt}_it${N}.pt" ]] || { echo "[ERROR] ${BUFFER%.pt}_it${N}.pt ya existe" >&2; exit 1; }
done

CFG="distill_${DATASET}_${METHOD}_ctc"
if [[ "$DATASET" == "iam" ]]; then CFG="distill_iam_${METHOD}_ctc_wr8"; fi
SCRIPT="src/distill_${METHOD}_ctc.py"
EXTRA=()
if [[ "$METHOD" == "dc" ]]; then EXTRA=("inner_lr=3e-3"); fi
# DC_EXTRA: extra overrides for DC only, e.g. "+match_distance=dc_original +skip_nonfinite_syn_grad=true"
if [[ "$METHOD" == "dc" && -n "${DC_EXTRA:-}" ]]; then read -r -a DC_EXTRA_ARR <<< "$DC_EXTRA"; EXTRA+=("${DC_EXTRA_ARR[@]}"); fi

echo "[INFO] === X1 ${METHOD} ${DATASET} K=${K} syn_lr=${LR} B=${B} its=${N3} snapshots=[${N1},${N2},${N3}] ==="
"$PYTHON_BIN" "$SCRIPT" --config-name "$CFG" \
  K="$K" \
  matching_batch_size="$B" \
  cluster_file="$CLUSTER_FILE" \
  +conditional_matching=false \
  +round_robin_syn=false \
  syn_lr="$LR" \
  outer_iterations="$N3" \
  "+snapshot_iterations=[${N1},${N2},${N3}]" \
  "${EXTRA[@]}" \
  buffer_path="$BUFFER" \
  hydra.run.dir="$RUN_DIR/hydra"

echo "[INFO] === Deriva frente a $ZEROITER ==="
"$PYTHON_BIN" src/buffer_drift_budget.py --init "$ZEROITER" \
  --buffers "${BUFFER%.pt}_it${N1}.pt" "${BUFFER%.pt}_it${N2}.pt" "${BUFFER%.pt}_it${N3}.pt"
echo "[INFO] X1 ${METHOD} ${DATASET} K=${K} syn_lr=${LR} completada."
