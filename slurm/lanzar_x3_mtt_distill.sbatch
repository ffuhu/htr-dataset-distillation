#!/bin/bash
#SBATCH --job-name=dd_x3_mtt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-7%4
#SBATCH --output=%x_%A_%a.log

set -euo pipefail

# ============================================================================
# E6 (X3): MTT with pixel_lr in {1, 10} (0.1 = the MTT of the sweep), X1 cells, seed 0 (same
# random init as MTT s0 of the sweep and as its 0-iter, E3). Otherwise identical to
# lanzar_mtt_distill_budget.sbatch (2000 iterations, same experts). Also used by E10 (GW seeds
# 7 and 123, pixel_lr 0.1) with MANIFEST=slurm/manifests/gw5_mtt_distill.txt.
#
#   sbatch slurm/lanzar_x3_mtt_distill.sbatch
# ============================================================================

REPO_ROOT="${REPO_ROOT:?Set REPO_ROOT to the repository root}"
RUNS_DIR="${RUNS_DIR:?Set RUNS_DIR to the directory for runs, buffers and experts}"
PYTHON_BIN="${PYTHON_BIN:?Set PYTHON_BIN to the Python interpreter}"
MANIFEST="${MANIFEST:-$REPO_ROOT/slurm/manifests/x3_distill.txt}"

export PYTHONUNBUFFERED=1
cd "$REPO_ROOT"

LINE="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")"
LINE="${LINE//\$RUNS_DIR/$RUNS_DIR}"  # manifests write paths as $RUNS_DIR/...
[[ -n "$LINE" ]] || { echo "[ERROR] Sin linea $((SLURM_ARRAY_TASK_ID + 1)) en $MANIFEST" >&2; exit 1; }
read -r DATASET K LR SEED EXPERT_DIR BUFFER <<< "$LINE"
RUN_DIR="$RUNS_DIR/x3_${DATASET}_mtt_k${K}_lr${LR}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$RUN_DIR"
[[ -d "$EXPERT_DIR" ]] || { echo "[ERROR] No existe $EXPERT_DIR" >&2; exit 1; }
[[ ! -e "$BUFFER" ]] || { echo "[ERROR] $BUFFER ya existe" >&2; exit 1; }

MTT_CFG="distill_${DATASET}_mtt_ctc"
if [[ "$DATASET" == "iam" ]]; then MTT_CFG="distill_iam_mtt_ctc_wr8"; fi

echo "[INFO] === X3 MTT ${DATASET} K=${K} pixel_lr=${LR} seed=${SEED} ==="
"$PYTHON_BIN" src/distill_mtt_ctc.py --config-name "$MTT_CFG" \
  expert_dir="$EXPERT_DIR" \
  K="$K" \
  seed="$SEED" \
  pixel_lr="$LR" \
  log_every=50 \
  buffer_path="$BUFFER" \
  hydra.run.dir="$RUN_DIR/hydra"
echo "[INFO] X3 MTT ${DATASET} K=${K} pixel_lr=${LR} completada."
