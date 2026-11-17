#!/bin/bash
#SBATCH --job-name=dd_e13_diag
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%A_%a.log

set -euo pipefail

# ============================================================================
# E13: reachability diagnostic of the MTT objective (src/diag_mtt_reachability.py).
# Manifest: one line per task with the script arguments (without --out or --seed: set by this launcher).
#
#   sbatch --array=0-N --export=ALL,MANIFEST=slurm/manifests/e13_diag_adam.txt slurm/lanzar_e13_diag.sbatch
# ============================================================================

REPO_ROOT="${REPO_ROOT:?Set REPO_ROOT to the repository root}"
RUNS_DIR="${RUNS_DIR:?Set RUNS_DIR to the directory for runs, buffers and experts}"
PYTHON_BIN="${PYTHON_BIN:?Set PYTHON_BIN to the Python interpreter}"
OUT_DIR="$RUNS_DIR/e13_diag"
: "${MANIFEST:?Set MANIFEST}"

export PYTHONUNBUFFERED=1
cd "$REPO_ROOT"
mkdir -p "$OUT_DIR"

LINE="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")"
LINE="${LINE//\$RUNS_DIR/$RUNS_DIR}"  # manifests write paths as $RUNS_DIR/...
[[ -n "$LINE" ]] || { echo "[ERROR] Sin linea $((SLURM_ARRAY_TASK_ID + 1)) en $MANIFEST" >&2; exit 1; }
read -r -a ARGS <<< "$LINE"
OUT="$OUT_DIR/$(basename "$MANIFEST" .txt)_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.jsonl"

echo "[INFO] === E13 diag tarea ${SLURM_ARRAY_TASK_ID}: ${LINE} ==="
"$PYTHON_BIN" src/diag_mtt_reachability.py "${ARGS[@]}" --seed 0 --out "$OUT"
echo "[INFO] Resultados en $OUT"
