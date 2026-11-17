#!/bin/bash
#SBATCH --job-name=dd_e13b_obj
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%A_%a.log

set -euo pipefail

# ============================================================================
# E13b: do DM and DC reduce their own objective? (src/eval_matching_objective.py)
# Manifest (one line per task): "tag method run_config initial_buffer final_buffer coadapt_steps".
#
#   sbatch --array=0-67%16 --export=ALL,MANIFEST=slurm/manifests/e13b_objective.txt slurm/lanzar_e13b_objective.sbatch
# ============================================================================

REPO_ROOT="${REPO_ROOT:?Set REPO_ROOT to the repository root}"
RUNS_DIR="${RUNS_DIR:?Set RUNS_DIR to the directory for runs, buffers and experts}"
PYTHON_BIN="${PYTHON_BIN:?Set PYTHON_BIN to the Python interpreter}"
OUT_DIR="$RUNS_DIR/e13b_objective"
: "${MANIFEST:?Set MANIFEST}"

export PYTHONUNBUFFERED=1
cd "$REPO_ROOT"
mkdir -p "$OUT_DIR"

LINE="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")"
LINE="${LINE//\$RUNS_DIR/$RUNS_DIR}"  # manifests write paths as $RUNS_DIR/...
[[ -n "$LINE" ]] || { echo "[ERROR] Sin linea $((SLURM_ARRAY_TASK_ID + 1)) en $MANIFEST" >&2; exit 1; }
read -r TAG METHOD RUN_CONFIG INIT FINAL COADAPT <<< "$LINE"
[[ -n "$COADAPT" ]] || { echo "[ERROR] Falta coadapt_steps en: $LINE" >&2; exit 1; }

echo "[INFO] === E13b ${TAG} (${METHOD}, coadapt=${COADAPT}) ==="
"$PYTHON_BIN" src/eval_matching_objective.py --method "$METHOD" --run-config "$RUN_CONFIG" \
  --init "$INIT" --final "$FINAL" --units 100 --draws 4 --seed 0 --coadapt-steps "$COADAPT" \
  --out "$OUT_DIR/${TAG}_c${COADAPT}_${SLURM_ARRAY_JOB_ID}.jsonl"
