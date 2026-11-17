#!/bin/bash
#SBATCH --job-name=dd_x2_distill
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-3%2
#SBATCH --output=%x_%A_%a.log

set -euo pipefail

# ============================================================================
# E5 (X2): DM/DC initialised with EXACTLY the lines of the random baseline
# (random_baseline_K{K}_indices.yaml), current recipe of the unconditional sweep
# (round_robin_syn=true, syn_lr=0.1, 2000 iterations). Per cell:
#   (1) pool positions verified line by line (make_random_init_positions.py);
#   (2) 0-iteration buffer with those lines (paired pipeline control);
#   (3) distilled DM and DC; (4) drift against (2).
#
#   sbatch slurm/lanzar_x2_distill.sbatch
# ============================================================================

REPO_ROOT="${REPO_ROOT:?Set REPO_ROOT to the repository root}"
RUNS_DIR="${RUNS_DIR:?Set RUNS_DIR to the directory for runs, buffers and experts}"
# Python interpreter of the environment (see README).
PYTHON_BIN="${PYTHON_BIN:?Set PYTHON_BIN to the Python interpreter}"
BUFFER_DIR="$RUNS_DIR/budget_buffers"

DATASETS=(iam gw read2016 lam)
KS=(308 36 418 991)
DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
K="${KS[$SLURM_ARRAY_TASK_ID]}"
RUN_DIR="$RUNS_DIR/x2_${DATASET}_k${K}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
POSITIONS="$BUFFER_DIR/x2_${DATASET}_random_init_positions_k${K}.yaml"
ZEROITER="$BUFFER_DIR/x2_${DATASET}_rand_zeroiter_k${K}.pt"
DM_BUFFER="$BUFFER_DIR/x2_${DATASET}_dm_randinit_k${K}.pt"
DC_BUFFER="$BUFFER_DIR/x2_${DATASET}_dc_randinit_k${K}.pt"

export PYTHONUNBUFFERED=1
cd "$REPO_ROOT"
mkdir -p "$RUN_DIR" "$BUFFER_DIR"
for f in "$POSITIONS" "$ZEROITER" "$DM_BUFFER" "$DC_BUFFER"; do
  [[ ! -e "$f" ]] || { echo "[ERROR] $f ya existe" >&2; exit 1; }
done

DM_CFG="distill_${DATASET}_dm_ctc"; DC_CFG="distill_${DATASET}_dc_ctc"
if [[ "$DATASET" == "iam" ]]; then DM_CFG="distill_iam_dm_ctc_wr8"; DC_CFG="distill_iam_dc_ctc_wr8"; fi
MATCH_BS="$K"; if [[ "$K" -gt 32 ]]; then MATCH_BS=32; fi

echo "[INFO] === (1) Posiciones de la init aleatoria: ${DATASET} K=${K} ==="
"$PYTHON_BIN" scripts_gen/make_random_init_positions.py --config-name "$DM_CFG" \
  --dataset "${DATASET^^}" --K "$K" --out "$POSITIONS"

COMMON=(K="$K" matching_batch_size="$MATCH_BS" "+init_positions_file=$POSITIONS")

echo "[INFO] === (2) Buffer de 0 iteraciones ==="
"$PYTHON_BIN" src/distill_dm_ctc.py --config-name "$DM_CFG" "${COMMON[@]}" \
  outer_iterations=0 buffer_path="$ZEROITER" hydra.run.dir="$RUN_DIR/hydra_zeroiter"

echo "[INFO] === (3a) DM init aleatoria (round_robin_syn=true, syn_lr=0.1, 2000 it) ==="
"$PYTHON_BIN" src/distill_dm_ctc.py --config-name "$DM_CFG" "${COMMON[@]}" \
  +round_robin_syn=true outer_iterations=2000 syn_lr=0.1 \
  buffer_path="$DM_BUFFER" hydra.run.dir="$RUN_DIR/hydra_dm"

echo "[INFO] === (3b) DC init aleatoria (round_robin_syn=true, syn_lr=0.1, inner_lr=3e-3, 2000 it) ==="
"$PYTHON_BIN" src/distill_dc_ctc.py --config-name "$DC_CFG" "${COMMON[@]}" \
  +round_robin_syn=true outer_iterations=2000 syn_lr=0.1 inner_lr=3e-3 \
  buffer_path="$DC_BUFFER" hydra.run.dir="$RUN_DIR/hydra_dc"

echo "[INFO] === (4) Deriva ==="
"$PYTHON_BIN" src/buffer_drift_budget.py --init "$ZEROITER" --buffers "$DM_BUFFER" "$DC_BUFFER"
echo "[INFO] X2 ${DATASET} K=${K} completada."
