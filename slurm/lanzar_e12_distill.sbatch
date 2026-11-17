#!/bin/bash
#SBATCH --job-name=dd_e12_distill
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%A_%a.log

set -euo pipefail

# ============================================================================
# E12: DM and DC initialised with the X5 random subset of each seed, 12 cells.
# X2 recipe (unconditional, round_robin_syn=true, syn_lr=0.1, 2000 it; DC inner_lr=3e-3) with seed=seed
# and, for DC, the original distance + guard (as the DC column of the table, E9). Per task (cell, seed):
#   (1) pool positions of the X5 indices (make_random_init_positions.py, verified line by line);
#   (2) 0-iteration buffer and a check that it is IDENTICAL to the MTT init buffer of that seed
#       (its evaluations are the paired 0-iter of E12);
#   (3) distilled DM and DC; (4) drift against (2).
# Manifest (one line per task): "dataset K seed indices_file mtt_init_buffer".
#
#   sbatch --array=0-41%4 --export=ALL,MANIFEST=slurm/manifests/e12_distill.txt slurm/lanzar_e12_distill.sbatch
# ============================================================================

REPO_ROOT="${REPO_ROOT:?Set REPO_ROOT to the repository root}"
RUNS_DIR="${RUNS_DIR:?Set RUNS_DIR to the directory for runs, buffers and experts}"
PYTHON_BIN="${PYTHON_BIN:?Set PYTHON_BIN to the Python interpreter}"
BUFFER_DIR="$RUNS_DIR/budget_buffers"
: "${MANIFEST:?Set MANIFEST}"

export PYTHONUNBUFFERED=1
cd "$REPO_ROOT"

LINE="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")"
LINE="${LINE//\$RUNS_DIR/$RUNS_DIR}"  # manifests write paths as $RUNS_DIR/...
[[ -n "$LINE" ]] || { echo "[ERROR] Sin linea $((SLURM_ARRAY_TASK_ID + 1)) en $MANIFEST" >&2; exit 1; }
read -r DATASET K SEED INDICES MTT_INIT <<< "$LINE"

RUN_DIR="$RUNS_DIR/e12_${DATASET}_k${K}_s${SEED}"
POSITIONS="$BUFFER_DIR/e12_${DATASET}_init_positions_k${K}_s${SEED}.yaml"
ZEROITER="$BUFFER_DIR/e12_${DATASET}_rand_zeroiter_k${K}_s${SEED}.pt"
DM_BUFFER="$BUFFER_DIR/e12_${DATASET}_dm_randinit_k${K}_s${SEED}.pt"
DC_BUFFER="$BUFFER_DIR/e12_${DATASET}_dc_randinit_k${K}_s${SEED}.pt"
mkdir -p "$RUN_DIR"
[[ -f "$INDICES" ]] || { echo "[ERROR] Falta $INDICES" >&2; exit 1; }
[[ -f "$MTT_INIT" ]] || { echo "[ERROR] Falta $MTT_INIT" >&2; exit 1; }
for f in "$POSITIONS" "$ZEROITER" "$DM_BUFFER" "$DC_BUFFER"; do
  [[ ! -e "$f" ]] || { echo "[ERROR] $f ya existe" >&2; exit 1; }
done

DM_CFG="distill_${DATASET}_dm_ctc"; DC_CFG="distill_${DATASET}_dc_ctc"
if [[ "$DATASET" == "iam" ]]; then DM_CFG="distill_iam_dm_ctc_wr8"; DC_CFG="distill_iam_dc_ctc_wr8"; fi
MATCH_BS="$K"; if [[ "$K" -gt 32 ]]; then MATCH_BS=32; fi

echo "[INFO] === E12 ${DATASET} K=${K} seed=${SEED}: (1) posiciones de ${INDICES} ==="
"$PYTHON_BIN" scripts_gen/make_random_init_positions.py --config-name "$DM_CFG" \
  --dataset "${DATASET^^}" --K "$K" --indices "$INDICES" --out "$POSITIONS"

COMMON=(K="$K" seed="$SEED" matching_batch_size="$MATCH_BS" "+init_positions_file=$POSITIONS")

echo "[INFO] === (2) Buffer de 0 iteraciones + comprobación frente a la init de MTT ==="
"$PYTHON_BIN" src/distill_dm_ctc.py --config-name "$DM_CFG" "${COMMON[@]}" \
  outer_iterations=0 buffer_path="$ZEROITER" hydra.run.dir="$RUN_DIR/hydra_zeroiter"
"$PYTHON_BIN" - "$ZEROITER" "$MTT_INIT" <<'EOF'
import sys, torch
a, b = (torch.load(p, map_location="cpu", weights_only=False) for p in sys.argv[1:3])
assert len(a["images"]) == len(b["images"]), (len(a["images"]), len(b["images"]))
for i, (xa, xb, ya, yb) in enumerate(zip(a["images"], b["images"], a["labels"], b["labels"])):
	assert xa.shape == xb.shape and torch.equal(xa, xb) and torch.equal(ya, yb), f"línea {i} distinta de la init de MTT"
print(f"[INFO] 0-iter idéntico a la init de MTT ({len(a['images'])} líneas, imágenes y etiquetas)")
EOF

echo "[INFO] === (3a) DM init aleatoria (round_robin_syn=true, syn_lr=0.1, 2000 it) ==="
"$PYTHON_BIN" src/distill_dm_ctc.py --config-name "$DM_CFG" "${COMMON[@]}" \
  +round_robin_syn=true outer_iterations=2000 syn_lr=0.1 \
  buffer_path="$DM_BUFFER" hydra.run.dir="$RUN_DIR/hydra_dm"

echo "[INFO] === (3b) DC init aleatoria (round_robin_syn=true, syn_lr=0.1, inner_lr=3e-3, 2000 it, distancia original + guard) ==="
"$PYTHON_BIN" src/distill_dc_ctc.py --config-name "$DC_CFG" "${COMMON[@]}" \
  +round_robin_syn=true outer_iterations=2000 syn_lr=0.1 inner_lr=3e-3 \
  +match_distance=dc_original +skip_nonfinite_syn_grad=true \
  buffer_path="$DC_BUFFER" hydra.run.dir="$RUN_DIR/hydra_dc"

echo "[INFO] === (4) Deriva ==="
"$PYTHON_BIN" src/buffer_drift_budget.py --init "$ZEROITER" --buffers "$DM_BUFFER" "$DC_BUFFER"
echo "[INFO] E12 ${DATASET} K=${K} seed=${SEED} completada."
