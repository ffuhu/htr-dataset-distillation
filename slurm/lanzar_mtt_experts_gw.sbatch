#!/bin/bash
#SBATCH --job-name=dd_gw_mtt_experts
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.log

set -euo pipefail

# ============================================================================
# E1 (MTT on GW): builds the expert trajectory buffer for GW -- N INDEPENDENT trainings
# (different seeds) on the FULL GW pool (357 lines), with exactly the same recipe as the GW
# wr8 full-pool model, so that the trajectories represent standard training, not an ad-hoc
# recipe. callbacks.ExpertSnapshotter saves the trainable parameters after every epoch.
#
# Expected cost: GW is tiny (357 lines, batch=8, 45 steps/epoch) -- the full-pool model
# converged in ~79 epochs / <1 min. 15 experts fit easily in a small fraction of 12h.
#
#   sbatch slurm/lanzar_mtt_experts_gw.sbatch   # from the repository root
# ============================================================================

REPO_ROOT="${REPO_ROOT:?Set REPO_ROOT to the repository root}"
RUNS_DIR="${RUNS_DIR:?Set RUNS_DIR to the directory for runs, buffers and experts}"
PYTHON_BIN="${PYTHON_BIN:?Set PYTHON_BIN to the Python interpreter}"
EXPERT_DIR="$RUNS_DIR/experts/gw"
RUN_DIR="$RUNS_DIR/gw_mtt_expert_training_${SLURM_JOB_ID}"

export PYTHONUNBUFFERED=1

cd "$REPO_ROOT"
mkdir -p "$EXPERT_DIR" "$RUN_DIR"

[[ -f configs/train_expert_gw_wr8.yaml ]] || { echo "[ERROR] Falta configs/train_expert_gw_wr8.yaml -- sincroniza el repo." >&2; exit 1; }
grep -q "class ExpertSnapshotter" src/callbacks/__init__.py || { echo "[ERROR] src/callbacks/__init__.py sin ExpertSnapshotter -- sincroniza src/callbacks/." >&2; exit 1; }

SEEDS=(42 43 44 45 46 47 48 49 50 51 52 53 54 55 56)

i=0
for seed in "${SEEDS[@]}"; do
  i=$((i + 1))
  echo "[INFO] === ${i}/${#SEEDS[@]}: entrenando experto GW seed=${seed} ==="
  "$PYTHON_BIN" src/run.py --config-name train_expert_gw_wr8 \
    seed="${seed}" \
    dataset_name="gw_mtt_expert_s${seed}" \
    trainer_callbacks.checkpointer.dirpath="$RUN_DIR/weights_s${seed}" \
    trainer_callbacks.expert_snapshotter.dirpath="$EXPERT_DIR/expert_s${seed}" \
    hydra.run.dir="$RUN_DIR/hydra_s${seed}"
done

echo "[INFO] Buffer de trayectorias expertas GW completo en $EXPERT_DIR"
echo "[INFO] Usar como expert_dir=$EXPERT_DIR en distill_gw_mtt_ctc.yaml"
