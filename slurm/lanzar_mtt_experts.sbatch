#!/bin/bash
#SBATCH --job-name=dd_iam_mtt_experts
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.log

set -euo pipefail

# ============================================================================
# E1 (MTT): builds the expert trajectory buffer that distill_mtt_ctc.py needs as its
# parameter-space matching target -- N INDEPENDENT trainings (different seeds) on the FULL
# IAM pool (6161 lines), with exactly the same recipe as the IAM wr8 full-pool model
# (iam_default + wr8, no lr/epoch overrides) so that the trajectories represent standard
# training, not an ad-hoc recipe. callbacks.ExpertSnapshotter saves the trainable
# parameters after every epoch.
#
# Expected cost: the full-pool model converged in ~35 min (epoch 114, stopped at 139):
# 15 experts x ~35-40 min = ~9-10h (estimate; for the paper the IAM experts took 13.6h and
# needed a second job).
#
#   sbatch slurm/lanzar_mtt_experts.sbatch   # from the repository root
# ============================================================================

REPO_ROOT="${REPO_ROOT:?Set REPO_ROOT to the repository root}"
RUNS_DIR="${RUNS_DIR:?Set RUNS_DIR to the directory for runs, buffers and experts}"
PYTHON_BIN="${PYTHON_BIN:?Set PYTHON_BIN to the Python interpreter}"
EXPERT_DIR="$RUNS_DIR/experts/iam"
RUN_DIR="$RUNS_DIR/mtt_expert_training_${SLURM_JOB_ID}"

export PYTHONUNBUFFERED=1

cd "$REPO_ROOT"
mkdir -p "$EXPERT_DIR" "$RUN_DIR"

# Consistency guards (fail loudly: an incomplete checkout must not silently train with the
# wrong or missing callback):
[[ -f configs/train_expert_wr8.yaml ]] || { echo "[ERROR] Falta configs/train_expert_wr8.yaml -- sincroniza el repo." >&2; exit 1; }
grep -q "class ExpertSnapshotter" src/callbacks/__init__.py || { echo "[ERROR] src/callbacks/__init__.py sin ExpertSnapshotter -- sincroniza src/callbacks/." >&2; exit 1; }

SEEDS=(42 43 44 45 46 47 48 49 50 51 52 53 54 55 56)

i=0
for seed in "${SEEDS[@]}"; do
  i=$((i + 1))
  echo "[INFO] === ${i}/${#SEEDS[@]}: entrenando experto seed=${seed} ==="
  "$PYTHON_BIN" src/run.py --config-name train_expert_wr8 \
    seed="${seed}" \
    dataset_name="iam_mtt_expert_s${seed}" \
    trainer_callbacks.checkpointer.dirpath="$RUN_DIR/weights_s${seed}" \
    trainer_callbacks.expert_snapshotter.dirpath="$EXPERT_DIR/expert_s${seed}" \
    hydra.run.dir="$RUN_DIR/hydra_s${seed}" \
    data_module.train_datasets.iam.num_workers=0
done

echo "[INFO] Buffer de trayectorias expertas completo en $EXPERT_DIR"
echo "[INFO] Usar como expert_dir=$EXPERT_DIR en distill_iam_mtt_ctc.yaml"
