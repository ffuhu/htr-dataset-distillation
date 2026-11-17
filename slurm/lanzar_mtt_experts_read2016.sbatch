#!/bin/bash
#SBATCH --job-name=dd_read2016_mtt_experts
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.log

set -euo pipefail

# ============================================================================
# E1 (MTT on READ2016): builds the expert trajectory buffer for READ2016 -- N INDEPENDENT
# trainings (different seeds) on the FULL READ2016 pool (8354 feasible lines), with exactly
# the same recipe as the READ2016 wr8 full-pool model, so that the trajectories represent
# standard training. callbacks.ExpertSnapshotter saves the trainable parameters after every
# epoch.
#
# Expected cost: the full-pool model converged in ~79 epochs / ~30 min of training: 15
# experts x ~30-40 min = ~7.5-10h.
#
#   sbatch slurm/lanzar_mtt_experts_read2016.sbatch   # from the repository root
# ============================================================================

REPO_ROOT="${REPO_ROOT:?Set REPO_ROOT to the repository root}"
RUNS_DIR="${RUNS_DIR:?Set RUNS_DIR to the directory for runs, buffers and experts}"
PYTHON_BIN="${PYTHON_BIN:?Set PYTHON_BIN to the Python interpreter}"
EXPERT_DIR="$RUNS_DIR/experts/read2016"
RUN_DIR="$RUNS_DIR/read2016_mtt_expert_training_${SLURM_JOB_ID}"

export PYTHONUNBUFFERED=1

cd "$REPO_ROOT"
mkdir -p "$EXPERT_DIR" "$RUN_DIR"

[[ -f configs/train_expert_read2016_wr8.yaml ]] || { echo "[ERROR] Falta configs/train_expert_read2016_wr8.yaml -- sincroniza el repo." >&2; exit 1; }
grep -q "class ExpertSnapshotter" src/callbacks/__init__.py || { echo "[ERROR] src/callbacks/__init__.py sin ExpertSnapshotter -- sincroniza src/callbacks/." >&2; exit 1; }

SEEDS=(42 43 44 45 46 47 48 49 50 51 52 53 54 55 56)

i=0
for seed in "${SEEDS[@]}"; do
  i=$((i + 1))
  echo "[INFO] === ${i}/${#SEEDS[@]}: entrenando experto READ2016 seed=${seed} ==="
  "$PYTHON_BIN" src/run.py --config-name train_expert_read2016_wr8 \
    seed="${seed}" \
    dataset_name="read2016_mtt_expert_s${seed}" \
    trainer_callbacks.checkpointer.dirpath="$RUN_DIR/weights_s${seed}" \
    trainer_callbacks.expert_snapshotter.dirpath="$EXPERT_DIR/expert_s${seed}" \
    hydra.run.dir="$RUN_DIR/hydra_s${seed}" \
    data_module.train_datasets.read2016.num_workers=0
done

echo "[INFO] Buffer de trayectorias expertas READ2016 completo en $EXPERT_DIR"
echo "[INFO] Usar como expert_dir=$EXPERT_DIR en distill_read2016_mtt_ctc.yaml"
