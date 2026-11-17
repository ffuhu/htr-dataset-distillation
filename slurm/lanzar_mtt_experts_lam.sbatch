#!/bin/bash
#SBATCH --job-name=dd_lam_mtt_experts
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.log

set -euo pipefail

# ============================================================================
# E1 (MTT on LAM): builds the expert trajectory buffer for LAM -- N INDEPENDENT trainings
# (different seeds) on the FULL LAM pool (19829 feasible lines), with exactly the same recipe
# as the LAM wr8 full-pool model, so that the trajectories represent standard training.
# callbacks.ExpertSnapshotter saves the trainable parameters after every epoch.
#
# Expected cost: the full-pool model converged in ~124-149 epochs, ~3:43 h of training
# because of the size of LAM (2479 steps/epoch at batch=8). MTT samples (expert, start,
# target) at random from the whole buffer at every distillation iteration, not one
# trajectory per evaluation seed.
#
#   sbatch slurm/lanzar_mtt_experts_lam.sbatch   # from the repository root
# ============================================================================

REPO_ROOT="${REPO_ROOT:?Set REPO_ROOT to the repository root}"
RUNS_DIR="${RUNS_DIR:?Set RUNS_DIR to the directory for runs, buffers and experts}"
PYTHON_BIN="${PYTHON_BIN:?Set PYTHON_BIN to the Python interpreter}"
EXPERT_DIR="$RUNS_DIR/experts/lam"
RUN_DIR="$RUNS_DIR/lam_mtt_expert_training_${SLURM_JOB_ID}"

export PYTHONUNBUFFERED=1

cd "$REPO_ROOT"
mkdir -p "$EXPERT_DIR" "$RUN_DIR"

[[ -f configs/train_expert_lam_wr8.yaml ]] || { echo "[ERROR] Falta configs/train_expert_lam_wr8.yaml -- sincroniza el repo." >&2; exit 1; }
grep -q "class ExpertSnapshotter" src/callbacks/__init__.py || { echo "[ERROR] src/callbacks/__init__.py sin ExpertSnapshotter -- sincroniza src/callbacks/." >&2; exit 1; }

# Paper: 5 trajectories (seeds 42-46). The original job hit its time limit during the 5th one
# (cut at epoch 90; MTT on LAM only uses epochs <= max_start_epoch + 1 = 41), so the paper's
# expert buffer contains only these 5 seeds.
SEEDS=(42 43 44 45 46)

i=0
for seed in "${SEEDS[@]}"; do
  i=$((i + 1))
  echo "[INFO] === ${i}/${#SEEDS[@]}: entrenando experto LAM seed=${seed} ==="
  "$PYTHON_BIN" src/run.py --config-name train_expert_lam_wr8 \
    seed="${seed}" \
    dataset_name="lam_mtt_expert_s${seed}" \
    trainer_callbacks.checkpointer.dirpath="$RUN_DIR/weights_s${seed}" \
    trainer_callbacks.expert_snapshotter.dirpath="$EXPERT_DIR/expert_s${seed}" \
    hydra.run.dir="$RUN_DIR/hydra_s${seed}" \
    data_module.train_datasets.lam.num_workers=0
done

echo "[INFO] Buffer de trayectorias expertas LAM completo en $EXPERT_DIR"
echo "[INFO] Usar como expert_dir=$EXPERT_DIR en distill_lam_mtt_ctc.yaml"
