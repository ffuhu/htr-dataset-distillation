# Dataset Distillation for Handwritten Text Recognition: An Empirical Study

Code to reproduce the experiments of the paper (WACV 2027 Workshops, VisionDocs). It adapts three
dataset-distillation methods to CTC line recognition — distribution matching (DM), gradient matching
(DC/GM) and trajectory matching (MTT) — and compares them with the same number of real lines (random and
text-medoid selection) on IAM, George Washington (GW), READ 2016 and LAM at 1 %, 5 % and 10 % budgets.

The training code (CRNN-CTC, data modules, Hydra configs) builds on the
OMR-CRNN codebase by Eric Ayllón.

Log and error messages printed by the scripts are in Spanish.

## Contents

```
src/            training (run.py), distillation (distill_{dm,dc,mtt}_ctc.py), diagnostics, models, data modules
src/data/prep_scripts/   dataset preparation (IAM, GW, READ 2016, LAM -> parquet)
configs/        Hydra configs (datasets, budgets, models, distillation, MTT experts)
data/           index files: selected subsets per budget/seed, text clusters, vocabularies, infeasible lines
slurm/          SLURM launchers (one per experiment step) and manifests/ (one line per array task)
scripts_gen/    index generation and helpers
tests/          unit tests
```

## Setup

```bash
pip install -r requirements.txt    # Python 3.13.7; exact environment of the runs (CUDA 12.6 wheels)
python src/code_version.py write   # once after cloning, and after every commit
```

Expert training, full-pool baselines and some distillations ran in a second environment that differs in the
CUDA 13.0 build of torch and in minor versions of other packages (e.g. numpy 2.4.6, pillow 12.2.0, pyarrow 24.0.0).

Every run checks that the tracked files of `src/ configs/ slurm/ scripts_gen/` match `CODE_VERSION` (commit + md5
per file) and stops otherwise. `code_version.py write` refuses to run on a dirty tree.

Runs log to Weights & Biases (project `OMR CRNN-CTC`): export `WANDB_API_KEY` and `WANDB_ENTITY` before
submitting jobs.

## Data

Datasets are not redistributed. All commands run from the repository root and write
`datasets/<DATASET>/parquet/{train,val,test}.parquet` (IAM: `train, val_1, val_2, test`).

| Dataset | Source | Command |
|---|---|---|
| IAM | IAM Handwriting Database (lines + `ascii/lines.txt`) and the split lists `trainset.txt`, `validationset1.txt`, `validationset2.txt`, `testset.txt` (Large Writer Independent Text Line Recognition Task) | `python src/data/prep_scripts/prepare_iam.py <iam_dir>` |
| GW | downloaded by the script (15 pages, line crops rebuilt from word boxes) | `python src/data/prep_scripts/prepare_gw.py datasets/GW` |
| READ 2016 | ICFHR 2016 competition, Zenodo 10.5281/zenodo.1164045 | `python src/data/prep_scripts/prepare_read2016.py <raw_dir> datasets/READ2016` |
| LAM | LAM dataset (Cascianelli et al., ICPR 2022), `basic` split | `python src/data/prep_scripts/prepare_lam.py <raw_dir> datasets/LAM` |

`data/` holds the exact subsets used in the paper: `medoids_k{K}_indices.yaml`, `random_baseline_K{K}_indices.yaml`
(single random subset), `x5_random_K{K}_s{seed}_indices.yaml` (one random subset per seed, the "Random" of the
paper), `text_clusters_k{K}.yaml` (pseudo-classes and medoids), vocabularies and
`READ2016/infeasible_wr8_indices.yaml` (12 lines removed from the READ 2016 pool). To regenerate them:
`PYTHON_BIN=... bash scripts_gen/gen_indices.sh`, then `python scripts_gen/fix_read2016_medoid_indices.py`
(translates READ 2016 medoid positions in the filtered pool to raw parquet indices).
`python scripts_gen/pool_feasibility.py --config-name <distill config> --dataset <DATASET>` counts CTC-infeasible
lines (Table 1).

Budgets (K = number of lines):

| | 1 % | 5 % | 10 % |
|---|---|---|---|
| IAM (`iam`) | 62 | 308 | 616 |
| GW (`gw`) | 4 | 18 | 36 |
| READ 2016 (`read2016`) | 84 | 418 | 835 |
| LAM (`lam`) | 198 | 991 | 1983 |

## Running the experiments

All launchers need three environment variables (no defaults):

```bash
export REPO_ROOT=/path/to/HTR-dataset-distillation
export RUNS_DIR=/path/to/runs        # run dirs, distilled buffers ($RUNS_DIR/budget_buffers), MTT experts
export PYTHON_BIN=/path/to/venv/bin/python
cd $REPO_ROOT
```

Choose the partition with `sbatch -p <partition>`. Manifest-based launchers take `MANIFEST=slurm/manifests/<name>.txt`
and one array task per manifest line (`--array=0-<lines-1>`); manifests write paths as `$RUNS_DIR/...`. Steps that
consume buffers must wait for the steps that write them (use `--dependency`). Seeds are 0, 42, 1337 (GW also 7, 123).

Experiment IDs as in the paper's supplementary material:

| ID | What | Commands |
|---|---|---|
| E1 | Main sweep: real lines (single random subset, medoids), DM-cond, DC-diag, MTT; 12 cells × 3 seeds | per (DATASET, K): `lanzar_nucleo_baseline_budget`; `lanzar_distill_budget` → `lanzar_distill_eval_budget` with `TAG=dm_cond BUFFER_PATH=$RUNS_DIR/budget_buffers/${DATASET}_dm_cond_k${K}.pt` and `TAG=dc_diag …_dc_diag_k${K}.pt`. MTT: `lanzar_mtt_experts{,_gw,_read2016,_lam}` (writes `$RUNS_DIR/experts/<dataset>`) → per seed `lanzar_mtt_distill_budget` (`EXPERT_DIR=$RUNS_DIR/experts/$DATASET SEED=…`) → `lanzar_mtt_eval_budget` (`BUFFER_PATH=$RUNS_DIR/budget_buffers/${DATASET}_mtt_ctc_k${K}_s${SEED}.pt`) |
| E2 | Full training pool, 3 seeds | `DATASET=… lanzar_full_baseline_budget` |
| E3 | 0-iteration controls (medoid init and MTT init), drift | `lanzar_distill_zeroiter_budget` → `lanzar_distill_eval_zeroiter_budget`; `lanzar_mtt_init_buffers` → `lanzar_mtt_init_eval_budget`; `lanzar_mtt_drift_budget` |
| E4 | DM/DC without pseudo-class conditioning | `lanzar_distill_noclass_budget` → `lanzar_distill_eval_noclass_budget` |
| E5 | DM/DC initialised from the random lines | `lanzar_x2_distill` → `lanzar_eval_manifest` (`x2_eval`) |
| E6 | Optimisation strength: DM (X1), DC with original distance + guard (X1b), MTT `pixel_lr` (X3); phase B | `lanzar_x1_distill` (`x1_distill`) → `lanzar_eval_manifest` (`x1_eval_u30`, `x1_eval_u300`, `x1_eval_u3000`); X1b: `MANIFEST=slurm/manifests/x1b_distill.txt DC_EXTRA="+match_distance=dc_original +skip_nonfinite_syn_grad=true" lanzar_x1_distill` → `x1b_eval_u{30,300,3000}`; phase B: `x1_phaseB_eval`, `x1_phaseB_eval_dcread`; X3: `lanzar_x3_mtt_distill` (`x3_distill`) → `x3_eval` |
| E7 | Random with one subset per seed (1 %) | `lanzar_real_manifest` (`x5_real`) |
| E8 | Pipeline (PIL vs tensor) vs hardware | `lanzar_real_manifest` (`ctl_real`) and `lanzar_eval_manifest` (`ctl_tensor`) on the same node |
| E9 | DC column with the original DC distance | `lanzar_dco_distill` → `lanzar_eval_manifest` (`dco_eval`) |
| E10 | GW with 5 seeds | `lanzar_real_manifest` (`gw5_real`); `lanzar_eval_manifest` (`gw5_tensor`); `MANIFEST=slurm/manifests/gw5_mtt_distill.txt lanzar_x3_mtt_distill` → `$PYTHON_BIN src/make_init_buffer.py --config-name distill_gw_mtt_ctc --pairs $(sed "s|\$RUNS_DIR|$RUNS_DIR|g" slurm/manifests/gw5_mtt_init_pairs.txt)` → `lanzar_eval_manifest` (`gw5_mtt_eval`) |
| E11 | READ 2016 medoids with corrected indices | included in E1: `data/READ2016/medoids_k*_indices.yaml` are the corrected indices |
| E12 | Random with one subset per seed in all cells; DM/DC initialised from those lines | `lanzar_real_manifest` (`e12_x5_real`); `lanzar_e12_distill` (`e12_distill`, needs the MTT init buffers of E3) → `lanzar_eval_manifest` (`e12_eval_dm`, `e12_eval_dc`) |
| E13 | Objective checks: MTT reachability with Adam/SGD experts; does DM/DC reduce its own objective | `lanzar_e13_experts` (`e13_g0_experts`); `lanzar_e13_diag` (`e13_diag_adam`, `e13_diag_sgd_g0`); `lanzar_e13b_objective` (`e13b_objective`: DM rows; `e13b_dc_v2`: DC with paired dropout masks, supersedes the DC rows of `e13b_objective`) |

Example (E1, IAM 5 %):

```bash
DATASET=iam K=308 sbatch -p <partition> slurm/lanzar_nucleo_baseline_budget.sbatch
DATASET=iam K=308 sbatch -p <partition> slurm/lanzar_distill_budget.sbatch
DATASET=iam K=308 TAG=dm_cond BUFFER_PATH=$RUNS_DIR/budget_buffers/iam_dm_cond_k308.pt \
  sbatch -p <partition> slurm/lanzar_distill_eval_budget.sbatch
sbatch -p <partition> --array=0-11 --export=ALL,MANIFEST=slurm/manifests/x5_real.txt slurm/lanzar_real_manifest.sbatch
```

## Notes on reproducing the reported numbers

- Test CER: runs before the fix in the test step compared against the reference without its last character; the
  paper reports the corrected CER, recomputed from the logged per-line predictions. This code computes the
  corrected CER directly.
- `src/utils/differentiable_ctc.py` zeroes CTC-infeasible samples. Before this fix, one infeasible IAM line was not
  zeroed in DC runs initialised from the IAM 10 % medoids (DC-diag, DC original distance, unconditioned) and in E12
  DC seed 0 at K = 308 and 616; re-running those cells may give different numbers.
- READ 2016 drops lines with transcriptions of at most one character during preparation.
- MTT on LAM uses 5 expert trajectories (seeds 42–46), the others 15. In the paper the fifth one was cut at epoch
  90 by the job time limit; MTT on LAM only reads expert epochs ≤ 41.
- Most evaluation runs used NVIDIA A100 GPUs; E8 measures a hardware effect of about 0.5 CER points on IAM 5 %.

## Citation

```bibtex
@inproceedings{fuenteshurtado2027ddhtr,
  title     = {Dataset Distillation for Handwritten Text Recognition: An Empirical Study},
  author    = {Fuentes-Hurtado, F{\'e}lix and Ayll{\'o}n, Eric and Alfaro, Juan C.},
  booktitle = {Proceedings of the Winter Conference on Applications of Computer Vision (WACV) Workshops},
  year      = {2027}
}
```

## License

MIT (see `LICENSE`).
