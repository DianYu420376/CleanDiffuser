#!/bin/bash
#SBATCH --job-name=uni_pnp_supp
#SBATCH --account=bgeo-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=/u/rzhang26/CleanDiffuser_rebuttal_pnp/experiments/rebuttal_pnp_2027/results/logs/unicycle_pnp_supp_%j.out
#SBATCH --error=/u/rzhang26/CleanDiffuser_rebuttal_pnp/experiments/rebuttal_pnp_2027/results/logs/unicycle_pnp_supp_%j.err
# Unicycle PnP-PGD post-selection supplementary job (reviewer fixes): Part A = K up to 20 reporting pass on
# test with the FROZEN selection (not used for selection); Part B = timing diagnostic for the PnP init overhead.
set -euo pipefail
W=/u/rzhang26/CleanDiffuser_rebuttal_pnp
E=$W/experiments/rebuttal_pnp_2027
PY=/u/rzhang26/venvs/CleanDiffuser/bin/python
export PYTHONPATH=$W/pipelines:$W:$E
export D4RL_SUPPRESS_IMPORT_ERROR=1 MUJOCO_GL=disable PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd $W
J=${SLURM_JOB_ID}
nvidia-smi || true
R=$E/results/unicycle
$PY $E/scripts/unicycle_pnp_supp.py --run-id supp_${J} \
  --selection-json $R/val_3565048/selection.json --ref-test-dir $R/test_3565048
echo "DONE"
