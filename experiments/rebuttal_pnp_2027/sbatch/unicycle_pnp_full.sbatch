#!/bin/bash
#SBATCH --job-name=uni_pnp_full
#SBATCH --account=bgeo-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/u/rzhang26/CleanDiffuser_rebuttal_pnp/experiments/rebuttal_pnp_2027/results/logs/unicycle_pnp_full_%j.out
#SBATCH --error=/u/rzhang26/CleanDiffuser_rebuttal_pnp/experiments/rebuttal_pnp_2027/results/logs/unicycle_pnp_full_%j.err
# Unicycle PnP-PGD rebuttal: (1) GPU smoke test, (2) validation grid + selection,
# (3) test evaluation of all methods with the frozen selection, all in ONE GPU job.
set -euo pipefail
W=/u/rzhang26/CleanDiffuser_rebuttal_pnp
E=$W/experiments/rebuttal_pnp_2027
PY=/u/rzhang26/venvs/CleanDiffuser/bin/python
export PYTHONPATH=$W/pipelines:$W:$E
export D4RL_SUPPRESS_IMPORT_ERROR=1 MUJOCO_GL=disable PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd $W
J=${SLURM_JOB_ID}
nvidia-smi || true
echo "=== (1) smoke test (dummy selection; 3 candidates, random_smooth_1) ==="
$PY $E/scripts/unicycle_pnp_test.py --run-id _smoke_gpu_test_${J} \
  --selection-json $E/results/unicycle/_smoke_dummy_selection.json \
  --references random_smooth_1 --n-candidates 3
echo "=== (2) validation ==="
$PY $E/scripts/unicycle_pnp_validate.py --run-id val_${J}
echo "=== (3) test ==="
$PY $E/scripts/unicycle_pnp_test.py --run-id test_${J} \
  --selection-json $E/results/unicycle/val_${J}/selection.json
echo "DONE"
