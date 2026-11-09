#!/bin/bash
#SBATCH --job-name=inverse_valtest
#SBATCH --account=bgeo-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/u/rzhang26/CleanDiffuser_rebuttal_pnp/experiments/rebuttal_pnp_2027/results/logs/inverse_valtest_%j.out
#SBATCH --error=/u/rzhang26/CleanDiffuser_rebuttal_pnp/experiments/rebuttal_pnp_2027/results/logs/inverse_valtest_%j.err
# Validation tuning (pre-registered selection, frozen) followed by the test evaluation in the same job.
# Usage: VAL_ID=<val run id> TEST_ID=<test run id> [DATA_DIR=...] sbatch inverse_val_then_test.sbatch
set -euo pipefail
WT=/u/rzhang26/CleanDiffuser_rebuttal_pnp
cd "${WT}"
export PYTHONPATH=${WT}/pipelines:${WT}:${WT}/experiments/rebuttal_pnp_2027
export D4RL_SUPPRESS_IMPORT_ERROR=1 MUJOCO_GL=disable PYTHONUNBUFFERED=1
DATA_DIR="${DATA_DIR:-${WT}/experiments/rebuttal_pnp_2027/results/inverse/data_v1}"
PY=/u/rzhang26/venvs/CleanDiffuser/bin/python
nvidia-smi --query-gpu=name,memory.total --format=csv || true
${PY} experiments/rebuttal_pnp_2027/scripts/inverse_run.py --phase val --data-dir "${DATA_DIR}" --run-id "${VAL_ID}"
${PY} experiments/rebuttal_pnp_2027/scripts/inverse_run.py --phase test --data-dir "${DATA_DIR}" \
  --val-run "${WT}/experiments/rebuttal_pnp_2027/results/inverse/${VAL_ID}" --run-id "${TEST_ID}"
