#!/bin/bash
#SBATCH --job-name=inverse_smoke
#SBATCH --account=bgeo-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=/u/rzhang26/CleanDiffuser_rebuttal_pnp/experiments/rebuttal_pnp_2027/results/logs/inverse_smoke_%j.out
#SBATCH --error=/u/rzhang26/CleanDiffuser_rebuttal_pnp/experiments/rebuttal_pnp_2027/results/logs/inverse_smoke_%j.err
# Sparse trajectory reconstruction (smoke phase). Usage:
#   DATA_DIR=<results/inverse/data_v1> [VAL_RUN=<val run dir>] [RUN_ID=<id>] sbatch inverse_smoke.sbatch
set -euo pipefail
WT=/u/rzhang26/CleanDiffuser_rebuttal_pnp
cd "${WT}"
export PYTHONPATH=${WT}/pipelines:${WT}:${WT}/experiments/rebuttal_pnp_2027
export D4RL_SUPPRESS_IMPORT_ERROR=1 MUJOCO_GL=disable PYTHONUNBUFFERED=1
DATA_DIR="${DATA_DIR:-${WT}/experiments/rebuttal_pnp_2027/results/inverse/data_v1}"
EXTRA=()
[ -n "${VAL_RUN:-}" ] && EXTRA+=(--val-run "${VAL_RUN}")
[ -n "${RUN_ID:-}" ] && EXTRA+=(--run-id "${RUN_ID}")
nvidia-smi --query-gpu=name,memory.total --format=csv || true
/u/rzhang26/venvs/CleanDiffuser/bin/python experiments/rebuttal_pnp_2027/scripts/inverse_run.py \
  --phase smoke --data-dir "${DATA_DIR}" "${EXTRA[@]}"
