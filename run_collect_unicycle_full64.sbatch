#!/bin/bash
#SBATCH --job-name=collect_unicycle_full64
#SBATCH --time=01:00:00
#SBATCH --account=bgeo-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=slurm_collect_unicycle_full64_%j.out
#SBATCH --error=slurm_collect_unicycle_full64_%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/u/rzhang26/CleanDiffuser}"
VENV_DIR="${VENV_DIR:-/u/rzhang26/venvs/CleanDiffuser}"
TARGET_FULL="${TARGET_FULL:-6000}"
SEED="${SEED:-0}"

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/pipelines:${REPO_DIR}:${PYTHONPATH:-}"
PYTHON="${VENV_DIR}/bin/python"

echo "Collecting ${TARGET_FULL} full-horizon (64-step) unicycle episodes..."
"${PYTHON}" pipelines/collect_unicycle_offline_data.py \
  --output-dir results/unicycle_offline/full64_6k \
  --require-full-horizon \
  --target-full-episodes "${TARGET_FULL}" \
  --horizon 64 \
  --seed "${SEED}" \
  --num-plots 12

echo "Done. Dataset: results/unicycle_offline/full64_6k/unicycle_offline.hdf5"
