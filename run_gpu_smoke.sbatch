#!/bin/bash
#SBATCH --job-name=cleandiffuser_gpu_smoke
#SBATCH --time=00:15:00
#SBATCH --account=bgeo-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=slurm_cleandiffuser_gpu_smoke_%j.out
#SBATCH --error=slurm_cleandiffuser_gpu_smoke_%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/u/rzhang26/CleanDiffuser}"
VENV_DIR="${VENV_DIR:-/u/rzhang26/venvs/cleandiffuser}"
CONDA_SH="${CONDA_SH:-/u/rzhang26/miniconda3/etc/profile.d/conda.sh}"

cd "${REPO_DIR}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "ERROR: venv python not found at ${VENV_DIR}/bin/python"
  exit 1
fi

# shellcheck source=/dev/null
source "${CONDA_SH}"
conda activate "${VENV_DIR}"
module load python/miniforge3_pytorch/2.10.0

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

echo "============================================================"
echo "CleanDiffuser GPU smoke test"
echo "============================================================"
echo "hostname: $(hostname)"
echo "repo: ${REPO_DIR}"
echo "venv: ${VENV_DIR}"
echo "python: $(which python)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "============================================================"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
else
  echo "nvidia-smi not found"
fi

python scripts/gpu_smoke_test.py

echo "Smoke test job completed."
