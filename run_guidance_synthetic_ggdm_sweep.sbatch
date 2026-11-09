#!/bin/bash
#SBATCH --job-name=ggdm_sweep
#SBATCH --time=04:00:00
#SBATCH --account=bgeo-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm_ggdm_sweep_%j.out
#SBATCH --error=slurm_ggdm_sweep_%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/u/rzhang26/CleanDiffuser}"
VENV_DIR="${VENV_DIR:-/u/rzhang26/venvs/CleanDiffuser}"
PYTHON="${VENV_DIR}/bin/python"

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/pipelines:${REPO_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

echo "GGDM-only hyperparameter sweep on synthetic Tests A/B"
nvidia-smi || true

"${PYTHON}" pipelines/guidance_synthetic_ggdm_sweep_test_a.py \
  --test both \
  --output-root results/guidance_synthetic \
  --device cuda:0 \
  --diffusion-steps 50 \
  --training-diffusion-steps 50 \
  --sampling-steps 50 \
  --n-paths 5 \
  --ggdm-solver ddpm \
  --ggdm-obs-noise-sigma 0.1 \
  --intervals 0.01 0.02 0.03 0.05 0.07 0.1 \
  --beta-coefs 0.5 0.75 1.0 1.5 2.0 3.0 \
  --ggdm-rounds 6 8 10

echo "Done. JSON: results/guidance_synthetic/test_*/ggdm_sweep_results.json"
echo "Figures: results/guidance_synthetic/test_*/figures/test_*_ggdm_sweep*.png"
