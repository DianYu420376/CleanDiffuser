#!/bin/bash
#SBATCH --job-name=ggdm_sweep_a
#SBATCH --time=03:00:00
#SBATCH --account=bgeo-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm_ggdm_sweep_test_a_%j.out
#SBATCH --error=slurm_ggdm_sweep_test_a_%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/u/rzhang26/CleanDiffuser}"
VENV_DIR="${VENV_DIR:-/u/rzhang26/venvs/CleanDiffuser}"
PYTHON="${VENV_DIR}/bin/python"

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/pipelines:${REPO_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

echo "GGDM-only sweep on synthetic Test A (no PCG/DCG/DRGD reruns)"
nvidia-smi || true

"${PYTHON}" pipelines/guidance_synthetic_ggdm_sweep_test_a.py \
  --output-root results/guidance_synthetic \
  --output-json results/guidance_synthetic/test_a/ggdm_sweep_results.json \
  --output-name test_a_ggdm_sweep \
  --device cuda:0 \
  --diffusion-steps 50 \
  --training-diffusion-steps 50 \
  --sampling-steps 50 \
  --n-paths 3 \
  --ggdm-solver ddpm \
  --ggdm-obs-noise-sigma 0.1 \
  --ggdm-rounds 10

echo "Done. Figures: results/guidance_synthetic/test_a/figures/test_a_ggdm_sweep*.png"
