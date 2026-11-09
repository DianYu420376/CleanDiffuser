#!/bin/bash
#SBATCH --job-name=ggdm_cmp_n20
#SBATCH --time=02:00:00
#SBATCH --account=bgeo-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm_ggdm_mean_compare_test_a_n20_%j.out
#SBATCH --error=slurm_ggdm_mean_compare_test_a_n20_%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/u/rzhang26/CleanDiffuser}"
VENV_DIR="${VENV_DIR:-/u/rzhang26/venvs/CleanDiffuser}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/guidance_synthetic}"
PYTHON="${VENV_DIR}/bin/python"

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/pipelines:${REPO_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

echo "GGDM particle vs mean comparison: Test A, n_paths=20, tuned best config"
nvidia-smi || true

"${PYTHON}" pipelines/guidance_synthetic_visualize_subspace.py \
  --test test_a \
  --compare-ggdm-mean \
  --output-root "${OUTPUT_ROOT}" \
  --device cuda:0 \
  --diffusion-steps 50 \
  --training-diffusion-steps 50 \
  --sampling-steps 50 \
  --optimization-guidance-last-steps 50 \
  --n-paths 20 \
  --w-cg 0.6 \
  --opt-scale 15 \
  --plot-limit-a 2.0 \
  --output-name test_a_ggdm_particle_vs_mean_s50_last50_n20

echo "Done."
