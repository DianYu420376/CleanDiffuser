#!/bin/bash
#SBATCH --job-name=guid_wcg_resweep
#SBATCH --time=02:00:00
#SBATCH --account=bgeo-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm_guidance_synthetic_wcg_resweep_%j.out
#SBATCH --error=slurm_guidance_synthetic_wcg_resweep_%j.err

set -euo pipefail
REPO_DIR="${REPO_DIR:-/u/rzhang26/CleanDiffuser}"
VENV_DIR="${VENV_DIR:-/u/rzhang26/venvs/CleanDiffuser}"
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/pipelines:${REPO_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

echo "Balanced w_cg resweep (score = reward - 2.0 * violation)"
nvidia-smi || true

"${VENV_DIR}/bin/python" pipelines/guidance_synthetic_wcg_resweep.py \
  --test both \
  --device cuda:0 \
  --violation-penalty 2.0 \
  --num-samples 1024 \
  --validation-samples 4096 \
  --coarse-step 0.5 \
  --refine-step 0.1 \
  --refine-radius 2.5

echo "Done."
