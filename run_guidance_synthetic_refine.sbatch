#!/bin/bash
#SBATCH --job-name=guid_synth_refine
#SBATCH --time=02:00:00
#SBATCH --account=bgeo-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm_guidance_synthetic_refine_%j.out
#SBATCH --error=slurm_guidance_synthetic_refine_%j.err

set -euo pipefail
REPO_DIR="${REPO_DIR:-/u/rzhang26/CleanDiffuser}"
VENV_DIR="${VENV_DIR:-/u/rzhang26/venvs/CleanDiffuser}"
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/pipelines:${REPO_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
"${VENV_DIR}/bin/python" pipelines/guidance_synthetic_refine.py
