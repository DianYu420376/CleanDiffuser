#!/bin/bash
#SBATCH --job-name=guid_paper_n5
#SBATCH --time=06:00:00
#SBATCH --account=bgeo-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm_guidance_synthetic_paper_n10_r50_%j.out
#SBATCH --error=slurm_guidance_synthetic_paper_n10_r50_%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/u/rzhang26/CleanDiffuser}"
VENV_DIR="${VENV_DIR:-/u/rzhang26/venvs/CleanDiffuser}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/guidance_synthetic}"
PYTHON="${VENV_DIR}/bin/python"

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/pipelines:${REPO_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

echo "Paper figures: n_paths=5, GGDM-mean R=50 beta=0.5, PCG/DCG/DRGD/GGDM"
nvidia-smi || true

"${PYTHON}" pipelines/guidance_synthetic_visualize_subspace.py \
  --test both \
  --resample \
  --no-use-legacy-pcg-dcg \
  --tuned-drgd-only \
  --output-root "${OUTPUT_ROOT}" \
  --device cuda:0 \
  --diffusion-steps 50 \
  --training-diffusion-steps 50 \
  --sampling-steps 50 \
  --optimization-guidance-last-steps 50 \
  --n-paths 5 \
  --ggdm-rounds 50 \
  --ggdm-beta-coef 0.5 \
  --ggdm-interval 0.03 \
  --ggdm-obs-noise-sigma 0.1 \
  --w-cg 0.6 \
  --opt-scale 15 \
  --w-cg-test-b 10 \
  --opt-scale-test-b 10 \
  --plot-limit-a 2.0 \
  --plot-limit-b 1.8

echo "Done. Figures under ${OUTPUT_ROOT}/test_*/figures/"
