#!/bin/bash
#SBATCH --job-name=guid_baselines
#SBATCH --time=02:30:00
#SBATCH --account=bgeo-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm_guidance_synthetic_baselines_%j.out
#SBATCH --error=slurm_guidance_synthetic_baselines_%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/u/rzhang26/CleanDiffuser}"
VENV_DIR="${VENV_DIR:-/u/rzhang26/venvs/CleanDiffuser}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/guidance_synthetic}"
PYTHON="${VENV_DIR}/bin/python"
TUNE_JSON="${OUTPUT_ROOT}/baseline_tune_best.json"

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/pipelines:${REPO_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

echo "Tune GGDM/DRGD, restore legacy PCG/DCG, regenerate 4-method paper figures"
nvidia-smi || true

echo "=== Phase 1: baseline hyperparameter sweep ==="
"${PYTHON}" pipelines/guidance_synthetic_tune_baselines.py \
  --test both \
  --output-root "${OUTPUT_ROOT}" \
  --output-json "${TUNE_JSON}" \
  --device cuda:0 \
  --diffusion-steps 50 \
  --training-diffusion-steps 50 \
  --sampling-steps 50 \
  --optimization-guidance-last-steps 50 \
  --drgd-steps 50 \
  --ggdm-beta-coef 0.5 \
  --n-paths 3 \
  --w-cg 0.6 \
  --w-cg-test-b 10

echo "=== Phase 2: paper figures with tuned baselines + legacy PCG/DCG ==="
"${PYTHON}" pipelines/guidance_synthetic_visualize_subspace.py \
  --test both \
  --resample \
  --use-legacy-pcg-dcg \
  --output-root "${OUTPUT_ROOT}" \
  --device cuda:0 \
  --diffusion-steps 50 \
  --training-diffusion-steps 50 \
  --sampling-steps 50 \
  --optimization-guidance-last-steps 50 \
  --n-paths 5 \
  --w-cg 0.6 \
  --opt-scale 15 \
  --w-cg-test-b 10 \
  --opt-scale-test-b 10 \
  --ggdm-obs-noise-sigma 0.1 \
  --plot-limit-a 2.0 \
  --plot-limit-b 1.8

echo "Done. Figures under ${OUTPUT_ROOT}/test_*/figures/"
