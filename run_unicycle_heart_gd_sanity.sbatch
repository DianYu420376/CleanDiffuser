#!/bin/bash
#SBATCH --job-name=heart_gd_sanity
#SBATCH --time=00:30:00
#SBATCH --account=bgeo-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=slurm_heart_gd_sanity_%j.out
#SBATCH --error=slurm_heart_gd_sanity_%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/u/rzhang26/CleanDiffuser}"
VENV_DIR="${VENV_DIR:-/u/rzhang26/venvs/CleanDiffuser}"
SEED="${SEED:-0}"
SAMPLING_STEPS="${SAMPLING_STEPS:-100}"
GD_LR="${GD_LR:-}"
OUTPUT_DIR="${OUTPUT_DIR:-results/unicycle_eval/heart_gd_sanity_s100}"

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/pipelines:${REPO_DIR}:${PYTHONPATH:-}"
export D4RL_SUPPRESS_IMPORT_ERROR=1
PYTHON="${VENV_DIR}/bin/python"

echo "Heart GD sanity: no diffusion, ${SAMPLING_STEPS} GD steps${GD_LR:+, lr=${GD_LR}}"

GD_EXTRA=()
if [[ -n "${GD_LR}" ]]; then
  GD_EXTRA+=(--gd-lr "${GD_LR}")
fi

"${PYTHON}" pipelines/unicycle_eval.py \
  --eval-task heart_gd \
  --seed "${SEED}" \
  --device cpu \
  --sampling-steps "${SAMPLING_STEPS}" \
  "${GD_EXTRA[@]}" \
  --temperature 1.0 \
  --heart-reference-steps 256 \
  --output-dir "${OUTPUT_DIR}"
