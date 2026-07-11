#!/bin/bash
#SBATCH --job-name=heart_plan_unguided
#SBATCH --time=00:45:00
#SBATCH --account=bgeo-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm_heart_plan_unguided_%j.out
#SBATCH --error=slurm_heart_plan_unguided_%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/u/rzhang26/CleanDiffuser}"
VENV_DIR="${VENV_DIR:-/u/rzhang26/venvs/CleanDiffuser}"
CKPT_PATH="${CKPT_PATH:-results/diffuser_unicycle/Unicycle-v0/full64_6k/diffusion_ckpt_latest.pt}"
SEED="${SEED:-0}"
CANDIDATES="${CANDIDATES:-100}"
SAMPLING_STEPS="${SAMPLING_STEPS:-100}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/unicycle_eval/heart_plan_unguided_100_candidates}"

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/pipelines:${REPO_DIR}:${PYTHONPATH:-}"
export D4RL_SUPPRESS_IMPORT_ERROR=1
PYTHON="${VENV_DIR}/bin/python"

echo "Heart plan unguided baseline (DDIM ${SAMPLING_STEPS} reverse steps, prior=init state)"
echo "checkpoint: ${CKPT_PATH}"
echo "candidates: ${CANDIDATES}"

"${PYTHON}" pipelines/unicycle_eval.py \
  --eval-task heart_plan \
  --ckpt-path "${CKPT_PATH}" \
  --seed "${SEED}" \
  --device cuda:0 \
  --solver ddim \
  --sample-step-schedule uniform \
  --sampling-steps "${SAMPLING_STEPS}" \
  --diffusion-steps 20 \
  --temperature 1.0 \
  --use-ema \
  --heart-config-names standard_w_cg0 \
  --heart-plan-candidates "${CANDIDATES}" \
  --heart-reference-steps 256 \
  --output-dir "${OUTPUT_ROOT}"

echo ""
echo "Done. Results under ${OUTPUT_ROOT}/"
