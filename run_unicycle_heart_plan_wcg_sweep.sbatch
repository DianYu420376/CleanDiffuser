#!/bin/bash
#SBATCH --job-name=heart_plan_wcg_sweep
#SBATCH --time=02:00:00
#SBATCH --account=bgeo-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm_heart_plan_wcg_sweep_%j.out
#SBATCH --error=slurm_heart_plan_wcg_sweep_%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/u/rzhang26/CleanDiffuser}"
VENV_DIR="${VENV_DIR:-/u/rzhang26/venvs/CleanDiffuser}"
CKPT_PATH="${CKPT_PATH:-results/diffuser_unicycle/Unicycle-v0/diffusion_ckpt_latest.pt}"
SEED="${SEED:-0}"
CANDIDATES="${CANDIDATES:-5}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/unicycle_eval/heart_plan_wcg_sweep}"

CONFIGS=(
  standard_w_cg0p3
  standard_w_cg0p5
  standard_w_cg1
  standard_w_cg2
  standard_w_cg5
)

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/pipelines:${REPO_DIR}:${PYTHONPATH:-}"
export D4RL_SUPPRESS_IMPORT_ERROR=1
PYTHON="${VENV_DIR}/bin/python"

echo "Heart plan w_cg sweep: ${CONFIGS[*]}"
echo "checkpoint: ${CKPT_PATH}"
echo "candidates per w_cg: ${CANDIDATES}"

for cfg in "${CONFIGS[@]}"; do
  echo ""
  echo "========== ${cfg} =========="
  "${PYTHON}" pipelines/unicycle_eval.py \
    --eval-task heart_plan \
    --ckpt-path "${CKPT_PATH}" \
    --seed "${SEED}" \
    --device cuda:0 \
    --solver ddim \
    --sample-step-schedule uniform \
    --sampling-steps 20 \
    --diffusion-steps 20 \
    --temperature 1.0 \
    --use-ema \
    --heart-config-names "${cfg}" \
    --heart-plan-candidates "${CANDIDATES}" \
    --heart-reference-steps 256 \
    --output-dir "${OUTPUT_ROOT}/${cfg}"
done

echo ""
echo "Sweep completed. Results under ${OUTPUT_ROOT}/"
