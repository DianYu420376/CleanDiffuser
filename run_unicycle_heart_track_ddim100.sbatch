#!/bin/bash
#SBATCH --job-name=heart_track_ddim100
#SBATCH --time=06:00:00
#SBATCH --account=bgeo-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm_heart_track_ddim100_%j.out
#SBATCH --error=slurm_heart_track_ddim100_%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/u/rzhang26/CleanDiffuser}"
VENV_DIR="${VENV_DIR:-/u/rzhang26/venvs/CleanDiffuser}"
CKPT_PATH="${CKPT_PATH:-results/diffuser_unicycle/Unicycle-v0/diffusion_ckpt_latest.pt}"
SEED="${SEED:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/unicycle_eval/heart_track_ddim100_rawgrad}"

CONFIGS=(
  standard_w_cg0p3
  standard_w_cg0p5
  standard_w_cg1
)

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/pipelines:${REPO_DIR}:${PYTHONPATH:-}"
export D4RL_SUPPRESS_IMPORT_ERROR=1
PYTHON="${VENV_DIR}/bin/python"

echo "Heart MPC tracking: DDIM 100 reverse steps, raw reward gradient"
echo "checkpoint: ${CKPT_PATH}"
echo "configs: ${CONFIGS[*]}"

for cfg in "${CONFIGS[@]}"; do
  echo ""
  echo "========== ${cfg} =========="
  "${PYTHON}" pipelines/unicycle_eval.py \
    --eval-task heart \
    --ckpt-path "${CKPT_PATH}" \
    --seed "${SEED}" \
    --device cuda:0 \
    --solver ddim \
    --sample-step-schedule uniform \
    --sampling-steps 100 \
    --diffusion-steps 20 \
    --temperature 1.0 \
    --use-ema \
    --tracking-steps 256 \
    --init-from-heart-start \
    --heart-mpc-execute-steps 10 \
    --heart-config-names "${cfg}" \
    --heart-checkpoint-steps 0 64 128 192 240 \
    --output-dir "${OUTPUT_ROOT}/${cfg}"
done

echo ""
echo "Done. Results under ${OUTPUT_ROOT}/"
