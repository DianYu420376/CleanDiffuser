#!/bin/bash
#SBATCH --job-name=unicycle_heart_wcg05
#SBATCH --time=04:00:00
#SBATCH --account=bgeo-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm_unicycle_heart_wcg05_%j.out
#SBATCH --error=slurm_unicycle_heart_wcg05_%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/u/rzhang26/CleanDiffuser}"
VENV_DIR="${VENV_DIR:-/u/rzhang26/venvs/CleanDiffuser}"

CKPT="${CKPT:-latest}"
RUN_SUFFIX="${RUN_SUFFIX:-}"
SEED="${SEED:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-results/unicycle_eval/heart_wcg0p5_256}"

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/pipelines:${REPO_DIR}:${PYTHONPATH:-}"
export D4RL_SUPPRESS_IMPORT_ERROR=1
PYTHON="${VENV_DIR}/bin/python"

EVAL_ARGS=(
  --eval-task heart
  --ckpt "${CKPT}"
  --seed "${SEED}"
  --device cuda:0
  --solver ddim
  --sample-step-schedule uniform
  --sampling-steps 100
  --diffusion-steps 20
  --temperature 1.0
  --use-ema
  --tracking-steps 256
  --init-from-heart-start
  --heart-mpc-execute-steps 10
  --heart-config-names standard_w_cg0p5
  --heart-checkpoint-steps 0 64 128 192 240
  --output-dir "${OUTPUT_DIR}"
)
if [[ -n "${RUN_SUFFIX}" ]]; then
  EVAL_ARGS+=(--run-suffix "${RUN_SUFFIX}")
fi

echo "Heart tracking eval: standard_w_cg0p5, 256 steps, DDIM 100 reverse steps"
"${PYTHON}" pipelines/unicycle_eval.py "${EVAL_ARGS[@]}"
