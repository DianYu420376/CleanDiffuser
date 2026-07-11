#!/bin/bash
#SBATCH --job-name=unicycle_eval
#SBATCH --time=04:00:00
#SBATCH --account=bgeo-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm_unicycle_eval_%j.out
#SBATCH --error=slurm_unicycle_eval_%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/u/rzhang26/CleanDiffuser}"
VENV_DIR="${VENV_DIR:-/u/rzhang26/venvs/CleanDiffuser}"

EVAL_TASK="${EVAL_TASK:-feasibility}"
CKPT="${CKPT:-latest}"
RUN_SUFFIX="${RUN_SUFFIX:-}"
SEED="${SEED:-0}"

cd "${REPO_DIR}"

export PYTHONPATH="${REPO_DIR}/pipelines:${REPO_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
PYTHON="${VENV_DIR}/bin/python"

echo "============================================================"
echo "unicycle_eval: dynamic feasibility (unguided) + optional heart tracking"
echo "============================================================"
echo "eval_task: ${EVAL_TASK}"
echo "ckpt: ${CKPT}"
echo "run_suffix: ${RUN_SUFFIX:-<default>}"
echo "============================================================"

EVAL_ARGS=(
  --eval-task "${EVAL_TASK}"
  --ckpt "${CKPT}"
  --seed "${SEED}"
  --device cuda:0
)
if [[ -n "${RUN_SUFFIX}" ]]; then
  EVAL_ARGS+=(--run-suffix "${RUN_SUFFIX}")
fi

"${PYTHON}" pipelines/unicycle_eval.py "${EVAL_ARGS[@]}"

echo ""
echo "Evaluation completed."
