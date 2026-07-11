#!/bin/bash
#SBATCH --job-name=unicycle_feas_xy6
#SBATCH --time=02:00:00
#SBATCH --account=bgeo-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm_unicycle_feas_xy6_%j.out
#SBATCH --error=slurm_unicycle_feas_xy6_%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/u/rzhang26/CleanDiffuser}"
VENV_DIR="${VENV_DIR:-/u/rzhang26/venvs/CleanDiffuser}"

CKPT="${CKPT:-latest}"
RUN_SUFFIX="${RUN_SUFFIX:-}"
SEED="${SEED:-0}"
MIN_TRAJ="${MIN_TRAJ:-30}"
XY_MIN="${XY_MIN:--6}"
XY_MAX="${XY_MAX:-6}"
OUTPUT_DIR="${OUTPUT_DIR:-results/unicycle_eval/ddim_det_feasibility_xy6}"

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/pipelines:${REPO_DIR}:${PYTHONPATH:-}"
export D4RL_SUPPRESS_IMPORT_ERROR=1
PYTHON="${VENV_DIR}/bin/python"

EVAL_ARGS=(
  --eval-task feasibility
  --ckpt "${CKPT}"
  --seed "${SEED}"
  --device cuda:0
  --solver ddim
  --temperature 1.0
  --min-trajectories "${MIN_TRAJ}"
  --num-feasibility-plots 12
  --feasibility-plan-xy-min "${XY_MIN}"
  --feasibility-plan-xy-max "${XY_MAX}"
  --output-dir "${OUTPUT_DIR}"
)
if [[ -n "${RUN_SUFFIX}" ]]; then
  EVAL_ARGS+=(--run-suffix "${RUN_SUFFIX}")
fi

echo "Feasibility eval with plan xy in [${XY_MIN}, ${XY_MAX}]"
"${PYTHON}" pipelines/unicycle_eval.py "${EVAL_ARGS[@]}"
