#!/bin/bash
#SBATCH --job-name=diffuser-mede-v4-mc
#SBATCH --time=6:00:00
#SBATCH --account=bgeo-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-50
#SBATCH --output=slurm_diffuser_mujoco_mc_medium_expert_v4_%A_%a.out
#SBATCH --error=slurm_diffuser_mujoco_mc_medium_expert_v4_%A_%a.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/u/rzhang26/CleanDiffuser}"
VENV_DIR="${VENV_DIR:-/u/rzhang26/venvs/CleanDiffuser}"
D4RL_DIR="${D4RL_DIR:-/u/rzhang26/D4RL}"

export TASK="hopper-medium-v2"
export CKPT="${CKPT:-latest}"
export RUN_SUFFIX="medium_expert_v4_reroll"
export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/results/diffuser_d4rl_mujoco/${TASK}/${RUN_SUFFIX}/guidance_comparison_monte_carlo/v4reroll_eval}"

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/pipelines:${REPO_DIR}:${D4RL_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export D4RL_SUPPRESS_IMPORT_ERROR=1
export MUJOCO_GL=disable

SEED="${SLURM_ARRAY_TASK_ID}"
NUM_EPISODES="${NUM_EPISODES:-1}"
NUM_CANDIDATES="${NUM_CANDIDATES:-64}"

echo "MC array medium_expert_v4_reroll seed=${SEED} ckpt=${CKPT}"

"${VENV_DIR}/bin/python" -u pipelines/guidance_comparison_eval.py \
  --repo-dir "${REPO_DIR}" \
  --python "${VENV_DIR}/bin/python" \
  --task "${TASK}" \
  --ckpt "${CKPT}" \
  --run-suffix "${RUN_SUFFIX}" \
  --seed-start "${SEED}" \
  --seed-end "${SEED}" \
  --num-episodes "${NUM_EPISODES}" \
  --num-candidates "${NUM_CANDIDATES}" \
  --configs "monte_carlo_w_cg0" \
  --output-dir "${OUTPUT_DIR}"
