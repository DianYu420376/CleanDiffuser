#!/bin/bash
#SBATCH --job-name=hopper-traj-eval
#SBATCH --account=bgfg-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=1-15
#SBATCH --output=slurm_hopper_traj_eval_%A_%a.out

set -euo pipefail

CKPT="${CKPT:-latest}"
NUM_EPISODES="${NUM_EPISODES:-1}"
MAX_STEPS="${MAX_STEPS:-1000}"
LOG_INTERVAL="${LOG_INTERVAL:-5}"
SEED="${SEED:-${SLURM_ARRAY_TASK_ID:-0}}"
TEMPERATURE="${TEMPERATURE:-1.0}"
W_CG="${W_CG:-0.5}"

source /u/rzhang26/venvs/cleandiffuser/activate_cleandiffuser.sh

cd /u/rzhang26/CleanDiffuser
export PYTHONPATH="/u/rzhang26/CleanDiffuser/pipelines:/u/rzhang26/CleanDiffuser:${PYTHONPATH:-}"
export D4RL_SUPPRESS_IMPORT_ERROR=1
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1

OUTPUT="results/diffuser_d4rl_mujoco/hopper-medium-v2/trajectory_eval_${CKPT}_seed${SEED}.json"

/u/rzhang26/venvs/cleandiffuser/bin/python -u pipelines/eval_hopper_trajectory.py \
  --task hopper-medium-v2 \
  --ckpt "${CKPT}" \
  --num_episodes "${NUM_EPISODES}" \
  --max_steps "${MAX_STEPS}" \
  --log_interval "${LOG_INTERVAL}" \
  --seed "${SEED}" \
  --temperature "${TEMPERATURE}" \
  --output "${OUTPUT}"\
  --w_cg "${W_CG}"
