#!/bin/bash
#SBATCH --job-name=hopper-traj-eval
#SBATCH --account=bgeo-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_hopper_traj_eval_%j.out

set -euo pipefail

CKPT="${CKPT:-latest}"
NUM_EPISODES="${NUM_EPISODES:-1}"
MAX_STEPS="${MAX_STEPS:-1000}"
LOG_INTERVAL="${LOG_INTERVAL:-5}"
SEED="${SEED:-0}"


source /u/rzhang26/miniconda3/etc/profile.d/conda.sh
conda activate /u/rzhang26/venvs/CleanDiffuser

cd /u/rzhang26/CleanDiffuser
export PYTHONPATH="/u/rzhang26/CleanDiffuser/pipelines:/u/rzhang26/CleanDiffuser:/u/rzhang26/D4RL:${PYTHONPATH:-}"
export D4RL_SUPPRESS_IMPORT_ERROR=1
export PYTHONUNBUFFERED=1

python -u pipelines/eval_hopper_trajectory.py \
  --task hopper-medium-v2 \
  --ckpt "${CKPT}" \
  --num_episodes "${NUM_EPISODES}" \
  --max_steps "${MAX_STEPS}" \
  --log_interval "${LOG_INTERVAL}"
  --seed "${SEED}"
