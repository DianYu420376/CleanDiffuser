#!/bin/bash
#SBATCH --job-name=diffuser-dyn-feas
#SBATCH --time=08:00:00
#SBATCH --account=bgeo-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm_dynamic_feasibility_medium_expert_v4_%j.out
#SBATCH --error=slurm_dynamic_feasibility_medium_expert_v4_%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/u/rzhang26/CleanDiffuser}"
VENV_DIR="${VENV_DIR:-/u/rzhang26/venvs/CleanDiffuser}"
D4RL_DIR="${D4RL_DIR:-/u/rzhang26/D4RL}"

export TASK="hopper-medium-v2"
export CKPT="${CKPT:-latest}"
export RUN_SUFFIX="medium_expert_v4_reroll"
export DATASET_H5PATH="${DATASET_H5PATH:-/u/rzhang26/.d4rl/datasets/hopper_medium_expert_v4_reroll.hdf5}"
export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/results/diffuser_d4rl_mujoco/${TASK}/${RUN_SUFFIX}/dynamic_feasibility}"

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/pipelines:${REPO_DIR}:${D4RL_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export D4RL_SUPPRESS_IMPORT_ERROR=1
export MUJOCO_GL=disable

MIN_TRAJECTORIES="${MIN_TRAJECTORIES:-100}"
SEED_START="${SEED_START:-0}"
SEED_END="${SEED_END:-50}"
NUM_CANDIDATES="${NUM_CANDIDATES:-64}"
CONFIGS="${CONFIGS:-standard_w_cg0p3,optimization_scale_0p1,optimization_scale_0p25}"

echo "Dynamic feasibility: task=${TASK} ckpt=${CKPT} min_trajectories=${MIN_TRAJECTORIES}"

"${VENV_DIR}/bin/python" -u pipelines/dynamic_feasibility_comparison.py \
  --repo-dir "${REPO_DIR}" \
  --task "${TASK}" \
  --ckpt "${CKPT}" \
  --run-suffix "${RUN_SUFFIX}" \
  --dataset-h5path "${DATASET_H5PATH}" \
  --seed-start "${SEED_START}" \
  --seed-end "${SEED_END}" \
  --min-trajectories "${MIN_TRAJECTORIES}" \
  --num-candidates "${NUM_CANDIDATES}" \
  --configs "${CONFIGS}" \
  --output-dir "${OUTPUT_DIR}"
