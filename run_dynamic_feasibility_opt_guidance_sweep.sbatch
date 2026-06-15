#!/bin/bash
#SBATCH --job-name=diffuser-dyn-feas-opt
#SBATCH --time=04:00:00
#SBATCH --account=bgeo-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm_dynamic_feasibility_opt_guidance_sweep_%j.out
#SBATCH --error=slurm_dynamic_feasibility_opt_guidance_sweep_%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/u/rzhang26/CleanDiffuser}"
VENV_DIR="${VENV_DIR:-/u/rzhang26/venvs/CleanDiffuser}"
D4RL_DIR="${D4RL_DIR:-/u/rzhang26/D4RL}"

export TASK="hopper-medium-v2"
export CKPT="${CKPT:-latest}"
export RUN_SUFFIX="medium_expert_v4_reroll"
export DATASET_H5PATH="${DATASET_H5PATH:-/u/rzhang26/.d4rl/datasets/hopper_medium_expert_v4_reroll.hdf5}"
export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/results/diffuser_d4rl_mujoco/${TASK}/${RUN_SUFFIX}/dynamic_feasibility_fixed_opt_guidance}"

export MIN_TRAJECTORIES="${MIN_TRAJECTORIES:-100}"
export SEED="${SEED:-0}"
export NUM_CANDIDATES="${NUM_CANDIDATES:-64}"
export INIT_MODE="${INIT_MODE:-dataset_episode_starts}"
export OPT_SCALES="${OPT_SCALES:-0.05,0.1,0.25,0.5}"
export SOLVER="${SOLVER:-ddim}"
export DIFFUSION_STEPS="${DIFFUSION_STEPS:-20}"
export SAMPLING_STEPS="${SAMPLING_STEPS:-20}"
export TRAINING_DIFFUSION_STEPS="${TRAINING_DIFFUSION_STEPS:-20}"
export OPT_GUIDANCE_LAST_STEPS="${OPT_GUIDANCE_LAST_STEPS:-}"
export CONFIGS="${CONFIGS:-monte_carlo_w_cg0,standard_w_cg0p3,optimization_scale_0p05,optimization_scale_0p1,optimization_scale_0p25,optimization_scale_0p5}"
export FEASIBILITY_SCRIPT="${FEASIBILITY_SCRIPT:-pipelines/dynamic_feasibility_comparison.py}"

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/pipelines:${REPO_DIR}:${D4RL_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export D4RL_SUPPRESS_IMPORT_ERROR=1
export MUJOCO_GL=disable

EXTRA_ARGS=()
if [[ -n "${OPT_GUIDANCE_LAST_STEPS}" ]]; then
  EXTRA_ARGS+=(--optimization-guidance-last-steps "${OPT_GUIDANCE_LAST_STEPS}")
fi

CANDIDATE_ARGS=()
if [[ "${FEASIBILITY_SCRIPT}" == *dynamic_feasibility_comparison.py ]]; then
  CANDIDATE_ARGS+=(--num-candidates "${NUM_CANDIDATES}")
fi

echo "Dynamic feasibility: script=${FEASIBILITY_SCRIPT} solver=${SOLVER} steps=${SAMPLING_STEPS} configs=${CONFIGS}"

"${VENV_DIR}/bin/python" -u "${FEASIBILITY_SCRIPT}" \
  --repo-dir "${REPO_DIR}" \
  --task "${TASK}" \
  --ckpt "${CKPT}" \
  --run-suffix "${RUN_SUFFIX}" \
  --dataset-h5path "${DATASET_H5PATH}" \
  --seed "${SEED}" \
  --min-trajectories "${MIN_TRAJECTORIES}" \
  "${CANDIDATE_ARGS[@]}" \
  --init-mode "${INIT_MODE}" \
  --solver "${SOLVER}" \
  --diffusion-steps "${DIFFUSION_STEPS}" \
  --sampling-steps "${SAMPLING_STEPS}" \
  --training-diffusion-steps "${TRAINING_DIFFUSION_STEPS}" \
  --opt-scales "${OPT_SCALES}" \
  --configs "${CONFIGS}" \
  --output-dir "${OUTPUT_DIR}" \
  "${EXTRA_ARGS[@]}"
