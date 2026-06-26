#!/bin/bash
#SBATCH --job-name=hopper-dyn-opt0
#SBATCH --account=bgfg-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=slurm_dynamic_feasibility_ep150_opt0_comparison_%j.out

set -euo pipefail

source /u/rzhang26/venvs/cleandiffuser/activate_cleandiffuser.sh

cd /u/rzhang26/CleanDiffuser
export PYTHONPATH="/u/rzhang26/CleanDiffuser/pipelines:/u/rzhang26/CleanDiffuser:${PYTHONPATH:-}"
export D4RL_SUPPRESS_IMPORT_ERROR=1
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1

TASK="${TASK:-hopper-medium-v2}"
CKPT="${CKPT:-latest}"
SIM_ENV_NAME="${SIM_ENV_NAME:-hopper-medium-v2}"
SEED_START="${SEED_START:-0}"
NUM_SEEDS="${NUM_SEEDS:-10}"
NUM_CANDIDATES="${NUM_CANDIDATES:-64}"
SAMPLING_STEPS="${SAMPLING_STEPS:-20}"
CONFIG_PRESET="${CONFIG_PRESET:-ep150_plus_opt0}"
RUN_TAG="${RUN_TAG:-${CONFIG_PRESET}_$(date +%Y%m%d_%H%M%S)_job${SLURM_JOB_ID:-local}}"
OUTPUT_DIR="results/diffuser_d4rl_mujoco/${TASK}/dynamic_feasibility_hopper_v2/${RUN_TAG}"

PYTHON=/u/rzhang26/venvs/cleandiffuser/bin/python

echo "============================================================"
echo "Dynamic feasibility: ep150 + opt scale=0"
echo "============================================================"
echo "hostname: $(hostname)"
echo "cuda_visible_devices: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L 2>/dev/null || true
echo "task: ${TASK}  ckpt: ${CKPT}  sim_env: ${SIM_ENV_NAME}"
echo "config_preset: ${CONFIG_PRESET}"
echo "configs: ep150 set + opt_scale0_optlast10"
echo "  (optimization mode, scale=0.0, ddim, temp=1.0, opt_last=10, eta=1, w_cg=0)"
echo "num_seeds: ${NUM_SEEDS} (seed_start=${SEED_START})"
echo "num_candidates: ${NUM_CANDIDATES}  sampling_steps: ${SAMPLING_STEPS}"
echo "output_dir: ${OUTPUT_DIR}"
echo "============================================================"

"${PYTHON}" -u pipelines/dynamic_feasibility_hopper_v2_comparison.py \
  --repo-dir /u/rzhang26/CleanDiffuser \
  --task "${TASK}" \
  --ckpt "${CKPT}" \
  --sim-env-name "${SIM_ENV_NAME}" \
  --seed-start "${SEED_START}" \
  --num-seeds "${NUM_SEEDS}" \
  --num-candidates "${NUM_CANDIDATES}" \
  --sampling-steps "${SAMPLING_STEPS}" \
  --config-preset "${CONFIG_PRESET}" \
  --output-dir "${OUTPUT_DIR}" \
  --output-json "${OUTPUT_DIR}/summary.json"

echo ""
echo "Done. Results in ${OUTPUT_DIR}/"
