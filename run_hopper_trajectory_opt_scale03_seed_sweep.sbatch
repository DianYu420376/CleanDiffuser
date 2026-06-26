#!/bin/bash
#SBATCH --job-name=hopper-opt-s01
#SBATCH --account=bgfg-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=1-15
#SBATCH --output=slurm_hopper_opt_scale03_seed_sweep_%A_%a.out

set -euo pipefail

CKPT="${CKPT:-latest}"
NUM_EPISODES="${NUM_EPISODES:-1}"
MAX_STEPS="${MAX_STEPS:-1000}"
LOG_INTERVAL="${LOG_INTERVAL:-5}"
SEED="${SLURM_ARRAY_TASK_ID}"
OPT_SCALE="${OPT_SCALE:-0.3}"
SIM_ENV_NAME="${SIM_ENV_NAME:-hopper-medium-v2}"
SOLVER="${SOLVER:-ddim}"
SAMPLING_STEPS="${SAMPLING_STEPS:-20}"
TEMPERATURE="${TEMPERATURE:-1.0}"
OPT_GUIDANCE_LAST_STEPS="${OPT_GUIDANCE_LAST_STEPS:-10}"
NOISE_SCHEDULE="${NOISE_SCHEDULE:-cosine}"
DDIM_ETA="${DDIM_ETA:-1}"
SEED_MIN="${SEED_MIN:-1}"
SEED_MAX="${SEED_MAX:-15}"
SWEEP_LOG_PREFIX="${SWEEP_LOG_PREFIX:-slurm_hopper_opt_scale03_seed_sweep}"
AGG_JOB_NAME="${AGG_JOB_NAME:-hopper-opt-s03-agg}"

source /u/rzhang26/venvs/cleandiffuser/activate_cleandiffuser.sh

cd /u/rzhang26/CleanDiffuser
export PYTHONPATH="/u/rzhang26/CleanDiffuser/pipelines:/u/rzhang26/CleanDiffuser:${PYTHONPATH:-}"
export D4RL_SUPPRESS_IMPORT_ERROR=1
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1

RUN_TAG="${RUN_TAG:-temp${TEMPERATURE//./p}_optlast${OPT_GUIDANCE_LAST_STEPS}_ns${NOISE_SCHEDULE}_eta${DDIM_ETA//./p}_array${SLURM_ARRAY_JOB_ID}}"
SWEEP_DIR="results/diffuser_d4rl_mujoco/hopper-medium-v2/opt_guidance_scale03_seed_sweep/${RUN_TAG}"
mkdir -p "${SWEEP_DIR}"

PYTHON=/u/rzhang26/venvs/cleandiffuser/bin/python
SCALE_LABEL="${OPT_SCALE//./p}"
RUN_NAME="seed${SEED}_optimization_scale_${SCALE_LABEL}"
OUT_FILE="${SWEEP_DIR}/${RUN_NAME}.json"

# Array task 1 schedules aggregation after all seeds finish (sbatch only).
if [[ "${SLURM_ARRAY_TASK_ID:-}" == "${SEED_MIN}" ]] && [[ -n "${SLURM_ARRAY_JOB_ID:-}" ]]; then
  AGG_OUT="${SWEEP_LOG_PREFIX}_aggregate_%j.out"
  AGG_JSON="${SWEEP_DIR}/seed_sweep_aggregate.json"
  AGG_JOB="$(
    sbatch --parsable \
      --job-name="${AGG_JOB_NAME}" \
      --account=bgfg-delta-cpu \
      --partition=cpu \
      --cpus-per-task=1 \
      --mem=4G \
      --time=00:15:00 \
      --output="${AGG_OUT}" \
      --dependency=afterany:"${SLURM_ARRAY_JOB_ID}" \
      --export=ALL,ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID}",SWEEP_LOG_PREFIX="${SWEEP_LOG_PREFIX}",SEED_MIN="${SEED_MIN}",SEED_MAX="${SEED_MAX}",OUTPUT_JSON="${AGG_JSON}" \
      /u/rzhang26/CleanDiffuser/run_hopper_opt_seed_sweep_aggregate.sbatch
  )"
  echo ">>> Scheduled aggregate job ${AGG_JOB} (after array ${SLURM_ARRAY_JOB_ID})"
  echo ">>> Aggregate log: ${SWEEP_LOG_PREFIX}_aggregate_${AGG_JOB}.out"
  echo ">>> Aggregate JSON: ${AGG_JSON}"
fi

echo "============================================================"
echo "hopper optimization-guidance scale=${OPT_SCALE} seed=${SEED}"
echo "============================================================"
echo "hostname: $(hostname)"
echo "array_job_id: ${SLURM_ARRAY_JOB_ID}"
echo "array_task_id: ${SLURM_ARRAY_TASK_ID}"
echo "cuda_visible_devices: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L 2>/dev/null || true
echo "ckpt: ${CKPT}"
echo "opt_scale: ${OPT_SCALE}"
echo "sim_env: ${SIM_ENV_NAME}"
echo "solver: ${SOLVER}"
echo "sampling_steps: ${SAMPLING_STEPS}"
echo "temperature: ${TEMPERATURE}"
echo "opt_guidance_last_steps: ${OPT_GUIDANCE_LAST_STEPS}"
echo "noise_schedule: ${NOISE_SCHEDULE}"
echo "ddim_eta: ${DDIM_ETA}"
echo "episodes: ${NUM_EPISODES}, max_steps: ${MAX_STEPS}"
echo "output_dir: ${SWEEP_DIR}"
echo "output_file: ${OUT_FILE}"
echo "============================================================"

"${PYTHON}" -u pipelines/eval_hopper_trajectory.py \
  --task hopper-medium-v2 \
  --ckpt "${CKPT}" \
  --seed "${SEED}" \
  --num_episodes "${NUM_EPISODES}" \
  --max_steps "${MAX_STEPS}" \
  --log_interval "${LOG_INTERVAL}" \
  --sim-env-name "${SIM_ENV_NAME}" \
  --solver "${SOLVER}" \
  --sampling-steps "${SAMPLING_STEPS}" \
  --temperature "${TEMPERATURE}" \
  --optimization-guidance-last-steps "${OPT_GUIDANCE_LAST_STEPS}" \
  --noise-schedule "${NOISE_SCHEDULE}" \
  --ddim-eta "${DDIM_ETA}" \
  --guidance_mode optimization \
  --optimization_guidance_scale "${OPT_SCALE}" \
  --w_cg 0.0 \
  --output "${OUT_FILE}" \
  2>&1 | tee "${SWEEP_DIR}/${RUN_NAME}.log"

echo ""
echo ">>> Finished ${RUN_NAME}"
echo "Results: ${OUT_FILE}"
