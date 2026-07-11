#!/bin/bash
#SBATCH --job-name=hopper-hybrid
#SBATCH --account=bgfg-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=1-15
#SBATCH --output=logs/slurm/hybrid_guidance_seed_sweep/%A_%a.out

set -euo pipefail

CKPT="${CKPT:-latest}"
NUM_EPISODES="${NUM_EPISODES:-1}"
MAX_STEPS="${MAX_STEPS:-1000}"
LOG_INTERVAL="${LOG_INTERVAL:-5}"
SEED="${SLURM_ARRAY_TASK_ID}"
W_CG="${W_CG:-0.3}"
OPT_SCALE="${OPT_SCALE:-0.9}"
SIM_ENV_NAME="${SIM_ENV_NAME:-hopper-medium-v2}"
SOLVER="${SOLVER:-ddim}"
SAMPLING_STEPS="${SAMPLING_STEPS:-20}"
TEMPERATURE="${TEMPERATURE:-1.0}"
OPT_GUIDANCE_LAST_STEPS="${OPT_GUIDANCE_LAST_STEPS:-20}"
NOISE_SCHEDULE="${NOISE_SCHEDULE:-cosine}"
DDIM_ETA="${DDIM_ETA:-1}"
OPT_ALPHA_SIGMA_SCALE="${OPT_ALPHA_SIGMA_SCALE:-1}"
SEED_MIN="${SEED_MIN:-1}"
SEED_MAX="${SEED_MAX:-15}"
SWEEP_LOG_DIR="${SWEEP_LOG_DIR:-logs/slurm/hybrid_guidance_seed_sweep}"
AGG_JOB_NAME="${AGG_JOB_NAME:-hopper-hybrid-agg}"

REPO_DIR="${REPO_DIR:-/u/rzhang26/CleanDiffuser}"
mkdir -p "${REPO_DIR}/logs/slurm/hybrid_guidance_seed_sweep"

source /u/rzhang26/venvs/cleandiffuser/activate_cleandiffuser.sh

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/pipelines:${REPO_DIR}:${PYTHONPATH:-}"
export D4RL_SUPPRESS_IMPORT_ERROR=1
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1

RUN_TAG="${RUN_TAG:-hybrid_wcg${W_CG}_opt${OPT_SCALE}_optlast${OPT_GUIDANCE_LAST_STEPS}_alphasig${OPT_ALPHA_SIGMA_SCALE}_array${SLURM_ARRAY_JOB_ID}}"
SWEEP_DIR="results/diffuser_d4rl_mujoco/hopper-medium-v2/hybrid_guidance_seed_sweep/${RUN_TAG}"
mkdir -p "${SWEEP_DIR}"

PYTHON=/u/rzhang26/venvs/cleandiffuser/bin/python
SCALE_LABEL="${OPT_SCALE//./p}"
WCG_LABEL="${W_CG//./p}"
RUN_NAME="seed${SEED}_hybrid_wcg${WCG_LABEL}_opt${SCALE_LABEL}"
OUT_FILE="${SWEEP_DIR}/${RUN_NAME}.json"

if [[ "${SLURM_ARRAY_TASK_ID:-}" == "${SEED_MIN}" ]] && [[ -n "${SLURM_ARRAY_JOB_ID:-}" ]]; then
  AGG_OUT="${REPO_DIR}/logs/slurm/hybrid_guidance_seed_sweep/${SLURM_ARRAY_JOB_ID}_aggregate_%j.out"
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
      --export=ALL,ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID}",LOG_DIR="${REPO_DIR}/${SWEEP_LOG_DIR}",RESULTS_DIR="${REPO_DIR}/${SWEEP_DIR}",SEED_MIN="${SEED_MIN}",SEED_MAX="${SEED_MAX}",OUTPUT_JSON="${AGG_JSON}" \
      /u/rzhang26/CleanDiffuser/run_hopper_opt_seed_sweep_aggregate.sbatch
  )"
  echo ">>> Scheduled aggregate job ${AGG_JOB} (after array ${SLURM_ARRAY_JOB_ID})"
fi

ALPHA_SIGMA_FLAG=()
if [[ "${OPT_ALPHA_SIGMA_SCALE}" == "1" ]]; then
  ALPHA_SIGMA_FLAG=(--optimization-guidance-alpha-sigma-scale)
fi

echo "============================================================"
echo "hopper hybrid guidance seed=${SEED}"
echo "============================================================"
echo "hostname: $(hostname)"
echo "array_job_id: ${SLURM_ARRAY_JOB_ID}"
echo "array_task_id: ${SLURM_ARRAY_TASK_ID}"
echo "ckpt: ${CKPT}"
echo "guidance_mode: hybrid"
echo "w_cg: ${W_CG}"
echo "opt_scale: ${OPT_SCALE}"
echo "opt_guidance_last_steps: ${OPT_GUIDANCE_LAST_STEPS}"
echo "optimization_guidance_alpha_sigma_scale: ${OPT_ALPHA_SIGMA_SCALE}"
echo "solver: ${SOLVER}, temp: ${TEMPERATURE}, eta: ${DDIM_ETA}"
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
  --guidance_mode hybrid \
  --optimization_guidance_scale "${OPT_SCALE}" \
  --w_cg "${W_CG}" \
  "${ALPHA_SIGMA_FLAG[@]}" \
  --output "${OUT_FILE}" \
  2>&1 | tee "${SWEEP_DIR}/${RUN_NAME}.log"

echo ""
echo ">>> Finished ${RUN_NAME}"
echo "Results: ${OUT_FILE}"
