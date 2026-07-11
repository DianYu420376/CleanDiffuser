#!/bin/bash
#SBATCH --job-name=ep150-std-smoke
#SBATCH --account=bgfg-delta-cpu
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/ep150_thorough_sweep/%x_%j.out

set -euo pipefail

: "${TASK:?Set TASK (walker2d-medium-v2 or halfcheetah-medium-v2)}"

case "${TASK}" in
  walker2d-medium-v2) SIM_ENV="Walker2d-v2" ;;
  halfcheetah-medium-v2) SIM_ENV="HalfCheetah-v2" ;;
  *) echo "ERROR: unsupported TASK=${TASK}"; exit 1 ;;
esac

REPO_DIR="${REPO_DIR:-/u/rzhang26/CleanDiffuser}"
SMOKE_DIR="${REPO_DIR}/results/diffuser_d4rl_mujoco/${TASK}/ep150_thorough_sweep/_cpu_smoke_job${SLURM_JOB_ID:-local}"
mkdir -p "${SMOKE_DIR}"

source /u/rzhang26/venvs/cleandiffuser/activate_cleandiffuser.sh
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/pipelines:${REPO_DIR}:${PYTHONPATH:-}"
export D4RL_SUPPRESS_IMPORT_ERROR=1
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""

PYTHON=/u/rzhang26/venvs/cleandiffuser/bin/python

echo "CPU smoke: task=${TASK} sim=${SIM_ENV} seeds=0 max_steps=1"

"${PYTHON}" -u pipelines/run_ep150_config_seed_sweep.py \
  --task "${TASK}" \
  --ckpt latest \
  --run-name "standard_repo_smoke" \
  --log-file "${SMOKE_DIR}/smoke.log" \
  --output "${SMOKE_DIR}/smoke.json" \
  --interim-dir "${SMOKE_DIR}/interim" \
  --seed-start 0 \
  --seed-end 0 \
  --stats-interval 1 \
  --max-steps 1 \
  --sim-env-name "${SIM_ENV}" \
  --solver ddpm \
  --sampling-steps 20 \
  --temperature 0.5 \
  --guidance_mode standard \
  --optimization_guidance_scale 0.0

"${PYTHON}" -u - <<PY
import json
from pathlib import Path

out = Path("${SMOKE_DIR}") / "smoke.json"
data = json.load(open(out))
sim = data["sim_env"]
assert "-v4" not in sim, sim
assert sim == "${SIM_ENV}", sim
assert data["num_seeds"] == 1
assert data["episodes"][0]["survival_steps"] == 1
print(f"OK task=${TASK} sim_env={sim} reward={data['mean_reward']:.2f}")
PY

echo ">>> CPU smoke passed: ${SMOKE_DIR}"
