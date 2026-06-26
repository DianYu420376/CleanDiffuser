#!/bin/bash
#SBATCH --job-name=hopper-ep150-cpu-smoke
#SBATCH --account=bgfg-delta-cpu
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_hopper_ep150_cpu_smoke_%j.out

set -euo pipefail

NUM_EPISODES="${NUM_EPISODES:-1}"
MAX_STEPS="${MAX_STEPS:-5}"
LOG_INTERVAL="${LOG_INTERVAL:-5}"
SEED="${SEED:-0}"
RUN_TAG="${RUN_TAG:-smoke_cpu_job${SLURM_JOB_ID}}"

source /u/rzhang26/venvs/cleandiffuser/activate_cleandiffuser.sh
cd /u/rzhang26/CleanDiffuser
export PYTHONPATH="/u/rzhang26/CleanDiffuser/pipelines:/u/rzhang26/CleanDiffuser:${PYTHONPATH:-}"
export D4RL_SUPPRESS_IMPORT_ERROR=1
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""

PYTHON=/u/rzhang26/venvs/cleandiffuser/bin/python
SWEEP_DIR="results/diffuser_d4rl_mujoco/hopper-medium-v2/ep150_thorough_sweep/${RUN_TAG}"
mkdir -p "${SWEEP_DIR}"

echo "CPU smoke: ${NUM_EPISODES} ep, ${MAX_STEPS} steps -> ${SWEEP_DIR}"

run_one() {
  local name="$1"
  shift
  echo ""
  echo "=== ${name} ==="
  "${PYTHON}" -u pipelines/eval_hopper_trajectory.py \
    --task hopper-medium-v2 \
    --ckpt latest \
    --seed "${SEED}" \
    --num_episodes "${NUM_EPISODES}" \
    --max_steps "${MAX_STEPS}" \
    --log_interval "${LOG_INTERVAL}" \
    --sim-env-name hopper-medium-v2 \
    --output "${SWEEP_DIR}/${name}.json" \
    "$@" \
    2>&1 | tee "${SWEEP_DIR}/${name}.log"
}

run_one opt_scale01_wcg0 \
  --solver ddim --sampling-steps 20 --temperature 1.0 \
  --optimization-guidance-last-steps 10 --noise-schedule cosine --ddim-eta 1 \
  --guidance_mode optimization --optimization_guidance_scale 0.1 --w_cg 0.0

run_one standard_wcg0p3 \
  --solver ddpm --sampling-steps 20 --temperature 0.5 \
  --guidance_mode standard --optimization_guidance_scale 0.0 --w_cg 0.3

run_one standard_wcg0p5_temp1 \
  --solver ddpm --sampling-steps 20 --temperature 1.0 \
  --guidance_mode standard --optimization_guidance_scale 0.0 --w_cg 0.5

run_one opt_scale01_optlast15 \
  --solver ddim --sampling-steps 20 --temperature 1.0 \
  --optimization-guidance-last-steps 15 --noise-schedule cosine --ddim-eta 1 \
  --guidance_mode optimization --optimization_guidance_scale 0.1 --w_cg 0.0

run_one opt_scale01_optlast20 \
  --solver ddim --sampling-steps 20 --temperature 1.0 \
  --optimization-guidance-last-steps 20 --noise-schedule cosine --ddim-eta 1 \
  --guidance_mode optimization --optimization_guidance_scale 0.1 --w_cg 0.0

SWEEP_DIR="${SWEEP_DIR}" "${PYTHON}" -u - <<'PY'
import json
from pathlib import Path
import os
sweep_dir = Path(os.environ["SWEEP_DIR"])
names = sorted(p.stem for p in sweep_dir.glob("*.json") if p.name != "sweep_summary.json")
print(f"CPU smoke OK: {len(names)} configs -> {names}")
for n in names:
    with open(sweep_dir / f"{n}.json") as f:
        d = json.load(f)
    assert "mean_reward" in d and d.get("episodes"), n
    print(f"  {n}: reward={d['mean_reward']:.2f} norm_x100={d['mean_normalized_score_x100']:.2f}")
PY

echo ">>> CPU smoke passed: ${SWEEP_DIR}"
