#!/bin/bash
#SBATCH --job-name=hopper-ep150
#SBATCH --account=bgfg-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/slurm/ep150_thorough_sweep/%x_%j.out
#SBATCH --error=logs/slurm/ep150_thorough_sweep/%x_%j.err

set -euo pipefail

# 150 seeds (0-149), 1 episode per seed, per config.
CKPT="${CKPT:-latest}"
SEED_START="${SEED_START:-0}"
SEED_END="${SEED_END:-149}"
STATS_INTERVAL="${STATS_INTERVAL:-10}"
MAX_STEPS="${MAX_STEPS:-1000}"
SIM_ENV_NAME="${SIM_ENV_NAME:-hopper-medium-v2}"

# Shared optimization-guidance settings (scale01 family).
OPT_SOLVER="${OPT_SOLVER:-ddim}"
OPT_SAMPLING_STEPS="${OPT_SAMPLING_STEPS:-20}"
OPT_TEMPERATURE="${OPT_TEMPERATURE:-1.0}"
OPT_NOISE_SCHEDULE="${OPT_NOISE_SCHEDULE:-cosine}"
OPT_DDIM_ETA="${OPT_DDIM_ETA:-1}"
OPT_SCALE01="${OPT_SCALE01:-0.1}"
OPT_LAST_SCALE01="${OPT_LAST_SCALE01:-8}"
OPT_LAST_SCALE01_ALT1="${OPT_LAST_SCALE01_ALT1:-10}"
OPT_LAST_SCALE01_ALT2="${OPT_LAST_SCALE01_ALT2:-12}"

# Shared standard-guidance settings.
STD_SOLVER="${STD_SOLVER:-ddpm}"
STD_SAMPLING_STEPS="${STD_SAMPLING_STEPS:-20}"
STD_TEMPERATURE="${STD_TEMPERATURE:-0.5}"
STD_TEMPERATURE_ALT="${STD_TEMPERATURE_ALT:-1.0}"

REPO_DIR="${REPO_DIR:-/u/rzhang26/CleanDiffuser}"
JOB_ID="${SLURM_JOB_ID:-local}"
NUM_SEEDS=$((SEED_END - SEED_START + 1))
RUN_TAG="${RUN_TAG:-seeds${SEED_START}-${SEED_END}_job${JOB_ID}}"

SWEEP_DIR="${REPO_DIR}/results/diffuser_d4rl_mujoco/hopper-medium-v2/ep150_thorough_sweep/${RUN_TAG}"
LOG_DIR="${SWEEP_DIR}/logs"
JSON_DIR="${SWEEP_DIR}/json"
INTERIM_DIR="${SWEEP_DIR}/interim"
SLURM_LOG_DIR="${REPO_DIR}/logs/slurm/ep150_thorough_sweep"

mkdir -p "${LOG_DIR}" "${JSON_DIR}" "${INTERIM_DIR}" "${SLURM_LOG_DIR}"

source /u/rzhang26/venvs/cleandiffuser/activate_cleandiffuser.sh

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/pipelines:${REPO_DIR}:${PYTHONPATH:-}"
export D4RL_SUPPRESS_IMPORT_ERROR=1
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1

PYTHON=/u/rzhang26/venvs/cleandiffuser/bin/python

echo "============================================================"
echo "Hopper ep150 seed sweep: ${NUM_SEEDS} seeds, 1 episode each, 4 GPUs"
echo "============================================================"
echo "hostname: $(hostname)"
echo "job_id: ${JOB_ID}"
echo "cuda_visible_devices: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L 2>/dev/null || true
echo "ckpt: ${CKPT}"
echo "seeds: ${SEED_START}..${SEED_END} (${NUM_SEEDS} total)"
echo "stats_interval: every ${STATS_INTERVAL} seeds"
echo "max_steps: ${MAX_STEPS}"
echo "sweep_dir: ${SWEEP_DIR}"
echo "method logs: ${LOG_DIR}/<config>.log"
echo "slurm logs: ${SLURM_LOG_DIR}/"
echo "configs:"
echo "  [GPU 0 wave1] opt_scale01 opt_last=${OPT_LAST_SCALE01}"
echo "  [GPU 1 wave1] standard w_cg=0.3 (solver=${STD_SOLVER}, temp=${STD_TEMPERATURE})"
echo "  [GPU 2 wave1] standard w_cg=0.5 (solver=${STD_SOLVER}, temp=${STD_TEMPERATURE_ALT})"
echo "  [GPU 3 wave1] opt_scale01 opt_last=${OPT_LAST_SCALE01_ALT1}"
echo "  [GPU 0 wave2] opt_scale01 opt_last=${OPT_LAST_SCALE01_ALT2}"
echo "============================================================"

run_config() {
  local gpu_id="$1"
  local run_name="$2"
  shift 2

  local log_file="${LOG_DIR}/${run_name}.log"
  local out_json="${JSON_DIR}/${run_name}.json"
  local gpu_log="${LOG_DIR}/${run_name}.gpu${gpu_id}.stdout"

  echo ""
  echo ">>> [GPU ${gpu_id}] Starting ${run_name}"
  echo "    method log: ${log_file}"
  echo "    gpu stdout: ${gpu_log}"
  echo "    output json: ${out_json}"

  CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON}" -u pipelines/run_ep150_config_seed_sweep.py \
    --task hopper-medium-v2 \
    --ckpt "${CKPT}" \
    --run-name "${run_name}" \
    --log-file "${log_file}" \
    --output "${out_json}" \
    --interim-dir "${INTERIM_DIR}" \
    --seed-start "${SEED_START}" \
    --seed-end "${SEED_END}" \
    --stats-interval "${STATS_INTERVAL}" \
    --max-steps "${MAX_STEPS}" \
    --sim-env-name "${SIM_ENV_NAME}" \
    "$@" \
    > "${gpu_log}" 2>&1

  echo ">>> [GPU ${gpu_id}] Finished ${run_name}"
}

run_config_bg() {
  run_config "$@" &
}

WAVE1_PIDS=()

run_config_bg 0 "opt_scale01_optlast${OPT_LAST_SCALE01}" \
  --solver "${OPT_SOLVER}" \
  --sampling-steps "${OPT_SAMPLING_STEPS}" \
  --temperature "${OPT_TEMPERATURE}" \
  --optimization-guidance-last-steps "${OPT_LAST_SCALE01}" \
  --noise-schedule "${OPT_NOISE_SCHEDULE}" \
  --ddim-eta "${OPT_DDIM_ETA}" \
  --guidance_mode optimization \
  --optimization_guidance_scale "${OPT_SCALE01}" \
  --w_cg 0.0
WAVE1_PIDS+=($!)

run_config_bg 1 "standard_wcg0p3" \
  --solver "${STD_SOLVER}" \
  --sampling-steps "${STD_SAMPLING_STEPS}" \
  --temperature "${STD_TEMPERATURE}" \
  --guidance_mode standard \
  --optimization_guidance_scale 0.0 \
  --w_cg 0.3
WAVE1_PIDS+=($!)

run_config_bg 2 "standard_wcg0p5_temp1" \
  --solver "${STD_SOLVER}" \
  --sampling-steps "${STD_SAMPLING_STEPS}" \
  --temperature "${STD_TEMPERATURE_ALT}" \
  --guidance_mode standard \
  --optimization_guidance_scale 0.0 \
  --w_cg 0.5
WAVE1_PIDS+=($!)

run_config_bg 3 "opt_scale01_optlast${OPT_LAST_SCALE01_ALT1}" \
  --solver "${OPT_SOLVER}" \
  --sampling-steps "${OPT_SAMPLING_STEPS}" \
  --temperature "${OPT_TEMPERATURE}" \
  --optimization-guidance-last-steps "${OPT_LAST_SCALE01_ALT1}" \
  --noise-schedule "${OPT_NOISE_SCHEDULE}" \
  --ddim-eta "${OPT_DDIM_ETA}" \
  --guidance_mode optimization \
  --optimization_guidance_scale "${OPT_SCALE01}" \
  --w_cg 0.0
WAVE1_PIDS+=($!)

echo ""
echo ">>> Wave 1 launched on GPUs 0-3 (pids: ${WAVE1_PIDS[*]})"
FAIL=0
for pid in "${WAVE1_PIDS[@]}"; do
  if ! wait "${pid}"; then
    echo ">>> ERROR: wave-1 pid ${pid} failed"
    FAIL=1
  fi
done

echo ""
echo ">>> Wave 2: opt_scale01_optlast${OPT_LAST_SCALE01_ALT2} on GPU 0"
if ! run_config 0 "opt_scale01_optlast${OPT_LAST_SCALE01_ALT2}" \
  --solver "${OPT_SOLVER}" \
  --sampling-steps "${OPT_SAMPLING_STEPS}" \
  --temperature "${OPT_TEMPERATURE}" \
  --optimization-guidance-last-steps "${OPT_LAST_SCALE01_ALT2}" \
  --noise-schedule "${OPT_NOISE_SCHEDULE}" \
  --ddim-eta "${OPT_DDIM_ETA}" \
  --guidance_mode optimization \
  --optimization_guidance_scale "${OPT_SCALE01}" \
  --w_cg 0.0; then
  echo ">>> ERROR: wave-2 opt_scale01_optlast${OPT_LAST_SCALE01_ALT2} failed"
  FAIL=1
fi

SWEEP_DIR="${SWEEP_DIR}" JSON_DIR="${JSON_DIR}" SEED_START="${SEED_START}" SEED_END="${SEED_END}" "${PYTHON}" -u - <<'PY'
import json
import math
import os
from pathlib import Path

sweep_dir = Path(os.environ["SWEEP_DIR"])
json_dir = Path(os.environ["JSON_DIR"])
seed_start = int(os.environ["SEED_START"])
seed_end = int(os.environ["SEED_END"])
num_seeds = seed_end - seed_start + 1

rows = []
for path in sorted(json_dir.glob("*.json")):
    with open(path) as f:
        data = json.load(f)
    std_reward = float(data.get("std_reward", float("nan")))
    std_norm = float(data.get("std_normalized_score_x100", float("nan")))
    se_reward = float(data.get("se_reward", std_reward / math.sqrt(num_seeds) if num_seeds else float("nan")))
    se_norm = float(data.get("se_normalized_score_x100", std_norm / math.sqrt(num_seeds) if num_seeds else float("nan")))
    rows.append(
        {
            "run": data.get("run_name", path.stem),
            "guidance_mode": data.get("guidance_mode"),
            "optimization_guidance_scale": data.get("optimization_guidance_scale"),
            "w_cg": data.get("w_cg"),
            "solver": data.get("solver"),
            "sampling_steps": data.get("sampling_steps"),
            "temperature": data.get("temperature"),
            "ddim_eta": data.get("ddim_eta"),
            "optimization_guidance_last_steps": data.get("optimization_guidance_last_steps"),
            "num_seeds": int(data.get("num_seeds", num_seeds)),
            "mean_reward": float(data.get("mean_reward", float("nan"))),
            "std_reward": std_reward,
            "se_reward": se_reward,
            "mean_norm_x100": float(data.get("mean_normalized_score_x100", float("nan"))),
            "std_norm_x100": std_norm,
            "se_norm_x100": se_norm,
            "mean_survival_steps": float(data.get("mean_survival_steps", float("nan"))),
            "result_file": str(path),
        }
    )

rows.sort(key=lambda r: r["mean_norm_x100"], reverse=True)
summary = {
    "sweep_dir": str(sweep_dir),
    "seed_start": seed_start,
    "seed_end": seed_end,
    "num_seeds": num_seeds,
    "episodes_per_seed": 1,
    "metric_primary": "mean_normalized_score_x100",
    "ranking": rows,
    "best": rows[0] if rows else None,
}

out = sweep_dir / "sweep_summary.json"
with open(out, "w") as f:
    json.dump(summary, f, indent=2)

print("\n=== Ep150 Seed Sweep Summary ===")
for i, row in enumerate(rows, 1):
    print(
        f"{i}. {row['run']:22s} "
        f"mode={row['guidance_mode']:<12} "
        f"scale={row['optimization_guidance_scale']!s:>4} "
        f"w_cg={row['w_cg']!s:>4} "
        f"reward={row['mean_reward']:8.2f} ± {row['se_reward']:6.2f} "
        f"(std={row['std_reward']:7.2f}) "
        f"norm_x100={row['mean_norm_x100']:6.2f} ± {row['se_norm_x100']:5.2f} "
        f"(std={row['std_norm_x100']:5.2f})"
    )
print(f"\nSaved {out}")
PY

echo ""
if [[ "${FAIL}" -ne 0 ]]; then
  echo ">>> Sweep finished with errors. See ${LOG_DIR}/ and ${SLURM_LOG_DIR}/"
  exit 1
fi
echo ">>> Sweep complete."
echo "    method logs: ${LOG_DIR}/"
echo "    json results: ${JSON_DIR}/"
echo "    interim snapshots: ${INTERIM_DIR}/"
echo "    slurm logs: ${SLURM_LOG_DIR}/"
