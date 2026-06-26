#!/bin/bash
#SBATCH --job-name=hopper-opt-sweep
#SBATCH --account=bgfg-delta-cpu
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm_hopper_opt_guidance_sweep_%j.out

set -euo pipefail

CKPT="${CKPT:-latest}"
NUM_EPISODES="${NUM_EPISODES:-1}"
MAX_STEPS="${MAX_STEPS:-1000}"
LOG_INTERVAL="${LOG_INTERVAL:-5}"
SEEDS="${SEEDS:-0}"
OPT_SCALES="${OPT_SCALES:-0.1 0.3 0.5 1 3}"
SIM_ENV_NAME="${SIM_ENV_NAME:-hopper-medium-v2}"
SOLVER="${SOLVER:-ddim}"
SAMPLING_STEPS="${SAMPLING_STEPS:-100}"

source /u/rzhang26/venvs/cleandiffuser/activate_cleandiffuser.sh

cd /u/rzhang26/CleanDiffuser
export PYTHONPATH="/u/rzhang26/CleanDiffuser/pipelines:/u/rzhang26/CleanDiffuser:${PYTHONPATH:-}"
export D4RL_SUPPRESS_IMPORT_ERROR=1
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1

RUN_TAG="$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-local}"
SWEEP_DIR="results/diffuser_d4rl_mujoco/hopper-medium-v2/opt_guidance_sweep/${RUN_TAG}"
mkdir -p "${SWEEP_DIR}"

PYTHON=/u/rzhang26/venvs/cleandiffuser/bin/python

echo "============================================================"
echo "hopper optimization-guidance sweep (single job)"
echo "============================================================"
echo "hostname: $(hostname)"
echo "ckpt: ${CKPT}"
echo "seeds: ${SEEDS}"
echo "opt_scales: ${OPT_SCALES}"
echo "sim_env: ${SIM_ENV_NAME}"
echo "solver: ${SOLVER}"
echo "sampling_steps: ${SAMPLING_STEPS}"
echo "episodes: ${NUM_EPISODES}, max_steps: ${MAX_STEPS}"
echo "output_dir: ${SWEEP_DIR}"
echo "============================================================"

run_eval() {
  local run_name="$1"
  local seed="$2"
  local scale="$3"
  local out_file="${SWEEP_DIR}/${run_name}.json"
  echo ""
  echo ">>> Running ${run_name} (seed=${seed}, scale=${scale})"
  echo "    output: ${out_file}"
  "${PYTHON}" -u pipelines/eval_hopper_trajectory.py \
    --task hopper-medium-v2 \
    --ckpt "${CKPT}" \
    --seed "${seed}" \
    --num_episodes "${NUM_EPISODES}" \
    --max_steps "${MAX_STEPS}" \
    --log_interval "${LOG_INTERVAL}" \
    --sim-env-name "${SIM_ENV_NAME}" \
    --solver "${SOLVER}" \
    --sampling-steps "${SAMPLING_STEPS}" \
    --guidance_mode optimization \
    --optimization_guidance_scale "${scale}" \
    --w_cg 0.0 \
    --output "${out_file}" \
    2>&1 | tee "${SWEEP_DIR}/${run_name}.log"
  echo ">>> Finished ${run_name}"
}

for seed in ${SEEDS}; do
  for scale in ${OPT_SCALES}; do
    scale_label="${scale//./p}"
    run_eval "seed${seed}_optimization_scale_${scale_label}" "${seed}" "${scale}"
  done
done

SWEEP_DIR="${SWEEP_DIR}" "${PYTHON}" -u - <<'PY'
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

sweep_dir = Path(os.environ["SWEEP_DIR"])
rows = []
by_scale = defaultdict(list)

for path in sorted(sweep_dir.glob("seed*_optimization_scale_*.json")):
    with open(path) as f:
        data = json.load(f)
    row = {
        "run": path.stem,
        "seed": int(data["seed"]),
        "guidance_mode": data.get("guidance_mode"),
        "optimization_guidance_scale": float(data.get("optimization_guidance_scale")),
        "w_cg": float(data.get("w_cg")),
        "sim_env": data.get("sim_env"),
        "solver": data.get("solver"),
        "sampling_steps": data.get("sampling_steps"),
        "mean_survival_steps": float(data.get("mean_survival_steps")),
        "mean_reward": float(data.get("mean_reward")),
        "mean_normalized_score_x100": float(data.get("mean_normalized_score_x100")),
        "classification": data["episodes"][0]["classification"] if data.get("episodes") else None,
        "fell": data["episodes"][0]["fell"] if data.get("episodes") else None,
        "result_file": str(path),
    }
    rows.append(row)
    by_scale[row["optimization_guidance_scale"]].append(row)

def mean_std(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr.mean()), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))

scale_summary = []
for scale in sorted(by_scale):
    entries = by_scale[scale]
    reward_mean, reward_std = mean_std([e["mean_reward"] for e in entries])
    norm_mean, norm_std = mean_std([e["mean_normalized_score_x100"] for e in entries])
    step_mean, step_std = mean_std([e["mean_survival_steps"] for e in entries])
    scale_summary.append(
        {
            "optimization_guidance_scale": scale,
            "n_seeds": len(entries),
            "seeds": [e["seed"] for e in entries],
            "mean_raw_reward": reward_mean,
            "std_raw_reward": reward_std,
            "mean_normalized_score_x100": norm_mean,
            "std_normalized_score_x100": norm_std,
            "mean_survival_steps": step_mean,
            "std_survival_steps": step_std,
            "per_seed": entries,
        }
    )

scale_summary.sort(key=lambda s: s["mean_normalized_score_x100"], reverse=True)
best = scale_summary[0] if scale_summary else None

summary = {
    "sweep_dir": str(sweep_dir),
    "guidance_mode": "optimization",
    "solver": rows[0].get("solver") if rows else None,
    "sampling_steps": rows[0].get("sampling_steps") if rows else None,
    "metric_primary": "mean_normalized_score_x100",
    "best_scale": best,
    "scale_summary": scale_summary,
    "all_runs": rows,
}

out = sweep_dir / "sweep_summary.json"
with open(out, "w") as f:
    json.dump(summary, f, indent=2)

print("\n=== Optimization Guidance Scale Sweep ===")
for item in scale_summary:
    print(
        f"scale={item['optimization_guidance_scale']!s:>4} "
        f"reward={item['mean_raw_reward']:8.1f} ± {item['std_raw_reward']:6.1f} "
        f"norm_x100={item['mean_normalized_score_x100']:6.2f} ± "
        f"{item['std_normalized_score_x100']:5.2f} "
        f"steps={item['mean_survival_steps']:6.0f} ± {item['std_survival_steps']:5.1f}"
    )
if best:
    print(
        f"\nBest scale by normalized score: {best['optimization_guidance_scale']} "
        f"(norm_x100={best['mean_normalized_score_x100']:.2f} ± "
        f"{best['std_normalized_score_x100']:.2f})"
    )
print(f"\nSaved {out}")
PY

echo ""
echo "Sweep complete. Results in ${SWEEP_DIR}/"
