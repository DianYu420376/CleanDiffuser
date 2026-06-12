#!/bin/bash
#SBATCH --job-name=hopper-traj-sweep
#SBATCH --account=bgeo-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_hopper_traj_guidance_sweep_%j.out

set -euo pipefail

CKPT="${CKPT:-latest}"
NUM_EPISODES="${NUM_EPISODES:-1}"
MAX_STEPS="${MAX_STEPS:-1000}"
LOG_INTERVAL="${LOG_INTERVAL:-5}"
SEED="${SEED:-0}"
# Standard baseline + optimization-guidance scales (w_cg=0 required for optimization).
OPT_SCALES="${OPT_SCALES:-0.1 0.25 0.5 0.75 1.0 1.5 2.0}"

source /u/rzhang26/miniconda3/etc/profile.d/conda.sh
conda activate /u/rzhang26/venvs/CleanDiffuser

cd /u/rzhang26/CleanDiffuser
export PYTHONPATH="/u/rzhang26/CleanDiffuser/pipelines:/u/rzhang26/CleanDiffuser:/u/rzhang26/D4RL:${PYTHONPATH:-}"
export D4RL_SUPPRESS_IMPORT_ERROR=1
export PYTHONUNBUFFERED=1

RUN_TAG="$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-local}"
SWEEP_DIR="results/diffuser_d4rl_mujoco/hopper-medium-v2/trajectory_guidance_sweep/${RUN_TAG}"
mkdir -p "${SWEEP_DIR}"

echo "============================================================"
echo "hopper trajectory eval: optimization_guidance_scale sweep"
echo "============================================================"
echo "hostname: $(hostname)"
echo "ckpt: ${CKPT}"
echo "seed: ${SEED}"
echo "episodes: ${NUM_EPISODES}, max_steps: ${MAX_STEPS}, log_interval: ${LOG_INTERVAL}"
echo "opt_scales: ${OPT_SCALES}"
echo "output_dir: ${SWEEP_DIR}"
echo "============================================================"

run_eval() {
  local run_name="$1"
  shift
  local out_file="${SWEEP_DIR}/${run_name}.json"
  echo ""
  echo ">>> Running ${run_name}"
  echo "    output: ${out_file}"
  python -u pipelines/eval_hopper_trajectory.py \
    --task hopper-medium-v2 \
    --ckpt "${CKPT}" \
    --seed "${SEED}" \
    --num_episodes "${NUM_EPISODES}" \
    --max_steps "${MAX_STEPS}" \
    --log_interval "${LOG_INTERVAL}" \
    --output "${out_file}" \
    "$@" \
    2>&1 | tee "${SWEEP_DIR}/${run_name}.log"
  echo ">>> Finished ${run_name}"
}

# Baseline: standard classifier guidance (w_cg from task yaml).
run_eval "standard_w_cg0p3" \
  --guidance_mode standard \
  --optimization_guidance_scale 0.0

for scale in ${OPT_SCALES}; do
  scale_label="${scale//./p}"
  run_eval "optimization_scale_${scale_label}" \
    --guidance_mode optimization \
    --optimization_guidance_scale "${scale}" \
    --w_cg 0.0
done

SWEEP_DIR="${SWEEP_DIR}" python -u - <<'PY'
import json
import os
from pathlib import Path

sweep_dir = Path(os.environ["SWEEP_DIR"])
rows = []
for path in sorted(sweep_dir.glob("*.json")):
    if path.name == "sweep_summary.json":
        continue
    with open(path) as f:
        data = json.load(f)
    rows.append({
        "run": path.stem,
        "guidance_mode": data.get("guidance_mode"),
        "optimization_guidance_scale": data.get("optimization_guidance_scale"),
        "w_cg": data.get("w_cg"),
        "survival_steps": data.get("mean_survival_steps"),
        "mean_reward": data.get("mean_reward"),
        "classification": data["episodes"][0]["classification"] if data.get("episodes") else None,
        "fell": data["episodes"][0]["fell"] if data.get("episodes") else None,
        "result_file": str(path),
    })

rows.sort(key=lambda r: (r["survival_steps"] or 0), reverse=True)
best = rows[0] if rows else None

summary = {
    "sweep_dir": str(sweep_dir),
    "metric": "mean_survival_steps",
    "best": best,
    "ranking": rows,
}
out = sweep_dir / "sweep_summary.json"
with open(out, "w") as f:
    json.dump(summary, f, indent=2)

print("\n=== Guidance Scale Sweep Summary (by survival steps) ===")
for i, row in enumerate(rows, 1):
    print(
        f"{i:2d}. {row['run']:30s} "
        f"mode={row['guidance_mode']:12s} "
        f"scale={row['optimization_guidance_scale']!s:>6} "
        f"w_cg={row['w_cg']!s:>5} "
        f"survival={row['survival_steps']:.0f} "
        f"reward={row['mean_reward']:.1f}"
    )
if best:
    print(
        f"\nBest hop survival: {best['run']} "
        f"(scale={best['optimization_guidance_scale']}, "
        f"{best['survival_steps']:.0f} steps)"
    )
print(f"\nSaved {out}")
PY

echo ""
echo "Sweep complete. Results in ${SWEEP_DIR}/"
