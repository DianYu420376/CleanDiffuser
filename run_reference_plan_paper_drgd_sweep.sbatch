#!/bin/bash
#SBATCH --job-name=paper_drgd_sw
#SBATCH --time=24:00:00
#SBATCH --account=bgeo-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm_reference_plan_paper_drgd_sweep_%j.out
#SBATCH --error=slurm_reference_plan_paper_drgd_sweep_%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/u/rzhang26/CleanDiffuser}"
VENV_DIR="${VENV_DIR:-/u/rzhang26/venvs/CleanDiffuser}"
PYTHON="${VENV_DIR}/bin/python"

# Match reference_plan_paper_v1 evaluation settings.
CKPT_PATH="${CKPT_PATH:-results/diffuser_unicycle/Unicycle-v0/full64_6k/diffusion_ckpt_latest.pt}"
SEED="${SEED:-0}"
CANDIDATES="${CANDIDATES:-100}"
SAMPLING_STEPS="${SAMPLING_STEPS:-20}"
PLAN_PLOT_TOP_K="${PLAN_PLOT_TOP_K:-5}"
REFERENCE_STEPS="${REFERENCE_STEPS:-64}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/unicycle_eval/reference_plan_paper_v1}"

TRAJECTORIES=(
  circle
  random_smooth_1
  random_smooth_3
  random_smooth_7
)
if [[ -n "${TRAJECTORIES_OVERRIDE:-}" ]]; then
  read -r -a TRAJECTORIES <<< "${TRAJECTORIES_OVERRIDE}"
fi

# DRGD refinement steps match total sampling / diffusion steps (20), not opt last_steps.
DRGD_STEPS="${DRGD_STEPS:-${SAMPLING_STEPS}}"
DRGD_GAMMAS=(5 10 15 20)

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/pipelines:${REPO_DIR}:${PYTHONPATH:-}"
export D4RL_SUPPRESS_IMPORT_ERROR=1
export PYTHONUNBUFFERED=1

COMMON_ARGS=(
  --eval-task reference_plan
  --ckpt-path "${CKPT_PATH}"
  --seed "${SEED}"
  --device cuda:0
  --solver ddim
  --sample-step-schedule uniform
  --sampling-steps "${SAMPLING_STEPS}"
  --diffusion-steps 20
  --training-diffusion-steps 20
  --temperature 1.0
  --use-ema
  --heart-plan-candidates "${CANDIDATES}"
  --plan-plot-top-k "${PLAN_PLOT_TOP_K}"
  --reference-steps "${REFERENCE_STEPS}"
  --heart-config-names drgd
)

echo "DRGD sweep under ${OUTPUT_ROOT} (paper_v1 settings)"
echo "drgd_steps=${DRGD_STEPS}, gammas=${DRGD_GAMMAS[*]}"
echo "trajectories: ${TRAJECTORIES[*]}"
nvidia-smi || true

for traj in "${TRAJECTORIES[@]}"; do
  for gamma in "${DRGD_GAMMAS[@]}"; do
    echo ""
    echo "========== ${traj} | drgd | gamma=${gamma} =========="
    "${PYTHON}" pipelines/unicycle_eval.py \
      "${COMMON_ARGS[@]}" \
      --reference-name "${traj}" \
      --drgd-steps "${DRGD_STEPS}" \
      --drgd-gamma "${gamma}" \
      --drgd-noise-index -1 \
      --output-dir "${OUTPUT_ROOT}/${traj}/drgd_gamma${gamma}"
  done
done

echo ""
echo "Aggregating DRGD sweep summary..."
OUTPUT_ROOT="${OUTPUT_ROOT}" "${PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["OUTPUT_ROOT"])
rows = []
for summary_path in sorted(root.glob("*/drgd_gamma*/summary.json")):
    traj = summary_path.parent.parent.name
    cfg = summary_path.parent.name
    gamma = cfg.replace("drgd_gamma", "")
    with open(summary_path) as f:
        data = json.load(f)
    block = data.get("reference_plan", {})
    stats = block.get("candidate_metrics_summary", {})
    rows.append({
        "trajectory": traj,
        "drgd_gamma": float(gamma),
        "plan_reward_mean": stats.get("plan_reward", {}).get("mean"),
        "plan_reward_std": stats.get("plan_reward", {}).get("std"),
        "rollout_reward_mean": stats.get("rollout_reward", {}).get("mean"),
        "feasibility_l2_mean": stats.get("feasibility_mean_l2_future", {}).get("mean"),
        "summary_path": str(summary_path),
    })

out_path = root / "drgd_gamma_sweep_summary.json"
with open(out_path, "w") as f:
    json.dump({"runs": rows}, f, indent=2)
print(f"Wrote {out_path} ({len(rows)} runs)")
PY

echo "Done. Results under ${OUTPUT_ROOT}/"
