#!/bin/bash
#SBATCH --job-name=drgd_lownoise
#SBATCH --time=06:00:00
#SBATCH --account=bgeo-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/u/rzhang26/CleanDiffuser_rebuttal_pnp/experiments/rebuttal_pnp_2027/results/logs/unicycle_drgd_lownoise_%j.out
#SBATCH --error=/u/rzhang26/CleanDiffuser_rebuttal_pnp/experiments/rebuttal_pnp_2027/results/logs/unicycle_drgd_lownoise_%j.err

# Landing DRGD on the unicycle reference_plan task, identical to the paper sweep
# (run_reference_plan_paper_drgd_sweep.sbatch / reference_plan_paper_v1) EXCEPT
# --drgd-noise-index 0 (least noise: schedule value 0, alpha=1.0, sigma=0.0064)
# instead of -1 (noisiest: schedule value 19, alpha=0.0084, sigma=1.0).
# Reads the original checkpoint / normalizer dataset read-only; writes only to a
# fresh directory under experiments/rebuttal_pnp_2027/results/.

set -euo pipefail

WT=/u/rzhang26/CleanDiffuser_rebuttal_pnp
OR=/u/rzhang26/CleanDiffuser
PYTHON=/u/rzhang26/venvs/CleanDiffuser/bin/python
CKPT_PATH="${OR}/results/diffuser_unicycle/Unicycle-v0/full64_6k/diffusion_ckpt_latest.pt"
# Paper reference_plan runs normalize with the old dataset (unicycle_eval default under repo_dir).
DATASET_H5="${OR}/results/unicycle_offline/unicycle_offline.hdf5"
RUN_ID="${RUN_ID:-drgd_noise0_${SLURM_JOB_ID:-manual}}"
OUTPUT_ROOT="${WT}/experiments/rebuttal_pnp_2027/results/unicycle_drgd_lownoise/${RUN_ID}"
NOISE_INDEX=0
DRGD_STEPS=20
DRGD_GAMMAS=(5 10 15 20)
TRAJECTORIES=(circle random_smooth_1 random_smooth_7 random_smooth_3)

if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "Refusing to overwrite existing ${OUTPUT_ROOT}" >&2
  exit 1
fi
mkdir -p "$(dirname "${OUTPUT_ROOT}")"
mkdir "${OUTPUT_ROOT}"

cd "${WT}"
export PYTHONPATH="${WT}/pipelines:${WT}:${PYTHONPATH:-}"
export D4RL_SUPPRESS_IMPORT_ERROR=1 MUJOCO_GL=disable PYTHONUNBUFFERED=1

COMMON_ARGS=(
  --eval-task reference_plan
  --ckpt-path "${CKPT_PATH}"
  --dataset-h5path "${DATASET_H5}"
  --seed 0
  --device cuda:0
  --solver ddim
  --sample-step-schedule uniform
  --sampling-steps 20
  --diffusion-steps 20
  --training-diffusion-steps 20
  --temperature 1.0
  --use-ema
  --heart-plan-candidates 100
  --plan-plot-top-k 5
  --reference-steps 64
  --heart-config-names drgd
)

{
  echo "git_commit=$(git -C "${WT}" rev-parse HEAD)"
  echo "ckpt_sha256=$(sha256sum "${CKPT_PATH}" | cut -d' ' -f1)"
  echo "dataset_sha256=$(sha256sum "${DATASET_H5}" | cut -d' ' -f1)"
  echo "noise_index=${NOISE_INDEX} drgd_steps=${DRGD_STEPS} gammas=${DRGD_GAMMAS[*]}"
  echo "trajectories=${TRAJECTORIES[*]}"
  echo "slurm_job=${SLURM_JOB_ID:-} host=$(hostname)"
} > "${OUTPUT_ROOT}/run_info.txt"
cat "${OUTPUT_ROOT}/run_info.txt"
nvidia-smi -L || true

for traj in "${TRAJECTORIES[@]}"; do
  for gamma in "${DRGD_GAMMAS[@]}"; do
    out="${OUTPUT_ROOT}/${traj}/drgd_noise${NOISE_INDEX}_gamma${gamma}"
    echo ""
    echo "========== ${traj} | drgd | noise_index=${NOISE_INDEX} | gamma=${gamma} =========="
    "${PYTHON}" pipelines/unicycle_eval.py \
      "${COMMON_ARGS[@]}" \
      --reference-name "${traj}" \
      --drgd-steps "${DRGD_STEPS}" \
      --drgd-gamma "${gamma}" \
      --drgd-noise-index "${NOISE_INDEX}" \
      --output-dir "${out}"
  done
done

echo ""
echo "Aggregating..."
OUTPUT_ROOT="${OUTPUT_ROOT}" "${PYTHON}" - <<'PY'
import json, os
from pathlib import Path

root = Path(os.environ["OUTPUT_ROOT"])
paper = Path("/u/rzhang26/CleanDiffuser/results/unicycle_eval/reference_plan_paper_v1")

def stats(summary_path):
    block = json.load(open(summary_path)).get("reference_plan", {})
    s = block.get("candidate_metrics_summary", {})
    return {
        "c_plan": -s["plan_reward"]["mean"], "c_plan_std": s["plan_reward"]["std"],
        "c_rollout": -s["rollout_reward"]["mean"], "c_rollout_std": s["rollout_reward"]["std"],
        "e_dyn": s["feasibility_mean_l2_future"]["mean"], "e_dyn_std": s["feasibility_mean_l2_future"]["std"],
    }

rows = []
for p in sorted(root.glob("*/drgd_noise*_gamma*/summary.json")):
    traj, cfg = p.parent.parent.name, p.parent.name
    gamma = cfg.split("_gamma")[-1]
    row = {"trajectory": traj, "config": cfg, "gamma": float(gamma), "noise_index": 0, **stats(p)}
    ref = paper / traj / f"drgd_gamma{gamma}" / "summary.json"
    if ref.exists():
        row["paper_noise_index_minus1"] = stats(ref)
    rows.append(row)

out = root / "drgd_lownoise_summary.json"
with open(out, "x") as f:  # exclusive create: never overwrite
    json.dump({"runs": rows}, f, indent=2)

lines = ["trajectory        gamma | noise0: c_plan   c_rollout  e_dyn  | paper(noise -1): c_plan  c_rollout  e_dyn"]
for r in rows:
    p = r.get("paper_noise_index_minus1", {})
    lines.append(
        f"{r['trajectory']:<17} {r['gamma']:>5.0f} | {r['c_plan']:8.4f} {r['c_rollout']:9.4f} {r['e_dyn']:7.4f} | "
        f"{p.get('c_plan', float('nan')):8.4f} {p.get('c_rollout', float('nan')):9.4f} {p.get('e_dyn', float('nan')):7.4f}"
    )
with open(root / "drgd_lownoise_summary.txt", "x") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"Wrote {out}")
PY

echo "Done. Results under ${OUTPUT_ROOT}/"
