#!/bin/bash
#SBATCH --job-name=hopper-s01-agg
#SBATCH --account=bgfg-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=slurm_hopper_opt_scale01_seed_sweep_aggregate_%j.out

set -euo pipefail

if [[ -z "${SWEEP_DIR:-}" ]]; then
  echo "ERROR: set SWEEP_DIR to the array run output directory before submitting."
  exit 1
fi

source /u/rzhang26/venvs/cleandiffuser/activate_cleandiffuser.sh
cd /u/rzhang26/CleanDiffuser

PYTHON=/u/rzhang26/venvs/cleandiffuser/bin/python
SWEEP_DIR="${SWEEP_DIR}" OPT_SCALE="${OPT_SCALE:-0.1}" "${PYTHON}" -u - <<'PY'
import json
import os
from pathlib import Path

import numpy as np

sweep_dir = Path(os.environ["SWEEP_DIR"])
opt_scale = float(os.environ.get("OPT_SCALE", "0.1"))

rows = []
for path in sorted(sweep_dir.glob("seed*_optimization_scale_*.json")):
    with open(path) as f:
        data = json.load(f)
    rows.append(
        {
            "run": path.stem,
            "seed": int(data["seed"]),
            "temperature": float(data.get("temperature", float("nan"))),
            "device": data.get("device"),
            "mean_reward": float(data["mean_reward"]),
            "mean_normalized_score_x100": float(data["mean_normalized_score_x100"]),
            "mean_survival_steps": float(data.get("mean_survival_steps", 0.0)),
            "result_file": str(path),
        }
    )

rows.sort(key=lambda r: r["seed"])

def mean_std(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr.mean()), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))

raw_mean, raw_std = mean_std([r["mean_reward"] for r in rows])
norm_mean, norm_std = mean_std([r["mean_normalized_score_x100"] for r in rows])

summary = {
    "sweep_dir": str(sweep_dir),
    "guidance_mode": "optimization",
    "optimization_guidance_scale": opt_scale,
    "n_completed": len(rows),
    "seeds_completed": [r["seed"] for r in rows],
    "mean_raw_reward": raw_mean,
    "std_raw_reward": raw_std,
    "mean_normalized_score_x100": norm_mean,
    "std_normalized_score_x100": norm_std,
    "per_seed": rows,
}

out = sweep_dir / "cumulative_summary.json"
with open(out, "w") as f:
    json.dump(summary, f, indent=2)

print(f"Completed seeds: {[r['seed'] for r in rows]} (n={len(rows)})")
print(f"raw_reward:      {raw_mean:.2f} ± {raw_std:.2f}")
print(f"normalized_x100: {norm_mean:.2f} ± {norm_std:.2f}")
print(f"Saved {out}")
PY
