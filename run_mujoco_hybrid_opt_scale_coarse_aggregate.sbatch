#!/bin/bash
#SBATCH --job-name=hyb-meta-agg
#SBATCH --account=bgfg-delta-cpu
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/hybrid_opt_scale_sweep/coarse_meta_agg_%j.out

set -euo pipefail

: "${SWEEP_ROOT:?Set SWEEP_ROOT}"

REPO_DIR="${REPO_DIR:-/u/rzhang26/CleanDiffuser}"
MANIFEST="${MANIFEST:-${SWEEP_ROOT}/sweep_manifest.json}"

source /u/rzhang26/venvs/cleandiffuser/activate_cleandiffuser.sh
cd "${REPO_DIR}"

PYTHON=/u/rzhang26/venvs/cleandiffuser/bin/python

echo "Meta-aggregate coarse sweep: ${SWEEP_ROOT}"

"${PYTHON}" -u pipelines/aggregate_hybrid_opt_scale_sweep.py \
  --sweep-root "${SWEEP_ROOT}" \
  --manifest "${MANIFEST}" \
  --output-json "${SWEEP_ROOT}/coarse_sweep_summary.json"

echo ">>> Meta-aggregate finished"
