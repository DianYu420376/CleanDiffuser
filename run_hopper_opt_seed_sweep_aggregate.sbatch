#!/bin/bash
#SBATCH --job-name=hopper-sweep-agg
#SBATCH --account=bgfg-delta-cpu
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=slurm_hopper_opt_seed_sweep_aggregate_%j.out

set -euo pipefail

: "${ARRAY_JOB_ID:?Set ARRAY_JOB_ID to the SLURM array job id}"
: "${SWEEP_LOG_PREFIX:?Set SWEEP_LOG_PREFIX (e.g. slurm_hopper_opt_scale005_seed_sweep)}"

SEED_MIN="${SEED_MIN:-1}"
SEED_MAX="${SEED_MAX:-15}"
REPO_DIR="${REPO_DIR:-/u/rzhang26/CleanDiffuser}"

source /u/rzhang26/venvs/cleandiffuser/activate_cleandiffuser.sh
cd "${REPO_DIR}"

PYTHON=/u/rzhang26/venvs/cleandiffuser/bin/python
OUTPUT_JSON="${OUTPUT_JSON:-${REPO_DIR}/${SWEEP_LOG_PREFIX}_${ARRAY_JOB_ID}_aggregate.json}"

echo "============================================================"
echo "Seed sweep aggregate (from .out logs)"
echo "============================================================"
echo "array_job_id: ${ARRAY_JOB_ID}"
echo "log_prefix:   ${SWEEP_LOG_PREFIX}"
echo "seed_range:   ${SEED_MIN}-${SEED_MAX}"
echo "output_json:  ${OUTPUT_JSON}"
echo "============================================================"

"${PYTHON}" -u pipelines/aggregate_seed_sweep_from_out.py \
  --repo-dir "${REPO_DIR}" \
  --seed-min "${SEED_MIN}" \
  --seed-max "${SEED_MAX}" \
  --job-ids "${ARRAY_JOB_ID}" \
  --only-prefix-substrings "${SWEEP_LOG_PREFIX}" \
  --no-merge-families \
  --output-json "${OUTPUT_JSON}"

echo ""
echo ">>> Aggregate finished"
