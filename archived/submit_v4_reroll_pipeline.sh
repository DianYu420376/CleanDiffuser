#!/bin/bash
# Submit v4-reroll training, then parallel guidance + Monte Carlo eval arrays.
set -euo pipefail

REPO_DIR="${REPO_DIR:-/u/rzhang26/CleanDiffuser}"
cd "${REPO_DIR}"

TRAIN_JOB="$(sbatch --parsable run_diffuser_mujoco_train_v4_reroll.sbatch)"
echo "Submitted training job: ${TRAIN_JOB}"

GUIDANCE_JOB="$(sbatch --parsable --dependency=afterok:"${TRAIN_JOB}" run_diffuser_mujoco_guidance_comparison_v4_reroll_array.sbatch)"
echo "Submitted guidance array (51 tasks, after train): ${GUIDANCE_JOB}"

MC_JOB="$(sbatch --parsable --dependency=afterok:"${TRAIN_JOB}" run_diffuser_mujoco_monte_carlo_comparison_v4_reroll_array.sbatch)"
echo "Submitted Monte Carlo array (51 tasks, after train): ${MC_JOB}"

cat <<EOF

Pipeline submitted (pretrain uses hopper_medium_v4_reroll.hdf5).
  Train:    ${TRAIN_JOB}
  Guidance: ${GUIDANCE_JOB}  (array 0-50, one seed per GPU)
  MC:       ${MC_JOB}        (array 0-50, one seed per GPU)

Checkpoints: results/diffuser_d4rl_mujoco/hopper-medium-v2/v4_reroll/
Guidance JSON: results/diffuser_d4rl_mujoco/hopper-medium-v2/v4_reroll/guidance_comparison/v4reroll_eval/per_run/
MC JSON:       results/diffuser_d4rl_mujoco/hopper-medium-v2/v4_reroll/guidance_comparison_monte_carlo/v4reroll_eval/per_run/

After arrays finish, aggregate with:
  PYTHONPATH=pipelines:\$PYTHONPATH python pipelines/scrape_guidance_comparison.py \\
    --json-dir results/diffuser_d4rl_mujoco/hopper-medium-v2/v4_reroll/guidance_comparison/v4reroll_eval/per_run \\
    --json-dir results/diffuser_d4rl_mujoco/hopper-medium-v2/v4_reroll/guidance_comparison_monte_carlo/v4reroll_eval/per_run
EOF
