#!/bin/bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/u/rzhang26/CleanDiffuser}"
cd "${REPO_DIR}"

TRAIN_JOB="$(sbatch --parsable run_diffuser_mujoco_train_medium_expert_v4_reroll.sbatch)"
echo "Submitted training job: ${TRAIN_JOB}"

GUIDANCE_JOB="$(sbatch --parsable --dependency=afterok:${TRAIN_JOB} run_diffuser_mujoco_guidance_medium_expert_v4_reroll_array.sbatch)"
echo "Submitted guidance array: ${GUIDANCE_JOB}"

MC_JOB="$(sbatch --parsable --dependency=afterok:${TRAIN_JOB} run_diffuser_mujoco_mc_medium_expert_v4_reroll_array.sbatch)"
echo "Submitted Monte Carlo array: ${MC_JOB}"

cat <<EOF

Pipeline submitted (hopper_medium_expert_v4_reroll.hdf5, truncate-on-fall).
  Train:    ${TRAIN_JOB}
  Guidance: ${GUIDANCE_JOB}
  MC:       ${MC_JOB}

Checkpoints: results/diffuser_d4rl_mujoco/hopper-medium-v2/medium_expert_v4_reroll/
EOF
