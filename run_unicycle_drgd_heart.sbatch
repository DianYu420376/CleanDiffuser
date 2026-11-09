#!/bin/bash
#SBATCH --job-name=paper_drgd_smoke
#SBATCH --time=02:00:00
#SBATCH --account=bgeo-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm_unicycle_baselines_v4_%j.out
#SBATCH --error=slurm_unicycle_baselines_v4_%j.err

set -euo pipefail

# Quick smoke: one trajectory, 4 DRGD gammas, paper_v1 settings.
export TRAJECTORIES_OVERRIDE="circle"
export CANDIDATES=3
export OUTPUT_ROOT="${OUTPUT_ROOT:-results/unicycle_eval/reference_plan_paper_v1}"

bash /u/rzhang26/CleanDiffuser/run_reference_plan_paper_drgd_sweep.sbatch
