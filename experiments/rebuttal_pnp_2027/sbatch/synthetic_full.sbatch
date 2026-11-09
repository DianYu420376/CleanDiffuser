#!/bin/bash
#SBATCH --job-name=syn_pnp_full
#SBATCH --account=bgeo-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=/u/rzhang26/CleanDiffuser_rebuttal_pnp/experiments/rebuttal_pnp_2027/results/logs/synthetic_full_%j.out
#SBATCH --error=/u/rzhang26/CleanDiffuser_rebuttal_pnp/experiments/rebuttal_pnp_2027/results/logs/synthetic_full_%j.err
set -euo pipefail
source /u/rzhang26/CleanDiffuser_rebuttal_pnp/experiments/rebuttal_pnp_2027/sbatch/env_synthetic.sh
nvidia-smi || true
$PY experiments/rebuttal_pnp_2027/scripts/synthetic_run.py --tag full "$@"
