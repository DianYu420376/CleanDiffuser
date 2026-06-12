#!/bin/bash
#SBATCH --job-name=diffuser-guidance-smoke
#SBATCH --time=01:00:00
#SBATCH --account=bgeo-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm_diffuser_mujoco_guidance_comparison_smoke_%j.out
#SBATCH --error=slurm_diffuser_mujoco_guidance_comparison_smoke_%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/u/rzhang26/CleanDiffuser}"
VENV_DIR="${VENV_DIR:-/u/rzhang26/venvs/CleanDiffuser}"
D4RL_DIR="${D4RL_DIR:-/u/rzhang26/D4RL}"

TASK="${TASK:-hopper-medium-v2}"
CKPT="${CKPT:-latest}"
RUN_SUFFIX="${RUN_SUFFIX:-}"

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/pipelines:${REPO_DIR}:${D4RL_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export D4RL_SUPPRESS_IMPORT_ERROR=1

echo "============================================================"
echo "Guidance comparison SMOKE test"
echo "============================================================"
echo "hostname: $(hostname)"
echo "task: ${TASK} ckpt: ${CKPT}"
echo "seeds: 0 only | num_episodes: 1 | num_candidates: 4"
echo "configs: standard + optimization scales 0.1"
echo "============================================================"

"${VENV_DIR}/bin/python" -u pipelines/guidance_comparison_eval.py \
  --repo-dir "${REPO_DIR}" \
  --python "${VENV_DIR}/bin/python" \
  --task "${TASK}" \
  --ckpt "${CKPT}" \
  --run-suffix "${RUN_SUFFIX}" \
  --seed-start 0 \
  --seed-end 0 \
  --num-episodes 1 \
  --num-candidates 4 \
  --opt-scales "0.1"

echo ""
echo "SMOKE PASS: guidance comparison completed without errors."
