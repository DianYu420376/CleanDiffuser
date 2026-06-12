#!/bin/bash
#SBATCH --job-name=diffuser_mujoco_render
#SBATCH --time=00:30:00
#SBATCH --account=bgeo-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm_diffuser_mujoco_render_%j.out
#SBATCH --error=slurm_diffuser_mujoco_render_%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/u/rzhang26/CleanDiffuser}"
VENV_DIR="${VENV_DIR:-/u/rzhang26/venvs/CleanDiffuser}"
D4RL_DIR="${D4RL_DIR:-/u/rzhang26/D4RL}"

cd "${REPO_DIR}"

export PYTHONPATH="${REPO_DIR}/pipelines:${REPO_DIR}:${D4RL_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export D4RL_SUPPRESS_IMPORT_ERROR=1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID="${CUDA_VISIBLE_DEVICES:-0}"
PYTHON="${VENV_DIR}/bin/python"

CKPT="${CKPT:-latest}"
NUM_EPISODES="${NUM_EPISODES:-1}"
NUM_CANDIDATES="${NUM_CANDIDATES:-8}"
SAMPLING_STEPS="${SAMPLING_STEPS:-5}"
# 20x default planning horizon (32 -> 640 env steps in one video)
MAX_RENDER_STEPS="${MAX_RENDER_STEPS:-640}"

echo "============================================================"
echo "diffuser_d4rl_mujoco render job"
echo "============================================================"
echo "hostname: $(hostname)"
echo "task: hopper-medium-v2"
echo "ckpt: ${CKPT}"
echo "============================================================"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
fi

"${PYTHON}" -m pip install -q gymnasium imageio imageio-ffmpeg 2>/dev/null || true

CKPT_PATH="results/diffuser_d4rl_mujoco/hopper-medium-v2/diffusion_ckpt_${CKPT}.pt"
if [[ ! -f "${CKPT_PATH}" ]]; then
  echo "Checkpoint ${CKPT_PATH} not found; running short training to create latest ckpt..."
  "${PYTHON}" pipelines/diffuser_d4rl_mujoco.py \
    task=hopper-medium-v2 \
    mode=train \
    device=cuda:0 \
    diffusion_gradient_steps=50 \
    classifier_gradient_steps=50 \
    log_interval=50 \
    save_interval=50 \
    batch_size=64
fi

"${PYTHON}" pipelines/diffuser_d4rl_mujoco.py \
  task=hopper-medium-v2 \
  mode=render \
  device=cuda:0 \
  ckpt="${CKPT}" \
  num_episodes="${NUM_EPISODES}" \
  num_candidates="${NUM_CANDIDATES}" \
  sampling_steps="${SAMPLING_STEPS}" \
  render_fps=30 \
  render_width=480 \
  render_height=480 \
  max_render_steps="${MAX_RENDER_STEPS}" \
  render_reset_on_done=false \
  render_ignore_termination=true

echo ""
echo "Render job completed. Videos under video_outputs/diffuser_d4rl_mujoco/hopper-medium-v2/"
