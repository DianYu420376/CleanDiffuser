#!/bin/bash
#SBATCH --job-name=hopper-gpu-smoke
#SBATCH --account=bgfg-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_hopper_opt_scale01_gpu_smoke_%j.out

set -euo pipefail

source /u/rzhang26/venvs/cleandiffuser/activate_cleandiffuser.sh

cd /u/rzhang26/CleanDiffuser
export PYTHONPATH="/u/rzhang26/CleanDiffuser/pipelines:/u/rzhang26/CleanDiffuser:${PYTHONPATH:-}"
export D4RL_SUPPRESS_IMPORT_ERROR=1
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1

PYTHON=/u/rzhang26/venvs/cleandiffuser/bin/python

echo "============================================================"
echo "GPU smoke test: hopper opt-guidance eval (temperature=1)"
echo "============================================================"
echo "hostname: $(hostname)"
echo "cuda_visible_devices: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L 2>/dev/null || true
"${PYTHON}" - <<'PY'
import torch
print(f"torch.cuda.is_available() = {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"torch.cuda.device_count() = {torch.cuda.device_count()}")
    print(f"torch.cuda.get_device_name(0) = {torch.cuda.get_device_name(0)}")
PY

SMOKE_OUT="results/diffuser_d4rl_mujoco/hopper-medium-v2/opt_guidance_scale01_seed_sweep/gpu_smoke_seed1.json"
mkdir -p "$(dirname "${SMOKE_OUT}")"

echo ""
echo ">>> Running short rollout eval on GPU (seed=1, max_steps=10)"
"${PYTHON}" -u pipelines/eval_hopper_trajectory.py \
  --task hopper-medium-v2 \
  --ckpt latest \
  --seed 1 \
  --num_episodes 1 \
  --max_steps 10 \
  --log_interval 5 \
  --sim-env-name hopper-medium-v2 \
  --solver ddim \
  --sampling-steps 100 \
  --temperature 1.0 \
  --guidance_mode optimization \
  --optimization_guidance_scale 0.1 \
  --w_cg 0.0 \
  --output "${SMOKE_OUT}"

echo ""
echo "GPU smoke test complete. Output: ${SMOKE_OUT}"
