#!/bin/bash
#SBATCH --job-name=diffuser-v4reroll-train
#SBATCH --time=2-00:00:00
#SBATCH --account=bgeo-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=slurm_diffuser_mujoco_train_v4_reroll_%j.out
#SBATCH --error=slurm_diffuser_mujoco_train_v4_reroll_%j.err

set -euo pipefail

export TASK="hopper-medium-v2"
export SEED="0"
export RUN_SUFFIX="v4_reroll"
export RUN_NAME="hopper_v4_reroll_seed0"
export DATASET_H5PATH="/u/rzhang26/.d4rl/datasets/hopper_medium_v4_reroll.hdf5"
export WANDB_MODE="${WANDB_MODE:-offline}"

if [[ ! -f "${DATASET_H5PATH}" ]]; then
  echo "ERROR: v4 reroll dataset not found at ${DATASET_H5PATH}"
  exit 1
fi

echo "Training on v4-reroll HDF5: ${DATASET_H5PATH}"
/u/rzhang26/venvs/CleanDiffuser/bin/python - <<'PY'
import h5py
path = "/u/rzhang26/.d4rl/datasets/hopper_medium_v4_reroll.hdf5"
with h5py.File(path, "r") as f:
    attrs = dict(f.attrs)
    n = f["observations"].shape[0]
print(f"[preflight] transitions={n} attrs={attrs}")
if attrs.get("reroll_sim_env") != "Hopper-v4":
    raise SystemExit("ERROR: dataset is not a Hopper-v4 reroll file")
PY

bash /u/rzhang26/CleanDiffuser/run_diffuser_mujoco_train.sbatch
