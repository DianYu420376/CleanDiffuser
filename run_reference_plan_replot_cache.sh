#!/bin/bash
# Fast replot from saved candidates.npz (no GPU, ~1 min for full paper_v1 set).

set -euo pipefail

REPO_DIR="${REPO_DIR:-/u/rzhang26/CleanDiffuser}"
VENV_DIR="${VENV_DIR:-/u/rzhang26/venvs/CleanDiffuser}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/unicycle_eval/reference_plan_paper_v1}"
PLAN_PLOT_TOP_K="${PLAN_PLOT_TOP_K:-5}"
TRAJECTORIES="${TRAJECTORIES:-}"

cd "${REPO_DIR}"
PYTHON="${VENV_DIR}/bin/python"

ARGS=(--root "${OUTPUT_ROOT}" --top-k "${PLAN_PLOT_TOP_K}")
if [[ -n "${TRAJECTORIES}" ]]; then
  read -r -a TRAJ_ARR <<< "${TRAJECTORIES}"
  ARGS+=(--trajectories "${TRAJ_ARR[@]}")
fi

"${PYTHON}" pipelines/reference_plan_replot.py "${ARGS[@]}"
