# CleanDiffuser (cleaned branch)

Focused fork for three experiment lines:

1. **D4RL MuJoCo v2** — `hopper-medium-v2`, `walker2d-medium-v2`, `halfcheetah-medium-v2`
2. **Unicycle** — offline planning / heart-track benchmarks
3. **Synthetic guidance** — subspace guidance sweeps and visualizations

Removed: v4 reroll, other CleanDiffuser algorithms (DQL, DP, Veteran, …), kitchen/antmaze/maze2d, redundant hopper-only sweep scripts, and `logs/`.

## Best ep150 hybrid configs

See [`docs/BEST_EP150_CONFIGS.md`](docs/BEST_EP150_CONFIGS.md) for recorded best `opt_scale` settings from 150-seed sweeps.

## MuJoCo v2 quick start

```bash
source /path/to/venvs/cleandiffuser/activate_cleandiffuser.sh
cd CleanDiffuser
export PYTHONPATH="pipelines:.:${PYTHONPATH:-}"

# Train
python pipelines/diffuser_d4rl_mujoco.py task=hopper-medium-v2 mode=train seed=0 device=cuda:0

# Ep150 standard guidance sweep
sbatch --export=ALL,TASK=walker2d-medium-v2 run_mujoco_ep150_standard_guidance.sbatch

# Ep150 hybrid opt-scale sweep (multi-GPU)
sbatch --export=ALL,TASK=halfcheetah-medium-v2,OPT_SCALES="0.00003 0.00005" run_mujoco_ep150_hybrid_sweep_3gpu.sbatch

# Dynamic feasibility (monte carlo + standard + best hybrid)
sbatch --export=ALL,TASK=hopper-medium-v2,SIM_ENV_NAME=Hopper-v2 run_dynamic_feasibility_ep150_std_vs_opt.sbatch
```

Sim envs use standalone Gym **v2** physics: `Hopper-v2`, `Walker2d-v2`, `HalfCheetah-v2`.

## Key entry points

| Purpose | Script |
|---------|--------|
| Train / infer / render | `pipelines/diffuser_d4rl_mujoco.py` |
| Trajectory eval | `pipelines/eval_hopper_trajectory.py` |
| Ep150 seed sweeps | `pipelines/run_ep150_config_seed_sweep.py` |
| Dynamic feasibility | `pipelines/dynamic_feasibility_hopper_v2_comparison.py` |
| Hybrid scale aggregation | `pipelines/aggregate_hybrid_opt_scale_sweep.py` |
| Unicycle train | `pipelines/diffuser_unicycle.py` |
| Unicycle eval | `pipelines/unicycle_eval.py` |
| Synthetic guidance | `pipelines/guidance_synthetic_subspace.py` |

## Tests

```bash
pytest tests/test_d4rl_render_utils_v2.py tests/test_optimization_guidance.py tests/test_d4rl_envs.py tests/test_diffusion_sde.py tests/test_janner_unet.py -q
```
