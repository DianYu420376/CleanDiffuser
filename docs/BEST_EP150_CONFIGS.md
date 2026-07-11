# Best ep150 configs (hopper / walker2d / halfcheetah medium-v2)

Recorded from completed 150-seed ep150 sweeps before repo cleanup.
Metric: `mean_normalized_score_x100` (higher is better).

## Standard baseline (repo default)

| Task | Config | w_cg | solver | temp | norm×100 |
|------|--------|------|--------|------|----------|
| hopper-medium-v2 | standard_repo | 0.3 | ddpm | 0.5 | 93.03 |
| walker2d-medium-v2 | standard_repo | 0.007 | ddpm | 0.5 | 76.70 |
| halfcheetah-medium-v2 | standard_repo | 0.0001 | ddpm | 0.5 | 44.47 |

Source: `results/diffuser_d4rl_mujoco/*/ep150_thorough_sweep/standard_repo_seeds0-149_job*/sweep_summary.json`

## Best hybrid opt_scale (ep150, 150 seeds)

### hopper-medium-v2

**Best hybrid:** `opt_scale=0.9`, `w_cg=0.9` (w_cg = opt_scale)

```yaml
guidance_mode: hybrid
optimization_guidance_scale: 0.9
w_cg: 0.9
solver: ddim
temperature: 1.0
sampling_steps: 20
optimization_guidance_last_steps: 20
ddim_eta: 1.0
optimization_guidance_alpha_sigma_scale: true
sim_env_name: Hopper-v2
```

- norm×100: **94.82** ± 3.53 (std)
- mean reward: 3065.8
- Source: `ep150_thorough_sweep/hybrid_optlast20_seeds0-149_job19748178`

Note: An older ep150 sweep (`seeds0-149_job19714526`) found `standard_wcg0p5_temp1` at norm×100 **95.75**, slightly above this hybrid config. That uses standard guidance (w_cg=0.5, temp=1.0), not hybrid opt.

### walker2d-medium-v2

**Best hybrid:** `opt_scale=0.05`, `w_cg=1.1` (fixed)

```yaml
guidance_mode: hybrid
optimization_guidance_scale: 0.05
w_cg: 1.1
solver: ddim
temperature: 1.0
sampling_steps: 20
optimization_guidance_last_steps: 20
ddim_eta: 1.0
optimization_guidance_alpha_sigma_scale: true
sim_env_name: Walker2d-v2
```

- norm×100: **78.14** ± 10.52 (std)
- mean reward: 3588.7
- Source: `ep150_thorough_sweep/hybrid_wcg1.1_optlast20_seeds0-149_job19803454`

### halfcheetah-medium-v2

**Best hybrid:** `opt_scale=0.00003`, `w_cg=1.1` (fixed)

```yaml
guidance_mode: hybrid
optimization_guidance_scale: 0.00003
w_cg: 1.1
solver: ddim
temperature: 1.0
sampling_steps: 20
optimization_guidance_last_steps: 20
ddim_eta: 1.0
optimization_guidance_alpha_sigma_scale: true
sim_env_name: HalfCheetah-v2
```

- norm×100: **44.80** ± 0.99 (std)
- mean reward: 5282.3
- Source: `ep150_thorough_sweep/hybrid_wcg1.1_optlast20_seeds0-149_job19808624`

Note: A 15-seed coarse sweep (`opt0p0001`) reported 44.72; the 150-seed fine sweep around 3e-5 is the authoritative best.

## Dynamic feasibility comparison configs (10 seeds, normalized L2)

Used in `run_dynamic_feasibility_ep150_std_vs_opt.sbatch`:

| Task | Opt config | opt_scale | w_cg |
|------|------------|-----------|------|
| hopper | hybrid_wcg0p09_opt0p09 | 0.09 | 0.09 |
| halfcheetah | hybrid_wcg1p1_opt0p0005 | 0.0005 | 1.1 |
| walker2d | hybrid_wcg1p1_opt0p05 | 0.05 | 1.1 |

These differ slightly from ep150 bests above for hopper (0.09 vs 0.9) and cheetah (0.0005 vs 0.00003).
