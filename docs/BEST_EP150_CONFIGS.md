# Best ep150 configs (hopper / walker2d / halfcheetah medium-v2)

**Canonical source:** `configs/diffuser/mujoco/task/<task>.yaml` → `best_hybrid` block.

Metric for hybrid selection: `mean_normalized_score_x100` from 150-seed ep150 sweeps (higher is better).

## Standard baseline (repo default)

| Task | w_cg | solver | temp | norm×100 |
|------|------|--------|------|----------|
| hopper-medium-v2 | 0.3 | ddpm | 0.5 | 93.03 |
| walker2d-medium-v2 | 0.007 | ddpm | 0.5 | 76.70 |
| halfcheetah-medium-v2 | 0.0001 | ddpm | 0.5 | 44.47 |

## Best hybrid (`best_hybrid` in task yaml)

With `optimization_guidance_last_steps=20` and `sampling_steps=20`, classifier guidance (`w_cg`) is inactive → **w_cg: 0.0**.

| Task | opt_scale | norm×100 | Source sweep |
|------|-----------|----------|--------------|
| hopper-medium-v2 | 0.9 | 94.82 | `ep150_thorough_sweep/hybrid_optlast20_seeds0-149_job19748178` |
| walker2d-medium-v2 | 0.05 | 78.14 | `ep150_thorough_sweep/hybrid_wcg1.1_optlast20_seeds0-149_job19803454` |
| halfcheetah-medium-v2 | 0.00003 | 44.80 | `ep150_thorough_sweep/hybrid_wcg1.1_optlast20_seeds0-149_job19808624` |

Note: An older hopper ep150 sweep found `standard_wcg0p5_temp1` at norm×100 **95.75** (standard guidance, not hybrid).
