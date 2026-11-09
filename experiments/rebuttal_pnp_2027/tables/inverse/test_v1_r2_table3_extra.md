# Table 3 (supplementary metrics)

| Method | Unobs. 4-dim obs MSE | Action MSE | f (normalized) | Unobs. xy MSE median | e_dyn (normalized) | o_0 pin err [phys] | batch-1 vs batched max|dx| | batched time/traj [ms] |
|---|---|---|---|---|---|---|---|---|
| Ground truth | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 | 0.000 ± 0.000 | 2.4e-08 | nan | nan |
| Unguided | 3.095 ± 2.449 | 0.733 ± 0.397 | 5.800 ± 5.664 | 3.837 | 0.122 ± 0.059 | 2.4e-08 | 2.0e-02 | 1.36 |
| PDG | 1.958 ± 1.531 | 0.655 ± 0.330 | 0.633 ± 0.452 | 2.121 | 0.253 ± 0.099 | 2.4e-08 | 6.0e-02 | 1.55 |
| DCG | 0.796 ± 0.766 | 0.654 ± 0.469 | 0.841 ± 1.029 | 0.633 | 0.278 ± 0.148 | 2.4e-08 | 1.4e-02 | 1.53 |
| PnP-PGD | 1.101 ± 0.797 | 0.758 ± 0.457 | 1.491 ± 1.530 | 1.074 | 0.161 ± 0.073 | 2.4e-08 | 4.4e-02 | 3.03 |
| PnP-PGD (DDIM-10 init, K=10) | 1.273 ± 0.899 | 0.676 ± 0.421 | 1.813 ± 1.750 | 1.313 | 0.153 ± 0.060 | 2.4e-08 | 1.6e-01 | 1.50 |

mean ± std (ddof=1), n=100. Action MSE over t=0..63 of clipped actions. Batched time = one batch of all trajectories / n (secondary convention).
