# Table 3 (supplementary metrics)

| Method | Unobs. 4-dim obs MSE | Action MSE | f (normalized) | Unobs. xy MSE median | e_dyn (normalized) | o_0 pin err [phys] | batch-1 vs batched max|dx| | batched time/traj [ms] |
|---|---|---|---|---|---|---|---|---|
| Ground truth | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 | 0.000 ± 0.000 | 7.2e-08 | nan | nan |
| Unguided | 2.001 ± 1.676 | 0.632 ± 0.523 | 4.098 ± 4.062 | 1.771 | 0.103 ± 0.050 | 7.2e-08 | 2.0e-03 | 24.84 |
| PDG | 1.498 ± 1.113 | 0.621 ± 0.529 | 0.673 ± 0.618 | 1.360 | 0.149 ± 0.070 | 7.2e-08 | 1.9e-03 | 28.69 |
| DCG | 1.240 ± 0.868 | 0.609 ± 0.533 | 2.159 ± 1.888 | 1.244 | 0.113 ± 0.055 | 7.2e-08 | 1.9e-03 | 27.85 |
| PnP-PGD | 1.086 ± 0.772 | 0.707 ± 0.507 | 1.599 ± 1.626 | 0.696 | 0.195 ± 0.040 | 7.2e-08 | 7.7e-03 | 40.75 |
| PnP-PGD (DDIM-10 init, K=10) | 1.116 ± 0.777 | 0.698 ± 0.506 | 1.647 ± 1.659 | 0.769 | 0.197 ± 0.048 | 7.2e-08 | 1.0e-02 | 27.63 |

mean ± std (ddof=1), n=5. Action MSE over t=0..63 of clipped actions. Batched time = one batch of all trajectories / n (secondary convention).
