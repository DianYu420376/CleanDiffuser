# Table 3: sparse trajectory reconstruction

| Method | Hyper-parameters (val-selected) | Unobs. xy MSE [m^2] | Obs. xy RMSE [m] | e_dyn [m] | NFE (init) | Grad evals | Net backward | Runtime/traj [ms] (init) |
|---|---|---|---|---|---|---|---|---|
| Ground truth | - | 0.000 ± 0.000 | 0.000 ± 0.000 | 1.5e-07 | - | - | - | - |
| Unguided |  | 3.240 ± 3.135 | 1.756 ± 1.008 | 0.263 ± 0.217 | 20 | 0 | 0 | 123.4 ± 0.3 |
| PDG | w_cg=1 | 2.259 ± 2.009 | 0.721 ± 0.387 | 0.453 ± 0.308 | 20 | 20 | 0 | 143.3 ± 2.0 |
| DCG | eta=1 | 1.731 ± 1.503 | 1.308 ± 0.654 | 0.302 ± 0.244 | 20 | 10 | 0 | 138.4 ± 1.5 |
| PnP-PGD | k=5, sigma_bar=0.451, eta=0.5/L, K=10 | 1.433 ± 1.299 | 1.110 ± 0.600 | 0.586 ± 0.084 | 30 (20) | 10 | 0 | 204.1 ± 1.5 (132.1) |
| PnP-PGD (DDIM-10 init, K=10) | k=5, sigma_bar=0.451, eta=0.5/L, K=10 | 1.479 ± 1.323 | 1.129 ± 0.602 | 0.589 ± 0.084 | 20 (10) | 10 | 0 | 139.9 ± 1.3 (68.1) |

n = 5 test trajectories; mean ± std (ddof=1) over trajectories. Physical units. NFE = denoiser network forward evaluations per trajectory (measured by forward hook, all equal to the analytic count: True); (init) = cost of the unguided DDIM initialization included in the total. Grad evals = evaluations of grad f; Net backward = backward passes through the denoiser. Runtime: batch size 1, 3 warm-up calls, torch.cuda.synchronize around each trajectory, model loading and CPU e_dyn rollout excluded, GPU NVIDIA GH200 120GB.
