# Table 3: sparse trajectory reconstruction

| Method | Hyper-parameters (val-selected) | Unobs. xy MSE [m^2] | Obs. xy RMSE [m] | e_dyn [m] | NFE (init) | Grad evals | Net backward | Runtime/traj [ms] (init) |
|---|---|---|---|---|---|---|---|---|
| Ground truth | - | 0.000 ± 0.000 | 0.000 ± 0.000 | 2.1e-07 | - | - | - | - |
| Unguided |  | 5.431 ± 4.707 | 2.084 ± 1.085 | 0.263 ± 0.111 | 20 | 0 | 0 | 128.5 ± 0.9 |
| PDG | w_cg=10 | 3.183 ± 2.859 | 0.729 ± 0.263 | 0.851 ± 0.394 | 20 | 20 | 0 | 148.9 ± 1.1 |
| DCG | eta=10 | 1.068 ± 1.314 | 0.785 ± 0.428 | 0.865 ± 0.509 | 20 | 10 | 0 | 145.5 ± 1.7 |
| PnP-PGD | k=1, sigma_bar=0.095, eta=1.5/L, K=20 | 1.550 ± 1.418 | 1.059 ± 0.545 | 0.432 ± 0.198 | 40 (20) | 20 | 0 | 288.2 ± 2.9 (139.6) |
| PnP-PGD (DDIM-10 init, K=10) | k=2, sigma_bar=0.179, eta=1.5/L, K=10 | 1.860 ± 1.606 | 1.180 ± 0.577 | 0.443 ± 0.192 | 20 (10) | 10 | 0 | 143.0 ± 2.1 (69.5) |

n = 100 test trajectories; mean ± std (ddof=1) over trajectories. Physical units. NFE = denoiser network forward evaluations per trajectory (measured by forward hook, all equal to the analytic count: True); (init) = cost of the unguided DDIM initialization included in the total. Grad evals = evaluations of grad f; Net backward = backward passes through the denoiser. Runtime: batch size 1, 3 warm-up calls, torch.cuda.synchronize around each trajectory, model loading and CPU e_dyn rollout excluded, GPU NVIDIA GH200 120GB.
