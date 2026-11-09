# Paired per-trajectory differences of unobserved-xy MSE (row minus column method), mean and 95% bootstrap CI (10000 resamples of trajectories), and #trajectories where row is better

| A | B | mean(A-B) [m^2] | 95% CI | #A<B of n |
|---|---|---|---|---|
| PnP-PGD | DCG | 0.4817 | [0.1282, 0.8460] | 35 of 100 |
| PnP-PGD | PDG | -1.6329 | [-2.0702, -1.2335] | 86 of 100 |
| PnP-PGD | Unguided | -3.8814 | [-4.6721, -3.1426] | 97 of 100 |
| DCG | PDG | -2.1147 | [-2.7059, -1.5668] | 83 of 100 |
| DCG | Unguided | -4.3631 | [-5.3139, -3.4595] | 90 of 100 |
| PDG | Unguided | -2.2485 | [-2.7255, -1.8015] | 95 of 100 |
| PnP-PGD (DDIM-10 init, K=10) | PnP-PGD | 0.3105 | [0.1706, 0.4408] | 20 of 100 |
| PnP-PGD (DDIM-10 init, K=10) | DCG | 0.7923 | [0.4195, 1.1757] | 33 of 100 |
