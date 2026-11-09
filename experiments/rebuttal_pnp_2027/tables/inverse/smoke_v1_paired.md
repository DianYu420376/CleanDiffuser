# Paired per-trajectory differences of unobserved-xy MSE (row minus column method), mean and 95% bootstrap CI (10000 resamples of trajectories), and #trajectories where row is better

| A | B | mean(A-B) [m^2] | 95% CI | #A<B of n |
|---|---|---|---|---|
| PnP-PGD | DCG | -0.2979 | [-1.1633, 0.6275] | 3 of 5 |
| PnP-PGD | PDG | -0.8254 | [-2.0676, 0.4167] | 3 of 5 |
| PnP-PGD | Unguided | -1.8063 | [-3.7708, 0.1865] | 3 of 5 |
| DCG | PDG | -0.5276 | [-0.9464, -0.1343] | 5 of 5 |
| DCG | Unguided | -1.5084 | [-2.8912, -0.3646] | 5 of 5 |
| PDG | Unguided | -0.9808 | [-1.9444, -0.2129] | 5 of 5 |
| PnP-PGD (DDIM-10 init, K=10) | PnP-PGD | 0.0453 | [0.0183, 0.0723] | 1 of 5 |
| PnP-PGD (DDIM-10 init, K=10) | DCG | -0.2526 | [-1.1155, 0.6851] | 3 of 5 |
