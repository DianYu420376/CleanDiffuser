| Setting | PnP variant | vs | median c_rollout PnP | median c_rollout other | win rate PnP (c_rollout lower) | mean diff c_rollout (PnP - other) [95% paired bootstrap CI] | median e_dyn PnP | median e_dyn other |
|---|---|---|---|---|---|---|---|---|
| Circle | PnP-PGD (init: Monte Carlo) | DCG | 0.7345 | 0.2054 | 0% | +0.5131 [+0.4708, +0.5547] | 0.6982 | 0.4925 |
| Circle | PnP-PGD (init: Monte Carlo) | DRGD (gamma=15) | 0.7345 | 0.1723 | 0% | +0.5562 [+0.5148, +0.5967] | 0.6982 | 0.7190 |
| Circle | PnP-PGD (init: Monte Carlo) | PDG | 0.7345 | 1.1697 | 99% | -0.4341 [-0.4891, -0.3877] | 0.6982 | 1.6759 |
| Circle | PnP-PGD (init: Monte Carlo) | Monte Carlo | 0.7345 | 1.4979 | 100% | -0.7524 [-0.7961, -0.7100] | 0.6982 | 0.2129 |
| Circle | PnP-PGD (init: x_T; abl. b) | DCG | 0.9286 | 0.2054 | 0% | +0.6839 [+0.6463, +0.7191] | 0.8257 | 0.4925 |
| Circle | PnP-PGD (init: x_T; abl. b) | DRGD (gamma=15) | 0.9286 | 0.1723 | 0% | +0.7270 [+0.6948, +0.7561] | 0.8257 | 0.7190 |
| Circle | PnP-PGD (init: x_T; abl. b) | PDG | 0.9286 | 1.1697 | 82% | -0.2633 [-0.3251, -0.2059] | 0.8257 | 1.6759 |
| Circle | PnP-PGD (init: x_T; abl. b) | Monte Carlo | 0.9286 | 1.4979 | 98% | -0.5816 [-0.6347, -0.5303] | 0.8257 | 0.2129 |
| Circle | PnP-PGD (a) budget-matched | DCG | 0.3857 | 0.2054 | 11% | +0.2491 [+0.2038, +0.2941] | 0.5172 | 0.4925 |
| Circle | PnP-PGD (a) budget-matched | DRGD (gamma=15) | 0.3857 | 0.1723 | 5% | +0.2921 [+0.2507, +0.3349] | 0.5172 | 0.7190 |
| Circle | PnP-PGD (a) budget-matched | PDG | 0.3857 | 1.1697 | 100% | -0.6981 [-0.7622, -0.6405] | 0.5172 | 1.6759 |
| Circle | PnP-PGD (a) budget-matched | Monte Carlo | 0.3857 | 1.4979 | 100% | -1.0165 [-1.0778, -0.9567] | 0.5172 | 0.2129 |
| Circle | PnP-PGD (c) no input scaling | DCG | 0.6836 | 0.2054 | 0% | +0.4706 [+0.4300, +0.5109] | 0.6910 | 0.4925 |
| Circle | PnP-PGD (c) no input scaling | DRGD (gamma=15) | 0.6836 | 0.1723 | 0% | +0.5137 [+0.4743, +0.5529] | 0.6910 | 0.7190 |
| Circle | PnP-PGD (c) no input scaling | PDG | 0.6836 | 1.1697 | 100% | -0.4766 [-0.5309, -0.4312] | 0.6910 | 1.6759 |
| Circle | PnP-PGD (c) no input scaling | Monte Carlo | 0.6836 | 1.4979 | 100% | -0.7949 [-0.8367, -0.7541] | 0.6910 | 0.2129 |
| Random Trajectory 1 | PnP-PGD (init: Monte Carlo) | DCG | 0.4454 | 0.0647 | 2% | +0.4440 [+0.3776, +0.5105] | 0.8032 | 0.3048 |
| Random Trajectory 1 | PnP-PGD (init: Monte Carlo) | DRGD (gamma=15) | 0.4454 | 0.1267 | 6% | +0.3924 [+0.3301, +0.4546] | 0.8032 | 0.4215 |
| Random Trajectory 1 | PnP-PGD (init: Monte Carlo) | PDG | 0.4454 | 0.2752 | 9% | +0.1608 [+0.1284, +0.1933] | 0.8032 | 0.7744 |
| Random Trajectory 1 | PnP-PGD (init: Monte Carlo) | Monte Carlo | 0.4454 | 0.5620 | 68% | -0.0539 [-0.0793, -0.0285] | 0.8032 | 0.1921 |
| Random Trajectory 1 | PnP-PGD (init: x_T; abl. b) | DCG | 0.3170 | 0.0647 | 5% | +0.3089 [+0.2579, +0.3631] | 0.6381 | 0.3048 |
| Random Trajectory 1 | PnP-PGD (init: x_T; abl. b) | DRGD (gamma=15) | 0.3170 | 0.1267 | 13% | +0.2574 [+0.2081, +0.3099] | 0.6381 | 0.4215 |
| Random Trajectory 1 | PnP-PGD (init: x_T; abl. b) | PDG | 0.3170 | 0.2752 | 44% | +0.0257 [-0.0261, +0.0778] | 0.6381 | 0.7744 |
| Random Trajectory 1 | PnP-PGD (init: x_T; abl. b) | Monte Carlo | 0.3170 | 0.5620 | 73% | -0.1890 [-0.2559, -0.1230] | 0.6381 | 0.1921 |
| Random Trajectory 1 | PnP-PGD (a) budget-matched | DCG | 0.5150 | 0.0647 | 1% | +0.5395 [+0.4626, +0.6154] | 0.8299 | 0.3048 |
| Random Trajectory 1 | PnP-PGD (a) budget-matched | DRGD (gamma=15) | 0.5150 | 0.1267 | 3% | +0.4879 [+0.4157, +0.5603] | 0.8299 | 0.4215 |
| Random Trajectory 1 | PnP-PGD (a) budget-matched | PDG | 0.5150 | 0.2752 | 7% | +0.2563 [+0.2131, +0.3008] | 0.8299 | 0.7744 |
| Random Trajectory 1 | PnP-PGD (a) budget-matched | Monte Carlo | 0.5150 | 0.5620 | 39% | +0.0415 [+0.0027, +0.0809] | 0.8299 | 0.1921 |
| Random Trajectory 1 | PnP-PGD (c) no input scaling | DCG | 0.4047 | 0.0647 | 6% | +0.4003 [+0.3354, +0.4654] | 0.7971 | 0.3048 |
| Random Trajectory 1 | PnP-PGD (c) no input scaling | DRGD (gamma=15) | 0.4047 | 0.1267 | 7% | +0.3487 [+0.2877, +0.4103] | 0.7971 | 0.4215 |
| Random Trajectory 1 | PnP-PGD (c) no input scaling | PDG | 0.4047 | 0.2752 | 19% | +0.1171 [+0.0858, +0.1485] | 0.7971 | 0.7744 |
| Random Trajectory 1 | PnP-PGD (c) no input scaling | Monte Carlo | 0.4047 | 0.5620 | 80% | -0.0976 [-0.1223, -0.0728] | 0.7971 | 0.1921 |
| Random Trajectory 2 | PnP-PGD (init: Monte Carlo) | DCG | 3.4291 | 0.1681 | 0% | +3.2086 [+3.1473, +3.2678] | 1.7384 | 0.5606 |
| Random Trajectory 2 | PnP-PGD (init: Monte Carlo) | DRGD (gamma=15) | 3.4291 | 1.1317 | 2% | +2.1032 [+1.9805, +2.2127] | 1.7384 | 0.8161 |
| Random Trajectory 2 | PnP-PGD (init: Monte Carlo) | PDG | 3.4291 | 3.9810 | 95% | -0.4893 [-0.5776, -0.3878] | 1.7384 | 2.9391 |
| Random Trajectory 2 | PnP-PGD (init: Monte Carlo) | Monte Carlo | 3.4291 | 6.2404 | 100% | -2.9716 [-3.1947, -2.7617] | 1.7384 | 0.1921 |
| Random Trajectory 2 | PnP-PGD (init: x_T; abl. b) | DCG | 0.1688 | 0.1681 | 48% | +0.0361 [-0.0107, +0.0983] | 0.4139 | 0.5606 |
| Random Trajectory 2 | PnP-PGD (init: x_T; abl. b) | DRGD (gamma=15) | 0.1688 | 1.1317 | 97% | -1.0693 [-1.1848, -0.9562] | 0.4139 | 0.8161 |
| Random Trajectory 2 | PnP-PGD (init: x_T; abl. b) | PDG | 0.1688 | 3.9810 | 100% | -3.6618 [-3.7683, -3.5383] | 0.4139 | 2.9391 |
| Random Trajectory 2 | PnP-PGD (init: x_T; abl. b) | Monte Carlo | 0.1688 | 6.2404 | 100% | -6.1441 [-6.3631, -5.9257] | 0.4139 | 0.1921 |
| Random Trajectory 2 | PnP-PGD (a) budget-matched | DCG | 2.6186 | 0.1681 | 0% | +2.4371 [+2.3508, +2.5234] | 1.4446 | 0.5606 |
| Random Trajectory 2 | PnP-PGD (a) budget-matched | DRGD (gamma=15) | 2.6186 | 1.1317 | 5% | +1.3317 [+1.1894, +1.4600] | 1.4446 | 0.8161 |
| Random Trajectory 2 | PnP-PGD (a) budget-matched | PDG | 2.6186 | 3.9810 | 97% | -1.2607 [-1.3674, -1.1462] | 1.4446 | 2.9391 |
| Random Trajectory 2 | PnP-PGD (a) budget-matched | Monte Carlo | 2.6186 | 6.2404 | 100% | -3.7431 [-3.9820, -3.5168] | 1.4446 | 0.1921 |
| Random Trajectory 2 | PnP-PGD (c) no input scaling | DCG | 3.3882 | 0.1681 | 0% | +3.1676 [+3.1036, +3.2293] | 1.7571 | 0.5606 |
| Random Trajectory 2 | PnP-PGD (c) no input scaling | DRGD (gamma=15) | 3.3882 | 1.1317 | 2% | +2.0621 [+1.9381, +2.1728] | 1.7571 | 0.8161 |
| Random Trajectory 2 | PnP-PGD (c) no input scaling | PDG | 3.3882 | 3.9810 | 95% | -0.5303 [-0.6198, -0.4286] | 1.7571 | 2.9391 |
| Random Trajectory 2 | PnP-PGD (c) no input scaling | Monte Carlo | 3.3882 | 6.2404 | 100% | -3.0126 [-3.2361, -2.8021] | 1.7571 | 0.1921 |
