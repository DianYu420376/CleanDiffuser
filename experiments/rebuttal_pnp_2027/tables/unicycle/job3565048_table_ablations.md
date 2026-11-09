| Setting | Algorithm | c_plan | c_rollout | e_dyn | NFE | Net bwd | Grad evals | Runtime/sample (ms) | Note |
|---|---|---|---|---|---|---|---|---|---|
| Circle | PnP-PGD | 0.3281 ± 0.0991 | 0.7354 ± 0.1988 | 0.7051 ± 0.1637 | 25 | 0 | 5 | 165.08 ± 1.42 | init 20 NFE / 129.80 ms; iters 5 NFE + 5 grads / 35.29 ms |
| Circle | PnP-PGD (a) budget-matched | 0.2098 ± 0.0912 | 0.4713 ± 0.2070 | 0.5708 ± 0.1784 | 20 | 0 | 10 | 134.44 ± 0.82 | init 10 NFE / 64.88 ms; iters 10 NFE + 10 grads / 69.56 ms |
| Circle | PnP-PGD (b) init x_T | 0.3823 ± 0.0919 | 0.9061 ± 0.1427 | 0.8262 ± 0.1123 | 10 | 0 | 10 | 69.65 ± 0.50 | init 0 NFE / 0.14 ms; iters 10 NFE + 10 grads / 69.51 ms |
| Circle | PnP-PGD (c) no input scaling | 0.3039 ± 0.0952 | 0.6928 ± 0.1904 | 0.6969 ± 0.1632 | 25 | 0 | 5 | 160.68 ± 1.45 | init 20 NFE / 126.05 ms; iters 5 NFE + 5 grads / 34.63 ms |
| Random Trajectory 1 | PnP-PGD | 0.0782 ± 0.0539 | 0.5184 ± 0.3157 | 0.7777 ± 0.2222 | 25 | 0 | 5 | 162.02 ± 1.26 | init 20 NFE / 127.13 ms; iters 5 NFE + 5 grads / 34.89 ms |
| Random Trajectory 1 | PnP-PGD (a) budget-matched | 0.1010 ± 0.0704 | 0.6139 ± 0.3646 | 0.8122 ± 0.2051 | 20 | 0 | 10 | 133.55 ± 0.98 | init 10 NFE / 64.29 ms; iters 10 NFE + 10 grads / 69.26 ms |
| Random Trajectory 1 | PnP-PGD (b) init x_T | 0.0687 ± 0.0447 | 0.3833 ± 0.2527 | 0.6433 ± 0.1899 | 10 | 0 | 10 | 70.00 ± 0.48 | init 0 NFE / 0.14 ms; iters 10 NFE + 10 grads / 69.86 ms |
| Random Trajectory 1 | PnP-PGD (c) no input scaling | 0.0655 ± 0.0494 | 0.4747 ± 0.3094 | 0.7714 ± 0.2321 | 25 | 0 | 5 | 162.17 ± 1.27 | init 20 NFE / 127.31 ms; iters 5 NFE + 5 grads / 34.86 ms |
| Random Trajectory 2 | PnP-PGD | 0.8673 ± 0.1049 | 3.3932 ± 0.3312 | 1.7217 ± 0.1147 | 25 | 0 | 5 | 162.06 ± 1.23 | init 20 NFE / 127.21 ms; iters 5 NFE + 5 grads / 34.85 ms |
| Random Trajectory 2 | PnP-PGD (a) budget-matched | 0.7709 ± 0.1076 | 2.6218 ± 0.4743 | 1.4149 ± 0.1773 | 20 | 0 | 10 | 134.27 ± 3.13 | init 10 NFE / 64.66 ms; iters 10 NFE + 10 grads / 69.61 ms |
| Random Trajectory 2 | PnP-PGD (b) init x_T | 0.1029 ± 0.0702 | 0.2207 ± 0.2790 | 0.4231 ± 0.2042 | 10 | 0 | 10 | 70.10 ± 0.55 | init 0 NFE / 0.14 ms; iters 10 NFE + 10 grads / 69.96 ms |
| Random Trajectory 2 | PnP-PGD (c) no input scaling | 0.8297 ± 0.1046 | 3.3522 ± 0.3452 | 1.7389 ± 0.1226 | 25 | 0 | 5 | 162.84 ± 1.27 | init 20 NFE / 127.88 ms; iters 5 NFE + 5 grads / 34.97 ms |
