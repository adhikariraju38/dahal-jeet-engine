# Multi-seed summary

Generated from `rl_*.json`. Expected 3 seeds per method.

| Method | seeds | vs Random (mean ± SD) | vs best heuristic (mean ± SD) | complete |
|---|---:|---|---|---|
| `distill` | 3 | 0.7358 ± 0.0068 | **0.527 ± 0.0096** | yes |
| `ppo-potential` | 3 | 0.6921 ± 0.0038 | **0.5039 ± 0.0086** | yes |
| `ppo-ten_shaped` | 3 | 0.6853 ± 0.0039 | **0.5028 ± 0.004** | yes |
| `ppo-trick_shaped` | 3 | 0.6946 ± 0.0083 | **0.5006 ± 0.0064** | yes |
| `ppo-terminal` | 3 | 0.6844 ± 0.0065 | **0.4883 ± 0.01** | yes |
| `mappo-potential` | 3 | 0.7187 ± 0.0026 | **0.4589 ± 0.0186** | yes |
| `a2c` | 3 | 0.6218 ± 0.0245 | **0.385 ± 0.0291** | yes |
| `deepcfr` | 3 | 0.6314 ± 0.0085 | **0.382 ± 0.013** | yes |
| `dqn-ten_shaped` | 3 | 0.6103 ± 0.0128 | **0.3772 ± 0.0193** | yes |
| `double_dueling` | 3 | 0.6132 ± 0.005 | **0.3506 ± 0.0109** | yes |
| `dqn-terminal` | 3 | 0.599 ± 0.0211 | **0.3458 ± 0.0284** | yes |
| `dqn-potential` | 3 | 0.614 ± 0.0061 | **0.3405 ± 0.0094** | yes |
| `double` | 3 | 0.5936 ± 0.0126 | **0.337 ± 0.0137** | yes |
| `dqn-trick_shaped` | 3 | 0.5906 ± 0.0137 | **0.3336 ± 0.0057** | yes |
| `nfsp` | 3 | 0.5333 ± 0.0022 | **0.263 ± 0.0034** | yes |

## Paired reward comparison (same seed, both arms)

Pairing on seed removes the seed's own contribution. `consistent` means every seed agreed on the direction.

| Algo | A | B | pairs | mean(A−B) | SD | consistent |
|---|---|---|---:|---:|---:|---|
| dqn | `dqn-potential` | `dqn-ten_shaped` | 3 | -0.0367 | 0.0133 | yes |
| dqn | `dqn-potential` | `dqn-terminal` | 3 | -0.0053 | 0.0191 | no |
| dqn | `dqn-potential` | `dqn-trick_shaped` | 3 | +0.0069 | 0.0041 | yes |
| dqn | `dqn-ten_shaped` | `dqn-terminal` | 3 | +0.0314 | 0.0204 | yes |
| dqn | `dqn-ten_shaped` | `dqn-trick_shaped` | 3 | +0.0436 | 0.0143 | yes |
| dqn | `dqn-terminal` | `dqn-trick_shaped` | 3 | +0.0122 | 0.0232 | no |
| ppo | `ppo-potential` | `ppo-ten_shaped` | 3 | +0.0011 | 0.0110 | no |
| ppo | `ppo-potential` | `ppo-terminal` | 3 | +0.0155 | 0.0131 | no |
| ppo | `ppo-potential` | `ppo-trick_shaped` | 3 | +0.0033 | 0.0147 | no |
| ppo | `ppo-ten_shaped` | `ppo-terminal` | 3 | +0.0144 | 0.0069 | yes |
| ppo | `ppo-ten_shaped` | `ppo-trick_shaped` | 3 | +0.0022 | 0.0041 | no |
| ppo | `ppo-terminal` | `ppo-trick_shaped` | 3 | -0.0122 | 0.0096 | yes |

## Compute

| Method | mean s/run | total s |
|---|---:|---:|
| `deepcfr` | 19485 | 58456 |
| `dqn-ten_shaped` | 6502 | 19505 |
| `dqn-trick_shaped` | 6497 | 19491 |
| `dqn-potential` | 6496 | 19490 |
| `dqn-terminal` | 6491 | 19473 |
| `double_dueling` | 4359 | 13076 |
| `double` | 4291 | 12874 |
| `mappo-potential` | 3456 | 10367 |
| `a2c` | 1785 | 5356 |
| `nfsp` | 1568 | 4704 |
| `ppo-potential` | 715 | 2144 |
| `ppo-terminal` | 713 | 2140 |
| `ppo-trick_shaped` | 713 | 2139 |
| `ppo-ten_shaped` | 712 | 2135 |
| `distill` | 67 | 202 |
