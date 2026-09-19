# Trained Models — Dahal Jeet

94 checkpoints, 223.1 MB total.

Released alongside the paper so every reported number can be reproduced without retraining.

## Verifying you have the right files

```bash
cd 05-engine && python archive_models.py --verify
```

Every checkpoint carries a SHA-256 in `manifest.json`. If a checksum does not match, the file is not the one the paper evaluated.

## Loading

```python
import torch, sys; sys.path.insert(0, '05-engine')
from train_rl import Net, TorchAgent

net = Net()
net.load_state_dict(torch.load('checkpoints/rl_distill_s0.pt'))
net.eval()
agent = TorchAgent(net, 'Distilled ISMCTS')
```

Architectures by file:

| Pattern | Class | Module |
|---|---|---|
| `rl_dqn_*`, `rl_distill_*`, `br_*` | `Net` | `train_rl.py` |
| `rl_ppo_*` | `Net(critic=True)` | `train_rl.py` |
| `rl_double*`, `rl_dueling*` | `DuelNet` | `train_more.py` |
| `rl_mappo_*` | `Actor` | `train_mappo.py` |
| `rl_nfsp_*` | `mlp(52)` | `nfsp.py` |
| `rl_deepcfr_*` | `mlp()` | `deepcfr.py` |
| `alphadj_*` | `PolicyValueNet` | `alphazero.py` |

`final_tournament.py` also has a loader that tries each architecture until one fits.

## Distillation

### `rl_distill_s0.pt` — Distilled ISMCTS
Supervised imitation of ISMCTS. Search-quality play at roughly 0.1 ms per decision.

**Config:** `method=distill`

**Measured:** Random 0.7262, TensThenTricks 0.5142

`2.4 MB` · `sha256:368977f37923327e…`

### `rl_distill_s1.pt` — Distilled ISMCTS
Supervised imitation of ISMCTS. Search-quality play at roughly 0.1 ms per decision.

**Config:** `method=distill`

**Measured:** Random 0.74, TensThenTricks 0.5375

`2.4 MB` · `sha256:e42e4164dd850625…`

### `rl_distill_s2.pt` — Distilled ISMCTS
Supervised imitation of ISMCTS. Search-quality play at roughly 0.1 ms per decision.

**Config:** `method=distill`

**Measured:** Random 0.7412, TensThenTricks 0.5292

`2.4 MB` · `sha256:b3ded64ab7b7e6f3…`

## Equilibrium

### `rl_deepcfr_s0.pt` — Deep CFR
Average-strategy network. No equilibrium guarantee in a 4-player partnership game -- a baseline, not a solution.

**Config:** `method=deepcfr`, `iters=40`, `traversals=120`, `seed=0`

**Measured:** Random 0.6433, TensThenTricks 0.3867

`2.4 MB` · `sha256:97cf2cd1d7bdfaed…`

### `rl_deepcfr_s1.pt` — Deep CFR
Average-strategy network. No equilibrium guarantee in a 4-player partnership game -- a baseline, not a solution.

**Config:** `method=deepcfr`, `iters=40`, `traversals=120`, `seed=1`

**Measured:** Random 0.6267, TensThenTricks 0.395

`2.4 MB` · `sha256:f5b099d38e585213…`

### `rl_deepcfr_s2.pt` — Deep CFR
Average-strategy network. No equilibrium guarantee in a 4-player partnership game -- a baseline, not a solution.

**Config:** `method=deepcfr`, `iters=40`, `traversals=120`, `seed=2`

**Measured:** Random 0.6242, TensThenTricks 0.3642

`2.4 MB` · `sha256:ae50ca1f61a471ca…`

### `rl_nfsp_s0.pt` — NFSP
Neural Fictitious Self-Play. The released network is the AVERAGE policy, which is the agent.

**Config:** `method=nfsp`, `eta=0.1`, `episodes=80000`, `seed=0`

**Measured:** Random 0.5358, TensThenTricks 0.265

`2.4 MB` · `sha256:3264605716568ae3…`

### `rl_nfsp_s1.pt` — NFSP
Neural Fictitious Self-Play. The released network is the AVERAGE policy, which is the agent.

**Config:** `method=nfsp`, `eta=0.1`, `episodes=80000`, `seed=1`

**Measured:** Random 0.5337, TensThenTricks 0.2583

`2.4 MB` · `sha256:20732440121b7233…`

### `rl_nfsp_s2.pt` — NFSP
Neural Fictitious Self-Play. The released network is the AVERAGE policy, which is the agent.

**Config:** `method=nfsp`, `eta=0.1`, `episodes=80000`, `seed=2`

**Measured:** Random 0.5304, TensThenTricks 0.2658

`2.4 MB` · `sha256:2d2e90821dbf8e35…`

## Exploiter

### `br_Adaptive_s0.pt` — best-response vs Adaptive
DQN trained solely to exploit Adaptive. Used for the exploitability analysis, not as a general agent.

**Config:** `episodes=50000`, `seed=0`

**Measured:** best_response_win 0.386, generic_baseline 0.58, exploitability -0.194, generic_agent TensThenTricks

`2.4 MB` · `sha256:b1b5db987f12c064…`

### `br_GreedyTricks_s0.pt` — best-response vs GreedyTricks
DQN trained solely to exploit GreedyTricks. Used for the exploitability analysis, not as a general agent.

**Config:** `episodes=50000`, `seed=0`

**Measured:** best_response_win 0.361, generic_baseline 0.55, exploitability -0.189, generic_agent TensThenTricks

`2.4 MB` · `sha256:e97ecee023109be8…`

### `br_TensThenTricks_s0.pt` — best-response vs TensThenTricks
DQN trained solely to exploit TensThenTricks. Used for the exploitability analysis, not as a general agent.

**Config:** `episodes=50000`, `seed=0`

**Measured:** best_response_win 0.3395, generic_baseline 0.45, exploitability -0.1105, generic_agent GreedyTricks

`2.4 MB` · `sha256:1cf6f7672e0f5889…`

### `br_Tuned_s0.pt` — best-response vs Tuned
DQN trained solely to exploit Tuned. Used for the exploitability analysis, not as a general agent.

**Config:** `episodes=50000`, `seed=0`

**Measured:** best_response_win 0.308, generic_baseline 0.5067, exploitability -0.1987, generic_agent TensThenTricks

`2.4 MB` · `sha256:835e8abb571c113b…`

## Multi-Agent Rl

### `rl_mappo_potential_s0.pt` — MAPPO (CTDE)
Shared actor across both partner seats; trained with a centralised critic and self-play.

**Config:** `algo=mappo`, `reward=potential`, `episodes=200000`

**Measured:** Random 0.7208, TensThenTricks 0.4483, GreedyTricks 0.4475

`2.4 MB` · `sha256:b0739b87ea579334…`

### `rl_mappo_potential_s1.pt` — MAPPO (CTDE)
Shared actor across both partner seats; trained with a centralised critic and self-play.

**Config:** `algo=mappo`, `reward=potential`, `episodes=200000`

**Measured:** Random 0.715, TensThenTricks 0.485, GreedyTricks 0.47

`2.4 MB` · `sha256:86b53c72c9213ce5…`

### `rl_mappo_potential_s2.pt` — MAPPO (CTDE)
Shared actor across both partner seats; trained with a centralised critic and self-play.

**Config:** `algo=mappo`, `reward=potential`, `episodes=200000`

**Measured:** Random 0.7204, TensThenTricks 0.4433, GreedyTricks 0.4333

`2.4 MB` · `sha256:390c9131a772edb3…`

## Neural Search

### `alphadj_s0.pt` — AlphaDJ
Policy+value network used as prior and leaf evaluator inside determinized information-set search.

`2.4 MB` · `sha256:45229a8d17387172…`

### `alphadj_s1.pt` — AlphaDJ
Policy+value network used as prior and leaf evaluator inside determinized information-set search.

`2.4 MB` · `sha256:e56626c0cfaf2c71…`

### `alphadj_s2.pt` — AlphaDJ
Policy+value network used as prior and leaf evaluator inside determinized information-set search.

`2.4 MB` · `sha256:8cf8d8cc1747a39c…`

## Single-Agent Rl

### `gen_dqn_Adaptive_s0.pt` — DQN
DQN agent. Reward scheme is recorded in `config`.

**Config:** `algo=dqn`, `reward=potential`, `episodes=120000`, `seed=0`

`2.4 MB` · `sha256:fb1583b4f9d64c9e…`

### `gen_dqn_Adaptive_s1.pt` — DQN
DQN agent. Reward scheme is recorded in `config`.

**Config:** `algo=dqn`, `reward=potential`, `episodes=120000`, `seed=1`

`2.4 MB` · `sha256:0ee3fffa2a677943…`

### `gen_dqn_Adaptive_s2.pt` — DQN
DQN agent. Reward scheme is recorded in `config`.

**Config:** `algo=dqn`, `reward=potential`, `episodes=120000`, `seed=2`

`2.4 MB` · `sha256:57e1e9b8ebf826ed…`

### `gen_dqn_GreedyTricks_s0.pt` — DQN
DQN agent. Reward scheme is recorded in `config`.

**Config:** `algo=dqn`, `reward=potential`, `episodes=120000`, `seed=0`

`2.4 MB` · `sha256:eda7aa15aa4e1556…`

### `gen_dqn_GreedyTricks_s1.pt` — DQN
DQN agent. Reward scheme is recorded in `config`.

**Config:** `algo=dqn`, `reward=potential`, `episodes=120000`, `seed=1`

`2.4 MB` · `sha256:7f114a5db792a871…`

### `gen_dqn_GreedyTricks_s2.pt` — DQN
DQN agent. Reward scheme is recorded in `config`.

**Config:** `algo=dqn`, `reward=potential`, `episodes=120000`, `seed=2`

`2.4 MB` · `sha256:906d2b9e2ec6833b…`

### `gen_dqn_Random_s0.pt` — DQN
DQN agent. Reward scheme is recorded in `config`.

**Config:** `algo=dqn`, `reward=potential`, `episodes=300`, `seed=0`

`2.4 MB` · `sha256:87ad8643164d8ce3…`

### `gen_dqn_Random_s1.pt` — DQN
DQN agent. Reward scheme is recorded in `config`.

**Config:** `algo=dqn`, `reward=potential`, `episodes=120000`, `seed=1`

`2.4 MB` · `sha256:01c3393495807770…`

### `gen_dqn_Random_s2.pt` — DQN
DQN agent. Reward scheme is recorded in `config`.

**Config:** `algo=dqn`, `reward=potential`, `episodes=120000`, `seed=2`

`2.4 MB` · `sha256:3a5b031b2a372c31…`

### `gen_dqn_TenAware_s0.pt` — DQN
DQN agent. Reward scheme is recorded in `config`.

**Config:** `algo=dqn`, `reward=potential`, `episodes=120000`, `seed=0`

`2.4 MB` · `sha256:e6369e95df2e72b5…`

### `gen_dqn_TenAware_s1.pt` — DQN
DQN agent. Reward scheme is recorded in `config`.

**Config:** `algo=dqn`, `reward=potential`, `episodes=120000`, `seed=1`

`2.4 MB` · `sha256:b44e9e400fd3292d…`

### `gen_dqn_TenAware_s2.pt` — DQN
DQN agent. Reward scheme is recorded in `config`.

**Config:** `algo=dqn`, `reward=potential`, `episodes=120000`, `seed=2`

`2.4 MB` · `sha256:8f4e9b4cdebb56e0…`

### `gen_dqn_TensThenTricks_s0.pt` — DQN
DQN agent. Reward scheme is recorded in `config`.

**Config:** `algo=dqn`, `reward=potential`, `episodes=120000`, `seed=0`

`2.4 MB` · `sha256:8baf3e14b77df56e…`

### `gen_dqn_TensThenTricks_s1.pt` — DQN
DQN agent. Reward scheme is recorded in `config`.

**Config:** `algo=dqn`, `reward=potential`, `episodes=120000`, `seed=1`

`2.4 MB` · `sha256:db83a17016dd2166…`

### `gen_dqn_TensThenTricks_s2.pt` — DQN
DQN agent. Reward scheme is recorded in `config`.

**Config:** `algo=dqn`, `reward=potential`, `episodes=120000`, `seed=2`

`2.4 MB` · `sha256:e656697bace5457f…`

### `gen_dqn_league_s0.pt` — DQN
DQN agent. Reward scheme is recorded in `config`.

**Config:** `algo=dqn`, `reward=potential`, `episodes=120000`, `seed=0`

`2.4 MB` · `sha256:2a5df8890a22c2c9…`

### `gen_dqn_league_s1.pt` — DQN
DQN agent. Reward scheme is recorded in `config`.

**Config:** `algo=dqn`, `reward=potential`, `episodes=120000`, `seed=1`

`2.4 MB` · `sha256:b6d60a417af8aa26…`

### `gen_dqn_league_s2.pt` — DQN
DQN agent. Reward scheme is recorded in `config`.

**Config:** `algo=dqn`, `reward=potential`, `episodes=120000`, `seed=2`

`2.4 MB` · `sha256:25e1e4f2a27c6d26…`

### `gen_dqn_self_s0.pt` — DQN
DQN agent. Reward scheme is recorded in `config`.

**Config:** `algo=dqn`, `reward=potential`, `episodes=120000`, `seed=0`

`2.4 MB` · `sha256:b5140a20d9be6e99…`

### `gen_dqn_self_s1.pt` — DQN
DQN agent. Reward scheme is recorded in `config`.

**Config:** `algo=dqn`, `reward=potential`, `episodes=120000`, `seed=1`

`2.4 MB` · `sha256:dcce3291dcd1aa08…`

### `gen_dqn_self_s2.pt` — DQN
DQN agent. Reward scheme is recorded in `config`.

**Config:** `algo=dqn`, `reward=potential`, `episodes=120000`, `seed=2`

`2.4 MB` · `sha256:8de60c0ef1a6b6ee…`

### `gen_ppo_Adaptive_s0.pt` — PPO
PPO agent. Reward scheme is recorded in `config`.

**Config:** `algo=ppo`, `reward=potential`, `episodes=120000`, `seed=0`

`2.4 MB` · `sha256:dc99b8593a2bdc31…`

### `gen_ppo_Adaptive_s1.pt` — PPO
PPO agent. Reward scheme is recorded in `config`.

**Config:** `algo=ppo`, `reward=potential`, `episodes=120000`, `seed=1`

`2.4 MB` · `sha256:44dd14298fafddad…`

### `gen_ppo_Adaptive_s2.pt` — PPO
PPO agent. Reward scheme is recorded in `config`.

**Config:** `algo=ppo`, `reward=potential`, `episodes=120000`, `seed=2`

`2.4 MB` · `sha256:e39d4f2eb2a08b0b…`

### `gen_ppo_GreedyTricks_s0.pt` — PPO
PPO agent. Reward scheme is recorded in `config`.

**Config:** `algo=ppo`, `reward=potential`, `episodes=120000`, `seed=0`

`2.4 MB` · `sha256:0680409195df986d…`

### `gen_ppo_GreedyTricks_s1.pt` — PPO
PPO agent. Reward scheme is recorded in `config`.

**Config:** `algo=ppo`, `reward=potential`, `episodes=120000`, `seed=1`

`2.4 MB` · `sha256:1459b3755174c98b…`

### `gen_ppo_GreedyTricks_s2.pt` — PPO
PPO agent. Reward scheme is recorded in `config`.

**Config:** `algo=ppo`, `reward=potential`, `episodes=120000`, `seed=2`

`2.4 MB` · `sha256:47d29f7be557a1e7…`

### `gen_ppo_Random_s0.pt` — PPO
PPO agent. Reward scheme is recorded in `config`.

**Config:** `algo=ppo`, `reward=potential`, `episodes=120000`, `seed=0`

`2.4 MB` · `sha256:554231368f3a498f…`

### `gen_ppo_Random_s1.pt` — PPO
PPO agent. Reward scheme is recorded in `config`.

**Config:** `algo=ppo`, `reward=potential`, `episodes=120000`, `seed=1`

`2.4 MB` · `sha256:f3c03831bddfef8b…`

### `gen_ppo_Random_s2.pt` — PPO
PPO agent. Reward scheme is recorded in `config`.

**Config:** `algo=ppo`, `reward=potential`, `episodes=120000`, `seed=2`

`2.4 MB` · `sha256:e3c19a5c4940a72c…`

### `gen_ppo_TenAware_s0.pt` — PPO
PPO agent. Reward scheme is recorded in `config`.

**Config:** `algo=ppo`, `reward=potential`, `episodes=120000`, `seed=0`

`2.4 MB` · `sha256:6d706359f7a866db…`

### `gen_ppo_TenAware_s1.pt` — PPO
PPO agent. Reward scheme is recorded in `config`.

**Config:** `algo=ppo`, `reward=potential`, `episodes=120000`, `seed=1`

`2.4 MB` · `sha256:90a1f04a2f21c8a1…`

### `gen_ppo_TenAware_s2.pt` — PPO
PPO agent. Reward scheme is recorded in `config`.

**Config:** `algo=ppo`, `reward=potential`, `episodes=120000`, `seed=2`

`2.4 MB` · `sha256:11b005072be93c16…`

### `gen_ppo_TensThenTricks_s0.pt` — PPO
PPO agent. Reward scheme is recorded in `config`.

**Config:** `algo=ppo`, `reward=potential`, `episodes=120000`, `seed=0`

`2.4 MB` · `sha256:d7eb089fa87c8122…`

### `gen_ppo_TensThenTricks_s1.pt` — PPO
PPO agent. Reward scheme is recorded in `config`.

**Config:** `algo=ppo`, `reward=potential`, `episodes=120000`, `seed=1`

`2.4 MB` · `sha256:5678dfe70b82fcd8…`

### `gen_ppo_TensThenTricks_s2.pt` — PPO
PPO agent. Reward scheme is recorded in `config`.

**Config:** `algo=ppo`, `reward=potential`, `episodes=120000`, `seed=2`

`2.4 MB` · `sha256:24d275f831f90d1f…`

### `gen_ppo_league_s0.pt` — PPO
PPO agent. Reward scheme is recorded in `config`.

**Config:** `algo=ppo`, `reward=potential`, `episodes=120000`, `seed=0`

`2.4 MB` · `sha256:1b0aa8611d94f70d…`

### `gen_ppo_league_s1.pt` — PPO
PPO agent. Reward scheme is recorded in `config`.

**Config:** `algo=ppo`, `reward=potential`, `episodes=120000`, `seed=1`

`2.4 MB` · `sha256:9f56aabbdf083137…`

### `gen_ppo_league_s2.pt` — PPO
PPO agent. Reward scheme is recorded in `config`.

**Config:** `algo=ppo`, `reward=potential`, `episodes=120000`, `seed=2`

`2.4 MB` · `sha256:d492aa741f2839b4…`

### `gen_ppo_self_s0.pt` — PPO
PPO agent. Reward scheme is recorded in `config`.

**Config:** `algo=ppo`, `reward=potential`, `episodes=300`, `seed=0`

`2.4 MB` · `sha256:d248dbb20c197a5f…`

### `gen_ppo_self_s1.pt` — PPO
PPO agent. Reward scheme is recorded in `config`.

**Config:** `algo=ppo`, `reward=potential`, `episodes=120000`, `seed=1`

`2.4 MB` · `sha256:37b3d55ed92518a6…`

### `gen_ppo_self_s2.pt` — PPO
PPO agent. Reward scheme is recorded in `config`.

**Config:** `algo=ppo`, `reward=potential`, `episodes=120000`, `seed=2`

`2.4 MB` · `sha256:0e9b7b7dccc9020a…`

### `rl_a2c_s0.pt` — A2C
A2C agent. Reward scheme is recorded in `config`.

**Config:** `method=a2c`

**Measured:** Random 0.6108, TensThenTricks 0.3967

`2.4 MB` · `sha256:6f9efb4809d01fc2…`

### `rl_a2c_s1.pt` — A2C
A2C agent. Reward scheme is recorded in `config`.

**Config:** `method=a2c`

**Measured:** Random 0.5988, TensThenTricks 0.345

`2.4 MB` · `sha256:8d594f779f799c72…`

### `rl_a2c_s2.pt` — A2C
A2C agent. Reward scheme is recorded in `config`.

**Config:** `method=a2c`

**Measured:** Random 0.6558, TensThenTricks 0.4133

`2.4 MB` · `sha256:14d850cdd5a38453…`

### `rl_double_dueling_s0.pt` — Double+Dueling DQN
Double+Dueling DQN agent. Reward scheme is recorded in `config`.

**Config:** `method=double_dueling`

**Measured:** Random 0.6142, TensThenTricks 0.3642

`2.4 MB` · `sha256:201ee46f80829a01…`

### `rl_double_dueling_s1.pt` — Double+Dueling DQN
Double+Dueling DQN agent. Reward scheme is recorded in `config`.

**Config:** `method=double_dueling`

**Measured:** Random 0.6188, TensThenTricks 0.35

`2.4 MB` · `sha256:6a144bf953f690e6…`

### `rl_double_dueling_s2.pt` — Double+Dueling DQN
Double+Dueling DQN agent. Reward scheme is recorded in `config`.

**Config:** `method=double_dueling`

**Measured:** Random 0.6067, TensThenTricks 0.3375

`2.4 MB` · `sha256:1e8e3458b2d664fd…`

### `rl_double_s0.pt` — Double DQN
Double DQN agent. Reward scheme is recorded in `config`.

**Config:** `method=double`

**Measured:** Random 0.6021, TensThenTricks 0.3525

`2.4 MB` · `sha256:d33e48cf118364e1…`

### `rl_double_s1.pt` — Double DQN
Double DQN agent. Reward scheme is recorded in `config`.

**Config:** `method=double`

**Measured:** Random 0.5758, TensThenTricks 0.3392

`2.4 MB` · `sha256:9c06996b00385c3a…`

### `rl_double_s2.pt` — Double DQN
Double DQN agent. Reward scheme is recorded in `config`.

**Config:** `method=double`

**Measured:** Random 0.6029, TensThenTricks 0.3192

`2.4 MB` · `sha256:a995b46b0a141d19…`

### `rl_dqn_potential_s0.pt` — DQN
DQN agent. Reward scheme is recorded in `config`.

**Config:** `algo=dqn`, `reward=potential`, `episodes=120000`, `seed=0`

**Measured:** Random 0.6183, TensThenTricks 0.3375

`2.4 MB` · `sha256:acd0ad0ec6935205…`

### `rl_dqn_potential_s1.pt` — DQN
DQN agent. Reward scheme is recorded in `config`.

**Config:** `algo=dqn`, `reward=potential`, `episodes=120000`, `seed=1`

**Measured:** Random 0.6054, TensThenTricks 0.3308

`2.4 MB` · `sha256:ba7326ab10f80074…`

### `rl_dqn_potential_s2.pt` — DQN
DQN agent. Reward scheme is recorded in `config`.

**Config:** `algo=dqn`, `reward=potential`, `episodes=120000`, `seed=2`

**Measured:** Random 0.6183, TensThenTricks 0.3533

`2.4 MB` · `sha256:59154c3e2dcd4059…`

### `rl_dqn_ten_shaped_s0.pt` — DQN
DQN agent. Reward scheme is recorded in `config`.

**Config:** `algo=dqn`, `reward=ten_shaped`, `episodes=120000`, `seed=0`

**Measured:** Random 0.5996, TensThenTricks 0.3558

`2.4 MB` · `sha256:760fc6103e3e05e9…`

### `rl_dqn_ten_shaped_s1.pt` — DQN
DQN agent. Reward scheme is recorded in `config`.

**Config:** `algo=dqn`, `reward=ten_shaped`, `episodes=120000`, `seed=1`

**Measured:** Random 0.6283, TensThenTricks 0.3733

`2.4 MB` · `sha256:f5f95847456e6e69…`

### `rl_dqn_ten_shaped_s2.pt` — DQN
DQN agent. Reward scheme is recorded in `config`.

**Config:** `algo=dqn`, `reward=ten_shaped`, `episodes=120000`, `seed=2`

**Measured:** Random 0.6029, TensThenTricks 0.4025

`2.4 MB` · `sha256:03cf1fdb2da1bf38…`

### `rl_dqn_terminal_s0.pt` — DQN
DQN agent. Reward scheme is recorded in `config`.

**Config:** `algo=dqn`, `reward=terminal`, `episodes=120000`, `seed=0`

**Measured:** Random 0.5988, TensThenTricks 0.3417

`2.4 MB` · `sha256:59612508dac0b9e2…`

### `rl_dqn_terminal_s1.pt` — DQN
DQN agent. Reward scheme is recorded in `config`.

**Config:** `algo=dqn`, `reward=terminal`, `episodes=120000`, `seed=1`

**Measured:** Random 0.5733, TensThenTricks 0.3133

`2.4 MB` · `sha256:f871c902eb15f39b…`

### `rl_dqn_terminal_s2.pt` — DQN
DQN agent. Reward scheme is recorded in `config`.

**Config:** `algo=dqn`, `reward=terminal`, `episodes=120000`, `seed=2`

**Measured:** Random 0.625, TensThenTricks 0.3825

`2.4 MB` · `sha256:d90916ed002a45cf…`

### `rl_dqn_trick_shaped_s0.pt` — DQN
DQN agent. Reward scheme is recorded in `config`.

**Config:** `algo=dqn`, `reward=trick_shaped`, `episodes=120000`, `seed=0`

**Measured:** Random 0.5983, TensThenTricks 0.33

`2.4 MB` · `sha256:059dd974417e7fd4…`

### `rl_dqn_trick_shaped_s1.pt` — DQN
DQN agent. Reward scheme is recorded in `config`.

**Config:** `algo=dqn`, `reward=trick_shaped`, `episodes=120000`, `seed=1`

**Measured:** Random 0.5713, TensThenTricks 0.3292

`2.4 MB` · `sha256:453c8f827994f564…`

### `rl_dqn_trick_shaped_s2.pt` — DQN
DQN agent. Reward scheme is recorded in `config`.

**Config:** `algo=dqn`, `reward=trick_shaped`, `episodes=120000`, `seed=2`

**Measured:** Random 0.6021, TensThenTricks 0.3417

`2.4 MB` · `sha256:33a0470f903caeb7…`

### `rl_ppo_potential_s0.pt` — PPO
PPO agent. Reward scheme is recorded in `config`.

**Config:** `algo=ppo`, `reward=potential`, `episodes=120000`, `seed=0`

**Measured:** Random 0.6971, TensThenTricks 0.5

`2.4 MB` · `sha256:ba3fa96bf4d9522d…`

### `rl_ppo_potential_s1.pt` — PPO
PPO agent. Reward scheme is recorded in `config`.

**Config:** `algo=ppo`, `reward=potential`, `episodes=120000`, `seed=1`

**Measured:** Random 0.6879, TensThenTricks 0.4958

`2.4 MB` · `sha256:5d05134ac806ddba…`

### `rl_ppo_potential_s2.pt` — PPO
PPO agent. Reward scheme is recorded in `config`.

**Config:** `algo=ppo`, `reward=potential`, `episodes=120000`, `seed=2`

**Measured:** Random 0.6913, TensThenTricks 0.5158

`2.4 MB` · `sha256:9e365f22a8a5e8ce…`

### `rl_ppo_ten_shaped_s0.pt` — PPO
PPO agent. Reward scheme is recorded in `config`.

**Config:** `algo=ppo`, `reward=ten_shaped`, `episodes=120000`, `seed=0`

**Measured:** Random 0.6825, TensThenTricks 0.5083

`2.4 MB` · `sha256:ec63a4c1bc53c846…`

### `rl_ppo_ten_shaped_s1.pt` — PPO
PPO agent. Reward scheme is recorded in `config`.

**Config:** `algo=ppo`, `reward=ten_shaped`, `episodes=120000`, `seed=1`

**Measured:** Random 0.6908, TensThenTricks 0.5008

`2.4 MB` · `sha256:029fb2257a31ffc2…`

### `rl_ppo_ten_shaped_s2.pt` — PPO
PPO agent. Reward scheme is recorded in `config`.

**Config:** `algo=ppo`, `reward=ten_shaped`, `episodes=120000`, `seed=2`

**Measured:** Random 0.6825, TensThenTricks 0.4992

`2.4 MB` · `sha256:eced318e85aeb2c0…`

### `rl_ppo_terminal_s0.pt` — PPO
PPO agent. Reward scheme is recorded in `config`.

**Config:** `algo=ppo`, `reward=terminal`, `episodes=120000`, `seed=0`

**Measured:** Random 0.6933, TensThenTricks 0.5017

`2.4 MB` · `sha256:ad383f1614c59c6c…`

### `rl_ppo_terminal_s1.pt` — PPO
PPO agent. Reward scheme is recorded in `config`.

**Config:** `algo=ppo`, `reward=terminal`, `episodes=120000`, `seed=1`

**Measured:** Random 0.6821, TensThenTricks 0.4775

`2.4 MB` · `sha256:8dbc02e847c06b2e…`

### `rl_ppo_terminal_s2.pt` — PPO
PPO agent. Reward scheme is recorded in `config`.

**Config:** `algo=ppo`, `reward=terminal`, `episodes=120000`, `seed=2`

**Measured:** Random 0.6779, TensThenTricks 0.4858

`2.4 MB` · `sha256:1bcbcb5347f936a2…`

### `rl_ppo_trick_shaped_s0.pt` — PPO
PPO agent. Reward scheme is recorded in `config`.

**Config:** `algo=ppo`, `reward=trick_shaped`, `episodes=120000`, `seed=0`

**Measured:** Random 0.7063, TensThenTricks 0.5067

`2.4 MB` · `sha256:5fc3194b52ca4c86…`

### `rl_ppo_trick_shaped_s1.pt` — PPO
PPO agent. Reward scheme is recorded in `config`.

**Config:** `algo=ppo`, `reward=trick_shaped`, `episodes=120000`, `seed=1`

**Measured:** Random 0.6892, TensThenTricks 0.5033

`2.4 MB` · `sha256:5e13df2aadbe03d6…`

### `rl_ppo_trick_shaped_s2.pt` — PPO
PPO agent. Reward scheme is recorded in `config`.

**Config:** `algo=ppo`, `reward=trick_shaped`, `episodes=120000`, `seed=2`

**Measured:** Random 0.6883, TensThenTricks 0.4917

`2.4 MB` · `sha256:a8fb85a70fa6114b…`

## Licence and citation

Code is released under the MIT licence; the models, data and figures under Creative Commons Attribution 4.0 International (CC BY 4.0).

If you use these, please cite the paper; see CITATION.cff in the repository root.

## Provenance

Every model was trained against the engine in `05-engine/`, which is tested to the rule specification in `01-rulebooks/` — 31 spec tests, one per rule ID, plus 7 tests proving no agent can observe hidden information.
