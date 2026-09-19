"""Phase 4 -- DQN and PPO with the reward ablation.

Run with the venv python (torch lives there):
    .venv/bin/python train_rl.py --algo dqn --reward potential --episodes 20000

The experiment is the REWARD COMPARISON, not the algorithm. Dahal Jeet's
scoring is unusually sparse -- only the four tens score, so a whole hand of
thirteen tricks yields at most four scoring events -- which makes the reward
scheme, not the optimiser, the thing most likely to decide whether learning
works. So the state layout is fixed and documented (dahaljeet/encode.py) and
the reward scheme is the independent variable. trick_shaped is included as a
deliberately MISALIGNED control: it rewards taking tricks, which is only the
secondary objective, and it should therefore underperform.
"""
from __future__ import annotations

import argparse, json, random, sys, time
from collections import deque

sys.path.insert(0, '.')
import runenv
import torch
import torch.nn as nn
import torch.nn.functional as F

from dahaljeet.agents import REGISTRY, Agent
from dahaljeet.encode import encode, legal_mask, OBS_SIZE, ACTION_SIZE
from dahaljeet.env import DahalJeetEnv
from dahaljeet.tournament import duplicate_match
from dahaljeet.view import make_view

DEV = torch.device("cpu")
NEG = -1e9


# Shared definitions live in nets.py; re-exported here because many scripts
# already do `from train_rl import Net, TorchAgent`.
from nets import DEV, NEG, Net, TorchAgent, evaluate  # noqa: F401

# The DQN and PPO loops now live in trainer.py, which takes the training
# opponent as a parameter. Keeping a second copy here would let the two drift;
# `python trainer.py --equivalence-test` asserts they are identical.
from trainer import train_dqn, train_ppo  # noqa: F401


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["dqn", "ppo"], default="dqn")
    ap.add_argument("--reward", default="potential")
    ap.add_argument("--episodes", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    print(f"[{a.algo} | reward={a.reward} | {a.episodes} episodes | seed {a.seed}]",
          flush=True)
    t0 = time.perf_counter()
    log = []
    net = (train_dqn if a.algo == "dqn" else train_ppo)(
        a.reward, a.episodes, a.seed, log)
    final = evaluate(TorchAgent(net, f"{a.algo}-{a.reward}"), deals=600)
    el = time.perf_counter() - t0
    print(f"FINAL {a.algo}/{a.reward}: {final}  ({el:.0f}s)", flush=True)
    tag = f"{a.algo}_{a.reward}_s{a.seed}"
    torch.save(net.state_dict(), f"rl_{tag}.pt")
    json.dump({"env": runenv.snapshot(), "algo": a.algo, "reward": a.reward, "episodes": a.episodes,
               "seed": a.seed, "final": final, "curve": log,
               "seconds": el}, open(f"rl_{tag}.json", "w"), indent=2)
    print(f"saved rl_{tag}.pt / .json")
