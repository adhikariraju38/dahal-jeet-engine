"""End-to-end proof: reproduce a stored checkpoint exactly with the new code.

The grid check in verify_optimization.py compares short runs of the old and new
trainer against each other. This is stronger: it takes a checkpoint that is
ALREADY ON DISK -- trained at full length by the original pre-refactor
train_rl.py, before the nets.py split and before the buffer optimisation -- and
retrains it with today's code at the same config and seed.

If the weights match bit for bit, then every artifact produced so far is
reproducible by the current code, and the refactor plus optimisation provably
changed nothing about the results.

    python reproduce_artifact.py --algo dqn --reward potential --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, ".")
import runenv
import torch

from trainer import train_dqn, train_ppo

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", default="dqn", choices=["dqn", "ppo"])
    ap.add_argument("--reward", default="potential")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=120000)
    a = ap.parse_args()

    ckpt = f"rl_{a.algo}_{a.reward}_s{a.seed}.pt"
    if not os.path.exists(ckpt):
        print(f"[abort] {ckpt} not on disk")
        sys.exit(1)
    stored = torch.load(ckpt, map_location="cpu")

    print(f"reproducing {ckpt} ({a.episodes} episodes) with current code",
          flush=True)
    t0 = time.perf_counter()
    fn = train_dqn if a.algo == "dqn" else train_ppo
    net = fn(a.reward, a.episodes, a.seed, [], opponent="TensThenTricks",
             eval_every=0)
    el = time.perf_counter() - t0

    fresh = net.state_dict()
    same = stored.keys() == fresh.keys() and all(
        torch.equal(stored[k], fresh[k]) for k in stored)
    worst = max((stored[k] - fresh[k]).abs().max().item() for k in stored
                if k in fresh)
    print(f"\n  bit-identical to stored checkpoint: {same}")
    print(f"  max |delta| = {worst:.3e}")
    print(f"  wall-clock now {el/60:.1f} min "
          f"(the stored run took ~108 min before the optimisation)")

    json.dump({"env": runenv.snapshot(), "checkpoint": ckpt,
               "algo": a.algo, "reward": a.reward, "seed": a.seed,
               "episodes": a.episodes, "bit_identical": same,
               "max_abs_delta": worst, "seconds_now": el,
               "meaning": ("The stored checkpoint was trained by the original "
                           "train_rl.py before the nets.py split and before the "
                           "replay-buffer optimisation. Reproducing it exactly "
                           "shows those changes altered no result, so artifacts "
                           "produced earlier remain valid and reproducible.")},
              open(f"reproduce_{a.algo}_{a.reward}_s{a.seed}.json", "w"),
              indent=2)
    sys.exit(0 if same else 1)
