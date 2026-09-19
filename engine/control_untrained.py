"""Untrained-network control.

A trained agent's win rate means nothing on its own: the architecture, the
action masking and the encoding already impose structure, so a randomly
initialised network is NOT equivalent to random play. Reporting "our agent
beats Random" without this control leaves open how much of the margin came from
learning and how much came from the harness.

This measures the floor: the same architectures, same masking, same evaluation
protocol, with weights straight from initialisation and no training at all.

    python control_untrained.py --deals 400 --seeds 3

Every trained result should be read as the distance from THIS line, not from
Random.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time

sys.path.insert(0, ".")
import runenv
import torch

from dahaljeet.agents import REGISTRY
from dahaljeet.tournament import duplicate_match
from train_rl import Net, TorchAgent

OPPONENTS = ("Random", "GreedyTricks", "TenAware", "TensThenTricks")


def build(arch, seed):
    """Randomly initialised net for each architecture we train."""
    torch.manual_seed(seed)
    if arch == "dqn":
        net = Net()
    elif arch == "ppo":
        net = Net(critic=True)
    elif arch == "dueling":
        try:
            from train_more import DuelNet
            net = DuelNet()
        except Exception:
            return None
    else:
        return None
    net.eval()
    return net


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--deals", type=int, default=400)
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()

    t0 = time.perf_counter()
    rows = []
    for arch in ("dqn", "ppo", "dueling"):
        for seed in range(a.seeds):
            net = build(arch, seed)
            if net is None:
                print(f"  [skip] {arch} — architecture unavailable")
                continue
            agent = TorchAgent(net, f"untrained-{arch}-s{seed}", greedy=True,
                               rng=random.Random(seed))
            row = {"arch": arch, "seed": seed}
            for opp in OPPONENTS:
                m = duplicate_match(agent, REGISTRY[opp](rng=random.Random(4)),
                                    n_deals=a.deals, seed=555, timed=False)
                lo, hi = m.bootstrap_ci()
                row[opp] = {"win": round(m.win_rate_a, 4),
                            "ci": [round(lo, 4), round(hi, 4)]}
            rows.append(row)
            print(f"  untrained {arch:8s} s{seed}  " + "  ".join(
                f"vs {o} {row[o]['win']:.4f}" for o in OPPONENTS), flush=True)

    # aggregate per architecture across seeds
    agg = {}
    for arch in sorted({r["arch"] for r in rows}):
        sub = [r for r in rows if r["arch"] == arch]
        agg[arch] = {}
        for opp in OPPONENTS:
            v = [r[opp]["win"] for r in sub]
            mu = sum(v) / len(v)
            sd = (sum((x - mu) ** 2 for x in v) / len(v)) ** 0.5 if len(v) > 1 else 0.0
            agg[arch][opp] = {"mean": round(mu, 4), "sd": round(sd, 4),
                              "n_seeds": len(v)}

    el = time.perf_counter() - t0
    json.dump({"env": runenv.snapshot(), "deals_per_pair": a.deals,
               "per_run": rows, "aggregated": agg, "seconds": el,
               "note": ("Randomly initialised weights, no training. This is the "
                        "floor a trained agent must be read against, because "
                        "architecture and action masking alone are not random "
                        "play.")},
              open("control_untrained.json", "w"), indent=2)
    print(f"\n{el:.0f}s -> control_untrained.json")
