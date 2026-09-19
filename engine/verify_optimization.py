"""Prove the replay-buffer optimisation changed no result.

The optimisation stores observations as float32 arrays at insertion instead of
rebuilding tensors from Python lists on every batch. That is a pure
representation change: the RNG stream is untouched (same deque, same
rng.sample) and float32 was already the dtype torch.tensor produced. So every
trained network must come out BIT-IDENTICAL, not merely close.

"Close" would not be good enough. A tolerance would let a real behavioural
change hide inside it, and the whole point is that published results predating
the optimisation remain valid.

    python verify_optimization.py --episodes 600

Writes optimization_equivalence.json.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time

sys.path.insert(0, ".")
import runenv
import torch

PREOPT = ("/private/tmp/claude-501/-Users-rajuyadav-Desktop-Card/"
          "f10058df-1a63-4647-95fa-7bbee0047bd8/scratchpad/trainer_preopt.py")

REWARDS = ("terminal", "ten_shaped", "potential", "trick_shaped")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=600)
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()

    # DQN needs 2000 buffered transitions before it updates and PPO needs a
    # 2048-step horizon; below that both return untouched initialisations and
    # would match trivially.
    if a.episodes * 13 < 3000:
        print(f"[abort] {a.episodes} episodes performs no gradient update; "
              f"the comparison would be vacuous. Use >= 260.")
        sys.exit(1)

    old = load_module("trainer_preopt", PREOPT)
    import trainer as new

    rows, all_same = [], True
    t0 = time.perf_counter()
    for algo in ("dqn", "ppo"):
        fo = old.train_dqn if algo == "dqn" else old.train_ppo
        fn = new.train_dqn if algo == "dqn" else new.train_ppo
        for reward in REWARDS:
            for seed in range(a.seeds):
                A = fo(reward, a.episodes, seed, [], opponent="TensThenTricks",
                       eval_every=0)
                B = fn(reward, a.episodes, seed, [], opponent="TensThenTricks",
                       eval_every=0)
                sa, sb = A.state_dict(), B.state_dict()
                same = sa.keys() == sb.keys() and all(
                    torch.equal(sa[k], sb[k]) for k in sa)
                worst = max((sa[k] - sb[k]).abs().max().item() for k in sa)
                # a run whose weights never moved would match for the wrong reason
                torch.manual_seed(seed)
                fresh = (new.Net(critic=True) if algo == "ppo"
                         else new.Net()).state_dict()
                trained = any(not torch.equal(sa[k], fresh[k])
                              for k in sa if k in fresh)
                ok = same and trained
                all_same &= ok
                rows.append({"algo": algo, "reward": reward, "seed": seed,
                             "bit_identical": same, "max_abs_delta": worst,
                             "weights_moved_from_init": trained, "ok": ok})
                print(f"  {algo:3s} {reward:13s} s{seed}  "
                      f"identical={same}  max|delta|={worst:.3e}  "
                      f"trained={trained}", flush=True)

    el = time.perf_counter() - t0
    json.dump({"env": runenv.snapshot(), "episodes": a.episodes,
               "n_configurations": len(rows), "all_bit_identical": all_same,
               "criterion": ("exact equality of every parameter tensor; no "
                             "tolerance is allowed, because a tolerance could "
                             "conceal a genuine behavioural change"),
               "results": rows, "seconds": el},
              open("optimization_equivalence.json", "w"), indent=2)
    print(f"\n{len(rows)} configurations, all bit-identical: {all_same}  "
          f"({el:.0f}s)")
    print("  -> optimization_equivalence.json")
    sys.exit(0 if all_same else 1)
