"""Opponent-generalisation matrix (gap C9).

Train against opponent A, evaluate against opponent B. If an agent only beats
the opponent it trained against, it learned that opponent rather than the game
-- and with every learner in this project trained against a single fixed
heuristic, we currently cannot tell those apart.

Two phases, so training can be parallelised by the pipeline:

    python generalise.py --train --algo dqn --opponent Random --episodes 120000 --seed 0
    python generalise.py --matrix

HEADLINE NUMBER

    overfitting gap = mean(win vs the opponent trained against)
                    - mean(win vs HELD-OUT opponents never trained against)

Held-out opponents are search agents (PIMC, ISMCTS), which no learner ever
trains against, so they cannot have been fitted. A large gap means the agents
fit an opponent; a small gap means they generalise. Both outcomes are
reportable, and the current design cannot distinguish them at all.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import sys
import time

sys.path.insert(0, ".")
import runenv
import torch

from dahaljeet.agents import REGISTRY
from dahaljeet.tournament import duplicate_match
from train_rl import Net, TorchAgent
from trainer import train_dqn, train_ppo

# opponents an agent may be TRAINED against
TRAIN_OPPONENTS = ["Random", "GreedyTricks", "TenAware", "Adaptive",
                   "TensThenTricks", "self", "league"]

# opponents used only for EVALUATION. The search agents are held out: no
# learner ever trains against them, so a good score there cannot be fitting.
HELD_OUT = ["PIMC:8", "ISMCTS:100"]


def make_eval_agent(spec):
    if spec.startswith("PIMC:"):
        from dahaljeet.search import PIMCAgent
        return PIMCAgent(worlds=int(spec.split(":")[1]), rng=random.Random(11))
    if spec.startswith("ISMCTS:"):
        from dahaljeet.search import ISMCTSAgent
        return ISMCTSAgent(iterations=int(spec.split(":")[1]),
                           rng=random.Random(12))
    return REGISTRY[spec](rng=random.Random(4))


def tag(algo, opponent, seed, prefix="gen"):
    return f"{prefix}_{algo}_{opponent.replace(':', '')}_s{seed}"


def do_train(algo, opponent, episodes, seed, reward="potential",
             prefix="gen", eval_every=None):
    t0 = time.perf_counter()
    log = []
    fn = train_dqn if algo == "dqn" else train_ppo
    net = fn(reward, episodes, seed, log, opponent=opponent,
             eval_every=eval_every)
    el = time.perf_counter() - t0
    t = tag(algo, opponent, seed, prefix)
    torch.save(net.state_dict(), f"{t}.pt")
    json.dump({"env": runenv.snapshot(), "algo": algo, "reward": reward,
               "train_opponent": opponent, "episodes": episodes, "seed": seed,
               "curve": log, "seconds": el},
              open(f"{t}.json", "w"), indent=2)
    print(f"saved {t}.pt ({el:.0f}s)", flush=True)


def load_net(path, algo):
    net = Net(critic=(algo == "ppo"))
    net.load_state_dict(torch.load(path, map_location="cpu"))
    net.eval()
    return net


def do_matrix(deals, deals_search, prefix="gen"):
    checkpoints = sorted(glob.glob(f"{prefix}_*.pt"))
    if not checkpoints:
        print(f"  [wait] no {prefix}_*.pt checkpoints yet")
        return False
    eval_names = [o for o in TRAIN_OPPONENTS if o not in ("self", "league")] \
        + HELD_OUT

    rows = []
    for path in checkpoints:
        m = re.match(rf"{prefix}_(dqn|ppo)_(.+)_s(\d+)\.pt$",
                     os.path.basename(path))
        if not m:
            print(f"  [skip] unparsable name: {path}")
            continue
        algo, train_opp, seed = m.group(1), m.group(2), int(m.group(3))
        agent = TorchAgent(load_net(path, algo), f"{algo}-{train_opp}",
                           greedy=True, rng=random.Random(7))
        row = {"algo": algo, "train_opponent": train_opp, "seed": seed,
               "vs": {}}
        for name in eval_names:
            n = deals_search if ":" in name else deals
            r = duplicate_match(agent, make_eval_agent(name), n_deals=n,
                                seed=515, timed=False)
            lo, hi = r.bootstrap_ci()
            row["vs"][name] = {"win": round(r.win_rate_a, 4),
                               "ci": [round(lo, 4), round(hi, 4)]}
        held = [row["vs"][h]["win"] for h in HELD_OUT if h in row["vs"]]
        fixed = [v["win"] for k, v in row["vs"].items() if k not in HELD_OUT]
        own = row["vs"].get(train_opp, {}).get("win")
        row["held_out_mean"] = round(sum(held) / len(held), 4) if held else None
        row["fixed_mean"] = round(sum(fixed) / len(fixed), 4) if fixed else None
        # `own` is undefined for self/league training -- there is no fixed
        # opponent to have overfitted TO. Their generalisation is judged on the
        # held-out column directly, which is the comparison that matters:
        # which training opponent produces the best transfer?
        row["vs_train_opponent"] = own
        if own is not None and row["held_out_mean"] is not None:
            row["overfitting_gap"] = round(own - row["held_out_mean"], 4)
        rows.append(row)
        print(f"  {algo:3s} trained-vs {train_opp:14s} s{seed}  "
              f"own {'n/a' if own is None else f'{own:.4f}'}  "
              f"fixed {row['fixed_mean']}  held-out {row['held_out_mean']}  "
              f"gap {row.get('overfitting_gap', 'n/a')}", flush=True)

    gaps = [r["overfitting_gap"] for r in rows if r.get("overfitting_gap") is not None]
    summary = {}
    if gaps:
        mu = sum(gaps) / len(gaps)
        sd = (sum((g - mu) ** 2 for g in gaps) / len(gaps)) ** 0.5 if len(gaps) > 1 else 0.0
        summary["raw_gap_CONFOUNDED"] = {
            "mean": round(mu, 4), "sd": round(sd, 4), "n": len(gaps),
            "warning": (
                "DO NOT report this as overfitting. own-minus-held-out compares "
                "performance against DIFFERENT opponents, so it mostly measures "
                "how much stronger the held-out search agents are than the "
                "training heuristic. An agent trained on Random scores ~0.61 vs "
                "Random and ~0.31 vs PIMC even when barely trained at all. Use "
                "opponent_specificity below.")}

    # --- the controlled measure
    #
    # specificity(X) = win(agent trained ON X, played vs X)
    #                - mean over Y != X of win(agent trained on Y, played vs X)
    #
    # Both terms are measured against the SAME opponent X, so opponent strength
    # cancels. What remains is the advantage that training specifically against
    # X confers when facing X -- which is opponent-fitting, and nothing else.
    spec = []
    fixed_train = [o for o in TRAIN_OPPONENTS if o not in ("self", "league")]
    for algo in sorted({r["algo"] for r in rows}):
        for X in fixed_train:
            diag = [r["vs"][X]["win"] for r in rows
                    if r["algo"] == algo and r["train_opponent"] == X
                    and X in r["vs"]]
            off = [r["vs"][X]["win"] for r in rows
                   if r["algo"] == algo and r["train_opponent"] != X
                   and X in r["vs"]]
            if not diag or not off:
                continue
            d = sum(diag) / len(diag)
            o = sum(off) / len(off)
            spec.append({"algo": algo, "opponent": X,
                         "trained_on_it": round(d, 4),
                         "trained_elsewhere": round(o, 4),
                         "specificity": round(d - o, 4),
                         "n_diag": len(diag), "n_off": len(off)})
    if spec:
        vals = [x["specificity"] for x in spec]
        mu2 = sum(vals) / len(vals)
        sd2 = (sum((v - mu2) ** 2 for v in vals) / len(vals)) ** 0.5 \
            if len(vals) > 1 else 0.0
        summary["opponent_specificity"] = {
            "mean": round(mu2, 4), "sd": round(sd2, 4), "n": len(vals),
            "per_opponent": spec,
            "reading": ("Advantage from having trained against this specific "
                        "opponent, with opponent strength cancelled. Near 0 "
                        "means the agents learned the game; clearly positive "
                        "means they learned their training opponent.")}
    json.dump({"env": runenv.snapshot(), "deals": deals,
               "deals_vs_search": deals_search,
               "train_opponents": TRAIN_OPPONENTS, "held_out": HELD_OUT,
               "rows": rows, "summary": summary},
              open(("generalisation_matrix" if prefix == "gen"
                    else prefix + "_matrix") + ".json", "w"), indent=2)
    out_name = ("generalisation_matrix" if prefix == "gen"
                else prefix + "_matrix") + ".json"
    sp = summary.get("opponent_specificity")
    if sp:
        print(f"\n  OPPONENT SPECIFICITY {sp['mean']:+.4f} +/- {sp['sd']:.4f} "
              f"over {sp['n']} opponents  (0 = learned the game, "
              f">0 = learned the opponent)")
        for x in sp["per_opponent"]:
            print(f"    {x['algo']:3s} vs {x['opponent']:14s} "
                  f"trained-on-it {x['trained_on_it']:.4f}  "
                  f"trained-elsewhere {x['trained_elsewhere']:.4f}  "
                  f"specificity {x['specificity']:+.4f}")
    elif rows:
        print("\n  [note] specificity needs >= 2 different training opponents "
              "per algorithm; not computable yet")
    # best transfer, across ALL training opponents including self-play
    best = sorted((r for r in rows if r["held_out_mean"] is not None),
                  key=lambda r: -r["held_out_mean"])
    if best:
        print("  best transfer to held-out search opponents:")
        for r in best[:4]:
            print(f"    {r['algo']:3s} trained-vs {r['train_opponent']:14s} "
                  f"held-out {r['held_out_mean']:.4f}")
    print(f"  -> {out_name}")
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--matrix", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny end-to-end run: train 2 configs, build a matrix")
    ap.add_argument("--algo", choices=["dqn", "ppo"], default="dqn")
    ap.add_argument("--opponent", default="TensThenTricks")
    ap.add_argument("--episodes", type=int, default=120000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--deals", type=int, default=600)
    ap.add_argument("--deals-search", type=int, default=150)
    a = ap.parse_args()

    if a.smoke:
        # Writes under a SEPARATE prefix so a crash or timeout can never touch
        # real gen_* checkpoints. An earlier version stashed and restored them,
        # which loses files if the run dies midway.
        print("SMOKE TEST — tiny train + matrix, separate prefix\n")
        P = "smokegen"
        for f in glob.glob(f"{P}_*"):
            os.remove(f)
        try:
            # >= 2 training opponents for the SAME algo, or specificity is
            # not computable and the smoke test would not exercise it
            for algo, opp in (("dqn", "Random"),
                              ("dqn", "TensThenTricks"),
                              ("dqn", "GreedyTricks"),
                              ("ppo", "self")):
                do_train(algo, opp, 300, 0, prefix=P, eval_every=0)
            ok = do_matrix(deals=30, deals_search=6, prefix=P)
            print(f"\n{'SMOKE OK' if ok else 'SMOKE FAILED'}")
        finally:
            for f in glob.glob(f"{P}_*"):
                os.remove(f)
        sys.exit(0 if ok else 1)

    if a.train:
        do_train(a.algo, a.opponent, a.episodes, a.seed)
    elif a.matrix:
        do_matrix(a.deals, a.deals_search)
    else:
        ap.error("choose --train, --matrix or --smoke")
