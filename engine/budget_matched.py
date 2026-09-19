"""Compare DQN and PPO at MATCHED compute, not matched episodes (gap C12).

Every learning result in this project matches algorithms on episodes. That is
not a fair comparison of optimisers: at 120k episodes DQN takes ~390,000
gradient steps and PPO ~3,000, a 128x difference, and DQN finishes in 108
minutes against PPO's 12. So "PPO beats DQN" currently conflates the algorithm
with the budget it happened to receive.

This measures both curves against two matched axes and reports both:

  WALL-CLOCK    the budget a practitioner actually has
  GRADIENT STEPS the budget an optimiser-focused reviewer will ask about

Matching is DIRECT, not interpolated: episode counts are chosen from a
calibration pass so the runs actually land on the target budget, and the
ACHIEVED budget is reported next to the target so the match can be checked
rather than trusted.

    python budget_matched.py --budgets 120,400,1200 --seeds 2
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time

sys.path.insert(0, ".")
import runenv

from dahaljeet.agents import REGISTRY
from dahaljeet.tournament import duplicate_match
from nets import TorchAgent
from trainer import STEPS, train_dqn, train_ppo

FN = {"dqn": train_dqn, "ppo": train_ppo}


def calibrate(algo, episodes, seed):
    """Seconds and gradient steps per episode, measured not assumed."""
    t0 = time.perf_counter()
    FN[algo]("potential", episodes, seed, [], opponent="TensThenTricks",
             eval_every=0)
    el = time.perf_counter() - t0
    return el / episodes, STEPS["updates"] / episodes


def run(algo, episodes, seed, deals):
    t0 = time.perf_counter()
    net = FN[algo]("potential", episodes, seed, [], opponent="TensThenTricks",
                   eval_every=0)
    el = time.perf_counter() - t0
    upd, smp = STEPS["updates"], STEPS["samples"]
    agent = TorchAgent(net, f"{algo}", greedy=True, rng=random.Random(7))
    out = {}
    for opp in ("Random", "TensThenTricks"):
        m = duplicate_match(agent, REGISTRY[opp](rng=random.Random(4)),
                            n_deals=deals, seed=808, timed=False)
        lo, hi = m.bootstrap_ci()
        out[opp] = {"win": round(m.win_rate_a, 4), "ci": [round(lo, 4), round(hi, 4)]}
    return {"algo": algo, "seed": seed, "episodes": episodes,
            "seconds": round(el, 1), "gradient_updates": upd,
            "samples_seen": smp, "eval": out}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", default="120,400,1200",
                    help="wall-clock seconds per run")
    ap.add_argument("--update-budgets", default="2000,8000,25000",
                    help="gradient steps per run")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--deals", type=int, default=400)
    ap.add_argument("--calib-episodes", type=int, default=600)
    a = ap.parse_args()

    t0 = time.perf_counter()
    print("calibration (measuring cost per episode)\n", flush=True)
    cal = {}
    for algo in ("dqn", "ppo"):
        sec_ep, upd_ep = calibrate(algo, a.calib_episodes, 0)
        cal[algo] = {"seconds_per_episode": sec_ep, "updates_per_episode": upd_ep}
        print(f"  {algo}: {sec_ep*1000:.2f} ms/episode, "
              f"{upd_ep:.3f} gradient steps/episode", flush=True)
    ratio = cal["dqn"]["updates_per_episode"] / max(cal["ppo"]["updates_per_episode"], 1e-9)
    print(f"\n  DQN takes {ratio:.0f}x more gradient steps per episode than PPO\n",
          flush=True)

    rows = []
    print("MATCHED WALL-CLOCK\n", flush=True)
    for budget in [float(x) for x in a.budgets.split(",")]:
        for algo in ("dqn", "ppo"):
            want = int(budget / cal[algo]["seconds_per_episode"])
            eps = max(200, want)
            floored = eps > want          # the floor bound; match will be poor
            for seed in range(a.seeds):
                r = run(algo, eps, seed, a.deals)
                r.update(matched_on="wall_clock", target=budget,
                         achieved=r["seconds"], episode_floor_bound=floored,
                         match_error_pct=round(100*(r["seconds"]-budget)/budget, 1))
                rows.append(r)
                warn = "  <-- 200-episode FLOOR bound; not a valid match" if floored else ""
                print(f"  {algo:3s} target {budget:6.0f}s  actual {r['seconds']:6.1f}s "
                      f"({r['match_error_pct']:+5.1f}%)  eps {eps:7,}  "
                      f"updates {r['gradient_updates']:7,}  "
                      f"vs TTT {r['eval']['TensThenTricks']['win']:.4f}{warn}", flush=True)

    print("\nMATCHED GRADIENT STEPS\n", flush=True)
    for budget in [float(x) for x in a.update_budgets.split(",")]:
        for algo in ("dqn", "ppo"):
            per = max(cal[algo]["updates_per_episode"], 1e-9)
            want = int(budget / per)
            eps = max(200, want)
            floored = eps > want
            for seed in range(a.seeds):
                r = run(algo, eps, seed, a.deals)
                r.update(matched_on="gradient_steps", target=budget,
                         achieved=r["gradient_updates"],
                         episode_floor_bound=floored,
                         match_error_pct=round(
                             100*(r["gradient_updates"]-budget)/budget, 1))
                rows.append(r)
                warn = "  <-- 200-episode FLOOR bound; not a valid match" if floored else ""
                print(f"  {algo:3s} target {budget:6.0f} updates  actual "
                      f"{r['gradient_updates']:7,} ({r['match_error_pct']:+5.1f}%)  "
                      f"eps {eps:7,}  {r['seconds']:6.1f}s  "
                      f"vs TTT {r['eval']['TensThenTricks']['win']:.4f}{warn}", flush=True)

    el = time.perf_counter() - t0
    json.dump({"env": runenv.snapshot(), "calibration": cal,
               "updates_per_episode_ratio_dqn_over_ppo": round(ratio, 1),
               "deals": a.deals, "runs": rows, "seconds": el,
               "validity_note": ("Runs with episode_floor_bound=true hit the "
                                 "200-episode minimum and did NOT reach the "
                                 "target budget; they are not valid matched "
                                 "comparisons and must be excluded or re-run "
                                 "at a larger budget."),
               "reading": ("Matched on EPISODES (the rest of this project) DQN "
                           "and PPO receive wildly different optimisation "
                           "budgets. Here they are matched on wall-clock and "
                           "again on gradient steps, and both are reported "
                           "because they can disagree and each answers a "
                           "different reviewer's question.")},
              open("budget_matched.json", "w"), indent=2)
    print(f"\n{el/60:.1f} min -> budget_matched.json")
