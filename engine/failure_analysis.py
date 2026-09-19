"""Where do agents actually lose? (gap E6)

A win rate says how often, never why. This conditions outcomes on properties of
the deal that are fixed before a card is played, so systematic weaknesses show
up as buckets where an agent underperforms its own average:

  tens_dealt      how many of the four tens started in the agent's team's hands
  holds_trump     whether the agent's team drew the trump card (T1/T2)
  trump_length    how many trumps the team held
  decided_by      tens majority, or the 2-2 trick tiebreak

It also answers a specific open question. Under PPO the deliberately MISALIGNED
`trick_shaped` reward did not underperform, and the suspicion was that
trick-taking is not really misaligned here because a 2-2 tens split is settled
on trick count. If that is right, trick_shaped should win a larger share of
TIEBREAK hands and no more tens-decided hands. That is checkable, so it is
checked rather than asserted.

    python failure_analysis.py --deals 400 --agents TensThenTricks,rl_ppo_potential_s0.pt
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict

sys.path.insert(0, ".")
import runenv
import torch

from dahaljeet.agents import REGISTRY
from dahaljeet.cards import is_ten, suit_of
from dahaljeet.hand import Hand, team_of
from dahaljeet.view import make_view
from nets import Net, TorchAgent


def make_agent(spec, rng):
    if spec.endswith(".pt"):
        sd = torch.load(spec, map_location="cpu")
        for critic in (False, True):
            try:
                net = Net(critic=critic)
                net.load_state_dict(sd)
                net.eval()
                return TorchAgent(net, os.path.basename(spec)[:-3], greedy=True,
                                  rng=rng)
            except Exception:
                continue
        raise SystemExit(f"cannot load {spec}")
    if spec.startswith("PIMC:"):
        from dahaljeet.search import PIMCAgent
        return PIMCAgent(worlds=int(spec.split(":")[1]), rng=rng)
    if spec.startswith("ISMCTS:"):
        from dahaljeet.search import ISMCTSAgent
        return ISMCTSAgent(iterations=int(spec.split(":")[1]), rng=rng)
    return REGISTRY[spec](rng=rng)


def analyse(a_spec, b_spec, n_deals, seed):
    base = random.Random(seed).randrange(1 << 30)
    buckets = defaultdict(lambda: [0, 0])      # key -> [wins, hands]

    def note(key, won):
        buckets[key][0] += int(won)
        buckets[key][1] += 1

    for d in range(n_deals):
        for r in range(4):
            a = make_agent(a_spec, random.Random(1000 + d))
            b = make_agent(b_spec, random.Random(2000 + d))
            seats = {s: (a if ((s - r) % 4) % 2 == 0 else b) for s in range(4)}
            h = Hand(dealer=0, rng=random.Random(base + d))
            h.deal()
            us = team_of(r)

            # pre-play features, fixed before any decision
            my_seats = [s for s in range(4) if team_of(s) == us]
            tens_dealt = sum(1 for s in my_seats for c in h.hands[s] if is_ten(c))
            holds_trump = team_of(h.trump_holder) == us
            trump_len = sum(1 for s in my_seats for c in h.hands[s]
                            if suit_of(c) == h.trump_suit)

            while not h.is_over:
                h.play(seats[h.to_act].act(make_view(h, h.to_act)))
            res = h.result()
            won = res.winning_team == us
            tiebreak = res.tens_by_team[0] == res.tens_by_team[1] == 2

            note(("overall", "all"), won)
            note(("tens_dealt", tens_dealt), won)
            note(("holds_trump", bool(holds_trump)), won)
            note(("trump_length", min(trump_len, 8)), won)
            note(("decided_by", "tiebreak" if tiebreak else "tens"), won)
    out = {}
    for (dim, val), (w, n) in sorted(buckets.items(), key=lambda kv: str(kv[0])):
        out.setdefault(dim, {})[str(val)] = {
            "win_rate": round(w / n, 4), "hands": n}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--deals", type=int, default=400)
    ap.add_argument("--opponent", default="TensThenTricks")
    ap.add_argument("--agents", default="TensThenTricks")
    ap.add_argument("--seed", type=int, default=515)
    a = ap.parse_args()

    t0 = time.perf_counter()
    results = {}
    for spec in a.agents.split(","):
        spec = spec.strip()
        if spec.endswith(".pt") and not os.path.exists(spec):
            print(f"  [skip] {spec} not on disk")
            continue
        r = analyse(spec, a.opponent, a.deals, a.seed)
        results[spec] = r
        ov = r["overall"]["all"]["win_rate"]
        print(f"\n{spec}  vs {a.opponent}   overall {ov:.4f} "
              f"({r['overall']['all']['hands']} hands)", flush=True)
        for dim in ("tens_dealt", "holds_trump", "trump_length", "decided_by"):
            if dim not in r:
                continue
            cells = "  ".join(
                f"{k}:{v['win_rate']:.3f}(n={v['hands']})"
                for k, v in sorted(r[dim].items()))
            print(f"    {dim:13s} {cells}", flush=True)

    el = time.perf_counter() - t0
    json.dump({"env": runenv.snapshot(), "deals": a.deals,
               "opponent": a.opponent, "results": results, "seconds": el,
               "note": ("Buckets are properties of the DEAL, fixed before play, "
                        "so a low win rate in a bucket is a weakness of the "
                        "policy in that situation rather than a consequence of "
                        "how it played.")},
              open("failure_analysis.json", "w"), indent=2)
    print(f"\n{el:.0f}s -> failure_analysis.json")
