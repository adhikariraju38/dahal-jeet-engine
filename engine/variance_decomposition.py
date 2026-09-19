"""How much of a Dahal Jeet result is the cards, the seat, or the play? (D10)

A game whose entire score rides on four cards invites the obvious question:
how much of the outcome is luck? Duplicate dealing lets it be answered rather
than argued, because the SAME deal is played from all four seat assignments.

Design: for each deal d and rotation r (0..3), record whether team A won. With
one observation per (d, r) cell this is a two-way layout without replication,
so total variation splits into

    SS_total = SS_deal + SS_rotation + SS_residual

  deal      the cards themselves -- luck
  rotation  which seats the agents occupied; seat 2 draws the trump and leads,
            so a non-zero share here is structural advantage, not noise
  residual  everything else: how the specific matchup played out on that deal

The residual is where skill lives. If the deal term dominates, the game is
mostly luck per hand, which matters for how many hands a human evaluation
needs before it means anything.

    python variance_decomposition.py --deals 400
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
from dahaljeet.hand import Hand, team_of
from dahaljeet.view import make_view


def make_agent(spec, rng):
    if spec.startswith("PIMC:"):
        from dahaljeet.search import PIMCAgent
        return PIMCAgent(worlds=int(spec.split(":")[1]), rng=rng)
    if spec.startswith("ISMCTS:"):
        from dahaljeet.search import ISMCTSAgent
        return ISMCTSAgent(iterations=int(spec.split(":")[1]), rng=rng)
    return REGISTRY[spec](rng=rng)


def play_grid(a_spec, b_spec, n_deals, seed):
    """y[d][r] = 1 if A's team won deal d under rotation r."""
    base = random.Random(seed).randrange(1 << 30)
    y = []
    for d in range(n_deals):
        row = []
        for r in range(4):
            a = make_agent(a_spec, random.Random(1000 + d))
            b = make_agent(b_spec, random.Random(2000 + d))
            seats = {s: (a if ((s - r) % 4) % 2 == 0 else b) for s in range(4)}
            h = Hand(dealer=0, rng=random.Random(base + d))
            h.deal()
            while not h.is_over:
                h.play(seats[h.to_act].act(make_view(h, h.to_act)))
            row.append(1.0 if h.result().winning_team == team_of(r) else 0.0)
        y.append(row)
    return y


def decompose(y):
    n, k = len(y), len(y[0])
    grand = sum(sum(r) for r in y) / (n * k)
    deal_means = [sum(r) / k for r in y]
    rot_means = [sum(y[d][r] for d in range(n)) / n for r in range(k)]

    ss_total = sum((y[d][r] - grand) ** 2 for d in range(n) for r in range(k))
    ss_deal = k * sum((m - grand) ** 2 for m in deal_means)
    ss_rot = n * sum((m - grand) ** 2 for m in rot_means)
    ss_res = ss_total - ss_deal - ss_rot

    f = lambda x: round(x / ss_total, 4) if ss_total > 0 else 0.0
    return {
        "n_deals": n, "n_rotations": k, "grand_mean": round(grand, 4),
        "ss_total": round(ss_total, 3),
        "share_deal": f(ss_deal), "share_rotation": f(ss_rot),
        "share_residual": f(ss_res),
        "rotation_means": [round(m, 4) for m in rot_means],
        "reading": ("share_deal is the fraction of outcome variation "
                    "attributable to which cards were dealt -- luck. "
                    "share_rotation is structural seat advantage, which "
                    "duplicate dealing removes from agent comparisons. "
                    "share_residual is where the matchup actually played out."),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--deals", type=int, default=400)
    ap.add_argument("--pairs", default="TensThenTricks:Random,"
                                       "TensThenTricks:TenAware,"
                                       "PIMC:16:TensThenTricks")
    ap.add_argument("--seed", type=int, default=321)
    a = ap.parse_args()

    t0 = time.perf_counter()
    out = []
    for spec in a.pairs.split(","):
        parts = spec.rsplit(":", 1) if spec.count(":") == 1 else None
        if parts is None:
            # handles PIMC:16:Opponent
            i = spec.rindex(":")
            x, yb = spec[:i], spec[i + 1:]
        else:
            x, yb = parts
        y = play_grid(x, yb, a.deals, a.seed)
        d = decompose(y)
        d.update(agent_a=x, agent_b=yb)
        out.append(d)
        print(f"  {x} vs {yb}:  deal {d['share_deal']:.3f}  "
              f"seat {d['share_rotation']:.3f}  residual {d['share_residual']:.3f}"
              f"   (mean {d['grand_mean']:.4f})", flush=True)
        print(f"      per-rotation win rates: {d['rotation_means']}", flush=True)

    el = time.perf_counter() - t0
    json.dump({"env": runenv.snapshot(), "deals": a.deals, "results": out,
               "seconds": el},
              open("variance_decomposition.json", "w"), indent=2)
    print(f"\n{el:.0f}s -> variance_decomposition.json")
