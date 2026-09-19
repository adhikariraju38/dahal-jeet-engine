"""Decompose wins into tens-decided vs tiebreak-decided (artifact for Figure 2).

The central structural finding -- that the game's two objectives pull against
each other and roughly cancel -- was previously recorded only in prose, and
Figure 2 had the numbers hardcoded. That breaks the rule that every number in
the paper traces to an artifact on disk, so this recomputes it and emits JSON.

    python3 decompose.py --hands 3000
"""
from __future__ import annotations
import argparse, json, random, sys, time
sys.path.insert(0, ".")
import runenv
from dahaljeet.agents import REGISTRY
from dahaljeet.hand import Hand, team_of
from dahaljeet.view import make_view


def decompose(name_a, name_b, n_deals, seed):
    """Duplicate-deal match, split by HOW each hand was decided."""
    A = REGISTRY[name_a](rng=random.Random(1))
    B = REGISTRY[name_b](rng=random.Random(2))
    base = random.Random(seed).randrange(1 << 30)
    s = {"tens_A": 0, "tens_B": 0, "tie_A": 0, "tie_B": 0,
         "coatA": 0, "coatB": 0, "tensA": 0, "tricksA": 0, "n": 0}
    for i in range(n_deals):
        for r in range(4):
            by = {q: (A if ((q - r) % 4) % 2 == 0 else B) for q in range(4)}
            h = Hand(dealer=0, rng=random.Random(base + i))
            h.deal()
            while not h.is_over:
                q = h.to_act
                h.play(by[q].act(make_view(h, q)))
            res = h.result()
            a_team = team_of(r)
            awon = res.winning_team == a_team
            tie = res.tens_by_team[0] == res.tens_by_team[1] == 2
            if tie:
                s["tie_A" if awon else "tie_B"] += 1
            else:
                s["tens_A" if awon else "tens_B"] += 1
            if res.coat:
                s["coatA" if awon else "coatB"] += 1
            s["tensA"] += res.tens_by_team[a_team]
            s["tricksA"] += res.tricks_by_team[a_team]
            s["n"] += 1
    n = s["n"]
    by_tens = s["tens_A"] + s["tens_B"]
    by_tie = s["tie_A"] + s["tie_B"]
    return {
        "agent_a": name_a, "agent_b": name_b, "hands": n,
        "overall_win_a": round((s["tens_A"] + s["tie_A"]) / n, 4),
        "share_decided_by_tens": round(by_tens / n, 4),
        "share_decided_by_tiebreak": round(by_tie / n, 4),
        "win_a_when_tens_decide": round(s["tens_A"] / by_tens, 4) if by_tens else None,
        "win_a_when_tiebreak_decides": round(s["tie_A"] / by_tie, 4) if by_tie else None,
        "mean_tens_a": round(s["tensA"] / n, 4),
        "mean_tricks_a": round(s["tricksA"] / n, 4),
        "coats_a": s["coatA"], "coats_b": s["coatB"],
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--deals", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args()
    t0 = time.perf_counter()
    pairs = [("TenAware", "GreedyTricks"), ("TensThenTricks", "GreedyTricks"),
             ("TenAware", "Random"), ("GreedyTricks", "Random")]
    out = []
    for x, y in pairs:
        d = decompose(x, y, a.deals, a.seed)
        out.append(d)
        print(f"  {x:15s} vs {y:14s} overall {d['overall_win_a']:.4f} | "
              f"tens-decided {d['share_decided_by_tens']:.1%} -> "
              f"{d['win_a_when_tens_decide']:.4f} | "
              f"tiebreak {d['share_decided_by_tiebreak']:.1%} -> "
              f"{d['win_a_when_tiebreak_decides']:.4f}", flush=True)
    json.dump({"env": runenv.snapshot(), "deals_per_pair": a.deals,
               "seed": a.seed, "decompositions": out,
               "seconds": round(time.perf_counter() - t0, 1)},
              open("decomposition.json", "w"), indent=2)
    print("saved decomposition.json")
