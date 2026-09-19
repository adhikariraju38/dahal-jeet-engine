"""Multiple-observer ISMCTS with team-aware backup (gap B8).

The single-observer agent in dahaljeet/search.py has a modelling flaw that is
easy to miss and materially optimistic: it backs up the ROOT team's payoff at
every node and selects by maximising it at every node -- including the
opponents' decision nodes. Opponents are therefore searched as if they were
trying to help the searcher win.

This variant keeps a SEPARATE tree per player and backs up each player's OWN
team payoff, so every node is selected by the player who actually moves there,
maximising what that player actually wants. That is the multiple-observer
formulation [Cowling, Powley & Whitehouse 2012, UNVERIFIED citation], reduced
to what this game needs: all actions here are public, so the players'
observation histories coincide and the trees differ only in whose payoff is
backed up -- which is precisely the part the single-observer version gets wrong.

The original agent is left untouched. It is the baseline in the published
latency curve, and changing it would silently move results that are already
recorded; the two are compared head to head instead.

    python mo_ismcts.py --deals 300 --iterations 200
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time

sys.path.insert(0, ".")

from dahaljeet.agents import Agent, _safe_discard
from dahaljeet.determinize import determinize, hand_from_view
from dahaljeet.hand import team_of
from dahaljeet.search import _Rollout, _payoff


class MOISMCTSAgent(Agent):
    """ISMCTS with one tree per player and team-aware backup."""

    name = "MO-ISMCTS"

    def __init__(self, iterations: int = 400, c: float = 0.7, rng=None):
        super().__init__(rng)
        self.iterations = iterations
        self.c = c
        self.name = f"MOISMCTS{iterations}"
        self._roll = _Rollout(self.rng)

    def act(self, v):
        if len(v.legal) == 1:
            return v.legal[0]

        # one tree per SEAT; nodes keyed by the public action sequence
        stats = [dict() for _ in range(4)]
        avail = [dict() for _ in range(4)]

        for _ in range(self.iterations):
            d = determinize(v, self.rng)
            if d is None:
                continue
            h = hand_from_view(v, d)
            path: tuple = ()
            visited = []                      # (seat, path_before, move)

            while not h.is_over:
                seat = h.to_act
                legal = h.legal_moves(seat)
                node = stats[seat].setdefault(path, {})
                av = avail[seat].setdefault(path, {})
                for m in legal:
                    av[m] = av.get(m, 0) + 1
                untried = [m for m in legal if m not in node]
                if untried:
                    move = self.rng.choice(untried)
                    node[move] = [0, 0.0]
                    visited.append((seat, path, move))
                    h.play(move)
                    path = path + (move,)
                    break
                logN = math.log(sum(av[m] for m in legal) + 1)
                move = max(legal, key=lambda m: (
                    node[m][1] / node[m][0]
                    + self.c * math.sqrt(logN / node[m][0])))
                visited.append((seat, path, move))
                h.play(move)
                path = path + (move,)

            result = self._roll.finish(h)
            # THE CORRECTION: each visited node is credited with the payoff of
            # the team that moved there, not the root's payoff.
            payoff = (_payoff(result, 0), _payoff(result, 1))
            for seat, pre, mv in visited:
                node = stats[seat].get(pre)
                if node and mv in node:
                    node[mv][0] += 1
                    node[mv][1] += payoff[team_of(seat)]

        root = stats[v.seat].get((), {})
        if not root:
            return _safe_discard(v)
        return max(root, key=lambda m: root[m][0])


if __name__ == "__main__":
    from dahaljeet.agents import REGISTRY
    from dahaljeet.search import ISMCTSAgent
    from dahaljeet.tournament import duplicate_match
    import runenv

    ap = argparse.ArgumentParser()
    ap.add_argument("--deals", type=int, default=300)
    ap.add_argument("--iterations", default="50,100,200")
    ap.add_argument("--opponent", default="TensThenTricks")
    a = ap.parse_args()

    t0 = time.perf_counter()
    rows, h2h = [], []
    for it in [int(x) for x in a.iterations.split(",")]:
        for label, mk in (("SO-ISMCTS", lambda i=it: ISMCTSAgent(iterations=i, rng=random.Random(3))),
                          ("MO-ISMCTS", lambda i=it: MOISMCTSAgent(iterations=i, rng=random.Random(3)))):
            opp = REGISTRY[a.opponent](rng=random.Random(4))
            r = duplicate_match(mk(), opp, n_deals=a.deals, seed=909, timed=True)
            lo, hi = r.bootstrap_ci()
            rows.append({"variant": label, "iterations": it,
                         "win": round(r.win_rate_a, 4),
                         "ci": [round(lo, 4), round(hi, 4)],
                         "ms_per_decision": round(r.ms_per_decision_a, 3),
                         "a_wins_per_deal": list(r.a_wins_per_deal)})
            print(f"  {label} {it:4d} iters  win {r.win_rate_a:.4f} "
                  f"[{lo:.4f},{hi:.4f}]  {r.ms_per_decision_a:7.2f} ms", flush=True)
        # head to head on the identical deals
        r = duplicate_match(MOISMCTSAgent(iterations=it, rng=random.Random(3)),
                            ISMCTSAgent(iterations=it, rng=random.Random(5)),
                            n_deals=a.deals, seed=910, timed=False)
        lo, hi = r.bootstrap_ci()
        v = "MO WINS" if lo > 0.5 else ("MO loses" if hi < 0.5 else "tie")
        h2h.append({"iterations": it, "mo_win_vs_so": round(r.win_rate_a, 4),
                    "ci": [round(lo, 4), round(hi, 4)], "verdict": v})
        print(f"    head-to-head MO vs SO at {it}: {r.win_rate_a:.4f} "
              f"[{lo:.4f},{hi:.4f}]  {v}\n", flush=True)

    el = time.perf_counter() - t0
    json.dump({"env": runenv.snapshot(), "deals": a.deals,
               "opponent": a.opponent, "arms": rows, "head_to_head": h2h,
               "seconds": el,
               "difference": ("SO-ISMCTS backs up the ROOT team's payoff at every "
                              "node, so opponents are searched as if helping the "
                              "searcher. MO-ISMCTS keeps a tree per seat and backs "
                              "up each seat's own team payoff.")},
              open("mo_ismcts.json", "w"), indent=2)
    print(f"{el:.0f}s -> mo_ismcts.json")
