"""A best-response oracle strong enough to bound exploitability.

Two earlier attempts failed, and both failures were informative:

  1. exploit.py trained a DQN exploiter. DQN caps around 0.34 against
     TensThenTricks while a plain heuristic reaches 0.45, so the "best
     response" lost to an off-the-shelf opponent and bounded nothing.
  2. exploitability2.py used the generalisation matrix's PPO agents, which are
     genuinely trained against one opponent. Better, but the strongest GENERIC
     agent is ISMCTS250, which beats every learner we have. The comparison
     therefore measured "is our best learner stronger than our best search
     agent", not "is the target exploitable".

The fix is to make the best response a SEARCH agent that knows the target's
policy exactly. During simulation the opponent seats are played by the target
itself rather than by tree selection, so the search plans against the true
opponent instead of an assumed one. That is a best response in the proper
sense: the strongest reply available given full knowledge of the opponent.

    python best_response_search.py --targets TensThenTricks,GreedyTricks --deals 300

exploitability(T) = win(oracle-opponent search vs T) - win(same search, generic, vs T)

Both arms are the SAME search at the SAME budget, so the difference isolates
what knowing the opponent is worth, with agent strength held constant. That is
the comparison the earlier versions failed to make.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time

sys.path.insert(0, ".")
import runenv

from dahaljeet.agents import Agent, REGISTRY, _safe_discard
from dahaljeet.determinize import determinize, hand_from_view
from dahaljeet.hand import team_of
from dahaljeet.search import ISMCTSAgent, _Rollout, _payoff
from dahaljeet.tournament import duplicate_match
from dahaljeet.view import make_view


class OracleOpponentISMCTS(Agent):
    """ISMCTS whose simulated opponents play the target's actual policy."""

    def __init__(self, target_name, iterations=200, c=0.7, rng=None):
        super().__init__(rng)
        self.iterations = iterations
        self.c = c
        self.target_name = target_name
        self.name = f"BR-ISMCTS{iterations}[{target_name}]"
        self._roll = _Rollout(self.rng)
        self._model = REGISTRY[target_name](rng=random.Random(4242))

    def act(self, v):
        if len(v.legal) == 1:
            return v.legal[0]
        me = team_of(v.seat)
        stats: dict = {}
        avail: dict = {}

        for _ in range(self.iterations):
            d = determinize(v, self.rng)
            if d is None:
                continue
            h = hand_from_view(v, d)
            path: tuple = ()
            while not h.is_over:
                seat = h.to_act
                if team_of(seat) != me:
                    # THE POINT: the opponent is simulated by its real policy,
                    # not by tree selection maximising our payoff.
                    h.play(self._model.act(make_view(h, seat)))
                    continue
                legal = h.legal_moves(seat)
                node = stats.setdefault(path, {})
                av = avail.setdefault(path, {})
                for m in legal:
                    av[m] = av.get(m, 0) + 1
                untried = [m for m in legal if m not in node]
                if untried:
                    move = self.rng.choice(untried)
                    node[move] = [0, 0.0]
                    h.play(move)
                    path = path + (move,)
                    break
                logN = math.log(sum(av[m] for m in legal) + 1)
                move = max(legal, key=lambda m: (
                    node[m][1] / node[m][0]
                    + self.c * math.sqrt(logN / node[m][0])))
                h.play(move)
                path = path + (move,)
            reward = _payoff(self._roll.finish(h), me)
            for i in range(len(path)):
                pre, mv = path[:i], path[i]
                if pre in stats and mv in stats[pre]:
                    stats[pre][mv][0] += 1
                    stats[pre][mv][1] += reward

        root = stats.get((), {})
        if not root:
            return _safe_discard(v)
        return max(root, key=lambda m: root[m][0])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets",
                    default="Random,GreedyTricks,TenAware,Adaptive,TensThenTricks")
    ap.add_argument("--iterations", type=int, default=200)
    ap.add_argument("--deals", type=int, default=300)
    ap.add_argument("--out", default="exploitability_search.json",
                    help="output file; per-target files let the five targets "
                         "run in parallel and make a crash cost one target, "
                         "not all five")
    a = ap.parse_args()

    t0 = time.perf_counter()
    rows = []
    for T in [x.strip() for x in a.targets.split(",")]:
        opp = REGISTRY[T](rng=random.Random(4))
        br = OracleOpponentISMCTS(T, iterations=a.iterations, rng=random.Random(3))
        r1 = duplicate_match(br, opp, n_deals=a.deals, seed=1717, timed=False)
        lo1, hi1 = r1.bootstrap_ci()

        opp = REGISTRY[T](rng=random.Random(4))
        gen = ISMCTSAgent(iterations=a.iterations, rng=random.Random(3))
        r2 = duplicate_match(gen, opp, n_deals=a.deals, seed=1717, timed=False)
        lo2, hi2 = r2.bootstrap_ci()

        # paired on identical deals
        va, vb = r1.a_wins_per_deal, r2.a_wins_per_deal
        n = min(len(va), len(vb))
        diffs = [(va[i] - vb[i]) / 4.0 for i in range(n)]
        mean = sum(diffs) / n
        rng = random.Random(0)
        hits = sum(1 for _ in range(20000)
                   if abs(sum(x if rng.random() < 0.5 else -x
                              for x in diffs) / n) >= abs(mean) - 1e-12)
        p = (hits + 1) / 20001

        rows.append({"target": T, "iterations": a.iterations,
                     "best_response_win": round(r1.win_rate_a, 4),
                     "best_response_ci": [round(lo1, 4), round(hi1, 4)],
                     "generic_search_win": round(r2.win_rate_a, 4),
                     "generic_search_ci": [round(lo2, 4), round(hi2, 4)],
                     "exploitability": round(r1.win_rate_a - r2.win_rate_a, 4),
                     "paired_p": round(p, 5)})
        print(f"  {T:16s} BR {r1.win_rate_a:.4f} [{lo1:.4f},{hi1:.4f}]   "
              f"generic {r2.win_rate_a:.4f} [{lo2:.4f},{hi2:.4f}]   "
              f"exploitability {r1.win_rate_a-r2.win_rate_a:+.4f}  p={p:.4f}",
              flush=True)

    el = time.perf_counter() - t0
    json.dump({"env": runenv.snapshot(), "deals": a.deals,
               "iterations": a.iterations, "targets": rows, "seconds": el,
               "definition": ("exploitability(T) = win(search that simulates T's "
                              "true policy for the opponent seats, vs T) - "
                              "win(the same search with no opponent model, vs T). "
                              "Identical algorithm and budget in both arms, so "
                              "the difference isolates the value of knowing the "
                              "opponent. Paired on identical deals.")},
              open(a.out, "w"), indent=2)
    print(f"\n{el:.0f}s -> {a.out}")
