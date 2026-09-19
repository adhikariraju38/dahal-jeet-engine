"""Search-based agents (Phase 3): PIMC and ISMCTS.

Both work from a PlayerView only. They sample hypothetical worlds with
determinize(), which respects the voids inferred from P2, so every rollout runs
in a world consistent with the observed play.

The determinization budget is an explicit parameter rather than a constant,
because search strength is bought with time and a single win rate hides the
price. eval_final.py sweeps it so strength can be reported AGAINST measured
latency instead of asserted at one undisclosed setting.
"""

from __future__ import annotations

import math
import random

from .agents import Agent, TensThenTricks, _safe_discard
from .determinize import determinize, hand_from_view
from .hand import team_of
from .view import make_view


def _payoff(result, team: int) -> float:
    """Hand value to `team`. Win is what counts; a coat is worth a nudge more
    because it also swings the deal (R3) and the tally (G3)."""
    if result.winning_team != team:
        return 0.0
    if result.double_coat:
        return 1.15
    if result.coat:
        return 1.05
    return 1.0


class _Rollout:
    """Shared rollout policy: the strongest heuristic we have."""

    def __init__(self, rng):
        self.pol = TensThenTricks(rng=rng)

    def finish(self, h) -> object:
        while not h.is_over:
            h.play(self.pol.act(make_view(h, h.to_act)))
        return h.result()


class PIMCAgent(Agent):
    """Perfect-Information Monte Carlo.

    For each legal move: sample `worlds` determinizations, play the move, roll
    the rest out with the heuristic, average the payoff. Take the best mean.

    Cheap and surprisingly strong, but it suffers the known PIMC pathology --
    it assumes every future decision is made with perfect information, so it
    cannot value information-gathering. ISMCTS below is the comparison.
    """

    name = "PIMC"

    def __init__(self, worlds: int = 20, rng=None):
        super().__init__(rng)
        self.worlds = worlds
        self.name = f"PIMC{worlds}"
        self._roll = _Rollout(self.rng)

    def act(self, v):
        if len(v.legal) == 1:
            return v.legal[0]
        team = team_of(v.seat)
        totals = {c: 0.0 for c in v.legal}
        counts = {c: 0 for c in v.legal}

        for _ in range(self.worlds):
            d = determinize(v, self.rng)
            if d is None:
                continue
            for card in v.legal:
                h = hand_from_view(v, d)
                h.play(card)
                totals[card] += _payoff(self._roll.finish(h), team)
                counts[card] += 1

        if not any(counts.values()):
            return _safe_discard(v)
        return max(v.legal,
                   key=lambda c: totals[c] / counts[c] if counts[c] else -1)


class ISMCTSAgent(Agent):
    """Information Set Monte Carlo Tree Search (single-observer).

    One determinization per iteration; the tree is keyed by action sequence
    from the root, so nodes correspond to information sets rather than states.
    Selection is UCB1 over the moves that are legal in the current world --
    which is what makes it an information-set search rather than plain MCTS.
    """

    name = "ISMCTS"

    def __init__(self, iterations: int = 400, c: float = 0.7, rng=None):
        super().__init__(rng)
        self.iterations = iterations
        self.c = c
        self.name = f"ISMCTS{iterations}"
        self._roll = _Rollout(self.rng)

    def act(self, v):
        if len(v.legal) == 1:
            return v.legal[0]
        team = team_of(v.seat)
        # stats[path][card] = [visits, total_reward]; availability counted
        # separately so UCB1 uses "times this move was legal", per Cowling.
        stats: dict[tuple, dict[int, list]] = {}
        avail: dict[tuple, dict[int, int]] = {}

        for _ in range(self.iterations):
            d = determinize(v, self.rng)
            if d is None:
                continue
            h = hand_from_view(v, d)
            path: tuple = ()
            # ---- selection / expansion
            while not h.is_over:
                legal = h.legal_moves(h.to_act)
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
            # ---- rollout + backprop
            reward = _payoff(self._roll.finish(h), team)
            for i in range(len(path)):
                pre = path[:i]
                mv = path[i]
                if pre in stats and mv in stats[pre]:
                    stats[pre][mv][0] += 1
                    stats[pre][mv][1] += reward

        root = stats.get((), {})
        if not root:
            return _safe_discard(v)
        return max(root, key=lambda m: root[m][0])   # most-visited move


SEARCH_AGENTS = [PIMCAgent, ISMCTSAgent]
