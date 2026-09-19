"""Evaluation harness (Phase 5), used from Phase 1 onward.

Two methodological choices, both deliberate:

1. DUPLICATE DEALS. Evaluating on independently shuffled hands makes deal
   luck a large variance component: a run of good cards is indistinguishable
   from a better policy. Here every deal is played FOUR times, with the seat
   assignment rotated by 0/1/2/3, so both agents hold the identical cards from
   every seat and the comparison is paired. This is duplicate scoring, borrowed
   from competitive bridge, where it exists for exactly this reason.

2. SEAT BALANCE. Rotating through all four seats removes seat as a confound
   and makes the comparison paired, which is the larger benefit: the paired
   difference has a much smaller variance than two independent win rates.

   An earlier version of this note asserted that the trump drawer's seat is
   structurally advantaged. That was never measured, and when it finally was,
   it turned out to be false: over 4,800 self-play hands the trump-drawing
   team wins 0.5008 (95% CI [0.4808, 0.5208]), and the seat term in the
   variance decomposition is ~0.000. Rotation is still worth doing -- it costs
   nothing and forecloses the question -- but not for the reason originally
   given. See failure_analysis.json and variance_decomposition.json.

Statistics are bootstrapped over DEALS, not hands: the four rotations of one
deal are correlated, so the deal is the independent unit.
"""

from __future__ import annotations

import random
import statistics
import time
from dataclasses import dataclass, field

from .hand import Hand, team_of
from .view import make_view


@dataclass
class MatchResult:
    """Outcome of a duplicate match between two agents."""

    name_a: str
    name_b: str
    n_deals: int
    #: Per deal: how many of the 4 rotations agent A's team won (0..4).
    a_wins_per_deal: list[int] = field(default_factory=list)
    coats_a: int = 0
    coats_b: int = 0
    double_coats_a: int = 0
    double_coats_b: int = 0
    tens_a: int = 0
    tens_b: int = 0
    tricks_a: int = 0
    tricks_b: int = 0
    tiebreaks: int = 0
    decisions_a: int = 0
    decisions_b: int = 0
    time_a: float = 0.0
    time_b: float = 0.0

    @property
    def n_hands(self) -> int:
        return self.n_deals * 4

    @property
    def win_rate_a(self) -> float:
        return sum(self.a_wins_per_deal) / (self.n_deals * 4)

    @property
    def ms_per_decision_a(self) -> float:
        return 1000 * self.time_a / max(1, self.decisions_a)

    @property
    def ms_per_decision_b(self) -> float:
        return 1000 * self.time_b / max(1, self.decisions_b)

    def bootstrap_ci(self, iters: int = 2000, seed: int = 0,
                     alpha: float = 0.05) -> tuple[float, float]:
        """Percentile bootstrap CI for A's win rate, resampling DEALS."""
        if not self.a_wins_per_deal:
            return (0.0, 0.0)
        rng = random.Random(seed)
        n = len(self.a_wins_per_deal)
        data = self.a_wins_per_deal
        means = []
        for _ in range(iters):
            s = sum(data[rng.randrange(n)] for _ in range(n))
            means.append(s / (n * 4))
        means.sort()
        lo = means[int(alpha / 2 * iters)]
        hi = means[min(iters - 1, int((1 - alpha / 2) * iters))]
        return (lo, hi)

    def cohens_d(self) -> float:
        """Effect size on the per-deal win count (0..4), vs the null of 2."""
        if len(self.a_wins_per_deal) < 2:
            return 0.0
        sd = statistics.pstdev(self.a_wins_per_deal)
        if sd == 0:
            return 0.0
        return (statistics.fmean(self.a_wins_per_deal) - 2.0) / sd

    def summary(self) -> str:
        lo, hi = self.bootstrap_ci()
        return (
            f"{self.name_a:>13s} vs {self.name_b:<13s} "
            f"win {self.win_rate_a:6.3f} [{lo:.3f},{hi:.3f}]  "
            f"d={self.cohens_d():+5.2f}  "
            f"coat {self.coats_a:4d}/{self.coats_b:<4d} "
            f"tens {self.tens_a / self.n_hands:4.2f}  "
            f"{self.ms_per_decision_a:6.3f}ms"
        )


def play_deal(agent_by_seat, dealer: int, deal_seed: int, timing=None):
    """Play one hand with a fixed deal. `agent_by_seat` maps seat -> Agent."""
    h = Hand(dealer=dealer, rng=random.Random(deal_seed))
    h.deal()
    while not h.is_over:
        seat = h.to_act
        agent = agent_by_seat[seat]
        v = make_view(h, seat)
        if timing is None:
            h.play(agent.act(v))
        else:
            t0 = time.perf_counter()
            card = agent.act(v)
            dt = time.perf_counter() - t0
            timing[team_of(seat)][0] += dt
            timing[team_of(seat)][1] += 1
            h.play(card)
    return h.result()


def duplicate_match(agent_a, agent_b, n_deals: int = 500, seed: int = 0,
                    dealer: int = 0, timed: bool = True) -> MatchResult:
    """Agent A's team vs agent B's team over `n_deals`, each played 4 ways.

    Rotation r assigns agent A to seats {r, r+2} and B to {r+1, r+3}. Over
    r = 0..3 each agent holds every seat's cards exactly once.
    """
    res = MatchResult(agent_a.name, agent_b.name, n_deals)
    base = random.Random(seed).randrange(1 << 30)

    for i in range(n_deals):
        deal_seed = base + i
        a_wins = 0
        for r in range(4):
            by_seat = {}
            for s in range(4):
                by_seat[s] = agent_a if ((s - r) % 4) % 2 == 0 else agent_b
            # timing[t] = [seconds, decisions] for the team at index t
            timing = [[0.0, 0], [0.0, 0]] if timed else None
            result = play_deal(by_seat, dealer, deal_seed, timing)

            # Which team index is agent A on this rotation?
            a_team = team_of(r)
            b_team = 1 - a_team
            if result.winning_team == a_team:
                a_wins += 1
                if result.coat:
                    res.coats_a += 1
                if result.double_coat:
                    res.double_coats_a += 1
            else:
                if result.coat:
                    res.coats_b += 1
                if result.double_coat:
                    res.double_coats_b += 1

            res.tens_a += result.tens_by_team[a_team]
            res.tens_b += result.tens_by_team[b_team]
            res.tricks_a += result.tricks_by_team[a_team]
            res.tricks_b += result.tricks_by_team[b_team]
            if result.tens_by_team[0] == result.tens_by_team[1] == 2:
                res.tiebreaks += 1
            if timing is not None:
                res.time_a += timing[a_team][0]
                res.decisions_a += timing[a_team][1]
                res.time_b += timing[b_team][0]
                res.decisions_b += timing[b_team][1]
        res.a_wins_per_deal.append(a_wins)
    return res


def round_robin(agents, n_deals: int = 300, seed: int = 0, timed: bool = True):
    """Every agent against every other. Returns {(a,b): MatchResult}."""
    out = {}
    for i, a in enumerate(agents):
        for b in agents[i + 1:]:
            out[(a.name, b.name)] = duplicate_match(
                a, b, n_deals=n_deals, seed=seed, timed=timed)
    return out


def leaderboard(results, agents):
    """Mean win rate of each agent across all its matchups."""
    acc = {a.name: [] for a in agents}
    for (na, nb), r in results.items():
        acc[na].append(r.win_rate_a)
        acc[nb].append(1.0 - r.win_rate_a)
    rows = [(n, statistics.fmean(v)) for n, v in acc.items() if v]
    rows.sort(key=lambda t: -t[1])
    return rows
