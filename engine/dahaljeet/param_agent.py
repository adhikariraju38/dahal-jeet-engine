"""A parameterised heuristic, for offline metaheuristic tuning.

One agent with a numeric genome, tuned by hill-climbing against a fixed
opponent panel using duplicate-deal evaluation.

The genome exists because hand-tuning failed. Measured over 10,000 hands,
pure ten-hunting won 0.646 of tens-decided hands but 0.192 of tiebreaks, and
pure trick-taking did the reverse; both sat at ~0.50 head to head. The balance
point is somewhere between, and it is easier to search for than to guess.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, asdict

from .agents import Agent, _lowest, _safe_discard, _cheapest_winner, _non_tens
from .cards import suit_of, rank_of, is_ten


@dataclass
class Genome:
    """Tunable knobs. Ranges are inclusive; all are integers or bools."""

    #: Take a ten-less trick if the cheapest winner's rank is <= this (0..12).
    #: Low value = the first author's "8 of hearts" discipline. High = greedy.
    free_trick_max_rank: int = 4
    #: Spend a trump on a trick with no ten? 0 = never, 1 = only late, 2 = always.
    trump_on_empty_trick: int = 0
    #: Don't lead trump before this trick index (0..13).
    lead_trump_after: int = 13
    #: Minimum seats-already-played required to dump a ten to a winning partner.
    ten_dump_min_pos: int = 3
    #: After this trick index, maximise tricks regardless of tens (0..13).
    late_switch: int = 13
    #: When on lead, prefer the longest suit rather than the lowest card.
    lead_from_longest: int = 0
    #: Contest a ten-less trick when the partner is losing and we play last.
    contest_last_seat: int = 1

    def vector(self):
        return list(asdict(self).values())

    @staticmethod
    def bounds():
        return {
            "free_trick_max_rank": (0, 12),
            "trump_on_empty_trick": (0, 2),
            "lead_trump_after": (0, 13),
            "ten_dump_min_pos": (0, 3),
            "late_switch": (0, 13),
            "lead_from_longest": (0, 1),
            "contest_last_seat": (0, 1),
        }


class ParamAgent(Agent):
    name = "Param"

    def __init__(self, genome: Genome | None = None,
                 rng: random.Random | None = None, name: str | None = None):
        super().__init__(rng)
        self.g = genome or Genome()
        if name:
            self.name = name

    def clone(self, genome):
        return ParamAgent(genome, self.rng, self.name)

    # -------------------------------------------------------------- policy

    def act(self, v):
        g = self.g
        pos = len(v.trick)
        last = pos == 3
        late = v.tricks_played >= g.late_switch
        stake = (not v.on_lead) and v.trick_has_ten()
        tens_live = bool(v.tens_outstanding)

        # ---- on lead -------------------------------------------------
        if v.on_lead:
            pool = [c for c in v.legal if not is_ten(c)] or list(v.legal)
            if v.tricks_played < g.lead_trump_after:
                plain = [c for c in pool if suit_of(c) != v.trump_suit]
                pool = plain or pool
            if g.lead_from_longest:
                counts = {}
                for c in pool:
                    counts[suit_of(c)] = counts.get(suit_of(c), 0) + 1
                best_suit = max(counts, key=lambda s: counts[s])
                pool = [c for c in pool if suit_of(c) == best_suit]
            return _lowest(pool)

        # ---- partner already winning ---------------------------------
        if v.partner_is_winning():
            tens = [c for c in v.legal if is_ten(c)]
            if tens and pos >= g.ten_dump_min_pos:
                return tens[0]
            return _safe_discard(v)

        win = _cheapest_winner(v)

        # ---- a ten is at stake, or the tens race is over -------------
        if stake or late or not tens_live:
            if win is not None:
                return win
            return _safe_discard(v)

        # ---- ten-less trick: the first author's discipline, parameterised
        if win is not None:
            is_trump = suit_of(win) == v.trump_suit
            allow_trump = (g.trump_on_empty_trick == 2 or
                           (g.trump_on_empty_trick == 1 and late))
            if is_trump and not allow_trump:
                return _safe_discard(v)
            if not is_trump:
                if rank_of(win) <= g.free_trick_max_rank:
                    return win
                if last and g.contest_last_seat:
                    return win
        return _safe_discard(v)


def random_genome(rng: random.Random) -> Genome:
    lo_hi = Genome.bounds()
    return Genome(**{k: rng.randint(*b) for k, b in lo_hi.items()})


def mutate(g: Genome, rng: random.Random, rate: float = 0.34) -> Genome:
    lo_hi = Genome.bounds()
    d = asdict(g)
    for k, (lo, hi) in lo_hi.items():
        if rng.random() < rate:
            step = rng.choice([-2, -1, 1, 2])
            d[k] = max(lo, min(hi, d[k] + step))
    return Genome(**d)
