"""An open-ended session of Dahal Jeet: dealer rotation and the tally.

The game has NO terminal state (rule G2) — play continues indefinitely and
what accumulates is a running tally (G3).  So `Session` deliberately does not
know when to stop.  Any episode boundary is imposed by the caller and is a
MODELLING ARTIFACT, not a rule of the game.  Keep it that way: the boundary
belongs in the experiment config so the paper can be honest about it.

Seats are 0..3 here and 1..4 in the rulebook (code seat i == rulebook i+1).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .hand import Hand, HandResult, PARTNER, team_of


def next_dealer(dealer: int, losing_team: int, coat: bool,
                double_coat: bool) -> int:
    """Who deals the next hand.  (R1-R4)

    R1  The dealer always comes from the losing team.
    R2  Default: the first seat, counting from the current dealer in turn
        order 0->1->2->3, that belongs to the losing team.
    R3  Coat skip: if the winners took all four tens, skip the R2 seat and
        give the deal to the OTHER member of the losing team.
    R4  A double coat cancels the R3 skip — revert to the R2 default.
    """
    default = -1
    for i in range(4):
        seat = (dealer + i) % 4
        if team_of(seat) == losing_team:
            default = seat
            break
    assert default >= 0

    if coat and not double_coat:
        return PARTNER[default]        # R3
    return default                     # R2, or R4 cancelling R3


@dataclass
class Tally:
    """The running score teams actually keep.  (G3)

    Not a points system — it exists so the winning side can keep reminding
    the losing side that they have not won.  (G4: purely social.)
    """

    hands_won: list[int] = field(default_factory=lambda: [0, 0])
    coats: list[int] = field(default_factory=lambda: [0, 0])
    double_coats: list[int] = field(default_factory=lambda: [0, 0])
    #: How many hands each team lost, i.e. how long they were stuck dealing.
    hands_lost: list[int] = field(default_factory=lambda: [0, 0])

    def record(self, r: HandResult) -> None:
        w, l = r.winning_team, r.losing_team
        self.hands_won[w] += 1
        self.hands_lost[l] += 1
        if r.coat:
            self.coats[w] += 1
        if r.double_coat:
            self.double_coats[w] += 1

    def describe(self) -> str:
        return (
            f"team0 {self.hands_won[0]}W/{self.coats[0]}C/{self.double_coats[0]}DC   "
            f"team1 {self.hands_won[1]}W/{self.coats[1]}C/{self.double_coats[1]}DC"
        )


class Session:
    """A sequence of hands with dealer rotation and a tally.

    `first_dealer` may be any seat — rule R0 places no constraint on the
    opening deal.
    """

    def __init__(self, first_dealer: int = 0, rng: random.Random | None = None):
        self.dealer = first_dealer                 # R0
        self.rng = rng if rng is not None else random.Random()
        self.tally = Tally()
        self.history: list[HandResult] = []

    def play_hand(self, policy) -> HandResult:
        """Play one hand.  `policy(hand) -> card` chooses for whoever is to act."""
        hand = Hand(dealer=self.dealer, rng=self.rng)
        hand.deal()
        while not hand.is_over:
            hand.play(policy(hand))
        result = hand.result()

        self.tally.record(result)
        self.history.append(result)
        self.dealer = next_dealer(
            self.dealer, result.losing_team, result.coat, result.double_coat
        )
        return result

    def play_hands(self, n: int, policy) -> list[HandResult]:
        """Play `n` hands.

        NOTE: `n` is an imposed episode boundary, not a rule of the game
        (G2).  Report it as such.
        """
        return [self.play_hand(policy) for _ in range(n)]
