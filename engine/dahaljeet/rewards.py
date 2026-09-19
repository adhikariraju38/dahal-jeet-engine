"""Reward schemes for Phase 4 -- the ablation, not an afterthought.

Dahal Jeet's reward is sparse and peaked: 48 of 52 cards are worthless (W1),
and the outcome is decided at the end of 13 tricks. A whole hand therefore
yields at most four scoring events, which puts most of the burden on credit
assignment. Rather than guess a shaping function and report whatever it gives,
we treat the choice as the experiment and compare four schemes, including one
deliberately misaligned control.

Potential-based shaping (Ng, Harada & Russell 1999) is policy-invariant: the
optimal policy is unchanged, only credit assignment gets easier. The `ten` and
`trick` schemes are NOT policy-invariant and can distort the objective -- which
is the point of including them.
"""

from __future__ import annotations

from .hand import team_of


def terminal(prev, cur, team, done, result):
    """Honest and maximally sparse: +1 win, -1 loss, nothing in between."""
    if not done:
        return 0.0
    return 1.0 if result.winning_team == team else -1.0


def ten_shaped(prev, cur, team, done, result):
    """+0.25 per ten captured, -0.25 per ten conceded, plus the terminal signal.

    Aligned with the real objective (W1) but NOT policy-invariant: it can
    over-value tens relative to the W5 tiebreak, which decides ~34% of hands.
    """
    r = 0.0
    other = 1 - team
    r += 0.25 * (cur.tens[team] - prev.tens[team])
    r -= 0.25 * (cur.tens[other] - prev.tens[other])
    if done:
        r += 1.0 if result.winning_team == team else -1.0
    return r


def potential(prev, cur, team, done, result):
    """Potential-based shaping: F = gamma*Phi(s') - Phi(s), gamma = 1.

    Phi blends the two objectives in the proportion the game actually uses:
    tens decide ~66% of hands, the trick tiebreak ~34%.
    """
    def phi(st):
        other = 1 - team
        tens = (st.tens[team] - st.tens[other]) / 4.0
        tricks = (st.tricks[team] - st.tricks[other]) / 13.0
        return 0.66 * tens + 0.34 * tricks
    r = phi(cur) - phi(prev)
    if done:
        r += 1.0 if result.winning_team == team else -1.0
    return r


def trick_shaped(prev, cur, team, done, result):
    """CONTROL -- deliberately misaligned. Rewards tricks, the obvious signal.

    Trick count is the wrong objective: 13.6% of hands are won by the side
    taking FEWER tricks. If the reward ablation is meaningful, this should
    underperform `potential`. A negative result here is a publishable finding.
    """
    r = 0.05 * (cur.tricks[team] - prev.tricks[team])
    if done:
        r += 1.0 if result.winning_team == team else -1.0
    return r


SCHEMES = {
    "terminal": terminal,
    "ten_shaped": ten_shaped,
    "potential": potential,
    "trick_shaped": trick_shaped,
}


class Snapshot:
    """Minimal state summary the reward functions need."""

    __slots__ = ("tens", "tricks")

    def __init__(self, hand):
        self.tens = tuple(hand.tens_by_team)
        self.tricks = tuple(hand.tricks_by_team)
