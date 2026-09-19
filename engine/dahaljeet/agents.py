"""Rule-based agents (Phase 2).

Each agent isolates one idea so the tournament can attribute credit. The
strategic core, from the first author:

    "try to win others' 10 and defend yours 10 ... if play card is 8 of heart
     no need to win over it by next player, he can easily play 2 of heart"

Which is to say: a trick with no ten in it is worth nothing (W1). Spending a
high card or a trump on it is pure waste. The whole game is about the four
tens, plus enough tricks to survive the W5 tiebreak -- which fires in ~34% of
hands, so trick count is a real secondary objective, not a throwaway.
"""

from __future__ import annotations

import random

from .cards import suit_of, rank_of, is_ten, TEN_RANK
from .view import PlayerView


class Agent:
    """Base class. `act` sees only a PlayerView -- never a Hand."""

    name = "agent"

    def __init__(self, rng: random.Random | None = None):
        self.rng = rng if rng is not None else random.Random()

    def act(self, v: PlayerView) -> int:
        raise NotImplementedError

    def __repr__(self) -> str:
        return self.name


# --------------------------------------------------------------- helpers

def _lowest(cards):
    """Lowest by rank, ignoring suit."""
    return min(cards, key=rank_of)


def _highest(cards):
    return max(cards, key=rank_of)


def _non_tens(cards):
    return [c for c in cards if not is_ten(c)]


def _cheapest_winner(v: PlayerView):
    """Weakest legal card that still takes the trick, or None."""
    wins = v.winning_moves()
    if not wins:
        return None
    # Prefer a non-trump win; among equals take the lowest rank. Spending a
    # trump on a trick a plain card would win is the same waste as spending an
    # ace on a trick with no ten in it.
    plain = [c for c in wins if suit_of(c) != v.trump_suit]
    pool = plain if plain else wins
    return min(pool, key=rank_of)


def _safe_discard(v: PlayerView):
    """Lowest card that is not a ten. Falls back to a ten only if forced."""
    nt = _non_tens(v.legal)
    return _lowest(nt) if nt else _lowest(v.legal)


# ---------------------------------------------------------------- agents

class RandomAgent(Agent):
    """Uniform over legal moves. The floor."""

    name = "Random"

    def act(self, v):
        return self.rng.choice(v.legal)


class LowestAgent(Agent):
    """Always the lowest legal card. Deterministic regression baseline."""

    name = "Lowest"

    def act(self, v):
        return _lowest(v.legal)


class GreedyTricks(Agent):
    """Wins every trick it can, regardless of whether a ten is at stake.

    A deliberate CONTROL, not a serious strategy. It optimises trick count --
    the obvious signal and the wrong objective. If the first author's principle is
    right, this must lose to TenAware. That is the experiment.
    """

    name = "GreedyTricks"

    def act(self, v):
        w = _cheapest_winner(v)
        return w if w is not None else _safe_discard(v)


class TenHunter(Agent):
    """Contests tricks that hold a ten; otherwise plays low.

    Isolates the offensive half: "win others' 10".
    """

    name = "TenHunter"

    def act(self, v):
        if v.on_lead:
            return _safe_discard(v)
        if v.trick_has_ten():
            w = _cheapest_winner(v)
            if w is not None:
                return w
        return _safe_discard(v)


class TenDefender(Agent):
    """Never spends a ten into a trick its team is not taking.

    Isolates the defensive half: "defend yours 10". A ten is only safe to play
    when the partner is already winning and this seat plays last, so the
    outcome cannot change.
    """

    name = "TenDefender"

    def act(self, v):
        last = len(v.trick) == 3
        tens_in_hand = [c for c in v.legal if is_ten(c)]
        if tens_in_hand and last and v.partner_is_winning():
            return tens_in_hand[0]        # safe dump: partner banks the ten
        return _safe_discard(v)


class PartnerAware(Agent):
    """Never overtakes its own partner; ducks when the partner is winning.

    Isolates the partnership signal (S1). If this beats TenHunter, then
    cooperating with the partner matters more than grabbing tens individually
    -- which is the question fixed partnerships make it possible to ask.
    """

    name = "PartnerAware"

    def act(self, v):
        if v.on_lead:
            return _safe_discard(v)
        if v.partner_is_winning():
            return _safe_discard(v)       # let the partner have it
        w = _cheapest_winner(v)
        return w if w is not None else _safe_discard(v)


class TenAware(Agent):
    """The first author's strategy, played properly.

    Combines both halves plus positional awareness:
      - a trick with no ten is not worth a high card (the 8-of-hearts rule)
      - contest a trick that holds a ten, cheaply
      - duck when the partner already has it
      - dump a ten only when the team is certain to take the trick
      - keep trumps for tricks that actually score
    """

    name = "TenAware"

    def act(self, v):
        last = len(v.trick) == 3
        partner_winning = v.partner_is_winning()
        tens_in_hand = [c for c in v.legal if is_ten(c)]

        # ---- leading ------------------------------------------------
        if v.on_lead:
            # Never lead a ten: three players get a shot at it.
            # Don't lead trump either -- trumps are for capturing tens.
            plain = [c for c in v.legal
                     if not is_ten(c) and suit_of(c) != v.trump_suit]
            if plain:
                return _lowest(plain)
            return _safe_discard(v)

        stake = v.trick_has_ten()

        # ---- partner already winning --------------------------------
        if partner_winning:
            # Certain outcome only when we play last; then a ten is safe to
            # bank. Otherwise keep it back.
            if last and tens_in_hand:
                return tens_in_hand[0]
            return _safe_discard(v)

        # ---- opponents winning --------------------------------------
        if stake:
            w = _cheapest_winner(v)
            if w is not None:
                return w                  # take the ten
            # Cannot win a trick that holds a ten -- do not feed it another.
            return _safe_discard(v)

        # No ten at stake. This is the 8-of-hearts case: concede cheaply.
        # Exception -- winning it costs nothing when the cheapest winner is
        # already our lowest card, and tricks matter for the W5 tiebreak
        # (~34% of hands).
        w = _cheapest_winner(v)
        low = _safe_discard(v)
        if w is not None and not is_ten(w) and suit_of(w) != v.trump_suit \
                and rank_of(w) <= rank_of(low):
            return w
        return low


class Balanced(TenAware):
    """TenAware plus a mild tiebreak-aware tilt.

    Because W5 decides ~34% of hands, a free trick is worth taking. This agent
    accepts a slightly more expensive win on a ten-less trick late in the hand,
    when trumps have less remaining value.
    """

    name = "Balanced"

    def act(self, v):
        if not v.on_lead and not v.trick_has_ten() \
                and not v.partner_is_winning() and v.tricks_played >= 9:
            w = _cheapest_winner(v)
            if w is not None and not is_ten(w) and suit_of(w) != v.trump_suit:
                return w
        return super().act(v)


#: Everything Phase 2 evaluates.
RULE_BASED = [
    RandomAgent, LowestAgent, GreedyTricks,
    TenHunter, TenDefender, PartnerAware, TenAware, Balanced,
]

REGISTRY = {cls.name: cls for cls in RULE_BASED}


class Adaptive(TenAware):
    """Hunts tens while they are live, then switches to tricks.

    Motivated by a measured result, not intuition. Over 10,000 hands,
    TenAware beat GreedyTricks 0.646 on hands decided by tens but only 0.192
    on hands decided by the W5 tiebreak, and the two effects cancelled almost
    exactly (0.502 overall). Conceding cheap tricks is correct while a ten is
    still live and wrong once the tens are settled.

    So track the tens and switch objective when they run out:

      * tens still outstanding  -> TenAware (protect and hunt)
      * all four gone, split 2-2 -> the hand is now a pure trick race (W5):
                                    take everything
      * all four gone to us, and we hold every trick so far -> chase the
                                    double coat (W3)
      * otherwise                -> the hand is already decided; play low and
                                    keep good cards for nothing
    """

    name = "Adaptive"

    def act(self, v):
        outstanding = v.tens_outstanding
        if outstanding:
            return super().act(v)          # tens still live

        mine, theirs = v.tens_by_team[v.team], v.tens_by_team[1 - v.team]

        if mine == 2 and theirs == 2:
            return self._grab(v)           # W5 race
        if mine == 4 and v.tricks_by_team[1 - v.team] == 0:
            return self._grab(v)           # W3 double coat still alive
        return _safe_discard(v)            # decided; nothing left to play for

    @staticmethod
    def _grab(v):
        if v.partner_is_winning() and len(v.trick) == 3:
            return _safe_discard(v)        # partner already has it
        w = _cheapest_winner(v)
        return w if w is not None else _safe_discard(v)


class AdaptiveGreedy(Adaptive):
    """Adaptive, but it also takes tricks that cost nothing while hunting.

    TenAware concedes ~7.6 tricks a hand, which throws away the tiebreak. This
    keeps the ten discipline but stops declining free tricks: if the cheapest
    winning card is a low plain card, take the trick.
    """

    name = "AdaptiveGreedy"

    def act(self, v):
        if v.tens_outstanding and not v.on_lead and not v.trick_has_ten() \
                and not v.partner_is_winning():
            w = _cheapest_winner(v)
            if w is not None and not is_ten(w) \
                    and suit_of(w) != v.trump_suit and rank_of(w) <= 8:
                return w                   # free trick, no ten spent
        return super().act(v)


RULE_BASED.extend([Adaptive, AdaptiveGreedy])
REGISTRY.update({cls.name: cls for cls in (Adaptive, AdaptiveGreedy)})


class TensThenTricks(TenAware):
    """Tens first, tricks ALWAYS second -- never as a late switch.

    From the first author: "if i can win just two tens my secondary aim should be
    always most tricks."

    That is the correction to Adaptive, which only started chasing tricks once
    all four tens were gone -- far too late to matter. Because a 2-2 split is
    the single most likely outcome (~34% of hands), the trick race is live from
    trick one and should be contested continuously, not eventually.

    Priority order at every decision:
      1. never spend a ten where the opponents can take it
      2. never overtake the partner
      3. if a ten is at stake, take the trick as cheaply as possible
      4. OTHERWISE TAKE THE TRICK ANYWAY -- unless doing so burns a trump or a
         high card that is still needed for an outstanding ten
    """

    name = "TensThenTricks"

    #: While a ten is still live, a plain card at or below this rank is
    #: considered expendable on a ten-less trick. Ace..Jack are held back only
    #: while they might still capture a ten.
    CHEAP = 10        # rank 10 == Queen

    def act(self, v):
        pos = len(v.trick)
        last = pos == 3

        if v.on_lead:
            return super().act(v)

        # (2) partner already winning -- bank a ten if the outcome is certain
        if v.partner_is_winning():
            tens = [c for c in v.legal if is_ten(c)]
            if last and tens:
                return tens[0]
            return _safe_discard(v)

        win = _cheapest_winner(v)
        if win is None:
            return _safe_discard(v)        # cannot win; concede cheaply

        # (3) a ten is on the table -- take it, whatever it costs
        if v.trick_has_ten():
            return win

        # (4) no ten at stake, but tricks are the standing secondary aim.
        tens_live = bool(v.tens_outstanding)
        if not tens_live:
            return win                     # tens settled: pure trick race

        is_trump = suit_of(win) == v.trump_suit
        if is_trump:
            # Trumps are the tool for capturing outstanding tens. Spend one on
            # an empty trick only from the last seat, where nothing is lost.
            return win if last else _safe_discard(v)
        if rank_of(win) <= self.CHEAP or last:
            return win                     # cheap enough, or free
        return _safe_discard(v)


RULE_BASED.append(TensThenTricks)
REGISTRY[TensThenTricks.name] = TensThenTricks
