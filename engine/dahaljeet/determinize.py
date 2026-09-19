"""Sampling hypothetical worlds consistent with what a seat has observed.

A determinization assigns every unseen card to some opponent seat such that:
  * each seat receives exactly the number of cards it still holds
  * no seat receives a suit it is provably void in  (P2 -> Hand.voids)

The void constraint is what makes sampling worth doing. A uniform assignment
of unseen cards ignores the play history, and most such assignments contradict
it -- a seat that discarded on a spade lead cannot hold spades (P2), yet an
unconstrained sample will keep dealing it spades. Rollouts in impossible worlds
are wasted. Filtering by inferred voids costs almost nothing per sample and
makes every sample informative.

Assignment is a bipartite feasibility problem. We place the most-constrained
cards first (fewest eligible seats) and retry on failure, which is fast enough
in practice -- 13 tricks give few voids early and tight hands late.
"""

from __future__ import annotations

import random

from .cards import suit_of
from .hand import Hand, PARTNER


def determinize(v, rng: random.Random, tries: int = 40):
    """Return `hands`: list of 4 card-lists consistent with view `v`.

    The acting seat gets its real cards; the others get a consistent guess.
    Returns None if no consistent assignment was found within `tries`.
    """
    unseen = list(v.unseen)
    need = {}
    for s in range(4):
        if s == v.seat:
            continue
        need[s] = len(v.hand) - (1 if _has_played_this_trick(v, s) else 0)
    # Correct count: cards still in hand = 13 - number already played.
    for s in range(4):
        if s == v.seat:
            continue
        need[s] = 13 - len(v.played_by_seat[s])

    assert sum(need.values()) == len(unseen), (
        f"need {sum(need.values())} != unseen {len(unseen)}")

    for _ in range(tries):
        rng.shuffle(unseen)
        remaining = dict(need)
        out = {s: [] for s in need}
        # Most-constrained card first: fewest seats that can legally hold it.
        def eligible(card):
            su = suit_of(card)
            return [s for s in need
                    if remaining[s] > 0 and su not in v.voids[s]]
        order = sorted(unseen, key=lambda c: len(eligible(c)))
        ok = True
        for card in order:
            cands = eligible(card)
            if not cands:
                ok = False
                break
            # Weight by remaining capacity so hands fill evenly.
            pick = rng.choices(cands, weights=[remaining[s] for s in cands])[0]
            out[pick].append(card)
            remaining[pick] -= 1
        if ok and all(r == 0 for r in remaining.values()):
            hands = [None] * 4
            hands[v.seat] = list(v.hand)
            for s, cards in out.items():
                hands[s] = cards
            return hands
    return None


def _has_played_this_trick(v, seat):
    return any(s == seat for s, _ in v.trick)


def hand_from_view(v, hands) -> Hand:
    """Rebuild a playable Hand from a view plus a determinization.

    The reconstructed Hand is a hypothetical world: legal, self-consistent,
    and identical to the real one in everything the acting seat can observe.
    """
    h = Hand(dealer=v.dealer)
    h.hands = [sorted(x) for x in hands]
    h.trump_suit = v.trump_suit
    h.trump_card = v.trump_card
    h.trump_holder = v.trump_holder
    h.trick = list(v.trick)
    h.trick_lead_suit = v.lead_suit
    h.tricks_played = v.tricks_played
    h.tens_by_team = list(v.tens_by_team)
    h.tricks_by_team = list(v.tricks_by_team)
    h.played_by_seat = [list(x) for x in v.played_by_seat]
    h.voids = [set(x) for x in v.voids]
    h.ten_owner = dict(v.ten_owner)
    h.to_act = v.seat
    h.leader = v.trick[0][0] if v.trick else v.seat
    h.trick_winners = []
    return h
