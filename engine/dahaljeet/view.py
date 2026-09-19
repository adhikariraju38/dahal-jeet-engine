"""What a player is allowed to know.

Agents receive a PlayerView, never a Hand.  This is a scientific-validity
guard, not a convenience: given a Hand an agent could read the opponents'
cards, and every result would be worthless.  Phase 3 search agents rebuild
hypothetical worlds from a view; they never peek at the real one.
"""

from __future__ import annotations

from dataclasses import dataclass

from .cards import suit_of, is_ten, TENS, HAND_SIZE
from .hand import PARTNER, team_of


@dataclass(frozen=True)
class PlayerView:
    """The information set of one seat at one decision point."""

    seat: int
    hand: tuple[int, ...]              # own cards only
    legal: tuple[int, ...]             # pre-filtered by P2/P3
    trump_suit: int
    trump_card: int                    # public since T3
    trump_holder: int
    lead_suit: int                     # -1 if on lead
    trick: tuple[tuple[int, int], ...]  # (seat, card) so far this trick
    played_by_seat: tuple[tuple[int, ...], ...]
    voids: tuple[frozenset[int], ...]  # inferred from P2, public knowledge
    tens_by_team: tuple[int, int]
    tricks_by_team: tuple[int, int]
    tricks_played: int
    dealer: int
    ten_owner: tuple[tuple[int, int], ...] = ()  # (card, team) pairs

    # ------------------------------------------------------------ helpers

    @property
    def team(self) -> int:
        return team_of(self.seat)

    @property
    def partner(self) -> int:
        return PARTNER[self.seat]

    @property
    def on_lead(self) -> bool:
        return not self.trick

    @property
    def seen(self) -> frozenset[int]:
        """Every card whose location this seat knows: own hand + all played."""
        s = set(self.hand)
        for cards in self.played_by_seat:
            s.update(cards)
        return frozenset(s)

    @property
    def unseen(self) -> frozenset[int]:
        """Cards that could be in any other hand. Basis for determinization."""
        return frozenset(range(52)) - self.seen

    @property
    def tens_outstanding(self) -> frozenset[int]:
        """Tens not yet played. These are the only cards that still matter."""
        played = set()
        for cards in self.played_by_seat:
            played.update(cards)
        return frozenset(TENS - played)

    def trick_has_ten(self) -> bool:
        """Is a ten already committed to this trick?  (W1 — the only stake.)"""
        return any(is_ten(c) for _, c in self.trick)

    def cards_left(self, seat: int) -> int:
        return HAND_SIZE - len(self.played_by_seat[seat])

    # ------------------------------------------------- trick arithmetic

    def beats(self, card: int, other: int) -> bool:
        """Would `card` beat `other`, given the led suit?  (P4)"""
        cs, os_ = suit_of(card), suit_of(other)
        if cs == self.trump_suit:
            return os_ != self.trump_suit or card > other
        if os_ == self.trump_suit:
            return False
        if cs == self.lead_suit:
            return os_ != self.lead_suit or card > other
        return False

    def current_winner(self) -> tuple[int, int] | None:
        """(seat, card) currently winning this trick, or None if on lead."""
        if not self.trick:
            return None
        best_seat, best_card = self.trick[0]
        for seat, card in self.trick[1:]:
            saved, self_lead = self.lead_suit, None
            del saved, self_lead
            if self.beats(card, best_card):
                best_seat, best_card = seat, card
        return best_seat, best_card

    def partner_is_winning(self) -> bool:
        w = self.current_winner()
        return w is not None and w[0] == self.partner

    def winning_moves(self) -> list[int]:
        """Legal cards that would currently take the trick."""
        w = self.current_winner()
        if w is None:
            return list(self.legal)
        return [c for c in self.legal if self.beats(c, w[1])]


def make_view(hand, seat: int | None = None) -> PlayerView:
    """Project a Hand down to what `seat` may legally see."""
    if seat is None:
        seat = hand.to_act
    return PlayerView(
        seat=seat,
        hand=tuple(hand.hands[seat]),
        legal=tuple(hand.legal_moves(seat)),
        trump_suit=hand.trump_suit,
        trump_card=hand.trump_card,
        trump_holder=hand.trump_holder,
        lead_suit=hand.trick_lead_suit,
        trick=tuple(hand.trick),
        played_by_seat=tuple(tuple(c) for c in hand.played_by_seat),
        voids=tuple(frozenset(v) for v in hand.voids),
        tens_by_team=tuple(hand.tens_by_team),
        tricks_by_team=tuple(hand.tricks_by_team),
        tricks_played=hand.tricks_played,
        dealer=hand.dealer,
        ten_owner=tuple(sorted(hand.ten_owner.items())),
    )
