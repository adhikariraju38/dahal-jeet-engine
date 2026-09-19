"""One hand of Dahal Jeet: the deal, thirteen tricks, and the score.

Rule IDs refer to the rule specification in the paper's Methods (draft 05).

Seats are 0..3 and correspond to the rulebook's seats 1..4 (seat i in code
== seat i+1 in the rulebook).  Turn order is 0 -> 1 -> 2 -> 3 -> 0 (S3).
Teams are seats {0, 2} and {1, 3} (S1): partners sit opposite.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .cards import (
    HAND_SIZE, N_CARDS, N_PLAYERS, TENS, TEN_RANK,
    is_ten, suit_of, card_name,
)

#: Seat -> partner seat (S1).  Partners sit opposite.
PARTNER = (2, 3, 0, 1)


def team_of(seat: int) -> int:
    """Team 0 is seats {0,2}; team 1 is seats {1,3}.  (S1)"""
    return seat & 1


@dataclass(frozen=True)
class HandResult:
    """Outcome of a completed hand."""

    winning_team: int
    tens_by_team: tuple[int, int]        # W1
    tricks_by_team: tuple[int, int]
    coat: bool                            # W2 — winners took all four tens
    double_coat: bool                     # W3 — all four tens AND all 13 tricks
    trump_suit: int
    trump_card: int
    dealer: int
    #: Seat that won each trick, in order.
    trick_winners: tuple[int, ...] = field(default=(), repr=False)

    @property
    def losing_team(self) -> int:
        return 1 - self.winning_team

    def describe(self) -> str:
        label = "DOUBLE COAT" if self.double_coat else ("COAT" if self.coat else "win")
        return (
            f"team {self.winning_team} {label} "
            f"(tens {self.tens_by_team[0]}-{self.tens_by_team[1]}, "
            f"tricks {self.tricks_by_team[0]}-{self.tricks_by_team[1]})"
        )


class Hand:
    """A single hand: deal, 13 tricks, score.

    Usage::

        h = Hand(dealer=0, rng=random.Random(7))
        h.deal()
        while not h.is_over:
            h.play(agent_choice(h))
        result = h.result()
    """

    __slots__ = (
        "dealer", "rng", "hands", "trump_suit", "trump_card", "trump_holder",
        "leader", "to_act", "trick", "trick_lead_suit", "tricks_played",
        "tens_by_team", "tricks_by_team", "played_by_seat", "trick_winners",
        "voids", "ten_owner",
    )

    def __init__(self, dealer: int = 0, rng: random.Random | None = None):
        self.dealer = dealer
        self.rng = rng if rng is not None else random.Random()
        self.hands: list[list[int]] = [[] for _ in range(N_PLAYERS)]
        self.trump_suit: int = -1
        self.trump_card: int = -1
        self.trump_holder: int = -1
        self.leader: int = -1
        self.to_act: int = -1
        self.trick: list[tuple[int, int]] = []      # (seat, card)
        self.trick_lead_suit: int = -1
        self.tricks_played: int = 0
        self.tens_by_team = [0, 0]
        self.tricks_by_team = [0, 0]
        self.played_by_seat: list[list[int]] = [[] for _ in range(N_PLAYERS)]
        self.trick_winners: list[int] = []
        #: card -> team that captured it. Public: everyone sees who takes
        #: each trick and what is in it.
        self.ten_owner: dict[int, int] = {}
        #: voids[seat] is the set of suits that seat has provably run out of.
        #: Derived from P2 — failing to follow suit proves a void.  Phase 3
        #: uses this to constrain determinization sampling.
        self.voids: list[set[int]] = [set() for _ in range(N_PLAYERS)]

    # ---------------------------------------------------------------- deal

    def deal(self) -> None:
        """Deal 5, fix trump, deal 4 + 4.  (D1, T1-T4, D2)"""
        deck = list(range(N_CARDS))
        self.rng.shuffle(deck)

        # D1 — five cards each, in seat order starting left of the dealer.
        order = [(self.dealer + 1 + i) % N_PLAYERS for i in range(N_PLAYERS)]
        pos = 0
        for seat in order:
            self.hands[seat] = deck[pos:pos + 5]
            pos += 5

        # T1/T2 — the seat after the dealer draws one of his own five at
        # random, WITHOUT looking.  The pick is a randomisation device, not a
        # decision, so the engine makes it rather than any agent.
        self.trump_holder = (self.dealer + 1) % N_PLAYERS
        idx = self.rng.randrange(5)
        self.trump_card = self.hands[self.trump_holder][idx]
        self.trump_suit = suit_of(self.trump_card)
        # T3 — shown to everyone; T4 — the card stays in his hand.

        # D2 — four more each, then four more.  13 apiece.
        for _ in range(2):
            for seat in order:
                self.hands[seat].extend(deck[pos:pos + 4])
                pos += 4
        assert pos == N_CARDS
        for seat in range(N_PLAYERS):
            assert len(self.hands[seat]) == HAND_SIZE
            self.hands[seat].sort()

        # P1 — the trump holder leads the first trick.
        self.leader = self.trump_holder
        self.to_act = self.leader

    # ------------------------------------------------------------- queries

    @property
    def is_over(self) -> bool:
        return self.tricks_played == HAND_SIZE

    def legal_moves(self, seat: int | None = None) -> list[int]:
        """Cards `seat` may legally play right now.  (P2, P3)

        Must follow the led suit if holding it; otherwise anything, trump
        included — there is no obligation to trump.
        """
        if seat is None:
            seat = self.to_act
        hand = self.hands[seat]
        if not self.trick:
            return list(hand)                      # leader is unconstrained
        follow = [c for c in hand if suit_of(c) == self.trick_lead_suit]
        return follow if follow else list(hand)    # P2 / P3

    # ---------------------------------------------------------------- play

    def play(self, card: int) -> int | None:
        """Play `card` for the seat to act.  Returns the trick winner if the
        trick completed, else None."""
        seat = self.to_act
        if card not in self.hands[seat]:
            raise ValueError(f"seat {seat} does not hold {card_name(card)}")
        legal = self.legal_moves(seat)
        if card not in legal:
            raise ValueError(
                f"illegal: seat {seat} must follow "
                f"{'CDHS'[self.trick_lead_suit]} (P2)"
            )

        if not self.trick:
            self.trick_lead_suit = suit_of(card)
        elif suit_of(card) != self.trick_lead_suit:
            # P2 — could not follow, so this seat is provably void.
            self.voids[seat].add(self.trick_lead_suit)

        self.hands[seat].remove(card)
        self.played_by_seat[seat].append(card)
        self.trick.append((seat, card))
        self.to_act = (seat + 1) % N_PLAYERS

        if len(self.trick) < N_PLAYERS:
            return None
        return self._resolve_trick()

    def _resolve_trick(self) -> int:
        """Highest trump, else highest of the led suit.  (P4, C1, P5)"""
        best_seat, best_card = self.trick[0]
        best_is_trump = suit_of(best_card) == self.trump_suit
        for seat, card in self.trick[1:]:
            s = suit_of(card)
            if best_is_trump:
                if s == self.trump_suit and card > best_card:
                    best_seat, best_card = seat, card
            elif s == self.trump_suit:
                best_seat, best_card, best_is_trump = seat, card, True
            elif s == self.trick_lead_suit and card > best_card:
                best_seat, best_card = seat, card

        # C1 — single-hand variant: the winner takes the cards immediately.
        team = team_of(best_seat)
        self.tricks_by_team[team] += 1
        for _, card in self.trick:
            if is_ten(card):
                self.tens_by_team[team] += 1        # W1
                self.ten_owner[card] = team

        self.trick_winners.append(best_seat)
        self.trick = []
        self.trick_lead_suit = -1
        self.tricks_played += 1
        self.leader = best_seat                     # P5
        self.to_act = best_seat
        return best_seat

    # -------------------------------------------------------------- result

    def result(self) -> HandResult:
        """Score the completed hand.  (W1-W5, W2/W3 for coat/double coat)"""
        if not self.is_over:
            raise RuntimeError("hand is not finished")
        t0, t1 = self.tens_by_team
        assert t0 + t1 == 4

        if t0 != t1:
            winner = 0 if t0 > t1 else 1            # W2/W4 — majority of tens
        else:
            # W5 — tens split 2-2, most tricks wins.  13 tricks, so no draw.
            winner = 0 if self.tricks_by_team[0] > self.tricks_by_team[1] else 1

        coat = self.tens_by_team[winner] == 4                    # W2
        double_coat = coat and self.tricks_by_team[winner] == HAND_SIZE  # W3

        return HandResult(
            winning_team=winner,
            tens_by_team=(t0, t1),
            tricks_by_team=tuple(self.tricks_by_team),
            coat=coat,
            double_coat=double_coat,
            trump_suit=self.trump_suit,
            trump_card=self.trump_card,
            dealer=self.dealer,
            trick_winners=tuple(self.trick_winners),
        )
