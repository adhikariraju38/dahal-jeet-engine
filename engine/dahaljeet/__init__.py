"""Dahal Jeet — a reference engine for the single-hand variant.

Implements the specification in the rule specification in the paper's Methods
(draft 05).  Every rule carries an ID (S1, T2, W5, R3 ...) and every rule ID
has a corresponding test in ../tests/.
"""

from .cards import (
    SUITS, RANKS, TENS, N_CARDS, N_PLAYERS, HAND_SIZE,
    suit_of, rank_of, is_ten, card_name, hand_str, parse_card,
)
from .hand import Hand, HandResult, team_of, PARTNER
from .session import Session, next_dealer, Tally

__all__ = [
    "SUITS", "RANKS", "TENS", "N_CARDS", "N_PLAYERS", "HAND_SIZE",
    "suit_of", "rank_of", "is_ten", "card_name", "hand_str", "parse_card",
    "Hand", "HandResult", "team_of", "PARTNER",
    "Session", "next_dealer", "Tally",
]
