"""Card representation for Dahal Jeet.

Cards are ints 0..51.  suit = c // 13, rank = c % 13.
Rank 0 is the Two and rank 12 is the Ace, so higher int == higher card
within a suit (rule S2).

Keeping cards as bare ints makes the trick loop allocation-free, which
matters because the evaluation in Phase 5 needs millions of hands.
"""

SUITS = ("C", "D", "H", "S")          # club, diamond, heart, spade
RANKS = ("2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A")

N_CARDS = 52
N_PLAYERS = 4
HAND_SIZE = 13

TEN_RANK = 8                           # RANKS[8] == "10"

#: The four scoring cards (rule W1).  Nothing else counts.
TENS = frozenset(s * 13 + TEN_RANK for s in range(4))

assert len(TENS) == 4


def suit_of(card: int) -> int:
    return card // 13


def rank_of(card: int) -> int:
    return card % 13


def is_ten(card: int) -> bool:
    return card % 13 == TEN_RANK


def card_name(card: int) -> str:
    return f"{RANKS[card % 13]}{SUITS[card // 13]}"


def hand_str(cards) -> str:
    return " ".join(card_name(c) for c in sorted(cards))


def parse_card(name: str) -> int:
    """Inverse of card_name, e.g. '10S' -> 47.  Used by tests and fixtures."""
    name = name.strip().upper()
    suit = SUITS.index(name[-1])
    rank = RANKS.index(name[:-1])
    return suit * 13 + rank
