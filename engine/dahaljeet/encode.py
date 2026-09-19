"""State encoding for learning agents (Phase 4).

When a learning agent underperforms on a card game, the cause is usually the
state representation rather than the algorithm -- and an undocumented encoding
makes that impossible to diagnose or to reproduce. So this module is
deliberately explicit: every field is named, sized and documented, the layout
is printed by describe() so it can go straight into the paper, and each block
is ablated in ablate_encoding.py so its contribution is measured rather than
assumed.

Two choices that matter:

1. SEAT-RELATIVE encoding. Everything is indexed relative to the acting seat
   (self / right-hand opponent / partner / left-hand opponent), never by
   absolute seat number. A policy learned in one seat then transfers to all
   four, which matters because seat 2 is structurally special (T5) and absolute
   encoding would force the network to learn four separate policies.

2. TEN STATUS is explicit. The four tens are the entire objective (W1), so
   each one gets a 3-way one-hot: still out, captured by us, captured by them.
   Making the network infer this from raw play histories is exactly the kind of
   avoidable difficulty that sinks sparse-reward learning.

Pure stdlib: returns a list[float]. Tensor conversion belongs to the trainer.
"""

from __future__ import annotations

from .cards import N_CARDS, TENS, suit_of
from .hand import PARTNER, team_of

# ---- layout ---------------------------------------------------------------
_F = []


def _field(name, size):
    off = sum(s for _, s in _F)
    _F.append((name, size))
    return off, size


OFF_HAND, SZ_HAND = _field("own_hand", 52)
OFF_PLAYED, SZ_PLAYED = _field("played_by_relseat", 4 * 52)
OFF_TRUMP, SZ_TRUMP = _field("trump_suit", 4)
OFF_TENS, SZ_TENS = _field("ten_status", 4 * 3)
OFF_TRICK, SZ_TRICK = _field("current_trick_cards", 52)
OFF_POS, SZ_POS = _field("position_in_trick", 4)
OFF_LEAD, SZ_LEAD = _field("lead_suit", 5)
OFF_VOID, SZ_VOID = _field("voids_by_relseat", 4 * 4)
OFF_WIN, SZ_WIN = _field("partner_winning", 2)
OFF_SCALARS, SZ_SCALARS = _field("scalars", 5)

OBS_SIZE = sum(s for _, s in _F)
ACTION_SIZE = N_CARDS


def describe() -> str:
    """Human-readable layout table -- paste into the paper's appendix."""
    lines = [f"observation vector: {OBS_SIZE} floats", ""]
    off = 0
    for name, size in _F:
        lines.append(f"  [{off:4d}:{off+size:4d}]  {size:4d}  {name}")
        off += size
    lines.append(f"\naction space: {ACTION_SIZE} (card id), masked to legal moves")
    return "\n".join(lines)


def relseat(seat: int, me: int) -> int:
    """0=self, 1=next (opponent), 2=partner, 3=previous (opponent)."""
    return (seat - me) % 4


def encode(v) -> list[float]:
    """PlayerView -> fixed-length observation."""
    x = [0.0] * OBS_SIZE
    me = v.seat

    for c in v.hand:
        x[OFF_HAND + c] = 1.0

    for s in range(4):
        base = OFF_PLAYED + relseat(s, me) * 52
        for c in v.played_by_seat[s]:
            x[base + c] = 1.0

    x[OFF_TRUMP + v.trump_suit] = 1.0

    # ten status, per ten: [still out, ours, theirs]. This is the whole
    # objective (W1), so it is given directly rather than left to be inferred.
    my_team = team_of(me)
    owner = dict(v.ten_owner)
    for i, ten in enumerate(sorted(TENS)):
        base = OFF_TENS + i * 3
        if ten in owner:
            x[base + (1 if owner[ten] == my_team else 2)] = 1.0
        else:
            x[base + 0] = 1.0

    for _, c in v.trick:
        x[OFF_TRICK + c] = 1.0
    x[OFF_POS + len(v.trick)] = 1.0

    x[OFF_LEAD + (4 if v.lead_suit < 0 else v.lead_suit)] = 1.0

    for s in range(4):
        base = OFF_VOID + relseat(s, me) * 4
        for suit in v.voids[s]:
            x[base + suit] = 1.0

    if v.trick:
        x[OFF_WIN + (0 if v.partner_is_winning() else 1)] = 1.0

    them = 1 - my_team
    x[OFF_SCALARS + 0] = v.tens_by_team[my_team] / 4.0
    x[OFF_SCALARS + 1] = v.tens_by_team[them] / 4.0
    x[OFF_SCALARS + 2] = v.tricks_by_team[my_team] / 13.0
    x[OFF_SCALARS + 3] = v.tricks_by_team[them] / 13.0
    x[OFF_SCALARS + 4] = v.tricks_played / 13.0
    return x


def legal_mask(v) -> list[float]:
    """1.0 for legal cards, 0.0 otherwise. Illegal moves are masked, never
    penalised -- the engine makes them impossible (P2/P3)."""
    m = [0.0] * ACTION_SIZE
    for c in v.legal:
        m[c] = 1.0
    return m
