"""NOT PART OF THE PAPER — parked as future work (2026-09-03).

Scope decision by the project owner: this work documents the SINGLE-HAND
(immediate capture) variant only, which is what the player communities actually
play and what the rulebook specifies. The deferred-capture "double hand" /
Double Sir variant is a possible future extension, not a current contribution.

Left here because the engine subclass is written and its self-tests pass (tens
conserved, 13 tricks counted, centre pile always cleared, and the variant
provably differs from the standard game on 55/60 identical deals).

KNOWN UNFINISHED: the variant-aware search arm fails an assertion. When search
reconstructs a world from a PlayerView, the centre pile is not represented in
the view, so tens lying in the centre are unaccounted for. Passing the pile via
a module global was a first attempt and did not fully fix it. Doing this
properly means representing the pending pile in PlayerView itself. Anyone
resuming this should start there.

Nothing in the pipeline imports this file.

"""
"""Rule-variant robustness: do the agents transfer to the documented relatives?

Every result in this project is measured on ONE ruleset -- the single-hand,
immediate-capture variant. A reviewer can reasonably ask whether the methods
are tuned to that ruleset or whether they capture the game. The comparative
identification already establishes, from fetched sources, that Dahal Jeet sits
in the Court Piece / Rang family and that its documented sibling differs on one
crisp mechanic, so the question can be answered rather than deflected.

THE VARIANT (Double Sir / "double hand" / deferred capture), from Pagat via
01-rulebooks/comparative-identification.md:

  * a trick's cards are NOT taken immediately; they accumulate in the centre
  * a player collects the accumulated pile only by winning TWO CONSECUTIVE
    tricks
  * "The player who wins the 13th and last trick takes in this and any tricks
    that have accumulated in the centre, even if he did not win the 12th trick."

Only capture timing changes. Dealing, trumps, following suit, and the win
condition are untouched, which is what makes it a clean test: the agents face
the same game with one documented rule altered, and nothing was retrained.

    python variants.py --deals 300

This is implemented as a SUBCLASS in its own module. dahaljeet/hand.py is not
modified, so nothing already running or already measured can be perturbed.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time

sys.path.insert(0, ".")
import runenv

from dahaljeet.agents import REGISTRY
from dahaljeet.cards import is_ten
from dahaljeet.hand import Hand, team_of
from dahaljeet.view import make_view


class DeferredCaptureHand(Hand):
    """Double Sir: the pile is won by taking two tricks in a row."""

    def __init__(self, dealer=0, rng=None):
        super().__init__(dealer=dealer, rng=rng)
        self.pending: list[int] = []      # cards lying in the centre
        self.piles_collected = [0, 0]

    def _resolve_trick(self) -> int:
        best_seat, best_card = self.trick[0]
        best_is_trump = (best_card // 13) == self.trump_suit
        for seat, card in self.trick[1:]:
            s = card // 13
            if best_is_trump:
                if s == self.trump_suit and card > best_card:
                    best_seat, best_card = seat, card
            elif s == self.trump_suit:
                best_seat, best_card, best_is_trump = seat, card, True
            elif s == self.trick_lead_suit and card > best_card:
                best_seat, best_card = seat, card

        team = team_of(best_seat)
        self.tricks_by_team[team] += 1
        cards = [c for _, c in self.trick]

        won_previous = bool(self.trick_winners) and self.trick_winners[-1] == best_seat
        is_last = self.tricks_played == 12          # this is the 13th trick

        if won_previous or is_last:
            # collect the centre as well as this trick
            self.piles_collected[team] += 1
            for card in self.pending + cards:
                if is_ten(card):
                    self.tens_by_team[team] += 1
                    self.ten_owner[card] = team
            self.pending = []
        else:
            self.pending.extend(cards)

        self.trick_winners.append(best_seat)
        self.trick = []
        self.trick_lead_suit = -1
        self.tricks_played += 1
        self.leader = best_seat
        self.to_act = best_seat
        return best_seat


def play(hand_cls, seats, deal_seed, dealer=0):
    global CURRENT_PENDING
    h = hand_cls(dealer=dealer, rng=random.Random(deal_seed))
    h.deal()
    while not h.is_over:
        CURRENT_PENDING = list(getattr(h, "pending", []))
        h.play(seats[h.to_act].act(make_view(h, h.to_act)))
    CURRENT_PENDING = []
    return h


from contextlib import contextmanager


#: The centre pile of the hand currently being played. This is PUBLIC
#: information -- those cards were played face up and every player can see
#: them -- so handing it to the search agent leaks nothing hidden. It is passed
#: this way because PlayerView has no notion of a pending pile, and extending
#: the view would mean editing dahaljeet/view.py, which other experiments and
#: running jobs depend on.
CURRENT_PENDING: list[int] = []


@contextmanager
def variant_aware_search():
    """Make PIMC/ISMCTS simulate the VARIANT instead of the standard rules.

    Without this, search agents playing the variant still plan with a
    standard-rules model, because dahaljeet.search rebuilds worlds with
    hand_from_view(), which constructs a plain Hand. That is worth measuring on
    its own -- it is what happens if you deploy an agent into a game whose rules
    changed underneath it -- but it is NOT "search re-plans under the new rule",
    which an earlier draft of this file wrongly claimed.

    Patched at runtime so dahaljeet/search.py is left untouched.
    """
    from dahaljeet import search as search_mod
    orig = search_mod.hand_from_view

    def variant_hand_from_view(v, hands):
        h = orig(v, hands)
        g = DeferredCaptureHand(dealer=h.dealer)
        g.__dict__.update({k: val for k, val in h.__dict__.items()})
        # restore the centre pile, or the tens lying in it are lost and the
        # simulated hand ends with fewer than four tens accounted for
        g.pending = list(CURRENT_PENDING)
        g.piles_collected = [0, 0]
        return g

    search_mod.hand_from_view = variant_hand_from_view
    try:
        yield
    finally:
        search_mod.hand_from_view = orig


def make_agent(spec, rng):
    if spec.startswith("PIMC:"):
        from dahaljeet.search import PIMCAgent
        return PIMCAgent(worlds=int(spec.split(":")[1]), rng=rng)
    if spec.startswith("ISMCTS:"):
        from dahaljeet.search import ISMCTSAgent
        return ISMCTSAgent(iterations=int(spec.split(":")[1]), rng=rng)
    if spec.endswith(".pt"):
        import torch
        from nets import Net, TorchAgent
        sd = torch.load(spec, map_location="cpu")
        for critic in (False, True):
            try:
                net = Net(critic=critic); net.load_state_dict(sd); net.eval()
                return TorchAgent(net, spec[3:-3], greedy=True, rng=rng)
            except Exception:
                continue
        return None
    return REGISTRY[spec](rng=rng)


def match(a_spec, b_spec, hand_cls, deals, seed):
    base = random.Random(seed).randrange(1 << 30)
    wins = []
    for d in range(deals):
        w = 0
        for r in range(4):
            a = make_agent(a_spec, random.Random(1000 + d))
            b = make_agent(b_spec, random.Random(2000 + d))
            if a is None or b is None:
                return None
            seats = {s: (a if ((s - r) % 4) % 2 == 0 else b) for s in range(4)}
            h = play(hand_cls, seats, base + d)
            if h.result().winning_team == team_of(r):
                w += 1
        wins.append(w)
    n = len(wins) * 4
    rate = sum(wins) / n
    rng = random.Random(0)
    boots = sorted(sum(wins[rng.randrange(len(wins))] for _ in wins) / n
                   for _ in range(2000))
    return {"win": round(rate, 4),
            "ci": [round(boots[50], 4), round(boots[-50], 4)],
            "per_deal": wins}


def selftest():
    print("SELF-TEST — the variant must still be a legal, consistent game\n")
    ok = True
    rng = random.Random(0)
    bad_tens = bad_tricks = pend = 0
    for _ in range(200):
        seats = {s: REGISTRY["TensThenTricks"](rng=random.Random(s)) for s in range(4)}
        h = play(DeferredCaptureHand, seats, rng.randrange(1 << 30))
        r = h.result()
        if sum(r.tens_by_team) != 4:
            bad_tens += 1
        if sum(r.tricks_by_team) != 13:
            bad_tricks += 1
        if h.pending:
            pend += 1
    print(f"  [{'PASS' if bad_tens == 0 else 'FAIL'}] all four tens always "
          f"accounted for ({bad_tens} bad of 200)")
    print(f"  [{'PASS' if bad_tricks == 0 else 'FAIL'}] 13 tricks always "
          f"counted ({bad_tricks} bad)")
    print(f"  [{'PASS' if pend == 0 else 'FAIL'}] centre pile always empty at "
          f"the end -- the 13th-trick rule must clear it ({pend} left over)")
    ok = bad_tens == 0 and bad_tricks == 0 and pend == 0

    # the variant must actually DIFFER from the standard game
    diff = 0
    for i in range(60):
        seed = 5000 + i
        s1 = {s: REGISTRY["TensThenTricks"](rng=random.Random(s)) for s in range(4)}
        s2 = {s: REGISTRY["TensThenTricks"](rng=random.Random(s)) for s in range(4)}
        a = play(Hand, s1, seed).result()
        b = play(DeferredCaptureHand, s2, seed).result()
        if (a.winning_team, a.tens_by_team) != (b.winning_team, b.tens_by_team):
            diff += 1
    r2 = diff > 0
    ok &= r2
    print(f"  [{'PASS' if r2 else 'FAIL'}] variant differs from the standard "
          f"game on {diff}/60 identical deals (0 would mean it is not "
          f"implemented)")
    print(f"\n{'ALL PASS' if ok else 'FAILURES PRESENT'}")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--deals", type=int, default=300)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--agents",
                    default="Random,GreedyTricks,TenAware,TensThenTricks,"
                            "rl_ppo_potential_s0.pt,rl_mappo_potential_s0.pt,"
                            "PIMC:16,ISMCTS:200")
    ap.add_argument("--opponent", default="TensThenTricks")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)

    t0 = time.perf_counter()
    rows = []
    print(f"transfer to the deferred-capture (Double Sir) variant, "
          f"{a.deals} deals x4\n", flush=True)
    for spec in [x.strip() for x in a.agents.split(",")]:
        std = match(spec, a.opponent, Hand, a.deals, 4242)
        var = match(spec, a.opponent, DeferredCaptureHand, a.deals, 4242)
        if std is None or var is None:
            print(f"  [skip] {spec}")
            continue
        row = {"agent": spec, "standard": std["win"], "standard_ci": std["ci"],
               "variant_stale_model": var["win"],
               "variant_stale_model_ci": var["ci"],
               "drop_stale": round(std["win"] - var["win"], 4)}
        line = (f"  {spec:28s} standard {std['win']:.4f}   "
                f"variant(stale model) {var['win']:.4f}  "
                f"drop {std['win']-var['win']:+.4f}")
        # third arm: search agents told the truth about the rule change
        if spec.startswith(("PIMC:", "ISMCTS:")):
            with variant_aware_search():
                aware = match(spec, a.opponent, DeferredCaptureHand, a.deals, 4242)
            if aware:
                row["variant_correct_model"] = aware["win"]
                row["variant_correct_model_ci"] = aware["ci"]
                row["value_of_knowing_the_rule"] = round(
                    aware["win"] - var["win"], 4)
                line += (f"   variant(correct model) {aware['win']:.4f}"
                         f"  re-planning buys {aware['win']-var['win']:+.4f}")
        rows.append(row)
        print(line, flush=True)

    el = time.perf_counter() - t0
    json.dump({"env": runenv.snapshot(), "deals": a.deals,
               "opponent": a.opponent, "results": rows, "seconds": el,
               "variant": ("Double Sir / deferred capture: tricks accumulate in "
                           "the centre and are collected only by winning two "
                           "consecutive tricks; the winner of the 13th trick "
                           "takes whatever remains. Documented in Pagat and "
                           "recorded in 01-rulebooks/comparative-identification.md."),
               "reading": ("NO AGENT WAS RETRAINED. Three arms: (1) standard "
                           "rules; (2) the variant with a STALE model -- every "
                           "agent, search included, still assumes immediate "
                           "capture, which is what happens when a deployed "
                           "agent meets a changed rule; (3) for search only, "
                           "the variant with a CORRECT model, which isolates "
                           "what re-planning under the true rule is worth. "
                           "Learned agents have no arm (3): they cannot re-plan "
                           "at all, and that asymmetry is the finding.")},
              open("variant_transfer.json", "w"), indent=2)
    print(f"\n{el:.0f}s -> variant_transfer.json")
