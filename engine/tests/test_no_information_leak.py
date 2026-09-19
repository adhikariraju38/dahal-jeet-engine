"""Prove that no agent can see hidden information.

If an agent could see the opponents' cards, every result in this project would
be worthless. Asserting "PlayerView only contains public fields" is not enough
-- a leak could hide in a helper, in encode(), or in determinize().

The definitive test is behavioural:

    Take a real game position. Redistribute the UNSEEN cards among the
    opponents in a different but equally consistent way. The acting seat's
    information set is unchanged, so a sound agent MUST make the same move.

Any agent whose move changes is reading something it should not be able to see.
"""
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dahaljeet.agents import REGISTRY                       # noqa: E402
from dahaljeet.cards import suit_of                          # noqa: E402
from dahaljeet.determinize import determinize                # noqa: E402
from dahaljeet.encode import encode, legal_mask              # noqa: E402
from dahaljeet.hand import Hand                              # noqa: E402
from dahaljeet.search import ISMCTSAgent, PIMCAgent          # noqa: E402
from dahaljeet.view import PlayerView, make_view             # noqa: E402


def position(seed, plies):
    """A real mid-hand position with the acting seat's view."""
    rng = random.Random(seed)
    h = Hand(dealer=seed % 4, rng=random.Random(seed))
    h.deal()
    for _ in range(plies):
        if h.is_over:
            break
        h.play(rng.choice(h.legal_moves()))
    return h


def reshuffle_hidden(h, me, rng):
    """Redistribute unseen cards among the other seats, keeping the position
    consistent: same hand sizes, same voids, same played cards, same own hand."""
    v = make_view(h, me)
    d = determinize(v, rng, tries=200)
    if d is None:
        return None
    h2 = Hand(dealer=h.dealer)
    h2.hands = [sorted(x) for x in d]
    h2.trump_suit, h2.trump_card = h.trump_suit, h.trump_card
    h2.trump_holder = h.trump_holder
    h2.trick = list(h.trick)
    h2.trick_lead_suit = h.trick_lead_suit
    h2.tricks_played = h.tricks_played
    h2.tens_by_team = list(h.tens_by_team)
    h2.tricks_by_team = list(h.tricks_by_team)
    h2.played_by_seat = [list(x) for x in h.played_by_seat]
    h2.voids = [set(x) for x in h.voids]
    h2.ten_owner = dict(h.ten_owner)
    h2.to_act, h2.leader = h.to_act, h.leader
    h2.trick_winners = list(h.trick_winners)
    return h2


class ViewIsPublicOnly(unittest.TestCase):

    def test_view_carries_no_other_hand(self):
        """No field of PlayerView may contain another seat's holding."""
        h = position(3, 9)
        me = h.to_act
        v = make_view(h, me)
        others = set()
        for s in range(4):
            if s != me:
                others |= set(h.hands[s])
        def cards_in(name, val):
            """Extract only genuine CARD ids -- seat indices are not cards."""
            if name in ("hand", "legal"):
                return set(val)
            if name == "trick":                      # (seat, card) pairs
                return {c for _, c in val}
            if name == "played_by_seat":             # tuple of per-seat tuples
                return {c for row in val for c in row}
            return set()

        for name in ("hand", "legal", "trick", "played_by_seat"):
            leak = cards_in(name, getattr(v, name)) & others
            self.assertFalse(
                leak, f"field {name} leaks opponents' cards {sorted(leak)}")

        # played_by_seat is public by definition: every card in it has been
        # played, so it cannot overlap anyone's REMAINING holding.
        for s in range(4):
            self.assertFalse(set(v.played_by_seat[s]) & set(h.hands[s]),
                             "a played card is still in hand")

    def test_unseen_is_exactly_the_unknown_cards(self):
        """`unseen` must equal the true hidden set -- no more, no less."""
        for seed in range(20):
            h = position(seed, 7 + seed % 11)
            if h.is_over:
                continue
            me = h.to_act
            v = make_view(h, me)
            truth = set()
            for s in range(4):
                if s != me:
                    truth |= set(h.hands[s])
            self.assertEqual(set(v.unseen), truth,
                             "unseen must be exactly the opponents' cards")


class AgentsCannotSeeHiddenCards(unittest.TestCase):
    """The behavioural proof."""

    HEURISTICS = ["Random", "Lowest", "GreedyTricks", "TenHunter",
                  "TenDefender", "PartnerAware", "TenAware", "Balanced",
                  "Adaptive", "AdaptiveGreedy", "TensThenTricks"]

    def test_heuristics_are_invariant_to_hidden_cards(self):
        checked = 0
        for name in self.HEURISTICS:
            for seed in range(12):
                h = position(seed * 7 + 1, 5 + seed % 15)
                if h.is_over:
                    continue
                me = h.to_act
                rng = random.Random(seed)
                h2 = reshuffle_hidden(h, me, rng)
                if h2 is None:
                    continue
                # confirm the worlds really do differ
                if all(sorted(h.hands[s]) == sorted(h2.hands[s])
                       for s in range(4) if s != me):
                    continue
                a1 = REGISTRY[name](rng=random.Random(0))
                a2 = REGISTRY[name](rng=random.Random(0))
                m1 = a1.act(make_view(h, me))
                m2 = a2.act(make_view(h2, me))
                self.assertEqual(
                    m1, m2,
                    f"{name} changed its move when only HIDDEN cards changed "
                    f"-- it is reading opponents' hands")
                checked += 1
        self.assertGreater(checked, 40, "not enough positions actually tested")

    def test_search_agents_are_invariant_to_hidden_cards(self):
        """Same test for PIMC/ISMCTS, with the RNG pinned so any difference
        must come from the world, not from sampling."""
        checked = 0
        for seed in range(6):
            h = position(seed * 13 + 5, 6 + seed % 9)
            if h.is_over:
                continue
            me = h.to_act
            h2 = reshuffle_hidden(h, me, random.Random(seed))
            if h2 is None:
                continue
            if all(sorted(h.hands[s]) == sorted(h2.hands[s])
                   for s in range(4) if s != me):
                continue
            for mk in (lambda: PIMCAgent(worlds=4, rng=random.Random(99)),
                       lambda: ISMCTSAgent(iterations=40, rng=random.Random(99))):
                m1 = mk().act(make_view(h, me))
                m2 = mk().act(make_view(h2, me))
                self.assertEqual(m1, m2,
                                 "search agent saw the real hidden hands")
                checked += 1
        self.assertGreater(checked, 4)

    def test_encoding_is_invariant_to_hidden_cards(self):
        """The RL observation must be identical too, or the network learns
        from a leak."""
        checked = 0
        for seed in range(15):
            h = position(seed * 5 + 2, 4 + seed % 13)
            if h.is_over:
                continue
            me = h.to_act
            h2 = reshuffle_hidden(h, me, random.Random(seed))
            if h2 is None:
                continue
            if all(sorted(h.hands[s]) == sorted(h2.hands[s])
                   for s in range(4) if s != me):
                continue
            self.assertEqual(encode(make_view(h, me)), encode(make_view(h2, me)),
                             "encode() differs on identical information sets")
            self.assertEqual(legal_mask(make_view(h, me)),
                             legal_mask(make_view(h2, me)))
            checked += 1
        self.assertGreater(checked, 8)


class DeterminizationIsHonest(unittest.TestCase):

    def test_determinize_never_reproduces_the_truth_by_peeking(self):
        """determinize() takes only a view. Sampled worlds must be consistent
        with public info, but must NOT systematically match the real deal."""
        matches = 0
        trials = 0
        rng = random.Random(4)
        for seed in range(40):
            h = position(seed, 6)
            if h.is_over:
                continue
            me = h.to_act
            v = make_view(h, me)
            d = determinize(v, rng)
            if d is None:
                continue
            trials += 1
            if all(sorted(d[s]) == sorted(h.hands[s])
                   for s in range(4) if s != me):
                matches += 1
        self.assertGreater(trials, 20)
        # With ~10^13 consistent worlds, matching the truth even once would be
        # astronomically unlikely unless it is peeking.
        self.assertEqual(matches, 0,
                         "determinize() reproduced the real deal -- it peeks")

    def test_determinize_respects_public_constraints(self):
        rng = random.Random(11)
        for seed in range(30):
            h = position(seed, 12)
            if h.is_over:
                continue
            me = h.to_act
            v = make_view(h, me)
            d = determinize(v, rng)
            if d is None:
                continue
            self.assertEqual(sorted(d[me]), sorted(v.hand))
            allc = sorted(c for x in d for c in x)
            self.assertEqual(len(set(allc)), len(allc))
            for s in range(4):
                self.assertEqual(len(d[s]), 13 - len(v.played_by_seat[s]))
                for c in d[s]:
                    self.assertNotIn(suit_of(c), v.voids[s])


if __name__ == "__main__":
    unittest.main(verbosity=2)
