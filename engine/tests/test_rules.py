"""Spec-compliance tests. Every rule ID in the rulebook gets a named test.

Run from 05-engine/:  python3 -m unittest discover -s tests -v
Rulebook: ../rulebook/dahal-jeet-rulebook.html (draft 05)
"""
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dahaljeet import (                                    # noqa: E402
    Hand, Session, next_dealer, PARTNER, team_of,
    TENS, N_CARDS, HAND_SIZE, suit_of, parse_card, RANKS,
)


def fresh(dealer=0, seed=0):
    h = Hand(dealer=dealer, rng=random.Random(seed))
    h.deal()
    return h


def play_out(h, rng):
    while not h.is_over:
        h.play(rng.choice(h.legal_moves()))
    return h.result()


class Setup(unittest.TestCase):

    def test_S1_partnerships_are_opposite(self):
        """S1 - teams are seats {0,2} and {1,3}; partners sit opposite."""
        self.assertEqual(PARTNER, (2, 3, 0, 1))
        for s in range(4):
            self.assertEqual(team_of(s), team_of(PARTNER[s]))
            self.assertNotEqual(team_of(s), team_of((s + 1) % 4))

    def test_S2_pack_is_52_with_ace_high(self):
        """S2 - 52 cards, A K Q J 10 ... 2 high to low within a suit."""
        self.assertEqual(N_CARDS, 52)
        self.assertEqual((RANKS[0], RANKS[-1]), ("2", "A"))
        self.assertGreater(parse_card("AS"), parse_card("KS"))
        self.assertGreater(parse_card("KS"), parse_card("10S"))
        self.assertGreater(parse_card("3S"), parse_card("2S"))
        self.assertEqual(TENS, {parse_card("10" + s) for s in "CDHS"})

    def test_S3_turn_order_is_cyclic_0123(self):
        """S3 - play proceeds 0 -> 1 -> 2 -> 3 -> 0."""
        h = fresh(seed=5)
        start = h.to_act
        seats = []
        for _ in range(4):
            seats.append(h.to_act)
            h.play(h.legal_moves()[0])
        self.assertEqual(seats, [(start + i) % 4 for i in range(4)])


class DealAndTrump(unittest.TestCase):

    def test_D1_D2_each_seat_gets_thirteen(self):
        """D1/D2 - 5 then 4 then 4; whole pack dealt, no duplicates."""
        h = fresh(seed=11)
        for s in range(4):
            self.assertEqual(len(h.hands[s]), HAND_SIZE)
        allc = sorted(c for seat in h.hands for c in seat)
        self.assertEqual(allc, list(range(N_CARDS)))

    def test_T1_T2_trump_holder_is_seat_after_dealer(self):
        """T1/T2 - the seat after the dealer supplies the trump card."""
        for seed in range(30):
            h = fresh(dealer=seed % 4, seed=seed)
            self.assertEqual(h.trump_holder, (h.dealer + 1) % 4)

    def test_T3_trump_suit_matches_shown_card(self):
        """T3 - the shown card's suit is trump, public to all."""
        for seed in range(30):
            h = fresh(seed=seed)
            self.assertEqual(h.trump_suit, suit_of(h.trump_card))

    def test_T4_T5_trump_card_stays_in_hand(self):
        """T4 - card returns to hand. T5 - so the holder always has a trump."""
        for seed in range(50):
            h = fresh(seed=seed)
            self.assertIn(h.trump_card, h.hands[h.trump_holder])
            trumps = [c for c in h.hands[h.trump_holder]
                      if suit_of(c) == h.trump_suit]
            self.assertGreaterEqual(len(trumps), 1)

    def test_P1_trump_holder_leads_first_trick(self):
        """P1 - the trump drawer leads."""
        for seed in range(20):
            h = fresh(dealer=seed % 4, seed=seed)
            self.assertEqual(h.leader, h.trump_holder)
            self.assertEqual(h.to_act, h.trump_holder)


class Play(unittest.TestCase):

    def test_P2_must_follow_suit_when_able(self):
        """P2 - holding the led suit, only that suit is legal."""
        checked = 0
        for seed in range(40):
            h = fresh(seed=seed)
            h.play(h.legal_moves()[0])
            seat = h.to_act
            holds = [c for c in h.hands[seat]
                     if suit_of(c) == h.trick_lead_suit]
            if holds:
                self.assertEqual(sorted(h.legal_moves()), sorted(holds))
                checked += 1
        self.assertGreater(checked, 0)

    def test_P2_illegal_move_is_rejected(self):
        """P2 - off-suit while holding the led suit must raise."""
        raised = 0
        for seed in range(40):
            h = fresh(seed=seed)
            h.play(h.legal_moves()[0])
            seat = h.to_act
            legal = set(h.legal_moves())
            offsuit = [c for c in h.hands[seat] if c not in legal]
            if offsuit:
                with self.assertRaises(ValueError):
                    h.play(offsuit[0])
                raised += 1
        self.assertGreater(raised, 0)

    def test_P3_void_player_may_play_anything(self):
        """P3 - unable to follow, the whole hand is legal (no forced trump)."""
        rng = random.Random(4)
        found = 0
        for seed in range(60):
            h = fresh(seed=seed)
            while not h.is_over:
                seat = h.to_act
                if h.trick and not any(suit_of(c) == h.trick_lead_suit
                                       for c in h.hands[seat]):
                    self.assertEqual(sorted(h.legal_moves()),
                                     sorted(h.hands[seat]))
                    found += 1
                    break
                h.play(rng.choice(h.legal_moves()))
        self.assertGreater(found, 0, "no void ever arose")

    def test_P4_highest_trump_else_highest_of_led_suit(self):
        """P4 - trick resolution, checked against an independent computation."""
        rng = random.Random(8)
        for seed in range(150):
            h = fresh(seed=seed)
            while not h.is_over:
                before = len(h.trick_winners)
                plays = []
                while len(h.trick_winners) == before:
                    seat = h.to_act
                    c = rng.choice(h.legal_moves())
                    plays.append((seat, c))
                    h.play(c)
                lead = suit_of(plays[0][1])
                trumps = [(s, c) for s, c in plays
                          if suit_of(c) == h.trump_suit]
                if trumps:
                    expect = max(trumps, key=lambda sc: sc[1])[0]
                else:
                    inled = [(s, c) for s, c in plays if suit_of(c) == lead]
                    expect = max(inled, key=lambda sc: sc[1])[0]
                self.assertEqual(h.trick_winners[-1], expect)

    def test_P5_winner_leads_next_trick(self):
        """P5 - the trick winner is on lead."""
        rng = random.Random(2)
        h = fresh(seed=33)
        while not h.is_over:
            n = len(h.trick_winners)
            h.play(rng.choice(h.legal_moves()))
            if len(h.trick_winners) > n and not h.is_over:
                self.assertEqual(h.to_act, h.trick_winners[-1])
                self.assertEqual(h.leader, h.trick_winners[-1])

    def test_C1_capture_is_immediate(self):
        """C1 - single-hand: counts land the moment the trick ends."""
        rng = random.Random(6)
        h = fresh(seed=44)
        while not h.is_over:
            n = len(h.trick_winners)
            h.play(rng.choice(h.legal_moves()))
            self.assertEqual(sum(h.tricks_by_team), len(h.trick_winners))
            if len(h.trick_winners) > n:
                self.assertEqual(sum(h.tricks_by_team), n + 1)

    def test_void_inference_is_sound(self):
        """Voids must be real - Phase 3 determinization depends on this."""
        rng = random.Random(17)
        for seed in range(40):
            h = fresh(seed=seed)
            recorded = [set() for _ in range(4)]
            while not h.is_over:
                seat, lead = h.to_act, h.trick_lead_suit
                held = set(h.hands[seat])
                c = rng.choice(h.legal_moves())
                h.play(c)
                if h.voids[seat] - recorded[seat]:
                    # engine just recorded a void; it must be true of `held`
                    for suit in h.voids[seat] - recorded[seat]:
                        self.assertFalse(
                            any(suit_of(x) == suit for x in held),
                            f"false void: seat {seat} suit {suit}")
                    recorded[seat] = set(h.voids[seat])
                del lead


class Scoring(unittest.TestCase):

    def test_W1_only_tens_score(self):
        """W1 - exactly four scoring cards; totals always 4 tens / 13 tricks."""
        rng = random.Random(3)
        for seed in range(300):
            r = play_out(fresh(seed=seed), rng)
            self.assertEqual(sum(r.tens_by_team), 4)
            self.assertEqual(sum(r.tricks_by_team), 13)

    def test_W4_majority_of_tens_wins(self):
        """W4 - 3-1 or 4-0 decides it regardless of trick count."""
        rng = random.Random(31)
        seen = 0
        for seed in range(400):
            r = play_out(fresh(seed=seed), rng)
            if r.tens_by_team[0] != r.tens_by_team[1]:
                seen += 1
                expect = 0 if r.tens_by_team[0] > r.tens_by_team[1] else 1
                self.assertEqual(r.winning_team, expect)
        self.assertGreater(seen, 0)

    def test_W5_two_two_split_decided_by_tricks(self):
        """W5 - tens 2-2, most tricks wins; 13 tricks means no draw."""
        rng = random.Random(77)
        seen = 0
        for seed in range(400):
            r = play_out(fresh(seed=seed), rng)
            if r.tens_by_team == (2, 2):
                seen += 1
                expect = 0 if r.tricks_by_team[0] > r.tricks_by_team[1] else 1
                self.assertEqual(r.winning_team, expect)
                self.assertNotEqual(r.tricks_by_team[0], r.tricks_by_team[1])
        self.assertGreater(seen, 0)

    def test_W2_W3_coat_and_double_coat(self):
        """W2 - coat is all four tens. W3 - double coat adds all 13 tricks."""
        rng = random.Random(5)
        saw_coat = 0
        for seed in range(1200):
            r = play_out(fresh(seed=seed), rng)
            self.assertEqual(r.coat, r.tens_by_team[r.winning_team] == 4)
            self.assertEqual(r.double_coat,
                             r.coat and r.tricks_by_team[r.winning_team] == 13)
            if r.double_coat:
                self.assertTrue(r.coat)
            saw_coat += r.coat
        self.assertGreater(saw_coat, 0)

    def test_W1_a_side_can_lose_with_most_tricks(self):
        """W1 - the point of the game: trick count is not the objective."""
        rng = random.Random(101)
        found = False
        for seed in range(3000):
            r = play_out(fresh(seed=seed), rng)
            w, l = r.winning_team, r.losing_team
            if r.tricks_by_team[l] > r.tricks_by_team[w]:
                found = True
                break
        self.assertTrue(found, "expected some hand won with fewer tricks")


class Rotation(unittest.TestCase):
    """R0-R4 - the rulebook's rotation table is the oracle."""

    ORACLE = {
        (0, 0, False, False): 0,   # 1+3 lose, ordinary       -> seat 1
        (0, 0, True,  False): 2,   # 1+3 lose, opponents coat -> seat 3
        (0, 0, True,  True):  0,   # 1+3 lose, double coat    -> seat 1
        (0, 1, False, False): 1,   # 1+3 win, ordinary        -> seat 2
        (0, 1, True,  False): 3,   # 1+3 win with coat        -> seat 4
        (0, 1, True,  True):  1,   # 1+3 win, double coat     -> seat 2
        (2, 1, True,  True):  3,   # seat 3 dealt, dbl coat   -> seat 4
    }

    def test_R1_R4_matches_rulebook_table(self):
        for args, expect in self.ORACLE.items():
            with self.subTest(args=args):
                self.assertEqual(next_dealer(*args), expect)

    def test_R1_dealer_always_from_losing_team(self):
        """R1 - exhaustive over all 4 x 2 x 2 x 2 combinations."""
        for dealer in range(4):
            for losing in range(2):
                for coat in (False, True):
                    for dc in (False, True):
                        nd = next_dealer(dealer, losing, coat, dc)
                        self.assertEqual(team_of(nd), losing)

    def test_R3_coat_skips_to_other_member(self):
        """R3 - a coat sends the deal to the partner of the R2 default."""
        for dealer in range(4):
            for losing in range(2):
                default = next_dealer(dealer, losing, False, False)
                skipped = next_dealer(dealer, losing, True, False)
                self.assertEqual(skipped, PARTNER[default])
                self.assertNotEqual(skipped, default)

    def test_R4_double_coat_cancels_the_skip(self):
        """R4 - a double coat reverts to the R2 default."""
        for dealer in range(4):
            for losing in range(2):
                default = next_dealer(dealer, losing, False, False)
                self.assertEqual(next_dealer(dealer, losing, True, True),
                                 default)

    def test_R0_any_seat_may_open(self):
        """R0 - no constraint on the first deal."""
        for d in range(4):
            self.assertEqual(Session(first_dealer=d).dealer, d)


class SessionRules(unittest.TestCase):

    def test_G2_session_has_no_terminal_state(self):
        """G2 - Session must not define its own stopping condition.

        The episode boundary is the caller's and is a modelling artifact.
        """
        self.assertFalse(hasattr(Session, "is_over"))
        self.assertFalse(hasattr(Session, "winner"))

    def test_G3_tally_records_wins_coats_and_losses(self):
        """G3 - the running tally."""
        rng = random.Random(12)
        s = Session(first_dealer=0, rng=rng)
        s.play_hands(200, lambda h: rng.choice(h.legal_moves()))
        t = s.tally
        self.assertEqual(sum(t.hands_won), 200)
        self.assertEqual(t.hands_won[0], t.hands_lost[1])
        self.assertEqual(t.hands_won[1], t.hands_lost[0])
        for i in (0, 1):
            self.assertLessEqual(t.double_coats[i], t.coats[i])
            self.assertLessEqual(t.coats[i], t.hands_won[i])

    def test_session_rotation_matches_next_dealer(self):
        """The session applies R1-R4 exactly as next_dealer states."""
        rng = random.Random(19)
        s = Session(first_dealer=0, rng=rng)
        for _ in range(120):
            before = s.dealer
            r = s.play_hand(lambda h: rng.choice(h.legal_moves()))
            self.assertEqual(s.dealer,
                             next_dealer(before, r.losing_team,
                                         r.coat, r.double_coat))

    def test_G1_losing_team_keeps_dealing(self):
        """G1/R1 - a team that keeps losing keeps dealing."""
        rng = random.Random(23)
        s = Session(first_dealer=0, rng=rng)
        for _ in range(300):
            before = s.dealer
            r = s.play_hand(lambda h: rng.choice(h.legal_moves()))
            if team_of(before) == r.losing_team and not r.coat:
                self.assertEqual(s.dealer, before,
                                 "dealer's team lost, so he must deal again")


class Determinism(unittest.TestCase):

    def test_same_seed_same_hand(self):
        a, b = fresh(seed=123), fresh(seed=123)
        self.assertEqual(a.hands, b.hands)
        self.assertEqual(a.trump_card, b.trump_card)

    def test_random_play_is_balanced(self):
        """Sanity: four random agents give ~50% per team. Bugs surface here."""
        rng = random.Random(2024)
        wins = [0, 0]
        for i in range(4000):
            h = Hand(dealer=i % 4, rng=rng)
            h.deal()
            wins[play_out(h, rng).winning_team] += 1
        self.assertAlmostEqual(wins[0] / 4000, 0.5, delta=0.03)


if __name__ == "__main__":
    unittest.main(verbosity=2)
