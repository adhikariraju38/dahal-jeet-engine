# Dahal Jeet Engine — Phase 0

Reference implementation of the single-hand variant.
Spec: the rule specification in the paper's Methods.

**Pure Python 3 stdlib — no dependencies.** (The environment has no numpy/torch;
Phases 0–3 are deliberately built to run without them.)

## Layout
```
dahaljeet/
  cards.py     card ints 0-51, suits, ranks, the four tens
  hand.py      one hand: deal, 13 tricks, scoring (S/D/T/P/C/W rules)
  session.py   dealer rotation R0-R4, the tally G1-G3
tests/
  test_rules.py   31 tests, one or more per rule ID
```

## Run
```bash
cd 05-engine
python3 -m unittest discover -s tests -v
```

## Usage
```python
import random
from dahaljeet import Hand, Session, card_name

h = Hand(dealer=0, rng=random.Random(7))
h.deal()
print(card_name(h.trump_card), "is trump; seat", h.trump_holder + 1, "leads")
while not h.is_over:
    h.play(random.choice(h.legal_moves()))
print(h.result().describe())     # -> "team 0 COAT (tens 4-0, tricks 11-2)"

s = Session(first_dealer=0, rng=random.Random(3))
s.play_hands(100, lambda hd: random.choice(hd.legal_moves()))
print(s.tally.describe())
```

Seats are **0–3** in code and **1–4** in the rulebook (code seat *i* = rulebook seat *i*+1).

## Design notes

- **Cards are ints.** `suit = c // 13`, `rank = c % 13`, rank 0 = Two … 12 = Ace, so a larger
  int is a higher card *within a suit* (S2). The four tens are `{8, 21, 34, 47}`.
- **Illegal moves are impossible, not penalised.** `legal_moves()` enforces P2/P3, and `play()`
  raises on anything else. RL agents therefore mask rather than learn legality.
- **`Session` has no terminal state.** Deliberate — rule **G2**: the real game never ends. There is
  no `is_over` and no `winner` on `Session`, and a test asserts their absence. Episode length is
  the caller's choice and a *modelling artifact* that the paper must declare.
- **Void inference is built in.** `hand.voids[seat]` is the set of suits that seat has provably run
  out of, derived from P2. Phase 3 determinization uses this to reject inconsistent samples
  cheaply — Malla 2026 did not do this, and it is a concrete improvement over that baseline.
- **Rotation is one function.** `next_dealer(dealer, losing_team, coat, double_coat)` implements
  R1–R4. It is the part most likely to be subtly wrong, so it is tested exhaustively over all
  4 × 2 × 2 × 2 combinations *and* against the rulebook's own table.

## Status against the Phase 0 acceptance criteria

| Criterion | Target | Actual |
|---|---|---|
| Rotation table reproduced | 6 cases | **7/7** (added a seat-3 case) |
| Rule IDs covered by a named test | 100 % | **29/30** — see below |
| Speed, single-threaded | ≥ 20 000 hands/s | **11 400 hands/s** (~149 k tricks/s) |
| Invariant violations over 20 k hands | 0 | **0** |
| Tests passing | all | **31/31** |

**On the two shortfalls, stated plainly:**

- **G4 has no test** because it has no mechanical content — "played for fun, not for money" is
  context for the write-up, not behaviour the engine can exhibit. Every rule with
  mechanical content (29/29) is covered.
- **Speed is 11.4 k/s, not the 20 k/s I targeted.** Adequate for Phases 1–3 (1 M hands ≈ 90 s) and
  RL will be compute-bound elsewhere, but the target was missed. Known costs: `legal_moves()`
  allocates a list per call and `list.remove()` is O(n). Worth revisiting before large-scale
  training, not before.

## Baseline statistics under uniform-random play

40 000 hands, seeded. These are the numbers every later agent must beat.

| Quantity | Value |
|---|---|
| Team-0 win rate | 0.4969 — balanced, as it must be |
| Tens split 2–2 | **0.342** |
| Coat rate (all four tens) | 0.161 |
| Double coat rate | 0.0014 (56 in 40 000) |
| **Hands won with fewer than 7 tricks** | **0.136** |

Two of these matter for the paper:

1. **The W5 tiebreak fires in ~34 % of hands.** Tens split 2–2 is not a corner case — it decides
   a third of all play. Any agent that ignores trick count is throwing away a third of the game.
2. **13.6 % of hands are won by the side taking fewer tricks.** Direct empirical support for the
   claim that trick count is the wrong objective (W1). This is the number to quote when motivating
   the reward-design experiments in Phase 4.

## Next: Phase 1
Baseline agents and the duplicate-deal tournament harness — see
the paper's Methods.
