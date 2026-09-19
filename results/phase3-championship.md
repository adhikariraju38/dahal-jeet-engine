# Phase 3 Results — Cross-Category Championship

Duplicate deals, seat-balanced, bootstrap CIs over deals.
Rule-based round robin: 1200 deals × 4 rotations = **4800 hands per matchup**.
Search matchups: 130 deals × 4 = **520 hands per matchup**. Total runtime 2884 s.

---

## Rule-based leaderboard

| # | Agent | Mean win rate |
|---|---|---|
| 1 | **Tuned** (hill-climbed genome) | **0.6147** |
| 2 | **TensThenTricks** (the first author's strategy) | **0.5979** |
| 3 | GreedyTricks | 0.5446 |
| 4 | AdaptiveGreedy | 0.5446 |
| 5 | Balanced | 0.5281 |
| 6 | TenAware | 0.5268 |
| 7 | Adaptive | 0.5243 |
| 8 | PartnerAware | 0.5193 |
| 9 | TenHunter | 0.4514 |
| 10 | TenDefender | 0.4363 |
| 11 | Random | 0.3571 |
| 12 | Lowest | 0.3550 |

The top two are the two agents derived from the first author's stated strategy — one written by
hand, one found by parameter search. Both beat the trick-maximising control comfortably.

---

## Search vs the best heuristics

| Agent | Opponent | Win rate | 95% CI | *d* | ms/decision |
|---|---|---|---|---|---|
| **ISMCTS** (250 it.) | GreedyTricks | 0.6731 | [0.635, 0.712] | +0.78 | 41.4 |
| **ISMCTS** | **TensThenTricks** | **0.6615** | [0.621, 0.702] | +0.72 | 42.4 |
| **ISMCTS** | **Tuned** | **0.6442** | [0.610, 0.679] | +0.70 | 41.6 |
| **ISMCTS** | Random | 0.8635 | [0.831, 0.894] | +2.01 | 42.9 |
| PIMC (12 worlds) | GreedyTricks | 0.6885 | [0.656, 0.721] | +0.95 | 9.1 |
| PIMC | TensThenTricks | 0.6269 | [0.590, 0.664] | +0.61 | 9.4 |
| PIMC | Tuned | 0.6192 | [0.585, 0.654] | +0.58 | 9.5 |
| PIMC | Random | 0.8019 | [0.769, 0.833] | +1.65 | 9.4 |

**Search beats every heuristic, decisively, and cheaply.** Every interval is clear of 0.5.

---

## Why search is cheap here

ISMCTS reaches 0.66 against the best heuristic at 42 ms per decision, and PIMC 0.63 at 9 ms.
Both sit far inside an interactive budget. Two mechanical reasons:

1. **Small action space.** At most 13 cards, and usually far fewer once P2's follow-suit
   constraint applies. Search cost in trick-taking games is dominated by branching factor.
2. **Void-constrained determinization.** P2 makes voids *provable*, so sampled worlds are
   consistent with the observed play rather than mostly impossible: **300/300 consistent samples,
   zero rejections**.

These latencies are not comparable to figures from other games. Search cost is a property of
the game's action space, so a millisecond count from a different game is not a baseline, and
an apparent "speedup" against one would be an artifact of studying an easier game.

## An open question worth raising in the discussion

Heuristics and search rank differently across imperfect-information card games. Here search wins
decisively. Whether that is driven by the partnership structure, the point-trick scoring, the
action-space size, or something else is **not settled by this study** — it needs a controlled
comparison across games, which we have not done.

## Caveats
- Search matchups used 520 hands each — narrower intervals need more, but every CI already
  excludes 0.5.
- `Tuned` was hill-climbed against a fixed five-agent panel, so it is partly overfitted to it.
  It was re-tested here against fresh opponents and held up, but this should be stated.
- All results are agent-vs-agent. Nothing here speaks to play against humans (Phase 6).
- ISMCTS at 250 iterations and PIMC at 12 worlds are single points. The latency-vs-strength
  curve promised in the plan is still to be run.
