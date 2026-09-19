# Phase 2 Findings — Rule-Based Agents

All results from duplicate-deal, seat-balanced tournaments;
CIs are percentile bootstrap over **deals** (the independent unit), 2000 resamples.

---

## Finding 1 — The two objectives are in direct tension, and they nearly cancel

`TenAware` (pure ten-hunting, implementing the "8 of hearts" discipline) versus
`GreedyTricks` (pure trick-maximising control), **10,000 hands**:

| | TenAware |
|---|---|
| Overall win rate | **0.5024** — a dead tie |
| Mean tens captured | **2.286** of 4 |
| Mean tricks won | **5.41** of 13 |
| Wins among hands decided by **tens** (68.4% of hands) | **0.6460** |
| Wins among hands decided by the **W5 tiebreak** (31.6%) | **0.1922** |
| Coats | 1412 vs 546 |

**Read that carefully.** Ten-hunting wins the tens battle decisively (0.646) and loses the
tiebreak battle just as decisively (0.192). At the game's natural ~32% tiebreak rate the two
effects cancel almost exactly. Neither pure strategy dominates.

This is a quantified structural property of Dahal Jeet, and it is the central Phase 2 result.

### Why trick-maximising is stronger than it looks
`GreedyTricks` was designed as a control that *should* underperform. It did not — it topped the
initial leaderboard (0.5888 mean win rate). The reason is mechanical: **tens arrive inside
tricks.** Taking more tricks captures more tens incidentally, so trick count is a decent proxy for
the real objective. Against Random, `GreedyTricks` averaged 2.481 tens without ever targeting them.

*The control failing to fail is itself a finding, and it is recorded rather than quietly dropped.*

---

## Finding 2 — Mode-switching too late does nothing

`Adaptive` switched to trick-maximising only once all four tens had been played.
Result vs `TenAware`: **0.5000** — literally no effect. By the time the tens are settled there
are too few tricks left to matter.

---

## Finding 3 — The first author's correction, and the agent that beat everything

The first author supplied the fix:

> "if i can win just two tens my secondary aim should be always most tricks"

That is, the trick race is **not** a fallback to switch into — it is a standing secondary
objective, live from trick one, because a 2–2 split is the single most likely outcome.

`TensThenTricks` implements exactly that: tens first, tricks always second, never a late switch.
It is the first agent to beat `GreedyTricks`.

| Matchup | Win rate | 95% CI | Cohen's *d* |
|---|---|---|---|
| TensThenTricks vs **GreedyTricks** | **0.5628** | [0.5528, 0.5726] | +0.25 |
| TensThenTricks vs TenAware | 0.5730 | [0.5622, 0.5834] | +0.28 |
| TensThenTricks vs Adaptive | 0.5740 | [0.5634, 0.5844] | +0.28 |
| TensThenTricks vs AdaptiveGreedy | 0.5560 | [0.5460, 0.5666] | +0.21 |
| TensThenTricks vs PartnerAware | 0.5568 | [0.5464, 0.5676] | +0.21 |
| TensThenTricks vs Random | 0.7366 | [0.7289, 0.7444] | +1.15 |

Every interval is clear of 0.5.

---

## Finding 4 — Independent confirmation by metaheuristic search

`ParamAgent` exposes the strategy as a 7-parameter genome, tuned by hill-climbing with restarts
against a fixed opponent panel (Malla 2026 §5.5 item 5 lists this as future work).

**The optimiser was given no hint about the first author's principle.** Across independent restarts
it converged on the same shape:

| Restart | Fitness | `late_switch` | `trump_on_empty_trick` | `free_trick_max_rank` |
|---|---|---|---|---|
| 0 | 0.5987 | 3 | 2 (always) | 0 |
| 1 | **0.6119** | **2** | 2 (always) | 8 |
| 2 | 0.6096 | **2** | 2 (always) | 9 |

`late_switch = 2` means *start maximising tricks at trick two* — the search independently
rediscovered "the secondary aim should be always most tricks."

**A player's tacit strategic knowledge and an automated parameter search converged on
the same policy.** That is a genuinely nice result for a cultural-game-preservation paper: it
gives an objective, reproducible check on the first author's domain knowledge.

---

## Finding 5 — Search latency is not a problem here

| Agent | ms / decision |
|---|---|
| PIMC (8 worlds) | **6.1** |
| ISMCTS (120 iterations) | **20.1** |
| *Malla 2026, ISMCTS on Dhumbal* | *1444.7 (stated limitation)* |

Two reasons ours is cheap: Dahal Jeet's action space is at most 13 cards and usually far fewer
under the follow-suit constraint (P2), and **void-constrained determinization** (P2 ⇒ provable
voids) means samples are consistent with the play history instead of mostly impossible.

Malla used three unconstrained determinizations. Constraining by voids costs almost nothing and
makes every rollout informative — a concrete, measurable improvement on the reference method.

*Cross-category championship results pending — see `championship_results.json`.*

---

## Caveats, stated plainly
- All results are **agent-vs-agent**. Nothing here says how any of this plays against people.
  That is Phase 6, and it is the gap the reference paper also left open.
- The heuristics are hand-written by someone who has never played Dahal Jeet. `TensThenTricks`
  encodes a stated strategy, not demonstrated expert play.
- `ParamAgent` was tuned against a fixed panel, so its fitness is panel-relative and will be
  partly overfitted to those five opponents. The championship re-tests it against fresh opponents.
- The ruleset rests on the first author's own long-term play across 7 Terai-Madhesh
  districts, not a single source as earlier drafts stated.
