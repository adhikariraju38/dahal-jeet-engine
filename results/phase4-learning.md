# Phase 4 Findings — Learning Agents and the Reward Ablation

9,000 training episodes per run (20,000 for MAPPO).
Evaluation: duplicate deals, seat-balanced, **4,000 hands per cell**, bootstrap CIs over deals.

---

## Why this phase exists

Malla 2026 §5.4.1, verbatim:
> the learning agents "did not surpass the random baseline… the baseline state representation and
> reward design were insufficient for stable policy learning."

That is a **design failure, not a fact about RL.** So we fixed the state representation (documented
field-by-field in `dahaljeet/encode.py`, 360 dims, seat-relative) and made the **reward scheme the
independent variable** rather than a guess.

---

## Result 1 — All learning agents beat random

Every configuration clears the random baseline. **This alone exceeds the reference paper**, where
DQN and PPO fell *below* random.

| Algo | Reward | vs Random | 95% CI | vs TensThenTricks |
|---|---|---|---|---|
| DQN | **ten_shaped** | **0.6118** | [0.5985, 0.6248] | 0.3420 |
| DQN | **potential** | **0.6008** | [0.5877, 0.6140] | **0.3440** |
| DQN | trick_shaped *(control)* | 0.5803 | [0.5663, 0.5938] | 0.3175 |
| DQN | terminal | 0.5565 | [0.5427, 0.5687] | 0.2935 |
| **MAPPO** (CTDE self-play) | potential | **0.5942** | — | **0.3342** |
| PPO | ten_shaped | 0.5583 | [0.5445, 0.5723] | 0.2820 |
| PPO | potential | 0.5475 | [0.5337, 0.5610] | 0.2480 |
| PPO | terminal | 0.5282 | [0.5152, 0.5410] | 0.2775 |
| PPO | trick_shaped | 0.5268 | [0.5145, 0.5393] | 0.2780 |

---

## Result 2 — Aligned shaping ≫ sparse terminal reward

DQN head-to-head, 3,200 hands each:

| Matchup | Win rate | 95% CI | Verdict |
|---|---|---|---|
| terminal vs ten_shaped | 0.4381 | [0.4188, 0.4581] | **loses** |
| terminal vs potential | 0.4619 | [0.4425, 0.4831] | **loses** |
| terminal vs trick_shaped | 0.5000 | [0.4794, 0.5200] | tie |
| ten_shaped vs potential | 0.4950 | [0.4750, 0.5150] | tie |
| ten_shaped vs trick_shaped | 0.5350 | [0.5150, 0.5550] | **wins** |
| potential vs trick_shaped | 0.5587 | [0.5394, 0.5775] | **wins** |

Grouping: **{ten_shaped ≈ potential} > {trick_shaped ≈ terminal}**.

Aligned shaping — whether principled (potential-based, policy-invariant) or naive (+0.25 per ten)
— beats both sparse terminal reward and misaligned shaping. Notably the two aligned schemes are
**indistinguishable**, so the theoretical guarantee of potential-based shaping bought no measurable
advantage here over the simpler heuristic shaping.

---

## Result 3 — The control failed to fail, and that is the interesting part

`trick_shaped` was designed to underperform: it rewards trick count, and **13.6% of hands are won
by the side taking fewer tricks** (Phase 2). The prediction was that it would come last.

**It did not.** It beat sparse terminal reward (0.5803 vs 0.5565) and tied it head-to-head.

**Honest interpretation:** at this training budget, *dense-but-misaligned* feedback is worth about
as much as *sparse-but-correct* feedback. The value of frequent credit assignment roughly cancels
the cost of optimising a subtly wrong objective. Only when shaping is **both dense and aligned**
does it clearly win.

This is a more useful finding than the one predicted, and a real caution for reward design in
sparse-reward card games: a shaping signal being "obviously reasonable" is not sufficient — it has
to track the actual win condition, and the failure mode is silent.

---

## Result 4 — CTDE self-play did not beat single-agent DQN

`MAPPO` (shared actor across both partner seats, centralised critic over both partners'
observations, 80% self-play against a frozen snapshot) reached **0.5942** vs Random and **0.3342**
vs TensThenTricks — statistically indistinguishable from the best DQN (0.3440).

Malla §5.5 item 4 proposes multi-agent RL as future work. We implemented it. **At this budget it
delivers no measurable gain.** Reported as-is rather than dropped.

Two candidate explanations, neither tested here:
- 20,000 episodes is small for self-play; the frozen-snapshot curriculum may not have advanced far.
- The partnership coordination signal may be weak in this game: partners cannot communicate, and
  much of the value may be capturable by a good individual policy.

---

## Result 5 — PPO is undertrained, and says so

PPO's two evaluations **disagree**. Against Random, `ten_shaped` (0.5583) beats `terminal` (0.5282).
Head-to-head, `terminal` *beats* `ten_shaped` (0.5294 [0.5100, 0.5494]).

Inconsistent orderings across evaluation methods are a signature of high-variance, undertrained
policies. **PPO's reward ordering should not be reported as a finding.** Only DQN's ablation is
trustworthy at this budget.

---

## Where learning lands overall

```
search  (ISMCTS 0.66, PIMC 0.63)
   >  heuristics  (Tuned 0.61, TensThenTricks 0.60)
      >  learning  (DQN/MAPPO ~0.34 vs the best heuristic)
         >  random
```

RL clears random but does not approach hand-written heuristics, let alone search. Stated plainly
rather than framed as a partial success.

## Caveats
- 9,000 episodes (20,000 for MAPPO) is a modest budget. These are **baselines, not ceilings** —
  the same caution Malla applied to his own RL results, and it applies to ours.
- All learning ran against a fixed strong opponent (TensThenTricks) except MAPPO. Learning from
  scratch against a strong opponent gives very sparse wins; a curriculum was not tested.
- Single seed per configuration. Multi-seed runs are needed before print.
- No hyperparameter search was performed. Architecture and learning rate were fixed across all
  runs so the reward comparison stays controlled.
