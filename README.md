# Dahal Jeet — game engine, agents and experimental record

Dahal Jeet is a four-player, fixed-partnership, trick-taking card game played
across the Terai-Madhesh region of Nepal, in which **only the four tens score**.
Its objective is lexicographic rather than additive: a majority of tens decides
most hands, and trick count settles the rest. It belongs to the documented
Mendikot branch of the Court Piece family, but the Nepali variant had no
published rule description.

This repository holds the formal rule specification, a reference engine, 26
agents spanning hand-written heuristics through determinized search and seven
families of learning agent, every trained model, and the complete result
artifacts behind the paper.

> **Paper status.** The accompanying manuscript, *"Search-based agents
> outperform reinforcement learning in Dahal Jeet, a Nepali
> imperfect-information card game"*, is **under review at Scientific Reports**.
> This repository is released so the results can be inspected and reproduced
> during and after review. Citation details will be updated on publication.

## What is here

| | |
|---|---|
| `rulebook/` | The 26-feature comparison against Mendikot, in machine-readable form, that places the game in the Court Piece family. The rule specification itself is in the paper's Methods |
| `engine/` | The game engine, all 26 agents, every experiment script, and the result artifacts they produced |
| `models/` | Every trained checkpoint, with a manifest |
| `results/` | Phase-by-phase findings written up as the work proceeded |
| `figures/` | The figures in the paper, and the script that draws them |
| `compute/` | The full compute and environment record |

The engine keeps code and artifacts in one directory on purpose: the analysis
scripts load artifacts by filename, and separating them would break
reproduction for no gain.

## Headline results

All 26 agents played a complete round-robin of 325 pairings at 300 duplicate
deals each, 390,000 hands in total.

- **Determinized search leads.** ISMCTS at 250 iterations tops the table at Elo
  1663.2, with PIMC second.
- **No learning method reaches a tuned heuristic.** The metaheuristically tuned
  heuristic takes third, ahead of every network trained here.
- **Search plays endgames close to optimally.** Against an exact double-dummy
  solver on a common position set, PIMC converts 0.923 of theoretically won
  endgames against 0.810 for the best heuristic.
- **A minimal state encoding beats a rich hand-designed one** by 0.0644 win
  rate, with no seed overlap — against the intent of the design.
- **Every deterministic heuristic is exploitable** by a search that knows its
  policy; random play is not.
- **Much of the game is luck.** The deal alone explains 29–39% of outcome
  variance, and trump length predicts the outcome far more strongly than the
  number of tens dealt.

## Reproducing

```sh
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
cd engine
```

Every number in the paper is generated, never typed:

```sh
python paper_numbers.py     # every reported value, as LaTeX macros
python paper_tables.py      # every results table
python verify_equations.py  # 30 manuscript identities against the artifacts
```

One artifact is not shipped: `engine/expert_data.pkl`, the 368 MB of
(state, expert-action) pairs used to distil a policy from ISMCTS. It exceeds
GitHub's file-size limit and is regenerated in one command:

```sh
python gen_expert_data.py
```

`verify_equations.py` re-runs no experiment. It reads the stored artifacts and
recomputes the identities the paper asserts; expect **30/30**.

To re-run the experiments themselves, start from `run_full.sh` and
`long_runs.sh`. They take a long time: the recorded results are **134.9
CPU-hours** across **81 artifacts**, all on the CPU of a single laptop with no
GPU (Apple M1 Pro, 10 logical cores, 16 GiB; Python 3.14.5, torch 2.13.0,
numpy 2.5.2). `compute/compute.json` has the per-experiment breakdown.

## Licence

This repository is dual-licensed, which is the usual arrangement for a
release that is part software and part data.

- **Code** — everything under `engine/`, and any script elsewhere — is released
  under the **MIT Licence**. See [`LICENSE`](LICENSE). MIT is approved by the
  Open Source Initiative, which is what Nature Portfolio asks for when custom
  code underpins a paper's conclusions.
- **Data, models, figures and the rule specification** — `models/`,
  `figures/`, `rulebook/`, `results/`, `compute/`, and the result artifacts in
  `engine/` — are released under **Creative Commons Attribution 4.0
  International (CC BY 4.0)**. See [`LICENSE-DATA`](LICENSE-DATA). CC BY 4.0
  matches the licence Scientific Reports publishes articles under, so the
  paper and the data it rests on can be reused on the same terms.

Both permit commercial use and redistribution; both require attribution.

## Citing

See [`CITATION.cff`](CITATION.cff). Please cite the paper as well as this
repository; the paper reference will be updated when it is published.

## Authors

**Raju Kumar Yadav** ([ORCID 0009-0009-4811-246X](https://orcid.org/0009-0009-4811-246X))
· **Ganesh Gautam** ([ORCID 0000-0002-1926-1492](https://orcid.org/0000-0002-1926-1492)),
corresponding author

Department of Electronics and Computer Engineering, Institute of Engineering,
Pulchowk Campus, Tribhuvan University, Lalitpur, Nepal
