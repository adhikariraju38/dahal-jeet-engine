"""Void-constraint ablation for determinized search.

Writes void_constraint_ablation.json -- deliberately NOT named ablate_*.json,
which is the glob the ENCODING ablation owns. Sharing that prefix made
make_figures.py try to read this file as an encoding-ablation artifact and fail,
and would have filed it under the wrong experiment in the compute record.

`determinize.py` claims that constraining sampled worlds by inferred voids (P2)
is the right design. We have never measured it. This ablates our own design
choice.

    python ablate_determinize.py --diagnostic-only
    python ablate_determinize.py --deals 300

TWO PROTOCOLS, BOTH REPORTED

1. MATCHED SAMPLES -- identical world/iteration counts. Answers "is a
   constrained sample worth more than an unconstrained one?"
2. MATCHED WALL-CLOCK -- constrained sampling costs more per sample, so this
   answers the question a practitioner actually has. A design that wins on
   matched samples but loses on matched time is NOT an improvement.

Reporting only the flattering protocol is the failure mode this file exists to
avoid, so both go in the artifact regardless of which way they fall.

PLUS A DIRECT DIAGNOSTIC
The fraction of UNCONSTRAINED samples that contradict the observed play. This
measures the waste directly instead of inferring it from win rates.

The unconstrained sampler is defined LOCALLY and patched in at runtime, so
`dahaljeet/determinize.py` is not modified and the production RNG stream is
untouched.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from contextlib import contextmanager

sys.path.insert(0, ".")
import runenv

from dahaljeet import search as search_mod
from dahaljeet.cards import suit_of
from dahaljeet.agents import REGISTRY
from dahaljeet.hand import Hand
from dahaljeet.search import ISMCTSAgent, PIMCAgent
from dahaljeet.tournament import duplicate_match
from dahaljeet.view import make_view


def unconstrained(v, rng: random.Random, tries: int = 40):
    """Deal the unseen cards at random, IGNORING the void constraints.

    Same signature as determinize() so it can be swapped in. Always succeeds,
    because without constraints any assignment respecting hand sizes is
    accepted -- including ones that contradict the play history.
    """
    unseen = list(v.unseen)
    need = {s: 13 - len(v.played_by_seat[s]) for s in range(4) if s != v.seat}
    rng.shuffle(unseen)
    hands = [None] * 4
    hands[v.seat] = list(v.hand)
    pos = 0
    for s, k in need.items():
        hands[s] = unseen[pos:pos + k]
        pos += k
    return hands


def consistent(v, hands) -> bool:
    """Does this world contradict anything the acting seat has observed?"""
    for s in range(4):
        if s == v.seat or hands[s] is None:
            continue
        for c in hands[s]:
            if suit_of(c) in v.voids[s]:
                return False
    return True


@contextmanager
def sampler(fn):
    old = search_mod.determinize
    search_mod.determinize = fn
    try:
        yield
    finally:
        search_mod.determinize = old


def diagnostic(n_hands, seed, policy=None):
    """How often does an unconstrained sample contradict the play history?

    `policy` selects the play used to GENERATE the positions. This matters: the
    waste rate depends on how fast voids get revealed, and skilled play does not
    reveal them at the same rate as random play. Both are measured rather than
    assuming they agree.
    """
    rng = random.Random(seed)
    agent = REGISTRY[policy](rng=random.Random(seed + 5)) if policy else None
    by_trick = {}
    for _ in range(n_hands):
        h = Hand(dealer=rng.randrange(4), rng=rng)
        h.deal()
        while not h.is_over:
            v = make_view(h, h.to_act)
            if h.trick_lead_suit < 0 or True:
                bucket = by_trick.setdefault(h.tricks_played, [0, 0])
                for _ in range(8):
                    w = unconstrained(v, rng)
                    bucket[1] += 1
                    if consistent(v, w):
                        bucket[0] += 1
            if agent is not None:
                h.play(agent.act(v))
            else:
                h.play(rng.choice(h.legal_moves(h.to_act)))
    rows = []
    for t in sorted(by_trick):
        good, tot = by_trick[t]
        rows.append({"trick": t + 1, "samples": tot,
                     "consistent_fraction": round(good / tot, 4),
                     "wasted_fraction": round(1 - good / tot, 4)})
    return rows


def run_arm(make_agent, label, opp_name, deals, seed):
    opp = REGISTRY[opp_name](rng=random.Random(4))
    a = make_agent()
    r = duplicate_match(a, opp, n_deals=deals, seed=seed, timed=True)
    lo, hi = r.bootstrap_ci()
    return {"arm": label, "win": round(r.win_rate_a, 4),
            "ci": [round(lo, 4), round(hi, 4)],
            "ms_per_decision": round(r.ms_per_decision_a, 3),
            # Both samplers face the SAME deal sequence (same match seed), so
            # keeping the per-deal vector makes a PAIRED test possible. Paired
            # comparison removes deal luck, which is the dominant variance
            # component here -- without it the arms' CIs overlap and nothing is
            # decidable at any affordable sample size.
            "a_wins_per_deal": list(r.a_wins_per_deal)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--deals", type=int, default=300)
    ap.add_argument("--opponent", default="TensThenTricks")
    ap.add_argument("--diagnostic-hands", type=int, default=150)
    ap.add_argument("--diagnostic-only", action="store_true")
    ap.add_argument("--seed", type=int, default=808)
    a = ap.parse_args()

    t0 = time.perf_counter()
    print("DIAGNOSTIC — fraction of unconstrained samples that contradict "
          "observed play\n", flush=True)
    diag = {}
    for lbl, pol in (("random_play", None), ("skilled_play", "TensThenTricks")):
        print(f"  --- positions generated by {lbl} ---", flush=True)
        diag[lbl] = diagnostic(a.diagnostic_hands, a.seed, pol)
        for r in diag[lbl]:
            print(f"    trick {r['trick']:2d}: {r['consistent_fraction']:.4f} "
                  f"consistent, {r['wasted_fraction']:.4f} wasted  "
                  f"(n={r['samples']:,})", flush=True)

    rows = []
    if not a.diagnostic_only:
        print("\nMATCHED SAMPLES — identical budgets, both samplers\n", flush=True)
        for w in (4, 8, 16, 32):
            for name, fn in (("constrained", None), ("unconstrained", unconstrained)):
                mk = lambda w=w: PIMCAgent(worlds=w, rng=random.Random(3))
                if fn is None:
                    res = run_arm(mk, f"PIMC-{w}-{name}", a.opponent, a.deals, a.seed)
                else:
                    with sampler(fn):
                        res = run_arm(mk, f"PIMC-{w}-{name}", a.opponent, a.deals, a.seed)
                res.update(algo="PIMC", budget=w, sampler=name)
                rows.append(res)
                print(f"  PIMC worlds={w:3d} {name:14s} win {res['win']:.4f} "
                      f"{res['ci']}  {res['ms_per_decision']:8.2f} ms", flush=True)
        for it in (50, 100, 200, 400):
            for name, fn in (("constrained", None), ("unconstrained", unconstrained)):
                mk = lambda it=it: ISMCTSAgent(iterations=it, rng=random.Random(3))
                if fn is None:
                    res = run_arm(mk, f"ISMCTS-{it}-{name}", a.opponent, a.deals, a.seed)
                else:
                    with sampler(fn):
                        res = run_arm(mk, f"ISMCTS-{it}-{name}", a.opponent, a.deals, a.seed)
                res.update(algo="ISMCTS", budget=it, sampler=name)
                rows.append(res)
                print(f"  ISMCTS iters={it:4d} {name:14s} win {res['win']:.4f} "
                      f"{res['ci']}  {res['ms_per_decision']:8.2f} ms", flush=True)

    el = time.perf_counter() - t0
    json.dump({"env": runenv.snapshot(), "deals": a.deals, "opponent": a.opponent,
               "diagnostic": diag,
               "diagnostic_note": ("Waste rate depends on how fast voids are revealed, which differs between random and skilled play, so both are reported."), "matched_samples": rows, "seconds": el,
               "matched_time_note": (
                   "Matched-wall-clock comparison is read off these curves: for "
                   "each constrained point, compare against the unconstrained "
                   "point with the nearest ms_per_decision. Both protocols are "
                   "reported because a design that wins on matched samples but "
                   "loses on matched time is not an improvement.")},
              open("void_constraint_ablation.json", "w"), indent=2)
    print(f"\n{el:.0f}s -> void_constraint_ablation.json")
