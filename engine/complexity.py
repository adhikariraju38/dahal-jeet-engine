"""Combinatorial complexity of Dahal Jeet.

Traditional-game papers are routinely asked for the size of the problem. Every
quantity here is either exact with a derivation, or estimated with a stated
error -- an earlier complexity figure in this project was wrong by seven orders
of magnitude and was caught only by computing it properly rather than
reasoning about it.

    python complexity.py --hands 3000
    python complexity.py --selftest

QUANTITIES

1. Deals. An ordered partition of 52 distinct cards into four labelled hands of
   13:  52! / (13!)^4.

2. Initial information set. A player sees 13 cards; the other 39 are split
   three ways:  39! / (13!)^3. This is what one seat cannot distinguish at
   trick 1, and it is the number PIMC and ISMCTS sample from.

3. Information set DURING play, WITH void constraints -- the interesting one.
   Once a seat fails to follow suit it is provably void (rule P2), so many
   assignments are impossible. Counting the legal ones is a constrained
   distribution problem, and it is computed here EXACTLY rather than sampled.

   Method: each unseen card has an eligibility set (which opponents may legally
   hold it). With three opponents there are at most 7 distinct eligibility
   patterns, so group the cards by pattern and run a DP over patterns with the
   remaining hand-size capacities as state. Exact, and small enough to be
   instant. Verified against brute-force enumeration in --selftest.

4. Branching factor and decision count, measured from real play rather than
   assumed, since follow-suit makes the legal move count highly variable.

5. Game-tree size, reported as a range from the measured branching factor. This
   one is an ESTIMATE and is labelled as such.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import sys
import time
from collections import defaultdict

sys.path.insert(0, ".")


def deals_total():
    return math.factorial(52) // (math.factorial(13) ** 4)


def initial_infoset():
    return math.factorial(39) // (math.factorial(13) ** 3)


def count_assignments(patterns, caps):
    """Exact number of ways to deal grouped cards to seats under constraints.

    patterns: dict mapping a frozenset of eligible seat indices -> card count
    caps:     tuple of remaining hand size per seat

    DP over patterns; state is the remaining capacity vector.
    """
    states = {tuple(caps): 1}
    for pat, cnt in patterns.items():
        seats = sorted(pat)
        if not seats:
            return 0 if cnt else sum(states.values())
        nxt = defaultdict(int)
        # every way to split `cnt` identical-slot cards among eligible seats
        for combo in _compositions(cnt, len(seats)):
            coef = math.factorial(cnt)
            for k in combo:
                coef //= math.factorial(k)
            for st, ways in states.items():
                new = list(st)
                ok = True
                for s, k in zip(seats, combo):
                    new[s] -= k
                    if new[s] < 0:
                        ok = False
                        break
                if ok:
                    nxt[tuple(new)] += ways * coef
        states = dict(nxt)
        if not states:
            return 0
    return states.get((0,) * len(caps), 0)


def _compositions(total, parts):
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in _compositions(total - first, parts - 1):
            yield (first,) + rest


def brute_count(elig, caps):
    """Reference implementation: enumerate every assignment. Small inputs only."""
    n = len(elig)
    total = 0
    for assign in itertools.product(range(len(caps)), repeat=n):
        if any(assign[i] not in elig[i] for i in range(n)):
            continue
        c = [0] * len(caps)
        for a in assign:
            c[a] += 1
        if tuple(c) == tuple(caps):
            total += 1
    return total


def infoset_with_voids(elig, caps):
    """Group by eligibility pattern, then count exactly."""
    pats = defaultdict(int)
    for e in elig:
        pats[frozenset(e)] += 1
    return count_assignments(dict(pats), caps)


def measure_play(hands, seed):
    """Branching factor and decision count, measured from real deals."""
    from dahaljeet.hand import Hand

    rng = random.Random(seed)
    per_trick = defaultdict(list)
    allb, decisions = [], []
    for _ in range(hands):
        h = Hand(dealer=rng.randrange(4), rng=rng)
        h.deal()
        n_dec = 0
        while not h.is_over:
            legal = h.legal_moves(h.to_act)
            allb.append(len(legal))
            per_trick[h.tricks_played].append(len(legal))
            n_dec += 1
            h.play(rng.choice(legal))
        decisions.append(n_dec)
    return allb, per_trick, decisions


def eligibility(h, seat):
    """Unseen cards from `seat`'s view, and which opponents may hold each.

    Two public facts constrain this, and both are applied:
      * P2 voids -- a seat that failed to follow a suit cannot hold it
      * T3/T4 -- the trump card was shown to everyone and STAYS in the
        drawer's hand, so while it is unplayed its location is known exactly
    """
    others = [(seat + k) % 4 for k in (1, 2, 3)]
    caps = tuple(len(h.hands[o]) for o in others)
    seen = set(h.hands[seat])
    for s in range(4):
        seen.update(h.played_by_seat[s])
    unseen = [c for c in range(52) if c not in seen]
    elig = []
    for c in unseen:
        suit = c // 13
        if c == h.trump_card and h.trump_holder != seat:
            # publicly known to be in the drawer's hand
            e = {i for i, o in enumerate(others) if o == h.trump_holder}
        else:
            e = {i for i, o in enumerate(others) if suit not in h.voids[o]}
        elig.append(e)
    return unseen, elig, caps


def selftest():
    print("SELF-TEST\n")
    ok = True
    rng = random.Random(0)
    cases = 0
    for trial in range(60):
        ncards = rng.randint(3, 9)
        caps = [0, 0, 0]
        for _ in range(ncards):
            caps[rng.randrange(3)] += 1
        elig = []
        for _ in range(ncards):
            k = rng.randint(1, 3)
            elig.append(set(rng.sample([0, 1, 2], k)))
        want = brute_count(elig, caps)
        got = infoset_with_voids(elig, caps)
        if want != got:
            print(f"  [FAIL] elig={elig} caps={caps}: exact={got} brute={want}")
            ok = False
        cases += 1
    print(f"  [{'PASS' if ok else 'FAIL'}] exact void-constrained count matches "
          f"brute force on {cases} random cases")

    # unconstrained special case must equal the plain multinomial
    elig = [{0, 1, 2}] * 9
    got = infoset_with_voids(elig, (3, 3, 3))
    want = math.factorial(9) // (math.factorial(3) ** 3)
    r = got == want
    ok &= r
    print(f"  [{'PASS' if r else 'FAIL'}] unconstrained case equals multinomial: "
          f"{got} == {want}")

    # a card eligible nowhere makes the count zero
    r2 = infoset_with_voids([set()] + [{0, 1, 2}] * 5, (2, 2, 2)) == 0
    ok &= r2
    print(f"  [{'PASS' if r2 else 'FAIL'}] impossible constraint yields 0")

    print(f"\n{'ALL PASS' if ok else 'FAILURES PRESENT'}")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)

    t0 = time.perf_counter()
    D, I0 = deals_total(), initial_infoset()
    print(f"deals                 {D:.4e}   ({D})")
    print(f"initial info set      {I0:.4e}   ({I0})")

    allb, per_trick, decisions = measure_play(a.hands, a.seed)
    mean_b = sum(allb) / len(allb)
    mean_dec = sum(decisions) / len(decisions)
    trick_tbl = {int(k): round(sum(v) / len(v), 3)
                 for k, v in sorted(per_trick.items())}
    print(f"mean legal moves      {mean_b:.3f}  (measured over {len(allb):,} decisions)")
    print(f"decisions per hand    {mean_dec:.1f}")
    print(f"branching by trick    {trick_tbl}")

    # game tree size, estimated
    plies = 52
    log10_tree = plies * math.log10(mean_b)
    print(f"game-tree size (est)  10^{log10_tree:.1f}")

    # exact info-set sizes at a few points in real deals, with voids applied
    from dahaljeet.hand import Hand
    rng = random.Random(a.seed + 1)
    samples = defaultdict(list)
    for _ in range(300):
        h = Hand(dealer=rng.randrange(4), rng=rng)
        h.deal()
        while not h.is_over:
            if not h.trick and h.tricks_played in (0, 3, 6, 9, 11):
                unseen, elig, caps = eligibility(h, h.to_act)
                if sum(caps) == len(unseen):
                    samples[h.tricks_played].append(
                        infoset_with_voids(elig, caps))
            h.play(rng.choice(h.legal_moves(h.to_act)))

    infoset_rows = []
    for t in sorted(samples):
        v = samples[t]
        if not v:
            continue
        gm = math.exp(sum(math.log(max(x, 1)) for x in v) / len(v))
        infoset_rows.append({"trick": t + 1, "n_samples": len(v),
                             "log10_geometric_mean": round(math.log10(max(gm, 1)), 3),
                             "min": str(min(v)), "max": str(max(v))})
        print(f"  info set at trick {t+1:2d}: 10^{math.log10(max(gm,1)):.2f} "
              f"(n={len(v)}, exact, voids + public trump applied)")

    el = time.perf_counter() - t0
    import runenv
    json.dump({"env": runenv.snapshot(),
               "deals_total": str(D), "deals_total_log10": round(math.log10(D), 3),
               "initial_infoset": str(I0),
               "initial_infoset_log10": round(math.log10(I0), 3),
               "mean_branching_factor": round(mean_b, 4),
               "branching_by_trick": trick_tbl,
               "decisions_per_hand": round(mean_dec, 2),
               "n_decisions_measured": len(allb),
               "game_tree_log10_estimate": round(log10_tree, 2),
               "game_tree_note": ("ESTIMATE: mean branching raised to 52 plies. "
                                  "Branching is not independent across plies "
                                  "(follow-suit correlates them), so treat this "
                                  "as an order-of-magnitude figure only."),
               "infoset_during_play_exact": infoset_rows,
               "infoset_method": ("Exact count of void-consistent deals of the "
                                  "unseen cards to the three opponents, by DP "
                                  "over eligibility patterns. Verified against "
                                  "brute-force enumeration in --selftest."),
               "hands_sampled": a.hands, "seconds": round(el, 1)},
              open("complexity.json", "w"), indent=2)
    print(f"\n{el:.0f}s -> complexity.json")
