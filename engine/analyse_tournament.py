"""Post-hoc analysis of the round-robin: corrections, ratings, cycles, power.

Consumes the per-deal outcome vectors persisted by final_tournament.py (P0.1)
and produces three artifacts:

  stats_corrections.json  per-pairing tests with FWER and FDR control
  ratings.json            Bradley-Terry ratings, Elo scale, intransitivity
  power.json              retrospective power / required sample size

    python analyse_tournament.py
    python analyse_tournament.py --selftest    # verify the machinery first

WHY THESE TESTS

A 26-agent round-robin is 325 simultaneous comparisons. At alpha=0.05 about 16
pairings are expected to look "significant" purely by chance, so an uncorrected
claim at that scale says nothing. Both a family-wise correction (Holm) and a
false-discovery-rate correction (Benjamini-Hochberg) are reported, because they
answer different questions and reviewers differ on which they want.

The per-pairing test is a SIGN-FLIP PERMUTATION TEST, which is exact for this
design rather than assuming normality. Under the null that two agents are
equally strong, relabelling them maps a deal outcome d (A's wins out of the 4
rotations) to 4-d, so the null distribution of d-2 is symmetric about zero and
sign-flipping is a valid randomisation. The duplicate design earns this: it is
only true because both agents played the identical cards from every seat.

All pairings share one deal sequence, so the bootstrap resamples DEALS (not
pairings), which keeps the correlation between pairings intact.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import sys

sys.path.insert(0, ".")

try:
    import numpy as np
except Exception:
    np = None

# Normal quantiles, hard-coded so scipy is not a dependency.
Z_TWO_SIDED_95 = 1.959964
Z_POWER = {80: 0.8416212, 90: 1.2815516}

ROT = 4  # rotations per deal


# ----------------------------------------------------------------- tests

def signflip_p(deal_wins, iters=20000, seed=0):
    """Two-sided sign-flip permutation p-value for 'A is as strong as B'.

    deal_wins: per deal, A's wins out of 4 rotations (0..4).
    Statistic: mean(d) - 2. Null: symmetric about 0 under relabelling.
    """
    n = len(deal_wins)
    if n == 0:
        return 1.0, 0.0
    if np is not None:
        d = np.asarray(deal_wins, dtype=np.float64) - (ROT / 2.0)
        obs = d.mean()
        rng = np.random.default_rng(seed)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(iters, n))
        null = (signs * d).mean(axis=1)
        p = (np.count_nonzero(np.abs(null) >= abs(obs) - 1e-12) + 1) / (iters + 1)
        return float(p), float(obs)
    rng = random.Random(seed)
    d = [x - ROT / 2.0 for x in deal_wins]
    obs = sum(d) / n
    hits = 0
    for _ in range(iters):
        s = sum(x if rng.random() < 0.5 else -x for x in d) / n
        if abs(s) >= abs(obs) - 1e-12:
            hits += 1
    return (hits + 1) / (iters + 1), obs


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (family-wise error rate)."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    run = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * pvals[i]
        run = max(run, val)
        adj[i] = min(1.0, run)
    return adj


def benjamini_hochberg(pvals):
    """BH adjusted p-values (false discovery rate)."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i], reverse=True)
    adj = [0.0] * m
    run = 1.0
    for rank, i in enumerate(order):
        k = m - rank
        run = min(run, pvals[i] * m / k)
        adj[i] = min(1.0, run)
    return adj


# ------------------------------------------------- Bradley-Terry & cycles

def bradley_terry(names, wins, games, iters=2000, tol=1e-10):
    """MM/Zermelo fit. wins[i][j] = wins of i over j (may be fractional)."""
    n = len(names)
    p = [1.0] * n
    W = [sum(wins[i]) for i in range(n)]
    for _ in range(iters):
        new = []
        for i in range(n):
            den = 0.0
            for j in range(n):
                if i == j or games[i][j] == 0:
                    continue
                den += games[i][j] / (p[i] + p[j])
            new.append(W[i] / den if den > 0 and W[i] > 0 else 1e-12)
        gm = math.exp(sum(math.log(max(x, 1e-300)) for x in new) / n)
        new = [x / gm for x in new]
        if max(abs(a - b) for a, b in zip(new, p)) < tol:
            p = new
            break
        p = new
    return p


def to_elo(p, anchor=1500.0):
    """Bradley-Terry strengths on the Elo scale (400 points per 10x odds)."""
    lg = [400.0 * math.log10(max(x, 1e-300)) for x in p]
    mid = sum(lg) / len(lg)
    return [anchor + (x - mid) for x in lg]


def intransitivity(names, winrate):
    """Cycles among agents, plus a Hodge-style transitive/cyclic split.

    On a COMPLETE comparison graph the least-squares transitive potential is
    just each agent's mean margin, so the decomposition is exact and needs no
    solver: Y = grad(s) + residual, and the residual is the cyclic part.
    """
    n = len(names)
    Y = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and winrate[i][j] is not None:
                Y[i][j] = winrate[i][j] - 0.5
    # Least-squares transitive potential. The complete-graph Laplacian is
    # L = nI - J, so the LS solution of L s = div(Y) is the row mean over ALL n
    # entries (the diagonal is 0 and contributes nothing). Dividing by n-1
    # instead inflates the potential by n/(n-1) and leaves a residual even when
    # Y is a perfect gradient -- for n=3 that is a spurious cyclic ratio of
    # exactly 0.25. The self-test pins this.
    s = [sum(Y[i][j] for j in range(n)) / n for i in range(n)]
    num = den = 0.0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            r = Y[i][j] - (s[i] - s[j])
            num += r * r
            den += Y[i][j] * Y[i][j]
    cyclic_ratio = num / den if den > 0 else 0.0

    cycles = []
    for i, j, k in itertools.combinations(range(n), 3):
        for a, b, c in ((i, j, k), (i, k, j)):
            if (winrate[a][b] or 0) > 0.5 and (winrate[b][c] or 0) > 0.5 \
               and (winrate[c][a] or 0) > 0.5:
                cycles.append((names[a], names[b], names[c]))
    total = n * (n - 1) * (n - 2) // 6
    return {"n_triples": total, "n_3cycles": len(cycles),
            "cycle_rate": round(len(cycles) / total, 4) if total else 0.0,
            "cyclic_energy_ratio": round(cyclic_ratio, 4),
            "transitive_energy_ratio": round(1 - cyclic_ratio, 4),
            "examples": cycles[:12],
            "interpretation": (
                "cyclic_energy_ratio is the share of pairwise-margin variance "
                "NOT explained by any single strength ordering. Near 0 means "
                "the agents form a clean ladder; a large value means genuine "
                "rock-paper-scissors structure, which would be a fact about "
                "the game rather than a defect of the agents.")}


def estimate_dispersion(pairs):
    """Intra-deal correlation rho, estimated from the data.

    The four rotations of one deal are POSITIVELY correlated -- they are the
    same cards. So the per-deal win count is over-dispersed relative to
    Binomial(4, p). For exchangeable binary outcomes,

        Var(d) = 4 p (1-p) [1 + 3 rho]

    so rho follows from the observed mean and variance. Pooled by median across
    pairings, which is robust to the pairings pinned near 0 or 1.
    """
    est = []
    for pr in pairs:
        v = pr.get("a_wins_per_deal") or []
        if len(v) < 3:
            continue
        mu = sum(v) / len(v)
        ph = mu / ROT
        if ph <= 0.02 or ph >= 0.98:
            continue
        var = sum((x - mu) ** 2 for x in v) / (len(v) - 1)
        binom = ROT * ph * (1 - ph)
        if binom <= 0:
            continue
        rho = (var / binom - 1.0) / (ROT - 1)
        est.append(max(0.0, min(0.95, rho)))
    if not est:
        return 0.0
    est.sort()
    return est[len(est) // 2]


def _betabinom(rng, p, rho):
    """One deal's win count (0..4) with mean 4p and intra-deal correlation rho."""
    if rho <= 1e-9:
        return sum(1 for _ in range(ROT) if rng.random() < p)
    conc = (1.0 - rho) / rho
    a, b = max(p * conc, 1e-6), max((1 - p) * conc, 1e-6)
    q = rng.betavariate(a, b)
    return sum(1 for _ in range(ROT) if rng.random() < q)


def intransitivity_null(names, strengths, pairs, idx, n_deals, reps=200, seed=0):
    """Noise floor for cyclic energy under a PERFECTLY TRANSITIVE truth.

    Sampling noise alone manufactures apparent cycles: with few deals a pure
    ladder still yields non-zero cyclic energy. Reporting an observed value
    without this floor invites the obvious question -- is it more than noise?

    Null model: each agent's true strength is its fitted Bradley-Terry value,
    which is transitive by construction. Deal outcomes are drawn with the
    intra-deal correlation ESTIMATED FROM THE DATA (beta-binomial), not as
    independent rotations.

    That detail decides the answer. Duplicate dealing correlates the four
    rotations of a deal, so real outcomes are over-dispersed relative to
    Binomial(4, p). A binomial null therefore under-states noise, sets the bar
    too LOW, and manufactures "significant" intransitivity -- anti-conservative,
    exactly the wrong direction for a claim like this one.
    """
    rng = random.Random(seed)
    rho = estimate_dispersion(pairs)
    n = len(names)
    prob = [[0.5] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                prob[i][j] = strengths[i] / (strengths[i] + strengths[j])
    observed_pairs = [(idx[p_["a"]], idx[p_["b"]]) for p_ in pairs]
    out = []
    for _ in range(reps):
        wr = [[None] * n for _ in range(n)]
        for i, j in observed_pairs:
            w = sum(_betabinom(rng, prob[i][j], rho) for _ in range(n_deals))
            v = w / (n_deals * ROT)
            wr[i][j], wr[j][i] = v, 1 - v
        out.append(intransitivity(names, wr)["cyclic_energy_ratio"])
    out.sort()
    return {"reps": reps,
            "intra_deal_correlation_rho": round(rho, 4),
            "null_mean": round(sum(out) / len(out), 4),
            "null_p95": round(out[int(0.95 * len(out)) - 1], 4),
            "null_max": round(out[-1], 4),
            "model": ("Bradley-Terry strengths (transitive by construction); "
                      "per-deal counts beta-binomial with rho estimated from "
                      "the observed per-deal vectors"),
            "_samples": out}


# ------------------------------------------------------------ power

def required_n(sd_per_deal, delta, power=80):
    """Deals needed to detect a win-rate shift of `delta`, two-sided a=0.05."""
    if delta <= 0 or sd_per_deal <= 0:
        return None
    z = Z_TWO_SIDED_95 + Z_POWER[power]
    return int(math.ceil((z * sd_per_deal / delta) ** 2))


# ------------------------------------------------------------ driver

def load_pairs(path):
    d = json.load(open(path))
    pairs = [p for p in d.get("pairs", []) if p.get("a_wins_per_deal")]
    return d, pairs


def analyse(path, iters, out_prefix=""):
    d, pairs = load_pairs(path)
    if not pairs:
        print(f"  [stop] {path} has no per-deal vectors. It predates P0.1 — "
              f"re-run final_tournament.py to record them.")
        return False
    names = d.get("roster") or sorted({p["a"] for p in pairs} | {p["b"] for p in pairs})
    idx = {n: i for i, n in enumerate(names)}
    n = len(names)
    ndeals = len(pairs[0]["a_wins_per_deal"])

    # ---- per-pairing tests
    rows, pvals = [], []
    for k, p in enumerate(pairs):
        pv, obs = signflip_p(p["a_wins_per_deal"], iters=iters, seed=k)
        rows.append({"a": p["a"], "b": p["b"], "win_a": p["win_a"],
                     "mean_margin_per_deal": round(obs, 4), "p_raw": pv})
        pvals.append(pv)
    h = holm(pvals)
    bh = benjamini_hochberg(pvals)
    for r, x, y in zip(rows, h, bh):
        r["p_holm"] = round(x, 6)
        r["p_bh"] = round(y, 6)
        r["sig_holm"] = x < 0.05
        r["sig_bh"] = y < 0.05
    raw_sig = sum(1 for v in pvals if v < 0.05)
    corrections = {
        "family": f"all {len(pairs)} pairings of the round-robin",
        "family_declared_before_analysis": True,
        "test": "two-sided sign-flip permutation on per-deal margin",
        "permutations": iters,
        "n_pairings": len(pairs),
        "n_significant_raw": raw_sig,
        "n_significant_holm": sum(1 for r in rows if r["sig_holm"]),
        "n_significant_bh": sum(1 for r in rows if r["sig_bh"]),
        "expected_false_positives_uncorrected": round(0.05 * len(pairs), 1),
        "pairings": rows,
    }
    json.dump(corrections, open(f"{out_prefix}stats_corrections.json", "w"), indent=2)

    # ---- ratings
    wins = [[0.0] * n for _ in range(n)]
    games = [[0.0] * n for _ in range(n)]
    winrate = [[None] * n for _ in range(n)]
    for p in pairs:
        i, j = idx[p["a"]], idx[p["b"]]
        w = sum(p["a_wins_per_deal"])
        g = len(p["a_wins_per_deal"]) * ROT
        wins[i][j] += w
        wins[j][i] += g - w
        games[i][j] += g
        games[j][i] += g
        winrate[i][j] = w / g
        winrate[j][i] = 1 - w / g
    strengths = bradley_terry(names, wins, games)
    elo = to_elo(strengths)

    # bootstrap over DEALS, shared across pairings
    boot = [[] for _ in range(n)]
    rng = random.Random(0)
    for _ in range(200):
        pick = [rng.randrange(ndeals) for _ in range(ndeals)]
        bw = [[0.0] * n for _ in range(n)]
        bg = [[0.0] * n for _ in range(n)]
        for p in pairs:
            i, j = idx[p["a"]], idx[p["b"]]
            v = p["a_wins_per_deal"]
            w = sum(v[t] for t in pick)
            g = len(pick) * ROT
            bw[i][j] += w
            bw[j][i] += g - w
            bg[i][j] += g
            bg[j][i] += g
        be = to_elo(bradley_terry(names, bw, bg, iters=300))
        for i in range(n):
            boot[i].append(be[i])
    ratings = []
    for i, nm in enumerate(names):
        b = sorted(boot[i])
        ratings.append({"agent": nm, "elo": round(elo[i], 1),
                        "ci95": [round(b[int(0.025 * len(b))], 1),
                                 round(b[int(0.975 * len(b)) - 1], 1)],
                        "bt_strength": round(strengths[i], 6)})
    ratings.sort(key=lambda r: -r["elo"])
    intr = intransitivity(names, winrate)
    null = intransitivity_null(names, strengths, pairs, idx, ndeals)
    obs = intr["cyclic_energy_ratio"]
    above = sum(1 for x in null["_samples"] if x >= obs)
    null.pop("_samples")
    intr["noise_floor"] = null
    intr["p_value_vs_transitive_null"] = round((above + 1) / (null["reps"] + 1), 4)
    intr["exceeds_noise"] = obs > null["null_p95"]
    intr["verdict"] = (
        f"cyclic energy {obs:.4f} vs noise floor {null['null_mean']:.4f} "
        f"(95th pct {null['null_p95']:.4f}) -- "
        + ("ABOVE the floor: evidence of real intransitivity"
           if obs > null["null_p95"] else
           "WITHIN noise: no evidence of intransitivity beyond sampling error"))

    json.dump({"deals_per_pair": ndeals, "n_agents": n,
               "scale_note": ("Elo here is a monotone transform of the "
                              "Bradley-Terry fit on THIS agent pool. It is not "
                              "comparable to Elo from any other pool."),
               "ratings": ratings,
               "intransitivity": intr},
              open(f"{out_prefix}ratings.json", "w"), indent=2)

    # ---- power
    sds = []
    for p in pairs:
        v = [x / ROT for x in p["a_wins_per_deal"]]
        mu = sum(v) / len(v)
        sds.append((sum((x - mu) ** 2 for x in v) / len(v)) ** 0.5)
    sds.sort()
    med = sds[len(sds) // 2]
    worst = sds[-1]
    power = {
        "deals_used": ndeals,
        "per_deal_sd_median": round(med, 4),
        "per_deal_sd_max": round(worst, 4),
        "required_deals": {
            f"delta={dl}": {f"power{pw}": required_n(med, dl, pw)
                            for pw in (80, 90)}
            for dl in (0.02, 0.03, 0.05, 0.10)},
        "required_deals_worst_case_sd": {
            f"delta={dl}": {f"power{pw}": required_n(worst, dl, pw)
                            for pw in (80, 90)}
            for dl in (0.02, 0.05)},
        "note": ("Computed from the OBSERVED per-deal SD of this tournament, "
                 "so it answers 'was our sample big enough' with a measurement "
                 "rather than a convention. Comparisons whose detectable effect "
                 "exceeds what we can resolve must be reported as inconclusive, "
                 "not as ties."),
    }
    json.dump(power, open(f"{out_prefix}power.json", "w"), indent=2)

    print(f"  {len(pairs)} pairings, {ndeals} deals each, {n} agents")
    print(f"  significant: raw {raw_sig}  ->  Holm {corrections['n_significant_holm']}"
          f"  BH {corrections['n_significant_bh']}"
          f"   (expected false positives uncorrected: "
          f"{corrections['expected_false_positives_uncorrected']})")
    it = json.load(open(f"{out_prefix}ratings.json"))["intransitivity"]
    print(f"  3-cycles: {it['n_3cycles']}/{it['n_triples']} "
          f"({it['cycle_rate']:.1%})   cyclic energy "
          f"{it['cyclic_energy_ratio']:.3f}")
    print(f"  {it['verdict']}  (p={it['p_value_vs_transitive_null']}, "
          f"rho={it['noise_floor']['intra_deal_correlation_rho']})")
    print(f"  power: median per-deal SD {med:.4f}; to detect delta=0.05 at 80% "
          f"needs {required_n(med, 0.05, 80)} deals")
    print(f"  -> {out_prefix}stats_corrections.json / ratings.json / power.json")
    return True


# ------------------------------------------------------------ self-test

def selftest():
    """Verify the machinery on data whose answer is known in advance."""
    print("SELF-TEST\n")
    ok = True

    # 1. sign-flip test on a known-null and known-effect vector
    rng = random.Random(1)
    null_v = [rng.choice([0, 1, 2, 3, 4]) for _ in range(400)]
    p_null, _ = signflip_p(null_v, iters=5000, seed=1)
    eff_v = [min(4, max(0, 2 + rng.choice([1, 1, 2, 0]))) for _ in range(400)]
    p_eff, _ = signflip_p(eff_v, iters=5000, seed=2)
    r1 = p_null > 0.05 and p_eff < 0.001
    ok &= r1
    print(f"  [{'PASS' if r1 else 'FAIL'}] sign-flip: null p={p_null:.3f} (>0.05), "
          f"strong effect p={p_eff:.4f} (<0.001)")

    # 2. Holm/BH must be monotone and >= raw
    pv = [0.001, 0.01, 0.02, 0.04, 0.5]
    hh, bb = holm(pv), benjamini_hochberg(pv)
    r2 = all(a >= b - 1e-12 for a, b in zip(hh, pv)) and \
         all(a >= b - 1e-12 for a, b in zip(bb, pv)) and \
         all(hh[i] >= bb[i] - 1e-12 for i in range(len(pv)))
    ok &= r2
    print(f"  [{'PASS' if r2 else 'FAIL'}] corrections: Holm >= BH >= raw   "
          f"Holm={[round(x,4) for x in hh]}")

    # 3. Bradley-Terry recovers a known ladder
    names = ["A", "B", "C", "D"]
    true_p = [8.0, 4.0, 2.0, 1.0]
    n = 4
    wins = [[0.0] * n for _ in range(n)]
    games = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i < j:
                g = 4000
                w = g * true_p[i] / (true_p[i] + true_p[j])
                wins[i][j], wins[j][i] = w, g - w
                games[i][j] = games[j][i] = g
    fit = bradley_terry(names, wins, games)
    ratio = [fit[i] / fit[-1] for i in range(n)]
    want = [true_p[i] / true_p[-1] for i in range(n)]
    r3 = all(abs(a - b) / b < 0.02 for a, b in zip(ratio, want))
    ok &= r3
    print(f"  [{'PASS' if r3 else 'FAIL'}] Bradley-Terry recovers ladder: "
          f"got {[round(x,2) for x in ratio]}, want {[round(x,2) for x in want]}")

    # 4. intransitivity: a pure ladder ~0 cyclic; a pure 3-cycle ~1
    wr_ladder = [[None] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            if i != j:
                wr_ladder[i][j] = 0.5 + 0.2 * (j - i) / 2.0 * -1
    lad = intransitivity(["A", "B", "C"], wr_ladder)
    wr_cycle = [[None, 0.8, 0.2], [0.2, None, 0.8], [0.8, 0.2, None]]
    cyc = intransitivity(["A", "B", "C"], wr_cycle)
    r4 = lad["cyclic_energy_ratio"] < 0.05 and cyc["cyclic_energy_ratio"] > 0.9 \
         and cyc["n_3cycles"] >= 1 and lad["n_3cycles"] == 0
    ok &= r4
    print(f"  [{'PASS' if r4 else 'FAIL'}] intransitivity: ladder cyclic="
          f"{lad['cyclic_energy_ratio']:.3f} (~0), cycle cyclic="
          f"{cyc['cyclic_energy_ratio']:.3f} (~1), cycles found={cyc['n_3cycles']}")

    # 5. power formula against a textbook value
    got = required_n(0.5, 0.1, 80)
    r5 = abs(got - 197) <= 2
    ok &= r5
    print(f"  [{'PASS' if r5 else 'FAIL'}] power: sd=0.5 delta=0.1 80% -> "
          f"n={got} (textbook 196-197)")

    print(f"\n{'ALL PASS' if ok else 'FAILURES PRESENT'}")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="final_tournament.json")
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if not os.path.exists(a.input):
        print(f"  [wait] {a.input} does not exist yet — stage 7 has not run.")
        sys.exit(0)
    analyse(a.input, a.iters)
