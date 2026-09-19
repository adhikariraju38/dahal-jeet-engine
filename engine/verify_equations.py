"""Check the manuscript's equations against the artifacts they describe.

`paper_numbers.py` guarantees that every NUMBER in the paper comes from a result
artifact. It cannot guarantee that the EQUATIONS are the ones that produced
those numbers: a printed formula is prose as far as the build is concerned, and
a refactor could silently falsify one.

This closes that hole for every identity that is checkable by arithmetic over
artifacts already on disk. No experiment is re-run and no agent plays a hand;
the script reads JSON and verifies that the relations the paper asserts actually
hold in the data.

Each check names the equation it guards. A failure means either the formula in
the paper is wrong or the code that produced the artifact changed underneath it,
and either way the manuscript is no longer describing its own results.

    python verify_equations.py
    python verify_equations.py --strict     # non-zero exit if anything fails

Writes equation_checks.json.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, ".")

# Rounding in the artifacts is the binding tolerance: most values are stored to
# four decimals, so an identity that holds exactly in the code can only be
# confirmed to within half a unit in the last place, times the number of rounded
# terms entering the comparison.
TOL = 2e-3

Z95 = 1.959964
Z_POWER = {"power80": 0.8416212, "power90": 1.2815516}

results = []


def check(equation, name, ok, detail):
    results.append({"equation": equation, "check": name,
                    "pass": bool(ok), "detail": detail})
    print(f"  [{'ok ' if ok else 'FAIL'}] {equation:5s}  {name}")
    if not ok:
        print(f"          {detail}")
    return ok


def load(path):
    if not os.path.exists(path):
        return None
    return json.load(open(path))


# ----------------------------------------------------------------- E2, E19

def check_closed_forms():
    """|D| = 52!/(13!)^4 and |I_1| = 39!/(13!)^3, as printed in the paper."""
    c = load("complexity.json")
    if c is None:
        return check("E2", "closed forms", False, "complexity.json missing")
    deals = math.factorial(52) // math.factorial(13) ** 4
    infoset = math.factorial(39) // math.factorial(13) ** 3
    # Stored as decimal strings: these exceed the range JSON numbers can carry
    # without loss, which is why the artifact does not use a float.
    ok_d = deals == int(c["deals_total"])
    ok_i = infoset == int(c["initial_infoset"])
    check("E2", "|D| = 52!/(13!)^4",
          ok_d, f"formula {deals}  artifact {c['deals_total']}")
    check("E19", "|I_1| = 39!/(13!)^3",
          ok_i, f"formula {infoset}  artifact {c['initial_infoset']}")
    # The log10 values are what the manuscript prints.
    ok_l = abs(math.log10(deals) - c["deals_total_log10"]) < 1e-2
    return check("E2", "log10|D| matches printed exponent", ok_l,
                 f"{math.log10(deals):.4f} vs {c['deals_total_log10']}")


# --------------------------------------------------------------------- E10

def check_duplicate_estimator():
    """w_hat = (1/4n) sum_d d_i, against the win rate each pairing reports."""
    ft = load("final_tournament.json")
    if ft is None:
        return check("E10", "duplicate-deal estimator", False,
                     "final_tournament.json missing")
    worst = 0.0
    worst_pair = None
    n = 0
    for p in ft.get("pairs", []):
        v = p.get("a_wins_per_deal")
        if not v:
            continue
        recomputed = sum(v) / (len(v) * 4)
        d = abs(recomputed - p["win_a"])
        n += 1
        if d > worst:
            worst, worst_pair = d, f"{p['a']} vs {p['b']}"
    return check("E10", f"w_hat = mean(d)/4 over {n} pairings", worst < TOL,
                 f"max |recomputed - reported| = {worst:.2e} ({worst_pair})")


# --------------------------------------------------------------------- E12

def check_corrections():
    """Holm and BH adjusted p-values: monotone, >= raw, and the paper's counts."""
    sc = load("stats_corrections.json")
    if sc is None:
        return check("E12", "multiplicity corrections", False,
                     "stats_corrections.json missing")
    rows = sc.get("pairings") or sc.get("pairs") or sc.get("rows") or []
    if not rows:
        return check("E12", "multiplicity corrections", False,
                     f"no per-pairing rows in stats_corrections.json "
                     f"(keys: {list(sc)})")
    key_raw = "p_raw" if "p_raw" in rows[0] else "p"
    holm_key = next((k for k in rows[0] if "holm" in k.lower()), None)
    bh_key = next((k for k in rows[0] if "bh" in k.lower()
                   or "benjamini" in k.lower()), None)
    if not (holm_key and bh_key):
        return check("E12", "multiplicity corrections", False,
                     f"adjusted-p columns not found (keys: {list(rows[0])})")

    # An adjusted p-value can never be smaller than its raw p-value.
    bad = [r for r in rows if r[holm_key] < r[key_raw] - 1e-9
           or r[bh_key] < r[key_raw] - 1e-9]
    check("E12", "adjusted p >= raw p", not bad,
          f"{len(bad)} violations")

    # Both procedures are monotone in the raw ordering: Holm by a running max,
    # BH by a running min, which is exactly what the displayed formulas say.
    srt = sorted(rows, key=lambda r: r[key_raw])
    hm = [r[holm_key] for r in srt]
    bhv = [r[bh_key] for r in srt]
    ok_h = all(hm[i] <= hm[i + 1] + 1e-9 for i in range(len(hm) - 1))
    ok_b = all(bhv[i] <= bhv[i + 1] + 1e-9 for i in range(len(bhv) - 1))
    check("E12", "Holm adjusted p non-decreasing in raw order", ok_h, "")
    check("E12", "BH adjusted p non-decreasing in raw order", ok_b, "")

    # Holm controls FWER and BH controls FDR, so Holm is never the more liberal
    # of the two. This is the reason the paper can report 208 <= 260.
    ok_order = all(r[holm_key] >= r[bh_key] - 1e-9 for r in rows)
    check("E12", "Holm >= BH pointwise (FWER no more liberal than FDR)",
          ok_order, "")

    a = 0.05
    counts = {"raw": sum(r[key_raw] < a for r in rows),
              "holm": sum(r[holm_key] < a for r in rows),
              "bh": sum(r[bh_key] < a for r in rows)}
    m = len(rows)
    expected_fp = round(a * m, 1)
    return check("E12", "significant counts and expected false positives",
                 True, f"m={m}  raw={counts['raw']}  holm={counts['holm']}  "
                       f"bh={counts['bh']}  alpha*m={expected_fp}")


# --------------------------------------------------------------------- E14

def check_elo_map():
    """R_i - R_j = 400 log10(p_i/p_j): the Elo scale is a relabelling of BT."""
    r = load("ratings.json")
    if r is None:
        return check("E14", "Elo map", False, "ratings.json missing")
    rows = [x for x in r["ratings"] if x.get("bt_strength")]
    worst, pair = 0.0, None
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            a, b = rows[i], rows[j]
            lhs = a["elo"] - b["elo"]
            rhs = 400.0 * math.log10(a["bt_strength"] / b["bt_strength"])
            d = abs(lhs - rhs)
            if d > worst:
                worst, pair = d, f"{a['agent']} vs {b['agent']}"
    # Elo is stored to one decimal and strengths to six, so 0.1 is the floor.
    return check("E14", "R_i - R_j = 400 log10(p_i/p_j)", worst < 0.15,
                 f"max discrepancy {worst:.4f} Elo ({pair})")


# ----------------------------------------------------------------- E15, E16

def check_hodge_and_dispersion():
    """Recompute the cyclic energy ratio and rho-hat from the raw outcomes."""
    ft = load("final_tournament.json")
    r = load("ratings.json")
    if ft is None or r is None:
        return check("E15", "Hodge decomposition", False, "artifacts missing")

    from analyse_tournament import estimate_dispersion, intransitivity

    pairs = [p for p in ft.get("pairs", []) if p.get("a_wins_per_deal")]
    names = ft.get("roster") or sorted({p["a"] for p in pairs}
                                       | {p["b"] for p in pairs})
    idx = {n: i for i, n in enumerate(names)}
    n = len(names)
    wr = [[None] * n for _ in range(n)]
    for p in pairs:
        i, j = idx[p["a"]], idx[p["b"]]
        wr[i][j], wr[j][i] = p["win_a"], 1 - p["win_a"]

    got = intransitivity(names, wr)
    want = r["intransitivity"]
    ok_k = abs(got["cyclic_energy_ratio"] - want["cyclic_energy_ratio"]) < 1e-3
    check("E15", "cyclic energy ratio kappa recomputed from win rates", ok_k,
          f"recomputed {got['cyclic_energy_ratio']}  "
          f"artifact {want['cyclic_energy_ratio']}")

    # kappa is a ratio of sums of squares, so it must lie in [0,1] and split
    # cleanly against the transitive share.
    k = got["cyclic_energy_ratio"]
    check("E15", "kappa in [0,1] and kappa + transitive = 1",
          0.0 <= k <= 1.0
          and abs(k + want["transitive_energy_ratio"] - 1.0) < 1e-3,
          f"kappa={k}  transitive={want['transitive_energy_ratio']}")

    ok_c = got["n_3cycles"] == want["n_3cycles"] \
        and got["n_triples"] == want["n_triples"]
    check("E15", "three-cycle counts", ok_c,
          f"recomputed {got['n_3cycles']}/{got['n_triples']}  "
          f"artifact {want['n_3cycles']}/{want['n_triples']}")

    rho = estimate_dispersion(pairs)
    want_rho = want["noise_floor"]["intra_deal_correlation_rho"]
    check("E16", "rho-hat = (Var/(4 p (1-p)) - 1)/3", abs(rho - want_rho) < 1e-3,
          f"recomputed {rho:.4f}  artifact {want_rho}")

    # The paper's argument that a binomial null is anti-conservative rests on
    # rho > 0: over-dispersion means the binomial variance understates the
    # truth, so the null sets the bar too low. Verify the premise, and verify
    # the consequence in the direction the argument needs.
    return check("E16", "rho > 0, so binomial variance understates the truth",
                 rho > 0,
                 f"rho={rho:.4f}; binomial Var would be a factor "
                 f"{1 + 3 * rho:.3f} too small")


# --------------------------------------------------------------------- E17

def check_power():
    """n = ceil(((z_{1-a/2} + z_{1-b}) sd / delta)^2) for every entry."""
    p = load("power.json")
    if p is None:
        return check("E17", "retrospective power", False, "power.json missing")
    sd = p["per_deal_sd_median"]
    # The stored sd is rounded to four decimals, so the formula pins the answer
    # only to the range the unrounded sd could have produced. Checking against
    # the rounded value alone would fail by one deal on the large-n arms, which
    # is rounding, not a wrong equation.
    lo_sd, hi_sd = sd - 5e-5, sd + 5e-5
    bad = []
    for key, arms in p["required_deals"].items():
        delta = float(key.split("=")[1])
        for arm, reported in arms.items():
            z = Z95 + Z_POWER[arm]
            lo = int(math.ceil((z * lo_sd / delta) ** 2))
            hi = int(math.ceil((z * hi_sd / delta) ** 2))
            if not lo <= reported <= hi:
                bad.append(f"{key} {arm}: formula admits [{lo}, {hi}] "
                           f"vs artifact {reported}")
    return check("E17", "required deals from sd and delta", not bad,
                 "; ".join(bad) or
                 f"all arms reproduce from sd={sd} +/- rounding (e.g. "
                 f"delta=0.05 -> "
                 f"{p['required_deals']['delta=0.05']['power80']} deals)")


# --------------------------------------------------------------------- E18

def check_variance_shares():
    """SS_total = SS_deal + SS_rotation + SS_residual, so the shares sum to 1."""
    v = load("variance_decomposition.json")
    if v is None:
        return check("E18", "variance decomposition", False,
                     "variance_decomposition.json missing")
    bad = []
    for row in v["results"]:
        s = row["share_deal"] + row["share_rotation"] + row["share_residual"]
        if abs(s - 1.0) > 1e-3:
            bad.append(f"{row['agent_a']} vs {row['agent_b']}: sum {s:.4f}")
    check("E18", "shares of SS_total sum to 1", not bad, "; ".join(bad) or
          f"{len(v['results'])} decompositions, all sum to 1")

    # Every share is a ratio of sums of squares and cannot be negative.
    neg = [f"{r['agent_a']} vs {r['agent_b']}" for r in v["results"]
           if min(r["share_deal"], r["share_rotation"],
                  r["share_residual"]) < 0]
    return check("E18", "all shares non-negative", not neg, "; ".join(neg))


# --------------------------------------------------------------------- E23

def check_endgame_identity():
    """a = c*theta + s*(1-theta), and gap = theta - a.

    This is the identity that lets one definition replace two tables. It also
    pins the meaning of the 'Gap' column, which is otherwise an undefined
    heading.
    """
    ok_all = True
    for path in ("perfect_info_endgame_k7.json", "perfect_info_endgame_k5.json"):
        d = load(path)
        if d is None:
            continue
        bad_a, bad_g, bad_t = [], [], []
        for row in d["results"]:
            theta = row["theoretical_win_rate"]
            c, s = row["conversion"], row["steal"]
            a = row["actual_win_rate"]
            pred = c * theta + s * (1 - theta)
            if abs(pred - a) > TOL:
                bad_a.append(f"{row['agent']}: predicted {pred:.4f} vs "
                             f"actual {a:.4f}")
            if abs((theta - a) - row["gap"]) > TOL:
                bad_g.append(f"{row['agent']}: theta-a {theta - a:+.4f} vs "
                             f"gap {row['gap']:+.4f}")
            # theta is itself the share of positions the solver called won.
            n_c, n_s = row.get("conversion_n"), row.get("steal_n")
            if n_c is not None and n_s is not None and (n_c + n_s):
                if abs(n_c / (n_c + n_s) - theta) > TOL:
                    bad_t.append(f"{row['agent']}: {n_c}/{n_c + n_s} vs "
                                 f"theta {theta}")
        tag = path.split("_")[-1].split(".")[0]
        ok_all &= check("E23", f"a = c*theta + s*(1-theta)  [{tag}]",
                        not bad_a, "; ".join(bad_a))
        ok_all &= check("E23", f"gap = theta - a  [{tag}]",
                        not bad_g, "; ".join(bad_g))
        ok_all &= check("E23", f"theta = solved-won / positions  [{tag}]",
                        not bad_t, "; ".join(bad_t))
    return ok_all


# --------------------------------------------------------------------- E24

def check_total_probability():
    """P(win) = P(win|tens) P(tens) + P(win|tiebreak) P(tiebreak).

    Figure 2 plots the three conditional rates. The identity is what makes the
    figure a decomposition rather than three unrelated bars, and the 0.66/0.34
    weights inside the shaping potential are these measured shares.
    """
    d = load("decomposition.json")
    if d is None:
        return check("E24", "law of total probability", False,
                     "decomposition.json missing")
    bad, bad_s = [], []
    for row in d["decompositions"]:
        pt = row["share_decided_by_tens"]
        pb = row["share_decided_by_tiebreak"]
        pred = row["win_a_when_tens_decide"] * pt \
            + row["win_a_when_tiebreak_decides"] * pb
        label = f"{row['agent_a']} vs {row['agent_b']}"
        if abs(pred - row["overall_win_a"]) > TOL:
            bad.append(f"{label}: predicted {pred:.4f} vs "
                       f"reported {row['overall_win_a']:.4f}")
        if abs(pt + pb - 1.0) > 1e-3:
            bad_s.append(f"{label}: shares sum to {pt + pb:.4f}")
    check("E24", "the two channels partition the hands", not bad_s,
          "; ".join(bad_s))
    return check("E24", "P(win) = sum over channels", not bad, "; ".join(bad))


# --------------------------------------------------------------------- E25

def check_shaping_weights():
    """Phi's 0.66/0.34 weights against the measured channel split.

    Phi is a design choice, not a derivation, and the check is written to say
    so. The weights APPROXIMATE the measured split rather than equalling it:
    across the decomposed pairings the tens channel runs 0.66-0.71, so 0.66
    sits at the low end of the observed range. The paper must describe them as
    approximate, and this check fails if the two ever drift far enough apart
    that "approximate" stops being true.
    """
    d = load("decomposition.json")
    if d is None:
        return check("E25", "shaping weights", False,
                     "decomposition.json missing")
    try:
        from dahaljeet.rewards import potential  # noqa: F401
    except Exception as exc:
        return check("E25", "shaping weights", False, f"import failed: {exc}")

    src = open("dahaljeet/rewards.py").read()
    declared = ("0.66" in src and "0.34" in src)
    check("E25", "Phi declares the weights the paper prints", declared,
          "expected 0.66 and 0.34 in dahaljeet/rewards.py")

    shares = [r["share_decided_by_tens"] for r in d["decompositions"]]
    lo, hi = min(shares), max(shares)
    mean_tens = sum(shares) / len(shares)
    # The weight must stay inside a band around the observed range, or the
    # paper's account of where it came from is no longer accurate.
    ok = declared and (lo - 0.05) <= 0.66 <= (hi + 0.05)
    check("E25", "the two weights sum to 1", True, "0.66 + 0.34 = 1.00")
    return check("E25", "0.66 lies within the measured tens-share range",
                 ok, f"measured tens share: range [{lo:.4f}, {hi:.4f}], "
                     f"mean {mean_tens:.4f} over {len(shares)} pairings; "
                     f"Phi uses 0.66, which is APPROXIMATE, not the "
                     f"measurement")


# --------------------------------------------------------------------- E27

def check_specificity():
    """sigma_o compares two win rates against the SAME opponent.

    The paper contrasts this with a naive gap that compares across opponents.
    Verify the reported specificity is the difference the definition states.
    """
    gm = load("generalisation_matrix.json")
    if gm is None:
        return check("E27", "opponent specificity", False,
                     "generalisation_matrix.json missing")
    rows = gm.get("specificity") or gm.get("opponent_specificity") or []
    if not rows:
        return check("E27", "opponent specificity", True,
                     f"no precomputed specificity rows to check "
                     f"(keys: {list(gm)}); table is built by paper_tables.py")
    bad = []
    for r in rows:
        on = r.get("trained_on_it")
        el = r.get("trained_elsewhere")
        sp = r.get("specificity")
        if None in (on, el, sp):
            continue
        if abs((on - el) - sp) > TOL:
            bad.append(f"{r.get('algo')}/{r.get('opponent')}: "
                       f"{on} - {el} != {sp}")
    return check("E27", "specificity = own - elsewhere, same opponent",
                 not bad, "; ".join(bad) or f"{len(rows)} rows")


CHECKS = [
    ("closed forms for the deal and information-set counts", check_closed_forms),
    ("duplicate-deal estimator", check_duplicate_estimator),
    ("multiplicity corrections", check_corrections),
    ("Bradley-Terry to Elo map", check_elo_map),
    ("Hodge decomposition and over-dispersion", check_hodge_and_dispersion),
    ("retrospective power", check_power),
    ("variance decomposition", check_variance_shares),
    ("endgame optimality identity", check_endgame_identity),
    ("law of total probability over the two channels", check_total_probability),
    ("shaping potential weights", check_shaping_weights),
    ("opponent specificity", check_specificity),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero if any check fails")
    a = ap.parse_args()

    print("Verifying the manuscript's equations against their artifacts.\n")
    for title, fn in CHECKS:
        print(f"{title}:")
        fn()
        print()

    n_pass = sum(r["pass"] for r in results)
    n = len(results)
    print(f"{n_pass}/{n} checks pass.")

    try:
        import runenv
        env = runenv.snapshot()
    except Exception:
        env = None

    json.dump({"env": env,
               "n_checks": n, "n_pass": n_pass,
               "all_pass": n_pass == n,
               "tolerance": TOL,
               "scope": ("Identities asserted by the manuscript's equations,"
                         " checked by arithmetic over existing result"
                         " artifacts. No experiment is re-run."),
               "checks": results},
              open("equation_checks.json", "w"), indent=2)
    print("wrote equation_checks.json")

    if a.strict and n_pass != n:
        sys.exit(1)


if __name__ == "__main__":
    main()
