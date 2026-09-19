"""Matched-wall-clock comparison for the void-constraint ablation.

The matched-SAMPLES comparison (same world/iteration count) flatters the
constrained sampler, because constraining costs time per sample. The question a
practitioner actually has is which is better AT EQUAL TIME. For PIMC the
constraint costs only ~3-4% more per decision so the two protocols agree; for
ISMCTS it costs 14-21%, and there the protocols can disagree.

Two levels of evidence, reported separately and honestly labelled:

  INTERPOLATED  read off the measured curves. Cheap, but it is an inference:
                the unconstrained win rate at the constrained arm's exact time
                was never actually measured. Reported under BOTH linear-time
                and log-time interpolation, because if the verdict flips
                between them the verdict is an artefact of the interpolation
                and must not be reported as a result.

  MEASURED      the unconstrained sampler is actually RUN at a budget chosen to
                match the constrained arm's wall-clock, and the achieved times
                of both arms are reported so the reader can check the match.
                This is the claim that survives review.

    python matched_time.py                     # interpolated, from the artifact
    python matched_time.py --measure --deals 300
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

sys.path.insert(0, ".")

ART = "void_constraint_ablation.json"
LEGACY = "ablate_determinize.json"


def load_rows(path=None):
    for p in ([path] if path else [ART, LEGACY]):
        if p and os.path.exists(p):
            d = json.load(open(p))
            rows = d.get("matched_samples", [])
            if rows:
                return d, rows, p
    return None, [], None


def interp(curve, t, log=False):
    """Win rate of `curve` at wall-clock `t`, or None if outside its range.

    Extrapolation is refused. An extrapolated number would be an invention.
    """
    pts = sorted((r["ms_per_decision"], r["win"]) for r in curve)
    if len(pts) < 2:
        return None
    lo, hi = pts[0][0], pts[-1][0]
    if t < lo or t > hi:
        return None
    f = (lambda x: math.log(x)) if log else (lambda x: x)
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= t <= x1:
            if x1 == x0:
                return y0
            w = (f(t) - f(x0)) / (f(x1) - f(x0))
            return y0 + w * (y1 - y0)
    return None


def interpolated_analysis(rows):
    out = []
    for algo in sorted({r["algo"] for r in rows}):
        con = [r for r in rows if r["algo"] == algo and r["sampler"] == "constrained"]
        unc = [r for r in rows if r["algo"] == algo and r["sampler"] == "unconstrained"]
        if len(unc) < 2:
            continue
        for c in sorted(con, key=lambda r: r["budget"]):
            t = c["ms_per_decision"]
            lin = interp(unc, t, log=False)
            lg = interp(unc, t, log=True)
            row = {"algo": algo, "budget": c["budget"],
                   "constrained_win": c["win"], "constrained_ci": c["ci"],
                   "matched_ms": t,
                   "unconstrained_at_same_time_linear": None if lin is None else round(lin, 4),
                   "unconstrained_at_same_time_log": None if lg is None else round(lg, 4)}
            if lin is not None and lg is not None:
                d_lin, d_log = c["win"] - lin, c["win"] - lg
                row["delta_linear"] = round(d_lin, 4)
                row["delta_log"] = round(d_log, 4)
                # A verdict that depends on the interpolation is not a verdict.
                row["agrees_across_interpolation"] = (d_lin > 0) == (d_log > 0)
                # Does the constrained CI exclude the interpolated value?
                row["ci_excludes_interpolated"] = (
                    c["ci"][0] > max(lin, lg) or c["ci"][1] < min(lin, lg))
            else:
                row["note"] = ("outside the measured unconstrained range; "
                               "extrapolation refused")
            out.append(row)
    return out


def paired_signflip(va, vb, iters=20000, seed=0):
    """Paired two-sided test on the SAME deals.

    Both arms play the identical deal sequence, so the per-deal difference
    removes deal luck entirely. Under the null that the two samplers are
    equally strong, swapping their labels on a deal flips the sign of that
    deal's difference, so sign-flipping is an exact randomisation test.
    """
    n = min(len(va), len(vb))
    if n < 5:
        return None, None
    d = [(va[i] - vb[i]) / 4.0 for i in range(n)]
    obs = sum(d) / n
    rng = random.Random(seed)
    hits = 0
    for _ in range(iters):
        s_ = sum(x if rng.random() < 0.5 else -x for x in d) / n
        if abs(s_) >= abs(obs) - 1e-12:
            hits += 1
    return (hits + 1) / (iters + 1), obs


def measured_analysis(deals, opponent, seed, rows, refine=2):
    """Actually run the unconstrained sampler at time-matched budgets.

    The budget is refined iteratively: the first estimate uses a calibration
    constant, then each attempt re-estimates ms-per-unit from what was actually
    achieved. Without this the match error ran to -12%, and always in the same
    direction -- the unconstrained arm under-resourced, which biases the
    comparison toward the constrained sampler being compared against it.
    """
    import runenv
    from dahaljeet.agents import REGISTRY
    from dahaljeet.search import ISMCTSAgent, PIMCAgent
    from dahaljeet.tournament import duplicate_match
    from ablate_determinize import sampler, unconstrained

    out = []
    for algo in sorted({r["algo"] for r in rows}):
        con = [r for r in rows if r["algo"] == algo and r["sampler"] == "constrained"]
        unc = [r for r in rows if r["algo"] == algo and r["sampler"] == "unconstrained"]
        if not unc:
            continue
        # unconstrained cost is very close to linear in budget; calibrate on the
        # measured points rather than assuming a constant
        k = sum(r["ms_per_decision"] / r["budget"] for r in unc) / len(unc)
        for c in sorted(con, key=lambda r: r["budget"]):
            target_ms = c["ms_per_decision"]
            attempts, tried = [], set()
            budget = max(1, int(round(target_ms / k)))
            for _ in range(1 + refine):
                if budget in tried:
                    break
                tried.add(budget)
                mk = (lambda b=budget: PIMCAgent(worlds=b, rng=random.Random(3))) \
                    if algo == "PIMC" else \
                    (lambda b=budget: ISMCTSAgent(iterations=b, rng=random.Random(3)))
                opp = REGISTRY[opponent](rng=random.Random(4))
                with sampler(unconstrained):
                    rr = duplicate_match(mk(), opp, n_deals=deals, seed=seed,
                                         timed=True)
                attempts.append((abs(rr.ms_per_decision_a - target_ms), budget, rr))
                # re-estimate the per-unit cost from what actually happened
                k_act = rr.ms_per_decision_a / budget
                nxt = max(1, int(round(target_ms / k_act)))
                if nxt == budget:
                    break
                budget = nxt
            attempts.sort(key=lambda t: t[0])
            _, budget, r = attempts[0]
            lo, hi = r.bootstrap_ci()
            achieved = r.ms_per_decision_a
            rec = {"algo": algo, "constrained_budget": c["budget"],
                   "constrained_win": c["win"], "constrained_ci": c["ci"],
                   "constrained_ms": target_ms,
                   "unconstrained_budget_for_time_match": budget,
                   "unconstrained_win": round(r.win_rate_a, 4),
                   "unconstrained_ci": [round(lo, 4), round(hi, 4)],
                   "unconstrained_ms": round(achieved, 3),
                   "time_match_error_pct": round(
                       100 * (achieved - target_ms) / target_ms, 1),
                   "budgets_tried": sorted(tried),
                   "delta_measured": round(c["win"] - r.win_rate_a, 4)}
            # paired test on the identical deal sequence
            cv = c.get("a_wins_per_deal")
            if cv:
                pv, md = paired_signflip(cv, list(r.a_wins_per_deal))
                rec["paired_p"] = None if pv is None else round(pv, 5)
                rec["paired_mean_diff"] = None if md is None else round(md, 4)
            else:
                rec["paired_p"] = None
                rec["paired_note"] = ("constrained arm predates per-deal "
                                      "storage; re-run ablate_determinize.py "
                                      "to enable the paired test")
            rec["ci_overlap"] = not (c["ci"][0] > rec["unconstrained_ci"][1]
                                     or c["ci"][1] < rec["unconstrained_ci"][0])
            out.append(rec)
            print(f"  {algo:6s} constrained b={c['budget']:>3} "
                  f"{c['win']:.4f} @{target_ms:7.2f}ms   vs   unconstrained "
                  f"b={budget:>3} {r.win_rate_a:.4f} @{achieved:7.2f}ms "
                  f"(time match {rec['time_match_error_pct']:+.1f}%)  "
                  f"delta {rec['delta_measured']:+.4f}"
                  + (f"  paired p={rec['paired_p']}" if rec.get("paired_p")
                     else "  [unpaired]"), flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None)
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--deals", type=int, default=300)
    ap.add_argument("--opponent", default="TensThenTricks")
    ap.add_argument("--seed", type=int, default=808)
    a = ap.parse_args()

    doc, rows, path = load_rows(a.input)
    if not rows:
        print("  [wait] no matched-samples artifact yet — run "
              "ablate_determinize.py first")
        sys.exit(0)
    print(f"source: {path}\n")

    print("INTERPOLATED (inference from the measured curves)\n")
    inter = interpolated_analysis(rows)
    for r in inter:
        if "delta_linear" not in r:
            print(f"  {r['algo']:6s} b={r['budget']:>3}  {r['note']}")
            continue
        flag = "" if r["agrees_across_interpolation"] else \
            "  <-- VERDICT FLIPS between linear and log interpolation"
        print(f"  {r['algo']:6s} b={r['budget']:>3} @{r['matched_ms']:7.2f}ms  "
              f"constrained {r['constrained_win']:.4f}  "
              f"unconstrained~ {r['unconstrained_at_same_time_log']:.4f}(log) "
              f"{r['unconstrained_at_same_time_linear']:.4f}(lin)  "
              f"delta {r['delta_log']:+.4f}{flag}")

    measured = []
    if a.measure:
        print("\nMEASURED (unconstrained actually run at time-matched budgets)\n")
        measured = measured_analysis(a.deals, a.opponent, a.seed, rows)

    import runenv
    json.dump({"env": runenv.snapshot(), "source_artifact": path,
               "interpolated": inter, "measured": measured,
               "reading": (
                   "delta > 0 means the void constraint is worth its cost at "
                   "equal wall-clock. INTERPOLATED entries are inferred, not "
                   "observed; where linear and log interpolation disagree the "
                   "comparison is unresolved and must not be reported as a "
                   "result. MEASURED entries were actually run and are the "
                   "claim to cite.")},
              open("matched_time.json", "w"), indent=2)
    print("\n-> matched_time.json")
