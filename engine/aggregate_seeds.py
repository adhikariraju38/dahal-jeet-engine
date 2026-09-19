"""Aggregate multi-seed results into the tables the paper needs.

Single-seed numbers are not publishable. This collects every `rl_*_s*.json`,
groups by method, and reports mean, SD, min and max across seeds -- plus a
paired comparison between reward schemes that respects the seed pairing rather
than pooling runs.

    python aggregate_seeds.py

Writes seed_summary.json and seed_summary.md. Any method with fewer than the
expected number of seeds is listed explicitly, so a gap is visible rather than
silently averaged away.
"""
from __future__ import annotations

import glob
import json
import os
import statistics
import sys
import time

EXPECTED_SEEDS = 3
OPPONENTS = ("Random", "TensThenTricks")


def method_key(d):
    m = d.get("method")
    if m and m not in ("dqn", "ppo"):
        return m
    algo = d.get("algo") or m or "?"
    rew = d.get("reward")
    return f"{algo}-{rew}" if rew else algo


def collect():
    groups = {}
    for f in sorted(glob.glob("rl_*.json")):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        fin = d.get("final")
        if not isinstance(fin, dict):
            continue
        key = method_key(d)
        rec = {"artifact": os.path.basename(f), "seed": d.get("seed"),
               "seconds": d.get("seconds"),
               "env_recorded": bool(d.get("env"))}
        for opp in OPPONENTS:
            v = fin.get(opp)
            if isinstance(v, dict):
                v = v.get("win")
            rec[opp] = v
        groups.setdefault(key, []).append(rec)
    return groups


def summarise(groups):
    out = []
    for key, runs in sorted(groups.items()):
        row = {"method": key, "n_seeds": len(runs),
               "seeds": sorted(r["seed"] for r in runs if r["seed"] is not None),
               "complete": len(runs) >= EXPECTED_SEEDS,
               "all_env_recorded": all(r["env_recorded"] for r in runs),
               "runs": runs}
        for opp in OPPONENTS:
            vals = [r[opp] for r in runs if isinstance(r[opp], (int, float))]
            if not vals:
                continue
            row[opp] = {
                "mean": round(statistics.fmean(vals), 4),
                "sd": round(statistics.pstdev(vals), 4) if len(vals) > 1 else 0.0,
                "min": round(min(vals), 4),
                "max": round(max(vals), 4),
                "n": len(vals),
            }
        secs = [r["seconds"] for r in runs if isinstance(r["seconds"], (int, float))]
        if secs:
            row["mean_seconds"] = round(statistics.fmean(secs), 1)
            row["total_seconds"] = round(sum(secs), 1)
        out.append(row)
    return out


def paired_reward_comparison(groups):
    """Compare reward schemes seed-by-seed, not by pooling runs.

    Pooling would let a lucky seed in one arm masquerade as a real effect.
    Pairing on seed removes the seed's contribution from the comparison.
    """
    out = []
    for algo in ("dqn", "ppo"):
        arms = {k: v for k, v in groups.items() if k.startswith(f"{algo}-")}
        if len(arms) < 2:
            continue
        by_seed = {}
        for name, runs in arms.items():
            for r in runs:
                if r["seed"] is None or r["TensThenTricks"] is None:
                    continue
                by_seed.setdefault(r["seed"], {})[name] = r["TensThenTricks"]
        names = sorted(arms)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                diffs = [s[a] - s[b] for s in by_seed.values()
                         if a in s and b in s]
                if len(diffs) < 2:
                    continue
                mean = statistics.fmean(diffs)
                sd = statistics.pstdev(diffs)
                out.append({
                    "algo": algo, "a": a, "b": b, "n_pairs": len(diffs),
                    "mean_diff": round(mean, 4), "sd_diff": round(sd, 4),
                    "all_same_sign": all(d > 0 for d in diffs)
                                     or all(d < 0 for d in diffs),
                })
    return out


def main():
    groups = collect()
    if not groups:
        print("no rl_*.json artifacts yet")
        return
    rows = summarise(groups)
    paired = paired_reward_comparison(groups)
    rec = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "expected_seeds": EXPECTED_SEEDS,
           "methods": rows, "paired_reward_comparison": paired}
    json.dump(rec, open("seed_summary.json", "w"), indent=2)

    L = ["# Multi-seed summary", "",
         f"Generated {rec['generated']} from `rl_*.json`. "
         f"Expected {EXPECTED_SEEDS} seeds per method.", "",
         "| Method | seeds | vs Random (mean ± SD) | vs best heuristic (mean ± SD) | complete |",
         "|---|---:|---|---|---|"]
    for r in sorted(rows, key=lambda x: -(x.get("TensThenTricks", {}).get("mean") or 0)):
        rnd = r.get("Random", {})
        ttt = r.get("TensThenTricks", {})
        L.append(
            f"| `{r['method']}` | {r['n_seeds']} | "
            f"{rnd.get('mean','—')} ± {rnd.get('sd','—')} | "
            f"**{ttt.get('mean','—')} ± {ttt.get('sd','—')}** | "
            f"{'yes' if r['complete'] else '⚠️ NO'} |")

    incomplete = [r for r in rows if not r["complete"]]
    if incomplete:
        L += ["", "## ⚠️ Incomplete — do not report these as multi-seed", ""]
        for r in incomplete:
            L.append(f"- `{r['method']}`: {r['n_seeds']} seed(s) "
                     f"{r['seeds']} — needs {EXPECTED_SEEDS}")

    noenv = [r for r in rows if not r["all_env_recorded"]]
    if noenv:
        L += ["", "## ⚠️ Missing environment capture", ""]
        for r in noenv:
            L.append(f"- `{r['method']}`")

    if paired:
        L += ["", "## Paired reward comparison (same seed, both arms)", "",
              "Pairing on seed removes the seed's own contribution. "
              "`consistent` means every seed agreed on the direction.", "",
              "| Algo | A | B | pairs | mean(A−B) | SD | consistent |",
              "|---|---|---|---:|---:|---:|---|"]
        for p in paired:
            L.append(f"| {p['algo']} | `{p['a']}` | `{p['b']}` | {p['n_pairs']} | "
                     f"{p['mean_diff']:+.4f} | {p['sd_diff']:.4f} | "
                     f"{'yes' if p['all_same_sign'] else 'no'} |")

    L += ["", "## Compute", "",
          "| Method | mean s/run | total s |", "|---|---:|---:|"]
    for r in sorted(rows, key=lambda x: -(x.get("total_seconds") or 0)):
        if r.get("total_seconds"):
            L.append(f"| `{r['method']}` | {r['mean_seconds']:.0f} | "
                     f"{r['total_seconds']:.0f} |")

    open("seed_summary.md", "w").write("\n".join(L) + "\n")
    done = sum(1 for r in rows if r["complete"])
    print(f"  {len(rows)} methods, {done} with all {EXPECTED_SEEDS} seeds")
    print("  -> seed_summary.json / seed_summary.md")


if __name__ == "__main__":
    main()
