"""Emit every number the paper quotes as a LaTeX macro.

The rule this project runs on is that no number is typed into prose. Each one
is pulled from its artifact here and written to 11-paper/numbers.tex as a
\\newcommand, so the manuscript cites \\EloISMCTS rather than "1663.2". If an
experiment is re-run, the paper updates; it cannot silently disagree with the
data.

    python paper_numbers.py
"""
from __future__ import annotations

import collections
import glob
import json
import os
import statistics

OUT = os.environ.get("DJ_NUMBERS_OUT", "../out/numbers.tex")
os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
M = {}


def sanitise(name):
    """LaTeX macro names may contain letters only -- no digits, no punctuation.

    \\newcommand{\\ConvISMCTS200} is a hard error, which is how this was found.
    Digits are spelled out so distinct budgets stay distinct macros.
    """
    words = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four",
             "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}
    return "".join(words.get(c, c) for c in name if c.isalnum())


def put(k, v):
    M[sanitise(k)] = v


def load(f):
    return json.load(open(f)) if os.path.exists(f) else None


# ---------------------------------------------------------------- complexity
d = load("complexity.json")
if d:
    def sci(x):
        """Proper LaTeX scientific notation. `5.364e+28` is not how a paper
        writes a number."""
        m, e = f"{x:.4e}".split("e")
        return f"{float(m):.3f}\\times 10^{{{int(e)}}}"
    put("DealCount", sci(float(d["deals_total"])))
    put("DealCountLog", f"{d['deals_total_log10']:.2f}")
    put("InitInfoSet", sci(float(d["initial_infoset"])))
    put("InitInfoSetLog", f"{d['initial_infoset_log10']:.2f}")
    put("Branching", f"{d['mean_branching_factor']:.3f}")
    put("BranchingN", f"{d['n_decisions_measured']:,}")
    put("GameTreeLog", f"{d['game_tree_log10_estimate']:.1f}")
    rows = {r["trick"]: r["log10_geometric_mean"] for r in d["infoset_during_play_exact"]}
    for t in (1, 4, 7, 10, 12):
        if t in rows:
            put(f"InfoSetTrick{t}", f"{rows[t]:.2f}")

# ---------------------------------------------------------------- ratings
d = load("ratings.json")
if d:
    r = d["ratings"]
    put("NAgents", str(d["n_agents"]))
    put("TournamentDeals", str(d["deals_per_pair"]))
    for i, x in enumerate(r[:6], 1):
        put(f"EloRank{i}Name", x["agent"].replace("_", "\\_"))
        put(f"EloRank{i}", f"{x['elo']:.1f}")
        put(f"EloRank{i}Lo", f"{x['ci95'][0]:.1f}")
        put(f"EloRank{i}Hi", f"{x['ci95'][1]:.1f}")
    put("EloLastName", r[-1]["agent"].replace("_", "\\_"))
    put("EloLast", f"{r[-1]['elo']:.1f}")
    it = d["intransitivity"]
    put("CyclicEnergy", f"{it['cyclic_energy_ratio']:.4f}")
    put("CyclicFloor", f"{it['noise_floor']['null_mean']:.4f}")
    put("CyclicFloorPNF", f"{it['noise_floor']['null_p95']:.4f}")
    put("CyclicP", f"{it['p_value_vs_transitive_null']:.3f}")
    put("CyclicRho", f"{it['noise_floor']['intra_deal_correlation_rho']:.3f}")
    put("NCycles", str(it["n_3cycles"]))
    put("NTriples", f"{it['n_triples']:,}")

# ---------------------------------------------------------------- statistics
d = load("stats_corrections.json")
if d:
    put("NPairings", str(d["n_pairings"]))
    put("SigRaw", str(d["n_significant_raw"]))
    put("SigHolm", str(d["n_significant_holm"]))
    put("SigBH", str(d["n_significant_bh"]))
    put("ExpectedFP", f"{d['expected_false_positives_uncorrected']:.1f}")

d = load("power.json")
if d:
    put("PowerSD", f"{d['per_deal_sd_median']:.4f}")
    put("PowerFive", str(d["required_deals"]["delta=0.05"]["power80"]))
    put("PowerTwo", str(d["required_deals"]["delta=0.02"]["power80"]))

# ---------------------------------------------------------------- variance
d = load("variance_decomposition.json")
if d and d["results"]:
    r = d["results"][0]
    put("VarDeal", f"{r['share_deal']:.3f}")
    put("VarSeat", f"{r['share_rotation']:.3f}")
    put("VarResid", f"{r['share_residual']:.3f}")
    put("VarPairA", r["agent_a"].replace("_", "\\_"))
    put("VarPairB", r["agent_b"].replace("_", "\\_"))
    shares = [x["share_deal"] for x in d["results"]]
    put("VarDealLo", f"{min(shares):.3f}")
    put("VarDealHi", f"{max(shares):.3f}")

# ---------------------------------------------------------------- endgame
d = load("perfect_info_endgame_k7.json")
if d:
    common = [r for r in d["results"] if r.get("position_source") not in (None, "own play")]
    put("EndgameTricks", str(d["tricks_remaining"]))
    for r in common:
        key = r["agent"].replace(":", "").replace("_", "")
        put(f"Conv{key}", f"{r['conversion']:.3f}")
        put(f"Steal{key}", f"{r['steal']:.3f}")
        put(f"Label{key}", r["agent"].replace(":", "("). replace("_", "\\_")
            + (")" if ":" in r["agent"] else ""))

d = load("perfect_info_probe.json")
if d:
    solved = [r for r in d["probe"] if r.get("n_aborted", 0) == 0]
    if solved:
        put("DDMaxDepth", str(max(r["tricks_remaining"] for r in solved)))

# ---------------------------------------------------------------- exploitability
parts = [f for f in glob.glob("exploitability_*.json") if f != "exploitability_search.json"]
rows = []
for f in parts:
    rows.extend(json.load(open(f)).get("targets", []))
if rows:
    put("ExploitN", str(len(rows)))
    put("ExploitSig", str(sum(1 for r in rows if r["paired_p"] < 0.05)))
    for r in rows:
        k = r["target"].replace("_", "")
        put(f"Exploit{k}", f"{r['exploitability']:+.4f}")
        put(f"ExploitP{k}", ("<0.001" if r["paired_p"] < 0.001
                             else f"{r['paired_p']:.3f}"))

# ---------------------------------------------------------------- ablation
g = collections.defaultdict(list)
for f in glob.glob("ablate_*_s*.json"):
    d = json.load(open(f))
    g[d["block"]].append(d["final"]["TensThenTricks"]["win"])
if g:
    put("AblSeeds", str(min(len(v) for v in g.values())))
    for k, v in g.items():
        key = "".join(p.capitalize() for p in k.split("_"))
        put(f"Abl{key}", f"{statistics.fmean(v):.4f}")
        put(f"Abl{key}SD", f"{statistics.pstdev(v):.4f}")
    put("AblGap", f"{statistics.fmean(g['minimal']) - statistics.fmean(g['none']):+.4f}")

# ---------------------------------------------------------------- learning
d = load("seed_summary.json")
if d:
    best = max((m for m in d["methods"] if m.get("TensThenTricks")),
               key=lambda m: m["TensThenTricks"]["mean"])
    put("BestLearnerName", best["method"].replace("_", "\\_"))
    put("BestLearner", f"{best['TensThenTricks']['mean']:.4f}")
    put("BestLearnerSD", f"{best['TensThenTricks']['sd']:.4f}")

d = load("control_untrained.json")
if d:
    v = [x["TensThenTricks"]["mean"] for x in d["aggregated"].values()]
    put("UntrainedLo", f"{min(v):.3f}")
    put("UntrainedHi", f"{max(v):.3f}")

d = load("generalisation_matrix.json")
if d and d.get("summary", {}).get("opponent_specificity"):
    s = d["summary"]["opponent_specificity"]
    put("SpecMean", f"{s['mean']:+.4f}")
    put("SpecSD", f"{s['sd']:.4f}")
    put("SpecN", str(s["n"]))

# ---------------------------------------------------------------- compute
# ---- numbers that used to be typed into the prose by hand -----------------
d = load("failure_analysis.json")
if d:
    r = list(d["results"].values())[0]
    ht = r["holds_trump"]
    # the rate is over the hands in which a side DREW the trump, not over all
    put("TrumpSeatWin", f"{ht['True']['win_rate']:.4f}")
    put("TrumpSeatN", f"{ht['True']['hands']:,}")
    put("TrumpSeatTotal", f"{ht['True']['hands'] + ht['False']['hands']:,}")
    half = 1.959964 * (0.25 / ht["True"]["hands"]) ** 0.5
    put("TrumpSeatCILo", f"{ht['True']['win_rate'] - half:.4f}")
    put("TrumpSeatCIHi", f"{ht['True']['win_rate'] + half:.4f}")

d = load("void_constraint_ablation.json")
if d and d.get("diagnostic"):
    g = d["diagnostic"]
    def at(seq, trick):
        rows = seq if isinstance(seq, list) else list(seq.values())
        for row in rows:
            if row.get("trick") == trick:
                return row["wasted_fraction"]
    put("VoidWasteOneRand", f"{at(g['random_play'], 1) * 100:.1f}")
    put("VoidWasteOneSkill", f"{at(g['skilled_play'], 1) * 100:.1f}")
    put("VoidWasteElevenRand", f"{at(g['random_play'], 11) * 100:.0f}")
    put("VoidWasteElevenSkill", f"{at(g['skilled_play'], 11) * 100:.0f}")

d = load("mo_ismcts.json")
if d:
    by = {}
    for a in d["arms"]:
        by.setdefault(a["iterations"], {})[a["variant"]] = a["win"]
    gains = sorted(v["MO-ISMCTS"] - v["SO-ISMCTS"] for v in by.values() if len(v) == 2)
    put("MOGainLo", f"{gains[0]:.4f}")
    put("MOGainHi", f"{gains[-1]:.4f}")

d = load("perfect_info_endgame_k7.json")
if d:
    own = [r["theoretical_win_rate"] for r in d["results"]
           if r.get("position_source") == "own play"]
    if own:
        put("EndgameTheoLo", f"{min(own):.3f}")
        put("EndgameTheoHi", f"{max(own):.3f}")

d = load("generalisation_matrix.json")
if d and d.get("summary", {}).get("raw_gap_CONFOUNDED"):
    put("NaiveGap", f"{d['summary']['raw_gap_CONFOUNDED']['mean']:+.3f}")

d = load("decomposition.json")
if d and d.get("decompositions"):
    x = d["decompositions"][0]
    put("DecompHands", f"{x['hands']:,}")
    put("DecompTensWin", f"{x['win_a_when_tens_decide']:.4f}")
    put("DecompTieWin", f"{x['win_a_when_tiebreak_decides']:.4f}")
    put("DecompHeadToHead", f"{x['overall_win_a']:.4f}")
    put("DecompAgentA", x["agent_a"])
    put("DecompAgentB", x["agent_b"])

d = load("failure_analysis.json")
if d:
    # Conditioning buckets are properties of the DEAL, fixed before play.
    # trump_length counts the TEAM's trumps and its top bucket is min(n, 8),
    # so it means "eight or more"; both facts are stated in the text.
    r = list(d["results"].values())[0]
    tl, td = r["trump_length"], r["tens_dealt"]
    put("TrumpEightWin", f"{tl['8']['win_rate']:.3f}")
    put("TrumpEightN", f"{tl['8']['hands']:,}")
    put("TrumpFourWin", f"{tl['4']['win_rate']:.3f}")
    put("TensFourWin", f"{td['4']['win_rate']:.3f}")
    put("TensFourN", f"{td['4']['hands']:,}")
    put("TensZeroWin", f"{td['0']['win_rate']:.3f}")
    put("TrumpSpread", f"{tl['8']['win_rate'] - tl['4']['win_rate']:.3f}")
    put("TensSpread", f"{td['4']['win_rate'] - td['0']['win_rate']:.3f}")
    put("FailureDeals", f"{d['deals']:,}")
    put("FailureOpponent", r["overall"]["all"] and d["opponent"])

d = load("../rulebook/feature-comparison.json")
if d:
    fs = d["features"]
    put("FeatMatch", str(sum(1 for x in fs if x["match"])))
    put("FeatTotal", str(len(fs)))
    put("FeatDiff", str(sum(1 for x in fs if not x["match"])))

d = load("budget_matched.json")
if d:
    put("UpdateRatio", f"{d['updates_per_episode_ratio_dqn_over_ppo']:.0f}")

d = load("../compute/compute.json")
if d:
    put("CPUHours", f"{d['totals']['measured_wall_clock_h']:.1f}")
    put("NArtifacts", str(d["totals"]["artifacts"]))
    e = d["environment"]
    # The artifact keeps the captured string verbatim; the paper reports the
    # hardware without the vendor name.
    put("Machine", " ".join(str(e.get("processor", "?"))
                            .replace("Apple", "").split()) or "?")
    put("RAMGiB", str(e.get("ram_gib", "?")))
    put("Cores", str(e.get("cpu_cores_logical", "?")))

# Hyperparameter sweep. The spread is what the manuscript quotes; the table
# itself moves to the supplement in the venue build, so the number has to be
# available to the prose independently.
d = load("sweep_s0.json")
if d:
    put("SweepSpread", f"{d['spread']:.4f}")
    put("SweepEpisodes", f"{d['episodes']:,}")
    put("SweepConfigs", str(len(d.get("results", []))))
    b = d.get("baseline", {})
    if b:
        put("SweepBaseLR", str(b.get("lr")))
        put("SweepBaseHidden", str(b.get("hidden")))

# Equation checks. The manuscript's displayed identities are verified against
# the artifacts by verify_equations.py, and the count comes from that run rather
# than being asserted in the text.
d = load("equation_checks.json")
if d:
    put("EqChecks", str(d["n_checks"]))
    put("EqChecksPass", str(d["n_pass"]))

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    f.write("% AUTO-GENERATED by 05-engine/paper_numbers.py -- do not edit.\n")
    f.write("% Every number in the manuscript comes from an artifact via these macros.\n")
    for k in sorted(M):
        f.write(f"\\newcommand{{\\{k}}}{{{M[k]}}}\n")
print(f"  {len(M)} macros -> {OUT}")
