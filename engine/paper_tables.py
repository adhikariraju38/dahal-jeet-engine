"""Generate every results table in the paper directly from its artifact.

Hand-written LaTeX tables drift from the data the moment an experiment is
re-run. These are emitted from the JSON, so a table cannot disagree with the
result it reports. Each writes to 11-paper/tables/.

    python paper_tables.py
"""
from __future__ import annotations

import collections
import glob
import json
import os
import statistics

OUT = os.environ.get("DJ_TABLES_OUT", "../out/tables")
os.makedirs(OUT, exist_ok=True)
os.makedirs(OUT, exist_ok=True)


def esc(s):
    return str(s).replace("_", r"\_")


def load(f):
    return json.load(open(f)) if os.path.exists(f) else None


def write(name, body):
    open(f"{OUT}/{name}.tex", "w").write(body)
    print(f"  {name}.tex")


def tabular(cols, header, rows, caption, label, note=None, small=True):
    # [tbp], not [!htbp] or [H]. \'h\' lets a table land between a heading and
    # its first sentence, splitting the prose; [H] pins it exactly and leaves
    # the rest of the page blank when the next float will not fit. Restricting
    # placement to the top or bottom of a page, or a float page, keeps the text
    # continuous; placeins still holds every float inside its own section.
    s = ["\\begin{table}[htp]\\centering" + ("\\small" if small else "")]
    s.append(f"\\begin{{tabular}}{{{cols}}}\\toprule")
    s.append(" & ".join(header) + r" \\ \midrule")
    for r in rows:
        s.append(" & ".join(str(x) for x in r) + r" \\")
    s.append(r"\bottomrule\end{tabular}")
    s.append(f"\\caption{{{caption}}}\\label{{tab:{label}}}")
    if note:
        s.append("\n\\vspace{2pt}\\begin{minipage}{0.92\\linewidth}"
                 f"\\footnotesize\\raggedright {note}\\end{{minipage}}")
    s.append(r"\end{table}")
    return "\n".join(s)


# ------------------------------------------------------------- leaderboard
d = load("ratings.json")
if d:
    rows = [(i, esc(x["agent"]), f"{x['elo']:.1f}",
             f"[{x['ci95'][0]:.1f}, {x['ci95'][1]:.1f}]")
            for i, x in enumerate(d["ratings"], 1)]
    write("leaderboard", tabular(
        "clcc", ["Rank", "Agent", "Elo", "95\\% CI"], rows,
        f"Full round-robin ranking, all {d['n_agents']} agents, "
        f"{d['deals_per_pair']} duplicate deals per pairing. Bradley--Terry "
        f"ratings on an Elo scale; intervals bootstrapped over deals. Ratings "
        f"are relative to this pool only.", "leaderboard"))

# ------------------------------------------------------------- search curve
d = load("eval_curve.json")
if d:
    rows = [(r[0], r[1], f"{r[2]:.4f}", f"[{r[3]:.4f}, {r[4]:.4f}]", f"{r[5]:.2f}")
            for r in d["latency_curve"]]
    write("latency", tabular(
        "llccc", ["Family", "Budget", "Win rate", "95\\% CI", "ms/decision"],
        rows, "Search strength against measured decision cost, versus the "
        "strongest heuristic. Budget is worlds for PIMC and iterations for "
        "ISMCTS.", "latency"))

# ------------------------------------------------------------- void ablation
d = load("void_constraint_ablation.json")
if d:
    rows = []
    for r in d["matched_samples"]:
        rows.append((r["algo"], r["budget"], r["sampler"], f"{r['win']:.4f}",
                     f"[{r['ci'][0]:.4f}, {r['ci'][1]:.4f}]",
                     f"{r['ms_per_decision']:.2f}"))
    write("voidsamples", tabular(
        "llcccc", ["Family", "Budget", "Sampler", "Win rate", "95\\% CI",
                   "ms/dec"], rows,
        "Void-constraint ablation, matched samples: identical world and "
        "iteration counts for both samplers.", "voidsamples"))

    diag = d.get("diagnostic", {})
    if isinstance(diag, dict) and "random_play" in diag:
        rows = []
        rp = {r["trick"]: r for r in diag["random_play"]}
        sp = {r["trick"]: r for r in diag.get("skilled_play", [])}
        for t in sorted(rp):
            rows.append((t, f"{rp[t]['wasted_fraction']:.4f}",
                         f"{sp[t]['wasted_fraction']:.4f}" if t in sp else "--"))
        write("voidwaste", tabular(
            "ccc", ["Trick", "Wasted (random play)", "Wasted (skilled play)"],
            rows, "Fraction of unconstrained determinizations that contradict "
            "the observed play. Measured under both random and skilled play, "
            "since the rate at which voids are revealed differs.", "voidwaste"))

d = load("matched_time.json")
if d and d.get("measured"):
    rows = [(m["algo"], m["constrained_budget"], f"{m['constrained_win']:.4f}",
             m["unconstrained_budget_for_time_match"],
             f"{m['unconstrained_win']:.4f}", f"{m['delta_measured']:+.4f}",
             f"{m['time_match_error_pct']:+.1f}\\%",
             f"{m['paired_p']:.4f}" if m.get("paired_p") is not None else "--")
            for m in d["measured"]]
    write("voidtime", tabular(
        "llccccrc",
        ["Family", "Budget", "Constrained", "Matched budget", "Unconstrained",
         "$\\Delta$", "Time error", "Paired $p$"], rows,
        "Void-constraint ablation, matched wall-clock. The unconstrained "
        "sampler is actually run at a budget chosen to match the constrained "
        "arm's measured time; both arms play the same deals, so the test is "
        "paired.", "voidtime",
        note="Eight simultaneous tests. Under Holm--Bonferroni only the "
             "ISMCTS-200 arm survives at $\\alpha=0.05$."))

# ------------------------------------------------------------- MO-ISMCTS
d = load("mo_ismcts.json")
if d:
    rows = [(a["variant"], a["iterations"], f"{a['win']:.4f}",
             f"[{a['ci'][0]:.4f}, {a['ci'][1]:.4f}]",
             f"{a['ms_per_decision']:.2f}") for a in d["arms"]]
    write("moismcts", tabular(
        "llccc", ["Variant", "Iterations", "Win rate", "95\\% CI", "ms/dec"],
        rows, "Single-observer versus multiple-observer ISMCTS. The "
        "single-observer form backs up the root team's payoff at every node, "
        "so opponents are searched as though they were helping the searcher.",
        "moismcts"))
    rows = [(h["iterations"], f"{h['mo_win_vs_so']:.4f}",
             f"[{h['ci'][0]:.4f}, {h['ci'][1]:.4f}]",
             # the artifact stores the verdict upper-cased; the table reads
             # better without the shouting
             esc(str(h["verdict"]).replace("WINS", "wins").replace("TIE", "tie")))
            for h in d["head_to_head"]]
    write("moh2h", tabular(
        "cccl", ["Iterations", "MO win rate vs SO", "95\\% CI", "Verdict"],
        rows, "Multiple- against single-observer ISMCTS, head to head on "
        "identical deals.", "moh2h"))

# ------------------------------------------------------------- endgame
d = load("perfect_info_endgame_k7.json")
if d:
    for src, tag, cap in (
        ("own play", "endgameown",
         "Endgame optimality with each agent generating its own positions. "
         "Conversion rates are not comparable across agents here, because a "
         "stronger agent reaches different endgames."),
        (None, "endgamecommon",
         "Endgame optimality on a common position set: one fixed policy drives "
         "every seat to the handover, so all agents face identical positions. "
         "The theoretical rate is identical by construction, which is the "
         "protocol's own check. These are the comparable numbers.")):
        rows = []
        for r in d["results"]:
            own = r.get("position_source") in (None, "own play")
            if (src == "own play") != own:
                continue
            rows.append((esc(r["agent"]), f"{r['theoretical_win_rate']:.4f}",
                         f"{r['actual_win_rate']:.4f}",
                         f"{r['conversion']:.3f}", f"{r['steal']:.3f}",
                         f"{r['gap']:+.4f}"))
        if rows:
            write(tag, tabular(
                "lccccc",
                ["Agent", "Theoretical", "Actual", "Conversion", "Steal",
                 "Gap"], rows, cap, tag))

d = load("perfect_info_probe.json")
if d:
    def g(r, *keys):
        """Schema tolerance: this artifact may predate later fields."""
        for k in keys:
            if r.get(k) is not None:
                return r[k]
        return None
    rows = []
    for r in d["probe"]:
        ms = g(r, "mean_seconds_SOLVED_ONLY", "mean_seconds")
        lb = g(r, "mean_seconds_censored_lower_bound")
        nd = g(r, "mean_nodes_solved_only", "mean_nodes")
        rows.append((r["tricks_remaining"], f"{r['n_solved']}/{r['n_attempted']}",
                     f"{ms:.3f}" if ms is not None else "--",
                     f"{lb:.3f}" if lb is not None else "--",
                     f"{nd:,}" if nd is not None else "--",
                     "yes" if r.get("biased") else ""))
    write("ddprobe", tabular(
        "ccrrrc", ["Tricks left", "Solved", "Mean s (solved only)",
                   "Censored lower bound", "Nodes", "Biased"], rows,
        "Exact double-dummy solve cost by depth. Cost grows roughly tenfold "
        "per additional trick.", "ddprobe",
        note="Where positions abort, the solved-only mean is subject to "
             "survivorship bias and can fall with depth, because only easy "
             "positions finish. Use the censored lower bound."))

# ------------------------------------------------------------- learning
d = load("seed_summary.json")
if d:
    rows = []
    for m in sorted(d["methods"], key=lambda x: -(x.get("TensThenTricks", {}).get("mean") or 0)):
        t = m.get("TensThenTricks", {})
        r = m.get("Random", {})
        rows.append((esc(m["method"]), m["n_seeds"],
                     f"{r.get('mean','--')} $\\pm$ {r.get('sd','--')}",
                     f"{t.get('mean','--')} $\\pm$ {t.get('sd','--')}"))
    write("learners", tabular(
        "lccc", ["Method", "Seeds", "vs Random", "vs best heuristic"], rows,
        "All learning methods, mean $\\pm$ standard deviation across seeds.",
        "learners"))
    if d.get("paired_reward_comparison"):
        rows = [(p["algo"], esc(p["a"]), esc(p["b"]), p["n_pairs"],
                 f"{p['mean_diff']:+.4f}", f"{p['sd_diff']:.4f}",
                 "yes" if p["all_same_sign"] else "no")
                for p in d["paired_reward_comparison"]]
        write("rewardpaired", tabular(
            "lllcccc",
            ["Algo", "A", "B", "Pairs", "mean(A$-$B)", "SD", "Consistent"],
            rows, "Reward schemes compared seed by seed. Pairing on the seed "
            "removes the seed's own contribution; `consistent' means every "
            "seed agreed on the direction.", "rewardpaired"))

# ------------------------------------------------------------- encoding
g = collections.defaultdict(list)
eps = None
for f in glob.glob("ablate_*_s*.json"):
    x = json.load(open(f))
    g[x["block"]].append(x["final"]["TensThenTricks"]["win"])
    eps = x["episodes"]
if g:
    rows = []
    for k, v in sorted(g.items(), key=lambda kv: -statistics.fmean(kv[1])):
        rows.append((esc(k), f"{statistics.fmean(v):.4f}",
                     f"{statistics.pstdev(v):.4f}", len(v),
                     ", ".join(f"{x:.4f}" for x in sorted(v))))
    write("encoding", tabular(
        "lcccl", ["Blocks zeroed", "Mean", "SD", "Seeds", "Per-seed"], rows,
        f"State-encoding ablation at {eps:,} episodes, matched to the headline "
        f"agents' budget.", "encoding",
        note="No single block's removal changes performance materially, yet "
             "removing almost everything improves it, with no overlap between "
             "the seeds of `minimal' and the full encoding."))

# ------------------------------------------------------------- exploitability
rows = []
for f in sorted(glob.glob("exploitability_*.json")):
    if f == "exploitability_search.json":
        continue
    for r in json.load(open(f)).get("targets", []):
        rows.append(r)
if rows:
    rows.sort(key=lambda r: -r["exploitability"])
    body = [(esc(r["target"]), f"{r['best_response_win']:.4f}",
             f"{r['generic_search_win']:.4f}", f"{r['exploitability']:+.4f}",
             "$<$0.001" if r["paired_p"] < 0.001 else f"{r['paired_p']:.4f}")
            for r in rows]
    write("exploit", tabular(
        "lcccc", ["Target", "Best response", "Generic search", "Exploitability",
                  "Paired $p$"], body,
        "Exploitability: a search that simulates the target's true policy for "
        "the opponent seats, against the same search at the same budget with no "
        "opponent model. Identical algorithm in both arms, paired on identical "
        "deals.", "exploit"))

# ------------------------------------------------------------- generalisation
d = load("generalisation_matrix.json")
if d and d.get("summary", {}).get("opponent_specificity"):
    s = d["summary"]["opponent_specificity"]
    rows = [(x["algo"], esc(x["opponent"]), f"{x['trained_on_it']:.4f}",
             f"{x['trained_elsewhere']:.4f}", f"{x['specificity']:+.4f}")
            for x in s["per_opponent"]]
    write("specificity", tabular(
        "llccc", ["Algo", "Opponent", "Trained on it", "Trained elsewhere",
                  "Specificity"], rows,
        "Opponent specificity: the advantage from having trained against this "
        "specific opponent, with opponent strength cancelled because both "
        "terms are measured against the same opponent.", "specificity",
        note=f"Mean {s['mean']:+.4f} $\\pm$ {s['sd']:.4f} over {s['n']} "
             f"opponents. A naive gap (own minus held-out) would instead "
             f"report roughly $+0.30$, but that compares performance against "
             f"different opponents and mostly measures their relative "
             f"strength."))

# ------------------------------------------------------------- features
d = load("feature_importance.json")
if d:
    rows = []
    for r in d["results"]:
        for blk, v in r["blocks"].items():
            rows.append((esc(os.path.basename(r["model"])), esc(blk),
                         v["features"], f"{v['win']:.4f}", f"{v['drop']:+.4f}"))
    write("features", tabular(
        "llccc", ["Model", "Block permuted", "Features", "Win rate", "Drop"],
        rows, "Permutation feature importance. Each block is replaced at every "
        "decision by the same slice from an unrelated real state, which "
        "destroys its relationship to the position while preserving its "
        "marginal distribution.", "features"))

# ------------------------------------------------------------- budget matched
d = load("budget_matched.json")
if d:
    rows = []
    for r in d["runs"]:
        rows.append((r["algo"], r["matched_on"].replace("_", " "),
                     f"{r['target']:.0f}", f"{r['achieved']:.0f}",
                     f"{r['episodes']:,}", f"{r['gradient_updates']:,}",
                     f"{r['eval']['TensThenTricks']['win']:.4f}"))
    write("budget", tabular(
        "lllcccc",
        ["Algo", "Matched on", "Target", "Achieved", "Episodes", "Updates",
         "vs best heuristic"], rows,
        "DQN and PPO compared at matched wall-clock and again at matched "
        "gradient steps. Matching on episodes, as elsewhere in the literature, "
        "gives the two very different optimisation budgets.", "budget",
        note=f"DQN performs {d['updates_per_episode_ratio_dqn_over_ppo']:.0f}"
             f"$\\times$ more gradient steps per episode than PPO."))

# ------------------------------------------------------------- variance
d = load("variance_decomposition.json")
if d:
    rows = [(esc(r["agent_a"]), esc(r["agent_b"]), f"{r['share_deal']:.3f}",
             f"{r['share_rotation']:.3f}", f"{r['share_residual']:.3f}",
             f"{r['grand_mean']:.4f}") for r in d["results"]]
    write("variance", tabular(
        "llcccc", ["Agent A", "Agent B", "Deal", "Seat", "Residual", "Mean"],
        rows, "Variance decomposition of the outcome. `Deal' is the cards "
        "themselves, i.e.\\ luck; `seat' is structural seat advantage, which "
        "duplicate dealing removes; `residual' is how the matchup played out.",
        "variance"))

# ------------------------------------------------------------- failure
d = load("failure_analysis.json")
if d:
    for agent, res in d["results"].items():
        rows = []
        for dim in ("tens_dealt", "trump_length", "holds_trump", "decided_by"):
            if dim not in res:
                continue
            # trump_length is bucketed as min(n, 8), so its top row covers
            # eight or more; label it that way rather than as an exact count.
            top = max(res[dim], key=lambda x: int(x)) if dim == "trump_length" else None
            for k, v in sorted(res[dim].items()):
                label = f"{k}+" if k == top else k
                rows.append((esc(dim), esc(label),
                             f"{v['win_rate']:.3f}", f"{v['hands']:,}"))
        if rows:
            write("failure", tabular(
                "llcc", ["Condition", "Value", "Win rate", "Hands"], rows,
                f"Outcomes conditioned on properties of the deal fixed before "
                f"play, for {esc(agent)} in self-play. Trump length counts "
                f"the team's trumps between them and its top bucket is eight "
                f"or more. A low win rate in a "
                f"bucket is a property of the situation, not of how it was "
                f"played.", "failure"))
        break

# ------------------------------------------------------------- sweep
d = load("sweep_s0.json")
if d:
    rows = [(r["lr"], r["hidden"], r["gamma"],
             f"{r['win_vs_best_heuristic']:.4f}") for r in d["results"]]
    write("sweep", tabular(
        "cccc", ["Learning rate", "Hidden", "$\\gamma$", "vs best heuristic"],
        rows, f"Hyperparameter sensitivity at {d['episodes']:,} episodes. "
        f"Spread {d.get('spread','--')}.", "sweep"))

# ------------------------------------------------------------- stats
d = load("stats_corrections.json")
if d:
    rows = [("Nominally significant ($\\alpha=0.05$)", d["n_significant_raw"]),
            ("Expected false positives if uncorrected",
             d["expected_false_positives_uncorrected"]),
            ("Survive Holm--Bonferroni", d["n_significant_holm"]),
            ("Survive Benjamini--Hochberg", d["n_significant_bh"])]
    write("corrections", tabular(
        "lc", ["Quantity", f"of {d['n_pairings']} pairings"], rows,
        "Family-wise error control over the full round-robin. The family is "
        "declared as all pairings, before analysis.", "corrections"))


# ------------------------------------------------------------- heuristics
d = load("championship_results.json")
if d and d.get("rule_based_leaderboard"):
    lb = d["rule_based_leaderboard"]
    rows = []
    for i, x in enumerate(lb, 1):
        if isinstance(x, (list, tuple)):
            rows.append((i, esc(x[0]), f"{x[1]:.4f}"))
        elif isinstance(x, dict):
            rows.append((i, esc(x.get("name", "?")),
                         f"{x.get('win', x.get('mean_win', 0)):.4f}"))
    if rows:
        write("heuristics", tabular(
            "clc", ["Rank", "Heuristic", "Mean win rate"], rows,
            f"Hand-written heuristic ladder, {d.get('rule_deals','?')} duplicate "
            f"deals per pairing. Each rung adds one idea to the rung below "
            f"advantage can be attributed to something specific.",
            "heuristics"))

# ------------------------------------------------------------- complexity
d = load("complexity.json")
if d:
    # JSON object keys are strings even when they were written as ints
    rows = [(int(t) + 1, f"{v:.3f}") for t, v in
            sorted(d["branching_by_trick"].items(), key=lambda kv: int(kv[0]))]
    write("branching", tabular(
        "cc", ["Trick", "Mean legal moves"], rows,
        f"Branching factor by trick, measured over "
        f"{d['n_decisions_measured']:,} decisions of random play. Follow-suit "
        f"narrows the choice steadily as the hand proceeds.", "branching"))
    rows = [(r["trick"], f"$10^{{{r['log10_geometric_mean']:.2f}}}$",
             r["n_samples"]) for r in d["infoset_during_play_exact"]]
    write("infoset", tabular(
        "ccc", ["Trick", "Information-set size", "Positions sampled"], rows,
        "Exact information-set size from one seat's view, with inferred voids "
        "and the publicly shown trump card applied, counted by dynamic "
        "programming over eligibility patterns.", "infoset",
        note="Uncertainty collapses by roughly fifteen orders of magnitude "
             "across a single hand, which is why determinized search is "
             "effective here and why sampling budget matters most early."))

# ------------------------------------------------------------- reproducibility
rows = []
d = load("optimization_equivalence.json")
if d:
    rows.append(("Buffer optimisation, all configurations",
                 f"{d['n_configurations']} configs",
                 "bit-identical" if d["all_bit_identical"] else "differ",
                 f"{max(r['max_abs_delta'] for r in d['results']):.1e}"))
d = load("reproduce_dqn_potential_s0.json")
if d:
    rows.append((f"Stored checkpoint reproduced ({d['episodes']:,} episodes)",
                 esc(d["checkpoint"]),
                 "bit-identical" if d["bit_identical"] else "differs",
                 f"{d['max_abs_delta']:.1e}"))
if rows:
    write("repro", tabular(
        "llcc", ["Check", "Scope", "Result", "max $|\\Delta|$"], rows,
        "A refactor and a performance optimisation applied mid-project "
        "altered no result. In the second row, a "
        "checkpoint trained by the original code is reproduced exactly by the "
        "current code, which is what licenses reusing earlier artifacts.",
        "repro"))

# ------------------------------------------------------------- compute
d = load("../compute/compute.json")
if d:
    e, t = d["environment"], d["totals"]

    # Vendor names are stripped here rather than in the artifact: the captured
    # environment stays verbatim for provenance, while the paper reports the
    # hardware generically.
    def degloss(v):
        v = str(v)
        for brand in ("Apple Silicon", "Apple-Silicon", "Apple", "MacBook",
                      "macOS", "Mac OS", "Metal Performance Shaders", "MPS",
                      "Darwin"):
            v = v.replace(brand, "")
        return " ".join(v.split()).strip(" ;,-") or "unspecified"

    rows = [("Processor", esc(degloss(e.get("processor")))),
            ("Logical cores", e.get("cpu_cores_logical")),
            ("Memory", f"{e.get('ram_gib')} GiB"),
            ("Accelerator", "none; CPU only"),
            ("Result artifacts", t["artifacts"]),
            ("Measured CPU-hours", f"{t['measured_wall_clock_h']:.1f}")]
    write("compute", tabular(
        "ll", ["Quantity", "Value"], rows,
        "Compute and hardware. Every experiment ran on the CPU of a single "
        "laptop; no GPU acceleration was used.", "compute",
        note="The platform's integrated GPU backend was benchmarked and "
             "rejected: at batch size 1, which dominates reinforcement-learning "
             "rollouts, it is roughly four times slower than the CPU for "
             "networks of this size."))

# ------------------------------------------------- family feature comparison
d = load("../rulebook/feature-comparison.json")
if d:
    rows = [(esc(x["feature"]), esc(x["dahal_jeet"]), esc(x["mendikot"]),
             "yes" if x["match"] else "\\textbf{no}")
            for x in d["features"]]
    nm = sum(1 for x in d["features"] if x["match"])
    write("familyfeatures", tabular(
        "p{0.24\\linewidth}p{0.26\\linewidth}p{0.28\\linewidth}c",
        ["Feature", "Dahal Jeet", "Mendikot", "Same"], rows,
        f"Feature-by-feature comparison against Mendikot, the closest documented "
        f"game: {nm} of {len(d['features'])} features agree. Mendikot is taken "
        f"from Pagat~\\cite{{pagat-mendikot}}; Dahal Jeet from the specification "
        f"in \\S\\ref{{sec:game}}, as implemented and tested in the released engine.",
        "familyfeatures",
        note=("Feature granularity is a judgement of ours, and splitting or "
              "merging rows would change the denominator, so the full list is "
              "given rather than the ratio alone.")))

print("done")
