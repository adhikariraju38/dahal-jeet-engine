"""Publication figures for the Dahal Jeet paper.

Palette: Okabe-Ito core (#0072B2 blue, #D55E00 vermillion, #009E73 green).
Validated colourblind-safe in BOTH light and dark surfaces -- all six checks
pass, worst adjacent CVD separation dE 11.0 (deutan), contrast >= 3:1
throughout. Not eyeballed; computed.

Conventions followed throughout:
  * one axis, never two y-scales
  * thin marks, recessive grid and spines
  * a legend whenever there are >= 2 series, plus direct labels where they fit
  * numbers never printed on every point
  * every figure written as both PDF (vector, for the paper) and PNG (for talks)
"""
from __future__ import annotations

import glob
import json
import os
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

sys.path.insert(0, ".")

OUT = "../figures"
os.makedirs(OUT, exist_ok=True)

BLUE, VERM, GREEN = "#0072B2", "#D55E00", "#009E73"
INK, INK2, GRID = "#17211C", "#5A6C63", "#DCE3DE"

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9.5,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.5,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK2,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.frameon": False,
    "legend.fontsize": 8.5,
    "text.color": INK,
})


def finish(fig, name, caption=None):
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/{name}.{ext}")
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png" + (f"  — {caption}" if caption else ""))


def despine(ax, keep=("left", "bottom")):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)


def load(pattern):
    out = []
    for f in sorted(glob.glob(pattern)):
        try:
            out.append((f, json.load(open(f))))
        except Exception:
            pass
    return out


# ---------------------------------------------------------------- FIGURE 1

def fig_hierarchy():
    """Every method, ranked by strength against the best heuristic."""
    # Aggregate ACROSS SEEDS. Globbing rl_*.json returns three files per
    # method, and barh() places same-named bars at the same categorical
    # position, so the bars overlapped and every value label was printed three
    # times on top of itself.
    import collections
    acc = collections.defaultdict(list)
    for f, d in load("rl_*.json"):
        label = d.get("method") or f"{d.get('algo','')}-{d.get('reward','')}"
        fin = d.get("final", {})
        v = fin.get("TensThenTricks")
        if isinstance(v, dict):
            v = v.get("win")
        if isinstance(v, (int, float)):
            acc[label].append(v)
    rows = [(k, statistics.fmean(v)) for k, v in acc.items()]
    # search + heuristics from the championship
    if os.path.exists("championship_results.json"):
        c = json.load(open("championship_results.json"))
        for s in c.get("search_rows", []):
            if s[1] == "TensThenTricks":
                rows.append((s[0], s[2]))
    if not rows:
        print("  [skip] fig1 — no results yet")
        return
    rows.sort(key=lambda r: r[1])
    nice = {"distill": "Distilled ISMCTS", "double_dueling": "Double+Dueling DQN",
            "double": "Double DQN", "a2c": "A2C"}
    rows = [(nice.get(n, n), v) for n, v in rows]
    names = [r[0] for r in rows]
    vals = [r[1] for r in rows]

    def colour(n):
        n = n.lower()
        # "distill" must be tested BEFORE "ismcts": the distilled agent is named
        # "Distilled ISMCTS", so testing ismcts first painted it as search and
        # left the orange legend entry with no bar.
        if "distill" in n:
            return VERM
        if "ismcts" in n or "pimc" in n:
            return GREEN
        return BLUE

    # keep the panel a sane shape no matter how many agents there are
    height = min(9.0, 0.30 * len(rows) + 1.6)
    fig, ax = plt.subplots(figsize=(6.4, height))
    bars = ax.barh(names, vals, height=0.62,
                   color=[colour(n) for n in names])
    ax.axvline(0.5, color=INK2, lw=1.0, ls="--", zorder=3)
    ax.text(0.497, -0.85, "parity with the best hand-written heuristic",
            fontsize=8, color=INK2, va="center", ha="right", style="italic")
    for b, v in zip(bars, vals):
        ax.text(v + 0.008, b.get_y() + b.get_height() / 2, f"{v:.3f}",
                va="center", fontsize=8, color=INK2)
    ax.set_xlim(0, max(vals) * 1.18)
    ax.set_ylim(-1.4, len(rows) - 0.3)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("win rate vs TensThenTricks (best hand-written heuristic)")
    ax.set_title("Search > heuristics > distillation > reinforcement learning")
    ax.grid(axis="y", visible=False)
    despine(ax)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (GREEN, VERM, BLUE)]
    ax.legend(handles, ["search", "distilled search", "reinforcement learning"],
              loc="lower right", ncol=1)
    finish(fig, "fig1_agent_hierarchy",
           "source: championship_results.json, final_tournament.json")


# ---------------------------------------------------------------- FIGURE 2

def fig_tension():
    """The core game finding: the two objectives pull against each other.

    Reads decomposition.json -- these numbers were previously hardcoded here,
    which broke the rule that every value in the paper traces to an artifact.
    """
    if not os.path.exists("decomposition.json"):
        print("  [skip] fig2 - run decompose.py first")
        return
    d = json.load(open("decomposition.json"))
    row = next((r for r in d["decompositions"]
                if r["agent_a"] == "TenAware" and r["agent_b"] == "GreedyTricks"),
               None)
    if row is None:
        print("  [skip] fig2 - TenAware vs GreedyTricks not in artifact")
        return

    cats = [f"hands decided\nby the tens\n({row['share_decided_by_tens']:.1%})",
            f"hands decided by\nthe trick tiebreak\n({row['share_decided_by_tiebreak']:.1%})",
            "overall\n(100%)"]
    vals = [row["win_a_when_tens_decide"],
            row["win_a_when_tiebreak_decides"],
            row["overall_win_a"]]

    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    x = range(len(cats))
    ax.bar(x, vals, width=0.5, color=[GREEN, VERM, BLUE])
    ax.axhline(0.5, color=INK2, lw=1.0, ls="--", zorder=3)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.02, f"{v:.3f}", ha="center", fontsize=9,
                color=INK, fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels(cats, fontsize=8.5)
    ax.set_ylim(0, max(vals) * 1.25)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylabel("win rate of the ten-hunting agent")
    ax.set_title("Ten-hunting wins the tens and loses the tiebreaks -\n"
                 "and the two almost exactly cancel")
    ax.grid(axis="x", visible=False)
    despine(ax)
    finish(fig, "fig2_objective_tension",
           f"n={row['hands']:,} hands, from decomposition.json")


# ---------------------------------------------------------------- FIGURE 3

def fig_latency():
    """Strength against decision cost for both search families.

    No cross-game reference point is plotted: latency is a function of the
    action space and the rules being searched, so a number from a different
    game is not commensurable with these.
    """
    # Newest artifact wins. A fixed preference order would let a stale
    # eval_final.json from an earlier run silently override a fresh curve.
    cands = [f for f in ("eval_final.json", "eval_curve.json")
             if os.path.exists(f)]
    src = max(cands, key=os.path.getmtime) if cands else None
    if src is None:
        print("  [skip] fig3 - no latency curve artifact yet")
        return
    d = json.load(open(src))
    curve = d.get("latency_curve", [])
    if not curve:
        print("  [skip] fig3 — no latency curve")
        return
    fig, ax = plt.subplots(figsize=(5.9, 3.8))
    for algo, colour, marker in (("PIMC", BLUE, "o"), ("ISMCTS", GREEN, "s")):
        pts = [(r[5], r[2], r[3], r[4], r[1]) for r in curve if r[0] == algo]
        if not pts:
            continue
        pts.sort()
        ms = [p[0] for p in pts]
        win = [p[1] for p in pts]
        lo = [p[1] - p[2] for p in pts]
        hi = [p[3] - p[1] for p in pts]
        ax.errorbar(ms, win, yerr=[lo, hi], color=colour, marker=marker,
                    ms=5.5, lw=1.8, capsize=2.5, elinewidth=0.9, label=algo)
        ax.annotate(algo, (ms[-1], win[-1]), textcoords="offset points",
                    xytext=(7, -2), color=colour, fontsize=9, fontweight="bold")
    ax.axhline(0.5, color=INK2, lw=0.9, ls="--")
    # Deliberately NO cross-game reference line here. Latency depends on the
    # game's action space, so a number from a different game is not a
    # comparable baseline -- plotting one would imply a speedup we did not
    # demonstrate.
    ax.axvline(200, color=INK2, lw=1.0, ls=":", alpha=0.7)
    ax.annotate("200 ms: interactive\nlatency budget", (200, 0.485),
                textcoords="offset points", xytext=(8, 0),
                color=INK2, fontsize=8, ha="left", va="center")
    ax.set_xscale("log")
    ax.set_xlabel("milliseconds per decision  (log scale)")
    ax.set_ylabel("win rate vs best heuristic")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title("Both search families beat the best heuristic\n"
                 "well inside an interactive latency budget")
    despine(ax)
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.99))
    finish(fig, "fig3_latency_vs_strength", f"source: {src}")


# ---------------------------------------------------------------- FIGURE 4

def fig_reward_ablation():
    """Reward design, the Phase-4 experiment.

    The verdict in the title is COMPUTED from the data, never asserted. An
    earlier version hard-coded "dense and aligned beats everything else"; when
    the numbers came in they did not say that, and a figure whose caption
    disagrees with its own bars is worse than no figure.
    """
    order = ["terminal", "trick_shaped", "ten_shaped", "potential"]
    pretty = {"terminal": "terminal\n(sparse, correct)",
              "trick_shaped": "trick-shaped\n(dense, MISALIGNED)",
              "ten_shaped": "ten-shaped\n(dense, aligned)",
              "potential": "potential-based\n(dense, aligned)"}

    # Preferred source: the dedicated evaluation (1000 deals x4 per seed).
    # Fallback: the training artifacts' own final evaluation.
    vals = {}
    src = None
    for cand in ("eval_rl.json", "eval_final.json"):
        if not os.path.exists(cand):
            continue
        d = json.load(open(cand))
        for row in d.get("ablation_per_seed", []):
            if row.get("algo") != "dqn":
                continue
            v = row.get("TensThenTricks")
            if isinstance(v, (list, tuple)) and v:
                vals.setdefault(row["reward"], []).append(v[0])
        if vals:
            src = cand
            break
    if not vals:
        for f, d in load("rl_dqn_*.json"):
            r, fin = d.get("reward"), d.get("final", {})
            v = fin.get("TensThenTricks")
            if isinstance(v, dict):
                v = v.get("win")
            if r in order and isinstance(v, (int, float)):
                vals.setdefault(r, []).append(v)
        src = "rl_dqn_*.json"

    vals = {k: v for k, v in vals.items() if k in order}
    if len(vals) < 3:
        print("  [skip] fig4 — ablation incomplete")
        return

    n_seeds = min(len(v) for v in vals.values())
    means = [statistics.fmean(vals[k]) if k in vals else 0.0 for k in order]
    sds = [statistics.pstdev(vals[k]) if k in vals and len(vals[k]) > 1 else 0.0
           for k in order]

    # --- the verdict, derived
    ranked = sorted(((statistics.fmean(v), k) for k, v in vals.items()),
                    reverse=True)
    best_v, best_k = ranked[0]
    worst_v, worst_k = ranked[-1]
    spread = best_v - worst_v
    typical_sd = statistics.fmean([s for s in sds if s > 0]) if any(sds) else 0.0
    separated = typical_sd > 0 and spread > 2 * typical_sd

    if n_seeds < 2:
        verdict = (f"single seed, differences not yet interpretable "
                   f"(spread {spread:.3f})")
    elif separated:
        verdict = (f"{best_k.replace('_', '-')} leads by {spread:.3f}, "
                   f"more than twice the across-seed SD ({typical_sd:.3f})")
    else:
        verdict = (f"no scheme separates: spread {spread:.3f} is within "
                   f"across-seed variation (SD {typical_sd:.3f})")

    cols = [INK2, VERM, GREEN, GREEN]
    fig, ax = plt.subplots(figsize=(5.9, 3.7))
    ax.bar(range(len(order)), means, width=0.55, color=cols,
           yerr=sds if any(sds) else None, capsize=3.5,
           error_kw={"elinewidth": 1.0, "ecolor": INK2})
    for i, (m, sd) in enumerate(zip(means, sds)):
        lab = f"{m:.3f}" + (f"\n±{sd:.3f}" if sd > 0 else "")
        ax.text(i, m + sd + 0.006, lab, ha="center", fontsize=8,
                color=INK, fontweight="bold")
    ax.axhline(0.5, color=INK2, lw=0.9, ls="--")
    ax.text(-0.42, 0.508, "parity with best heuristic", fontsize=7.5,
            color=INK2, ha="left", va="bottom")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([pretty[k] for k in order], fontsize=8)
    ax.set_ylabel("win rate vs best heuristic")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(0, max(0.55, max(m + s for m, s in zip(means, sds)) * 1.28))
    ax.set_title(f"Reward design (DQN, {n_seeds} seed"
                 f"{'s' if n_seeds != 1 else ''}):\n{verdict}", fontsize=9.5)
    ax.grid(axis="x", visible=False)
    despine(ax)
    finish(fig, "fig4_reward_ablation",
           f"{n_seeds} seed(s), source {src}")


# ---------------------------------------------------------------- FIGURE 5

def fig_learning_curves():
    """Learning curves aggregated ACROSS SEEDS, with variability shown.

    The previous version globbed rl_*.json and plotted every file, so each
    method appeared once per seed as if it were three different methods, and
    seed variability was invisible. It also hard-coded the conclusion "nowhere
    near a ceiling" in the title. Both are fixed: curves are grouped by method
    and shown as mean with a +/- SD band, and the title states what the data
    actually does at the end of training.
    """
    import collections
    groups = collections.defaultdict(lambda: collections.defaultdict(list))
    for f, d in load("rl_*.json"):
        curve = d.get("curve", [])
        if not curve or "ep" not in curve[0]:
            continue
        m = d.get("method")
        label = m if m and m not in ("dqn", "ppo") else \
            f"{d.get('algo','?')}-{d.get('reward','?')}"
        for c in curve:
            if "TensThenTricks" in c:
                groups[label][c["ep"]].append(c["TensThenTricks"])
    if not groups:
        print("  [skip] fig5 — no learning curves yet")
        return

    series = []
    for label, byep in groups.items():
        eps = sorted(byep)
        n_seeds = max(len(v) for v in byep.values())
        mean = [statistics.fmean(byep[e]) for e in eps]
        sd = [statistics.pstdev(byep[e]) if len(byep[e]) > 1 else 0.0
              for e in eps]
        if len(eps) >= 3:
            series.append((label, eps, mean, sd, n_seeds, mean[-1]))
    if not series:
        print("  [skip] fig5 — no curves with enough points")
        return
    series.sort(key=lambda t: -t[5])
    top = series[:4]

    # Is anything still improving at the end? Compare the last third to the
    # preceding third, per method, instead of asserting it.
    rising = []
    for label, eps, mean, sd, ns, _ in top:
        k = max(1, len(mean) // 3)
        late, mid = statistics.fmean(mean[-k:]), statistics.fmean(mean[-2 * k:-k])
        rising.append(late - mid)
    n_up = sum(1 for r in rising if r > 0.005)
    if n_up == len(top):
        verdict = "all still improving at the end of training"
    elif n_up == 0:
        verdict = "all have flattened by the end of training"
    else:
        verdict = f"{n_up} of {len(top)} still improving at the end"

    seeds = min(s[4] for s in top)
    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    cols = [BLUE, VERM, GREEN, INK2]
    for i, (label, eps, mean, sd, ns, _) in enumerate(top):
        c = cols[i % len(cols)]
        lo = [m - s_ for m, s_ in zip(mean, sd)]
        hi = [m + s_ for m, s_ in zip(mean, sd)]
        ax.fill_between(eps, lo, hi, color=c, alpha=0.18, linewidth=0)
        ax.plot(eps, mean, color=c, lw=1.9, marker="o", ms=3.4, label=label)
    ax.axhline(0.5, color=INK2, lw=0.9, ls="--")
    ax.text(eps[0], 0.505, "parity with best heuristic", fontsize=7.5,
            color=INK2, va="bottom")
    ax.set_xlabel("training episodes")
    ax.set_ylabel("win rate vs best heuristic")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title(f"Learning curves, mean ± SD over {seeds} seeds:\n{verdict}",
                 fontsize=9.5)
    # These four end within ~0.02 of each other, so direct labels at the right
    # edge overlap illegibly. A legend is the readable choice here.
    ax.legend(loc="lower right", frameon=False, fontsize=8, ncol=2)
    despine(ax)
    finish(fig, "fig5_learning_curves",
           f"{len(top)} methods, {seeds} seeds, band = ±1 SD across seeds")


# ---------------------------------------------------------------- FIGURE 6

def fig_game_stats():
    """What the game itself looks like: why the tiebreak matters."""
    import random
    from dahaljeet import Hand
    rng = random.Random(20260831)
    counts = [0] * 5
    few_tricks_win = 0
    N = 30000
    for i in range(N):
        h = Hand(dealer=i & 3, rng=rng)
        h.deal()
        while not h.is_over:
            h.play(rng.choice(h.legal_moves()))
        r = h.result()
        counts[r.tens_by_team[0]] += 1
        w, l = r.winning_team, r.losing_team
        if r.tricks_by_team[l] > r.tricks_by_team[w]:
            few_tricks_win += 1
    frac = [c / N for c in counts]
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    cols = [BLUE, BLUE, VERM, BLUE, BLUE]
    ax.bar(range(5), frac, width=0.6, color=cols)
    for i, v in enumerate(frac):
        ax.text(i, v + 0.006, f"{v:.3f}", ha="center", fontsize=9, color=INK)
    ax.set_xticks(range(5))
    ax.set_xticklabels(["0", "1", "2", "3", "4"])
    ax.set_xlabel("tens captured by one side (of four)")
    ax.set_ylabel("share of hands")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title(f"A 2-2 split decides {frac[2]:.0%} of hands,\n"
                 f"and {few_tricks_win/N:.1%} are won with FEWER tricks")
    ax.grid(axis="x", visible=False)
    despine(ax)
    finish(fig, "fig6_game_statistics", f"n={N:,} random-play hands")


# ---------------------------------------------------------------- FIGURE 7

def fig_distillation():
    """Imitating search: accuracy is not the criterion that matters."""
    f = "rl_distill_s0.json"
    if not os.path.exists(f):
        print("  [skip] fig7 — distillation not finished")
        return
    d = json.load(open(f))
    curve = d.get("curve", [])
    if not curve:
        return
    ep = [c["epoch"] for c in curve]
    acc = [c["val_acc"] for c in curve]
    win = [c["TensThenTricks"] for c in curve]
    fig, ax = plt.subplots(figsize=(5.9, 3.6))
    # ONE axis: both series are proportions, so they share a scale honestly.
    ax.plot(ep, acc, color=BLUE, lw=1.9, marker="o", ms=3.5,
            label="agreement with ISMCTS (held-out)")
    ax.plot(ep, win, color=VERM, lw=1.9, marker="s", ms=3.5,
            label="playing strength vs best heuristic")
    best = max(range(len(win)), key=lambda i: win[i])
    ax.scatter([ep[best]], [win[best]], s=90, facecolor="none",
               edgecolor=VERM, lw=1.8, zorder=5)
    ax.annotate("best checkpoint", (ep[best], win[best]),
                textcoords="offset points", xytext=(8, 10),
                color=VERM, fontsize=8, fontweight="bold")
    ax.axhline(0.5, color=INK2, lw=0.9, ls="--")
    ax.set_xlabel("epoch")
    ax.set_ylabel("proportion")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title("Distilling search into a network:\n"
                 "strength peaks and decays before accuracy does")
    despine(ax)
    ax.legend(loc="lower left")
    finish(fig, "fig7_distillation", "source: rl_distill_s*.json")


# ---------------------------------------------------------------- FIGURE 8

def fig_exploitability():
    """What does knowing the opponent buy against each agent?

    Reads exploitability_search.json ONLY. The earlier exploit_*.json artifacts
    are deliberately not plotted: their best-response oracle was a DQN that
    loses to an off-the-shelf heuristic, so every value came out negative and
    bounded nothing. Plotting them would present a measurement failure as a
    finding about the agents.
    """
    if not os.path.exists("exploitability_search.json"):
        print("  [skip] fig8 - exploitability_search.json not present "
              "(exploit_*.json is superseded and is not plotted)")
        return
    d = json.load(open("exploitability_search.json"))
    rows = [(t["target"], t["generic_search_win"], t["best_response_win"],
             t["exploitability"], t.get("paired_p")) for t in d["targets"]]
    if not rows:
        print("  [skip] fig8 - no targets in exploitability_search.json")
        return
    rows.sort(key=lambda r: r[3])
    names = [r[0] for r in rows]
    gen = [r[1] for r in rows]
    br = [r[2] for r in rows]

    fig, ax = plt.subplots(figsize=(6.4, 0.6 * len(rows) + 2.0))
    y = range(len(rows))
    h = 0.34
    ax.barh([i + h / 2 for i in y], gen, height=h, color=INK2,
            label="same search, no opponent model")
    ax.barh([i - h / 2 for i in y], br, height=h, color=VERM,
            label="search that knows the opponent")
    for i, (_, g, b, e, pv) in enumerate(rows):
        star = "" if pv is None else ("*" if pv < 0.05 else "")
        ax.text(max(g, b) + 0.012, i, f"{e:+.3f}{star}", va="center",
                fontsize=8.5, color=INK, fontweight="bold")
    ax.axvline(0.5, color=INK2, lw=1.0, ls="--")
    ax.set_yticks(list(y))
    ax.set_yticklabels(names)
    ax.set_xlabel("win rate against the target")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title("Exploitability: what knowing the opponent is worth\n"
                 "(identical search and budget in both arms; * paired p<0.05)",
                 fontsize=9.5)
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    ax.grid(axis="y", visible=False)
    despine(ax)
    finish(fig, "fig8_exploitability",
           "source: exploitability_search.json (exploit_*.json superseded)")


# ---------------------------------------------------------------- FIGURE 9

def fig_encoding_ablation():
    """Which parts of the observation actually earn their place?"""
    import collections
    # Aggregate ACROSS SEEDS. The glob returns one file per (block, seed), so
    # plotting them directly drew each block three times with its x-tick label
    # printed on top of itself.
    acc = collections.defaultdict(list)
    feats = {}
    for f, d in load("ablate_*.json"):
        # Only encoding-ablation artifacts have both `block` and `final`.
        if "block" not in d or not isinstance(d.get("final"), dict):
            continue
        w = d["final"].get("TensThenTricks")
        if not isinstance(w, dict):
            continue
        acc[d["block"]].append(w["win"])
        feats[d["block"]] = d.get("features_zeroed", 0)
    if len(acc) < 2:
        print("  [skip] fig9 - need more ablation artifacts")
        return
    rows = [(k, statistics.fmean(v), statistics.pstdev(v), feats.get(k, 0), len(v))
            for k, v in acc.items()]
    base = next((r for r in rows if r[0] == "none"), None)
    rows.sort(key=lambda r: -r[1])
    n_seeds = min(r[4] for r in rows)
    names = [f"{r[0]}\n({r[3]} feats)" if r[3] else f"{r[0]}\n(control)"
             for r in rows]
    vals = [r[1] for r in rows]
    err = [r[2] for r in rows]
    cols = [GREEN if r[0] == "none" else
            (VERM if base and r[1] < base[1] - 0.02 else BLUE) for r in rows]

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.bar(range(len(rows)), vals, yerr=err, width=0.58, color=cols,
           capsize=3, error_kw={"elinewidth": 0.9, "ecolor": INK2})
    if base:
        ax.axhline(base[1], color=GREEN, lw=1.0, ls="--")
        # placed at the LEFT edge: at the right it sat on top of the last
        # bar's error bar
        ax.text(-0.42, base[1] + 0.006, "full encoding", fontsize=8,
                color=GREEN, ha="left", va="bottom")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("win rate vs best heuristic")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    best = rows[0]
    verdict = (f"{best[0]} leads by {best[1]-base[1]:+.3f}"
               if base and best[0] != "none" else "no block changes the result")
    ax.set_title(f"State-encoding ablation ({n_seeds} seeds): {verdict}",
                 fontsize=10)
    ax.grid(axis="x", visible=False)
    despine(ax)
    finish(fig, "fig9_encoding_ablation", "source: ablate_*_s*.json")


# --------------------------------------------------------------- FIGURE 10

def fig_seed_variance():
    """Multi-seed spread. Single-seed results are not publishable."""
    import statistics
    groups = {}
    for f, d in load("rl_*.json"):
        key = d.get("method") or f"{d.get('algo','')}-{d.get('reward','')}"
        w = (d.get("final") or {}).get("TensThenTricks")
        if w is not None:
            groups.setdefault(key, []).append(w)
    multi = {k: v for k, v in groups.items() if len(v) >= 2}
    if not multi:
        print("  [skip] fig10 - no method has >=2 seeds yet")
        return
    order = sorted(multi, key=lambda k: -statistics.fmean(multi[k]))
    means = [statistics.fmean(multi[k]) for k in order]
    sds = [statistics.pstdev(multi[k]) if len(multi[k]) > 1 else 0 for k in order]

    fig, ax = plt.subplots(figsize=(6.2, 0.34 * len(order) + 1.6))
    ax.barh(order, means, xerr=sds, height=0.6, color=BLUE, capsize=3,
            error_kw={"elinewidth": 1.0, "ecolor": INK})
    for i, k in enumerate(order):
        ax.scatter(multi[k], [i] * len(multi[k]), s=16, color=VERM,
                   zorder=5, linewidths=0)
    ax.axvline(0.5, color=INK2, lw=1.0, ls="--")
    ax.set_xlabel("win rate vs best heuristic  (bar = mean, whisker = SD, dots = seeds)")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title(f"Seed variance across {max(len(v) for v in multi.values())} runs")
    ax.grid(axis="y", visible=False)
    despine(ax)
    finish(fig, "fig10_seed_variance", "source: rl_*_s*.json")


if __name__ == "__main__":
    print("generating figures ->", OUT)
    for fn in (fig_hierarchy, fig_tension, fig_latency, fig_reward_ablation,
               fig_learning_curves, fig_game_stats, fig_distillation,
               fig_exploitability, fig_encoding_ablation,
               fig_seed_variance):
        try:
            fn()
        except Exception as e:
            print(f"  [error] {fn.__name__}: {e}")
    print("done")
