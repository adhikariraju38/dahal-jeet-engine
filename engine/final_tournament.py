"""Final all-play-all across every agent family (gap #6).

Learning agents were previously evaluated only against Random and the best
heuristic, while search and heuristics got a full round-robin. That asymmetry
is indefensible in a paper that claims to compare methods, so this runs every
agent that exists against every other under one protocol.

Protocol (identical for all pairs):
  * duplicate deals -- the same deal played four times with rotated seating,
    so both agents hold the identical cards from every seat
  * seat-balanced -- seat 2 draws the trump and leads, so it is structurally
    advantaged; rotation removes it as a confound
  * bootstrap CIs resampled over DEALS, the independent unit

    python final_tournament.py --deals 400
"""
from __future__ import annotations

import argparse
import glob
import multiprocessing as mp
import itertools
import json
import os
import random
import re
import sys
import time

sys.path.insert(0, ".")
import runenv
import torch

from dahaljeet.agents import REGISTRY
from dahaljeet.search import ISMCTSAgent, PIMCAgent
from dahaljeet.tournament import duplicate_match
from train_rl import Net, TorchAgent


def try_load(path):
    """Load a checkpoint under whichever architecture actually fits.

    Different methods were trained with different heads; rather than hard-code
    a mapping that silently rots, try each and keep the one that loads cleanly.
    """
    sd = torch.load(path, map_location="cpu")
    name = os.path.basename(path).replace(".pt", "")

    candidates = []
    candidates.append(("Net", lambda: Net()))
    candidates.append(("Net+critic", lambda: Net(critic=True)))
    try:
        from train_more import DuelNet
        candidates.append(("DuelNet", lambda: DuelNet()))
    except Exception:
        pass
    try:
        from nfsp import mlp as nfsp_mlp
        candidates.append(("nfsp-mlp", lambda: nfsp_mlp(52)))
    except Exception:
        pass
    try:
        from deepcfr import mlp as cfr_mlp
        candidates.append(("cfr-mlp", lambda: cfr_mlp()))
    except Exception:
        pass
    try:
        from train_mappo import Actor
        candidates.append(("Actor", lambda: Actor()))
    except Exception:
        pass

    for label, mk in candidates:
        try:
            net = mk()
            net.load_state_dict(sd)
            net.eval()
            return net, label
        except Exception:
            continue
    return None, None


def agent_from_spec(spec, seed):
    """Build one agent from a picklable spec, with a deterministic seed.

    Specs rather than live objects because every pairing must construct its own
    agents. The previous version built the roster once and reused the same
    objects for all 325 pairings, so each agent's RNG advanced as it played and
    a pairing's result depended on how many pairings had run before it. That
    makes no single pairing independently reproducible, and it is why the
    tournament could not be parallelised. Seeding per pairing fixes both.
    """
    rng = random.Random(seed)
    kind, _, arg = spec.partition(":")
    if kind == "REG":
        return REGISTRY[arg](rng=rng)
    if kind == "TUNED":
        from dahaljeet.param_agent import ParamAgent, Genome
        g = json.load(open("tuned_genome.json"))["genome"]
        return ParamAgent(Genome(**g), rng, name="Tuned")
    if kind == "PT":
        net, _ = try_load(arg)
        label = re.sub(r"_s\d+\.pt$", "", os.path.basename(arg)[3:]).replace(".pt", "")
        return TorchAgent(net, label, greedy=True, rng=rng)
    if kind == "ALPHADJ":
        from alphazero import PolicyValueNet, NeuralISMCTS
        pv = PolicyValueNet()
        pv.load_state_dict(torch.load(arg, map_location="cpu"))
        pv.eval()
        return NeuralISMCTS(pv, sims=60, rng=rng, name="AlphaDJ60")
    if kind == "PIMC":
        return PIMCAgent(worlds=int(arg), rng=rng)
    if kind == "ISMCTS":
        return ISMCTSAgent(iterations=int(arg), rng=rng)
    raise ValueError(f"unknown spec: {spec}")


def _run_pair(job):
    """One pairing, in its own process. Deterministic given (specs, seeds)."""
    (ia, spec_a, name_a), (ib, spec_b, name_b), deals, pair_seed = job
    a = agent_from_spec(spec_a, 10_000 + pair_seed)
    b = agent_from_spec(spec_b, 20_000 + pair_seed)
    r = duplicate_match(a, b, n_deals=deals, seed=2026, timed=True)
    lo, hi = r.bootstrap_ci()
    return {"a": name_a, "b": name_b, "ia": ia, "ib": ib,
            "win_a": round(r.win_rate_a, 4),
            "ci": [round(lo, 4), round(hi, 4)],
            "cohens_d": round(r.cohens_d(), 3),
            "ms_a": round(r.ms_per_decision_a, 3),
            "ms_b": round(r.ms_per_decision_b, 3),
            "a_wins_per_deal": list(r.a_wins_per_deal),
            "tens_a": r.tens_a, "tens_b": r.tens_b,
            "tricks_a": r.tricks_a, "tricks_b": r.tricks_b,
            "tiebreaks": r.tiebreaks,
            "coats_a": r.coats_a, "coats_b": r.coats_b,
            "double_coats_a": r.double_coats_a,
            "double_coats_b": r.double_coats_b}


def build_specs(include_search=True, sims=250, worlds=16,
                roster_seed=0, all_seeds=False):
    """The roster, as (spec, display name) pairs."""
    out = [(f"REG:{n}", n) for n in
           ["Random", "GreedyTricks", "PartnerAware", "TenAware",
            "Adaptive", "AdaptiveGreedy", "TensThenTricks"]]
    if os.path.exists("tuned_genome.json"):
        out.append(("TUNED:", "Tuned"))
    paths = sorted(glob.glob("rl_*.pt"))
    if not all_seeds:
        keep = [p for p in paths if p.endswith(f"_s{roster_seed}.pt")]
        if len(keep) < len(paths):
            print(f"  [roster] one seed per method: keeping seed {roster_seed}; "
                  f"{len(paths) - len(keep)} other-seed checkpoint(s) excluded")
        paths = keep
    for p in paths:
        net, _ = try_load(p)
        if net is None:
            print(f"  [skip] {p} — no matching architecture")
            continue
        label = re.sub(r"_s\d+\.pt$", "", os.path.basename(p)[3:]).replace(".pt", "")
        out.append((f"PT:{p}", label))
    adj = f"alphadj_s{roster_seed}.pt"
    if os.path.exists(adj):
        out.append((f"ALPHADJ:{adj}", "AlphaDJ60"))
    if include_search:
        out.append((f"PIMC:{worlds}", f"PIMC{worlds}"))
        out.append((f"ISMCTS:{sims}", f"ISMCTS{sims}"))
    return out


def build_roster(include_search=True, sims=250, worlds=16,
                 roster_seed=0, all_seeds=False):
    roster = []

    # --- heuristics
    for n in ["Random", "GreedyTricks", "PartnerAware", "TenAware",
              "Adaptive", "AdaptiveGreedy", "TensThenTricks"]:
        roster.append(REGISTRY[n](rng=random.Random(hash(n) % 9999)))

    # --- tuned genome
    if os.path.exists("tuned_genome.json"):
        try:
            from dahaljeet.param_agent import ParamAgent, Genome
            g = json.load(open("tuned_genome.json"))["genome"]
            roster.append(ParamAgent(Genome(**g), random.Random(9),
                                     name="Tuned"))
        except Exception as e:
            print(f"  [warn] tuned genome: {e}")

    # --- learning agents
    #
    # ONE SEED PER METHOD by default. Every learning method is trained on 3
    # seeds, so globbing rl_*.pt puts 3 near-identical copies of each method in
    # the round-robin: the pairing count is quadratic, so 45 checkpoints is
    # 1540 pairings rather than ~230, and the leaderboard becomes 3 adjacent
    # rows per method with no added information. Seed variation belongs in the
    # multi-seed table (aggregate_seeds.py), not in the head-to-head matrix.
    #
    # The representative seed is FIXED in advance (seed 0), never chosen by
    # score -- picking the best-performing seed per method would be
    # cherry-picking dressed up as a tournament.
    paths = sorted(glob.glob("rl_*.pt"))
    if not all_seeds:
        keep = [p for p in paths if p.endswith(f"_s{roster_seed}.pt")]
        if len(keep) < len(paths):
            print(f"  [roster] one seed per method: keeping seed {roster_seed}; "
                  f"{len(paths) - len(keep)} checkpoint(s) from other seeds are "
                  f"excluded from the round-robin (their variation is reported "
                  f"by aggregate_seeds.py)")
        paths = keep
    for path in paths:
        net, arch = try_load(path)
        if net is None:
            print(f"  [skip] {path} — no matching architecture")
            continue
        label = re.sub(r"_s\d+\.pt$", "", os.path.basename(path)[3:])
        label = label.replace(".pt", "")
        roster.append(TorchAgent(net, label, greedy=True,
                                 rng=random.Random(7)))

    # --- neural-guided search
    adj = f"alphadj_s{roster_seed}.pt"
    if os.path.exists(adj):
        try:
            from alphazero import PolicyValueNet, NeuralISMCTS
            pv = PolicyValueNet()
            pv.load_state_dict(torch.load(adj, map_location="cpu"))
            pv.eval()
            roster.append(NeuralISMCTS(pv, sims=60, rng=random.Random(5),
                                       name="AlphaDJ60"))
        except Exception as e:
            print(f"  [warn] alphadj: {e}")

    # --- classical search
    if include_search:
        roster.append(PIMCAgent(worlds=worlds, rng=random.Random(11)))
        roster.append(ISMCTSAgent(iterations=sims, rng=random.Random(12)))
    return roster


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--deals", type=int, default=400)
    ap.add_argument("--sims", type=int, default=250)
    ap.add_argument("--worlds", type=int, default=16)
    ap.add_argument("--no-search", action="store_true")
    ap.add_argument("--roster-seed", type=int, default=0,
                    help="training seed that represents each learning method")
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 4) - 2),
                    help="parallel pairings; each is independent and seeded")
    ap.add_argument("--all-seeds", action="store_true",
                    help="enter every seed as its own competitor (quadratic cost)")
    a = ap.parse_args()

    specs = build_specs(not a.no_search, a.sims, a.worlds,
                        a.roster_seed, a.all_seeds)
    names = [n for _, n in specs]
    print(f"roster ({len(specs)}): {', '.join(names)}\n", flush=True)
    npairs = len(specs) * (len(specs) - 1) // 2
    hands = npairs * a.deals * 4
    print(f"{npairs} pairings x {a.deals} deals x 4 rotations = "
          f"{hands:,} hands   ({a.procs} processes)", flush=True)
    print(flush=True)

    # Jobs carry a per-pairing seed so each pairing is reproducible on its own
    # and the result does not depend on execution order.
    jobs = []
    for k, ((ia, (sa, na)), (ib, (sb, nb))) in enumerate(
            itertools.combinations(list(enumerate(specs)), 2)):
        jobs.append(((ia, sa, na), (ib, sb, nb), a.deals, k))

    # Longest pairings first: without this the tail of the run is one lone
    # ISMCTS pairing while every other core sits idle.
    COST = {"ISMCTS": 47.0, "PIMC": 14.5, "ALPHADJ": 20.0}
    def cost(j):
        return sum(COST.get(x[1].split(":")[0], 0.85) for x in (j[0], j[1]))
    jobs.sort(key=cost, reverse=True)

    t0 = time.perf_counter()
    matrix = {n: {} for n in names}
    rows = []
    if a.procs > 1:
        with mp.get_context("spawn").Pool(a.procs) as pool:
            for i, r in enumerate(pool.imap_unordered(_run_pair, jobs), 1):
                rows.append(r)
                matrix[r["a"]][r["b"]] = r["win_a"]
                matrix[r["b"]][r["a"]] = round(1 - r["win_a"], 4)
                print(f"  [{i:3d}/{npairs}] {r['a']:18s} vs {r['b']:18s} "
                      f"{r['win_a']:.4f} {r['ci']}", flush=True)
    else:
        for i, j in enumerate(jobs, 1):
            r = _run_pair(j)
            rows.append(r)
            matrix[r["a"]][r["b"]] = r["win_a"]
            matrix[r["b"]][r["a"]] = round(1 - r["win_a"], 4)
            print(f"  [{i:3d}/{npairs}] {r['a']:18s} vs {r['b']:18s} "
                  f"{r['win_a']:.4f} {r['ci']}", flush=True)
    rows.sort(key=lambda r: (r["ia"], r["ib"]))   # stable, order-independent

    board = []
    for n in names:
        v = list(matrix[n].values())
        board.append((n, sum(v) / len(v) if v else 0.0))
    board.sort(key=lambda t: -t[1])

    print(f"\n{'='*64}\nLEADERBOARD (mean win rate, all pairings)\n{'='*64}")
    for i, (n, w) in enumerate(board, 1):
        print(f"  {i:2d}. {n:20s} {w:.4f}")

    el = time.perf_counter() - t0
    json.dump({"env": runenv.snapshot(), "deals_per_pair": a.deals, "roster": names,
               "deal_seed": 2026, "procs": a.procs,
               "determinism": ("Every pairing builds its own agents from a spec "
                               "with a seed derived from the pairing index, so a "
                               "pairing is reproducible in isolation and results "
                               "do not depend on execution order or process count."),
               "per_deal_note": ("Every pairing is played on the SAME deal "
                                 "sequence (deal_seed), so per-deal vectors are "
                                 "aligned across pairings and comparisons "
                                 "between agents are paired on the deal."),
               "leaderboard": board, "pairs": rows, "matrix": matrix,
               "seconds": el},
              open("final_tournament.json", "w"), indent=2)
    print(f"\n{el:.0f}s -> final_tournament.json")
