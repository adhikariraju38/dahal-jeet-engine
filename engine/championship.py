"""Cross-category championship: rule-based vs search-based.

Duplicate deals throughout, seat-balanced, bootstrap CIs over deals.
"""
import json, random, sys, time
sys.path.insert(0, '.')
from dahaljeet.agents import REGISTRY
from dahaljeet.search import PIMCAgent, ISMCTSAgent
from dahaljeet.tournament import duplicate_match, round_robin, leaderboard

RULE_N = 1200      # deals per rule-based matchup (fast agents)
SEARCH_N = 130     # deals per matchup involving search (expensive)

def build():
    rb = [REGISTRY[n](rng=random.Random(300 + i)) for i, n in enumerate(
        ["Random", "Lowest", "GreedyTricks", "TenHunter", "TenDefender",
         "PartnerAware", "TenAware", "Balanced", "Adaptive",
         "AdaptiveGreedy", "TensThenTricks"])]
    try:
        g = json.load(open("tuned_genome.json"))
        from dahaljeet.param_agent import ParamAgent, Genome
        rb.append(ParamAgent(Genome(**g["genome"]), random.Random(9),
                             name="Tuned"))
        print(f"loaded tuned genome (fitness {g['fitness']:.4f})", flush=True)
    except Exception as e:
        print("no tuned genome yet:", e, flush=True)
    search = [PIMCAgent(worlds=12, rng=random.Random(11)),
              ISMCTSAgent(iterations=250, rng=random.Random(12))]
    return rb, search

if __name__ == "__main__":
    rb, search = build()
    t0 = time.perf_counter()

    print(f"\n{'='*96}\nSTAGE 1 - rule-based round robin "
          f"({RULE_N} deals x4 = {RULE_N*4} hands per matchup)\n{'='*96}", flush=True)
    res = round_robin(rb, n_deals=RULE_N, seed=77, timed=True)
    board = leaderboard(res, rb)
    for i, (n, wr) in enumerate(board, 1):
        print(f"  {i:2d}. {n:16s} {wr:.4f}", flush=True)
    print(f"  [{time.perf_counter()-t0:.0f}s]", flush=True)

    top = [a for a in rb if a.name in {board[0][0], board[1][0], board[2][0]}]
    print(f"\n{'='*96}\nSTAGE 2 - search vs top heuristics "
          f"({SEARCH_N} deals x4 = {SEARCH_N*4} hands per matchup)\n{'='*96}", flush=True)
    rows = []
    for s in search:
        for h in top + [REGISTRY["Random"](rng=random.Random(5))]:
            r = duplicate_match(s, h, n_deals=SEARCH_N, seed=91, timed=True)
            lo, hi = r.bootstrap_ci()
            v = "WINS" if lo > 0.5 else ("loses" if hi < 0.5 else "tie")
            rows.append((s.name, h.name, r.win_rate_a, lo, hi, r.cohens_d(),
                         r.ms_per_decision_a, v))
            print(f"  {s.name:12s} vs {h.name:16s} win={r.win_rate_a:.4f} "
                  f"[{lo:.4f},{hi:.4f}] d={r.cohens_d():+.2f} "
                  f"{r.ms_per_decision_a:7.2f}ms  {v}", flush=True)

    json.dump({
        "rule_based_leaderboard": board,
        "search_rows": rows,
        "rule_deals": RULE_N, "search_deals": SEARCH_N,
        "elapsed_s": time.perf_counter() - t0,
    }, open("championship_results.json", "w"), indent=2)
    print(f"\nTotal {time.perf_counter()-t0:.0f}s -> championship_results.json")
