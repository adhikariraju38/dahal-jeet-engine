"""Hill-climbing tuner for ParamAgent (Phase 2.5 / metaheuristic).

Fitness = mean duplicate-deal win rate against a fixed opponent panel.
Duplicate deals + a fixed panel + a fixed seed keep the comparison paired, so
small genome differences are measurable without huge sample sizes.
"""
import random, sys, json, time
sys.path.insert(0, '.')
from dahaljeet.agents import REGISTRY
from dahaljeet.param_agent import ParamAgent, Genome, random_genome, mutate
from dahaljeet.tournament import duplicate_match

PANEL = ["GreedyTricks", "TenAware", "PartnerAware", "AdaptiveGreedy", "Random"]
DEALS = 260


def fitness(genome, seed=5):
    a = ParamAgent(genome, random.Random(0), name="Param")
    tot = 0.0
    for i, opp_name in enumerate(PANEL):
        opp = REGISTRY[opp_name](rng=random.Random(50 + i))
        r = duplicate_match(a, opp, n_deals=DEALS, seed=seed + i, timed=False)
        tot += r.win_rate_a
    return tot / len(PANEL)


def hill_climb(iters=140, restarts=4, seed=1):
    rng = random.Random(seed)
    best_g, best_f = None, -1.0
    for rs in range(restarts):
        g = Genome() if rs == 0 else random_genome(rng)
        f = fitness(g)
        stall = 0
        for it in range(iters // restarts):
            cand = mutate(g, rng)
            cf = fitness(cand)
            if cf > f:
                g, f, stall = cand, cf, 0
            else:
                stall += 1
                if stall > 12:
                    break
        print(f"  restart {rs}: fitness {f:.4f}  {g}", flush=True)
        if f > best_f:
            best_g, best_f = g, f
    return best_g, best_f


if __name__ == "__main__":
    t0 = time.perf_counter()
    print(f"panel={PANEL}  deals/match={DEALS}", flush=True)
    g, f = hill_climb()
    print(f"\nBEST fitness {f:.4f} after {time.perf_counter()-t0:.0f}s")
    print("genome:", g)
    json.dump({"fitness": f, "genome": g.__dict__}, open("tuned_genome.json", "w"), indent=2)
    print("saved -> tuned_genome.json")
