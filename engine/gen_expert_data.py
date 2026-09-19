"""Generate (state, expert-action) pairs from ISMCTS for policy distillation.

ISMCTS is our strongest agent but costs ~42 ms/decision. If a network can be
trained to imitate it, we get search-quality play at ~0.1 ms -- which is the
difference between a research result and something deployable on a phone.

Parallelised across cores; ISMCTS is the bottleneck.
"""
import argparse, json, os, pickle, random, sys, time
from multiprocessing import Pool
sys.path.insert(0, '.')
from dahaljeet.encode import encode, legal_mask
from dahaljeet.hand import Hand
from dahaljeet.search import ISMCTSAgent
from dahaljeet.agents import REGISTRY
from dahaljeet.view import make_view


def worker(args):
    seed, hands, iters = args
    rng = random.Random(seed)
    expert = ISMCTSAgent(iterations=iters, rng=random.Random(seed + 7))
    # Opponents vary so the data covers a range of situations, not just
    # ISMCTS-vs-ISMCTS lines.
    pool = ["TensThenTricks", "GreedyTricks", "Random", "PartnerAware"]
    out = []
    for h_i in range(hands):
        opp = REGISTRY[rng.choice(pool)](rng=random.Random(rng.random()))
        expert_team = h_i % 2
        h = Hand(dealer=h_i % 4, rng=rng)
        h.deal()
        while not h.is_over:
            seat = h.to_act
            v = make_view(h, seat)
            if seat % 2 == expert_team:
                a = expert.act(v)
                out.append((encode(v), legal_mask(v), a))
            else:
                a = opp.act(v)
            h.play(a)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", type=int, default=4000)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--out", default="expert_data.pkl")
    a = ap.parse_args()
    per = max(1, a.hands // a.procs)
    jobs = [(1000 + i, per, a.iters) for i in range(a.procs)]
    t0 = time.perf_counter()
    print(f"generating ~{per*a.procs} hands with ISMCTS{a.iters} on {a.procs} procs",
          flush=True)
    with Pool(a.procs) as p:
        chunks = p.map(worker, jobs)
    data = [x for c in chunks for x in c]
    with open(a.out, "wb") as f:
        pickle.dump(data, f)
    el = time.perf_counter() - t0
    print(f"{len(data):,} expert decisions in {el:.0f}s -> {a.out}", flush=True)
