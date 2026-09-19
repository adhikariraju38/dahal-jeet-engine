"""Evaluation of trained RL agents, plus the search latency curve.

    python eval_final.py --part curve   # latency curve only -- needs NO models
    python eval_final.py --part rl      # reward ablation + head-to-head
    python eval_final.py                # both

The split matters for pipeline ordering: the latency curve depends on nothing
and can run before any training, while the RL parts require checkpoints.
Running the whole script early fails on a missing .pt file.
"""
import argparse, glob, json, random, re, sys, time, itertools
sys.path.insert(0, '.')
import runenv
import torch

_ap = argparse.ArgumentParser()
_ap.add_argument("--part", choices=["curve", "rl", "both"], default="both")
_ap.add_argument("--deals", type=int, default=1000,
                 help="deals per ablation evaluation (x4 rotations)")
_ap.add_argument("--h2h-deals", type=int, default=800,
                 help="deals per head-to-head comparison")
_ap.add_argument("--curve-deals", type=int, default=120,
                 help="deals per point on the latency curve")
ARGS = _ap.parse_args()

rows, h2h, curve = [], [], []
from train_rl import Net, TorchAgent
from dahaljeet.agents import REGISTRY
from dahaljeet.search import PIMCAgent, ISMCTSAgent
from dahaljeet.tournament import duplicate_match

ALGOS = ["dqn", "ppo"]
REWARDS = ["terminal", "ten_shaped", "potential", "trick_shaped"]

def seeds_for(algo, reward):
    """Which seeds actually have a checkpoint on disk.

    Discovered rather than assumed: reporting a seed that was never trained is
    worse than reporting that it is missing.
    """
    out = []
    for p in glob.glob(f"rl_{algo}_{reward}_s*.pt"):
        m = re.search(r"_s(\d+)\.pt$", p)
        if m:
            out.append(int(m.group(1)))
    return sorted(out)


def load(algo, reward, seed):
    net = Net(critic=(algo == "ppo"))
    net.load_state_dict(torch.load(f"rl_{algo}_{reward}_s{seed}.pt",
                                   map_location="cpu"))
    net.eval()
    return TorchAgent(net, f"{algo}-{reward}-s{seed}")


def mean_sd(vals):
    n = len(vals)
    mu = sum(vals) / n
    sd = (sum((v - mu) ** 2 for v in vals) / n) ** 0.5 if n > 1 else 0.0
    return round(mu, 4), round(sd, 4)


agg, missing = [], []

if ARGS.part in ('rl', 'both'):
    print("="*100)
    print(f"PART 1 - REWARD ABLATION, every trained seed, {ARGS.deals} deals "
          f"x4 = {ARGS.deals*4} hands each, bootstrap CI over deals")
    print("="*100)
    for algo in ALGOS:
        for rew in REWARDS:
            seeds = seeds_for(algo, rew)
            if not seeds:
                missing.append(f"{algo}-{rew}")
                print(f"  [MISSING] {algo}-{rew}: no checkpoint on disk", flush=True)
                continue
            per_seed = {o: [] for o in ("Random", "TensThenTricks")}
            for sd_ in seeds:
                a = load(algo, rew, sd_)
                line = {"algo": algo, "reward": rew, "seed": sd_}
                for opp_name in ("Random", "TensThenTricks"):
                    opp = REGISTRY[opp_name](rng=random.Random(4))
                    r = duplicate_match(a, opp, n_deals=ARGS.deals, seed=555, timed=False)
                    lo, hi = r.bootstrap_ci()
                    line[opp_name] = (round(r.win_rate_a, 4), round(lo, 4), round(hi, 4))
                    per_seed[opp_name].append(r.win_rate_a)
                rows.append(line)
                print(f"  {algo:4s} {rew:13s} s{sd_}  vs Random {line['Random'][0]:.4f} "
                      f"[{line['Random'][1]:.4f},{line['Random'][2]:.4f}]   "
                      f"vs TensThenTricks {line['TensThenTricks'][0]:.4f} "
                      f"[{line['TensThenTricks'][1]:.4f},{line['TensThenTricks'][2]:.4f}]",
                      flush=True)
            row = {"algo": algo, "reward": rew, "n_seeds": len(seeds), "seeds": seeds}
            for opp_name in ("Random", "TensThenTricks"):
                mu, sd2 = mean_sd(per_seed[opp_name])
                row[opp_name] = {"mean": mu, "sd": sd2,
                                 "min": round(min(per_seed[opp_name]), 4),
                                 "max": round(max(per_seed[opp_name]), 4)}
            agg.append(row)
            t = row["TensThenTricks"]
            print(f"    -> {algo}-{rew}: {len(seeds)} seed(s), "
                  f"vs TensThenTricks {t['mean']:.4f} +/- {t['sd']:.4f}\n", flush=True)

    print("\n" + "="*100)
    print("PART 2 - HEAD-TO-HEAD between reward schemes (same algo, SAME SEED), 800 deals")
    print("="*100)
    print("Pairing on seed matters: comparing arm A's seed 0 with arm B's seed 1")
    print("would confound the reward scheme with the seed.\n")
    for algo in ALGOS:
        for r1, r2 in itertools.combinations(REWARDS, 2):
            common = sorted(set(seeds_for(algo, r1)) & set(seeds_for(algo, r2)))
            if not common:
                continue
            verdicts = []
            for sd_ in common:
                a, b = load(algo, r1, sd_), load(algo, r2, sd_)
                r = duplicate_match(a, b, n_deals=ARGS.h2h_deals, seed=606, timed=False)
                lo, hi = r.bootstrap_ci()
                v = "WINS" if lo > 0.5 else ("loses" if hi < 0.5 else "tie")
                verdicts.append(v)
                h2h.append((algo, r1, r2, sd_, round(r.win_rate_a, 4),
                            round(lo, 4), round(hi, 4), v))
                print(f"  {algo:4s} {r1:13s} vs {r2:13s} s{sd_}  {r.win_rate_a:.4f} "
                      f"[{lo:.4f},{hi:.4f}]  {v}", flush=True)
            if len(set(verdicts)) > 1:
                print(f"    ^ seeds DISAGREE ({verdicts}) - not a reportable "
                      f"difference\n", flush=True)

if ARGS.part in ('curve', 'both'):
    print("\n" + "="*100)
    print("PART 3 - SEARCH LATENCY vs STRENGTH  (vs TensThenTricks, 120 deals each)")
    print("="*100)
    opp = REGISTRY["TensThenTricks"](rng=random.Random(4))
    for w in [2, 4, 8, 16, 32]:
        a = PIMCAgent(worlds=w, rng=random.Random(3))
        r = duplicate_match(a, opp, n_deals=ARGS.curve_deals, seed=707, timed=True)
        lo, hi = r.bootstrap_ci()
        curve.append(("PIMC", w, round(r.win_rate_a,4), round(lo,4), round(hi,4),
                      round(r.ms_per_decision_a,2)))
        print(f"  PIMC   worlds={w:3d}  win {r.win_rate_a:.4f} [{lo:.4f},{hi:.4f}]  "
              f"{r.ms_per_decision_a:8.2f} ms/decision", flush=True)
    for it in [50, 100, 200, 400, 800]:
        a = ISMCTSAgent(iterations=it, rng=random.Random(3))
        r = duplicate_match(a, opp, n_deals=ARGS.curve_deals, seed=707, timed=True)
        lo, hi = r.bootstrap_ci()
        curve.append(("ISMCTS", it, round(r.win_rate_a,4), round(lo,4), round(hi,4),
                      round(r.ms_per_decision_a,2)))
        print(f"  ISMCTS iters={it:4d}  win {r.win_rate_a:.4f} [{lo:.4f},{hi:.4f}]  "
              f"{r.ms_per_decision_a:8.2f} ms/decision", flush=True)

out = f"eval_final.json" if ARGS.part == "both" else f"eval_{ARGS.part}.json"
json.dump({"env": runenv.snapshot(), "part": ARGS.part,
           "ablation_per_seed": rows, "ablation_aggregated": agg,
           "methods_without_checkpoints": missing,
           "head_to_head_per_seed": h2h, "latency_curve": curve,
           "h2h_schema": "algo, reward_a, reward_b, seed, win_a, ci_lo, ci_hi, verdict"},
          open(out, "w"), indent=2)
print(f"\nsaved {out}")
