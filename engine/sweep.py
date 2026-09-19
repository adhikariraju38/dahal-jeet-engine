"""Hyperparameter sensitivity (gap #7).

Every learning result so far used one fixed configuration. A reviewer can
dismiss weak RL numbers as under-tuned, and would be entitled to. This sweeps
the parameters most likely to matter and reports the spread, so the paper can
say how much of the RL/search gap is tuning and how much is real.

    python sweep.py --episodes 15000
"""
from __future__ import annotations
import argparse, itertools, json, random, sys, time
sys.path.insert(0, '.')
import runenv
import torch, torch.nn as nn, torch.nn.functional as F
from collections import deque

from dahaljeet.agents import REGISTRY
from dahaljeet.encode import OBS_SIZE, ACTION_SIZE
from dahaljeet.env import DahalJeetEnv
from dahaljeet.tournament import duplicate_match
from train_rl import TorchAgent

NEG = -1e9

GRID = {
    "lr":     [1e-4, 3e-4, 1e-3],
    "hidden": [256, 512, 1024],
    "gamma":  [0.95, 0.99, 1.0],
}


def net_of(hidden):
    return nn.Sequential(
        nn.Linear(OBS_SIZE, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden // 2), nn.ReLU(),
        nn.Linear(hidden // 2, ACTION_SIZE))


def run(lr, hidden, gamma, episodes, seed):
    rng = random.Random(seed); torch.manual_seed(seed)
    q, tgt = net_of(hidden), net_of(hidden)
    tgt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=lr)
    buf = deque(maxlen=100_000)
    opp = REGISTRY["TensThenTricks"](rng=random.Random(seed + 1))
    eps, batch, step = 1.0, 128, 0
    for ep in range(episodes):
        env = DahalJeetEnv(opp, reward="potential", learner_team=ep % 2,
                           dealer=ep % 4, rng=rng)
        s = env.reset()
        while s is not None:
            obs, mask = s
            legal = [i for i, m in enumerate(mask) if m]
            if rng.random() < eps:
                a = rng.choice(legal)
            else:
                with torch.no_grad():
                    ql = q(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
                    a = int(ql.masked_fill(torch.tensor(mask).unsqueeze(0) == 0,
                                           NEG).argmax(-1))
            s2, r, done, _ = env.step(a)
            buf.append((obs, a, r, s2[0] if s2 else None,
                        s2[1] if s2 else None, done))
            s = s2; step += 1
            if len(buf) >= 2000 and step % 4 == 0:
                bt = rng.sample(buf, batch)
                bo = torch.tensor([b[0] for b in bt], dtype=torch.float32)
                ba = torch.tensor([b[1] for b in bt], dtype=torch.long)
                tv = torch.tensor([b[2] for b in bt], dtype=torch.float32)
                nz = [i for i, b in enumerate(bt) if not b[5] and b[3] is not None]
                if nz:
                    no = torch.tensor([bt[i][3] for i in nz], dtype=torch.float32)
                    nm = torch.tensor([bt[i][4] for i in nz], dtype=torch.float32)
                    with torch.no_grad():
                        sel = q(no).masked_fill(nm == 0, NEG).argmax(-1, keepdim=True)
                        tv[nz] += gamma * tgt(no).gather(1, sel).squeeze(1)
                pred = q(bo).gather(1, ba.unsqueeze(1)).squeeze(1)
                loss = F.smooth_l1_loss(pred, tv)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), 10.0); opt.step()
            if step % 2000 == 0:
                tgt.load_state_dict(q.state_dict())
        eps = max(0.05, eps * 0.9997)
    ag = TorchAgent(q, "sweep")
    m = duplicate_match(ag, REGISTRY["TensThenTricks"](rng=random.Random(4)),
                        n_deals=400, seed=313, timed=False)
    return round(m.win_rate_a, 4)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=15000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--axis", default="all", choices=["all", "lr", "hidden", "gamma"])
    a = ap.parse_args()
    base = {"lr": 3e-4, "hidden": 512, "gamma": 0.99}
    combos = []
    if a.axis == "all":
        # one-at-a-time around the baseline: 7 runs, not 27
        for k, vals in GRID.items():
            for v in vals:
                c = dict(base); c[k] = v
                if c not in combos:
                    combos.append(c)
    else:
        for v in GRID[a.axis]:
            c = dict(base); c[a.axis] = v
            combos.append(c)

    print(f"[sweep: {len(combos)} configs x {a.episodes} episodes]", flush=True)
    out = []
    t0 = time.perf_counter()
    for c in combos:
        w = run(c["lr"], c["hidden"], c["gamma"], a.episodes, a.seed)
        out.append({**c, "win_vs_best_heuristic": w})
        print(f"  lr={c['lr']:<7} hidden={c['hidden']:<5} gamma={c['gamma']:<5} "
              f"-> {w:.4f}", flush=True)
    ws = [o["win_vs_best_heuristic"] for o in out]
    print(f"\n  spread: {min(ws):.4f} - {max(ws):.4f}  (range {max(ws)-min(ws):.4f})",
          flush=True)
    json.dump({"env": runenv.snapshot(), "episodes": a.episodes, "seed": a.seed, "baseline": base,
               "results": out, "spread": round(max(ws) - min(ws), 4),
               "seconds": time.perf_counter() - t0},
              open(f"sweep_s{a.seed}.json", "w"), indent=2)
