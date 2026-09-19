"""Phase 4b -- the remaining learning methods, for full coverage.

Adds to DQN / PPO / MAPPO:
  * Double DQN      -- decouples action selection from evaluation
  * Dueling DQN     -- separates state value from action advantage
  * A2C             -- synchronous actor-critic
  * Behaviour cloning / policy DISTILLATION from ISMCTS

Distillation is the one with a real shot at being useful rather than just
complete: ISMCTS plays well but costs ~35 ms a move. A network that imitates it
runs in ~0.1 ms, which is the difference between a paper result and something
that runs on a phone.
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
import time
from collections import deque

sys.path.insert(0, '.')
import runenv
import torch
import torch.nn as nn
import torch.nn.functional as F

from dahaljeet.agents import REGISTRY
from dahaljeet.encode import OBS_SIZE, ACTION_SIZE
from dahaljeet.env import DahalJeetEnv
from train_rl import Net, TorchAgent, evaluate

NEG = -1e9


# ------------------------------------------------------------ architectures

class DuelNet(nn.Module):
    """Dueling: Q(s,a) = V(s) + A(s,a) - mean_a A(s,a)."""

    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(OBS_SIZE, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU())
        self.val = nn.Linear(256, 1)
        self.adv = nn.Linear(256, ACTION_SIZE)

    def forward(self, x):
        h = self.body(x)
        a = self.adv(h)
        return self.val(h) + a - a.mean(dim=-1, keepdim=True)


# -------------------------------------------------------------- DQN family

def train_dqn_variant(kind, episodes, seed, reward="potential"):
    """kind: 'double' | 'dueling' | 'double_dueling'."""
    rng = random.Random(seed)
    torch.manual_seed(seed)
    mk = DuelNet if "dueling" in kind else Net
    q, tgt = mk(), mk()
    tgt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=3e-4)
    buf = deque(maxlen=100_000)
    opp = REGISTRY["TensThenTricks"](rng=random.Random(seed + 1))
    eps, gamma, batch, step = 1.0, 0.99, 128, 0
    double = "double" in kind
    log = []

    for ep in range(episodes):
        env = DahalJeetEnv(opp, reward=reward, learner_team=ep % 2,
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
                    a = int(ql.masked_fill(
                        torch.tensor(mask).unsqueeze(0) == 0, NEG).argmax(-1))
            s2, r, done, _ = env.step(a)
            nobs, nmask = s2 if s2 is not None else (None, None)
            buf.append((obs, a, r, nobs, nmask, done))
            s = s2
            step += 1

            if len(buf) >= 2000 and step % 4 == 0:
                bt = rng.sample(buf, batch)
                bo = torch.tensor([b[0] for b in bt], dtype=torch.float32)
                ba = torch.tensor([b[1] for b in bt], dtype=torch.long)
                target = torch.tensor([b[2] for b in bt], dtype=torch.float32)
                nz = [i for i, b in enumerate(bt)
                      if not b[5] and b[3] is not None]
                if nz:
                    no = torch.tensor([bt[i][3] for i in nz], dtype=torch.float32)
                    nm = torch.tensor([bt[i][4] for i in nz], dtype=torch.float32)
                    with torch.no_grad():
                        if double:
                            # select with the online net, evaluate with target
                            sel = q(no).masked_fill(
                                nm == 0, NEG).argmax(-1, keepdim=True)
                            nq = tgt(no).gather(1, sel).squeeze(1)
                        else:
                            nq = tgt(no).masked_fill(nm == 0, NEG).max(-1).values
                    target[nz] += gamma * nq
                pred = q(bo).gather(1, ba.unsqueeze(1)).squeeze(1)
                loss = F.smooth_l1_loss(pred, target)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), 10.0)
                opt.step()
            if step % 2000 == 0:
                tgt.load_state_dict(q.state_dict())
        eps = max(0.05, eps * 0.9997)

        if (ep + 1) % max(1, episodes // 6) == 0:
            res = evaluate(TorchAgent(q, kind), deals=150)
            print(f"    {kind} ep {ep+1}: {res}", flush=True)
            log.append({"ep": ep + 1, **res})
    return q, log


# --------------------------------------------------------------------- A2C

def train_a2c(episodes, seed, reward="potential"):
    rng = random.Random(seed)
    torch.manual_seed(seed)
    net = Net(critic=True)
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    opp = REGISTRY["TensThenTricks"](rng=random.Random(seed + 1))
    gamma, log = 0.99, []

    for ep in range(episodes):
        env = DahalJeetEnv(opp, reward=reward, learner_team=ep % 2,
                           dealer=ep % 4, rng=rng)
        s = env.reset()
        traj = []
        while s is not None:
            obs, mask = s
            ot = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            mt = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
            lg, v = net(ot)
            lg = lg.masked_fill(mt == 0, NEG)
            d = torch.distributions.Categorical(logits=lg)
            a = d.sample()
            s2, r, done, _ = env.step(int(a))
            traj.append((d.log_prob(a), v.squeeze(), r, d.entropy()))
            s = s2

        R, pl, vl, el = 0.0, [], [], []
        for lp, v, r, ent in reversed(traj):
            R = r + gamma * R
            adv = R - v.detach()
            pl.append(-lp * adv)
            vl.append(F.mse_loss(v, torch.tensor(R)))
            el.append(-0.01 * ent)
        loss = (torch.stack(pl).sum() + torch.stack(vl).sum()
                + torch.stack(el).sum())
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 0.5)
        opt.step()

        if (ep + 1) % max(1, episodes // 6) == 0:
            res = evaluate(TorchAgent(net, "A2C"), deals=150)
            print(f"    a2c ep {ep+1}: {res}", flush=True)
            log.append({"ep": ep + 1, **res})
    return net, log


# ----------------------------------------------------- behaviour cloning

def train_distill(path, epochs, seed):
    """Supervised imitation of ISMCTS -- the practical payoff of the project.

    Distillation overfits here: training loss keeps falling while validation
    accuracy and, more importantly, PLAYING STRENGTH peak early and then decay.
    So we select the best checkpoint by held-out win rate rather than taking
    the last epoch, and stop early once it stops improving.
    """
    torch.manual_seed(seed)
    with open(path, "rb") as f:
        data = pickle.load(f)
    print(f"    distilling from {len(data):,} ISMCTS decisions", flush=True)

    X = torch.tensor([d[0] for d in data], dtype=torch.float32)
    M = torch.tensor([d[1] for d in data], dtype=torch.float32)
    Y = torch.tensor([d[2] for d in data], dtype=torch.long)
    idx = torch.randperm(len(X))
    cut = int(len(X) * 0.9)
    tr, va = idx[:cut], idx[cut:]

    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    log, bs = [], 512
    best = {"score": -1.0, "state": None, "epoch": 0}
    patience, since_best = 6, 0

    for e in range(epochs):
        net.train()
        perm = tr[torch.randperm(len(tr))]
        tot = 0.0
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            lg = net(X[b]).masked_fill(M[b] == 0, NEG)
            loss = F.cross_entropy(lg, Y[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(b)

        net.eval()
        with torch.no_grad():
            lg = net(X[va]).masked_fill(M[va] == 0, NEG)
            acc = float((lg.argmax(-1) == Y[va]).float().mean())
        res = evaluate(TorchAgent(net, "Distill"), deals=250)
        score = res["TensThenTricks"]          # model-selection criterion

        star = ""
        if score > best["score"]:
            best = {"score": score,
                    "state": {k: v.clone() for k, v in net.state_dict().items()},
                    "epoch": e + 1}
            since_best = 0
            star = "   <- best"
        else:
            since_best += 1

        print(f"    distill epoch {e+1}: loss {tot/len(perm):.4f} "
              f"val-acc {acc:.4f} {res}{star}", flush=True)
        log.append({"epoch": e + 1, "loss": tot / len(perm),
                    "val_acc": acc, "is_best": bool(star), **res})

        if since_best >= patience:
            print(f"    early stop: no gain in {patience} epochs", flush=True)
            break

    if best["state"] is not None:
        net.load_state_dict(best["state"])
        print(f"    restored best checkpoint: epoch {best['epoch']} "
              f"({best['score']:.4f} vs TensThenTricks)", flush=True)
    return net, log


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True,
                    choices=["double", "dueling", "double_dueling",
                             "a2c", "distill"])
    ap.add_argument("--episodes", type=int, default=9000)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--data", default="expert_data.pkl")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    t0 = time.perf_counter()
    print(f"[{a.method}]", flush=True)
    if a.method == "distill":
        net, log = train_distill(a.data, a.epochs, a.seed)
    elif a.method == "a2c":
        net, log = train_a2c(a.episodes, a.seed)
    else:
        net, log = train_dqn_variant(a.method, a.episodes, a.seed)

    final = evaluate(TorchAgent(net, a.method), deals=600)
    el = time.perf_counter() - t0
    print(f"FINAL {a.method}: {final} ({el:.0f}s)", flush=True)
    torch.save(net.state_dict(), f"rl_{a.method}_s{a.seed}.pt")
    json.dump({"env": runenv.snapshot(), "method": a.method, "final": final, "curve": log,
               "seconds": el},
              open(f"rl_{a.method}_s{a.seed}.json", "w"), indent=2)
