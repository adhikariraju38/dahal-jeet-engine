"""Neural Fictitious Self-Play (gap #3).

NFSP (Heinrich & Silver) is one of the two canonical deep methods for
imperfect-information games. Its absence from a paper on an imperfect-
information card game is the first thing a specialist reviewer notices.

Reference: Heinrich & Silver, arXiv:1603.01121 (verified; in references.bib).

THE IDEA. Each player keeps two policies:
  * a BEST-RESPONSE net (Q-learning) that exploits the opponents' current
    average behaviour, and
  * an AVERAGE-POLICY net (supervised) that imitates its own past
    best-response actions.
Acting from a mixture -- best response with probability eta, average policy
otherwise -- makes the average policy converge toward a fixed point of the
best-response map. The average policy, not the best response, is the agent.

Two buffers, and the distinction matters:
  * circular replay buffer for the RL half (recent transitions)
  * RESERVOIR buffer for the supervised half, so the average policy sees a
    uniform sample of the WHOLE history rather than only recent play.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import deque

sys.path.insert(0, ".")
import runenv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dahaljeet.agents import Agent, REGISTRY
from dahaljeet.encode import ACTION_SIZE, OBS_SIZE, encode, legal_mask
from dahaljeet.hand import Hand, team_of
from dahaljeet.tournament import duplicate_match
from dahaljeet.view import make_view

NEG = -1e9


def mlp(out):
    return nn.Sequential(
        nn.Linear(OBS_SIZE, 512), nn.ReLU(),
        nn.Linear(512, 512), nn.ReLU(),
        nn.Linear(512, 256), nn.ReLU(),
        nn.Linear(256, out))


class Reservoir:
    """Uniform sample of the entire stream, in bounded memory.

    Vital for NFSP: the average policy must average over ALL past behaviour,
    so a recency-biased circular buffer would defeat the algorithm.
    """

    def __init__(self, cap, rng):
        self.cap, self.rng, self.n, self.data = cap, rng, 0, []

    def add(self, item):
        # float32 arrays, not Python float lists -- see the note in deepcfr.py
        item = tuple(np.asarray(x, dtype=np.float32) if isinstance(x, list)
                     else x for x in item)
        self.n += 1
        if len(self.data) < self.cap:
            self.data.append(item)
        else:
            j = self.rng.randrange(self.n)
            if j < self.cap:
                self.data[j] = item

    def sample(self, k):
        return self.rng.sample(self.data, min(k, len(self.data)))

    def __len__(self):
        return len(self.data)


class AveragePolicyAgent(Agent):
    """The NFSP agent proper: the AVERAGE policy, not the best response."""

    name = "NFSP"

    def __init__(self, pi, rng=None, greedy=True):
        super().__init__(rng)
        self.pi, self.greedy = pi, greedy

    @torch.no_grad()
    def act(self, v):
        x = torch.tensor(encode(v), dtype=torch.float32).unsqueeze(0)
        m = torch.tensor(legal_mask(v), dtype=torch.float32).unsqueeze(0)
        lg = self.pi(x).masked_fill(m == 0, NEG)
        if self.greedy:
            return int(lg.argmax(-1))
        p = torch.softmax(lg, dim=-1)
        return int(torch.multinomial(p, 1))


def train(episodes, seed, eta=0.1, anticipatory_eval=200):
    rng = random.Random(seed)
    torch.manual_seed(seed)
    q, qt = mlp(ACTION_SIZE), mlp(ACTION_SIZE)     # best response
    qt.load_state_dict(q.state_dict())
    pi = mlp(ACTION_SIZE)                          # average policy
    opt_q = torch.optim.Adam(q.parameters(), lr=1e-4)
    opt_p = torch.optim.Adam(pi.parameters(), lr=1e-3)

    replay = deque(maxlen=100_000)
    reservoir = Reservoir(150_000, rng)
    eps, gamma, batch, step = 0.12, 0.99, 128, 0
    log = []

    for ep in range(episodes):
        # Anticipatory dynamics: with prob eta this episode is played by the
        # best response (and those actions train the average policy);
        # otherwise by the average policy itself.
        use_br = rng.random() < eta
        h = Hand(dealer=ep % 4, rng=rng)
        h.deal()
        learner_team = ep % 2
        pending = {}

        while not h.is_over:
            seat = h.to_act
            v = make_view(h, seat)
            obs, mask = encode(v), legal_mask(v)
            legal = [i for i, m in enumerate(mask) if m]

            if team_of(seat) == learner_team:
                if use_br:
                    if rng.random() < eps:
                        a = rng.choice(legal)
                    else:
                        with torch.no_grad():
                            ql = q(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
                            a = int(ql.masked_fill(
                                torch.tensor(mask).unsqueeze(0) == 0, NEG).argmax(-1))
                    # only best-response actions feed the average policy
                    reservoir.add((obs, mask, a))
                else:
                    with torch.no_grad():
                        lg = pi(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
                        p = torch.softmax(lg.masked_fill(
                            torch.tensor(mask).unsqueeze(0) == 0, NEG), dim=-1)
                        a = int(torch.multinomial(p, 1))
                pending[seat] = (obs, mask, a)
            else:
                # opponents play the current average policy: self-play
                with torch.no_grad():
                    lg = pi(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
                    p = torch.softmax(lg.masked_fill(
                        torch.tensor(mask).unsqueeze(0) == 0, NEG), dim=-1)
                    a = int(torch.multinomial(p, 1))
            h.play(a)

        r = h.result()
        reward = 1.0 if r.winning_team == learner_team else -1.0
        for seat, (obs, mask, a) in pending.items():
            replay.append((obs, mask, a, reward))
        step += 1

        # ---- RL half (best response)
        if len(replay) >= 2000 and step % 2 == 0:
            bt = rng.sample(replay, batch)
            bo = torch.tensor([b[0] for b in bt], dtype=torch.float32)
            ba = torch.tensor([b[2] for b in bt], dtype=torch.long)
            br = torch.tensor([b[3] for b in bt], dtype=torch.float32)
            pred = q(bo).gather(1, ba.unsqueeze(1)).squeeze(1)
            loss = F.smooth_l1_loss(pred, br)
            opt_q.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(q.parameters(), 10.0); opt_q.step()

        # ---- supervised half (average policy)
        if len(reservoir) >= 2000 and step % 2 == 0:
            bt = reservoir.sample(batch)
            bo = torch.from_numpy(np.stack([b[0] for b in bt]))
            bm = torch.from_numpy(np.stack([b[1] for b in bt]))
            ba = torch.tensor([int(b[2]) for b in bt], dtype=torch.long)
            lg = pi(bo).masked_fill(bm == 0, NEG)
            loss = F.cross_entropy(lg, ba)
            opt_p.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(pi.parameters(), 10.0); opt_p.step()

        if step % 2000 == 0:
            qt.load_state_dict(q.state_dict())

        if (ep + 1) % max(1, episodes // 8) == 0:
            ag = AveragePolicyAgent(pi, rng=random.Random(1))
            res = {}
            for opp in ("Random", "TensThenTricks"):
                m = duplicate_match(ag, REGISTRY[opp](rng=random.Random(4)),
                                    n_deals=anticipatory_eval, seed=555,
                                    timed=False)
                res[opp] = round(m.win_rate_a, 4)
            print(f"    ep {ep+1:6d} reservoir={len(reservoir):,} {res}",
                  flush=True)
            log.append({"ep": ep + 1, **res})
    return pi, q, log


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=60000)
    ap.add_argument("--eta", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    t0 = time.perf_counter()
    print(f"[NFSP | eta={a.eta} | {a.episodes} episodes | seed {a.seed}]",
          flush=True)
    pi, q, log = train(a.episodes, a.seed, a.eta)
    ag = AveragePolicyAgent(pi, rng=random.Random(1))
    final = {}
    for opp in ("Random", "TensThenTricks"):
        m = duplicate_match(ag, REGISTRY[opp](rng=random.Random(4)),
                            n_deals=600, seed=555, timed=False)
        final[opp] = round(m.win_rate_a, 4)
    el = time.perf_counter() - t0
    print(f"FINAL nfsp: {final} ({el:.0f}s)", flush=True)
    torch.save(pi.state_dict(), f"rl_nfsp_s{a.seed}.pt")
    json.dump({"env": runenv.snapshot(), "method": "nfsp", "eta": a.eta, "episodes": a.episodes,
               "seed": a.seed, "final": final, "curve": log,
               "seconds": el}, open(f"rl_nfsp_s{a.seed}.json", "w"), indent=2)
