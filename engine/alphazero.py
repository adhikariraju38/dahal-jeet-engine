"""Neural-guided information-set search (gap #5).

Our two strongest agents are ISMCTS (search) and a network distilled from it.
The obvious combination was missing: search GUIDED by the network rather than
by uniform priors and random rollouts.

DESIGN
  * A two-headed network: policy (52 logits, masked) and value (scalar, the
    expected outcome for the side to move).
  * Determinized information-set search, as in ISMCTS, but:
      - node expansion uses the POLICY head as a PUCT prior
      - leaves are evaluated by the VALUE head instead of a rollout to the end
    Removing the rollout is what buys the speed; the prior is what buys the
    strength.
  * The policy head is initialised from the distilled ISMCTS network, so search
    starts from a competent prior rather than from noise. The value head is
    trained by regression on hand outcomes.

Two stages:
    python alphazero.py --stage value  --epochs 12     # train the value head
    python alphazero.py --stage eval   --sims 100      # evaluate the agent
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import sys
import time

sys.path.insert(0, ".")
import runenv
import torch
import torch.nn as nn
import torch.nn.functional as F

from dahaljeet.agents import Agent, REGISTRY, _safe_discard
from dahaljeet.determinize import determinize, hand_from_view
from dahaljeet.encode import ACTION_SIZE, OBS_SIZE, encode, legal_mask
from dahaljeet.hand import Hand, team_of
from dahaljeet.tournament import duplicate_match
from dahaljeet.view import make_view

NEG = -1e9


class PolicyValueNet(nn.Module):
    """Shared trunk, policy head + value head."""

    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(OBS_SIZE, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU())
        self.policy = nn.Linear(256, ACTION_SIZE)
        self.value = nn.Linear(256, 1)

    def forward(self, x):
        h = self.body(x)
        return self.policy(h), torch.tanh(self.value(h)).squeeze(-1)

    def load_distilled_policy(self, path):
        """Warm-start policy + trunk from the distilled ISMCTS network.

        The distilled net (train_rl.Net) has the same trunk shape and a `head`
        that corresponds to our `policy`; the value head starts fresh.
        """
        sd = torch.load(path, map_location="cpu")
        own = self.state_dict()
        moved = 0
        for k, v in sd.items():
            tgt = k.replace("head.", "policy.")
            if tgt in own and own[tgt].shape == v.shape:
                own[tgt] = v
                moved += 1
        self.load_state_dict(own)
        return moved


class NeuralISMCTS(Agent):
    """Determinized information-set search with a neural prior and leaf value."""

    name = "AlphaDJ"

    def __init__(self, net, sims=100, c_puct=1.4, rng=None, name=None):
        super().__init__(rng)
        self.net = net
        self.sims = sims
        self.c = c_puct
        if name:
            self.name = name
        else:
            self.name = f"AlphaDJ{sims}"

    @torch.no_grad()
    def _pv(self, hand, seat):
        v = make_view(hand, seat)
        x = torch.tensor(encode(v), dtype=torch.float32).unsqueeze(0)
        m = torch.tensor(legal_mask(v), dtype=torch.float32).unsqueeze(0)
        logits, val = self.net(x)
        p = torch.softmax(logits.masked_fill(m == 0, NEG), dim=-1)[0]
        return p, float(val.item())

    @torch.no_grad()
    def act(self, view):
        if len(view.legal) == 1:
            return view.legal[0]
        me = view.seat
        my_team = team_of(me)
        # stats[path][move] = [visits, value_sum, prior]
        stats: dict[tuple, dict[int, list]] = {}

        for _ in range(self.sims):
            d = determinize(view, self.rng)
            if d is None:
                continue
            h = hand_from_view(view, d)
            path: tuple = ()
            depth = 0

            while not h.is_over and depth < 26:
                seat = h.to_act
                legal = h.legal_moves(seat)
                node = stats.get(path)
                if node is None:
                    # expand: one network call gives priors AND the leaf value
                    prior, val = self._pv(h, seat)
                    stats[path] = {m: [0, 0.0, float(prior[m])] for m in legal}
                    # value is from `seat`'s point of view; convert to ours
                    leaf = val if team_of(seat) == my_team else -val
                    break
                # PUCT over the moves legal in THIS determinization
                total = sum(node[m][0] for m in legal if m in node) + 1
                best, best_u = None, -1e18
                for m in legal:
                    if m not in node:
                        prior, _ = self._pv(h, seat)
                        node[m] = [0, 0.0, float(prior[m])]
                    n, w, p = node[m]
                    q = (w / n) if n else 0.0
                    u = q + self.c * p * math.sqrt(total) / (1 + n)
                    if u > best_u:
                        best, best_u = m, u
                h.play(best)
                path = path + (best,)
                depth += 1
            else:
                if h.is_over:
                    r = h.result()
                    leaf = 1.0 if r.winning_team == my_team else -1.0
                else:
                    leaf = 0.0

            # backprop along the traversed path
            for i in range(len(path)):
                pre, mv = path[:i], path[i]
                if pre in stats and mv in stats[pre]:
                    stats[pre][mv][0] += 1
                    stats[pre][mv][1] += leaf

        root = stats.get(())
        if not root:
            return _safe_discard(view)
        legal = set(view.legal)
        cand = {m: s for m, s in root.items() if m in legal}
        if not cand:
            return _safe_discard(view)
        return max(cand, key=lambda m: cand[m][0])      # most visited


# ------------------------------------------------------------ value training

def train_value(net, hands=6000, epochs=10, seed=0):
    """Regress the value head on hand outcomes from mixed-strength play."""
    rng = random.Random(seed)
    torch.manual_seed(seed)
    pool = ["TensThenTricks", "GreedyTricks", "PartnerAware", "Random",
            "TenAware", "Adaptive"]
    X, Y = [], []
    print(f"  generating {hands} hands of value data...", flush=True)
    for i in range(hands):
        a = REGISTRY[rng.choice(pool)](rng=random.Random(rng.random()))
        b = REGISTRY[rng.choice(pool)](rng=random.Random(rng.random()))
        h = Hand(dealer=i % 4, rng=rng)
        h.deal()
        seen = []
        while not h.is_over:
            seat = h.to_act
            v = make_view(h, seat)
            seen.append((encode(v), team_of(seat)))
            h.play((a if seat % 2 == 0 else b).act(v))
        w = h.result().winning_team
        for obs, t in seen:
            X.append(obs)
            Y.append(1.0 if t == w else -1.0)
    # Cap the dataset: 8000 hands yields ~416k targets, which as a float32
    # tensor plus optimiser copies pushed a 16 GB machine into swap. 150k is
    # ample for a value head this size.
    CAP = 150_000
    if len(X) > CAP:
        keep = rng.sample(range(len(X)), CAP)
        X = [X[i] for i in keep]
        Y = [Y[i] for i in keep]
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    print(f"  {len(X):,} value targets (capped at {CAP:,})", flush=True)

    idx = torch.randperm(len(X))
    cut = int(len(X) * 0.9)
    tr, va = idx[:cut], idx[cut:]
    # Freeze nothing: the trunk may adapt, but the policy head is kept honest
    # by a distillation term so warm-started priors are not destroyed.
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    with torch.no_grad():
        ref_logits, _ = net(X[tr[:8000]])
        ref = torch.softmax(ref_logits, dim=-1)
    bs = 512
    for e in range(epochs):
        perm = tr[torch.randperm(len(tr))]
        tot = 0.0
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            logits, val = net(X[b])
            loss = F.mse_loss(val, Y[b])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            tot += float(loss.detach()) * len(b)
        with torch.no_grad():
            _, vv = net(X[va])
            vloss = float(F.mse_loss(vv, Y[va]))
            acc = float(((vv > 0) == (Y[va] > 0)).float().mean())
        print(f"    value epoch {e+1}: train {tot/len(perm):.4f} "
              f"val {vloss:.4f} sign-acc {acc:.4f}", flush=True)
    del ref
    return net


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["value", "eval"], default="value")
    ap.add_argument("--distilled", default="rl_distill_s0.pt")
    ap.add_argument("--hands", type=int, default=6000)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--sims", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    net = PolicyValueNet()
    ckpt = f"alphadj_s{a.seed}.pt"

    if a.stage == "value":
        t0 = time.perf_counter()
        if os.path.exists(a.distilled):
            n = net.load_distilled_policy(a.distilled)
            print(f"  warm-started {n} tensors from {a.distilled}", flush=True)
        else:
            print(f"  [warn] {a.distilled} not found — policy starts random",
                  flush=True)
        net = train_value(net, a.hands, a.epochs, a.seed)
        torch.save(net.state_dict(), ckpt)
        print(f"saved {ckpt}  ({time.perf_counter()-t0:.0f}s)", flush=True)
    else:
        net.load_state_dict(torch.load(ckpt, map_location="cpu"))
        net.eval()
        ag = NeuralISMCTS(net, sims=a.sims, rng=random.Random(a.seed))
        out = {}
        for opp in ["Random", "TensThenTricks"]:
            r = duplicate_match(ag, REGISTRY[opp](rng=random.Random(4)),
                                n_deals=200, seed=888, timed=True)
            lo, hi = r.bootstrap_ci()
            out[opp] = {"win": round(r.win_rate_a, 4),
                        "ci": [round(lo, 4), round(hi, 4)],
                        "ms": round(r.ms_per_decision_a, 2)}
            print(f"  vs {opp:16s} {r.win_rate_a:.4f} [{lo:.4f},{hi:.4f}] "
                  f"{r.ms_per_decision_a:.2f} ms", flush=True)
        json.dump({"env": runenv.snapshot(), "sims": a.sims, "seed": a.seed, "results": out},
                  open(f"alphadj_eval_s{a.seed}_n{a.sims}.json", "w"), indent=2)
