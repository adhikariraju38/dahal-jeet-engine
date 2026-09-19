"""Deep Counterfactual Regret Minimisation (gap #2).

CFR and its deep variants are the canonical family for imperfect-information
games. Their absence from a paper on an imperfect-information card game is the
first thing a specialist reviewer looks for.

Reference: Brown, Lerer, Gross & Sandholm, arXiv:1811.00164 (verified; in
references.bib). The underlying CFR paper (row D12) is still unverified.

⚠️ AN HONEST LIMITATION, WHICH MUST BE STATED IN THE PAPER
CFR's convergence guarantee holds for **two-player zero-sum** games. Dahal Jeet
is four-player with fixed partnerships. Running CFR here is standard practice
and produces a strong baseline, but it carries **no equilibrium guarantee**.
It is included because it is the expected baseline for this class of game, and
reported as such — not as a solution concept.

METHOD (external-sampling Deep CFR)
  * One ADVANTAGE network per team, predicting counterfactual regret for each
    action at an information set.
  * Traverse with external sampling: at the traverser's nodes explore every
    action; at other nodes sample one from the current strategy.
  * Regrets go to a reservoir buffer; the advantage net is refit each
    iteration from scratch (as Deep CFR prescribes).
  * A separate STRATEGY network is fit on the running average strategy — that
    average, not the final iterate, is the agent.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time

sys.path.insert(0, ".")
import runenv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dahaljeet.agents import Agent, REGISTRY
from dahaljeet.determinize import determinize, hand_from_view
from dahaljeet.encode import ACTION_SIZE, OBS_SIZE, encode, legal_mask
from dahaljeet.hand import Hand, team_of
from dahaljeet.tournament import duplicate_match
from dahaljeet.view import make_view

NEG = -1e9


def mlp():
    return nn.Sequential(
        nn.Linear(OBS_SIZE, 512), nn.ReLU(),
        nn.Linear(512, 512), nn.ReLU(),
        nn.Linear(512, 256), nn.ReLU(),
        nn.Linear(256, ACTION_SIZE))


class Reservoir:
    """Uniform sample of the whole stream, in bounded memory.

    Entries are stored as float32 numpy arrays. Python lists of floats cost
    ~28 bytes per element, so a 400k x 360 buffer of lists is over 2 GB -- it
    exhausted a 16 GB machine during development. float32 arrays cut that by
    roughly 20x.
    """

    def __init__(self, cap, rng):
        self.cap, self.rng, self.n, self.data = cap, rng, 0, []

    def add(self, item):
        item = tuple(np.asarray(x, dtype=np.float32) if isinstance(x, list)
                     else x for x in item)
        self.n += 1
        if len(self.data) < self.cap:
            self.data.append(item)
        else:
            j = self.rng.randrange(self.n)
            if j < self.cap:
                self.data[j] = item

    def __len__(self):
        return len(self.data)


def strategy_from_advantage(adv, mask):
    """Regret matching: positive regrets, normalised; uniform if none."""
    pos = torch.clamp(adv, min=0.0) * mask
    tot = pos.sum()
    if float(tot) <= 1e-8:
        return mask / mask.sum()
    return pos / tot


class CFRAgent(Agent):
    """Plays the AVERAGE strategy — the CFR solution concept, not the last iterate."""

    name = "DeepCFR"

    def __init__(self, strat_net, rng=None, greedy=False):
        super().__init__(rng)
        self.net, self.greedy = strat_net, greedy

    @torch.no_grad()
    def act(self, v):
        x = torch.tensor(encode(v), dtype=torch.float32).unsqueeze(0)
        m = torch.tensor(legal_mask(v), dtype=torch.float32).unsqueeze(0)
        p = torch.softmax(self.net(x).masked_fill(m == 0, NEG), dim=-1)[0]
        if self.greedy:
            return int(p.argmax())
        return int(torch.multinomial(p, 1))


#: nodes expanded in the current traversal batch -- instrumentation, so cost is
#: measured rather than discovered after twelve hours
NODES = {"expanded": 0, "rollouts": 0, "rollout_plies": 0}

_ROLLOUT_AGENT = None


def rollout_value(h, traverser_team, rng):
    """Finish the hand with a fixed cheap policy and return the true payoff.

    This replaces the old `return 0.0` truncation. Returning 0.0 asserted that
    every position beyond the depth limit is a draw, which is false and biases
    every regret that passes through the cutoff -- and in a game decided by four
    cards, positions near the cutoff are frequently already decided.

    Playing the hand out to a real terminal instead gives a value that is at
    least the true value UNDER THIS POLICY. That is an approximation and is
    declared as one: it is the rollout policy's value, not the current
    strategy's. It is stated in the artifact so no reader mistakes this for
    exact external-sampling CFR.
    """
    global _ROLLOUT_AGENT
    if _ROLLOUT_AGENT is None:
        _ROLLOUT_AGENT = REGISTRY["TensThenTricks"](rng=random.Random(1234))
    NODES["rollouts"] += 1
    while not h.is_over:
        h.play(_ROLLOUT_AGENT.act(make_view(h, h.to_act)))
        NODES["rollout_plies"] += 1
    return 1.0 if h.result().winning_team == traverser_team else -1.0


def traverse(h, traverser_team, adv_nets, adv_buf, strat_buf, rng, depth=0,
             max_depth=10):
    """Depth-limited external-sampling traversal.

    COST. At traverser-team nodes every legal action is expanded; at the other
    team's nodes a single action is sampled. Over a full 52-ply hand the
    traverser acts ~26 times with branching ~4, so unbounded expansion is
    ~4^26 -- which is why the original 30-ply version had not completed 6 of 40
    iterations after 12.75 hours. `max_depth` bounds the expanded prefix; the
    remainder is evaluated by rollout. Cost is then ~b^(max_depth/2), which is
    measurable and controllable.
    """
    if h.is_over:
        r = h.result()
        return 1.0 if r.winning_team == traverser_team else -1.0
    if depth >= max_depth:
        return rollout_value(h, traverser_team, rng)

    NODES["expanded"] += 1
    seat = h.to_act
    v = make_view(h, seat)
    obs = encode(v)
    mask = legal_mask(v)
    mt = torch.tensor(mask, dtype=torch.float32)
    legal = list(v.legal)

    with torch.no_grad():
        a = adv_nets[team_of(seat)](
            torch.tensor(obs, dtype=torch.float32).unsqueeze(0))[0]
    sigma = strategy_from_advantage(a, mt)

    if team_of(seat) == traverser_team:
        # explore every action, then store the regret vector
        vals = torch.zeros(ACTION_SIZE)
        node_val = 0.0
        for m in legal:
            snap = _clone(h)
            snap.play(m)
            vals[m] = traverse(snap, traverser_team, adv_nets,
                               adv_buf, strat_buf, rng, depth + 1, max_depth)
            node_val += float(sigma[m]) * float(vals[m])
        regret = (vals - node_val) * mt
        adv_buf.add((obs, mask, regret.tolist()))
        return node_val
    else:
        # sample a single action from the current strategy
        strat_buf.add((obs, mask, sigma.tolist()))
        m = int(torch.multinomial(sigma, 1))
        h.play(m)
        return traverse(h, traverser_team, adv_nets, adv_buf,
                        strat_buf, rng, depth + 1, max_depth)


def _clone(h):
    """Cheap deep-ish copy of a Hand for branching."""
    g = Hand(dealer=h.dealer)
    g.hands = [list(x) for x in h.hands]
    g.trump_suit, g.trump_card = h.trump_suit, h.trump_card
    g.trump_holder = h.trump_holder
    g.trick = list(h.trick)
    g.trick_lead_suit = h.trick_lead_suit
    g.tricks_played = h.tricks_played
    g.tens_by_team = list(h.tens_by_team)
    g.tricks_by_team = list(h.tricks_by_team)
    g.played_by_seat = [list(x) for x in h.played_by_seat]
    g.trick_winners = list(h.trick_winners)
    g.voids = [set(x) for x in h.voids]
    g.ten_owner = dict(h.ten_owner)
    g.to_act, g.leader = h.to_act, h.leader
    return g


def fit(net, buf, epochs, rng, kind):
    """Refit a network from scratch on its reservoir (Deep CFR prescribes this)."""
    if len(buf) < 512:
        return net
    net = mlp()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    X = torch.from_numpy(np.stack([b[0] for b in buf.data]))
    M = torch.from_numpy(np.stack([b[1] for b in buf.data]))
    Y = torch.from_numpy(np.stack([b[2] for b in buf.data]))
    n, bs = len(X), 512
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            out = net(X[b])
            if kind == "adv":
                loss = (((out - Y[b]) ** 2) * M[b]).sum() / M[b].sum().clamp(min=1)
            else:
                logp = torch.log_softmax(out.masked_fill(M[b] == 0, NEG), -1)
                loss = -(Y[b] * logp).sum(-1).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
    return net


def train(iters, traversals, seed, max_depth=10):
    rng = random.Random(seed)
    torch.manual_seed(seed)
    adv_nets = [mlp(), mlp()]
    strat_net = mlp()
    strat_buf = Reservoir(150_000, rng)
    log = []

    for it in range(iters):
        t_iter = time.perf_counter()
        NODES.update(expanded=0, rollouts=0, rollout_plies=0)
        for team in (0, 1):
            adv_buf = Reservoir(80_000, rng)
            for t in range(traversals):
                h = Hand(dealer=t % 4, rng=rng)
                h.deal()
                traverse(h, team, adv_nets, adv_buf, strat_buf, rng,
                         max_depth=max_depth)
            adv_nets[team] = fit(adv_nets[team], adv_buf, 4, rng, "adv")
        strat_net = fit(strat_net, strat_buf, 4, rng, "strat")
        # Per-iteration progress. The previous version printed only every
        # iters//6 iterations, so a run that had not finished one sixth of its
        # work looked identical to a hung process.
        dt = time.perf_counter() - t_iter
        print(f"    iter {it+1:3d}/{iters}  {dt:7.1f}s  "
              f"expanded {NODES['expanded']:>9,}  rollouts {NODES['rollouts']:>8,}"
              f"  strat_buf {len(strat_buf):,}"
              f"  eta {dt*(iters-it-1)/60:6.1f} min", flush=True)

        if (it + 1) % max(1, iters // 6) == 0 and len(strat_buf) >= 512:
            ag = CFRAgent(strat_net, rng=random.Random(2), greedy=True)
            res = {}
            for opp in ("Random", "TensThenTricks"):
                m = duplicate_match(ag, REGISTRY[opp](rng=random.Random(4)),
                                    n_deals=150, seed=606, timed=False)
                res[opp] = round(m.win_rate_a, 4)
            print(f"    iter {it+1:4d} strat_buf={len(strat_buf):,} {res}",
                  flush=True)
            log.append({"iter": it + 1, **res})
    return strat_net, log


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--traversals", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--depth", type=int, default=10,
                    help="plies expanded before switching to rollout evaluation")
    a = ap.parse_args()
    t0 = time.perf_counter()
    print(f"[Deep CFR | {a.iters} iters x {a.traversals} traversals | "
          f"depth {a.depth} | seed {a.seed}]", flush=True)
    print("  NOTE: no equilibrium guarantee in a 4-player partnership game — "
          "reported as a baseline, not a solution.", flush=True)
    net, log = train(a.iters, a.traversals, a.seed, max_depth=a.depth)
    ag = CFRAgent(net, rng=random.Random(2), greedy=True)
    final = {}
    for opp in ("Random", "TensThenTricks"):
        m = duplicate_match(ag, REGISTRY[opp](rng=random.Random(4)),
                            n_deals=600, seed=606, timed=False)
        final[opp] = round(m.win_rate_a, 4)
    el = time.perf_counter() - t0
    print(f"FINAL deepcfr: {final} ({el:.0f}s)", flush=True)
    torch.save(net.state_dict(), f"rl_deepcfr_s{a.seed}.pt")
    json.dump({"env": runenv.snapshot(), "method": "deepcfr", "iters": a.iters,
               "traversals": a.traversals, "seed": a.seed,
               "max_depth": a.depth,
               "sampling_scheme": (
                   "Depth-limited external sampling. All traverser actions are "
                   "expanded for the first `max_depth` plies; beyond that the "
                   "hand is completed by a fixed heuristic rollout "
                   "(TensThenTricks) and the true terminal payoff is returned. "
                   "This is an APPROXIMATION of external-sampling Deep CFR, not "
                   "the exact algorithm: values past the cutoff are the rollout "
                   "policy's, not the current strategy's. Unbounded expansion "
                   "is ~4^26 per traversal and is not computable."),
               "final": final, "curve": log, "seconds": el},
              open(f"rl_deepcfr_s{a.seed}.json", "w"), indent=2)
